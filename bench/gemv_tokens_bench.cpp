// Q8_0 GEMVs over 1 to 8 tokens (a verify's projections), through Gemv::encode and through a one-job
// Gemv::encode_jobs: GPU time a GEMV over distinct weight copies (so the weights come from memory), median of 7
// chains, and whether every token's result is the one-token GEMV's bit for bit.
//   gemv_tokens_bench [k n ...]
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <print>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    using mako::gguf::Type;
    using mako::mtl::Ref;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<std::pair<uint32_t, uint32_t>> shapes;
    for (int i = 1; i + 1 < argc; i += 2)
        shapes.emplace_back(std::stoul(argv[i]), std::stoul(argv[i + 1]));
    if (shapes.empty())
        shapes = {{16384, 3072}, {3072, 16384}, {3072, 1536}, {1536, 3072}, {1024, 16384}};
    mako::mtl::Context ctx;
    mako::mtl::Gemv gemv(ctx);
    auto buffer = [&](uint64_t bytes) {
        return Ref<MTL::Buffer>(ctx.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    auto table = buffer(mako::mtl::Gemv::kJobBytes * 64 * 32);
    int failures = 0;
    for (const auto [k, n] : shapes) {
        const uint64_t mat = mako::mtl::Gemv::row_bytes(Type::Q8_0, k) * n;
        const uint32_t copies = uint32_t(std::clamp<uint64_t>((1ull << 30) / mat, 1, 32));
        auto w = buffer(mat * copies);
        {
            // Blocks of an f16 scale (finite, small) and 32 int8.
            auto* p = static_cast<uint8_t*>(w->contents());
            for (uint64_t b = 0; b < w->length() / 34; ++b) {
                const __fp16 d = __fp16(0.01f * (1 + rng() % 8));
                std::memcpy(p + b * 34, &d, 2);
                for (int i = 0; i < 32; ++i)
                    p[b * 34 + 2 + i] = uint8_t(rng());
            }
        }
        auto x = buffer(uint64_t(8) * k * 4), y = buffer(uint64_t(8) * n * 4 * copies), y1 = buffer(uint64_t(8) * n * 4);
        for (uint32_t i = 0; i < 8 * k; ++i)
            static_cast<float*>(x->contents())[i] = nd(rng);
        auto run = [&](const std::function<void(MTL::ComputeCommandEncoder*, uint32_t)>& f) {
            std::vector<double> ms;
            for (int r = 0; r < 7; ++r) {
                auto* cb = ctx.queue()->commandBuffer();
                auto* e = cb->computeCommandEncoder();
                for (uint32_t c = 0; c < copies; ++c)
                    f(e, c);
                e->endEncoding();
                cb->commit();
                cb->waitUntilCompleted();
                ms.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / copies);
            }
            std::ranges::sort(ms);
            return ms[3];
        };
        std::string line = std::format("{:6}x{:<6}", k, n);
        for (const bool jobs : {false, true}) {
            line += jobs ? "  jobs:" : "  encode:";
            for (const uint32_t t : {1u, 2u, 4u, 8u}) {
                mako::mtl::Gemv::Args a{Type::Q8_0, k, n, t};
                const auto f = [&](MTL::ComputeCommandEncoder* e, uint32_t c) {
                    if (jobs) {
                        mako::mtl::Gemv::Job j;
                        j.a = a;
                        j.w = w.get();
                        j.w_off = c * mat;
                        j.x = x.get();
                        j.y = y.get();
                        j.y_off = uint64_t(c) * 8 * n * 4;
                        gemv.encode_jobs(e, std::span(&j, 1), table.get(), uint64_t(c) * mako::mtl::Gemv::kJobBytes);
                    } else {
                        gemv.encode(e, a, w.get(), c * mat, x.get(), 0, y.get(), uint64_t(c) * 8 * n * 4);
                    }
                };
                const double ms = run(f);
                // Each token against the one-token GEMV of its row.
                bool same = true;
                for (uint32_t tk = 0; tk < t; ++tk) {
                    auto* cb = ctx.queue()->commandBuffer();
                    auto* e = cb->computeCommandEncoder();
                    gemv.encode(e, {Type::Q8_0, k, n, 1}, w.get(), 0, x.get(), uint64_t(tk) * k * 4, y1.get(), 0);
                    e->endEncoding();
                    cb->commit();
                    cb->waitUntilCompleted();
                    same = same && std::memcmp(static_cast<const float*>(y->contents()) + uint64_t(tk) * n,
                                               y1->contents(), uint64_t(n) * 4) == 0;
                }
                failures += !same;
                line += std::format(" t{} {:.3f}{}", t, ms, same ? "" : " (DIFFERS)");
            }
        }
        std::println("{} ms", line);
    }
    std::println("{}", failures ? std::format("{} differ", failures) : std::string("every token as the one-token GEMV's"));
    return failures ? 1 : 0;
}
