// Decode GEMV bandwidth per weight type: one token through a k x n matrix, repeated in one command buffer over
// distinct copies (so the weights come from memory, not cache); median of 50 runs after a warm-up. Timing only.
//   gemv_bench [k] [n] [copies] [types, e.g. q8_0,q4_k,iq2_xxs] [batch]
// batch > 1: that many matrices in one dispatch reading the same x (as a token's routed experts).
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"
#include "quant/quantize.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

// Q8_0 rows: per 32, an f16 scale and 32 int8.
void q8_0(const float* x, std::byte* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 32; ++b, x += 32, dst += 34) {
        float amax = 0;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(x[i]));
        const float d = amax / 127.0f;
        const _Float16 h = _Float16(d);
        std::memcpy(dst, &h, 2);
        for (int i = 0; i < 32; ++i)
            reinterpret_cast<int8_t*>(dst + 2)[i] = int8_t(d > 0 ? std::lrintf(x[i] / d) : 0);
    }
}

} // namespace

int main(int argc, char** argv)
{
    using mako::gguf::Type;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t k = argc > 1 ? std::stoul(argv[1]) : 5120;
    const uint32_t n = argc > 2 ? std::stoul(argv[2]) : 8192;
    const uint32_t copies = argc > 3 ? std::stoul(argv[3]) : 24;
    std::vector<Type> types{Type::Q8_0, Type::Q6_K, Type::Q5_K, Type::Q4_K};
    if (argc > 4) {
        types.clear();
        std::string list = argv[4];
        for (size_t p = 0; p <= list.size();) {
            const size_t e = std::min(list.find(',', p), list.size());
            const std::string t = list.substr(p, e - p);
            p = e + 1;
            for (Type c : {Type::Q8_0, Type::Q6_K, Type::Q5_K, Type::Q4_K, Type::IQ1_S, Type::IQ1_M, Type::IQ2_XXS,
                           Type::IQ2_XS, Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S, Type::F32})
                if (std::string(mako::gguf::traits(c).name) == t || [&] {
                        std::string u = t;
                        for (auto& ch : u)
                            ch = char(std::toupper(ch));
                        return u == mako::gguf::traits(c).name;
                    }())
                    types.push_back(c);
        }
    }
    const uint32_t batch = argc > 5 ? std::stoul(argv[5]) : 1;
    mako::mtl::Context ctx;
    mako::mtl::Gemv gemv(ctx);
    std::mt19937 rng(9);
    std::normal_distribution<float> nd;
    std::vector<float> row(k);
    for (Type type : types) {
        const uint64_t rb = type == Type::F32 ? uint64_t(k) * 4 : mako::mtl::Gemv::row_bytes(type, k);
        const uint64_t mat = rb * n * batch;
        mako::mtl::Ref<MTL::Buffer> w(ctx.device()->newBuffer(mat * copies, MTL::ResourceStorageModeShared));
        auto* p = static_cast<std::byte*>(w->contents());
        for (uint64_t r = 0; r < uint64_t(n) * batch; ++r) {
            for (auto& v : row)
                v = nd(rng) * 0.02f;
            if (type == Type::Q8_0)
                q8_0(row.data(), p + r * rb, k);
            else if (mako::quant::can_quantize(type))
                mako::quant::quantize_row(type, row.data(), p + r * rb, k);
            else if (type == Type::F32)
                std::memcpy(p + r * rb, row.data(), rb);
            else // timing only: bytes with finite f16 scales
                for (uint64_t i = 0; i < rb; ++i)
                    p[r * rb + i] = std::byte(rng() & 0x37);
        }
        for (uint32_t c = 1; c < copies; ++c)
            std::memcpy(p + c * mat, p, mat);
        mako::mtl::Ref<MTL::Buffer> x(ctx.device()->newBuffer(k * 4, MTL::ResourceStorageModeShared));
        mako::mtl::Ref<MTL::Buffer> y(
            ctx.device()->newBuffer(uint64_t(n) * 4 * copies * batch, MTL::ResourceStorageModeShared));
        std::memset(x->contents(), 0, k * 4);
        // Warm-up (~0.3 s of GPU work, so the clocks are up as in decode), then the median of 50 runs.
        std::vector<double> times;
        double warm = 0;
        for (int it = 0; it < 2000 && (it < 56 || warm < 0.3); ++it) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            for (uint32_t c = 0; c < copies; ++c)
                gemv.encode(enc, mako::mtl::Gemv::Args{type, k, n, 1, batch, true}, w.get(), c * mat, x.get(), 0,
                            y.get(), uint64_t(c) * n * 4 * batch);
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            const double t = cb->GPUEndTime() - cb->GPUStartTime();
            warm += t;
            times.push_back(t);
        }
        std::vector<double> last(times.end() - 50, times.end());
        std::ranges::sort(last);
        const double best = last[25];
        std::println("{:7} {}x{} x{}: {:.1f} MB a dispatch, {:.3f} ms each, {:.0f} GB/s", mako::gguf::traits(type).name, k,
                     n, batch, mat / 1e6, best * 1e3 / copies, mat * copies / best / 1e9);
    }
    return 0;
}
