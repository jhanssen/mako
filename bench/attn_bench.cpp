// Flash attention on random half inputs shaped like Qwen-Image's transformer at 1024x1024: time and TFLOP/s (more
// variants in the list are compared with the first's output).
//   attn_bench [queries] [keys] [heads] [block]   (block: the bidirectional image block's first position)
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using mako::mtl::Ref;

struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t n_q = argc > 1 ? std::stoul(argv[1]) : 4096;
    const uint32_t n_k = argc > 2 ? std::stoul(argv[2]) : 4122;
    const uint32_t heads = argc > 3 ? std::stoul(argv[3]) : 32;
    const uint32_t block = argc > 4 ? std::stoul(argv[4]) : n_k - n_q;
    constexpr uint32_t D = 128;
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    const uint32_t q_pad = round_up(n_q, 128), k_pad = round_up(n_k, 128);
    auto buffer = [&](uint64_t bytes) {
        return Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    auto q = buffer(uint64_t(heads) * q_pad * D * 2), k = buffer(uint64_t(heads) * k_pad * D * 2),
         v = buffer(uint64_t(heads) * k_pad * D * 2);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    for (auto* b : {q.get(), k.get(), v.get()}) {
        auto* h = static_cast<__fp16*>(b->contents());
        for (uint64_t i = 0; i < b->length() / 2; ++i)
            h[i] = __fp16(nd(rng));
    }
    for (uint32_t h = 0; h < heads; ++h) // padded keys and values are zero, as the engine leaves them
        for (uint32_t r = n_k; r < k_pad; ++r)
            for (auto* b : {k.get(), v.get()})
                std::memset(static_cast<char*>(b->contents()) + (uint64_t(h) * k_pad + r) * D * 2, 0, D * 2);
    const double flop = 4.0 * n_q * double(n_k) * D * heads;
    std::vector<float> base;
    // name, queries a threadgroup, threads a threadgroup
    for (const auto& [name, b, threads] : std::vector<std::tuple<std::string, uint32_t, uint32_t>>{
             {"flash_attn_sg_128_16x32_4", 64, 128}}) {
        auto pso = ctx.pipeline(lib.get(), name);
        auto o = buffer(uint64_t(n_q) * heads * D * 4);
        const AttnArgs a{n_q, round_up(n_q, b), n_k, k_pad, heads, heads, n_k - n_q, block, heads * D,
                         1.0f / std::sqrt(float(D))};
        auto run = [&](int reps) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(q.get(), 0, 0);
            enc->setBuffer(k.get(), 0, 1);
            enc->setBuffer(v.get(), 0, 2);
            enc->setBuffer(o.get(), 0, 3);
            enc->setBytes(&a, sizeof(a), 4);
            for (int i = 0; i < reps; ++i)
                enc->dispatchThreadgroups(MTL::Size(a.q_pad / b, heads, 1), MTL::Size(threads, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / reps;
        };
        run(3);
        const double ms = run(20);
        const auto* of = static_cast<const float*>(o->contents());
        std::string agree = "the reference";
        if (base.empty()) {
            base.assign(of, of + uint64_t(n_q) * heads * D);
        } else {
            double e = 0, r = 0;
            for (size_t i = 0; i < base.size(); ++i) {
                e += (double(of[i]) - base[i]) * (double(of[i]) - base[i]);
                r += double(base[i]) * base[i];
            }
            agree = std::format("rel {:.1e} to the first", std::sqrt(e / r));
        }
        std::println("{:24} {:7.2f} ms  {:5.1f} TFLOP/s  {}", name, ms, flop / ms / 1e9, agree);
    }
    return 0;
}
