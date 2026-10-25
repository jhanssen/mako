// Expert GEMMs shaped like a prefill MoE layer: per expert its own weights and `tokens` rows, gate and up (k x n,
// one type) then down (n x k, another type), in one concurrent encoder as the MoE prefill issues them. Timing only
// (weights are random bytes).
//   gemm_bench [experts] [tokens] [k] [n] [gate/up type] [down type]   (types: iq2_xxs iq2_s iq3_xxs q8_0 f16)
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"

#include <cstdint>
#include <cstdio>
#include <format>
#include <print>
#include <random>
#include <string>

namespace {

using mako::mtl::Ref;
namespace gguf = mako::gguf;

gguf::Type type_of(const std::string& s)
{
    if (s == "iq2_xxs")
        return gguf::Type::IQ2_XXS;
    if (s == "iq2_s")
        return gguf::Type::IQ2_S;
    if (s == "iq3_xxs")
        return gguf::Type::IQ3_XXS;
    if (s == "q8_0")
        return gguf::Type::Q8_0;
    if (s == "f16")
        return gguf::Type::F16;
    throw std::runtime_error("unknown type " + s);
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t experts = argc > 1 ? std::stoul(argv[1]) : 288;
    const uint32_t tokens = argc > 2 ? std::stoul(argv[2]) : 114;
    const uint32_t k = argc > 3 ? std::stoul(argv[3]) : 4096;
    const uint32_t n = argc > 4 ? std::stoul(argv[4]) : 2048;
    const gguf::Type tu = type_of(argc > 5 ? argv[5] : "iq2_xxs"), td = type_of(argc > 6 ? argv[6] : "iq2_s");
    mako::mtl::Context ctx;
    auto* dev = ctx.device();
    mako::mtl::Gemm gemm(ctx);
    auto bytes_of = [](gguf::Type t, uint64_t elems) {
        const auto tr = gguf::traits(t);
        return elems / tr.block_elems * tr.block_bytes;
    };
    const uint64_t up_bytes = bytes_of(tu, uint64_t(k) * n), down_bytes = bytes_of(td, uint64_t(k) * n);
    std::mt19937 rng(7);
    auto random_buffer = [&](uint64_t bytes) {
        Ref<MTL::Buffer> b(dev->newBuffer(bytes, MTL::ResourceStorageModeShared));
        auto* p = static_cast<uint32_t*>(b->contents());
        // low bits random, high bits clear in every byte: finite half scales, moderate values
        for (uint64_t i = 0; i < bytes / 4; ++i)
            p[i] = rng() & 0x37373737u;
        return b;
    };
    Ref<MTL::Buffer> wg = random_buffer(up_bytes * experts), wu = random_buffer(up_bytes * experts);
    Ref<MTL::Buffer> wd = random_buffer(down_bytes * experts);
    const uint64_t rows = uint64_t(experts) * tokens;
    Ref<MTL::Buffer> x = random_buffer(rows * k * 2), m = random_buffer(rows * n * 2);
    Ref<MTL::Buffer> g(dev->newBuffer(rows * n * 4, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> u(dev->newBuffer(rows * n * 4, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> y(dev->newBuffer(rows * k * 4, MTL::ResourceStorageModeShared));
    double best[2] = {1e9, 1e9};
    for (int it = 0; it < 6; ++it) {
        for (int phase = 0; phase < 2; ++phase) {
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
            for (uint32_t e = 0; e < experts; ++e) {
                const uint64_t r = uint64_t(e) * tokens;
                if (phase == 0) {
                    gemm.encode(enc, tu, k, n, tokens, wg.get(), e * up_bytes, x.get(), r * k * 2, g.get(), r * n * 4);
                    gemm.encode(enc, tu, k, n, tokens, wu.get(), e * up_bytes, x.get(), r * k * 2, u.get(), r * n * 4);
                } else {
                    gemm.encode(enc, td, n, k, tokens, wd.get(), e * down_bytes, m.get(), r * n * 2, y.get(), r * k * 4);
                }
            }
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            if (it > 0)
                best[phase] = std::min(best[phase], cb->GPUEndTime() - cb->GPUStartTime());
            pool->release();
        }
    }
    const double flop = 2.0 * rows * k * n;
    std::println("{} experts x {} tokens, {}x{}: gate+up {} {:.1f} ms ({:.1f} TFLOP/s), down {} {:.1f} ms ({:.1f} TFLOP/s)",
                 experts, tokens, k, n, gguf::traits(tu).name, best[0] * 1e3, 2 * flop / best[0] / 1e12,
                 gguf::traits(td).name, best[1] * 1e3, flop / best[1] / 1e12);
    return 0;
}
