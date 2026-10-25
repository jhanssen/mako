// Throughput of matmul2d with each operand type pair it accepts (half activations against int8 / int4 / int2 /
// fp8 / fp4 weights, both-low-precision pairs, bfloat) and with relaxed precision: which lower-precision paths the
// GPU's matrix units run faster than half x half. Timing only (operand values are random bytes).
//   mm_types_bench [M] [N] [K]
#include "mtl/context.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using mako::mtl::Ref;

constexpr const char* kSource = R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Args { int M, N, K, pad; };

template <typename TA, typename TB, typename TY, bool RELAXED>
kernel void mm(device uchar* A [[buffer(0)]], device uchar* B [[buffer(1)]], device uchar* Y [[buffer(2)]],
               constant Args& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]])
{
    using tA_t = tensor<device TA, dextents<int32_t, 2>, tensor_inline>;
    using tB_t = tensor<device TB, dextents<int32_t, 2>, tensor_inline>;
    using tY_t = tensor<device TY, dextents<int32_t, 2>, tensor_inline>;
    tA_t tA(reinterpret_cast<typename tA_t::data_handle_type>(A), dextents<int32_t, 2>(a.K, a.M));
    tB_t tB(reinterpret_cast<typename tB_t::data_handle_type>(B), dextents<int32_t, 2>(a.K, a.N));
    tY_t tY(reinterpret_cast<typename tY_t::data_handle_type>(Y), dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(64, 64, dynamic_length_v<int>, false, true, RELAXED);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, int(tg.y) * 64);
    auto mB = tB.slice(0, int(tg.x) * 64);
    auto mY = tY.slice(int(tg.x) * 64, int(tg.y) * 64);
    op.run(mA, mB, mY);
}

#define MM(name, TA, TB, TY, R) template [[host_name(name)]] kernel void mm<TA, TB, TY, R>( \
    device uchar*, device uchar*, device uchar*, constant Args&, uint3);
MM("half_half_float", half, half, float, false)
MM("half_half_float_relaxed", half, half, float, true)
MM("half_half_half", half, half, half, false)
MM("bfloat_bfloat_float", bfloat, bfloat, float, false)
MM("half_int8_float", half, int8_t, float, false)
MM("half_int8_float_relaxed", half, int8_t, float, true)
MM("int8_int8_int32", int8_t, int8_t, int32_t, false)
MM("half_int4_float", half, int4b_format, float, false)
MM("half_int2_float", half, int2b_format, float, false)
MM("int8_int4_int32", int8_t, int4b_format, int32_t, false)
MM("int8_int2_int32", int8_t, int2b_format, int32_t, false)
MM("half_fp8_float", half, metal_fp8_e4m3_format, float, false)
MM("half_fp4_float", half, metal_fp4_e2m1_format, float, false)
MM("fp8_fp8_float", metal_fp8_e4m3_format, metal_fp8_e4m3_format, float, false)
MM("fp4_fp4_float", metal_fp4_e2m1_format, metal_fp4_e2m1_format, float, false)
)";

struct Variant {
    const char* name;
    double a_bytes, b_bytes; // per element
};

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int M = argc > 1 ? std::stoi(argv[1]) : 4096;
    const int N = argc > 2 ? std::stoi(argv[2]) : 4096;
    const int K = argc > 3 ? std::stoi(argv[3]) : 4096;
    if (M % 64 || N % 64 || K % 64) {
        std::println(stderr, "M, N and K must be multiples of 64");
        return 2;
    }
    mako::mtl::Context ctx;
    auto* dev = ctx.device();
    std::println("{}  apple9={} apple10={}  M={} N={} K={}", ctx.info().name, ctx.info().apple9, ctx.info().apple10, M,
                 N, K);
    Ref<MTL::CompileOptions> opts(MTL::CompileOptions::alloc()->init());
    opts->setLanguageVersion(static_cast<MTL::LanguageVersion>(0x40001)); // Metal 4.1: fp8 / fp4 types
    NS::Error* err = nullptr;
    Ref<MTL::Library> lib(dev->newLibrary(NS::String::string(kSource, NS::UTF8StringEncoding), opts.get(), &err));
    if (!lib) {
        std::println(stderr, "compile failed: {}", err ? err->localizedDescription()->utf8String() : "?");
        return 1;
    }
    std::mt19937 rng(7);
    auto random_buffer = [&](size_t bytes) {
        Ref<MTL::Buffer> b(dev->newBuffer(bytes, MTL::ResourceStorageModeShared));
        auto* p = static_cast<uint8_t*>(b->contents());
        // small values in every format: low bits random, high bits clear (finite fp8/fp4, moderate half)
        for (size_t i = 0; i < bytes; ++i)
            p[i] = uint8_t(rng() & 0x37);
        return b;
    };
    Ref<MTL::Buffer> A = random_buffer(size_t(M) * K * 2), B = random_buffer(size_t(N) * K * 2);
    Ref<MTL::Buffer> Y(dev->newBuffer(size_t(M) * N * 4, MTL::ResourceStorageModeShared));
    const Variant variants[] = {
        {"half_half_float", 2, 2},     {"half_half_float_relaxed", 2, 2}, {"half_half_half", 2, 2},
        {"bfloat_bfloat_float", 2, 2}, {"half_int8_float", 2, 1},         {"half_int8_float_relaxed", 2, 1},
        {"int8_int8_int32", 1, 1},     {"half_int4_float", 2, 0.5},       {"half_int2_float", 2, 0.25},
        {"int8_int4_int32", 1, 0.5},   {"int8_int2_int32", 1, 0.25},      {"half_fp8_float", 2, 1},
        {"half_fp4_float", 2, 0.5},    {"fp8_fp8_float", 1, 1},           {"fp4_fp4_float", 0.5, 0.5},
    };
    const int args[4] = {M, N, K, 0};
    const double flop = 2.0 * M * N * K;
    for (const auto& v : variants) {
        Ref<MTL::ComputePipelineState> pso;
        try {
            pso = ctx.pipeline(lib.get(), v.name);
        } catch (const std::exception& e) {
            std::println("{:26} unavailable: {}", v.name, e.what());
            continue;
        }
        double best = 1e9;
        for (int it = 0; it < 8; ++it) {
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(A.get(), 0, 0);
            enc->setBuffer(B.get(), 0, 1);
            enc->setBuffer(Y.get(), 0, 2);
            enc->setBytes(args, sizeof(args), 3);
            enc->dispatchThreadgroups(MTL::Size(N / 64, M / 64, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            if (it > 0)
                best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime());
            pool->release();
        }
        std::println("{:26} {:7.3f} ms  {:6.1f} T(FL)OP/s", v.name, best * 1e3, flop / best / 1e12);
    }
    return 0;
}
