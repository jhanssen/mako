// Prefill-shaped GEMM: Y[M][N] = A[M][K] . B[N][K]^T, half operands into fp32 or int8 operands into int32.
// Compares MetalPerformancePrimitives matmul2d (the path that reaches M5's matrix units) against a
// simdgroup_matrix kernel that loads fragments straight from device memory (no threadgroup staging).
//   mm_bench [M] [N] [K]
#include "mtl/context.hpp"

#include <cmath>
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
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Args { int M, N, K, pad; };

template <typename T, typename TY, int TM, int TN, int SG>
kernel void mpp_gemm(device T* A [[buffer(0)]], device T* B [[buffer(1)]], device TY* Y [[buffer(2)]],
                     constant Args& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]])
{
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device TY, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}

#define MPP_F16(TM, TN, SG) template [[host_name("mpp_" #TM "x" #TN "_sg" #SG)]] kernel void \
    mpp_gemm<half, float, TM, TN, SG>(device half*, device half*, device float*, constant Args&, uint3);
#define MPP_I8(TM, TN, SG) template [[host_name("mpp_i8_" #TM "x" #TN "_sg" #SG)]] kernel void \
    mpp_gemm<int8_t, int32_t, TM, TN, SG>(device int8_t*, device int8_t*, device int32_t*, constant Args&, uint3);
MPP_F16(32, 32, 1) MPP_F16(64, 32, 4) MPP_F16(64, 64, 4) MPP_F16(128, 64, 4)
MPP_I8(32, 32, 1) MPP_I8(64, 32, 4) MPP_I8(64, 64, 4) MPP_I8(128, 64, 4)

// 4 simdgroups per threadgroup, each a 32x32 block of Y as 4x4 fragments of 8x8.
kernel void sg_gemm(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                    constant Args& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]])
{
    const int m0 = int(tg.y) * 64 + (sg / 2) * 32;
    const int n0 = int(tg.x) * 64 + (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (int k = 0; k < a.K; k += 8) {
        simdgroup_half8x8 fa[4], fb[4];
        for (int i = 0; i < 4; ++i)
            simdgroup_load(fa[i], A + ulong(m0 + i * 8) * a.K + k, a.K);
        for (int j = 0; j < 4; ++j)
            simdgroup_load(fb[j], B + ulong(n0 + j * 8) * a.K + k, a.K, ulong2(0, 0), true);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], Y + ulong(m0 + i * 8) * a.N + n0 + j * 8, a.N);
}
)";

struct Args {
    int M, N, K, pad;
};

struct Variant {
    const char* name;
    int tm, tn, threads;
    bool int8;
};

uint16_t to_half(float f)
{
    _Float16 h = static_cast<_Float16>(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

float from_half(uint16_t u)
{
    _Float16 h;
    std::memcpy(&h, &u, 2);
    return static_cast<float>(h);
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int M = argc > 1 ? std::stoi(argv[1]) : 2048;
    const int N = argc > 2 ? std::stoi(argv[2]) : 4096;
    const int K = argc > 3 ? std::stoi(argv[3]) : 4096;
    if (M % 128 || N % 64 || K % 8) {
        std::println(stderr, "M must be a multiple of 128, N of 64, K of 8");
        return 2;
    }

    mako::mtl::Context ctx;
    std::println("{}  apple10={}  M={} N={} K={}", ctx.info().name, ctx.info().apple10, M, N, K);
    auto lib = ctx.compile(kSource);
    auto* dev = ctx.device();

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    Ref<MTL::Buffer> A(dev->newBuffer(size_t(M) * K * 2, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> B(dev->newBuffer(size_t(N) * K * 2, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> A8(dev->newBuffer(size_t(M) * K, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> B8(dev->newBuffer(size_t(N) * K, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> Y(dev->newBuffer(size_t(M) * N * 4, MTL::ResourceStorageModeShared));
    auto* a16 = static_cast<uint16_t*>(A->contents());
    auto* b16 = static_cast<uint16_t*>(B->contents());
    auto* a8 = static_cast<int8_t*>(A8->contents());
    auto* b8 = static_cast<int8_t*>(B8->contents());
    std::uniform_int_distribution<int> u8(-127, 127);
    for (size_t i = 0; i < size_t(M) * K; ++i) {
        a16[i] = to_half(u(rng));
        a8[i] = int8_t(u8(rng));
    }
    for (size_t i = 0; i < size_t(N) * K; ++i) {
        b16[i] = to_half(u(rng));
        b8[i] = int8_t(u8(rng));
    }

    auto check = [&]() {
        auto* y = static_cast<const float*>(Y->contents());
        double worst = 0;
        std::mt19937 pick(11);
        for (int s = 0; s < 64; ++s) {
            int m = pick() % M, n = pick() % N;
            double ref = 0;
            for (int k = 0; k < K; ++k)
                ref += double(from_half(a16[size_t(m) * K + k])) * from_half(b16[size_t(n) * K + k]);
            worst = std::max(worst, std::abs(ref - y[size_t(m) * N + n]) / (std::sqrt(double(K)) * 1e-2 + std::abs(ref)));
        }
        return worst;
    };
    // Integer results must be exact; returns the number of mismatching samples.
    auto check_i8 = [&]() {
        auto* y = static_cast<const int32_t*>(Y->contents());
        int bad = 0;
        std::mt19937 pick(11);
        for (int s = 0; s < 64; ++s) {
            int m = pick() % M, n = pick() % N;
            int64_t ref = 0;
            for (int k = 0; k < K; ++k)
                ref += int64_t(a8[size_t(m) * K + k]) * b8[size_t(n) * K + k];
            bad += ref != y[size_t(m) * N + n];
        }
        return bad;
    };

    const Args args{M, N, K, 0};
    const Variant variants[] = {
        {"mpp_32x32_sg1", 32, 32, 32, false},       {"mpp_64x32_sg4", 64, 32, 128, false},
        {"mpp_64x64_sg4", 64, 64, 128, false},      {"mpp_128x64_sg4", 128, 64, 128, false},
        {"sg_gemm", 64, 64, 128, false},            {"mpp_i8_32x32_sg1", 32, 32, 32, true},
        {"mpp_i8_64x32_sg4", 64, 32, 128, true},    {"mpp_i8_64x64_sg4", 64, 64, 128, true},
        {"mpp_i8_128x64_sg4", 128, 64, 128, true},
    };
    const double flop = 2.0 * M * N * K;
    for (const auto& v : variants) {
        Ref<MTL::ComputePipelineState> pso;
        try {
            pso = ctx.pipeline(lib.get(), v.name);
        } catch (const std::exception& e) {
            std::println("{:18} unavailable: {}", v.name, e.what());
            continue;
        }
        std::memset(Y->contents(), 0, size_t(M) * N * 4);
        double best = 1e9;
        for (int it = 0; it < 8; ++it) {
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(v.int8 ? A8.get() : A.get(), 0, 0);
            enc->setBuffer(v.int8 ? B8.get() : B.get(), 0, 1);
            enc->setBuffer(Y.get(), 0, 2);
            enc->setBytes(&args, sizeof(args), 3);
            enc->dispatchThreadgroups(MTL::Size(N / v.tn, M / v.tm, 1), MTL::Size(v.threads, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            if (it > 0)
                best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime());
            pool->release();
        }
        if (v.int8)
            std::println("{:18} {:7.3f} ms  {:6.2f} TOP/s    mismatches {}/64", v.name, best * 1e3, flop / best / 1e12,
                         check_i8());
        else
            std::println("{:18} {:7.3f} ms  {:6.2f} TFLOP/s  max rel err {:.2e}", v.name, best * 1e3, flop / best / 1e12,
                         check());
    }
    return 0;
}
