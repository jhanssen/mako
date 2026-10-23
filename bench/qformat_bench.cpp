// Decode GEMV throughput per weight format: y[T][N] = x[T][K] . W[N][K]^T with synthetic weights.
// Kernels are the upstream ggml Metal mul_mv kernels, compiled offline. Each command buffer walks a
// pool of distinct matrices (>= 512 MiB) so the weights come from DRAM, not the GPU caches.
//   qformat_bench [T...]        token counts per GEMV (default 1 4)
#include "mtl/context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using mako::mtl::Ref;

// Layout must match ggml_metal_kargs_mul_mv.
struct MulMvArgs {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int32_t nr0;
    int16_t r2, r3;
};

enum class Scale { F16, E8M0 };

struct Format {
    const char* name;
    const char* kernel;
    int block_elems, block_bytes;
    int nsg, nr0, smem;
    bool row_per_sg_grid; // grid.x counts nr0 rows per threadgroup instead of nr0 * nsg
    std::vector<int> scale_offsets;
    Scale scale = Scale::F16;
};

// nsg/nr0/smem follow the upstream defaults for these types at K >= 2048 (no split rows).
const std::vector<Format> kFormats = {
    {"Q8_0", "kernel_mul_mv_q8_0_f32", 32, 34, 4, 2, 32 * 4 * 2, true, {0}},
    {"Q6_K", "kernel_mul_mv_q6_K_f32", 256, 210, 2, 2, 0, false, {208}},
    {"Q5_K", "kernel_mul_mv_q5_K_f32", 256, 176, 2, 1, 0, false, {0, 2}},
    {"Q4_K", "kernel_mul_mv_q4_K_f32", 256, 144, 2, 2, 0, false, {0, 2}},
    {"Q4_0", "kernel_mul_mv_q4_0_f32", 32, 18, 2, 4, 0, false, {0}},
    {"IQ4_XS", "kernel_mul_mv_iq4_xs_f32", 256, 136, 2, 2, 32 * 4, false, {0}},
    {"MXFP4", "kernel_mul_mv_mxfp4_f32", 32, 17, 2, 2, 32 * 4, false, {0}, Scale::E8M0},
    {"IQ3_S", "kernel_mul_mv_iq3_s_f32", 256, 110, 2, 4, 512 * 4, false, {0}},
    {"IQ3_XXS", "kernel_mul_mv_iq3_xxs_f32", 256, 98, 2, 4, 256 * 4 + 128, false, {0}},
    {"Q2_K", "kernel_mul_mv_q2_K_f32", 256, 84, 2, 4, 0, false, {80, 82}},
    {"IQ2_S", "kernel_mul_mv_iq2_s_f32", 256, 82, 2, 4, 0, false, {0}},
    {"IQ2_XS", "kernel_mul_mv_iq2_xs_f32", 256, 74, 2, 4, 512 * 8 + 128, false, {0}},
    {"IQ2_XXS", "kernel_mul_mv_iq2_xxs_f32", 256, 66, 2, 4, 256 * 8 + 128, false, {0}},
    {"Q2_0", "kernel_mul_mv_q2_0_f32", 64, 18, 2, 8, 0, false, {0}},
};

struct Shape {
    const char* what;
    int K, N;
};

const std::vector<Shape> kShapes = {
    {"expert gate/up", 4096, 2048},
    {"expert down", 2048, 4096},
    {"kda q/k/v", 4096, 8192},
    {"kda out", 8192, 4096},
    {"mla q_b", 1536, 16384},
    {"mla out", 16384, 4096},
};

uint16_t f16_bits(float f)
{
    _Float16 h = static_cast<_Float16>(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

constexpr const char* kPeakSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void read_bw(device const uint4* src [[buffer(0)]], device uint* sink [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]])
{
    uint4 acc = 0;
    for (uint i = gid; i < n; i += grid)
        acc ^= src[i];
    if (acc.x == 0x12345678u && acc.y == 0x9abcdef0u)
        sink[0] = acc.z;
}
)";

double measure_peak(const mako::mtl::Context& ctx)
{
    auto lib = ctx.compile(kPeakSource);
    auto pso = ctx.pipeline(lib.get(), "read_bw");
    const uint64_t bytes = 1ull << 30;
    Ref<MTL::Buffer> src(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    std::memset(src->contents(), 1, bytes);
    Ref<MTL::Buffer> sink(ctx.device()->newBuffer(16, MTL::ResourceStorageModeShared));
    const uint32_t n = static_cast<uint32_t>(bytes / 16);
    double best = 1e9;
    for (int i = 0; i < 6; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso.get());
        enc->setBuffer(src.get(), 0, 0);
        enc->setBuffer(sink.get(), 0, 1);
        enc->setBytes(&n, sizeof(n), 2);
        enc->dispatchThreads(MTL::Size(1u << 20, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (i > 0)
            best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime());
        pool->release();
    }
    return bytes / best / 1e9;
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<int> tokens;
    for (int i = 1; i < argc; ++i)
        tokens.push_back(std::stoi(argv[i]));
    if (tokens.empty())
        tokens = {1, 4};

    mako::mtl::Context ctx;
    auto* dev = ctx.device();
    const double peak = measure_peak(ctx);
    std::println("{}  read peak {:.1f} GB/s (streaming kernel)", ctx.info().name, peak);

    auto lib = ctx.load(LLAMA_MUL_MV_METALLIB);
    constexpr uint64_t kPoolBytes = 512ull << 20;
    Ref<MTL::Buffer> pool(dev->newBuffer(kPoolBytes + (64ull << 20), MTL::ResourceStorageModeShared));
    {
        auto* p = static_cast<uint64_t*>(pool->contents());
        std::mt19937_64 rng(5);
        for (size_t i = 0; i < pool->length() / 8; ++i)
            p[i] = rng();
    }
    const int max_t = *std::ranges::max_element(tokens);
    Ref<MTL::Buffer> x(dev->newBuffer(size_t(16384) * max_t * 4, MTL::ResourceStorageModeShared));
    {
        auto* xf = static_cast<float*>(x->contents());
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (size_t i = 0; i < size_t(16384) * max_t; ++i)
            xf[i] = u(rng);
    }
    Ref<MTL::Buffer> y(dev->newBuffer(size_t(16384) * max_t * 4, MTL::ResourceStorageModeShared));

    std::print("{:8} {:>5}", "format", "bpw");
    for (const auto& s : kShapes)
        std::print("  {:>16}", s.what);
    std::println("");
    std::print("{:8} {:>5}", "", "");
    for (const auto& s : kShapes)
        std::print("  {:>16}", std::to_string(s.K) + "->" + std::to_string(s.N));
    std::println("");

    for (int T : tokens) {
        std::println("-- T={} (GB/s of weights read; % of read peak)", T);
        for (const auto& f : kFormats) {
            // Valid finite scales in every block of the pool, rewritten per format.
            auto* bytes = static_cast<uint8_t*>(pool->contents());
            const uint64_t blocks = pool->length() / f.block_bytes;
            const uint16_t d = f16_bits(0.001f);
            for (uint64_t b = 0; b < blocks; ++b)
                for (int off : f.scale_offsets) {
                    uint8_t* p = bytes + b * f.block_bytes + off;
                    if (f.scale == Scale::E8M0)
                        *p = 120;
                    else
                        std::memcpy(p, &d, 2);
                }

            Ref<MTL::FunctionConstantValues> fc(MTL::FunctionConstantValues::alloc()->init());
            const int16_t nsg = static_cast<int16_t>(f.nsg), nxpsg = 16, one = 1;
            const bool split = false;
            fc->setConstantValue(&nsg, MTL::DataTypeShort, 600);
            fc->setConstantValue(&nxpsg, MTL::DataTypeShort, 601);
            fc->setConstantValue(&one, MTL::DataTypeShort, 602);
            fc->setConstantValue(&one, MTL::DataTypeShort, 603);
            fc->setConstantValue(&one, MTL::DataTypeShort, 604);
            fc->setConstantValue(&split, MTL::DataTypeBool, 605);
            Ref<MTL::ComputePipelineState> pso;
            try {
                pso = ctx.pipeline(lib.get(), f.kernel, fc.get());
            } catch (const std::exception& e) {
                std::println("{:8} unavailable: {}", f.name, e.what());
                continue;
            }

            const double bpw = 8.0 * f.block_bytes / f.block_elems;
            std::print("{:8} {:5.2f}", f.name, bpw);
            for (const auto& s : kShapes) {
                const uint64_t row_bytes = uint64_t(s.K) / f.block_elems * f.block_bytes;
                const uint64_t mat_bytes = (row_bytes * s.N + 255) / 256 * 256;
                const int mats = static_cast<int>(std::max<uint64_t>(8, kPoolBytes / mat_bytes));
                if (uint64_t(mats) * mat_bytes > pool->length()) {
                    std::print("  {:>16}", "-");
                    continue;
                }
                MulMvArgs a{};
                a.ne00 = s.K;
                a.ne01 = s.N;
                a.ne02 = 1;
                a.nb00 = f.block_bytes;
                a.nb01 = row_bytes;
                a.nb02 = a.nb03 = row_bytes * s.N;
                a.ne10 = s.K;
                a.ne11 = T;
                a.ne12 = 1;
                a.nb10 = 4;
                a.nb11 = uint64_t(s.K) * 4;
                a.nb12 = a.nb13 = a.nb11 * T;
                a.ne0 = s.N;
                a.ne1 = T;
                a.nr0 = f.nr0;
                a.r2 = a.r3 = 1;
                const int rows_per_tg = f.row_per_sg_grid ? f.nr0 : f.nr0 * f.nsg;
                const MTL::Size grid((s.N + rows_per_tg - 1) / rows_per_tg, T, 1);

                double best = 1e9;
                for (int it = 0; it < 5; ++it) {
                    NS::AutoreleasePool* ap = NS::AutoreleasePool::alloc()->init();
                    auto* cb = ctx.queue()->commandBuffer();
                    auto* enc = cb->computeCommandEncoder();
                    enc->setComputePipelineState(pso.get());
                    enc->setBytes(&a, sizeof(a), 0);
                    enc->setBuffer(x.get(), 0, 2);
                    enc->setBuffer(y.get(), 0, 3);
                    if (f.smem)
                        enc->setThreadgroupMemoryLength(f.smem, 0);
                    for (int m = 0; m < mats; ++m) {
                        enc->setBuffer(pool.get(), uint64_t(m) * mat_bytes, 1);
                        enc->dispatchThreadgroups(grid, MTL::Size(32, f.nsg, 1));
                    }
                    enc->endEncoding();
                    cb->commit();
                    cb->waitUntilCompleted();
                    if (it > 0)
                        best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime());
                    ap->release();
                }
                const double gbps = double(mats) * row_bytes * s.N / best / 1e9;
                std::print("  {:9.1f} ({:3.0f}%)", gbps, 100.0 * gbps / peak);
            }
            std::println("");
            const float* yo = static_cast<const float*>(y->contents());
            if (!std::isfinite(yo[0]))
                std::println("         warning: non-finite output");
        }
    }
    return 0;
}
