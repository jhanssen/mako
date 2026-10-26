// Dependent dispatches without encoder barriers. A chain of GEMVs (f16 4096x4096, 32 MB each, one's y the next's
// x) and a chain of tiny kernels, each run four ways in one command buffer: a serial encoder; a concurrent encoder
// with a Metal memory barrier between dispatches; a concurrent encoder where each dispatch waits on a device
// counter that the previous dispatch's threadgroups bump after a device-scope fence (the weights are read into
// registers before the wait, so they stream during the predecessor's tail); and a concurrent encoder with no
// synchronization at all (wrong output, the most overlap possible). Median GPU times of interleaved runs, a
// bitwise check of the chain's last output against the serial run, and the count of waits that gave up.
//   chain_bench [depth] [prefetch uint4s per lane: 0..8] [fence: 1 = barrier(mem_device), 2 = atomic fences]
//               [dependent loads in the small kernel] [rows per simdgroup in the GEMV]
#include "mtl/context.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

const char* kSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Sync { uint wait_idx, expected, signal_idx, budget, my_tg; };
constant uint K = 4096;
constant uint NSG = 4;

static inline void wait_on(device atomic_uint* ctr, constant Sync& s, device uint* bad, uint tid)
{
#if SYNC
    // 32 counters a dispatch: producer threadgroup i bumps counter i % 32; lane l waits for counter l.
    if (tid < 32) {
        const uint want = (s.expected + 31 - tid) / 32;
        uint n = 0;
        while (!simd_all(atomic_load_explicit(ctr + 32 * s.wait_idx + tid, memory_order_relaxed) >= want))
            if (++n > s.budget) { bad[0] = 1; break; }
    }
#if FENCE == 1
    threadgroup_barrier(mem_flags::mem_device);
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
#endif
}

static inline void signal_done(device atomic_uint* ctr, constant Sync& s, uint tid, uint my_tg)
{
#if SYNC
#if FENCE == 1
    threadgroup_barrier(mem_flags::mem_device);
#else
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    if (tid == 0)
        atomic_fetch_add_explicit(ctr + 32 * s.signal_idx + (my_tg % 32), 1u, memory_order_relaxed);
#endif
}

// ROWS rows per simdgroup, 8 halves a lane a step; the first row's first PRE steps are loaded before the wait.
kernel void chain_gemv(device const uint4* W [[buffer(0)]], device const float4* x [[buffer(1)]],
                       device float* y [[buffer(2)]], device atomic_uint* ctr [[buffer(3)]],
                       constant Sync& s [[buffer(4)]], device uint* bad [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]])
{
    const uint row0 = (tg * NSG + sg) * ROWS;
    device const uint4* w = W + uint64_t(row0) * (K / 8);
    uint4 pre[PRE > 0 ? PRE : 1];
#pragma unroll
    for (uint j = 0; j < PRE; ++j)
        pre[j] = w[lane + 32 * j];
    wait_on(ctr, s, bad, tid);
    for (uint r = 0; r < ROWS; ++r, w += K / 8) {
        float acc = 0;
        if (r == 0) {
#pragma unroll
            for (uint j = 0; j < PRE; ++j) {
                const uint i = lane + 32 * j;
                acc += dot(float4(as_type<half4>(pre[j].xy)), x[2 * i]) + dot(float4(as_type<half4>(pre[j].zw)), x[2 * i + 1]);
            }
#pragma unroll
            for (uint j = PRE; j < K / 256; ++j) {
                const uint i = lane + 32 * j;
                const uint4 v = w[i];
                acc += dot(float4(as_type<half4>(v.xy)), x[2 * i]) + dot(float4(as_type<half4>(v.zw)), x[2 * i + 1]);
            }
        } else {
#pragma unroll
            for (uint j = 0; j < K / 256; ++j) {
                const uint i = lane + 32 * j;
                const uint4 v = w[i];
                acc += dot(float4(as_type<half4>(v.xy)), x[2 * i]) + dot(float4(as_type<half4>(v.zw)), x[2 * i + 1]);
            }
        }
        acc = simd_sum(acc);
        if (lane == 0)
            y[row0 + r] = acc * 0.015625f;
    }
    signal_done(ctr, s, tid, tg);
}

// A small kernel with a latency chain: HOPS dependent device loads a thread, then y = f(x).
kernel void chain_small(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        device atomic_uint* ctr [[buffer(3)]], constant Sync& s [[buffer(4)]],
                        device uint* bad [[buffer(5)]], constant uint& hops [[buffer(6)]],
                        uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    wait_on(ctr, s, bad, tid);
    uint i = gid;
    float v = 0.0f;
    for (uint h = 0; h < hops; ++h) {
        v = x[i];
        i = (i + uint(as_type<uint>(v)) * 37u + 1024u) & 4095u;
    }
    y[gid] = x[gid] * 1.0001f + 1.0f + v * 0.0f;
    signal_done(ctr, s, tid, gid / 256);
}

kernel void chain_tiny(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       device atomic_uint* ctr [[buffer(3)]], constant Sync& s [[buffer(4)]],
                       device uint* bad [[buffer(5)]], uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    wait_on(ctr, s, bad, tid);
    y[gid] = x[gid] * 1.0001f + 1.0f;
    signal_done(ctr, s, tid, gid / 256);
}
)";

struct Sync {
    uint32_t wait_idx, expected, signal_idx, budget, my_tg;
};

enum class Mode { serial, barrier, flags, free };
const char* name(Mode m)
{
    switch (m) {
    case Mode::serial: return "serial";
    case Mode::barrier: return "barrier";
    case Mode::flags: return "flags";
    default: return "free";
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t depth = argc > 1 ? std::stoul(argv[1]) : 64;
    const uint32_t pre = argc > 2 ? std::stoul(argv[2]) : 4;
    const uint32_t fence = argc > 3 ? std::stoul(argv[3]) : 2;
    const uint32_t rows = argc > 5 ? std::stoul(argv[5]) : 1;
    constexpr uint32_t K = 4096, N = 4096, NSG = 4;
    mako::mtl::Context ctx;
    auto make = [&](bool sync) {
        const std::string src = std::format("#define SYNC {}\n#define PRE {}\n#define FENCE {}\n#define ROWS {}\n", sync ? 1 : 0, pre,
                                            fence, rows) + kSource;
        return ctx.compile(src);
    };
    mako::mtl::Ref<MTL::Library> lib_plain = make(false), lib_sync = make(true);
    mako::mtl::Ref<MTL::ComputePipelineState> gemv_plain = ctx.pipeline(lib_plain.get(), "chain_gemv");
    mako::mtl::Ref<MTL::ComputePipelineState> gemv_sync = ctx.pipeline(lib_sync.get(), "chain_gemv");
    mako::mtl::Ref<MTL::ComputePipelineState> tiny_plain = ctx.pipeline(lib_plain.get(), "chain_tiny");
    mako::mtl::Ref<MTL::ComputePipelineState> tiny_sync = ctx.pipeline(lib_sync.get(), "chain_tiny");
    mako::mtl::Ref<MTL::ComputePipelineState> small_plain = ctx.pipeline(lib_plain.get(), "chain_small");
    mako::mtl::Ref<MTL::ComputePipelineState> small_sync = ctx.pipeline(lib_sync.get(), "chain_small");
    const uint32_t hops = argc > 4 ? std::stoul(argv[4]) : 4;

    const uint64_t mat = uint64_t(K) * N * 2;
    mako::mtl::Ref<MTL::Buffer> w(ctx.device()->newBuffer(mat * depth, MTL::ResourceStorageModeShared));
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd;
        auto* p = static_cast<_Float16*>(w->contents());
        for (uint64_t i = 0; i < uint64_t(K) * N; ++i)
            p[i] = _Float16(nd(rng) * 0.02f);
        for (uint32_t c = 1; c < depth; ++c)
            std::memcpy(p + c * uint64_t(K) * N, p, mat);
    }
    // x0 and a y per link (the next link's x), per mode so the outputs can be compared after.
    const uint64_t vec = uint64_t(N) * 4;
    std::vector<mako::mtl::Ref<MTL::Buffer>> ys;
    for (int m = 0; m < 4; ++m)
        ys.emplace_back(ctx.device()->newBuffer(vec * (depth + 1), MTL::ResourceStorageModeShared));
    for (auto& y : ys) {
        auto* p = static_cast<float*>(y->contents());
        for (uint32_t i = 0; i < N; ++i)
            p[i] = float(i % 17) - 8.0f;
    }
    mako::mtl::Ref<MTL::Buffer> ctr(ctx.device()->newBuffer((depth + 2) * 32 * 4, MTL::ResourceStorageModeShared));
    mako::mtl::Ref<MTL::Buffer> bad(ctx.device()->newBuffer(64, MTL::ResourceStorageModeShared));
    std::memset(bad->contents(), 0, 64);

    // chain: 0 = gemv links, 1 = tiny links, 2 = small (latency) and gemv links alternating
    auto run = [&](Mode mode, int chain) {
        std::memset(ctr->contents(), 0, (depth + 2) * 32 * 4);
        MTL::Buffer* y = ys[int(mode)].get();
        std::memset(static_cast<char*>(y->contents()) + vec, 0, vec * depth); // stale values would pass as right
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder(mode == Mode::serial ? MTL::DispatchTypeSerial : MTL::DispatchTypeConcurrent);
        const bool sync = mode == Mode::flags;
        uint32_t prev_tgs = 0;
        for (uint32_t d = 0; d < depth; ++d) {
            const int kind = chain == 2 ? (d % 2 == 0 ? 2 : 0) : chain; // 0 gemv, 1 tiny, 2 small
            const bool gemv = kind == 0;
            enc->setComputePipelineState(kind == 0 ? (sync ? gemv_sync.get() : gemv_plain.get())
                                         : kind == 1 ? (sync ? tiny_sync.get() : tiny_plain.get())
                                                     : (sync ? small_sync.get() : small_plain.get()));
            const uint32_t tgs = gemv ? N / (NSG * rows) : N / 256;
            if (mode == Mode::barrier && d > 0)
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
            const Sync s{d, d == 0 ? 0u : prev_tgs, d + 1, 1u << 22, 0};
            if (gemv)
                enc->setBuffer(w.get(), d * mat, 0);
            enc->setBuffer(y, d * vec, gemv ? 1 : 0);
            enc->setBuffer(y, (d + 1) * vec, gemv ? 2 : 1);
            enc->setBuffer(ctr.get(), 0, 3);
            enc->setBytes(&s, sizeof(s), 4);
            enc->setBuffer(bad.get(), 0, 5);
            enc->setBytes(&hops, sizeof(hops), 6);
            enc->dispatchThreadgroups(MTL::Size(tgs, 1, 1), MTL::Size(gemv ? 32 * NSG : 256, 1, 1));
            prev_tgs = tgs;
        }
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (cb->status() != MTL::CommandBufferStatusCompleted)
            throw std::runtime_error("command buffer failed");
        return cb->GPUEndTime() - cb->GPUStartTime();
    };

    for (int chain : {0, 1, 2}) {
        const bool tiny = chain == 1;
        const Mode modes[] = {Mode::serial, Mode::barrier, Mode::flags, Mode::free};
        std::vector<double> times[4];
        double warm = 0;
        for (int it = 0; it < 400 && (it < 40 || warm < 0.6); ++it)
            for (int m = 0; m < 4; ++m) {
                const double t = run(modes[m], chain);
                warm += t;
                times[m].push_back(t);
            }
        std::println("{} chain, depth {}, prefetch {} uint4 a lane, {} rows a simdgroup, fence {}, {} hops:",
                     chain == 0 ? "gemv (32 MB a link)" : chain == 1 ? "tiny" : "small+gemv alternating", depth, pre,
                     rows, fence, hops);
        for (int m = 0; m < 4; ++m) {
            std::vector<double> last(times[m].end() - 30, times[m].end());
            std::ranges::sort(last);
            const double med = last[15];
            const float* ref = static_cast<const float*>(ys[0]->contents()) + depth * N;
            const float* out = static_cast<const float*>(ys[m]->contents()) + depth * N;
            const bool same = std::memcmp(ref, out, vec) == 0;
            const uint32_t gemvs = chain == 0 ? depth : chain == 2 ? depth / 2 : 0;
            if (tiny)
                std::println("  {:8}: {:7.1f} us total, {:6.2f} us a link, last y {} serial", name(modes[m]), med * 1e6,
                             med * 1e6 / depth, same ? "==" : "!=");
            else
                std::println("  {:8}: {:7.3f} ms total, {:6.1f} us a link, {:4.0f} GB/s over the GEMVs, last y {} serial",
                             name(modes[m]), med * 1e3, med * 1e6 / depth, mat * gemvs / med / 1e9, same ? "==" : "!=");
        }
        std::println("  waits that gave up: {}", static_cast<const uint32_t*>(bad->contents())[0]);
    }
    return 0;
}
