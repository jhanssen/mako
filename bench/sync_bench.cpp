// CPU<->GPU synchronization latency on one queue.
//   wait      commit a tiny dispatch, waitUntilCompleted
//   signal    commit a tiny dispatch + signal, CPU spins on signaledValue
//   handshake GPU waits on an event the CPU sets, runs a tiny dispatch, signals back;
//             measured from the CPU set to the CPU seeing the reply (the per-layer cost of
//             a CPU-mediated expert binding)
//   poll      a kernel spins on a shared-memory flag the CPU writes (bounded spin)
#include "mtl/context.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <print>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void touch(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    out[i] += 1u;
}

kernel void poll(device atomic_uint* flag [[buffer(0)]], device uint* out [[buffer(1)]],
                 constant uint& target [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    uint n = 0;
    while (atomic_load_explicit(flag, memory_order_relaxed) < target && n < 20000000u)
        ++n;
    out[i] = n;
}
)";

struct Stats {
    double p50, p90, p99, min;
};

Stats stats(std::vector<double> v)
{
    std::ranges::sort(v);
    auto at = [&](double p) { return v[static_cast<size_t>(p * (v.size() - 1))]; };
    return {at(0.5), at(0.9), at(0.99), v.front()};
}

void report(const char* name, const std::vector<double>& us)
{
    Stats s = stats(us);
    std::println("{:10} min {:7.1f} us  p50 {:7.1f} us  p90 {:7.1f} us  p99 {:7.1f} us  (n={})",
                 name, s.min, s.p50, s.p90, s.p99, us.size());
}

double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

void encode_touch(MTL::CommandBuffer* cb, MTL::ComputePipelineState* p, MTL::Buffer* buf)
{
    auto* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(buf, 0, 0);
    enc->dispatchThreads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    enc->endEncoding();
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int iters = argc > 1 ? std::stoi(argv[1]) : 2000;

    mako::mtl::Context ctx;
    const auto& info = ctx.info();
    std::println("{}  apple9={} apple10={} metal4={}  maxBuffer={:.1f} GB  workingSet={:.1f} GB  tgMem={} B",
                 info.name, info.apple9, info.apple10, info.metal4, info.max_buffer / 1e9,
                 info.recommended_working_set / 1e9, info.max_threadgroup_memory);

    auto lib = ctx.compile(kSource);
    auto touch = ctx.pipeline(lib.get(), "touch");
    auto poll = ctx.pipeline(lib.get(), "poll");
    mako::mtl::Ref<MTL::Buffer> buf(ctx.device()->newBuffer(4096, MTL::ResourceStorageModeShared));
    mako::mtl::Ref<MTL::Buffer> flag(ctx.device()->newBuffer(16384, MTL::ResourceStorageModeShared));
    mako::mtl::Ref<MTL::SharedEvent> ev(ctx.device()->newSharedEvent());
    auto* q = ctx.queue();

    for (int i = 0; i < 50; ++i) {
        auto* cb = q->commandBuffer();
        encode_touch(cb, touch.get(), buf.get());
        cb->commit();
        cb->waitUntilCompleted();
    }

    std::vector<double> wait_us, gpu_us;
    for (int i = 0; i < iters; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto* cb = q->commandBuffer();
        encode_touch(cb, touch.get(), buf.get());
        auto t = Clock::now();
        cb->commit();
        cb->waitUntilCompleted();
        wait_us.push_back(us_since(t));
        gpu_us.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e6);
        pool->release();
    }
    report("wait", wait_us);
    report("gpu-time", gpu_us);

    uint64_t v = ev->signaledValue();
    std::vector<double> sig_us;
    for (int i = 0; i < iters; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto* cb = q->commandBuffer();
        encode_touch(cb, touch.get(), buf.get());
        cb->encodeSignalEvent(ev.get(), ++v);
        auto t = Clock::now();
        cb->commit();
        while (ev->signaledValue() < v) {
        }
        sig_us.push_back(us_since(t));
        cb->waitUntilCompleted();
        pool->release();
    }
    report("signal", sig_us);

    // The command buffer is committed ahead of time and parked on the event, as an engine
    // would park the MoE half of a layer behind the expert binding.
    std::vector<double> hs_us;
    for (int i = 0; i < iters; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto* cb = q->commandBuffer();
        const uint64_t go = v + 1, done = v + 2;
        v = done;
        cb->encodeWait(ev.get(), go);
        encode_touch(cb, touch.get(), buf.get());
        cb->encodeSignalEvent(ev.get(), done);
        cb->commit();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        auto t = Clock::now();
        ev->setSignaledValue(go);
        while (ev->signaledValue() < done) {
        }
        hs_us.push_back(us_since(t));
        cb->waitUntilCompleted();
        pool->release();
    }
    report("handshake", hs_us);

    // One command buffer of `segs` segments, each a dispatch, a signal to the CPU and a wait for its reply,
    // as a decode step with one CPU binding per MoE layer; the CPU serves the segments while it runs.
    // Two events (GPU->CPU, CPU->GPU) or one shared by both directions.
    for (auto [two, delay_us] : {std::pair{true, 0}, {false, 0}, {true, 200}, {true, 1000}, {true, 3000}}) {
        constexpr int segs = 42;
        mako::mtl::Ref<MTL::SharedEvent> up(ctx.device()->newSharedEvent()), down(ctx.device()->newSharedEvent());
        MTL::SharedEvent* to_cpu = up.get();
        MTL::SharedEvent* to_gpu = two ? down.get() : up.get();
        std::vector<double> seg_us, step_ms;
        uint64_t u = 0;
        for (int it = 0; it < std::min(iters, 50); ++it) {
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            auto* cb = q->commandBuffer();
            std::vector<std::pair<uint64_t, uint64_t>> marks; // (GPU signals, CPU replies)
            for (int s = 0; s < segs; ++s) {
                encode_touch(cb, touch.get(), buf.get());
                const uint64_t sig = two ? ++u : (u += 2) - 1, rep = two ? u : u;
                cb->encodeSignalEvent(to_cpu, sig);
                cb->encodeWait(to_gpu, rep);
                marks.emplace_back(sig, rep);
            }
            encode_touch(cb, touch.get(), buf.get());
            const auto t0 = Clock::now();
            cb->commit();
            auto t = Clock::now();
            for (auto [sig, rep] : marks) {
                while (to_cpu->signaledValue() < sig) {
                }
                seg_us.push_back(us_since(t));
                for (auto d = Clock::now(); us_since(d) < delay_us;) {
                }
                t = Clock::now();
                to_gpu->setSignaledValue(rep);
            }
            cb->waitUntilCompleted();
            step_ms.push_back(us_since(t0) / 1e3);
            pool->release();
        }
        std::println("chain, {} event(s), CPU replies after {} us:", two ? 2 : 1, delay_us);
        report("  GPU seg", seg_us);
        std::ranges::sort(step_ms);
        std::println("  {} segments per command buffer: p50 {:.2f} ms per buffer", segs, step_ms[step_ms.size() / 2]);
    }

    auto* f = static_cast<std::atomic<uint32_t>*>(flag->contents());
    auto* out = static_cast<uint32_t*>(buf->contents());
    uint32_t target = 0;
    std::vector<double> poll_us;
    int timeouts = 0;
    const int poll_iters = std::min(iters, 10);
    for (int i = 0; i < poll_iters; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        ++target;
        auto* cb = q->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(poll.get());
        enc->setBuffer(flag.get(), 0, 0);
        enc->setBuffer(buf.get(), 0, 1);
        enc->setBytes(&target, sizeof(target), 2);
        enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        enc->endEncoding();
        cb->encodeSignalEvent(ev.get(), ++v);
        cb->commit();
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        auto t = Clock::now();
        f->store(target, std::memory_order_release);
        while (ev->signaledValue() < v) {
        }
        poll_us.push_back(us_since(t));
        cb->waitUntilCompleted();
        std::println("  poll spins {} wall {:.0f} us", out[0], poll_us.back());
        if (out[0] >= 20000000u)
            ++timeouts;
        pool->release();
    }
    report("poll", poll_us);
    std::println("poll timeouts: {} of {}", timeouts, poll_iters);
    return 0;
}
