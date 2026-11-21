// The router GEMV and the routing in isolation, as a decode MoE layer runs them: the F32 router GEMV alone; with the
// moe_route dispatch after it; fused (the GEMV's last threadgroup routes); each followed by the routed experts'
// dispatch, which reads the selection (n_used IQ2_XXS experts of n_embd x ff); and with the encoder ended and an
// event signalled between the routing and the experts, as a layer with misses possible does. Each link reads its own
// copy of the router weights and the experts its routing picked, so the weights come from memory as in a step. Median
// GPU time a link over a chain in one command buffer, after a warm-up; then the same between 34 MB Q8_0 GEMVs, as in
// a step, where the routing's inputs are not in the cache. Timing only.
//   route_bench [n_embd] [n_expert] [n_used] [ff] [copies]
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"
#include "mtl/gemv.hpp"
#include "kernels/args.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using mako::kernels::RouteArgs;

using Link = std::function<void(MTL::CommandBuffer*, MTL::ComputeCommandEncoder*&, int)>;

} // namespace

int main(int argc, char** argv)
{
    using mako::gguf::Type;
    using mako::mtl::Ref;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t n_embd = argc > 1 ? std::stoul(argv[1]) : 4096;
    const uint32_t n_expert = argc > 2 ? std::stoul(argv[2]) : 288;
    const uint32_t n_used = argc > 3 ? std::stoul(argv[3]) : 8;
    const uint32_t ff = argc > 4 ? std::stoul(argv[4]) : 2048;
    const uint32_t copies = argc > 5 ? std::stoul(argv[5]) : 64;
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    mako::mtl::Gemv gemv(ctx);
    auto buffer = [&](uint64_t bytes) {
        Ref<MTL::Buffer> b(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        std::memset(b->contents(), 0, bytes);
        return b;
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    // Router weights: a different random matrix a copy, so each link routes to other experts.
    const uint64_t rw_bytes = uint64_t(n_expert) * n_embd * 4;
    auto rw = buffer(rw_bytes * copies);
    {
        auto* f = static_cast<float*>(rw->contents());
        for (uint64_t i = 0; i < rw->length() / 4; ++i)
            f[i] = nd(rng) * 0.02f;
    }
    auto x = buffer(uint64_t(n_embd) * 4);
    for (uint32_t i = 0; i < n_embd; ++i)
        static_cast<float*>(x->contents())[i] = nd(rng);
    // Expert weights: random bytes with finite f16 scales (timing only).
    const uint64_t erow = mako::mtl::Gemv::row_bytes(Type::IQ2_XXS, n_embd), estride = erow * ff;
    auto ew = buffer(estride * n_expert);
    {
        auto* p = static_cast<std::byte*>(ew->contents());
        for (uint64_t i = 0; i < ew->length(); ++i)
            p[i] = std::byte(rng() & 0x37);
    }
    auto rlog = buffer(uint64_t(n_expert) * 4), bias = buffer(uint64_t(n_expert) * 4), sel = buffer(16 * 4),
         w = buffer(16 * 4), biased = buffer(uint64_t(n_expert) * 4), slots = buffer(uint64_t(n_expert) * 4),
         hit_ids = buffer(16 * 4), miss_rank = buffer(16 * 4), indirect = buffer(9 * 4),
         counts = buffer(uint64_t(n_expert) * 4), route_counters = buffer(4), y = buffer(uint64_t(n_used) * ff * 4),
         fixed_ids = buffer(uint64_t(copies) * 16 * 4);
    for (uint32_t e = 0; e < n_expert; ++e)
        static_cast<int32_t*>(slots->contents())[e] = int32_t(e); // every expert resident, slot = expert
    for (uint32_t c = 0; c < copies; ++c)
        for (uint32_t i = 0; i < n_used; ++i)
            static_cast<int32_t*>(fixed_ids->contents())[c * 16 + i] = int32_t(rng() % n_expert);
    const RouteArgs ra{n_expert, n_used, 1, 2.5f, 2, 0, 0.0f};
    auto route_pso = ctx.pipeline(lib.get(), "moe_route");
    Ref<MTL::Event> event(ctx.device()->newEvent());
    uint64_t event_value = 0;
    const mako::mtl::Gemv::Args router{Type::F32, n_embd, n_expert, 1, 1, false};
    const mako::mtl::Gemv::Args experts{Type::IQ2_XXS, n_embd, ff, 1, n_used, true};

    auto gemv_link = [&](MTL::ComputeCommandEncoder* enc, int i) {
        gemv.encode(enc, router, rw.get(), uint64_t(i % copies) * rw_bytes, x.get(), 0, rlog.get(), 0);
    };
    auto route_link = [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(route_pso.get());
        enc->setBuffer(rlog.get(), 0, 0);
        enc->setBuffer(bias.get(), 0, 1);
        enc->setBuffer(sel.get(), 0, 2);
        enc->setBuffer(w.get(), 0, 3);
        enc->setBuffer(biased.get(), 0, 4);
        enc->setBytes(&ra, sizeof(ra), 5);
        enc->setBuffer(slots.get(), 0, 6);
        enc->setBuffer(hit_ids.get(), 0, 7);
        enc->setBuffer(miss_rank.get(), 0, 8);
        enc->setBuffer(indirect.get(), 0, 9);
        enc->setBuffer(counts.get(), 0, 10);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
    };
    auto fused_link = [&](MTL::ComputeCommandEncoder* enc, int i) {
        const bool ok = gemv.encode_router(enc, router, rw.get(), uint64_t(i % copies) * rw_bytes, x.get(), 0, rlog.get(),
                                           0, [&](MTL::ComputeCommandEncoder* e) {
                                               e->setBuffer(bias.get(), 0, 4);
                                               e->setBuffer(sel.get(), 0, 5);
                                               e->setBuffer(w.get(), 0, 6);
                                               e->setBuffer(biased.get(), 0, 7);
                                               e->setBytes(&ra, sizeof(ra), 8);
                                               e->setBuffer(slots.get(), 0, 9);
                                               e->setBuffer(hit_ids.get(), 0, 10);
                                               e->setBuffer(miss_rank.get(), 0, 11);
                                               e->setBuffer(indirect.get(), 0, 12);
                                               e->setBuffer(counts.get(), 0, 13);
                                               e->setBuffer(route_counters.get(), 0, 14);
                                           });
        if (!ok)
            throw std::runtime_error("the fused router does not take this shape");
    };
    auto experts_link = [&](MTL::ComputeCommandEncoder* enc, const MTL::Buffer* ids, uint64_t ids_off) {
        gemv.encode_indexed(enc, experts, ew.get(), 0, estride, ids, ids_off, x.get(), 0, y.get(), 0);
    };
    // The encoder ends and a new one begins, with or without an event signalled between them.
    auto split = [&](MTL::CommandBuffer* cb, MTL::ComputeCommandEncoder*& enc, bool signal) {
        enc->endEncoding();
        if (signal)
            cb->encodeSignalEvent(event.get(), ++event_value);
        enc = cb->computeCommandEncoder();
    };
    const uint32_t bcopies = 16;
    const uint64_t bbytes = mako::mtl::Gemv::row_bytes(Type::Q8_0, n_embd) * 8192;
    auto big = buffer(bbytes * bcopies), yb = buffer(8192 * 4);
    {
        auto* p = static_cast<std::byte*>(big->contents());
        for (uint64_t i = 0; i < big->length(); ++i)
            p[i] = std::byte(rng() & 0x37);
    }
    const mako::mtl::Gemv::Args big_args{Type::Q8_0, n_embd, 8192, 1, 1, false};
    const Link big_link = [&](MTL::CommandBuffer*, MTL::ComputeCommandEncoder*& enc, int i) {
        gemv.encode(enc, big_args, big.get(), uint64_t(i % bcopies) * bbytes, x.get(), 0, yb.get(), 0);
    };

    auto time = [&](const Link& link, int reps) {
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        for (int i = 0; i < reps; ++i)
            link(cb, enc, i);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e6 / reps;
    };
    auto median = [&](const Link& link) {
        double warm = 0;
        while (warm < 0.3)
            warm += time(link, 200) * 200e-6;
        std::vector<double> us;
        for (int i = 0; i < 7; ++i)
            us.push_back(time(link, 200));
        std::ranges::sort(us);
        return us[3];
    };
    std::println("router F32 {}x{} ({:.1f} MB), top {} of {}, experts IQ2_XXS {}x{} ({:.1f} MB a link)", n_embd, n_expert,
                 rw_bytes / 1e6, n_used, n_expert, n_embd, ff, estride * n_used / 1e6);
    using Enc = MTL::ComputeCommandEncoder*&;
    const std::vector<std::pair<std::string, Link>> modes{
        {"router gemv", [&](MTL::CommandBuffer*, Enc enc, int i) { gemv_link(enc, i); }},
        {"router gemv, moe_route", [&](MTL::CommandBuffer*, Enc enc, int i) { gemv_link(enc, i); route_link(enc); }},
        {"fused router", [&](MTL::CommandBuffer*, Enc enc, int i) { fused_link(enc, i); }},
        {"router gemv, experts (fixed ids)",
         [&](MTL::CommandBuffer*, Enc enc, int i) {
             gemv_link(enc, i);
             experts_link(enc, fixed_ids.get(), uint64_t(i % copies) * 64);
         }},
        {"router gemv, moe_route, experts",
         [&](MTL::CommandBuffer*, Enc enc, int i) {
             gemv_link(enc, i);
             route_link(enc);
             experts_link(enc, hit_ids.get(), 0);
         }},
        {"fused router, experts",
         [&](MTL::CommandBuffer*, Enc enc, int i) {
             fused_link(enc, i);
             experts_link(enc, hit_ids.get(), 0);
         }},
        {"fused router, new encoder, experts",
         [&](MTL::CommandBuffer* cb, Enc enc, int i) {
             fused_link(enc, i);
             split(cb, enc, false);
             experts_link(enc, hit_ids.get(), 0);
         }},
        {"fused router, signal, new encoder, experts",
         [&](MTL::CommandBuffer* cb, Enc enc, int i) {
             fused_link(enc, i);
             split(cb, enc, true);
             experts_link(enc, hit_ids.get(), 0);
         }},
    };
    for (const auto& [name, link] : modes)
        std::println("{:44} {:7.1f} us a link", name, median(link));
    const double alone = median(big_link);
    std::println("between {:.0f} MB Q8_0 GEMVs ({:.1f} us each), over the GEMV alone:", bbytes / 1e6, alone);
    for (const auto& [name, link] : modes) {
        const Link cold = [&](MTL::CommandBuffer* cb, Enc enc, int i) {
            big_link(cb, enc, i);
            link(cb, enc, i);
        };
        std::println("{:44} {:+7.1f} us a link", name, median(cold) - alone);
    }
    return 0;
}
