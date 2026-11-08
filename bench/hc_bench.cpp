// The hyper-connection step in isolation: hc_step_norm (partial dots over `parts` threadgroups, then the last one's
// mix and norm) against hc_step_a and hc_step_b apart, at 8-64 threadgroups a token. Each dispatch reads its own copy
// of the mixing weights, so they come from memory as in a step. Median GPU time a dispatch over a chain in one command
// buffer, after a warm-up; also hc_step_norm's cost between 34 MB Q8_0 GEMVs, as in a step, where its inputs are not
// in the cache. Timing only.
//   hc_bench [n_embd] [tokens] [copies]
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"
#include "mtl/gemv.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

struct HcStepArgs {
    uint32_t n_embd, expand, parts, iters;
    float eps, rms_eps;
    uint32_t wn;
};

} // namespace

int main(int argc, char** argv)
{
    using mako::mtl::Ref;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t n_embd = argc > 1 ? std::stoul(argv[1]) : 4096;
    const uint32_t T = argc > 2 ? std::stoul(argv[2]) : 1;
    const uint32_t copies = argc > 3 ? std::stoul(argv[3]) : 64;
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto buffer = [&](uint64_t bytes) {
        Ref<MTL::Buffer> b(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        std::memset(b->contents(), 0, bytes);
        return b;
    };
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    auto fill = [&](MTL::Buffer* b, float s) {
        auto* f = static_cast<float*>(b->contents());
        for (uint64_t i = 0; i < b->length() / 4; ++i)
            f[i] = nd(rng) * s;
    };
    const uint64_t hd = 4 * n_embd, fn_bytes = 24 * hd * 4;
    auto fn = buffer(fn_bytes * copies);
    fill(fn.get(), 0.02f);
    auto x_post = buffer(uint64_t(T) * n_embd * 4), R = buffer(uint64_t(T) * hd * 4), out = buffer(uint64_t(T) * hd * 4),
         pc = buffer(uint64_t(T) * 24 * 4), part = buffer(uint64_t(T) * 64 * 25 * 4), scale = buffer(3 * 4),
         base = buffer(24 * 4), x = buffer(uint64_t(T) * n_embd * 4), w = buffer(n_embd * 4),
         xn = buffer(uint64_t(T) * n_embd * 4), counters = buffer(T * 4);
    fill(x_post.get(), 1.0f);
    fill(R.get(), 1.0f);
    for (uint32_t t = 0; t < T; ++t) {
        auto* p = static_cast<float*>(pc->contents()) + t * 24;
        for (int i = 0; i < 8; ++i)
            p[i] = 1.0f;
        for (int i = 8; i < 24; ++i)
            p[i] = 0.25f;
    }
    for (auto* b : {scale.get(), w.get()})
        for (uint64_t i = 0; i < b->length() / 4; ++i)
            static_cast<float*>(b->contents())[i] = 1.0f;
    // A 34 MB Q8_0 GEMV (random bytes, finite scales) between links: the step's cache state.
    mako::mtl::Gemv gemv(ctx);
    const uint32_t bcopies = 16;
    const uint64_t bbytes = mako::mtl::Gemv::row_bytes(mako::gguf::Type::Q8_0, 4096) * 8192;
    auto big = buffer(bbytes * bcopies), xb = buffer(4096 * 4), yb = buffer(8192 * 4);
    {
        auto* p = static_cast<std::byte*>(big->contents());
        for (uint64_t i = 0; i < big->length(); ++i)
            p[i] = std::byte(rng() & 0x37);
    }
    const mako::mtl::Gemv::Args big_args{mako::gguf::Type::Q8_0, 4096, 8192, 1, 1, false};
    auto big_link = [&](MTL::ComputeCommandEncoder* enc, int i) {
        gemv.encode(enc, big_args, big.get(), uint64_t(i % bcopies) * bbytes, xb.get(), 0, yb.get(), 0);
    };
    auto norm_pso = ctx.pipeline(lib.get(), "hc_step_norm");
    auto a_pso = ctx.pipeline(lib.get(), "hc_step_a");
    auto b_pso = ctx.pipeline(lib.get(), "hc_step_b");

    // GPU time of `reps` links in one command buffer, in us a link.
    auto time = [&](const std::function<void(MTL::ComputeCommandEncoder*, int)>& link, int reps) {
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        for (int i = 0; i < reps; ++i)
            link(enc, i);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e6 / reps;
    };
    auto median = [&](const std::function<void(MTL::ComputeCommandEncoder*, int)>& link) {
        double warm = 0;
        while (warm < 0.3)
            warm += time(link, 200) * 200e-6;
        std::vector<double> us;
        for (int i = 0; i < 7; ++i)
            us.push_back(time(link, 200));
        std::ranges::sort(us);
        return us[3];
    };
    std::println("hc step, n_embd {}, {} token(s), mixing weights {:.2f} MB a dispatch, {} copies", n_embd, T, fn_bytes / 1e6,
                 copies);
    for (uint32_t parts : {8u, 16u, 32u, 64u}) {
        const HcStepArgs a{n_embd, 1, parts, 20, 1e-6f, 1e-6f, n_embd};
        auto norm = [&](MTL::ComputeCommandEncoder* enc, int i) {
            enc->setComputePipelineState(norm_pso.get());
            enc->setBuffer(x_post.get(), 0, 0);
            enc->setBuffer(R.get(), 0, 1);
            enc->setBuffer(pc.get(), 0, 2);
            enc->setBuffer(out.get(), 0, 3);
            enc->setBuffer(fn.get(), uint64_t(i % copies) * fn_bytes, 4);
            enc->setBuffer(part.get(), 0, 5);
            enc->setBytes(&a, sizeof(a), 6);
            enc->setBuffer(scale.get(), 0, 7);
            enc->setBuffer(base.get(), 0, 8);
            enc->setBuffer(x.get(), 0, 9);
            enc->setBuffer(w.get(), 0, 10);
            enc->setBuffer(xn.get(), 0, 11);
            enc->setBuffer(counters.get(), 0, 12);
            enc->dispatchThreadgroups(MTL::Size(parts, T, 1), MTL::Size(256, 1, 1));
        };
        auto step_a = [&](MTL::ComputeCommandEncoder* enc, int i) {
            enc->setComputePipelineState(a_pso.get());
            enc->setBuffer(x_post.get(), 0, 0);
            enc->setBuffer(R.get(), 0, 1);
            enc->setBuffer(pc.get(), 0, 2);
            enc->setBuffer(out.get(), 0, 3);
            enc->setBuffer(fn.get(), uint64_t(i % copies) * fn_bytes, 4);
            enc->setBuffer(part.get(), 0, 5);
            enc->setBytes(&a, sizeof(a), 6);
            enc->dispatchThreadgroups(MTL::Size(parts, T, 1), MTL::Size(256, 1, 1));
        };
        auto step_b = [&](MTL::ComputeCommandEncoder* enc, int) {
            enc->setComputePipelineState(b_pso.get());
            enc->setBuffer(part.get(), 0, 0);
            enc->setBuffer(out.get(), 0, 1);
            enc->setBuffer(scale.get(), 0, 2);
            enc->setBuffer(base.get(), 0, 3);
            enc->setBuffer(pc.get(), 0, 4);
            enc->setBuffer(x.get(), 0, 5);
            enc->setBytes(&a, sizeof(a), 6);
            enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(1024, 1, 1));
        };
        auto both = [&](MTL::ComputeCommandEncoder* enc, int i) {
            step_a(enc, i);
            step_b(enc, i);
        };
        auto cold = [&](MTL::ComputeCommandEncoder* enc, int i) {
            big_link(enc, i);
            norm(enc, i);
        };
        const double alone = median(big_link);
        std::println("parts {:2}: hc_step_norm {:6.1f} us   hc_step_a {:6.1f}   hc_step_b {:6.1f}   a then b {:6.1f}   "
                     "between 34 MB GEMVs +{:5.1f}",
                     parts, median(norm), median(step_a), median(step_b), median(both), median(cold) - alone);
    }
    return 0;
}
