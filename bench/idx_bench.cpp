// Hy4-flash's indexer kernels in isolation: the pool scores (hy4_idx_score) and the top-k selection, the
// single-threadgroup idx_select against the many-threadgroup hy4_sel_* chain. Random keys and queries; the two
// selections' lists are compared word for word (also over scores rounded to few values, so ties at the threshold
// decide). Median GPU time of a chain of runs in one command buffer.
//   idx_bench [tokens] [pools...]
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

struct ScoreArgs {
    uint32_t dim, heads, first, kpool;
    float w_scale;
    uint32_t stride;
};
struct SelectArgs {
    uint32_t first, n_top, kpool, score_stride, list_stride, full_tail;
};
struct SelArgs {
    uint32_t first, n_top, kpool, score_stride, list_stride, slice, slices, shift, bits;
};
constexpr uint32_t kDim = 128, kHeads = 16, kPool = 4, kTop = 511, kListStride = kTop * kPool + kPool;
constexpr uint32_t kSelWords = 2052;

} // namespace

int main(int argc, char** argv)
{
    using mako::mtl::Ref;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t T = argc > 1 ? std::stoul(argv[1]) : 1;
    std::vector<uint32_t> sizes;
    for (int i = 2; i < argc; ++i)
        sizes.push_back(std::stoul(argv[i]));
    if (sizes.empty())
        sizes = {2048, 17000, 35000, 95000, 262000};
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto buffer = [&](uint64_t bytes) {
        Ref<MTL::Buffer> b(ctx.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
        std::memset(b->contents(), 0, b->length());
        return b;
    };
    auto score_pso = ctx.pipeline(lib.get(), "hy4_idx_score");
    auto mm_pso = ctx.pipeline(lib.get(), "hy4_idx_score_mm");
    auto select_pso = ctx.pipeline(lib.get(), "idx_select");
    auto hist_pso = ctx.pipeline(lib.get(), "hy4_sel_hist");
    auto count_pso = ctx.pipeline(lib.get(), "hy4_sel_count");
    auto write_pso = ctx.pipeline(lib.get(), "hy4_sel_write");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    auto state = buffer(uint64_t(std::max(T, 128u)) * kSelWords * 4);
    auto queue = ctx.queue();
    int failures = 0;
    for (const uint32_t n_pool : sizes) {
        const uint32_t pos = n_pool * kPool, stride = (pos + T - 1) / kPool + 1;
        auto pooled = buffer(uint64_t(stride) * kDim * 4), q = buffer(uint64_t(T) * kHeads * kDim * 4),
             w = buffer(uint64_t(T) * kHeads * 4), score = buffer(uint64_t(T) * stride * 4),
             list_a = buffer(uint64_t(T) * kListStride * 4), list_b = buffer(uint64_t(T) * kListStride * 4),
             count_a = buffer(T * 4), count_b = buffer(T * 4), cnt = buffer(uint64_t(T) * 128 * 2 * 4),
             score_mm = buffer(uint64_t(T) * stride * 4);
        for (auto* b : {pooled.get(), q.get(), w.get()}) {
            auto* f = static_cast<float*>(b->contents());
            for (uint64_t i = 0; i < b->length() / 4; ++i)
                f[i] = nd(rng);
        }
        const uint32_t last_pools = (pos + T - 1) / kPool;
        const ScoreArgs sa{kDim, kHeads, pos, kPool, 1.0f / std::sqrt(float(kDim * kHeads)), stride};
        const SelectArgs sel{pos + 1, kTop, kPool, stride, kListStride, 1};
        const uint32_t slice = std::max(512u, (last_pools + 127) / 128), slices = (last_pools + slice - 1) / slice;
        using Enc = MTL::ComputeCommandEncoder;
        const auto barrier = [](Enc* e) { e->memoryBarrier(MTL::BarrierScopeBuffers); };
        const std::function<void(Enc*)> run_score = [&](Enc* e) {
            e->setComputePipelineState(score_pso.get());
            e->setBuffer(q.get(), 0, 0);
            e->setBuffer(w.get(), 0, 1);
            e->setBuffer(pooled.get(), 0, 2);
            e->setBuffer(score.get(), 0, 3);
            e->setBytes(&sa, sizeof(sa), 4);
            e->dispatchThreadgroups(MTL::Size((last_pools + 7) / 8, T, 1), MTL::Size(256, 1, 1));
            barrier(e);
        };
        const std::function<void(Enc*)> run_mm = [&](Enc* e) {
            e->setComputePipelineState(mm_pso.get());
            e->setBuffer(q.get(), 0, 0);
            e->setBuffer(w.get(), 0, 1);
            e->setBuffer(pooled.get(), 0, 2);
            e->setBuffer(score_mm.get(), 0, 3);
            e->setBytes(&sa, sizeof(sa), 4);
            const uint32_t tj = T >= 8 ? 8 : T >= 4 ? 4 : T >= 2 ? 2 : 1;
            const uint32_t mm[2] = {T, tj};
            e->setBytes(mm, sizeof(mm), 5);
            e->dispatchThreadgroups(MTL::Size((last_pools + 256 / tj - 1) / (256 / tj), (T + tj - 1) / tj, 1),
                                    MTL::Size(256, 1, 1));
            barrier(e);
        };
        const std::function<void(Enc*)> run_old = [&](Enc* e) {
            e->setComputePipelineState(select_pso.get());
            e->setBuffer(score.get(), 0, 0);
            e->setBytes(&sel, sizeof(sel), 1);
            e->setBuffer(list_a.get(), 0, 2);
            e->setBuffer(count_a.get(), 0, 3);
            e->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
            barrier(e);
        };
        const std::function<void(Enc*)> run_new = [&](Enc* e) {
            SelArgs a{pos + 1, kTop, kPool, stride, kListStride, slice, slices, 21, 11};
            const MTL::Size grid(slices, T, 1), tg(256, 1, 1);
            for (const auto [shift, bits] : {std::pair{21u, 11u}, {10u, 11u}, {0u, 10u}}) {
                a.shift = shift;
                a.bits = bits;
                e->setComputePipelineState(hist_pso.get());
                e->setBuffer(score.get(), 0, 0);
                e->setBytes(&a, sizeof(a), 1);
                e->setBuffer(state.get(), 0, 2);
                e->dispatchThreadgroups(grid, tg);
                barrier(e);
            }
            e->setComputePipelineState(count_pso.get());
            e->setBuffer(score.get(), 0, 0);
            e->setBytes(&a, sizeof(a), 1);
            e->setBuffer(state.get(), 0, 2);
            e->setBuffer(cnt.get(), 0, 3);
            e->dispatchThreadgroups(grid, tg);
            barrier(e);
            e->setComputePipelineState(write_pso.get());
            e->setBuffer(score.get(), 0, 0);
            e->setBytes(&a, sizeof(a), 1);
            e->setBuffer(state.get(), 0, 2);
            e->setBuffer(cnt.get(), 0, 3);
            e->setBuffer(list_b.get(), 0, 4);
            e->setBuffer(count_b.get(), 0, 5);
            e->dispatchThreadgroups(grid, tg);
            barrier(e);
        };
        // Median GPU ms of one run over chains of 20.
        const auto time = [&](const std::function<void(Enc*)>& f) {
            std::vector<double> ms;
            for (int r = 0; r < 7; ++r) {
                auto* cb = queue->commandBuffer();
                auto* e = cb->computeCommandEncoder();
                for (int i = 0; i < 20; ++i)
                    f(e);
                e->endEncoding();
                cb->commit();
                cb->waitUntilCompleted();
                ms.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / 20);
            }
            std::ranges::sort(ms);
            return ms[3];
        };
        const auto once = [&](const std::function<void(Enc*)>& f) {
            auto* cb = queue->commandBuffer();
            auto* e = cb->computeCommandEncoder();
            f(e);
            e->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
        };
        const auto same = [&](const char* what) {
            once(run_old);
            once(run_new);
            const auto* ca = static_cast<const uint32_t*>(count_a->contents());
            const auto* cb = static_cast<const uint32_t*>(count_b->contents());
            const auto* la = static_cast<const uint32_t*>(list_a->contents());
            const auto* lb = static_cast<const uint32_t*>(list_b->contents());
            bool ok = std::memcmp(ca, cb, T * 4) == 0;
            for (uint32_t j = 0; ok && j < T; ++j)
                ok = std::memcmp(la + uint64_t(j) * kListStride, lb + uint64_t(j) * kListStride, ca[j] * 4) == 0;
            failures += !ok;
            std::println("  {} {} pools, {} tokens, {}: lists {}", ok ? "ok  " : "FAIL", n_pool, T, what,
                         ok ? "the same" : "differ");
        };
        once(run_score);
        once(run_mm);
        {
            // The matrix scores against the others: the largest difference relative to the row's largest score, and
            // the cells the selection then keeps otherwise.
            const auto* a = static_cast<const float*>(score->contents());
            const auto* b = static_cast<const float*>(score_mm->contents());
            double worst = 0;
            for (uint32_t j = 0; j < T; ++j) {
                const uint32_t n = (pos + j) / kPool;
                float mx = 0;
                for (uint32_t p = 0; p < n; ++p)
                    mx = std::max(mx, std::abs(a[uint64_t(j) * stride + p]));
                for (uint32_t p = 0; p < n; ++p)
                    worst = std::max(worst, double(std::abs(a[uint64_t(j) * stride + p] - b[uint64_t(j) * stride + p])) / mx);
            }
            once(run_old);
            std::vector<uint32_t> la(static_cast<const uint32_t*>(list_a->contents()),
                                     static_cast<const uint32_t*>(list_a->contents()) + uint64_t(T) * kListStride);
            std::vector<uint32_t> ca(static_cast<const uint32_t*>(count_a->contents()),
                                     static_cast<const uint32_t*>(count_a->contents()) + T);
            std::swap(score, score_mm);
            once(run_old);
            std::swap(score, score_mm);
            const auto* lb = static_cast<const uint32_t*>(list_a->contents());
            uint64_t differ = 0, total = 0;
            for (uint32_t j = 0; j < T; ++j) {
                std::vector<uint32_t> x(la.begin() + uint64_t(j) * kListStride, la.begin() + uint64_t(j) * kListStride + ca[j]);
                std::vector<uint32_t> y(lb + uint64_t(j) * kListStride, lb + uint64_t(j) * kListStride + ca[j]);
                std::vector<uint32_t> d;
                std::ranges::set_difference(x, y, std::back_inserter(d));
                differ += d.size();
                total += x.size();
            }
            std::println("  matrix scores: largest difference {:.2e} of the row's largest; {} of {} cells kept otherwise",
                         worst, differ, total);
        }
        same("random scores");
        // Few distinct values: the threshold's ties decide.
        {
            auto* f = static_cast<float*>(score->contents());
            for (uint64_t i = 0; i < score->length() / 4; ++i)
                f[i] = std::round(f[i] * 4.0f) / 4.0f;
        }
        same("scores in quarters (ties)");
        once(run_score);
        const double t_score = time(run_score), t_mm = time(run_mm), t_old = time(run_old), t_new = time(run_new);
        std::println("{:7} pools x {} tokens: score {:.3f} ms, matrix score {:.3f} ms, idx_select {:.3f} ms, hy4_sel "
                     "{:.3f} ms ({} slices of {})",
                     n_pool, T, t_score, t_mm, t_old, t_new, slices, slice);
    }
    std::println("{}", failures ? std::format("{} failures", failures) : std::string("all the same"));
    return failures ? 1 : 0;
}
