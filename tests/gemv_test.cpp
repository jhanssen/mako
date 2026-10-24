// GPU GEMV against CPU dequantization on real tensors of a model file.
//   gemv_test <model shard.gguf> [--time] [tensor ...]
#include "gguf/gguf.hpp"
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"
#include "quant/dequant.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <print>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: gemv_test <model.gguf> [tensor ...]");
        return 2;
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    std::vector<std::string> names;
    bool timed = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--time")
            timed = true;
        else
            names.emplace_back(argv[i]);
    }
    if (names.empty())
        names = {"blk.0.attn_q.weight",       "blk.0.ssm_f_b.weight",       "blk.0.ssm_f_a.weight",
                 "blk.0.hc_attn_fn.weight",   "blk.3.attn_k_b.weight",      "blk.3.attn_v_b.weight",
                 "blk.0.ffn_down.weight",     "blk.0.ssm_beta.weight",      "blk.0.ssm_g_a.weight",
                 "blk.3.ffn_gate_inp.weight"};

    mako::mtl::Context ctx;
    mako::mtl::Gemv gemv(ctx);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    int failures = 0;
    for (const auto& name : names) {
        const mako::gguf::Tensor* t = f->tensor(name);
        if (!t) {
            std::println("{:28} missing", name);
            ++failures;
            continue;
        }
        const uint32_t k = uint32_t(t->dims[0]), n = uint32_t(t->dims[1]);
        const uint32_t batch = t->dims.size() > 2 ? uint32_t(t->dims[2]) : 1;
        auto data = f->data(*t);
        mako::mtl::Ref<MTL::Buffer> w(ctx.device()->newBuffer(data.size(), MTL::ResourceStorageModeShared));
        std::memcpy(w->contents(), data.data(), data.size());
        mako::mtl::Ref<MTL::Buffer> x(ctx.device()->newBuffer(uint64_t(k) * batch * 4, MTL::ResourceStorageModeShared));
        mako::mtl::Ref<MTL::Buffer> y(ctx.device()->newBuffer(uint64_t(n) * batch * 4, MTL::ResourceStorageModeShared));
        auto* xf = static_cast<float*>(x->contents());
        for (uint64_t i = 0; i < uint64_t(k) * batch; ++i)
            xf[i] = nd(rng);

        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        mako::mtl::Gemv::Args a{t->type, k, n, 1, batch, false};
        gemv.encode(enc, a, w.get(), 0, x.get(), 0, y.get(), 0);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();

        const uint64_t rb = mako::mtl::Gemv::row_bytes(t->type, k);
        std::vector<float> row(k);
        double err = 0, ref = 0;
        const auto* yf = static_cast<const float*>(y->contents());
        for (uint32_t b = 0; b < batch; ++b)
            for (uint32_t r = 0; r < n; ++r) {
                mako::quant::dequant_row(t->type, data.data() + (uint64_t(b) * n + r) * rb, row.data(), k);
                double acc = 0;
                for (uint32_t i = 0; i < k; ++i)
                    acc += double(row[i]) * xf[uint64_t(b) * k + i];
                const double e = yf[uint64_t(b) * n + r] - acc;
                err += e * e;
                ref += acc * acc;
            }
        const double rel = std::sqrt(err / ref);
        const bool ok = rel < 1e-4;
        failures += !ok;
        std::string timing;
        if (timed) {
            // GPU time per dispatch over a chain of dependent dispatches, as in a decode step, cycling through
            // copies of the matrix that together exceed the GPU's caches (decode reads every weight from DRAM).
            const uint64_t copies = std::max<uint64_t>(2, (256ull << 20) / data.size());
            mako::mtl::Ref<MTL::Buffer> wc(ctx.device()->newBuffer(data.size() * copies, MTL::ResourceStorageModeShared));
            for (uint64_t c = 0; c < copies; ++c)
                std::memcpy(static_cast<char*>(wc->contents()) + c * data.size(), data.data(), data.size());
            for (bool own : {false, true}) {
                gemv.set_own_kernels(own);
                double best = 1e9;
                for (int rep = 0; rep < 3; ++rep) {
                    auto* tcb = ctx.queue()->commandBuffer();
                    auto* tenc = tcb->computeCommandEncoder();
                    for (int r = 0; r < 200; ++r)
                        gemv.encode(tenc, a, wc.get(), (r % copies) * data.size(), x.get(), 0, y.get(), 0);
                    tenc->endEncoding();
                    tcb->commit();
                    tcb->waitUntilCompleted();
                    best = std::min(best, (tcb->GPUEndTime() - tcb->GPUStartTime()) * 1e6 / 200);
                }
                timing += std::format("  {} {:6.1f} us", own ? "own" : "per-format", best);
            }
            gemv.set_own_kernels(true);
        }
        std::println("{:28} {:5} k={:5} n={:6} batch={:3}  rel err {:.2e}  {}{}", name,
                     mako::gguf::traits(t->type).name, k, n, batch, rel, ok ? "ok" : "FAIL", timing);
    }

    // Indexed GEMV over expert slots (padded stride, ids not in slot order) against the batched GEMV over the same
    // experts copied contiguously: checks the slot addressing (expert types have no CPU dequantization).
    if (names.size() == 10) {
        const int experts[] = {5, 287, 0, 100, 42, 7, 200, 13};
        const int32_t slot_ids[] = {3, 0, 7, 1, 6, 2, 5, 4};
        constexpr int n_sel = 8;
        for (auto [name, shared] : {std::pair{"blk.3.ffn_gate_exps.weight", true}, {"blk.3.ffn_down_exps.weight", false}}) {
            const mako::gguf::Tensor* t = f->tensor(name);
            if (!t) {
                std::println("{:28} missing", name);
                ++failures;
                continue;
            }
            const uint32_t k = uint32_t(t->dims[0]), n = uint32_t(t->dims[1]);
            const uint64_t eb = mako::mtl::Gemv::row_bytes(t->type, k) * n;
            const uint64_t stride = (eb + 16383) / 16384 * 16384;
            auto data = f->data(*t);
            auto buffer = [&](uint64_t bytes) {
                return mako::mtl::Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
            };
            auto slots = buffer(stride * n_sel), packed = buffer(eb * n_sel), ids = buffer(sizeof(slot_ids));
            for (int i = 0; i < n_sel; ++i) {
                std::memcpy(static_cast<char*>(slots->contents()) + slot_ids[i] * stride, data.data() + experts[i] * eb, eb);
                std::memcpy(static_cast<char*>(packed->contents()) + i * eb, data.data() + experts[i] * eb, eb);
            }
            std::memcpy(ids->contents(), slot_ids, sizeof(slot_ids));
            const uint32_t nx = shared ? 1 : n_sel;
            auto x = buffer(uint64_t(k) * nx * 4), y = buffer(uint64_t(n) * n_sel * 4), yref = buffer(uint64_t(n) * n_sel * 4);
            auto* xf = static_cast<float*>(x->contents());
            for (uint64_t i = 0; i < uint64_t(k) * nx; ++i)
                xf[i] = nd(rng);

            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            mako::mtl::Gemv::Args a{t->type, k, n, 1, uint32_t(n_sel), shared};
            gemv.encode_indexed(enc, a, slots.get(), 0, stride, ids.get(), 0, x.get(), 0, y.get(), 0);
            gemv.encode(enc, a, packed.get(), 0, x.get(), 0, yref.get(), 0);
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();

            const auto* yf = static_cast<const float*>(y->contents());
            const auto* rf = static_cast<const float*>(yref->contents());
            double err = 0, ref = 0;
            for (uint64_t i = 0; i < uint64_t(n) * n_sel; ++i) {
                err += double(yf[i] - rf[i]) * (yf[i] - rf[i]);
                ref += double(rf[i]) * rf[i];
            }
            // Slots from `split` on in a second buffer (the arena's lendable slots); the main buffer's copies of
            // those are zeroed so reading the wrong one shows.
            constexpr uint32_t split = 5;
            auto main = buffer(stride * n_sel), tail = buffer(stride * (n_sel - split) + 16384);
            std::memcpy(main->contents(), slots->contents(), stride * split);
            std::memset(static_cast<char*>(main->contents()) + stride * split, 0, stride * (n_sel - split));
            std::memcpy(static_cast<char*>(tail->contents()) + 16384, static_cast<char*>(slots->contents()) + stride * split,
                        stride * (n_sel - split));
            auto y2 = buffer(uint64_t(n) * n_sel * 4);
            cb = ctx.queue()->commandBuffer();
            enc = cb->computeCommandEncoder();
            gemv.encode_indexed(enc, a, main.get(), 0, stride, ids.get(), 0, x.get(), 0, y2.get(), 0, nullptr, 0,
                                {tail.get(), 16384, split});
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            const auto* y2f = static_cast<const float*>(y2->contents());
            double err2 = 0;
            for (uint64_t i = 0; i < uint64_t(n) * n_sel; ++i)
                err2 += double(y2f[i] - rf[i]) * (y2f[i] - rf[i]);
            const double rel = std::sqrt(std::max(err, err2) / ref);
            const bool ok = rel < 1e-6;
            failures += !ok;
            std::string timing;
            if (timed) {
                const uint64_t bytes = stride * n_sel, copies = std::max<uint64_t>(2, (256ull << 20) / bytes);
                auto many = buffer(bytes * copies);
                for (uint64_t c = 0; c < copies; ++c)
                    std::memcpy(static_cast<char*>(many->contents()) + c * bytes, slots->contents(), bytes);
                for (bool own : {false, true}) {
                    gemv.set_own_kernels(own);
                    double best = 1e9;
                    for (int rep = 0; rep < 3; ++rep) {
                        auto* tcb = ctx.queue()->commandBuffer();
                        auto* tenc = tcb->computeCommandEncoder();
                        for (int r = 0; r < 100; ++r)
                            gemv.encode_indexed(tenc, a, many.get(), (r % copies) * bytes, stride, ids.get(), 0, x.get(),
                                                0, y.get(), 0);
                        tenc->endEncoding();
                        tcb->commit();
                        tcb->waitUntilCompleted();
                        best = std::min(best, (tcb->GPUEndTime() - tcb->GPUStartTime()) * 1e6 / 100);
                    }
                    timing += std::format("  {} {:6.1f} us ({:.0f} GB/s)", own ? "own" : "per-format", best,
                                          double(eb) * n_sel / best / 1e3);
                }
                gemv.set_own_kernels(true);
            }
            std::println("{:28} {:7} k={:5} n={:6} indexed x{} {:8}  rel err vs batched {:.2e}  {}{}", name,
                         mako::gguf::traits(t->type).name, k, n, n_sel, shared ? "shared x" : "own x", rel,
                         ok ? "ok" : "FAIL", timing);
        }
    }
    return failures ? 1 : 0;
}
