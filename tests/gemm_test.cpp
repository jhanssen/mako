// Prefill GEMM against CPU dequantization on real tensors, for several token counts, plus its speed. For expert
// tensors (one expert's matrix) and formats without CPU dequantization the reference is the per-format GEMV.
//   gemm_test <model shard.gguf> [tensor ...]
#include "gguf/gguf.hpp"
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "mtl/gemv.hpp"
#include "quant/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <print>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: gemm_test <model.gguf> [tensor ...]");
        return 2;
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    std::vector<std::string> names(argv + 2, argv + argc);
    if (names.empty())
        names = {"blk.0.attn_q.weight",       "blk.0.ffn_down.weight",      "blk.0.ssm_f_a.weight",
                 "blk.0.hc_attn_fn.weight",   "blk.3.ffn_gate_inp.weight",  "blk.3.ffn_gate_exps.weight",
                 "blk.3.ffn_down_exps.weight", "blk.10.ffn_down_exps.weight"};

    mako::mtl::Context ctx;
    mako::mtl::Gemm gemm(ctx);
    mako::mtl::Gemv gemv(ctx);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    int failures = 0;
    for (const auto& name : names) {
        const mako::gguf::Tensor* t = f->tensor(name);
        if (!t || t->dims.size() < 2) {
            std::println("{:28} missing or not a matrix", name);
            ++failures;
            continue;
        }
        const uint32_t k = uint32_t(t->dims[0]), n = uint32_t(t->dims[1]);
        const auto tr = mako::gguf::traits(t->type);
        const uint64_t rb = uint64_t(k) / tr.block_elems * tr.block_bytes;
        auto data = f->data(*t).subspan(t->dims.size() > 2 ? 7 * rb * n : 0, rb * n); // expert 7
        const bool cpu_ref = mako::quant::can_dequant(t->type);
        auto buffer = [&](uint64_t bytes) {
            return mako::mtl::Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        };
        auto w = buffer(data.size());
        std::memcpy(w->contents(), data.data(), data.size());

        // Reference rows: a sample spread over the matrix, its last row included.
        std::vector<uint32_t> rows;
        for (uint32_t i = 0; i < 48; ++i)
            rows.push_back(uint32_t(uint64_t(i) * (n - 1) / 47));
        std::vector<std::vector<float>> deq(rows.size(), std::vector<float>(k));
        if (cpu_ref)
            for (size_t i = 0; i < rows.size(); ++i)
                mako::quant::dequant_row(t->type, data.data() + rows[i] * rb, deq[i].data(), k);

        for (bool exact : {false, true})
        for (uint32_t tokens : {1u, 37u, 64u, 200u}) {
            gemm.set_exact_activations(exact);
            auto x = buffer(uint64_t(k) * tokens * 4), y = buffer(uint64_t(n) * tokens * 4);
            auto* xf = static_cast<float*>(x->contents());
            for (uint64_t i = 0; i < uint64_t(k) * tokens; ++i)
                xf[i] = nd(rng);
            auto yref = buffer(uint64_t(n) * tokens * 4);
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            gemm.encode(enc, t->type, k, n, tokens, w.get(), 0, x.get(), 0, y.get(), 0);
            if (!cpu_ref) {
                mako::mtl::Gemv::Args a{t->type, k, n, tokens, 1, false};
                gemv.encode(enc, a, w.get(), 0, x.get(), 0, yref.get(), 0);
            }
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            const auto* yf = static_cast<const float*>(y->contents());
            const auto* rf = static_cast<const float*>(yref->contents());
            double err = 0, ref = 0;
            for (uint32_t tok = 0; tok < tokens; ++tok)
                for (size_t i = 0; i < rows.size(); ++i) {
                    double acc = 0;
                    if (cpu_ref)
                        for (uint32_t j = 0; j < k; ++j)
                            acc += double(deq[i][j]) * xf[uint64_t(tok) * k + j];
                    else
                        acc = rf[uint64_t(tok) * n + rows[i]];
                    const double e = yf[uint64_t(tok) * n + rows[i]] - acc;
                    err += e * e;
                    ref += acc * acc;
                }
            const double rel = std::sqrt(err / ref);
            const bool ok = rel < 5e-3;
            failures += !ok;
            std::println("{:28} {:7} k={:5} n={:6} t={:4} {}  rel err {:.2e} vs {}  {}", name, tr.name, k, n,
                         tokens, exact ? "x32" : "x16", rel, cpu_ref ? "CPU" : "GEMV", ok ? "ok" : "FAIL");
        }

        // Speed at 512 tokens.
        for (bool exact : {false, true}) {
        gemm.set_exact_activations(exact);
        const uint32_t tokens = 512;
        auto x = buffer(uint64_t(k) * tokens * 4), y = buffer(uint64_t(n) * tokens * 4);
        std::fill_n(static_cast<float*>(x->contents()), uint64_t(k) * tokens, 0.01f);
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            gemm.encode(enc, t->type, k, n, tokens, w.get(), 0, x.get(), 0, y.get(), 0);
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime());
        }
        std::println("{:28} t=512 {}: {:8.1f} us, {:6.2f} TFLOP/s", name, exact ? "x32" : "x16", best * 1e6,
                     2.0 * k * n * tokens / best / 1e12);
        }
    }
    return failures ? 1 : 0;
}
