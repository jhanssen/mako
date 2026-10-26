// DeepSeek-V4.1 engine checks on a short prompt.
//   dsv41_test <model-00001.gguf> <engram.gguf> [ablation.abl]
// Ablation (sites attn_out and shexp_out), with one layer's vectors at one site at a time:
// - attn_out: the layer's attention output loses exactly its component along the vectors, and equals W'·x with
//   W' = W - Q(QᵀW) for the output projection W = attn_output_b on its input x (CPU dequantized), so both the site and
//   the scale convention (strength in the vectors' norms) hold;
// - shexp_out: the FFN output moves by exactly the shared expert output's component along the vectors;
// - an all-zero file and switching ablation off give the same logits as none, bit for bit;
// - with a real file (third argument): prefill and step-by-step decode agree as closely as they do without it.
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"
#include "quant/dequant.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "  ok  " : "  FAIL", what);
    failures += !ok;
}

double rel(const std::vector<double>& a, const std::vector<double>& b)
{
    double e = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e += (a[i] - b[i]) * (a[i] - b[i]);
        r += b[i] * b[i];
    }
    return std::sqrt(e / std::max(r, 1e-300));
}

std::vector<double> to_d(const std::vector<float>& v) { return {v.begin(), v.end()}; }

double kl(const std::vector<float>& p_logits, const std::vector<float>& q_logits)
{
    auto lsm = [](const std::vector<float>& x) {
        double m = -1e300;
        for (float v : x)
            m = std::max(m, double(v));
        double s = 0;
        for (float v : x)
            s += std::exp(double(v) - m);
        std::vector<double> y(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = double(x[i]) - m - std::log(s);
        return y;
    };
    const auto lp = lsm(p_logits), lq = lsm(q_logits);
    double k = 0;
    for (size_t i = 0; i < lp.size(); ++i)
        k += std::exp(lp[i]) * (lp[i] - lq[i]);
    return k;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: dsv41_test <model-00001.gguf> <engram.gguf> [ablation.abl]");
        return 2;
    }
    if (!std::filesystem::exists(argv[1]) || !std::filesystem::exists(argv[2])) {
        std::println("model {} or Engram tables {} not found: skipped", argv[1], argv[2]);
        return 77;
    }
    const std::string abl_path = argc > 3 ? argv[3] : "";
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m || !m->add(argv[2])) {
        std::println(stderr, "cannot open the model files");
        return 1;
    }
    auto cfg = mako::dsv41::Config::load(*m);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::mtl::Context ctx;
    mako::dsv41::Options opts;
    opts.max_ctx = 256;
    mako::dsv41::Engine engine(*m, *cfg, opts, ctx);
    const uint32_t E = cfg->n_embd, NL = cfg->n_layer, V = cfg->n_vocab;
    const std::vector<int32_t> ids = {0, 671, 6102, 294, 8760, 344, 11111, 16, 455, 6102,
                                      294, 10322, 344, 17575, 16, 455, 6102, 294, 14251, 344};
    constexpr uint32_t kLayer = 20; // has vectors at both sites in the files made so far

    // One run step by step: every step's logits, and the probed rows of kLayer at the last step.
    struct Run {
        std::vector<std::vector<float>> logits;
        std::map<std::string, std::vector<float>> at;
    };
    auto run = [&](std::optional<mako::mtl::Ablation> a) {
        engine.reset();
        engine.set_ablation(std::move(a));
        Run r;
        engine.set_probe([&](const std::string& name, int layer, const float* d, size_t n) {
            if (layer == int(kLayer))
                r.at[name].assign(d, d + n);
        });
        for (int32_t t : ids)
            r.logits.push_back(engine.step(t));
        engine.set_probe({});
        return r;
    };

    // Two orthogonal directions with norms like the real file's (~1.25): random, then Gram-Schmidt.
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    const uint32_t K = 2;
    std::vector<std::vector<double>> dirs(K, std::vector<double>(E));
    for (uint32_t j = 0; j < K; ++j) {
        for (auto& x : dirs[j])
            x = nd(rng);
        for (uint32_t i = 0; i < j; ++i) { // the earlier ones are unit vectors here
            double d = 0;
            for (uint32_t e = 0; e < E; ++e)
                d += dirs[j][e] * dirs[i][e];
            for (uint32_t e = 0; e < E; ++e)
                dirs[j][e] -= d * dirs[i][e];
        }
        double n = 0;
        for (double x : dirs[j])
            n += x * x;
        for (auto& x : dirs[j])
            x /= std::sqrt(n);
    }
    for (auto& v : dirs)
        for (auto& x : v)
            x *= 1.25;
    const std::vector<std::string> names{"attn_out", "shexp_out"};
    auto ablation = [&](int site) { // kLayer's vectors at `site` only; -1: all zero
        std::vector<float> v(uint64_t(NL) * names.size() * K * E, 0.0f);
        if (site >= 0)
            for (uint32_t j = 0; j < K; ++j)
                for (uint32_t e = 0; e < E; ++e)
                    v[((uint64_t(kLayer) * names.size() + site) * K + j) * E + e] = float(dirs[j][e]);
        return mako::mtl::Ablation::from_vectors(ctx, E, NL, names, K, v, mako::dsv41::Engine::kAblationSites);
    };
    // y minus its components along the directions, in turn (the kernel's order).
    auto project = [&](std::vector<double> y) {
        for (const auto& v : dirs) {
            double d = 0;
            for (uint32_t e = 0; e < E; ++e)
                d += y[e] * v[e];
            for (uint32_t e = 0; e < E; ++e)
                y[e] -= d * v[e];
        }
        return y;
    };

    Run plain = run(std::nullopt);
    if (!plain.at.contains("attn_out") || !plain.at.contains("attn_low") || !plain.at.contains("shexp_out")) {
        std::println(stderr, "probes missing");
        return 1;
    }

    {
        Run a = run(ablation(0));
        check(a.at["attn_low"] == plain.at["attn_low"], "attn_out: the output projection's input is unchanged");
        const auto want = project(to_d(plain.at["attn_out"]));
        const double e1 = rel(to_d(a.at["attn_out"]), want);
        check(e1 < 1e-5, std::format("attn_out: the layer's attention output loses its component along the vectors "
                                     "(rel err {:.1e})",
                                     e1));
        // W' x on the CPU, W' = W - sum_j v_j (v_jᵀ W), W = attn_output_b dequantized.
        const auto ref = m->find(std::format("blk.{}.attn_output_b.weight", kLayer));
        const auto& t = *ref.tensor;
        const uint32_t k = uint32_t(t.dims[0]), n = uint32_t(t.dims[1]);
        const auto tr = mako::gguf::traits(t.type);
        const uint64_t rb = uint64_t(k) / tr.block_elems * tr.block_bytes;
        const auto data = ref.file->data(t);
        std::vector<float> W(uint64_t(n) * k);
        for (uint32_t r = 0; r < n; ++r)
            mako::quant::dequant_row(t.type, data.data() + uint64_t(r) * rb, W.data() + uint64_t(r) * k, k);
        std::vector<double> vtw(uint64_t(K) * k, 0.0); // v_jᵀ W
        for (uint32_t j = 0; j < K; ++j)
            for (uint32_t r = 0; r < n; ++r)
                for (uint32_t c = 0; c < k; ++c)
                    vtw[uint64_t(j) * k + c] += dirs[j][r] * W[uint64_t(r) * k + c];
        const auto& x = plain.at["attn_low"];
        // The GPU's W x against the CPU's (no ablation): the precision any match can have.
        std::vector<double> w0(n);
        for (uint32_t r = 0; r < n; ++r) {
            double acc = 0;
            for (uint32_t c = 0; c < k; ++c)
                acc += double(W[uint64_t(r) * k + c]) * x[c];
            w0[r] = acc;
        }
        const double e0 = rel(to_d(plain.at["attn_out"]), w0);
        std::vector<double> wx(n);
        for (uint32_t r = 0; r < n; ++r) {
            double acc = 0;
            for (uint32_t c = 0; c < k; ++c) {
                double w = W[uint64_t(r) * k + c];
                for (uint32_t j = 0; j < K; ++j)
                    w -= dirs[j][r] * vtw[uint64_t(j) * k + c];
                acc += w * x[c];
            }
            wx[r] = acc;
        }
        const double e2 = rel(to_d(a.at["attn_out"]), wx);
        check(e2 < 2 * e0 + 1e-6, std::format("attn_out: equals W'x with W' = W - Q(QᵀW) on attn_output_b (rel err "
                                              "{:.1e}; the unablated GPU product vs the CPU's: {:.1e})",
                                              e2, e0));
        double moved = 0, ref_norm = 0;
        for (uint32_t i = 0; i < E; ++i) {
            moved += std::pow(double(a.at["attn_out"][i]) - plain.at["attn_out"][i], 2);
            ref_norm += std::pow(double(plain.at["attn_out"][i]), 2);
        }
        check(moved > 1e-12 * ref_norm, std::format("attn_out: the ablation changed the output ({:.2e} of its norm)",
                                                    std::sqrt(moved / ref_norm)));
    }
    {
        Run a = run(ablation(1));
        check(a.at["shexp_out"] == plain.at["shexp_out"], "shexp_out: the shared expert's output before it is unchanged");
        check(a.at["attn_out"] == plain.at["attn_out"], "shexp_out: the attention output is untouched");
        // ffn_out moves by -(component of the shared expert output along the vectors).
        const auto sh = to_d(plain.at["shexp_out"]);
        const auto sh_proj = project(sh);
        std::vector<double> want(E), got(E);
        for (uint32_t i = 0; i < E; ++i) {
            want[i] = sh_proj[i] - sh[i];
            got[i] = double(a.at["ffn_out"][i]) - plain.at["ffn_out"][i];
        }
        const double e = rel(got, want);
        check(e < 1e-3, std::format("shexp_out: the FFN output moves by the shared expert output's component along "
                                    "the vectors (rel err {:.1e})",
                                    e));
    }
    {
        const Run z = run(ablation(-1));
        check(z.logits == plain.logits, "an all-zero file: the same logits, bit for bit");
        engine.set_ablation(ablation(0));
        const Run off = run(std::nullopt);
        check(off.logits == plain.logits, "switched off: the same logits, bit for bit");
    }

    // Reading experts ahead changes where they are read from, never the results.
    {
        engine.set_ablation(std::nullopt);
        auto steps = [&]() {
            engine.reset();
            std::vector<std::vector<float>> l;
            for (int32_t t : ids)
                l.push_back(engine.step(t));
            return l;
        };
        engine.set_prefetch(0, 2);
        const auto off = steps();
        engine.set_prefetch(4, 2);
        const auto on = steps();
        check(on == off, std::format("reading ahead: the same logits, bit for bit ({} experts read ahead, {} used)",
                                     engine.arena().stats().prefetched, engine.arena().stats().prefetch_used));
    }

    // Speculative verification: rows of a verify() against the same tokens step by step, and keep()'s rollback
    // (continuing after it as if the rejected tokens had never run), with the next position odd and even (a
    // ratio-2 compressor's pair open or closed).
    {
        engine.set_ablation(std::nullopt);
        const uint32_t at = 10; // verify ids[at .. at + width) after a prefill of the ones before
        // The reference: the same prefill, then step by step.
        Run steps;
        steps.logits.resize(ids.size());
        engine.reset();
        engine.prefill(std::span<const int32_t>(ids).first(at));
        for (size_t i = at; i < ids.size(); ++i)
            steps.logits[i] = engine.step(ids[i]);
        for (const uint32_t width : {2u, 4u, 5u}) {
            for (const uint32_t keep : {1u, 2u}) {
                engine.reset();
                engine.prefill(std::span<const int32_t>(ids).first(at));
                std::vector<int32_t> run_ids(ids.begin() + at, ids.begin() + at + width);
                for (uint32_t i = keep; i < width; ++i)
                    run_ids[i] = 9 + int32_t(i); // rejected drafts
                const auto rows = engine.verify(run_ids);
                double kv = 0;
                for (uint32_t i = 0; i < keep; ++i)
                    kv += kl(steps.logits[at + i], std::vector<float>(rows.begin() + uint64_t(i) * V,
                                                                      rows.begin() + uint64_t(i + 1) * V));
                engine.keep(keep);
                double kc = 0;
                for (size_t i = at + keep; i < ids.size(); ++i)
                    kc += kl(steps.logits[i], engine.step(ids[i]));
                kc /= double(ids.size() - at - keep);
                check(engine.position() == ids.size() && kv / keep < 0.02 && kc < 0.02,
                      std::format("verify of {} keeping {} (next position {}): rows vs steps KL {:.1e}, continuing "
                                  "after the rollback KL {:.1e}",
                                  width, keep, at + keep, kv / keep, kc));
            }
        }
    }

    if (!abl_path.empty() && std::filesystem::exists(abl_path)) {
        auto prefill_vs_decode = [&](std::optional<mako::mtl::Ablation> a) {
            const Run dec = run(std::move(a));
            std::vector<std::vector<float>> pre(ids.size());
            engine.reset();
            engine.set_logits_sink([&](uint32_t pos, const float* l) { pre[pos].assign(l, l + V); });
            engine.prefill(ids);
            engine.set_logits_sink({});
            double k = 0;
            for (size_t i = 0; i < ids.size(); ++i)
                k += kl(dec.logits[i], pre[i]);
            return std::pair{k / double(ids.size()), dec};
        };
        const auto [k0, d0] = prefill_vs_decode(std::nullopt);
        auto real = mako::mtl::Ablation::load(abl_path, ctx, E, NL, mako::dsv41::Engine::kAblationSites);
        const auto [k1, d1] = prefill_vs_decode(std::move(real));
        double effect = 0;
        for (size_t i = 0; i < ids.size(); ++i)
            effect += kl(d0.logits[i], d1.logits[i]);
        effect /= double(ids.size());
        check(k1 < 5 * k0 + 1e-3, std::format("prefill and decode agree with {} (KL {:.2e}; without {:.2e}); its "
                                              "effect KL {:.2e}",
                                              std::filesystem::path(abl_path).filename().string(), k1, k0, effect));
        engine.set_ablation(std::nullopt);
    } else {
        std::println("  (no ablation file given: prefill/decode agreement with one not checked)");
    }
    std::println("{} failures", failures);
    return failures ? 1 : 0;
}
