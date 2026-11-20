// Hy4-flash ablation (sites stream0..3 and head0..3) on a short prompt.
//   hy4f_ablation_test <model-00001.gguf> [ablation.abl]
// - Random directions, different for each stream, at one layer and at the head: each stream entering the layer
//   (after its Engram) and before the head is the CPU's projection of the unablated stream, with its own directions;
// - an all-zero file and switching ablation off give the same logits as none, bit for bit;
// - with a real file: prefill and step-by-step decode agree as closely as they do without it.
#include "hy4f/config.hpp"
#include "hy4f/engine.hpp"

#include <cmath>
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
    if (argc < 2) {
        std::println(stderr, "usage: hy4f_ablation_test <model-00001.gguf> [ablation.abl]");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    const std::string abl_path = argc > 2 ? argv[2] : "";
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m) {
        std::println(stderr, "{}", m.error());
        return 1;
    }
    auto cfg = mako::hy4f::Config::load(*m);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::mtl::Context ctx;
    mako::hy4f::Options opts;
    opts.max_ctx = 512;
    mako::hy4f::Engine engine(*m, *cfg, opts, ctx);
    const uint32_t E = cfg->n_embd, NL = cfg->n_layer, HC = cfg->hc, V = cfg->n_vocab;
    const uint32_t S = uint32_t(mako::hy4f::Engine::kAblationSites.size());
    // tests/data/hy4f_short_ids.txt (text; random ids make prefill and decode disagree far more).
    const std::vector<int32_t> ids = {802, 5466, 19405, 63357, 52392, 1278, 269, 34262,
                                      8988, 13,   499,   8778,  299,   12749, 341};
    constexpr uint32_t kLayer = 20;

    // Every step's logits, step by step.
    struct Run {
        std::vector<std::vector<float>> logits;
    };
    auto run = [&](std::optional<mako::mtl::Ablation> a) {
        engine.reset();
        engine.set_ablation(std::move(a));
        Run r;
        for (int32_t t : ids)
            r.logits.push_back(engine.step(t));
        return r;
    };

    // Two orthonormal directions for each stream at kLayer (stream sites) and for the head (head sites, at the last
    // layer's index).
    const uint32_t K = 2;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    auto orthonormal = [&] {
        std::vector<std::vector<double>> d(K, std::vector<double>(E));
        for (uint32_t j = 0; j < K; ++j) {
            for (auto& x : d[j])
                x = nd(rng);
            for (uint32_t i = 0; i < j; ++i) {
                double p = 0;
                for (uint32_t e = 0; e < E; ++e)
                    p += d[j][e] * d[i][e];
                for (uint32_t e = 0; e < E; ++e)
                    d[j][e] -= p * d[i][e];
            }
            double n = 0;
            for (double x : d[j])
                n += x * x;
            for (auto& x : d[j])
                x /= std::sqrt(n);
        }
        return d;
    };
    std::vector<std::vector<std::vector<double>>> dirs(S); // [site][k][E]
    for (auto& d : dirs)
        d = orthonormal();
    const std::vector<std::string> names(mako::hy4f::Engine::kAblationSites.begin(),
                                         mako::hy4f::Engine::kAblationSites.end());
    auto ablation = [&](bool zero) {
        std::vector<float> v(uint64_t(NL) * S * K * E, 0.0f);
        if (!zero)
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t l = s < 4 ? kLayer : NL - 1;
                for (uint32_t k = 0; k < K; ++k)
                    for (uint32_t e = 0; e < E; ++e)
                        v[((uint64_t(l) * S + s) * K + k) * E + e] = float(dirs[s][k][e]);
            }
        return mako::mtl::Ablation::from_vectors(ctx, E, NL, names, K, v, mako::hy4f::Engine::kAblationSites);
    };

    const Run plain = run(std::nullopt);
    {
        // The first step's streams, unablated and ablated (later steps differ in what came before).
        auto first_step = [&](std::optional<mako::mtl::Ablation> a) {
            engine.reset();
            engine.set_ablation(std::move(a));
            std::map<std::pair<std::string, int>, std::vector<float>> at;
            engine.set_probe([&](const std::string& name, int layer, const float* d, size_t n) {
                at[{name, layer}].assign(d, d + n);
            });
            engine.step(ids[0]);
            engine.set_probe({});
            return at;
        };
        const auto p1 = first_step(std::nullopt), a1 = first_step(ablation(false));
        // in's streams projected on the CPU with the sites' directions from site0 on, against got: the relative
        // error and how far the projection moved them.
        auto compare = [&](const std::vector<float>& in, const std::vector<float>& got, uint32_t site0) {
            double e = 0, r = 0, moved = 0;
            for (uint32_t s = 0; s < HC; ++s) {
                std::vector<double> x(in.begin() + uint64_t(s) * E, in.begin() + uint64_t(s + 1) * E);
                const std::vector<double> x0 = x;
                for (const auto& d : dirs[site0 + s]) {
                    double p = 0;
                    for (uint32_t i = 0; i < E; ++i)
                        p += x[i] * d[i];
                    for (uint32_t i = 0; i < E; ++i)
                        x[i] -= p * d[i];
                }
                for (uint32_t i = 0; i < E; ++i) {
                    const double g = got[uint64_t(s) * E + i];
                    e += (g - x[i]) * (g - x[i]);
                    r += x[i] * x[i];
                    moved += (x0[i] - x[i]) * (x0[i] - x[i]);
                }
            }
            return std::pair{std::sqrt(e / r), std::sqrt(moved / r)};
        };
        // kLayer's input: its Engram's output when it has one, else the previous layer's.
        const auto eg = p1.find({"engram_out", int(kLayer)});
        const auto& in = eg != p1.end() ? eg->second : p1.at({"l_out", int(kLayer) - 1});
        const auto [e_s, m_s] = compare(in, a1.at({"stream_in", int(kLayer)}), 0);
        check(e_s < 1e-5 && m_s > 1e-4,
              std::format("stream0..3: each stream entering layer {} loses its own directions (rel err {:.1e}, "
                          "moved {:.1e} of its norm)",
                          kLayer, e_s, m_s));
        // The head's input follows kLayer's ablation: against the ablated run's last layer.
        const auto [e_h, m_h] = compare(a1.at({"l_out", int(NL) - 1}), a1.at({"head_in", -1}), 4);
        check(e_h < 1e-5 && m_h > 1e-4,
              std::format("head0..3: each stream before the head loses its own directions (rel err {:.1e}, moved "
                          "{:.1e} of its norm)",
                          e_h, m_h));
        const Run a = run(ablation(false));
        double effect = 0;
        for (size_t i = 0; i < ids.size(); ++i)
            effect += kl(plain.logits[i], a.logits[i]);
        check(effect > 0, std::format("random directions change the logits (KL {:.2e} a step)", effect / ids.size()));
    }
    {
        const Run z = run(ablation(true));
        check(z.logits == plain.logits, "an all-zero file: the same logits, bit for bit");
        engine.set_ablation(ablation(false));
        const Run off = run(std::nullopt);
        check(off.logits == plain.logits, "switched off: the same logits, bit for bit");
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
        auto real = mako::mtl::Ablation::load(abl_path, ctx, E, NL, mako::hy4f::Engine::kAblationSites);
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
