// Runs token ids through the DeepSeek-V4.1 engine one step at a time and compares each position's logits with a
// reference's (float32 [positions][vocab]).
//
//   ds_run <model-00001.gguf> <engram.gguf> --ids I,J,... [--ref logits.f32] [--logits-out FILE] [--arena-gb N]
//          [--stage-gb N] [--max-ctx N] [--ablate FILE] [--top-k N] [--prefill 1] [--chunk N] [--group N] [--telemetry layers|parts|kernels] [--profile F]
//          [--save-profile F|default] [--split S --width W] [--dspark DIR [--gen N] [--spec 0|1] [--max-drafts N]
//          [--draft-conf F]] [--dense-quant q4_k|q5_k|q6_k] [--miss-penalty F] [--prefetch K] [--prefill-skip 0|1] [-n N]
// --tail N: the ids but the last N through prefill(), then the last N timed through prefill() and, from the same
// checkpoint, as decode steps of up to 8 tokens; the misses and expert reads of each.
// --ref-from N: the reference's first position is N (a tail of a long prompt's logits). --steps-from N: the first N ids
// through prefill(), the rest one step each. Comparisons also report the perplexity of the ids' own next tokens.
// --imatrix-out FILE (with --gen N): the mean square of each input channel of the dense matrices' inputs over the
// generated steps, for --imatrix (activation-weighted rounding in --dense-quant).
// --route-study N: N greedy tokens without drafts, then how well each expert-miss predictor would have done: layer
// l+1's router on layer l's FFN input (one and two layers ahead), the previous token's experts, and the experts the
// same input token used earlier (misses caught, and reads of experts neither resident nor selected wasted).
// --gen N: N greedy tokens through a generation session after the ids (with the drafter when --dspark gives one and
// --spec is not 0), with the tokens, speed and drafts kept.
// --split S --width W: prefills the first S ids, then runs the rest W at a time (as a speculative verify step whose
// drafts were all right would) and reports the time and expert misses per step and per token.
// --prefill 1: the ids through prefill() (every position's logits from its sink) instead of one step each.
// -n N: then N greedy tokens.
#include "app/presets.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"
#include "gen/session.hpp"

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <map>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> log_softmax(const float* x, size_t n)
{
    const float m = *std::max_element(x, x + n);
    double s = 0;
    for (size_t i = 0; i < n; ++i)
        s += std::exp(double(x[i] - m));
    const float lse = m + float(std::log(s));
    std::vector<float> y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = x[i] - lse;
    return y;
}

} // namespace

int main(int argc, char** argv)
{
    mako::app::Args expanded = [&] {
        try {
            return mako::app::Args(argc, argv);
        } catch (const std::exception& e) {
            std::println(stderr, "{}", e.what());
            std::exit(2);
        }
    }();
    argc = expanded.argc();
    argv = expanded.argv();
    if (argc < 4) {
        std::println(stderr, "usage: ds_run <model-00001.gguf> <engram.gguf> --ids I,J,... [--ref F] [--logits-out F] "
                             "[--arena-gb N] [--max-ctx N] [--ablate F] [-n N]");
        return 2;
    }
    std::vector<int32_t> ids;
    std::string ref, out, ablate, save_profile;
    mako::dsv41::Options opts;
    int gen = 0;
    bool use_prefill = false;
    uint32_t split = 0, width = 0, session_n = 0, study_n = 0;
    bool part_timing = false;
    size_t ref_from = 0, steps_from = 0, tail = 0;
    std::string imatrix_out;
    uint64_t step_misses = 0;
    bool spec = true;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--ids") {
            std::stringstream ss(v);
            for (std::string s; std::getline(ss, s, ',');)
                ids.push_back(std::stoi(s));
        } else if (a == "--ref")
            ref = v;
        else if (a == "--logits-out")
            out = v;
        else if (a == "--arena-gb")
            opts.arena_gb = std::stod(v);
        else if (a == "--stage-gb")
            opts.stage_gb = std::stod(v);
        else if (a == "--max-ctx")
            opts.max_ctx = uint32_t(std::stoul(v));
        else if (a == "--prefill")
            use_prefill = v != "0";
        else if (a == "--chunk")
            opts.prefill_chunk = uint32_t(std::stoul(v));
        else if (a == "--group")
            opts.prefill_group = uint32_t(std::stoul(v));
        else if (a == "--telemetry") {
            opts.telemetry = v == "summary" ? mako::core::Telemetry::summary
                             : v == "kernels" ? mako::core::Telemetry::kernels
                                              : mako::core::Telemetry::layers;
            part_timing = v == "parts";
        }
        else if (a == "--save-profile")
            save_profile = v;
        else if (a == "--profile")
            opts.expert_counts = v;
        else if (a == "--prefetch")
            opts.prefetch_k = uint32_t(std::stoul(v));
        else if (a == "--prefetch-ahead")
            opts.prefetch_ahead = uint32_t(std::stoul(v));
        else if (a == "--prefill-skip")
            opts.prefill_skip = v != "0";
        else if (a == "--miss-penalty")
            opts.miss_penalty = std::stof(v);
        else if (a == "--dense-quant") {
            if (v == "q4_k")
                opts.dense_quant = mako::gguf::Type::Q4_K;
            else if (v == "q5_k")
                opts.dense_quant = mako::gguf::Type::Q5_K;
            else if (v == "q6_k")
                opts.dense_quant = mako::gguf::Type::Q6_K;
            else {
                std::println(stderr, "--dense-quant takes q4_k, q5_k or q6_k");
                return 2;
            }
        } else if (a == "--dspark")
            opts.dspark = v;
        else if (a == "--gen")
            session_n = uint32_t(std::stoul(v));
        else if (a == "--tail")
            tail = std::stoul(v);
        else if (a == "--ref-from")
            ref_from = std::stoul(v);
        else if (a == "--steps-from")
            steps_from = std::stoul(v);
        else if (a == "--imatrix-out")
            imatrix_out = v;
        else if (a == "--imatrix")
            opts.dense_imatrix = v;
        else if (a == "--route-study")
            study_n = uint32_t(std::stoul(v));
        else if (a == "--spec")
            spec = v != "0";
        else if (a == "--max-drafts")
            opts.max_drafts = uint32_t(std::stoul(v));
        else if (a == "--draft-conf")
            opts.draft_conf = std::stof(v);
        else if (a == "--split")
            split = uint32_t(std::stoul(v));
        else if (a == "--width")
            width = uint32_t(std::stoul(v));
        else if (a == "--top-k")
            opts.indexer_top_k = uint32_t(std::stoul(v));
        else if (a == "--ablate")
            ablate = v;
        else if (a == "-n")
            gen = std::stoi(v);
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m) {
        std::println(stderr, "{}", m.error());
        return 1;
    }
    if (auto r = m->add(argv[2]); !r) {
        std::println(stderr, "{}", r.error());
        return 1;
    }
    auto cfg = mako::dsv41::Config::load(*m);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::mtl::Context ctx;
    const auto t0 = std::chrono::steady_clock::now();
    // Every position from --ref-from on is compared (and scored), so the prefill computes those exactly.
    if (ids.size() > ref_from)
        opts.prefill_exact_tail = std::max<uint32_t>(opts.prefill_exact_tail, uint32_t(ids.size() - ref_from));
    mako::dsv41::Engine engine(*m, *cfg, opts, ctx);
    engine.set_part_timing(part_timing);
    std::println(stderr, "loaded in {:.1f} s: arena {:.1f} GB, {}-{} slots a layer, profile {}",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), engine.arena().bytes() / 1e9,
                 engine.load_stats().slots_min, engine.load_stats().slots_max,
                 engine.loaded_profile().empty() ? "none" : engine.loaded_profile().string());
    if (!ablate.empty()) {
        auto a = mako::mtl::Ablation::load(ablate, ctx, cfg->n_embd, cfg->n_layer, mako::dsv41::Engine::kAblationSites);
        engine.set_ablation(std::move(a));
    }
    const size_t V = cfg->n_vocab;
    std::vector<float> want;
    if (!ref.empty()) {
        std::ifstream in(ref, std::ios::binary);
        in.seekg(0, std::ios::end);
        want.resize(size_t(in.tellg()) / 4);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(want.data()), std::streamsize(want.size() * 4));
    }
    std::ofstream lo;
    if (!out.empty())
        lo.open(out, std::ios::binary);
    double kl_sum = 0, nll_sum = 0, ref_nll_sum = 0;
    int agree = 0, compared = 0, scored = 0;
    auto check = [&](size_t t, const float* lg, double ms) {
        if (lo)
            lo.write(reinterpret_cast<const char*>(lg), std::streamsize(V * 4));
        const size_t top = size_t(std::max_element(lg, lg + V) - lg);
        if (t + 1 < ids.size() && t >= ref_from) {
            const auto lq = log_softmax(lg, V);
            nll_sum -= double(lq[size_t(ids[t + 1])]);
            ++scored;
        }
        if (t >= ref_from && (t - ref_from + 1) * V <= want.size()) {
            const float* w = want.data() + (t - ref_from) * V;
            const auto lp = log_softmax(w, V), lq = log_softmax(lg, V);
            double kl = 0;
            for (size_t i = 0; i < V; ++i)
                kl += std::exp(double(lp[i])) * (double(lp[i]) - double(lq[i]));
            const size_t wtop = size_t(std::max_element(w, w + V) - w);
            kl_sum += kl;
            agree += wtop == top;
            if (t + 1 < ids.size())
                ref_nll_sum -= double(lp[size_t(ids[t + 1])]);
            ++compared;
            std::println("pos {:4} token {:6}: top {:6} (ref {:6}) KL {:.5f}  {:.1f} ms", t, ids[t], top, wtop, kl, ms);
        } else {
            std::println("pos {:4} token {:6}: top {:6} ({:.3f})  {:.1f} ms", t, ids[t], top, lg[top], ms);
        }
    };
    if (std::getenv("DS_TIME_DRAFT") && engine.has_drafter()) {
        engine.prefill(ids);
        const int32_t y = int32_t(std::max_element(engine.logits().begin(), engine.logits().end()) - engine.logits().begin());
        for (int i = 0; i < 3; ++i)
            engine.draft(y);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i)
            engine.draft(y);
        std::println("draft(): {:.2f} ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20);
        const std::vector<int32_t> v2{y, 1}, v4{y, 1, 2, 3}, v5{y, 1, 2, 3, 4};
        for (const auto* v : {&v2, &v4, &v5}) {
            const uint32_t p0 = engine.position();
            const auto s0 = std::chrono::steady_clock::now();
            engine.verify(*v);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            engine.keep(1);
            std::println("verify of {}: {:.1f} ms (position {} -> {})", v->size(), ms, p0, engine.position());
            engine.reset();
            engine.prefill(ids);
        }
        return 0;
    }
    if (tail > 0 && tail < ids.size()) {
        const std::span<const int32_t> all(ids);
        engine.prefill(all.first(ids.size() - tail));
        const auto cp = engine.checkpoint();
        const auto rest = all.last(tail);
        // DS_TAIL_ORDER=steps: the steps first (each method's first run then starts from the same cache).
        const bool steps_first = std::getenv("DS_TAIL_ORDER") && std::string_view(std::getenv("DS_TAIL_ORDER")) == "steps";
        if (steps_first) {
            engine.restore(cp);
            const auto s0 = engine.arena().stats();
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < rest.size(); i += 8)
                engine.prefill(rest.subspan(i, std::min<size_t>(8, rest.size() - i)));
            const auto s1 = engine.arena().stats();
            std::println("{} tokens after {}: steps of 8 first {:.0f} ms ({} misses, {:.0f} ms reading)", tail,
                         ids.size() - tail,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                         s1.misses - s0.misses, s1.read_ms - s0.read_ms);
            return 0;
        }
        for (int rep = 0; rep < 2; ++rep) {
            engine.restore(cp);
            auto st0 = engine.arena().stats();
            auto t0 = std::chrono::steady_clock::now();
            engine.prefill(rest);
            const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto st1 = engine.arena().stats();
            engine.restore(cp);
            t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < rest.size(); i += 8)
                engine.prefill(rest.subspan(i, std::min<size_t>(8, rest.size() - i)));
            const double dms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto st2 = engine.arena().stats();
            std::println("{} tokens after {}: prefill {:.0f} ms ({} misses, {:.0f} ms reading), steps of 8 {:.0f} ms ({} misses, "
                         "{:.0f} ms reading)",
                         tail, ids.size() - tail, pms, st1.misses - st0.misses, st1.read_ms - st0.read_ms, dms,
                         st2.misses - st1.misses, st2.read_ms - st1.read_ms);
        }
        return 0;
    }
    if (study_n > 0) {
        const uint32_t NL = cfg->n_layer, NE = cfg->n_expert, K = cfg->n_expert_used, E = cfg->n_embd;
        struct LayerRec {
            std::vector<float> x;      // FFN input
            std::vector<int> sel;      // selected experts
            std::vector<char> resident; // before this layer's misses were read
        };
        std::vector<std::vector<LayerRec>> steps; // [step][layer]
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (layer < 0 || uint32_t(layer) >= NL)
                return;
            if (name == "moe_sel") {
                if (layer == 0)
                    steps.emplace_back(NL);
                steps.back()[layer].sel.assign(data, data + n);
            } else if (name == "moe_resident") {
                steps.back()[layer].resident.assign(n, 0);
                for (size_t e = 0; e < n; ++e)
                    steps.back()[layer].resident[e] = data[e] > 0.5f;
            } else if (name == "ffn_in" && !steps.empty()) {
                steps.back()[layer].x.assign(data, data + n);
            }
        });
        mako::gen::Session session(engine);
        mako::gen::GenerateParams gp;
        gp.sampling.temperature = 0;
        gp.max_tokens = study_n;
        gp.speculative = false;
        const auto r = session.generate(ids, gp);
        engine.set_probe({});
        std::vector<int32_t> inputs{ids.back()};
        inputs.insert(inputs.end(), r.tokens.begin(), r.tokens.end());
        // Router weights and biases, read from the model file.
        std::vector<std::vector<float>> W(NL), B(NL);
        for (uint32_t l = 0; l < NL; ++l) {
            for (auto [name, dst] : {std::pair{std::format("blk.{}.ffn_gate_inp.weight", l), &W[l]},
                                     std::pair{std::format("blk.{}.exp_probs_b.bias", l), &B[l]}}) {
                const auto ref = m->find(name);
                if (!ref || ref.tensor->type != mako::gguf::Type::F32)
                    throw std::runtime_error("need F32 " + name);
                const auto d = ref.file->data(*ref.tensor);
                dst->resize(d.size() / 4);
                std::memcpy(dst->data(), d.data(), d.size());
            }
        }
        auto predict = [&](uint32_t l, const std::vector<float>& x, uint32_t topk) {
            std::vector<std::pair<float, int>> sc(NE);
            for (uint32_t e = 0; e < NE; ++e) {
                const float* w = W[l].data() + uint64_t(e) * E;
                double dot = 0;
                for (uint32_t i = 0; i < E; ++i)
                    dot += double(w[i]) * x[i];
                const double sp = dot > 20 ? dot : std::log1p(std::exp(dot));
                sc[e] = {float(std::sqrt(sp) + B[l][e]), int(e)};
            }
            std::partial_sort(sc.begin(), sc.begin() + topk, sc.end(), std::greater{});
            std::vector<int> out;
            for (uint32_t i = 0; i < topk; ++i)
                out.push_back(sc[i].second);
            return out;
        };
        struct Score {
            double caught = 0, wasted = 0;
        };
        std::map<std::string, Score> scores;
        double misses = 0;
        uint64_t counted = 0;
        std::map<std::pair<int32_t, uint32_t>, std::map<int, int>> by_token; // (input token, layer) -> expert counts
        for (size_t s = 0; s < steps.size(); ++s) {
            for (uint32_t l = 0; l < NL; ++l) {
                const LayerRec& rec = steps[s][l];
                if (rec.sel.size() != K || rec.resident.size() != NE)
                    continue;
                auto is_sel = [&](int e) { return std::ranges::find(rec.sel, e) != rec.sel.end(); };
                std::vector<int> miss;
                for (int e : rec.sel)
                    if (!rec.resident[e])
                        miss.push_back(e);
                auto score = [&](const std::string& name, const std::vector<int>& pred) {
                    Score& sc = scores[name];
                    for (int e : pred) {
                        if (std::ranges::find(miss, e) != miss.end())
                            sc.caught += 1;
                        else if (!rec.resident[e] && !is_sel(e))
                            sc.wasted += 1;
                    }
                };
                std::vector<int> a1, a2;
                if (l >= 1 && steps[s][l - 1].x.size() == E) {
                    a1 = predict(l, steps[s][l - 1].x, 8);
                    for (uint32_t k : {2u, 3u, 4u, 6u, 8u})
                        score(std::format("router 1 layer ahead, top {}", k), {a1.begin(), a1.begin() + k});
                }
                if (l >= 2 && steps[s][l - 2].x.size() == E) {
                    a2 = predict(l, steps[s][l - 2].x, 8);
                    for (uint32_t k : {2u, 4u, 6u})
                        score(std::format("router 2 layers ahead, top {}", k), {a2.begin(), a2.begin() + k});
                }
                if (!a1.empty() && !a2.empty()) {
                    std::vector<int> both;
                    for (int e : a2)
                        if (std::ranges::find(a1, e) != a1.end())
                            both.push_back(e);
                    score("2-ahead top 8 also in 1-ahead top 8", both);
                }
                if (s >= 1 && steps[s - 1][l].sel.size() == K)
                    score("previous token's experts", steps[s - 1][l].sel);
                const int32_t tok = s < inputs.size() ? inputs[s] : -1;
                auto& hist = by_token[{tok, l}];
                if (!hist.empty()) {
                    std::vector<std::pair<int, int>> v;
                    for (auto [e, c] : hist)
                        v.push_back({c, e});
                    std::ranges::sort(v, std::greater{});
                    std::vector<int> pred;
                    for (size_t i = 0; i < v.size() && i < 8; ++i)
                        pred.push_back(v[i].second);
                    score("same input token earlier, up to 8", pred);
                }
                for (int e : rec.sel)
                    ++hist[e];
                misses += double(miss.size());
                ++counted;
            }
        }
        const double n_steps = double(steps.size());
        std::println("{} steps, {:.1f} misses a step (layers with records: {})", steps.size(), misses / n_steps, counted);
        for (const auto& [name, sc] : scores)
            std::println("  {:38} catches {:5.1f}% of misses ({:.2f} a step), wastes {:.2f} reads a step", name,
                         misses > 0 ? 100.0 * sc.caught / misses : 0.0, sc.caught / n_steps, sc.wasted / n_steps);
        return 0;
    }
    // name -> layer -> (sum of squares per channel, steps)
    std::map<std::string, std::map<int, std::pair<std::vector<double>, uint64_t>>> imx;
    if (!imatrix_out.empty())
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (!name.starts_with("imx_") && name != "ffn_in")
                return;
            auto& [sum, count] = imx[name == "ffn_in" ? "imx_ffn" : name][layer];
            sum.resize(n);
            for (size_t i = 0; i < n; ++i)
                sum[i] += double(data[i]) * data[i];
            ++count;
        });
    if (session_n > 0) {
        mako::gen::Session session(engine);
        mako::gen::GenerateParams gp;
        gp.sampling.temperature = 0;
        gp.max_tokens = session_n;
        gp.speculative = spec;
        const auto r = session.generate(ids, gp);
        std::print("generated:");
        for (int32_t t : r.tokens)
            std::print(" {}", t);
        std::println("\n{} tokens in {:.0f} ms ({:.2f} tok/s); prompt {:.0f} ms; drafts {} kept {} ({:.0f}%)", r.tokens.size(),
                     r.decode_ms, r.tokens.size() / r.decode_ms * 1e3, r.prefill_ms, r.drafts, r.accepted,
                     r.drafts ? 100.0 * r.accepted / r.drafts : 0.0);
        const auto& ss = engine.spec_stats();
        std::println("drafts {} ({:.1f} ms each), verifies {} of {:.1f} tokens ({:.1f} ms, {:.1f} misses, {:.1f} ms making "
                     "them resident each), steps {} ({:.1f} ms, {:.1f} misses, {:.1f} ms each)",
                     ss.drafts, ss.drafts ? ss.draft_ms / ss.drafts : 0.0, ss.verifies,
                     ss.verifies ? double(ss.verified_tokens) / ss.verifies : 0.0, ss.verifies ? ss.verify_ms / ss.verifies : 0.0,
                     ss.verifies ? double(ss.verify_misses) / ss.verifies : 0.0,
                     ss.verifies ? ss.verify_read_ms / ss.verifies : 0.0, ss.steps, ss.steps ? ss.step_ms / ss.steps : 0.0,
                     ss.steps ? double(ss.step_misses) / ss.steps : 0.0, ss.steps ? ss.step_read_ms / ss.steps : 0.0);
        if (!imatrix_out.empty()) {
            engine.set_probe({});
            std::ofstream out(imatrix_out, std::ios::binary);
            out.write("MAKOIMX1", 8);
            uint32_t entries = 0;
            for (const auto& [name, layers] : imx)
                entries += uint32_t(layers.size());
            out.write(reinterpret_cast<const char*>(&entries), 4);
            for (const auto& [name, layers] : imx)
                for (const auto& [layer, rec] : layers) {
                    const uint32_t len = uint32_t(name.size()), n = uint32_t(rec.first.size());
                    const int32_t l = layer;
                    out.write(reinterpret_cast<const char*>(&len), 4);
                    out.write(name.data(), len);
                    out.write(reinterpret_cast<const char*>(&l), 4);
                    out.write(reinterpret_cast<const char*>(&n), 4);
                    for (double v : rec.first) {
                        const float m = float(v / double(rec.second));
                        out.write(reinterpret_cast<const char*>(&m), 4);
                    }
                }
            std::println("wrote {} activation statistics to {}", entries, imatrix_out);
        }
        if (const auto& as = engine.arena().stats(); as.prefetched)
            std::println("read ahead {} experts, {} selected before eviction ({:.0f}%), waited {:.0f} ms for reads in flight",
                         as.prefetched, as.prefetch_used, 100.0 * double(as.prefetch_used) / double(as.prefetched),
                         as.prefetch_wait_ms);
        if (ss.steps) {
            const double n = double(ss.steps);
            std::println("per step: wall {:.1f} ms = inputs read {:.1f} + encoding {:.1f} + ...; GPU {:.1f} ms; CPU waiting "
                         "for routing {:.1f} ms; making experts resident {:.1f} ms ({:.1f} misses)",
                         ss.step_wall_ms / n, ss.step_prep_ms / n, ss.step_encode_ms / n, ss.step_gpu_ms / n,
                         ss.step_wait_ms / n, ss.step_read_ms / n, double(ss.step_misses) / n);
            const double sec = ss.step_attn_ms + ss.step_ffn_ms + ss.step_hit_ms + ss.step_miss_ms;
            if (sec > 0 && ss.parts.empty()) // with parts, empty section encoders' timestamps overlap the parts'
                std::println("GPU sections per step: attention {:.2f}, FFN to routing {:.2f}, shared + resident experts "
                             "{:.2f}, misses + rest {:.2f} = {:.2f} ms busy; GPU time not in a section {:.2f} ms",
                             ss.step_attn_ms / n, ss.step_ffn_ms / n, ss.step_hit_ms / n, ss.step_miss_ms / n, sec / n,
                             (ss.step_gpu_ms - sec) / n);
            auto parts = ss.parts;
            std::ranges::sort(parts, std::greater{}, &mako::core::KernelTime::ms);
            for (const auto& k : parts)
                std::println("    {:36} {:6.2f} ms per step  ({:.0f} a step)", k.kernel, k.ms / n, k.count / n);
        }
        return 0;
    }
    if (split > 0 && width > 0) {
        engine.prefill(std::span<const int32_t>(ids).first(std::min<size_t>(split, ids.size())));
        double ms = 0;
        uint64_t misses = 0, steps = 0, tokens = 0;
        for (size_t t = split; t + width <= ids.size(); t += width) {
            const auto s0 = std::chrono::steady_clock::now();
            const std::span<const int32_t> part(ids.data() + t, width);
            const auto& st = width == 1 ? (engine.step(part[0]), engine.last_step()) : (engine.prefill(part), engine.last_prefill());
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            misses += st.misses;
            ++steps;
            tokens += width;
        }
        std::println("width {}: {} steps, {:.1f} ms a step ({:.1f} ms a token), {:.1f} misses a step ({:.1f} a token)", width,
                     steps, ms / double(steps), ms / double(tokens), double(misses) / double(steps),
                     double(misses) / double(tokens));
        return 0;
    }
    if (use_prefill) {
        engine.set_logits_sink([&](uint32_t pos, const float* lg) { check(pos, lg, 0); });
        const auto s0 = std::chrono::steady_clock::now();
        engine.prefill(ids);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        engine.set_logits_sink({});
        std::println("prefill: {} tokens in {:.0f} ms ({:.1f} tok/s)", ids.size(), ms, ids.size() / ms * 1e3);
        const auto& st = engine.last_prefill();
        std::println("  GPU {:.0f} ms, making experts resident {:.0f} ms (reads {:.0f} ms, {} misses, {:.1f} GB)", st.gpu_ms,
                     st.ensure_ms, st.read_ms, st.misses, st.read_bytes / 1e9);
        double at = 0, ff = 0, hit = 0, miss = 0;
        for (const auto& l : st.layers)
            at += l.attn_ms, ff += l.ffn_ms, hit += l.hit_ms, miss += l.miss_ms;
        if (!st.layers.empty())
            std::println("  per section (GPU): attention {:.0f} ms, FFN to routing {:.0f}, resident experts {:.0f}, "
                         "misses and the rest {:.0f}",
                         at, ff, hit, miss);
    } else {
        if (steps_from) {
            engine.prefill(std::span<const int32_t>(ids).first(std::min(steps_from, ids.size())));
            std::println("prefilled {} ids", engine.position());
        }
        for (size_t t = steps_from; t < ids.size(); ++t) {
            const auto s0 = std::chrono::steady_clock::now();
            const auto& lg = engine.step(ids[t]);
            step_misses += engine.last_step().misses;
            check(t, lg.data(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
        }
    }
    if (compared)
        std::println("{} positions: top-1 agreement {:.1f}%, KL mean {:.5f}, {:.2f} expert misses a step", compared,
                     100.0 * agree / compared, kl_sum / compared,
                     double(step_misses) / double(ids.size() - std::min(steps_from, ids.size())));
    if (scored)
        std::println("perplexity of the next ids over {} positions: {:.4f} (mean NLL {:.5f}){}", scored,
                     std::exp(nll_sum / scored), nll_sum / scored,
                     compared ? std::format("; the reference's {:.4f}", std::exp(ref_nll_sum / compared)) : "");
    int32_t next = int32_t(std::max_element(engine.logits().begin(), engine.logits().end()) - engine.logits().begin());
    std::vector<int32_t> gen_ids;
    const auto g0 = std::chrono::steady_clock::now();
    double wall = 0, gpu = 0, wait = 0, ensure = 0, read = 0;
    uint64_t misses = 0;
    for (int i = 0; i < gen; ++i) {
        gen_ids.push_back(next);
        const auto& lg = engine.step(next);
        next = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
        const auto& st = engine.last_step();
        wall += st.wall_ms, gpu += st.gpu_ms, wait += st.route_wait_ms, ensure += st.ensure_ms, read += st.read_ms;
        misses += st.misses;
    }
    if (gen > 0)
        std::println("per step: wall {:.1f} ms, GPU {:.1f}, CPU waiting for routing {:.1f}, making experts resident {:.1f} "
                     "(reads {:.1f}, {:.1f} misses)",
                     wall / gen, gpu / gen, wait / gen, ensure / gen, read / gen, double(misses) / gen);
    if (!save_profile.empty()) {
        const auto path = save_profile == "default" ? engine.default_profile() : std::filesystem::path(save_profile);
        engine.save_profile(path, true);
        std::println(stderr, "routing counts added to {}", path.string());
    }
    if (gen > 0) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
        std::print("generated:");
        for (int32_t g : gen_ids)
            std::print(" {}", g);
        std::println("\n{} tokens in {:.2f} s ({:.2f} tok/s)", gen, s, gen / s);
    }
}
