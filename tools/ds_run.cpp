// Runs token ids through the DeepSeek-V4.1 engine one step at a time and compares each position's logits with a
// reference's (float32 [positions][vocab]).
//
//   ds_run <model-00001.gguf> <engram.gguf> --ids I,J,... [--ref logits.f32] [--logits-out FILE] [--arena-gb N]
//          [--stage-gb N] [--max-ctx N] [--ablate FILE] [--top-k N] [--prefill 1] [--chunk N] [--group N] [--telemetry layers] [--profile F]
//          [--save-profile F|default] [-n N]
// --prefill 1: the ids through prefill() (every position's logits from its sink) instead of one step each.
// -n N: then N greedy tokens.
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"

#include <algorithm>
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
        else if (a == "--telemetry")
            opts.telemetry = v == "layers" ? mako::core::Telemetry::layers : mako::core::Telemetry::summary;
        else if (a == "--save-profile")
            save_profile = v;
        else if (a == "--profile")
            opts.expert_counts = v;
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
    mako::dsv41::Engine engine(*m, *cfg, opts, ctx);
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
    double kl_sum = 0;
    int agree = 0, compared = 0;
    auto check = [&](size_t t, const float* lg, double ms) {
        if (lo)
            lo.write(reinterpret_cast<const char*>(lg), std::streamsize(V * 4));
        const size_t top = size_t(std::max_element(lg, lg + V) - lg);
        if ((t + 1) * V <= want.size()) {
            const float* w = want.data() + t * V;
            const auto lp = log_softmax(w, V), lq = log_softmax(lg, V);
            double kl = 0;
            for (size_t i = 0; i < V; ++i)
                kl += std::exp(double(lp[i])) * (double(lp[i]) - double(lq[i]));
            const size_t wtop = size_t(std::max_element(w, w + V) - w);
            kl_sum += kl;
            agree += wtop == top;
            ++compared;
            std::println("pos {:4} token {:6}: top {:6} (ref {:6}) KL {:.5f}  {:.1f} ms", t, ids[t], top, wtop, kl, ms);
        } else {
            std::println("pos {:4} token {:6}: top {:6} ({:.3f})  {:.1f} ms", t, ids[t], top, lg[top], ms);
        }
    };
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
        for (size_t t = 0; t < ids.size(); ++t) {
            const auto s0 = std::chrono::steady_clock::now();
            const auto& lg = engine.step(ids[t]);
            check(t, lg.data(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
        }
    }
    if (compared)
        std::println("{} positions: top-1 agreement {:.1f}%, KL mean {:.5f}", compared, 100.0 * agree / compared,
                     kl_sum / compared);
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
