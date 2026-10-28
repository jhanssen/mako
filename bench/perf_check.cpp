// Catches speed regressions: times a fixed prefill and decode of GLM-5.3 or DeepSeek-V4.1 and compares them with a
// baseline taken on this machine.
//   perf_check <model.gguf> [--prompt N] [--decode N] [--pause S] [--save] [--baseline FILE]
// Four runs, each after a pause and once macOS reports the nominal thermal state (throttling slows everything by
// ~1.3x): the prefill of the first N tokens of bench/data/perf_prompt.txt and N greedy decode steps after a short
// prompt, each once for its time and once with part timing for the GPU time by part and section. Shares of the GPU
// time hardly move when the whole machine runs slower, so they are compared even when the totals are not.
// Flags: a total more than 10% slower, a part or section at least 1% of the GPU time whose share grew by more than
// 25%. --save writes this run as the baseline (default: ~/Library/Application Support/mako/perf/<model>.json).
// Exits 1 when something is flagged.

#include "app/cli.hpp"
#include "core/telemetry.hpp"
#include "dsv41/engine.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <print>
#include <sstream>
#include <thread>

namespace perf {

struct Phase {
    double ms = 0;                        // prefill: the whole prompt; decode: median step
    double misses = 0;                    // per prefill or per step
    std::map<std::string, double> shares; // part or "section ..." -> share of the GPU time
    std::string thermal;                  // at the end of the timed run
};

struct Result {
    std::string commit, chip, model;
    uint32_t prompt = 0, decode = 0;
    Phase prefill, steps;
};

} // namespace perf

namespace {

using perf::Phase;
using perf::Result;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Waits until the thermal state is nominal (at most 10 minutes), then `pause` seconds more.
void settle(uint32_t pause)
{
    const auto t0 = Clock::now();
    bool said = false;
    while (mako::core::thermal_state() != "nominal" && ms_since(t0) < 600e3) {
        if (!said)
            std::println("waiting for the nominal thermal state ({} now)", mako::core::thermal_state());
        said = true;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    std::this_thread::sleep_for(std::chrono::seconds(pause));
}

// Decode steps leave the sections out: their stage timestamps overlap (the step is committed while it is still
// being encoded), so they do not divide its GPU time.
void add_shares(const mako::core::StepStats& s, std::map<std::string, double>& ms, bool sections)
{
    for (const auto& p : s.parts)
        ms[p.kernel] += p.ms;
    for (const auto& l : sections ? s.layers : std::vector<mako::core::LayerTimes>{}) {
        ms[l.mla ? "section MLA attention" : "section KDA attention"] += l.attn_ms;
        ms["section FFN to routing"] += l.ffn_ms;
        ms["section experts"] += l.hit_ms;
        ms["section misses and combine"] += l.miss_ms;
    }
}

std::map<std::string, double> to_shares(const std::map<std::string, double>& ms, double gpu_ms)
{
    std::map<std::string, double> out;
    for (const auto& [k, v] : ms)
        if (v > 0)
            out[k] = v / gpu_ms;
    return out;
}

struct Model {
    mako::gen::Model* gen = nullptr;
    mako::core::EngineBase* base = nullptr;
    std::function<const mako::core::StepStats&()> last_prefill;
};

int32_t argmax(const std::vector<float>& v)
{
    return int32_t(std::ranges::max_element(v) - v.begin());
}

Phase run_prefill(Model& m, std::span<const int32_t> ids, bool parts)
{
    m.base->set_part_timing(parts);
    m.gen->reset();
    const auto t = Clock::now();
    m.gen->prefill(ids);
    Phase p;
    p.ms = ms_since(t);
    const auto& s = m.last_prefill();
    p.misses = double(s.misses);
    if (parts) {
        std::map<std::string, double> ms;
        add_shares(s, ms, true);
        p.shares = to_shares(ms, s.gpu_ms);
    }
    p.thermal = mako::core::thermal_state();
    return p;
}

// n greedy steps after a 256-token prompt; the first two left out (first-use costs).
Phase run_steps(Model& m, std::span<const int32_t> ids, uint32_t n, bool parts)
{
    m.base->set_part_timing(parts);
    m.gen->reset();
    int32_t tok = argmax(m.gen->prefill(ids.first(256)));
    std::vector<double> wall;
    std::map<std::string, double> ms;
    double gpu = 0, misses = 0;
    for (uint32_t i = 0; i < n + 2; ++i) {
        tok = argmax(m.gen->step(tok));
        if (i < 2)
            continue;
        const auto& s = m.base->last_step();
        wall.push_back(s.wall_ms);
        gpu += s.gpu_ms;
        misses += double(s.misses);
        add_shares(s, ms, false);
    }
    std::ranges::sort(wall);
    Phase p;
    p.ms = wall[wall.size() / 2];
    p.misses = misses / n;
    if (parts)
        p.shares = to_shares(ms, gpu);
    p.thermal = mako::core::thermal_state();
    return p;
}

struct Checker {
    bool flagged = false;

    void total(const char* what, double now, double base, bool comparable)
    {
        const double change = now / base - 1;
        const bool flag = comparable && change > 0.10;
        flagged |= flag;
        std::println("  {:34} {:9.2f} ms  baseline {:9.2f}  {:+6.1f}%{}", what, now, base, 100 * change,
                     flag ? "  SLOWER" : comparable ? "" : "  (not comparable: throttled)");
    }

    void shares(const char* phase, const std::map<std::string, double>& now, const std::map<std::string, double>& base)
    {
        std::map<std::string, std::pair<double, double>> all;
        for (const auto& [k, v] : now)
            all[k].first = v;
        for (const auto& [k, v] : base)
            all[k].second = v;
        std::println("{} GPU time by part (share now / baseline):", phase);
        for (const auto& [k, v] : all) {
            const auto [n, b] = v;
            if (std::max(n, b) < 0.01)
                continue;
            const bool flag = b > 0 ? n / b > 1.25 : true;
            flagged |= flag;
            std::println("  {:34} {:5.1f}% / {:5.1f}%{}", k, 100 * n, 100 * b,
                         flag ? (b > 0 ? "  GREW" : "  NEW") : "");
        }
    }
};

} // namespace

int main(int argc, char** argv)
{
    uint32_t prompt = 4096, decode = 64, pause = 30;
    bool save = false;
    std::string baseline_path;
    const auto args = mako::app::Cli("perf_check", "<model.gguf>", 1, 1)
                          .option("--prompt", "N", "prefill tokens (default 4096)", prompt)
                          .option("--decode", "N", "decode steps timed (default 64)", decode)
                          .option("--pause", "S", "seconds of rest before each run (default 30)", pause)
                          .flag("--save", "write this run as the baseline", save)
                          .option("--baseline", "FILE", "the baseline (default: per model, see above)", baseline_path)
                          .parse(argc, argv);
    auto files = mako::gguf::ModelFiles::open(args[0]);
    if (!files) {
        std::println(stderr, "{}", files.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(files->meta());
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    std::ifstream in(MAKO_SOURCE_DIR "/bench/data/perf_prompt.txt");
    std::stringstream ss;
    ss << in.rdbuf();
    const auto ids = tok->encode(ss.str(), false);
    if (ids.size() < std::max(prompt, 256u)) {
        std::println(stderr, "the prompt has {} tokens, fewer than --prompt {}", ids.size(), prompt);
        return 1;
    }

    mako::mtl::Context ctx;
    std::unique_ptr<mako::glm::Engine> glm;
    std::unique_ptr<mako::dsv41::Engine> ds;
    Model m;
    const uint32_t max_ctx = std::max(prompt, 256 + decode + 2) + 16;
    if (files->meta().get_string("general.architecture").value_or("") == "deepseek41") {
        auto cfg = mako::dsv41::Config::load(*files);
        if (!cfg) {
            std::println(stderr, "{}", cfg.error());
            return 1;
        }
        mako::dsv41::Options opts;
        opts.max_ctx = max_ctx;
        opts.telemetry = mako::core::Telemetry::layers;
        ds = std::make_unique<mako::dsv41::Engine>(*files, *cfg, opts, ctx);
        m = {ds.get(), ds.get(), [&]() -> const mako::core::StepStats& { return ds->last_prefill(); }};
    } else {
        auto cfg = mako::glm::Config::load(*files);
        if (!cfg) {
            std::println(stderr, "{}", cfg.error());
            return 1;
        }
        mako::glm::Options opts;
        opts.max_ctx = max_ctx;
        opts.telemetry = mako::core::Telemetry::layers;
        glm = std::make_unique<mako::glm::Engine>(*files, *cfg, opts, ctx);
        m = {glm.get(), glm.get(), [&]() -> const mako::core::StepStats& { return glm->last_prefill(); }};
    }

    const auto machine = mako::core::query_machine(ctx.info());
    Result r{machine.commit, machine.chip, std::filesystem::path(args[0]).filename().string(), prompt, decode, {}, {}};
    const std::span<const int32_t> span(ids);
    settle(pause);
    r.prefill = run_prefill(m, span.first(prompt), false);
    std::println("prefill {} tokens: {:.0f} ms ({:.0f} tok/s), thermal {}", prompt, r.prefill.ms,
                 prompt / r.prefill.ms * 1e3, r.prefill.thermal);
    settle(pause);
    r.prefill.shares = run_prefill(m, span.first(prompt), true).shares;
    settle(pause);
    r.steps = run_steps(m, span, decode, false);
    std::println("decode: {:.2f} ms/step median, {:.1f} misses a step, thermal {}", r.steps.ms, r.steps.misses,
                 r.steps.thermal);
    settle(pause);
    r.steps.shares = run_steps(m, span, decode, true).shares;

    if (baseline_path.empty()) {
        const char* home = std::getenv("HOME");
        baseline_path = (std::filesystem::path(home ? home : ".") / "Library/Application Support/mako/perf" /
                         (std::filesystem::path(args[0]).stem().string() + ".json"))
                            .string();
    }
    if (save) {
        std::filesystem::create_directories(std::filesystem::path(baseline_path).parent_path());
        std::string out;
        if (auto ec = glz::write<glz::opts{.prettify = true}>(r, out)) {
            std::println(stderr, "writing the baseline: {}", glz::format_error(ec, out));
            return 1;
        }
        std::ofstream(baseline_path) << out;
        std::println("baseline written: {}", baseline_path);
        return 0;
    }
    Result b;
    std::string text;
    if (std::ifstream bin(baseline_path); bin) {
        std::stringstream bs;
        bs << bin.rdbuf();
        text = bs.str();
    }
    if (text.empty()) {
        std::println("no baseline at {} (--save writes one)", baseline_path);
        return 0;
    }
    if (auto ec = glz::read_json(b, text)) {
        std::println(stderr, "{}: {}", baseline_path, glz::format_error(ec, text));
        return 1;
    }
    if (b.chip != r.chip || b.model != r.model || b.prompt != r.prompt || b.decode != r.decode)
        std::println("the baseline was taken with other settings ({}, {}, prompt {}, decode {})", b.chip, b.model,
                     b.prompt, b.decode);
    std::println("against the baseline of {}:", b.commit);
    Checker c;
    const bool cool = r.prefill.thermal == "nominal" && r.steps.thermal == "nominal" && b.prefill.thermal == "nominal" &&
                      b.steps.thermal == "nominal";
    c.total("prefill", r.prefill.ms, b.prefill.ms, cool);
    c.total("decode step (median)", r.steps.ms, b.steps.ms, cool);
    if (r.prefill.misses != b.prefill.misses || std::abs(r.steps.misses - b.steps.misses) > 0.5)
        std::println("  misses differ (prefill {} / {}, a step {:.1f} / {:.1f}): reads from the SSD change the totals",
                     r.prefill.misses, b.prefill.misses, r.steps.misses, b.steps.misses);
    c.shares("prefill", r.prefill.shares, b.prefill.shares);
    c.shares("decode", r.steps.shares, b.steps.shares);
    std::println("{}", c.flagged ? "FLAGGED: look at the lines marked above" : "no regression found");
    return c.flagged ? 1 : 0;
}
