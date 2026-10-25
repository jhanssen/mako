// Prefill time of reads of several sizes on top of a base context, as an agent turn reading a tool result.
//   prefill_sweep <model.gguf> [--base N] [--sizes a,b,...] [--repeats N] [--layers] [--steps N] [--prefill-group N]
//                 [--stage-gb N]                 [--decode N] [--turns N] [--prompt-file F] [--temp T] [--no-spec]
// The text is this repository's sources, tokenized. Each size runs `repeats` times (default 2), each from a checkpoint
// of the base context; the experts left resident by the runs before stay. --layers adds per-layer GPU timestamps
// (GPU time by section; extra encoder boundaries slow it somewhat). --steps N also times sizes up to N as decode
// steps, one token at a time. --decode N instead generates N tokens after the base context through a Session
// (greedy and speculative unless --temp / --no-spec) and summarizes its steps. --prompt-file takes the text from F
// (all of it, up to --base tokens) instead of the sources. --turns N simulates an agent: an answer of 150 tokens after the
// base context, then N turns that each read 1000 new tokens and answer 150 (misses per step in each answer's halves).
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gen/session.hpp"
#include "gguf/model_files.hpp"
#include "text/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string sources()
{
    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::recursive_directory_iterator(MAKO_SOURCE_DIR "/src"))
        if (e.is_regular_file() && (e.path().extension() == ".cpp" || e.path().extension() == ".hpp" ||
                                    e.path().extension() == ".metal"))
            files.push_back(e.path());
    std::ranges::sort(files);
    std::string all;
    for (const auto& f : files) {
        std::ifstream in(f);
        std::stringstream ss;
        ss << in.rdbuf();
        all += "// " + f.filename().string() + "\n" + ss.str() + "\n";
    }
    return all;
}

double ms_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// Generates n tokens after the prompt (greedy, speculative) and summarizes the engine steps: per step its wall,
// GPU, routing-wait and expert-read times, misses and (with layer telemetry) GPU time by section; the time between
// steps (drafting, sampling) is the rest of the decode's wall time.
int run_decode(mako::glm::Engine& engine, std::span<const int32_t> prompt, uint32_t n, float temp, bool spec)
{
    mako::gen::Session session(engine);
    mako::gen::GenerateParams gp;
    gp.sampling.temperature = temp;
    gp.sampling.top_p = 0.95f;
    gp.seed = 1;
    gp.speculative = spec;
    gp.max_tokens = n;
    std::vector<mako::core::StepStats> steps;
    uint32_t seen = uint32_t(prompt.size());
    const auto r = session.generate(prompt, gp, [&](int32_t) {
        if (engine.position() != seen) {
            steps.push_back(engine.last_step());
            seen = engine.position();
        }
        return true;
    });
    std::println("prompt {} tokens in {:.0f} ms; {} tokens in {:.0f} ms ({:.1f} ms/token), {} steps, {} of {} drafts kept "
                 "({:.0f}%)",
                 prompt.size(), r.prefill_ms, r.tokens.size(), r.decode_ms, r.decode_ms / r.tokens.size(), steps.size(),
                 r.accepted, r.drafts, 100.0 * r.accepted / std::max(1u, r.drafts));
    // The first steps after a long prompt pay for the experts it displaced; both halves are shown.
    auto summary = [&](size_t b, size_t e, const char* what) {
        double wall = 0, gpu = 0, wait = 0, ensure = 0, read = 0, enc = 0, misses = 0;
        double kda = 0, mla = 0, ffn = 0, hit = 0, miss = 0;
        for (size_t i = b; i < e; ++i) {
            const auto& s = steps[i];
            wall += s.wall_ms, gpu += s.gpu_ms, wait += s.route_wait_ms, ensure += s.ensure_ms, read += s.read_ms;
            enc += s.encode_ms, misses += double(s.misses);
            for (const auto& l : s.layers) {
                (l.mla ? mla : kda) += l.attn_ms;
                ffn += l.ffn_ms, hit += l.hit_ms, miss += l.miss_ms;
            }
        }
        const double k = double(e - b);
        std::println("{} (steps {}-{}), per step: wall {:.1f} ms, GPU {:.1f}, CPU waiting for routing {:.1f}, making "
                     "experts resident {:.1f} (reads {:.1f}, {:.1f} misses), encode {:.2f}",
                     what, b, e, wall / k, gpu / k, wait / k, ensure / k, read / k, misses / k, enc / k);
        if (kda + mla > 0)
            std::println("  GPU by section per step: KDA attention {:.1f} ms, MLA attention {:.1f}, FFN to routing {:.1f}, "
                         "resident experts {:.1f}, misses and combine {:.1f}",
                         kda / k, mla / k, ffn / k, hit / k, miss / k);
    };
    const size_t half = steps.size() / 2;
    summary(1, half, "first half");
    summary(half, steps.size(), "second half");
    double step_wall = 0;
    for (const auto& s : steps)
        step_wall += s.wall_ms;
    std::println("between steps (drafting, sampling): {:.1f} ms per step", (r.decode_ms - step_wall) / steps.size());
    return 0;
}

int run_turns(mako::glm::Engine& engine, const std::vector<int32_t>& ids, uint32_t base, uint32_t turns, uint32_t read,
              uint32_t gen)
{
    mako::gen::Session session(engine);
    mako::gen::GenerateParams gp;
    gp.sampling.temperature = 0;
    gp.max_tokens = gen;
    std::vector<int32_t> ctx(ids.begin(), ids.begin() + base);
    for (uint32_t turn = 0; turn <= turns; ++turn) {
        if (turn)
            ctx.insert(ctx.end(), ids.begin() + base + (turn - 1) * read, ids.begin() + base + turn * read);
        std::vector<mako::core::StepStats> steps;
        uint32_t seen = uint32_t(ctx.size());
        const auto r = session.generate(ctx, gp, [&](int32_t) {
            if (engine.position() != seen) {
                steps.push_back(engine.last_step());
                seen = engine.position();
            }
            return true;
        });
        auto half = [&](size_t b, size_t e) {
            double misses = 0, read_ms = 0, wall = 0;
            for (size_t i = b; i < e; ++i)
                misses += double(steps[i].misses), read_ms += steps[i].read_ms, wall += steps[i].wall_ms;
            const double k = double(std::max<size_t>(e - b, 1));
            return std::format("{:.1f} misses, reads {:.1f} ms, wall {:.1f} ms", misses / k, read_ms / k, wall / k);
        };
        const size_t h = steps.size() / 2;
        std::println("turn {}: prefill {} new tokens in {:.0f} ms; {} tokens in {:.0f} ms ({:.1f} ms/token); per step, "
                     "first half: {}; second half: {}",
                     turn, r.prompt_tokens - r.reused_tokens, r.prefill_ms, r.tokens.size(), r.decode_ms,
                     r.decode_ms / r.tokens.size(), half(1, h), half(h, steps.size()));
        ctx.insert(ctx.end(), r.tokens.begin(), r.tokens.end());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: prefill_sweep <model.gguf> [--base N] [--sizes a,b,...] [--repeats N] [--layers] "
                             "[--steps N] [--prefill-group N]");
        return 2;
    }
    uint32_t base = 15000, repeats = 2, step_max = 0, decode = 0, turns = 0;
    std::string prompt_file;
    float temp = 0.0f;
    bool spec = true;
    std::vector<uint32_t> sizes{16, 64, 256, 512, 1024, 2048, 4096};
    mako::glm::Options opts;
    opts.telemetry = mako::core::Telemetry::summary;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers") {
            opts.telemetry = mako::core::Telemetry::layers;
            continue;
        }
        if (a == "--no-spec") {
            spec = false;
            continue;
        }
        if (i + 1 >= argc) {
            std::println(stderr, "option {} needs a value", a);
            return 2;
        }
        const std::string v = argv[++i];
        if (a == "--base")
            base = uint32_t(std::stoul(v));
        else if (a == "--repeats")
            repeats = uint32_t(std::stoul(v));
        else if (a == "--steps")
            step_max = uint32_t(std::stoul(v));
        else if (a == "--decode")
            decode = uint32_t(std::stoul(v));
        else if (a == "--turns")
            turns = uint32_t(std::stoul(v));
        else if (a == "--prompt-file")
            prompt_file = v;
        else if (a == "--temp")
            temp = std::stof(v);
        else if (a == "--prefill-group")
            opts.prefill_group = uint32_t(std::stoul(v));
        else if (a == "--stage-gb")
            opts.stage_gb = std::stod(v);
        else if (a == "--sizes") {
            sizes.clear();
            std::stringstream ss(v);
            for (std::string x; std::getline(ss, x, ',');)
                sizes.push_back(uint32_t(std::stoul(x)));
        } else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model) {
        std::println(stderr, "{}", model.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(model->meta());
    auto cfg = mako::glm::Config::load(*model);
    if (!tok || !cfg) {
        std::println(stderr, "{}", !tok ? tok.error() : cfg.error());
        return 1;
    }
    const uint32_t most = *std::ranges::max_element(sizes);
    std::vector<int32_t> ids;
    std::string src = sources();
    if (!prompt_file.empty()) {
        std::ifstream in(prompt_file);
        std::stringstream ss;
        ss << in.rdbuf();
        src = ss.str();
    }
    const auto text = tok->encode(src, false);
    if (!prompt_file.empty())
        base = std::min<uint32_t>(base, uint32_t(text.size()));
    while (ids.size() < base + most)
        ids.insert(ids.end(), text.begin(), text.end());
    opts.max_ctx = base + most + 16;
    constexpr uint32_t kTurnRead = 1000, kTurnGen = 150;
    while (ids.size() < base + turns * kTurnRead)
        ids.insert(ids.end(), text.begin(), text.end());
    opts.mtp = decode > 0 || turns > 0;
    opts.max_ctx = std::max(opts.max_ctx, base + decode + (turns + 1) * (kTurnRead + kTurnGen) + 16);
    mako::mtl::Context ctx;
    mako::glm::Engine engine(*model, *cfg, opts, ctx);
    if (decode)
        return run_decode(engine, std::span(ids).first(base), decode, temp, spec);
    if (turns)
        return run_turns(engine, ids, base, turns, kTurnRead, kTurnGen);

    auto t = std::chrono::steady_clock::now();
    engine.prefill(std::span(ids).first(base));
    std::println("base context: {} tokens in {:.0f} ms", base, ms_since(t));
    const auto cp = engine.checkpoint();
    const std::span<const int32_t> more = std::span(ids).subspan(base);

    std::println("| tokens | run | wall ms | tok/s | GPU ms | resident ms | misses | GB read | read ms |{}",
                 opts.telemetry == mako::core::Telemetry::layers ? " KDA | MLA | FFN | experts | rest |" : "");
    std::println("|---|---|---|---|---|---|---|---|---|{}",
                 opts.telemetry == mako::core::Telemetry::layers ? "---|---|---|---|---|" : "");
    for (uint32_t n : sizes)
        for (uint32_t r = 0; r < repeats; ++r) {
            engine.restore(cp);
            t = std::chrono::steady_clock::now();
            engine.prefill(more.first(n));
            const double wall = ms_since(t);
            const auto& p = engine.last_prefill();
            std::string sections;
            if (!p.layers.empty()) {
                double kda = 0, mla = 0, ffn = 0, ex = 0, rest = 0;
                for (const auto& l : p.layers) {
                    (l.mla ? mla : kda) += l.attn_ms;
                    ffn += l.ffn_ms;
                    ex += l.hit_ms;
                    rest += l.miss_ms;
                }
                sections = std::format(" {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |", kda, mla, ffn, ex, rest);
            }
            std::println("| {} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {} | {:.2f} | {:.0f} |{}", n, r + 1, wall,
                         n / wall * 1e3, p.gpu_ms, p.ensure_ms, p.misses, p.read_bytes / 1e9, p.read_ms, sections);
        }
    for (uint32_t n : sizes) {
        if (n > step_max)
            continue;
        engine.restore(cp);
        t = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n; ++i)
            engine.step(more[i]);
        const double wall = ms_since(t);
        std::println("decode steps: {} tokens in {:.0f} ms ({:.0f} tok/s)", n, wall, n / wall * 1e3);
    }
    return 0;
}
