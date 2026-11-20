// Runs token ids (or text) through the Hy4-flash engine and compares each position's logits with a reference's
// (ref_dump's <dir>/logits.<pos>.f32), then optionally generates greedily.
//
//   hy4f_run <model-00001.gguf> (--ids I,J,... | --text T | --text-file F) [--ref DIR] [--prefill 0|1] [-n N] [--max-ctx N]
//            [--arena-gb N] [--chunk N] [--group N] [--probe-out DIR] [--telemetry summary|layers|kernels]
// --prefill 1: the ids through prefill() (every position's logits from its sink) instead of one step each.
// --probe-out DIR: one step at a time, each probe the engine reports written as DIR/<name>-<layer>.<pos>.f32.
#include "app/cli.hpp"
#include "hy4f/config.hpp"
#include "hy4f/engine.hpp"
#include "text/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<double> log_softmax(const float* x, size_t n)
{
    const float m = *std::max_element(x, x + n);
    double s = 0;
    for (size_t i = 0; i < n; ++i)
        s += std::exp(double(x[i] - m));
    const double lse = m + std::log(s);
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = x[i] - lse;
    return y;
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<int32_t> ids;
    std::string text, ref, probe_out;
    mako::hy4f::Options opts;
    int gen = 0;
    bool use_prefill = false;
    size_t steps_from = 0, out_from = 0;
    std::string logits_out, compare;
    const auto args =
        mako::app::Cli("hy4f_run", "<model-00001.gguf>", 1, 1)
            .option("--ids", "I,J,...", "the prompt's token ids",
                    [&](const std::string& v) {
                        std::stringstream ss(v);
                        for (std::string x; std::getline(ss, x, ',');)
                            ids.push_back(mako::app::Cli::parse<int32_t>(x));
                    })
            .option("--text", "T", "the prompt as text (control tokens parsed)", text)
            .option("--text-file", "FILE", "the prompt from a file (trailing newlines dropped, as $(cat FILE) does)",
                    [&](const std::string& v) {
                        std::ifstream f(v, std::ios::binary);
                        text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                        while (!text.empty() && text.back() == '\n')
                            text.pop_back();
                    })
            .option("--ref", "DIR", "compare the logits against ref_dump's", ref)
            .option("--prefill", "0|1", "the ids through prefill instead of one step each", use_prefill)
            .option("--steps-from", "N", "the first N ids through prefill, the rest one step each", steps_from)
            .option("--logits-out", "FILE", "write the logits of the positions from --out-from on", logits_out)
            .option("--out-from", "P", "the first position --logits-out writes and --compare reads", out_from)
            .option("--compare", "FILE", "KL against logits --logits-out wrote", compare)
            .option("-n", "N", "then N greedy tokens", gen)
            .option("--max-ctx", "N", "the context length in tokens", opts.max_ctx)
            .option("--arena-gb", "N", "the expert cache's size in GB", opts.arena_gb)
            .option("--chunk", "N", "prompt rows a prefill step", opts.prefill_chunk)
            .option("--group", "N", "prompt rows that read the missing experts once", opts.prefill_group)
            .option("--probe-out", "DIR", "write the engine's probes", probe_out)
            .option("--telemetry", "summary|layers|kernels", "telemetry",
                    [&](const std::string& v) {
                        opts.telemetry = v == "summary"   ? mako::core::Telemetry::summary
                                         : v == "kernels" ? mako::core::Telemetry::kernels
                                                          : mako::core::Telemetry::layers;
                    })
            .parse(argc, argv);
    auto m = mako::gguf::ModelFiles::open(args[0]);
    if (!m) {
        std::println(stderr, "{}", m.error());
        return 1;
    }
    auto cfg = mako::hy4f::Config::load(*m);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(m->meta());
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    if (!text.empty())
        ids = tok->encode(text, true);
    if (ids.empty()) {
        std::println(stderr, "no ids");
        return 2;
    }
    mako::mtl::Context ctx;
    const auto t0 = std::chrono::steady_clock::now();
    mako::hy4f::Engine engine(*m, *cfg, opts, ctx);
    std::println(stderr, "loaded in {:.1f} s: arena {:.1f} GB",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 engine.arena().bytes() / 1e9);

    const uint32_t V = cfg->n_vocab;
    std::vector<std::vector<float>> logits;
    if (steps_from) {
        logits.assign(ids.size(), {});
        steps_from = std::min(steps_from, ids.size());
        engine.set_logits_sink([&](uint32_t pos, const float* l) { logits[pos].assign(l, l + V); });
        engine.prefill(std::span(ids).first(steps_from));
        engine.set_logits_sink({});
        const auto tp = std::chrono::steady_clock::now();
        for (size_t p = steps_from; p < ids.size(); ++p) {
            const auto& l = engine.step(ids[p]);
            logits[p].assign(l.begin(), l.end());
        }
        std::println(stderr, "{} steps in {:.2f} s", ids.size() - steps_from,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    } else if (use_prefill) {
        logits.assign(ids.size(), {});
        engine.set_logits_sink([&](uint32_t pos, const float* l) { logits[pos].assign(l, l + V); });
        const auto tp = std::chrono::steady_clock::now();
        engine.prefill(ids);
        std::println(stderr, "prefill of {} in {:.2f} s", ids.size(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
        engine.set_logits_sink({});
    } else {
        uint32_t pos = 0;
        if (!probe_out.empty()) {
            std::filesystem::create_directories(probe_out);
            engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
                std::ofstream f(std::filesystem::path(probe_out) / std::format("{}-{}.{}.f32", name, layer, pos),
                                std::ios::binary);
                f.write(reinterpret_cast<const char*>(data), std::streamsize(n * 4));
            });
        }
        const auto tp = std::chrono::steady_clock::now();
        for (int32_t id : ids) {
            const auto& l = engine.step(id);
            logits.emplace_back(l.begin(), l.end());
            ++pos;
        }
        std::println(stderr, "{} steps in {:.2f} s", ids.size(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    }

    double kl_sum = 0, nll = 0;
    uint32_t agree = 0, compared = 0;
    for (size_t p = 0; p < logits.size(); ++p) {
        const auto& l = logits[p];
        const uint32_t best = uint32_t(std::max_element(l.begin(), l.end()) - l.begin());
        const auto lp = log_softmax(l.data(), V);
        if (p + 1 < ids.size())
            nll -= lp[ids[p + 1]];
        std::string line = std::format("pos {:4} in {:6} top {:6} {:>14}", p, ids[p], best, tok->decode({&ids[p], 1}));
        if (!ref.empty()) {
            std::ifstream f(std::filesystem::path(ref) / std::format("logits.{}.f32", p), std::ios::binary);
            std::vector<float> r(V);
            if (f.read(reinterpret_cast<char*>(r.data()), std::streamsize(V) * 4)) {
                const auto rp = log_softmax(r.data(), V);
                double kl = 0;
                for (uint32_t v = 0; v < V; ++v)
                    kl += std::exp(rp[v]) * (rp[v] - lp[v]);
                const uint32_t rbest = uint32_t(std::max_element(r.begin(), r.end()) - r.begin());
                agree += rbest == best;
                kl_sum += kl;
                ++compared;
                line += std::format("  ref top {:6}  KL {:.4f}", rbest, kl);
            }
        }
        std::println("{}", line);
    }
    if (!logits_out.empty()) {
        std::ofstream f(logits_out, std::ios::binary);
        for (size_t p = out_from; p < logits.size(); ++p)
            f.write(reinterpret_cast<const char*>(logits[p].data()), std::streamsize(V) * 4);
    }
    if (!compare.empty()) {
        std::ifstream f(compare, std::ios::binary);
        std::vector<float> r(V);
        double sum = 0, worst = 0;
        size_t n = 0;
        for (size_t p = out_from; p < logits.size() && f.read(reinterpret_cast<char*>(r.data()), std::streamsize(V) * 4); ++p) {
            const auto rp = log_softmax(r.data(), V), lp = log_softmax(logits[p].data(), V);
            double kl = 0;
            for (uint32_t v = 0; v < V; ++v)
                kl += std::exp(rp[v]) * (rp[v] - lp[v]);
            sum += kl;
            worst = std::max(worst, kl);
            ++n;
        }
        std::println("against {}: {} positions, mean KL {:.3e}, max {:.3e}", compare, n, n ? sum / double(n) : 0.0, worst);
    }
    if (ids.size() > 1)
        std::println("perplexity of the ids' next tokens: {:.3f}", std::exp(nll / double(ids.size() - 1)));
    if (compared)
        std::println("against the reference: {} positions, top-1 agreement {:.1f}%, mean KL {:.5f}", compared,
                     100.0 * agree / compared, kl_sum / compared);

    if (gen > 0) {
        std::vector<int32_t> out;
        int32_t next = int32_t(std::max_element(logits.back().begin(), logits.back().end()) - logits.back().begin());
        const auto tg = std::chrono::steady_clock::now();
        for (int i = 0; i < gen; ++i) {
            out.push_back(next);
            const auto& l = engine.step(next);
            next = int32_t(std::max_element(l.begin(), l.end()) - l.begin());
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
        std::println("generated: {}", tok->decode(out));
        std::println("{} tokens in {:.2f} s: {:.1f} tokens/s, last step {:.1f} ms ({} misses)", gen, s, gen / s,
                     engine.last_step().wall_ms, engine.last_step().misses);
    }
    return 0;
}
