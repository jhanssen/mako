// Hy4-flash's verify() and keep(), bit for bit against the same tokens step by step: a verify's rows, and continuing
// after keep() as if the rejected tokens had never run, at a position inside the dense window and among the indexer's
// selecting positions with rejected tokens completing a pool group; then verifies of 2-8 keeping 1-8 one after another
// across the window layers' buffer moves (their 1600 rows fill at 1600 positions).
// With the DFly2 drafter (second argument): greedy decoding drafting and verifying gives the tokens decoding one
// step at a time does, and on a code prompt it keeps most of its drafts.
//   hy4f_spec_test <model-00001.gguf> [dfly2.safetensors]
#include "hy4f/config.hpp"
#include "hy4f/engine.hpp"

#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "  ok  " : "  FAIL", what);
    failures += !ok;
}

double kl(const float* p_logits, const float* q_logits, size_t n)
{
    auto lsm = [n](const float* x) {
        double m = -1e300;
        for (size_t i = 0; i < n; ++i)
            m = std::max(m, double(x[i]));
        double s = 0;
        for (size_t i = 0; i < n; ++i)
            s += std::exp(double(x[i]) - m);
        std::vector<double> y(n);
        for (size_t i = 0; i < n; ++i)
            y[i] = double(x[i]) - m - std::log(s);
        return y;
    };
    const auto lp = lsm(p_logits), lq = lsm(q_logits);
    double k = 0;
    for (size_t i = 0; i < n; ++i)
        k += std::exp(lp[i]) * (lp[i] - lq[i]);
    return k;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: hy4f_spec_test <model-00001.gguf> [dfly2.safetensors]");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
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
    std::vector<int32_t> seq;
    {
        std::ifstream f(std::filesystem::path(MAKO_TEST_DATA) / "hy4f_long_ids.txt");
        std::stringstream ss;
        ss << f.rdbuf();
        for (std::string x; std::getline(ss, x, ',');)
            if (!x.empty() && x != "\n")
                seq.push_back(std::stoi(x));
    }
    mako::mtl::Context ctx;
    mako::hy4f::Options opts;
    opts.max_ctx = 4096;
    const bool with_drafter = argc > 2 && std::string_view(argv[2]).size() && std::filesystem::exists(argv[2]);
    if (with_drafter)
        opts.dfly2 = argv[2];
    mako::hy4f::Engine engine(*m, *cfg, opts, ctx);
    const size_t V = cfg->n_vocab;

    // 10: dense; 2101: selecting, the next position 2 before a group boundary (a verify of 3 or more completes the
    // group, a keep of 1 leaves it open).
    for (const uint32_t at : {10u, 2101u}) {
        const size_t end = at + 12;
        std::vector<std::vector<float>> steps(end);
        engine.reset();
        engine.prefill(std::span<const int32_t>(seq).first(at));
        for (size_t i = at; i < end; ++i)
            steps[i] = engine.step(seq[i]);
        for (const uint32_t width : {2u, 4u, 5u, 8u}) {
            for (const uint32_t keep : {1u, 2u}) {
                engine.reset();
                engine.prefill(std::span<const int32_t>(seq).first(at));
                std::vector<int32_t> run_ids(seq.begin() + at, seq.begin() + at + width);
                for (uint32_t i = keep; i < width; ++i)
                    run_ids[i] = 1000 + int32_t(i); // rejected drafts
                const std::vector<float> rows = engine.verify(run_ids);
                double kv = 0;
                bool same = true;
                for (uint32_t i = 0; i < keep; ++i) {
                    kv = std::max(kv, kl(steps[at + i].data(), rows.data() + uint64_t(i) * V, V));
                    same = same && std::equal(rows.begin() + uint64_t(i) * V, rows.begin() + uint64_t(i + 1) * V,
                                              steps[at + i].begin());
                }
                engine.keep(keep);
                double kc = 0;
                bool same_after = true;
                for (size_t i = at + keep; i < end; ++i) {
                    const auto& l = engine.step(seq[i]);
                    kc = std::max(kc, kl(steps[i].data(), l.data(), V));
                    same_after = same_after && std::equal(l.begin(), l.end(), steps[i].begin());
                }
                check(engine.position() == end && same && same_after,
                      std::format("position {}: verify of {} keeping {}: rows vs steps KL max {:.1e}{}, continuing "
                                  "after keep KL max {:.1e}{}",
                                  at, width, keep, kv, same ? " (bit-identical)" : "", kc,
                                  same_after ? " (bit-identical)" : ""));
            }
        }
    }
    {
        constexpr uint32_t from = 1500, n = 200;
        std::vector<std::vector<float>> steps(n);
        engine.reset();
        engine.prefill(std::span<const int32_t>(seq).first(from));
        for (uint32_t i = 0; i < n; ++i)
            steps[i] = engine.step(seq[from + i]);
        engine.reset();
        engine.prefill(std::span<const int32_t>(seq).first(from));
        uint32_t done = 0, rounds = 0;
        bool same = true;
        while (done < n) {
            const uint32_t width = std::min(2 + rounds % 7, n - done), keep = 1 + (rounds * 3) % width;
            std::vector<int32_t> run_ids(seq.begin() + from + done, seq.begin() + from + done + width);
            for (uint32_t i = keep; i < width; ++i)
                run_ids[i] = 2000 + int32_t(i);
            const std::vector<float> rows = engine.verify(run_ids);
            for (uint32_t i = 0; i < keep; ++i)
                same = same && std::equal(rows.begin() + uint64_t(i) * V, rows.begin() + uint64_t(i + 1) * V,
                                          steps[done + i].begin());
            engine.keep(keep);
            done += keep;
            ++rounds;
        }
        check(same && engine.position() == from + n,
              std::format("positions {}-{} in {} verifies of 2-8 keeping 1-8: every kept row as the steps' "
                          "(bit for bit)",
                          from, from + n - 1, rounds));
    }
    if (with_drafter) {
        // "Write a Python implementation of a red-black tree with insert, delete and search, with comments." through
        // the chat template, thinking off.
        const std::vector<int32_t> prompt = {120000, 13251, 120001, 120039, 59785, 287, 62, 4138, 506, 25, 3202, 32980,
                                             1121, 120025, 120000, 3717, 120001, 15434, 259, 8010, 8143, 299, 259, 4180,
                                             68321, 7338, 430, 7231, 11, 10691, 303, 4997, 11, 430, 10977, 13, 120025,
                                             120000, 611, 10372, 120001, 120029, 120030};
        const auto argmax = [&](const float* l) { return int32_t(std::max_element(l, l + V) - l); };
        constexpr uint32_t n = 160;
        engine.reset();
        engine.prefill(std::span(prompt).first(prompt.size() - 1));
        std::vector<int32_t> plain;
        int32_t next = argmax(engine.step(prompt.back()).data());
        for (uint32_t i = 0; i < n; ++i) {
            plain.push_back(next);
            next = argmax(engine.step(next).data());
        }
        engine.reset();
        engine.prefill(std::span(prompt).first(prompt.size() - 1));
        std::vector<int32_t> spec;
        next = argmax(engine.step(prompt.back()).data());
        uint64_t offered = 0, kept = 0;
        while (spec.size() < n) {
            const std::vector<int32_t> drafts = engine.draft(next);
            std::vector<int32_t> run{next};
            run.insert(run.end(), drafts.begin(), drafts.end());
            const std::vector<float> rows = drafts.empty() ? engine.step(next) : engine.verify(run);
            uint32_t k = 1;
            int32_t t = argmax(rows.data());
            while (k < run.size() && t == run[k])
                t = argmax(rows.data() + uint64_t(k++) * V);
            if (!drafts.empty())
                engine.keep(k);
            spec.insert(spec.end(), run.begin(), run.begin() + k);
            offered += drafts.size();
            kept += k - 1;
            next = t;
        }
        spec.resize(n);
        check(spec == plain,
              std::format("DFly2: {} greedy tokens drafting and verifying, the same as step by step", n));
        check(offered > 0 && kept >= offered / 2,
              std::format("DFly2 on code: {} of {} drafts kept", kept, offered));
    } else {
        std::println("  (no DFly2 drafter given: its checks skipped)");
    }
    std::println("{} failures", failures);
    return failures ? 1 : 0;
}
