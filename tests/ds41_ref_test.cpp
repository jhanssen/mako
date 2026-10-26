// DeepSeek-V4.1 against the reference engine's logits (top 64 per position, tests/data/ds41_*_ref.bin), one mode per
// ctest entry:
//   short    1300 positions of prose, step by step and through prefill
//   long     the last 1024 of 24576 positions (the candidate stage active), through prefill and as decode steps
//   penalty  the miss penalty at 0.1: KL bounded and fewer misses than without
//   q6k      dense weights re-quantized to Q6_K: KL bounded
//   dspark   with the drafter (a directory; skipped without one): greedy output with drafts equals it without
// and `import <logits.f32> <out.bin> [positions]` (rows of n_vocab floats, the last `positions` of them) to write the
// data from a full reference dump.
//   ds41_ref_test <model-00001.gguf> <engram.gguf> <mode> [dspark dir]
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"
#include "gen/session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <numeric>
#include <print>
#include <string>
#include <vector>

#ifndef MAKO_TEST_DATA
#define MAKO_TEST_DATA "tests/data"
#endif

namespace {

namespace fs = std::filesystem;
const fs::path kData = MAKO_TEST_DATA;
constexpr uint32_t kTopK = 64;
int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

std::vector<double> log_softmax(const float* x, size_t n)
{
    double m = x[0];
    for (size_t i = 1; i < n; ++i)
        m = std::max(m, double(x[i]));
    double s = 0;
    for (size_t i = 0; i < n; ++i)
        s += std::exp(double(x[i]) - m);
    const double lse = m + std::log(s);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = double(x[i]) - lse;
    return out;
}

std::vector<int32_t> read_ids(const fs::path& p)
{
    std::ifstream in(p);
    std::vector<int32_t> ids;
    for (std::string tok; std::getline(in, tok, ',');)
        if (!tok.empty())
            ids.push_back(std::stoi(tok));
    return ids;
}

struct RefTopK {
    uint32_t positions = 0, k = 0;
    std::vector<float> lse;
    std::vector<std::vector<std::pair<int32_t, float>>> top;
};

bool load_ref(const fs::path& p, RefTopK& r)
{
    std::ifstream in(p, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(&r.positions), 4) || !in.read(reinterpret_cast<char*>(&r.k), 4))
        return false;
    r.lse.resize(r.positions);
    r.top.assign(r.positions, std::vector<std::pair<int32_t, float>>(r.k));
    for (uint32_t i = 0; i < r.positions; ++i) {
        in.read(reinterpret_cast<char*>(&r.lse[i]), 4);
        for (auto& [id, v] : r.top[i]) {
            in.read(reinterpret_cast<char*>(&id), 4);
            in.read(reinterpret_cast<char*>(&v), 4);
        }
    }
    return bool(in);
}

int import_ref(const std::string& src, const std::string& dst, uint32_t n_vocab, uint32_t last)
{
    std::ifstream in(src, std::ios::binary | std::ios::ate);
    if (!in)
        return std::println(stderr, "cannot open {}", src), 1;
    const uint64_t rows = uint64_t(in.tellg()) / 4 / n_vocab, n = last && last < rows ? last : rows;
    in.seekg(int64_t((rows - n) * n_vocab * 4));
    std::ofstream out(dst, std::ios::binary);
    const uint32_t n32 = uint32_t(n), k = kTopK;
    out.write(reinterpret_cast<const char*>(&n32), 4);
    out.write(reinterpret_cast<const char*>(&k), 4);
    std::vector<float> x(n_vocab);
    std::vector<int32_t> order(n_vocab);
    for (uint64_t pos = 0; pos < n; ++pos) {
        in.read(reinterpret_cast<char*>(x.data()), int64_t(n_vocab) * 4);
        const auto ls = log_softmax(x.data(), n_vocab);
        const float lse = float(x[0] - ls[0]);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) { return x[a] > x[b]; });
        out.write(reinterpret_cast<const char*>(&lse), 4);
        for (uint32_t i = 0; i < k; ++i) {
            out.write(reinterpret_cast<const char*>(&order[i]), 4);
            out.write(reinterpret_cast<const char*>(&x[order[i]]), 4);
        }
    }
    std::println("wrote {} positions of top-{} reference logits to {}", n, k, dst);
    return 0;
}

// KL(ref || ours) over the reference's top-k and top-1 agreement; reference row i is position from + i.
struct Parity {
    const RefTopK& ref;
    uint32_t from = 0;
    double kl_sum = 0, agree = 0, mass = 0;
    uint32_t n = 0;

    void add(uint32_t pos, const float* logits, size_t n_vocab)
    {
        if (pos < from || pos - from >= ref.positions)
            return;
        const uint32_t r = pos - from;
        const auto ours = log_softmax(logits, n_vocab);
        double kl = 0, m = 0;
        for (const auto& [id, v] : ref.top[r]) {
            const double lp = v - ref.lse[r];
            kl += std::exp(lp) * (lp - ours[id]);
            m += std::exp(lp);
        }
        kl_sum += kl;
        mass += m;
        agree += int(std::max_element(logits, logits + n_vocab) - logits) == ref.top[r][0].first;
        ++n;
    }
    double kl() const { return n ? kl_sum / n : 0; }
    double top1() const { return n ? agree / n : 0; }
    void check_limits(const std::string& what, double kl_limit, double top1_limit) const
    {
        check(n == ref.positions && kl() < kl_limit && top1() > top1_limit,
              std::format("{} ({} positions): KL mean {:.5f} (limit {}), top-1 {:.2f}% (limit {}%), top-{} mass {:.3f}",
                          what, n, kl(), kl_limit, 100 * top1(), 100 * top1_limit, ref.k, n ? mass / n : 0));
    }
};

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc >= 4 && std::string_view(argv[1]) == "import")
        return import_ref(argv[2], argv[3], 129280, argc > 4 ? uint32_t(std::stoul(argv[4])) : 0);
    if (argc < 4) {
        std::println(stderr, "usage: ds41_ref_test <model-00001.gguf> <engram.gguf> short|long|penalty|q6k|dspark [dspark dir]");
        return 2;
    }
    const std::string mode = argv[3];
    if (!fs::exists(argv[1]) || !fs::exists(argv[2])) {
        std::println("model {} or Engram tables {} not found: skipped", argv[1], argv[2]);
        return 77;
    }
    const std::string dspark = argc > 4 ? argv[4] : "";
    if (mode == "dspark" && (dspark.empty() || !fs::exists(fs::path(dspark) / "dspark.json"))) {
        std::println("no drafter directory: skipped");
        return 77;
    }
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
    opts.max_ctx = mode == "long" ? 25600 : 4096;
    opts.prefill_exact_tail = mode == "long" ? 1024 : 4096; // the checked positions (all of a short prompt)
    if (mode == "q6k")
        opts.dense_quant = mako::gguf::Type::Q6_K;
    if (mode == "dspark")
        opts.dspark = dspark;
    mako::dsv41::Engine engine(*m, *cfg, opts, ctx);
    const size_t V = cfg->n_vocab;
    std::println("{}", mode);

    if (mode == "short" || mode == "penalty" || mode == "q6k") {
        const auto ids = read_ids(kData / "ds41_short_ids.txt");
        RefTopK ref;
        if (ids.empty() || !load_ref(kData / "ds41_short_ref.bin", ref) || ref.positions != ids.size()) {
            std::println(stderr, "missing or mismatched ds41_short data");
            return 1;
        }
        auto steps = [&](float penalty) {
            engine.reset();
            engine.set_miss_penalty(penalty);
            Parity p{ref};
            uint64_t misses = 0;
            for (uint32_t t = 0; t < ids.size(); ++t) {
                const auto& lg = engine.step(ids[t]);
                misses += engine.last_step().misses;
                p.add(t, lg.data(), V);
            }
            return std::pair{p, double(misses) / double(ids.size())};
        };
        if (mode == "short") {
            // Measured when set (full-vocabulary KL): steps 0.00572 / 98.2%, prefill 0.0061.
            const auto [p, misses] = steps(0);
            p.check_limits("decode steps vs reference", 0.0075, 0.97);
            engine.reset();
            Parity q{ref};
            engine.set_logits_sink([&](uint32_t pos, const float* lg) { q.add(pos, lg, V); });
            engine.prefill(ids);
            engine.set_logits_sink({});
            q.check_limits("prefill vs reference", 0.0080, 0.97);
        } else if (mode == "penalty") {
            // Measured when set: KL 0.0081 / 97.7%, misses 11.8 -> 4.6 a step.
            const auto [base, base_misses] = steps(0);
            const auto [pen, pen_misses] = steps(0.1f);
            pen.check_limits("miss penalty 0.1 vs reference", 0.0105, 0.965);
            check(pen_misses < 0.7 * base_misses,
                  std::format("miss penalty 0.1: {:.1f} misses a step, {:.1f} without", pen_misses, base_misses));
        } else {
            // Measured when set: KL 0.0068 / 98.2%.
            const auto [p, misses] = steps(0);
            p.check_limits("dense Q6_K steps vs reference", 0.0090, 0.965);
        }
    } else if (mode == "long") {
        const auto ids = read_ids(kData / "ds41_long_ids.txt");
        RefTopK ref;
        if (ids.empty() || !load_ref(kData / "ds41_long_ref.bin", ref) || ref.positions >= ids.size()) {
            std::println(stderr, "missing or mismatched ds41_long data");
            return 1;
        }
        const uint32_t from = uint32_t(ids.size() - ref.positions);
        // Measured when set (full-vocabulary KL): prefill 0.0102 / 96.9%, steps 0.0093 / 97.3%.
        for (const bool skip : {true, false}) {
            engine.reset();
            engine.set_prefill_skip(skip);
            Parity p{ref, from};
            engine.set_logits_sink([&](uint32_t pos, const float* lg) { p.add(pos, lg, V); });
            const auto t0 = std::chrono::steady_clock::now();
            engine.prefill(ids);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            engine.set_logits_sink({});
            p.check_limits(std::format("prefill of {} tokens ({}, {:.1f} s) vs reference at {}..", ids.size(),
                                       skip ? "early rows skip layers" : "every row", s, from),
                           0.0130, 0.955);
        }
        engine.set_prefill_skip(true);
        engine.reset();
        engine.prefill(std::span<const int32_t>(ids).first(from));
        Parity q{ref, from};
        for (uint32_t t = from; t < ids.size(); ++t)
            q.add(t, engine.step(ids[t]).data(), V);
        q.check_limits(std::format("decode steps after a {}-token prefill vs reference", from), 0.0125, 0.96);
    } else if (mode == "dspark") {
        const auto ids = read_ids(kData / "ds41_short_ids.txt");
        const std::vector<int32_t> prompt(ids.begin(), ids.begin() + 300);
        mako::gen::GenerateParams gp;
        gp.sampling.temperature = 0;
        gp.max_tokens = 96;
        auto run = [&](bool spec) {
            engine.reset();
            mako::gen::Session session(engine);
            gp.speculative = spec;
            return session.generate(prompt, gp);
        };
        const auto plain = run(false), drafted = run(true);
        check(drafted.tokens == plain.tokens,
              std::format("greedy tokens with drafts equal those without ({} tokens)", plain.tokens.size()));
        check(drafted.drafts > 0 && drafted.accepted * 4 >= drafted.drafts,
              std::format("drafts kept: {} of {}", drafted.accepted, drafted.drafts));
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
