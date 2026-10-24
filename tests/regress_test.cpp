// End-to-end regression tests on the real model, one engine for all cases:
//   decode       greedy tokens of a short prompt match the recorded ones
//   prefill      prefill at several chunk sizes gives token-by-token decode's logits and continuation
//   parity       per-position logits stay close to the reference implementation's (stored top-k)
//   long         a prompt past the dense-attention window (sparse selection, lent arena slots) against the
//                reference, then decode
//   regress_test <model.gguf> [case ...] [--update]
//   regress_test - --import-ref <logits dir> parity|long
// --update rewrites the recorded tokens and logits from this run; --import-ref turns a reference run's logits
// (logits.<pos>.f32) for a case's prompt (data/<case>_ids.txt) into its stored top-k (data/<case>_ref.bin).
// Exit 77: the model is missing (skipped).
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "text/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
const fs::path kData = MAKO_TEST_DATA;
const fs::path kExpected = kData / "regress_expected.txt";

// Tolerances, from the spread measured when they were set (see each use).
// Prefill vs decode on the short prompt: KL between their distributions and the largest difference among decode's
// top-20 logits (a maximum over the whole vocabulary measures f16 noise in the far tail instead). Measured when
// set: KL 1.3e-4, top-20 difference 0.052, the same at every chunk size.
constexpr double kPrefillKl = 2e-4;
constexpr double kPrefillTopDiff = 0.06;
constexpr double kLongTopDiff = 0.05; // the long prompt's top logit vs recorded
constexpr int kTopK = 64;                // reference logits stored per position

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

std::vector<int32_t> parse_ids(const std::string& s)
{
    std::vector<int32_t> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty())
            v.push_back(std::stoi(t));
    return v;
}

std::string join(const std::vector<int32_t>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i)
        s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

std::string read_file(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int argmax(const std::vector<float>& v) { return int(std::max_element(v.begin(), v.end()) - v.begin()); }

std::vector<double> log_softmax(const float* x, size_t n)
{
    const double m = *std::max_element(x, x + n);
    double s = 0;
    for (size_t i = 0; i < n; ++i)
        s += std::exp(x[i] - m);
    const double lse = m + std::log(s);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] - lse;
    return out;
}

// Recorded values: one "key value" per line.
std::map<std::string, std::string> load_expected()
{
    std::map<std::string, std::string> m;
    std::ifstream in(kExpected);
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#')
            continue;
        const auto sp = line.find(' ');
        m[line.substr(0, sp)] = sp == std::string::npos ? "" : line.substr(sp + 1);
    }
    return m;
}

struct RefTopK {
    uint32_t positions = 0, k = 0;
    std::vector<float> lse;                          // [pos]
    std::vector<std::vector<std::pair<int32_t, float>>> top; // [pos] (id, logit), best first
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

int import_ref(const fs::path& dir, const std::string& name)
{
    const uint32_t n = uint32_t(parse_ids(read_file(kData / (name + "_ids.txt"))).size()), k = kTopK;
    if (!n) {
        std::println(stderr, "no prompt {}", name);
        return 1;
    }
    std::ofstream out(kData / (name + "_ref.bin"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(reinterpret_cast<const char*>(&k), 4);
    for (uint32_t pos = 0; pos < n; ++pos) {
        const std::string raw = read_file(dir / std::format("logits.{}.f32", pos));
        if (raw.empty()) {
            std::println(stderr, "missing reference logits at position {}", pos);
            return 1;
        }
        const auto* x = reinterpret_cast<const float*>(raw.data());
        const size_t nv = raw.size() / 4;
        const auto ls = log_softmax(x, nv);
        const float lse = float(x[0] - ls[0]);
        std::vector<int32_t> order(nv);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) { return x[a] > x[b]; });
        out.write(reinterpret_cast<const char*>(&lse), 4);
        for (uint32_t i = 0; i < k; ++i) {
            out.write(reinterpret_cast<const char*>(&order[i]), 4);
            out.write(reinterpret_cast<const char*>(&x[order[i]]), 4);
        }
    }
    std::println("wrote {} positions of top-{} reference logits to {}_ref.bin", n, k, name);
    return 0;
}

// KL(ref || ours) over the reference's top-k (the rest of its mass is left out) and top-1 agreement, over the
// positions added from `from` on.
struct Parity {
    const RefTopK& ref;
    uint32_t from = 0;
    double kl_sum = 0, kl_max = 0, agree = 0, mass = 0;
    uint32_t n = 0;

    void add(uint32_t pos, const float* logits, size_t n_vocab)
    {
        if (pos < from || pos >= ref.positions)
            return;
        const auto ours = log_softmax(logits, n_vocab);
        double kl = 0, m = 0;
        for (const auto& [id, v] : ref.top[pos]) {
            const double lp = v - ref.lse[pos];
            kl += std::exp(lp) * (lp - ours[id]);
            m += std::exp(lp);
        }
        kl_sum += kl;
        kl_max = std::max(kl_max, kl);
        mass += m;
        agree += int(std::max_element(logits, logits + n_vocab) - logits) == ref.top[pos][0].first;
        ++n;
    }
    double kl_mean() const { return n ? kl_sum / n : 0; }
    double top1() const { return n ? agree / n : 0; }
    // Checks KL mean and top-1 agreement against limits.
    void check_limits(const std::string& what, double kl_limit, double top1_limit) const
    {
        check(n > 0 && kl_mean() < kl_limit && top1() > top1_limit,
              std::format("{} ({} positions): KL mean {:.5f} (limit {}), max {:.4f}, top-1 {:.2f}% (limit {}%), "
                          "top-{} mass {:.3f}",
                          what, n, kl_mean(), kl_limit, kl_max, 100 * top1(), 100 * top1_limit, ref.k,
                          n ? mass / n : 0));
    }
};

std::vector<int32_t> greedy(mako::glm::Engine& e, std::vector<float> logits, int n)
{
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        out.push_back(argmax(logits));
        if (i + 1 < n)
            logits = e.step(out.back());
    }
    return out;
}

std::vector<float> run_steps(mako::glm::Engine& e, const std::vector<int32_t>& ids,
                             std::vector<std::vector<float>>* all = nullptr)
{
    std::vector<float> l;
    for (int32_t t : ids) {
        l = e.step(t);
        if (all)
            all->push_back(l);
    }
    return l;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: regress_test <model.gguf> [case ...] [--update] [--import-ref <dir>]");
        return 2;
    }
    const std::string model_path = argv[1];
    bool update = false;
    std::string ref_dir;
    std::vector<std::string> cases;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--update")
            update = true;
        else if (a == "--import-ref" && i + 1 < argc) // the case name (parity, long) follows as a case
            ref_dir = argv[++i];
        else
            cases.push_back(a);
    }
    const auto want = [&](const char* c) { return cases.empty() || std::ranges::find(cases, c) != cases.end(); };
    if (!ref_dir.empty())
        return cases.size() == 1 ? import_ref(ref_dir, cases[0]) : 2;
    const auto parity_ids = parse_ids(read_file(kData / "parity_ids.txt"));
    if (model_path.empty() || !fs::exists(model_path)) {
        std::println("model {} not found: skipped", model_path);
        return 77;
    }

    auto model = mako::gguf::ModelFiles::open(model_path);
    auto cfg = model ? mako::glm::Config::load(*model) : std::unexpected(model.error());
    auto tok = model ? mako::text::Tokenizer::load(model->meta()) : std::unexpected(model.error());
    if (!model || !cfg || !tok) {
        std::println(stderr, "loading {} failed", model_path);
        return 1;
    }
    mako::mtl::Context ctx;
    mako::glm::Options opts;
    opts.prefill_chunk = 1024;
    mako::glm::Engine engine(*model, *cfg, opts, ctx);
    auto expected = load_expected();
    std::map<std::string, std::string> recorded;
    const auto compare = [&](const std::string& key, const std::string& got, const std::string& what) {
        recorded[key] = got;
        if (update)
            return;
        auto it = expected.find(key);
        check(it != expected.end() && it->second == got,
              it == expected.end() ? what + ": nothing recorded (run --update)"
                                   : what + (it->second == got ? "" : "\n         got " + got + "\n    expected " + it->second));
    };

    if (want("decode")) {
        std::println("decode");
        engine.reset();
        const std::vector<int32_t> prompt{785, 6722, 315};
        compare("decode_greedy", join(greedy(engine, run_steps(engine, prompt), 20)), "greedy tokens after 3-token prompt");
    }

    if (want("prefill")) {
        std::println("prefill");
        const std::vector<int32_t> prompt{785, 3840, 315, 279, 12745, 20406, 44038, 1657, 23566, 11, 504, 279,
                                          35056, 315, 279, 3283, 311, 279, 4399, 315, 90427, 1164, 304, 220,
                                          117721, 18, 11, 323, 1181, 10168, 374, 2058, 6476, 3351, 13};
        engine.reset();
        const auto ref = run_steps(engine, prompt);
        const auto ref_greedy = greedy(engine, ref, 12);
        compare("prefill_greedy", join(ref_greedy), "token-by-token greedy tokens");
        for (uint32_t chunk : {1024u, 512u, 8u}) {
            engine.reset();
            engine.set_prefill_chunk(chunk);
            const auto got = engine.prefill(prompt);
            const auto lr = log_softmax(ref.data(), ref.size()), lg = log_softmax(got.data(), got.size());
            double kl = 0;
            for (size_t i = 0; i < lr.size(); ++i)
                kl += std::exp(lr[i]) * (lr[i] - lg[i]);
            std::vector<int32_t> order(ref.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 20, order.end(), [&](int a, int b) { return ref[a] > ref[b]; });
            double diff = 0;
            for (int i = 0; i < 20; ++i)
                diff = std::max(diff, double(std::abs(got[order[i]] - ref[order[i]])));
            check(kl <= kPrefillKl && diff <= kPrefillTopDiff,
                  std::format("chunk {}: KL to decode {:.2e} (limit {:.0e}), top-20 logit difference {:.4f} (limit {})",
                              chunk, kl, kPrefillKl, diff, kPrefillTopDiff));
            check(greedy(engine, got, 12) == ref_greedy, std::format("chunk {}: same greedy continuation", chunk));
        }
        engine.set_prefill_chunk(1024);
    }

    if (want("parity")) {
        std::println("parity ({} positions)", parity_ids.size());
        RefTopK ref;
        if (!load_ref(kData / "parity_ref.bin", ref) || ref.positions != parity_ids.size()) {
            check(false, "reference top-k missing or not for this prompt");
        } else {
            Parity pf{ref};
            engine.reset();
            engine.set_logits_sink([&](uint32_t pos, const float* l) { pf.add(pos, l, cfg->n_vocab); });
            engine.prefill(parity_ids);
            engine.set_logits_sink({});
            // Limits ~1.1x the measured KL and a few positions of top-1; measured when set: KL 0.0086, 95.3%.
            pf.check_limits("prefill vs reference", 0.0095, 0.945);
            Parity dec{ref};
            engine.reset();
            for (uint32_t pos = 0; pos < 128; ++pos) {
                const auto& l = engine.step(parity_ids[pos]);
                dec.add(pos, l.data(), l.size());
            }
            // Measured when set: KL 0.0119, 94.5% (one position is 0.8%).
            dec.check_limits("decode vs reference", 0.013, 0.937);
        }
    }

    if (want("long")) {
        const auto ids = parse_ids(read_file(kData / "long_ids.txt"));
        std::println("long ({} tokens)", ids.size());
        check(tok->encode(read_file(MAKO_CALIBRATION), true) == ids, "corpus tokenizes to the stored ids");
        RefTopK ref;
        const bool have_ref = load_ref(kData / "long_ref.bin", ref) && ref.positions == ids.size();
        Parity all{ref}, sparse{ref, 2048};
        engine.reset();
        engine.set_logits_sink([&](uint32_t pos, const float* l) {
            if (have_ref) {
                all.add(pos, l, cfg->n_vocab);
                sparse.add(pos, l, cfg->n_vocab);
            }
        });
        const auto l = engine.prefill(ids);
        engine.set_logits_sink({});
        check(!engine.arena().lent(), "arena slots reclaimed after prefill");
        if (!have_ref) {
            check(false, "reference top-k missing or not for this prompt");
        } else {
            // Measured when set: all positions KL 0.0076, 96.8%; positions >= 2048 KL 0.0072, 95.6% (one is 0.3%).
            all.check_limits("prefill vs reference", 0.0084, 0.963);
            sparse.check_limits("prefill vs reference, sparse attention", 0.0079, 0.948);
        }
        const int top = argmax(l);
        recorded["long_top"] = std::format("{} {:.3f}", top, l[top]);
        if (!update) {
            int etop = -1;
            double ev = 0;
            std::sscanf(expected["long_top"].c_str(), "%d %lf", &etop, &ev);
            check(top == etop && std::abs(l[top] - ev) <= kLongTopDiff,
                  std::format("top logit {}:{:.3f}, recorded {}:{:.3f} (limit {})", top, l[top], etop, ev, kLongTopDiff));
        }
        compare("long_greedy", join(greedy(engine, l, 8)), "greedy tokens after the prompt");

        // Groups of 1024 tokens (experts loaded per group, in different batches): the same logits, bit for bit. With
        // a sink both times: the last position's logits then come from the same kernel.
        engine.reset();
        engine.set_prefill_group(1024);
        engine.set_logits_sink([](uint32_t, const float*) {});
        const auto l2 = engine.prefill(ids);
        engine.set_logits_sink({});
        engine.set_prefill_group(4096);
        check(l2 == l, "identical logits with 1024-token groups");
    }

    if (update) {
        for (const auto& [k, v] : recorded)
            expected[k] = v;
        std::ofstream out(kExpected, std::ios::trunc);
        out << "# Recorded by regress_test --update.\n";
        for (const auto& [k, v] : expected)
            out << k << ' ' << v << '\n';
        std::println("updated {}", kExpected.string());
        return 0;
    }
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
