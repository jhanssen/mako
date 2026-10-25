// End-to-end regression tests on the real model, one engine for all cases:
//   decode       greedy tokens of a short prompt match the recorded ones
//   prefill      prefill at several chunk sizes gives token-by-token decode's logits and continuation
//   parity       per-position logits stay close to the reference implementation's (stored top-k)
//   spec         a two-token step (speculative verification), and its rollback, match single-token steps
//   session      generation through Session: greedy and speculative tokens, prefix reuse from the current state
//                and from a checkpoint (the same logits as a fresh prefill), a repeated prompt, a read cancelled
//                between groups and resumed, drafts after a restore
//   ablate       per-layer direction removal (sites ffn_out and stream): the streams after an ablated layer have
//                no component along its directions, prefill and decode agree, all-zero directions and switching it
//                off change nothing
//   long         a prompt past the dense-attention window (sparse selection, lent arena slots) against the
//                reference, then decode
//   regress_test <model.gguf> [case ...] [--update]
//   regress_test - --import-ref <logits dir> parity|long
// --update rewrites the recorded tokens and logits from this run; --import-ref turns a reference run's logits
// (logits.<pos>.f32) for a case's prompt (data/<case>_ids.txt) into its stored top-k (data/<case>_ref.bin).
// Exit 77: the model is missing (skipped).
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gen/session.hpp"
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
#include <random>
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
    opts.mtp = true; // for spec; results do not depend on which experts are resident
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
            // Limits: ~1.1x the measured KL, and top-1 a point below the measured (a rounding-level change, such as a
            // reordered sum, moves it by ~0.4 points over a long prompt). Measured when set: KL 0.0086, 95.3%.
            pf.check_limits("prefill vs reference", 0.0095, 0.943);
            Parity dec{ref};
            engine.reset();
            for (uint32_t pos = 0; pos < 128; ++pos) {
                const auto& l = engine.step(parity_ids[pos]);
                dec.add(pos, l.data(), l.size());
            }
            // Measured when set: KL 0.0119, 94.5% (one position is 0.8%).
            dec.check_limits("decode vs reference", 0.013, 0.935);
        }
    }

    if (want("spec")) {
        std::println("spec");
        const std::vector<int32_t> prompt{785, 3840, 315, 279, 12745, 20406, 44038, 1657, 23566, 11, 504, 279};
        const int32_t a = 35056, b = 315, c = 279, wrong = 13;
        const size_t nv = cfg->n_vocab;
        auto close = [&](const float* got, const std::vector<float>& want, const std::string& what) {
            const auto lr = log_softmax(want.data(), nv), lg = log_softmax(got, nv);
            double kl = 0;
            for (size_t i = 0; i < nv; ++i)
                kl += std::exp(lr[i]) * (lr[i] - lg[i]);
            std::vector<int32_t> order(nv);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 20, order.end(),
                              [&](int x, int y) { return want[x] > want[y]; });
            double diff = 0;
            for (int i = 0; i < 20; ++i)
                diff = std::max(diff, double(std::abs(got[order[i]] - want[order[i]])));
            // Measured when set: KL <= 7e-9, top-20 difference <= 0.0005.
            check(kl <= 1e-6 && diff <= 0.01,
                  std::format("{}: KL {:.2e}, top-20 logit difference {:.4f}", what, kl, diff));
        };
        engine.reset();
        engine.prefill(prompt);
        const auto ra = engine.step(a), rb = engine.step(b), rc = engine.step(c);
        engine.reset();
        engine.prefill(prompt);
        const auto two = engine.step2(a, b);
        close(two.data(), ra, "two-token step, first token");
        close(two.data() + nv, rb, "two-token step, second token");
        engine.reset();
        engine.prefill(prompt);
        engine.step2(a, wrong);
        engine.rollback();
        const auto again = engine.step2(b, c);
        close(again.data(), rb, "after a rollback, first token");
        close(again.data() + nv, rc, "after a rollback, second token");

        // Speculative greedy decoding: every emitted token is the verifier's greedy choice; the next-token block's
        // drafts are accepted at a plausible rate.
        if (engine.has_mtp()) {
            auto argmax_row = [&](const float* l) { return int32_t(std::max_element(l, l + nv) - l); };
            engine.reset();
            const auto& lp = engine.prefill(prompt);
            int32_t pending = argmax_row(lp.data()), draft = engine.mtp_draft(pending);
            int rounds = 0, accepted = 0, emitted = 0;
            while (emitted < 48) {
                const auto& two = engine.step2(pending, draft);
                const int32_t g1 = argmax_row(two.data());
                ++emitted;
                ++rounds;
                if (g1 == draft) {
                    ++accepted;
                    ++emitted;
                    const int32_t g2 = argmax_row(two.data() + nv);
                    const int32_t both[2] = {draft, g2};
                    draft = engine.mtp_draft(std::span<const int32_t>(both));
                    pending = g2;
                } else {
                    engine.rollback();
                    draft = engine.mtp_draft(g1);
                    pending = g1;
                }
            }
            // Measured when set: ~67-75% on similar text.
            check(accepted > rounds / 2, std::format("drafts accepted: {} of {} rounds (limit half)", accepted, rounds));
        }
    }

    if (want("session")) {
        std::println("session");
        const std::vector<int32_t> prompt{785, 3840, 315, 279, 12745, 20406, 44038, 1657, 23566, 11, 504, 279,
                                          35056, 315, 279, 3283, 311, 279, 4399, 315, 90427, 1164, 304, 220,
                                          117721, 18, 11, 323, 1181, 10168, 374, 2058, 6476, 3351, 13};
        const uint32_t nv = cfg->n_vocab;
        auto kl_to = [&](const std::vector<float>& want, const std::vector<float>& got) {
            const auto lw = log_softmax(want.data(), nv), lg = log_softmax(got.data(), nv);
            double kl = 0;
            for (uint32_t i = 0; i < nv; ++i)
                kl += std::exp(lw[i]) * (lw[i] - lg[i]);
            return kl;
        };
        auto fresh = [&](const std::vector<int32_t>& p) {
            engine.reset();
            return engine.prefill(p);
        };
        mako::gen::GenerateParams greedy_p;
        greedy_p.sampling.temperature = 0;
        greedy_p.max_tokens = 24;
        greedy_p.speculative = false;

        // As a session reads a prompt: all but its last token, then that one as a step (the checkpoint goes between).
        auto as_session = [&](std::span<const int32_t> p) -> const std::vector<float>& {
            if (p.size() > 1)
                engine.prefill(p.first(p.size() - 1));
            return engine.step(p.back());
        };

        // Greedy without drafts: the plain decode loop's tokens.
        engine.reset();
        const auto want = greedy(engine, as_session(prompt), 24);
        mako::gen::Session session(engine);
        auto r = session.generate(prompt, greedy_p);
        check(r.tokens == want && r.finish == mako::gen::Finish::max_tokens, "greedy session tokens match the decode loop");

        // Speculative greedy: each token is the argmax of the logits before it (teacher-forced over prompt + output),
        // except at near-ties, where the two-token step's rounding may choose the other.
        if (engine.has_mtp()) {
            session.clear();
            auto sp = greedy_p;
            sp.speculative = true;
            sp.max_tokens = 48;
            const auto rs = session.generate(prompt, sp);
            std::vector<int32_t> all = prompt;
            all.insert(all.end(), rs.tokens.begin(), rs.tokens.end());
            std::map<uint32_t, std::vector<float>> forced;
            engine.reset();
            engine.set_logits_sink([&](uint32_t pos, const float* l) {
                if (pos + 1 >= prompt.size())
                    forced[pos].assign(l, l + nv);
            });
            engine.prefill(all);
            engine.set_logits_sink({});
            int off = 0, ties = 0;
            for (size_t i = 0; i < rs.tokens.size(); ++i) {
                const auto& l = forced[uint32_t(prompt.size() - 1 + i)];
                const int32_t best = argmax(l);
                if (best != rs.tokens[i])
                    ++(l[best] - l[rs.tokens[i]] < 0.05f ? ties : off);
            }
            check(off == 0 && rs.tokens.size() == 48,
                  std::format("speculative greedy tokens are the teacher-forced argmax ({} near-ties, {} not)", ties, off));
            // Measured when set: ~67-75% on similar text.
            check(rs.accepted * 2 > rs.drafts, std::format("drafts accepted: {} of {}", rs.accepted, rs.drafts));
            session.clear();
        }

        // Prefix reuse. Continuing the context (prompt, its output, more text) keeps every token; a prompt diverging
        // right after the first one, or the same prompt again, restores the checkpoint taken before its last token.
        // Each gives the logits of the same tokens run without a session, bit for bit (prefill split at any position
        // gives the same logits; the generated tokens ran as decode steps, whose rounding differs from prefill's).
        session.clear();
        r = session.generate(prompt, greedy_p);
        std::vector<int32_t> cont = prompt;
        cont.insert(cont.end(), r.tokens.begin(), r.tokens.end());
        cont.insert(cont.end(), {13, 785, 6722, 315});
        const size_t held = session.context().size();
        auto zero = greedy_p;
        zero.max_tokens = 0;
        auto rc = session.generate(cont, zero);
        const auto l_cont = engine.logits();
        engine.reset();
        as_session(std::span(cont).first(prompt.size()));
        for (size_t i = prompt.size(); i < held; ++i)
            engine.step(cont[i]);
        const double kl_cont = kl_to(as_session(std::span(cont).subspan(held)), l_cont);
        std::vector<int32_t> branch = prompt;
        branch.insert(branch.end(), {576, 1052, 374, 264, 1273});
        auto rb = session.generate(branch, zero);
        const auto l_branch = engine.logits();
        engine.reset();
        const double kl_branch = kl_to(as_session(branch), l_branch);
        auto ra = session.generate(branch, zero);
        const auto l_again = engine.logits();
        check(rc.reused_tokens == held, std::format("continuation reuses the {} tokens held ({})", held, rc.reused_tokens));
        check(rb.reused_tokens == prompt.size() - 1,
              std::format("diverging prompt restores the checkpoint at {} ({})", prompt.size() - 1, rb.reused_tokens));
        check(ra.reused_tokens == branch.size() - 1,
              std::format("the same prompt again restores the checkpoint at {} ({})", branch.size() - 1, ra.reused_tokens));
        check(kl_cont == 0 && kl_branch == 0 && kl_to(l_branch, l_again) == 0,
              std::format("logits as without a session: continued KL {:.2e}, restored KL {:.2e}, again KL {:.2e}",
                          kl_cont, kl_branch, kl_to(l_branch, l_again)));

        // Reading a prompt cancelled between groups keeps the groups read; the next request continues from there,
        // with the logits of an uninterrupted read.
        {
            const auto corpus = parse_ids(read_file(kData / "long_ids.txt"));
            engine.set_prefill_group(1024);
            session.clear();
            auto stop_now = zero;
            int polls = 0;
            stop_now.cancelled = [&] { return ++polls > 0; };
            const auto rx = session.generate(corpus, stop_now);
            const size_t kept = session.context().size();
            const auto ry = session.generate(corpus, zero);
            const auto l_resumed = engine.logits();
            engine.reset();
            engine.prefill(std::span(corpus).first(kept));
            const double kl_resumed = kl_to(as_session(std::span(corpus).subspan(kept)), l_resumed);
            engine.set_prefill_group(opts.prefill_group);
            check(rx.finish == mako::gen::Finish::cancelled && rx.tokens.empty() && polls == 1 &&
                      kept == 1024 && ry.reused_tokens == kept && kl_resumed == 0,
                  std::format("cancelled read keeps {} of {} tokens ({} polls), resumed with {} reused, KL {:.2e}", kept,
                              corpus.size(), polls, ry.reused_tokens, kl_resumed));
        }

        // After a restore, the next-token block's cache at the checkpoint's last position is refilled for the new
        // next token: its guesses as after a fresh prefill (that row came from prefill there, from a decode-path pass
        // here, so not bit for bit).
        if (engine.has_mtp()) {
            const std::vector<int32_t> probes{13, 11, 279, 315, 323, 264, 374, 220};
            auto drafts = [&] {
                std::vector<std::vector<float>> d;
                for (int32_t x : probes) {
                    engine.mtp_draft(x);
                    const auto l = engine.mtp_logits();
                    d.emplace_back(l.begin(), l.end());
                }
                return d;
            };
            fresh(branch);
            const auto want_d = drafts();
            auto restored = [&](bool seed) {
                engine.reset();
                engine.prefill(prompt);
                const auto cp = engine.checkpoint();
                engine.mtp_draft(13); // as if the answer had started with another token
                engine.step(576);
                engine.mtp_draft(1);
                engine.step(1);
                engine.mtp_draft(2);
                engine.restore(cp);
                if (seed)
                    engine.mtp_seed(branch[prompt.size()]);
                engine.prefill(std::span(branch).subspan(prompt.size()));
                const auto d = drafts();
                double worst = 0;
                for (size_t i = 0; i < d.size(); ++i)
                    worst = std::max(worst, kl_to(want_d[i], d[i]));
                return worst;
            };
            const double seeded = restored(true), unseeded = restored(false);
            // Measured when set: 2.5e-7 (stale row: 4.6e-2).
            check(seeded <= 1e-5 && seeded * 10 < unseeded,
                  std::format("drafts after a restore and seed: worst KL to a fresh prefill's {:.2e} (unseeded {:.2e})",
                              seeded, unseeded));
        }
        session.clear();
    }

    if (want("ablate")) {
        std::println("ablate");
        const std::vector<int32_t> prompt{785, 3840, 315, 279, 12745, 20406, 44038, 1657, 23566, 11, 504, 279,
                                          35056, 315, 279, 3283, 311, 279, 4399, 315, 90427, 1164, 304, 220};
        const uint32_t E = cfg->n_embd, NL = cfg->n_layer, nv = cfg->n_vocab, k = 2;
        auto kl = [&](const std::vector<float>& want, const std::vector<float>& got) {
            const auto lw = log_softmax(want.data(), nv), lg = log_softmax(got.data(), nv);
            double d = 0;
            for (uint32_t i = 0; i < nv; ++i)
                d += std::exp(lw[i]) * (lw[i] - lg[i]);
            return d;
        };
        auto split = [&]() -> std::vector<float> { // prefill all but the last token, step that one
            engine.reset();
            engine.prefill(std::span(prompt).first(prompt.size() - 1));
            return engine.step(prompt.back());
        };
        auto whole = [&]() -> std::vector<float> {
            engine.reset();
            return engine.prefill(prompt);
        };
        const auto base = whole(), base_split = split();

        // Random orthonormal directions for every layer but the first, which stays all zero.
        std::mt19937 rng(11);
        std::normal_distribution<float> nd;
        std::vector<float> dirs(uint64_t(NL) * k * E, 0.0f);
        for (uint32_t l = 1; l < NL; ++l)
            for (uint32_t d = 0; d < k; ++d) {
                float* v = dirs.data() + (uint64_t(l) * k + d) * E;
                for (uint32_t i = 0; i < E; ++i)
                    v[i] = nd(rng);
                for (uint32_t p = 0; p < d; ++p) {
                    const float* u = dirs.data() + (uint64_t(l) * k + p) * E;
                    double dot = 0;
                    for (uint32_t i = 0; i < E; ++i)
                        dot += double(v[i]) * u[i];
                    for (uint32_t i = 0; i < E; ++i)
                        v[i] -= float(dot) * u[i];
                }
                double n = 0;
                for (uint32_t i = 0; i < E; ++i)
                    n += double(v[i]) * v[i];
                for (uint32_t i = 0; i < E; ++i)
                    v[i] /= float(std::sqrt(n));
            }

        // The engine's file layout: [layer][site][k][n_embd], the same directions at both sites.
        auto ablation = [&](const std::vector<float>& per_layer, uint32_t kk) {
            const std::vector<std::string> names{"ffn_out", "stream"};
            std::vector<float> v(uint64_t(NL) * names.size() * kk * E);
            for (uint32_t l = 0; l < NL; ++l)
                for (size_t s = 0; s < names.size(); ++s)
                    std::copy_n(per_layer.data() + uint64_t(l) * kk * E, uint64_t(kk) * E,
                                v.data() + (uint64_t(l) * names.size() + s) * kk * E);
            return mako::mtl::Ablation::from_vectors(ctx, E, NL, names, kk, v, mako::glm::Engine::kAblationSites);
        };
        engine.set_ablation(ablation(std::vector<float>(uint64_t(NL) * E, 0.0f), 1));
        const bool zero_same = whole() == base && split() == base_split;
        engine.set_ablation(ablation(dirs, k));
        const auto abl = whole(), abl_split = split();

        // The streams after layer l: probed while the last token runs as a step.
        std::map<int, std::vector<float>> streams;
        engine.reset();
        engine.prefill(std::span(prompt).first(prompt.size() - 1));
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (name == "l_out")
                streams[layer].assign(data, data + n);
        });
        engine.step(prompt.back());
        engine.set_probe({});
        double worst_on = 0, worst_off = 0; // |stream . v| / |stream|, ablated layers and the first
        for (const auto& [l, r] : streams)
            for (uint32_t h = 0; h < r.size() / E; ++h)
                for (uint32_t d = 0; d < k; ++d) {
                    const float* x = r.data() + uint64_t(h) * E;
                    const uint32_t ld = l == 0 ? 1 : uint32_t(l); // the first layer: measured against layer 1's
                    const float* v = dirs.data() + (uint64_t(ld) * k + d) * E;
                    double dot = 0, n = 0;
                    for (uint32_t i = 0; i < E; ++i) {
                        dot += double(x[i]) * v[i];
                        n += double(x[i]) * x[i];
                    }
                    (l == 0 ? worst_off : worst_on) = std::max(l == 0 ? worst_off : worst_on, std::abs(dot) / std::sqrt(n));
                }
        engine.set_ablation(std::nullopt);
        const bool off_same = whole() == base;

        check(zero_same, "all-zero directions: the same logits, bit for bit");
        check(streams.size() == NL && worst_on < 1e-4 && worst_off > 1e-3,
              std::format("streams after an ablated layer along its directions: worst {:.1e} of their norm (a "
                          "layer left alone: {:.1e})",
                          worst_on, worst_off));
        const double kl_paths = kl(abl, abl_split), kl_paths_base = kl(base, base_split), effect = kl(base, abl);
        // Measured when set: prefill vs decode 2.8e-9 (6.4e-10 without); the random directions' effect 1.2e-4.
        check(kl_paths < 10 * kl_paths_base + 1e-7 && effect > 1000 * kl_paths,
              std::format("prefill and decode agree when ablating (KL {:.1e}; without {:.1e}); effect KL {:.1e}",
                          kl_paths, kl_paths_base, effect));
        check(off_same, "switched off: the logits as before, bit for bit");
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
            all.check_limits("prefill vs reference", 0.0084, 0.958);
            sparse.check_limits("prefill vs reference, sparse attention", 0.0079, 0.946);
        }
        // Exact tokens are not recorded here: over this many positions, rounding-level changes alone move the last
        // logits by tenths. Decode after the long prefill (reclaimed arena slots) must still work.
        std::vector<float> dl = l;
        for (int i = 0; i < 8; ++i)
            dl = engine.step(argmax(dl));
        check(std::ranges::all_of(dl, [](float v) { return std::isfinite(v); }), "decode after the long prefill");

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
