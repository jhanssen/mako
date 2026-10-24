// Sampler: distributions of sample() and verify() against the target probabilities (Monte Carlo), truncation and
// penalties. No model needed.
#include "gen/sampler.hpp"

#include <cmath>
#include <cstdio>
#include <print>
#include <vector>

using mako::gen::Sampler;
using mako::gen::SamplingParams;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

std::vector<double> softmax(const std::vector<float>& l, float t)
{
    std::vector<double> p(l.size());
    double m = -1e30, s = 0;
    for (float v : l)
        m = std::max(m, double(v));
    for (size_t i = 0; i < l.size(); ++i)
        s += p[i] = std::exp((l[i] - m) / t);
    for (double& v : p)
        v /= s;
    return p;
}

template <typename F> std::vector<double> histogram(size_t n, int draws, F&& f)
{
    std::vector<double> h(n, 0);
    for (int i = 0; i < draws; ++i)
        h[size_t(f())] += 1.0 / draws;
    return h;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

} // namespace

int main()
{
    const std::vector<float> logits = {2.0f, 1.5f, 0.3f, -0.5f, 1.9f, -2.0f, 0.0f, 1.0f};
    const size_t n = logits.size();
    constexpr int kDraws = 400000; // probabilities to within ~0.003
    constexpr double kTol = 0.004;

    {
        Sampler s({.temperature = 0.0f}, uint32_t(n), 1);
        check(s.sample(logits.data()) == 0, "greedy picks the largest logit");
        bool acc = true;
        check(s.verify(logits.data(), 4, acc) == 0 && !acc, "greedy verification rejects a non-argmax draft");
        check(s.verify(logits.data(), 0, acc) == 0 && acc, "greedy verification accepts the argmax draft");
    }
    for (float t : {1.0f, 0.7f}) {
        Sampler s({.temperature = t}, uint32_t(n), 2);
        const auto want = softmax(logits, t);
        const auto got = histogram(n, kDraws, [&] { return s.sample(logits.data()); });
        char what[96];
        std::snprintf(what, sizeof what, "sample() at temperature %.1f matches softmax (max diff %.4f)", t,
                      max_diff(got, want));
        check(max_diff(got, want) < kTol, what);
        for (int draft : {0, 4, 5}) {
            Sampler v({.temperature = t}, uint32_t(n), 3 + draft);
            int accepted = 0;
            const auto vh = histogram(n, kDraws, [&] {
                bool acc = false;
                const int32_t r = v.verify(logits.data(), draft, acc);
                accepted += acc;
                return r;
            });
            std::snprintf(what, sizeof what, "verify(draft %d) matches softmax (max diff %.4f), accepts %.3f of %.3f",
                          draft, max_diff(vh, want), double(accepted) / kDraws, want[size_t(draft)]);
            check(max_diff(vh, want) < kTol && std::abs(double(accepted) / kDraws - want[size_t(draft)]) < kTol,
                  what);
        }
    }
    {
        // top_k 3 keeps tokens 0, 4, 1 (logits 2.0, 1.9, 1.5), renormalized.
        Sampler s({.temperature = 1.0f, .top_k = 3}, uint32_t(n), 7);
        auto want = softmax({2.0f, 1.5f, -1e9f, -1e9f, 1.9f, -1e9f, -1e9f, -1e9f}, 1.0f);
        const auto got = histogram(n, kDraws, [&] { return s.sample(logits.data()); });
        check(max_diff(got, want) < kTol, "top_k keeps the k most likely, renormalized");
        Sampler v({.temperature = 1.0f, .top_k = 3}, uint32_t(n), 8);
        const auto vh = histogram(n, kDraws, [&] {
            bool acc = false;
            return v.verify(logits.data(), 7, acc); // a draft outside the top 3 is never accepted
        });
        check(max_diff(vh, want) < kTol, "verify() of a truncated-away draft samples the truncated distribution");
    }
    {
        // Probabilities at T=1: 0.297 0.180 0.054 0.024 0.269 0.004 0.040 0.109; top_p 0.7 needs 0, 4, 1.
        Sampler s({.temperature = 1.0f, .top_p = 0.7f}, uint32_t(n), 9);
        auto want = softmax({2.0f, 1.5f, -1e9f, -1e9f, 1.9f, -1e9f, -1e9f, -1e9f}, 1.0f);
        const auto got = histogram(n, kDraws, [&] { return s.sample(logits.data()); });
        check(max_diff(got, want) < kTol, "top_p keeps the smallest set reaching p");
    }
    {
        // min_p 0.3: tokens with p >= 0.3 * p(0): logits >= 2 + ln 0.3 = 0.796 -> 0, 1, 4, 7.
        Sampler s({.temperature = 1.0f, .min_p = 0.3f}, uint32_t(n), 10);
        auto want = softmax({2.0f, 1.5f, -1e9f, -1e9f, 1.9f, -1e9f, -1e9f, 1.0f}, 1.0f);
        const auto got = histogram(n, kDraws, [&] { return s.sample(logits.data()); });
        check(max_diff(got, want) < kTol, "min_p drops tokens below p times the most likely");
    }
    {
        Sampler s({.temperature = 1.0f, .presence_penalty = 0.5f, .frequency_penalty = 0.25f}, uint32_t(n), 11);
        s.accept(0);
        s.accept(0);
        auto pen = logits;
        pen[0] -= 0.5f + 0.25f * 2;
        const auto got = histogram(n, kDraws, [&] { return s.sample(logits.data()); });
        check(max_diff(got, softmax(pen, 1.0f)) < kTol, "presence and frequency penalties");
        Sampler r({.temperature = 1.0f, .repetition_penalty = 2.0f}, uint32_t(n), 12);
        r.accept(0);
        r.accept(5);
        auto rp = logits;
        rp[0] /= 2.0f;
        rp[5] *= 2.0f;
        const auto rh = histogram(n, kDraws, [&] { return r.sample(logits.data()); });
        check(max_diff(rh, softmax(rp, 1.0f)) < kTol, "repetition penalty");
    }
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
