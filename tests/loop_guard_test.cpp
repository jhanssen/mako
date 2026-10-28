// LoopGuard: exact and near loops found where they start, varied and structured text left alone. No model needed.
#include "gen/loop_guard.hpp"

#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

using mako::gen::LoopGuard;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

std::mt19937 rng(7);

std::vector<int32_t> random_tokens(size_t n)
{
    std::uniform_int_distribution<int32_t> d(0, 150000);
    std::vector<int32_t> v(n);
    for (auto& t : v)
        t = d(rng);
    return v;
}

// Pushes tokens until the guard fires: the count pushed then, 0 if it never did.
size_t first_hit(LoopGuard& g, const std::vector<int32_t>& tokens)
{
    for (size_t i = 0; i < tokens.size(); ++i)
        if (g.push(tokens[i]))
            return i + 1;
    return 0;
}

} // namespace

int main()
{
    {
        LoopGuard g(256);
        check(first_hit(g, random_tokens(20000)) == 0, "20000 varied tokens: no loop");
    }
    {
        // 100 tokens, then one 60-token paragraph over and over.
        auto v = random_tokens(100);
        const auto para = random_tokens(60);
        for (int i = 0; i < 20; ++i)
            v.insert(v.end(), para.begin(), para.end());
        LoopGuard g(256);
        const size_t at = first_hit(g, v);
        check(at == 100 + 60 + 256 && g.start() == 100 && g.period() == 60,
              std::format("a repeated paragraph: found after {} tokens, from {} with period {}", at, g.start(),
                          g.period()));
    }
    {
        // The same paragraph with a counter that changes every time.
        auto v = random_tokens(100);
        auto para = random_tokens(60);
        for (int i = 0; i < 20; ++i) {
            para[5] = 1000 + i;
            v.insert(v.end(), para.begin(), para.end());
        }
        LoopGuard g(256);
        const size_t at = first_hit(g, v);
        check(at > 0 && at <= 100 + 60 * 11 && g.period() == 60 && g.start() == 100,
              std::format("a paragraph with a counter: found after {} tokens, from {} with period {}", at, g.start(),
                          g.period()));
    }
    {
        // Table rows of 10 tokens, 3 of them different in each row (70% the same): structured, not a loop.
        std::vector<int32_t> v;
        const auto row = random_tokens(10);
        for (int i = 0; i < 300; ++i) {
            auto r = row;
            for (int j : {1, 4, 7})
                r[size_t(j)] = int32_t(rng() % 150000);
            v.insert(v.end(), r.begin(), r.end());
        }
        LoopGuard g(256);
        check(first_hit(g, v) == 0, "3000 tokens of table rows differing in 3 of 10 tokens: no loop");
    }
    {
        LoopGuard g(256);
        const size_t at = first_hit(g, std::vector<int32_t>(400, 42));
        check(at == 257 && g.period() == 1 && g.start() == 0, std::format("one token repeated: found after {}", at));
    }
    {
        auto v = random_tokens(50);
        const auto para = random_tokens(60);
        for (int i = 0; i < 3; ++i)
            v.insert(v.end(), para.begin(), para.end());
        v.insert(v.end(), v.begin(), v.begin() + 50);
        LoopGuard g(256);
        check(first_hit(g, v) == 0, "a paragraph three times (fewer than 4): no loop");
    }
    {
        LoopGuard g(256);
        first_hit(g, std::vector<int32_t>(200, 42));
        g.reset();
        check(first_hit(g, std::vector<int32_t>(200, 42)) == 0, "reset forgets the run so far");
        LoopGuard off(0);
        check(first_hit(off, std::vector<int32_t>(5000, 42)) == 0, "min_tokens 0: off");
    }
    std::println("{}", failures ? std::format("{} failed", failures) : "all passed");
    return failures ? 1 : 0;
}
