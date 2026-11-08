// gen::LookupDrafter: drafts from repeats of the context. A repeated passage proposes its continuation (as many
// drafts as its match length allows, never past the current position); a trigram alone is below min_match; the gate
// stops a match length whose first drafts keep being rejected, probing one in eight, and recovers; the index is
// rebuilt after the context is cut back.
//   lookup_test
#include "gen/lookup.hpp"

#include <print>
#include <string>
#include <vector>

namespace {

int failed = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "ok  " : "FAIL", what);
    if (!ok)
        ++failed;
}

std::vector<int32_t> seq(std::initializer_list<int32_t> l) { return l; }

} // namespace

int main()
{
    using mako::gen::LookupDrafter;
    // A passage of 20 tokens, other text, then the passage's first 9 tokens again.
    std::vector<int32_t> passage;
    for (int32_t t = 100; t < 120; ++t)
        passage.push_back(t);
    std::vector<int32_t> ctx = passage;
    ctx.insert(ctx.end(), {7, 8, 9, 10, 11, 12});
    ctx.insert(ctx.end(), passage.begin(), passage.begin() + 9);
    {
        LookupDrafter d;
        const auto got = d.propose(ctx, 109, 4); // the passage's 10th token: a match of 10
        check(got == seq({110, 111, 112, 113, 114, 115, 116}), "a long repeat proposes 7 drafts, the passage's continuation");
        check(d.propose(ctx, 50, 4).empty(), "an unseen trigram proposes nothing");
    }
    {
        // Only the trigram repeats (3 tokens): below min_match 4, proposed at 3.
        std::vector<int32_t> c{1, 2, 3, 4, 5, 6, 7, 8, 2, 3};
        LookupDrafter d;
        check(d.propose(c, 4, 4).empty(), "a 3-token match is below min_match 4");
        check(d.propose(c, 4, 3) == seq({5, 6, 7}), "at min_match 3 it proposes 3 drafts");
    }
    {
        // A match of 5 proposes 3; of 6, 5; drafts never reach the current position.
        std::vector<int32_t> c{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4};
        LookupDrafter d;
        check(d.propose(c, 5, 4) == seq({6, 7, 8}), "a match of 5 proposes 3");
        c.push_back(5);
        check(d.propose(c, 6, 4) == seq({7, 8, 9, 10, 1}), "a match of 6 proposes 5");
        // Periodic text: the best match (of 6) ends 2 tokens before the current position, so 2 drafts, not 5.
        std::vector<int32_t> tail{1, 2, 3, 1, 2, 3, 1, 2};
        check(d.propose(tail, 3, 4) == seq({1, 2}), "drafts stop before the current position");
    }
    {
        // The gate: rejections close the bucket (probes one in eight), acceptances reopen it.
        LookupDrafter d;
        int offered = 0, probes = 0;
        for (int r = 0; r < 40; ++r) {
            const auto got = d.propose(ctx, 109, 4);
            if (got.size() == 7)
                ++offered;
            else if (got.size() == 1)
                ++probes;
            d.outcome(0);
        }
        check(offered >= 2 && offered <= 4 && probes >= 3,
              std::format("rejections close the bucket after {} offers, then {} probes in 40 rounds", offered, probes));
        int reopened = 0;
        for (int r = 0; r < 40; ++r) {
            const auto got = d.propose(ctx, 109, 4);
            if (got.size() == 7)
                ++reopened;
            d.outcome(uint32_t(got.size()));
        }
        check(reopened >= 28, std::format("a kept probe reopens it ({} full offers in 40 rounds)", reopened));
    }
    {
        // The index follows the context: cut back and continued differently, proposals come from the new text.
        LookupDrafter d;
        d.propose(ctx, 109, 4);
        std::vector<int32_t> cut(ctx.begin(), ctx.begin() + 22);
        cut.insert(cut.end(), {40, 41, 42, 43, 44, 45, 40, 41, 42, 43});
        check(d.propose(cut, 44, 4) == seq({45, 40, 41}), "after a cut the index is rebuilt and the new repeat found");
        check(d.propose(cut, 109, 4).empty(), "the old continuation past the cut is gone");
    }
    std::println("{}", failed ? "FAILED" : "all passed");
    return failed ? 1 : 0;
}
