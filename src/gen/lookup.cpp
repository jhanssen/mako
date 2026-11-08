#include "gen/lookup.hpp"

#include <algorithm>

namespace mako::gen {

namespace {

constexpr size_t kPositionsPerKey = 4; // the most recent occurrences kept per trigram
constexpr uint32_t kLongest = 64;      // a backward match is not extended further
constexpr double kThreshold = 0.55;    // first drafts kept, below which a bucket offers only probes
constexpr double kAlpha = 0.2;
constexpr uint32_t kProbeEvery = 8;

} // namespace

uint64_t LookupDrafter::key(int32_t a, int32_t b, int32_t c)
{
    uint64_t h = uint64_t(uint32_t(a));
    h = (h * 1000003u) ^ uint64_t(uint32_t(b));
    h = (h * 1000003u) ^ uint64_t(uint32_t(c));
    return h;
}

void LookupDrafter::clear()
{
    seen_.clear();
    last_.clear();
}

void LookupDrafter::index(uint32_t pos)
{
    std::vector<uint32_t>& at = last_[key(seen_[pos - 2], seen_[pos - 1], seen_[pos])];
    if (at.size() == kPositionsPerKey)
        at.erase(at.begin());
    at.push_back(pos);
}

// seen_ is a prefix of ctx (checked at its last tokens): extended; else rebuilt.
void LookupDrafter::sync(std::span<const int32_t> ctx)
{
    const size_t n = seen_.size();
    bool prefix = n <= ctx.size();
    for (size_t i = n > 8 ? n - 8 : 0; prefix && i < n; ++i)
        prefix = seen_[i] == ctx[i];
    if (!prefix)
        clear();
    const size_t from = seen_.size();
    seen_.insert(seen_.end(), ctx.begin() + from, ctx.end());
    for (size_t pos = std::max<size_t>(from, 2); pos < seen_.size(); ++pos)
        index(uint32_t(pos));
}

std::vector<int32_t> LookupDrafter::propose(std::span<const int32_t> ctx, int32_t next, uint32_t min_match,
                                            uint32_t max_drafts)
{
    last_bucket_ = -1;
    min_match = std::max<uint32_t>(min_match, 3);
    if (ctx.size() < min_match)
        return {};
    sync(ctx);
    // The sequence is seen_ then next: the trigram ending at i = seen_.size().
    const uint32_t i = uint32_t(seen_.size());
    const auto it = last_.find(key(seen_[i - 2], seen_[i - 1], next));
    if (it == last_.end())
        return {};
    uint32_t best = 0, best_match = 0;
    for (const uint32_t j : it->second) {
        if (seen_[j - 2] != seen_[i - 2] || seen_[j - 1] != seen_[i - 1] || seen_[j] != next)
            continue;
        uint32_t m = 3;
        while (m < kLongest && m <= j && seen_[j - m] == seen_[i - m])
            ++m;
        if (m >= best_match) { // ties: the most recent
            best_match = m;
            best = j;
        }
    }
    if (best_match < min_match || best + 1 >= i)
        return {};
    const uint32_t b = bucket_of(best_match);
    uint32_t n = std::min<uint32_t>({b == 2 ? 7u : b == 1 ? 5u : 3u, max_drafts, i - 1 - best});
    if (acc_[b] < kThreshold) {
        if (++refused_[b] % kProbeEvery != 0)
            return {};
        n = 1;
    }
    last_bucket_ = int(b);
    return std::vector<int32_t>(seen_.begin() + best + 1, seen_.begin() + best + 1 + n);
}

void LookupDrafter::outcome(uint32_t kept)
{
    if (last_bucket_ < 0)
        return;
    double& acc = acc_[last_bucket_];
    acc = acc * (1 - kAlpha) + (kept > 0 ? kAlpha : 0.0);
    if (kept > 0 && acc < kThreshold)
        acc = kThreshold; // a kept probe reopens the bucket; one rejection closes it again
    last_bucket_ = -1;
}

} // namespace mako::gen
