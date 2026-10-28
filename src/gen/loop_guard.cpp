#include "gen/loop_guard.hpp"

#include <algorithm>

namespace mako::gen {

LoopGuard::LoopGuard(uint32_t min_tokens, uint32_t max_period)
    : min_(min_tokens), max_p_(std::max(1u, max_period)), window_(2 * min_tokens)
{
    reset();
}

void LoopGuard::reset()
{
    tokens_.clear();
    run_.assign(max_p_ + 1, 0);
    hits_.assign(max_p_ + 1, 0);
    matched_.assign(size_t(max_p_ + 1) * window_, 0);
    start_ = 0;
    period_ = 0;
}

bool LoopGuard::push(int32_t token)
{
    tokens_.push_back(token);
    const size_t n = tokens_.size();
    if (min_ == 0)
        return false;
    const size_t slot = (n - 1) % window_;
    bool found = false;
    for (uint32_t p = 1; p <= max_p_ && p < n; ++p) {
        const bool same = tokens_[n - 1] == tokens_[n - 1 - p];
        run_[p] = same ? run_[p] + 1 : 0;
        uint8_t& bit = matched_[size_t(p) * window_ + slot];
        hits_[p] += uint32_t(same) - bit;
        bit = same;
        if (found)
            continue;
        // Exactly periodic over run + p tokens, at least 4 periods.
        if (run_[p] >= std::max(min_, 3 * p)) {
            start_ = n - run_[p] - p;
            period_ = p;
            found = true;
        } else if (4 * p <= window_ && n >= window_ + p && 10 * uint64_t(hits_[p]) >= 9 * uint64_t(window_)) {
            // From the window's first match: the window may start before the loop.
            size_t k = n - window_;
            while (tokens_[k] != tokens_[k - p])
                ++k;
            start_ = k - p;
            period_ = p;
            found = true;
        }
    }
    return found;
}

} // namespace mako::gen
