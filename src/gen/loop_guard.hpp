#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mako::gen {

// Repetition loops in generated tokens: the tokens so far ending in a run that repeats with some period p
// (1..max_period) at least 4 times, either exactly over at least min_tokens, or with at least 90% of its positions
// matching the token p before over a window of 2 * min_tokens (a loop with a counter or a changing word). O(max_period)
// a token.
class LoopGuard {
public:
    explicit LoopGuard(uint32_t min_tokens, uint32_t max_period = 512);

    // Takes the next token; true when the tokens so far end in a loop (start() and period() say where).
    bool push(int32_t token);
    // Forgets the tokens so far.
    void reset();
    // The index (in pushed tokens since the last reset) of the loop's first token, and its period.
    size_t start() const { return start_; }
    uint32_t period() const { return period_; }

private:
    uint32_t min_, max_p_, window_;
    std::vector<int32_t> tokens_;
    std::vector<uint32_t> run_;    // per period: trailing positions matching the token p before
    std::vector<uint32_t> hits_;   // per period: matches within the last window_ positions
    std::vector<uint8_t> matched_; // per period, a ring of window_ match bits
    size_t start_ = 0;
    uint32_t period_ = 0;
};

} // namespace mako::gen
