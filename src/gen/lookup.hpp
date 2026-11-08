#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace mako::gen {

// Drafts from the context itself: when the tokens being written repeat an earlier stretch of it (a file sent back
// with edits, quoted tool output), the tokens that followed that stretch are the draft. Each trigram of the context
// is indexed by the positions it ended at; a proposal takes the earlier occurrence of the current trigram whose
// backward match is longest (at least min_match tokens) and offers the tokens after it: 3 for a match of 4-5, 5 for
// 6-7, 7 from 8. A rejected wide verify costs about as much as a step, so each match length keeps the share of its
// first drafts that were accepted and stops offering (one probe draft in eight) while it is below the threshold; a
// kept probe reopens it.
// Costs no memory beyond the index and no model work; the engine's verify keeps the output distribution.
class LookupDrafter {
public:
    static constexpr uint32_t kMaxDrafts = 7; // with the token verified, the 8 rows an engine verifies at most

    // The tokens likely to follow `next`, which comes after `ctx` (the index follows ctx: extended, or rebuilt after
    // a rewind). Empty: no match of min_match tokens, or its length is gated off. min_match >= 3.
    std::vector<int32_t> propose(std::span<const int32_t> ctx, int32_t next, uint32_t min_match,
                                 uint32_t max_drafts = kMaxDrafts);
    // How many of the last proposal's drafts were kept, for the gate.
    void outcome(uint32_t kept);
    // The index is dropped (a parked session), rebuilt by the next propose().
    void clear();
    // The share of first drafts kept, per bucket (match 4-5, 6-7, 8 and more).
    double acceptance(uint32_t bucket) const { return bucket < 3 ? acc_[bucket] : 0.0; }

private:
    static uint32_t bucket_of(uint32_t match) { return match >= 8 ? 2 : match >= 6 ? 1 : 0; }
    static uint64_t key(int32_t a, int32_t b, int32_t c);
    void sync(std::span<const int32_t> ctx);
    void index(uint32_t pos); // the trigram ending at seen_[pos]

    std::vector<int32_t> seen_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> last_; // trigram -> the last positions it ended at
    double acc_[3] = {0.8, 0.8, 0.8};
    uint32_t refused_[3] = {0, 0, 0};
    int last_bucket_ = -1;
};

} // namespace mako::gen
