#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace mako::gen {

// What a Session needs of an engine: tokens in, logits out, and the state to keep or go back to.
class Model {
public:
    virtual ~Model() = default;

    // Runs one token at the next position and returns its logits.
    virtual const std::vector<float>& step(int32_t token) = 0;
    // Runs tokens at the next positions, as many steps would; returns the logits after the last. `cancelled` is
    // polled between groups of tokens: true stops there, with the tokens so far in the context (position()).
    virtual const std::vector<float>& prefill(std::span<const int32_t> tokens,
                                              const std::function<bool()>& cancelled = {}) = 0;
    virtual uint32_t position() const = 0;
    virtual uint32_t max_context() const = 0;
    virtual uint32_t n_vocab() const = 0;
    // Back to position 0.
    virtual void reset() = 0;

    // The state at a position that the positional caches do not hold, to continue from it later: restore() is valid
    // while the tokens before pos are the same as when it was taken.
    struct Checkpoint {
        uint32_t pos = 0;
        std::vector<std::byte> state;
        std::vector<float> hidden; // a draft block's input at pos - 1, when the engine has one
        uint64_t bytes() const { return state.size() + hidden.size() * 4; }
    };
    virtual Checkpoint checkpoint() = 0;
    virtual void restore(const Checkpoint& cp) = 0;
    // Checkpoints the engine's memory budget reserved room for.
    virtual uint32_t checkpoint_slots() const = 0;

    // Speculative decoding with a one-token drafter (a next-token block); without one, has_mtp() is false and the
    // rest is not called. mtp_draft(next): the guess for the token after `next`, which follows the last step's
    // token (over several: the last step's first next.size() positions, each followed by next[i]); mtp_seed(next):
    // the token after position() - 1 is `next`; step2: two tokens in one step, their logits a row each; rollback:
    // undoes step2's second token.
    virtual bool has_mtp() const { return false; }
    virtual int32_t mtp_draft(int32_t) { return -1; }
    virtual int32_t mtp_draft(std::span<const int32_t>) { return -1; }
    virtual void mtp_seed(int32_t) {}
    virtual const std::vector<float>& step2(int32_t, int32_t) { throw std::logic_error("no two-token steps"); }
    virtual void rollback() { throw std::logic_error("no two-token steps"); }
};

} // namespace mako::gen
