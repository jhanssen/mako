#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace mako::gen {

// Rows that stand in for tokens' embeddings (an image's encoder output): sequence position pos + i takes
// rows[i * width, (i + 1) * width), width the model's embedding width.
struct InputRows {
    uint32_t pos = 0, count = 0;
    const float* rows = nullptr;
};

// What a Session needs of an engine: tokens in, logits out, and the state to keep or go back to.
class Model {
public:
    virtual ~Model() = default;

    // Positions that take these rows instead of their tokens' embeddings in the prefills and steps that follow,
    // until replaced (empty: none). The rows must outlive their use. Engines without such input throw.
    virtual void set_input_rows(std::span<const InputRows> rows)
    {
        if (!rows.empty())
            throw std::runtime_error("this engine takes no input rows");
    }

    // Runs one token at the next position and returns its logits.
    virtual const std::vector<float>& step(int32_t token) = 0;
    // Runs tokens at the next positions, as many steps would; returns the logits after the last. `progress` gets the
    // number run so far between groups of tokens, and within a group an estimate (its share of the layers done).
    // `cancelled` is polled between groups: true stops there, with the tokens so far in the context (position()).
    virtual const std::vector<float>& prefill(std::span<const int32_t> tokens,
                                              const std::function<bool()>& cancelled = {},
                                              const std::function<void(size_t)>& progress = {}) = 0;
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
    // The positional caches' contents for positions before pos (what a checkpoint leaves out) and back: with a
    // checkpoint taken at pos, they carry a context out of the engine and, after restore(), into it again.
    virtual std::vector<std::byte> save_positions(uint32_t pos) const = 0;
    virtual void load_positions(uint32_t pos, std::span<const std::byte> data) = 0;
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

    // Speculative decoding with a multi-token drafter: draft(next) proposes the tokens likely to follow `next`, the
    // token at position() not yet run (empty when the drafter is unsure); verify(tokens) runs `next` and drafts at
    // the next positions, returning their logits (a row each); keep(n) keeps the first n of them (n >= 1), undoing
    // the rest. Without a drafter has_drafter() is false and the rest is not called.
    virtual bool has_drafter() const { return false; }
    virtual std::vector<int32_t> draft(int32_t) { return {}; }
    virtual const std::vector<float>& verify(std::span<const int32_t>) { throw std::logic_error("no drafter"); }
    virtual void keep(uint32_t) { throw std::logic_error("no drafter"); }
};

} // namespace mako::gen
