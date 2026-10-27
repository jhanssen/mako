#pragma once

#include "gen/model.hpp"
#include "gen/sampler.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace mako::gen {

struct GenerateParams {
    SamplingParams sampling;
    uint64_t seed = 0;
    uint32_t max_tokens = 256;
    std::vector<int32_t> stop_tokens; // generation ends after one of these (it is part of the output)
    bool speculative = true;          // draft with the next-token block, when the engine has it
    // Polled while the prompt is read (between groups of tokens): true stops (Finish::cancelled, no tokens), keeping
    // the part read so far in the context.
    std::function<bool()> cancelled;
    // Prompt positions whose embeddings come from rows (images), each span with a key for its content: a prefix the
    // context holds is reused only where its keys match too. The rows must outlive generate().
    struct PromptRows {
        InputRows rows;
        uint64_t key = 0;
    };
    std::vector<PromptRows> input_rows;
    // Asked before each token is chosen (after the previous one reached on_token), with that position's logits
    // (n_vocab): a token >= 0 is taken instead of sampling, and a draft is kept only if it matches. -1: sample.
    std::function<int32_t(const float* logits)> forced;
};

enum class Finish { stop_token, max_tokens, context_full, cancelled };

struct GenerateResult {
    std::vector<int32_t> tokens; // generated
    Finish finish = Finish::max_tokens;
    uint32_t prompt_tokens = 0, reused_tokens = 0; // reused: the prompt's leading tokens the context already held
    double prefill_ms = 0, decode_ms = 0;
    uint32_t drafts = 0, accepted = 0; // speculative verifications and drafts kept
};

// A conversation's context on an engine. Each generate() gets the whole prompt (earlier turns included); the
// longest prefix the engine already holds is kept, from the current state or from a checkpoint of the recurrent
// state taken before the last token of an earlier prompt (the same prompt again, or a chat template that rewrites
// the last answer, diverges there).
class Session {
public:
    // max_checkpoints beyond the engine's checkpoint_slots() use memory its budget did not reserve.
    explicit Session(Model& model) : Session(model, model.checkpoint_slots()) {}
    Session(Model& model, size_t max_checkpoints) : e_(model), max_checkpoints_(max_checkpoints) {}

    // on_token gets each generated token once decided; returning false stops (Finish::cancelled).
    GenerateResult generate(std::span<const int32_t> prompt, const GenerateParams& p,
                            const std::function<bool(int32_t)>& on_token = {});
    // The tokens at the engine's positions 0..position() - 1.
    std::span<const int32_t> context() const { return ctx_; }
    // Forgets the context and the checkpoints.
    void clear();

    // The prompt's leading tokens this session can keep: its whole context when the prompt continues it, else the
    // latest checkpoint within the part they share (0: none).
    size_t usable_prefix(std::span<const int32_t> prompt, std::span<const GenerateParams::PromptRows> rows) const;
    // Takes the context out of the engine (another session is about to use it) and puts it back; generate() needs
    // it in. Parking a session the engine no longer holds forgets its context.
    void park();
    void unpark();
    bool parked() const { return parked_.has_value(); }
    // Memory the session holds outside the engine: its parked context and its checkpoints.
    uint64_t bytes() const;

private:
    struct Saved {
        Model::Checkpoint cp;
        uint64_t used = 0;
    };
    struct Parked {
        Model::Checkpoint cp;
        std::vector<std::byte> positions;
    };
    GenerateResult run(std::span<const int32_t> prompt, const GenerateParams& p,
                       const std::function<bool(int32_t)>& on_token);
    void rewind(size_t keep);
    void save_checkpoint();
    // The key of ctx_[i]: its span's in keys_, else 0.
    uint64_t key_at(size_t i) const;
    // Keys only of positions before n.
    void cut_keys(size_t n);

    Model& e_;
    size_t max_checkpoints_;
    std::vector<int32_t> ctx_;
    std::vector<GenerateParams::PromptRows> keys_; // the input-row spans within ctx_ (their rows are not kept)
    std::vector<Saved> saved_;
    std::optional<Parked> parked_;
    uint64_t clock_ = 0;
    int32_t mtp_next_ = -1; // the token the next-token block's cache assumes after ctx_.back(); -1: none
};

} // namespace mako::gen
