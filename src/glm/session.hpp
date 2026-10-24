#pragma once

#include "gen/sampler.hpp"
#include "glm/engine.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace mako::glm {

struct GenerateParams {
    gen::SamplingParams sampling;
    uint64_t seed = 0;
    uint32_t max_tokens = 256;
    std::vector<int32_t> stop_tokens; // generation ends after one of these (it is part of the output)
    bool speculative = true;          // draft with the next-token block, when the engine has it
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
// state taken at the end of an earlier prompt (a chat template that rewrites the last answer diverges there).
class Session {
public:
    // max_checkpoints beyond the engine's checkpoint_slots() use memory its budget did not reserve.
    explicit Session(Engine& engine) : Session(engine, engine.checkpoint_slots()) {}
    Session(Engine& engine, size_t max_checkpoints) : e_(engine), max_checkpoints_(max_checkpoints) {}

    // on_token gets each generated token once decided; returning false stops (Finish::cancelled).
    GenerateResult generate(std::span<const int32_t> prompt, const GenerateParams& p,
                            const std::function<bool(int32_t)>& on_token = {});
    // The tokens at the engine's positions 0..position() - 1.
    std::span<const int32_t> context() const { return ctx_; }
    // Forgets the context and the checkpoints.
    void clear();

private:
    struct Saved {
        Engine::Checkpoint cp;
        uint64_t used = 0;
    };
    GenerateResult run(std::span<const int32_t> prompt, const GenerateParams& p,
                       const std::function<bool(int32_t)>& on_token);
    void rewind(size_t keep);
    void save_checkpoint();

    Engine& e_;
    size_t max_checkpoints_;
    std::vector<int32_t> ctx_;
    std::vector<Saved> saved_;
    uint64_t clock_ = 0;
    int32_t mtp_next_ = -1; // the token the next-token block's cache assumes after ctx_.back(); -1: none
};

} // namespace mako::glm
