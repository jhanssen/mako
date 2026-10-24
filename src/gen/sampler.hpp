#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mako::gen {

// Applied in this order: penalties, temperature, min_p, top_k, top_p.
struct SamplingParams {
    float temperature = 1.0f; // <= 0: greedy (the most likely token; the other fields have no effect)
    uint32_t top_k = 0;       // 0: no limit
    float top_p = 1.0f;       // the smallest set of most likely tokens whose probabilities sum to at least top_p
    float min_p = 0.0f;       // drops tokens less likely than min_p times the most likely one
    // Over the tokens generated so far: logit -= presence if generated, and -= frequency * count.
    float presence_penalty = 0.0f, frequency_penalty = 0.0f;
    float repetition_penalty = 1.0f; // > 1: a generated token's logit is divided by it (multiplied if negative)
};

class Sampler {
public:
    Sampler(const SamplingParams& p, uint32_t n_vocab, uint64_t seed);

    bool greedy() const { return p_.temperature <= 0.0f; }
    int32_t sample(const float* logits);
    // Speculative verification of a draft that was chosen deterministically: accepts it with its probability
    // under sample()'s distribution, else returns a token drawn from that distribution with the draft removed. The
    // result is distributed exactly as sample()'s.
    int32_t verify(const float* logits, int32_t draft, bool& accepted);
    // Records a generated token (for the penalties).
    void accept(int32_t token);

private:
    void distribution(const float* logits);
    int32_t draw(float total);

    SamplingParams p_;
    uint32_t n_vocab_;
    std::mt19937_64 rng_;
    std::unordered_map<int32_t, uint32_t> counts_;
    std::vector<float> work_;
    std::vector<std::pair<float, int32_t>> cand_; // (probability, token), unnormalized
};

} // namespace mako::gen
