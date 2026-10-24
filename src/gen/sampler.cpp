#include "gen/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mako::gen {

Sampler::Sampler(const SamplingParams& p, uint32_t n_vocab, uint64_t seed) : p_(p), n_vocab_(n_vocab), rng_(seed)
{
    if (n_vocab == 0)
        throw std::invalid_argument("sampler: empty vocabulary");
}

void Sampler::accept(int32_t token)
{
    ++counts_[token];
}

void Sampler::distribution(const float* logits)
{
    work_.assign(logits, logits + n_vocab_);
    for (const auto& [t, n] : counts_) {
        float& l = work_[size_t(t)];
        if (p_.repetition_penalty != 1.0f)
            l = l > 0 ? l / p_.repetition_penalty : l * p_.repetition_penalty;
        l -= p_.presence_penalty + p_.frequency_penalty * float(n);
    }
    const float inv_t = 1.0f / p_.temperature, m = *std::ranges::max_element(work_);
    cand_.clear();
    cand_.reserve(n_vocab_);
    for (uint32_t i = 0; i < n_vocab_; ++i) {
        const float q = std::exp((work_[i] - m) * inv_t); // the most likely token gets 1
        if (q >= std::min(p_.min_p, 1.0f) && q > 0.0f)
            cand_.emplace_back(q, int32_t(i));
    }
    auto desc = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
    if (p_.top_k > 0 && p_.top_k < cand_.size()) {
        std::ranges::nth_element(cand_, cand_.begin() + p_.top_k, desc);
        cand_.resize(p_.top_k);
    }
    if (p_.top_p < 1.0f) {
        double total = 0;
        for (const auto& c : cand_)
            total += c.first;
        const double want = p_.top_p * total;
        // Sorting a growing prefix: the nucleus is usually a small part of the vocabulary.
        for (size_t k = std::min<size_t>(64, cand_.size());; k = std::min(cand_.size(), k * 4)) {
            std::ranges::partial_sort(cand_, cand_.begin() + k, desc);
            double sum = 0;
            size_t n = 0;
            while (n < k && sum < want)
                sum += cand_[n++].first;
            if (sum >= want || k == cand_.size()) {
                cand_.resize(std::max<size_t>(n, 1));
                break;
            }
        }
    }
}

int32_t Sampler::draw(float total)
{
    float r = std::uniform_real_distribution<float>(0.0f, total)(rng_);
    for (const auto& [q, t] : cand_) {
        if (r < q)
            return t;
        r -= q;
    }
    return cand_.back().second; // rounding
}

int32_t Sampler::sample(const float* logits)
{
    if (greedy())
        return int32_t(std::max_element(logits, logits + n_vocab_) - logits);
    distribution(logits);
    float total = 0;
    for (const auto& c : cand_)
        total += c.first;
    return draw(total);
}

int32_t Sampler::verify(const float* logits, int32_t draft, bool& accepted)
{
    if (greedy()) {
        const int32_t t = int32_t(std::max_element(logits, logits + n_vocab_) - logits);
        accepted = t == draft;
        return t;
    }
    distribution(logits);
    float total = 0, q_draft = 0;
    for (const auto& [q, t] : cand_) {
        total += q;
        if (t == draft)
            q_draft = q;
    }
    accepted = std::uniform_real_distribution<float>(0.0f, total)(rng_) < q_draft;
    if (accepted)
        return draft;
    std::erase_if(cand_, [&](const auto& c) { return c.second == draft; });
    if (cand_.empty()) // the draft had all the probability; not reachable through the comparison above
        return draft;
    return draw(total - q_draft);
}

} // namespace mako::gen
