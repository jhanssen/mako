#include "gen/session.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace mako::gen {

namespace {

double ms_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

void Session::clear()
{
    e_.reset();
    ctx_.clear();
    saved_.clear();
    mtp_next_ = -1;
}

// Back to the best state that holds at most ctx_[0, keep): the current one, a checkpoint, or the start.
void Session::rewind(size_t keep)
{
    if (keep >= ctx_.size())
        return;
    Saved* best = nullptr;
    for (Saved& s : saved_)
        if (s.cp.pos <= keep && (!best || s.cp.pos > best->cp.pos))
            best = &s;
    if (best) {
        e_.restore(best->cp);
        best->used = ++clock_;
    } else {
        e_.reset();
    }
    ctx_.resize(e_.position());
    // Checkpoints past the divergence would describe tokens that are about to be overwritten.
    std::erase_if(saved_, [&](const Saved& s) { return s.cp.pos > keep; });
    mtp_next_ = -1;
}

void Session::save_checkpoint()
{
    if (max_checkpoints_ == 0 || std::ranges::any_of(saved_, [&](const Saved& s) { return s.cp.pos == ctx_.size(); }))
        return;
    if (saved_.size() >= max_checkpoints_)
        saved_.erase(std::ranges::min_element(saved_, {}, &Saved::used));
    saved_.push_back({e_.checkpoint(), ++clock_});
}

GenerateResult Session::generate(std::span<const int32_t> prompt, const GenerateParams& p,
                                 const std::function<bool(int32_t)>& on_token)
{
    try {
        return run(prompt, p, on_token);
    } catch (...) {
        // The context no longer matches what is known about it.
        clear();
        throw;
    }
}

GenerateResult Session::run(std::span<const int32_t> prompt, const GenerateParams& p,
                            const std::function<bool(int32_t)>& on_token)
{
    if (prompt.empty())
        throw std::invalid_argument("empty prompt");
    if (prompt.size() > e_.max_context())
        throw std::invalid_argument("prompt longer than the context");
    GenerateResult res;
    res.prompt_tokens = uint32_t(prompt.size());

    // At least the prompt's last token runs, for its logits.
    const size_t limit = prompt.size() - 1;
    size_t common = 0;
    while (common < std::min(limit, ctx_.size()) && ctx_[common] == prompt[common])
        ++common;
    rewind(common);
    res.reused_tokens = uint32_t(ctx_.size());

    const auto t_prefill = std::chrono::steady_clock::now();
    const auto rest = prompt.subspan(ctx_.size());
    if (!ctx_.empty() && mtp_next_ != rest[0])
        e_.mtp_seed(rest[0]);
    // The checkpoint is taken before the prompt's last token, so that the same prompt again (reuse stops short of
    // its end) or one continuing differently right after it restores it.
    if (rest.size() > 1) {
        e_.prefill(rest.first(rest.size() - 1), p.cancelled);
        if (const size_t read = e_.position(); read < prompt.size() - 1) {
            // The next-token block's cache already holds the last position read with the prompt's next token.
            ctx_.assign(prompt.begin(), prompt.begin() + read);
            mtp_next_ = e_.has_mtp() ? prompt[read] : -1;
            res.finish = Finish::cancelled;
            res.prefill_ms = ms_since(t_prefill);
            return res;
        }
        if (e_.has_mtp())
            e_.mtp_seed(prompt.back());
    }
    ctx_.assign(prompt.begin(), prompt.end() - 1);
    if (!ctx_.empty())
        save_checkpoint();
    const std::vector<float>& lp = e_.step(prompt.back());
    ctx_.push_back(prompt.back());
    res.prefill_ms = ms_since(t_prefill);

    const auto t_decode = std::chrono::steady_clock::now();
    const uint32_t nv = e_.n_vocab();
    Sampler sampler(p.sampling, nv, p.seed);
    const bool spec = p.speculative && e_.has_mtp();
    // Takes a decided token; false when generation ends with it.
    auto emit = [&](int32_t t) {
        sampler.accept(t);
        res.tokens.push_back(t);
        if (on_token && !on_token(t)) {
            res.finish = Finish::cancelled;
            return false;
        }
        if (std::ranges::find(p.stop_tokens, t) != p.stop_tokens.end()) {
            res.finish = Finish::stop_token;
            return false;
        }
        if (res.tokens.size() >= p.max_tokens) {
            res.finish = Finish::max_tokens;
            return false;
        }
        if (e_.position() >= e_.max_context()) {
            res.finish = Finish::context_full;
            return false;
        }
        return true;
    };

    auto choose = [&](const float* logits) {
        const int32_t f = p.forced ? p.forced() : -1;
        return f >= 0 ? f : sampler.sample(logits);
    };
    int32_t token = choose(lp.data());
    int32_t draft = spec ? e_.mtp_draft(token) : -1;
    mtp_next_ = spec ? token : -1;
    while (p.max_tokens > 0 && emit(token)) {
        if (draft >= 0 && e_.position() + 2 <= e_.max_context()) {
            // Verify [token, draft] in one two-token step.
            const std::vector<float>& two = e_.step2(token, draft);
            ctx_.push_back(token);
            ctx_.push_back(draft);
            ++res.drafts;
            bool accepted = false;
            const int32_t f = p.forced ? p.forced() : -1;
            const int32_t t1 = f >= 0 ? (accepted = f == draft, f) : sampler.verify(two.data(), draft, accepted);
            if (!accepted) {
                e_.rollback();
                ctx_.pop_back();
                token = t1;
                draft = e_.mtp_draft(token);
                mtp_next_ = token;
                continue;
            }
            ++res.accepted;
            if (!emit(draft)) {
                // The block's cache at token's position still needs the draft as its next token.
                e_.mtp_draft(draft);
                mtp_next_ = -1;
                break;
            }
            token = choose(two.data() + nv);
            const int32_t both[2] = {draft, token};
            draft = e_.mtp_draft(std::span<const int32_t>(both));
            mtp_next_ = token;
        } else {
            const std::vector<float>& l = e_.step(token);
            ctx_.push_back(token);
            token = choose(l.data());
            draft = spec ? e_.mtp_draft(token) : -1;
            mtp_next_ = spec ? token : -1;
        }
    }
    res.decode_ms = ms_since(t_decode);
    return res;
}

} // namespace mako::gen
