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

uint64_t Session::key_at(size_t i) const
{
    for (const auto& k : keys_)
        if (i >= k.rows.pos && i - k.rows.pos < k.rows.count)
            return k.key;
    return 0;
}

void Session::cut_keys(size_t n)
{
    std::erase_if(keys_, [&](const auto& k) { return k.rows.pos >= n; });
    for (auto& k : keys_) {
        k.rows.count = uint32_t(std::min<size_t>(k.rows.count, n - k.rows.pos));
        k.rows.rows = nullptr;
    }
}

void Session::clear()
{
    if (parked_)
        parked_.reset(); // the engine holds another session's context
    else
        e_.reset();
    ctx_.clear();
    keys_.clear();
    saved_.clear();
    mtp_next_ = -1;
}

namespace {

uint64_t key_in(std::span<const GenerateParams::PromptRows> rows, size_t i)
{
    for (const auto& r : rows)
        if (i >= r.rows.pos && i - r.rows.pos < r.rows.count)
            return r.key;
    return 0;
}

} // namespace

Session::Match Session::match(std::span<const int32_t> prompt, std::span<const GenerateParams::PromptRows> rows) const
{
    Match m;
    if (prompt.empty())
        return m;
    const size_t limit = std::min(prompt.size() - 1, ctx_.size());
    size_t common = 0;
    while (common < limit && ctx_[common] == prompt[common] && key_at(common) == key_in(rows, common))
        ++common;
    if (common == ctx_.size()) {
        m.keep = common;
        return m;
    }
    size_t best = 0, latest = 0;
    for (const Saved& sv : saved_) {
        latest = std::max<size_t>(latest, sv.cp->pos);
        if (sv.cp->pos <= common)
            best = std::max<size_t>(best, sv.cp->pos);
    }
    (best == latest ? m.keep : m.fork) = best;
    return m;
}

void Session::fork(const Session& src, size_t pos)
{
    const auto it = std::ranges::find_if(src.saved_, [&](const Saved& s) { return s.cp->pos == pos; });
    if (it == src.saved_.end() || pos > src.ctx_.size())
        throw std::logic_error("fork from a position without a checkpoint");
    ctx_.assign(src.ctx_.begin(), src.ctx_.begin() + pos);
    keys_ = src.keys_;
    cut_keys(pos);
    saved_ = {{it->cp, ++clock_, it->fork}}; // shared with src
    parked_ = Parked{it->cp, e_.save_positions(uint32_t(pos))};
    mtp_next_ = -1;
}

void Session::park()
{
    if (parked_)
        return;
    if (e_.position() != ctx_.size() || ctx_.empty()) {
        ctx_.clear();
        keys_.clear();
        saved_.clear();
        mtp_next_ = -1;
        return;
    }
    Parked p{std::make_shared<const Model::Checkpoint>(e_.checkpoint()), e_.save_positions(uint32_t(ctx_.size()))};
    parked_ = std::move(p);
    // The newest checkpoint stays (another try at the last prompt) and the fork point's; the parked state continues
    // the context.
    size_t newest = 0;
    for (const Saved& sv : saved_)
        newest = std::max<size_t>(newest, sv.cp->pos);
    std::erase_if(saved_, [&](const Saved& s) { return s.cp->pos != newest && !s.fork; });
}

void Session::unpark()
{
    if (!parked_) {
        clear();
        return;
    }
    e_.restore(*parked_->cp);
    e_.load_positions(parked_->cp->pos, parked_->positions);
    parked_.reset();
}

uint64_t Session::bytes(std::unordered_set<const void*>* seen) const
{
    const auto once = [&](const std::shared_ptr<const Model::Checkpoint>& cp) {
        return !seen || seen->insert(cp.get()).second ? cp->bytes() : 0;
    };
    uint64_t n = parked_ ? once(parked_->cp) + parked_->positions.size() : 0;
    for (const Saved& sv : saved_)
        n += once(sv.cp);
    return n;
}

// Back to the best state that holds at most ctx_[0, keep): the current one, a checkpoint, or the start.
void Session::rewind(size_t keep)
{
    // The model moved outside this session (another user, or a new session over a used model): nothing is known.
    if (e_.position() != ctx_.size())
        clear();
    if (keep >= ctx_.size())
        return;
    Saved* best = nullptr;
    for (Saved& s : saved_)
        if (s.cp->pos <= keep && (!best || s.cp->pos > best->cp->pos))
            best = &s;
    if (best) {
        e_.restore(*best->cp);
        best->used = ++clock_;
    } else {
        e_.reset();
    }
    ctx_.resize(e_.position());
    cut_keys(ctx_.size());
    // Checkpoints past the divergence would describe tokens that are about to be overwritten.
    std::erase_if(saved_, [&](const Saved& s) { return s.cp->pos > keep; });
    mtp_next_ = -1;
}

void Session::save_checkpoint(bool fork)
{
    if (max_checkpoints_ == 0)
        return;
    if (const auto it = std::ranges::find_if(saved_, [&](const Saved& s) { return s.cp->pos == ctx_.size(); });
        it != saved_.end()) {
        it->fork |= fork;
        return;
    }
    if (saved_.size() >= max_checkpoints_) {
        auto victim = saved_.end();
        for (auto it = saved_.begin(); it != saved_.end(); ++it)
            if (!it->fork && (victim == saved_.end() || it->used < victim->used))
                victim = it;
        if (victim != saved_.end())
            saved_.erase(victim);
    }
    saved_.push_back({std::make_shared<const Model::Checkpoint>(e_.checkpoint()), ++clock_, fork});
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
    if (parked_)
        throw std::logic_error("generate() on a parked session");
    if (prompt.size() > e_.max_context())
        throw std::invalid_argument("prompt longer than the context");
    GenerateResult res;
    res.prompt_tokens = uint32_t(prompt.size());

    // The prompt's input-row spans, the keys its positions are compared by.
    const auto prompt_key = [&](size_t i) { return key_in(p.input_rows, i); };
    const auto keep_keys = [&](size_t n) {
        keys_ = p.input_rows;
        cut_keys(n);
    };
    // At least the prompt's last token runs, for its logits.
    const size_t limit = prompt.size() - 1;
    size_t common = 0;
    while (common < std::min(limit, ctx_.size()) && ctx_[common] == prompt[common] &&
           key_at(common) == prompt_key(common))
        ++common;
    rewind(common);
    res.reused_tokens = uint32_t(ctx_.size());
    // The engine reads the rows only while the prompt runs.
    std::vector<InputRows> rows;
    for (const auto& r : p.input_rows)
        rows.push_back(r.rows);
    e_.set_input_rows(rows);
    struct Unset {
        Model& e;
        ~Unset() { e.set_input_rows({}); }
    } unset{e_};

    const auto t_prefill = std::chrono::steady_clock::now();
    const auto rest = prompt.subspan(ctx_.size());
    if (!ctx_.empty() && mtp_next_ != rest[0])
        e_.mtp_seed(rest[0]);
    // Runs the prompt up to `end` (exclusive) and records it; false when cancelled there (the result says so).
    const auto run_to = [&](size_t end) {
        if (end <= ctx_.size())
            return true;
        // Pieces end at multiples of prefill_piece more than 8 tokens from either end (shorter runs take the decode
        // path, which rounds differently).
        constexpr size_t kMin = 8;
        const size_t piece = p.prefill_piece;
        size_t at = ctx_.size();
        while (true) {
            size_t to = end;
            if (piece) {
                const size_t cut = (at + kMin) / piece * piece + piece;
                if (cut + kMin < end)
                    to = cut;
            }
            e_.prefill(prompt.subspan(at, to - at), p.cancelled);
            if (e_.position() != to || to == end)
                break;
            at = to;
            ctx_.assign(prompt.begin(), prompt.begin() + at);
            keep_keys(at);
            if (e_.has_mtp())
                e_.mtp_seed(prompt[at]);
            if (p.cancelled && p.cancelled())
                break;
            if (p.yield)
                p.yield();
        }
        const size_t read = e_.position();
        ctx_.assign(prompt.begin(), prompt.begin() + read);
        keep_keys(read);
        if (read == end) {
            // The next-token block learns the token after the part (within it, prefill gave it each next token).
            if (e_.has_mtp())
                e_.mtp_seed(prompt[read]);
            return true;
        }
        // The next-token block's cache already holds the last position read with the prompt's next token.
        mtp_next_ = e_.has_mtp() ? prompt[read] : -1;
        res.finish = Finish::cancelled;
        res.prefill_ms = ms_since(t_prefill);
        return false;
    };
    // A checkpoint at the fork point, for other conversations starting the same way; one at the end of the last
    // image, for other questions about it; and one before the prompt's last token, so that the same prompt again
    // (reuse stops short of its end) or one continuing differently right after it restores it.
    size_t image_end = 0;
    for (const auto& r : p.input_rows)
        image_end = std::max<size_t>(image_end, r.rows.pos + r.rows.count);
    std::vector<std::pair<size_t, bool>> marks{{p.fork_point, true}, {image_end, false}};
    std::ranges::sort(marks);
    for (const auto& [at, fork] : marks)
        if (at > ctx_.size() && at < prompt.size() - 1) {
            if (!run_to(at))
                return res;
            save_checkpoint(fork);
        }
    if (!run_to(prompt.size() - 1))
        return res;
    ctx_.assign(prompt.begin(), prompt.end() - 1);
    keep_keys(prompt.size());
    if (!ctx_.empty())
        save_checkpoint();
    const std::vector<float>& lp = e_.step(prompt.back());
    ctx_.push_back(prompt.back());
    e_.set_input_rows({});
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
        const int32_t f = p.forced ? p.forced(logits) : -1;
        return f >= 0 ? f : sampler.sample(logits);
    };
    int32_t token = choose(lp.data());
    if (p.speculative && e_.has_drafter()) {
        // Each round: drafts after `token`, then token and the drafts in one verify; drafts are kept up to the first
        // the sampler rejects (it then gives the token in its place), and after the last kept the trunk's next.
        while (p.max_tokens > 0 && emit(token)) {
            if (p.yield)
                p.yield();
            std::vector<int32_t> drafts = e_.draft(token);
            const uint32_t room = std::min<uint32_t>(e_.max_context() - e_.position(),
                                                     p.max_tokens - uint32_t(res.tokens.size()) + 1);
            if (drafts.size() + 1 > room)
                drafts.resize(room > 0 ? room - 1 : 0);
            if (drafts.empty()) {
                const std::vector<float>& l = e_.step(token);
                ctx_.push_back(token);
                token = choose(l.data());
                continue;
            }
            std::vector<int32_t> run{token};
            run.insert(run.end(), drafts.begin(), drafts.end());
            const std::vector<float>& rows = e_.verify(run);
            res.drafts += uint32_t(drafts.size());
            uint32_t kept = 0;
            int32_t next = -1;
            bool more = true;
            for (; kept < drafts.size(); ++kept) {
                bool accepted = false;
                const int32_t f = p.forced ? p.forced(rows.data() + uint64_t(kept) * nv) : -1;
                const int32_t t = f >= 0 ? (accepted = f == drafts[kept], f)
                                         : sampler.verify(rows.data() + uint64_t(kept) * nv, drafts[kept], accepted);
                if (!accepted) {
                    next = t;
                    break;
                }
                ++res.accepted;
                if (!emit(drafts[kept])) {
                    more = false;
                    ++kept;
                    break;
                }
            }
            e_.keep(kept + 1);
            ctx_.insert(ctx_.end(), run.begin(), run.begin() + kept + 1);
            if (!more)
                break;
            token = next >= 0 ? next : choose(rows.data() + uint64_t(kept) * nv);
        }
        res.decode_ms = ms_since(t_decode);
        return res;
    }
    int32_t draft = spec ? e_.mtp_draft(token) : -1;
    mtp_next_ = spec ? token : -1;
    while (p.max_tokens > 0 && emit(token)) {
        if (p.yield)
            p.yield();
        if (draft >= 0 && e_.position() + 2 <= e_.max_context()) {
            // Verify [token, draft] in one two-token step.
            const std::vector<float>& two = e_.step2(token, draft);
            ctx_.push_back(token);
            ctx_.push_back(draft);
            ++res.drafts;
            bool accepted = false;
            const int32_t f = p.forced ? p.forced(two.data()) : -1;
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
