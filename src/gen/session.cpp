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
        e_->reset();
    ctx_.clear();
    keys_.clear();
    saved_.clear();
    lookup_.clear();
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

Session::Match Session::match(std::span<const int32_t> prompt, std::span<const GenerateParams::PromptRows> rows,
                              const GenerateParams::Text* text) const
{
    Match m;
    if (prompt.empty())
        return m;
    const size_t limit = std::min(prompt.size() - 1, ctx_.size());
    size_t common = 0;
    while (common < limit && ctx_[common] == prompt[common] && key_at(common) == key_in(rows, common))
        ++common;
    std::optional<Respelled> r;
    std::optional<std::pair<size_t, size_t>> stop;
    if (common < ctx_.size() && text)
        if ((r = respell(prompt, rows, *text, &stop))) {
            // As run() will: the tokens it has in common with the context past the part kept.
            common = r->kept;
            while (common < std::min(r->prompt.size() - 1, ctx_.size()) && ctx_[common] == r->prompt[common] &&
                   key_at(common) == key_in(r->rows, common))
                ++common;
        }
    if (common == ctx_.size()) {
        m.keep = ctx_.size();
        return m;
    }
    m.diverged = stop ? divergence(stop->first, prompt, stop->second)
                      : divergence(common, r ? std::span<const int32_t>(r->prompt) : prompt, common);
    // The last turn's checkpoints rewind this session: where its reply ended, before its prompt's last token (another
    // try at it) and where the reply before it ended (an edited last message), the three newest. An older one, or
    // the fork point's, starts another session from there.
    size_t best = 0;
    bool best_fork = false;
    std::vector<size_t> at;
    for (const Saved& sv : saved_) {
        at.push_back(sv.cp->pos);
        if (sv.cp->pos <= common && sv.cp->pos >= best) {
            best = sv.cp->pos;
            best_fork = sv.fork;
        }
    }
    std::ranges::sort(at, std::greater<>());
    const bool last_turn = best > 0 && !best_fork && (at.size() < 3 || best >= at[2]);
    (last_turn ? m.keep : m.fork) = best;
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
    parked_ = Parked{it->cp, e_->save_positions(uint32_t(pos))};
    mtp_next_ = -1;
}

void Session::park()
{
    if (parked_)
        return;
    if (e_->position() != ctx_.size() || ctx_.empty()) {
        ctx_.clear();
        keys_.clear();
        saved_.clear();
        mtp_next_ = -1;
        return;
    }
    // The parked state is the checkpoint where the reply ended, when one was taken.
    const auto at_end = std::ranges::find_if(saved_, [&](const Saved& s) { return s.cp->pos == ctx_.size(); });
    Parked p{at_end != saved_.end() ? at_end->cp : std::make_shared<const Model::Checkpoint>(e_->checkpoint()),
             e_->save_positions(uint32_t(ctx_.size()))};
    parked_ = std::move(p);
    lookup_.clear();
    // The last turn's checkpoints stay (the three newest: where its reply ended, which the parked state shares;
    // before its prompt's last token; where the reply before it ended) and the fork point's.
    std::vector<size_t> at;
    for (const Saved& sv : saved_)
        at.push_back(sv.cp->pos);
    std::ranges::sort(at, std::greater<>());
    const size_t keep_from = at.size() > 3 ? at[2] : 0;
    std::erase_if(saved_, [&](const Saved& s) { return s.cp->pos < keep_from && !s.fork; });
}

void Session::rebind(Model& model)
{
    if (!parked_ && !ctx_.empty())
        throw std::logic_error("rebind() of a session the engine holds");
    e_ = &model;
}

void Session::unpark()
{
    if (!parked_) {
        clear();
        return;
    }
    e_->restore(*parked_->cp);
    e_->load_positions(parked_->cp->pos, parked_->positions);
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
    if (e_->position() != ctx_.size())
        clear();
    if (keep >= ctx_.size())
        return;
    Saved* best = nullptr;
    for (Saved& s : saved_)
        if (s.cp->pos <= keep && (!best || s.cp->pos > best->cp->pos))
            best = &s;
    if (best) {
        e_->restore(*best->cp);
        best->used = ++clock_;
    } else {
        e_->reset();
    }
    ctx_.resize(e_->position());
    cut_keys(ctx_.size());
    // Checkpoints past the divergence would describe tokens that are about to be overwritten.
    std::erase_if(saved_, [&](const Saved& s) { return s.cp->pos > keep; });
    lookup_.clear(); // its index, too
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
    saved_.push_back({std::make_shared<const Model::Checkpoint>(e_->checkpoint()), ++clock_, fork});
}

std::optional<Session::Respelled> Session::respell(std::span<const int32_t> prompt,
                                                   std::span<const GenerateParams::PromptRows> rows,
                                                   const GenerateParams::Text& t,
                                                   std::optional<std::pair<size_t, size_t>>* stop) const
{
    const size_t held = ctx_.size(), n = prompt.size();
    size_t d = 0;
    while (d < std::min(held, n) && ctx_[d] == prompt[d] && key_at(d) == key_in(rows, d))
        ++d;
    if (d >= held || d >= n)
        return std::nullopt;
    // From d on: control tokens must be the same, plain tokens spell the same text. `a` and `b` hold the context's
    // and the prompt's text past the last place both are at a token boundary (the furthest before the prompt's end:
    // upto, at); the context's text may end inside one of the prompt's tokens (the rest of its text, encoded again).
    size_t i = d, j = d, upto = d, at = d, same = 0;
    bool whole = false;
    std::string a, b;
    while (true) {
        if (a.size() == b.size()) {
            a.clear(), b.clear(), same = 0;
            if (i == held) {
                whole = true;
                break;
            }
            if (j == n)
                break;
            upto = i, at = j;
            if (key_at(i) || key_in(rows, j))
                break;
            if (t.control(ctx_[i]) || t.control(prompt[j])) {
                if (ctx_[i] != prompt[j])
                    break;
                ++i, ++j;
                continue;
            }
            a += t.piece(ctx_[i++]);
        } else if (a.size() < b.size()) {
            if (i == held) {
                whole = true;
                break;
            }
            if (key_at(i) || t.control(ctx_[i]))
                break;
            a += t.piece(ctx_[i++]);
        } else {
            if (j == n || key_in(rows, j) || t.control(prompt[j]))
                break;
            b += t.piece(prompt[j++]);
        }
        const size_t m = std::min(a.size(), b.size());
        if (a.compare(same, m - same, b, same, m - same) != 0)
            break;
        same = m;
    }
    Respelled r;
    r.at = d;
    if (whole) {
        r.kept = held;
        r.prompt.assign(ctx_.begin(), ctx_.end());
        if (b.size() > a.size()) {
            const std::vector<int32_t> more = t.encode(std::string_view(b).substr(a.size()));
            r.prompt.insert(r.prompt.end(), more.begin(), more.end());
        }
        if (r.prompt.size() + (n - j) > held)
            at = j;
        else
            whole = false; // the prompt ends within the context
    }
    if (!whole) {
        // The context's tokens up to the last place the prompt still spelled it, when a checkpoint there keeps more.
        if (std::ranges::none_of(saved_, [&](const Saved& s) { return s.cp->pos > d && s.cp->pos <= upto; })) {
            if (stop)
                *stop = std::pair{upto, at};
            return std::nullopt;
        }
        r.kept = upto;
        r.prompt.assign(ctx_.begin(), ctx_.begin() + upto);
    }
    r.prompt.insert(r.prompt.end(), prompt.begin() + at, prompt.end());
    const int64_t shift = int64_t(r.prompt.size()) - int64_t(n);
    r.rows.assign(rows.begin(), rows.end());
    for (auto& row : r.rows)
        if (row.rows.pos >= at)
            row.rows.pos = uint32_t(int64_t(row.rows.pos) + shift);
    return r;
}

GenerateResult::Divergence Session::divergence(size_t at, std::span<const int32_t> prompt, size_t prompt_at) const
{
    GenerateResult::Divergence d;
    d.at = uint32_t(at);
    d.held = uint32_t(ctx_.size());
    d.from = uint32_t(at - std::min<size_t>(at, 8));
    d.context.assign(ctx_.begin() + d.from, ctx_.begin() + std::min(ctx_.size(), at + 16));
    d.prompt_at = uint32_t(prompt_at);
    d.prompt_from = uint32_t(prompt_at - std::min<size_t>(prompt_at, 8));
    d.prompt.assign(prompt.begin() + d.prompt_from, prompt.begin() + std::min(prompt.size(), prompt_at + 16));
    return d;
}

GenerateResult Session::generate(std::span<const int32_t> prompt, const GenerateParams& p,
                                 const std::function<bool(int32_t)>& on_token)
{
    try {
        std::optional<Respelled> r;
        std::optional<std::pair<size_t, size_t>> stop;
        if (p.text && !parked_ && e_->position() == ctx_.size())
            r = respell(prompt, p.input_rows, *p.text, &stop);
        if (!r && stop) {
            const auto d = divergence(stop->first, prompt, stop->second);
            GenerateResult res = run(prompt, p, on_token);
            if (res.diverged)
                res.diverged = d;
            return res;
        }
        if (r) {
            GenerateParams q = p;
            q.input_rows = std::move(r->rows);
            GenerateResult res = run(r->prompt, q, on_token);
            res.respelled = uint32_t(r->kept - r->at);
            res.respelled_at = uint32_t(r->at);
            return res;
        }
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
    if (prompt.size() > e_->max_context())
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
    if (common < ctx_.size() && common < limit)
        res.diverged = divergence(common, prompt, common);
    rewind(common);
    res.reused_tokens = uint32_t(ctx_.size());
    // The engine reads the rows only while the prompt runs.
    std::vector<GenerateParams::PromptRows> given = p.input_rows;
    if (p.fill_rows)
        p.fill_rows(ctx_.size(), given);
    std::vector<InputRows> rows;
    for (const auto& r : given) {
        if (!r.rows.rows && r.rows.pos + r.rows.count > ctx_.size())
            throw std::logic_error("input rows the prompt reads are missing");
        rows.push_back(r.rows);
    }
    e_->set_input_rows(rows);
    struct Unset {
        Model& e;
        ~Unset() { e.set_input_rows({}); }
    } unset{*e_};

    const auto t_prefill = std::chrono::steady_clock::now();
    const auto rest = prompt.subspan(ctx_.size());
    if (!ctx_.empty() && mtp_next_ != rest[0])
        e_->mtp_seed(rest[0]);
    if (p.progress)
        p.progress(ctx_.size(), prompt.size());
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
            std::function<void(size_t)> progress;
            if (p.progress)
                progress = [&, at](size_t n) { p.progress(at + n, prompt.size()); };
            e_->prefill(prompt.subspan(at, to - at), p.cancelled, progress);
            if (p.progress && e_->position() + 1 < prompt.size())
                p.progress(e_->position(), prompt.size());
            if (e_->position() != to || to == end)
                break;
            at = to;
            ctx_.assign(prompt.begin(), prompt.begin() + at);
            keep_keys(at);
            if (e_->has_mtp())
                e_->mtp_seed(prompt[at]);
            if (p.cancelled && p.cancelled())
                break;
            if (p.yield)
                p.yield();
        }
        const size_t read = e_->position();
        ctx_.assign(prompt.begin(), prompt.begin() + read);
        keep_keys(read);
        if (read == end) {
            // The next-token block learns the token after the part (within it, prefill gave it each next token).
            if (e_->has_mtp())
                e_->mtp_seed(prompt[read]);
            return true;
        }
        // The next-token block's cache already holds the last position read with the prompt's next token.
        mtp_next_ = e_->has_mtp() ? prompt[read] : -1;
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
    const std::vector<float>& lp = e_->step(prompt.back());
    ctx_.push_back(prompt.back());
    e_->set_input_rows({});
    res.prefill_ms = ms_since(t_prefill);

    const auto t_decode = std::chrono::steady_clock::now();
    const uint32_t nv = e_->n_vocab();
    Sampler sampler(p.sampling, nv, p.seed);
    const bool spec = p.speculative && e_->has_mtp();
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
        if (e_->position() >= e_->max_context()) {
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
    const bool lookup = p.speculative && p.lookup_match && e_->can_verify();
    if (p.speculative && (e_->has_drafter() || lookup)) {
        // Each round: drafts after `token` (a repeat of the context when one matches, else the engine's drafter),
        // then token and the drafts in one verify; drafts are kept up to the first the sampler rejects (it then
        // gives the token in its place), and after the last kept the trunk's next.
        while (p.max_tokens > 0 && emit(token)) {
            if (p.yield)
                p.yield();
            std::vector<int32_t> drafts;
            bool from_lookup = false;
            if (lookup) {
                drafts = lookup_.propose(ctx_, token, p.lookup_match);
                from_lookup = !drafts.empty();
            }
            if (!from_lookup && e_->has_drafter())
                drafts = e_->draft(token);
            const uint32_t room = std::min<uint32_t>(e_->max_context() - e_->position(),
                                                     p.max_tokens - uint32_t(res.tokens.size()) + 1);
            if (drafts.size() + 1 > room)
                drafts.resize(room > 0 ? room - 1 : 0);
            if (drafts.empty()) {
                const std::vector<float>& l = e_->step(token);
                ctx_.push_back(token);
                token = choose(l.data());
                continue;
            }
            std::vector<int32_t> run{token};
            run.insert(run.end(), drafts.begin(), drafts.end());
            const std::vector<float>& rows = e_->verify(run);
            res.drafts += uint32_t(drafts.size());
            if (from_lookup)
                res.lookup_drafts += uint32_t(drafts.size());
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
            e_->keep(kept + 1);
            ctx_.insert(ctx_.end(), run.begin(), run.begin() + kept + 1);
            if (from_lookup) {
                lookup_.outcome(kept);
                res.lookup_accepted += kept;
            }
            if (!more)
                break;
            token = next >= 0 ? next : choose(rows.data() + uint64_t(kept) * nv);
        }
        res.decode_ms = ms_since(t_decode);
        save_checkpoint(); // where the reply ended
        return res;
    }
    int32_t draft = spec ? e_->mtp_draft(token) : -1;
    mtp_next_ = spec ? token : -1;
    while (p.max_tokens > 0 && emit(token)) {
        if (p.yield)
            p.yield();
        if (draft >= 0 && e_->position() + 2 <= e_->max_context()) {
            // Verify [token, draft] in one two-token step.
            const std::vector<float>& two = e_->step2(token, draft);
            ctx_.push_back(token);
            ctx_.push_back(draft);
            ++res.drafts;
            bool accepted = false;
            const int32_t f = p.forced ? p.forced(two.data()) : -1;
            const int32_t t1 = f >= 0 ? (accepted = f == draft, f) : sampler.verify(two.data(), draft, accepted);
            if (!accepted) {
                e_->rollback();
                ctx_.pop_back();
                token = t1;
                draft = e_->mtp_draft(token);
                mtp_next_ = token;
                continue;
            }
            ++res.accepted;
            if (!emit(draft)) {
                // The block's cache at token's position still needs the draft as its next token.
                e_->mtp_draft(draft);
                mtp_next_ = -1;
                break;
            }
            token = choose(two.data() + nv);
            const int32_t both[2] = {draft, token};
            draft = e_->mtp_draft(std::span<const int32_t>(both));
            mtp_next_ = token;
        } else {
            const std::vector<float>& l = e_->step(token);
            ctx_.push_back(token);
            token = choose(l.data());
            draft = spec ? e_->mtp_draft(token) : -1;
            mtp_next_ = spec ? token : -1;
        }
    }
    res.decode_ms = ms_since(t_decode);
    save_checkpoint(); // where the reply ended
    return res;
}

} // namespace mako::gen
