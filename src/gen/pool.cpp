#include "gen/pool.hpp"

#include <algorithm>

namespace mako::gen {

SessionPool::SessionPool(Model& model, uint64_t budget_bytes, size_t max_sessions, std::chrono::milliseconds slice)
    : model_(model), budget_(budget_bytes), max_sessions_(std::max<size_t>(1, max_sessions)), slice_(slice)
{
}

uint64_t SessionPool::parked_bytes() const
{
    uint64_t n = 0;
    for (const Slot& s : slots_)
        if (s.session.get() != live_)
            n += s.session->bytes();
    return n;
}

uint32_t SessionPool::trim()
{
    uint32_t evicted = 0;
    while (slots_.size() > max_sessions_ || parked_bytes() > budget_) {
        auto victim = slots_.end();
        for (auto it = slots_.begin(); it != slots_.end(); ++it)
            if (it->session.get() != live_ && !it->busy && (victim == slots_.end() || it->used < victim->used))
                victim = it;
        if (victim == slots_.end())
            break;
        slots_.erase(victim);
        ++evicted;
    }
    return evicted;
}

void SessionPool::make_live(Session* s)
{
    if (live_ == s)
        return;
    if (live_)
        live_->park();
    s->unpark();
    live_ = s;
}

SessionPool::Pick SessionPool::acquire(std::span<const int32_t> prompt,
                                       std::span<const GenerateParams::PromptRows> rows)
{
    Pick pick;
    Slot* best = nullptr;
    for (Slot& s : slots_) {
        if (s.busy)
            continue;
        const size_t u = s.session->usable_prefix(prompt, rows);
        if (u > 0 && (!best || u > pick.usable || (u == pick.usable && s.used > best->used))) {
            best = &s;
            pick.usable = u;
        }
    }
    if (!best) {
        pick.fresh = true;
        slots_.push_back({std::make_unique<Session>(model_), next_index_++, 0, false});
        best = &slots_.back();
    }
    best->used = ++clock_;
    best->busy = true;
    Session* s = best->session.get();
    pick.session = s;
    pick.index = best->index;
    make_live(s);
    pick.evicted = trim();
    return pick;
}

void SessionPool::release(const Session* s)
{
    for (Slot& slot : slots_)
        if (slot.session.get() == s)
            slot.busy = false;
    trim();
}

bool SessionPool::take_turn(const std::function<bool()>& cancelled)
{
    std::unique_lock lock(turn_mutex_);
    const uint64_t ticket = next_ticket_++;
    queue_.push_back(ticket);
    while (held_ || queue_.front() != ticket) {
        turn_cv_.wait_for(lock, std::chrono::milliseconds(100));
        if (cancelled && cancelled() && (held_ || queue_.front() != ticket)) {
            std::erase(queue_, ticket);
            turn_cv_.notify_all();
            return false;
        }
    }
    queue_.pop_front();
    held_ = true;
    return true;
}

void SessionPool::give_turn()
{
    {
        std::lock_guard lock(turn_mutex_);
        held_ = false;
    }
    turn_cv_.notify_all();
}

bool SessionPool::others_waiting()
{
    std::lock_guard lock(turn_mutex_);
    return !queue_.empty();
}

bool SessionPool::generate(std::span<const int32_t> prompt, GenerateParams p,
                           const std::function<bool(int32_t)>& on_token, GenerateResult& result, Pick& pick,
                           const std::function<bool(GenerateParams&)>& start)
{
    if (!take_turn(p.cancelled))
        return false;
    struct Turn {
        SessionPool& pool;
        ~Turn() { pool.give_turn(); }
    } turn{*this};
    if (start && !start(p))
        return false;
    pick = acquire(prompt, p.input_rows);
    struct Release {
        SessionPool& pool;
        const Session* s;
        ~Release() { pool.release(s); }
    } release{*this, pick.session};
    auto slice_start = std::chrono::steady_clock::now();
    p.yield = [&] {
        if (std::chrono::steady_clock::now() - slice_start < slice_ || !others_waiting())
            return;
        give_turn();
        take_turn({});
        const auto t = std::chrono::steady_clock::now();
        make_live(pick.session);
        pick.switch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        ++pick.switches;
        slice_start = std::chrono::steady_clock::now();
    };
    result = pick.session->generate(prompt, p, on_token);
    pick.conversations = slots_.size();
    pick.parked = parked_bytes();
    return true;
}

} // namespace mako::gen
