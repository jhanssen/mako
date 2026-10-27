#include "gen/pool.hpp"

#include <algorithm>

namespace mako::gen {

SessionPool::SessionPool(Model& model, uint64_t budget_bytes, size_t max_sessions)
    : model_(model), budget_(budget_bytes), max_sessions_(std::max<size_t>(1, max_sessions))
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

uint32_t SessionPool::trim(const Session* keep)
{
    uint32_t evicted = 0;
    while (true) {
        const bool over = slots_.size() > max_sessions_ || parked_bytes() > budget_;
        if (!over)
            return evicted;
        auto victim = slots_.end();
        for (auto it = slots_.begin(); it != slots_.end(); ++it)
            if (it->session.get() != live_ && it->session.get() != keep &&
                (victim == slots_.end() || it->used < victim->used))
                victim = it;
        if (victim == slots_.end())
            return evicted;
        slots_.erase(victim);
        ++evicted;
    }
}

SessionPool::Pick SessionPool::acquire(std::span<const int32_t> prompt,
                                       std::span<const GenerateParams::PromptRows> rows)
{
    Pick pick;
    Slot* best = nullptr;
    for (Slot& s : slots_) {
        const size_t u = s.session->usable_prefix(prompt, rows);
        if (u > 0 && (!best || u > pick.usable || (u == pick.usable && s.used > best->used))) {
            best = &s;
            pick.usable = u;
        }
    }
    if (!best) {
        pick.fresh = true;
        slots_.push_back({std::make_unique<Session>(model_), next_index_++, 0});
        best = &slots_.back();
    }
    best->used = ++clock_;
    Session* s = best->session.get();
    pick.session = s;
    pick.index = best->index;
    if (live_ != s) {
        if (live_)
            live_->park();
        s->unpark();
        live_ = s;
    }
    pick.evicted = trim(s);
    return pick;
}

} // namespace mako::gen
