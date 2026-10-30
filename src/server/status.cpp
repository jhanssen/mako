#include "server/status.hpp"

#include "core/telemetry.hpp"
#include "server/json.hpp"

#include <algorithm>
#include <optional>
#include <vector>

namespace mako::server {

namespace status_json { // named: glaze reflects only types with linkage

struct CacheJson {
    double gb = 0, resident = 0;
    uint64_t hits = 0, misses = 0;
    std::optional<double> hit_share;
};
struct ConversationJson {
    size_t index = 0, tokens = 0;
    std::string state; // live (the engine's context) or parked
    bool busy = false;
    double parked_mb = 0, idle_s = 0;
};
struct ConversationsJson {
    double parked_gb = 0, budget_gb = 0;
    std::vector<ConversationJson> list;
};
struct RequestJson {
    std::string id, phase; // waiting, reading (the prompt), generating
    size_t prompt = 0, generated = 0;
    double elapsed_s = 0;
    std::optional<double> tokens_per_s;
};
struct RecentJson {
    std::string id, finish;
    size_t prompt = 0, reused = 0, generated = 0, conversation = 0;
    std::optional<double> prefill_tokens_per_s, tokens_per_s, misses_per_token;
    double ago_s = 0;
};
struct StatusJson {
    std::string model, thermal, loaded;
    uint32_t context = 0, in_flight = 0;
    size_t waiting = 0;
    uint64_t served = 0;
    double uptime_s = 0;
    CacheJson expert_cache;
    ConversationsJson conversations;
    std::vector<RequestJson> requests;
    std::vector<RecentJson> recent;
};

} // namespace status_json

namespace {

using namespace status_json;
constexpr size_t kRecent = 16;

double seconds(StatusBoard::Clock::duration d)
{
    return std::chrono::duration<double>(d).count();
}

std::optional<double> rate(double n, double ms)
{
    return ms > 0 ? std::optional(n / ms * 1e3) : std::nullopt;
}

} // namespace

StatusBoard::Active* StatusBoard::find(const std::string& id)
{
    const auto it = std::ranges::find(active_, id, &Active::id);
    return it == active_.end() ? nullptr : &*it;
}

void StatusBoard::arrived(const std::string& id, size_t prompt)
{
    std::lock_guard lock(mutex_);
    active_.push_back({.id = id, .prompt = prompt, .arrived = Clock::now()});
}

void StatusBoard::started(const std::string& id)
{
    std::lock_guard lock(mutex_);
    if (Active* a = find(id)) {
        a->running = true;
        a->started = Clock::now();
    }
}

void StatusBoard::decoding(const std::string& id, const CacheCounters& cache)
{
    std::lock_guard lock(mutex_);
    cache_ = cache;
    if (Active* a = find(id)) {
        a->decoding_now = true;
        a->decoding = Clock::now();
        a->cache = cache;
    }
}

void StatusBoard::generated(const std::string& id, size_t n)
{
    std::lock_guard lock(mutex_);
    if (Active* a = find(id))
        a->generated = n;
}

void StatusBoard::cache(const CacheCounters& c)
{
    std::lock_guard lock(mutex_);
    cache_ = c;
}

void StatusBoard::finished(const std::string& id, const gen::GenerateResult& r, const char* finish,
                           size_t conversation, const CacheCounters& cache)
{
    std::lock_guard lock(mutex_);
    const auto it = std::ranges::find(active_, id, &Active::id);
    if (it == active_.end())
        return;
    if (finish) {
        Done d{.id = id,
               .finish = finish,
               .prompt = r.prompt_tokens,
               .reused = r.reused_tokens,
               .generated = r.tokens.size(),
               .conversation = conversation,
               .prefill_ms = r.prefill_ms,
               .decode_ms = r.decode_ms};
        if (it->decoding_now && !r.tokens.empty())
            d.misses_per_token = double(cache.misses - it->cache.misses) / double(r.tokens.size());
        d.ended = Clock::now();
        recent_.push_front(std::move(d));
        if (recent_.size() > kRecent)
            recent_.pop_back();
        cache_ = cache;
        ++served_;
    }
    active_.erase(it);
}

std::string StatusBoard::json(const StatusHeader& h, const gen::SessionPool::Info& pool) const
{
    const auto now = Clock::now();
    StatusJson s;
    s.model = h.model;
    s.loaded = h.loaded;
    s.thermal = core::thermal_state();
    s.context = h.context;
    s.in_flight = h.in_flight;
    s.waiting = pool.waiting;
    s.conversations.parked_gb = pool.parked / 1e9;
    s.conversations.budget_gb = h.park_budget / 1e9;
    for (const auto& c : pool.conversations)
        s.conversations.list.push_back(
            {c.index, c.tokens, c.live ? "live" : "parked", c.busy, c.bytes / 1e6, seconds(now - c.used)});
    std::lock_guard lock(mutex_);
    s.uptime_s = seconds(now - up_);
    s.served = served_;
    s.expert_cache = {.gb = h.cache_gb, .resident = h.resident, .hits = cache_.hits, .misses = cache_.misses};
    if (cache_.hits + cache_.misses)
        s.expert_cache.hit_share = double(cache_.hits) / double(cache_.hits + cache_.misses);
    for (const Active& a : active_) {
        RequestJson q{.id = a.id,
                      .phase = !a.running ? "waiting" : a.decoding_now ? "generating" : "reading",
                      .prompt = a.prompt,
                      .generated = a.generated,
                      .elapsed_s = seconds(now - a.arrived)};
        if (a.decoding_now && a.generated > 1)
            q.tokens_per_s = double(a.generated - 1) / seconds(now - a.decoding);
        s.requests.push_back(std::move(q));
    }
    for (const Done& d : recent_) {
        RecentJson j{.id = d.id,
                     .finish = d.finish,
                     .prompt = d.prompt,
                     .reused = d.reused,
                     .generated = d.generated,
                     .conversation = d.conversation};
        j.prefill_tokens_per_s = rate(double(d.prompt - d.reused), d.prefill_ms);
        j.tokens_per_s = rate(double(d.generated), d.decode_ms);
        if (d.generated)
            j.misses_per_token = d.misses_per_token;
        j.ago_s = seconds(now - d.ended);
        s.recent.push_back(std::move(j));
    }
    return json::write(s);
}

} // namespace mako::server
