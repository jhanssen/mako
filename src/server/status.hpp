#pragma once

#include "gen/pool.hpp"
#include "gen/session.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <string>

namespace mako::server {

// The expert cache's hit and miss counts so far.
struct CacheCounters {
    uint64_t hits = 0, misses = 0;
};

// What GET /v1/status reports besides the board: the server's fixed facts and the moment's.
struct StatusHeader {
    std::string model;
    uint32_t context = 0;
    double cache_gb = 0, resident = 0; // the expert cache's size and the share of the experts it holds
    uint64_t park_budget = 0;
    uint32_t in_flight = 0;
    std::string loaded = "llm"; // the model in memory: llm, image or none (swapping)
};

// Requests in progress and recently finished, and the expert cache's counters; any thread.
class StatusBoard {
public:
    using Clock = std::chrono::steady_clock;

    void arrived(const std::string& id, size_t prompt);
    void started(const std::string& id);                              // its turn came
    void decoding(const std::string& id, const CacheCounters& cache); // its first token
    void generated(const std::string& id, size_t n);
    // Done: moves it to the recent ones (finish empty: it gave up waiting).
    void finished(const std::string& id, const gen::GenerateResult& r, const char* finish, size_t conversation,
                  const CacheCounters& cache);
    void cache(const CacheCounters& c);

    std::string json(const StatusHeader& h, const gen::SessionPool::Info& pool) const;

private:
    struct Active {
        std::string id;
        size_t prompt = 0, generated = 0;
        Clock::time_point arrived, started, decoding;
        bool running = false, decoding_now = false;
        CacheCounters cache;
    };
    struct Done {
        std::string id, finish;
        size_t prompt = 0, reused = 0, generated = 0, conversation = 0;
        double prefill_ms = 0, decode_ms = 0, misses_per_token = 0;
        Clock::time_point ended;
    };
    Active* find(const std::string& id);

    mutable std::mutex mutex_;
    Clock::time_point up_ = Clock::now();
    std::list<Active> active_;
    std::deque<Done> recent_; // newest first
    CacheCounters cache_;
    uint64_t served_ = 0;
};

} // namespace mako::server
