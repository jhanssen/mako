#pragma once

#include "gen/session.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace mako::gen {

// Conversations over one engine: the session generating holds the engine's context, the others theirs parked in
// memory (Session::park) within a byte budget, the least recently used given up first. Requests from several
// threads take turns: one holds the engine at a time, and while others wait it gives the engine up every time slice
// (between decode steps or prefill pieces) and queues again.
class SessionPool {
public:
    SessionPool(Model& model, uint64_t budget_bytes, size_t max_sessions = 16,
                std::chrono::milliseconds slice = std::chrono::milliseconds(1000));

    struct Pick {
        Session* session = nullptr;
        size_t index = 0;   // stable while the session lives
        size_t usable = 0;  // prompt tokens it can keep
        bool fresh = false; // no session could keep any: an empty one
        bool forked = false; // a new session starting from another's checkpoint (usable tokens of it)
        size_t forked_from = 0;
        uint32_t evicted = 0;
        uint32_t switches = 0; // times the request gave the engine up and took it back
        double switch_ms = 0;  // taking its context back in those
        size_t conversations = 0; // kept when the request ended, and their parked bytes
        uint64_t parked = 0;
    };

    // Runs a request: waits for its turn (arrival order; p.cancelled true while waiting gives up), runs `start`
    // with the engine (it may complete p, e.g. its input rows; false: give up), then generates on the conversation
    // acquire() picks. Returns false without generating when it gave up.
    bool generate(std::span<const int32_t> prompt, GenerateParams p, const std::function<bool(int32_t)>& on_token,
                  GenerateResult& result, Pick& pick, const std::function<bool(GenerateParams&)>& start = {});

    // The session for this prompt, made the engine's: the one not in use that can keep most of it in place (ties:
    // the most recently used); or, when another's earlier checkpoint holds more of it, a new session forked from
    // there; else an empty one. The previous one is parked, and parked sessions past the budget or the count are
    // given up, oldest first. Marks it in use until release(). The caller holds the turn (or is the only thread).
    Pick acquire(std::span<const int32_t> prompt, std::span<const GenerateParams::PromptRows> rows);
    void release(const Session* s);

    size_t size() const { return slots_.size(); }
    uint64_t parked_bytes() const;
    uint64_t budget() const { return budget_; }

private:
    struct Slot {
        std::unique_ptr<Session> session;
        size_t index = 0;
        uint64_t used = 0;
        bool busy = false;
    };
    uint32_t trim();
    void make_live(Session* s);
    bool take_turn(const std::function<bool()>& cancelled);
    void give_turn();
    bool others_waiting();

    Model& model_;
    uint64_t budget_;
    size_t max_sessions_;
    std::chrono::milliseconds slice_;
    std::vector<Slot> slots_;
    Session* live_ = nullptr;
    uint64_t clock_ = 0;
    size_t next_index_ = 0;

    std::mutex turn_mutex_;
    std::condition_variable turn_cv_;
    std::deque<uint64_t> queue_; // waiting requests' tickets, in arrival order
    bool held_ = false;
    uint64_t next_ticket_ = 0;
};

} // namespace mako::gen
