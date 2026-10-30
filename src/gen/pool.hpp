#pragma once

#include "gen/session.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <span>
#include <unordered_set>
#include <vector>

namespace mako::gen {

// Conversations over one engine: the session generating holds the engine's context, the others theirs parked in
// memory (Session::park) within a byte budget, the least recently used given up first. Requests from several
// threads take turns: one holds the engine at a time, and while others wait it gives the engine up every time slice
// (between decode steps or prefill pieces) and queues again.
class SessionPool {
public:
    // identity: the model and the options its state depends on (resume() keeps the conversations only for the same).
    SessionPool(Model& model, uint64_t budget_bytes, size_t max_sessions = 16,
                std::chrono::milliseconds slice = std::chrono::milliseconds(1000), std::string identity = {});

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

    // Takes the engine away: waits for the turn (requests that come meanwhile wait for resume()), parks the
    // conversation the engine holds, and lets go of the engine, which may then be destroyed. The conversations stay
    // in memory.
    void suspend();
    // Gives the pool an engine again (a new one of the same model) and lets requests run. With another identity
    // the conversations are given up.
    void resume(Model& model, const std::string& identity);
    bool suspended() const { return model_ == nullptr; }
    // The conversations as of the last change to them, and the requests waiting for a turn; any thread.
    struct Info {
        struct Conversation {
            size_t index = 0, tokens = 0;
            bool live = false, busy = false; // the engine's context; a request's
            uint64_t bytes = 0;              // parked
            std::chrono::steady_clock::time_point used;
        };
        std::vector<Conversation> conversations;
        uint64_t parked = 0;
        size_t waiting = 0;
    };
    Info info() const;
    // Records the conversations for info(); by the thread generating (between tokens), for a growing context.
    void snapshot();

private:
    struct Slot {
        std::unique_ptr<Session> session;
        size_t index = 0;
        uint64_t used = 0;
        bool busy = false;
        std::chrono::steady_clock::time_point used_at = std::chrono::steady_clock::now();
    };
    uint32_t trim();
    void make_live(Session* s);
    bool take_turn(const std::function<bool()>& cancelled);
    void give_turn();
    bool others_waiting();

    Model* model_;
    std::string identity_;
    uint64_t budget_;
    size_t max_sessions_;
    std::chrono::milliseconds slice_;
    std::vector<Slot> slots_;
    Session* live_ = nullptr;
    uint64_t clock_ = 0;
    size_t next_index_ = 0;

    mutable std::mutex info_mutex_;
    Info info_;

    mutable std::mutex turn_mutex_;
    std::condition_variable turn_cv_;
    std::deque<uint64_t> queue_; // waiting requests' tickets, in arrival order
    bool held_ = false;
    uint64_t next_ticket_ = 0;
};

} // namespace mako::gen
