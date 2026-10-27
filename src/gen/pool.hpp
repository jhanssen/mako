#pragma once

#include "gen/session.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mako::gen {

// Conversations over one engine: the session generating holds the engine's context, the others theirs parked in
// memory (Session::park) within a byte budget, the least recently used given up first.
class SessionPool {
public:
    SessionPool(Model& model, uint64_t budget_bytes, size_t max_sessions = 16);

    struct Pick {
        Session* session = nullptr;
        size_t index = 0;   // stable while the session lives
        size_t usable = 0;  // prompt tokens it can keep
        bool fresh = false; // no session could keep any: an empty one
        uint32_t evicted = 0;
    };
    // The session for this prompt, made the engine's: the one that can keep most of it (ties: the most recently
    // used), else an empty one. The previous one is parked, and parked sessions past the budget or the count are
    // given up, oldest first.
    Pick acquire(std::span<const int32_t> prompt, std::span<const GenerateParams::PromptRows> rows);

    size_t size() const { return slots_.size(); }
    uint64_t parked_bytes() const;
    uint64_t budget() const { return budget_; }

private:
    struct Slot {
        std::unique_ptr<Session> session;
        size_t index = 0;
        uint64_t used = 0;
    };
    uint32_t trim(const Session* keep);

    Model& model_;
    uint64_t budget_;
    size_t max_sessions_;
    std::vector<Slot> slots_;
    Session* live_ = nullptr;
    uint64_t clock_ = 0;
    size_t next_index_ = 0;
};

} // namespace mako::gen
