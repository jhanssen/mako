#pragma once

#include "core/arena.hpp"
#include "mtl/context.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace mako::core {

// Reads routed experts that a prefill group will need but the arena lacks into a ring of staging slots, ahead of
// the GPU: a large group routes to nearly every expert of every layer, so the reads need no routing and start
// before a layer's router has run. Jobs are read in plan order by a few threads, each into slot (job % slots) once
// the job a ring-length earlier is released.
class Stager {
public:
    // `bytes`: the ring's size; a slot holds one expert's three parts.
    Stager(MTL::Device* dev, const ExpertArena& arena, uint64_t bytes);
    ~Stager();
    Stager(const Stager&) = delete;
    Stager& operator=(const Stager&) = delete;

    bool ok() const { return slots_ > 0; }
    uint64_t bytes() const { return buf_ ? buf_->length() : 0; }

    // Starts reading `plan`: per arena row (layer), the experts to read, in the order they will be taken. Any
    // earlier plan is dropped (wait for its GPU work first).
    void start(std::vector<std::pair<uint32_t, std::vector<int>>> plan);
    // Stops the readers (jobs not read are dropped).
    void stop();

    // Where expert e of `layer` is staged, waiting until it is read; -1 when it is not in the plan. The slot stays
    // until release(job). `job` identifies it. A layer's experts must be taken in plan order: the jobs before e
    // not taken are released.
    struct Where {
        int64_t job = -1;
        MTL::Buffer* buf = nullptr;
        std::array<uint64_t, 3> off{}; // gate, up, down
    };
    Where take(uint32_t layer, int e);
    // Thread-safe (a command buffer's completion handler releases the slots it read).
    void release(int64_t job);
    // Releases the jobs of `layer` never taken (experts the routing did not select).
    void release_untaken(uint32_t layer);

    double read_ms() const { return read_ms_.load(); }
    // From start() to the last read's end.
    double span_ms() const { return span_ms_.load(); }
    uint64_t read_bytes() const { return read_bytes_.load(); }

private:
    struct Job {
        uint32_t layer;
        int expert;
    };
    void work(std::stop_token st);
    int64_t find(uint32_t layer, int e) const;

    const ExpertArena& arena_;
    mtl::Ref<MTL::Buffer> buf_;
    uint64_t slot_bytes_ = 0;
    uint32_t slots_ = 0;

    mutable std::mutex m_;
    std::condition_variable_any cv_;
    std::vector<Job> jobs_;
    std::map<std::pair<uint32_t, int>, int64_t> index_;
    std::vector<char> ready_, released_, taken_;
    size_t next_ = 0;            // the next job to read
    int64_t released_upto_ = 0;  // every job before it is released
    uint64_t generation_ = 0;
    std::exception_ptr error_;
    std::atomic<double> read_ms_{0}, span_ms_{0};
    std::chrono::steady_clock::time_point t_start_;
    std::atomic<uint64_t> read_bytes_{0};
    std::vector<std::jthread> threads_;
};

} // namespace mako::core
