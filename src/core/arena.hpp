#pragma once

#include "gguf/gguf.hpp"
#include "gguf/read_pool.hpp"
#include "mtl/context.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mako::core {

// What the arena budget depends on, in bytes.
struct MemInfo {
    double phys = 0;          // RAM
    double working_set = 0;   // the Metal device's recommended working set
    double footprint = 0;     // what the process holds now (phys_footprint)
    double filecache_min = 0; // the file cache macOS keeps before it compresses anonymous memory
    double available = 0;     // free + purgeable + clean file cache
};
MemInfo query_memory(uint64_t working_set);

// Bytes the expert arena may take: what is available now (less a reserve of 15% of RAM), within the GPU working set,
// and the engine's whole anonymous footprint under RAM less a margin: a quarter of RAM from 96 GB up, 40% from 64 GB
// (and at least the file-cache floor plus 12 GB). A large file write beside a bigger footprint once pushed the
// engine into the compressor and froze the machine. Never less than 1 GB.
double arena_budget(const MemInfo& m);

// Anonymous memory locked for the object's lifetime (to make the machine look smaller than it is).
class LockedMemory {
public:
    explicit LockedMemory(uint64_t bytes);
    ~LockedMemory();
    LockedMemory(const LockedMemory&) = delete;
    LockedMemory& operator=(const LockedMemory&) = delete;

private:
    void* mem_ = nullptr;
    uint64_t bytes_ = 0;
};

// Routed experts kept resident in mlock'ed anonymous memory, wrapped no-copy into Metal buffers. Each MoE layer
// and part (gate, up, down) has a region of equal-stride slots: slot s < split is at buf + off + s * stride, a
// lendable slot s >= split at tail + tail_off + (s - split) * stride; the same s holds one expert's three parts.
// Kernels consume (region, slot ids), whichever family they target.
// Each layer's lendable slots are one buffer that lend() can release, returning its memory to the system (a wrapped
// page stays held by Metal while its buffer lives), and reclaim() rebuilds.
class ExpertArena {
public:
    struct Part {
        int fd = -1;              // F_NOCACHE descriptor of the shard
        uint64_t file_off = 0;    // expert 0's bytes in the file
        uint64_t bytes = 0;       // one expert's bytes
        gguf::Type type = gguf::Type::F32;
        uint32_t k = 0, n = 0;
    };
    using LayerParts = std::array<Part, 3>; // gate, up, down; bytes == 0: not a MoE layer
    struct Region {
        MTL::Buffer* buf = nullptr;
        uint64_t off = 0, stride = 0;
        MTL::Buffer* tail = nullptr; // null while lent
        uint64_t tail_off = 0;
        uint32_t split = 0;
        std::pair<MTL::Buffer*, uint64_t> at(uint32_t slot) const
        {
            return slot < split ? std::pair{buf, off + uint64_t(slot) * stride}
                                : std::pair{tail, tail_off + uint64_t(slot - split) * stride};
        }
    };
    struct Stats {
        uint64_t hits = 0, misses = 0, read_bytes = 0;
        double read_ms = 0, prefill_ms = 0;
    };

    // `counts` (per layer, n_expert routing counts from a profile; may be empty) decides which experts start
    // resident and how many slots each layer gets: slots go to the most used (layer, expert) pairs per byte, at
    // least n_used per layer. Without counts every layer gets the same number of slots, filled in id order.
    // At least `lendable_bytes` of the budget become lendable slots, the same number in every layer.
    ExpertArena(MTL::Device* dev, uint64_t max_buffer, std::vector<LayerParts> layers, uint32_t n_expert,
                uint32_t n_used, double budget_bytes, const std::vector<std::vector<uint64_t>>& counts = {},
                double lendable_bytes = 0);
    ~ExpertArena();
    ExpertArena(const ExpertArena&) = delete;
    ExpertArena& operator=(const ExpertArena&) = delete;

    uint32_t slots(uint32_t layer) const { return slots_[layer]; }
    uint64_t bytes() const { return bytes_; }
    const Part& part(uint32_t layer, int p) const { return layers_[layer][p]; }
    const Region& region(uint32_t layer, int p) const { return regions_[layer][p]; }
    // Expert -> slot (-1: not resident), [n_layer][n_expert] int32, kept current for GPU-side lookup.
    MTL::Buffer* slot_table() const { return table_buf_.get(); }
    const std::vector<mtl::Ref<MTL::Buffer>>& buffers() const { return bufs_; }
    std::vector<MTL::Buffer*> tail_buffers() const;
    uint64_t lendable_bytes() const { return tail_bytes_; }

    // Evicts the experts in whole layers' lendable slots (the last layers first) until at least `bytes` are lent,
    // and frees their memory; ensure() then uses only those layers' other slots.
    void lend(uint64_t bytes = UINT64_MAX);
    // Rebuilds the lent slots and fills the free slots with the best-scoring experts not resident. Returns false,
    // leaving some lent, when their memory cannot be locked again.
    bool reclaim();
    bool lent() const;

    // Loads each MoE layer's first slots(layer) experts in profile order (id order without a profile).
    void prefill();
    // Makes the selected experts of a layer resident, reading the misses into the slots of the least used experts
    // (profile counts plus selections so far; ties: least recently used), and writes their slots to `slots_out`.
    // Never evicts a selected or pinned expert, so the GPU may read those meanwhile; no other slot of the layer may
    // be in use. With `pin` the selected slots stay pinned until unpin(). Returns false, changing nothing, when the
    // misses do not fit in the unpinned slots.
    // `count_usage`: the selection counts once in usage() (prefill passes false and reports its per-token counts with
    // note_prefill); the eviction scores are updated either way.
    bool ensure(uint32_t layer, std::span<const int> experts, int32_t* slots_out, bool pin = false,
                bool count_usage = true);
    // The eviction scores' update of an ensure() of these experts, without making them resident (experts a prefill
    // runs from staging).
    void touch(uint32_t layer, std::span<const int> experts);
    // A prefill group's routing of a layer for usage() (the profile): users[e] tokens selected expert e.
    void note_prefill(uint32_t layer, std::span<const uint32_t> users);
    void unpin(uint32_t layer);
    void unpin(uint32_t layer, std::span<const int32_t> slots);

    const Stats& stats() const { return stats_; }
    // How often each expert was selected since construction, [layer][expert].
    const std::vector<std::vector<uint64_t>>& usage() const { return usage_; }

private:
    void read_into(uint32_t layer, std::span<const std::pair<int, int32_t>> loads); // (expert, slot)

    std::vector<LayerParts> layers_;
    std::vector<std::array<Region, 3>> regions_;
    uint32_t n_expert_ = 0;
    std::vector<uint32_t> slots_;              // per layer
    std::vector<std::vector<int32_t>> order_;  // per layer, experts in prefill order
    std::vector<std::vector<uint64_t>> usage_;
    std::vector<std::vector<double>> score_;   // decayed usage, seeded from the profile: the least scored resident is evicted
    uint64_t bytes_ = 0;
    std::vector<std::pair<void*, uint64_t>> chunks_;
    std::vector<mtl::Ref<MTL::Buffer>> bufs_;
    MTL::Device* dev_ = nullptr;
    struct Tail { // a layer's lendable slots
        void* mem = nullptr; // null while lent
        uint64_t bytes = 0;
        mtl::Ref<MTL::Buffer> buf;
    };
    bool map_tail(uint32_t layer); // false: the memory could not be locked
    std::vector<Tail> tails_;
    uint64_t tail_bytes_ = 0; // all layers'
    mtl::Ref<MTL::Buffer> table_buf_;
    std::vector<std::vector<int32_t>> slot_expert_; // [layer][slot] -> expert, -1 free
    std::vector<std::vector<uint64_t>> last_use_;   // [layer][slot]
    std::vector<std::vector<char>> pinned_;         // [layer][slot]
    uint64_t clock_ = 0;
    Stats stats_;
    std::unique_ptr<gguf::ReadPool> readers_;
};

} // namespace mako::core
