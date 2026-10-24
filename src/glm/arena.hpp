#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::glm {

// What the arena budget depends on, in bytes.
struct MemInfo {
    double phys = 0;          // RAM
    double working_set = 0;   // the Metal device's recommended working set
    double footprint = 0;     // what the process holds now (phys_footprint)
    double filecache_min = 0; // the file cache macOS keeps before it compresses anonymous memory
    double available = 0;     // free + purgeable + clean file cache
};
MemInfo query_memory(uint64_t working_set);

// Bytes the expert arena may take. The engine's whole anonymous footprint stays under ~60% of a big Mac's RAM
// (and leaves the file-cache floor plus headroom): above that, a large file write beside the engine pushed its
// memory into the compressor and froze the machine. Never less than 1 GB.
double arena_budget(const MemInfo& m);

// Routed experts kept resident in mlock'ed anonymous memory, wrapped no-copy into Metal buffers. Each MoE layer
// and part (gate, up, down) has a region of equal-stride slots: slot s is at region.buf + region.off + s * stride,
// and the same s holds one expert's three parts. Kernels consume (region, slot ids), whichever family they target.
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
    };
    struct Stats {
        uint64_t hits = 0, misses = 0, read_bytes = 0;
        double read_ms = 0, prefill_ms = 0;
    };

    // `counts` (per layer, n_expert routing counts from a profile; may be empty) decides which experts start
    // resident and how many slots each layer gets: slots go to the most used (layer, expert) pairs per byte, at
    // least n_used per layer. Without counts every layer gets the same number of slots, filled in id order.
    ExpertArena(MTL::Device* dev, uint64_t max_buffer, std::vector<LayerParts> layers, uint32_t n_expert,
                uint32_t n_used, double budget_bytes, const std::vector<std::vector<uint64_t>>& counts = {});
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

    // Loads each MoE layer's first slots(layer) experts in profile order (id order without a profile).
    void prefill();
    // Makes the selected experts of a layer resident, reading the misses into the slots of the least used experts
    // (profile counts plus selections so far; ties: least recently used), and writes their slots to `slots_out`.
    // Never evicts a selected expert, so the GPU may read those meanwhile; no other slot of the layer may be in use.
    void ensure(uint32_t layer, std::span<const int> experts, int32_t* slots_out);

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
    std::vector<std::vector<uint64_t>> score_; // profile counts plus usage: the least scored resident is evicted
    uint64_t bytes_ = 0;
    std::vector<std::pair<void*, uint64_t>> chunks_;
    std::vector<mtl::Ref<MTL::Buffer>> bufs_;
    mtl::Ref<MTL::Buffer> table_buf_;
    std::vector<std::vector<int32_t>> slot_expert_; // [layer][slot] -> expert, -1 free
    std::vector<std::vector<uint64_t>> last_use_;   // [layer][slot]
    uint64_t clock_ = 0;
    Stats stats_;
};

} // namespace mako::glm
