#pragma once

#include "mtl/context.hpp"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace mako::glm {

// How much the engine measures and reports: summary (machine, load, per-token averages), steps (every step too),
// layers (every step's per-layer GPU time too), kernels (GPU time per kernel instead of per layer; every dispatch
// gets its own encoder, which adds overhead, so absolute times run high).
enum class Telemetry { off, summary, steps, layers, kernels };
std::optional<Telemetry> parse_telemetry(std::string_view s);

struct MachineInfo {
    std::string commit, chip, macos, macos_build;
    uint32_t perf_cores = 0, eff_cores = 0;
    uint64_t ram = 0, wired_limit_mb = 0; // 0: macOS default
    mtl::DeviceInfo gpu;
};
MachineInfo query_machine(const mtl::DeviceInfo& gpu);

struct LoadStats {
    double dense_ms = 0, key_ms = 0, arena_ms = 0, prefill_ms = 0, warm_ms = 0, total_ms = 0;
    uint64_t dense_bytes = 0, arena_bytes = 0;
    double arena_budget = 0;
    uint32_t slots_min = 0, slots_max = 0;
    std::string profile; // routing profile in use; empty: none
};

// GPU time of one layer's sections, from encoder-boundary timestamps.
struct LayerTimes {
    uint32_t layer = 0;
    double attn_ms = 0, ffn_ms = 0; // ffn: up to routing for MoE layers, the whole FFN for dense ones
    double hit_ms = 0, miss_ms = 0; // MoE: resident experts, then the misses and the combine
};

struct KernelTime {
    std::string kernel;
    double ms = 0;
    uint32_t count = 0;
};

struct StepStats {
    uint32_t pos = 0;
    int32_t token = 0;
    double wall_ms = 0, encode_ms = 0, gpu_ms = 0;
    double route_wait_ms = 0; // CPU waiting for the GPU's routing results
    double ensure_ms = 0;     // CPU making selected experts resident, misses read included
    double read_ms = 0;
    uint64_t hits = 0, misses = 0, read_bytes = 0;
    std::vector<LayerTimes> layers;
    std::vector<KernelTime> kernels;
};

// Collects telemetry and writes it as JSON lines (one object per record) and as a Markdown summary.
class Report {
public:
    Report(Telemetry level, std::ostream* jsonl) : level_(level), out_(jsonl) {}

    void machine(const MachineInfo& m);
    void load(const LoadStats& l);
    void step(const StepStats& s);
    // Averages leave out the first `skip` steps (first-use costs).
    std::string markdown(size_t skip = 1) const;

private:
    Telemetry level_;
    std::ostream* out_;
    MachineInfo machine_;
    LoadStats load_;
    std::vector<StepStats> steps_;
};

} // namespace mako::glm
