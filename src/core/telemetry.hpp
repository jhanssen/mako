#pragma once

#include "gpu/runtime.hpp"
#include "mtl/context.hpp"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace mako::core {

using gpu::KernelTime;
using gpu::parse_telemetry;
using gpu::Telemetry;

struct MachineInfo {
    std::string commit, chip, macos, macos_build;
    uint32_t perf_cores = 0, eff_cores = 0;
    uint64_t ram = 0, wired_limit_mb = 0; // 0: macOS default
    mtl::DeviceInfo gpu;
};
MachineInfo query_machine(const mtl::DeviceInfo& gpu);
// macOS's thermal state now: "nominal", "fair", "serious" or "critical". Timings taken above nominal run throttled.
std::string thermal_state();

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
    bool mla = false;               // attention kind: MLA, else KDA
    double attn_ms = 0, ffn_ms = 0; // ffn: up to routing for MoE layers, the whole FFN for dense ones
    double hit_ms = 0, miss_ms = 0; // MoE: resident experts, then the misses and the combine (prefill: all
                                    // experts, then the combine)
};

struct StepStats {
    uint32_t pos = 0;
    int32_t token = 0;
    uint32_t tokens = 1; // run in the step (a drafted step runs its drafts too, kept or not)
    double wall_ms = 0, encode_ms = 0, gpu_ms = 0;
    double prep_ms = 0;       // CPU work before encoding (inputs read from files)
    double route_wait_ms = 0; // CPU waiting for the GPU's routing results
    double ensure_ms = 0;     // CPU making selected experts resident, misses read included
    double read_ms = 0;
    uint64_t hits = 0, misses = 0, read_bytes = 0;
    std::vector<LayerTimes> layers;
    std::vector<KernelTime> parts; // layers level with part timing on: GPU time by named part of the layers
    std::vector<KernelTime> kernels;
};

// Collects telemetry and writes it as JSON lines (one object per record) and as a Markdown summary.
class Report {
public:
    Report(Telemetry level, std::ostream* jsonl) : level_(level), out_(jsonl) {}

    void machine(const MachineInfo& m);
    void load(const LoadStats& l);
    void step(const StepStats& s);
    // One prefill: its token count and totals (pos: first position; layers summed over its chunks).
    void prefill(const StepStats& s, uint32_t tokens);
    // Averages leave out the first `skip` steps (first-use costs).
    std::string markdown(size_t skip = 1) const;

private:
    Telemetry level_;
    std::ostream* out_;
    MachineInfo machine_;
    LoadStats load_;
    std::vector<StepStats> steps_;
    std::vector<std::pair<StepStats, uint32_t>> prefills_;
};

} // namespace mako::core
