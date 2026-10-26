#pragma once

#include "core/arena.hpp"
#include "core/profile.hpp"
#include "core/stager.hpp"
#include "core/telemetry.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "mtl/gemv.hpp"

#include <dispatch/dispatch.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mako::core {

// What the shared runtime needs to know of a model.
struct Dims {
    uint32_t n_embd = 0;
    uint32_t n_layer = 0; // trunk layers; MoE rows past them (a draft block) count as the rest in telemetry
    uint32_t hc = 4, hc_iters = 20;
    float hc_eps = 1e-6f, rms_eps = 1e-6f;
    uint32_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
};

// How a MoE layer turns router logits into the selected experts and their weights.
struct Routing {
    enum class Score : uint32_t { sigmoid = 0, sqrt_softplus = 1 };
    Score score = Score::sigmoid;
    bool norm = false; // weights normalized over the selected experts
    float scale = 1.0f;
    // Decode: subtracted from a non-resident expert's selection score (its weight unchanged), trading routing
    // fidelity for fewer expert reads; 0: off.
    float miss_penalty = 0.0f;
};

// The GPU runtime models share: resident dense weights, scratch buffers, kernel dispatch with telemetry and probes,
// and the routed-expert arena with decode's GPU routing served by the CPU (misses read between a layer's router and
// its experts). A model's engine derives from it and adds its layer graph.
class EngineBase {
public:
    struct DevTensor {
        MTL::Buffer* buf = nullptr;
        uint64_t off = 0;
        gguf::Type type = gguf::Type::F32;
        std::vector<uint64_t> dims;
    };
    // One MoE layer's dense parts and SwiGLU clamps (<= 0: none).
    struct MoeWeights {
        DevTensor router, bias, sh_gate, sh_up, sh_down;
        float clamp_exp = 0, clamp_sh = 0;
    };

    // Called after each named stage of a step with the fp32 values it produced (for parity checks).
    using Probe = std::function<void(const std::string& name, int layer, const float* data, size_t n)>;
    void set_probe(Probe probe) { probe_ = std::move(probe); }
    // Called after routing a MoE layer; may overwrite the selected experts and their final weights (k each).
    using RouteOverride = std::function<void(int layer, int* sel, float* w, size_t k)>;
    void set_route_override(RouteOverride f) { route_override_ = std::move(f); }

    struct Stats {
        double expert_read_ms = 0, gpu_wait_ms = 0, gpu_exec_ms = 0; // exec: GPU start to end, summed
        double encode_ms = 0, step_ms = 0; // encode: step start to the first commit
        uint64_t expert_bytes = 0;
    };
    const Stats& stats() const { return stats_; }

    // The memory figures the arena was sized from and its size (bytes).
    struct ArenaPlan {
        MemInfo mem;
        double budget = 0;
        double lendable = 0; // arena bytes prefill's scratch buffers borrow
        double warm_ms = 0;  // first GPU pass over the arena, at load
    };
    const ArenaPlan& arena_plan() const { return arena_plan_; }
    const ExpertArena& arena() const { return *arena_; }
    const LoadStats& load_stats() const { return load_; }
    const StepStats& last_step() const { return last_step_; }
    // Prefill at the layers level: also time named parts of each layer (StepStats::parts); adds encoders.
    void set_part_timing(bool on);
    // Bytes of all scratch buffers allocated so far.
    uint64_t scratch_bytes() const
    {
        uint64_t n = 0;
        for (const auto& [name, b] : scratch_)
            n += b->length();
        return n;
    }

    const std::string& routing_key() const { return routing_key_; }
    const std::filesystem::path& default_profile() const { return default_profile_; }
    const std::filesystem::path& loaded_profile() const { return loaded_profile_; } // empty: none
    // Writes the routing counts of this engine's steps to `path`, added to the file's counts when `merge`.
    void save_profile(const std::filesystem::path& path, bool merge) const;

protected:
    static constexpr uint32_t kMaxDecodeTokens = 8;
    // proj() of at least this many tokens uses the GEMM (fewer: the GEMV, which reads each weight once for them and
    // gives every token the same result as alone).
    uint32_t gemm_min_tokens_ = 5;

    EngineBase(const gguf::ModelFiles& model, mtl::Context& ctx, Telemetry telemetry);
    ~EngineBase();
    EngineBase(const EngineBase&) = delete;
    EngineBase& operator=(const EngineBase&) = delete;

    // Reads a dense tensor into the resident weight buffers (BF16 becomes F32).
    // Dense weights; Q8_0 matrices are re-quantized to dense_quant_ when it is set.
    DevTensor upload(const std::string& name);
    DevTensor upload_requant(int fd, const gguf::Tensor& t, gguf::Type to, const std::string& name);
    std::optional<gguf::Type> dense_quant_;
    // Per tensor name: the importance of each input column (k values, or k per group of rows for a tensor whose row
    // groups read different inputs), weighting the re-quantization's rounding.
    std::map<std::string, std::vector<float>> importance_;
    uint64_t requant_bytes_ = 0; // bytes the re-quantization saved
    // The same from `bytes` at `offset` of a file (tensors outside the GGUF, e.g. a drafter's own file).
    DevTensor upload_raw(int fd, uint64_t offset, uint64_t bytes, gguf::Type type, std::vector<uint64_t> dims,
                         const std::string& what);
    // Room for `bytes` in the resident weight buffers, for the caller to fill.
    DevTensor reserve(uint64_t bytes, gguf::Type type, std::vector<uint64_t> dims);
    // A routed experts tensor [k, n, n_expert] as the arena reads it; `prefill`: it also needs the prefill GEMM.
    ExpertArena::Part expert(const std::string& name, bool prefill);

    struct ArenaInit {
        std::vector<ExpertArena::LayerParts> parts; // per arena row (layer)
        std::vector<uint32_t> moe_layers;           // for the routing key
        uint64_t context_bytes = 0;                 // caches and checkpoints, taken from the budget
        double lendable = 0;
        double arena_gb = -1;      // < 0: from the memory budget
        double simulate_ram_gb = 0;
        std::string expert_counts; // profile; empty: the default one when it exists
        uint32_t draft_rows = 0;   // arena rows after the trunk (n_layer) that a profile may lack
    };
    // Sizes and fills the expert arena, then sets up telemetry and the memory-pressure source. Call last in the
    // derived constructor, after every upload() and with load_.dense_ms set.
    void init_runtime(ArenaInit init);

    MTL::Buffer* scratch(const std::string& name, uint64_t bytes);
    MTL::ComputePipelineState* pso(const std::string& fn);

    // batch > 1: that many matrices from w on, each its own y block of t rows; x_shared: all read the same x.
    void gemv(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t batch = 1, uint32_t t = 1, bool x_shared = false);
    // b's rows follow a's in memory, same type and shape: one batched GEMV can run both.
    static bool adjacent(const DevTensor& a, const DevTensor& b);
    void proj(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t T, uint32_t ldy = 0);
    void dispatch(MTL::ComputeCommandEncoder*& enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                  const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads = true);
    void norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const DevTensor* w, MTL::Buffer* y, uint32_t rows,
              uint32_t n, uint32_t wn, float eps, uint64_t x_off = 0, uint64_t y_off = 0, uint32_t ystride = 0);
    // x = mHC pre-mix of the streams R for T tokens; pc keeps each token's pre/post/comb for hc_post. R (and out)
    // at byte offset r_off: a chunk's rows of a longer stream.
    void hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T, uint64_t r_off = 0);
    void hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* R, MTL::Buffer* out, MTL::Buffer* pc,
                 uint32_t T, uint64_t r_off = 0, uint64_t x_off = 0, uint64_t pc_off = 0);
    // T tokens: hc_post(post_x, R -> R_out, pc) when post_x is set (else R_out is R), then hc_pre of the next block
    // on the new stream (fn, scale, base) into x and pc, in two dispatches.
    // With norm_w and xn: one dispatch that also writes xn = rms_norm(x) * norm_w (the token's last threadgroup).
    void hc_step(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* post_x, MTL::Buffer* R, MTL::Buffer* R_out,
                 const DevTensor& fn, const DevTensor& scale, const DevTensor& base, MTL::Buffer* x, MTL::Buffer* pc,
                 uint32_t T = 1, const DevTensor* norm_w = nullptr, MTL::Buffer* xn = nullptr);
    void ffn_dense(MTL::ComputeCommandEncoder*& enc, const DevTensor& gate, const DevTensor& up,
                   const DevTensor& down, MTL::Buffer* xn, MTL::Buffer* out, float limit, uint32_t T,
                   uint64_t x_off = 0, uint64_t out_off = 0);
    // T (<= kMaxDecodeTokens) tokens' routed FFN plus shared expert of arena row il, rows of xn into rows of ffo.
    // With `sho_out` the shared expert's output stays there and goes through `between` (an encoder hook, e.g. an
    // ablation) before it is combined.
    // With `next` (arena row il + ahead's weights) and predict_k, one token's step also predicts that layer's top
    // predict_k experts from this layer's FFN input, and the ones not resident are read ahead (ExpertArena::prefetch).
    void moe_decode(MTL::CommandBuffer* cb, MTL::ComputeCommandEncoder*& enc, const MoeWeights& w, uint32_t il,
                    MTL::Buffer* xn, MTL::Buffer* ffo, uint32_t T,
                    const std::function<void(MTL::ComputeCommandEncoder*&, MTL::Buffer* sho)>& between = {},
                    const MoeWeights* next = nullptr, uint32_t predict_k = 0, uint32_t ahead = 1);
    // block false: only the layers whose routing has already been signaled, stopping at the first that has not.
    void serve_moe(MTL::CommandBuffer* cb, bool block = true);
    // G tokens' routed experts of arena row il (rows of xn_g) added into ffo, which holds their shared expert's
    // output. Command buffers it commits go to `committed`: they must complete before the row's slots are used again.
    void moe_prefill(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc, const MoeWeights& w, uint32_t il,
                     MTL::Buffer* xn_g, MTL::Buffer* ffo, uint32_t G, std::vector<MTL::CommandBuffer*>& committed);
    static constexpr uint32_t kBatchRows = 8192; // prefill expert rows (token, expert pairs) run and combined together
    // Staged prefill: for a group of at least kStageMinTokens (which routes to nearly every expert of every layer),
    // the arena's missing experts of every layer are read ahead into a ring (Stager) while the GPU works, and the
    // layer's MoE runs them from there instead of making them resident (so a prompt leaves the arena as the steps
    // shaped it). end_staging() once the group's GPU work is done. staging_bytes(): the ring's size, also for scratch
    // estimates.
    static constexpr uint32_t kStageMinTokens = 512;
    void begin_staging(uint32_t tokens);
    void end_staging();
    // Two layers' worth of the arena's missing experts, at most MAKO_STAGE_GB (3 GB); before the arena exists, that
    // most.
    uint64_t staging_bytes() const;
    // The ring's limit in GB (0: no staging; < 0: MAKO_STAGE_GB, else 3). Call in the derived constructor before
    // init_runtime().
    void set_staging_limit(double gb);
    // For a prefill: scratch beyond 256 MB takes the memory of lendable arena slots meanwhile; at the end the scratch
    // buffers allocated since go and the slots are read again (unless memory pressure holds them).
    class ScratchLoan {
    public:
        ScratchLoan(EngineBase& e, uint64_t need);
        ~ScratchLoan();
        ScratchLoan(const ScratchLoan&) = delete;
        ScratchLoan& operator=(const ScratchLoan&) = delete;

    private:
        EngineBase& e_;
        bool lend_ = false;
        std::map<std::string, MTL::Buffer*> kept_;
    };
    // A half copy of rows x cols of x (rows ld floats apart, 0: cols) for the GEMM, in one shared scratch buffer.
    MTL::Buffer* halves(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                        uint32_t ld);
    // A compute encoder, timed as `section` of `layer` when per-layer telemetry is on.
    MTL::ComputeCommandEncoder* encoder(MTL::CommandBuffer* cb, uint32_t layer, int section,
                                        MTL::DispatchType type = MTL::DispatchTypeSerial);
    // At the kernels telemetry level, closes the encoder holding the dispatch just encoded, labelled `kernel`,
    // and opens the next one.
    void after(MTL::ComputeCommandEncoder*& enc, std::string_view kernel);
    void resolve_layer_times(uint64_t cpu0, uint64_t gpu0);
    void finish(MTL::CommandBuffer* cb);
    void submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc);
    // Commits what cb holds, so the GPU starts on it while the rest of the step is encoded into a new command
    // buffer (finish() accounts for both). Not while a MoE layer waits for the CPU: finish() serves those.
    void commit_ahead(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc);
    void probe(MTL::ComputeCommandEncoder*& enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
               uint64_t off = 0);
    void deliver_probes();
    void apply_pressure();
    bool timing() const { return !samples_.empty() && tel_ >= Telemetry::layers; }
    // At the layers level with part timing on: the dispatches from here to the next part() (or the end of the
    // section's encoder) are timed as `name`, within the current section.
    void part(MTL::ComputeCommandEncoder*& enc, std::string_view name);
    // MAKO_REPEAT=<label substrings, comma-separated>:<n> encodes matching GEMVs, norms and helper dispatches n times, so the extra
    // step time measures their cost in place (they all recompute their output from their inputs). Repeats read
    // weights the cache may still hold; n = 0 leaves them out instead (the output is then meaningless).
    int repeats(std::string_view label) const
    {
        if (repeat_n_ == 1)
            return 1;
        for (size_t p = 0; p <= repeat_label_.size();) { // comma-separated label substrings
            const size_t e = std::min(repeat_label_.find(',', p), repeat_label_.size());
            if (e > p && label.find(std::string_view(repeat_label_).substr(p, e - p)) != std::string_view::npos)
                return repeat_n_;
            p = e + 1;
        }
        return 1;
    }

    const gguf::ModelFiles& model_;
    mtl::Context& ctx_;
    Dims dims_;
    Routing routing_;
    std::vector<char> layer_mla_; // telemetry: the layer's attention counts as MLA (else KDA)
    std::unique_ptr<mtl::Gemv> gemv_;
    std::unique_ptr<mtl::Gemm> gemm_;
    mtl::Ref<MTL::Library> lib_;
    std::map<std::string, mtl::Ref<MTL::ComputePipelineState>> psos_;

    std::vector<mtl::Ref<MTL::Buffer>> weight_chunks_;
    uint64_t chunk_used_ = 0;
    uint64_t dense_bytes_ = 0;
    std::map<std::string, mtl::Ref<MTL::Buffer>> scratch_;
    std::map<std::string, int> nocache_fds_; // shard path -> descriptor for expert reads
    std::unique_ptr<LockedMemory> simulated_; // before arena_: released after it
    // macOS memory-pressure level (0 normal, 1 warning, 2 critical), set from a dispatch source; applied between
    // steps by apply_pressure(), which lends the arena's lendable slots under pressure and takes them back after.
    std::atomic<int> pressure_{0};
    bool pressure_lent_ = false;
    dispatch_queue_t pressure_queue_ = nullptr;
    dispatch_source_t pressure_src_ = nullptr;
    std::unique_ptr<ExpertArena> arena_;
    ArenaPlan arena_plan_;
    std::string routing_key_;
    std::filesystem::path default_profile_, loaded_profile_;

    // Each MoE layer signals routed_ after its router ran and waits for ready_, which the CPU signals once the
    // selected experts are resident and their slots written (serve_moe).
    mtl::Ref<MTL::SharedEvent> routed_, ready_;
    uint64_t event_value_ = 0;
    struct PendingMoe {
        uint32_t layer;
        uint64_t value; // event value
        uint32_t tokens;
        uint32_t ahead; // nonzero: arena row layer + ahead's likely misses are in scratch "mpred" (read them ahead)
    };
    std::vector<PendingMoe> pending_moe_; // not yet served
    // Layers whose experts are all resident skip that round trip: moe_route counts their selections into
    // resident_counts_ ([arena row][n_expert]), which finish() passes on to the arena.
    mtl::Ref<MTL::Buffer> resident_counts_;
    std::vector<uint32_t> resident_seen_;  // the counts already passed on
    std::vector<uint32_t> resident_calls_; // per arena row, steps since then
    void note_resident();
    std::unique_ptr<Stager> stager_;
    uint64_t stage_bytes_ = 3ull << 30;

    Probe probe_;
    RouteOverride route_override_;
    std::chrono::steady_clock::time_point step_start_;
    LoadStats load_;
    StepStats last_step_;
    Telemetry tel_ = Telemetry::off; // the level in effect (prefill runs kernel-level telemetry as summary)
    // Per-layer telemetry: GPU timestamps at the start and end of timed encoders.
    bool part_timing_ = false;
    void add_samples(uint64_t want); // timestamp sample buffers for at least `want` samples
    struct Timed {
        uint32_t layer;
        int section; // 0 attention, 1 FFN (to routing), 2 resident experts, 3 misses and the rest; -1 one kernel
        NS::UInteger start, end;
        std::string kernel; // kernels level; empty: the encoder held no dispatch
    };
    std::string repeat_label_;
    int repeat_n_ = 1;
    std::string last_pso_; // the kernel a direct dispatch runs, for repeats()
    MTL::CommandBuffer* cur_cb_ = nullptr;
    std::vector<MTL::CommandBuffer*> ahead_; // committed by commit_ahead, waited for in finish()
    mtl::Ref<MTL::Buffer> hc_counters_;      // hc_step_norm: a threadgroup counter per token, zero between dispatches
    mtl::Ref<MTL::Buffer> route_counters_;   // the routing router GEMV: the same
    uint32_t cur_layer_ = 0;
    std::vector<mtl::Ref<MTL::CounterSampleBuffer>> samples_; // kSamplesPerBuffer each; index / that picks one
    std::vector<Timed> timed_;
    NS::UInteger next_sample_ = 0;
    bool encoding_ = false;
    struct PendingProbe {
        std::string name;
        int layer;
        mtl::Ref<MTL::Buffer> copy;
        size_t n;
    };
    std::vector<PendingProbe> pending_probes_;
    Stats stats_;
};

} // namespace mako::core
