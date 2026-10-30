#pragma once

#include "core/arena.hpp"
#include "core/profile.hpp"
#include "core/stager.hpp"
#include "core/telemetry.hpp"
#include "gen/model.hpp"
#include "gguf/model_files.hpp"
#include "gpu/runtime.hpp"

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

// What the routed-expert LLMs share on top of the GPU runtime: weights from the model's GGUF files, the
// routed-expert arena with decode's GPU routing served by the CPU (misses read between a layer's router and its
// experts), staged prefill, hyper-connections, input rows. A model's engine derives from it and adds its layer graph.
class EngineBase : public gpu::Runtime {
public:
    using DevTensor = gpu::Runtime::Tensor;
    // One MoE layer's dense parts and SwiGLU clamps (<= 0: none).
    struct MoeWeights {
        DevTensor router, bias, sh_gate, sh_up, sh_down;
        DevTensor bias_vl; // routes the rows taking input rows (an image) when present
        float clamp_exp = 0, clamp_sh = 0;
    };

    // Called after routing a MoE layer; may overwrite the selected experts and their final weights (k each).
    using RouteOverride = std::function<void(int layer, int* sel, float* w, size_t k)>;
    void set_route_override(RouteOverride f) { route_override_ = std::move(f); }

    // The memory figures the arena was sized from and its size (bytes).
    struct ArenaPlan {
        MemInfo mem;
        double budget = 0;
        double park = 0;     // bytes for parked conversations (gen::SessionPool), taken from the budget
        double lendable = 0; // arena bytes prefill's scratch buffers borrow
        double warm_ms = 0;  // first GPU pass over the arena, at load
    };
    const ArenaPlan& arena_plan() const { return arena_plan_; }
    const ExpertArena& arena() const { return *arena_; }
    const LoadStats& load_stats() const { return load_; }
    const StepStats& last_step() const { return last_step_; }
    // Prefill at the layers level: also time named parts of each layer (StepStats::parts); adds encoders.
    void set_part_timing(bool on);
    const std::string& routing_key() const { return routing_key_; }
    const std::filesystem::path& default_profile() const { return default_profile_; }
    const std::filesystem::path& loaded_profile() const { return loaded_profile_; } // empty: none
    // Writes the routing counts of this engine's steps to `path`, added to the file's counts when `merge`.
    void save_profile(const std::filesystem::path& path, bool merge) const;

protected:
    static constexpr uint32_t kMaxDecodeTokens = 8;

    EngineBase(const gguf::ModelFiles& model, mtl::Context& ctx, Telemetry telemetry);
    ~EngineBase() override;
    EngineBase(const EngineBase&) = delete;
    EngineBase& operator=(const EngineBase&) = delete;

    // Reads a dense tensor into the resident weight buffers (BF16 becomes F32); Q8_0 and Q6_K matrices are
    // re-quantized to dense_quant_ when it is set.
    DevTensor upload(const std::string& name);
    DevTensor upload_requant(int fd, const gguf::Tensor& t, gguf::Type to, const std::string& name);
    std::optional<gguf::Type> dense_quant_;
    // Per tensor name: the importance of each input column (k values, or k per group of rows for a tensor whose row
    // groups read different inputs), weighting the re-quantization's rounding.
    std::map<std::string, std::vector<float>> importance_;
    uint64_t requant_bytes_ = 0; // bytes the re-quantization saved
    // A routed experts tensor [k, n, n_expert] as the arena reads it; `prefill`: it also needs the prefill GEMM.
    ExpertArena::Part expert(const std::string& name, bool prefill);

    struct ArenaInit {
        std::vector<ExpertArena::LayerParts> parts; // per arena row (layer)
        std::vector<uint32_t> moe_layers;           // for the routing key
        uint64_t context_bytes = 0;                 // caches and checkpoints, taken from the budget
        double lendable = 0;
        double arena_gb = -1;      // < 0: from the memory budget
        double park_gb = -1;       // parked conversations; < 0: what full expert residency leaves, 1-8 GB
        double simulate_ram_gb = 0;
        std::string expert_counts; // profile; empty: the default one when it exists
        uint32_t draft_rows = 0;   // arena rows after the trunk (n_layer) that a profile may lack
    };
    // Sizes and fills the expert arena, then sets up telemetry and the memory-pressure source. Call last in the
    // derived constructor, after every upload() and with load_.dense_ms set.
    void init_runtime(ArenaInit init);

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
    // G tokens' routed experts of arena row il (rows of xn_g, the first at position pos0) added into ffo, which holds
    // their shared expert's output. Command buffers it commits go to `committed`: they must complete before the row's
    // slots are used again.
    void moe_prefill(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc, const MoeWeights& w, uint32_t il,
                     MTL::Buffer* xn_g, MTL::Buffer* ffo, uint32_t G, uint32_t pos0,
                     std::vector<MTL::CommandBuffer*>& committed);
    static constexpr uint32_t kBatchRows = 8192; // prefill expert rows (token, expert pairs) run and combined together
    // Staged prefill: for a group of at least kStageMinTokens, the arena's missing experts of every layer are read
    // ahead into a ring (Stager) while the GPU works, and the layer's MoE runs them from there instead of making them
    // resident (so a prompt leaves the arena as the steps shaped it). end_staging() once the group's GPU work is done.
    // staging_bytes(): the ring's size, also for scratch estimates. Smaller groups read only the experts they route
    // to, which then stay resident: a DeepSeek group of 512-1024 tokens routes to well under all the missing ones
    // (ISTA IQ2_XS at 512: ~2000 of ~5400), and the steps after it use them.
    static constexpr uint32_t kStageMinTokens = 2048;
    // Groups under this many tokens route with the decode steps' miss penalty.
    static constexpr uint32_t kPenaltyMaxTokens = 512;
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
    // The step's encoder timestamps into last_step_: per-layer section times (sections: 0 attention, 1 FFN to
    // routing, 2 resident experts, 3 misses and the rest), parts and kernels.
    void resolve_layer_times(uint64_t cpu0, uint64_t gpu0);
    // finish() serves the MoE layers waiting for the CPU, then passes on the resident layers' counts; no commit
    // ahead while a MoE layer waits.
    void on_committed(MTL::CommandBuffer* cb) override { serve_moe(cb); }
    void on_completed() override { note_resident(); }
    bool commit_ahead_ok() const override { return pending_moe_.empty(); }
    // dst[i] = the embedding of tokens[i] at sequence position pos0 + i (E floats each): its input row when one
    // stands there, else its row of token_embd.
    void embed(const gguf::ModelFiles::Ref& token_embd, std::span<const int32_t> tokens, uint32_t pos0, float* dst) const;
    // Whether an input row stands at position pos.
    bool input_at(uint32_t pos) const;
    void apply_pressure();

    const gguf::ModelFiles& model_;
    Dims dims_;
    Routing routing_;
    std::vector<char> layer_mla_; // telemetry: the layer's attention counts as MLA (else KDA)
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
    std::vector<gen::InputRows> input_rows_; // gen::Model::set_input_rows
    // Layers whose experts are all resident skip that round trip: moe_route counts their selections into
    // resident_counts_ ([arena row][n_expert]), which finish() passes on to the arena.
    mtl::Ref<MTL::Buffer> resident_counts_;
    std::vector<uint32_t> resident_seen_;  // the counts already passed on
    std::vector<uint32_t> resident_calls_; // per arena row, steps since then
    void note_resident();
    std::unique_ptr<Stager> stager_;
    uint64_t stage_bytes_ = 3ull << 30;

    RouteOverride route_override_;
    LoadStats load_;
    StepStats last_step_;
    mtl::Ref<MTL::Buffer> hc_counters_;    // hc_step_norm: a threadgroup counter per token, zero between dispatches
    mtl::Ref<MTL::Buffer> route_counters_; // the routing router GEMV: the same
};

} // namespace mako::core
