#pragma once

#include "glm/arena.hpp"
#include "glm/config.hpp"
#include "glm/profile.hpp"
#include "glm/telemetry.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "mtl/gemv.hpp"

#include <array>
#include <chrono>
#include <functional>
#include <dispatch/dispatch.h>

#include <atomic>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mako::glm {

struct Options {
    uint32_t max_ctx = 4096;
    uint32_t indexer_top_k = 0; // nonzero: override the model's (for testing the sparse selection)
    double arena_gb = -1;       // expert arena size; < 0: from the memory budget
    uint32_t prefill_chunk = 512; // tokens prefill() runs through each layer's attention together
    uint32_t prefill_group = 4096; // tokens whose routed experts run together, each loaded once per layer
    std::string expert_counts;  // routing profile placing experts in the arena; empty: the model's default profile
                                // (default_profile_path), when it exists
    Telemetry telemetry = Telemetry::off; // layers: per-layer GPU timestamps (extra encoder boundaries)
    bool mtp = false; // load the next-token block (speculative decoding): its weights, caches and expert slots
    // Nonzero: behave as on a Mac with this much RAM (GB): the difference is locked before loading, and the memory
    // budget uses this RAM size and a proportionally smaller GPU working set.
    double simulate_ram_gb = 0;
};

// Decode, one token per step. Dense weights are resident; routed experts live in an expert arena, and the ones
// a token selects but the arena lacks are read from the model files after each MoE layer's router runs.
class Engine {
public:
    Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx);
    ~Engine();

    // Runs one token at the next position and returns its logits.
    const std::vector<float>& step(int32_t token);
    // Two consecutive tokens in one step (speculative verification): their logits, a row each.
    const std::vector<float>& step2(int32_t a, int32_t b);
    // Undoes the second token of the last step2(): its position becomes the next one again.
    void rollback();
    // The next-token block's guess for the token after `next`, where `next` follows the token of the last step():
    // from that step's final hidden state and `next`'s embedding. Also fills the block's own attention cache at the
    // last step's position, so it must run after every step for later drafts to see the whole context. -1 without
    // the block.
    int32_t mtp_draft(int32_t next);
    // The same over the positions of the last step's first next.size() tokens (rows of its hidden state), each
    // followed by next[i]; the guess comes from the last.
    int32_t mtp_draft(std::span<const int32_t> next);
    // Tells the next-token block that the token after position() - 1 is `next`: refills its attention cache there
    // (the last draft may have assumed another token). No-op without the block or at position 0.
    void mtp_seed(int32_t next);
    // The logits behind the last draft (valid until the next one).
    std::span<const float> mtp_logits();
    bool has_mtp() const { return layers_.size() > cfg_.n_layer; }

    // The recurrent state at a position, to continue from it later (prefix reuse). The attention caches are
    // positional and not part of it: restore() is valid while the tokens before cp.pos are the same as when the
    // checkpoint was taken.
    struct Checkpoint {
        uint32_t pos = 0;
        std::vector<std::byte> state; // the KDA layers' scan and convolution states
        std::vector<float> hidden;    // position pos - 1's final hidden state (the next-token block's input)
        uint64_t bytes() const { return state.size() + hidden.size() * 4; }
    };
    Checkpoint checkpoint();
    void restore(const Checkpoint& cp);
    // Runs tokens at the next positions, as many steps would, faster; returns the logits after the last.
    const std::vector<float>& prefill(std::span<const int32_t> tokens);
    // When set, prefill() computes the logits of every position and passes each here (for evaluations).
    using LogitsSink = std::function<void(uint32_t pos, const float* logits)>;
    void set_logits_sink(LogitsSink sink) { logits_sink_ = std::move(sink); }

    uint32_t position() const { return pos_; }
    // The logits after the last step's (or prefill's) last token.
    const std::vector<float>& logits() const { return logits_; }
    uint32_t max_context() const { return opts_.max_ctx; }
    const Config& config() const { return cfg_; }
    // Back to position 0 with empty recurrent state (the expert arena keeps its contents).
    void reset();
    // At most the chunk size the engine was built with, which sized the arena's lendable slots.
    void set_prefill_chunk(uint32_t chunk);
    // At most the group size the engine was built with.
    void set_prefill_group(uint32_t group);

    // Called after each named stage of a step with the fp32 values it produced (for parity checks).
    using Probe = std::function<void(const std::string& name, int layer, const float* data, size_t n)>;
    void set_probe(Probe probe) { probe_ = std::move(probe); }

    // Teacher forcing for parity checks. Called before each layer with its input residual streams
    // (hc * n_embd values), which it may overwrite. Costs a GPU sync per layer.
    using StreamOverride = std::function<void(int layer, float* streams, size_t n)>;
    void set_stream_override(StreamOverride f) { stream_override_ = std::move(f); }
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
        double warm_ms = 0; // first GPU pass over the arena, at load
    };
    const ArenaPlan& arena_plan() const { return arena_plan_; }
    const ExpertArena& arena() const { return *arena_; }
    const LoadStats& load_stats() const { return load_; }
    const StepStats& last_step() const { return last_step_; }
    const StepStats& last_prefill() const { return last_prefill_; } // layers summed over its chunks
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

private:
    struct DevTensor {
        MTL::Buffer* buf = nullptr;
        uint64_t off = 0;
        gguf::Type type = gguf::Type::F32;
        std::vector<uint64_t> dims;
    };
    struct Layer;

    int32_t mtp_run(std::span<const int32_t> next, uint32_t row0);
    DevTensor upload(const std::string& name);
    DevTensor upload_optional(const std::string& name);
    ExpertArena::Part expert(const std::string& name);
    MTL::Buffer* scratch(const std::string& name, uint64_t bytes);
    MTL::ComputePipelineState* pso(const std::string& fn);

    void gemv(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t batch = 1, uint32_t t = 1);
    void proj(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t T, uint32_t ldy = 0);
    void dispatch(MTL::ComputeCommandEncoder*& enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                  const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads = true);
    void norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const DevTensor* w, MTL::Buffer* y, uint32_t rows,
              uint32_t n, uint32_t wn, float eps, uint64_t x_off = 0, uint64_t y_off = 0, uint32_t ystride = 0);
    // R (and out) at byte offset r_off: a chunk's rows of a longer stream.
    void hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T, uint64_t r_off = 0);
    void hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* R, MTL::Buffer* out, MTL::Buffer* pc,
                 uint32_t T, uint64_t r_off = 0);
    // T tokens: hc_post(post_x, R -> R_out, pc) when post_x is set (else R_out is R), then hc_pre of the next block
    // on the new stream (fn, scale, base) into x and pc, in two dispatches.
    void hc_step(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* post_x, MTL::Buffer* R, MTL::Buffer* R_out,
                 const DevTensor& fn, const DevTensor& scale, const DevTensor& base, MTL::Buffer* x, MTL::Buffer* pc,
                 uint32_t T = 1);
    void kda(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
             uint32_t T);
    void mla(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
             uint32_t T, uint32_t pos0);
    void attend(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off, MTL::Buffer* out,
                uint64_t out_off, uint32_t tokens, uint32_t first_kv);
    void indexer_project(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                         MTL::Buffer* qrn, uint32_t T, uint32_t pos0);
    bool indexer_select(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t t, uint32_t pos, uint32_t count = 1);
    void attend_sparse(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off,
                       MTL::Buffer* out, uint64_t out_off, uint32_t tokens);
    // Entries per token of the indexer's cell lists: its selected pools plus an incomplete one.
    uint32_t sparse_list_stride() const { return cfg_.indexer_top_k + cfg_.indexer_kpool; }
    // Query tokens per sparse-attention group and the scratch bytes such a group uses.
    uint32_t sparse_group() const;
    uint64_t sparse_scratch() const;
    void ffn_dense(MTL::ComputeCommandEncoder*& enc, const DevTensor& gate, const DevTensor& up,
                   const DevTensor& down, MTL::Buffer* xn, MTL::Buffer* out, float limit, uint32_t T,
                   uint64_t x_off = 0, uint64_t out_off = 0);
    void serve_moe(MTL::CommandBuffer* cb);
    void decode_tokens(std::span<const int32_t> tokens);
    void mtp_fill(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* h, uint64_t h_off, MTL::Buffer* e, uint64_t e_off,
                  uint32_t T, uint32_t pos0);
    // T (<= kMaxDecodeTokens) tokens' routed FFN plus shared expert, rows of xn into rows of ffo.
    void moe_decode(MTL::CommandBuffer* cb, MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il,
                    MTL::Buffer* xn, MTL::Buffer* ffo, uint32_t T);
    static constexpr uint32_t kMaxDecodeTokens = 4;
    // A half copy of rows x cols of x (rows ld floats apart, 0: cols) for the GEMM, in one shared scratch buffer.
    MTL::Buffer* halves(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                        uint32_t ld);
    uint64_t prefill_scratch(uint32_t chunk, uint32_t group) const;
    // A compute encoder, timed as `section` of `layer` when per-layer telemetry is on.
    MTL::ComputeCommandEncoder* encoder(MTL::CommandBuffer* cb, uint32_t layer, int section,
                                        MTL::DispatchType type = MTL::DispatchTypeSerial);
    // At the kernels telemetry level, closes the encoder holding the dispatch just encoded, labelled `kernel`,
    // and opens the next one.
    void after(MTL::ComputeCommandEncoder*& enc, std::string_view kernel);
    void resolve_layer_times(uint64_t cpu0, uint64_t gpu0);
    void finish(MTL::CommandBuffer* cb);
    void probe(MTL::ComputeCommandEncoder*& enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
               uint64_t off = 0);
    void deliver_probes();
    void submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc);

    const gguf::ModelFiles& model_;
    Config cfg_;
    Options opts_;
    mtl::Context& ctx_;
    std::unique_ptr<mtl::Gemv> gemv_;
    std::unique_ptr<mtl::Gemm> gemm_;
    mtl::Ref<MTL::Library> lib_;
    std::map<std::string, mtl::Ref<MTL::ComputePipelineState>> psos_;

    std::vector<mtl::Ref<MTL::Buffer>> weight_chunks_;
    uint64_t chunk_used_ = 0;
    std::map<std::string, mtl::Ref<MTL::Buffer>> scratch_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::map<std::string, int> nocache_fds_; // shard path -> descriptor for expert reads
    std::unique_ptr<LockedMemory> simulated_; // before arena_: released after it
    // macOS memory-pressure level (0 normal, 1 warning, 2 critical), set from a dispatch source; applied between
    // steps by apply_pressure(), which lends the arena's lendable slots under pressure and takes them back after.
    std::atomic<int> pressure_{0};
    bool pressure_lent_ = false;
    dispatch_queue_t pressure_queue_ = nullptr;
    dispatch_source_t pressure_src_ = nullptr;
    void apply_pressure();
    std::unique_ptr<ExpertArena> arena_;
    ArenaPlan arena_plan_;
    std::string routing_key_;
    std::filesystem::path default_profile_, loaded_profile_;

    gguf::ModelFiles::Ref token_embd_;
    DevTensor output_norm_, output_;
    std::vector<mtl::Ref<MTL::Buffer>> kda_state_, conv_state_, kv_cache_, idx_cache_, idx_pooled_;
    // A speculative step's KDA and conv states after its first snap_ tokens, swapped in by rollback().
    std::vector<mtl::Ref<MTL::Buffer>> kda_state_alt_, conv_state_alt_;
    uint32_t snap_ = 0;
    uint32_t last_tokens_ = 0, last_pos0_ = 0; // the last decode step's token count and first position
    std::vector<float> logits_all_;            // its logits, a row per token

    uint32_t pos_ = 0;
    uint32_t built_chunk_ = 0, built_group_ = 0;
    std::vector<float> logits_;
    // Each MoE layer signals routed_ after its router ran and waits for ready_, which the CPU signals once the
    // selected experts are resident and their slots written (serve_moe).
    mtl::Ref<MTL::SharedEvent> routed_, ready_;
    uint64_t event_value_ = 0;
    struct PendingMoe {
        uint32_t layer;
        uint64_t value; // event value
        uint32_t tokens;
    };
    std::vector<PendingMoe> pending_moe_; // not yet served

    Probe probe_;
    LogitsSink logits_sink_;
    StreamOverride stream_override_;
    std::chrono::steady_clock::time_point step_start_;
    LoadStats load_;
    StepStats last_step_, last_prefill_;
    // Per-layer telemetry: GPU timestamps at the start and end of timed encoders.
    struct Timed {
        uint32_t layer;
        int section; // 0 attention, 1 FFN (to routing), 2 resident experts, 3 misses and the rest; -1 one kernel
        NS::UInteger start, end;
        std::string kernel; // kernels level; empty: the encoder held no dispatch
    };
    // MAKO_REPEAT=<label substring>:<n> encodes matching GEMVs, norms and helper dispatches n times, so the extra
    // step time measures their cost in place (they all recompute their output from their inputs).
    std::string repeat_label_;
    int repeat_n_ = 1;
    std::string last_pso_; // the kernel a direct dispatch runs, for repeats()
    int repeats(std::string_view label) const
    {
        return repeat_n_ > 1 && label.find(repeat_label_) != std::string_view::npos ? repeat_n_ : 1;
    }
    MTL::CommandBuffer* cur_cb_ = nullptr;
    uint32_t cur_layer_ = 0;
    std::vector<mtl::Ref<MTL::CounterSampleBuffer>> samples_; // kSamplesPerBuffer each; index / that picks one
    std::vector<Timed> timed_;
    NS::UInteger next_sample_ = 0;
    uint64_t dense_bytes_ = 0;
    bool encoding_ = false;
    RouteOverride route_override_;
    struct PendingProbe {
        std::string name;
        int layer;
        mtl::Ref<MTL::Buffer> copy;
        size_t n;
    };
    std::vector<PendingProbe> pending_probes_;
    Stats stats_;
};

} // namespace mako::glm
