#pragma once

#include "core/engine_base.hpp"
#include "gen/model.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"
#include "mtl/ablation.hpp"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mako::dsv41 {

struct Options {
    uint32_t max_ctx = 8192;
    uint32_t indexer_top_k = 0;    // nonzero: override the model's (for testing the sparse selection)
    uint32_t prefill_chunk = 256;  // tokens prefill() runs through each layer's attention together
    uint32_t prefill_group = 8192; // tokens whose routed experts run together, each loaded once per layer
    uint32_t checkpoints = 4;      // checkpoints (checkpoint()) the memory budget reserves room for
    double stage_gb = -1;       // prefill staging ring limit (0: off; < 0: automatic, EngineBase::set_staging_limit)
    double arena_gb = -1;          // expert arena size; < 0: from the memory budget
    std::string expert_counts;     // routing profile; empty: the model's default one when it exists
    core::Telemetry telemetry = core::Telemetry::off;
    double simulate_ram_gb = 0;
    // The DSpark drafter (a directory with dspark.json and dspark.bin); empty: none. Drafts are offered while each
    // one's confidence is at least draft_conf and the product of theirs at least draft_cum, at most max_drafts.
    std::string dspark;
    float draft_conf = 0.5f, draft_cum = 0.5f;
    uint32_t max_drafts = 4;
    // Dense Q8_0 matrices re-quantized at load to this type (Q4_K, Q5_K or Q6_K); unset: as stored.
    std::optional<gguf::Type> dense_quant;
    // Activation statistics (ds_run --imatrix-out) weighting dense_quant's rounding; empty: unweighted.
    std::string dense_imatrix;
    float miss_penalty = 0.0f; // core::Routing::miss_penalty
    // Decode: each layer predicts the top prefetch_k experts of the layer prefetch_ahead later from its FFN input and
    // reads the missing ones ahead (0: off). Results are unchanged.
    uint32_t prefetch_k = 4;
    uint32_t prefetch_ahead = 2;
    // Prefill: from the last KV source layer on, a long prompt's earlier rows only add that layer's compressed rows
    // and index keys; the later layers run only the rows the last prefill_exact_tail positions (at least a window,
    // 0: a window) and the window rings depend on. The logits sink gets only those positions.
    bool prefill_skip = true;
    uint32_t prefill_exact_tail = 0;
};

// DeepSeek-V4.1: dense weights resident, routed experts in the shared expert arena, Engram rows read per token from
// their tables. Decode and prefill run the same layer code over one or many tokens.
class Engine : public core::EngineBase, public gen::Model {
public:
    // `model` holds the Engram tables too (ModelFiles::add).
    Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx);
    ~Engine();

    // Runs one token at the next position and returns its logits.
    const std::vector<float>& step(int32_t token) override;
    // Runs tokens at the next positions, as many steps would, faster; returns the logits after the last.
    // `cancelled` is polled between groups of tokens (Options::prefill_group): true stops there, with the tokens so
    // far in the context (position() tells how many).
    const std::vector<float>& prefill(std::span<const int32_t> tokens,
                                      const std::function<bool()>& cancelled = {}) override;
    // When set, prefill() computes the logits of every position and passes each here (for evaluations).
    using LogitsSink = std::function<void(uint32_t pos, const float* logits)>;
    void set_logits_sink(LogitsSink sink) { logits_sink_ = std::move(sink); }

    uint32_t position() const override { return pos_; }
    const std::vector<float>& logits() const { return logits_; }
    uint32_t max_context() const override { return opts_.max_ctx; }
    uint32_t n_vocab() const override { return cfg_.n_vocab; }
    const Config& config() const { return cfg_; }
    // Back to position 0 (the expert arena keeps its contents).
    void reset() override;
    // state: every layer's window rows, the compressors' pending halves and the Engram history (the compressed rows
    // and index keys are positional).
    Checkpoint checkpoint() override;
    void restore(const Checkpoint& cp) override;
    uint32_t checkpoint_slots() const override { return opts_.checkpoints; }
    uint64_t checkpoint_bytes() const;
    const core::StepStats& last_prefill() const { return last_prefill_; }

    // Abliteration sites: a layer's attention output (attn_out, after the output projection) and its shared
    // expert's output (shexp_out), each before it is added in.
    static constexpr std::array<std::string_view, 2> kAblationSites{"attn_out", "shexp_out"};
    void set_ablation(std::optional<mtl::Ablation> ablation);
    // Options::prefill_skip.
    void set_prefill_skip(bool on) { opts_.prefill_skip = on; }
    // Options::miss_penalty.
    void set_miss_penalty(float penalty)
    {
        opts_.miss_penalty = penalty;
        routing_.miss_penalty = penalty;
    }
    // Options::prefetch_k and prefetch_ahead.
    void set_prefetch(uint32_t k, uint32_t ahead)
    {
        opts_.prefetch_k = k;
        opts_.prefetch_ahead = ahead;
    }

    // DSpark: three small blocks fed the trunk's streams entering its last layers, proposing the tokens after the
    // next one, each with a confidence.
    bool has_drafter() const override { return drafter_ != nullptr; }
    std::vector<int32_t> draft(int32_t next) override;
    const std::vector<float>& verify(std::span<const int32_t> tokens) override;
    void keep(uint32_t n) override;
    // The last draft()'s confidences (as many as it offered, then the ones it cut).
    const std::vector<float>& draft_confidences() const { return draft_conf_; }
    // Totals since load: draft() and verify() calls and their wall times, tokens verified.
    struct SpecStats {
        uint64_t drafts = 0, verifies = 0, verified_tokens = 0, steps = 0, verify_misses = 0, step_misses = 0;
        double draft_ms = 0, verify_ms = 0, step_ms = 0, verify_read_ms = 0, step_read_ms = 0;
        double step_gpu_ms = 0, step_wait_ms = 0, step_encode_ms = 0, step_prep_ms = 0, step_wall_ms = 0;
        // Layer telemetry: GPU time by section (attention; FFN to routing; shared and resident experts; misses and
        // the rest) and by named part.
        double step_attn_ms = 0, step_ffn_ms = 0, step_hit_ms = 0, step_miss_ms = 0;
        std::vector<core::KernelTime> parts;
    };
    const SpecStats& spec_stats() const { return spec_stats_; }

private:
    struct Layer;
    struct Drafter;

    // G tokens at positions pos_..: every layer (attention in chunks of prefill_chunk), then the logits of the last
    // (of all, to the sink, when all_logits).
    // prompt_end: the position after the prefill's last token, for Options::prefill_skip (0: every row).
    void run(std::span<const int32_t> tokens, bool all_logits, uint32_t prompt_end = 0);
    // Attention of T tokens at positions pos0.. (rows of xn into rows of att); g0: their first row in the group's
    // selection rows.
    // `ring`: the window's ring (null: layer il's); `block`: the tokens are a draft block (all see each other and the
    // ring's positions before pos0; their rows do not go into the ring).
    void attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
                   uint32_t T, uint32_t pos0, uint32_t g0, MTL::Buffer* ring = nullptr, bool block = false,
                   bool compress_only = false);
    void load_drafter(const std::string& dir);
    void load_importance(const std::string& path);
    // The drafter's MoE of block b on T rows of xn into ffo (its experts all resident).
    void draft_moe(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t b, MTL::Buffer* xn, MTL::Buffer* ffo,
                   uint32_t T);
    // The drafter's window rows for T positions from pos0 from the trunk's taps (rows of 3 * n_embd).
    void seed_drafter(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* taps, uint64_t taps_off, uint32_t T,
                      uint32_t pos0);
    // T tokens: x = sum_h pre[h] R[h] with the previous sublayer's pre (pc_prev rows); the new mixes of R go to pc
    // rows. R, pc and pc_prev at byte offsets r_off / pc_off (the same rows of each).
    void hc_lagged(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                   MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, MTL::Buffer* pc_prev, uint32_t T,
                   uint64_t r_off, uint64_t pc_off);
    // T rows (stride heads * head_dim) at positions pos0, pos0 + step, ...
    void rope(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, bool compressed, uint32_t head_dim,
              uint32_t heads, uint32_t pos0, uint32_t T, uint32_t step = 1, bool inverse = false);
    void round(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y, uint64_t y_off,
               uint32_t n, uint32_t kind, bool out_half = false);
    void ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* x, uint64_t off, uint32_t rows);
    uint64_t prefill_scratch(uint32_t chunk, uint32_t group) const;

    Config cfg_;
    Options opts_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::optional<Engram> engram_;
    std::unique_ptr<gguf::ReadPool> engram_pool_;
    std::vector<int32_t> engram_history_;

    gguf::ModelFiles::Ref token_embd_;
    DevTensor output_norm_, output_;
    mtl::Ref<MTL::Buffer> freq_window_, freq_comp_; // RoPE frequencies (n_rot / 2) of the two kinds of layer
    std::vector<mtl::Ref<MTL::Buffer>> ring_;       // per layer: the window's kv rows, float [window][head_dim]
    // Per KV source: compressed rows (half, head_dim) and index keys (float, indexer_dim); the ratio-2 sources'
    // pending (k, g) of a group's first token.
    std::vector<mtl::Ref<MTL::Buffer>> ckv_, ik_, comp_state_;
    std::vector<int32_t> kv_sources_;
    uint32_t skip_from_ = 0; // the last KV source layer (Options::prefill_skip)
    uint32_t pos_ = 0;
    std::vector<float> logits_;
    std::vector<float> logits_all_; // verify(): a row per token
    std::unique_ptr<Drafter> drafter_;
    std::vector<float> draft_conf_;
    // The last verify(): its first position and tokens, the Engram history before it; the ratio-2 compressors'
    // state after each of its tokens (comp_hist_, per KV source).
    uint32_t verify_pos0_ = 0;
    std::vector<int32_t> verify_tokens_, verify_history_;
    std::vector<mtl::Ref<MTL::Buffer>> comp_hist_;
    bool verifying_ = false;
    SpecStats spec_stats_;
    LogitsSink logits_sink_;
    core::StepStats last_prefill_;
    std::optional<mtl::Ablation> ablation_;
    int ablate_attn_ = -1, ablate_shexp_ = -1;
};

} // namespace mako::dsv41
