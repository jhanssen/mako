#pragma once

#include "core/engine_base.hpp"
#include "gen/model.hpp"
#include "glm/config.hpp"
#include "mtl/ablation.hpp"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mako::glm {

using core::Counts;
using core::ExpertArena;
using core::LayerTimes;
using core::LoadStats;
using core::LockedMemory;
using core::MemInfo;
using core::StepStats;
using core::Telemetry;

struct Options {
    uint32_t max_ctx = 4096;
    uint32_t indexer_top_k = 0; // nonzero: override the model's (for testing the sparse selection)
    double stage_gb = -1;       // prefill staging ring limit (0: off; < 0: automatic, EngineBase::set_staging_limit)
    double arena_gb = -1;       // expert arena size; < 0: from the memory budget
    double park_gb = -1;        // parked conversations (gen::SessionPool); < 0: what full expert residency leaves, 1-8 GB
    uint32_t prefill_chunk = 512; // tokens prefill() runs through each layer's attention together
    uint32_t prefill_group = 4096; // tokens whose routed experts run together, each loaded once per layer
    std::string expert_counts;  // routing profile placing experts in the arena; empty: the model's default profile
                                // (default_profile_path), when it exists
    Telemetry telemetry = Telemetry::off; // layers: per-layer GPU timestamps (extra encoder boundaries)
    bool mtp = false; // load the next-token block (speculative decoding): its weights, caches and expert slots
    uint32_t checkpoints = 4; // checkpoints (checkpoint()) the memory budget reserves room for
    // Dense Q6_K and Q8_0 matrices re-quantized to this type at load (fewer bytes a token; measured in
    // docs/performance.md). Env MAKO_DENSE_QUANT=q4_k|q5_k|q6_k when unset.
    std::optional<gguf::Type> dense_quant;
    // Nonzero: behave as on a Mac with this much RAM (GB): the difference is locked before loading, and the memory
    // budget uses this RAM size and a proportionally smaller GPU working set.
    double simulate_ram_gb = 0;
};

// Decode, one token per step. Dense weights are resident; routed experts live in an expert arena, and the ones
// a token selects but the arena lacks are read from the model files after each MoE layer's router runs.
class Engine : public core::EngineBase, public gen::Model {
public:
    Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx);
    ~Engine();

    // Runs one token at the next position and returns its logits.
    const std::vector<float>& step(int32_t token) override;
    // Two consecutive tokens in one step (speculative verification): their logits, a row each.
    const std::vector<float>& step2(int32_t a, int32_t b) override;
    // Undoes the second token of the last step2(): its position becomes the next one again.
    void rollback() override;
    // The next-token block's guess for the token after `next`, where `next` follows the token of the last step():
    // from that step's final hidden state and `next`'s embedding. Also fills the block's own attention cache at the
    // last step's position, so it must run after every step for later drafts to see the whole context. -1 without
    // the block.
    int32_t mtp_draft(int32_t next) override;
    // The same over the positions of the last step's first next.size() tokens (rows of its hidden state), each
    // followed by next[i]; the guess comes from the last.
    int32_t mtp_draft(std::span<const int32_t> next) override;
    // Tells the next-token block that the token after position() - 1 is `next`: refills its attention cache there
    // (the last draft may have assumed another token). No-op without the block or at position 0.
    void mtp_seed(int32_t next) override;
    // The logits behind the last draft (valid until the next one).
    std::span<const float> mtp_logits();
    bool has_mtp() const override { return layers_.size() > cfg_.n_layer; }

    // The recurrent state at a position, to continue from it later (prefix reuse). The attention caches are
    // positional and not part of it: restore() is valid while the tokens before cp.pos are the same as when the
    // checkpoint was taken.
    // state: the KDA layers' scan and convolution states; hidden: position pos - 1's final hidden state (the
    // next-token block's input).
    Checkpoint checkpoint() override;
    void restore(const Checkpoint& cp) override;
    // The MLA layers' latent rows, indexer keys and complete pools before pos (the next-token block's included).
    std::vector<std::byte> save_positions(uint32_t pos) const override;
    void load_positions(uint32_t pos, std::span<const std::byte> data) override;
    // Runs tokens at the next positions, as many steps would, faster; returns the logits after the last.
    // `cancelled` is polled between groups of tokens (Options::prefill_group): true stops there, with the tokens so
    // far in the context (position() tells how many) and the logits after the last of them.
    void set_input_rows(std::span<const gen::InputRows> rows) override { input_rows_.assign(rows.begin(), rows.end()); }
    const std::vector<float>& prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled = {},
                                      const std::function<void(size_t)>& progress = {}) override;
    // When set, prefill() computes the logits of every position and passes each here (for evaluations).
    using LogitsSink = std::function<void(uint32_t pos, const float* logits)>;
    void set_logits_sink(LogitsSink sink) { logits_sink_ = std::move(sink); }

    uint32_t position() const override { return pos_; }
    // The logits after the last step's (or prefill's) last token.
    const std::vector<float>& logits() const { return logits_; }
    uint32_t max_context() const override { return opts_.max_ctx; }
    uint32_t n_vocab() const override { return cfg_.n_vocab; }
    // Abliteration sites: before layer l's FFN output is mixed into the residual streams, it (ffn_out) and the
    // streams (stream) lose their components along the layer's directions, in prefill and decode; the next-token
    // block is left alone. A file may hold either or both.
    static constexpr std::array<std::string_view, 2> kAblationSites{"ffn_out", "stream"};
    void set_ablation(std::optional<mtl::Ablation> ablation);
    bool ablating() const { return ablation_.has_value(); }
    uint32_t checkpoint_slots() const override { return opts_.checkpoints; }
    // Size of one checkpoint().
    uint64_t checkpoint_bytes() const;
    const Config& config() const { return cfg_; }
    // Back to position 0 with empty recurrent state (the expert arena keeps its contents).
    void reset() override;
    // At most the chunk size the engine was built with, which sized the arena's lendable slots.
    void set_prefill_chunk(uint32_t chunk);
    // At most the group size the engine was built with.
    void set_prefill_group(uint32_t group);

    // Teacher forcing for parity checks. Called before each layer with its input residual streams
    // (hc * n_embd values), which it may overwrite. Costs a GPU sync per layer.
    using StreamOverride = std::function<void(int layer, float* streams, size_t n)>;
    void set_stream_override(StreamOverride f) { stream_override_ = std::move(f); }

    const StepStats& last_prefill() const { return last_prefill_; } // layers summed over its chunks

private:
    struct Layer;

    int32_t mtp_run(std::span<const int32_t> next, uint32_t row0);
    void kda(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
             uint32_t T);
    void mla(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
             uint32_t T, uint32_t pos0);
    void attend(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off, MTL::Buffer* out,
                uint64_t out_off, uint32_t tokens, uint32_t first_kv);
    // With projected, the query, key, gate and weight projections are already in iq, ikr, the cache and iw.
    void indexer_project(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                         MTL::Buffer* qrn, uint32_t T, uint32_t pos0, bool projected = false);
    bool indexer_select(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t t, uint32_t pos, uint32_t count = 1);
    void attend_sparse(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off,
                       MTL::Buffer* out, uint64_t out_off, uint32_t tokens);
    // Entries per token of the indexer's cell lists: its selected pools plus an incomplete one.
    uint32_t sparse_list_stride() const { return cfg_.indexer_top_k + cfg_.indexer_kpool; }
    // Query tokens per sparse-attention group and the scratch bytes such a group uses.
    uint32_t sparse_group() const;
    uint64_t sparse_scratch() const;
    void decode_tokens(std::span<const int32_t> tokens);
    void mtp_fill(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* h, uint64_t h_off, MTL::Buffer* e, uint64_t e_off,
                  uint32_t T, uint32_t pos0);
    uint64_t prefill_scratch(uint32_t chunk, uint32_t group) const;
    // The positional caches and the bytes of each that hold positions before pos.
    std::vector<std::pair<MTL::Buffer*, uint64_t>> positional(uint32_t pos) const;

    Config cfg_;
    Options opts_;
    std::vector<std::unique_ptr<Layer>> layers_;

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

    LogitsSink logits_sink_;
    StreamOverride stream_override_;
    StepStats last_prefill_;
    std::optional<mtl::Ablation> ablation_;
    int ablate_ffn_ = -1, ablate_stream_ = -1; // the sites' indices in ablation_
    void ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* x, uint32_t rows);
};

} // namespace mako::glm
