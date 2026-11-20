#pragma once

#include "core/engine_base.hpp"
#include "dsv41/engram.hpp"
#include "gen/model.hpp"
#include "hy4f/config.hpp"
#include "mtl/ablation.hpp"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mako::hy4f {

struct Options {
    uint32_t max_ctx = 8192;
    uint32_t prefill_chunk = 512;  // tokens prefill() runs through each layer's attention together
    uint32_t prefill_group = 4096; // tokens whose routed experts run together, each loaded once per layer
    uint32_t checkpoints = 4;      // checkpoints (checkpoint()) the memory budget reserves room for
    double stage_gb = -1;          // prefill staging ring limit (0: off; < 0: automatic)
    double arena_gb = -1;          // expert arena size; < 0: from the memory budget
    double park_gb = -1;           // parked conversations; < 0: what full expert residency leaves, 1-8 GB
    std::string expert_counts;     // routing profile; empty: the model's default one when it exists
    core::Telemetry telemetry = core::Telemetry::off;
    double simulate_ram_gb = 0;
    bool spin_wait = true;
    bool page_cache = true;
    bool keepalive = false;
    std::optional<gguf::Type> dense_quant;
    float miss_penalty = 0.0f; // core::Routing::miss_penalty
    uint32_t prefetch_k = 4;   // decode: experts of the layer prefetch_ahead later predicted and read ahead (0: off)
    uint32_t prefetch_ahead = 2;
};

// Hy4-flash: dense weights resident, routed experts (at the latent width) in the shared expert arena, Engram rows
// read per token from their table. Decode and prefill run the same layer code over one or many tokens.
class Engine : public core::EngineBase, public gen::Model {
public:
    // `model` holds the Engram table too (its second shard).
    Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx);
    ~Engine();

    // Image spans with cols set take 2D positions (their rows and columns); an image of r rows of c tokens advances
    // the sequence position by max(r, c).
    void set_input_rows(std::span<const gen::InputRows> rows) override;
    // The RoPE positions (sequence, row, column, 0) of positions lo..end-1 into out, given the images among `rows`
    // and `delta` (the sequence position of pos minus pos) at a position pos >= lo; returns the delta at end.
    static int32_t rope_positions(std::span<const gen::InputRows> rows, int32_t delta, uint32_t pos, uint32_t lo,
                                  uint32_t end, int32_t* out);
    const std::vector<float>& step(int32_t token) override;
    const std::vector<float>& prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled = {},
                                      const std::function<void(size_t)>& progress = {}) override;
    // When set, prefill() computes the logits of every position and passes each here (for evaluations).
    using LogitsSink = std::function<void(uint32_t pos, const float* logits)>;
    void set_logits_sink(LogitsSink sink) { logits_sink_ = std::move(sink); }

    uint32_t position() const override { return pos_; }
    const std::vector<float>& logits() const { return logits_; }
    uint32_t max_context() const override { return opts_.max_ctx; }
    uint32_t n_vocab() const override { return cfg_.n_vocab; }
    const Config& config() const { return cfg_; }
    void reset() override;
    // state: the Engram history, each window layer's last sliding_window rows and each indexer's rows of the group
    // the position is in (what the sliding buffers hold that the positional caches do not).
    Checkpoint checkpoint() override;
    void restore(const Checkpoint& cp) override;
    uint32_t checkpoint_slots() const override { return opts_.checkpoints; }
    uint64_t checkpoint_bytes() const;
    // The indexer layers' latent and rope rows and pooled keys before pos.
    std::vector<std::byte> save_positions(uint32_t pos) const override;
    void load_positions(uint32_t pos, std::span<const std::byte> data) override;
    const core::StepStats& last_prefill() const { return last_prefill_; }
    // At most the chunk the engine was built with (which sized its sliding buffers and scratch). Results do not depend
    // on it.
    void set_prefill_chunk(uint32_t chunk);
    // Ablation sites: stream0..3, each hyper-connection stream entering a layer (after its Engram, before the
    // attention's mix), and head0..3, the streams before the output head (at the last layer's index). The four of
    // a kind come in order in the file.
    static constexpr std::array<std::string_view, 8> kAblationSites{"stream0", "stream1", "stream2", "stream3",
                                                                    "head0",   "head1",   "head2",   "head3"};
    void set_ablation(std::optional<mtl::Ablation> ablation);
    void set_miss_penalty(float penalty)
    {
        opts_.miss_penalty = penalty;
        routing_.miss_penalty = penalty;
    }

private:
    struct Layer;

    // G tokens at positions pos_..: every layer (attention in chunks of prefill_chunk), then the logits of the last
    // (of all, to the sink, when all_logits).
    void run(std::span<const int32_t> tokens, bool all_logits);
    // Attention of T tokens at positions pos0.. (rows of xn into rows of att).
    void attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
                   uint32_t T, uint32_t pos0);
    // The indexer's rows and new pools for T tokens at pos0 (always), and its query and weights (with qrn).
    // projected: the key, query and weight projections are already in their scratch buffers.
    void indexer_project(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                         MTL::Buffer* qrn, uint32_t T, uint32_t pos0, bool projected = false);
    // Cell lists of tokens t..t+count-1 (positions pos..) of the last indexer_project: their top pools and own group.
    void indexer_select(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t t, uint32_t pos, uint32_t count);
    // x = the pre-mix of the streams R (T rows from byte offset r_off) and pc their pre / post weights, then
    // xn = rms_norm(x) * norm_w.
    void hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                MTL::Buffer* R, uint64_t r_off, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T, const DevTensor& norm_w,
                MTL::Buffer* xn);
    // R rows (from r_off) += post * out, after out = rms_norm(out) * post_norm.
    // `probe`: the normed output's probe name (one token).
    void hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* out, const DevTensor& post_norm, MTL::Buffer* R,
                 uint64_t r_off, MTL::Buffer* pc, uint32_t T, const char* probe_name = nullptr, int layer = 0);
    // T rows (stride heads * head_dim) roped on their last n_rot values, at the RoPE positions of positions pos0,
    // pos0 + step, ... (from pos3_lo_ on, set by run()).
    void rope(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* freqs, uint32_t head_dim,
              uint32_t heads, uint32_t pos0, uint32_t T, uint32_t step = 1);
    // The RoPE positions of positions lo..end-1 into the scratch "hy_pos3"; returns rope_delta_ at end.
    int32_t rope_positions(uint32_t lo, uint32_t end);
    uint64_t prefill_scratch(uint32_t chunk, uint32_t group) const;
    // Before layer il's rows of T tokens at pos0 go into its sliding buffers (a window layer's latent and rope rows,
    // an indexer's value | gate rows): when they would not fit, the rows still needed move to the front.
    void slide(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t pos0, uint32_t T);
    void move_rows(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* b, uint64_t row_bytes, uint32_t from, uint32_t n);
    std::vector<std::pair<MTL::Buffer*, uint64_t>> positional(uint32_t pos) const;
    uint32_t list_stride() const { return n_top_ * cfg_.indexer_kpool + cfg_.indexer_kpool; }

    Config cfg_;
    Options opts_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::optional<dsv41::Engram> engram_;
    std::unique_ptr<gguf::ReadPool> engram_pool_;
    std::vector<int32_t> engram_history_;
    uint32_t n_top_ = 0; // pools a selecting token keeps (its own group comes on top)
    std::optional<mtl::Ablation> ablation_;
    int ablate_stream_ = -1, ablate_head_ = -1; // the first of the four sites' indices in ablation_
    // The streams of `rows` tokens of R (from byte offset off) lose their directions at `site`'s four sites.
    void ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* R, uint64_t off, uint32_t rows);

    gguf::ModelFiles::Ref token_embd_;
    DevTensor output_norm_, output_, head_fn_, head_base_, head_scale_;
    mtl::Ref<MTL::Buffer> freq_full_, freq_swa_; // RoPE frequencies of the two kinds of layer
    // Per layer: latent rows (half, kv_lora_rank) and rope rows (half, n_rot), of every position for an indexer
    // layer, of the last swa_cap_ from row_base_ on for a window layer; per indexer layer its value | gate rows
    // (float, 2 x dim; idx_cap_ from idx_base_ on) and pooled keys (float, dim, every complete group).
    std::vector<mtl::Ref<MTL::Buffer>> lat_, pe_, idx_rows_, idx_pooled_;
    std::vector<uint32_t> row_base_, idx_base_;
    uint32_t swa_cap_ = 0, idx_cap_ = 0, built_chunk_ = 0;
    uint32_t pos_ = 0;
    // The sequence position of pos_ minus pos_ (images take fewer positions than tokens); in the checkpoints.
    int32_t rope_delta_ = 0;
    uint32_t pos3_lo_ = 0;
    std::vector<float> logits_;
    LogitsSink logits_sink_;
    std::function<void(uint32_t)> layer_done_;
    core::StepStats last_prefill_;
};

} // namespace mako::hy4f
