#pragma once

#include "core/engine_base.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"
#include "mtl/ablation.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace mako::dsv41 {

struct Options {
    uint32_t max_ctx = 8192;
    double arena_gb = -1; // expert arena size; < 0: from the memory budget
    std::string expert_counts; // routing profile; empty: the model's default one when it exists
    core::Telemetry telemetry = core::Telemetry::off;
    double simulate_ram_gb = 0;
};

// DeepSeek-V4.1 decode, one token per step: dense weights resident, routed experts in the shared expert arena,
// Engram rows read per token from their tables.
class Engine : public core::EngineBase {
public:
    // `model` holds the Engram tables too (ModelFiles::add).
    Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx);
    ~Engine();

    // Runs one token at the next position and returns its logits.
    const std::vector<float>& step(int32_t token);
    uint32_t position() const { return pos_; }
    const std::vector<float>& logits() const { return logits_; }
    uint32_t max_context() const { return opts_.max_ctx; }
    const Config& config() const { return cfg_; }
    // Back to position 0 (the expert arena keeps its contents).
    void reset();

    // Abliteration sites: a layer's attention output (attn_out, after the output projection) and its shared
    // expert's output (shexp_out), each before it is added in.
    static constexpr std::array<std::string_view, 2> kAblationSites{"attn_out", "shexp_out"};
    void set_ablation(std::optional<mtl::Ablation> ablation);

private:
    struct Layer;

    void attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att);
    // x = sum_h pre[h] R[h] with the previous sublayer's pre (pc_prev); the new mixes of R go to pc.
    void hc_lagged(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                   MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, MTL::Buffer* pc_prev);
    void rope(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, bool compressed, uint32_t head_dim,
              uint32_t heads, uint32_t pos, bool inverse = false);
    void round(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y, uint64_t y_off,
               uint32_t n, uint32_t kind, bool out_half = false);
    void ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* x);

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
    uint32_t pos_ = 0;
    std::vector<float> logits_;
    std::optional<mtl::Ablation> ablation_;
    int ablate_attn_ = -1, ablate_shexp_ = -1;
};

} // namespace mako::dsv41
