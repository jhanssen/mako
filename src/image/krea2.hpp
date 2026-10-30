#pragma once

#include "image/engine.hpp"

namespace mako::image {

// Krea 2: twelve of the text encoder's hidden states per token, fused by a small transformer, condition a
// single-stream transformer over [text, image] with grouped-query, gated attention.
class Krea2 final : public Engine {
public:
    Krea2(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    const char* name() const override { return "Krea 2"; }
    uint32_t default_steps() const override { return distilled_ ? 8 : 28; }
    // The base model: the reference's scale (its guidance_scale 4.5, counted from 0 where this counts from 1). The
    // distilled one runs without guidance; with a negative prompt, 2 was the last clean scale tried (3: flat and
    // oversaturated, 5.5: broken).
    float default_guidance() const override { return distilled_ ? 2.0f : 5.5f; }
    // The text encoder's tapped hidden states of the prompt's tokens, [S][12][2560].
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // Fuses the tapped states into the text rows once; every step then runs the whole [text, image] sequence.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;

    // The prompt template around `prompt`, and the token count of its system part.
    static std::string templated(const std::string& prompt);
    uint32_t system_tokens() const { return drop_; }
    static constexpr uint32_t kTaps[12] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};

protected:
    LoraTarget lora_target(const std::string& module) override;

private:
    // Attention (q, k, v and the output gate fused) and a SwiGLU MLP (gate, up fused), RMS norms with their scales + 1.
    struct Layer {
        Tensor norm1, norm2, qkvg, o, norm_q, norm_k, gate_up, down;
    };
    // A fusion block: pre-norm, ungated residuals. groups: attention within each run of `groups` rows (0: all rows).
    void fusion_block(MTL::ComputeCommandEncoder*& enc, const Layer& L, MTL::Buffer* x, uint32_t rows, uint32_t groups);
    void load_transformer(const safetensors::Files& f);
    std::string key(const std::string& original) const;

    Tensor img_in_, img_in_b_, t1_, t1_b_, t2_, t2_b_, tproj_, tproj_b_, txt_norm_, txt1_, txt1_b_, txt2_, txt2_b_;
    Tensor out_norm_, out_table_, out_, out_b_;
    std::vector<Layer> fusion_, blocks_;
    // Each block's six modulation rows ([blocks][6][D]) and its two norm scales + 1 ([blocks][2][D]).
    Tensor tables_, norms_;
    float projector_[12] = {};
    bool diffusers_names_ = false, distilled_ = true;
    std::string prefix_; // of the original names ("model.diffusion_model." in single-file checkpoints)
    static constexpr uint32_t kDim = 6144, kHeads = 48, kKvHeads = 12, kText = 2560, kTextHeads = 20, kTimeDim = 256;
    static constexpr float kEps = 1e-5f;
    static constexpr const char* kSystem =
        "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the "
        "objects and background:";
    static constexpr uint32_t kMaxPrompt = 507; // the reference's 512 text rows less the template's last 5
    uint32_t drop_ = 0;
    // The generation begun: the latent grid and each slot's text rows (their fused text and rotary table in
    // k_txt<slot>, k_cs<slot>).
    uint32_t gh_ = 0, gw_ = 0, S_[2] = {};
};

} // namespace mako::image
