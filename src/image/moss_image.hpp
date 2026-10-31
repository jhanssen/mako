#pragma once

#include "image/engine.hpp"

namespace mako::image {

// MOSS-Image: the last hidden state of a Qwen3.5 language model, refined by two blocks, conditions a single-stream
// transformer over [image, text] whose blocks share one timestep modulation; guided by default against an empty
// negative prompt whose pass skips one block.
class MossImage final : public Engine {
public:
    MossImage(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    const char* name() const override { return "MOSS-Image"; }
    uint32_t default_steps() const override { return 50; }
    float default_guidance() const override { return 3.0f; }
    std::optional<std::string> default_negative() const override { return ""; }
    // The text encoder's final hidden states of the templated prompt, [S][4096].
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // Refines the text rows once; every step then runs the whole [image, text] sequence. Slot 1 (the negative
    // prompt's) skips the block the reference skips there.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;
    // The shifted linear schedule snapped to the 1000 training noise levels (repeats dropped).
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const override;

    // The chat template around a prompt (a user turn, the assistant's opened with thinking).
    static std::string templated(const std::string& prompt);

protected:
    LoraTarget lora_target(const std::string& module) override;

private:
    // q | k | v in one matrix; the FFN's activation input and its gate in another.
    struct Block {
        Tensor q, o, norm_q, norm_k, up_gate, down;
    };
    // The text refiner's blocks: biases throughout, ungated residuals.
    struct TextBlock {
        Tensor norm1, norm2, qkv, qkv_b, norm_q, norm_k, o, o_b, gate_up, gate_up_b, down, down_b;
    };
    void load_transformer(const safetensors::Files& f);
    // Rotary angles for an h x w image followed by S text tokens: (0, S, y, x) and (0, t, 0, 0), image centred.
    void rope_table(MTL::Buffer* cs, uint32_t S, uint32_t h, uint32_t w) const;

    Tensor img_in_, img_in_b_, txt_proj_, txt_proj_b_, t1_, t1_b_, t2_, t2_b_, mod_, mod_b_, head_mod_, head_mod_b_;
    Tensor out_norm_, out_, norms_; // norms_: each block's two norm scales, [blocks][2][D]
    std::vector<Block> blocks_;
    std::vector<TextBlock> text_blocks_;
    float bands_[16] = {};
    static constexpr uint32_t kDim = 3072, kHeads = 24, kText = 4096, kTimeDim = 256, kLatentWidth = 128;
    static constexpr uint32_t kSkip = 22; // the block the negative prompt's pass skips
    static constexpr float kEps = 1e-6f, kShift = 5.0f;
    // The generation begun: the latent grid and each slot's text rows (refined into k_txt<slot>).
    uint32_t gh_ = 0, gw_ = 0, S_[2] = {};
};

} // namespace mako::image
