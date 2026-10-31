#pragma once

#include "image/engine.hpp"
#include "text/unigram.hpp"

#include <utility>

namespace mako::image {

// Anima (Cosmos-Predict2's transformer): Qwen3-0.6B's final hidden states, read by an adapter whose queries are the
// prompt's T5 tokens, condition a transformer of self-attention over the image, cross-attention to that text and an
// MLP, each modulated by the timestep (AdaLN with a low-rank shared term). Its files as published: diffusion_models/
// (one transformer), text_encoders/, vae/ (the first Qwen-Image VAE in its original names), plus tokenizers/qwen3 and
// tokenizers/t5 (tokenizer.json each).
class Anima final : public Engine {
public:
    Anima(const std::string& dir, mtl::Context& ctx, const Options& opts = {});
    // Whether dir (or its split_files/) holds Anima's files.
    static bool detect(const std::string& dir);

    const char* name() const override { return turbo_ ? "Anima Turbo" : "Anima"; }
    // The model card's: Turbo 8-12 steps without guidance, the others 30-50 at 4-5.
    uint32_t default_steps() const override { return turbo_ ? 10 : 30; }
    float default_guidance() const override { return turbo_ ? 1.0f : 4.0f; }
    std::optional<std::string> default_negative() const override;
    // The prompt's Qwen3 states and T5 tokens with their weights, packed: S, M, [S][1024], M ids, M weights.
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // Runs the adapter into the text rows (at least 512, zero padded) and their keys and values for every block.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;
    // Shift 3 over 1000 levels, `steps` of them evenly spaced from the top, then 0.
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const override;

    // A prompt's parts and their weights: "(text)" times 1.1, "(text:w)" w, nested ones multiplied, "\(" and "\)"
    // literal parentheses.
    static std::vector<std::pair<std::string, float>> weighted(const std::string& prompt);
    // The two token sequences of a prompt: Qwen3's (the pad token when empty) and T5's with the end token, its weights.
    std::vector<int32_t> qwen_ids(const std::string& prompt) const;
    std::pair<std::vector<int32_t>, std::vector<float>> t5_ids(const std::string& prompt) const;

protected:
    LoraTarget lora_target(const std::string& module) override;

private:
    struct Attn {
        Tensor q, k, v, o, norm_q, norm_k; // q, k, v fused into q where they read the same input
    };
    struct Block {
        Attn self, cross;
        Tensor mlp1, mlp2;
    };
    struct AdapterBlock {
        Tensor norm_self, norm_cross, norm_mlp, mlp1, mlp1_b, mlp2, mlp2_b;
        Attn self, cross;
    };
    void load_transformer(const safetensors::Files& f);

    Tensor x_in_, t1_, t2_, t_norm_, out_, embed_, a_out_, a_out_b_, a_norm_;
    Tensor mod1_, mod2_; // each block's three modulations' down and up projections, then the final layer's
    std::vector<Block> blocks_;
    std::vector<AdapterBlock> adapter_;
    safetensors::Files tfiles_;
    safetensors::Files::Ref embed_ref_; // the adapter's token embeddings, read per prompt
    text::Unigram t5_;
    bool turbo_ = false;
    static constexpr uint32_t kDim = 2048, kHeads = 16, kCtx = 1024, kAdapterHeads = 16, kAdapterHead = 64;
    static constexpr uint32_t kLora = 256, kMinRows = 512;
    static constexpr int32_t kQwenPad = 151643;
    static constexpr float kEps = 1e-6f;
    // The generation begun: the latent grid and each slot's text rows, kept as each block's keys and values.
    uint32_t gh_ = 0, gw_ = 0, rows_[2] = {};
};

} // namespace mako::image
