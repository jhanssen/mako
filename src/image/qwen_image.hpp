#pragma once

#include "image/engine.hpp"

namespace mako::image {

// Qwen-Image 2.1: the text encoder's last layer (a Qwen3-VL language model) conditions a dual-stream transformer.
class QwenImage final : public Engine {
public:
    QwenImage(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    const char* name() const override { return "Qwen-Image 2.1"; }
    uint32_t default_steps() const override { return 40; }
    // The reference pipeline runs without guidance by default; 4 as Qwen-Image's first release suggested (2 to 6
    // tried: clean, the negative prompt followed).
    float default_guidance() const override { return 4.0f; }
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // The prompt embeddings [S][4096]. The first step runs the whole sequence and keeps the text's keys and values,
    // the later ones only the image's.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;

    // The prompt template around `prompt`, and the token count of its system part.
    static std::string templated(const std::string& prompt);
    uint32_t system_tokens() const { return drop_; }

protected:
    LoraTarget lora_target(const std::string& module) override;

private:
    struct Block {
        Tensor qkv, o, norm_q, norm_k, gate_up, down; // q, k, v and gate, up side by side: one GEMM each
    };
    void load_transformer(const safetensors::Files& f);

    Tensor img_in_, txt_norm_, txt_in1_, txt_in2_, t_lin1_, t_lin2_, mod_, out_lin_, proj_out_;
    std::vector<Block> blocks_;
    static constexpr uint32_t kDim = 4096, kHeads = 32, kTimeDim = 256;
    static constexpr float kEps = 1e-6f;
    static constexpr const char* kSystem = "Comprehend and analyze the provided prompt.";
    uint32_t drop_ = 0;
    // The generation begun: the latent grid; for each slot, its text rows, steps run and the text's keys and values
    // of each block (half [heads][S][128]).
    uint32_t gh_ = 0, gw_ = 0;
    struct Text {
        uint32_t S = 0, steps = 0;
        std::vector<float> prompt;
        std::vector<mtl::Ref<MTL::Buffer>> k, v;
    } texts_[2];
};

} // namespace mako::image
