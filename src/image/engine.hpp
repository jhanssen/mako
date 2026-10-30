#pragma once

#include "gpu/runtime.hpp"
#include "safetensors/safetensors.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

// Qwen-Image 2.1 from its original files (a directory with text_encoder/, transformer/, vae/): the text encoder (a
// Qwen3-VL language model) and the diffusion transformer so far.
class Engine : public gpu::Runtime {
public:
    Engine(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);

    // The text encoder over a prompt's token ids: its last layer's output before the final norm, rows [drop, n)
    // (drop: the template's system part), [n - drop][4096].
    std::vector<float> encode_text(std::span<const int32_t> ids, uint32_t drop);

    // A generation's text and image size: the prompt embeddings ([S][4096]) and the latent grid (h x w tokens).
    // Then step() for each denoising step, the first running the whole sequence and keeping the text's keys and
    // values, the later ones only the image's.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w);
    // The velocity the transformer predicts for latents [h * w][64] at noise level sigma: [h * w][64].
    std::vector<float> step(std::span<const float> latents, float sigma);

private:
    struct TextLayer {
        Tensor attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate, up, down;
    };
    Tensor upload(const safetensors::Files& files, const std::string& name, bool matrix);
    // Rows of the token embedding table (BF16, read from the file).
    void embed(std::span<const int32_t> ids, float* out) const;
    // y = x + z over n floats.
    void add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n);
    // Per-head q / k / v half tiles of T rows (rope: 0 none, 1 halves, 2 adjacent pairs; cs: the angle table).
    void heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
               uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope, MTL::Buffer* y);
    void attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v, MTL::Buffer* o,
                   uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t heads, uint32_t kv_heads,
                   uint32_t q_pos0, uint32_t block);

    struct Block {
        Tensor q, k, v, o, norm_q, norm_k, gate, up, down;
    };
    void load_transformer(const std::string& dir);

    safetensors::Files text_files_, dit_files_;
    safetensors::Files::Ref embd_;
    std::vector<TextLayer> text_;
    static constexpr uint32_t kTextEmbd = 4096, kTextHeads = 32, kTextKvHeads = 8, kHeadDim = 128;
    static constexpr float kTextEps = 1e-6f, kTextTheta = 5e6f;

    Tensor img_in_, txt_norm_, txt_in1_, txt_in2_, t_lin1_, t_lin2_, mod_, out_lin_, proj_out_;
    std::vector<Block> blocks_;
    static constexpr uint32_t kDim = 4096, kHeads = 32, kLatent = 64, kTimeDim = 256;
    static constexpr float kEps = 1e-6f;
    // The generation begun: text rows, the latent grid, steps run, the rotary table of the whole sequence and the
    // text's keys and values of each block (half [heads][S][128]).
    uint32_t S_ = 0, gh_ = 0, gw_ = 0, steps_ = 0;
    std::vector<float> prompt_;
    std::vector<mtl::Ref<MTL::Buffer>> text_k_, text_v_;
};

} // namespace mako::image
