#pragma once

#include "gpu/runtime.hpp"
#include "safetensors/safetensors.hpp"
#include "text/tokenizer.hpp"

#include <functional>
#include <optional>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

// Qwen-Image 2.1 from its original files (a directory with text_encoder/, transformer/, vae/): the text encoder (a
// Qwen3-VL language model), the diffusion transformer and the VAE decoder.
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

    // The VAE decoder: latents [h * w][64] (the transformer's, normalized) to RGBA [16h][16w][4] in [-1, 1].
    std::vector<float> decode(std::span<const float> latents, uint32_t h, uint32_t w);

    // The noise levels of `steps` denoising steps for an image of `tokens` latent tokens, then 0 (the scheduler's
    // dynamic shift, stretched to end at its terminal value).
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens) const;

    struct Request {
        std::string prompt;
        uint32_t width = 1024, height = 1024; // multiples of 32
        uint32_t steps = 40;
        uint64_t seed = 0;
        std::span<const float> noise; // instead of the seed's: [h * w][64]
        std::function<void(uint32_t step, uint32_t steps)> progress;
    };
    // Text to image: RGBA [height][width][4] in [-1, 1].
    std::vector<float> generate(const Request& r);
    const text::Tokenizer& tokenizer() const { return *tok_; }
    // The prompt template around `prompt`, and the token count of its system part.
    static std::string templated(const std::string& prompt);
    uint32_t system_tokens() const { return drop_; }

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

    // The VAE's convolutions as GEMMs over channels-last images: weights [cout][kpad] (cin * k * k columns, zero
    // padded to a multiple of 32), F32.
    struct Conv {
        Tensor w, b;
        uint32_t cin = 0, cout = 0, k = 3, kpad = 0;
    };
    struct Res {
        Tensor g1, g2; // the RMS norms' scales
        Conv c1, c2, sc;
        bool shortcut = false;
    };
    struct Up {
        std::vector<Res> res;
        bool up = false;
        Conv conv;
        uint32_t cin = 0, cout = 0, ft = 1;
    };
    void load_vae();
    Conv load_conv(const std::string& name, uint32_t k);
    void conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t h, uint32_t w, MTL::Buffer* y);
    // y = conv2(silu(norm2(conv1(silu(norm1(x)))))) + shortcut(x); t1, t2 scratch.
    void res_block(MTL::ComputeCommandEncoder*& enc, const Res& r, MTL::Buffer* x, uint32_t h, uint32_t w,
                   MTL::Buffer* y, MTL::Buffer* t1, MTL::Buffer* t2);
    void norm_silu(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, MTL::Buffer* y, uint32_t rows,
                   uint32_t c);

    safetensors::Files text_files_, dit_files_, vae_files_;
    safetensors::Files::Ref embd_;
    std::vector<TextLayer> text_;
    static constexpr uint32_t kTextEmbd = 4096, kTextHeads = 32, kTextKvHeads = 8, kHeadDim = 128;
    static constexpr float kTextEps = 1e-6f, kTextTheta = 5e6f;

    Tensor img_in_, txt_norm_, txt_in1_, txt_in2_, t_lin1_, t_lin2_, mod_, out_lin_, proj_out_;
    std::vector<Block> blocks_;
    static constexpr uint32_t kDim = 4096, kHeads = 32, kLatent = 64, kTimeDim = 256;
    static constexpr float kEps = 1e-6f;
    static constexpr const char* kSystem = "Comprehend and analyze the provided prompt.";
    // The generation begun: text rows, the latent grid, steps run, the rotary table of the whole sequence and the
    // text's keys and values of each block (half [heads][S][128]).
    uint32_t S_ = 0, gh_ = 0, gw_ = 0, steps_ = 0;
    std::vector<float> prompt_;
    std::vector<mtl::Ref<MTL::Buffer>> text_k_, text_v_;

    Conv post_quant_, conv_in_, conv_out_, qkv_, attn_proj_;
    Tensor attn_g_, out_g_;
    Res mid0_, mid1_;
    std::vector<Up> ups_;
    std::vector<float> lat_mean_, lat_std_;

    std::optional<text::Tokenizer> tok_;
    uint32_t drop_ = 0;
    struct Schedule {
        double base_seq = 256, max_seq = 4096, base_shift = 0.5, max_shift = 1.15, terminal = 0;
    } sched_;
};

} // namespace mako::image
