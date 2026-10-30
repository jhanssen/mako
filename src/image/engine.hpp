#pragma once

#include "gpu/runtime.hpp"
#include "safetensors/safetensors.hpp"
#include "text/tokenizer.hpp"

#include <functional>
#include <memory>
#include <optional>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

struct Options {
    std::string text_encoder; // another directory of the text encoder's safetensors (the same model's, changed)
    std::string transformer;  // another file or directory of the transformer's weights
    gpu::Telemetry telemetry = gpu::Telemetry::off;
};

// An image model from its original files (a directory with model_index.json, text_encoder/, transformer/, vae/,
// scheduler/): the parts the models share, the Qwen3-VL text encoder, the VAE decoder, the flow-matching schedule and
// the denoising loop. QwenImage and Krea2 add their diffusion transformers; open() picks one from model_index.json.
class Engine : public gpu::Runtime {
public:
    // The model in `dir`, by the pipeline model_index.json names. Throws on a model it does not know.
    static std::unique_ptr<Engine> open(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    virtual ~Engine() = default;
    virtual const char* name() const = 0;
    virtual uint32_t default_steps() const = 0;
    // The guidance scale with a negative prompt: the step's velocity is neg + scale * (pos - neg).
    virtual float default_guidance() const = 0;

    struct Request {
        std::string prompt;
        uint32_t width = 1024, height = 1024; // multiples of 32
        uint32_t steps = 0;                   // 0: the model's default
        uint64_t seed = 0;
        std::span<const float> noise; // instead of the seed's: [h * w][64]
        std::optional<float> terminal; // the schedule's last noise level (0: not stretched); unset: the model's
        // What the image should not be: classifier-free guidance, a second transformer pass a step (unset: none).
        std::optional<std::string> negative;
        std::optional<float> guidance; // its scale (unset: the model's)
        std::function<void(uint32_t step, uint32_t steps)> progress;
    };
    // Text to image: RGBA [height][width][4] in [-1, 1].
    std::vector<float> generate(const Request& r);

    // The transformer's text input for a prompt.
    virtual std::vector<float> prompt_embeddings(const std::string& prompt) = 0;
    // A generation's text and latent grid (h x w tokens of 64), then step() for each denoising step: the velocity the
    // transformer predicts for latents [h * w][64] at noise level sigma. Slots keep separate texts (1: the negative
    // prompt's) over the same grid.
    virtual void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) = 0;
    virtual std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) = 0;

    // The text encoder over token ids: rows [drop, n) of its last layer's output before the final norm ([n - drop][E]),
    // or with taps, of the hidden states listed (0: the embeddings, i: layer i's output), [n - drop][taps][E].
    std::vector<float> encode_text(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps = {});
    uint32_t text_width() const { return tE_; }

    // The VAE decoder: transformer latents [h * w][64] (normalized) to RGBA [16h][16w][4] in [-1, 1].
    std::vector<float> decode(std::span<const float> latents, uint32_t h, uint32_t w);

    // The noise levels of `steps` denoising steps for an image of `tokens` latent tokens, then 0 (the schedule's
    // shift, stretched to end at its terminal value).
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal = std::nullopt) const;

    // Merges a LoRA into the transformer's weights for good: W += scale * (alpha / rank) * B A for each matrix it
    // adapts (rounded back to BF16). Keys as Diffusers, PEFT or ComfyUI name them; alpha from the adapter's .alpha
    // tensors, else its PEFT config in the metadata, else the rank. Returns the matrices adapted; throws on a file
    // it cannot use (keys it does not know are an error).
    uint32_t add_lora(const std::string& path, float scale = 1.0f);
    const text::Tokenizer& tokenizer() const { return *tok_; }

protected:
    // dir's text encoder (or opts.text_encoder), VAE, scheduler and tokenizer (processor/ or tokenizer/).
    Engine(const std::string& dir, mtl::Context& ctx, const Options& opts);

    // The weight a LoRA module adapts: its matrix and the rows it has there (fused matrices share one).
    struct LoraTarget {
        Tensor* w = nullptr;
        uint32_t row0 = 0, rows = 0;
    };
    // `module` with the file's prefixes stripped (transformer., diffusion_model., ...).
    virtual LoraTarget lora_target(const std::string& module) = 0;

    Tensor upload(const safetensors::Files& files, const std::string& name, bool matrix);
    // BF16 (or F32, rounded to BF16) matrices of one width, their rows one after another in one tensor.
    Tensor upload_rows(const safetensors::Files& files, std::span<const std::string> names);
    // An RMS norm's scale stored zero-centred: weight + 1, as floats.
    Tensor upload_plus_one(const safetensors::Files& files, const std::string& name);
    void add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n);
    // Per-head q / k / v half tiles of T rows (rope: 0 none, 1 halves, 2 adjacent pairs; cs: the angle table).
    void heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
               uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope, MTL::Buffer* y);
    // heads() with each head RMS-normed first (times w).
    void heads_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
                    uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint64_t cs_off, uint32_t rope, const Tensor& w,
                    float eps, uint32_t row0, MTL::Buffer* y);
    // block: positions from here on see each other both ways; below it, causal (0: all both ways).
    void attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v, MTL::Buffer* o,
                   uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t heads, uint32_t kv_heads,
                   uint32_t q_pos0, uint32_t block);
    // y[r][i] += b[i] over rows of n.
    void bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n, uint32_t rows,
              uint64_t y_off = 0);
    // A tensor's values as floats.
    static std::vector<float> floats(const safetensors::Files& files, const std::string& name);
    void unary(MTL::ComputeCommandEncoder*& enc, const char* fn, MTL::Buffer* x, uint32_t n);

    static constexpr uint32_t kHeadDim = 128;
    static constexpr uint32_t kAttnRows = 64; // queries a threadgroup of the attention kernel
    static constexpr uint32_t kLatent = 64;   // a transformer token's latent values

    std::optional<text::Tokenizer> tok_;
    struct Schedule {
        double base_seq = 256, max_seq = 4096, base_shift = 0.5, max_shift = 1.15, terminal = 0;
        std::optional<double> mu; // a fixed shift (distilled models) instead of one by image size
    } sched_;
    std::string dir_;

private:
    struct TextLayer {
        Tensor attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate, up, down;
    };
    // Rows of the token embedding table (BF16, read from the file).
    void embed(std::span<const int32_t> ids, float* out) const;

    // The VAE's convolutions as GEMMs over channels-last images: weights F16 [cout][kpad] (cin * k * k columns, a 3x3's
    // zero padded to a multiple of 64). A causal 3D convolution's weights are its last time tap's (a single frame).
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

    safetensors::Files text_files_, vae_files_;
    safetensors::Files::Ref embd_;
    std::vector<TextLayer> text_;
    uint32_t tE_ = 0, tHeads_ = 0, tKv_ = 0;
    float tEps_ = 1e-6f, tTheta_ = 5e6f;

    Conv post_quant_, conv_in_, conv_out_, qkv_, attn_proj_;
    Tensor attn_g_, out_g_;
    Res mid0_, mid1_;
    std::vector<Up> ups_;
    std::vector<float> lat_mean_, lat_std_;
    // The VAE's latent channels and the 2x2 patches a token packs (16 channels: Qwen-Image's 8x VAE), and whether its
    // up blocks have the duplicating shortcut (Qwen-Image 2.1's).
    uint32_t z_ = kLatent, patch_ = 1;
    bool residual_ = false;
};

// "FILE" or "FILE:SCALE" (the scale a number after the last colon). Throws std::invalid_argument.
std::pair<std::string, float> parse_lora(const std::string& spec);

} // namespace mako::image
