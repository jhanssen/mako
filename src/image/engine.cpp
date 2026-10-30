#include "image/engine.hpp"

#include "gguf/read_pool.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>

namespace mako::image {

namespace {

struct HeadsArgs {
    uint32_t rows, ld, col0, dim, n_pad, row0, rope;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};
struct ModArgs {
    uint32_t n;
    float eps;
    uint32_t split, mod_stride, mod_off;
};
struct HeadRowsArgs {
    uint32_t dim, src_pad, dst_pad, rows;
};
struct Im2colArgs {
    uint32_t h, w, c, p0, rows, kpad;
};
struct UpArgs {
    uint32_t h, w, c, c_out, ft, repeats;
};
struct TransposeArgs {
    uint32_t rows, cols, ldx, col0;
};

// Pairs i of the rotary angle table for (frame, height, width) positions: 8 pairs on the frame axis, then 28 each on
// height and width (axes 16 / 56 / 56 of the 128 dims), theta 10000.
void rope3(float* cs, int64_t f, int64_t hpos, int64_t wpos)
{
    constexpr int dims[3] = {16, 56, 56};
    const int64_t pos[3] = {f, hpos, wpos};
    uint32_t i = 0;
    for (int a = 0; a < 3; ++a)
        for (int j = 0; j < dims[a] / 2; ++j, ++i) {
            const float inv = 1.0f / std::pow(10000.0f, float(2 * j) / float(dims[a]));
            const float ang = float(pos[a]) * inv;
            cs[2 * i] = std::cos(ang);
            cs[2 * i + 1] = std::sin(ang);
        }
}

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

safetensors::Files open_files(const std::string& dir)
{
    auto f = safetensors::Files::open(dir);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

Engine::Engine(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry)
    : gpu::Runtime(ctx, telemetry), text_files_(open_files(dir + "/text_encoder")),
      dit_files_(open_files(dir + "/transformer")), vae_files_(open_files(dir + "/vae"))
{
    const std::string p = "model.language_model.";
    embd_ = text_files_.find(p + "embed_tokens.weight");
    if (!embd_ || embd_.tensor->type != gguf::Type::BF16 || embd_.tensor->dims[0] != kTextEmbd)
        throw std::runtime_error("text encoder: embed_tokens missing or not BF16 [4096 x vocab]");
    for (uint32_t i = 0;; ++i) {
        const std::string l = std::format("{}layers.{}.", p, i);
        if (!text_files_.find(l + "input_layernorm.weight"))
            break;
        TextLayer t;
        t.attn_norm = upload(text_files_, l + "input_layernorm.weight", false);
        t.q = upload(text_files_, l + "self_attn.q_proj.weight", true);
        t.k = upload(text_files_, l + "self_attn.k_proj.weight", true);
        t.v = upload(text_files_, l + "self_attn.v_proj.weight", true);
        t.o = upload(text_files_, l + "self_attn.o_proj.weight", true);
        t.q_norm = upload(text_files_, l + "self_attn.q_norm.weight", false);
        t.k_norm = upload(text_files_, l + "self_attn.k_norm.weight", false);
        t.ffn_norm = upload(text_files_, l + "post_attention_layernorm.weight", false);
        t.gate = upload(text_files_, l + "mlp.gate_proj.weight", true);
        t.up = upload(text_files_, l + "mlp.up_proj.weight", true);
        t.down = upload(text_files_, l + "mlp.down_proj.weight", true);
        text_.push_back(std::move(t));
    }
    if (text_.empty())
        throw std::runtime_error("text encoder: no layers");
    load_transformer(dir);
    load_vae();
    // The latents' normalization, from the VAE's config.
    std::ifstream in(dir + "/vae/config.json", std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic j;
    if (glz::read_json(j, text) || !j.contains("latents_mean") || !j.contains("latents_std"))
        throw std::runtime_error("vae/config.json: no latents_mean / latents_std");
    for (const auto& v : j["latents_mean"].get_array())
        lat_mean_.push_back(float(v.get_number()));
    for (const auto& v : j["latents_std"].get_array())
        lat_std_.push_back(float(v.get_number()));
    if (lat_mean_.size() != kLatent || lat_std_.size() != kLatent)
        throw std::runtime_error("vae/config.json: latents_mean / latents_std not 64 wide");

    std::ifstream sin(dir + "/scheduler/scheduler_config.json", std::ios::binary);
    const std::string stext{std::istreambuf_iterator<char>(sin), std::istreambuf_iterator<char>()};
    glz::generic sj;
    if (glz::read_json(sj, stext) || !sj.is_object())
        throw std::runtime_error("scheduler/scheduler_config.json missing or unreadable");
    auto num = [&](const char* key, double& v) {
        if (sj.contains(key) && sj[key].is_number())
            v = sj[key].get_number();
    };
    num("base_image_seq_len", sched_.base_seq);
    num("max_image_seq_len", sched_.max_seq);
    num("base_shift", sched_.base_shift);
    num("max_shift", sched_.max_shift);
    num("shift_terminal", sched_.terminal);

    auto t = text::Tokenizer::load_json(dir + "/processor/tokenizer.json");
    if (!t)
        throw std::runtime_error(t.error());
    tok_ = std::move(*t);
    drop_ = uint32_t(tok_->encode("<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n", true).size());
}

std::string Engine::templated(const std::string& prompt)
{
    return "<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n" +
           (prompt.empty() ? " " : prompt) + "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<float> Engine::sigmas(uint32_t steps, uint32_t tokens) const
{
    const double m = (sched_.max_shift - sched_.base_shift) / (sched_.max_seq - sched_.base_seq);
    const double mu = double(tokens) * m + (sched_.base_shift - m * sched_.base_seq);
    std::vector<float> s(steps);
    for (uint32_t i = 0; i < steps; ++i) {
        // linspace(1, 1 / steps, steps), then exp(mu) / (exp(mu) + 1 / t - 1), in float as the scheduler does.
        const float t = steps == 1 ? 1.0f : float(1.0 + (1.0 / steps - 1.0) * double(i) / double(steps - 1));
        s[i] = float(std::exp(mu) / (std::exp(mu) + (1.0 / double(t) - 1.0)));
    }
    if (sched_.terminal > 0 && steps > 0) {
        const float scale = (1.0f - s.back()) / (1.0f - float(sched_.terminal));
        for (float& v : s)
            v = 1.0f - (1.0f - v) / scale;
    }
    s.push_back(0.0f);
    return s;
}

std::vector<float> Engine::generate(const Request& r)
{
    if (r.width % 32 || r.height % 32 || !r.width || !r.height)
        throw std::runtime_error("width and height must be multiples of 32");
    const uint32_t h = r.height / 16, w = r.width / 16, N = h * w;
    const auto ids = tok_->encode(templated(r.prompt), true);
    const auto prompt = encode_text(ids, drop_);
    std::vector<float> x(uint64_t(N) * kLatent);
    if (!r.noise.empty()) {
        if (r.noise.size() != x.size())
            throw std::runtime_error("noise must be [h * w][64]");
        std::copy(r.noise.begin(), r.noise.end(), x.begin());
    } else {
        std::mt19937_64 rng(r.seed);
        std::normal_distribution<float> nd;
        for (float& v : x)
            v = nd(rng);
    }
    const auto sig = sigmas(r.steps, N);
    begin(prompt, h, w);
    for (uint32_t i = 0; i < r.steps; ++i) {
        const auto v = step(x, sig[i]);
        const float dt = sig[i + 1] - sig[i];
        for (size_t j = 0; j < x.size(); ++j)
            x[j] += dt * v[j];
        if (r.progress)
            r.progress(i + 1, r.steps);
    }
    return decode(x, h, w);
}

Engine::Conv Engine::load_conv(const std::string& name, uint32_t k)
{
    const auto r = vae_files_.find(name + ".weight");
    const auto rb = vae_files_.find(name + ".bias");
    if (!r || !rb || r.tensor->type != gguf::Type::F32 || r.tensor->dims.size() != 4)
        throw std::runtime_error(name + ": missing, or not an F32 convolution");
    Conv c;
    c.k = k;
    c.cout = uint32_t(r.tensor->dims[3]);
    c.cin = uint32_t(r.tensor->dims[2]);
    if (r.tensor->dims[0] != k || r.tensor->dims[1] != k)
        throw std::runtime_error(name + ": unexpected kernel size");
    const uint32_t cols = c.cin * k * k;
    c.kpad = round_up(cols, 32);
    c.w = reserve(uint64_t(c.cout) * c.kpad * 4, gguf::Type::F32, {c.kpad, c.cout});
    auto* dst = reinterpret_cast<float*>(static_cast<char*>(c.w.buf->contents()) + c.w.off);
    std::vector<float> row(cols);
    for (uint32_t o = 0; o < c.cout; ++o) {
        gguf::read_exact(r.file->fd(), row.data(), uint64_t(cols) * 4, r.tensor->offset + uint64_t(o) * cols * 4);
        std::copy(row.begin(), row.end(), dst + uint64_t(o) * c.kpad);
        std::fill(dst + uint64_t(o) * c.kpad + cols, dst + uint64_t(o + 1) * c.kpad, 0.0f);
    }
    c.b = upload(vae_files_, name + ".bias", false);
    return c;
}

void Engine::load_vae()
{
    auto res = [&](const std::string& p) {
        Res r;
        r.g1 = upload(vae_files_, p + ".norm1.gamma", false);
        r.g2 = upload(vae_files_, p + ".norm2.gamma", false);
        r.c1 = load_conv(p + ".conv1", 3);
        r.c2 = load_conv(p + ".conv2", 3);
        if (vae_files_.find(p + ".conv_shortcut.weight")) {
            r.sc = load_conv(p + ".conv_shortcut", 1);
            r.shortcut = true;
        }
        return r;
    };
    post_quant_ = load_conv("post_quant_conv", 1);
    conv_in_ = load_conv("decoder.conv_in", 3);
    mid0_ = res("decoder.mid_block.resnets.0");
    mid1_ = res("decoder.mid_block.resnets.1");
    attn_g_ = upload(vae_files_, "decoder.mid_block.attentions.0.norm.gamma", false);
    qkv_ = load_conv("decoder.mid_block.attentions.0.to_qkv", 1);
    attn_proj_ = load_conv("decoder.mid_block.attentions.0.proj", 1);
    for (uint32_t i = 0;; ++i) {
        const std::string p = std::format("decoder.up_blocks.{}", i);
        if (!vae_files_.find(p + ".resnets.0.conv1.weight"))
            break;
        Up u;
        for (uint32_t j = 0; vae_files_.find(std::format("{}.resnets.{}.conv1.weight", p, j)); ++j)
            u.res.push_back(res(std::format("{}.resnets.{}", p, j)));
        u.cin = u.res.front().c1.cin;
        u.cout = u.res.back().c2.cout;
        if (vae_files_.find(p + ".upsampler.resample.1.weight")) {
            u.up = true;
            u.conv = load_conv(p + ".upsampler.resample.1", 3);
            // Blocks that also upsample in time (a time convolution) keep the second of two time copies.
            u.ft = vae_files_.find(p + ".upsampler.time_conv.weight") ? 2 : 1;
        }
        ups_.push_back(std::move(u));
    }
    out_g_ = upload(vae_files_, "decoder.norm_out.gamma", false);
    conv_out_ = load_conv("decoder.conv_out", 3);
}

void Engine::load_transformer(const std::string&)
{
    const auto& f = dit_files_;
    img_in_ = upload(f, "img_in.weight", true);
    txt_norm_ = upload(f, "txt_in.text_norm.weight", false);
    // Stored zero-centred: the scale is weight + 1.
    auto* tn = reinterpret_cast<float*>(static_cast<char*>(txt_norm_.buf->contents()) + txt_norm_.off);
    for (uint32_t i = 0; i < kDim; ++i)
        tn[i] += 1.0f;
    txt_in1_ = upload(f, "txt_in.in_layer.weight", true);
    txt_in2_ = upload(f, "txt_in.out_layer.weight", true);
    t_lin1_ = upload(f, "time_text_embed.timestep_embedder.linear_1.weight", true);
    t_lin2_ = upload(f, "time_text_embed.timestep_embedder.linear_2.weight", true);
    mod_ = upload(f, "modulation.1.weight", true);
    out_lin_ = upload(f, "norm_out.linear.weight", true);
    proj_out_ = upload(f, "proj_out.weight", true);
    for (uint32_t i = 0;; ++i) {
        const std::string b = std::format("transformer_blocks.{}.", i);
        if (!f.find(b + "attn.to_q.weight"))
            break;
        Block B;
        B.q = upload(f, b + "attn.to_q.weight", true);
        B.k = upload(f, b + "attn.to_k.weight", true);
        B.v = upload(f, b + "attn.to_v.weight", true);
        B.o = upload(f, b + "attn.to_out.0.weight", true);
        B.norm_q = upload(f, b + "attn.norm_q.weight", false);
        B.norm_k = upload(f, b + "attn.norm_k.weight", false);
        B.gate = upload(f, b + "img_mlp.gate_layer.weight", true);
        B.up = upload(f, b + "img_mlp.proj.weight", true);
        B.down = upload(f, b + "img_mlp.out.weight", true);
        blocks_.push_back(std::move(B));
    }
    if (blocks_.empty())
        throw std::runtime_error("transformer: no blocks");
}

Engine::Tensor Engine::upload(const safetensors::Files& files, const std::string& name, bool matrix)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    return upload_raw(r.file->fd(), r.tensor->offset, r.tensor->bytes, r.tensor->type, r.tensor->dims, name, matrix);
}

void Engine::embed(std::span<const int32_t> ids, float* out) const
{
    const uint64_t vocab = embd_.tensor->dims[1], row = uint64_t(kTextEmbd) * 2;
    std::vector<uint16_t> raw(kTextEmbd);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || uint64_t(ids[i]) >= vocab)
            throw std::runtime_error("token out of range");
        gguf::read_exact(embd_.file->fd(), raw.data(), row, embd_.tensor->offset + uint64_t(ids[i]) * row);
        auto* o = reinterpret_cast<uint32_t*>(out + i * kTextEmbd);
        for (uint32_t j = 0; j < kTextEmbd; ++j)
            o[j] = uint32_t(raw[j]) << 16;
    }
}

void Engine::add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n)
{
    dispatch(enc, "add", {x, z, y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void Engine::heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t n_heads,
                   uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope, MTL::Buffer* y)
{
    const HeadsArgs a{T, ld, col0, kHeadDim, n_pad, 0, rope};
    dispatch(enc, "heads_half", {x, cs ? cs : x, y}, &a, sizeof(a), MTL::Size(kHeadDim / 2, T, n_heads),
             MTL::Size(64, 1, 1));
    if (n_pad > T) {
        const uint32_t c[4] = {kHeadDim, n_pad, T, 0};
        dispatch(enc, "heads_clear", {y}, c, sizeof(c), MTL::Size(kHeadDim, n_pad - T, n_heads), MTL::Size(128, 1, 1));
    }
}

void Engine::attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v,
                       MTL::Buffer* o, uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t n_heads,
                       uint32_t kv_heads, uint32_t q_pos0, uint32_t block)
{
    const AttnArgs a{n_q, q_pad, n_k, k_pad, n_heads, kv_heads, q_pos0, block, n_heads * kHeadDim,
                     1.0f / std::sqrt(float(kHeadDim))};
    dispatch(enc, "flash_attn_128", {q, k, v, o}, &a, sizeof(a), MTL::Size(q_pad / 32, n_heads, 1),
             MTL::Size(256, 1, 1), false);
}

std::vector<float> Engine::encode_text(std::span<const int32_t> ids, uint32_t drop)
{
    const uint32_t T = uint32_t(ids.size()), E = kTextEmbd, kv = kTextKvHeads * kHeadDim;
    if (T <= drop)
        throw std::runtime_error("the prompt is all template");
    const uint32_t q_pad = round_up(T, 32), k_pad = round_up(T, 128);
    MTL::Buffer* x = scratch("t_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("t_xn", uint64_t(T) * E * 4);
    MTL::Buffer* q = scratch("t_q", uint64_t(T) * E * 4);
    MTL::Buffer* k = scratch("t_k", uint64_t(T) * kv * 4);
    MTL::Buffer* v = scratch("t_v", uint64_t(T) * kv * 4);
    MTL::Buffer* qh = scratch("t_qh", uint64_t(kTextHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("t_kh", uint64_t(kTextKvHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("t_vh", uint64_t(kTextKvHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("t_att", uint64_t(T) * E * 4);
    MTL::Buffer* y = scratch("t_y", uint64_t(T) * E * 4);
    MTL::Buffer* cs = scratch("t_cs", uint64_t(T) * kHeadDim / 2 * 8);
    embed(ids, static_cast<float*>(x->contents()));
    // Text positions are the same on the three rotary axes, which makes the multi-axis rotation the plain one.
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = 0; i < kHeadDim / 2; ++i) {
            const float inv = 1.0f / std::pow(kTextTheta, float(2 * i) / float(kHeadDim));
            const float ang = float(t) * inv;
            c[(uint64_t(t) * kHeadDim / 2 + i) * 2] = std::cos(ang);
            c[(uint64_t(t) * kHeadDim / 2 + i) * 2 + 1] = std::sin(ang);
        }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    for (uint32_t il = 0; il < text_.size(); ++il) {
        const TextLayer& L = text_[il];
        cur_layer_ = il;
        norm(enc, x, &L.attn_norm, xn, T, E, E, kTextEps);
        proj(enc, L.q, xn, 0, q, 0, T);
        proj(enc, L.k, xn, 0, k, 0, T);
        proj(enc, L.v, xn, 0, v, 0, T);
        norm(enc, q, &L.q_norm, q, T * kTextHeads, kHeadDim, kHeadDim, kTextEps);
        norm(enc, k, &L.k_norm, k, T * kTextKvHeads, kHeadDim, kHeadDim, kTextEps);
        heads(enc, q, E, 0, kTextHeads, T, q_pad, cs, 1, qh);
        heads(enc, k, kv, 0, kTextKvHeads, T, k_pad, cs, 1, kh);
        heads(enc, v, kv, 0, kTextKvHeads, T, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, kTextHeads, kTextKvHeads, 0, UINT32_MAX);
        proj(enc, L.o, att, 0, y, 0, T);
        add(enc, x, y, x, T * E);
        norm(enc, x, &L.ffn_norm, xn, T, E, E, kTextEps);
        ffn_dense(enc, L.gate, L.up, L.down, xn, y, 0.0f, T);
        add(enc, x, y, x, T * E);
    }
    enc->endEncoding();
    finish(cb);
    const auto* xf = static_cast<const float*>(x->contents());
    return std::vector<float>(xf + uint64_t(drop) * E, xf + uint64_t(T) * E);
}

void Engine::begin(std::span<const float> prompt, uint32_t h, uint32_t w)
{
    if (prompt.empty() || prompt.size() % kDim)
        throw std::runtime_error("prompt embeddings must be rows of 4096");
    S_ = uint32_t(prompt.size() / kDim);
    gh_ = h;
    gw_ = w;
    steps_ = 0;
    prompt_.assign(prompt.begin(), prompt.end());
    text_k_.clear();
    text_v_.clear();
    for (size_t b = 0; b < blocks_.size(); ++b) {
        const uint64_t bytes = uint64_t(kHeads) * S_ * kHeadDim * 2;
        text_k_.emplace_back(ctx_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        text_v_.emplace_back(ctx_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    }
    // Text positions advance on all three axes; the image block takes the next frame position and a height / width
    // grid centred on zero.
    MTL::Buffer* cs = scratch("d_cs", uint64_t(S_ + h * w) * kHeadDim / 2 * 8);
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < S_; ++t)
        rope3(c + uint64_t(t) * kHeadDim, t, t, t);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            rope3(c + uint64_t(S_ + y * w + x) * kHeadDim, S_, int64_t(y) - int64_t(h - h / 2),
                  int64_t(x) - int64_t(w - w / 2));
}

std::vector<float> Engine::step(std::span<const float> latents, float sigma)
{
    const uint32_t N = gh_ * gw_, D = kDim;
    if (latents.size() != uint64_t(N) * kLatent)
        throw std::runtime_error("latents must be [h * w][64]");
    const bool first = steps_++ == 0;
    // The first step runs the text rows too; later ones only the image's, against the text's cached keys and values.
    const uint32_t S = S_, T = first ? S + N : N, row0 = first ? 0 : S, n_k = S + N;
    const uint32_t q_pad = round_up(T, 32), k_pad = round_up(n_k, 128), split = first ? S : 0;
    MTL::Buffer* x = scratch("d_x", uint64_t(T) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(T) * D * 4);
    MTL::Buffer* q = scratch("d_q", uint64_t(T) * D * 4);
    MTL::Buffer* k = scratch("d_k", uint64_t(T) * D * 4);
    MTL::Buffer* v = scratch("d_v", uint64_t(T) * D * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(T) * D * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(T) * D * 4);
    MTL::Buffer* lat = scratch("d_lat", uint64_t(N) * kLatent * 4);
    MTL::Buffer* out = scratch("d_out", uint64_t(T) * kLatent * 4);
    MTL::Buffer* tp = scratch("d_tproj", 2 * kTimeDim * 4);
    MTL::Buffer* temb = scratch("d_temb", 2 * D * 4);
    MTL::Buffer* temb_s = scratch("d_temb_s", 2 * D * 4);
    MTL::Buffer* mod = scratch("d_mod", 2 * 4 * D * 4);
    MTL::Buffer* omod = scratch("d_omod", 2 * D * 4);
    MTL::Buffer* cs = scratch("d_cs", uint64_t(S + N) * kHeadDim / 2 * 8);
    std::memcpy(lat->contents(), latents.data(), latents.size() * 4);
    // Sinusoidal timestep embeddings (cos half first) of 1000 * sigma and of 0, the text's.
    auto* tpf = static_cast<float*>(tp->contents());
    for (int r = 0; r < 2; ++r)
        for (uint32_t i = 0; i < kTimeDim / 2; ++i) {
            const float freq = std::exp(-std::log(10000.0f) * float(i) / float(kTimeDim / 2));
            const float arg = (r == 0 ? 1000.0f * sigma : 0.0f) * freq;
            tpf[r * kTimeDim + i] = std::cos(arg);
            tpf[r * kTimeDim + kTimeDim / 2 + i] = std::sin(arg);
        }
    if (first) {
        MTL::Buffer* pr = scratch("d_prompt", uint64_t(S) * D * 4);
        std::memcpy(pr->contents(), prompt_.data(), prompt_.size() * 4);
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, t_lin1_, tp, 0, temb, 0, 2);
    uint32_t n2 = 2 * D;
    dispatch(enc, "silu", {temb}, &n2, sizeof(n2), MTL::Size(n2, 1, 1), MTL::Size(256, 1, 1));
    proj(enc, t_lin2_, temb, 0, temb_s, 0, 2);
    probe(enc, "temb", -1, temb_s, 2 * D);
    // silu(temb) feeds the block modulation and the final norm's.
    dispatch(enc, "silu", {temb_s}, &n2, sizeof(n2), MTL::Size(n2, 1, 1), MTL::Size(256, 1, 1));
    proj(enc, mod_, temb_s, 0, mod, 0, 2);
    probe(enc, "modulation", -1, mod, 2 * 4 * D);
    proj(enc, out_lin_, temb_s, 0, omod, 0, 2);
    if (first) {
        MTL::Buffer* pr = scratch("d_prompt", uint64_t(S) * D * 4);
        norm(enc, pr, &txt_norm_, xn, S, D, D, kEps);
        proj(enc, txt_in1_, xn, 0, y, 0, S);
        uint32_t n = S * D;
        dispatch(enc, "gelu_tanh", {y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        proj(enc, txt_in2_, y, 0, x, 0, S);
        probe(enc, "txt_in", -1, x, uint64_t(S) * D);
    }
    proj(enc, img_in_, lat, 0, x, uint64_t(T - N) * D * 4, N);
    probe(enc, "img_in", -1, x, uint64_t(N) * D, uint64_t(T - N) * D * 4);

    const uint32_t mstride = 4 * D;
    for (uint32_t il = 0; il < blocks_.size(); ++il) {
        const Block& B = blocks_[il];
        cur_layer_ = il;
        ModArgs ma{D, kEps, split, mstride, 0};
        dispatch(enc, "ln_modulate", {x, mod, xn}, &ma, sizeof(ma), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
        proj(enc, B.q, xn, 0, q, 0, T);
        proj(enc, B.k, xn, 0, k, 0, T);
        proj(enc, B.v, xn, 0, v, 0, T);
        norm(enc, q, &B.norm_q, q, T * kHeads, kHeadDim, kHeadDim, kEps);
        norm(enc, k, &B.norm_k, k, T * kHeads, kHeadDim, kHeadDim, kEps);
        // The rotary table's rows from this step's first token on.
        HeadsArgs ha{T, D, 0, kHeadDim, q_pad, 0, 2};
        enc->setComputePipelineState(pso("heads_half"));
        enc->setBuffer(q, 0, 0);
        enc->setBuffer(cs, uint64_t(row0) * kHeadDim / 2 * 8, 1);
        enc->setBuffer(qh, 0, 2);
        enc->setBytes(&ha, sizeof(ha), 3);
        enc->dispatchThreads(MTL::Size(kHeadDim / 2, T, kHeads), MTL::Size(64, 1, 1));
        after(enc, "heads_half");
        ha.n_pad = k_pad;
        ha.row0 = row0;
        enc->setBuffer(k, 0, 0);
        enc->setBuffer(kh, 0, 2);
        enc->setBytes(&ha, sizeof(ha), 3);
        enc->dispatchThreads(MTL::Size(kHeadDim / 2, T, kHeads), MTL::Size(64, 1, 1));
        after(enc, "heads_half");
        ha.rope = 0;
        enc->setBuffer(v, 0, 0);
        enc->setBuffer(vh, 0, 2);
        enc->setBytes(&ha, sizeof(ha), 3);
        enc->dispatchThreads(MTL::Size(kHeadDim / 2, T, kHeads), MTL::Size(64, 1, 1));
        after(enc, "heads_half");
        HeadRowsArgs hr{kHeadDim, k_pad, S, S};
        if (first) { // keep the text's keys and values
            dispatch(enc, "heads_rows", {kh, text_k_[il].get()}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_rows", {vh, text_v_[il].get()}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
        } else {
            hr = {kHeadDim, S, k_pad, S};
            dispatch(enc, "heads_rows", {text_k_[il].get(), kh}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_rows", {text_v_[il].get(), vh}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
        }
        if (k_pad > n_k) {
            const uint32_t c[4] = {kHeadDim, k_pad, n_k, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - n_k, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_clear", {vh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - n_k, kHeads),
                     MTL::Size(128, 1, 1));
        }
        attention(enc, qh, kh, vh, att, T, q_pad, n_k, k_pad, kHeads, kHeads, row0, S);
        proj(enc, B.o, att, 0, y, 0, T);
        ma.mod_off = D;
        dispatch(enc, "gate_add", {x, y, mod}, &ma, sizeof(ma), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        ma.mod_off = 2 * D;
        dispatch(enc, "ln_modulate", {x, mod, xn}, &ma, sizeof(ma), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
        ffn_dense(enc, B.gate, B.up, B.down, xn, y, 0.0f, T);
        ma.mod_off = 3 * D;
        dispatch(enc, "gate_add", {x, y, mod}, &ma, sizeof(ma), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        probe(enc, "block", int(il), x, uint64_t(T) * D);
    }
    ModArgs oa{D, kEps, split, D, 0};
    dispatch(enc, "ln_modulate", {x, omod, xn}, &oa, sizeof(oa), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
    probe(enc, "norm_out", -1, xn, uint64_t(T) * D);
    proj(enc, proj_out_, xn, 0, out, 0, T);
    enc->endEncoding();
    finish(cb);
    const auto* o = static_cast<const float*>(out->contents()) + uint64_t(T - N) * kLatent;
    return std::vector<float>(o, o + uint64_t(N) * kLatent);
}

void Engine::conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t h, uint32_t w,
                  MTL::Buffer* y)
{
    const uint32_t rows = h * w;
    if (c.k == 1) {
        gemm_->encode(enc, gguf::Type::F32, c.kpad, c.cout, rows, c.w.buf, c.w.off,
                      halves(enc, x, 0, rows, c.cin, 0), 0, y, 0);
        after(enc, "conv1x1");
    } else {
        // Image rows a pass, so the unfolded copy stays near 256 MB.
        const uint32_t chunk = std::max(32u, uint32_t((256ull << 20) / (uint64_t(c.kpad) * 2)) / 32 * 32);
        MTL::Buffer* col = scratch("v_col", uint64_t(std::min(chunk, rows)) * c.kpad * 2);
        for (uint32_t p0 = 0; p0 < rows; p0 += chunk) {
            const uint32_t n = std::min(chunk, rows - p0);
            const Im2colArgs a{h, w, c.cin, p0, n, c.kpad};
            dispatch(enc, "im2col3x3", {x, col}, &a, sizeof(a), MTL::Size(c.kpad, n, 1), MTL::Size(256, 1, 1));
            gemm_->encode(enc, gguf::Type::F32, c.kpad, c.cout, n, c.w.buf, c.w.off, col, 0, y,
                          uint64_t(p0) * c.cout * 4);
            after(enc, "conv3x3");
        }
    }
    enc->setComputePipelineState(pso("bias_rows"));
    enc->setBuffer(y, 0, 0);
    enc->setBuffer(c.b.buf, c.b.off, 1); // within the weight buffers
    enc->setBytes(&c.cout, sizeof(c.cout), 2);
    enc->dispatchThreads(MTL::Size(c.cout, rows, 1), MTL::Size(64, 1, 1));
    after(enc, "bias_rows");
}

void Engine::norm_silu(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, MTL::Buffer* y,
                       uint32_t rows, uint32_t c)
{
    // x / ||x|| * sqrt(c) * g over each pixel's channels: an RMS norm (eps as the reference's 1e-12 on the norm).
    norm(enc, x, &g, y, rows, c, c, 1e-24f / float(c));
    uint32_t n = rows * c;
    dispatch(enc, "silu", {y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void Engine::res_block(MTL::ComputeCommandEncoder*& enc, const Res& r, MTL::Buffer* x, uint32_t h, uint32_t w,
                       MTL::Buffer* y, MTL::Buffer* t1, MTL::Buffer* t2)
{
    const uint32_t rows = h * w, cin = r.c1.cin, cout = r.c2.cout;
    norm_silu(enc, x, r.g1, t1, rows, cin);
    conv(enc, r.c1, t1, h, w, t2);
    norm_silu(enc, t2, r.g2, t2, rows, cout);
    conv(enc, r.c2, t2, h, w, y);
    if (r.shortcut) {
        conv(enc, r.sc, x, h, w, t1);
        add(enc, y, t1, y, rows * cout);
    } else {
        add(enc, y, x, y, rows * cout);
    }
}

std::vector<float> Engine::decode(std::span<const float> latents, uint32_t h, uint32_t w)
{
    if (latents.size() != uint64_t(h) * w * kLatent)
        throw std::runtime_error("latents must be [h * w][64]");
    // The largest image: the last up block's input, at the output's size, and the widest channels at its scale.
    uint64_t most = uint64_t(h) * w * kLatent;
    uint32_t hh = h, ww = w;
    most = std::max(most, uint64_t(hh) * ww * conv_in_.cout);
    for (const Up& u : ups_) {
        most = std::max(most, uint64_t(hh) * ww * std::max(u.cin, u.cout));
        if (u.up) {
            hh *= 2;
            ww *= 2;
            most = std::max(most, uint64_t(hh) * ww * u.cout);
        }
    }
    const uint32_t H = hh, W = ww;
    MTL::Buffer* a = scratch("v_a", most * 4);
    MTL::Buffer* b = scratch("v_b", most * 4);
    MTL::Buffer* t1 = scratch("v_t1", most * 4);
    MTL::Buffer* t2 = scratch("v_t2", most * 4);
    MTL::Buffer* keep = scratch("v_keep", most * 4);
    auto* lat = static_cast<float*>(a->contents());
    for (uint64_t i = 0; i < uint64_t(h) * w; ++i)
        for (uint32_t ch = 0; ch < kLatent; ++ch)
            lat[i * kLatent + ch] = latents[i * kLatent + ch] * lat_std_[ch] + lat_mean_[ch];

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    probe(enc, "vae_in", -1, a, uint64_t(h) * w * kLatent);
    conv(enc, post_quant_, a, h, w, b);
    probe(enc, "vae_pq", -1, b, uint64_t(h) * w * kLatent);
    conv(enc, conv_in_, b, h, w, a);
    probe(enc, "vae_conv_in", -1, a, uint64_t(h) * w * conv_in_.cout);
    hh = h;
    ww = w;
    res_block(enc, mid0_, a, hh, ww, b, t1, t2);
    {
        // Single-head attention over the pixels, channels wide.
        const uint32_t n = hh * ww, c = qkv_.cin;
        norm(enc, b, &attn_g_, t1, n, c, c, 1e-24f / float(c));
        conv(enc, qkv_, t1, hh, ww, t2); // [n][3c]: q, k, v
        MTL::Buffer* kc = scratch("v_k", uint64_t(n) * c * 4);
        MTL::Buffer* vt = scratch("v_vt", uint64_t(n) * c * 4);
        MTL::Buffer* sc = scratch("v_s", uint64_t(n) * n * 4);
        const TransposeArgs ka{n, c, 3 * c, c}, va{n, c, 3 * c, 2 * c};
        dispatch(enc, "copy_cols", {t2, kc}, &ka, sizeof(ka), MTL::Size(c, n, 1), MTL::Size(256, 1, 1));
        dispatch(enc, "transpose_f32", {t2, vt}, &va, sizeof(va), MTL::Size(c, n, 1), MTL::Size(256, 1, 1));
        gemm_->encode(enc, gguf::Type::F32, c, n, n, kc, 0, halves(enc, t2, 0, n, c, 3 * c), 0, sc, 0);
        after(enc, "vae scores");
        const float sa[2] = {float(n), 1.0f / std::sqrt(float(c))};
        dispatch(enc, "softmax_rows", {sc}, sa, sizeof(sa), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1), false);
        gemm_->encode(enc, gguf::Type::F32, n, c, n, vt, 0, halves(enc, sc, 0, n, n, 0), 0, t1, 0);
        after(enc, "vae pv");
        conv(enc, attn_proj_, t1, hh, ww, a);
        add(enc, a, b, b, n * c);
    }
    res_block(enc, mid1_, b, hh, ww, a, t1, t2);
    probe(enc, "vae_mid", -1, a, uint64_t(hh) * ww * mid1_.c2.cout);
    MTL::Buffer* x = a;
    MTL::Buffer* y = b;
    for (const Up& u : ups_) {
        const uint32_t rows = hh * ww;
        if (u.up) { // the block's input, for the duplicating shortcut
            const uint32_t n = rows * u.cin;
            dispatch(enc, "copy_f32", {x, keep}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        }
        for (const Res& r : u.res) {
            res_block(enc, r, x, hh, ww, y, t1, t2);
            std::swap(x, y);
        }
        if (u.up) {
            const UpArgs ua{hh, ww, u.cout, u.cout, u.ft, u.cout * u.ft * 4 / u.cin};
            dispatch(enc, "upsample2x", {x, t1}, &ua, sizeof(ua), MTL::Size(u.cout, 2 * ww, 2 * hh),
                     MTL::Size(64, 1, 1));
            conv(enc, u.conv, t1, 2 * hh, 2 * ww, y);
            const UpArgs da{hh, ww, u.cin, u.cout, u.ft, u.cout * u.ft * 4 / u.cin};
            dispatch(enc, "dup_up_add", {keep, y}, &da, sizeof(da), MTL::Size(u.cout, 2 * ww, 2 * hh),
                     MTL::Size(64, 1, 1));
            std::swap(x, y);
            hh *= 2;
            ww *= 2;
        }
        probe(enc, "vae_up", int(&u - ups_.data()), x, uint64_t(hh) * ww * u.cout);
    }
    norm_silu(enc, x, out_g_, t1, hh * ww, conv_out_.cin);
    conv(enc, conv_out_, t1, hh, ww, y);
    enc->endEncoding();
    finish(cb);
    const auto* o = static_cast<const float*>(y->contents());
    std::vector<float> img(uint64_t(H) * W * 4);
    for (size_t i = 0; i < img.size(); ++i)
        img[i] = std::clamp(o[i], -1.0f, 1.0f);
    return img;
}

} // namespace mako::image
