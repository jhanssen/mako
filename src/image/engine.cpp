#include "image/engine.hpp"

#include "core/log.hpp"
#include "gguf/read_pool.hpp"
#include "image/krea2.hpp"
#include "image/qwen_image.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>

namespace mako::image {

namespace {

struct HeadsArgs {
    uint32_t rows, ld, col0, dim, n_pad, row0, rope;
};
struct HeadsNormArgs {
    uint32_t ld, col0, dim, n_pad, row0, rope;
    float eps;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
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
struct LoraArgs {
    uint32_t k, rows, rank;
    float scale;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

glz::generic read_json_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic j;
    if (glz::read_json(j, text) || !j.is_object())
        throw std::runtime_error(path + ": not a JSON object");
    return j;
}

// Round to nearest even BF16.
uint16_t to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// A tensor's values as floats (F32, F16 or BF16).
std::vector<float> read_floats(const safetensors::File& f, const gguf::Tensor& t)
{
    const uint64_t n = t.elements();
    std::vector<float> out(n);
    std::vector<std::byte> raw(t.bytes);
    gguf::read_exact(f.fd(), raw.data(), t.bytes, t.offset);
    for (uint64_t i = 0; i < n; ++i) {
        if (t.type == gguf::Type::F32) {
            std::memcpy(&out[i], raw.data() + i * 4, 4);
        } else if (t.type == gguf::Type::F16) {
            __fp16 h;
            std::memcpy(&h, raw.data() + i * 2, 2);
            out[i] = float(h);
        } else {
            uint16_t b;
            std::memcpy(&b, raw.data() + i * 2, 2);
            const uint32_t u = uint32_t(b) << 16;
            std::memcpy(&out[i], &u, 4);
        }
    }
    return out;
}

} // namespace

std::unique_ptr<Engine> Engine::open(const std::string& dir, mtl::Context& ctx, const Options& opts)
{
    const auto j = read_json_file(dir + "/model_index.json");
    const std::string cls = j.contains("_class_name") && j["_class_name"].is_string() ? j["_class_name"].get_string() : "";
    if (cls == "QwenImage21Pipeline")
        return std::make_unique<QwenImage>(dir, ctx, opts);
    if (cls == "Krea2Pipeline")
        return std::make_unique<Krea2>(dir, ctx, opts);
    throw std::runtime_error(std::format("{}: an image model mako does not know ({})", dir, cls.empty() ? "?" : cls));
}

Engine::Engine(const std::string& dir, mtl::Context& ctx, const Options& opts)
    : gpu::Runtime(ctx, opts.telemetry), dir_(dir),
      text_files_(open_files(opts.text_encoder.empty() ? dir + "/text_encoder" : opts.text_encoder)),
      vae_files_(open_files(dir + "/vae"))
{
    // The language model's tensors, under the full model's prefix or the bare one's.
    std::string p = "model.language_model.";
    if (!text_files_.find(p + "embed_tokens.weight"))
        p = "language_model.";
    embd_ = text_files_.find(p + "embed_tokens.weight");
    if (!embd_ || embd_.tensor->type != gguf::Type::BF16)
        throw std::runtime_error("text encoder: embed_tokens missing or not BF16");
    tE_ = uint32_t(embd_.tensor->dims[0]);
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
    if (text_[0].q_norm.dims[0] != kHeadDim)
        throw std::runtime_error("text encoder: heads not 128 wide");
    tHeads_ = uint32_t(text_[0].q.dims[1] / kHeadDim);
    tKv_ = uint32_t(text_[0].k.dims[1] / kHeadDim);
    // eps and the rotary base from the encoder's config, where it has one beside the weights.
    const std::string tdir = opts.text_encoder.empty() ? dir + "/text_encoder" : opts.text_encoder;
    if (std::filesystem::exists(tdir + "/config.json")) {
        auto j = read_json_file(tdir + "/config.json");
        const auto& t = j.contains("text_config") ? j["text_config"] : j;
        if (t.contains("rms_norm_eps") && t["rms_norm_eps"].is_number())
            tEps_ = float(t["rms_norm_eps"].get_number());
        if (t.contains("rope_theta") && t["rope_theta"].is_number())
            tTheta_ = float(t["rope_theta"].get_number());
        if (t.contains("rope_parameters") && t["rope_parameters"].contains("rope_theta"))
            tTheta_ = float(t["rope_parameters"]["rope_theta"].get_number());
    }

    load_vae();
    const auto vj = read_json_file(dir + "/vae/config.json");
    if (!vj.contains("latents_mean") || !vj.contains("latents_std"))
        throw std::runtime_error("vae/config.json: no latents_mean / latents_std");
    for (const auto& v : vj["latents_mean"].get_array())
        lat_mean_.push_back(float(v.get_number()));
    for (const auto& v : vj["latents_std"].get_array())
        lat_std_.push_back(float(v.get_number()));
    z_ = uint32_t(post_quant_.cin);
    patch_ = z_ == kLatent ? 1 : z_ * 4 == kLatent ? 2 : 0;
    if (!patch_ || lat_mean_.size() != z_ || lat_std_.size() != z_)
        throw std::runtime_error(std::format("vae: {} latent channels, {} means, {} deviations", z_, lat_mean_.size(),
                                             lat_std_.size()));
    residual_ = vj.contains("is_residual") && vj["is_residual"].is_boolean() && vj["is_residual"].get_boolean();

    const auto sj = read_json_file(dir + "/scheduler/scheduler_config.json");
    auto num = [&](const char* key, double& v) {
        if (sj.contains(key) && sj[key].is_number())
            v = sj[key].get_number();
    };
    num("base_image_seq_len", sched_.base_seq);
    num("max_image_seq_len", sched_.max_seq);
    num("base_shift", sched_.base_shift);
    num("max_shift", sched_.max_shift);
    num("shift_terminal", sched_.terminal);

    const std::string tp = std::filesystem::exists(dir + "/processor/tokenizer.json") ? dir + "/processor/tokenizer.json"
                                                                                       : dir + "/tokenizer/tokenizer.json";
    auto t = text::Tokenizer::load_json(tp);
    if (!t)
        throw std::runtime_error(t.error());
    tok_ = std::move(*t);
}

std::vector<float> Engine::sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const
{
    const double term = terminal ? double(*terminal) : sched_.terminal;
    const double m = (sched_.max_shift - sched_.base_shift) / (sched_.max_seq - sched_.base_seq);
    const double mu = sched_.mu ? *sched_.mu : double(tokens) * m + (sched_.base_shift - m * sched_.base_seq);
    std::vector<float> s(steps);
    for (uint32_t i = 0; i < steps; ++i) {
        // linspace(1, 1 / steps, steps), then exp(mu) / (exp(mu) + 1 / t - 1), in float as the scheduler does.
        const float t = steps == 1 ? 1.0f : float(1.0 + (1.0 / steps - 1.0) * double(i) / double(steps - 1));
        s[i] = float(std::exp(mu) / (std::exp(mu) + (1.0 / double(t) - 1.0)));
    }
    if (term > 0 && steps > 0) {
        const float scale = (1.0f - s.back()) / (1.0f - float(term));
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
    const uint32_t h = r.height / 16, w = r.width / 16, N = h * w, steps = r.steps ? r.steps : default_steps();
    const auto prompt = prompt_embeddings(r.prompt);
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
    const auto sig = sigmas(steps, N, r.terminal);
    begin(prompt, h, w);
    const float scale = r.guidance.value_or(default_guidance());
    if (r.negative)
        begin(prompt_embeddings(*r.negative), h, w, 1);
    for (uint32_t i = 0; i < steps; ++i) {
        auto v = step(x, sig[i]);
        if (r.negative) {
            const auto neg = step(x, sig[i], 1);
            for (size_t j = 0; j < v.size(); ++j)
                v[j] = neg[j] + scale * (v[j] - neg[j]);
        }
        const float dt = sig[i + 1] - sig[i];
        for (size_t j = 0; j < x.size(); ++j)
            x[j] += dt * v[j];
        if (r.progress)
            r.progress(i + 1, steps);
    }
    return decode(x, h, w);
}

Engine::Tensor Engine::upload_rows(const safetensors::Files& files, std::span<const std::string> names)
{
    uint64_t rows = 0, k = 0;
    for (const auto& n : names) {
        const auto r = files.find(n);
        if (!r || (r.tensor->type != gguf::Type::BF16 && r.tensor->type != gguf::Type::F32) ||
            r.tensor->dims.size() != 2 || (k && r.tensor->dims[0] != k))
            throw std::runtime_error(n + ": missing, or not a BF16 or F32 matrix of the others' width");
        k = r.tensor->dims[0];
        rows += r.tensor->dims[1];
    }
    Tensor t = reserve(rows * k * 2, gguf::Type::BF16, {k, rows});
    char* dst = static_cast<char*>(t.buf->contents()) + t.off;
    for (const auto& n : names) {
        const auto r = files.find(n);
        if (r.tensor->type == gguf::Type::BF16) {
            gguf::read_exact(r.file->fd(), dst, r.tensor->bytes, r.tensor->offset);
            dst += r.tensor->bytes;
        } else {
            const auto f = read_floats(*r.file, *r.tensor);
            auto* o = reinterpret_cast<uint16_t*>(dst);
            for (size_t i = 0; i < f.size(); ++i)
                o[i] = to_bf16(f[i]);
            dst += f.size() * 2;
        }
    }
    return t;
}

Engine::Tensor Engine::upload(const safetensors::Files& files, const std::string& name, bool matrix)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    if (matrix && r.tensor->type == gguf::Type::F32 && r.tensor->dims.size() == 2) {
        const std::string n[1] = {name};
        return upload_rows(files, n);
    }
    return upload_raw(r.file->fd(), r.tensor->offset, r.tensor->bytes, r.tensor->type, r.tensor->dims, name, matrix);
}

std::vector<float> Engine::floats(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    return read_floats(*r.file, *r.tensor);
}

Engine::Tensor Engine::upload_plus_one(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    const auto f = read_floats(*r.file, *r.tensor);
    Tensor t = reserve(f.size() * 4, gguf::Type::F32, {f.size()});
    auto* o = reinterpret_cast<float*>(static_cast<char*>(t.buf->contents()) + t.off);
    for (size_t i = 0; i < f.size(); ++i)
        o[i] = f[i] + 1.0f;
    return t;
}

void Engine::embed(std::span<const int32_t> ids, float* out) const
{
    const uint64_t vocab = embd_.tensor->dims[1], row = uint64_t(tE_) * 2;
    std::vector<uint16_t> raw(tE_);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || uint64_t(ids[i]) >= vocab)
            throw std::runtime_error("token out of range");
        gguf::read_exact(embd_.file->fd(), raw.data(), row, embd_.tensor->offset + uint64_t(ids[i]) * row);
        auto* o = reinterpret_cast<uint32_t*>(out + i * tE_);
        for (uint32_t j = 0; j < tE_; ++j)
            o[j] = uint32_t(raw[j]) << 16;
    }
}

void Engine::add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n)
{
    dispatch(enc, "add", {x, z, y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void Engine::unary(MTL::ComputeCommandEncoder*& enc, const char* fn, MTL::Buffer* x, uint32_t n)
{
    dispatch(enc, fn, {x}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void Engine::bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n, uint32_t rows,
                  uint64_t y_off)
{
    enc->setComputePipelineState(pso("bias_rows"));
    enc->setBuffer(y, y_off, 0);
    enc->setBuffer(b.buf, b.off, 1);
    enc->setBytes(&n, sizeof(n), 2);
    enc->dispatchThreads(MTL::Size(n, rows, 1), MTL::Size(64, 1, 1));
    after(enc, "bias_rows");
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

void Engine::heads_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t n_heads,
                        uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint64_t cs_off, uint32_t rope, const Tensor& w,
                        float eps, uint32_t row0, MTL::Buffer* y)
{
    const HeadsNormArgs a{ld, col0, kHeadDim, n_pad, row0, rope, eps};
    enc->setComputePipelineState(pso("heads_norm_half"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(cs ? cs : x, cs ? cs_off : 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBuffer(w.buf, w.off, 3);
    enc->setBytes(&a, sizeof(a), 4);
    enc->dispatchThreadgroups(MTL::Size(T, n_heads, 1), MTL::Size(kHeadDim / 2, 1, 1));
    after(enc, "heads_norm_half");
}

void Engine::attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v,
                       MTL::Buffer* o, uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t n_heads,
                       uint32_t kv_heads, uint32_t q_pos0, uint32_t block)
{
    const AttnArgs a{n_q, q_pad, n_k, k_pad, n_heads, kv_heads, q_pos0, block, n_heads * kHeadDim,
                     1.0f / std::sqrt(float(kHeadDim))};
    dispatch(enc, "flash_attn_sg_128_16x32_4", {q, k, v, o}, &a, sizeof(a),
             MTL::Size(q_pad / kAttnRows, n_heads, 1), MTL::Size(128, 1, 1), false);
}

std::vector<float> Engine::encode_text(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps)
{
    const uint32_t T = uint32_t(ids.size()), E = tE_, kv = tKv_ * kHeadDim, qw = tHeads_ * kHeadDim;
    if (T <= drop)
        throw std::runtime_error("the prompt is all template");
    uint32_t layers = uint32_t(text_.size());
    if (!taps.empty()) {
        layers = *std::max_element(taps.begin(), taps.end());
        if (layers > text_.size())
            throw std::runtime_error("text encoder: a tapped layer past its last");
    }
    const uint32_t nt = taps.empty() ? 1 : uint32_t(taps.size());
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    MTL::Buffer* x = scratch("t_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("t_xn", uint64_t(T) * E * 4);
    MTL::Buffer* q = scratch("t_q", uint64_t(T) * qw * 4);
    MTL::Buffer* k = scratch("t_k", uint64_t(T) * kv * 4);
    MTL::Buffer* v = scratch("t_v", uint64_t(T) * kv * 4);
    MTL::Buffer* qh = scratch("t_qh", uint64_t(tHeads_) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("t_kh", uint64_t(tKv_) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("t_vh", uint64_t(tKv_) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("t_att", uint64_t(T) * qw * 4);
    MTL::Buffer* y = scratch("t_y", uint64_t(T) * E * 4);
    MTL::Buffer* cs = scratch("t_cs", uint64_t(T) * kHeadDim / 2 * 8);
    MTL::Buffer* out = scratch("t_out", uint64_t(T) * nt * E * 4);
    embed(ids, static_cast<float*>(x->contents()));
    // Text positions are the same on the three rotary axes, which makes the multi-axis rotation the plain one.
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = 0; i < kHeadDim / 2; ++i) {
            const float inv = 1.0f / std::pow(tTheta_, float(2 * i) / float(kHeadDim));
            const float ang = float(t) * inv;
            c[(uint64_t(t) * kHeadDim / 2 + i) * 2] = std::cos(ang);
            c[(uint64_t(t) * kHeadDim / 2 + i) * 2 + 1] = std::sin(ang);
        }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // Hidden state i (0: the embeddings) into its column of the taps' rows.
    auto tap = [&](uint32_t i) {
        for (uint32_t j = 0; j < taps.size(); ++j)
            if (taps[j] == i) {
                const uint32_t a[2] = {E, nt * E};
                enc->setComputePipelineState(pso("copy_rows"));
                enc->setBuffer(x, 0, 0);
                enc->setBuffer(out, uint64_t(j) * E * 4, 1);
                enc->setBytes(a, sizeof(a), 2);
                enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
                after(enc, "copy_rows");
            }
    };
    tap(0);
    for (uint32_t il = 0; il < layers; ++il) {
        const TextLayer& L = text_[il];
        cur_layer_ = il;
        norm(enc, x, &L.attn_norm, xn, T, E, E, tEps_);
        proj(enc, L.q, xn, 0, q, 0, T);
        proj(enc, L.k, xn, 0, k, 0, T);
        proj(enc, L.v, xn, 0, v, 0, T);
        norm(enc, q, &L.q_norm, q, T * tHeads_, kHeadDim, kHeadDim, tEps_);
        norm(enc, k, &L.k_norm, k, T * tKv_, kHeadDim, kHeadDim, tEps_);
        heads(enc, q, qw, 0, tHeads_, T, q_pad, cs, 1, qh);
        heads(enc, k, kv, 0, tKv_, T, k_pad, cs, 1, kh);
        heads(enc, v, kv, 0, tKv_, T, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, tHeads_, tKv_, 0, UINT32_MAX);
        proj(enc, L.o, att, 0, y, 0, T);
        add(enc, x, y, x, T * E);
        norm(enc, x, &L.ffn_norm, xn, T, E, E, tEps_);
        ffn_dense(enc, L.gate, L.up, L.down, xn, y, 0.0f, T);
        add(enc, x, y, x, T * E);
        tap(il + 1);
    }
    enc->endEncoding();
    finish(cb);
    const auto* src = static_cast<const float*>((taps.empty() ? x : out)->contents());
    return std::vector<float>(src + uint64_t(drop) * nt * E, src + uint64_t(T) * nt * E);
}

Engine::Conv Engine::load_conv(const std::string& name, uint32_t k)
{
    const auto r = vae_files_.find(name + ".weight");
    const auto rb = vae_files_.find(name + ".bias");
    const size_t nd = r ? r.tensor->dims.size() : 0;
    if (!r || !rb || r.tensor->type != gguf::Type::F32 || (nd != 4 && nd != 5))
        throw std::runtime_error(name + ": missing, or not an F32 convolution");
    Conv c;
    c.k = k;
    c.cout = uint32_t(r.tensor->dims[nd - 1]);
    c.cin = uint32_t(r.tensor->dims[nd - 2]);
    // A causal 3D convolution over one frame sees only its last time tap (the earlier ones read the zero padding).
    const uint32_t kt = nd == 5 ? uint32_t(r.tensor->dims[2]) : 1;
    if (r.tensor->dims[0] != k || r.tensor->dims[1] != k)
        throw std::runtime_error(name + ": unexpected kernel size");
    const uint32_t cols = c.cin * k * k, kk = k * k;
    // 3x3: the unfolded rows padded for the direct GEMM; 1x1: the input's own rows (the staged GEMM where they are not
    // a multiple of 64).
    c.kpad = round_up(cols, k == 3 ? 64 : 32);
    // Half, as the activations it multiplies are (the direct GEMM reads it unstaged).
    c.w = reserve(uint64_t(c.cout) * c.kpad * 2, gguf::Type::F16, {c.kpad, c.cout});
    auto* dst = reinterpret_cast<__fp16*>(static_cast<char*>(c.w.buf->contents()) + c.w.off);
    std::vector<float> row(uint64_t(cols) * kt);
    for (uint32_t o = 0; o < c.cout; ++o) {
        gguf::read_exact(r.file->fd(), row.data(), row.size() * 4, r.tensor->offset + uint64_t(o) * row.size() * 4);
        for (uint32_t i = 0; i < c.kpad; ++i) {
            float v = 0.0f;
            if (i < cols)
                v = row[(uint64_t(i / kk) * kt + kt - 1) * kk + i % kk];
            dst[uint64_t(o) * c.kpad + i] = __fp16(v);
        }
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
        for (const char* up : {".upsampler", ".upsamplers.0"})
            if (vae_files_.find(p + up + ".resample.1.weight")) {
                u.up = true;
                u.conv = load_conv(p + up + ".resample.1", 3);
                // Blocks that also upsample in time (a time convolution) keep the second of two time copies.
                u.ft = vae_files_.find(p + up + ".time_conv.weight") ? 2 : 1;
            }
        ups_.push_back(std::move(u));
    }
    out_g_ = upload(vae_files_, "decoder.norm_out.gamma", false);
    conv_out_ = load_conv("decoder.conv_out", 3);
}

void Engine::conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t h, uint32_t w,
                  MTL::Buffer* y)
{
    const uint32_t rows = h * w;
    if (c.k == 1) {
        MTL::Buffer* in = x;
        if (c.kpad != c.cin) { // rows zero padded to the GEMM's
            in = scratch("v_pad", uint64_t(rows) * c.kpad * 4);
            std::memset(in->contents(), 0, uint64_t(rows) * c.kpad * 4);
            const uint32_t a[2] = {c.cin, c.kpad};
            dispatch(enc, "copy_rows", {x, in}, a, sizeof(a), MTL::Size(c.cin, rows, 1), MTL::Size(256, 1, 1));
        }
        gemm_->encode(enc, gguf::Type::F16, c.kpad, c.cout, rows, c.w.buf, c.w.off,
                      halves(enc, in, 0, rows, c.kpad, 0), 0, y, 0, 0, 0, c.kpad % 64 == 0);
        after(enc, "conv1x1");
    } else {
        // Image rows a pass, so the unfolded copy stays near 256 MB, in multiples of 256 (the GEMM's largest tiles).
        const uint32_t chunk = std::max(256u, uint32_t((256ull << 20) / (uint64_t(c.kpad) * 2)) / 256 * 256);
        MTL::Buffer* col = scratch("v_col", uint64_t(std::min(chunk, rows)) * c.kpad * 2);
        for (uint32_t p0 = 0; p0 < rows; p0 += chunk) {
            const uint32_t n = std::min(chunk, rows - p0);
            const Im2colArgs a{h, w, c.cin, p0, n, c.kpad};
            dispatch(enc, "im2col3x3", {x, col}, &a, sizeof(a), MTL::Size(c.kpad, n, 1), MTL::Size(256, 1, 1));
            gemm_->encode(enc, gguf::Type::F16, c.kpad, c.cout, n, c.w.buf, c.w.off, col, 0, y,
                          uint64_t(p0) * c.cout * 4, 0, 0, true);
            after(enc, "conv3x3");
        }
    }
    bias(enc, y, c.b, c.cout, rows);
}

void Engine::norm_silu(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, MTL::Buffer* y,
                       uint32_t rows, uint32_t c)
{
    // x / ||x|| * sqrt(c) * g over each pixel's channels: an RMS norm (eps as the reference's 1e-12 on the norm).
    norm(enc, x, &g, y, rows, c, c, 1e-24f / float(c));
    unary(enc, "silu", y, rows * c);
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
    // The VAE's own latent image: a token's 2x2 patch spread over pixels where it packs one.
    const uint32_t lh = h * patch_, lw = w * patch_;
    // The largest image: the last up block's input, at the output's size, and the widest channels at its scale.
    uint64_t most = uint64_t(lh) * lw * z_;
    uint32_t hh = lh, ww = lw;
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
    if (H != 16 * h || W != 16 * w)
        throw std::runtime_error("vae: not a 16x decoder for its tokens");
    MTL::Buffer* a = scratch("v_a", most * 4);
    MTL::Buffer* b = scratch("v_b", most * 4);
    MTL::Buffer* t1 = scratch("v_t1", most * 4);
    MTL::Buffer* t2 = scratch("v_t2", most * 4);
    MTL::Buffer* keep = scratch("v_keep", most * 4);
    auto* lat = static_cast<float*>(a->contents());
    const uint32_t p = patch_;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (uint32_t ch = 0; ch < z_; ++ch)
                for (uint32_t py = 0; py < p; ++py)
                    for (uint32_t px = 0; px < p; ++px) {
                        const float v = latents[(uint64_t(y) * w + x) * kLatent + (ch * p + py) * p + px];
                        lat[((uint64_t(y) * p + py) * lw + x * p + px) * z_ + ch] = v * lat_std_[ch] + lat_mean_[ch];
                    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    probe(enc, "vae_in", -1, a, uint64_t(lh) * lw * z_);
    conv(enc, post_quant_, a, lh, lw, b);
    probe(enc, "vae_pq", -1, b, uint64_t(lh) * lw * z_);
    conv(enc, conv_in_, b, lh, lw, a);
    probe(enc, "vae_conv_in", -1, a, uint64_t(lh) * lw * conv_in_.cout);
    hh = lh;
    ww = lw;
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
        if (u.up && residual_) { // the block's input, for the duplicating shortcut
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
            if (residual_) {
                const UpArgs da{hh, ww, u.cin, u.cout, u.ft, u.cout * u.ft * 4 / u.cin};
                dispatch(enc, "dup_up_add", {keep, y}, &da, sizeof(da), MTL::Size(u.cout, 2 * ww, 2 * hh),
                         MTL::Size(64, 1, 1));
            }
            std::swap(x, y);
            hh *= 2;
            ww *= 2;
        }
        probe(enc, "vae_up", int(&u - ups_.data()), x, uint64_t(hh) * ww * (u.up ? u.conv.cout : u.cout));
    }
    norm_silu(enc, x, out_g_, t1, hh * ww, conv_out_.cin);
    conv(enc, conv_out_, t1, hh, ww, y);
    enc->endEncoding();
    finish(cb);
    // RGB (opaque) or RGBA.
    const auto* o = static_cast<const float*>(y->contents());
    const uint32_t oc = conv_out_.cout;
    std::vector<float> img(uint64_t(H) * W * 4);
    for (uint64_t px = 0; px < uint64_t(H) * W; ++px)
        for (uint32_t ch = 0; ch < 4; ++ch)
            img[px * 4 + ch] = ch < oc ? std::clamp(o[px * oc + ch], -1.0f, 1.0f) : 1.0f;
    return img;
}

uint32_t Engine::add_lora(const std::string& path, float scale)
{
    auto file = safetensors::File::open(path);
    if (!file)
        throw std::runtime_error(file.error());
    // The adapter's alpha for every matrix, from a PEFT config in the metadata.
    std::optional<double> alpha;
    if (const auto it = file->metadata().find("lora_adapter_metadata"); it != file->metadata().end()) {
        glz::generic j;
        if (!glz::read_json(j, it->second) && j.is_object())
            for (const char* k : {"transformer.lora_alpha", "lora_alpha"})
                if (j.contains(k) && j[k].is_number())
                    alpha = j[k].get_number();
    }
    struct Pair {
        const gguf::Tensor *a = nullptr, *b = nullptr, *alpha = nullptr;
    };
    std::map<std::string, Pair> mods;
    std::vector<std::string> unknown;
    static constexpr std::pair<const char*, int> kSuffixes[] = {
        {".lora.down.weight", 0}, {".lora.up.weight", 1},  {".lora_down.weight", 0},       {".lora_up.weight", 1},
        {".lora_A.weight", 0},    {".lora_B.weight", 1},   {".lora_A.default.weight", 0}, {".lora_B.default.weight", 1},
        {".alpha", 2}};
    for (const auto& t : file->tensors()) {
        std::string_view name = t.name;
        for (const char* p : {"base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."})
            if (name.starts_with(p))
                name.remove_prefix(std::strlen(p));
        bool known = false;
        for (const auto& [suffix, role] : kSuffixes)
            if (name.ends_with(suffix)) {
                Pair& p = mods[std::string(name.substr(0, name.size() - std::strlen(suffix)))];
                (role == 0 ? p.a : role == 1 ? p.b : p.alpha) = &t;
                known = true;
                break;
            }
        if (!known)
            unknown.push_back(t.name);
    }
    for (const auto& [m, p] : mods)
        if (!p.a || !p.b || !lora_target(m).w)
            unknown.push_back(m);
    if (!unknown.empty())
        throw std::runtime_error(std::format("{}: {} keys {} does not adapt, e.g. {}", path, unknown.size(), name(),
                                             unknown.front()));
    uint32_t done = 0, rank_seen = 0;
    for (const auto& [m, p] : mods) {
        const LoraTarget tg = lora_target(m);
        const uint32_t k = uint32_t(tg.w->dims[0]), rank = uint32_t(p.a->dims[1]);
        // A is [rank][k] and B [rows][rank] (row-major), so their contiguous dimensions come first here.
        if (p.a->dims[0] != k || p.b->dims[0] != rank || p.b->dims[1] != tg.rows || tg.w->type != gguf::Type::BF16)
            throw std::runtime_error(std::format("{}: {} does not fit the weight it adapts", path, m));
        double a = alpha.value_or(double(rank));
        if (p.alpha)
            a = read_floats(*file, *p.alpha).front();
        const auto av = read_floats(*file, *p.a), bv = read_floats(*file, *p.b);
        MTL::Buffer* ab = scratch("l_a", av.size() * 4);
        MTL::Buffer* bb = scratch("l_b", bv.size() * 4);
        std::memcpy(ab->contents(), av.data(), av.size() * 4);
        std::memcpy(bb->contents(), bv.data(), bv.size() * 4);
        const LoraArgs la{k, tg.rows, rank, float(double(scale) * a / double(rank))};
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
        enc->setComputePipelineState(pso("lora_merge"));
        enc->setBuffer(tg.w->buf, tg.w->off + uint64_t(tg.row0) * k * 2, 0);
        enc->setBuffer(ab, 0, 1);
        enc->setBuffer(bb, 0, 2);
        enc->setBytes(&la, sizeof(la), 3);
        enc->dispatchThreads(MTL::Size(k, tg.rows, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        finish(cb);
        ++done;
        rank_seen = rank;
    }
    if (!done)
        throw std::runtime_error(path + ": no LoRA weights in it");
    log::info("LoRA {}: {} matrices adapted (rank {}, alpha {}), scale {}", path, done, rank_seen,
              alpha ? std::format("{}", *alpha) : std::string("= rank"), scale);
    return done;
}

std::pair<std::string, float> parse_lora(const std::string& spec)
{
    const size_t colon = spec.rfind(':');
    if (colon != std::string::npos && colon + 1 < spec.size()) {
        char* end = nullptr;
        const float v = std::strtof(spec.c_str() + colon + 1, &end);
        if (end && *end == '\0')
            return {spec.substr(0, colon), v};
    }
    if (spec.empty())
        throw std::invalid_argument("names no LoRA file");
    return {spec, 1.0f};
}

} // namespace mako::image
