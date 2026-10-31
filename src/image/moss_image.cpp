#include "image/moss_image.hpp"

#include "core/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::image {

namespace {

struct MossModArgs {
    uint32_t n, blocks;
};
struct RmsAffineArgs {
    uint32_t n;
    float eps;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

// A linear grid in float as the reference builds it: from the start for the first half, from the end for the rest.
std::vector<float> linspace(float start, float end, uint32_t n)
{
    std::vector<float> v(n);
    if (n == 1) {
        v[0] = start;
        return v;
    }
    const float step = (end - start) / float(n - 1);
    for (uint32_t i = 0; i < n; ++i)
        v[i] = i < n / 2 ? start + step * float(i) : end - step * float(n - 1 - i);
    return v;
}

// shift * v / (1 + (shift - 1) v) over linspace(1, shift / (1000 + shift - 1), n).
std::vector<float> shifted_grid(uint32_t n, float shift)
{
    auto v = linspace(1.0f, float(double(shift) / (1000.0 + double(shift) - 1.0)), n);
    for (float& x : v)
        x = shift * x / (1.0f + (shift - 1.0f) * x);
    return v;
}

} // namespace

MossImage::MossImage(const std::string& dir, mtl::Context& ctx, const Options& opts) : Engine(dir, ctx, opts)
{
    if (text_width() != kText)
        throw std::runtime_error("text encoder: not 4096 wide");
    if (latent_width() != kLatentWidth)
        throw std::runtime_error("vae: tokens not 128 values");
    load_transformer(open_files(opts.transformer.empty() ? dir + "/transformer" : opts.transformer));
    // Frequencies as the reference derives them (its stored copy may be rounded).
    for (uint32_t i = 0; i < 16; ++i)
        bands_[i] = 1.0f / std::pow(10000.0f, float(i) / 16.0f);
}

std::string MossImage::templated(const std::string& prompt)
{
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
}

void MossImage::load_transformer(const safetensors::Files& f)
{
    if (!f.find("video_embedder.weight") || !f.find("shared_block_modulation.weight"))
        throw std::runtime_error("transformer: not MOSS-Image's weights");
    auto mat = [&](const std::string& n) { return upload(f, n, true); };
    auto vec = [&](const std::string& n) { return upload(f, n, false); };
    // Biases of fused matrices, one after another.
    auto biases = [&](std::initializer_list<std::string> names) {
        std::vector<float> all;
        for (const auto& n : names) {
            const auto b = floats(f, n);
            all.insert(all.end(), b.begin(), b.end());
        }
        Tensor t = reserve(all.size() * 4, gguf::Type::F32, {all.size()});
        std::memcpy(static_cast<char*>(t.buf->contents()) + t.off, all.data(), all.size() * 4);
        return t;
    };
    img_in_ = mat("video_embedder.weight");
    img_in_b_ = vec("video_embedder.bias");
    txt_proj_ = mat("text_embedder.proj.weight");
    txt_proj_b_ = vec("text_embedder.proj.bias");
    for (uint32_t i = 0; f.find(std::format("text_embedder.blocks.{}.qkv.weight", i)); ++i) {
        const std::string b = std::format("text_embedder.blocks.{}.", i);
        TextBlock t;
        t.norm1 = vec(b + "norm1.weight");
        t.norm2 = vec(b + "norm2.weight");
        t.qkv = mat(b + "qkv.weight");
        t.qkv_b = vec(b + "qkv.bias");
        t.norm_q = vec(b + "q_norm.weight");
        t.norm_k = vec(b + "k_norm.weight");
        t.o = mat(b + "proj.weight");
        t.o_b = vec(b + "proj.bias");
        // The activation's input, then its gate, as the blocks' fused matrix has them.
        const std::string ug[2] = {b + "ffn_up.weight", b + "ffn_gate.weight"};
        t.gate_up = upload_rows(f, ug);
        t.gate_up_b = biases({b + "ffn_up.bias", b + "ffn_gate.bias"});
        t.down = mat(b + "ffn_down.weight");
        t.down_b = vec(b + "ffn_down.bias");
        text_blocks_.push_back(std::move(t));
    }
    t1_ = mat("time_embedding.0.weight");
    t1_b_ = vec("time_embedding.0.bias");
    t2_ = mat("time_embedding.2.weight");
    t2_b_ = vec("time_embedding.2.bias");
    mod_ = mat("shared_block_modulation.weight");
    mod_b_ = vec("shared_block_modulation.bias");
    head_mod_ = mat("video_head_modulation.weight");
    head_mod_b_ = vec("video_head_modulation.bias");
    for (uint32_t i = 0; f.find(std::format("layers.{}.proj.q.weight", i)); ++i) {
        const std::string b = std::format("layers.{}.", i);
        Block B;
        const std::string qkv[3] = {b + "proj.q.weight", b + "proj.k.weight", b + "proj.v.weight"};
        B.q = upload_rows(f, qkv);
        B.o = mat(b + "proj.o.weight");
        B.norm_q = vec(b + "q_norm.weight");
        B.norm_k = vec(b + "k_norm.weight");
        B.up_gate = mat(b + "proj.ffn_up_gate.weight");
        B.down = mat(b + "proj.ffn_down.weight");
        blocks_.push_back(std::move(B));
    }
    if (blocks_.size() <= kSkip || text_blocks_.empty())
        throw std::runtime_error(std::format("transformer: {} blocks, {} text blocks", blocks_.size(),
                                             text_blocks_.size()));
    const uint32_t nb = uint32_t(blocks_.size());
    norms_ = reserve(uint64_t(nb) * 2 * kDim * 4, gguf::Type::F32, {2 * kDim, nb});
    auto* nrm = reinterpret_cast<float*>(static_cast<char*>(norms_.buf->contents()) + norms_.off);
    for (uint32_t b = 0; b < nb; ++b)
        for (int k = 0; k < 2; ++k) {
            const auto s = floats(f, std::format("layers.{}.norm{}.weight", b, k + 1));
            std::copy(s.begin(), s.end(), nrm + (uint64_t(b) * 2 + k) * kDim);
        }
    out_norm_ = vec("final_norm_video.weight");
    out_ = mat("final_linear_video.weight");
    log::info("MOSS-Image transformer: {} blocks, {} text blocks", nb, text_blocks_.size());
}

std::vector<float> MossImage::prompt_embeddings(const std::string& prompt)
{
    auto ids = tok_->encode(templated(prompt), true);
    if (ids.size() > 4096) // the reference's longest
        ids.resize(4096);
    return encode_text(ids, 0, {}, true);
}

std::vector<float> MossImage::sigmas(uint32_t steps, uint32_t, std::optional<float>) const
{
    constexpr uint32_t kTotal = 1000;
    const auto s = shifted_grid(steps, kShift);
    auto training = shifted_grid(kTotal, 1.0f);
    std::reverse(training.begin(), training.end());
    std::vector<float> out;
    for (float v : s) {
        const uint32_t upper =
            std::min(uint32_t(std::lower_bound(training.begin(), training.end(), v) - training.begin()), kTotal - 1);
        const uint32_t lower = upper ? upper - 1 : 0;
        const float snapped = std::abs(v - training[lower]) <= std::abs(training[upper] - v) ? training[lower]
                                                                                             : training[upper];
        if (out.empty() || out.back() != snapped)
            out.push_back(snapped);
    }
    out.push_back(0.0f);
    return out;
}

void MossImage::rope_table(MTL::Buffer* cs, uint32_t S, uint32_t h, uint32_t w) const
{
    // Pair i rotates by coordinate i / 16 times band i % 16; coordinates (r, f, y, x).
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < h * w + S; ++t) {
        float pos[4] = {0, 0, 0, 0};
        if (t < h * w) {
            pos[1] = float(S);
            pos[2] = float(t / w) - float(h - 1) / 2.0f;
            pos[3] = float(t % w) - float(w - 1) / 2.0f;
        } else {
            pos[1] = float(t - h * w);
        }
        for (uint32_t i = 0; i < kHeadDim / 2; ++i) {
            const float ang = pos[i / 16] * bands_[i % 16];
            c[(uint64_t(t) * kHeadDim / 2 + i) * 2] = std::cos(ang);
            c[(uint64_t(t) * kHeadDim / 2 + i) * 2 + 1] = std::sin(ang);
        }
    }
}

void MossImage::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    const uint32_t E = kText, D = kDim;
    if (prompt.empty() || prompt.size() % E)
        throw std::runtime_error("prompt states must be rows of 4096");
    const uint32_t S = uint32_t(prompt.size() / E), N = h * w;
    S_[slot] = S;
    gh_ = h;
    gw_ = w;
    const uint32_t q_pad = round_up(S, kAttnRows), k_pad = round_up(S, 128);
    const uint64_t nff = text_blocks_.front().down.dims[0];
    MTL::Buffer* in = scratch("m_in", prompt.size() * 4);
    MTL::Buffer* x = scratch(std::format("k_txt{}", slot), uint64_t(S) * D * 4);
    MTL::Buffer* xn = scratch("m_xn", uint64_t(S) * D * 4);
    MTL::Buffer* qkv = scratch("m_qkv", uint64_t(S) * 3 * D * 4);
    MTL::Buffer* qh = scratch("m_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("m_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("m_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("m_att", uint64_t(S) * D * 4);
    MTL::Buffer* y = scratch("m_y", uint64_t(S) * D * 4);
    MTL::Buffer* gu = scratch("m_gu", uint64_t(S) * 2 * nff * 4);
    MTL::Buffer* ffm = scratch("m_ffm", uint64_t(S) * nff * 4);
    MTL::Buffer* cs = scratch(std::format("k_cs{}", slot), uint64_t(N + S) * kHeadDim / 2 * 8);
    rope_table(cs, S, h, w);
    std::memcpy(in->contents(), prompt.data(), prompt.size() * 4);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, txt_proj_, in, 0, x, 0, S);
    bias(enc, x, txt_proj_b_, D, S);
    const uint64_t cs_text = uint64_t(N) * kHeadDim / 2 * 8;
    for (const TextBlock& B : text_blocks_) {
        norm(enc, x, &B.norm1, xn, S, D, D, kEps);
        proj(enc, B.qkv, xn, 0, qkv, 0, S);
        bias(enc, qkv, B.qkv_b, 3 * D, S);
        heads_norm(enc, qkv, 3 * D, 0, kHeads, S, q_pad, cs, cs_text, 1, B.norm_q, kEps, 0, qh);
        heads_norm(enc, qkv, 3 * D, D, kHeads, S, k_pad, cs, cs_text, 1, B.norm_k, kEps, 0, kh);
        if (k_pad > S) {
            const uint32_t c[4] = {kHeadDim, k_pad, S, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - S, kHeads),
                     MTL::Size(128, 1, 1));
        }
        heads(enc, qkv, 3 * D, 2 * D, kHeads, S, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, S, q_pad, S, k_pad, kHeads, kHeads, 0, 0);
        proj(enc, B.o, att, 0, y, 0, S);
        bias(enc, y, B.o_b, D, S);
        add(enc, x, y, x, S * D);
        norm(enc, x, &B.norm2, xn, S, D, D, kEps);
        proj(enc, B.gate_up, xn, 0, gu, 0, S);
        bias(enc, gu, B.gate_up_b, uint32_t(2 * nff), S);
        uint32_t f = uint32_t(nff);
        dispatch(enc, "situ_rows", {gu, ffm}, &f, sizeof(f), MTL::Size(f, S, 1), MTL::Size(256, 1, 1));
        proj(enc, B.down, ffm, 0, y, 0, S);
        bias(enc, y, B.down_b, D, S);
        add(enc, x, y, x, S * D);
    }
    probe(enc, "text_emb", -1, x, uint64_t(S) * D);
    enc->endEncoding();
    finish(cb);
}

std::vector<float> MossImage::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    const uint32_t N = gh_ * gw_, D = kDim, S = S_[slot], T = N + S, nb = uint32_t(blocks_.size());
    if (latents.size() != uint64_t(N) * kLatentWidth)
        throw std::runtime_error("latents must be [h * w][128]");
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    // The GEMMs run a multiple of 256 rows (the rest ignored), which keeps them on their largest tiles.
    const uint32_t Tg = T > 256 ? round_up(T, 256) : T;
    const uint64_t nff = blocks_.front().down.dims[0];
    MTL::Buffer* x = scratch("d_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkv = scratch("d_qkv", uint64_t(Tg) * 3 * D * 4);
    MTL::Buffer* gu = scratch("d_gu", uint64_t(Tg) * 2 * nff * 4);
    MTL::Buffer* ffm = scratch("d_ffm", uint64_t(Tg) * nff * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(Tg) * D * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* lat = scratch("d_lat", uint64_t(N) * kLatentWidth * 4);
    MTL::Buffer* out = scratch("d_out", uint64_t(N) * kLatentWidth * 4);
    MTL::Buffer* tp = scratch("d_tproj", kTimeDim * 4);
    MTL::Buffer* temb = scratch("d_temb", D * 4);
    MTL::Buffer* tg = scratch("d_tg", D * 4);
    MTL::Buffer* tm = scratch("d_tm", 6 * D * 4);
    MTL::Buffer* hm = scratch("d_hm", 2 * D * 4);
    MTL::Buffer* mods = scratch("d_mods", uint64_t(nb) * 6 * D * 4);
    MTL::Buffer* fmod = scratch("d_fmod", 2 * D * 4);
    MTL::Buffer* txt = scratch(std::format("k_txt{}", slot), uint64_t(S) * D * 4);
    MTL::Buffer* cs = scratch(std::format("k_cs{}", slot), uint64_t(T) * kHeadDim / 2 * 8);
    std::memcpy(lat->contents(), latents.data(), latents.size() * 4);
    std::memcpy(static_cast<char*>(x->contents()) + uint64_t(N) * D * 4, txt->contents(), uint64_t(S) * D * 4);
    // Sinusoidal embedding (cos half first) of the timestep, in double as the reference computes it.
    const double t = double(sigma * 1000.0f);
    auto* tpf = static_cast<float*>(tp->contents());
    for (uint32_t i = 0; i < kTimeDim / 2; ++i) {
        const double arg = t * std::pow(10000.0, -double(i) / double(kTimeDim / 2));
        tpf[i] = float(std::cos(arg));
        tpf[kTimeDim / 2 + i] = float(std::sin(arg));
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, t1_, tp, 0, temb, 0, 1);
    bias(enc, temb, t1_b_, D, 1);
    unary(enc, "silu", temb, D);
    proj(enc, t2_, temb, 0, tg, 0, 1);
    bias(enc, tg, t2_b_, D, 1);
    proj(enc, mod_, tg, 0, tm, 0, 1);
    bias(enc, tm, mod_b_, 6 * D, 1);
    probe(enc, "mod", -1, tm, 6 * D);
    proj(enc, head_mod_, tg, 0, hm, 0, 1);
    bias(enc, hm, head_mod_b_, 2 * D, 1);
    const MossModArgs ma{D, nb};
    enc->setComputePipelineState(pso("moss_mod"));
    enc->setBuffer(tm, 0, 0);
    enc->setBuffer(norms_.buf, norms_.off, 1);
    enc->setBuffer(mods, 0, 2);
    enc->setBytes(&ma, sizeof(ma), 3);
    enc->dispatchThreads(MTL::Size(D, nb, 1), MTL::Size(256, 1, 1));
    after(enc, "moss_mod");
    {
        enc->setComputePipelineState(pso("moss_head_mod"));
        enc->setBuffer(hm, 0, 0);
        enc->setBuffer(out_norm_.buf, out_norm_.off, 1);
        enc->setBuffer(fmod, 0, 2);
        enc->setBytes(&D, sizeof(D), 3);
        enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "moss_head_mod");
    }
    proj(enc, img_in_, lat, 0, x, 0, N);
    bias(enc, x, img_in_b_, D, N);
    probe(enc, "img_in", -1, x, uint64_t(N) * D);

    // y = x / rms(x) * scale + shift over `rows` rows, with scale, shift at m offsets (in floats).
    auto rms = [&](MTL::Buffer* m, uint64_t s_off, uint64_t b_off, uint32_t rows) {
        const RmsAffineArgs a{D, kEps};
        enc->setComputePipelineState(pso("rms_affine"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(m, s_off * 4, 1);
        enc->setBuffer(m, b_off * 4, 2);
        enc->setBuffer(xn, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "rms_affine");
    };
    auto gated = [&](uint64_t g_off) {
        enc->setComputePipelineState(pso("scale_add"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(y, 0, 1);
        enc->setBuffer(mods, g_off * 4, 2);
        enc->setBytes(&D, sizeof(D), 3);
        enc->dispatchThreads(MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        after(enc, "scale_add");
    };
    for (uint32_t il = 0; il < nb; ++il) {
        if (slot == 1 && il == kSkip)
            continue;
        const Block& B = blocks_[il];
        cur_layer_ = il;
        const uint64_t mo = uint64_t(il) * 6 * D;
        rms(mods, mo, mo + D, T);
        proj(enc, B.q, xn, 0, qkv, 0, Tg);
        heads_norm(enc, qkv, 3 * D, 0, kHeads, T, q_pad, cs, 0, 1, B.norm_q, kEps, 0, qh);
        heads_norm(enc, qkv, 3 * D, D, kHeads, T, k_pad, cs, 0, 1, B.norm_k, kEps, 0, kh);
        if (k_pad > T) {
            const uint32_t c[4] = {kHeadDim, k_pad, T, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - T, kHeads),
                     MTL::Size(128, 1, 1));
        }
        heads(enc, qkv, 3 * D, 2 * D, kHeads, T, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, kHeads, kHeads, 0, 0);
        proj(enc, B.o, att, 0, y, 0, Tg);
        gated(mo + 2 * D);
        rms(mods, mo + 3 * D, mo + 4 * D, T);
        proj(enc, B.up_gate, xn, 0, gu, 0, Tg);
        uint32_t f = uint32_t(nff);
        dispatch(enc, "situ_rows", {gu, ffm}, &f, sizeof(f), MTL::Size(f, Tg, 1), MTL::Size(256, 1, 1));
        proj(enc, B.down, ffm, 0, y, 0, Tg);
        gated(mo + 5 * D);
        probe(enc, "block", int(il), x, uint64_t(T) * D);
    }
    rms(fmod, 0, D, N);
    proj(enc, out_, xn, 0, out, 0, N);
    enc->endEncoding();
    finish(cb);
    const auto* o = static_cast<const float*>(out->contents());
    return std::vector<float>(o, o + uint64_t(N) * kLatentWidth);
}

Engine::LoraTarget MossImage::lora_target(const std::string& module)
{
    unsigned i = 0;
    int used = 0;
    if (std::sscanf(module.c_str(), "layers.%u.%n", &i, &used) == 1 && used > 0 && i < blocks_.size()) {
        Block& B = blocks_[i];
        const std::string_view rest = std::string_view(module).substr(size_t(used));
        if (rest == "proj.qkv")
            return {&B.q, 0, 3 * kDim};
        if (rest == "proj.q")
            return {&B.q, 0, kDim};
        if (rest == "proj.k")
            return {&B.q, kDim, kDim};
        if (rest == "proj.v")
            return {&B.q, 2 * kDim, kDim};
        if (rest == "proj.o")
            return {&B.o, 0, kDim};
        if (rest == "proj.ffn_up_gate")
            return {&B.up_gate, 0, uint32_t(B.up_gate.dims[1])};
        if (rest == "proj.ffn_down")
            return {&B.down, 0, kDim};
        return {};
    }
    Tensor* t = module == "video_embedder"             ? &img_in_
                : module == "text_embedder.proj"        ? &txt_proj_
                : module == "time_embedding.0"          ? &t1_
                : module == "time_embedding.2"          ? &t2_
                : module == "shared_block_modulation"   ? &mod_
                : module == "video_head_modulation"     ? &head_mod_
                : module == "final_linear_video"        ? &out_
                                                        : nullptr;
    return t ? LoraTarget{t, 0, uint32_t(t->dims[1])} : LoraTarget{};
}

} // namespace mako::image
