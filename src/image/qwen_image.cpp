#include "image/qwen_image.hpp"

#include "core/log.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::image {

namespace {

struct HeadsArgs {
    uint32_t rows, ld, col0, dim, n_pad, row0, rope;
};
struct ModArgs {
    uint32_t n;
    float eps;
    uint32_t split, mod_stride, mod_off;
};
struct HeadRowsArgs {
    uint32_t dim, src_pad, dst_pad, rows;
};
struct HeadsNormArgs {
    uint32_t ld, col0, dim, n_pad, row0, rope;
    float eps;
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

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

QwenImage::QwenImage(const std::string& dir, mtl::Context& ctx, const Options& opts) : Engine(dir, ctx, opts)
{
    if (text_width() != kDim)
        throw std::runtime_error("text encoder: not 4096 wide");
    load_transformer(open_files(opts.transformer.empty() ? dir + "/transformer" : opts.transformer));
    drop_ = uint32_t(tok_->encode("<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n", true).size());
}

std::string QwenImage::templated(const std::string& prompt)
{
    return "<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n" +
           (prompt.empty() ? " " : prompt) + "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<float> QwenImage::prompt_embeddings(const std::string& prompt)
{
    const auto ids = tok_->encode(templated(prompt), true);
    return encode_text(ids, drop_);
}

void QwenImage::load_transformer(const safetensors::Files& f)
{
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
        const std::string qkv[3] = {b + "attn.to_q.weight", b + "attn.to_k.weight", b + "attn.to_v.weight"};
        B.qkv = upload_rows(f, qkv);
        B.o = upload(f, b + "attn.to_out.0.weight", true);
        B.norm_q = upload(f, b + "attn.norm_q.weight", false);
        B.norm_k = upload(f, b + "attn.norm_k.weight", false);
        const std::string gu[2] = {b + "img_mlp.gate_layer.weight", b + "img_mlp.proj.weight"};
        B.gate_up = upload_rows(f, gu);
        B.down = upload(f, b + "img_mlp.out.weight", true);
        blocks_.push_back(std::move(B));
    }
    if (blocks_.empty())
        throw std::runtime_error("transformer: no blocks");
}

void QwenImage::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    Text& g = texts_[slot];
    if (prompt.empty() || prompt.size() % kDim)
        throw std::runtime_error("prompt embeddings must be rows of 4096");
    g.S = uint32_t(prompt.size() / kDim);
    gh_ = h;
    gw_ = w;
    g.steps = 0;
    g.prompt.assign(prompt.begin(), prompt.end());
    g.k.clear();
    g.v.clear();
    for (size_t b = 0; b < blocks_.size(); ++b) {
        const uint64_t bytes = uint64_t(kHeads) * g.S * kHeadDim * 2;
        g.k.emplace_back(ctx_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        g.v.emplace_back(ctx_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    }
    // Text positions advance on all three axes; the image block takes the next frame position and a height / width
    // grid centred on zero.
    MTL::Buffer* cs = scratch(std::format("d_cs{}", slot), uint64_t(g.S + h * w) * kHeadDim / 2 * 8);
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < g.S; ++t)
        rope3(c + uint64_t(t) * kHeadDim, t, t, t);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            rope3(c + uint64_t(g.S + y * w + x) * kHeadDim, g.S, int64_t(y) - int64_t(h - h / 2),
                  int64_t(x) - int64_t(w - w / 2));
}

std::vector<float> QwenImage::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    Text& g = texts_[slot];
    const uint32_t N = gh_ * gw_, D = kDim;
    if (latents.size() != uint64_t(N) * kLatent)
        throw std::runtime_error("latents must be [h * w][64]");
    const bool first = g.steps++ == 0;
    // The first step runs the text rows too; later ones only the image's, against the text's cached keys and values.
    const uint32_t S = g.S, T = first ? S + N : N, row0 = first ? 0 : S, n_k = S + N;
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(n_k, 128), split = first ? S : 0;
    // The GEMMs run a multiple of 256 rows (the rest ignored): fewer rows, as the first step's text and image make,
    // would take 64-row tiles that read the weights four times as often (that step took twice as long).
    const uint32_t Tg = T > 256 ? round_up(T, 256) : T;
    MTL::Buffer* x = scratch("d_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkv = scratch("d_qkv", uint64_t(Tg) * 3 * D * 4);
    const uint64_t nff = blocks_.front().down.dims[0];
    MTL::Buffer* gu = scratch("d_gu", uint64_t(Tg) * 2 * nff * 4);
    MTL::Buffer* ffm = scratch("d_ffm", uint64_t(Tg) * nff * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(Tg) * D * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* lat = scratch("d_lat", uint64_t(N) * kLatent * 4);
    MTL::Buffer* out = scratch("d_out", uint64_t(Tg) * kLatent * 4);
    MTL::Buffer* tp = scratch("d_tproj", 2 * kTimeDim * 4);
    MTL::Buffer* temb = scratch("d_temb", 2 * D * 4);
    MTL::Buffer* temb_s = scratch("d_temb_s", 2 * D * 4);
    MTL::Buffer* mod = scratch("d_mod", 2 * 4 * D * 4);
    MTL::Buffer* omod = scratch("d_omod", 2 * D * 4);
    MTL::Buffer* cs = scratch(std::format("d_cs{}", slot), uint64_t(S + N) * kHeadDim / 2 * 8);
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
        std::memcpy(pr->contents(), g.prompt.data(), g.prompt.size() * 4);
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
        proj(enc, B.qkv, xn, 0, qkv, 0, Tg);
        // q and k RMS-normed per head and rotated (the table's rows from this step's first token on), v as it is.
        HeadsNormArgs hn{3 * D, 0, kHeadDim, q_pad, 0, 2, kEps};
        enc->setComputePipelineState(pso("heads_norm_half"));
        enc->setBuffer(qkv, 0, 0);
        enc->setBuffer(cs, uint64_t(row0) * kHeadDim / 2 * 8, 1);
        enc->setBuffer(qh, 0, 2);
        enc->setBuffer(B.norm_q.buf, B.norm_q.off, 3);
        enc->setBytes(&hn, sizeof(hn), 4);
        enc->dispatchThreadgroups(MTL::Size(T, kHeads, 1), MTL::Size(kHeadDim / 2, 1, 1));
        after(enc, "heads_norm_half");
        hn.col0 = D;
        hn.n_pad = k_pad;
        hn.row0 = row0;
        enc->setBuffer(kh, 0, 2);
        enc->setBuffer(B.norm_k.buf, B.norm_k.off, 3);
        enc->setBytes(&hn, sizeof(hn), 4);
        enc->dispatchThreadgroups(MTL::Size(T, kHeads, 1), MTL::Size(kHeadDim / 2, 1, 1));
        after(enc, "heads_norm_half");
        HeadsArgs ha{T, 3 * D, 2 * D, kHeadDim, k_pad, row0, 0};
        enc->setComputePipelineState(pso("heads_half"));
        enc->setBuffer(qkv, 0, 0);
        enc->setBuffer(cs, 0, 1);
        enc->setBuffer(vh, 0, 2);
        enc->setBytes(&ha, sizeof(ha), 3);
        enc->dispatchThreads(MTL::Size(kHeadDim / 2, T, kHeads), MTL::Size(64, 1, 1));
        after(enc, "heads_half");
        HeadRowsArgs hr{kHeadDim, k_pad, S, S};
        if (first) { // keep the text's keys and values
            dispatch(enc, "heads_rows", {kh, g.k[il].get()}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_rows", {vh, g.v[il].get()}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
        } else {
            hr = {kHeadDim, S, k_pad, S};
            dispatch(enc, "heads_rows", {g.k[il].get(), kh}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_rows", {g.v[il].get(), vh}, &hr, sizeof(hr), MTL::Size(kHeadDim, S, kHeads),
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
        proj(enc, B.o, att, 0, y, 0, Tg);
        ma.mod_off = D;
        dispatch(enc, "gate_add", {x, y, mod}, &ma, sizeof(ma), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        ma.mod_off = 2 * D;
        dispatch(enc, "ln_modulate", {x, mod, xn}, &ma, sizeof(ma), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
        proj(enc, B.gate_up, xn, 0, gu, 0, Tg);
        uint32_t nff = uint32_t(B.down.dims[0]);
        dispatch(enc, "swiglu_rows", {gu, ffm}, &nff, sizeof(nff), MTL::Size(nff, Tg, 1), MTL::Size(256, 1, 1));
        proj(enc, B.down, ffm, 0, y, 0, Tg);
        ma.mod_off = 3 * D;
        dispatch(enc, "gate_add", {x, y, mod}, &ma, sizeof(ma), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        probe(enc, "block", int(il), x, uint64_t(T) * D);
    }
    ModArgs oa{D, kEps, split, D, 0};
    dispatch(enc, "ln_modulate", {x, omod, xn}, &oa, sizeof(oa), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
    probe(enc, "norm_out", -1, xn, uint64_t(T) * D);
    proj(enc, proj_out_, xn, 0, out, 0, Tg);
    enc->endEncoding();
    finish(cb);
    const auto* o = static_cast<const float*>(out->contents()) + uint64_t(T - N) * kLatent;
    return std::vector<float>(o, o + uint64_t(N) * kLatent);
}

Engine::LoraTarget QwenImage::lora_target(const std::string& m)
{
    const uint32_t D = kDim, nff = uint32_t(blocks_.front().down.dims[0]);
    unsigned b = 0;
    int used = 0;
    if (std::sscanf(m.c_str(), "transformer_blocks.%u.%n", &b, &used) == 1 && used > 0 && b < blocks_.size()) {
        Block& B = blocks_[b];
        const std::string_view rest = std::string_view(m).substr(size_t(used));
        if (rest == "attn.to_q")
            return {&B.qkv, 0, D};
        if (rest == "attn.to_k")
            return {&B.qkv, D, D};
        if (rest == "attn.to_v")
            return {&B.qkv, 2 * D, D};
        if (rest == "attn.to_out.0")
            return {&B.o, 0, D};
        if (rest == "img_mlp.gate_layer")
            return {&B.gate_up, 0, nff};
        if (rest == "img_mlp.proj")
            return {&B.gate_up, nff, nff};
        if (rest == "img_mlp.gate_up") // gate rows first
            return {&B.gate_up, 0, 2 * nff};
        if (rest == "img_mlp.out")
            return {&B.down, 0, D};
        return {};
    }
    Tensor* t = m == "img_in" ? &img_in_
                : m == "txt_in.in_layer" ? &txt_in1_
                : m == "txt_in.out_layer" ? &txt_in2_
                : m == "proj_out" ? &proj_out_
                : m == "norm_out.linear" ? &out_lin_
                : m == "modulation.1" ? &mod_
                : m == "time_text_embed.timestep_embedder.linear_1" ? &t_lin1_
                : m == "time_text_embed.timestep_embedder.linear_2" ? &t_lin2_
                                                                     : nullptr;
    return t ? LoraTarget{t, 0, uint32_t(t->dims[1])} : LoraTarget{};
}

} // namespace mako::image
