#include "image/krea2.hpp"

#include "core/log.hpp"

#include <glaze/glaze.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace mako::image {

namespace {

struct KreaModArgs {
    uint32_t n, blocks;
};
struct RmsAffineArgs {
    uint32_t n;
    float eps;
};
struct GateArgs {
    uint32_t n, ld, col0;
};
struct GroupAttnArgs {
    uint32_t rows, group, heads, ld;
    float eps;
};
struct LayerMixArgs {
    uint32_t n, layers;
    float p[16];
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

// The original checkpoint's names as the Diffusers layout has them.
std::string diffusers_name(const std::string& original)
{
    static constexpr std::pair<const char*, const char*> kPrefixes[] = {
        {"first.", "img_in."},
        {"tmlp.0.", "time_embed.linear_1."},
        {"tmlp.2.", "time_embed.linear_2."},
        {"tproj.1.", "time_mod_proj."},
        {"txtmlp.0.scale", "txt_in.norm.weight"},
        {"txtmlp.1.", "txt_in.linear_1."},
        {"txtmlp.3.", "txt_in.linear_2."},
        {"txtfusion.", "text_fusion."},
        {"blocks.", "transformer_blocks."},
        {"last.linear.", "final_layer.linear."},
        {"last.norm.scale", "final_layer.norm.weight"},
        {"last.modulation.lin", "final_layer.scale_shift_table"}};
    static constexpr std::pair<const char*, const char*> kParts[] = {
        {".attn.wq.", ".attn.to_q."},
        {".attn.wk.", ".attn.to_k."},
        {".attn.wv.", ".attn.to_v."},
        {".attn.wo.", ".attn.to_out.0."},
        {".attn.gate.", ".attn.to_gate."},
        {".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"},
        {".attn.qknorm.knorm.scale", ".attn.norm_k.weight"},
        {".mlp.", ".ff."},
        {".prenorm.scale", ".norm1.weight"},
        {".postnorm.scale", ".norm2.weight"},
        {".mod.lin", ".scale_shift_table"}};
    std::string n = original;
    for (const auto& [o, d] : kPrefixes)
        if (n.starts_with(o)) {
            n = d + n.substr(std::strlen(o));
            break;
        }
    for (const auto& [o, d] : kParts)
        if (const size_t at = n.find(o); at != std::string::npos)
            n.replace(at, std::strlen(o), d);
    return n;
}

} // namespace

Krea2::Krea2(const std::string& dir, mtl::Context& ctx, const Options& opts) : Engine(dir, ctx, opts)
{
    if (text_width() != kText)
        throw std::runtime_error("text encoder: not 2560 wide");
    {
        std::ifstream in(dir + "/model_index.json", std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (!glz::read_json(j, text) && j.contains("is_distilled") && j["is_distilled"].is_boolean())
            distilled_ = j["is_distilled"].get_boolean();
    }
    // The distilled checkpoint's schedule takes a fixed shift.
    if (distilled_)
        sched_.mu = 1.15;
    load_transformer(open_files(opts.transformer.empty() ? dir + "/transformer" : opts.transformer));
    drop_ = uint32_t(tok_->encode("<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n", true)
                         .size());
}

std::string Krea2::templated(const std::string& prompt)
{
    return "<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n" + prompt +
           "<|im_end|>\n<|im_start|>assistant\n";
}

std::string Krea2::key(const std::string& original) const
{
    return diffusers_names_ ? diffusers_name(original) : prefix_ + original;
}

void Krea2::load_transformer(const safetensors::Files& f)
{
    diffusers_names_ = bool(f.find("transformer_blocks.0.attn.to_q.weight"));
    prefix_ = f.find("model.diffusion_model.first.weight") ? "model.diffusion_model." : "";
    if (!diffusers_names_ && !f.find(prefix_ + "first.weight"))
        throw std::runtime_error("transformer: not Krea 2's weights");
    auto mat = [&](const std::string& n) { return upload(f, key(n), true); };
    auto vec = [&](const std::string& n) { return upload(f, key(n), false); };
    auto layer = [&](const std::string& b) {
        Layer L;
        L.norm1 = upload_plus_one(f, key(b + "prenorm.scale"));
        L.norm2 = upload_plus_one(f, key(b + "postnorm.scale"));
        const std::string qkvg[4] = {key(b + "attn.wq.weight"), key(b + "attn.wk.weight"), key(b + "attn.wv.weight"),
                                     key(b + "attn.gate.weight")};
        L.qkvg = upload_rows(f, qkvg);
        L.o = mat(b + "attn.wo.weight");
        L.norm_q = upload_plus_one(f, key(b + "attn.qknorm.qnorm.scale"));
        L.norm_k = upload_plus_one(f, key(b + "attn.qknorm.knorm.scale"));
        const std::string gu[2] = {key(b + "mlp.gate.weight"), key(b + "mlp.up.weight")};
        L.gate_up = upload_rows(f, gu);
        L.down = mat(b + "mlp.down.weight");
        return L;
    };
    img_in_ = mat("first.weight");
    img_in_b_ = vec("first.bias");
    t1_ = mat("tmlp.0.weight");
    t1_b_ = vec("tmlp.0.bias");
    t2_ = mat("tmlp.2.weight");
    t2_b_ = vec("tmlp.2.bias");
    tproj_ = mat("tproj.1.weight");
    tproj_b_ = vec("tproj.1.bias");
    txt_norm_ = upload_plus_one(f, key("txtmlp.0.scale"));
    txt1_ = mat("txtmlp.1.weight");
    txt1_b_ = vec("txtmlp.1.bias");
    txt2_ = mat("txtmlp.3.weight");
    txt2_b_ = vec("txtmlp.3.bias");
    for (const char* kind : {"layerwise_blocks", "refiner_blocks"})
        for (uint32_t i = 0; f.find(key(std::format("txtfusion.{}.{}.attn.wq.weight", kind, i))); ++i)
            fusion_.push_back(layer(std::format("txtfusion.{}.{}.", kind, i)));
    if (fusion_.size() != 4)
        throw std::runtime_error(std::format("transformer: {} text fusion blocks, not 2 + 2", fusion_.size()));
    const auto p = floats(f, key("txtfusion.projector.weight"));
    if (p.size() != 12)
        throw std::runtime_error("transformer: the text projector is not over 12 layers");
    std::copy(p.begin(), p.end(), projector_);
    for (uint32_t i = 0; f.find(key(std::format("blocks.{}.attn.wq.weight", i))); ++i)
        blocks_.push_back(layer(std::format("blocks.{}.", i)));
    if (blocks_.empty())
        throw std::runtime_error("transformer: no blocks");
    const uint32_t nb = uint32_t(blocks_.size());
    tables_ = reserve(uint64_t(nb) * 6 * kDim * 4, gguf::Type::F32, {6 * kDim, nb});
    norms_ = reserve(uint64_t(nb) * 2 * kDim * 4, gguf::Type::F32, {2 * kDim, nb});
    auto* tab = reinterpret_cast<float*>(static_cast<char*>(tables_.buf->contents()) + tables_.off);
    auto* nrm = reinterpret_cast<float*>(static_cast<char*>(norms_.buf->contents()) + norms_.off);
    for (uint32_t b = 0; b < nb; ++b) {
        const auto t = floats(f, key(std::format("blocks.{}.mod.lin", b)));
        if (t.size() != 6 * kDim)
            throw std::runtime_error("transformer: a block's modulation table is not 6 x 6144");
        std::copy(t.begin(), t.end(), tab + uint64_t(b) * 6 * kDim);
        for (int k = 0; k < 2; ++k) {
            const auto s = floats(f, key(std::format("blocks.{}.{}.scale", b, k ? "postnorm" : "prenorm")));
            for (uint32_t i = 0; i < kDim; ++i)
                nrm[(uint64_t(b) * 2 + k) * kDim + i] = s[i] + 1.0f;
        }
    }
    out_norm_ = upload_plus_one(f, key("last.norm.scale"));
    out_table_ = vec("last.modulation.lin");
    out_ = mat("last.linear.weight");
    out_b_ = vec("last.linear.bias");
    log::info("Krea 2 transformer: {} blocks, {} names{}", nb, diffusers_names_ ? "Diffusers" : "original",
              distilled_ ? ", distilled" : "");
}

std::vector<float> Krea2::prompt_embeddings(const std::string& prompt)
{
    // The template's system and user parts and the prompt, at most kMaxPrompt tokens of it, then the assistant turn.
    auto ids = tok_->encode("<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n" + prompt,
                            true);
    if (ids.size() > drop_ + kMaxPrompt)
        ids.resize(drop_ + kMaxPrompt);
    const auto tail = tok_->encode("<|im_end|>\n<|im_start|>assistant\n", true);
    ids.insert(ids.end(), tail.begin(), tail.end());
    return encode_text(ids, drop_, kTaps);
}

void Krea2::fusion_block(MTL::ComputeCommandEncoder*& enc, const Layer& L, MTL::Buffer* x, uint32_t rows,
                         uint32_t groups)
{
    const uint32_t E = kText, nff = uint32_t(L.down.dims[0]);
    MTL::Buffer* xn = scratch("f_xn", uint64_t(rows) * E * 4);
    MTL::Buffer* qkvg = scratch("f_qkvg", uint64_t(rows) * 4 * E * 4);
    MTL::Buffer* att = scratch("f_att", uint64_t(rows) * E * 4);
    MTL::Buffer* y = scratch("f_y", uint64_t(rows) * E * 4);
    MTL::Buffer* gu = scratch("f_gu", uint64_t(rows) * 2 * nff * 4);
    MTL::Buffer* ffm = scratch("f_ffm", uint64_t(rows) * nff * 4);
    norm(enc, x, &L.norm1, xn, rows, E, E, kEps);
    proj(enc, L.qkvg, xn, 0, qkvg, 0, rows);
    if (groups) {
        const GroupAttnArgs a{rows, groups, kTextHeads, 4 * E, kEps};
        enc->setComputePipelineState(pso("attn_groups"));
        enc->setBuffer(qkvg, 0, 0);
        enc->setBuffer(L.norm_q.buf, L.norm_q.off, 1);
        enc->setBuffer(L.norm_k.buf, L.norm_k.off, 2);
        enc->setBuffer(att, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreads(MTL::Size(rows, kTextHeads, 1), MTL::Size(64, 1, 1));
        after(enc, "attn_groups");
    } else {
        const uint32_t q_pad = round_up(rows, kAttnRows), k_pad = round_up(rows, 128);
        MTL::Buffer* qh = scratch("f_qh", uint64_t(kTextHeads) * q_pad * kHeadDim * 2);
        MTL::Buffer* kh = scratch("f_kh", uint64_t(kTextHeads) * k_pad * kHeadDim * 2);
        MTL::Buffer* vh = scratch("f_vh", uint64_t(kTextHeads) * k_pad * kHeadDim * 2);
        heads_norm(enc, qkvg, 4 * E, 0, kTextHeads, rows, q_pad, nullptr, 0, 0, L.norm_q, kEps, 0, qh);
        heads_norm(enc, qkvg, 4 * E, E, kTextHeads, rows, k_pad, nullptr, 0, 0, L.norm_k, kEps, 0, kh);
        if (k_pad > rows) {
            const uint32_t c[4] = {kHeadDim, k_pad, rows, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - rows, kTextHeads),
                     MTL::Size(128, 1, 1));
        }
        heads(enc, qkvg, 4 * E, 2 * E, kTextHeads, rows, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, rows, q_pad, rows, k_pad, kTextHeads, kTextHeads, 0, 0);
    }
    const GateArgs ga{E, 4 * E, 3 * E};
    dispatch(enc, "sigmoid_mul", {att, qkvg}, &ga, sizeof(ga), MTL::Size(E, rows, 1), MTL::Size(256, 1, 1));
    proj(enc, L.o, att, 0, y, 0, rows);
    add(enc, x, y, x, rows * E);
    norm(enc, x, &L.norm2, xn, rows, E, E, kEps);
    proj(enc, L.gate_up, xn, 0, gu, 0, rows);
    dispatch(enc, "swiglu_rows", {gu, ffm}, &nff, sizeof(nff), MTL::Size(nff, rows, 1), MTL::Size(256, 1, 1));
    proj(enc, L.down, ffm, 0, y, 0, rows);
    add(enc, x, y, x, rows * E);
}

void Krea2::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    const uint32_t L = 12, E = kText, D = kDim;
    if (prompt.empty() || prompt.size() % (L * E))
        throw std::runtime_error("prompt states must be rows of 12 x 2560");
    S_[slot] = uint32_t(prompt.size() / (L * E));
    const uint32_t S = S_[slot];
    gh_ = h;
    gw_ = w;
    MTL::Buffer* x = scratch("f_x", prompt.size() * 4);
    MTL::Buffer* mixed = scratch("f_mixed", uint64_t(S) * E * 4);
    MTL::Buffer* xn = scratch("f_tn", uint64_t(S) * E * 4);
    MTL::Buffer* y = scratch("f_ty", uint64_t(S) * D * 4);
    MTL::Buffer* txt = scratch(std::format("k_txt{}", slot), uint64_t(S) * D * 4);
    std::memcpy(x->contents(), prompt.data(), prompt.size() * 4);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // Across each token's tapped layers, the layers mixed into one, then across the tokens.
    fusion_block(enc, fusion_[0], x, S * L, L);
    fusion_block(enc, fusion_[1], x, S * L, L);
    LayerMixArgs lm{E, L, {}};
    std::copy(std::begin(projector_), std::end(projector_), lm.p);
    dispatch(enc, "layer_mix", {x, mixed}, &lm, sizeof(lm), MTL::Size(E, S, 1), MTL::Size(256, 1, 1));
    fusion_block(enc, fusion_[2], mixed, S, 0);
    fusion_block(enc, fusion_[3], mixed, S, 0);
    probe(enc, "fusion", -1, mixed, uint64_t(S) * E);
    norm(enc, mixed, &txt_norm_, xn, S, E, E, kEps);
    proj(enc, txt1_, xn, 0, y, 0, S);
    bias(enc, y, txt1_b_, D, S);
    unary(enc, "gelu_tanh", y, S * D);
    proj(enc, txt2_, y, 0, txt, 0, S);
    bias(enc, txt, txt2_b_, D, S);
    probe(enc, "txt_in", -1, txt, uint64_t(S) * D);
    enc->endEncoding();
    finish(cb);

    // Rotary angles of [text, image]: the text at the origin, the image at (0, y, x); axes 32 / 48 / 48, theta 1000.
    MTL::Buffer* cs = scratch(std::format("k_cs{}", slot), uint64_t(S + h * w) * kHeadDim / 2 * 8);
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < S + h * w; ++t) {
        const int64_t pos[3] = {0, t < S ? 0 : (t - S) / w, t < S ? 0 : (t - S) % w};
        constexpr int dims[3] = {32, 48, 48};
        uint32_t i = 0;
        for (int a = 0; a < 3; ++a)
            for (int j = 0; j < dims[a] / 2; ++j, ++i) {
                const double ang = double(pos[a]) / std::pow(1000.0, double(2 * j) / double(dims[a]));
                c[(uint64_t(t) * kHeadDim / 2 + i) * 2] = float(std::cos(ang));
                c[(uint64_t(t) * kHeadDim / 2 + i) * 2 + 1] = float(std::sin(ang));
            }
    }
}

std::vector<float> Krea2::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    const uint32_t N = gh_ * gw_, D = kDim, S = S_[slot], T = S + N, nb = uint32_t(blocks_.size());
    if (latents.size() != uint64_t(N) * kLatent)
        throw std::runtime_error("latents must be [h * w][64]");
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    // The GEMMs run a multiple of 256 rows (the rest ignored), which keeps them on their largest tiles.
    const uint32_t Tg = T > 256 ? round_up(T, 256) : T;
    const uint32_t kv = kKvHeads * kHeadDim, wq = 2 * D + 2 * kv; // q | k | v | gate
    const uint64_t nff = blocks_.front().down.dims[0];
    MTL::Buffer* x = scratch("d_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkvg = scratch("d_qkvg", uint64_t(Tg) * wq * 4);
    MTL::Buffer* gu = scratch("d_gu", uint64_t(Tg) * 2 * nff * 4);
    MTL::Buffer* ffm = scratch("d_ffm", uint64_t(Tg) * nff * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kKvHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kKvHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(Tg) * D * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* lat = scratch("d_lat", uint64_t(N) * kLatent * 4);
    MTL::Buffer* out = scratch("d_out", uint64_t(N) * kLatent * 4);
    MTL::Buffer* tp = scratch("d_tproj", kTimeDim * 4);
    MTL::Buffer* temb = scratch("d_temb", D * 4);
    MTL::Buffer* tg = scratch("d_tg", D * 4);
    MTL::Buffer* tm = scratch("d_tm", 6 * D * 4);
    MTL::Buffer* mods = scratch("d_mods", uint64_t(nb) * 6 * D * 4);
    MTL::Buffer* fmod = scratch("d_fmod", 2 * D * 4);
    MTL::Buffer* txt = scratch(std::format("k_txt{}", slot), uint64_t(S) * D * 4);
    MTL::Buffer* cs = scratch(std::format("k_cs{}", slot), uint64_t(T) * kHeadDim / 2 * 8);
    std::memcpy(lat->contents(), latents.data(), latents.size() * 4);
    std::memcpy(x->contents(), txt->contents(), uint64_t(S) * D * 4);
    // Sinusoidal embedding (cos half first) of 1000 * the noise level.
    auto* tpf = static_cast<float*>(tp->contents());
    for (uint32_t i = 0; i < kTimeDim / 2; ++i) {
        const float freq = std::exp(-std::log(10000.0f) * float(i) / float(kTimeDim / 2));
        const float arg = 1000.0f * sigma * freq;
        tpf[i] = std::cos(arg);
        tpf[kTimeDim / 2 + i] = std::sin(arg);
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, t1_, tp, 0, temb, 0, 1);
    bias(enc, temb, t1_b_, D, 1);
    unary(enc, "gelu_tanh", temb, D);
    proj(enc, t2_, temb, 0, tg, 0, 1);
    bias(enc, tg, t2_b_, D, 1);
    probe(enc, "temb", -1, tg, D);
    // tg: the embedding (the final layer's); temb: gelu of it, into the blocks' shared modulation.
    uint32_t n = D;
    dispatch(enc, "copy_f32", {tg, temb}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    unary(enc, "gelu_tanh", temb, D);
    proj(enc, tproj_, temb, 0, tm, 0, 1);
    bias(enc, tm, tproj_b_, 6 * D, 1);
    probe(enc, "tmod", -1, tm, 6 * D);
    const KreaModArgs ka{D, nb};
    enc->setComputePipelineState(pso("krea_mod"));
    enc->setBuffer(tm, 0, 0);
    enc->setBuffer(tables_.buf, tables_.off, 1);
    enc->setBuffer(norms_.buf, norms_.off, 2);
    enc->setBuffer(mods, 0, 3);
    enc->setBytes(&ka, sizeof(ka), 4);
    enc->dispatchThreads(MTL::Size(D, nb, 1), MTL::Size(256, 1, 1));
    after(enc, "krea_mod");
    {
        enc->setComputePipelineState(pso("krea_final_mod"));
        enc->setBuffer(tg, 0, 0);
        enc->setBuffer(out_table_.buf, out_table_.off, 1);
        enc->setBuffer(out_norm_.buf, out_norm_.off, 2);
        enc->setBuffer(fmod, 0, 3);
        enc->setBytes(&n, sizeof(n), 4);
        enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "krea_final_mod");
    }
    proj(enc, img_in_, lat, 0, x, uint64_t(S) * D * 4, N);
    bias(enc, x, img_in_b_, D, N, uint64_t(S) * D * 4);
    probe(enc, "img_in", -1, x, uint64_t(N) * D, uint64_t(S) * D * 4);

    // y = x / rms(x) * scale + shift over `rows` rows from `row0`, with scale, shift at mods offsets (in floats).
    auto rms = [&](MTL::Buffer* src, MTL::Buffer* dst, MTL::Buffer* m, uint64_t s_off, uint64_t b_off, uint32_t row0,
                   uint32_t rows) {
        const RmsAffineArgs a{D, kEps};
        enc->setComputePipelineState(pso("rms_affine"));
        enc->setBuffer(src, uint64_t(row0) * D * 4, 0);
        enc->setBuffer(m, s_off * 4, 1);
        enc->setBuffer(m, b_off * 4, 2);
        enc->setBuffer(dst, uint64_t(row0) * D * 4, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "rms_affine");
    };
    auto gated = [&](MTL::Buffer* m, uint64_t g_off) {
        enc->setComputePipelineState(pso("scale_add"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(y, 0, 1);
        enc->setBuffer(m, g_off * 4, 2);
        enc->setBytes(&D, sizeof(D), 3);
        enc->dispatchThreads(MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        after(enc, "scale_add");
    };
    for (uint32_t il = 0; il < nb; ++il) {
        const Layer& B = blocks_[il];
        cur_layer_ = il;
        const uint64_t mo = uint64_t(il) * 6 * D;
        rms(x, xn, mods, mo, mo + D, 0, T);
        proj(enc, B.qkvg, xn, 0, qkvg, 0, Tg);
        heads_norm(enc, qkvg, wq, 0, kHeads, T, q_pad, cs, 0, 2, B.norm_q, kEps, 0, qh);
        heads_norm(enc, qkvg, wq, D, kKvHeads, T, k_pad, cs, 0, 2, B.norm_k, kEps, 0, kh);
        if (k_pad > T) {
            const uint32_t c[4] = {kHeadDim, k_pad, T, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - T, kKvHeads),
                     MTL::Size(128, 1, 1));
        }
        heads(enc, qkvg, wq, D + kv, kKvHeads, T, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, kHeads, kKvHeads, 0, 0);
        const GateArgs ga{D, wq, D + 2 * kv};
        dispatch(enc, "sigmoid_mul", {att, qkvg}, &ga, sizeof(ga), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        proj(enc, B.o, att, 0, y, 0, Tg);
        gated(mods, mo + 2 * D);
        rms(x, xn, mods, mo + 3 * D, mo + 4 * D, 0, T);
        proj(enc, B.gate_up, xn, 0, gu, 0, Tg);
        uint32_t f = uint32_t(nff);
        dispatch(enc, "swiglu_rows", {gu, ffm}, &f, sizeof(f), MTL::Size(f, Tg, 1), MTL::Size(256, 1, 1));
        proj(enc, B.down, ffm, 0, y, 0, Tg);
        gated(mods, mo + 5 * D);
        probe(enc, "block", int(il), x, uint64_t(T) * D);
    }
    rms(x, xn, fmod, 0, D, S, N);
    proj(enc, out_, xn, uint64_t(S) * D * 4, out, 0, N);
    bias(enc, out, out_b_, kLatent, N);
    enc->endEncoding();
    finish(cb);
    const auto* o = static_cast<const float*>(out->contents());
    return std::vector<float>(o, o + uint64_t(N) * kLatent);
}

Engine::LoraTarget Krea2::lora_target(const std::string& module)
{
    // Diffusers names as the original checkpoint's.
    static constexpr std::pair<const char*, const char*> kNames[] = {
        {"transformer_blocks.", "blocks."},  {"text_fusion.", "txtfusion."},
        {"img_in", "first"},                 {"time_embed.linear_1", "tmlp.0"},
        {"time_embed.linear_2", "tmlp.2"},   {"time_mod_proj", "tproj.1"},
        {"txt_in.linear_1", "txtmlp.1"},     {"txt_in.linear_2", "txtmlp.3"},
        {"final_layer.linear", "last.linear"}};
    static constexpr std::pair<const char*, const char*> kParts[] = {
        {".attn.to_q", ".attn.wq"}, {".attn.to_k", ".attn.wk"},       {".attn.to_v", ".attn.wv"},
        {".attn.to_gate", ".attn.gate"}, {".attn.to_out.0", ".attn.wo"}, {".ff.", ".mlp."}};
    std::string m = module;
    for (const auto& [d, o] : kNames)
        if (m.starts_with(d)) {
            m = o + m.substr(std::strlen(d));
            break;
        }
    for (const auto& [d, o] : kParts)
        if (const size_t at = m.find(d); at != std::string::npos)
            m.replace(at, std::strlen(d), o);

    Layer* L = nullptr;
    std::string_view rest;
    uint32_t width = kDim;
    for (const auto& [prefix, base, count, w] :
         {std::tuple{"blocks.", 0u, uint32_t(blocks_.size()), uint32_t(kDim)},
          std::tuple{"txtfusion.layerwise_blocks.", 0u, 2u, uint32_t(kText)},
          std::tuple{"txtfusion.refiner_blocks.", 2u, 2u, uint32_t(kText)}}) {
        unsigned i = 0;
        int used = 0;
        if (std::sscanf(m.c_str(), (std::string(prefix) + "%u.%n").c_str(), &i, &used) == 1 && used > 0 && i < count) {
            L = std::string_view(prefix) == "blocks." ? &blocks_[i] : &fusion_[base + i];
            rest = std::string_view(m).substr(size_t(used));
            width = w;
            break;
        }
    }
    if (L) {
        const uint32_t kv = uint32_t(L->qkvg.dims[1] - 2 * width) / 2, nff = uint32_t(L->down.dims[0]);
        if (rest == "attn.wq")
            return {&L->qkvg, 0, width};
        if (rest == "attn.wk")
            return {&L->qkvg, width, kv};
        if (rest == "attn.wv")
            return {&L->qkvg, width + kv, kv};
        if (rest == "attn.gate")
            return {&L->qkvg, width + 2 * kv, width};
        if (rest == "attn.wo")
            return {&L->o, 0, width};
        if (rest == "mlp.gate")
            return {&L->gate_up, 0, nff};
        if (rest == "mlp.up")
            return {&L->gate_up, nff, nff};
        if (rest == "mlp.down")
            return {&L->down, 0, width};
        return {};
    }
    Tensor* t = m == "first"         ? &img_in_
                : m == "tmlp.0"      ? &t1_
                : m == "tmlp.2"      ? &t2_
                : m == "tproj.1"     ? &tproj_
                : m == "txtmlp.1"    ? &txt1_
                : m == "txtmlp.3"    ? &txt2_
                : m == "last.linear" ? &out_
                                     : nullptr;
    return t ? LoraTarget{t, 0, uint32_t(t->dims[1])} : LoraTarget{};
}

} // namespace mako::image
