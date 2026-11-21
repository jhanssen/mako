#include "vision/encoder.hpp"

#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"
#include "kernels/args.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace mako::vision {

namespace {

// Kernel argument structs, as in vision.metal and glm.metal.
struct PatchOrder {
    uint32_t px, blocks;
};
struct PatchArgs {
    uint32_t width, height, patch, n_embd, n_patches;
    PatchOrder order;
};
struct QkvArgs {
    uint32_t n_tok, n_pad, heads;
    float eps;
    uint32_t qk_norm;
    float theta;
    PatchOrder order;
};
struct FlashArgs {
    uint32_t n_tok, n_pad, width;
    float scale;
};
struct RowsArgs {
    uint32_t n, rows;
};
struct UnfoldArgs {
    uint32_t px, py, cells_x, n_cells, n_embd;
};
using kernels::NormArgs;
using kernels::LayerNormArgs;
using kernels::SwigluArgs;
struct SplitQkvArgs {
    uint32_t n_tok, n_pad, heads, dim, pad;
};

constexpr uint32_t kHead = 64; // dims a head
constexpr uint32_t kHy4Head = 72, kHy4Pad = 80; // hy4f: dims a head, and as the attention kernel takes them
constexpr float kRopeTheta = 10000.0f;
constexpr float kProjectorNormEps = 1e-5f;    // glm5v's LayerNorm after the merger

uint16_t to_f16(float v) { return std::bit_cast<uint16_t>(static_cast<_Float16>(v)); }

std::vector<float> host(const gguf::File& f, std::string_view name, uint64_t n)
{
    const gguf::Tensor* t = f.tensor(name);
    if (!t || t->elements() != n)
        throw std::runtime_error(std::format("vision projector: no tensor {} of {} values", name, n));
    std::vector<float> v(n);
    quant::dequant_row(t->type, f.data(*t).data(), v.data(), n);
    return v;
}

} // namespace

Encoder::Encoder(const mtl::Context& ctx, const Mmproj& mm)
    : ctx_(ctx), mm_(mm), gemm_(ctx), lib_(ctx.load(mtl::kernels_metallib())), out_embd_(mm.out_embd)
{
    if (mm.prep.kind == Projector::hy4f) {
        if (mm.n_embd / mm.n_head != kHy4Head)
            throw std::runtime_error(std::format("vision heads of {} dims (72 supported)", mm.n_embd / mm.n_head));
        load_hy4f();
        return;
    }
    if (mm.n_embd / mm.n_head != kHead)
        throw std::runtime_error(std::format("vision heads of {} dims (64 supported)", mm.n_embd / mm.n_head));
    const bool glm = mm.prep.kind == Projector::glm5v;
    // glm5v's two patch convolutions see the same image: one with the summed kernel.
    patch_w_ = weight("v.patch_embd.weight", false);
    if (glm) {
        const auto w1 = host(mm.file, "v.patch_embd.weight.1", uint64_t(mm.n_embd) * 3 * mm.prep.patch * mm.prep.patch);
        auto* w = static_cast<float*>(patch_w_->contents());
        for (size_t i = 0; i < w1.size(); ++i)
            w[i] += w1[i];
    }
    patch_b_ = weight("v.patch_embd.bias", false);
    post_ln_ = weight("v.post_ln.weight", false);
    for (uint32_t i = 0; i < mm.n_layer; ++i) {
        const auto n = [&](const char* s) { return std::format("v.blk.{}.{}", i, s); };
        layers_.push_back({weight(n("ln1.weight"), false), weight(n("attn_qkv.weight")),
                           weight(n("attn_qkv.bias"), false), weight(n("attn_q_norm.weight"), false, glm),
                           weight(n("attn_k_norm.weight"), false, glm), weight(n("attn_out.weight")),
                           weight(n("attn_out.bias"), false), weight(n("ln2.weight"), false),
                           weight(n("ffn_gate.weight")), weight(n("ffn_gate.bias"), false, false),
                           weight(n("ffn_up.weight")), weight(n("ffn_up.bias"), false, false),
                           weight(n("ffn_down.weight")), weight(n("ffn_down.bias"), false, false)});
    }
    if (!glm) {
        mm1_ = weight("mm.1.weight");
        mm1_b_ = weight("mm.1.bias", false);
        mm2_ = weight("mm.2.weight");
        mm2_b_ = weight("mm.2.bias", false);
        start_ = host(mm.file, "v.token_embd.img_start", out_embd_);
        end_ = host(mm.file, "v.token_embd.img_end", out_embd_);
        newline_ = host(mm.file, "v.image_newline", out_embd_);
        return;
    }
    // The merger's kernel [out][c][dy][dx] as rows over the 4 consecutive tokens of a 2x2 block (token j = 2dy + dx):
    // w'[o][j * n_embd + c].
    const uint32_t E = mm.n_embd;
    const auto mw = host(mm.file, "mm.patch_merger.weight", uint64_t(out_embd_) * E * 4);
    merger_ = ctx_.device()->newBuffer(mw.size() * 2, MTL::ResourceStorageModeShared);
    weights_.emplace_back(merger_);
    auto* dst = static_cast<uint16_t*>(merger_->contents());
    for (uint32_t o = 0; o < out_embd_; ++o)
        for (uint32_t c = 0; c < E; ++c)
            for (uint32_t j = 0; j < 4; ++j)
                dst[(uint64_t(o) * 4 + j) * E + c] = to_f16(mw[(uint64_t(o) * E + c) * 4 + j]);
    merger_b_ = weight("mm.patch_merger.bias", false);
    fc_ = weight("mm.model.fc.weight");
    norm_w_ = weight("mm.post_norm.weight", false);
    norm_b_ = weight("mm.post_norm.bias", false);
    pgate_ = weight("mm.gate.weight");
    pup_ = weight("mm.up.weight");
    pdown_ = weight("mm.down.weight");
    const gguf::Tensor* g = mm.file.tensor("mm.gate.weight");
    proj_ff_ = uint32_t(g->dims[1]);
}

// hy4f: q, k and v as one matrix, the output projection's inputs at the attention's padded head width.
void Encoder::load_hy4f()
{
    const uint32_t E = mm_.n_embd, H = mm_.n_head, FF = mm_.n_ff;
    ff_ = (FF + 31) / 32 * 32;
    patch_w_ = weight("v.patch_embd.weight", false);
    patch_b_ = weight("v.patch_embd.bias", false);
    const gguf::Tensor* pg = mm_.file.tensor("v.position_embd.weight");
    if (!pg || pg->dims.size() != 2 || pg->dims[0] != E)
        throw std::runtime_error("vision projector: no position grid");
    grid_ = uint32_t(std::lround(std::sqrt(double(pg->dims[1]))));
    if (uint64_t(grid_) * grid_ != pg->dims[1])
        throw std::runtime_error("vision projector: the position grid is not square");
    pos_grid_ = host(mm_.file, "v.position_embd.weight", pg->elements());
    for (uint32_t i = 0; i < mm_.n_layer; ++i) {
        const auto n = [&](const char* s) { return std::format("v.blk.{}.{}", i, s); };
        Layer L{};
        L.ln1 = weight(n("ln1.weight"), false);
        L.ln1_b = weight(n("ln1.bias"), false);
        L.ln2 = weight(n("ln2.weight"), false);
        L.ln2_b = weight(n("ln2.bias"), false);
        std::vector<float> qkv, qkv_b;
        for (const char* part : {"attn_q", "attn_k", "attn_v"}) {
            const auto w = host(mm_.file, n(std::format("{}.weight", part).c_str()), uint64_t(E) * E);
            const auto b = host(mm_.file, n(std::format("{}.bias", part).c_str()), E);
            qkv.insert(qkv.end(), w.begin(), w.end());
            qkv_b.insert(qkv_b.end(), b.begin(), b.end());
        }
        L.qkv = f16(qkv);
        L.qkv_b = ctx_.device()->newBuffer(qkv_b.data(), qkv_b.size() * 4, MTL::ResourceStorageModeShared);
        weights_.emplace_back(L.qkv_b);
        const auto wo = host(mm_.file, n("attn_out.weight"), uint64_t(E) * E);
        std::vector<float> wp(uint64_t(E) * H * kHy4Pad, 0.0f);
        for (uint32_t o = 0; o < E; ++o)
            for (uint32_t h = 0; h < H; ++h)
                std::copy_n(wo.data() + uint64_t(o) * E + h * kHy4Head, kHy4Head,
                            wp.data() + (uint64_t(o) * H + h) * kHy4Pad);
        L.out = f16(wp);
        L.out_b = weight(n("attn_out.bias"), false);
        // The MLP's width padded to whole GEMM steps with zeros (gelu(0) = 0 adds nothing).
        std::vector<float> up = host(mm_.file, n("ffn_up.weight"), uint64_t(FF) * E), up_b = host(mm_.file, n("ffn_up.bias"), FF);
        up.resize(uint64_t(ff_) * E, 0.0f);
        up_b.resize(ff_, 0.0f);
        L.up = f16(up);
        L.up_b = ctx_.device()->newBuffer(up_b.data(), up_b.size() * 4, MTL::ResourceStorageModeShared);
        weights_.emplace_back(L.up_b);
        const auto down = host(mm_.file, n("ffn_down.weight"), uint64_t(E) * FF);
        std::vector<float> dp(uint64_t(E) * ff_, 0.0f);
        for (uint32_t o = 0; o < E; ++o)
            std::copy_n(down.data() + uint64_t(o) * FF, FF, dp.data() + uint64_t(o) * ff_);
        L.down = f16(dp);
        L.down_b = weight(n("ffn_down.bias"), false);
        layers_.push_back(L);
    }
    pre_norm_ = weight("mm.pre_norm.weight", false);
    // The 2x2 convolution's kernel [out][c][dy][dx] as rows over a block's 4 consecutive tokens (j = 2dy + dx).
    const uint32_t M4 = 4 * E;
    const auto mw = host(mm_.file, "mm.0.weight", uint64_t(M4) * E * 4);
    std::vector<float> r(mw.size());
    for (uint32_t o = 0; o < M4; ++o)
        for (uint32_t c = 0; c < E; ++c)
            for (uint32_t j = 0; j < 4; ++j)
                r[(uint64_t(o) * 4 + j) * E + c] = mw[(uint64_t(o) * E + c) * 4 + j];
    merger_ = f16(r);
    merger_b_ = weight("mm.0.bias", false);
    mm2_ = weight("mm.2.weight");
    mm2_b_ = weight("mm.2.bias", false);
    newline_ = host(mm_.file, "v.image_newline", out_embd_);
}

Encoder::Tensor Encoder::f16(const std::vector<float>& v)
{
    auto* buf = ctx_.device()->newBuffer(std::max<size_t>(v.size() * 2, 16), MTL::ResourceStorageModeShared);
    weights_.emplace_back(buf);
    auto* dst = static_cast<uint16_t*>(buf->contents());
    for (size_t i = 0; i < v.size(); ++i)
        dst[i] = to_f16(v[i]);
    return buf;
}

Encoder::Tensor Encoder::upload(const gguf::Tensor& t, bool half)
{
    const auto data = mm_.file.data(t);
    if (data.empty())
        throw std::runtime_error(std::format("{}: truncated", t.name));
    if (!quant::can_dequant(t.type))
        throw std::runtime_error(std::format("{}: weight type {} not supported", t.name, gguf::traits(t.type).name));
    const uint64_t n = t.elements();
    auto* buf = ctx_.device()->newBuffer(n * (half ? 2 : 4), MTL::ResourceStorageModeShared);
    weights_.emplace_back(buf);
    if (half) {
        std::vector<float> v(n);
        quant::dequant_row(t.type, data.data(), v.data(), n);
        auto* dst = static_cast<uint16_t*>(buf->contents());
        for (uint64_t i = 0; i < n; ++i)
            dst[i] = to_f16(v[i]);
    } else {
        quant::dequant_row(t.type, data.data(), static_cast<float*>(buf->contents()), n);
    }
    return buf;
}

Encoder::Tensor Encoder::weight(std::string_view name, bool half, bool required)
{
    const gguf::Tensor* t = mm_.file.tensor(name);
    if (!t) {
        if (required)
            throw std::runtime_error(std::format("vision projector: no tensor {}", name));
        return nullptr;
    }
    return upload(*t, half);
}

MTL::Buffer* Encoder::scratch(const std::string& name, uint64_t bytes)
{
    auto& b = scratch_[name];
    if (!b || b->length() < bytes)
        b = mtl::Ref<MTL::Buffer>(
            ctx_.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
    return b.get();
}

MTL::ComputePipelineState* Encoder::pso(const std::string& name)
{
    auto it = psos_.find(name);
    if (it == psos_.end())
        it = psos_.emplace(name, ctx_.pipeline(lib_.get(), name)).first;
    return it->second.get();
}

void Encoder::linear(MTL::ComputeCommandEncoder* enc, Tensor w, uint32_t k, uint32_t n, uint32_t t, MTL::Buffer* xh,
                     uint64_t xh_off, MTL::Buffer* y, uint64_t y_off, uint32_t ldy)
{
    gemm_.encode(enc, gguf::Type::F16, k, n, t, w, 0, xh, xh_off, y, y_off, ldy);
}

void Encoder::rows_op(MTL::ComputeCommandEncoder* enc, const char* fn, std::initializer_list<MTL::Buffer*> bufs,
                      const void* args, size_t size, uint32_t n, uint32_t rows, int flag)
{
    enc->setComputePipelineState(pso(fn));
    NS::UInteger i = 0;
    for (MTL::Buffer* b : bufs)
        enc->setBuffer(b, 0, i++);
    enc->setBytes(args, size, i++);
    if (flag >= 0) {
        const uint32_t f = uint32_t(flag);
        enc->setBytes(&f, sizeof(f), i);
    }
    enc->dispatchThreads(MTL::Size(n, rows, 1), MTL::Size(std::min(n, 256u), 1, 1));
}

void Encoder::sync(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc, std::string_view name,
                   MTL::Buffer* buf, size_t n)
{
    if (!probe)
        return;
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    probe(name, static_cast<const float*>(buf->contents()), n);
    cb = ctx_.queue()->commandBuffer();
    enc = cb->computeCommandEncoder();
}

uint32_t Encoder::rows_for(const Mmproj& mm, const Prepared& img)
{
    const uint32_t p = mm.prep.patch, m = mm.prep.merge, px = img.width / p, py = img.height / p;
    if (mm.prep.kind == Projector::glm5v)
        return (px / 2) * (py / 2);
    if (mm.prep.kind == Projector::hy4f)
        return (py / 2) * (px / 2 + 1);
    const uint32_t cx = (px + m - 1) / m, cy = (py + m - 1) / m;
    return cy * (cx + 1) + 2;
}

uint32_t Encoder::cols_for(const Mmproj& mm, const Prepared& img)
{
    return mm.prep.kind == Projector::hy4f ? img.width / mm.prep.patch / 2 + 1 : 0;
}

Encoder::Output Encoder::encode(const Prepared& img)
{
    if (mm_.prep.kind == Projector::hy4f)
        return encode_hy4f(img);
    const bool glm = mm_.prep.kind == Projector::glm5v;
    const uint32_t P = mm_.prep.patch, px = img.width / P, py = img.height / P, T = px * py;
    const uint32_t E = mm_.n_embd, H = mm_.n_head, FF = mm_.n_ff, OUT = out_embd_;
    const uint32_t n_pad = (T + 127) / 128 * 128; // whole tiles of the attention kernel
    if (!T || img.width % P || img.height % P || (glm && (px % 2 || py % 2)) || img.chw.size() != 3ull * img.width * img.height)
        throw std::runtime_error("image not prepared for this projector");
    const PatchOrder order{px, glm ? 1u : 0u};
    auto* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Buffer* image = scratch("v_img", img.chw.size() * 4);
    std::memcpy(image->contents(), img.chw.data(), img.chw.size() * 4);
    MTL::Buffer* x = scratch("v_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("v_xn", uint64_t(T) * E * 4);
    MTL::Buffer* xh = scratch("v_xh", uint64_t(T) * std::max(E, FF) * 2);
    MTL::Buffer* qkv = scratch("v_qkv", uint64_t(T) * 3 * E * 4);
    MTL::Buffer* q = scratch("v_q", uint64_t(H) * n_pad * kHead * 2);
    MTL::Buffer* k = scratch("v_k", uint64_t(H) * n_pad * kHead * 2);
    MTL::Buffer* v = scratch("v_v", uint64_t(H) * n_pad * kHead * 2);
    MTL::Buffer* attn = scratch("v_attn", uint64_t(n_pad) * E * 4);
    MTL::Buffer* tmp = scratch("v_tmp", uint64_t(T) * E * 4);
    MTL::Buffer* g = scratch("v_g", uint64_t(T) * FF * 4);
    MTL::Buffer* u = scratch("v_u", uint64_t(T) * FF * 4);
    // Padding rows stay zero (the attention leaves those keys out; their queries' rows are not used).
    std::memset(q->contents(), 0, q->length());
    std::memset(k->contents(), 0, k->length());
    std::memset(v->contents(), 0, v->length());

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    {
        const PatchArgs a{img.width, img.height, P, E, T, order};
        enc->setComputePipelineState(pso("vit_patch_embed"));
        enc->setBuffer(image, 0, 0);
        enc->setBuffer(patch_w_, 0, 1);
        enc->setBuffer(patch_b_, 0, 2);
        enc->setBuffer(x, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    }
    sync(cb, enc, "patch_bias", x, uint64_t(T) * E);
    std::string stage_name = "patch";
    const auto stage = [&](const char* next) {
        if (!timing)
            return;
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        stage_ms[stage_name] += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        stage_name = next;
        cb = ctx_.queue()->commandBuffer();
        enc = cb->computeCommandEncoder();
    };
    const auto rms = [&](MTL::Buffer* in, Tensor w, MTL::Buffer* out, uint32_t rows, uint32_t n) {
        const NormArgs a{n, n, mm_.eps, 0};
        enc->setComputePipelineState(pso("rms_norm"));
        enc->setBuffer(in, 0, 0);
        enc->setBuffer(w, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    };
    const auto swiglu = [&](MTL::Buffer* gate, MTL::Buffer* up, MTL::Buffer* out, uint64_t n) {
        const SwigluArgs a{uint32_t(n), mm_.swiglu_clamp};
        enc->setComputePipelineState(pso("swiglu_half"));
        enc->setBuffer(gate, 0, 0);
        enc->setBuffer(up, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    };
    const auto residual = [&](MTL::Buffer* y, Tensor b, uint32_t rows, uint32_t n) {
        const RowsArgs a{n, rows};
        rows_op(enc, "vit_residual", {x, y, b ? b : y}, &a, sizeof(a), n, rows, b ? 1 : 0);
    };
    const auto bias = [&](MTL::Buffer* y, Tensor b, uint32_t rows, uint32_t n) {
        if (!b)
            return;
        const RowsArgs a{n, rows};
        rows_op(enc, "vit_bias", {y, b}, &a, sizeof(a), n, rows);
    };

    for (uint32_t il = 0; il < layers_.size(); ++il) {
        const Layer& L = layers_[il];
        stage("qkv");
        rms(x, L.ln1, xn, T, E);
        gemm_.to_half(enc, xn, 0, T, E, 0, xh, 0);
        linear(enc, L.qkv, E, 3 * E, T, xh, 0, qkv);
        {
            const QkvArgs a{T, n_pad, H, mm_.eps, L.qn ? 1u : 0u, kRopeTheta, order};
            enc->setComputePipelineState(pso("vit_qkv_rope"));
            enc->setBuffer(qkv, 0, 0);
            enc->setBuffer(L.qkv_b, 0, 1);
            enc->setBuffer(L.qn ? L.qn : L.qkv_b, 0, 2);
            enc->setBuffer(L.kn ? L.kn : L.qkv_b, 0, 3);
            enc->setBuffer(q, 0, 4);
            enc->setBuffer(k, 0, 5);
            enc->setBuffer(v, 0, 6);
            enc->setBytes(&a, sizeof(a), 7);
            enc->dispatchThreads(MTL::Size(32, uint64_t(T) * H * 3, 1), MTL::Size(32, 1, 1));
        }
        stage("attention");
        {
            const FlashArgs fa{T, n_pad, E, 1.0f / 8.0f};
            enc->setComputePipelineState(pso("vit_flash_attn"));
            enc->setBuffer(q, 0, 0);
            enc->setBuffer(k, 0, 1);
            enc->setBuffer(v, 0, 2);
            enc->setBuffer(attn, 0, 3);
            enc->setBytes(&fa, sizeof(fa), 4);
            enc->dispatchThreadgroups(MTL::Size(n_pad / 32, H, 1), MTL::Size(32, 8, 1));
        }
        stage("out_ffn");
        gemm_.to_half(enc, attn, 0, T, E, 0, xh, 0);
        linear(enc, L.out, E, E, T, xh, 0, tmp);
        residual(tmp, L.out_b, T, E);
        rms(x, L.ln2, xn, T, E);
        gemm_.to_half(enc, xn, 0, T, E, 0, xh, 0);
        linear(enc, L.gate, E, FF, T, xh, 0, g);
        linear(enc, L.up, E, FF, T, xh, 0, u);
        bias(g, L.gate_b, T, FF);
        bias(u, L.up_b, T, FF);
        swiglu(g, u, xh, uint64_t(T) * FF);
        linear(enc, L.down, FF, E, T, xh, 0, tmp);
        residual(tmp, L.down_b, T, E);
        if (probe)
            sync(cb, enc, std::format("layer_out-{}", il), x, uint64_t(T) * E);
    }
    stage("merge");
    rms(x, post_ln_, xn, T, E);
    sync(cb, enc, "vit_out", xn, uint64_t(T) * E);

    Output out;
    MTL::Buffer* y = scratch("v_y", uint64_t(T) * OUT * 4);
    if (!glm) {
        const uint32_t m = mm_.prep.merge, cx = (px + m - 1) / m, cy = (py + m - 1) / m, cells = cx * cy;
        const UnfoldArgs ua{px, py, cx, cells, E};
        MTL::Buffer* cols = scratch("v_cols", uint64_t(cells) * E * 9 * 2);
        rows_op(enc, "vit_unfold3", {xn, cols}, &ua, sizeof(ua), E * 9, cells);
        MTL::Buffer* h1 = scratch("v_h1", uint64_t(cells) * OUT * 4);
        MTL::Buffer* h1h = scratch("v_h1h", uint64_t(cells) * OUT * 2);
        linear(enc, mm1_, E * 9, OUT, cells, cols, 0, h1);
        const RowsArgs ra{OUT, cells};
        rows_op(enc, "vit_gelu", {h1, mm1_b_, h1h}, &ra, sizeof(ra), OUT, cells, 1);
        linear(enc, mm2_, OUT, OUT, cells, h1h, 0, y);
        bias(y, mm2_b_, cells, OUT);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (cb->status() != MTL::CommandBufferStatusCompleted)
            throw std::runtime_error("vision encoder: command buffer failed");
        if (timing)
            stage_ms[stage_name] += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        if (probe)
            probe("projected", static_cast<const float*>(y->contents()), uint64_t(cells) * OUT);
        // [start] (row of cells, newline)* [end]
        out.cells_x = cx;
        out.cells_y = cy;
        out.rows = cy * (cx + 1) + 2;
        out.embd.reserve(uint64_t(out.rows) * OUT);
        const auto* rows = static_cast<const float*>(y->contents());
        out.embd.insert(out.embd.end(), start_.begin(), start_.end());
        for (uint32_t r = 0; r < cy; ++r) {
            out.embd.insert(out.embd.end(), rows + uint64_t(r) * cx * OUT, rows + uint64_t(r + 1) * cx * OUT);
            out.embd.insert(out.embd.end(), newline_.begin(), newline_.end());
        }
        out.embd.insert(out.embd.end(), end_.begin(), end_.end());
    } else {
        const uint32_t cells = T / 4;
        // A 2x2 block's 4 features are 4 consecutive rows: one row of 4 * E halves.
        gemm_.to_half(enc, xn, 0, T, E, 0, xh, 0);
        MTL::Buffer* m1 = scratch("v_m1", uint64_t(cells) * OUT * 4);
        MTL::Buffer* m1h = scratch("v_m1h", uint64_t(cells) * std::max(OUT, proj_ff_) * 2);
        linear(enc, merger_, 4 * E, OUT, cells, xh, 0, m1);
        bias(m1, merger_b_, cells, OUT);
        sync(cb, enc, "merged", m1, uint64_t(cells) * OUT);
        gemm_.to_half(enc, m1, 0, cells, OUT, 0, m1h, 0);
        MTL::Buffer* f = scratch("v_f", uint64_t(cells) * OUT * 4);
        linear(enc, fc_, OUT, OUT, cells, m1h, 0, f);
        {
            const LayerNormArgs la{OUT, kProjectorNormEps, OUT};
            enc->setComputePipelineState(pso("layer_norm"));
            enc->setBuffer(f, 0, 0);
            enc->setBuffer(norm_w_, 0, 1);
            enc->setBuffer(norm_b_, 0, 2);
            enc->setBuffer(m1, 0, 3);
            enc->setBytes(&la, sizeof(la), 4);
            enc->dispatchThreadgroups(MTL::Size(cells, 1, 1), MTL::Size(256, 1, 1));
        }
        const RowsArgs ra{OUT, cells};
        rows_op(enc, "vit_gelu", {m1, m1, m1h}, &ra, sizeof(ra), OUT, cells, 0);
        MTL::Buffer* pg = scratch("v_pg", uint64_t(cells) * proj_ff_ * 4);
        MTL::Buffer* pu = scratch("v_pu", uint64_t(cells) * proj_ff_ * 4);
        linear(enc, pgate_, OUT, proj_ff_, cells, m1h, 0, pg);
        linear(enc, pup_, OUT, proj_ff_, cells, m1h, 0, pu);
        MTL::Buffer* ph = scratch("v_ph", uint64_t(cells) * proj_ff_ * 2);
        swiglu(pg, pu, ph, uint64_t(cells) * proj_ff_);
        linear(enc, pdown_, proj_ff_, OUT, cells, ph, 0, y);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (cb->status() != MTL::CommandBufferStatusCompleted)
            throw std::runtime_error("vision encoder: command buffer failed");
        if (timing)
            stage_ms[stage_name] += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        out.cells_x = px / 2;
        out.cells_y = py / 2;
        out.rows = cells;
        const auto* rows = static_cast<const float*>(y->contents());
        out.embd.assign(rows, rows + uint64_t(cells) * OUT);
    }
    pool->release();
    return out;
}

// SigLIP-shaped: the patch embedding plus the position grid resampled to the patches (bilinear, scale (n + 0.1) /
// grid), pre-LayerNorm layers (biased q, k, v and output, exact GELU MLP), then an RMSNorm, the 2x2 convolution,
// exact GELU, the 1x1 convolution and a newline row after each row of cells. The ViT runs on the patches in 2x2
// block order (attention does not depend on the order), so a block's 4 rows are the convolution's input row.
Encoder::Output Encoder::encode_hy4f(const Prepared& img)
{
    const uint32_t P = mm_.prep.patch, px = img.width / P, py = img.height / P, T = px * py;
    const uint32_t E = mm_.n_embd, H = mm_.n_head, FF = ff_, OUT = out_embd_, HP = H * kHy4Pad;
    const uint32_t n_pad = (T + 127) / 128 * 128;
    if (!T || img.width % P || img.height % P || px % 2 || py % 2 || img.chw.size() != 3ull * img.width * img.height)
        throw std::runtime_error("image not prepared for this projector");
    const PatchOrder order{px, 1u};
    auto* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Buffer* image = scratch("v_img", img.chw.size() * 4);
    std::memcpy(image->contents(), img.chw.data(), img.chw.size() * 4);
    // The position grid at each patch (in block order).
    MTL::Buffer* pos = scratch("v_pos", uint64_t(T) * E * 4);
    {
        auto* dst = static_cast<float*>(pos->contents());
        const uint32_t n = grid_;
        // Source coordinate of output index i for an output of `out` cells: (i + 0.5) / scale - 0.5, from 0.
        const auto src = [n](uint32_t i, uint32_t out) {
            const float scale = (float(out) + 0.1f) / float(n);
            const float f = std::max(0.0f, (float(i) + 0.5f) / scale - 0.5f);
            const uint32_t i0 = std::min(uint32_t(f), n - 1), i1 = std::min(i0 + 1, n - 1);
            return std::tuple{i0, i1, f - float(i0)};
        };
        const uint32_t bw = px / 2;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t b = t / 4, j = t % 4, x = (b % bw) * 2 + (j & 1), y = (b / bw) * 2 + (j >> 1);
            const auto [y0, y1, fy] = src(y, py);
            const auto [x0, x1, fx] = src(x, px);
            const float* s00 = pos_grid_.data() + (uint64_t(y0) * n + x0) * E;
            const float* s01 = pos_grid_.data() + (uint64_t(y0) * n + x1) * E;
            const float* s10 = pos_grid_.data() + (uint64_t(y1) * n + x0) * E;
            const float* s11 = pos_grid_.data() + (uint64_t(y1) * n + x1) * E;
            float* d = dst + uint64_t(t) * E;
            for (uint32_t c = 0; c < E; ++c)
                d[c] = (1.0f - fy) * ((1.0f - fx) * s00[c] + fx * s01[c]) + fy * ((1.0f - fx) * s10[c] + fx * s11[c]);
        }
    }
    MTL::Buffer* x = scratch("v_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("v_xn", uint64_t(T) * E * 4);
    MTL::Buffer* xh = scratch("v_xh", uint64_t(T) * std::max({E * 4, FF, HP}) * 2);
    MTL::Buffer* qkv = scratch("v_qkv", uint64_t(T) * 3 * E * 4);
    MTL::Buffer* q = scratch("v_q", uint64_t(H) * n_pad * kHy4Pad * 2);
    MTL::Buffer* k = scratch("v_k", uint64_t(H) * n_pad * kHy4Pad * 2);
    MTL::Buffer* v = scratch("v_v", uint64_t(H) * n_pad * kHy4Pad * 2);
    MTL::Buffer* attn = scratch("v_attn", uint64_t(n_pad) * HP * 4);
    MTL::Buffer* tmp = scratch("v_tmp", uint64_t(T) * E * 4);
    MTL::Buffer* g = scratch("v_g", uint64_t(T) * std::max(FF, 4 * E) * 4);
    std::memset(q->contents(), 0, q->length());
    std::memset(k->contents(), 0, k->length());
    std::memset(v->contents(), 0, v->length());

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    std::string stage_name = "patch";
    const auto stage = [&](const char* next) {
        if (!timing)
            return;
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        stage_ms[stage_name] += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        stage_name = next;
        cb = ctx_.queue()->commandBuffer();
        enc = cb->computeCommandEncoder();
    };
    const auto residual = [&](MTL::Buffer* y, Tensor b, uint32_t n) {
        const RowsArgs a{n, T};
        rows_op(enc, "vit_residual", {x, y, b ? b : y}, &a, sizeof(a), n, T, b ? 1 : 0);
    };
    const auto layer_norm = [&](Tensor w, Tensor b) {
        const LayerNormArgs a{E, mm_.eps, E};
        enc->setComputePipelineState(pso("layer_norm"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(w, 0, 1);
        enc->setBuffer(b, 0, 2);
        enc->setBuffer(xn, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
    };
    {
        const PatchArgs a{img.width, img.height, P, E, T, order};
        enc->setComputePipelineState(pso("vit_patch_embed"));
        enc->setBuffer(image, 0, 0);
        enc->setBuffer(patch_w_, 0, 1);
        enc->setBuffer(patch_b_, 0, 2);
        enc->setBuffer(x, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    }
    residual(pos, nullptr, E);
    sync(cb, enc, "patch_bias", x, uint64_t(T) * E);
    for (uint32_t il = 0; il < layers_.size(); ++il) {
        const Layer& L = layers_[il];
        stage("qkv");
        layer_norm(L.ln1, L.ln1_b);
        gemm_.to_half(enc, xn, 0, T, E, 0, xh, 0);
        linear(enc, L.qkv, E, 3 * E, T, xh, 0, qkv);
        {
            const SplitQkvArgs a{T, n_pad, H, kHy4Head, kHy4Pad};
            enc->setComputePipelineState(pso("vit_qkv_split"));
            enc->setBuffer(qkv, 0, 0);
            enc->setBuffer(L.qkv_b, 0, 1);
            enc->setBuffer(q, 0, 2);
            enc->setBuffer(k, 0, 3);
            enc->setBuffer(v, 0, 4);
            enc->setBytes(&a, sizeof(a), 5);
            enc->dispatchThreads(MTL::Size(kHy4Pad, H * 3, T), MTL::Size(kHy4Pad, 1, 1));
        }
        stage("attention");
        {
            const FlashArgs fa{T, n_pad, HP, 1.0f / std::sqrt(float(kHy4Head))};
            enc->setComputePipelineState(pso("vit_flash_attn80"));
            enc->setBuffer(q, 0, 0);
            enc->setBuffer(k, 0, 1);
            enc->setBuffer(v, 0, 2);
            enc->setBuffer(attn, 0, 3);
            enc->setBytes(&fa, sizeof(fa), 4);
            enc->dispatchThreadgroups(MTL::Size(n_pad / 32, H, 1), MTL::Size(32, 8, 1));
        }
        stage("out_ffn");
        gemm_.to_half(enc, attn, 0, T, HP, 0, xh, 0);
        linear(enc, L.out, HP, E, T, xh, 0, tmp);
        residual(tmp, L.out_b, E);
        layer_norm(L.ln2, L.ln2_b);
        gemm_.to_half(enc, xn, 0, T, E, 0, xh, 0);
        linear(enc, L.up, E, FF, T, xh, 0, g);
        {
            const RowsArgs ra{FF, T};
            rows_op(enc, "vit_gelu", {g, L.up_b, xh}, &ra, sizeof(ra), FF, T, 1);
        }
        linear(enc, L.down, FF, E, T, xh, 0, tmp);
        residual(tmp, L.down_b, E);
        if (probe)
            sync(cb, enc, std::format("layer_out-{}", il), x, uint64_t(T) * E);
    }
    stage("merge");
    {
        const NormArgs a{E, E, mm_.eps, 0};
        enc->setComputePipelineState(pso("rms_norm"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(pre_norm_, 0, 1);
        enc->setBuffer(xn, 0, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
    }
    sync(cb, enc, "vit_out", xn, uint64_t(T) * E);
    const uint32_t cells = T / 4, M4 = 4 * E;
    gemm_.to_half(enc, xn, 0, T, E, 0, xh, 0);
    linear(enc, merger_, M4, M4, cells, xh, 0, g);
    {
        const RowsArgs ra{M4, cells};
        rows_op(enc, "vit_gelu", {g, merger_b_, xh}, &ra, sizeof(ra), M4, cells, 1);
    }
    MTL::Buffer* y = scratch("v_y", uint64_t(cells) * OUT * 4);
    linear(enc, mm2_, M4, OUT, cells, xh, 0, y);
    {
        const RowsArgs ra{OUT, cells};
        rows_op(enc, "vit_bias", {y, mm2_b_}, &ra, sizeof(ra), OUT, cells);
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted)
        throw std::runtime_error("vision encoder: command buffer failed");
    if (timing)
        stage_ms[stage_name] += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
    if (probe)
        probe("projected", static_cast<const float*>(y->contents()), uint64_t(cells) * OUT);
    Output out;
    out.cells_x = px / 2;
    out.cells_y = py / 2;
    out.rows = out.cells_y * (out.cells_x + 1);
    out.embd.reserve(uint64_t(out.rows) * OUT);
    const auto* rows = static_cast<const float*>(y->contents());
    for (uint32_t r = 0; r < out.cells_y; ++r) {
        out.embd.insert(out.embd.end(), rows + uint64_t(r) * out.cells_x * OUT, rows + uint64_t(r + 1) * out.cells_x * OUT);
        out.embd.insert(out.embd.end(), newline_.begin(), newline_.end());
    }
    pool->release();
    return out;
}

} // namespace mako::vision
