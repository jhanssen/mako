// Vision encoder kernels (both projector types). Activations are fp32 rows [token][width] unless noted.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Patch grid order: raster, or 2x2 blocks in raster order of the block grid, each block (x, y), (x+1, y), (x, y+1),
// (x+1, y+1).
struct PatchOrder {
    uint px;     // patches per row
    uint blocks; // 0: raster, 1: 2x2 blocks
};

static uint2 patch_xy(uint t, PatchOrder o)
{
    if (o.blocks == 0)
        return uint2(t % o.px, t / o.px);
    const uint b = t / 4, j = t % 4, bw = o.px / 2;
    return uint2((b % bw) * 2 + (j & 1), (b / bw) * 2 + (j >> 1));
}

struct PatchArgs {
    uint width, height; // image, pixels
    uint patch;
    uint n_embd;
    uint n_patches;
    PatchOrder order;
};

// y[t][o] = b[o] + sum_{c,kh,kw} w[o][c][kh][kw] * img[c][py*patch + kh][px*patch + kw], img planar [3][height][width];
// grid (n_embd, n_patches).
kernel void vit_patch_embed(device const float* img [[buffer(0)]], device const float* w [[buffer(1)]],
                            device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                            constant PatchArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint o = gid.x, t = gid.y;
    if (o >= a.n_embd || t >= a.n_patches)
        return;
    const uint2 p = patch_xy(t, a.order);
    const uint plane = a.width * a.height, k = a.patch * a.patch;
    device const float* wo = w + uint64_t(o) * 3 * k;
    float acc = 0.0f;
    for (uint c = 0; c < 3; ++c)
        for (uint kh = 0; kh < a.patch; ++kh) {
            device const float* row = img + uint64_t(c) * plane + uint64_t(p.y * a.patch + kh) * a.width + p.x * a.patch;
            device const float* wr = wo + c * k + kh * a.patch;
            for (uint kw = 0; kw < a.patch; ++kw)
                acc += wr[kw] * row[kw];
        }
    y[uint64_t(t) * a.n_embd + o] = acc + b[o];
}

struct QkvArgs {
    uint n_tok, n_pad; // tokens, and the key count rounded up for the attention GEMMs
    uint heads;
    float eps;       // per-head q/k RMS norm
    uint qk_norm;    // 0: none
    float theta;     // RoPE base
    PatchOrder order;
};

// One simdgroup per (token, head, part): adds the QKV bias, RMS-normalises q and k per head (with weights) when
// asked, applies the 2D RoPE (lane k < 16 rotates dims k, k+32 by the patch row times theta^(-k/16), lanes 16..31
// by the column times theta^(-(k-16)/16)), and writes half q, k and v, each [head][n_pad][64]. qkv rows hold q, k,
// v of all heads, 64 dims a head. Grid (32, n_tok * heads * 3).
kernel void vit_qkv_rope(device const float* qkv [[buffer(0)]], device const float* bias [[buffer(1)]],
                         device const float* qn [[buffer(2)]], device const float* kn [[buffer(3)]],
                         device half* q [[buffer(4)]], device half* k [[buffer(5)]], device half* vt [[buffer(6)]],
                         constant QkvArgs& a [[buffer(7)]], uint2 gid [[thread_position_in_grid]])
{
    const uint lane = gid.x, job = gid.y;
    if (job >= a.n_tok * a.heads * 3)
        return;
    const uint part = job % 3, h = (job / 3) % a.heads, t = job / (3 * a.heads);
    const uint width = a.heads * 64, off = part * width + h * 64;
    device const float* src = qkv + uint64_t(t) * 3 * width + off;
    float x0 = src[lane] + bias[off + lane], x1 = src[lane + 32] + bias[off + lane + 32];
    if (part == 2) {
        device half* dst = vt + (uint64_t(h) * a.n_pad + t) * 64;
        dst[lane] = half(x0);
        dst[lane + 32] = half(x1);
        return;
    }
    if (a.qk_norm) {
        const float inv = rsqrt(simd_sum(x0 * x0 + x1 * x1) / 64.0f + a.eps);
        device const float* w = part == 0 ? qn : kn;
        x0 = x0 * inv * w[lane];
        x1 = x1 * inv * w[lane + 32];
    }
    const uint2 p = patch_xy(t, a.order);
    const float pos = lane < 16 ? float(p.y) : float(p.x);
    const float angle = pos * pow(a.theta, -float(lane % 16) / 16.0f);
    const float c = cos(angle), s = sin(angle);
    const float r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
    device half* dst = (part == 0 ? q : k) + (uint64_t(h) * a.n_pad + t) * 64;
    dst[lane] = half(r0);
    dst[lane + 32] = half(r1);
}

struct RowsArgs {
    uint n;    // row length
    uint rows;
};

// y[r][i] += b[i]; grid (n, rows).
kernel void vit_bias(device float* y [[buffer(0)]], device const float* b [[buffer(1)]],
                     constant RowsArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.n && gid.y < a.rows)
        y[uint64_t(gid.y) * a.n + gid.x] += b[gid.x];
}

// x[r][i] += y[r][i] (+ b[i] when given: has_bias); grid (n, rows).
kernel void vit_residual(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                         device const float* b [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                         constant uint& has_bias [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.n || gid.y >= a.rows)
        return;
    const uint64_t i = uint64_t(gid.y) * a.n + gid.x;
    x[i] += y[i] + (has_bias ? b[gid.x] : 0.0f);
}

// erf, max error ~1.2e-7 (Chebyshev fit of erfc).
static float erf_approx(float x)
{
    const float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
    const float r = t * exp(-z * z - 1.26551223f +
                            t * (1.00002368f +
                                 t * (0.37409196f +
                                      t * (0.09678418f +
                                           t * (-0.18628806f +
                                                t * (0.27886807f +
                                                     t * (-1.13520398f +
                                                          t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

// y = gelu(x + b) with the erf form, as half; x [rows][n]; grid (n, rows).
kernel void vit_gelu(device const float* x [[buffer(0)]], device const float* b [[buffer(1)]],
                     device half* y [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                     constant uint& has_bias [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.n || gid.y >= a.rows)
        return;
    const uint64_t i = uint64_t(gid.y) * a.n + gid.x;
    const float v = x[i] + (has_bias ? b[gid.x] : 0.0f);
    y[i] = half(0.5f * v * (1.0f + erf_approx(v * 0.70710678f)));
}

struct UnfoldArgs {
    uint px, py;    // patch grid
    uint cells_x;   // ceil(px / 3)
    uint n_cells;
    uint n_embd;
};

// The 3x3 merge: cell (cx, cy) gets y[cell][c * 9 + kh * 3 + kw] = x[(3cy + kh) * px + 3cx + kw][c], 0 outside the
// grid, as half; x raster order. Grid (n_embd * 9, n_cells).
kernel void vit_unfold3(device const float* x [[buffer(0)]], device half* y [[buffer(1)]],
                        constant UnfoldArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, cell = gid.y;
    if (i >= a.n_embd * 9 || cell >= a.n_cells)
        return;
    const uint c = i / 9, kh = (i % 9) / 3, kw = i % 3;
    const uint px = (cell % a.cells_x) * 3 + kw, py = (cell / a.cells_x) * 3 + kh;
    const float v = px < a.px && py < a.py ? x[uint64_t(py * a.px + px) * a.n_embd + c] : 0.0f;
    y[uint64_t(cell) * a.n_embd * 9 + i] = half(v);
}


struct FlashArgs {
    uint n_tok;  // valid keys
    uint n_pad;  // rows of q, k, v a head (a multiple of 128)
    uint width;  // o row length (heads * 64)
    float scale;
};

// o[q][h * 64 + d] = sum_j softmax_j(scale * q . k_j) v_j[d] over keys j < n_tok, for a block of 32 queries of head
// h; q, k, v half [head][n_pad][64], n_pad a multiple of 128. The scores of 128 keys at a time go to threadgroup
// memory for the running softmax (eight threads a row), the output stays in registers (rescaled when a row's
// maximum grows) and both products run through matmul2d with float accumulation. Grid (n_pad / 32, heads), 8
// simdgroups.
kernel void vit_flash_attn(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                           device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                           constant FlashArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]])
{
    constexpr int B = 32, K = 128;
    threadgroup float s[B * K];
    threadgroup half p[B * K];
    threadgroup float row_scale[B];
    const uint q0 = tg.x * B, h = tg.y;
    const uint64_t head = uint64_t(h) * a.n_pad * 64;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + head + uint64_t(q0) * 64,
                                                                dextents<int32_t, 2>(64, B));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device half*>(k) + head,
                                                                dextents<int32_t, 2>(64, int(a.n_pad)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(const_cast<device half*>(v) + head,
                                                                dextents<int32_t, 2>(64, int(a.n_pad)));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> ts(s, dextents<int32_t, 2>(K, B));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(p, dextents<int32_t, 2>(K, B));
    constexpr auto qk_desc = matmul2d_descriptor(B, K, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(B, 64, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<8>> qk;
    matmul2d<pv_desc, execution_simdgroups<8>> pv;
    auto kt0 = tk.slice(0, 0);
    auto sc = qk.get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto vt0 = tv.slice(0, 0);
    auto acc = pv.get_destination_cooperative_tensor<decltype(tp), decltype(vt0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    // Eight threads a row, 16 columns each; each keeps the row's running maximum and sum.
    const uint r = tid / 8, c0 = (tid % 8) * 16;
    float m = -INFINITY, l = 0.0f;
    for (uint j0 = 0; j0 < a.n_tok; j0 += K) {
        for (uint16_t i = 0; i < sc.get_capacity(); ++i)
            if (sc.is_valid_element(i))
                sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        qk.run(tq, kt, sc);
        sc.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[16], mt = -INFINITY;
        for (uint c = 0; c < 16; ++c) {
            x[c] = j0 + c0 + c < a.n_tok ? s[r * K + c0 + c] * a.scale : -INFINITY;
            mt = max(mt, x[c]);
        }
        mt = max(mt, simd_shuffle_xor(mt, 1));
        mt = max(mt, simd_shuffle_xor(mt, 2));
        mt = max(mt, simd_shuffle_xor(mt, 4));
        const float mn = max(m, mt), f = exp(m - mn);
        float sum = 0.0f;
        for (uint c = 0; c < 16; ++c) {
            const float e = exp(x[c] - mn);
            p[r * K + c0 + c] = half(e);
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 4);
        l = l * f + sum;
        m = mn;
        if (tid % 8 == 0)
            row_scale[r] = f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i))
                acc[i] *= row_scale[acc.get_multidimensional_index(i)[1]];
        auto vt = tv.slice(0, int(j0));
        pv.run(tp, vt, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid % 8 == 0)
        row_scale[r] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] *= row_scale[acc.get_multidimensional_index(i)[1]];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * 64,
                                                                 dextents<int32_t, 2>(int(a.width), B));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}
