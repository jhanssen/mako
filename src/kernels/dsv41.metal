// DeepSeek-V4.1 kernels: partial RoPE, the FP8 / FP4 activation roundings, the compressor, the sparse attention
// with its sink, the indexer and Engram. Activations are fp32.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Nearest E4M3 value (ties to even), saturating at 448; a must be >= 0.
static float e4m3_mag(float a)
{
    a = min(a, 448.0f);
    if (a < 0.015625f) // subnormal: steps of 2^-9
        return rint(a * 512.0f) / 512.0f;
    int e;
    frexp(a, e); // a = m * 2^e, m in [0.5, 1)
    const float step = ldexp(1.0f, e - 4);
    return min(rint(a / step) * step, 448.0f);
}

// ceil(log2(r)) for r > 0, exact at powers of two.
static int ceil_log2(float r)
{
    int e;
    const float m = frexp(r, e);
    return m == 0.5f ? e - 1 : e;
}

// Nearest E2M1 value (ties to even), a in [0, 6].
static float e2m1_mag(float a)
{
    if (a < 2.0f)
        return rint(a * 2.0f) * 0.5f;
    if (a < 4.0f)
        return rint(a);
    return min(rint(a * 0.5f) * 2.0f, 6.0f);
}

struct RopeArgs {
    uint head_dim; // row of one head
    uint heads;
    uint n_rot;    // the head's last n_rot dims (adjacent pairs)
    uint stride;   // floats between tokens
    uint pos0, pos_step;
    uint inverse;
};

// Rotates the last n_rot dims of each head of each token (token t at position pos0 + t * pos_step) by the angles
// pos * freq[i], pairs (2i, 2i + 1) as complex numbers; inverse: by the conjugate. Grid (n_rot / 2, heads, tokens).
kernel void ds_rope(device float* x [[buffer(0)]], device const float* freq [[buffer(1)]],
                    constant RopeArgs& a [[buffer(2)]], uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, h = gid.y, t = gid.z;
    if (i >= a.n_rot / 2 || h >= a.heads)
        return;
    device float* p = x + uint64_t(t) * a.stride + uint64_t(h) * a.head_dim + (a.head_dim - a.n_rot) + 2 * i;
    const float ang = float(a.pos0 + t * a.pos_step) * freq[i];
    const float c = precise::cos(ang), s = a.inverse ? -precise::sin(ang) : precise::sin(ang);
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

struct RoundArgs {
    uint n;       // values (a multiple of 32)
    uint kind;    // 0: FP8 E4M3, power-of-two scale per 32; 1: FP4 E2M1, E4M3 scale per 16; 2: FP4, power-of-two per 32
    uint out_half; // write y as half (else float)
};

// y = the value x takes after the rounding (in place when y is x). One simdgroup per 32 values.
kernel void ds_round(device float* x [[buffer(0)]], device void* y [[buffer(1)]], constant RoundArgs& a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    if (gid >= a.n)
        return;
    const float v = x[gid];
    const float av = fabs(v);
    float q;
    if (a.kind == 0) {
        const float amax = max(simd_max(av), 1e-4f);
        const float s = ldexp(1.0f, ceil_log2(amax / 448.0f));
        q = e4m3_mag(av / s) * s;
    } else if (a.kind == 1) {
        // blocks of 16: the two halves of the simdgroup
        float m = av;
        for (uint o = 1; o < 16; o <<= 1)
            m = max(m, simd_shuffle_xor(m, o));
        const float s = e4m3_mag(max(m, 6.0f * 0.001953125f) / 6.0f);
        q = e2m1_mag(min(av / s, 6.0f)) * s;
    } else {
        const float amax = max(simd_max(av), 6.0f * 1.1754943508222875e-38f);
        const float s = ldexp(1.0f, ceil_log2(amax / 6.0f));
        q = e2m1_mag(min(av / s, 6.0f)) * s;
    }
    q = copysign(q, v);
    if (a.out_half)
        static_cast<device half*>(y)[gid] = half(q);
    else
        static_cast<device float*>(y)[gid] = q;
    (void)lane;
}

struct PoolArgs {
    uint dim;
    uint tokens;
    uint pos0; // position of the first token
    uint hist; // the first this many tokens also leave the state after them in hist (rows of k then g)
};

// Ratio-2 compressor over consecutive tokens: per channel, a group's two tokens (positions 2j, 2j + 1) give row j,
// the kv projections weighted by the softmax over the two of the gate. state holds the pending first token's k then
// g (from before pos0, or left for the next call). Rows go to y in order, one per completed group. Grid (dim).
// (After an odd position the state is stale and unused; hist keeps it as it was.)
kernel void ds_pool2(device const float* k [[buffer(0)]], device const float* g [[buffer(1)]],
                     device float* state [[buffer(2)]], device float* y [[buffer(3)]], constant PoolArgs& a [[buffer(4)]],
                     device float* hist [[buffer(5)]], uint c [[thread_position_in_grid]])
{
    if (c >= a.dim)
        return;
    float k0 = state[c], g0 = state[a.dim + c];
    uint row = 0;
    for (uint t = 0; t < a.tokens; ++t) {
        const float kt = k[uint64_t(t) * a.dim + c], gt = g[uint64_t(t) * a.dim + c];
        if (t < a.hist) { // the state after token t, should the tokens after it be undone
            hist[uint64_t(t) * 2 * a.dim + c] = (a.pos0 + t) % 2 == 0 ? kt : k0;
            hist[(uint64_t(t) * 2 + 1) * a.dim + c] = (a.pos0 + t) % 2 == 0 ? gt : g0;
        }
        if ((a.pos0 + t) % 2 == 0) {
            k0 = kt;
            g0 = gt;
            continue;
        }
        const float m = max(g0, gt);
        const float e0 = exp(g0 - m), e1 = exp(gt - m);
        y[uint64_t(row++) * a.dim + c] = (k0 * e0 + kt * e1) / (e0 + e1);
    }
    state[c] = k0;
    state[a.dim + c] = g0;
}

struct AttnArgs {
    uint dim;     // 512
    uint heads;   // 64
    uint stride;  // score row stride (keys per head) of a token
    uint ring;    // window rows (sliding window)
    uint pos0;    // position of the chunk's first token (its rows of kv_chunk)
    uint t0;      // first token (of the chunk) this dispatch covers
    uint ratio;   // compressed rows: one per `ratio` tokens; 0: none
    uint top;     // at most this many compressed rows; more visible: the selection's
    float scale;
    uint block;   // nonzero: a draft block of this many rows (see keys_of)
};

// Token t's keys: its window (positions max(0, p - ring + 1) .. p, from the chunk's rows at or after pos0, else the
// ring), then its compressed rows (all visible ones while at most `top`, else sel's). A draft block's rows instead
// all see the ring's positions before pos0 and every row of the block (the chunk's).
struct Keys {
    uint p, nw, first, nc;
    bool identity, block;
};

static Keys keys_of(constant AttnArgs& a, uint t)
{
    Keys k;
    if (a.block) {
        k.p = a.pos0 + t;
        k.nw = min(a.pos0, a.ring);
        k.first = a.pos0 - k.nw;
        k.nc = a.block;
        k.identity = true;
        k.block = true;
        return k;
    }
    k.block = false;
    k.p = a.pos0 + t;
    k.nw = min(k.p + 1, a.ring);
    k.first = k.p + 1 - k.nw;
    const uint n_comp = a.ratio ? (k.p + 1) / a.ratio : 0;
    k.nc = min(n_comp, a.top);
    k.identity = n_comp <= a.top;
    return k;
}

// Where key j of token t lives: kind 0 the chunk's rows, 1 the ring, 2 the compressed rows (half), 3 none (zeros);
// row its row there.
static uint2 key_row(constant AttnArgs& a, Keys kk, uint t, uint j, device const int* sel)
{
    if (j < kk.nw) {
        const uint pos = kk.first + j;
        return pos >= a.pos0 ? uint2(0, pos - a.pos0) : uint2(1, pos % a.ring);
    }
    if (j < kk.nw + kk.nc) {
        const uint c = j - kk.nw;
        if (kk.block)
            return uint2(0, c);
        return uint2(2, kk.identity ? c : uint(sel[uint64_t(t) * a.top + c]));
    }
    return uint2(3, 0);
}

static float4 key4(uint2 kr, uint d4, uint dim, device const float* chunk, device const float* ringb,
                   device const half* comp)
{
    switch (kr.x) {
    case 0: return reinterpret_cast<device const float4*>(chunk + uint64_t(kr.y) * dim)[d4];
    case 1: return reinterpret_cast<device const float4*>(ringb + uint64_t(kr.y) * dim)[d4];
    case 2: return float4(reinterpret_cast<device const half4*>(comp + uint64_t(kr.y) * dim)[d4]);
    default: return float4(0.0f);
    }
}

// scores[t][h][j] = q_t,h . key_j(t) (the scale goes in the softmax): 64 heads x 32 keys a threadgroup of 4
// simdgroups through matmul2d, the keys staged as half in threadgroup memory 256 dims at a time, half q, float
// accumulation. Grid (stride / 32, tokens); heads 64, dim a multiple of 256.
kernel void ds_attn_scores(device const half* q [[buffer(0)]], device const float* chunk [[buffer(1)]],
                           device const float* ringb [[buffer(2)]], device const half* comp [[buffer(3)]],
                           device const int* sel [[buffer(4)]], device float* scores [[buffer(5)]],
                           constant AttnArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]])
{
    constexpr int GN = 32, GK = 256;
    threadgroup half ws[GN * GK];
    threadgroup uint2 rows[GN];
    const uint t = a.t0 + tg.y, j0 = tg.x * GN;
    const Keys kk = keys_of(a, t);
    if (j0 >= kk.nw + kk.nc)
        return;
    if (tid < GN)
        rows[tid] = key_row(a, kk, t, j0 + tid, sel);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + uint64_t(t) * a.heads * a.dim,
                                                                dextents<int32_t, 2>(a.dim, a.heads));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw(ws, dextents<int32_t, 2>(GK, GN));
    constexpr auto desc = matmul2d_descriptor(64, GN, GK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mq = tq.slice(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mq), decltype(tw), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    for (uint d = 0; d < a.dim; d += GK) {
        for (uint i = tid; i < GN * GK / 4; i += 128) {
            const uint r = i / (GK / 4), c4 = i % (GK / 4);
            reinterpret_cast<threadgroup half4*>(ws)[i] = half4(key4(rows[r], d / 4 + c4, a.dim, chunk, ringb, comp));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto qd = tq.slice(int(d), 0);
        op.run(qd, tw, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> ts(scores + uint64_t(tg.y) * a.heads * a.stride,
                                                                 dextents<int32_t, 2>(a.stride, a.heads));
    acc.store(ts.slice(int(j0), 0));
}

// Per token and head, s scaled first: p = exp(s - m) / (sum exp(s - m) + exp(sink - m)), m = max(max s, sink), as
// half into ph (zeros past the token's keys, up to stride). Threadgroups
// of 256 threads; grid (heads, tokens).
kernel void ds_attn_softmax(device float* scores [[buffer(0)]], device const float* sink [[buffer(1)]],
                            constant AttnArgs& a [[buffer(2)]], device half* ph [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tid2 [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    threadgroup float red[8];
    const uint h = tg.x;
    const Keys kk = keys_of(a, a.t0 + tg.y);
    const uint n = kk.nw + kk.nc;
    device float* s = scores + (uint64_t(tg.y) * a.heads + h) * a.stride;
    float m = -INFINITY;
    for (uint j = tid; j < n; j += 256) {
        s[j] *= a.scale;
        m = max(m, s[j]);
    }
    m = simd_max(m);
    if (tid % 32 == 0)
        red[tid / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = sink[h];
    for (uint i = 0; i < 8; ++i)
        m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < n; j += 256) {
        const float e = exp(s[j] - m);
        s[j] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (tid % 32 == 0)
        red[tid / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = exp(sink[h] - m);
    for (uint i = 0; i < 8; ++i)
        total += red[i];
    const float inv = 1.0f / total;
    device half* pr = ph + (uint64_t(tg.y) * a.heads + h) * a.stride;
    for (uint j = tid; j < a.stride; j += 256)
        pr[j] = j < n ? half(s[j] * inv) : 0.0h;
}

// o[t][h][d] = sum_j p[t][h][j] key_j(t)[d]: 64 heads x 64 dims a threadgroup of 4 simdgroups through matmul2d,
// probabilities (half, from the softmax) and keys staged 64 keys at a time in threadgroup memory, float
// accumulation. Grid (dim / 64, tokens).
kernel void ds_attn_out(device const half* p [[buffer(0)]], device const float* chunk [[buffer(1)]],
                        device const float* ringb [[buffer(2)]], device const half* comp [[buffer(3)]],
                        device const int* sel [[buffer(4)]], device float* o [[buffer(5)]],
                        constant AttnArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]])
{
    constexpr int TK = 64;
    threadgroup half ps[64 * TK], ks[TK * 64];
    threadgroup uint2 rows[TK];
    const uint t = a.t0 + tg.y, d0 = tg.x * 64;
    const Keys kk = keys_of(a, t);
    const uint n = kk.nw + kk.nc;
    device const half* prow = p + uint64_t(tg.y) * a.heads * a.stride;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(ps, dextents<int32_t, 2>(TK, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tk(ks, dextents<int32_t, 2>(64, TK));
    constexpr auto desc = matmul2d_descriptor(64, 64, TK, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    for (uint j0 = 0; j0 < n; j0 += TK) {
        if (tid < TK)
            rows[tid] = key_row(a, kk, t, j0 + tid, sel);
        // p tile: heads x keys j0 .. (zeros past n come from the softmax, up to stride)
        for (uint i = tid; i < 64 * TK / 4; i += 128) {
            const uint h = i / (TK / 4), c4 = i % (TK / 4);
            reinterpret_cast<threadgroup half4*>(ps)[i] =
                reinterpret_cast<device const half4*>(prow + uint64_t(h) * a.stride + j0)[c4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TK * 64 / 4; i += 128) {
            const uint r = i / 16, c4 = i % 16;
            reinterpret_cast<threadgroup half4*>(ks)[i] = half4(key4(rows[r], d0 / 4 + c4, a.dim, chunk, ringb, comp));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tp, tk, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(t) * a.heads * a.dim,
                                                                 dextents<int32_t, 2>(a.dim, a.heads));
    acc.store(to.slice(int(d0), 0));
}

// The last min(tokens, ring) of the chunk's rows into the ring (row pos % ring). Grid (dim, rows).
kernel void ds_ring_store(device const float* chunk [[buffer(0)]], device float* ringb [[buffer(1)]],
                          constant uint4& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint dim = a.x, tokens = a.y, pos0 = a.z, ring = a.w;
    const uint first = tokens > ring ? tokens - ring : 0;
    const uint t = first + gid.y;
    if (gid.x >= dim || t >= tokens)
        return;
    ringb[uint64_t((pos0 + t) % ring) * dim + gid.x] = chunk[uint64_t(t) * dim + gid.x];
}

struct IdxArgs {
    uint dim;    // 128
    uint heads;  // 32
    uint stride; // score row stride
    uint pos0, t0, ratio;
    float w_scale;
};

// score[t][i] = sum_h relu(q_t,h . k_i) * w_t,h * w_scale for the token's visible rows i < (p + 1) / ratio. One
// simdgroup per row: lane l holds dims l*4..l*4+3. Grid (stride * 32, tokens).
kernel void ds_idx_score(device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* keys [[buffer(2)]], device float* score [[buffer(3)]],
                         constant IdxArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]])
{
    const uint i = gid.x / 32, t = a.t0 + gid.y;
    const uint n = (a.pos0 + t + 1) / a.ratio;
    if (i >= n)
        return;
    const float4 k = reinterpret_cast<device const float4*>(keys + uint64_t(i) * a.dim)[lane];
    device const float* qt = q + uint64_t(t) * a.heads * a.dim;
    device const float* wt = w + uint64_t(t) * a.heads;
    float acc = 0.0f;
    for (uint h = 0; h < a.heads; ++h) {
        const float4 qh = reinterpret_cast<device const float4*>(qt + uint64_t(h) * a.dim)[lane];
        const float d = simd_sum(dot(qh, k));
        acc += max(d, 0.0f) * wt[h];
    }
    if (lane == 0)
        score[uint64_t(gid.y) * a.stride + i] = acc * a.w_scale;
}

struct BlockArgs {
    uint stride;       // score row stride
    uint blk_stride;   // block row stride
    uint pos0, t0, ratio, block;
    uint kblocks;      // blocks kept (tokens with at most this many keep them all)
};

// blk[t][b] = max of the token's scores over block b's visible rows; the block holding its last row is +inf
// (always kept). Grid (blk_stride, tokens).
kernel void ds_block_max(device const float* score [[buffer(0)]], device float* blk [[buffer(1)]],
                         constant BlockArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint b = gid.x, n = (a.pos0 + a.t0 + gid.y + 1) / a.ratio;
    const uint nb = (n + a.block - 1) / a.block;
    if (b >= nb)
        return;
    device const float* s = score + uint64_t(gid.y) * a.stride;
    float m = -INFINITY;
    for (uint i = b * a.block; i < min(n, (b + 1) * a.block); ++i)
        m = max(m, s[i]);
    blk[uint64_t(gid.y) * a.blk_stride + b] = b == nb - 1 ? INFINITY : m;
}

// score[t][i] = -inf where the token's candidate blocks (a flag per block) drop row i; nothing for a token with at
// most kblocks blocks. Grid (stride, tokens).
kernel void ds_block_mask(device float* score [[buffer(0)]], device const uchar* keep [[buffer(1)]],
                          constant BlockArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, n = (a.pos0 + a.t0 + gid.y + 1) / a.ratio;
    if ((n + a.block - 1) / a.block <= a.kblocks)
        return;
    if (i < n && !keep[uint64_t(gid.y) * a.blk_stride + i / a.block])
        score[uint64_t(gid.y) * a.stride + i] = -INFINITY;
}

struct TopkArgs {
    uint stride;      // v row stride
    uint k;
    uint ids_stride;  // ids row stride (rows: tokens from t0)
    uint flags_stride; // 0: no flags
    uint pos0, t0, ratio;
    uint block;       // n counts blocks of this many rows (1: rows)
};

// Order-preserving map of a float to uint (larger float, larger key); NaN sorts lowest.
static uint fkey(float f)
{
    if (isnan(f))
        return 0;
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// For each token (one threadgroup of 1024 threads each; grid (tokens)): the k largest of its n values (n: its
// visible rows / block, rounded up), their indices in increasing order into its row of ids (and a 1 per kept index
// into flags). Values equal to the k-th largest are taken in index order. Nothing when n <= k (all are kept).
kernel void ds_topk(device const float* vall [[buffer(0)]], device int* idsall [[buffer(1)]],
                    device uchar* flagsall [[buffer(2)]], constant TopkArgs& a [[buffer(3)]],
                    uint tok [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    constexpr uint NT = 1024;
    threadgroup uint cnt[32];
    threadgroup uint total;
    threadgroup uint base_out, base_eq;
    threadgroup uint sg_eq[32], sg_keep[32];
    const uint rows = (a.pos0 + a.t0 + tok + 1) / a.ratio;
    const uint n = (rows + a.block - 1) / a.block;
    if (n <= a.k)
        return;
    device const float* v = vall + uint64_t(tok) * a.stride;
    device int* ids = idsall + uint64_t(tok) * a.ids_stride;
    device uchar* flags = flagsall + uint64_t(tok) * a.flags_stride;
    // Largest t with count(key >= t) >= k.
    uint lo = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint th = lo | (1u << bit);
        uint c = 0;
        for (uint i = tid; i < n; i += NT)
            c += fkey(v[i]) >= th ? 1u : 0u;
        c = simd_sum(c);
        if (tid % 32 == 0)
            cnt[tid / 32] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint s = 0;
            for (uint i = 0; i < NT / 32; ++i)
                s += cnt[i];
            total = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (total >= a.k)
            lo = th;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint kth = lo;
    // Strictly above the k-th key: all kept; equal: the first (k - above) in index order.
    uint c = 0;
    for (uint i = tid; i < n; i += NT)
        c += fkey(v[i]) > kth ? 1u : 0u;
    c = simd_sum(c);
    if (tid % 32 == 0)
        cnt[tid / 32] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0;
        for (uint i = 0; i < NT / 32; ++i)
            s += cnt[i];
        total = s;
        base_out = 0;
        base_eq = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint equal_quota = a.k - total;
    // Stable compaction in chunks of NT indices.
    for (uint c0 = 0; c0 < n; c0 += NT) {
        const uint i = c0 + tid;
        const uint key = i < n ? fkey(v[i]) : 0;
        const bool eq = i < n && key == kth;
        const uint eq_rank = simd_prefix_exclusive_sum(eq ? 1u : 0u);
        const uint eq_sg = simd_sum(eq ? 1u : 0u);
        if (tid % 32 == 0)
            sg_eq[tid / 32] = eq_sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eq_before = base_eq;
        for (uint s = 0; s < tid / 32; ++s)
            eq_before += sg_eq[s];
        const bool keep = i < n && (key > kth || (eq && eq_before + eq_rank < equal_quota));
        const uint keep_rank = simd_prefix_exclusive_sum(keep ? 1u : 0u);
        const uint keep_sg = simd_sum(keep ? 1u : 0u);
        if (tid % 32 == 0)
            sg_keep[tid / 32] = keep_sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint keep_before = base_out;
        for (uint s = 0; s < tid / 32; ++s)
            keep_before += sg_keep[s];
        if (keep)
            ids[keep_before + keep_rank] = int(i);
        if (i < n && a.flags_stride)
            flags[i] = keep ? 1 : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint e = 0, kk = 0;
            for (uint s = 0; s < NT / 32; ++s) {
                e += sg_eq[s];
                kk += sg_keep[s];
            }
            base_eq += e;
            base_out += kk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct EngramDecodeArgs {
    uint dim, tokens, layers, cols;
    uint q6k; // Q6_K rows instead of MXFP8
};

// Engram rows from their bytes to float: MXFP8 (per 32 values an E8M0 scale byte, then 32 E4M3 codes) or Q6_K (per
// 256: ql[128], qh[64], int8 scales[16] per 16 values, half d). Raw rows are [token][layer][column]; y rows
// [layer][token][column] of dim values (each layer's rows contiguous). Grid (dim, rows).
kernel void ds_engram_decode(device const uchar* raw [[buffer(0)]], device float* y [[buffer(1)]],
                             constant EngramDecodeArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, r = gid.y, dim = a.dim;
    if (i >= dim)
        return;
    const uint t = r / (a.layers * a.cols), l = (r / a.cols) % a.layers, c = r % a.cols;
    const uint out = (l * a.tokens + t) * a.cols + c;
    if (a.q6k) {
        device const uchar* blk = raw + uint64_t(r) * (dim / 256 * 210) + (i / 256) * 210;
        const uint j = i % 256, n = j / 128, qt = (j % 128) / 32, l32 = j % 32;
        const uchar ql = blk[64 * n + (qt & 1) * 32 + l32], qh = blk[128 + 32 * n + l32];
        const int q = int(((qt < 2 ? ql & 0xF : ql >> 4)) | (((qh >> (2 * qt)) & 3) << 4)) - 32;
        const float d = float(*reinterpret_cast<device const half*>(blk + 208));
        const char sc = reinterpret_cast<device const char*>(blk + 192)[8 * n + 2 * qt + l32 / 16];
        y[uint64_t(out) * dim + i] = d * float(sc) * float(q);
        return;
    }
    device const uchar* blk = raw + uint64_t(r) * (dim / 32 * 33) + (i / 32) * 33;
    const uchar code = blk[1 + i % 32];
    const int e = (code >> 3) & 15, m = code & 7;
    float v = e ? ldexp(float(8 + m), e - 10) : ldexp(float(m), -9);
    v = (code & 128) ? -v : v;
    y[uint64_t(out) * dim + i] = ldexp(v, int(blk[0]) - 127);
}

struct EngramArgs {
    uint dim; // n_embd
    float eps;
};

// Engram for each of a token's 4 streams (one threadgroup of 256 threads per stream k and token; grid (4, tokens)):
// with key_k and the shared value from kv ([4 keys | value] of dim each per token), d = sum h_k * G_k * key_k *
// rstd(h_k) * rstd(key_k) / sqrt(dim), h_k += sigmoid(sign(d) sqrt(max(|d|, 1e-6))) * value.
kernel void ds_engram_gate(device float* R [[buffer(0)]], device const float* kv [[buffer(1)]],
                           device const float* G [[buffer(2)]], constant EngramArgs& a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    threadgroup float red[3][8];
    const uint k = tg.x;
    device float* h = R + (uint64_t(tg.y) * 4 + k) * a.dim;
    device const float* key = kv + (uint64_t(tg.y) * 5 + k) * a.dim;
    device const float* value = kv + (uint64_t(tg.y) * 5 + 4) * a.dim;
    device const float* g = G + uint64_t(k) * a.dim;
    float hh = 0.0f, kk = 0.0f, hgk = 0.0f;
    for (uint i = tid; i < a.dim; i += 256) {
        const float hv = h[i], kvv = key[i];
        hh += hv * hv;
        kk += kvv * kvv;
        hgk += hv * g[i] * kvv;
    }
    hh = simd_sum(hh);
    kk = simd_sum(kk);
    hgk = simd_sum(hgk);
    if (tid % 32 == 0) {
        red[0][tid / 32] = hh;
        red[1][tid / 32] = kk;
        red[2][tid / 32] = hgk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        sh += red[0][i];
        sk += red[1][i];
        sd += red[2][i];
    }
    const float rstd = rsqrt(sh / float(a.dim) + a.eps) * rsqrt(sk / float(a.dim) + a.eps);
    const float d = sd * rstd * rsqrt(float(a.dim));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(fabs(d), 1e-6f)), d)));
    for (uint i = tid; i < a.dim; i += 256)
        h[i] += gate * value[i];
}

// taps[t][slot] = the mean of token t's 4 streams (rows of 4 * dim), into rows of `width` floats at slot * dim.
// Grid (dim, tokens).
kernel void ds_tap(device const float* R [[buffer(0)]], device float* taps [[buffer(1)]], constant uint4& a [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]])
{
    const uint dim = a.x, width = a.y, slot = a.z;
    if (gid.x >= dim)
        return;
    device const float* r = R + uint64_t(gid.y) * 4 * dim;
    taps[uint64_t(gid.y) * width + slot * dim + gid.x] = 0.25f * (r[gid.x] + r[dim + gid.x] + r[2 * dim + gid.x] + r[3 * dim + gid.x]);
}

struct MarkovArgs {
    uint vocab, rank;
    uint step; // the draft being chosen: prev is drafts[step - 1], or y at step 0
    int y;
};

// scores[v] = logits[v] + head[v] . embed[prev] (the Markov bias of the previous draft). One simdgroup per 8 rows.
kernel void ds_markov(device const float* logits [[buffer(0)]], device const float* head [[buffer(1)]],
                      device const float* embed [[buffer(2)]], device const int* drafts [[buffer(3)]],
                      device float* scores [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]],
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    const uint v0 = (gid / 32) * 8;
    const int prev = a.step ? drafts[a.step - 1] : a.y;
    device const float* e = embed + uint64_t(prev) * a.rank;
    for (uint r = 0; r < 8; ++r) {
        const uint v = v0 + r;
        if (v >= a.vocab)
            return;
        float acc = 0.0f;
        for (uint i = lane; i < a.rank; i += 32)
            acc += head[uint64_t(v) * a.rank + i] * e[i];
        acc = simd_sum(acc);
        if (lane == 0)
            scores[v] = logits[v] + acc;
    }
}

// drafts[step] = argmax of scores (lowest index on ties); conf[step] = sigmoid([x, embed[prev]] . w), x the block's
// row `step` of dim floats. One threadgroup of 1024 threads.
kernel void ds_draft_pick(device const float* scores [[buffer(0)]], device int* drafts [[buffer(1)]],
                          device float* conf [[buffer(2)]], device const float* x [[buffer(3)]],
                          device const float* embed [[buffer(4)]], device const float* w [[buffer(5)]],
                          constant MarkovArgs& a [[buffer(6)]], constant uint& dim [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    threadgroup float part[32];
    float bv = -INFINITY;
    uint bi = 0;
    for (uint v = tid; v < a.vocab; v += 1024) {
        const float s = scores[v];
        if (s > bv) {
            bv = s;
            bi = v;
        }
    }
    for (uint o = 16; o > 0; o >>= 1) {
        const float ov = simd_shuffle_down(bv, o);
        const uint oi = simd_shuffle_down(bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int prev = a.step ? drafts[a.step - 1] : a.y;
    device const float* e = embed + uint64_t(prev) * a.rank;
    device const float* xr = x + uint64_t(a.step) * dim;
    float d = 0.0f;
    for (uint i = tid; i < dim + a.rank; i += 1024)
        d += w[i] * (i < dim ? xr[i] : e[i - dim]);
    d = simd_sum(d);
    if (tid % 32 == 0) {
        best_v[tid / 32] = bv;
        best_i[tid / 32] = bi;
        part[tid / 32] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float v = best_v[0], s = part[0];
        uint i = best_i[0];
        for (uint k = 1; k < 32; ++k) {
            if (best_v[k] > v || (best_v[k] == v && best_i[k] < i)) {
                v = best_v[k];
                i = best_i[k];
            }
            s += part[k];
        }
        drafts[a.step] = int(i);
        conf[a.step] = 1.0f / (1.0f + exp(-s));
    }
}
