// DeepSeek-V4.1 kernels: partial RoPE, the FP8 / FP4 activation roundings, the compressor, the sparse attention
// with its sink, the indexer and Engram. Activations are fp32.
#include <metal_stdlib>
using namespace metal;

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
    uint emit; // 0: stash (k, g) of the group's first token; 1: pool with the stash into y
};

// Ratio-2 compressor: per channel, softmax over the group's two tokens of the gate, applied to the kv projections.
// state holds the first token's k then g; k, g the current token's. Grid (dim).
kernel void ds_pool2(device const float* k [[buffer(0)]], device const float* g [[buffer(1)]],
                     device float* state [[buffer(2)]], device float* y [[buffer(3)]], constant PoolArgs& a [[buffer(4)]],
                     uint c [[thread_position_in_grid]])
{
    if (c >= a.dim)
        return;
    if (!a.emit) {
        state[c] = k[c];
        state[a.dim + c] = g[c];
        return;
    }
    const float g0 = state[a.dim + c], g1 = g[c];
    const float m = max(g0, g1);
    const float e0 = exp(g0 - m), e1 = exp(g1 - m);
    y[c] = (state[c] * e0 + k[c] * e1) / (e0 + e1);
}

struct GatherArgs {
    uint dim;
    uint n_window;  // window rows first: positions first_pos .. first_pos + n_window - 1, ring row pos % ring
    uint first_pos, ring;
    uint n_comp;    // then compressed rows: ids[j] (or j itself when ids_identity)
    uint ids_identity;
};

// K = the window's rows of the ring (float) followed by the selected compressed rows (half). Grid (dim, rows).
kernel void ds_gather_keys(device const float* ringb [[buffer(0)]], device const half* comp [[buffer(1)]],
                           device const int* ids [[buffer(2)]], device float* K [[buffer(3)]],
                           constant GatherArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, j = gid.y;
    if (d >= a.dim || j >= a.n_window + a.n_comp)
        return;
    float v;
    if (j < a.n_window) {
        v = ringb[uint64_t((a.first_pos + j) % a.ring) * a.dim + d];
    } else {
        const uint c = j - a.n_window;
        const uint row = a.ids_identity ? c : uint(ids[c]);
        v = float(comp[uint64_t(row) * a.dim + d]);
    }
    K[uint64_t(j) * a.dim + d] = v;
}

struct ScoreArgs {
    uint dim;   // 512
    uint heads; // 64
    uint n;     // keys
    float scale;
};

// scores[h][j] = scale * q_h . K_j. Threadgroups of 256 threads take 8 keys (staged in threadgroup memory); each
// simdgroup takes heads h = sg, sg + 8, ... Grid (ceil(n / 8)) threadgroups. dim <= 512.
kernel void ds_attn_scores(device const float* q [[buffer(0)]], device const float* K [[buffer(1)]],
                           device float* scores [[buffer(2)]], constant ScoreArgs& a [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float ks[8][512];
    const uint j0 = tg * 8;
    for (uint i = tid; i < 8 * a.dim; i += 256) {
        const uint r = i / a.dim, d = i % a.dim;
        ks[r][d] = j0 + r < a.n ? K[uint64_t(j0 + r) * a.dim + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = sg; h < a.heads; h += 8) {
        float acc[8] = {};
        device const float* qh = q + uint64_t(h) * a.dim;
        for (uint d = lane; d < a.dim; d += 32) {
            const float qv = qh[d];
            for (uint r = 0; r < 8; ++r)
                acc[r] += qv * ks[r][d];
        }
        for (uint r = 0; r < 8; ++r) {
            const float s = simd_sum(acc[r]);
            if (lane == 0 && j0 + r < a.n)
                scores[uint64_t(h) * a.n + j0 + r] = s * a.scale;
        }
    }
}

// Per head: p = exp(s - m) / (sum exp(s - m) + exp(sink - m)), m = max(max s, sink), in place. One threadgroup of
// 256 threads per head.
kernel void ds_attn_softmax(device float* scores [[buffer(0)]], device const float* sink [[buffer(1)]],
                            constant ScoreArgs& a [[buffer(2)]], uint h [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float red[8];
    device float* s = scores + uint64_t(h) * a.n;
    float m = -INFINITY;
    for (uint j = tid; j < a.n; j += 256)
        m = max(m, s[j]);
    m = simd_max(m);
    if (tid % 32 == 0)
        red[tid / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = sink[h];
    for (uint i = 0; i < 8; ++i)
        m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < a.n; j += 256) {
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
    for (uint j = tid; j < a.n; j += 256)
        s[j] *= inv;
}

// o[h][d] = sum_j p[h][j] K[j][d]. Threadgroups of 256 threads own 64 dims: thread (d % 64, head group of 16);
// grid (dim / 64) threadgroups.
kernel void ds_attn_out(device const float* p [[buffer(0)]], device const float* K [[buffer(1)]],
                        device float* o [[buffer(2)]], constant ScoreArgs& a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    const uint d = tg * 64 + tid % 64, h0 = (tid / 64) * 16;
    float acc[16] = {};
    for (uint j = 0; j < a.n; ++j) {
        const float k = K[uint64_t(j) * a.dim + d];
        for (uint i = 0; i < 16; ++i)
            acc[i] += p[uint64_t(h0 + i) * a.n + j] * k;
    }
    for (uint i = 0; i < 16; ++i)
        if (h0 + i < a.heads)
            o[uint64_t(h0 + i) * a.dim + d] = acc[i];
}

struct IdxArgs {
    uint dim;   // 128
    uint heads; // 32
    uint n;     // compressed rows visible
    float w_scale;
};

// score[i] = sum_h relu(q_h . k_i) * w_h * w_scale for i < n. One simdgroup per row: lane l holds dims l*4..l*4+3.
kernel void ds_idx_score(device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* keys [[buffer(2)]], device float* score [[buffer(3)]],
                         constant IdxArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]])
{
    const uint i = gid / 32;
    if (i >= a.n)
        return;
    const float4 k = reinterpret_cast<device const float4*>(keys + uint64_t(i) * a.dim)[lane];
    float acc = 0.0f;
    for (uint h = 0; h < a.heads; ++h) {
        const float4 qh = reinterpret_cast<device const float4*>(q + uint64_t(h) * a.dim)[lane];
        const float d = simd_sum(dot(qh, k));
        acc += max(d, 0.0f) * w[h];
    }
    if (lane == 0)
        score[i] = acc * a.w_scale;
}

struct BlockArgs {
    uint n;     // rows
    uint block; // rows per block
};

// blk[b] = max of score over the block's rows; the block holding the last row is +inf (always kept). Grid (blocks).
kernel void ds_block_max(device const float* score [[buffer(0)]], device float* blk [[buffer(1)]],
                         constant BlockArgs& a [[buffer(2)]], uint b [[thread_position_in_grid]])
{
    const uint nb = (a.n + a.block - 1) / a.block;
    if (b >= nb)
        return;
    float m = -INFINITY;
    for (uint i = b * a.block; i < min(a.n, (b + 1) * a.block); ++i)
        m = max(m, score[i]);
    blk[b] = b == nb - 1 ? INFINITY : m;
}

// score[i] = -inf where the candidate mask (kept blocks, a flag per block) drops row i. Grid (n).
kernel void ds_block_mask(device float* score [[buffer(0)]], device const uchar* keep [[buffer(1)]],
                          constant BlockArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < a.n && !keep[i / a.block])
        score[i] = -INFINITY;
}

struct TopkArgs {
    uint n, k;
    uint with_flags;
};

// Order-preserving map of a float to uint (larger float, larger key); -inf and NaN sort lowest.
static uint fkey(float f)
{
    if (isnan(f))
        return 0;
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The k largest of v[0..n) (k < n): ids gets their indices in increasing order, flags (with_flags) a 1 per kept
// index. Values equal to the k-th largest are taken in index order. One threadgroup of 1024 threads: the k-th key
// by bisection on its bits, then a stable compaction.
kernel void ds_topk(device const float* v [[buffer(0)]], device int* ids [[buffer(1)]], device uchar* flags [[buffer(2)]],
                    constant TopkArgs& a [[buffer(3)]], uint tid [[thread_position_in_threadgroup]])
{
    constexpr uint NT = 1024;
    threadgroup uint cnt[32];
    threadgroup uint total;
    threadgroup uint base_out, base_eq;
    threadgroup uint sg_eq[32], sg_keep[32];
    // Largest t with count(key >= t) >= k.
    uint lo = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint t = lo | (1u << bit);
        uint c = 0;
        for (uint i = tid; i < a.n; i += NT)
            c += fkey(v[i]) >= t ? 1u : 0u;
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
            lo = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint kth = lo;
    // Strictly above the k-th key: all kept; equal: the first (k - above) in index order.
    uint c = 0;
    for (uint i = tid; i < a.n; i += NT)
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint equal_quota = a.k - total;
    // Stable compaction in chunks of NT indices.
    if (tid == 0) {
        base_out = 0;
        base_eq = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c0 = 0; c0 < a.n; c0 += NT) {
        const uint i = c0 + tid;
        const uint key = i < a.n ? fkey(v[i]) : 0;
        const bool eq = i < a.n && key == kth;
        const uint eq_rank = simd_prefix_exclusive_sum(eq ? 1u : 0u);
        const uint eq_sg = simd_sum(eq ? 1u : 0u);
        if (tid % 32 == 0)
            sg_eq[tid / 32] = eq_sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eq_before = base_eq;
        for (uint s = 0; s < tid / 32; ++s)
            eq_before += sg_eq[s];
        const bool keep = i < a.n && (key > kth || (eq && eq_before + eq_rank < equal_quota));
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
        if (i < a.n && a.with_flags)
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

struct EngramArgs {
    uint dim; // n_embd
    float eps;
};

// Engram for each of the 4 streams (one threadgroup of 256 threads per stream k): with key_k and the shared value
// from kv ([4 keys | value] of dim each), d = sum h_k * G_k * key_k * rstd(h_k) * rstd(key_k) / sqrt(dim),
// h_k += sigmoid(sign(d) sqrt(max(|d|, 1e-6))) * value.
kernel void ds_engram_gate(device float* R [[buffer(0)]], device const float* kv [[buffer(1)]],
                           device const float* G [[buffer(2)]], constant EngramArgs& a [[buffer(3)]],
                           uint k [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float red[3][8];
    device float* h = R + uint64_t(k) * a.dim;
    device const float* key = kv + uint64_t(k) * a.dim;
    device const float* value = kv + uint64_t(4) * a.dim;
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
