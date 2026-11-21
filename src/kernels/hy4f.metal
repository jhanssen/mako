// Hy4-flash kernels. Activations are fp32.
#include <metal_stdlib>
using namespace metal;

#include "rows.h"

static float tg_max_hy(float v, threadgroup float* scratch, uint tid)
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here (the default fast math assumes none: MSL 1.6.3)
    v = simd_max(v);
    if (tid % 32 == 0)
        scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid % 32;
    float m = lane < TG / 32 ? scratch[lane] : -INFINITY;
    m = simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

struct IhcArgs {
    uint n_embd;
    float eps;       // added after each sigmoid
    float magnitude; // the post weights' range
    uint tokens;
};

// Identity hyper-connections over 4 streams. From a token's 8 mixes: pre[h] = sigmoid(m[h] * s0 + b[h]) + eps,
// post[h] = magnitude * sigmoid(m[4 + h] * s1 + b[4 + h]) + eps (pc rows: pre, post); x = sum_h pre[h] R[h].
// Grid (n_embd, tokens).
kernel void ihc_pre(device const float* mixes [[buffer(0)]], device const float* scale [[buffer(1)]],
                    device const float* base [[buffer(2)]], device const float* R [[buffer(3)]],
                    device float* x [[buffer(4)]], device float* pc [[buffer(5)]], constant IhcArgs& a [[buffer(6)]],
                    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n_embd)
        return;
    device const float* m = mixes + t * 8;
    float pre[4];
    for (uint h = 0; h < 4; ++h)
        pre[h] = sigmoidf(m[h] * scale[0] + base[h]) + a.eps;
    if (i == 0)
        for (uint h = 0; h < 4; ++h) {
            pc[t * 8 + h] = pre[h];
            pc[t * 8 + 4 + h] = a.magnitude * sigmoidf(m[4 + h] * scale[1] + base[4 + h]) + a.eps;
        }
    R += uint64_t(t) * 4 * a.n_embd;
    float acc = 0.0f;
    for (uint h = 0; h < 4; ++h)
        acc += R[h * a.n_embd + i] * pre[h];
    x[uint64_t(t) * a.n_embd + i] = acc;
}

// R_out[h] = R[h] + post[h] * x (each stream keeps its own residual); in place allowed. Grid (n_embd, tokens).
kernel void ihc_post(device const float* x [[buffer(0)]], device const float* R [[buffer(1)]],
                     device float* R_out [[buffer(2)]], device const float* pc [[buffer(3)]],
                     constant IhcArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n_embd)
        return;
    const float v = x[uint64_t(t) * a.n_embd + i];
    const uint64_t r = uint64_t(t) * 4 * a.n_embd + i;
    for (uint h = 0; h < 4; ++h)
        R_out[r + h * a.n_embd] = R[r + h * a.n_embd] + pc[t * 8 + 4 + h] * v;
}

// The output head's reduction: x = sum_h (sigmoid(m[h] * s + b[h]) + eps) R[h], from 4 mixes a token.
// Grid (n_embd, tokens).
kernel void ihc_head(device const float* mixes [[buffer(0)]], device const float* scale [[buffer(1)]],
                     device const float* base [[buffer(2)]], device const float* R [[buffer(3)]],
                     device float* x [[buffer(4)]], constant IhcArgs& a [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n_embd)
        return;
    R += uint64_t(t) * 4 * a.n_embd;
    float acc = 0.0f;
    for (uint h = 0; h < 4; ++h)
        acc += R[h * a.n_embd + i] * (sigmoidf(mixes[t * 4 + h] * scale[0] + base[h]) + a.eps);
    x[uint64_t(t) * a.n_embd + i] = acc;
}

struct EngramRowsArgs {
    uint dim;          // values a row (<= 128)
    uint cols;         // rows a token
    uint token_stride; // floats between tokens' first rows in `rows`
    float eps;
};

// y[t][c] = rms_norm(rows[t][c]) * w[c] (a zero row stays zero). Threadgroups (cols, tokens) of 32.
kernel void hy4_engram_rows(device const float* rows [[buffer(0)]], device const float* w [[buffer(1)]],
                            device float* y [[buffer(2)]], constant EngramRowsArgs& a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    const uint c = tg.x, t = tg.y;
    device const float* r = rows + uint64_t(t) * a.token_stride + c * a.dim;
    float v[4], ss = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        const uint d = lane + 32 * j;
        v[j] = d < a.dim ? r[d] : 0.0f;
        ss += v[j] * v[j];
    }
    const float inv = rsqrt(simd_sum(ss) / float(a.dim) + a.eps);
    device float* o = y + (uint64_t(t) * a.cols + c) * a.dim;
    for (uint j = 0; j < 4; ++j) {
        const uint d = lane + 32 * j;
        if (d < a.dim)
            o[d] = v[j] * inv * w[c * a.dim + d];
    }
}

struct EngramArgs {
    uint n_embd;
    float eps;
};

// Per token (one threadgroup of TG): the Engram projection kv (4 keys then a value, plus bias kv_b) gates each
// stream by dot = sum(rms(R[h]) * rms(key[h]) * gate[h]) / sqrt(n_embd), g = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6)));
// R[h] += rms(value) * gamma * (g[h] + hc_scale[h]) when the token has an n-gram (valid[t]).
kernel void hy4_engram_apply(device float* R [[buffer(0)]], device const float* kv [[buffer(1)]],
                             device const float* kv_b [[buffer(2)]], device const float* gate [[buffer(3)]],
                             device const float* hc_scale [[buffer(4)]], device const float* gamma [[buffer(5)]],
                             device const uint* valid [[buffer(6)]], constant EngramArgs& a [[buffer(7)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    if (!valid[t])
        return;
    const uint E = a.n_embd;
    R += uint64_t(t) * 4 * E;
    kv += uint64_t(t) * 5 * E;
    float ss_x[4] = {}, ss_k[4] = {}, dot[4] = {}, ss_v = 0.0f;
    for (uint i = tid; i < E; i += TG) {
        for (uint h = 0; h < 4; ++h) {
            const float x = R[h * E + i], k = kv[h * E + i] + kv_b[h * E + i];
            ss_x[h] += x * x;
            ss_k[h] += k * k;
            dot[h] += x * k * gate[h * E + i];
        }
        const float v = kv[4 * E + i] + kv_b[4 * E + i];
        ss_v += v * v;
    }
    float g[4];
    for (uint h = 0; h < 4; ++h) {
        const float rx = rsqrt(tg_sum(ss_x[h], scratch, tid) / float(E) + a.eps);
        const float rk = rsqrt(tg_sum(ss_k[h], scratch, tid) / float(E) + a.eps);
        const float d = tg_sum(dot[h], scratch, tid) * rx * rk / sqrt(float(E));
        const float mag = sqrt(max(fabs(d), 1e-6f));
        g[h] = sigmoidf(d < 0.0f ? -mag : mag);
    }
    const float rv = rsqrt(tg_sum(ss_v, scratch, tid) / float(E) + a.eps);
    for (uint i = tid; i < E; i += TG) {
        const float s = (kv[4 * E + i] + kv_b[4 * E + i]) * rv * gamma[i];
        for (uint h = 0; h < 4; ++h)
            R[h * E + i] += s * (g[h] + hc_scale[h * E + i]);
    }
}

struct SplitArgs {
    uint heads, nope, rope; // each head of q: nope values, then rope
};

// q rows [heads][nope + rope] into nope [heads][nope] and rope [heads][rope] rows. Grid (nope + rope, heads, tokens).
kernel void hy4_split_q(device const float* q [[buffer(0)]], device float* nope [[buffer(1)]],
                        device float* rope [[buffer(2)]], constant SplitArgs& a [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, h = gid.y, t = gid.z, w = a.nope + a.rope;
    if (d >= w)
        return;
    const uint64_t row = uint64_t(t) * a.heads + h;
    const float v = q[row * w + d];
    if (d < a.nope)
        nope[row * a.nope + d] = v;
    else
        rope[row * a.rope + d - a.nope] = v;
}

struct AttnSoftmaxArgs {
    uint n_kv;    // row length: keys kbase .. kbase + n_kv
    float scale;
    uint vis0;    // keys row 0's token sees from kbase (itself included); token j sees vis0 + j
    uint heads;   // rows per token
    uint window;  // 0: none; else token j sees only its last `window` keys
};

// In place over rows (tokens x heads): s = (lat + rope) * scale over the row's visible keys, softmax with the head's
// sink in the denominator, zero elsewhere. One threadgroup per row.
kernel void hy4_attn_softmax(device float* lat [[buffer(0)]], device const float* rope [[buffer(1)]],
                             device const float* sinks [[buffer(2)]], constant AttnSoftmaxArgs& a [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here (the default fast math assumes none: MSL 1.6.3)
    threadgroup float scratch[TG / 32];
    device float* s = lat + uint64_t(row) * a.n_kv;
    device const float* r = rope + uint64_t(row) * a.n_kv;
    const uint j = row / a.heads, h = row % a.heads;
    const uint hi = min(a.n_kv, a.vis0 + j), lo = a.window && hi > a.window ? hi - a.window : 0;
    float m = sinks[h];
    for (uint k = lo + tid; k < hi; k += TG)
        m = max(m, (s[k] + r[k]) * a.scale);
    m = tg_max_hy(m, scratch, tid);
    float sum = 0.0f;
    for (uint k = tid; k < a.n_kv; k += TG) {
        const float e = k >= lo && k < hi ? exp((s[k] + r[k]) * a.scale - m) : 0.0f;
        s[k] = e;
        sum += e;
    }
    const float inv = 1.0f / (tg_sum(sum, scratch, tid) + exp(sinks[h] - m));
    for (uint k = tid; k < a.n_kv; k += TG)
        s[k] *= inv;
}

struct GatherRopeArgs {
    uint heads, dim, stride; // dim: rope values a key (64)
};

// scores[t][h][s] = q[t][h] . pe[list[t][s]] over each token's count[t] entries (0 past it). Threadgroups (ceil(stride /
// 64), heads, tokens) of 64: an entry per thread.
kernel void hy4_gather_rope(device const float* q [[buffer(0)]], device const half* pe [[buffer(1)]],
                            device const uint* list [[buffer(2)]], device const uint* count [[buffer(3)]],
                            device float* scores [[buffer(4)]], constant GatherRopeArgs& a [[buffer(5)]],
                            uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float qs[64];
    const uint t = tg.z, h = tg.y, s = tg.x * 64 + tid;
    device const float* qr = q + (uint64_t(t) * a.heads + h) * a.dim;
    if (tid < a.dim)
        qs[tid] = qr[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s >= a.stride)
        return;
    device float* out = scores + (uint64_t(t) * a.heads + h) * a.stride;
    if (s >= count[t]) {
        out[s] = 0.0f;
        return;
    }
    device const half* k = pe + uint64_t(list[uint64_t(t) * a.stride + s]) * a.dim;
    float acc = 0.0f;
    for (uint d = 0; d < a.dim; ++d)
        acc += qs[d] * float(k[d]);
    out[s] = acc;
}

struct ListSoftmaxArgs {
    uint stride; // row length
    float scale;
    uint heads;  // rows per token
};

// The same softmax over each (token, head) row's first count[token] entries of lat + rope (in place in lat).
kernel void hy4_softmax_list(device float* lat [[buffer(0)]], device const float* rope [[buffer(1)]],
                             device const uint* count [[buffer(2)]], device const float* sinks [[buffer(3)]],
                             constant ListSoftmaxArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here (the default fast math assumes none: MSL 1.6.3)
    threadgroup float scratch[TG / 32];
    device float* s = lat + uint64_t(row) * a.stride;
    device const float* r = rope + uint64_t(row) * a.stride;
    const uint n = count[row / a.heads], h = row % a.heads;
    float m = sinks[h];
    for (uint k = tid; k < n; k += TG)
        m = max(m, (s[k] + r[k]) * a.scale);
    m = tg_max_hy(m, scratch, tid);
    float sum = 0.0f;
    for (uint k = tid; k < a.stride; k += TG) {
        const float e = k < n ? exp((s[k] + r[k]) * a.scale - m) : 0.0f;
        s[k] = e;
        sum += e;
    }
    const float inv = 1.0f / (tg_sum(sum, scratch, tid) + exp(sinks[h] - m));
    for (uint k = tid; k < a.stride; k += TG)
        s[k] *= inv;
}

// y[i] *= sigmoid(g[i]).
kernel void hy4_gate_mul(device float* y [[buffer(0)]], device const float* g [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] *= sigmoidf(g[i]);
}

// y[i] = x[i] over n 32-bit words (rows moved within a cache; the ranges do not overlap).
kernel void hy4_copy_words(device const uint* x [[buffer(0)]], device uint* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                           uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = x[i];
}

struct Rope3Args {
    uint head_dim, heads, n_rot, stride;
    uint pos_step; // row t's position at pos[t * pos_step]
    uint sec_h, sec_w;
};

// Rotates the last n_rot dims of each head of each row, pairs (2i, 2i + 1), by pos_a * freq[i]: a the row (pos.y)
// for i % 3 == 1 below 3 * sec_h, the column (pos.z) for i % 3 == 2 below 3 * sec_w, else the sequence position
// (pos.x). Grid (n_rot / 2, heads, rows).
kernel void hy4_rope3(device float* x [[buffer(0)]], device const float* freq [[buffer(1)]],
                      device const int4* pos [[buffer(2)]], constant Rope3Args& a [[buffer(3)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, h = gid.y, t = gid.z;
    if (i >= a.n_rot / 2 || h >= a.heads)
        return;
    device float* p = x + uint64_t(t) * a.stride + uint64_t(h) * a.head_dim + (a.head_dim - a.n_rot) + 2 * i;
    const int4 ps = pos[t * a.pos_step];
    const int at = i % 3 == 1 && i < 3 * a.sec_h ? ps.y : i % 3 == 2 && i < 3 * a.sec_w ? ps.z : ps.x;
    const float ang = float(at) * freq[i];
    const float c = precise::cos(ang), s = precise::sin(ang);
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

// DFly2 drafter.

// Rows of a Q8_0 table (blocks of a float16 scale and 32 int8 values) as floats: out[r] = table[ids[r]]. Grid
// (width, rows).
kernel void hy4_embed_q8(device const uchar* table [[buffer(0)]], device const int* ids [[buffer(1)]],
                         device float* out [[buffer(2)]], constant uint& width [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, r = gid.y;
    if (i >= width)
        return;
    device const uchar* blk = table + (uint64_t(ids[r]) * (width / 32) + i / 32) * 34;
    const float d = float(*reinterpret_cast<device const half*>(blk));
    out[uint64_t(r) * width + i] = d * float(reinterpret_cast<device const char*>(blk + 2)[i % 32]);
}

// y[i] += g[i] * f[i].
kernel void hy4_gate_add(device float* y [[buffer(0)]], device const float* g [[buffer(1)]],
                         device const float* f [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] += g[i] * f[i];
}

struct DraftPickArgs {
    uint n_vocab, width, slot;
    float conf_bias;
};

// One threadgroup of 1024: draft `slot` is the largest logit (the lowest index on ties), also written to
// prev[slot + 1] for the next step's embedding; its confidence sigmoid(conf_w . h + bias).
kernel void hy4_draft_pick(device const float* logits [[buffer(0)]], device const float* h [[buffer(1)]],
                           device const float* conf_w [[buffer(2)]], device int* drafts [[buffer(3)]],
                           device float* conf [[buffer(4)]], device int* prev [[buffer(5)]],
                           constant DraftPickArgs& a [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here (the default fast math assumes none: MSL 1.6.3)
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    threadgroup float dots[32];
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = tid; i < a.n_vocab; i += 1024) {
        const float v = logits[i];
        if (v > bv) { // ascending i within the thread: strict > keeps the lowest index
            bv = v;
            bi = i;
        }
    }
    float d = 0.0f;
    for (uint i = tid; i < a.width; i += 1024)
        d += conf_w[i] * h[i];
    const float m = simd_max(bv);
    const uint mi = simd_min(bv == m ? bi : 0xffffffffu);
    d = simd_sum(d);
    if (lane == 0) {
        best_v[sg] = m;
        best_i[sg] = mi;
        dots[sg] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float v = best_v[lane];
        const float mm = simd_max(v);
        const uint ii = simd_min(v == mm ? best_i[lane] : 0xffffffffu);
        const float dd = simd_sum(dots[lane]);
        if (lane == 0) {
            const int id = ii < a.n_vocab ? int(ii) : 0;
            drafts[a.slot] = id;
            prev[a.slot + 1] = id;
            conf[a.slot] = 1.0f / (1.0f + precise::exp(-(dd + a.conf_bias)));
        }
    }
}

struct IdxScoreArgs {
    uint dim, heads, first, kpool;
    float w_scale;
    uint stride; // floats between tokens' score rows
};

// idx_bench's reference for hy4_idx_score_mm (the engine runs that one).
// score[j][p] = w_scale * sum_h relu(q[j][h] . pooled[p]) * w[j][h] over token j's complete pools: a simdgroup a pool
// (8 a threadgroup), lane l holding dims 4l .. 4l + 3 (dim 128) of every head's query; grid (pools of the last token
// / 8, tokens).
kernel void hy4_idx_score(device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
                          device const float* pooled [[buffer(2)]], device float* score [[buffer(3)]],
                          constant IdxScoreArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint j = tg.y, p = tg.x * 8 + sg;
    if (p >= (a.first + j) / a.kpool)
        return;
    device const float4* q4 = reinterpret_cast<device const float4*>(q + uint64_t(j) * a.heads * a.dim);
    const float4 k = reinterpret_cast<device const float4*>(pooled + uint64_t(p) * a.dim)[lane];
    float total = 0.0f;
    for (uint h = 0; h < a.heads; ++h) {
        const float d = simd_sum(dot(q4[h * (a.dim / 4) + lane], k));
        total += max(d, 0.0f) * w[uint64_t(j) * a.heads + h];
    }
    if (lane == 0)
        score[uint64_t(j) * a.stride + p] = total * a.w_scale;
}

// The same scores as matrix products. A threadgroup (8 simdgroups) takes tj tokens (1, 2, 4 or 8) and 256 / tj
// pools: simdgroup s token s / (8 / tj), 32 pools; the keys and queries staged 16 dims at a time, each token's 16
// heads times the pools in 8 x 8 blocks, then relu, the head weights and w_scale (heads in order). Every score's sums
// run in the same order whatever tj. 16 heads, dim a multiple of 16; grid (ceil(pools / (256 / tj)),
// ceil(tokens / tj)), 256 threads.
struct IdxMmArgs {
    uint tokens, tj;
};
kernel void hy4_idx_score_mm(device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
                             device const float* pooled [[buffer(2)]], device float* score [[buffer(3)]],
                             constant IdxScoreArgs& a [[buffer(4)]], constant IdxMmArgs& m [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float ks[256 * 16]; // pools x 16 dims; then each simdgroup's 16 x 32 results
    threadgroup float qs[8 * 16 * 16]; // tokens x heads x 16 dims
    const uint spt = 8 / m.tj, np = 32 * spt, ts = sg / spt, ps = sg % spt;
    const uint jt0 = tg.y * m.tj, j = jt0 + ts, pool_tg = tg.x * np, pool0 = pool_tg + ps * 32;
    const uint jl = min(m.tokens, jt0 + m.tj) - 1, last_pools = (a.first + jl) / a.kpool;
    const uint n_pool = j < m.tokens ? (a.first + j) / a.kpool : 0;
    const bool live = pool0 < n_pool;
    simdgroup_float8x8 acc[2][4];
    for (uint h = 0; h < 2; ++h)
        for (uint c = 0; c < 4; ++c)
            acc[h][c] = simdgroup_float8x8(0.0f);
    for (uint d0 = 0; d0 < a.dim; d0 += 16) {
        for (uint i = tid; i < np * 16; i += 256) {
            const uint p = pool_tg + i / 16;
            ks[i] = p < last_pools ? pooled[uint64_t(p) * a.dim + d0 + i % 16] : 0.0f;
        }
        for (uint i = tid; i < m.tj * 256; i += 256) {
            const uint jj = jt0 + i / 256;
            qs[i] = jj < m.tokens ? q[(uint64_t(jj) * 16 + (i / 16) % 16) * a.dim + d0 + i % 16] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live)
            for (uint ds = 0; ds < 16; ds += 8) {
                simdgroup_float8x8 qa[2];
                for (uint h = 0; h < 2; ++h)
                    simdgroup_load(qa[h], qs + (ts * 16 + h * 8) * 16 + ds, 16);
                for (uint c = 0; c < 4; ++c) {
                    simdgroup_float8x8 b;
                    simdgroup_load(b, ks + (ps * 32 + c * 8) * 16 + ds, 16, ulong2(0, 0), true);
                    for (uint h = 0; h < 2; ++h)
                        simdgroup_multiply_accumulate(acc[h][c], qa[h], b, acc[h][c]);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* out = ks + sg * 512;
    for (uint h = 0; h < 2; ++h)
        for (uint c = 0; c < 4; ++c)
            simdgroup_store(acc[h][c], out + h * 8 * 32 + c * 8, 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (live && pool0 + lane < n_pool) {
        float total = 0.0f;
        for (uint h = 0; h < 16; ++h)
            total += max(out[h * 32 + lane], 0.0f) * w[uint64_t(j) * 16 + h];
        score[uint64_t(j) * a.stride + pool0 + lane] = total * a.w_scale;
    }
}

// The indexer's top-k over many threadgroups a token (the single-threadgroup idx_select's latency grows with the
// pools): the same cells. Each token's pools split into `slices` slices of `slice`; hy4_sel_hist three times (key
// bits 31-21, 20-10, 9-0 under the prefix so far), then hy4_sel_count and hy4_sel_write. st holds a token's histogram
// (2048 words, zero between uses) and its counter, prefix and need; cnt each slice's counts above and at the key.
struct SelArgs {
    uint first;  // cells token 0 sees, itself included; token j sees first + j (its own group the tail)
    uint n_top;  // pools to keep
    uint kpool;
    uint score_stride, list_stride;
    uint slice, slices;
    uint shift, bits; // hy4_sel_hist: the digit
};
constant uint kSelWords = 2052;

static uint sel_key(float f)
{
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sum over the threadgroup (TG threads), and the total.
static uint sel_prefix(uint v, threadgroup uint* sums, uint tid, thread uint& total)
{
    const uint in_simd = simd_prefix_exclusive_sum(v);
    if (tid % 32 == 31)
        sums[tid / 32] = in_simd + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    total = 0;
    for (uint w = 0; w < TG / 32; ++w) {
        base += w < tid / 32 ? sums[w] : 0;
        total += sums[w];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return base + in_simd;
}

kernel void hy4_sel_hist(device const float* score [[buffer(0)]], constant SelArgs& a [[buffer(1)]],
                         device atomic_uint* st [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint h[2048];
    threadgroup uint sums[TG / 32];
    threadgroup uint last[1];
    const uint s = tg.x, j = tg.y, n_pool = (a.first + j - 1) / a.kpool, nb = 1u << a.bits;
    device atomic_uint* hist = st + uint64_t(j) * kSelWords;
    device atomic_uint* ctrl = hist + 2048;
    const bool all = n_pool <= a.n_top, top = a.shift + a.bits >= 32;
    const uint mask = top ? 0u : ~0u << (a.shift + a.bits);
    const uint prefix = top ? 0u : atomic_load_explicit(ctrl + 1, memory_order_relaxed);
    for (uint i = tid; i < nb; i += TG)
        atomic_store_explicit(&h[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!all) {
        score += uint64_t(j) * a.score_stride;
        const uint p1 = min(n_pool, (s + 1) * a.slice);
        for (uint p = s * a.slice + tid; p < p1; p += TG) {
            const uint k = sel_key(score[p]);
            if ((k & mask) == prefix)
                atomic_fetch_add_explicit(&h[(k >> a.shift) & (nb - 1)], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!all)
        for (uint i = tid; i < nb; i += TG)
            if (const uint v = atomic_load_explicit(&h[i], memory_order_relaxed))
                atomic_fetch_add_explicit(hist + i, v, memory_order_relaxed);
    // The token's last threadgroup finds the digit (as in ihc's mixes: the fences pass the others' sums on).
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last[0] = atomic_fetch_add_explicit(ctrl, 1u, memory_order_relaxed) + 1 == a.slices ? 1u : 0u;
        if (last[0])
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!last[0])
        return;
    if (all) {
        if (tid == 0) {
            atomic_store_explicit(ctrl, 0, memory_order_relaxed);
            atomic_store_explicit(ctrl + 1, 0, memory_order_relaxed);
            atomic_store_explicit(ctrl + 2, 0, memory_order_relaxed);
        }
        return;
    }
    const uint need = top ? a.n_top : atomic_load_explicit(ctrl + 2, memory_order_relaxed);
    // Thread tid the bins from nb - (tid + 1) * per up, in descending order.
    const uint per = nb / TG, b0 = nb - (tid + 1) * per;
    uint c[8], n = 0;
    for (uint i = 0; i < per; ++i) {
        c[i] = atomic_load_explicit(hist + b0 + per - 1 - i, memory_order_relaxed);
        atomic_store_explicit(hist + b0 + per - 1 - i, 0, memory_order_relaxed);
        n += c[i];
    }
    uint total;
    uint before = sel_prefix(n, sums, tid, total);
    if (before < need && need <= before + n)
        for (uint i = 0; i < per; ++i) {
            if (need <= before + c[i]) {
                atomic_store_explicit(ctrl + 1, prefix | ((b0 + per - 1 - i) << a.shift), memory_order_relaxed);
                atomic_store_explicit(ctrl + 2, need - before, memory_order_relaxed);
                break;
            }
            before += c[i];
        }
    if (tid == 0)
        atomic_store_explicit(ctrl, 0, memory_order_relaxed);
}

// Each slice's pools above the key and at it (all pools when the token keeps every one).
kernel void hy4_sel_count(device const float* score [[buffer(0)]], constant SelArgs& a [[buffer(1)]],
                          device const uint* st [[buffer(2)]], device uint* cnt [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]])
{
    threadgroup uint sums[TG / 32];
    const uint s = tg.x, j = tg.y, n_pool = (a.first + j - 1) / a.kpool;
    const bool all = n_pool <= a.n_top;
    const uint thresh = st[uint64_t(j) * kSelWords + 2049];
    score += uint64_t(j) * a.score_stride;
    const uint p1 = min(n_pool, (s + 1) * a.slice);
    uint gt = 0, eq = 0;
    for (uint p = s * a.slice + tid; p < p1; p += TG) {
        const uint k = sel_key(score[p]);
        gt += all || k > thresh;
        eq += !all && k == thresh;
    }
    uint total_gt, total_eq;
    sel_prefix(gt, sums, tid, total_gt);
    sel_prefix(eq, sums, tid, total_eq);
    if (tid == 0) {
        cnt[(uint64_t(j) * a.slices + s) * 2] = total_gt;
        cnt[(uint64_t(j) * a.slices + s) * 2 + 1] = total_eq;
    }
}

// The kept pools' cells in pool order (ties at the key: the lowest pools), then the tail; the count.
kernel void hy4_sel_write(device const float* score [[buffer(0)]], constant SelArgs& a [[buffer(1)]],
                          device const uint* st [[buffer(2)]], device const uint* cnt [[buffer(3)]],
                          device uint* list [[buffer(4)]], device uint* count [[buffer(5)]],
                          uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]])
{
    threadgroup uint sums[TG / 32];
    const uint s = tg.x, j = tg.y, n_pool = (a.first + j - 1) / a.kpool, n_kv = a.first + j;
    const bool all = n_pool <= a.n_top;
    const uint thresh = st[uint64_t(j) * kSelWords + 2049], ties = all ? 0u : st[uint64_t(j) * kSelWords + 2050];
    // The slices before this one, and all of them.
    cnt += uint64_t(j) * a.slices * 2;
    uint gt_b = 0, eq_b = 0, gt_t = 0, eq_t = 0;
    for (uint i = tid; i < a.slices; i += TG) {
        gt_t += cnt[i * 2];
        eq_t += cnt[i * 2 + 1];
        gt_b += i < s ? cnt[i * 2] : 0;
        eq_b += i < s ? cnt[i * 2 + 1] : 0;
    }
    uint gt_before, eq_before_slice, gt_total, eq_total;
    sel_prefix(gt_b, sums, tid, gt_before);
    sel_prefix(eq_b, sums, tid, eq_before_slice);
    sel_prefix(gt_t, sums, tid, gt_total);
    sel_prefix(eq_t, sums, tid, eq_total);
    const uint kept_total = gt_total + min(ties, eq_total);
    list += uint64_t(j) * a.list_stride;
    score += uint64_t(j) * a.score_stride;
    // Each thread a contiguous range of the slice.
    const uint q0 = min(n_pool, s * a.slice), q1 = min(n_pool, q0 + a.slice);
    const uint per = (q1 - q0 + TG - 1) / TG, p0 = min(q1, q0 + tid * per), p1 = min(q1, p0 + per);
    uint gt = 0, eq = 0;
    for (uint p = p0; p < p1; ++p) {
        const uint k = sel_key(score[p]);
        gt += all || k > thresh;
        eq += !all && k == thresh;
    }
    uint tot;
    const uint eq_before = eq_before_slice + sel_prefix(eq, sums, tid, tot);
    const uint eq_kept = ties > eq_before ? min(eq, ties - eq_before) : 0;
    const uint kept_before = gt_before + min(ties, eq_before_slice) + sel_prefix(gt + eq_kept, sums, tid, tot);
    uint at = kept_before * a.kpool, rank = eq_before;
    for (uint p = p0; p < p1; ++p) {
        const uint k = sel_key(score[p]);
        bool keep = all || k > thresh;
        if (!all && k == thresh)
            keep = rank++ < ties;
        if (keep)
            for (uint c = 0; c < a.kpool; ++c)
                list[at++] = p * a.kpool + c;
    }
    if (s + 1 == a.slices) {
        const uint tail0 = n_pool * a.kpool;
        for (uint c = tail0 + tid; c < n_kv; c += TG)
            list[kept_total * a.kpool + (c - tail0)] = c;
        if (tid == 0)
            count[j] = kept_total * a.kpool + (n_kv - tail0);
    }
}
