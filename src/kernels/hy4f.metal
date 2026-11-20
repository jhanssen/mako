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
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
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
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
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
                             uint tid [[thread_position_in_threadgroup]])
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
