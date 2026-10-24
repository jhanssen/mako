// GLM-5.3-Flash decode kernels (one token). Activations are fp32.
#include <metal_stdlib>
using namespace metal;

constant uint TG = 256;

// Sum of `v` over a threadgroup of TG threads; every thread gets the total.
static float tg_sum(float v, threadgroup float* scratch, uint tid)
{
    v = simd_sum(v);
    if (tid % 32 == 0)
        scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid % 32;
    float total = lane < TG / 32 ? scratch[lane] : 0.0f;
    total = simd_sum(total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

static float tg_max(float v, threadgroup float* scratch, uint tid)
{
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

static float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

struct NormArgs {
    uint n;       // row length
    uint wn;      // weight length (the weight repeats every wn elements); 0: no weight
    float eps;
};

// y = x / sqrt(mean(x^2) + eps) * w, one threadgroup per row.
kernel void rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.n;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        s += xr[i] * xr[i];
    const float inv = rsqrt(tg_sum(s, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += TG)
        yr[i] = xr[i] * inv * (a.wn ? w[i % a.wn] : 1.0f);
}

// y = (x - mean) / sqrt(var + eps) * w + b, one threadgroup per row.
kernel void layer_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                       device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                       constant NormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.n;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        s += xr[i];
    const float mean = tg_sum(s, scratch, tid) / float(a.n);
    float v = 0.0f;
    for (uint i = tid; i < a.n; i += TG) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = rsqrt(tg_sum(v, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += TG)
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

struct HcArgs {
    uint n_embd;
    uint iters;
    float eps;
    uint tokens; // hc_mix
};

// From the 24 mixes of one token: pre (4), post (4) and the Sinkhorn-normalized 4x4 comb.
// comb is stored dst-fastest: comb[src * 4 + dst].
// One thread per token: mixes [tokens][24] -> pc [tokens][24] = pre[4], post[4], comb[16].
kernel void hc_mix(device const float* mixes_all [[buffer(0)]], device const float* scale [[buffer(1)]],
                   device const float* base [[buffer(2)]], device float* pc [[buffer(3)]],
                   constant HcArgs& a [[buffer(4)]], uint t [[thread_position_in_grid]])
{
    if (t >= a.tokens)
        return;
    device const float* mixes = mixes_all + t * 24;
    device float* pre = pc + t * 24;
    device float* post = pre + 4;
    device float* comb = pre + 8;
    for (uint i = 0; i < 4; ++i) {
        pre[i] = sigmoidf(mixes[i] * scale[0] + base[i]) + a.eps;
        post[i] = 2.0f * sigmoidf(mixes[4 + i] * scale[1] + base[4 + i]);
    }
    float c[16];
    for (uint i = 0; i < 16; ++i)
        c[i] = mixes[8 + i] * scale[2] + base[8 + i];
    // softmax over dst for each src, then +eps
    for (uint s = 0; s < 4; ++s) {
        float m = -INFINITY;
        for (uint d = 0; d < 4; ++d)
            m = max(m, c[s * 4 + d]);
        float sum = 0.0f;
        for (uint d = 0; d < 4; ++d) {
            c[s * 4 + d] = exp(c[s * 4 + d] - m);
            sum += c[s * 4 + d];
        }
        for (uint d = 0; d < 4; ++d)
            c[s * 4 + d] = c[s * 4 + d] / sum + a.eps;
    }
    auto norm_cols = [&]() {
        for (uint d = 0; d < 4; ++d) {
            float sum = 0.0f;
            for (uint s = 0; s < 4; ++s)
                sum += c[s * 4 + d];
            sum += a.eps;
            for (uint s = 0; s < 4; ++s)
                c[s * 4 + d] /= sum;
        }
    };
    auto norm_rows = [&]() {
        for (uint s = 0; s < 4; ++s) {
            float sum = 0.0f;
            for (uint d = 0; d < 4; ++d)
                sum += c[s * 4 + d];
            sum += a.eps;
            for (uint d = 0; d < 4; ++d)
                c[s * 4 + d] /= sum;
        }
    };
    norm_cols();
    for (uint i = 1; i < a.iters; ++i) {
        norm_rows();
        norm_cols();
    }
    for (uint i = 0; i < 16; ++i)
        comb[i] = c[i];
}

// x = sum_h R[h] * pre[h]; grid (n_embd, tokens), R [tokens][4][n_embd], pc and x per token.
kernel void hc_collapse(device const float* R [[buffer(0)]], device const float* pc [[buffer(1)]],
                        device float* x [[buffer(2)]], constant HcArgs& a [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n_embd)
        return;
    R += uint64_t(t) * 4 * a.n_embd;
    x += uint64_t(t) * a.n_embd;
    device const float* pre = pc + t * 24;
    float acc = 0.0f;
    for (uint h = 0; h < 4; ++h)
        acc += R[h * a.n_embd + i] * pre[h];
    x[i] = acc;
}

// out[dst] = x * post[dst] + sum_src comb[src][dst] * R[src]; grid (n_embd, tokens); out may not alias R.
kernel void hc_expand(device const float* x [[buffer(0)]], device const float* R [[buffer(1)]],
                      device const float* pc [[buffer(2)]], device float* out [[buffer(3)]],
                      constant HcArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n_embd)
        return;
    x += uint64_t(t) * a.n_embd;
    R += uint64_t(t) * 4 * a.n_embd;
    out += uint64_t(t) * 4 * a.n_embd;
    device const float* post = pc + t * 24 + 4;
    device const float* comb = pc + t * 24 + 8;
    float r[4];
    for (uint s = 0; s < 4; ++s)
        r[s] = R[s * a.n_embd + i];
    const float xi = x[i];
    for (uint d = 0; d < 4; ++d) {
        float acc = xi * post[d];
        for (uint s = 0; s < 4; ++s)
            acc += comb[s * 4 + d] * r[s];
        out[d * a.n_embd + i] = acc;
    }
}

// Mean of the 4 streams; grid (n_embd, tokens).
kernel void hc_mean(device const float* R [[buffer(0)]], device float* x [[buffer(1)]],
                    constant HcArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= a.n_embd)
        return;
    R += uint64_t(gid.y) * 4 * a.n_embd;
    x += uint64_t(gid.y) * a.n_embd;
    x[i] = 0.25f * (R[i] + R[a.n_embd + i] + R[2 * a.n_embd + i] + R[3 * a.n_embd + i]);
}

struct ConvArgs {
    uint channels;   // per projection
    uint kernel_len; // taps, the last one multiplies the current token (at most 8)
    uint tokens;     // x and y are [tokens][3 * channels]
};

// Causal depthwise conv over q|k|v (3 * channels), then SiLU, for consecutive tokens. The state holds the
// previous kernel_len - 1 inputs per channel, oldest first; one thread per channel walks the tokens.
kernel void kda_conv(device const float* x [[buffer(0)]], device const float* wq [[buffer(1)]],
                     device const float* wk [[buffer(2)]], device const float* wv [[buffer(3)]],
                     device float* state [[buffer(4)]], device float* y [[buffer(5)]],
                     constant ConvArgs& a [[buffer(6)]], uint c [[thread_position_in_grid]])
{
    const uint width = 3 * a.channels;
    if (c >= width)
        return;
    const uint proj = c / a.channels, ch = c % a.channels, taps = a.kernel_len - 1;
    device const float* w = (proj == 0 ? wq : proj == 1 ? wk : wv) + uint64_t(ch) * a.kernel_len;
    device float* s = state + uint64_t(c) * taps;
    float hist[7];
    for (uint j = 0; j < taps; ++j)
        hist[j] = s[j];
    for (uint t = 0; t < a.tokens; ++t) {
        const float xc = x[uint64_t(t) * width + c];
        float acc = w[taps] * xc;
        for (uint j = 0; j < taps; ++j)
            acc += w[j] * hist[j];
        for (uint j = 0; j + 1 < taps; ++j)
            hist[j] = hist[j + 1];
        hist[taps - 1] = xc;
        y[uint64_t(t) * width + c] = acc * sigmoidf(acc);
    }
    for (uint j = 0; j < taps; ++j)
        s[j] = hist[j];
}

struct GateArgs {
    uint head_dim;
    uint heads;
    float lower_bound;
};

// g = lower_bound * sigmoid(-(A[h] * (raw + dt_bias))) per channel; beta = sigmoid(raw_beta) per head.
// Grid (channels, tokens); raw, g [tokens][channels], raw_beta, beta [tokens][heads].
kernel void kda_gate(device const float* raw [[buffer(0)]], device const float* dt_bias [[buffer(1)]],
                     device const float* A [[buffer(2)]], device const float* raw_beta [[buffer(3)]],
                     device float* g [[buffer(4)]], device float* beta [[buffer(5)]],
                     constant GateArgs& a [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, inner = a.head_dim * a.heads;
    raw += uint64_t(gid.y) * inner;
    g += uint64_t(gid.y) * inner;
    raw_beta += gid.y * a.heads;
    beta += gid.y * a.heads;
    if (c < a.heads)
        beta[c] = sigmoidf(raw_beta[c]);
    if (c >= a.head_dim * a.heads)
        return;
    const float v = (raw[c] + dt_bias[c]) * A[c / a.head_dim];
    g[c] = a.lower_bound * sigmoidf(-v);
}

struct L2Args {
    uint n;
    float eps;
    uint token_stride; // elements between tokens
};

// x / max(||x||, eps) per row of n, in place; threadgroups (rows, tokens).
kernel void l2_norm(device float* x [[buffer(0)]], constant L2Args& a [[buffer(1)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    const uint tid = tid2.x;
    device float* xr = x + uint64_t(tg.y) * a.token_stride + uint64_t(tg.x) * a.n;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        s += xr[i] * xr[i];
    const float scale = 1.0f / max(sqrt(tg_sum(s, scratch, tid)), a.eps);
    for (uint i = tid; i < a.n; i += TG)
        xr[i] *= scale;
}

struct ScanArgs {
    uint head_dim;   // 128
    uint tokens;
    uint qkv_stride; // q, k, v elements between tokens
    uint inner;      // g and o elements between tokens
    uint heads;      // beta elements between tokens
};

// One delta-rule step with per-channel decay. M is the state transposed (M[j][i] = S[i][j]):
//   S[i][:] *= exp(g[i]); delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta
//   S[i][j] += k[i] * delta[j];   o[j] = sum_i S[i][j] q[i] / sqrt(head_dim)
// One threadgroup of head_dim threads per head, stepping through the tokens; thread j owns row j of M.
kernel void kda_scan(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                     device const float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                     device float* o [[buffer(6)]], constant ScanArgs& a [[buffer(7)]],
                     uint h [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]])
{
    threadgroup float qs[128], ks[128], eg[128];
    const uint D = a.head_dim;
    const uint base = h * D;
    device float* M = state + uint64_t(h) * D * D + uint64_t(j) * D;
    for (uint t = 0; t < a.tokens; ++t) {
        const uint64_t qo = uint64_t(t) * a.qkv_stride + base, go = uint64_t(t) * a.inner + base;
        qs[j] = q[qo + j];
        ks[j] = k[qo + j];
        eg[j] = exp(g[go + j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint i = 0; i < D; ++i) {
            const float m = M[i] * eg[i];
            M[i] = m;
            sum += m * ks[i];
        }
        const float delta = (v[qo + j] - sum) * beta[t * a.heads + h];
        float out = 0.0f;
        for (uint i = 0; i < D; ++i) {
            const float m = M[i] + ks[i] * delta;
            M[i] = m;
            out += m * qs[i];
        }
        o[go + j] = out * rsqrt(float(D));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// y = rms_norm(x) * w * sigmoid(gate), one threadgroup per head row.
kernel void rms_norm_gate(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                          device const float* gate [[buffer(2)]], device float* y [[buffer(3)]],
                          constant NormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    const uint64_t off = uint64_t(row) * a.n;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        s += x[off + i] * x[off + i];
    const float inv = rsqrt(tg_sum(s, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += TG)
        y[off + i] = x[off + i] * inv * w[i % a.wn] * sigmoidf(gate[off + i]);
}

struct AttnArgs {
    uint dim;     // latent width, 512
    uint n_kv;    // cached tokens, the current one included
    float scale;
};

// Decode attention of one head over the latent cache: p = softmax(q . K * scale), out = p . K.
kernel void mla_attn(device const float* q [[buffer(0)]], device const half* kv [[buffer(1)]],
                     device float* out [[buffer(2)]], device float* scores [[buffer(3)]],
                     constant AttnArgs& a [[buffer(4)]], uint h [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* qh = q + uint64_t(h) * a.dim;
    device float* sh = scores + uint64_t(h) * a.n_kv;
    float m = -INFINITY;
    for (uint t = tid; t < a.n_kv; t += TG) {
        device const half* kt = kv + uint64_t(t) * a.dim;
        float d = 0.0f;
        for (uint i = 0; i < a.dim; ++i)
            d += qh[i] * float(kt[i]);
        d *= a.scale;
        sh[t] = d;
        m = max(m, d);
    }
    m = tg_max(m, scratch, tid);
    float s = 0.0f;
    for (uint t = tid; t < a.n_kv; t += TG) {
        const float e = exp(sh[t] - m);
        sh[t] = e;
        s += e;
    }
    threadgroup_barrier(mem_flags::mem_device);
    const float inv = 1.0f / tg_sum(s, scratch, tid);
    for (uint i = tid; i < a.dim; i += TG) {
        float acc = 0.0f;
        for (uint t = 0; t < a.n_kv; ++t)
            acc += sh[t] * float(kv[uint64_t(t) * a.dim + i]);
        out[uint64_t(h) * a.dim + i] = acc * inv;
    }
}

kernel void store_half(device const float* x [[buffer(0)]], device half* y [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = half(x[i]);
}

struct SwigluArgs {
    uint n;
    float limit;   // <= 0: no clamp
};

// y = silu(min(g, limit)) * clamp(u, -limit, limit); g and u are [rows][n].
kernel void swiglu(device const float* gt [[buffer(0)]], device const float* up [[buffer(1)]],
                   device float* y [[buffer(2)]], constant SwigluArgs& a [[buffer(3)]],
                   uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    float g = gt[i], u = up[i];
    if (a.limit > 0.0f) {
        g = min(g, a.limit);
        u = clamp(u, -a.limit, a.limit);
    }
    y[i] = g * sigmoidf(g) * u;
}

struct CombineArgs {
    uint n;
    uint n_used;
};

// y = shared + sum_e w[e] * x[e]
kernel void moe_combine(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device const float* shared [[buffer(2)]], device float* y [[buffer(3)]],
                        constant CombineArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    float acc = shared[i];
    for (uint e = 0; e < a.n_used; ++e)
        acc += w[e] * x[uint64_t(e) * a.n + i];
    y[i] = acc;
}

struct RouteArgs {
    uint n_expert, n_used, norm;
    float scale;
};

// Router top-k over sigmoid scores, the bias added for the selection only; ties keep the lower expert index.
// Weights are the unbiased scores of the selected experts, normalized when a.norm, times a.scale.
// Then each selected expert is looked up in `slots` (this layer's expert -> arena slot, -1 absent): a resident
// one gets its slot in hit_ids and miss_rank -1; a missing one slot 0 (its result is not used) and its index
// among the misses. The miss count goes to the z of the three indirect dispatches (gate, up, down).
// One threadgroup of TG threads; sel, w, hit_ids, miss_rank get n_used (<= 16) entries, biased n_expert.
kernel void moe_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device int* sel [[buffer(2)]], device float* w [[buffer(3)]], device float* biased [[buffer(4)]],
                      constant RouteArgs& a [[buffer(5)]], device const int* slots [[buffer(6)]],
                      device int* hit_ids [[buffer(7)]], device int* miss_rank [[buffer(8)]],
                      device uint* indirect [[buffer(9)]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float best_v[TG / 32];
    threadgroup uint best_i[TG / 32];
    threadgroup uint chosen[16];
    for (uint e = tid; e < a.n_expert; e += TG)
        biased[e] = 1.0f / (1.0f + precise::exp(-logits[e])) + bias[e];
    threadgroup_barrier(mem_flags::mem_device);
    for (uint r = 0; r < a.n_used; ++r) {
        float v = -INFINITY;
        uint idx = 0xffffffffu;
        for (uint e = tid; e < a.n_expert; e += TG) {
            bool taken = false;
            for (uint j = 0; j < r; ++j)
                taken |= chosen[j] == e;
            const float b = biased[e];
            if (!taken && (b > v || (b == v && e < idx))) {
                v = b;
                idx = e;
            }
        }
        const float m = simd_max(v);
        const uint mi = simd_min(v == m ? idx : 0xffffffffu);
        if (tid % 32 == 0) {
            best_v[tid / 32] = m;
            best_i[tid / 32] = mi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float bv = -INFINITY;
            uint bi = 0xffffffffu;
            for (uint s = 0; s < TG / 32; ++s)
                if (best_v[s] > bv || (best_v[s] == bv && best_i[s] < bi)) {
                    bv = best_v[s];
                    bi = best_i[s];
                }
            chosen[r] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float p[16], sum = 0.0f;
        for (uint j = 0; j < a.n_used; ++j) {
            p[j] = 1.0f / (1.0f + precise::exp(-logits[chosen[j]]));
            sum += p[j];
        }
        if (a.norm)
            sum = max(sum, 6.103515625e-5f);
        uint misses = 0;
        for (uint j = 0; j < a.n_used; ++j) {
            sel[j] = int(chosen[j]);
            w[j] = (a.norm ? p[j] / sum : p[j]) * a.scale;
            const int s = slots[chosen[j]];
            hit_ids[j] = max(s, 0);
            miss_rank[j] = s >= 0 ? -1 : int(misses++);
        }
        for (uint d = 0; d < 3; ++d)
            indirect[d * 3 + 2] = misses;
    }
}

// moe_combine over results split into resident experts (x_hit, row e) and misses (x_miss, row miss_rank[e]).
kernel void moe_combine_split(device const float* x_hit [[buffer(0)]], device const float* x_miss [[buffer(1)]],
                              device const int* miss_rank [[buffer(2)]], device const float* w [[buffer(3)]],
                              device const float* shared [[buffer(4)]], device float* y [[buffer(5)]],
                              constant CombineArgs& a [[buffer(6)]], uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    float acc = shared[i];
    for (uint e = 0; e < a.n_used; ++e) {
        const int r = miss_rank[e];
        acc += w[e] * (r < 0 ? x_hit[uint64_t(e) * a.n + i] : x_miss[uint64_t(r) * a.n + i]);
    }
    y[i] = acc;
}

kernel void add(device const float* x [[buffer(0)]], device const float* z [[buffer(1)]],
                device float* y [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = x[i] + z[i];
}

// R[h] = x for every stream h.
// Grid (n_embd, tokens).
kernel void hc_init(device const float* x [[buffer(0)]], device float* R [[buffer(1)]],
                    constant HcArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= a.n_embd)
        return;
    x += uint64_t(gid.y) * a.n_embd;
    R += uint64_t(gid.y) * 4 * a.n_embd;
    for (uint h = 0; h < 4; ++h)
        R[h * a.n_embd + i] = x[i];
}

struct PoolArgs {
    uint dim;      // indexer head dim, 128
    uint kpool;    // tokens per pool
    uint pool;     // the pool to build
};

// pooled[d] = sum_m softmax_m(gate[m][d] + ape[m][d]) * key[m][d] over the pool's members m.
// cache rows are key | gate per token.
kernel void idx_pool(device const float* cache [[buffer(0)]], device const float* ape [[buffer(1)]],
                     device float* pooled [[buffer(2)]], constant PoolArgs& a [[buffer(3)]],
                     uint d [[thread_position_in_grid]])
{
    if (d >= a.dim)
        return;
    const uint first = a.pool * a.kpool;
    float m = -INFINITY;
    for (uint j = 0; j < a.kpool; ++j)
        m = max(m, cache[uint64_t(first + j) * 2 * a.dim + a.dim + d] + ape[j * a.dim + d]);
    float s = 0.0f, acc = 0.0f;
    for (uint j = 0; j < a.kpool; ++j) {
        device const float* row = cache + uint64_t(first + j) * 2 * a.dim;
        const float e = exp(row[a.dim + d] + ape[j * a.dim + d] - m);
        s += e;
        acc += e * row[d];
    }
    pooled[uint64_t(a.pool) * a.dim + d] = acc / s;
}

struct ScoreArgs {
    uint dim;
    uint heads;
    uint n_pool;
    float w_scale;
};

// score[p] = sum_h relu(q_h . pooled_p) * w_h
kernel void idx_score(device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* pooled [[buffer(2)]], device float* score [[buffer(3)]],
                      constant ScoreArgs& a [[buffer(4)]], uint p [[thread_position_in_grid]])
{
    if (p >= a.n_pool)
        return;
    device const float* k = pooled + uint64_t(p) * a.dim;
    float total = 0.0f;
    for (uint h = 0; h < a.heads; ++h) {
        float dot = 0.0f;
        for (uint i = 0; i < a.dim; ++i)
            dot += q[h * a.dim + i] * k[i];
        total += max(dot, 0.0f) * w[h];
    }
    score[p] = total * a.w_scale;
}

struct SelectArgs {
    uint n_pool;   // complete pools
    uint n_top;    // pools to keep
    uint kpool;
    uint n_kv;     // cells, the current token included
};

static uint order_key(float f)
{
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// mask[cell] = 1 for the members of the n_top best pools and for the incomplete tail; one threadgroup.
// Ties at the threshold keep the lowest pool indices.
kernel void idx_select(device const float* score [[buffer(0)]], device uchar* mask [[buffer(1)]],
                       constant SelectArgs& a [[buffer(2)]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup uint sh_prefix, sh_need;
    const uint all = a.n_pool <= a.n_top;
    if (tid == 0) {
        sh_prefix = 0;
        sh_need = a.n_top;
    }
    uint prefix_mask = 0;
    for (int shift = 24; shift >= 0 && !all; shift -= 8) {
        for (uint i = tid; i < 256; i += TG)
            atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = sh_prefix;
        for (uint p = tid; p < a.n_pool; p += TG) {
            const uint k = order_key(score[p]);
            if ((k & prefix_mask) == prefix)
                atomic_fetch_add_explicit(&hist[(k >> shift) & 255], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint need = sh_need;
            for (int b = 255; b >= 0; --b) {
                const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (c >= need) {
                    sh_prefix = prefix | (uint(b) << shift);
                    break;
                }
                need -= c;
            }
            sh_need = need;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix_mask |= 255u << shift;
    }
    const uint thresh = sh_prefix, ties = sh_need;
    // keep everything above the threshold key, and the first `ties` pools equal to it
    if (tid == 0) {
        uint c = 0;
        for (uint p = 0; p < a.n_pool && !all; ++p)
            if (order_key(score[p]) == thresh) {
                mask[p * a.kpool] = c < ties ? 2 : 0;
                ++c;
            }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint p = tid; p < a.n_pool; p += TG) {
        uchar keep;
        if (all)
            keep = 1;
        else {
            const uint k = order_key(score[p]);
            keep = k > thresh ? 1 : (k == thresh ? (mask[p * a.kpool] == 2 ? 1 : 0) : 0);
        }
        for (uint j = 0; j < a.kpool; ++j)
            mask[p * a.kpool + j] = keep;
    }
    for (uint c = a.n_pool * a.kpool + tid; c < a.n_kv; c += TG)
        mask[c] = 1;
}

// Decode attention like mla_attn over the cells whose mask is set.
kernel void mla_attn_masked(device const float* q [[buffer(0)]], device const half* kv [[buffer(1)]],
                            device float* out [[buffer(2)]], device float* scores [[buffer(3)]],
                            device const uchar* mask [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
                            uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* qh = q + uint64_t(h) * a.dim;
    device float* sh = scores + uint64_t(h) * a.n_kv;
    float m = -INFINITY;
    for (uint t = tid; t < a.n_kv; t += TG) {
        if (!mask[t]) {
            sh[t] = -INFINITY;
            continue;
        }
        device const half* kt = kv + uint64_t(t) * a.dim;
        float d = 0.0f;
        for (uint i = 0; i < a.dim; ++i)
            d += qh[i] * float(kt[i]);
        d *= a.scale;
        sh[t] = d;
        m = max(m, d);
    }
    m = tg_max(m, scratch, tid);
    float s = 0.0f;
    for (uint t = tid; t < a.n_kv; t += TG) {
        const float e = mask[t] ? exp(sh[t] - m) : 0.0f;
        sh[t] = e;
        s += e;
    }
    threadgroup_barrier(mem_flags::mem_device);
    const float inv = 1.0f / tg_sum(s, scratch, tid);
    for (uint i = tid; i < a.dim; i += TG) {
        float acc = 0.0f;
        for (uint t = 0; t < a.n_kv; ++t)
            if (mask[t])
                acc += sh[t] * float(kv[uint64_t(t) * a.dim + i]);
        out[uint64_t(h) * a.dim + i] = acc * inv;
    }
}

kernel void copy_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = x[i];
}
