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
    uint ystride; // rms_norm: y row stride in elements (0: n)
};

// y = x / sqrt(mean(x^2) + eps) * w (w of n, or none when wn is 0), one threadgroup per row; n a multiple of 4.
kernel void rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float4* xr = reinterpret_cast<device const float4*>(x + uint64_t(row) * a.n);
    device float4* yr = reinterpret_cast<device float4*>(y + uint64_t(row) * (a.ystride ? a.ystride : a.n));
    device const float4* w4 = reinterpret_cast<device const float4*>(w);
    const uint n4 = a.n / 4;
    float s = 0.0f;
    for (uint i = tid; i < n4; i += TG) {
        const float4 v = xr[i];
        s += dot(v, v);
    }
    const float inv = rsqrt(tg_sum(s, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < n4; i += TG)
        yr[i] = xr[i] * inv * (a.wn ? w4[i] : float4(1.0f));
}

// y = (x - mean) / sqrt(var + eps) * w + b, one threadgroup per row.
struct LayerNormArgs {
    uint n;
    float eps;
    uint ystride; // y row stride in elements
};

kernel void layer_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                       device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                       constant LayerNormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.ystride;
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

// A few tokens' stream step, as hc_post then the next hc_pre (norm, mixing GEMV, hc_mix, hc_collapse) would do it,
// in two dispatches. A, grid (parts, tokens): threadgroups over each token's flattened stream (j = h * n_embd + i):
// with `expand`, R_new[j] = x[i] * post[h] + sum_s comb[s][h] * R[s][i] is written to out (else R_new = R); each
// writes its partial sum of R_new^2 and of the 24 dot products fn[m] . R_new to part[token][tg][25]. Per token: x
// rows of n_embd, R rows of 4 * n_embd, pc rows of 24.
struct HcStepArgs {
    uint n_embd;
    uint expand;
    uint parts; // threadgroups of A, at most 64
    uint iters;
    float eps;  // hc
    float rms_eps;
};
constant uint HC_TG = 256;

kernel void hc_step_a(device const float* x [[buffer(0)]], device const float* R [[buffer(1)]],
                      device const float* pc [[buffer(2)]], device float* out [[buffer(3)]],
                      device const float* fn [[buffer(4)]], device float* part [[buffer(5)]],
                      constant HcStepArgs& a [[buffer(6)]], uint2 tg2 [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup float red[25][HC_TG / 32];
    const uint tg = tg2.x, tok = tg2.y, tid = tid2.x;
    const uint hd = 4 * a.n_embd, per = (hd + a.parts - 1) / a.parts;
    x += uint64_t(tok) * a.n_embd;
    R += uint64_t(tok) * hd;
    pc += tok * 24;
    out += uint64_t(tok) * hd;
    part += uint64_t(tok) * a.parts * 25;
    const uint j0 = tg * per, j1 = min(hd, j0 + per);
    float acc[25] = {};
    for (uint j = j0 + tid; j < j1; j += HC_TG) {
        float r;
        if (a.expand) {
            const uint h = j / a.n_embd, i = j % a.n_embd;
            r = x[i] * pc[4 + h];
            for (uint src = 0; src < 4; ++src)
                r += pc[8 + src * 4 + h] * R[src * a.n_embd + i];
            out[j] = r;
        } else {
            r = R[j];
        }
        acc[24] += r * r;
        for (uint m = 0; m < 24; ++m)
            acc[m] += fn[uint64_t(m) * hd + j] * r;
    }
    for (uint m = 0; m < 25; ++m) {
        const float v = simd_sum(acc[m]);
        if (tid % 32 == 0)
            red[m][tid / 32] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 25) {
        float v = 0.0f;
        for (uint w = 0; w < HC_TG / 32; ++w)
            v += red[tid][w];
        part[tg * 25 + tid] = v;
    }
}

// B, one threadgroup per token: mixes = (sum of the partial dots) / rms(R_new), then hc_mix -> pc, then x = sum_h
// pre[h] * R_new[h].
kernel void hc_step_b(device const float* part [[buffer(0)]], device const float* Rn [[buffer(1)]],
                      device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                      device float* pc [[buffer(4)]], device float* x [[buffer(5)]],
                      constant HcStepArgs& a [[buffer(6)]], uint tid [[thread_position_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]], uint tok [[threadgroup_position_in_grid]])
{
    part += uint64_t(tok) * a.parts * 25;
    Rn += uint64_t(tok) * 4 * a.n_embd;
    pc += tok * 24;
    x += uint64_t(tok) * a.n_embd;
    threadgroup float mixes[25];
    threadgroup float pre[4];
    threadgroup float parts[64 * 25];
    // One load per thread (parts <= 64), then the sums from threadgroup memory: dependent device loads are slow.
    for (uint i = tid; i < a.parts * 25; i += nt)
        parts[i] = part[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 25) {
        float v = 0.0f;
        for (uint p = 0; p < a.parts; ++p)
            v += parts[p * 25 + tid];
        mixes[tid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Lanes 0..15 of the first simdgroup hold comb element (src, dst) = (lane / 4, lane % 4); row and column sums
    // go through shuffles. Lanes 0..3 also produce pre and post.
    if (tid < 32) {
        const uint lane = tid, s = (lane / 4) % 4, d = lane % 4;
        const float inv = rsqrt(mixes[24] / float(4 * a.n_embd) + a.rms_eps);
        if (lane < 4) {
            pre[lane] = sigmoidf(mixes[lane] * inv * scale[0] + base[lane]) + a.eps;
            pc[lane] = pre[lane];
            pc[4 + lane] = 2.0f * sigmoidf(mixes[4 + lane] * inv * scale[1] + base[4 + lane]);
        }
        float c = mixes[8 + (lane % 16)] * inv * scale[2] + base[8 + (lane % 16)];
        // softmax over dst within each src row (lanes 4s..4s+3), then +eps
        const float mx = max(max(simd_shuffle(c, s * 4), simd_shuffle(c, s * 4 + 1)),
                             max(simd_shuffle(c, s * 4 + 2), simd_shuffle(c, s * 4 + 3)));
        c = exp(c - mx);
        float sum = simd_shuffle(c, s * 4) + simd_shuffle(c, s * 4 + 1) + simd_shuffle(c, s * 4 + 2) +
                    simd_shuffle(c, s * 4 + 3);
        c = c / sum + a.eps;
        for (uint it = 0; it < a.iters; ++it) {
            if (it > 0) {
                sum = simd_shuffle(c, s * 4) + simd_shuffle(c, s * 4 + 1) + simd_shuffle(c, s * 4 + 2) +
                      simd_shuffle(c, s * 4 + 3);
                c /= sum + a.eps;
            }
            sum = simd_shuffle(c, d) + simd_shuffle(c, 4 + d) + simd_shuffle(c, 8 + d) + simd_shuffle(c, 12 + d);
            c /= sum + a.eps;
        }
        if (lane < 16)
            pc[8 + lane] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.n_embd; i += nt)
        x[i] = Rn[i] * pre[0] + Rn[a.n_embd + i] * pre[1] + Rn[2 * a.n_embd + i] * pre[2] + Rn[3 * a.n_embd + i] * pre[3];
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
    uint snap;       // nonzero: also write the state after this many tokens to snap_state (for a rollback)
};

// Causal depthwise conv over q|k|v (3 * channels), then SiLU, for consecutive tokens. The state holds the
// previous kernel_len - 1 inputs per channel, oldest first; one thread per channel walks the tokens.
kernel void kda_conv(device const float* x [[buffer(0)]], device const float* wq [[buffer(1)]],
                     device const float* wk [[buffer(2)]], device const float* wv [[buffer(3)]],
                     device float* state [[buffer(4)]], device float* y [[buffer(5)]],
                     constant ConvArgs& a [[buffer(6)]], device float* snap_state [[buffer(7)]],
                     uint c [[thread_position_in_grid]])
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
        if (t + 1 == a.snap)
            for (uint j = 0; j < taps; ++j)
                snap_state[uint64_t(c) * taps + j] = hist[j];
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
    uint snap;       // nonzero: also write the state after this many tokens to snap_state (for a rollback)
};

// One delta-rule step with per-channel decay. M is the state transposed (M[j][i] = S[i][j]):
//   S[i][:] *= exp(g[i]); delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta
//   S[i][j] += k[i] * delta[j];   o[j] = sum_i S[i][j] q[i] / sqrt(head_dim)
// Threadgroups (heads, head_dim / 32) of 128 threads: each keeps rows j0..j0+31 of its head's M in threadgroup
// memory for all the tokens (rows are independent: row j needs only the shared q, k, g and its own v[j]); four
// threads per row, thread lane covering columns lane, lane + 4, ... . head_dim must be 128.
kernel void kda_scan(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                     device const float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                     device float* o [[buffer(6)]], constant ScanArgs& a [[buffer(7)]],
                     device float* snap_state [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint2 tid2 [[thread_position_in_threadgroup]])
{
    constexpr uint D = 128, ROWS = 32, STRIDE = 132; // padded rows: the 4 lanes x 8 rows of a simdgroup hit distinct banks
    threadgroup float Ms[ROWS * STRIDE];
    threadgroup float qs[D], ks[D], eg[D];
    const uint tid = tid2.x, h = tg.x, j0 = tg.y * ROWS, row = tid / 4, lane = tid % 4;
    const uint base = h * D;
    device float* Mg = state + uint64_t(h) * D * D + uint64_t(j0) * D;
    for (uint idx = tid; idx < ROWS * D; idx += 128)
        Ms[(idx / D) * STRIDE + idx % D] = Mg[idx];
    threadgroup float* M = Ms + row * STRIDE;
    for (uint t = 0; t < a.tokens; ++t) {
        const uint64_t qo = uint64_t(t) * a.qkv_stride + base, go = uint64_t(t) * a.inner + base;
        qs[tid] = q[qo + tid];
        ks[tid] = k[qo + tid];
        eg[tid] = exp(g[go + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint i = lane; i < D; i += 4) {
            const float m = M[i] * eg[i];
            M[i] = m;
            sum += m * ks[i];
        }
        sum = quad_sum(sum);
        const float delta = (v[qo + j0 + row] - sum) * beta[t * a.heads + h];
        float out = 0.0f;
        for (uint i = lane; i < D; i += 4) {
            const float m = M[i] + ks[i] * delta;
            M[i] = m;
            out += m * qs[i];
        }
        out = quad_sum(out);
        if (lane == 0)
            o[go + j0 + row] = out * rsqrt(float(D));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t + 1 == a.snap) {
            device float* Ss = snap_state + uint64_t(h) * D * D + uint64_t(j0) * D;
            for (uint idx = tid; idx < ROWS * D; idx += 128)
                Ss[idx] = Ms[(idx / D) * STRIDE + idx % D];
        }
    }
    for (uint idx = tid; idx < ROWS * D; idx += 128)
        Mg[idx] = Ms[(idx / D) * STRIDE + idx % D];
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

struct SoftmaxArgs {
    uint n_kv;      // row length
    float scale;
    uint first_kv;  // keys visible to row 0; row r sees first_kv + r / heads (causal across query tokens)
    uint heads;
};

// In place, per row of n_kv scores (rows: query tokens x heads): p = softmax(scores * scale) over the row's
// visible keys, zero beyond. One threadgroup per row.
kernel void attn_softmax(device float* scores [[buffer(0)]], constant SoftmaxArgs& a [[buffer(1)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device float* sh = scores + uint64_t(row) * a.n_kv;
    const uint vis = min(a.n_kv, a.first_kv + row / a.heads);
    float m = -INFINITY;
    for (uint t = tid; t < vis; t += TG)
        m = max(m, sh[t] * a.scale);
    m = tg_max(m, scratch, tid);
    float s = 0.0f;
    for (uint t = tid; t < a.n_kv; t += TG) {
        const float e = t < vis ? exp(sh[t] * a.scale - m) : 0.0f;
        sh[t] = e;
        s += e;
    }
    const float inv = 1.0f / tg_sum(s, scratch, tid);
    for (uint t = tid; t < a.n_kv; t += TG)
        sh[t] *= inv;
}

// Softmax of each (token, head) score row over its token's count of list entries, scaled; entries past the count
// become zero. Threadgroup per row.
struct ListSoftmaxArgs {
    uint stride; // row length
    float scale;
    uint heads;  // rows per token
};
kernel void attn_softmax_list(device float* scores [[buffer(0)]], device const uint* count [[buffer(1)]],
                              constant ListSoftmaxArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device float* sh = scores + uint64_t(row) * a.stride;
    const uint n = count[row / a.heads];
    float m = -INFINITY;
    for (uint t = tid; t < n; t += TG)
        m = max(m, sh[t] * a.scale);
    m = tg_max(m, scratch, tid);
    float s = 0.0f;
    for (uint t = tid; t < a.stride; t += TG) {
        const float e = t < n ? exp(sh[t] * a.scale - m) : 0.0f;
        sh[t] = e;
        s += e;
    }
    const float inv = 1.0f / tg_sum(s, scratch, tid);
    for (uint t = tid; t < a.stride; t += TG)
        sh[t] *= inv;
}

// y[i] = sum over splits of partial[split][i].
kernel void sum_splits(device const float* partial [[buffer(0)]], device float* y [[buffer(1)]],
                       constant uint2& a [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i >= a.x)
        return;
    float acc = 0.0f;
    for (uint s = 0; s < a.y; ++s)
        acc += partial[uint64_t(s) * a.x + i];
    y[i] = acc;
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
template <typename T>
kernel void swiglu(device const float* gt [[buffer(0)]], device const float* up [[buffer(1)]],
                   device T* y [[buffer(2)]], constant SwigluArgs& a [[buffer(3)]],
                   uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    float g = gt[i], u = up[i];
    if (a.limit > 0.0f) {
        g = min(g, a.limit);
        u = clamp(u, -a.limit, a.limit);
    }
    y[i] = T(g * sigmoidf(g) * u);
}
template [[host_name("swiglu")]] kernel void swiglu<float>(device const float*, device const float*, device float*,
                                                          constant SwigluArgs&, uint);
template [[host_name("swiglu_half")]] kernel void swiglu<half>(device const float*, device const float*, device half*,
                                                              constant SwigluArgs&, uint);

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
    uint lookup; // look the selection up in slots (one token)
};

// Router top-k over sigmoid scores, the bias added for the selection only; ties keep the lower expert index.
// Weights are the unbiased scores of the selected experts, normalized when a.norm, times a.scale.
// Then each selected expert is looked up in `slots` (this layer's expert -> arena slot, -1 absent): a resident
// one gets its slot in hit_ids and miss_rank -1; a missing one slot 0 (its result is not used) and its index
// among the misses. The miss count goes to the z of the three indirect dispatches (gate, up, down).
// One simdgroup (32 threads) per token t: logits and biased rows of n_expert (<= 512), sel, w, hit_ids and
// miss_rank rows of 16 (n_used <= 16 used), indirect rows of 9 (the last three only with a.lookup). Lane l keeps experts l, l + 32, ... in
// registers; each of the n_used rounds takes the largest biased score left (lowest index on ties).
kernel void moe_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device int* sel [[buffer(2)]], device float* w [[buffer(3)]], device float* biased [[buffer(4)]],
                      constant RouteArgs& a [[buffer(5)]], device const int* slots [[buffer(6)]],
                      device int* hit_ids [[buffer(7)]], device int* miss_rank [[buffer(8)]],
                      device uint* indirect [[buffer(9)]], uint lane [[thread_position_in_threadgroup]],
                      uint t [[threadgroup_position_in_grid]])
{
    constexpr uint K = 16;
    logits += uint64_t(t) * a.n_expert;
    biased += uint64_t(t) * a.n_expert;
    sel += t * 16;
    w += t * 16;
    hit_ids += t * 16;
    miss_rank += t * 16;
    indirect += t * 9;
    float v[K];
    for (uint k = 0; k < K; ++k) {
        const uint e = lane + 32 * k;
        v[k] = -INFINITY;
        if (e < a.n_expert) {
            v[k] = 1.0f / (1.0f + precise::exp(-logits[e])) + bias[e];
            biased[e] = v[k];
        }
    }
    uint mine = 0xffffffffu; // the expert chosen in round `lane`
    for (uint r = 0; r < a.n_used; ++r) {
        float bv = -INFINITY;
        uint bk = 0;
        for (uint k = 0; k < K; ++k)
            if (v[k] > bv) { // ascending e within the lane: strict > keeps the lowest index on ties
                bv = v[k];
                bk = k;
            }
        const float m = simd_max(bv);
        const uint e = simd_min(bv == m && bv > -INFINITY ? lane + 32 * bk : 0xffffffffu);
        if (e % 32 == lane)
            v[e / 32] = -INFINITY;
        if (lane == r)
            mine = e;
    }
    const bool used = lane < a.n_used;
    const float p = used ? 1.0f / (1.0f + precise::exp(-logits[mine])) : 0.0f;
    float sum = 0.0f;
    for (uint j = 0; j < a.n_used; ++j)
        sum += simd_shuffle(p, j);
    if (a.norm)
        sum = max(sum, 6.103515625e-5f);
    if (used) {
        sel[lane] = int(mine);
        w[lane] = (a.norm ? p / sum : p) * a.scale;
    }
    if (a.lookup) {
        const int s = used ? slots[mine] : 0;
        const uint miss = used && s < 0 ? 1u : 0u;
        const uint rank = simd_prefix_exclusive_sum(miss);
        if (used) {
            hit_ids[lane] = max(s, 0);
            miss_rank[lane] = miss ? int(rank) : -1;
        }
        const uint misses = simd_sum(miss);
        if (lane < 3)
            indirect[lane * 3 + 2] = misses;
    }
}

// moe_combine over results split into resident experts (x_hit, row e) and misses (x_miss, row miss_rank[e]).
// Grid (n, tokens): token t's rows are x_hit[t * n_used + e], x_miss[t * n_used + rank], miss_rank and w rows of
// 16, shared and y rows of n.
kernel void moe_combine_split(device const float* x_hit [[buffer(0)]], device const float* x_miss [[buffer(1)]],
                              device const int* miss_rank [[buffer(2)]], device const float* w [[buffer(3)]],
                              device const float* shared [[buffer(4)]], device float* y [[buffer(5)]],
                              constant CombineArgs& a [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n)
        return;
    x_hit += uint64_t(t) * a.n_used * a.n;
    x_miss += uint64_t(t) * a.n_used * a.n;
    miss_rank += t * 16;
    w += t * 16;
    shared += uint64_t(t) * a.n;
    y += uint64_t(t) * a.n;
    float acc = shared[i];
    for (uint e = 0; e < a.n_used; ++e) {
        const int r = miss_rank[e];
        acc += w[e] * (r < 0 ? x_hit[uint64_t(e) * a.n + i] : x_miss[uint64_t(r) * a.n + i]);
    }
    y[i] = acc;
}

// y[i] = x[idx[i]], rows of n; grid (n, rows of y).
template <typename T>
kernel void gather_rows(device const float* x [[buffer(0)]], device const int* idx [[buffer(1)]],
                        device T* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n)
        y[uint64_t(gid.y) * n + gid.x] = T(x[uint64_t(idx[gid.y]) * n + gid.x]);
}
template [[host_name("gather_rows")]] kernel void gather_rows<float>(device const float*, device const int*,
                                                                    device float*, constant uint&, uint2);
template [[host_name("gather_rows_half")]] kernel void gather_rows<half>(device const float*, device const int*,
                                                                        device half*, constant uint&, uint2);

// Per token t: y[t] = shared[t] + sum_e w[t][e] * x[pos[t][e]], over n_used experts (w, pos rows of 16);
// grid (n, tokens).
kernel void moe_combine_rows(device const float* x [[buffer(0)]], device const int* pos [[buffer(1)]],
                             device const float* w [[buffer(2)]], device const float* shared [[buffer(3)]],
                             device float* y [[buffer(4)]], constant CombineArgs& a [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= a.n)
        return;
    float acc = shared[uint64_t(t) * a.n + i];
    for (uint e = 0; e < a.n_used; ++e)
        acc += w[t * 16 + e] * x[uint64_t(pos[t * 16 + e]) * a.n + i];
    y[uint64_t(t) * a.n + i] = acc;
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
// Pools a.pool, a.pool + 1, ... ; grid (dim, pools).
kernel void idx_pool(device const float* cache [[buffer(0)]], device const float* ape [[buffer(1)]],
                     device float* pooled [[buffer(2)]], constant PoolArgs& a [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, pool = a.pool + gid.y;
    if (d >= a.dim)
        return;
    const uint first = pool * a.kpool;
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
    pooled[uint64_t(pool) * a.dim + d] = acc / s;
}

struct ScoreArgs {
    uint dim;
    uint heads;
    uint first;  // cells token 0 sees (its position + 1); token j sees first + j, so (first + j) / kpool pools
    uint kpool;
    float w_scale;
    uint stride; // floats between tokens' score rows
};

// For consecutive tokens j (q, w rows): score[j][p] = sum_h relu(q_h . pooled_p) * w_h over token j's complete
// pools; grid (pools of the last token, tokens).
kernel void idx_score(device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* pooled [[buffer(2)]], device float* score [[buffer(3)]],
                      constant ScoreArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]])
{
    // Threadgroups (pools / 256, tokens): the token's q and w in threadgroup memory, a pool per thread with one
    // accumulator per head (heads <= 32, dim <= 128), so each key element is read once.
    threadgroup float qs[32 * 128], ws[32];
    const uint tid = tid2.x, j = tg.y, p = tg.x * 256 + tid;
    q += uint64_t(j) * a.heads * a.dim;
    w += uint64_t(j) * a.heads;
    for (uint i = tid; i < a.heads * a.dim; i += 256)
        qs[i] = q[i];
    if (tid < a.heads)
        ws[tid] = w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p >= (a.first + j) / a.kpool)
        return;
    device const float* k = pooled + uint64_t(p) * a.dim;
    float dot[32] = {};
    for (uint i = 0; i < a.dim; ++i) {
        const float kv = k[i];
        for (uint h = 0; h < a.heads; ++h)
            dot[h] += qs[h * a.dim + i] * kv;
    }
    float total = 0.0f;
    for (uint h = 0; h < a.heads; ++h)
        total += max(dot[h], 0.0f) * ws[h];
    score[uint64_t(j) * a.stride + p] = total * a.w_scale;
}

// score[j][p] = w_scale * sum_h relu(dots[j][h][p]) * w[j][h] over token j's complete pools; grid (pools of the
// last token, tokens).
kernel void idx_reduce(device const float* dots [[buffer(0)]], device const float* w [[buffer(1)]],
                       device float* score [[buffer(2)]], constant ScoreArgs& a [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    const uint p = gid.x, j = gid.y;
    if (p >= (a.first + j) / a.kpool)
        return;
    dots += uint64_t(j) * a.heads * a.stride + p;
    w += uint64_t(j) * a.heads;
    float total = 0.0f;
    for (uint h = 0; h < a.heads; ++h)
        total += max(dots[uint64_t(h) * a.stride], 0.0f) * w[h];
    score[uint64_t(j) * a.stride + p] = total * a.w_scale;
}

struct SelectArgs {
    uint first;        // cells token 0 sees, itself included; token j sees first + j
    uint n_top;        // pools to keep
    uint kpool;
    uint score_stride; // floats between tokens' score rows
    uint list_stride;  // entries between tokens' cell lists
};
struct SelectRow {
    uint n_pool, n_top, kpool, n_kv;
};

static uint order_key(float f)
{
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sum of v over the threadgroup (TG threads); sums holds TG / 32 entries. Also returns the total.
static uint tg_prefix(uint v, threadgroup uint* sums, uint tid, thread uint& total)
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

// One threadgroup per token: the cells of its n_top best pools (ties at the threshold keep the lowest pool
// indices) and of the incomplete tail, as a sorted list, and their count.
kernel void idx_select(device const float* score [[buffer(0)]], constant SelectArgs& sa [[buffer(1)]],
                       device uint* list [[buffer(2)]], device uint* count [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]], uint j [[threadgroup_position_in_grid]])
{
    score += uint64_t(j) * sa.score_stride;
    list += uint64_t(j) * sa.list_stride;
    const SelectRow a{(sa.first + j) / sa.kpool, sa.n_top, sa.kpool, sa.first + j};
    threadgroup atomic_uint hist[256];
    threadgroup uint sums[TG / 32];
    threadgroup uint sh_prefix, sh_need;
    const bool all = a.n_pool <= a.n_top;
    if (tid == 0) {
        sh_prefix = 0;
        sh_need = a.n_top;
    }
    // Radix select of the n_top-th largest key, 8 bits at a time: the bucket where the count from the top reaches
    // what is still needed (a prefix sum over the buckets in descending order).
    uint prefix_mask = 0, total = 0;
    for (int shift = 24; shift >= 0 && !all; shift -= 8) {
        atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = sh_prefix, need = sh_need;
        for (uint p = tid; p < a.n_pool; p += TG) {
            const uint k = order_key(score[p]);
            if ((k & prefix_mask) == prefix)
                atomic_fetch_add_explicit(&hist[(k >> shift) & 255], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint b = 255 - tid, c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        const uint before = tg_prefix(c, sums, tid, total);
        if (before < need && need <= before + c) {
            sh_prefix = prefix | (b << shift);
            sh_need = need - before;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix_mask |= 255u << shift;
    }
    const uint thresh = sh_prefix, ties = sh_need;

    // Each thread takes a contiguous range of pools: count the kept ones (above the threshold, or equal to it with a
    // rank below `ties`), then write their cells at its offset.
    const uint per = (a.n_pool + TG - 1) / TG, p0 = min(a.n_pool, tid * per), p1 = min(a.n_pool, p0 + per);
    uint gt = 0, eq = 0;
    for (uint p = p0; p < p1; ++p) {
        const uint k = order_key(score[p]);
        gt += all || k > thresh;
        eq += !all && k == thresh;
    }
    const uint eq_before = tg_prefix(eq, sums, tid, total);
    const uint eq_kept = ties > eq_before ? min(eq, ties - eq_before) : 0;
    uint kept_total = 0;
    const uint kept_before = tg_prefix(gt + eq_kept, sums, tid, kept_total);
    uint at = kept_before * a.kpool, rank = eq_before;
    for (uint p = p0; p < p1; ++p) {
        const uint k = order_key(score[p]);
        bool keep = all || k > thresh;
        if (!all && k == thresh)
            keep = rank++ < ties;
        if (keep)
            for (uint c = 0; c < a.kpool; ++c)
                list[at++] = p * a.kpool + c;
    }
    const uint tail0 = a.n_pool * a.kpool;
    for (uint c = tail0 + tid; c < a.n_kv; c += TG)
        list[kept_total * a.kpool + (c - tail0)] = c;
    if (tid == 0)
        count[j] = kept_total * a.kpool + (a.n_kv - tail0);
}

// y[t * ldy + i] = x[t * n + i]; grid (n, rows).
kernel void copy_rows(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint2& a [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.x)
        y[uint64_t(gid.y) * a.y + gid.x] = x[uint64_t(gid.y) * a.x + gid.x];
}

kernel void copy_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = x[i];
}

struct AblateArgs {
    uint n_embd;
    uint k;      // directions, orthonormal
    float scale; // 1: the components are removed entirely
};

// Each row loses its components along k directions (x -= scale (x . v) v, one direction after another); one
// threadgroup of 256 a row.
kernel void ablate_rows(device float* x [[buffer(0)]], device const float* v [[buffer(1)]],
                        constant AblateArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float part[8];
    x += uint64_t(row) * a.n_embd;
    for (uint k = 0; k < a.k; ++k) {
        device const float* d = v + uint64_t(k) * a.n_embd;
        float acc = 0.0f;
        for (uint i = tid; i < a.n_embd; i += 256)
            acc += x[i] * d[i];
        acc = simd_sum(acc);
        if (lane == 0)
            part[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float c = 0.0f;
        for (uint s = 0; s < 8; ++s)
            c += part[s];
        c *= a.scale;
        for (uint i = tid; i < a.n_embd; i += 256)
            x[i] -= c * d[i];
        threadgroup_barrier(mem_flags::mem_threadgroup); // part is rewritten for the next direction
    }
}
