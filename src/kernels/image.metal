// Qwen-Image's diffusion transformer: modulated layer norms and gated residuals. Each token takes one of two
// modulation rows: the sampled timestep's (the image) or t = 0's (the text, rows before `split`).
#include <metal_stdlib>
using namespace metal;

constant uint TG = 256;

static float tg_sum(float v, threadgroup float* scratch, uint tid)
{
    v = simd_sum(v);
    if (tid % 32 == 0)
        scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = 0; i < TG / 32; ++i)
        s += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

struct ModArgs {
    uint n;          // row width
    float eps;
    uint split;      // rows before it take modulation row 1 (t = 0), the rest row 0
    uint mod_stride; // between the two modulation rows
    uint mod_off;    // of this scale (or gate) within a row
};

// y = layer_norm(x) * (1 + scale), no learned weights; one threadgroup per row.
kernel void ln_modulate(device const float* x [[buffer(0)]], device const float* mod [[buffer(1)]],
                        device float* y [[buffer(2)]], constant ModArgs& a [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.n;
    device const float* sc = mod + (row < a.split ? a.mod_stride : 0) + a.mod_off;
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
        yr[i] = (xr[i] - mean) * inv * (1.0f + sc[i]);
}

// x += tanh(gate) * y. Grid (n, rows).
kernel void gate_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                     device const float* mod [[buffer(2)]], constant ModArgs& a [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, row = gid.y;
    if (i >= a.n)
        return;
    const float g = mod[(row < a.split ? a.mod_stride : 0) + a.mod_off + i];
    x[uint64_t(row) * a.n + i] += precise::tanh(g) * y[uint64_t(row) * a.n + i];
}

// The VAE decoder's pieces; images are channels-last, [height][width][channels] (a pixel a row).

struct Im2colArgs {
    uint h, w, c;   // input image
    uint p0, rows;  // output pixels [p0, p0 + rows)
    uint kpad;      // columns, 9 * c padded to a multiple of 32
};

// out[r][ch * 9 + ky * 3 + kx] = x[y + ky - 1][x + kx - 1][ch] for output pixel p0 + r = (y, x), zero outside the
// image and past 9 * c. Grid (kpad, rows).
kernel void im2col3x3(device const float* x [[buffer(0)]], device half* out [[buffer(1)]],
                      constant Im2colArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x, r = gid.y;
    if (col >= a.kpad || r >= a.rows)
        return;
    float v = 0.0f;
    if (col < 9 * a.c) {
        const uint ch = col / 9, kk = col % 9, p = a.p0 + r;
        const int y = int(p / a.w) + int(kk / 3) - 1, xx = int(p % a.w) + int(kk % 3) - 1;
        if (y >= 0 && y < int(a.h) && xx >= 0 && xx < int(a.w))
            v = x[(uint64_t(y) * a.w + uint(xx)) * a.c + ch];
    }
    out[uint64_t(r) * a.kpad + col] = half(v);
}

// y[r][i] += b[i]. Grid (n, rows).
kernel void bias_rows(device float* y [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n)
        y[uint64_t(gid.y) * n + gid.x] += b[gid.x];
}

struct UpArgs {
    uint h, w, c;    // input
    uint c_out;      // dup_up: output channels
    uint ft;         // dup_up: temporal factor (1 or 2; the last temporal copy is kept)
    uint repeats;    // dup_up: c_out * ft * 4 / c
};

// Nearest 2x: y[2y + i][2x + j][ch] = x[y][x][ch]. Grid (c, 2w, 2h).
kernel void upsample2x(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       constant UpArgs& a [[buffer(2)]], uint3 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x, ox = gid.y, oy = gid.z;
    if (ch >= a.c || ox >= 2 * a.w || oy >= 2 * a.h)
        return;
    y[(uint64_t(oy) * 2 * a.w + ox) * a.c + ch] = x[(uint64_t(oy / 2) * a.w + ox / 2) * a.c + ch];
}

// The duplicating shortcut of an up block, added into y: input channels repeated, the copies laid out as
// (out channel, time, i, j) and the 2x2 (i, j) spread over pixels; only the last time copy is kept.
// Grid (c_out, 2w, 2h).
kernel void dup_up_add(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       constant UpArgs& a [[buffer(2)]], uint3 gid [[thread_position_in_grid]])
{
    const uint o = gid.x, ox = gid.y, oy = gid.z;
    if (o >= a.c_out || ox >= 2 * a.w || oy >= 2 * a.h)
        return;
    const uint i = oy % 2, j = ox % 2, t = a.ft - 1;
    const uint cc = ((o * a.ft + t) * 2 + i) * 2 + j;
    y[(uint64_t(oy) * 2 * a.w + ox) * a.c_out + o] += x[(uint64_t(oy / 2) * a.w + ox / 2) * a.c + cc / a.repeats];
}

// Row softmax of scale * x, in place; one threadgroup per row.
kernel void softmax_rows(device float* x [[buffer(0)]], constant float2& a [[buffer(1)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    const uint n = uint(a.x);
    const float scale = a.y;
    device float* r = x + uint64_t(row) * n;
    float m = -INFINITY;
    for (uint i = tid; i < n; i += TG)
        m = max(m, r[i] * scale);
    m = simd_max(m);
    if (tid % 32 == 0)
        scratch[tid / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = scratch[0];
    for (uint i = 1; i < TG / 32; ++i)
        m = max(m, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = tid; i < n; i += TG) {
        const float e = exp(r[i] * scale - m);
        r[i] = e;
        s += e;
    }
    const float inv = 1.0f / tg_sum(s, scratch, tid);
    for (uint i = tid; i < n; i += TG)
        r[i] *= inv;
}

// y[c][r] = x[r][col0 + c] for rows x cols of x (rows ldx apart). Grid (cols, rows).
struct TransposeArgs {
    uint rows, cols, ldx, col0;
};
kernel void transpose_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant TransposeArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.cols && gid.y < a.rows)
        y[uint64_t(gid.x) * a.rows + gid.y] = x[uint64_t(gid.y) * a.ldx + a.col0 + gid.x];
}

// y[r][c] = x[r][col0 + c] for rows x cols of x (rows ldx apart). Grid (cols, rows).
kernel void copy_cols(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant TransposeArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.cols && gid.y < a.rows)
        y[uint64_t(gid.y) * a.cols + gid.x] = x[uint64_t(gid.y) * a.ldx + a.col0 + gid.x];
}

struct LoraArgs {
    uint k, rows, rank; // the matrix's row length, the rows adapted, the adapter's rank
    float scale;
};

// W[r][c] += scale * sum_j B[r][j] * A[j][c] over BF16 rows of W (from the adapted ones on), in float, rounded back.
// Grid (k, rows).
kernel void lora_merge(device bfloat* w [[buffer(0)]], device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]], constant LoraArgs& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, r = gid.y;
    if (c >= p.k || r >= p.rows)
        return;
    float d = 0.0f;
    for (uint j = 0; j < p.rank; ++j)
        d += b[uint64_t(r) * p.rank + j] * a[uint64_t(j) * p.k + c];
    device bfloat* x = w + uint64_t(r) * p.k + c;
    *x = bfloat(float(*x) + p.scale * d);
}

// Krea 2's transformer: RMS norms with per-step scales and shifts, gated residuals and attention outputs, and the text
// fusion's attention across each token's tapped layers.

struct KreaModArgs {
    uint n;      // model width
    uint blocks;
};

// Each block's modulation from the step's shared vector tm [6][n] plus the block's table [6][n], with the block's two
// norm scales (w [blocks][2][n], already + 1) folded in: out [blocks][6][n] = (w1 (1 + prescale), preshift, pregate,
// w2 (1 + postscale), postshift, postgate). Grid (n, blocks).
kernel void krea_mod(device const float* tm [[buffer(0)]], device const float* table [[buffer(1)]],
                     device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                     constant KreaModArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, b = gid.y;
    if (i >= a.n || b >= a.blocks)
        return;
    device const float* t = table + uint64_t(b) * 6 * a.n;
    device float* o = out + uint64_t(b) * 6 * a.n;
    float m[6];
    for (uint k = 0; k < 6; ++k)
        m[k] = tm[k * a.n + i] + t[k * a.n + i];
    o[i] = w[(uint64_t(b) * 2) * a.n + i] * (1.0f + m[0]);
    o[a.n + i] = m[1];
    o[2 * a.n + i] = m[2];
    o[3 * a.n + i] = w[(uint64_t(b) * 2 + 1) * a.n + i] * (1.0f + m[3]);
    o[4 * a.n + i] = m[4];
    o[5 * a.n + i] = m[5];
}

// The final layer's: out [2][n] = (w (1 + temb + table[0]), temb + table[1]). Grid n.
kernel void krea_final_mod(device const float* temb [[buffer(0)]], device const float* table [[buffer(1)]],
                           device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                           constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]])
{
    if (i >= n)
        return;
    out[i] = w[i] * (1.0f + temb[i] + table[i]);
    out[n + i] = temb[i] + table[n + i];
}

struct RmsAffineArgs {
    uint n;
    float eps;
};

// y = x / rms(x) * s + b over each row (s, b: [n]); one threadgroup per row.
kernel void rms_affine(device const float* x [[buffer(0)]], device const float* s [[buffer(1)]],
                       device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                       constant RmsAffineArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.n;
    float v = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        v += xr[i] * xr[i];
    const float inv = rsqrt(tg_sum(v, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += TG)
        yr[i] = xr[i] * inv * s[i] + b[i];
}

// x += g * y over rows of n (g: [n]). Grid (n, rows).
kernel void scale_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                      device const float* g [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n)
        x[uint64_t(gid.y) * n + gid.x] += g[gid.x] * y[uint64_t(gid.y) * n + gid.x];
}

struct GateArgs {
    uint n, ld, col0;
};

// x[r][i] *= sigmoid(src[r][col0 + i]) (src rows ld apart). Grid (n, rows).
kernel void sigmoid_mul(device float* x [[buffer(0)]], device const float* src [[buffer(1)]],
                        constant GateArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.n)
        return;
    const float g = src[uint64_t(gid.y) * a.ld + a.col0 + gid.x];
    x[uint64_t(gid.y) * a.n + gid.x] *= 1.0f / (1.0f + exp(-g));
}

struct GroupAttnArgs {
    uint rows;  // all rows, `group` per token
    uint group;
    uint heads; // of 128, as many key / value heads
    uint ld;    // of the q | k | v rows
    float eps;
};

// Attention within each run of `group` rows (no mask, no rotation), q and k RMS-normed per head (times wq, wk) first.
// qkv rows [q | k | v] of heads * 128; out [rows][heads * 128]. One thread per (row, head).
kernel void attn_groups(device const float* qkv [[buffer(0)]], device const float* wq [[buffer(1)]],
                        device const float* wk [[buffer(2)]], device float* out [[buffer(3)]],
                        constant GroupAttnArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint r = gid.x, h = gid.y;
    if (r >= a.rows || h >= a.heads)
        return;
    const uint w = a.heads * 128, r0 = r / a.group * a.group;
    device const float* q = qkv + uint64_t(r) * a.ld + h * 128;
    float qs = 0.0f;
    for (uint d = 0; d < 128; ++d)
        qs += q[d] * q[d];
    const float qinv = rsqrt(qs / 128.0f + a.eps) * rsqrt(128.0f);
    float score[32];
    float m = -INFINITY;
    for (uint j = 0; j < a.group; ++j) {
        device const float* k = qkv + uint64_t(r0 + j) * a.ld + w + h * 128;
        float ks = 0.0f, dot = 0.0f;
        for (uint d = 0; d < 128; ++d) {
            ks += k[d] * k[d];
            dot += q[d] * wq[d] * k[d] * wk[d];
        }
        score[j] = dot * qinv * rsqrt(ks / 128.0f + a.eps);
        m = max(m, score[j]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < a.group; ++j) {
        score[j] = exp(score[j] - m);
        sum += score[j];
    }
    device float* o = out + uint64_t(r) * w + h * 128;
    for (uint d = 0; d < 128; ++d) {
        float acc = 0.0f;
        for (uint j = 0; j < a.group; ++j)
            acc += score[j] * qkv[uint64_t(r0 + j) * a.ld + 2 * w + h * 128 + d];
        o[d] = acc / sum;
    }
}

struct LayerMixArgs {
    uint n, layers;
    float p[16];
};

// out[s][i] = sum_l p[l] x[s][l][i]. Grid (n, tokens).
kernel void layer_mix(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                      constant LayerMixArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.n)
        return;
    float acc = 0.0f;
    for (uint l = 0; l < a.layers; ++l)
        acc += a.p[l] * x[(uint64_t(gid.y) * a.layers + l) * a.n + gid.x];
    out[uint64_t(gid.y) * a.n + gid.x] = acc;
}

// Qwen3.5's text layers: the gated delta rule's input and output, and the gated attention's heads.

struct GdnConvArgs {
    uint channels;
    uint taps; // the last one multiplies the current token (at most 8)
    uint tokens;
};

// Causal depthwise convolution over each channel's tokens (zeros before the first), then SiLU: x, y [tokens][channels],
// w [channels][taps]. One thread per channel.
kernel void gdn_conv(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant GdnConvArgs& a [[buffer(3)]],
                     uint c [[thread_position_in_grid]])
{
    if (c >= a.channels)
        return;
    device const float* wc = w + uint64_t(c) * a.taps;
    float hist[8] = {};
    for (uint t = 0; t < a.tokens; ++t) {
        const float xc = x[uint64_t(t) * a.channels + c];
        for (uint j = 0; j + 1 < a.taps; ++j)
            hist[j] = hist[j + 1];
        hist[a.taps - 1] = xc;
        float acc = 0.0f;
        for (uint j = 0; j < a.taps; ++j)
            acc += wc[j] * hist[j];
        y[uint64_t(t) * a.channels + c] = acc / (1.0f + exp(-acc));
    }
}

struct GdnPrepArgs {
    uint k_heads, v_heads; // of 128: each key head serves v_heads / k_heads value heads
    uint conv_width;       // a conv row: q (k_heads), k (k_heads), v (v_heads)
    float eps;             // of the l2 norm: x / sqrt(sum x^2 + eps)
};

// For each (token, value head) from the convolved q | k | v and the b | a projections: q and k l2-normed and repeated
// to the value heads, v, the decay g = -exp(A_log) softplus(a + dt_bias) on every channel, beta = sigmoid(b), as the
// delta-rule scan reads them ([tokens][v_heads * 128]; beta [tokens][v_heads]). Threadgroups (v_heads, tokens) of 128.
kernel void gdn_prep(device const float* conv [[buffer(0)]], device const float* ba [[buffer(1)]],
                     device const float* a_log [[buffer(2)]], device const float* dt_bias [[buffer(3)]],
                     device float* q [[buffer(4)]], device float* k [[buffer(5)]], device float* v [[buffer(6)]],
                     device float* g [[buffer(7)]], device float* beta [[buffer(8)]],
                     constant GdnPrepArgs& a [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint i [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[2][4];
    const uint h = tg.x, t = tg.y, kh = h / (a.v_heads / a.k_heads), kw = a.k_heads * 128, vw = a.v_heads * 128;
    device const float* row = conv + uint64_t(t) * a.conv_width;
    const float qv = row[kh * 128 + i], kv = row[kw + kh * 128 + i];
    const float sq = simd_sum(qv * qv), sk = simd_sum(kv * kv);
    if (lane == 0) {
        part[0][sg] = sq;
        part[1][sg] = sk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float nq = part[0][0] + part[0][1] + part[0][2] + part[0][3];
    const float nk = part[1][0] + part[1][1] + part[1][2] + part[1][3];
    const uint64_t o = uint64_t(t) * vw + h * 128 + i;
    q[o] = qv * rsqrt(nq + a.eps);
    k[o] = kv * rsqrt(nk + a.eps);
    v[o] = row[2 * kw + h * 128 + i];
    device const float* bar = ba + uint64_t(t) * 2 * a.v_heads;
    const float x = bar[a.v_heads + h] + dt_bias[h];
    const float sp = x > 20.0f ? x : log(1.0f + exp(x));
    g[o] = -exp(a_log[h]) * sp;
    if (i == 0)
        beta[uint64_t(t) * a.v_heads + h] = 1.0f / (1.0f + exp(-bar[h]));
}

struct GatedRmsArgs {
    uint heads; // of 128
    float eps;
};

// x = rms_norm(x) * w * silu(z) over each 128-wide head of each row. Threadgroups (heads, rows) of 128.
kernel void gated_rms(device float* x [[buffer(0)]], device const float* z [[buffer(1)]],
                      device const float* w [[buffer(2)]], constant GatedRmsArgs& a [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[4];
    const uint64_t o = uint64_t(tg.y) * a.heads * 128 + tg.x * 128 + i;
    const float v = x[o], s = simd_sum(v * v);
    if (lane == 0)
        part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt((part[0] + part[1] + part[2] + part[3]) / 128.0f + a.eps);
    const float zz = z[o];
    x[o] = v * inv * w[i] * (zz / (1.0f + exp(-zz)));
}

struct HeadsPartialArgs {
    uint ld, col0; // x rows ld apart; head h at col0 + h * stride
    uint stride;
    uint dim;      // per head (at most 256)
    uint rot;      // the first rot values rotate, i with i + rot / 2 (0: none)
    uint n_pad;    // rows of a head in y
    float eps;     // 0: no norm
};

// y[h][t][d] = half of head h of row t, RMS-normed (times w) when eps is set, its first rot values rotated by
// cs[t][i] = (cos, sin). Threadgroups (rows, heads) of dim threads.
kernel void heads_partial(device const float* x [[buffer(0)]], device const float2* cs [[buffer(1)]],
                          device half* y [[buffer(2)]], device const float* w [[buffer(3)]],
                          constant HeadsPartialArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint i [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float vals[256];
    threadgroup float part[8];
    const uint t = tg.x, h = tg.y;
    float v = x[uint64_t(t) * a.ld + a.col0 + h * a.stride + i];
    if (a.eps > 0.0f) {
        const float s = simd_sum(v * v);
        if (lane == 0)
            part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < a.dim / 32; ++j)
            total += part[j];
        v *= rsqrt(total / float(a.dim) + a.eps) * w[i];
    }
    vals[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < a.rot) {
        const uint half_rot = a.rot / 2, p = i % half_rot;
        const float2 c = cs[uint64_t(t) * half_rot + p];
        v = i < half_rot ? vals[i] * c.x - vals[i + half_rot] * c.y : vals[i] * c.x + vals[i - half_rot] * c.y;
    }
    y[(uint64_t(h) * a.n_pad + t) * a.dim + i] = half(v);
}

struct GateHeadsArgs {
    uint heads, dim;
    uint ld, col0; // the gate of head h at src[t][col0 + h * stride]
    uint stride;
};

// x[t][h * dim + d] *= sigmoid(src[t][col0 + h * stride + d]). Grid (dim, heads, rows).
kernel void gate_heads(device float* x [[buffer(0)]], device const float* src [[buffer(1)]],
                       constant GateHeadsArgs& a [[buffer(2)]], uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, h = gid.y, t = gid.z;
    if (d >= a.dim || h >= a.heads)
        return;
    const float gv = src[uint64_t(t) * a.ld + a.col0 + h * a.stride + d];
    x[(uint64_t(t) * a.heads + h) * a.dim + d] *= 1.0f / (1.0f + exp(-gv));
}

// Group norm over channels-last images [rows][c]: per-chunk channel sums, then per-group statistics, then the norm.

struct GnArgs {
    uint rows, c, groups, chunk; // chunk: rows a threadgroup of gn_partial sums
    float eps;
    uint silu;
};

// part[k][ch] = (sum, sum of squares) of channel ch over rows [k * chunk, (k + 1) * chunk). Threadgroups (chunks) of c.
kernel void gn_partial(device const float* x [[buffer(0)]], device float2* part [[buffer(1)]],
                       constant GnArgs& a [[buffer(2)]], uint k [[threadgroup_position_in_grid]],
                       uint ch [[thread_index_in_threadgroup]])
{
    if (ch >= a.c)
        return;
    const uint r1 = min(a.rows, (k + 1) * a.chunk);
    float s = 0.0f, q = 0.0f;
    for (uint r = k * a.chunk; r < r1; ++r) {
        const float v = x[uint64_t(r) * a.c + ch];
        s += v;
        q += v * v;
    }
    part[uint64_t(k) * a.c + ch] = float2(s, q);
}

// stats[g] = (mean, 1 / sqrt(var + eps)) of group g from the chunks' sums. One thread per group.
kernel void gn_stats(device const float2* part [[buffer(0)]], device float2* stats [[buffer(1)]],
                     constant GnArgs& a [[buffer(2)]], uint g [[thread_position_in_grid]])
{
    if (g >= a.groups)
        return;
    const uint per = a.c / a.groups, chunks = (a.rows + a.chunk - 1) / a.chunk;
    float s = 0.0f, q = 0.0f;
    for (uint k = 0; k < chunks; ++k)
        for (uint ch = g * per; ch < (g + 1) * per; ++ch) {
            const float2 p = part[uint64_t(k) * a.c + ch];
            s += p.x;
            q += p.y;
        }
    const float n = float(a.rows) * float(per), mean = s / n;
    stats[g] = float2(mean, rsqrt(max(q / n - mean * mean, 0.0f) + a.eps));
}

// y = (x - mean) / sd * w + b per group, then SiLU when silu is set. Grid (c, rows).
kernel void gn_apply(device const float* x [[buffer(0)]], device const float2* stats [[buffer(1)]],
                     device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                     device float* y [[buffer(4)]], constant GnArgs& a [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    if (ch >= a.c || gid.y >= a.rows)
        return;
    const float2 st = stats[ch / (a.c / a.groups)];
    const uint64_t i = uint64_t(gid.y) * a.c + ch;
    float v = (x[i] - st.x) * st.y * w[ch] + b[ch];
    if (a.silu)
        v = v / (1.0f + exp(-v));
    y[i] = v;
}

// MOSS-Image: the blocks' modulation from the step's shared vector and the gated FFN activation.

struct MossModArgs {
    uint n, blocks;
};

// tm [6][n] = (shift, scale, gate) for attention, then for the FFN; with each block's two norm scales (w [blocks][2][n])
// folded in: out [blocks][6][n] = (w1 (1 + scale1), shift1, gate1, w2 (1 + scale2), shift2, gate2). Grid (n, blocks).
kernel void moss_mod(device const float* tm [[buffer(0)]], device const float* w [[buffer(1)]],
                     device float* out [[buffer(2)]], constant MossModArgs& a [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, b = gid.y;
    if (i >= a.n || b >= a.blocks)
        return;
    device float* o = out + uint64_t(b) * 6 * a.n;
    for (uint k = 0; k < 2; ++k) {
        o[3 * k * a.n + i] = w[(uint64_t(b) * 2 + k) * a.n + i] * (1.0f + tm[(3 * k + 1) * a.n + i]);
        o[(3 * k + 1) * a.n + i] = tm[3 * k * a.n + i];
        o[(3 * k + 2) * a.n + i] = tm[(3 * k + 2) * a.n + i];
    }
}

// The output head's: hm [2][n] = (shift, scale); out [2][n] = (w (1 + scale), shift). Grid n.
kernel void moss_head_mod(device const float* hm [[buffer(0)]], device const float* w [[buffer(1)]],
                          device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]])
{
    if (i >= n)
        return;
    out[i] = w[i] * (1.0f + hm[n + i]);
    out[n + i] = hm[i];
}

// m[t][i] = 4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25) with h = hg[t][i], g = hg[t][n + i]. Grid (n, rows).
kernel void situ_rows(device const float* hg [[buffer(0)]], device float* m [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= n)
        return;
    device const float* r = hg + uint64_t(gid.y) * 2 * n;
    const float h = r[gid.x], g = r[n + gid.x];
    m[uint64_t(gid.y) * n + gid.x] = 4.0f * precise::tanh(g / 4.0f) / (1.0f + exp(-g)) * (25.0f * precise::tanh(h / 25.0f));
}
