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
