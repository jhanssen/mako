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
