// The fused hyper-connection kernels against the dispatches they replace, on random data: GLM-5.3's hc_step_norm
// against hc_step_a + hc_step_b + rms_norm (with and without the expand of the previous sublayer's output), and
// DeepSeek-V4.1's hc_lagged_norm + hc_mix against rms_norm + the F32 mixing GEMV + hc_mix + hc_collapse + rms_norm.
// Bit for bit, the same over repeated runs (the fused kernels hand data to a token's last threadgroup), counters left
// zero; both paths also against a CPU reference in double.
//   hc_test
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"
#include "mtl/gemv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using mako::mtl::Ref;

// Layouts must match glm.metal's.
struct NormArgs {
    uint32_t n, wn;
    float eps;
    uint32_t ystride;
};
struct HcArgs {
    uint32_t n_embd, iters;
    float eps;
    uint32_t tokens;
};
struct HcStepArgs {
    uint32_t n_embd, expand, parts, iters;
    float eps, rms_eps;
    uint32_t wn;
};
struct HcLaggedArgs {
    uint32_t n_embd;
    float rms_eps;
    uint32_t wn, row_groups, groups;
};

constexpr uint32_t kIters = 20, kParts = 32, kRowGroups = 6, kRuns = 10;
constexpr float kEps = 1e-6f, kRmsEps = 1e-6f;

// hc_mix in double: pre (4), post (4) and the Sinkhorn-normalized comb (16, [src][dst]) from 24 normalized mixes.
std::vector<double> mix_ref(const double* m, const float* scale, const float* base)
{
    std::vector<double> pc(24);
    const auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    for (int i = 0; i < 4; ++i) {
        pc[i] = sig(m[i] * scale[0] + base[i]) + kEps;
        pc[4 + i] = 2.0 * sig(m[4 + i] * scale[1] + base[4 + i]);
    }
    double c[16];
    for (int i = 0; i < 16; ++i)
        c[i] = m[8 + i] * scale[2] + base[8 + i];
    for (int s = 0; s < 4; ++s) {
        const double mx = std::max({c[s * 4], c[s * 4 + 1], c[s * 4 + 2], c[s * 4 + 3]});
        double sum = 0;
        for (int d = 0; d < 4; ++d)
            sum += c[s * 4 + d] = std::exp(c[s * 4 + d] - mx);
        for (int d = 0; d < 4; ++d)
            c[s * 4 + d] = c[s * 4 + d] / sum + kEps;
    }
    const auto cols = [&] {
        for (int d = 0; d < 4; ++d) {
            double sum = kEps;
            for (int s = 0; s < 4; ++s)
                sum += c[s * 4 + d];
            for (int s = 0; s < 4; ++s)
                c[s * 4 + d] /= sum;
        }
    };
    const auto rows = [&] {
        for (int s = 0; s < 4; ++s) {
            double sum = kEps;
            for (int d = 0; d < 4; ++d)
                sum += c[s * 4 + d];
            for (int d = 0; d < 4; ++d)
                c[s * 4 + d] /= sum;
        }
    };
    cols();
    for (uint32_t i = 1; i < kIters; ++i) {
        rows();
        cols();
    }
    std::copy_n(c, 16, pc.begin() + 8);
    return pc;
}

// x / rms(x) * w in double.
std::vector<double> norm_ref(const std::vector<double>& x, const float* w)
{
    double ss = 0;
    for (double v : x)
        ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / double(x.size()) + kRmsEps);
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] * inv * (w ? w[i] : 1.0);
    return y;
}

struct Fixture {
    mako::mtl::Context& ctx;
    Ref<MTL::Library> lib;
    std::mt19937 rng;
    std::normal_distribution<float> nd;

    Ref<MTL::Buffer> buffer(uint64_t floats)
    {
        auto b = Ref<MTL::Buffer>(ctx.device()->newBuffer(std::max<uint64_t>(floats, 1) * 4,
                                                          MTL::ResourceStorageModeShared));
        std::memset(b->contents(), 0, b->length());
        return b;
    }
    Ref<MTL::Buffer> random(uint64_t floats, float sd, float mean = 0.0f)
    {
        auto b = buffer(floats);
        auto* p = static_cast<float*>(b->contents());
        for (uint64_t i = 0; i < floats; ++i)
            p[i] = mean + sd * nd(rng);
        return b;
    }
    Ref<MTL::Buffer> copy(MTL::Buffer* src)
    {
        auto b = buffer(src->length() / 4);
        std::memcpy(b->contents(), src->contents(), src->length());
        return b;
    }
    Ref<MTL::ComputePipelineState> pso(const char* name) { return ctx.pipeline(lib.get(), name); }

    template <typename F>
    void run(F&& encode)
    {
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        encode(enc);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }
};

const float* f(const Ref<MTL::Buffer>& b) { return static_cast<const float*>(b->contents()); }

uint64_t differing(const Ref<MTL::Buffer>& a, const Ref<MTL::Buffer>& b, uint64_t floats)
{
    uint64_t d = 0;
    for (uint64_t i = 0; i < floats; ++i)
        d += std::memcmp(f(a) + i, f(b) + i, 4) != 0;
    return d;
}

// The largest difference to the reference, relative to the reference's largest magnitude.
double rel_err(const float* got, const std::vector<double>& want)
{
    double err = 0, mag = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        err = std::max(err, std::abs(double(got[i]) - want[i]));
        mag = std::max(mag, std::abs(want[i]));
    }
    return mag > 0 ? err / mag : err;
}

bool zero(const Ref<MTL::Buffer>& counters)
{
    const auto* c = static_cast<const uint32_t*>(counters->contents());
    return std::all_of(c, c + counters->length() / 4, [](uint32_t v) { return v == 0; });
}

constexpr double kTol = 1e-4;

// hc_step_norm against hc_step_a + hc_step_b + rms_norm.
int step_case(Fixture& fx, uint32_t E, uint32_t T, bool expand)
{
    const uint32_t hd = 4 * E;
    auto x_post = fx.random(uint64_t(T) * E, 1.0f), R = fx.random(uint64_t(T) * hd, 1.0f);
    auto fn = fx.random(24 * uint64_t(hd), 0.02f), scale = fx.random(3, 0.3f, 1.0f), base = fx.random(24, 0.5f);
    auto w = fx.random(E, 0.1f, 1.0f);
    // The previous step's pc: positive post and comb.
    auto pc_in = fx.buffer(uint64_t(T) * 24);
    for (uint32_t i = 0; i < T * 24; ++i)
        static_cast<float*>(pc_in->contents())[i] = 0.25f + 0.5f * std::abs(fx.nd(fx.rng));
    const HcStepArgs a{E, expand ? 1u : 0u, kParts, kIters, kEps, kRmsEps, E};
    auto counters = fx.buffer(8);
    auto step_norm = fx.pso("hc_step_norm"), step_a = fx.pso("hc_step_a"), step_b = fx.pso("hc_step_b"),
         rms = fx.pso("rms_norm");

    struct Out {
        Ref<MTL::Buffer> out, pc, x, xn;
    };
    auto fused = [&] {
        Out o{fx.buffer(uint64_t(T) * hd), fx.copy(pc_in.get()), fx.buffer(uint64_t(T) * E), fx.buffer(uint64_t(T) * E)};
        auto part = fx.buffer(uint64_t(T) * kParts * 25);
        fx.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(step_norm.get());
            enc->setBuffer(expand ? x_post.get() : R.get(), 0, 0);
            enc->setBuffer(R.get(), 0, 1);
            enc->setBuffer(o.pc.get(), 0, 2);
            enc->setBuffer(expand ? o.out.get() : R.get(), 0, 3);
            enc->setBuffer(fn.get(), 0, 4);
            enc->setBuffer(part.get(), 0, 5);
            enc->setBytes(&a, sizeof(a), 6);
            enc->setBuffer(scale.get(), 0, 7);
            enc->setBuffer(base.get(), 0, 8);
            enc->setBuffer(o.x.get(), 0, 9);
            enc->setBuffer(w.get(), 0, 10);
            enc->setBuffer(o.xn.get(), 0, 11);
            enc->setBuffer(counters.get(), 0, 12);
            enc->dispatchThreadgroups(MTL::Size(kParts, T, 1), MTL::Size(256, 1, 1));
        });
        return o;
    };
    Out u{fx.buffer(uint64_t(T) * hd), fx.copy(pc_in.get()), fx.buffer(uint64_t(T) * E), fx.buffer(uint64_t(T) * E)};
    {
        auto part = fx.buffer(uint64_t(T) * kParts * 25);
        const NormArgs na{E, E, kRmsEps, 0};
        fx.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(step_a.get());
            enc->setBuffer(expand ? x_post.get() : R.get(), 0, 0);
            enc->setBuffer(R.get(), 0, 1);
            enc->setBuffer(u.pc.get(), 0, 2);
            enc->setBuffer(expand ? u.out.get() : part.get(), 0, 3);
            enc->setBuffer(fn.get(), 0, 4);
            enc->setBuffer(part.get(), 0, 5);
            enc->setBytes(&a, sizeof(a), 6);
            enc->dispatchThreadgroups(MTL::Size(kParts, T, 1), MTL::Size(256, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(step_b.get());
            enc->setBuffer(part.get(), 0, 0);
            enc->setBuffer(expand ? u.out.get() : R.get(), 0, 1);
            enc->setBuffer(scale.get(), 0, 2);
            enc->setBuffer(base.get(), 0, 3);
            enc->setBuffer(u.pc.get(), 0, 4);
            enc->setBuffer(u.x.get(), 0, 5);
            enc->setBytes(&a, sizeof(a), 6);
            enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(1024, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(rms.get());
            enc->setBuffer(u.x.get(), 0, 0);
            enc->setBuffer(w.get(), 0, 1);
            enc->setBuffer(u.xn.get(), 0, 2);
            enc->setBytes(&na, sizeof(na), 3);
            enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
        });
    }

    uint64_t d_out = 0, d_pc = 0, d_x = 0, d_xn = 0, unstable = 0;
    const Out first = fused();
    for (uint32_t run = 1; run < kRuns; ++run) {
        const Out o = fused();
        unstable += differing(o.pc, first.pc, T * 24) + differing(o.x, first.x, uint64_t(T) * E) +
                    differing(o.xn, first.xn, uint64_t(T) * E);
    }
    if (expand)
        d_out = differing(first.out, u.out, uint64_t(T) * hd);
    d_pc = differing(first.pc, u.pc, T * 24);
    d_x = differing(first.x, u.x, uint64_t(T) * E);
    d_xn = differing(first.xn, u.xn, uint64_t(T) * E);

    // CPU reference.
    double worst = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const float* pci = f(pc_in) + t * 24;
        std::vector<double> rn(hd);
        for (uint32_t j = 0; j < hd; ++j) {
            const float* Rt = f(R) + uint64_t(t) * hd;
            if (expand) {
                const uint32_t h = j / E, i = j % E;
                double r = double(f(x_post)[uint64_t(t) * E + i]) * pci[4 + h];
                for (uint32_t s = 0; s < 4; ++s)
                    r += double(pci[8 + s * 4 + h]) * Rt[s * E + i];
                rn[j] = r;
            } else {
                rn[j] = Rt[j];
            }
        }
        double ss = 0;
        for (double v : rn)
            ss += v * v;
        const double inv = 1.0 / std::sqrt(ss / hd + kRmsEps);
        double m[24];
        for (uint32_t r = 0; r < 24; ++r) {
            double s = 0;
            for (uint32_t j = 0; j < hd; ++j)
                s += double(f(fn)[uint64_t(r) * hd + j]) * rn[j];
            m[r] = s * inv;
        }
        const auto pc = mix_ref(m, f(scale), f(base));
        std::vector<double> x(E);
        for (uint32_t i = 0; i < E; ++i)
            x[i] = pc[0] * rn[i] + pc[1] * rn[E + i] + pc[2] * rn[2 * E + i] + pc[3] * rn[3 * E + i];
        const auto xn = norm_ref(x, f(w));
        for (const Out* o : {&first, static_cast<const Out*>(&u)}) {
            if (expand)
                worst = std::max(worst, rel_err(f(o->out) + uint64_t(t) * hd, rn));
            worst = std::max({worst, rel_err(f(o->pc) + t * 24, pc), rel_err(f(o->x) + uint64_t(t) * E, x),
                              rel_err(f(o->xn) + uint64_t(t) * E, xn)});
        }
    }
    const bool ok = !d_out && !d_pc && !d_x && !d_xn && !unstable && zero(counters) && worst < kTol;
    std::println("{} hc_step_norm  E={} T={} expand={}  vs unfused out/pc/x/xn differing: {}/{}/{}/{}  across runs: {}  "
                 "counters zero: {}  vs CPU: {:.2e}",
                 ok ? "ok  " : "FAIL", E, T, int(expand), d_out, d_pc, d_x, d_xn, unstable, zero(counters), worst);
    return ok ? 0 : 1;
}

// hc_lagged_norm + hc_mix against rms_norm + the F32 mixing GEMV + hc_mix + hc_collapse + rms_norm.
int lagged_case(Fixture& fx, mako::mtl::Gemv& gemv, uint32_t E, uint32_t T)
{
    const uint32_t hd = 4 * E;
    auto R = fx.random(uint64_t(T) * hd, 1.0f), fn = fx.random(24 * uint64_t(hd), 0.02f);
    auto scale = fx.random(3, 0.3f, 1.0f), base = fx.random(24, 0.5f), w = fx.random(E, 0.1f, 1.0f);
    auto pc_prev = fx.buffer(uint64_t(T) * 24);
    for (uint32_t i = 0; i < T * 24; ++i)
        static_cast<float*>(pc_prev->contents())[i] = 0.25f + 0.5f * std::abs(fx.nd(fx.rng));
    auto counters = fx.buffer(8);
    const HcArgs ha{E, kIters, kEps, T};
    auto lagged = fx.pso("hc_lagged_norm"), mix = fx.pso("hc_mix"), collapse = fx.pso("hc_collapse"),
         rms = fx.pso("rms_norm");

    struct Out {
        Ref<MTL::Buffer> mixes, pc, x, xn;
    };
    auto fused = [&] {
        Out o{fx.buffer(8 * 24), fx.buffer(uint64_t(T) * 24), fx.buffer(uint64_t(T) * E), fx.buffer(uint64_t(T) * E)};
        const HcLaggedArgs a{E, kRmsEps, E, kRowGroups, kRowGroups + (E + 255) / 256};
        fx.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(lagged.get());
            enc->setBuffer(R.get(), 0, 0);
            enc->setBuffer(fn.get(), 0, 1);
            enc->setBuffer(pc_prev.get(), 0, 2);
            enc->setBuffer(o.mixes.get(), 0, 3);
            enc->setBuffer(o.x.get(), 0, 4);
            enc->setBuffer(w.get(), 0, 5);
            enc->setBuffer(o.xn.get(), 0, 6);
            enc->setBuffer(counters.get(), 0, 7);
            enc->setBytes(&a, sizeof(a), 8);
            enc->dispatchThreadgroups(MTL::Size(a.groups, T, 1), MTL::Size(256, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(mix.get());
            enc->setBuffer(o.mixes.get(), 0, 0);
            enc->setBuffer(scale.get(), 0, 1);
            enc->setBuffer(base.get(), 0, 2);
            enc->setBuffer(o.pc.get(), 0, 3);
            enc->setBytes(&ha, sizeof(ha), 4);
            enc->dispatchThreads(MTL::Size(T, 1, 1), MTL::Size(T, 1, 1));
        });
        return o;
    };
    Out u{fx.buffer(uint64_t(T) * 24), fx.buffer(uint64_t(T) * 24), fx.buffer(uint64_t(T) * E),
          fx.buffer(uint64_t(T) * E)};
    {
        auto flat = fx.buffer(uint64_t(T) * hd);
        const NormArgs flat_a{hd, 0, kRmsEps, 0}, x_a{E, E, kRmsEps, 0};
        fx.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(rms.get());
            enc->setBuffer(R.get(), 0, 0);
            enc->setBuffer(R.get(), 0, 1);
            enc->setBuffer(flat.get(), 0, 2);
            enc->setBytes(&flat_a, sizeof(flat_a), 3);
            enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            gemv.encode(enc, mako::mtl::Gemv::Args{mako::gguf::Type::F32, hd, 24, T, 1, false}, fn.get(), 0,
                        flat.get(), 0, u.mixes.get(), 0);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(mix.get());
            enc->setBuffer(u.mixes.get(), 0, 0);
            enc->setBuffer(scale.get(), 0, 1);
            enc->setBuffer(base.get(), 0, 2);
            enc->setBuffer(u.pc.get(), 0, 3);
            enc->setBytes(&ha, sizeof(ha), 4);
            enc->dispatchThreads(MTL::Size(T, 1, 1), MTL::Size(std::min<uint32_t>(T, 64), 1, 1));
            enc->setComputePipelineState(collapse.get());
            enc->setBuffer(R.get(), 0, 0);
            enc->setBuffer(pc_prev.get(), 0, 1);
            enc->setBuffer(u.x.get(), 0, 2);
            enc->setBytes(&ha, sizeof(ha), 3);
            enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(rms.get());
            enc->setBuffer(u.x.get(), 0, 0);
            enc->setBuffer(w.get(), 0, 1);
            enc->setBuffer(u.xn.get(), 0, 2);
            enc->setBytes(&x_a, sizeof(x_a), 3);
            enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
        });
    }

    uint64_t unstable = 0;
    const Out first = fused();
    for (uint32_t run = 1; run < kRuns; ++run) {
        const Out o = fused();
        unstable += differing(o.mixes, first.mixes, T * 24) + differing(o.x, first.x, uint64_t(T) * E) +
                    differing(o.xn, first.xn, uint64_t(T) * E);
    }
    const uint64_t d_mix = differing(first.mixes, u.mixes, T * 24), d_pc = differing(first.pc, u.pc, T * 24),
                   d_x = differing(first.x, u.x, uint64_t(T) * E), d_xn = differing(first.xn, u.xn, uint64_t(T) * E);

    double worst = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const float* Rt = f(R) + uint64_t(t) * hd;
        double ss = 0;
        for (uint32_t j = 0; j < hd; ++j)
            ss += double(Rt[j]) * Rt[j];
        const double inv = 1.0 / std::sqrt(ss / hd + kRmsEps);
        double m[24];
        std::vector<double> mv(24);
        for (uint32_t r = 0; r < 24; ++r) {
            double s = 0;
            for (uint32_t j = 0; j < hd; ++j)
                s += double(f(fn)[uint64_t(r) * hd + j]) * Rt[j];
            mv[r] = m[r] = s * inv;
        }
        const auto pc = mix_ref(m, f(scale), f(base));
        const float* pre = f(pc_prev) + t * 24;
        std::vector<double> x(E);
        for (uint32_t i = 0; i < E; ++i)
            x[i] = double(pre[0]) * Rt[i] + double(pre[1]) * Rt[E + i] + double(pre[2]) * Rt[2 * E + i] +
                   double(pre[3]) * Rt[3 * E + i];
        const auto xn = norm_ref(x, f(w));
        for (const Out* o : {&first, static_cast<const Out*>(&u)})
            worst = std::max({worst, rel_err(f(o->mixes) + t * 24, mv), rel_err(f(o->pc) + t * 24, pc),
                              rel_err(f(o->x) + uint64_t(t) * E, x), rel_err(f(o->xn) + uint64_t(t) * E, xn)});
    }
    const bool ok = !d_mix && !d_pc && !d_x && !d_xn && !unstable && zero(counters) && worst < kTol;
    std::println("{} hc_lagged_norm E={} T={}  vs unfused mixes/pc/x/xn differing: {}/{}/{}/{}  across runs: {}  "
                 "counters zero: {}  vs CPU: {:.2e}",
                 ok ? "ok  " : "FAIL", E, T, d_mix, d_pc, d_x, d_xn, unstable, zero(counters), worst);
    return ok ? 0 : 1;
}

} // namespace

int main()
{
    mako::mtl::Context ctx;
    Fixture fx{ctx, ctx.load(mako::mtl::kernels_metallib()), std::mt19937(29), {}};
    mako::mtl::Gemv gemv(ctx);
    int failures = 0;
    for (const uint32_t T : {1u, 3u, 8u})
        for (const bool expand : {false, true})
            failures += step_case(fx, 4096, T, expand); // GLM-5.3's n_embd
    for (const uint32_t T : {1u, 3u, 8u})
        failures += lagged_case(fx, gemv, 5120, T); // DeepSeek-V4.1's
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
