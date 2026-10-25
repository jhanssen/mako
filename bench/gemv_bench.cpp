// Decode GEMV bandwidth per weight type: one token through a k x n matrix, repeated in one command buffer over
// distinct copies (so the weights come from memory, not cache); median of 50 runs after a warm-up. Timing only.
//   gemv_bench [k] [n] [copies]
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"
#include "quant/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

// Q8_0 rows: per 32, an f16 scale and 32 int8.
void q8_0(const float* x, std::byte* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 32; ++b, x += 32, dst += 34) {
        float amax = 0;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(x[i]));
        const float d = amax / 127.0f;
        const _Float16 h = _Float16(d);
        std::memcpy(dst, &h, 2);
        for (int i = 0; i < 32; ++i)
            reinterpret_cast<int8_t*>(dst + 2)[i] = int8_t(d > 0 ? std::lrintf(x[i] / d) : 0);
    }
}

} // namespace

int main(int argc, char** argv)
{
    using mako::gguf::Type;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t k = argc > 1 ? std::stoul(argv[1]) : 5120;
    const uint32_t n = argc > 2 ? std::stoul(argv[2]) : 8192;
    const uint32_t copies = argc > 3 ? std::stoul(argv[3]) : 24;
    mako::mtl::Context ctx;
    mako::mtl::Gemv gemv(ctx);
    std::mt19937 rng(9);
    std::normal_distribution<float> nd;
    std::vector<float> row(k);
    for (Type type : {Type::Q8_0, Type::Q6_K, Type::Q5_K, Type::Q4_K}) {
        const uint64_t rb = mako::mtl::Gemv::row_bytes(type, k), mat = rb * n;
        mako::mtl::Ref<MTL::Buffer> w(ctx.device()->newBuffer(mat * copies, MTL::ResourceStorageModeShared));
        auto* p = static_cast<std::byte*>(w->contents());
        for (uint32_t r = 0; r < n; ++r) {
            for (auto& v : row)
                v = nd(rng) * 0.02f;
            if (type == Type::Q8_0)
                q8_0(row.data(), p + r * rb, k);
            else
                mako::quant::quantize_row(type, row.data(), p + r * rb, k);
        }
        for (uint32_t c = 1; c < copies; ++c)
            std::memcpy(p + c * mat, p, mat);
        mako::mtl::Ref<MTL::Buffer> x(ctx.device()->newBuffer(k * 4, MTL::ResourceStorageModeShared));
        mako::mtl::Ref<MTL::Buffer> y(ctx.device()->newBuffer(uint64_t(n) * 4 * copies, MTL::ResourceStorageModeShared));
        std::memset(x->contents(), 0, k * 4);
        // Warm-up (~0.3 s of GPU work, so the clocks are up as in decode), then the median of 50 runs.
        std::vector<double> times;
        double warm = 0;
        for (int it = 0; it < 2000 && (it < 56 || warm < 0.3); ++it) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            for (uint32_t c = 0; c < copies; ++c)
                gemv.encode(enc, mako::mtl::Gemv::Args{type, k, n, 1, 1, false}, w.get(), c * mat, x.get(), 0, y.get(),
                            uint64_t(c) * n * 4);
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            const double t = cb->GPUEndTime() - cb->GPUStartTime();
            warm += t;
            times.push_back(t);
        }
        std::vector<double> last(times.end() - 50, times.end());
        std::ranges::sort(last);
        const double best = last[25];
        std::println("{:5} {}x{}: {:.1f} MB a matrix, {:.3f} ms each, {:.0f} GB/s", mako::gguf::traits(type).name, k, n,
                     mat / 1e6, best * 1e3 / copies, mat * copies / best / 1e9);
    }
    return 0;
}
