// DeepSeek-V4.1's Engram row decode kernel (ds_engram_decode) against the CPU dequantization on random Q6_K and IQ4_NL
// rows, including the [token][layer][column] -> [layer][token][column] reordering (exact up to the float multiply
// order), and the CPU dequantization against the per-format GEMV kernel on the same rows (an independent decoder).
//   engram_decode_test
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"
#include "mtl/gemv.hpp"
#include "quant/dequant.hpp"

#include <cmath>
#include <cstring>
#include <print>
#include <random>
#include <vector>

namespace {

// Layout must match EngramDecodeArgs in dsv41.metal.
struct EngramDecodeArgs {
    uint32_t dim, tokens, layers, cols, format;
};

} // namespace

int main()
{
    using mako::gguf::Type;
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto pso = ctx.pipeline(lib.get(), "ds_engram_decode");
    mako::mtl::Gemv gemv(ctx);
    constexpr uint32_t dim = 256, tokens = 5, layers = 2, cols = 24, rows = tokens * layers * cols;
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd;
    auto buf = [&](uint64_t bytes) {
        return mako::mtl::Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    int failures = 0;
    for (const auto [type, format] : {std::pair{Type::Q6_K, 1u}, std::pair{Type::IQ4_NL, 2u}}) {
        const auto tr = mako::gguf::traits(type);
        const uint32_t row_bytes = uint32_t(dim / tr.block_elems * tr.block_bytes);
        // the half d: at the end of a Q6_K block, at the start of an IQ4_NL one
        const uint32_t d_at = type == Type::Q6_K ? 208 : 0;
        std::vector<std::byte> raw(uint64_t(rows) * row_bytes);
        for (auto& b : raw)
            b = std::byte(byte(rng));
        for (uint64_t blk = 0; blk < raw.size() / tr.block_bytes; ++blk) { // finite d of a realistic size
            const __fp16 d = __fp16(0.0005f * (1 + blk % 5));
            std::memcpy(raw.data() + blk * tr.block_bytes + d_at, &d, 2);
        }
        auto in = buf(raw.size()), out = buf(uint64_t(rows) * dim * 4);
        auto x = buf(dim * 4), y = buf(rows * 4);
        std::memcpy(in->contents(), raw.data(), raw.size());
        auto* xf = static_cast<float*>(x->contents());
        for (uint32_t i = 0; i < dim; ++i)
            xf[i] = nd(rng);
        const EngramDecodeArgs a{dim, tokens, layers, cols, format};
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso.get());
        enc->setBuffer(in.get(), 0, 0);
        enc->setBuffer(out.get(), 0, 1);
        enc->setBytes(&a, sizeof(a), 2);
        enc->dispatchThreads(MTL::Size(dim, rows, 1), MTL::Size(256, 1, 1));
        gemv.encode(enc, mako::mtl::Gemv::Args{type, dim, rows, 1, 1, false}, in.get(), 0, x.get(), 0, y.get(), 0);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();

        const auto* got = static_cast<const float*>(out->contents());
        const auto* yg = static_cast<const float*>(y->contents());
        std::vector<float> want(dim);
        double worst = 0, err = 0, ref = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = r / (layers * cols), l = (r / cols) % layers, c = r % cols;
            mako::quant::dequant_row(type, raw.data() + uint64_t(r) * row_bytes, want.data(), dim);
            const float* g = got + (uint64_t(l * tokens + t) * cols + c) * dim;
            double dot = 0;
            for (uint32_t i = 0; i < dim; ++i) {
                worst = std::max(worst, double(std::abs(g[i] - want[i])) / (std::abs(want[i]) + 1e-6));
                dot += double(want[i]) * xf[i];
            }
            err += (yg[r] - dot) * (yg[r] - dot);
            ref += dot * dot;
        }
        const double rel = std::sqrt(err / ref);
        const bool ok = worst < 1e-6 && rel < 1e-5;
        failures += !ok;
        std::println("{} Engram rows: {} rows of {}, kernel vs CPU worst relative difference {:.2e}, CPU vs the GEMV "
                     "{:.2e}  {}",
                     tr.name, rows, dim, worst, rel, ok ? "ok" : "FAIL");
    }
    return failures ? 1 : 0;
}
