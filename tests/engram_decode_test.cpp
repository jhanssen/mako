// DeepSeek-V4.1's Engram row decode kernel (ds_engram_decode) against the CPU dequantization on random Q6_K rows,
// including the [token][layer][column] -> [layer][token][column] reordering. Exact up to the float multiply order.
//   engram_decode_test
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"

#include <cmath>
#include <cstring>
#include <print>
#include <random>
#include <vector>

namespace {

// Layout must match EngramDecodeArgs in dsv41.metal.
struct EngramDecodeArgs {
    uint32_t dim, tokens, layers, cols, q6k;
};

} // namespace

int main()
{
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto pso = ctx.pipeline(lib.get(), "ds_engram_decode");
    constexpr uint32_t dim = 256, tokens = 5, layers = 2, cols = 24, row_bytes = dim / 256 * 210;
    const uint32_t rows = tokens * layers * cols;
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<std::byte> raw(uint64_t(rows) * row_bytes);
    for (auto& b : raw)
        b = std::byte(byte(rng));
    for (uint32_t r = 0; r < rows; ++r) { // finite d of a realistic size
        const __fp16 d = __fp16(0.0005f * (1 + r % 5));
        std::memcpy(raw.data() + uint64_t(r) * row_bytes + 208, &d, 2);
    }
    auto buf = [&](uint64_t bytes) {
        return mako::mtl::Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    auto in = buf(raw.size()), out = buf(uint64_t(rows) * dim * 4);
    std::memcpy(in->contents(), raw.data(), raw.size());
    const EngramDecodeArgs a{dim, tokens, layers, cols, 1};
    auto* cb = ctx.queue()->commandBuffer();
    auto* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso.get());
    enc->setBuffer(in.get(), 0, 0);
    enc->setBuffer(out.get(), 0, 1);
    enc->setBytes(&a, sizeof(a), 2);
    enc->dispatchThreads(MTL::Size(dim, rows, 1), MTL::Size(256, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    const auto* y = static_cast<const float*>(out->contents());
    std::vector<float> want(dim);
    double worst = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t t = r / (layers * cols), l = (r / cols) % layers, c = r % cols;
        mako::quant::dequant_row(mako::gguf::Type::Q6_K, raw.data() + uint64_t(r) * row_bytes, want.data(), dim);
        const float* got = y + (uint64_t(l * tokens + t) * cols + c) * dim;
        for (uint32_t i = 0; i < dim; ++i)
            worst = std::max(worst, double(std::abs(got[i] - want[i])) / (std::abs(want[i]) + 1e-6));
    }
    const bool ok = worst < 1e-6;
    std::println("Q6_K Engram rows: {} rows of {}, worst relative difference {:.2e}  {}", rows, dim, worst,
                 ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
