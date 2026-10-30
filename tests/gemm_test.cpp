// Prefill GEMM against CPU dequantization on real tensors, for several token counts, plus its speed. For expert
// tensors (one expert's matrix) and formats without CPU dequantization the reference is the per-format GEMV.
//   gemm_test <model shard.gguf> [tensor ...]
//   gemm_test --synthetic   random blocks of formats no test model carries (IQ4_XS, IQ2_XS, IQ1_S, IQ1_M), against
//                           the GEMV; random BF16 weights with bfloat activations, against the CPU
#include "gguf/gguf.hpp"
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "mtl/gemv.hpp"
#include "quant/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <expected>
#include <span>
#include <filesystem>
#include <print>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: gemm_test <model.gguf> [tensor ...] | --synthetic");
        return 2;
    }
    const bool synthetic = std::string_view(argv[1]) == "--synthetic";
    if (!synthetic && !std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    std::expected<mako::gguf::File, std::string> f = std::unexpected("synthetic");
    if (!synthetic && !(f = mako::gguf::File::open(argv[1]))) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    std::vector<std::string> names(argv + 2, argv + argc);
    if (synthetic)
        names = {"iq4_xs", "iq2_xs", "iq1_s", "iq1_m", "bf16"};
    else if (names.empty())
        names = {"blk.0.attn_q.weight",       "blk.0.ffn_down.weight",      "blk.0.ssm_f_a.weight",
                 "blk.0.hc_attn_fn.weight",   "blk.3.ffn_gate_inp.weight",  "blk.3.ffn_gate_exps.weight",
                 "blk.3.ffn_down_exps.weight", "blk.10.ffn_down_exps.weight"};

    mako::mtl::Context ctx;
    mako::mtl::Gemm gemm(ctx);
    mako::mtl::Gemv gemv(ctx);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    int failures = 0;
    std::vector<std::byte> fake;
    for (const auto& name : names) {
        mako::gguf::Tensor synth;
        const mako::gguf::Tensor* t = &synth;
        std::span<const std::byte> data;
        if (synthetic && name == "bf16") {
            // Weights of the scale a model's dense matrices have.
            synth.type = mako::gguf::Type::BF16;
            synth.dims = {2048, 4096};
            fake.resize(2048ull * 4096 * 2);
            auto* b = reinterpret_cast<uint16_t*>(fake.data());
            for (uint64_t i = 0; i < 2048ull * 4096; ++i) {
                const float v = 0.02f * nd(rng);
                uint32_t u;
                std::memcpy(&u, &v, 4);
                b[i] = uint16_t(u >> 16);
            }
        } else if (synthetic) {
            // An expert-down-sized matrix of random blocks, the block scale d small enough for half activations
            // (IQ1_M keeps d in the top nibbles of its four 16-bit scale words).
            using T = mako::gguf::Type;
            synth.type = name == "iq4_xs" ? T::IQ4_XS : name == "iq2_xs" ? T::IQ2_XS : name == "iq1_s" ? T::IQ1_S : T::IQ1_M;
            synth.dims = {2048, 4096};
            const uint32_t bb = uint32_t(mako::gguf::traits(synth.type).block_bytes);
            const uint64_t blocks = 2048 / 256 * 4096;
            fake.resize(blocks * bb);
            std::uniform_int_distribution<int> byte(0, 255);
            for (auto& b : fake)
                b = std::byte(byte(rng));
            for (uint64_t i = 0; i < blocks; ++i) {
                const __fp16 d = __fp16(0.001f * (1 + i % 7));
                if (synth.type != T::IQ1_M) {
                    std::memcpy(fake.data() + i * bb, &d, 2);
                    continue;
                }
                uint16_t bits, sc[4];
                std::memcpy(&bits, &d, 2);
                std::memcpy(sc, fake.data() + i * bb + 48, 8);
                for (int w = 0; w < 4; ++w)
                    sc[w] = uint16_t((sc[w] & 0x0fff) | (((bits >> (4 * w)) & 0xf) << 12));
                std::memcpy(fake.data() + i * bb + 48, sc, 8);
            }
        } else if (!(t = f->tensor(name)) || t->dims.size() < 2) {
            std::println("{:28} missing or not a matrix", name);
            ++failures;
            continue;
        }
        const uint32_t k = uint32_t(t->dims[0]), n = uint32_t(t->dims[1]);
        const auto tr = mako::gguf::traits(t->type);
        const uint64_t rb = uint64_t(k) / tr.block_elems * tr.block_bytes;
        if (synthetic)
            data = fake;
        else
            data = f->data(*t).subspan(t->dims.size() > 2 ? 7 * rb * n : 0, rb * n); // expert 7
        const bool cpu_ref = mako::quant::can_dequant(t->type);
        auto buffer = [&](uint64_t bytes) {
            return mako::mtl::Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        };
        auto w = buffer(data.size());
        std::memcpy(w->contents(), data.data(), data.size());

        // Reference rows: a sample spread over the matrix, its last row included.
        std::vector<uint32_t> rows;
        for (uint32_t i = 0; i < 48; ++i)
            rows.push_back(uint32_t(uint64_t(i) * (n - 1) / 47));
        std::vector<std::vector<float>> deq(rows.size(), std::vector<float>(k));
        if (cpu_ref)
            for (size_t i = 0; i < rows.size(); ++i)
                mako::quant::dequant_row(t->type, data.data() + rows[i] * rb, deq[i].data(), k);

        for (uint32_t tokens : {1u, 2u, 5u, 8u, 13u, 16u, 37u, 64u, 200u}) {
            auto x = buffer(uint64_t(k) * tokens * 4), xh = buffer(uint64_t(k) * tokens * 2);
            auto y = buffer(uint64_t(n) * tokens * 4);
            auto* xf = static_cast<float*>(x->contents());
            const bool bf16 = mako::mtl::Gemm::bf16_input(t->type);
            for (uint64_t i = 0; i < uint64_t(k) * tokens; ++i) {
                xf[i] = nd(rng);
                if (bf16) { // the values the GPU's conversion keeps, so the reference sees them too
                    uint32_t u;
                    std::memcpy(&u, &xf[i], 4);
                    u &= 0xffff0000u;
                    std::memcpy(&xf[i], &u, 4);
                }
            }
            auto yref = buffer(uint64_t(n) * tokens * 4);
            gemm.set_few_tokens(tokens <= 2 ? 2 : 0); // up to 2 tokens the few-token kernel, where it has one
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            if (bf16)
                gemm.to_bf16(enc, x.get(), 0, tokens, k, 0, xh.get(), 0);
            else
                gemm.to_half(enc, x.get(), 0, tokens, k, 0, xh.get(), 0);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            gemm.encode(enc, t->type, k, n, tokens, w.get(), 0, xh.get(), 0, y.get(), 0);
            if (!cpu_ref) {
                mako::mtl::Gemv::Args a{t->type, k, n, tokens, 1, false};
                gemv.encode(enc, a, w.get(), 0, x.get(), 0, yref.get(), 0);
            }
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            const auto* yf = static_cast<const float*>(y->contents());
            const auto* rf = static_cast<const float*>(yref->contents());
            double err = 0, ref = 0;
            for (uint32_t tok = 0; tok < tokens; ++tok)
                for (size_t i = 0; i < rows.size(); ++i) {
                    double acc = 0;
                    if (cpu_ref)
                        for (uint32_t j = 0; j < k; ++j)
                            acc += double(deq[i][j]) * xf[uint64_t(tok) * k + j];
                    else
                        acc = rf[uint64_t(tok) * n + rows[i]];
                    const double e = yf[uint64_t(tok) * n + rows[i]] - acc;
                    err += e * e;
                    ref += acc * acc;
                }
            const double rel = std::sqrt(err / ref);
            const bool ok = rel < 5e-3;
            failures += !ok;
            std::println("{:28} {:7} k={:5} n={:6} t={:4}  rel err {:.2e} vs {}  {}", name, tr.name, k, n, tokens,
                         rel, cpu_ref ? "CPU" : "GEMV", ok ? "ok" : "FAIL");
        }

        // Speed (random activations: the matrix units' power, and so clocks, depend on the data). The matrix is
        // repeated 64 times and the dispatches, concurrent, cycle through the copies, so its weights come from
        // memory, not cache, as a batch of experts' do in prefill. Formats with a few-token kernel are also timed
        // without it.
        constexpr uint32_t copies = 64;
        auto wc = buffer(data.size() * copies);
        for (uint32_t c = 0; c < copies; ++c)
            std::memcpy(static_cast<char*>(wc->contents()) + c * data.size(), data.data(), data.size());
        const bool few = t->type == mako::gguf::Type::IQ2_XXS || t->type == mako::gguf::Type::IQ2_S ||
                         t->type == mako::gguf::Type::IQ3_XXS;
        for (const uint32_t tokens : {1u, 2u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 64u, 512u}) {
            auto xh = buffer(uint64_t(k) * tokens * 2), y = buffer(uint64_t(n) * tokens * 4);
            for (uint64_t i = 0; i < uint64_t(k) * tokens; ++i)
                static_cast<__fp16*>(xh->contents())[i] = __fp16(nd(rng));
            auto time = [&](uint32_t few_tokens) {
                gemm.set_few_tokens(few_tokens);
                // Sustained: the GPU clocks down between short, spaced-out dispatches, so time back to back after a
                // warm-up of about half a second.
                double best = 1e9;
                for (int rep = 0; rep < 8; ++rep) {
                    auto* cb = ctx.queue()->commandBuffer();
                    // Concurrent, as prefill runs a batch of experts.
                    auto* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
                    const int n_disp = rep < 3 ? 200 : 64;
                    for (int d = 0; d < n_disp; ++d)
                        gemm.encode(enc, t->type, k, n, tokens, wc.get(), uint64_t(d % copies) * data.size(), xh.get(),
                                    0, y.get(), 0);
                    enc->endEncoding();
                    cb->commit();
                    cb->waitUntilCompleted();
                    if (rep >= 3)
                        best = std::min(best, (cb->GPUEndTime() - cb->GPUStartTime()) / n_disp);
                }
                return best;
            };
            const double tiled = time(0), fast = few ? time(1u << 30) : 0;
            gemm.set_few_tokens(0);
            std::println("{:28} t={:3}: tiled {:7.1f} us ({:4.0f} GB/s){}", name, tokens, tiled * 1e6,
                         data.size() / tiled / 1e9,
                         few && tokens <= 64 ? std::format(", few-token {:7.1f} us ({:4.0f} GB/s)", fast * 1e6,
                                                           data.size() / fast / 1e9)
                                             : "");
        }
    }
    return failures ? 1 : 0;
}
