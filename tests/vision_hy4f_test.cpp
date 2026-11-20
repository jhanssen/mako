// Hy4-flash's vision encoder against the reference's rows for a test image (data/vision/hy4f_rgb_200x150_rows.f16:
// the reference run with the projector's weights in F32, rounded to fp16 here), per row cosine and relative RMS error.
//   vision_hy4f_test <mmproj.gguf> <tests/data/vision>
// Exit 77: the projector file is missing (skipped).
#include "mtl/context.hpp"
#include "vision/encoder.hpp"
#include "vision/image.hpp"
#include "vision/mmproj.hpp"
#include "vision/preprocess.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <print>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: vision_hy4f_test <mmproj.gguf> <tests/data/vision>");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("projector {} not found: skipped", argv[1]);
        return 77;
    }
    const std::filesystem::path dir = argv[2];
    auto mm = mako::vision::open_mmproj(argv[1]);
    if (!mm) {
        std::println(stderr, "{}", mm.error());
        return 1;
    }
    std::ifstream f(dir / "rgb_200x150.png", std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(f), {}};
    const auto prep = mako::vision::prepare(mm->prep, mako::vision::decode_image(std::as_bytes(std::span(bytes))));
    mako::mtl::Context ctx;
    mako::vision::Encoder enc(ctx, *mm);
    const auto out = enc.encode(prep);

    std::ifstream rf(dir / "hy4f_rgb_200x150_rows.f16", std::ios::binary);
    const std::vector<char> raw{std::istreambuf_iterator<char>(rf), {}};
    std::vector<float> ref(raw.size() / 2);
    for (size_t i = 0; i < ref.size(); ++i) {
        _Float16 h;
        std::memcpy(&h, raw.data() + 2 * i, 2);
        ref[i] = float(h);
    }
    const uint32_t W = enc.out_embd();
    if (out.embd.size() != ref.size() || out.rows * W != ref.size()) {
        std::println("FAIL {} rows, the reference {}", out.rows, ref.size() / W);
        return 1;
    }
    double min_cos = 1, num = 0, den = 0;
    for (uint32_t r = 0; r < out.rows; ++r) {
        double d = 0, na = 0, nb = 0;
        for (uint32_t i = 0; i < W; ++i) {
            const double a = out.embd[size_t(r) * W + i], b = ref[size_t(r) * W + i];
            d += a * b;
            na += a * a;
            nb += b * b;
            num += (a - b) * (a - b);
        }
        den += nb;
        min_cos = std::min(min_cos, d / std::sqrt(na * nb));
    }
    // Measured: min cosine 0.999998, relative RMS error 1.00e-3 (fp16 matrix products against the reference's F32).
    const double rel = std::sqrt(num / den);
    const bool ok = min_cos > 0.99995 && rel < 1.5e-3;
    std::println("{}  {} rows: min cosine {:.6f} (limit 0.99995), relative RMS error {:.2e} (limit 1.5e-3)",
                 ok ? "ok  " : "FAIL", out.rows, min_cos, rel);
    return ok ? 0 : 1;
}
