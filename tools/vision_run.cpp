// Runs an image through a vision projector file.
//   vision_run <mmproj.gguf> <image> [--prep-out FILE] [--out FILE] [--probe PREFIX] [--timing 1]
// --prep-out: the encoder's prepared input as raw fp32, planar [3][height][width].
// --out: the encoder's rows for the text model as raw fp32 [rows][width]; --probe: each stage's output to
// PREFIX.<stage>.f32 (patch_bias, layer_out-N, vit_out, merged / projected); --timing: GPU time by stage.
#include "mtl/context.hpp"
#include "vision/encoder.hpp"
#include "vision/image.hpp"
#include "vision/mmproj.hpp"
#include "vision/preprocess.hpp"

#include <chrono>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: vision_run <mmproj.gguf> <image> [--prep-out FILE]");
        return 2;
    }
    std::string prep_out, out, probe;
    bool timing = false;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--prep-out")
            prep_out = argv[i + 1];
        else if (a == "--out")
            out = argv[i + 1];
        else if (a == "--probe")
            probe = argv[i + 1];
        else if (a == "--timing")
            timing = std::string_view(argv[i + 1]) != "0";
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    auto mm = mako::vision::open_mmproj(argv[1]);
    if (!mm) {
        std::println(stderr, "{}", mm.error());
        return 1;
    }
    std::ifstream in(argv[2], std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
    if (!in && !in.eof()) {
        std::println(stderr, "cannot read {}", argv[2]);
        return 1;
    }
    try {
        const auto img = mako::vision::decode_image(std::as_bytes(std::span(bytes)));
        const auto prep = mako::vision::prepare(mm->prep, img);
        std::println("image {} x {} -> {} x {}", img.width, img.height, prep.width, prep.height);
        if (!prep_out.empty()) {
            std::FILE* f = std::fopen(prep_out.c_str(), "wb");
            if (!f || std::fwrite(prep.chw.data(), 4, prep.chw.size(), f) != prep.chw.size()) {
                std::println(stderr, "cannot write {}", prep_out);
                return 1;
            }
            std::fclose(f);
        }
        if (out.empty() && probe.empty() && !timing)
            return 0;
        mako::mtl::Context ctx;
        mako::vision::Encoder enc(ctx, *mm);
        if (!probe.empty())
            enc.probe = [&](std::string_view name, const float* data, size_t n) {
                const std::string path = std::format("{}.{}.f32", probe, name);
                if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                    std::fwrite(data, 4, n, f);
                    std::fclose(f);
                }
            };
        enc.encode(prep); // the first run builds the pipelines
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = enc.encode(prep);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::println("{} rows x {} ({} x {} cells) in {:.1f} ms", r.rows, enc.out_embd(), r.cells_x, r.cells_y, ms);
        if (timing) {
            enc.timing = true;
            enc.stage_ms.clear();
            enc.encode(prep);
            double sum = 0;
            for (const auto& [name, t] : enc.stage_ms)
                sum += t;
            for (const auto& [name, t] : enc.stage_ms)
                std::println("  {:10} {:8.1f} ms GPU ({:.0f}%)", name, t, 100.0 * t / sum);
        }
        if (!out.empty()) {
            std::FILE* f = std::fopen(out.c_str(), "wb");
            if (!f || std::fwrite(r.embd.data(), 4, r.embd.size(), f) != r.embd.size()) {
                std::println(stderr, "cannot write {}", out);
                return 1;
            }
            std::fclose(f);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
    return 0;
}
