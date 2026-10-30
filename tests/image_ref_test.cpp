// Qwen-Image 2.1 against the reference pipeline's tensors (fp32 on the CPU, 512x512, 3 steps, seed 42; written by a
// script beside the model, manifest.json listing them):
//   tokens   the prompt template's token ids
//   text     the text encoder's output (the transformer's prompt embeddings)
//   dit      the transformer on the reference's prompt embeddings and latents: each stage of the first two steps (the
//            whole sequence, then the image's rows against the cached text) and every step's prediction
//   image_ref_test <model dir> <reference dir> <mode>
#include "image/engine.hpp"
#include "mtl/context.hpp"
#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <print>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

std::string read_file(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<float> read_f32(const fs::path& p)
{
    const std::string s = read_file(p);
    std::vector<float> v(s.size() / 4);
    std::memcpy(v.data(), s.data(), v.size() * 4);
    return v;
}

// Relative RMS error over everything, and the lowest cosine similarity of a row of `width`.
struct Diff {
    double rel = 0, min_cos = 1;
};
Diff diff(const std::vector<float>& got, const std::vector<float>& want, size_t width)
{
    Diff d;
    double e = 0, r = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        e += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        r += double(want[i]) * want[i];
    }
    d.rel = std::sqrt(e / r);
    for (size_t row = 0; row < want.size() / width; ++row) {
        double gw = 0, gg = 0, ww = 0;
        for (size_t i = row * width; i < (row + 1) * width; ++i) {
            gw += double(got[i]) * want[i];
            gg += double(got[i]) * got[i];
            ww += double(want[i]) * want[i];
        }
        d.min_cos = std::min(d.min_cos, gw / std::sqrt(gg * ww));
    }
    return d;
}

// The pipeline's text-to-image prompt template.
std::string templated(const std::string& prompt)
{
    return "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n" + prompt +
           "<|im_end|>\n<|im_start|>assistant\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || !fs::exists(fs::path(argv[1]) / "transformer") || !fs::exists(fs::path(argv[2]) / "manifest.json")) {
        std::println("model {} or reference {} not found: skipped", argc > 1 ? argv[1] : "", argc > 2 ? argv[2] : "");
        return 77;
    }
    const fs::path model = argv[1], ref = argv[2];
    const std::string mode = argv[3];
    glz::generic manifest;
    if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    auto tok = mako::text::Tokenizer::load_json((model / "processor" / "tokenizer.json").string());
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    const std::string prompt = manifest["prompt"].get_string();
    std::vector<int32_t> want;
    for (const auto& v : manifest["ids"].get_array())
        want.push_back(int32_t(v.get_number()));

    if (mode == "tokens") {
        const auto ids = tok->encode(templated(prompt), true);
        check(ids == want, std::format("the template's {} token ids (the reference's {})", ids.size(), want.size()));
        check(tok->decode(ids) == templated(prompt), "they decode to the template");
    } else if (mode == "text") {
        mako::mtl::Context ctx;
        mako::image::Engine engine(model.string(), ctx);
        const auto t0 = std::chrono::steady_clock::now();
        const auto got = engine.encode_text(want, uint32_t(manifest["drop_idx"].get_number()));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto ref_e = read_f32(ref / "prompt_embeds.f32");
        if (got.size() != ref_e.size()) {
            check(false, std::format("{} values, the reference {}", got.size(), ref_e.size()));
        } else {
            // The reference pipeline in BF16 (its usual precision) is 7.8-8.6e-2 off its fp32 output (lowest row
            // cosine 0.91-0.92); mako keeps the residual stream in fp32 (measured 1.2e-2, 0.998).
            const Diff d = diff(got, ref_e, 4096);
            check(d.rel < 0.03 && d.min_cos > 0.995,
                  std::format("{} rows: rel error {:.2e}, lowest row cosine {:.6f} ({:.0f} ms)", got.size() / 4096,
                              d.rel, d.min_cos, ms));
        }
    } else if (mode == "dit") {
        mako::mtl::Context ctx;
        mako::image::Engine engine(model.string(), ctx);
        const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16, N = g * g;
        std::map<std::string, std::vector<float>> got;
        uint32_t step = 0;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (step < 2)
                got[std::format("s{}_{}{}", step, name, layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        engine.begin(read_f32(ref / "prompt_embeds.f32"), g, g);
        const auto& ts = manifest["timesteps"].get_array();
        for (step = 0; step < ts.size(); ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto pred = engine.step(read_f32(ref / std::format("step{}_latents.f32", step)), float(ts[step].get_number()));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            auto want_p = read_f32(ref / std::format("step{}_noise_pred.f32", step));
            want_p.erase(want_p.begin(), want_p.end() - ptrdiff_t(N) * 64); // the image's rows
            const Diff d = diff(pred, want_p, 64);
            check(d.rel < 0.05, std::format("step {} (sigma {:.3f}) prediction: rel error {:.2e}, lowest row cosine "
                                            "{:.4f} ({:.0f} ms)",
                                            step, ts[step].get_number(), d.rel, d.min_cos, ms));
        }
        for (const auto& [name, width] : std::vector<std::pair<std::string, size_t>>{
                 {"s0_temb", 4096}, {"s0_modulation", 16384}, {"s0_txt_in", 4096}, {"s0_img_in", 4096},
                 {"s0_block0", 4096}, {"s0_block1", 4096}, {"s0_block15", 4096}, {"s0_block31", 4096},
                 {"s0_norm_out", 4096}, {"s1_img_in", 4096}, {"s1_block0", 4096}, {"s1_block31", 4096},
                 {"s1_norm_out", 4096}}) {
            const auto want_s = read_f32(ref / (name + ".f32"));
            const auto it = got.find(name);
            if (it == got.end() || it->second.size() != want_s.size()) {
                check(false, std::format("{}: {} values, the reference {}", name,
                                         it == got.end() ? 0 : it->second.size(), want_s.size()));
                continue;
            }
            const Diff d = diff(it->second, want_s, width);
            std::println("  {:4} {}: rel error {:.2e}, lowest row cosine {:.5f}", "", name, d.rel, d.min_cos);
        }
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
