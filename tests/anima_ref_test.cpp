// Anima against the reference implementation's tensors (fp32 on the CPU, 256x256, 2 steps, Euler with its simple
// schedule, seed 42; written by a script beside the model, manifest.json listing them):
//   tokens   the Qwen3 and T5 token ids of the prompt and of the empty negative prompt
//   text     the text encoder's final hidden states
//   adapter  the adapter's context rows (512, zero padded) from the reference's states
//   dit      the transformer on the reference's context and latents: stages of the first two passes and every
//            pass's prediction
//   vae      the VAE decoder on the reference's final latents
//   pipeline the prompt to the image (the reference's starting noise)
//   anima_ref_test <model dir> <transformer file> <reference dir> <mode>
#include "image/anima.hpp"
#include "mtl/context.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
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
        if (ww > 0)
            d.min_cos = std::min(d.min_cos, gw / std::sqrt(gg * ww));
    }
    return d;
}

// Latents [16][2g][2g] to tokens [g * g][64] (a token's channel, row, column) and back.
std::vector<float> to_tokens(const std::vector<float>& v, uint32_t g)
{
    std::vector<float> o(v.size());
    for (uint32_t c = 0; c < 16; ++c)
        for (uint32_t y = 0; y < 2 * g; ++y)
            for (uint32_t x = 0; x < 2 * g; ++x)
                o[((y / 2) * g + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2] = v[(uint64_t(c) * 2 * g + y) * 2 * g + x];
    return o;
}
std::vector<float> from_tokens(const std::vector<float>& t, uint32_t g)
{
    std::vector<float> o(t.size());
    for (uint32_t c = 0; c < 16; ++c)
        for (uint32_t y = 0; y < 2 * g; ++y)
            for (uint32_t x = 0; x < 2 * g; ++x)
                o[(uint64_t(c) * 2 * g + y) * 2 * g + x] = t[((y / 2) * g + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2];
    return o;
}

std::vector<int32_t> ids(const glz::generic& v)
{
    std::vector<int32_t> out;
    for (const auto& x : v.get_array())
        out.push_back(int32_t(x.get_number()));
    return out;
}

// The first Qwen-Image VAE's latent statistics.
constexpr float kMean[16] = {-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                             0.4134,  -0.0715, 0.5517,  -0.3632, -0.1922, -0.9497, 0.2503, -0.2921};
constexpr float kStd[16] = {2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916};

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5 || !fs::exists(argv[1]) || !fs::exists(argv[2]) || !fs::exists(fs::path(argv[3]) / "manifest.json")) {
        std::println("model {}, transformer {} or reference {} not found: skipped", argc > 1 ? argv[1] : "",
                     argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");
        return 77;
    }
    const fs::path ref = argv[3];
    const std::string mode = argv[4];
    glz::generic manifest;
    if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    const std::string prompt = manifest["prompt"].get_string();
    const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16, N = g * g;
    mako::mtl::Context ctx;
    mako::image::Anima engine(argv[1], ctx, {.transformer = argv[2]});
    // The adapter's input as prompt_embeddings packs it, from the reference's states and token ids.
    auto packed = [&](const std::string& name) {
        const auto states = read_f32(ref / (name + "_qwen.f32"));
        const auto t5 = ids(manifest[name + "_t5_ids"]);
        std::vector<float> p{float(states.size() / 1024), float(t5.size())};
        p.insert(p.end(), states.begin(), states.end());
        for (int32_t id : t5)
            p.push_back(float(id));
        p.insert(p.end(), t5.size(), 1.0f);
        return p;
    };

    if (mode == "tokens") {
        for (const auto& [name, text] : {std::pair{"pos", prompt}, std::pair{"neg", std::string()}}) {
            const auto q = engine.qwen_ids(text);
            const auto wq = ids(manifest[std::string(name) + "_qwen_ids"]);
            check(q == wq, std::format("{}: {} Qwen3 ids (the reference's {})", name, q.size(), wq.size()));
            const auto t = engine.t5_ids(text).first;
            const auto wt = ids(manifest[std::string(name) + "_t5_ids"]);
            check(t == wt, std::format("{}: {} T5 ids (the reference's {})", name, t.size(), wt.size()));
        }
        const auto w = mako::image::Anima::weighted("a (b) ((c:1.5)) \\(d\\) (e:x)");
        check(w.size() == 6 && w[0].first == "a " && w[1] == std::pair<std::string, float>{"b", 1.1f} &&
                  w[3] == std::pair<std::string, float>{"c", 1.5f} && w[4].first == " (d) " &&
                  w[5] == std::pair<std::string, float>{"e:x", 1.1f},
              "prompt weights: (b) 1.1, nested (c:1.5) 1.5, escaped parentheses, a weight that is not a number");
    } else if (mode == "text") {
        for (const char* name : {"pos", "neg"}) {
            const auto got = engine.encode_text(ids(manifest[std::string(name) + "_qwen_ids"]), 0, {}, true);
            const auto want = read_f32(ref / std::format("{}_qwen.f32", name));
            const Diff d = diff(got, want, 1024);
            check(got.size() == want.size() && d.rel < 0.02,
                  std::format("{}: {} rows: rel error {:.2e}, lowest row cosine {:.5f}", name, want.size() / 1024,
                              d.rel, d.min_cos));
        }
    } else if (mode == "adapter") {
        for (const char* name : {"pos", "neg"}) {
            std::vector<float> got;
            engine.set_probe([&](const std::string& n, int, const float* data, size_t k) {
                if (n == "context")
                    got.assign(data, data + k);
            });
            engine.begin(packed(name), g, g);
            const auto want = read_f32(ref / std::format("{}_context.f32", name));
            const Diff d = diff(got, want, 1024);
            check(got.size() == want.size() && d.rel < 0.02,
                  std::format("{}: {} rows: rel error {:.2e}, lowest row cosine {:.5f}", name, want.size() / 1024,
                              d.rel, d.min_cos));
        }
    } else if (mode == "dit") {
        std::map<std::string, std::vector<float>> got;
        uint32_t call = 0;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (call < 2)
                got[std::format("c{}_{}{}", call, name, layer >= 0 ? std::to_string(layer) : "")].assign(data,
                                                                                                    data + n);
        });
        // With guidance the reference runs the prompt's pass, then the negative prompt's, each step.
        const bool guided = manifest["cfg"].get_number() > 1;
        engine.begin(packed("pos"), g, g, 0);
        if (guided)
            engine.begin(packed("neg"), g, g, 1);
        const auto& ts = manifest["timesteps"].get_array();
        for (call = 0; call < ts.size(); ++call) {
            const auto x = to_tokens(read_f32(ref / std::format("c{}_x.f32", call)), g);
            const auto pred = from_tokens(engine.step(x, float(ts[call].get_number()), guided ? call % 2 : 0), g);
            const Diff d = diff(pred, read_f32(ref / std::format("c{}_out.f32", call)), size_t(4 * N));
            check(d.rel < 0.05, std::format("pass {} (sigma {:.3f}): rel error {:.2e}, lowest channel cosine {:.4f}",
                                            call, ts[call].get_number(), d.rel, d.min_cos));
        }
        for (const auto& [name, width] : std::vector<std::pair<std::string, size_t>>{
                 {"c0_t_emb", 2048}, {"c0_x_embed", 2048}, {"c0_block0", 2048}, {"c0_block13", 2048},
                 {"c0_block27", 2048}, {"c1_t_emb", 2048}, {"c1_block27", 2048}}) {
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
    } else if (mode == "vae" || mode == "pipeline") {
        std::vector<float> img;
        if (mode == "vae") {
            auto lat = read_f32(ref / "final_latents.f32");
            const size_t px = lat.size() / 16;
            for (size_t i = 0; i < lat.size(); ++i)
                lat[i] = (lat[i] - kMean[i / px]) / kStd[i / px];
            img = engine.decode(to_tokens(lat, g), g, g);
        } else {
            std::vector<float> sig = engine.sigmas(uint32_t(manifest["steps"].get_number()), N, std::nullopt);
            const auto& ws = manifest["sigmas"].get_array();
            double worst = 0;
            for (size_t i = 0; i < ws.size() && i < sig.size(); ++i)
                worst = std::max(worst, std::abs(double(sig[i]) - ws[i].get_number()));
            check(sig.size() == ws.size() && worst < 1e-6,
                  std::format("{} noise levels (the reference's {}), largest difference {:.1e}", sig.size(), ws.size(),
                              worst));
            mako::image::Engine::Request r;
            r.prompt = prompt;
            r.width = r.height = size;
            r.steps = uint32_t(manifest["steps"].get_number());
            if (manifest["cfg"].get_number() > 1) {
                r.negative = "";
                r.guidance = float(manifest["cfg"].get_number());
            }
            const auto noise = to_tokens(read_f32(ref / "noise.f32"), g);
            r.noise = noise;
            std::vector<float> vae_in;
            engine.set_probe([&](const std::string& name, int, const float* data, size_t n) {
                if (name == "vae_in")
                    vae_in.assign(data, data + n);
            });
            img = engine.generate(r);
            // The final latents (channels last there, [16][2g][2g] in the reference).
            std::vector<float> planar(vae_in.size());
            const size_t px = vae_in.size() / 16;
            for (size_t p = 0; p < px; ++p)
                for (size_t c = 0; c < 16; ++c)
                    planar[c * px + p] = vae_in[p * 16 + c];
            std::println("        final latents: rel error {:.2e}",
                         diff(planar, read_f32(ref / "final_latents.f32"), px).rel);
        }
        // The reference's image is RGB in [0, 1].
        std::vector<float> rgb;
        for (size_t p = 0; p < size_t(size) * size; ++p)
            for (int c = 0; c < 3; ++c)
                rgb.push_back((img[p * 4 + c] + 1.0f) / 2.0f);
        // The turbo model at its second step (sigma 0.75) moves its prediction 2.3e-2 for a 2.2e-3 change in its
        // input (mako against mako), so the first step's 6e-3 comes out near 4e-2 in the final latents; mako measured
        // 4.6e-2 there and 6.7e-2 in the image.
        const Diff d = diff(rgb, read_f32(ref / "image.f32"), 3);
        check(d.rel < (mode == "vae" ? 0.01 : 0.1),
              std::format("{}x{} RGB: rel error {:.2e}, lowest pixel cosine {:.4f}", size, size, d.rel, d.min_cos));
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
