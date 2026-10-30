// Krea 2 against the reference pipeline's tensors (fp32 on the CPU, 256x256, 2 steps, seed 42; written by a script
// beside the model, manifest.json listing them):
//   tokens   the prompt template's token ids
//   text     the text encoder's twelve tapped hidden states
//   dit      the transformer on the reference's tapped states and latents: stages of the first two steps and every
//            step's prediction
//   vae      the VAE decoder on the reference's final latents
//   pipeline the prompt to the image (the reference's starting noise)
//   krea_ref_test <model dir> <transformer file> <reference dir> <mode>
#include "image/krea2.hpp"
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

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5 || !fs::exists(fs::path(argv[1]) / "model_index.json") || !fs::exists(argv[2]) ||
        !fs::exists(fs::path(argv[3]) / "manifest.json")) {
        std::println("model {}, transformer {} or reference {} not found: skipped", argc > 1 ? argv[1] : "",
                     argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");
        return 77;
    }
    const fs::path model = argv[1], ref = argv[3];
    const std::string transformer = argv[2], mode = argv[4];
    glz::generic manifest;
    if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    const std::string prompt = manifest["prompt"].get_string();
    std::vector<int32_t> want;
    for (const auto& v : manifest["ids"].get_array())
        want.push_back(int32_t(v.get_number()));
    const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16, N = g * g;
    mako::mtl::Context ctx;
    mako::image::Krea2 engine(model.string(), ctx, {.transformer = transformer});

    if (mode == "tokens") {
        check(engine.system_tokens() == uint32_t(manifest["drop_idx"].get_number()),
              std::format("the template's system part: {} tokens", engine.system_tokens()));
        const auto ids = engine.tokenizer().encode(mako::image::Krea2::templated(prompt), true);
        check(ids == want, std::format("the template's {} token ids (the reference's {})", ids.size(), want.size()));
    } else if (mode == "text") {
        const auto got = engine.encode_text(want, uint32_t(manifest["drop_idx"].get_number()), mako::image::Krea2::kTaps);
        const auto ref_e = read_f32(ref / "text_taps.f32");
        if (got.size() != ref_e.size()) {
            check(false, std::format("{} values, the reference {}", got.size(), ref_e.size()));
        } else {
            const Diff d = diff(got, ref_e, 2560);
            check(d.rel < 0.03, std::format("{} rows x 12 taps: rel error {:.2e}, lowest row cosine {:.6f}",
                                            got.size() / 2560 / 12, d.rel, d.min_cos));
            for (uint32_t t = 0; t < 12; ++t) {
                std::vector<float> a, b;
                for (size_t r = 0; r < got.size() / 2560 / 12; ++r) {
                    a.insert(a.end(), got.begin() + (r * 12 + t) * 2560, got.begin() + (r * 12 + t + 1) * 2560);
                    b.insert(b.end(), ref_e.begin() + (r * 12 + t) * 2560, ref_e.begin() + (r * 12 + t + 1) * 2560);
                }
                const Diff dt = diff(a, b, 2560);
                std::println("        tap {}: rel error {:.2e}, lowest row cosine {:.5f}", t, dt.rel, dt.min_cos);
            }
        }
    } else if (mode == "dit") {
        std::map<std::string, std::vector<float>> got;
        uint32_t step = 0;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (step < 2)
                got[std::format("s{}_{}{}", step, name, layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        engine.begin(read_f32(ref / "text_taps.f32"), g, g);
        const auto& ts = manifest["timesteps"].get_array();
        for (step = 0; step < ts.size(); ++step) {
            const auto pred = engine.step(read_f32(ref / std::format("step{}_latents.f32", step)), float(ts[step].get_number()));
            const Diff d = diff(pred, read_f32(ref / std::format("step{}_noise_pred.f32", step)), 64);
            check(d.rel < 0.05, std::format("step {} (sigma {:.3f}) prediction: rel error {:.2e}, lowest row cosine {:.4f}",
                                            step, ts[step].get_number(), d.rel, d.min_cos));
        }
        got["s1_fusion"] = got["s0_fusion"];
        got["s1_txt_in"] = got["s0_txt_in"];
        for (const auto& [name, width] : std::vector<std::pair<std::string, size_t>>{
                 {"s0_fusion", 2560}, {"s0_txt_in", 6144}, {"s0_temb", 6144}, {"s0_tmod", 36864}, {"s0_img_in", 6144},
                 {"s0_block0", 6144}, {"s0_block1", 6144}, {"s0_block13", 6144}, {"s0_block27", 6144},
                 {"s1_temb", 6144}, {"s1_block27", 6144}}) {
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
    } else if (mode == "vae") {
        std::vector<float> vae_in;
        std::map<std::string, std::vector<float>> stages;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (name == "vae_in")
                vae_in.assign(data, data + n);
            stages[name.substr(4) + (layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        const auto img = engine.decode(read_f32(ref / "final_latents.f32"), g, g);
        auto planar = [](const std::vector<float>& v, size_t pixels, size_t c, size_t stride) {
            std::vector<float> o(pixels * c);
            for (size_t p = 0; p < pixels; ++p)
                for (size_t ch = 0; ch < c; ++ch)
                    o[ch * pixels + p] = v[p * stride + ch];
            return o;
        };
        const Diff din = diff(planar(vae_in, size_t(2 * g) * 2 * g, 16, 16), read_f32(ref / "vae_in.f32"), size_t(4 * N));
        check(din.rel < 1e-5, std::format("denormalized latents: rel error {:.2e}", din.rel));
        const Diff d = diff(planar(img, size_t(size) * size, 3, 4), read_f32(ref / "vae_out.f32"), size_t(size) * size);
        // The decoder's stages, where the reference wrote them (channels last).
        for (const char* st : {"pq", "conv_in", "mid", "up0", "up1", "up2", "up3"}) {
            const auto p = ref / std::format("vae_{}.f32", st);
            if (!fs::exists(p) || !stages.contains(st))
                continue;
            const auto w = read_f32(p);
            const auto& gs = stages[st];
            if (gs.size() != w.size()) {
                std::println("        {}: {} values, the reference {}", st, gs.size(), w.size());
                continue;
            }
            const Diff ds = diff(gs, w, w.size());
            std::println("        {}: rel error {:.2e}", st, ds.rel);
        }
        check(d.rel < 0.02, std::format("{}x{} RGB: rel error {:.2e}, lowest channel cosine {:.5f}", size, size, d.rel,
                                        d.min_cos));
    } else if (mode == "pipeline") {
        const auto sig = engine.sigmas(uint32_t(manifest["steps"].get_number()), N);
        double worst = 0;
        const auto& ws = manifest["sigmas"].get_array();
        for (size_t i = 0; i < ws.size() && i < sig.size(); ++i)
            worst = std::max(worst, std::abs(double(sig[i]) - ws[i].get_number()));
        check(sig.size() == ws.size() && worst < 1e-6, std::format("{} noise levels, largest difference {:.1e}", sig.size(), worst));
        mako::image::Engine::Request r;
        r.prompt = prompt;
        r.width = r.height = size;
        r.steps = uint32_t(manifest["steps"].get_number());
        const auto noise = read_f32(ref / "step0_latents.f32");
        r.noise = noise;
        std::vector<float> vae_in;
        engine.set_probe([&](const std::string& name, int, const float* data, size_t n) {
            if (name == "vae_in")
                vae_in.assign(data, data + n);
        });
        const auto img = engine.generate(r);
        {
            std::vector<float> planar_in(vae_in.size());
            const size_t px = vae_in.size() / 16;
            for (size_t p = 0; p < px; ++p)
                for (size_t ch = 0; ch < 16; ++ch)
                    planar_in[ch * px + p] = vae_in[p * 16 + ch];
            const Diff dl = diff(planar_in, read_f32(ref / "vae_in.f32"), px);
            std::println("        final latents: rel error {:.2e}", dl.rel);
        }
        std::vector<float> planar(size_t(size) * size * 3);
        const size_t px = size_t(size) * size;
        for (size_t p = 0; p < px; ++p)
            for (size_t ch = 0; ch < 3; ++ch)
                planar[ch * px + p] = img[p * 4 + ch];
        // Two large steps carry the transformer's rounding into a 256x256 image: the reference pipeline in BF16 (its
        // usual precision) from the same noise is 5.4e-2 off its fp32 latents and 0.158 off its image; mako measured
        // 4.7e-2 and 0.133.
        const Diff d = diff(planar, read_f32(ref / "vae_out.f32"), px);
        check(d.rel < 0.15, std::format("{}x{}, {} steps: rel error {:.2e}, lowest channel cosine {:.5f}", size, size,
                                        r.steps, d.rel, d.min_cos));
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
