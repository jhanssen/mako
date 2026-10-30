// Text to image with Qwen-Image 2.1 from its original files.
//   mako_image <model dir> <prompt> [-o out.png] [--width N] [--height N] [--steps N] [--seed N] [--count N] [--rgb]
//              [--lora FILE[:SCALE] ...] [--terminal F]
// --lora merges a LoRA into the weights (repeatable; SCALE its strength, default 1). --terminal: the schedule's last
// noise level (0: not stretched there, as distilled few-step LoRAs are trained; default: the model's).
#include "app/cli.hpp"
#include "core/log.hpp"
#include "core/telemetry.hpp"
#include "image/engine.hpp"
#include "image/png.hpp"
#include "mtl/context.hpp"

#include <chrono>
#include <format>
#include <optional>
#include <print>
#include <utility>
#include <vector>

int main(int argc, char** argv)
{
    std::string out = "image.png";
    uint32_t width = 1024, height = 1024, steps = 40;
    uint64_t seed = 0;
    uint32_t count = 1;
    std::vector<std::pair<std::string, float>> loras;
    std::optional<float> terminal;
    bool rgb = false;
    mako::app::Cli cli("mako_image", "<model dir> <prompt>", 2, 2);
    cli.option("-o", "FILE", "the PNG to write (default image.png)", out)
        .option("--width", "N", "pixels, a multiple of 32 (default 1024)", width)
        .option("--height", "N", "pixels, a multiple of 32 (default 1024)", height)
        .option("--steps", "N", "denoising steps (default 40)", steps)
        .option("--seed", "N", "the starting noise's seed (default 0)", seed)
        .option("--count", "N", "images, seeds counting up from --seed, out-N.png after the first (default 1)", count)
        .flag("--rgb", "leave the alpha channel out", rgb)
        .option("--lora", "FILE[:SCALE]", "merge a LoRA (repeatable)",
                [&](const std::string& v) { loras.push_back(mako::image::parse_lora(v)); })
        .option("--terminal", "F", "the schedule's last noise level (0: none)",
                [&](const std::string& v) { terminal = mako::app::Cli::parse<float>(v); });
    const auto pos = cli.parse(argc, argv);
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    try {
        const auto t_load = Clock::now();
        mako::mtl::Context ctx;
        mako::image::Engine engine(pos[0], ctx);
        for (const auto& [path, scale] : loras)
            engine.add_lora(path, scale);
        mako::log::info("loaded in {:.1f} s", ms_since(t_load) / 1e3);
        for (uint32_t n = 0; n < count; ++n) {
            mako::image::Engine::Request r;
            r.prompt = pos[1];
            r.width = width;
            r.height = height;
            r.steps = steps;
            r.seed = seed + n;
            r.terminal = terminal;
            auto t_step = Clock::now();
            r.progress = [&](uint32_t i, uint32_t total) {
                mako::log::info("step {} of {}: {:.0f} ms, thermal {}", i, total, ms_since(t_step),
                                mako::core::thermal_state());
                t_step = Clock::now();
            };
            const auto t0 = Clock::now();
            const auto img = engine.generate(r);
            const double gen = ms_since(t0);
            std::string path = out;
            if (n > 0) {
                const size_t dot = out.rfind('.');
                path = dot == std::string::npos ? std::format("{}-{}", out, n)
                                                : std::format("{}-{}{}", out.substr(0, dot), n, out.substr(dot));
            }
            mako::image::write_file(path, mako::image::encode_png(mako::image::to_rgba8(img), width, height, rgb));
            mako::log::info("{}x{}, {} steps in {:.1f} s: {}", width, height, steps, gen / 1e3, path);
        }
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    return 0;
}
