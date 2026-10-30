// Text to image with Qwen-Image 2.1 from its original files.
//   mako_image <model dir> <prompt> [-o out.png] [--width N] [--height N] [--steps N] [--seed N] [--rgb]
#include "app/cli.hpp"
#include "core/log.hpp"
#include "image/engine.hpp"
#include "image/png.hpp"
#include "mtl/context.hpp"

#include <chrono>
#include <print>

int main(int argc, char** argv)
{
    std::string out = "image.png";
    uint32_t width = 1024, height = 1024, steps = 40;
    uint64_t seed = 0;
    bool rgb = false;
    mako::app::Cli cli("mako_image", "<model dir> <prompt>", 2, 2);
    cli.option("-o", "FILE", "the PNG to write (default image.png)", out)
        .option("--width", "N", "pixels, a multiple of 32 (default 1024)", width)
        .option("--height", "N", "pixels, a multiple of 32 (default 1024)", height)
        .option("--steps", "N", "denoising steps (default 40)", steps)
        .option("--seed", "N", "the starting noise's seed (default 0)", seed)
        .flag("--rgb", "leave the alpha channel out", rgb);
    const auto pos = cli.parse(argc, argv);
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    try {
        const auto t_load = Clock::now();
        mako::mtl::Context ctx;
        mako::image::Engine engine(pos[0], ctx);
        mako::log::info("loaded in {:.1f} s", ms_since(t_load) / 1e3);
        mako::image::Engine::Request r;
        r.prompt = pos[1];
        r.width = width;
        r.height = height;
        r.steps = steps;
        r.seed = seed;
        auto t_step = Clock::now();
        r.progress = [&](uint32_t i, uint32_t n) {
            mako::log::info("step {} of {}: {:.0f} ms", i, n, ms_since(t_step));
            t_step = Clock::now();
        };
        const auto t0 = Clock::now();
        const auto img = engine.generate(r);
        const double gen = ms_since(t0);
        const auto png = mako::image::encode_png(mako::image::to_rgba8(img), width, height, rgb);
        mako::image::write_file(out, png);
        mako::log::info("{}x{}, {} steps in {:.1f} s: {}", width, height, steps, gen / 1e3, out);
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    return 0;
}
