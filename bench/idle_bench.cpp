// The first GPU work after the engine sat idle: measured ~350 ms on an M5 Max after 3 s or more (the driver's cost of
// the first pass over the expert arena's buffer again, docs/performance.md). Times a decode step after `--idle`
// seconds of rest without a keepalive and with the runtime's (gpu::Runtime::set_keepalive) at several strides, to
// see what keeps the mappings alive. An empty kernel a second did not (measured here: 279-305 ms against 290-327).
//   idle_bench <GLM-5.3 .gguf> [--idle S] [--reps N]
#include "app/cli.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"

#include <chrono>
#include <format>
#include <print>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

} // namespace

int main(int argc, char** argv)
{
    double idle = 10;
    uint32_t reps = 2;
    const auto args = mako::app::Cli("idle_bench", "<model.gguf>", 1, 1)
                          .option("--idle", "S", "seconds of rest before the timed step (default 10)", idle)
                          .option("--reps", "N", "timed steps per mode (default 2)", reps)
                          .parse(argc, argv);
    auto model = mako::gguf::ModelFiles::open(args[0]);
    if (!model) {
        std::println(stderr, "{}", model.error());
        return 1;
    }
    auto cfg = mako::glm::Config::load(*model);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::mtl::Context ctx;
    mako::glm::Options opts;
    opts.max_ctx = 4096;
    mako::glm::Engine engine(*model, *cfg, opts, ctx);
    std::println("arena {:.1f} GB; idle {} s", engine.arena().bytes() / 1e9, idle);
    const std::vector<int32_t> prompt{785, 3840, 315, 279, 12745, 20406, 44038, 1657, 23566, 11, 504, 279};
    engine.prefill(prompt);
    int32_t tok = 13;
    auto step = [&] {
        const auto t0 = Clock::now();
        engine.step(tok);
        tok = tok == 13 ? 11 : 13;
        return ms_since(t0);
    };
    for (int i = 0; i < 4; ++i)
        step();
    std::println("warm step: {:.1f} ms", step());
    // 0: no keepalive; else the runtime's, reading a word per this many bytes of the weights and the arena.
    for (const uint32_t stride : {0u, 1u << 20, 1u << 16, 1u << 14, 0u}) {
        engine.set_keepalive(stride != 0, std::chrono::milliseconds(1000), stride ? stride : 1);
        for (uint32_t r = 0; r < reps; ++r) {
            step();
            std::this_thread::sleep_for(std::chrono::duration<double>(idle));
            const double first = step(), second = step();
            std::println("{:>14}: after {} s idle the step took {:.1f} ms, the next {:.1f} ms",
                         stride ? std::format("stride {} KiB", stride >> 10) : "none", idle, first, second);
        }
    }
    engine.set_keepalive(false);
    return 0;
}
