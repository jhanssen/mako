// The threadgroup-size guard. A dispatch whose threadgroup has more threads than the pipeline's
// maxTotalThreadsPerThreadgroup is not run by Metal and raises no error: its outputs stay stale (another engine saw
// 448 for a 512-thread kernel on an M2 Max, and every answer was garbage). Checks that the kernels mako dispatches
// with 1024 threads can take them on this GPU (they declare it), and that gpu::Runtime::check_fit refuses a dispatch
// past the limit and names the kernel (MAKO_THREADGROUP_CAP lowers the limit it sees).
//   tg_limit_test
#include "gpu/runtime.hpp"
#include "kernels/args.h"
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <cstdlib>
#include <print>
#include <stdexcept>
#include <string>

namespace {

struct Probe : mako::gpu::Runtime {
    explicit Probe(mako::mtl::Context& c) : Runtime(c, mako::gpu::Telemetry::off) {}
    // One rms_norm of 16 floats with `threads` a threadgroup, never committed.
    void try_dispatch(uint32_t threads)
    {
        MTL::Buffer* b = scratch("probe", 64);
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        const mako::kernels::NormArgs args{16, 0, 0.0f, 0};
        try {
            dispatch(enc, "rms_norm", {b, b, b}, &args, sizeof(args), MTL::Size(1, 1, 1), MTL::Size(threads, 1, 1),
                     false);
        } catch (...) {
            enc->endEncoding();
            throw;
        }
        enc->endEncoding();
    }
};

int failed = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "ok  " : "FAIL", what);
    if (!ok)
        ++failed;
}

} // namespace

int main()
{
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    for (const char* k : {"hc_step_b", "ds_topk", "ds_draft_pick"}) {
        auto p = ctx.pipeline(lib.get(), k);
        check(p->maxTotalThreadsPerThreadgroup() >= 1024,
              std::format("{}: {} threads a threadgroup (dispatched with 1024)", k, p->maxTotalThreadsPerThreadgroup()));
    }
    {
        Probe r(ctx);
        bool threw = false;
        try {
            r.try_dispatch(256);
        } catch (const std::exception& e) {
            threw = true;
            std::println("     {}", e.what());
        }
        check(!threw, "a dispatch within the limit passes");
    }
    setenv("MAKO_THREADGROUP_CAP", "512", 1);
    {
        Probe r(ctx);
        std::string msg;
        try {
            r.try_dispatch(1024);
        } catch (const std::exception& e) {
            msg = e.what();
        }
        check(msg.find("rms_norm") != std::string::npos && msg.find("1024") != std::string::npos,
              "a dispatch past the limit is refused, naming the kernel: " + msg);
    }
    std::println("{}", failed ? "FAILED" : "all passed");
    return failed ? 1 : 0;
}
