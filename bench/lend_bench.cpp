// Can part of an mlock'ed, no-copy-wrapped arena give its pages back while the Metal buffer lives, and take them
// again later? Prints phys_footprint at each step and checks the GPU sees the refilled contents.
//   lend_bench [total_gb] [lend_gb] [reusable|remap|release]
#include "mtl/context.hpp"

#include <mach/mach.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <print>
#include <string>

namespace {

double footprint_gb()
{
    task_vm_info_data_t vi{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vi), &count);
    return vi.phys_footprint / 1e9;
}

// System-wide free pages: footprint alone could miss pages someone else (the GPU driver) still holds.
double free_gb()
{
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &count);
    return double(vs.free_count) * vm_page_size / 1e9;
}

} // namespace

int main(int argc, char** argv)
{
    const uint64_t total = uint64_t((argc > 1 ? std::stod(argv[1]) : 4.0) * 1e9) / 16384 * 16384;
    const uint64_t lend = uint64_t((argc > 2 ? std::stod(argv[2]) : 2.0) * 1e9) / 16384 * 16384;
    mako::mtl::Context ctx;
    auto* mem = static_cast<uint8_t*>(::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
    std::println("start                {:6.2f} GB", footprint_gb());
    ::mlock(mem, total);
    std::memset(mem, 1, total);
    const std::string mode = argc > 3 ? argv[3] : "reusable";
    const bool remap = mode == "remap", release = mode == "release";
    // release: the lent tail is a Metal buffer of its own, released before its pages are unmapped.
    MTL::Buffer* buf = ctx.device()->newBuffer(mem, release ? total - lend : total, MTL::ResourceStorageModeShared, nullptr);
    MTL::Buffer* tail_buf = release ? ctx.device()->newBuffer(mem + total - lend, lend, MTL::ResourceStorageModeShared, nullptr) : nullptr;
    MTL::Buffer* out = ctx.device()->newBuffer(16384, MTL::ResourceStorageModeShared);
    auto gpu_read = [&](uint64_t off) {
        auto* cb = ctx.queue()->commandBuffer();
        auto* bl = cb->blitCommandEncoder();
        if (tail_buf && off >= total - lend)
            bl->copyFromBuffer(tail_buf, off - (total - lend), out, 0, 16384);
        else
            bl->copyFromBuffer(buf, off, out, 0, 16384);
        bl->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        return static_cast<uint8_t*>(out->contents())[100];
    };
    std::println("locked, wrapped      {:6.2f} GB  (GPU reads {})", footprint_gb(), gpu_read(total - lend));
    uint8_t* tail = mem + total - lend;
    const double free_before = free_gb();
    ::munlock(tail, lend);
    int r = 0;
    if (release) {
        tail_buf->release();
        tail_buf = nullptr;
        r = ::munmap(tail, lend);
    } else if (remap) {
        r = ::mmap(tail, lend, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0) == tail ? 0 : -1;
    } else {
        r = ::madvise(tail, lend, MADV_FREE_REUSABLE);
    }
    for (int i = 0; i < 6; ++i) {
        std::println("  +{:.1f} s: system free {:.2f} GB, footprint {:.2f} GB", i * 0.5, free_gb(), footprint_gb());
        ::usleep(500000);
    }
    std::println("lent {:.2f} GB         {:6.2f} GB  (madvise {}; system free {:.2f} GB before, {:.2f} GB after)", lend / 1e9,
                 footprint_gb(), r, free_before, free_gb());
    auto* other = static_cast<uint8_t*>(::mmap(nullptr, lend, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
    std::memset(other, 2, lend);
    std::println("scratch allocated    {:6.2f} GB", footprint_gb());
    ::munmap(other, lend);
    std::println("scratch freed        {:6.2f} GB", footprint_gb());
    if (release) {
        tail = static_cast<uint8_t*>(::mmap(nullptr, lend, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
        tail_buf = ctx.device()->newBuffer(tail, lend, MTL::ResourceStorageModeShared, nullptr);
    } else if (!remap) {
        ::madvise(tail, lend, MADV_FREE_REUSE);
    }
    ::mlock(tail, lend);
    std::println("taken back           {:6.2f} GB  (old byte {})", footprint_gb(), tail[0]);
    std::memset(tail, 3, lend);
    std::println("refilled             {:6.2f} GB  (GPU reads {}, head {})", footprint_gb(), gpu_read(total - lend),
                 gpu_read(0));
    buf->release();
    if (tail_buf)
        tail_buf->release();
    out->release();
    return 0;
}
