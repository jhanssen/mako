// SSD -> GPU-visible memory streaming (use a file made by `io_bench create`, larger than RAM).
//   stream_bench mtlio <file> [seconds]       MTLIOCommandQueue loads, swept over block size and loads in flight
//   stream_bench contend <file> [threads] [block_KiB] [GiB]
//       GPU read bandwidth alone, then while pread threads and then MTLIO stream into another buffer
// MTLIO loads populate the page cache, so every run reads each block of the file at most once: each
// measurement gets its own slice of the file, visits it in shuffled order and skips blocks already resident.
#include "mtl/context.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <format>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using mako::mtl::Ref;

double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

uint64_t file_size(const char* path)
{
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::println(stderr, "open {}: {}", path, std::strerror(errno));
        std::exit(1);
    }
    off_t n = ::lseek(fd, 0, SEEK_END);
    ::close(fd);
    return static_cast<uint64_t>(n);
}

Ref<MTL::IOFileHandle> io_handle(MTL::Device* dev, const char* path)
{
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    Ref<MTL::IOFileHandle> h(dev->newIOFileHandle(url, &err));
    if (!h) {
        std::println(stderr, "newIOFileHandle: {}", err ? err->localizedDescription()->utf8String() : "?");
        std::exit(1);
    }
    return h;
}

Ref<MTL::IOCommandQueue> io_queue(MTL::Device* dev)
{
    Ref<MTL::IOCommandQueueDescriptor> d(MTL::IOCommandQueueDescriptor::alloc()->init());
    d->setType(MTL::IOCommandQueueTypeConcurrent);
    d->setPriority(MTL::IOPriorityHigh);
    d->setMaxCommandsInFlight(64);
    NS::Error* err = nullptr;
    Ref<MTL::IOCommandQueue> q(dev->newIOCommandQueue(d.get(), &err));
    if (!q) {
        std::println(stderr, "newIOCommandQueue: {}", err ? err->localizedDescription()->utf8String() : "?");
        std::exit(1);
    }
    return q;
}

// Read-only mapping of the file, used only to ask which pages are in the page cache.
class Residency {
public:
    explicit Residency(const char* path)
    {
        int fd = ::open(path, O_RDONLY);
        size_ = static_cast<uint64_t>(::lseek(fd, 0, SEEK_END));
        base_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (base_ == MAP_FAILED) {
            std::println(stderr, "mmap: {}", std::strerror(errno));
            std::exit(1);
        }
    }
    ~Residency() { ::munmap(base_, size_); }
    // True when any page of [off, off + len) is resident.
    bool any(uint64_t off, uint64_t len) const
    {
        const uint64_t page = uint64_t(::getpagesize());
        std::vector<char> vec((len + page - 1) / page);
        if (::mincore(static_cast<char*>(base_) + off, len, vec.data()) != 0)
            return true;
        return std::ranges::any_of(vec, [](char c) { return c & MINCORE_INCORE; });
    }

private:
    void* base_ = nullptr;
    uint64_t size_ = 0;
};

// Block offsets of one slice of the file, each handed out once, in shuffled order, skipping blocks that
// already have pages in the page cache.
class Slice {
public:
    Slice(const Residency& res, uint64_t begin, uint64_t bytes, size_t block)
        : res_(res), block_(block), order_(bytes / block)
    {
        for (uint64_t i = 0; i < order_.size(); ++i)
            order_[i] = begin + i * block;
        std::ranges::shuffle(order_, std::mt19937_64(begin ^ block));
    }
    // Next unread, non-resident offset, or nothing when the slice is used up.
    std::optional<uint64_t> next()
    {
        for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= order_.size()) {
                exhausted_ = true;
                return std::nullopt;
            }
            if (!res_.any(order_[i], block_))
                return order_[i];
            skipped_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    bool exhausted() const { return exhausted_; }
    uint64_t skipped() const { return skipped_; }

private:
    const Residency& res_;
    size_t block_;
    std::vector<uint64_t> order_;
    std::atomic<size_t> next_{0};
    std::atomic<uint64_t> skipped_{0};
    std::atomic<bool> exhausted_{false};
};

// Keeps `inflight` single-load command buffers outstanding until `stop` or the slice runs out; returns bytes loaded.
uint64_t mtlio_stream(MTL::IOCommandQueue* q, MTL::IOFileHandle* h, MTL::Buffer* dst, Slice& slice, size_t block,
                      int inflight, const std::atomic<bool>& stop, std::vector<double>* lat_us = nullptr)
{
    const uint64_t slots = dst->length() / block;
    struct Pending {
        MTL::IOCommandBuffer* cb;
        Clock::time_point t;
    };
    std::deque<Pending> pending;
    uint64_t bytes = 0, slot = 0;
    auto issue = [&] {
        const auto off = slice.next();
        if (!off)
            return;
        auto* cb = q->commandBuffer();
        cb->retain();
        cb->loadBuffer(dst, (slot++ % slots) * block, block, h, *off);
        auto t = Clock::now();
        cb->commit();
        pending.push_back({cb, t});
    };
    for (int i = 0; i < inflight; ++i)
        issue();
    while (!pending.empty()) {
        Pending p = pending.front();
        pending.pop_front();
        p.cb->waitUntilCompleted();
        if (p.cb->status() != MTL::IOStatusComplete) {
            std::println(stderr, "MTLIO load failed: status {}", static_cast<int>(p.cb->status()));
            std::exit(1);
        }
        if (lat_us)
            lat_us->push_back(std::chrono::duration<double, std::micro>(Clock::now() - p.t).count());
        p.cb->release();
        bytes += block;
        if (!stop.load(std::memory_order_relaxed))
            issue();
    }
    return bytes;
}

int cmd_mtlio(const char* path, double seconds)
{
    mako::mtl::Context ctx;
    const uint64_t fsize = file_size(path);
    auto h = io_handle(ctx.device(), path);
    auto q = io_queue(ctx.device());
    Ref<MTL::Buffer> dst(ctx.device()->newBuffer(512ull << 20, MTL::ResourceStorageModeShared));
    const Residency res(path);
    const size_t kibs[] = {128, 512, 1024, 2048, 4096, 8192};
    const int inflights[] = {1, 2, 4, 8, 16, 32};
    const uint64_t slice_bytes = fsize / (std::size(kibs) * std::size(inflights)) / (8 << 20) * (8 << 20);
    uint64_t begin = 0;
    for (size_t kib : kibs) {
        for (int inflight : inflights) {
            Slice slice(res, begin, slice_bytes, kib << 10);
            begin += slice_bytes;
            std::atomic<bool> stop{false};
            std::vector<double> lat;
            auto t0 = Clock::now();
            std::jthread timer([&] {
                std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
                stop = true;
            });
            uint64_t bytes = mtlio_stream(q.get(), h.get(), dst.get(), slice, kib << 10, inflight, stop, &lat);
            double s = secs_since(t0);
            stop = true;
            std::ranges::sort(lat);
            std::println("mtlio inflight {:3}  block {:6} KiB  {:6.2f} GB/s  p50 {:8.0f} us  p99 {:8.0f} us{}{}",
                         inflight, kib, bytes / s / 1e9, lat[lat.size() / 2], lat[(lat.size() - 1) * 99 / 100],
                         slice.skipped() ? std::format("  ({} cached blocks skipped)", slice.skipped()) : "",
                         slice.exhausted() ? "  (slice used up early)" : "");
        }
    }
    return 0;
}

constexpr const char* kReadKernel = R"(
#include <metal_stdlib>
using namespace metal;

kernel void read_bw(device const uint4* src [[buffer(0)]], device uint* sink [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]])
{
    uint4 acc = 0;
    for (uint i = gid; i < n; i += grid)
        acc ^= src[i];
    if (acc.x == 0x12345678u && acc.y == 0x9abcdef0u)
        sink[0] = acc.z;
}
)";

int cmd_contend(const char* path, int threads, size_t block, uint64_t gib)
{
    mako::mtl::Context ctx;
    auto* dev = ctx.device();
    const uint64_t fsize = file_size(path);
    auto lib = ctx.compile(kReadKernel);
    auto pso = ctx.pipeline(lib.get(), "read_bw");
    const Residency res(path);

    const uint64_t src_bytes = gib << 30;
    Ref<MTL::Buffer> src(dev->newBuffer(src_bytes, MTL::ResourceStorageModeShared));
    std::memset(src->contents(), 1, src_bytes);
    Ref<MTL::Buffer> sink(dev->newBuffer(16, MTL::ResourceStorageModeShared));
    Ref<MTL::Buffer> landing(dev->newBuffer(512ull << 20, MTL::ResourceStorageModeShared));
    const uint32_t n = static_cast<uint32_t>(src_bytes / 16);

    auto gpu_pass = [&] {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso.get());
        enc->setBuffer(src.get(), 0, 0);
        enc->setBuffer(sink.get(), 0, 1);
        enc->setBytes(&n, sizeof(n), 2);
        enc->dispatchThreads(MTL::Size(1u << 20, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        double s = cb->GPUEndTime() - cb->GPUStartTime();
        pool->release();
        return s;
    };
    auto gpu_bw = [&](int passes) {
        double total = 0;
        for (int i = 0; i < passes; ++i)
            total += gpu_pass();
        return passes * static_cast<double>(src_bytes) / total / 1e9;
    };

    // ~2 s of GPU reads per measurement at ~550 GB/s.
    const int passes = std::max(20, int(1.1e12 / double(src_bytes)));
    gpu_bw(3);
    std::println("GPU read alone:        {:6.1f} GB/s", gpu_bw(passes));

    {
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> bytes{0};
        const uint64_t slots = landing->length() / block;
        Slice slice(res, 0, fsize / 2 / block * block, block);
        auto* base = static_cast<char*>(landing->contents());
        std::vector<std::jthread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                int fd = ::open(path, O_RDONLY);
                ::fcntl(fd, F_NOCACHE, 1);
                uint64_t i = t;
                while (!stop.load(std::memory_order_relaxed)) {
                    const auto off = slice.next();
                    if (!off)
                        break;
                    char* out = base + (i % slots) * block;
                    i += threads;
                    if (::pread(fd, out, block, static_cast<off_t>(*off)) == static_cast<ssize_t>(block))
                        bytes += block;
                }
                ::close(fd);
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        auto t0 = Clock::now();
        uint64_t b0 = bytes;
        double bw = gpu_bw(passes);
        double ssd = (bytes - b0) / secs_since(t0) / 1e9;
        stop = true;
        std::println("GPU read + pread:      {:6.1f} GB/s   (SSD {:.2f} GB/s, {} threads x {} KiB){}{}", bw, ssd,
                     threads, block >> 10,
                     slice.skipped() ? std::format("  ({} cached blocks skipped)", slice.skipped()) : "",
                     slice.exhausted() ? "  (slice used up early)" : "");
    }

    {
        auto h = io_handle(dev, path);
        auto q = io_queue(dev);
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> loaded{0};
        const uint64_t half = fsize / 2 / block * block;
        Slice slice(res, half, (fsize - half) / block * block, block);
        std::jthread io([&] { loaded = mtlio_stream(q.get(), h.get(), landing.get(), slice, block, threads, stop); });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        auto t0 = Clock::now();
        double bw = gpu_bw(passes);
        double s = secs_since(t0);
        stop = true;
        io.join();
        std::println("GPU read + MTLIO:      {:6.1f} GB/s   (SSD ~{:.2f} GB/s over the whole run){}{}", bw,
                     loaded / (s + 0.3) / 1e9,
                     slice.skipped() ? std::format("  ({} cached blocks skipped)", slice.skipped()) : "",
                     slice.exhausted() ? "  (slice used up early)" : "");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3) {
        std::println(stderr, "usage: stream_bench mtlio <file> [seconds] | contend <file> [threads] [block_KiB] [GiB]");
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "mtlio")
        return cmd_mtlio(argv[2], argc > 3 ? std::stod(argv[3]) : 2.0);
    if (cmd == "contend")
        return cmd_contend(argv[2], argc > 3 ? std::stoi(argv[3]) : 4, (argc > 4 ? std::stoull(argv[4]) : 2048) << 10,
                           argc > 5 ? std::stoull(argv[5]) : 4);
    std::println(stderr, "unknown command {}", cmd);
    return 2;
}
