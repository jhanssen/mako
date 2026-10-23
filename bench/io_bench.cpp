// SSD read throughput: pread thread pool, random block-aligned reads, F_NOCACHE.
//   io_bench create <file> <GiB>
//   io_bench read <file> [threads] [block_KiB] [seconds]
//   io_bench sweep <file> [seconds]
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int open_nocache(const char* path, int flags, int mode = 0)
{
    int fd = ::open(path, flags, mode);
    if (fd < 0) {
        std::println(stderr, "open {}: {}", path, std::strerror(errno));
        std::exit(1);
    }
    ::fcntl(fd, F_NOCACHE, 1);
    ::fcntl(fd, F_RDAHEAD, 0);
    return fd;
}

void* map_anon(size_t bytes)
{
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        std::println(stderr, "mmap {} bytes failed", bytes);
        std::exit(1);
    }
    std::memset(p, 0, bytes);
    return p;
}

int create(const char* path, uint64_t gib)
{
    int fd = open_nocache(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    constexpr size_t chunk = 64ull << 20;
    auto* buf = static_cast<uint64_t*>(map_anon(chunk));
    std::mt19937_64 rng(1);
    for (size_t i = 0; i < chunk / 8; ++i)
        buf[i] = rng();
    const uint64_t total = gib << 30;
    auto t0 = Clock::now();
    for (uint64_t off = 0; off < total; off += chunk) {
        buf[0] = off;
        if (::pwrite(fd, buf, chunk, static_cast<off_t>(off)) != static_cast<ssize_t>(chunk)) {
            std::println(stderr, "pwrite: {}", std::strerror(errno));
            return 1;
        }
    }
    ::fsync(fd);
    ::close(fd);
    double s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::println("wrote {} GiB in {:.1f} s ({:.2f} GB/s)", gib, s, total / s / 1e9);
    return 0;
}

struct Result {
    double gbps;
    double p50_us, p99_us;
    uint64_t reads;
};

Result run_read(const char* path, int threads, size_t block, double seconds)
{
    int probe = open_nocache(path, O_RDONLY);
    const off_t size = ::lseek(probe, 0, SEEK_END);
    ::close(probe);
    const uint64_t nblocks = static_cast<uint64_t>(size) / block;

    auto* dst = static_cast<char*>(map_anon(block * threads));
    std::atomic<bool> stop{false};
    std::vector<std::vector<float>> lat(threads);
    std::vector<uint64_t> bytes(threads, 0);

    auto worker = [&](int t) {
        int fd = open_nocache(path, O_RDONLY);
        std::mt19937_64 rng(1234 + t);
        std::uniform_int_distribution<uint64_t> pick(0, nblocks - 1);
        char* out = dst + block * t;
        lat[t].reserve(1 << 16);
        while (!stop.load(std::memory_order_relaxed)) {
            const off_t off = static_cast<off_t>(pick(rng) * block);
            auto a = Clock::now();
            ssize_t n = ::pread(fd, out, block, off);
            auto b = Clock::now();
            if (n != static_cast<ssize_t>(block)) {
                std::println(stderr, "pread: {}", n < 0 ? std::strerror(errno) : "short read");
                std::exit(1);
            }
            bytes[t] += block;
            lat[t].push_back(std::chrono::duration<float, std::micro>(b - a).count());
        }
        ::close(fd);
    };

    auto t0 = Clock::now();
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back(worker, t);
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop = true;
    }
    double s = std::chrono::duration<double>(Clock::now() - t0).count();

    std::vector<float> all;
    uint64_t total = 0;
    for (int t = 0; t < threads; ++t) {
        all.insert(all.end(), lat[t].begin(), lat[t].end());
        total += bytes[t];
    }
    std::ranges::sort(all);
    ::munmap(dst, block * threads);
    auto pct = [&](double p) { return all.empty() ? 0.0 : all[static_cast<size_t>(p * (all.size() - 1))]; };
    return {total / s / 1e9, pct(0.5), pct(0.99), all.size()};
}

void print_row(int threads, size_t block, const Result& r)
{
    std::println("threads {:3}  block {:6} KiB  {:6.2f} GB/s  p50 {:8.0f} us  p99 {:8.0f} us  reads {}",
                 threads, block >> 10, r.gbps, r.p50_us, r.p99_us, r.reads);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: io_bench create <file> <GiB> | read <file> [threads] [block_KiB] [seconds] | sweep <file> [seconds]");
        return 2;
    }
    const std::string cmd = argv[1];
    const char* path = argv[2];
    if (cmd == "create")
        return create(path, argc > 3 ? std::stoull(argv[3]) : 20);
    if (cmd == "read") {
        int threads = argc > 3 ? std::stoi(argv[3]) : 8;
        size_t block = (argc > 4 ? std::stoull(argv[4]) : 1024) << 10;
        double secs = argc > 5 ? std::stod(argv[5]) : 5.0;
        print_row(threads, block, run_read(path, threads, block, secs));
        return 0;
    }
    if (cmd == "sweep") {
        double secs = argc > 3 ? std::stod(argv[3]) : 3.0;
        for (size_t kib : {128, 512, 1024, 2048, 4096, 8192})
            for (int threads : {1, 2, 4, 8, 16, 32})
                print_row(threads, kib << 10, run_read(path, threads, kib << 10, secs));
        return 0;
    }
    std::println(stderr, "unknown command {}", cmd);
    return 2;
}
