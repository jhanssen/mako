#include "glm/arena.hpp"

#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <format>
#include <stdexcept>
#include <thread>

namespace mako::glm {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kPage = 16384;
constexpr int kReaders = 8; // enough reads in flight for the SSD's plateau at 2 MiB blocks

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

uint64_t sysctl_u64(const char* name)
{
    uint64_t v = 0; // a 4-byte value fills the low half
    size_t len = sizeof(v);
    if (::sysctlbyname(name, &v, &len, nullptr, 0) != 0)
        return 0;
    return v;
}

void read_exact(int fd, void* dst, uint64_t bytes, uint64_t offset)
{
    auto* p = static_cast<char*>(dst);
    while (bytes) {
        const ssize_t n = ::pread(fd, p, bytes, static_cast<off_t>(offset));
        if (n <= 0)
            throw std::runtime_error(std::format("pread at {}: {}", offset, n < 0 ? std::strerror(errno) : "EOF"));
        p += n;
        bytes -= static_cast<uint64_t>(n);
        offset += static_cast<uint64_t>(n);
    }
}

// Runs task(i) for i in [0, n) on up to kReaders threads.
template <typename F>
void parallel(size_t n, F&& task)
{
    std::atomic<size_t> next{0};
    std::vector<std::jthread> workers;
    for (size_t t = 0; t < std::min<size_t>(n, kReaders); ++t)
        workers.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < n;)
                task(i);
        });
}

} // namespace

MemInfo query_memory(uint64_t working_set)
{
    MemInfo m;
    m.phys = double(sysctl_u64("hw.memsize"));
    m.working_set = double(working_set);
    const uint64_t page = sysctl_u64("vm.pagesize");
    // The unit of vm_page_filecache_min is not documented; reading it as pages of vm.pagesize gives the larger floor.
    m.filecache_min = double(sysctl_u64("vm.vm_page_filecache_min") * page);

    task_vm_info_data_t vi{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vi), &count) == KERN_SUCCESS)
        m.footprint = double(vi.phys_footprint);

    vm_statistics64_data_t vs{};
    count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &count) == KERN_SUCCESS) {
        const double clean = std::min<double>(double(vs.inactive_count) + vs.speculative_count, vs.external_page_count);
        m.available = (double(vs.free_count) + vs.purgeable_count + clean) * double(page);
    }
    return m;
}

double arena_budget(const MemInfo& m)
{
    constexpr double GB = 1e9;
    const double keep = std::max(7 * GB, 0.15 * m.phys);
    double leave = keep;
    if (m.phys >= 64 * GB)
        leave = std::max({leave, 0.40 * m.phys, m.filecache_min > 0 ? m.filecache_min + 12 * GB : 0.0});
    const double cap = std::min(m.working_set, m.phys - leave);
    const double gpu = m.working_set - m.footprint - 1 * GB;
    const double ram = m.available - keep;
    return std::max(1 * GB, std::min({gpu, ram, cap - m.footprint}));
}

ExpertArena::ExpertArena(MTL::Device* dev, uint64_t max_buffer, std::vector<LayerParts> layers, uint32_t n_expert,
                         uint32_t n_used, double budget_bytes)
    : layers_(std::move(layers)), regions_(layers_.size()), n_expert_(n_expert)
{
    uint64_t per_slot = 0;
    for (const auto& L : layers_)
        for (const Part& p : L)
            per_slot += round_up(p.bytes, kPage);
    if (per_slot == 0)
        throw std::runtime_error("no MoE layers");
    slots_ = uint32_t(std::min<uint64_t>(n_expert, uint64_t(std::max(0.0, budget_bytes)) / per_slot));
    if (slots_ < n_used)
        throw std::runtime_error(std::format("expert arena of {:.1f} GB holds {} slots per layer, fewer than the {} "
                                             "experts a token uses",
                                             budget_bytes / 1e9, slots_, n_used));

    // Regions in order, each inside one chunk; a chunk is at most one Metal buffer.
    const uint64_t limit = max_buffer / kPage * kPage;
    std::vector<uint64_t> chunk_sizes{0};
    std::vector<std::array<std::pair<size_t, uint64_t>, 3>> place(layers_.size()); // (chunk, offset)
    for (size_t l = 0; l < layers_.size(); ++l)
        for (int p = 0; p < 3; ++p) {
            const uint64_t stride = round_up(layers_[l][p].bytes, kPage);
            const uint64_t size = stride * slots_;
            if (size == 0)
                continue;
            if (size > limit)
                throw std::runtime_error("expert region larger than the largest Metal buffer");
            if (chunk_sizes.back() + size > limit)
                chunk_sizes.push_back(0);
            place[l][p] = {chunk_sizes.size() - 1, chunk_sizes.back()};
            regions_[l][p].stride = stride;
            chunk_sizes.back() += size;
        }

    for (uint64_t size : chunk_sizes) {
        void* mem = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (mem == MAP_FAILED)
            throw std::runtime_error(std::format("mmap {} bytes: {}", size, std::strerror(errno)));
        chunks_.emplace_back(mem, size);
        // Unwired, idle arena memory gets compressed under pressure; the GPU only wires it while it runs.
        if (::mlock(mem, size) != 0)
            throw std::runtime_error(std::format("mlock {} bytes: {}", size, std::strerror(errno)));
        mtl::Ref<MTL::Buffer> b(dev->newBuffer(mem, size, MTL::ResourceStorageModeShared, nullptr));
        if (!b)
            throw std::runtime_error("wrapping the expert arena in a Metal buffer failed");
        bufs_.push_back(std::move(b));
        bytes_ += size;
    }
    for (size_t l = 0; l < layers_.size(); ++l)
        for (int p = 0; p < 3; ++p)
            if (regions_[l][p].stride) {
                regions_[l][p].buf = bufs_[place[l][p].first].get();
                regions_[l][p].off = place[l][p].second;
            }

    table_buf_ = mtl::Ref<MTL::Buffer>(
        dev->newBuffer(std::max<uint64_t>(layers_.size() * n_expert_ * 4, 16), MTL::ResourceStorageModeShared));

    std::fill_n(static_cast<int32_t*>(table_buf_->contents()), layers_.size() * n_expert_, -1);
    slot_expert_.assign(layers_.size(), std::vector<int32_t>(slots_, -1));
    last_use_.assign(layers_.size(), std::vector<uint64_t>(slots_, 0));
}

ExpertArena::~ExpertArena()
{
    bufs_.clear();
    for (auto [mem, size] : chunks_) {
        ::munlock(mem, size);
        ::munmap(mem, size);
    }
}

void ExpertArena::read_into(uint32_t layer, std::span<const std::pair<int, int32_t>> loads)
{
    const auto t0 = Clock::now();
    const LayerParts& L = layers_[layer];
    parallel(loads.size() * 3, [&](size_t i) {
        const auto [e, s] = loads[i / 3];
        const Part& p = L[i % 3];
        const Region& r = regions_[layer][i % 3];
        read_exact(p.fd, static_cast<char*>(r.buf->contents()) + r.off + uint64_t(s) * r.stride, p.bytes,
                   p.file_off + uint64_t(e) * p.bytes);
    });
    stats_.read_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    for (const Part& p : L)
        stats_.read_bytes += loads.size() * p.bytes;
}

void ExpertArena::prefill()
{
    const auto t0 = Clock::now();
    auto* table = static_cast<int32_t*>(table_buf_->contents());
    std::vector<std::pair<int, int32_t>> loads;
    for (uint32_t l = 0; l < layers_.size(); ++l) {
        if (!layers_[l][0].bytes)
            continue;
        loads.clear();
        for (uint32_t s = 0; s < slots_; ++s) {
            loads.emplace_back(int(s), int32_t(s));
            slot_expert_[l][s] = int32_t(s);
            table[l * n_expert_ + s] = int32_t(s);
        }
        read_into(l, loads);
    }
    stats_.prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    stats_.read_ms = 0;
    stats_.read_bytes = 0;
}

void ExpertArena::ensure(uint32_t layer, std::span<const int> experts, int32_t* slots_out)
{
    auto* table = static_cast<int32_t*>(table_buf_->contents()) + uint64_t(layer) * n_expert_;
    auto& owner = slot_expert_[layer];
    auto& last = last_use_[layer];
    ++clock_;
    std::vector<char> taken(slots_, 0);
    std::vector<size_t> missed;
    for (size_t i = 0; i < experts.size(); ++i) {
        const int32_t s = table[experts[i]];
        if (s >= 0) {
            slots_out[i] = s;
            taken[s] = 1;
            last[s] = clock_;
            ++stats_.hits;
        } else {
            missed.push_back(i);
        }
    }
    if (missed.empty())
        return;
    std::vector<std::pair<int, int32_t>> loads;
    for (size_t i : missed) {
        int32_t victim = -1;
        for (uint32_t s = 0; s < slots_; ++s) {
            if (taken[s])
                continue;
            if (owner[s] < 0) {
                victim = int32_t(s);
                break;
            }
            if (victim < 0 || last[s] < last[victim])
                victim = int32_t(s);
        }
        if (owner[victim] >= 0)
            table[owner[victim]] = -1;
        owner[victim] = experts[i];
        table[experts[i]] = victim;
        taken[victim] = 1;
        last[victim] = clock_;
        slots_out[i] = victim;
        loads.emplace_back(experts[i], victim);
        ++stats_.misses;
    }
    read_into(layer, loads);
}

} // namespace mako::glm
