#include "glm/arena.hpp"

#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <exception>
#include <cstring>
#include <format>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace mako::glm {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kPage = 16384;
constexpr int kReaders = 8;              // reads in flight: the SSD's plateau needs 4-8 at 1 MiB
constexpr uint64_t kReadChunk = 1 << 20; // a miss's parts are read in chunks of this size

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

} // namespace

// Persistent reader threads; run() hands them tasks 0..n-1 and returns when all are done.
class ExpertArena::Readers {
public:
    Readers()
    {
        for (int t = 0; t < kReaders; ++t)
            threads_.emplace_back([this](std::stop_token st) { work(st); });
    }
    ~Readers()
    {
        for (auto& t : threads_)
            t.request_stop();
        cv_.notify_all();
    }

    void run(size_t n, const std::function<void(size_t)>& task)
    {
        std::unique_lock lock(m_);
        task_ = &task;
        n_ = n;
        next_ = 0;
        done_ = 0;
        error_ = nullptr;
        ++generation_;
        cv_.notify_all();
        done_cv_.wait(lock, [&] { return done_ == n_; });
        task_ = nullptr;
        if (error_)
            std::rethrow_exception(error_);
    }

private:
    void work(std::stop_token st)
    {
        uint64_t seen = 0;
        std::unique_lock lock(m_);
        for (;;) {
            cv_.wait(lock, st, [&] { return generation_ != seen; });
            if (st.stop_requested())
                return;
            seen = generation_;
            while (task_ && next_ < n_) {
                const size_t i = next_++;
                lock.unlock();
                std::exception_ptr err;
                try {
                    (*task_)(i);
                } catch (...) {
                    err = std::current_exception();
                }
                lock.lock();
                if (err && !error_)
                    error_ = err;
                if (++done_ == n_)
                    done_cv_.notify_one();
            }
        }
    }

    std::mutex m_;
    std::condition_variable_any cv_, done_cv_;
    const std::function<void(size_t)>* task_ = nullptr;
    size_t n_ = 0, next_ = 0, done_ = 0;
    uint64_t generation_ = 0;
    std::exception_ptr error_;
    std::vector<std::jthread> threads_;
};

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
                         uint32_t n_used, double budget_bytes, const std::vector<std::vector<uint64_t>>& counts)
    : layers_(std::move(layers)), regions_(layers_.size()), n_expert_(n_expert), readers_(std::make_unique<Readers>())
{
    const size_t nl = layers_.size();
    const uint64_t budget = uint64_t(std::max(0.0, budget_bytes));
    std::vector<uint64_t> slot_bytes(nl, 0);
    uint64_t per_slot = 0;
    for (size_t l = 0; l < nl; ++l) {
        for (const Part& p : layers_[l])
            slot_bytes[l] += round_up(p.bytes, kPage);
        per_slot += slot_bytes[l];
    }
    if (per_slot == 0)
        throw std::runtime_error("no MoE layers");
    slots_.assign(nl, 0);
    order_.assign(nl, {});
    usage_.assign(nl, std::vector<uint64_t>(n_expert, 0));
    score_.assign(nl, std::vector<uint64_t>(n_expert, 0));
    for (size_t l = 0; l < nl && l < counts.size(); ++l)
        if (counts[l].size() == n_expert)
            score_[l] = counts[l];
    const bool profiled = std::ranges::any_of(counts, [&](const auto& c) { return c.size() == n_expert; });
    for (size_t l = 0; l < nl; ++l) {
        if (!slot_bytes[l])
            continue;
        order_[l].resize(n_expert);
        std::iota(order_[l].begin(), order_[l].end(), 0);
        if (l < counts.size() && counts[l].size() == n_expert)
            std::ranges::stable_sort(order_[l], [&](int a, int b) { return counts[l][a] > counts[l][b]; });
    }
    if (!profiled) {
        const uint32_t s = uint32_t(std::min<uint64_t>(n_expert, budget / per_slot));
        for (size_t l = 0; l < nl; ++l)
            slots_[l] = slot_bytes[l] ? s : 0;
    } else {
        // Every MoE layer gets n_used slots, then the rest go to (layer, rank) by count per byte; a layer's ranks
        // are taken in order, since its counts only fall with rank.
        uint64_t used = 0;
        struct Cand {
            double value;
            uint32_t layer, rank;
        };
        std::vector<Cand> cands;
        for (size_t l = 0; l < nl; ++l) {
            if (!slot_bytes[l])
                continue;
            slots_[l] = n_used;
            used += uint64_t(n_used) * slot_bytes[l];
            for (uint32_t r = n_used; r < n_expert; ++r) {
                const uint64_t c = l < counts.size() && counts[l].size() == n_expert ? counts[l][order_[l][r]] : 0;
                cands.push_back({double(c) / double(slot_bytes[l]), uint32_t(l), r});
            }
        }
        std::ranges::stable_sort(cands, [](const Cand& a, const Cand& b) { return a.value > b.value; });
        std::vector<char> blocked(nl, 0);
        for (const Cand& c : cands) {
            if (blocked[c.layer] || c.rank != slots_[c.layer])
                continue;
            if (used + slot_bytes[c.layer] > budget) {
                blocked[c.layer] = 1;
                continue;
            }
            used += slot_bytes[c.layer];
            ++slots_[c.layer];
        }
    }
    for (size_t l = 0; l < nl; ++l)
        if (slot_bytes[l] && slots_[l] < n_used)
            throw std::runtime_error(std::format("expert arena of {:.1f} GB holds {} slots in layer {}, fewer than the "
                                                 "{} experts a token uses",
                                                 budget_bytes / 1e9, slots_[l], l, n_used));

    // Regions in order, each inside one chunk; a chunk is at most one Metal buffer.
    const uint64_t limit = max_buffer / kPage * kPage;
    std::vector<uint64_t> chunk_sizes{0};
    std::vector<std::array<std::pair<size_t, uint64_t>, 3>> place(layers_.size()); // (chunk, offset)
    for (size_t l = 0; l < layers_.size(); ++l)
        for (int p = 0; p < 3; ++p) {
            const uint64_t stride = round_up(layers_[l][p].bytes, kPage);
            const uint64_t size = stride * slots_[l];
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
    slot_expert_.resize(nl);
    last_use_.resize(nl);
    for (size_t l = 0; l < nl; ++l) {
        slot_expert_[l].assign(slots_[l], -1);
        last_use_[l].assign(slots_[l], 0);
    }
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
    struct Chunk {
        uint32_t load;
        int part;
        uint64_t off, bytes;
    };
    std::vector<Chunk> chunks;
    for (uint32_t i = 0; i < loads.size(); ++i)
        for (int p = 0; p < 3; ++p)
            for (uint64_t off = 0; off < L[p].bytes; off += kReadChunk)
                chunks.push_back({i, p, off, std::min(kReadChunk, L[p].bytes - off)});
    readers_->run(chunks.size(), [&](size_t i) {
        const Chunk& c = chunks[i];
        const auto [e, s] = loads[c.load];
        const Part& p = L[c.part];
        const Region& r = regions_[layer][c.part];
        read_exact(p.fd, static_cast<char*>(r.buf->contents()) + r.off + uint64_t(s) * r.stride + c.off, c.bytes,
                   p.file_off + uint64_t(e) * p.bytes + c.off);
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
        for (uint32_t s = 0; s < slots_[l]; ++s) {
            const int e = order_[l][s];
            loads.emplace_back(e, int32_t(s));
            slot_expert_[l][s] = e;
            table[l * n_expert_ + e] = int32_t(s);
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
    const uint32_t n_slots = slots_[layer];
    for (int e : experts) {
        ++usage_[layer][e];
        ++score_[layer][e];
    }
    std::vector<char> taken(n_slots, 0);
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
        for (uint32_t s = 0; s < n_slots; ++s) {
            if (taken[s])
                continue;
            if (owner[s] < 0) {
                victim = int32_t(s);
                break;
            }
            if (victim < 0)
                victim = int32_t(s);
            const uint64_t us = score_[layer][owner[s]], uv = score_[layer][owner[victim]];
            if (us < uv || (us == uv && last[s] < last[victim]))
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
