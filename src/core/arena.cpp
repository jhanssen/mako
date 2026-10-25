#include "core/arena.hpp"

#include "gguf/read_pool.hpp"

#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <exception>
#include <cstdio>
#include <cstring>
#include <format>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace mako::core {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kPage = 16384;
constexpr int kReaders = 8;              // reads in flight: the SSD's plateau needs 4-8 at 1 MiB
constexpr uint64_t kReadChunk = 1 << 20; // a miss's parts are read in chunks of this size

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// mlock in bounded pieces: one call over tens of GB is long, uninterruptible VM work and fails without saying
// where. On failure the locked part is unlocked again; returns errno or 0.
int lock_chunked(void* mem, uint64_t size)
{
    constexpr uint64_t kPiece = 256ull << 20;
    auto* p = static_cast<char*>(mem);
    for (uint64_t off = 0; off < size; off += kPiece) {
        if (::mlock(p + off, std::min(kPiece, size - off)) != 0) {
            const int err = errno;
            if (off)
                ::munlock(p, off);
            return err;
        }
    }
    return 0;
}

uint64_t sysctl_u64(const char* name)
{
    uint64_t v = 0; // a 4-byte value fills the low half
    size_t len = sizeof(v);
    if (::sysctlbyname(name, &v, &len, nullptr, 0) != 0)
        return 0;
    return v;
}

} // namespace

// Persistent reader threads; run() hands them tasks 0..n-1 and returns when all are done.
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

LockedMemory::LockedMemory(uint64_t bytes)
{
    if (!bytes)
        return;
    void* mem = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (mem == MAP_FAILED)
        throw std::runtime_error(std::format("mmap {:.1f} GB: {}", bytes / 1e9, std::strerror(errno)));
    if (const int err = lock_chunked(mem, bytes)) {
        ::munmap(mem, bytes);
        throw std::runtime_error(std::format("locking {:.1f} GB failed: {}", bytes / 1e9, std::strerror(err)));
    }
    mem_ = mem;
    bytes_ = bytes;
}

LockedMemory::~LockedMemory()
{
    if (mem_) {
        ::munlock(mem_, bytes_);
        ::munmap(mem_, bytes_);
    }
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
                         uint32_t n_used, double budget_bytes, const std::vector<std::vector<uint64_t>>& counts,
                         double lendable_bytes)
    : layers_(std::move(layers)), regions_(layers_.size()), n_expert_(n_expert), dev_(dev),
      readers_(std::make_unique<gguf::ReadPool>(kReaders))
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
    // The profile only seeds the scores (a tenth of a use per count), so the session's recent routing soon
    // outweighs it.
    score_.assign(nl, std::vector<double>(n_expert, 0.0));
    for (size_t l = 0; l < nl && l < counts.size(); ++l)
        if (counts[l].size() == n_expert)
            for (uint32_t e = 0; e < n_expert; ++e)
                score_[l][e] = 0.1 * double(counts[l][e]);
    const bool profiled = std::ranges::any_of(counts, [&](const auto& c) { return c.size() == n_expert; });
    for (size_t l = 0; l < nl; ++l) {
        if (!slot_bytes[l])
            continue;
        order_[l].resize(n_expert);
        std::iota(order_[l].begin(), order_[l].end(), 0);
        if (l < counts.size() && counts[l].size() == n_expert)
            std::ranges::stable_sort(order_[l], [&](int a, int b) { return counts[l][a] > counts[l][b]; });
    }
    // Every MoE layer keeps n_used slots plus its share of the lendable ones.
    const uint32_t want_lend = uint32_t(std::ceil(std::max(0.0, lendable_bytes) / double(per_slot)));
    const uint32_t floor_slots = std::min(n_expert, n_used + want_lend);
    if (!profiled) {
        const uint32_t s = uint32_t(std::min<uint64_t>(n_expert, budget / per_slot));
        for (size_t l = 0; l < nl; ++l)
            slots_[l] = slot_bytes[l] ? s : 0;
    } else {
        // Every MoE layer gets floor_slots, then the rest go to (layer, rank) by count per byte; a layer's ranks
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
            slots_[l] = floor_slots;
            used += uint64_t(floor_slots) * slot_bytes[l];
            for (uint32_t r = floor_slots; r < n_expert; ++r) {
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

    // The last `lend` slots of every MoE layer are lendable, as long as n_used others remain.
    uint32_t lend = want_lend;
    for (size_t l = 0; l < nl; ++l)
        if (slot_bytes[l])
            lend = std::min(lend, slots_[l] - n_used);

    // Regions in order, each inside one chunk; a chunk is at most one Metal buffer.
    const uint64_t limit = max_buffer / kPage * kPage;
    std::vector<uint64_t> chunk_sizes{0};
    tails_.resize(layers_.size());
    std::vector<std::array<std::pair<size_t, uint64_t>, 3>> place(layers_.size()); // (chunk, offset)
    for (size_t l = 0; l < layers_.size(); ++l)
        for (int p = 0; p < 3; ++p) {
            const uint64_t stride = round_up(layers_[l][p].bytes, kPage);
            uint32_t main_slots = 0;
            if (slot_bytes[l]) {
                main_slots = slots_[l] - lend;
                regions_[l][p].split = lend ? main_slots : UINT32_MAX; // UINT32_MAX: no lendable slots
                regions_[l][p].tail_off = tails_[l].bytes;
                tails_[l].bytes += stride * lend;
                tail_bytes_ += stride * lend;
            }
            const uint64_t size = stride * main_slots;
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
        if (const int err = lock_chunked(mem, size))
            throw std::runtime_error(std::format("locking {:.1f} GB of expert arena failed ({}); try a smaller "
                                                 "--arena-gb",
                                                 size / 1e9, std::strerror(err)));
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
    for (uint32_t l = 0; l < layers_.size(); ++l)
        if (tails_[l].bytes && !map_tail(l))
            throw std::runtime_error("locking the lendable expert slots failed");
    bytes_ += tail_bytes_;

    table_buf_ = mtl::Ref<MTL::Buffer>(
        dev->newBuffer(std::max<uint64_t>(layers_.size() * n_expert_ * 4, 16), MTL::ResourceStorageModeShared));

    std::fill_n(static_cast<int32_t*>(table_buf_->contents()), layers_.size() * n_expert_, -1);
    slot_expert_.resize(nl);
    last_use_.resize(nl);
    pinned_.resize(nl);
    for (size_t l = 0; l < nl; ++l) {
        slot_expert_[l].assign(slots_[l], -1);
        last_use_[l].assign(slots_[l], 0);
        pinned_[l].assign(slots_[l], 0);
    }
}

ExpertArena::~ExpertArena()
{
    bufs_.clear();
    for (Tail& t : tails_) {
        t.buf = {};
        if (t.mem) {
            ::munlock(t.mem, t.bytes);
            ::munmap(t.mem, t.bytes);
        }
    }
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
        const auto [buf, off] = r.at(uint32_t(s));
        gguf::read_exact(p.fd, static_cast<char*>(buf->contents()) + off + c.off, c.bytes, p.file_off + uint64_t(e) * p.bytes + c.off);
    });
    stats_.read_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    for (const Part& p : L)
        stats_.read_bytes += loads.size() * p.bytes;
}

void ExpertArena::note_prefill(uint32_t layer, std::span<const uint32_t> users)
{
    for (size_t e = 0; e < users.size() && e < usage_[layer].size(); ++e)
        usage_[layer][e] += users[e];
}

void ExpertArena::unpin(uint32_t layer) { std::ranges::fill(pinned_[layer], 0); }

void ExpertArena::unpin(uint32_t layer, std::span<const int32_t> slots)
{
    for (int32_t s : slots)
        pinned_[layer][s] = 0;
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

bool ExpertArena::ensure(uint32_t layer, std::span<const int> experts, int32_t* slots_out, bool pin, bool count_usage)
{
    auto* table = static_cast<int32_t*>(table_buf_->contents()) + uint64_t(layer) * n_expert_;
    auto& owner = slot_expert_[layer];
    auto& last = last_use_[layer];
    const uint32_t n_slots = slots_[layer];
    std::vector<char> taken(pinned_[layer]);
    if (tails_[layer].bytes && !tails_[layer].mem)
        std::fill(taken.begin() + regions_[layer][0].split, taken.end(), 1);
    size_t misses = 0;
    for (int e : experts) {
        if (table[e] >= 0)
            taken[table[e]] = 1;
        else
            ++misses;
    }
    if (size_t(std::ranges::count(taken, 0)) < misses)
        return false;
    ++clock_;
    // Each use adds 1; scores halve every 256 calls, so usage long past counts for little. (Measured on GLM-5.3:
    // leaving prefill out of the scores, or one update per prefill group, raised steady decode misses ~20%.)
    static const double decay = std::exp2(-1.0 / 256);
    for (double& c : score_[layer])
        c *= decay;
    for (int e : experts) {
        if (count_usage)
            ++usage_[layer][e];
        score_[layer][e] += 1.0;
    }
    std::vector<size_t> missed;
    for (size_t i = 0; i < experts.size(); ++i) {
        const int32_t s = table[experts[i]];
        if (s >= 0) {
            slots_out[i] = s;
            last[s] = clock_;
            ++stats_.hits;
        } else {
            missed.push_back(i);
        }
    }
    if (pin)
        for (size_t i = 0; i < experts.size(); ++i)
            if (table[experts[i]] >= 0)
                pinned_[layer][table[experts[i]]] = 1;
    if (missed.empty())
        return true;
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
            const double us = score_[layer][owner[s]], uv = score_[layer][owner[victim]];
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
        if (pin)
            pinned_[layer][victim] = 1;
        loads.emplace_back(experts[i], victim);
        ++stats_.misses;
    }
    read_into(layer, loads);
    return true;
}

std::vector<MTL::Buffer*> ExpertArena::tail_buffers() const
{
    std::vector<MTL::Buffer*> v;
    for (const Tail& t : tails_)
        if (t.buf)
            v.push_back(t.buf.get());
    return v;
}

bool ExpertArena::lent() const
{
    return std::ranges::any_of(tails_, [](const Tail& t) { return t.bytes && !t.mem; });
}

void ExpertArena::lend(uint64_t bytes)
{
    uint64_t lent = 0;
    for (const Tail& t : tails_)
        if (t.bytes && !t.mem)
            lent += t.bytes;
    auto* table = static_cast<int32_t*>(table_buf_->contents());
    for (uint32_t l = uint32_t(layers_.size()); l-- > 0 && lent < bytes;) {
        Tail& t = tails_[l];
        if (!t.mem)
            continue;
        for (uint32_t s = regions_[l][0].split; s < slots_[l]; ++s) {
            if (pinned_[l][s])
                throw std::runtime_error("lending a pinned expert slot");
            if (const int e = slot_expert_[l][s]; e >= 0) {
                table[l * n_expert_ + e] = -1;
                slot_expert_[l][s] = -1;
            }
        }
        for (Region& r : regions_[l])
            r.tail = nullptr;
        // The buffer goes first: Metal keeps the pages it wraps until then (the system gets them back shortly after).
        t.buf = {};
        ::munlock(t.mem, t.bytes);
        ::munmap(t.mem, t.bytes);
        t.mem = nullptr;
        lent += t.bytes;
    }
}

bool ExpertArena::map_tail(uint32_t layer)
{
    Tail& t = tails_[layer];
    void* mem = ::mmap(nullptr, t.bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    const int err = mem == MAP_FAILED ? errno : lock_chunked(mem, t.bytes);
    if (err) {
        if (mem != MAP_FAILED)
            ::munmap(mem, t.bytes);
        return false;
    }
    t.mem = mem;
    t.buf = mtl::Ref<MTL::Buffer>(dev_->newBuffer(mem, t.bytes, MTL::ResourceStorageModeShared, nullptr));
    if (!t.buf)
        throw std::runtime_error("wrapping the lendable expert slots in a Metal buffer failed");
    for (Region& r : regions_[layer])
        if (r.stride)
            r.tail = t.buf.get();
    return true;
}

bool ExpertArena::reclaim()
{
    if (!lent())
        return true;
    bool all = true;
    for (uint32_t l = 0; l < layers_.size(); ++l)
        if (tails_[l].bytes && !tails_[l].mem && !map_tail(l)) {
            // They can stay lent (unused) and come back later.
            std::fprintf(stderr, "mako: taking back %.1f GB of lent expert slots failed; running without them\n",
                         tails_[l].bytes / 1e9);
            all = false;
            break;
        }
    // Free slots take the best-scoring experts not resident (the prefill may have loaded some lend() evicted
    // elsewhere, displacing others).
    auto* table = static_cast<int32_t*>(table_buf_->contents());
    std::vector<std::pair<int, int32_t>> loads;
    std::vector<int> missing;
    for (uint32_t l = 0; l < layers_.size(); ++l) {
        if (!layers_[l][0].bytes)
            continue;
        const uint32_t usable = tails_[l].bytes && !tails_[l].mem ? regions_[l][0].split : slots_[l];
        missing.clear();
        for (int e = 0; e < int(n_expert_); ++e)
            if (table[l * n_expert_ + e] < 0)
                missing.push_back(e);
        std::ranges::stable_sort(missing, [&](int x, int y) { return score_[l][x] > score_[l][y]; });
        loads.clear();
        size_t next = 0;
        for (uint32_t s = 0; s < usable && next < missing.size(); ++s) {
            if (slot_expert_[l][s] >= 0)
                continue;
            const int e = missing[next++];
            slot_expert_[l][s] = e;
            table[l * n_expert_ + e] = int32_t(s);
            loads.emplace_back(e, int32_t(s));
        }
        if (!loads.empty())
            read_into(l, loads);
    }
    return all;
}

} // namespace mako::core
