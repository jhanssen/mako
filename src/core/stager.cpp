#include "core/stager.hpp"

#include "gguf/read_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace mako::core {

namespace {

constexpr int kReaders = 8;             // the SSD's plateau needs 4-8 reads of 1 MiB in flight
constexpr uint64_t kReadChunk = 1 << 20;
constexpr uint64_t kAlign = 16384;

} // namespace

Stager::Stager(MTL::Device* dev, const ExpertArena& arena, uint64_t bytes) : arena_(arena)
{
    const auto& usage = arena.usage();
    for (uint32_t l = 0; l < usage.size(); ++l) {
        uint64_t b = 0;
        for (int p = 0; p < 3; ++p)
            b += (arena.part(l, p).bytes + kAlign - 1) / kAlign * kAlign;
        slot_bytes_ = std::max(slot_bytes_, b);
    }
    if (slot_bytes_ == 0)
        return;
    slots_ = uint32_t(bytes / slot_bytes_);
    if (slots_ < 2) {
        slots_ = 0;
        return;
    }
    buf_ = mtl::Ref<MTL::Buffer>(dev->newBuffer(uint64_t(slots_) * slot_bytes_, MTL::ResourceStorageModeShared));
    if (!buf_)
        slots_ = 0;
}

Stager::~Stager()
{
    stop();
}

void Stager::stop()
{
    for (auto& t : threads_)
        t.request_stop();
    cv_.notify_all();
    threads_.clear(); // joins
    std::lock_guard lock(m_);
    jobs_.clear();
    index_.clear();
}

void Stager::start(std::vector<std::pair<uint32_t, std::vector<int>>> plan)
{
    stop();
    {
        std::lock_guard lock(m_);
        for (auto& [layer, experts] : plan)
            for (int e : experts) {
                index_[{layer, e}] = int64_t(jobs_.size());
                jobs_.push_back({layer, e});
            }
        ready_.assign(jobs_.size(), 0);
        released_.assign(jobs_.size(), 0);
        taken_.assign(jobs_.size(), 0);
        next_ = 0;
        released_upto_ = 0;
        error_ = nullptr;
        ++generation_;
        t_start_ = std::chrono::steady_clock::now();
    }
    if (!slots_ || jobs_.empty())
        return;
    for (int t = 0; t < kReaders; ++t)
        threads_.emplace_back([this](std::stop_token st) { work(st); });
}

void Stager::work(std::stop_token st)
{
    std::unique_lock lock(m_);
    for (;;) {
        if (st.stop_requested() || next_ >= jobs_.size())
            return;
        const size_t i = next_++;
        // Slot i % slots_ is free once job i - slots_ is released.
        cv_.wait(lock, st, [&] { return int64_t(i) < released_upto_ + int64_t(slots_); });
        if (st.stop_requested())
            return;
        const Job job = jobs_[i];
        lock.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        std::exception_ptr err;
        uint64_t n = 0;
        try {
            uint64_t off = uint64_t(i % slots_) * slot_bytes_;
            for (int p = 0; p < 3; ++p) {
                const ExpertArena::Part& part = arena_.part(job.layer, p);
                for (uint64_t c = 0; c < part.bytes; c += kReadChunk)
                    gguf::read_exact(part.fd, static_cast<char*>(buf_->contents()) + off + c,
                                     std::min(kReadChunk, part.bytes - c),
                                     part.file_off + uint64_t(job.expert) * part.bytes + c);
                off += (part.bytes + kAlign - 1) / kAlign * kAlign;
                n += part.bytes;
            }
        } catch (...) {
            err = std::current_exception();
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        lock.lock();
        read_ms_.store(read_ms_.load() + ms);
        read_bytes_ += n;
        span_ms_.store(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start_).count());
        if (err && !error_)
            error_ = err;
        ready_[i] = 1;
        cv_.notify_all();
    }
}

int64_t Stager::find(uint32_t layer, int e) const
{
    const auto it = index_.find({layer, e});
    return it == index_.end() ? -1 : it->second;
}

Stager::Where Stager::take(uint32_t layer, int e)
{
    std::unique_lock lock(m_);
    Where w;
    const int64_t i = find(layer, e);
    if (i < 0)
        return w;
    // Experts are taken in plan order: this layer's jobs before i not taken never will be, so their slots go now
    // (a layer can need more slots than the ring has).
    for (int64_t j = i - 1; j >= released_upto_ && jobs_[size_t(j)].layer == layer; --j)
        if (!taken_[size_t(j)])
            released_[size_t(j)] = 1;
    while (size_t(released_upto_) < released_.size() && released_[size_t(released_upto_)])
        ++released_upto_;
    cv_.notify_all();
    while (!cv_.wait_for(lock, std::chrono::seconds(5), [&] { return ready_[size_t(i)] || error_ || threads_.empty(); }))
        std::fprintf(stderr, "stager: waiting for job %lld (layer %u expert %d): next %zu, released up to %lld, %u slots\n",
                     (long long)i, layer, e, next_, (long long)released_upto_, slots_);
    if (error_)
        std::rethrow_exception(error_);
    if (!ready_[size_t(i)])
        throw std::logic_error("staged expert not read");
    taken_[size_t(i)] = 1;
    w.job = i;
    w.buf = buf_.get();
    uint64_t off = uint64_t(i % slots_) * slot_bytes_;
    for (int p = 0; p < 3; ++p) {
        w.off[p] = off;
        off += (arena_.part(layer, p).bytes + kAlign - 1) / kAlign * kAlign;
    }
    return w;
}

void Stager::release(int64_t job)
{
    std::lock_guard lock(m_);
    if (job < 0 || size_t(job) >= released_.size())
        return;
    released_[size_t(job)] = 1;
    while (size_t(released_upto_) < released_.size() && released_[size_t(released_upto_)])
        ++released_upto_;
    cv_.notify_all();
}

void Stager::release_untaken(uint32_t layer)
{
    std::lock_guard lock(m_);
    for (size_t i = 0; i < jobs_.size(); ++i)
        if (jobs_[i].layer == layer && !taken_[i])
            released_[i] = 1;
    while (size_t(released_upto_) < released_.size() && released_[size_t(released_upto_)])
        ++released_upto_;
    cv_.notify_all();
}

} // namespace mako::core
