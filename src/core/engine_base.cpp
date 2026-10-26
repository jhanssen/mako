#include "core/engine_base.hpp"

#include "gguf/read_pool.hpp"
#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"
#include "quant/quantize.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <sys/sysctl.h>

#include <algorithm>
#include <deque>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <stdexcept>
#include <thread>

namespace mako::core {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr uint64_t kAlign = 256;
constexpr uint64_t kChunkBytes = 1ull << 30;
constexpr NS::UInteger kSamplesPerBuffer = 4096; // the largest counter sample buffer is 32 KiB

struct NormArgs {
    uint32_t n, wn;
    float eps;
    uint32_t ystride;
};
struct HcArgs {
    uint32_t n_embd, iters;
    float eps;
    uint32_t tokens;
};
struct HcStepArgs {
    uint32_t n_embd, expand, parts, iters;
    float eps, rms_eps;
};
struct SwigluArgs {
    uint32_t n;
    float limit;
};
struct CombineArgs {
    uint32_t n, n_used;
};
struct RouteArgs {
    uint32_t n_expert, n_used, norm;
    float scale;
    uint32_t lookup, score;
    float miss_penalty;
};

} // namespace

EngineBase::EngineBase(const gguf::ModelFiles& model, mtl::Context& ctx, Telemetry telemetry)
    : model_(model), ctx_(ctx), tel_(telemetry)
{
    if (const char* r = std::getenv("MAKO_REPEAT")) {
        const std::string_view spec(r);
        const size_t colon = spec.rfind(':');
        if (colon == std::string_view::npos)
            throw std::runtime_error("MAKO_REPEAT wants <label substring>:<n>");
        repeat_label_ = spec.substr(0, colon);
        repeat_n_ = std::max(1, std::atoi(r + colon + 1));
    }
    gemv_ = std::make_unique<mtl::Gemv>(ctx_);
    gemm_ = std::make_unique<mtl::Gemm>(ctx_);
    lib_ = ctx_.load(mtl::kernels_metallib());
    routed_ = mtl::Ref<MTL::SharedEvent>(ctx_.device()->newSharedEvent());
    ready_ = mtl::Ref<MTL::SharedEvent>(ctx_.device()->newSharedEvent());
    if (const char* g = std::getenv("MAKO_STAGE_GB"))
        stage_bytes_ = uint64_t(std::max(0.0, std::atof(g)) * double(1ull << 30));
}

void EngineBase::set_staging_limit(double gb)
{
    if (gb >= 0)
        stage_bytes_ = uint64_t(gb * double(1ull << 30));
}

uint64_t EngineBase::staging_bytes() const
{
    if (stage_bytes_ == 0 || !arena_)
        return stage_bytes_;
    // Measured: GLM-5.3 (~38 missing experts a layer) with a 3 GB ring -11% prefill (the borrowed memory evicts
    // resident experts), with 0.6 GB the same prefill and ~17% fewer decode misses after the prompt.
    const auto* table = static_cast<const int32_t*>(arena_->slot_table()->contents());
    const uint32_t n_rows = std::min<uint32_t>(uint32_t(arena_->usage().size()), dims_.n_layer);
    uint64_t most = 0;
    for (uint32_t l = 0; l < n_rows; ++l) {
        uint64_t per = 0, missing = 0;
        for (int p = 0; p < 3; ++p)
            per += arena_->part(l, p).bytes;
        for (uint32_t e = 0; e < dims_.n_expert; ++e)
            missing += table[uint64_t(l) * dims_.n_expert + e] < 0;
        most = std::max(most, missing * per);
    }
    return std::min(stage_bytes_, 2 * most);
}

void EngineBase::begin_staging(uint32_t tokens)
{
    stager_.reset();
    const uint64_t bytes = staging_bytes();
    if (tokens < kStageMinTokens || bytes == 0)
        return;
    const uint32_t n_rows = std::min<uint32_t>(uint32_t(arena_->usage().size()), dims_.n_layer);
    auto st = std::make_unique<Stager>(ctx_.device(), *arena_, bytes);
    if (!st->ok())
        return;
    const auto* table = static_cast<const int32_t*>(arena_->slot_table()->contents());
    std::vector<std::pair<uint32_t, std::vector<int>>> plan;
    for (uint32_t l = 0; l < n_rows; ++l) {
        if (!arena_->part(l, 0).bytes)
            continue;
        std::vector<int> missing;
        for (uint32_t e = 0; e < dims_.n_expert; ++e)
            if (table[uint64_t(l) * dims_.n_expert + e] < 0)
                missing.push_back(int(e));
        plan.emplace_back(l, std::move(missing));
    }
    st->start(std::move(plan));
    stager_ = std::move(st);
}

void EngineBase::end_staging()
{
    if (stager_ && std::getenv("MAKO_STAGE_STATS"))
        std::fprintf(stderr, "staging: %.1f GB in %.0f ms (%.1f GB/s; readers busy %.0f ms)\n",
                     stager_->read_bytes() / 1e9, stager_->span_ms(),
                     stager_->read_bytes() / 1e6 / std::max(1.0, stager_->span_ms()), stager_->read_ms());
    stager_.reset();
}

EngineBase::~EngineBase()
{
    if (pressure_src_) {
        dispatch_source_cancel(pressure_src_);
        dispatch_sync_f(pressure_queue_, nullptr, [](void*) {}); // a running handler finishes first
        dispatch_release(pressure_src_);
        dispatch_release(pressure_queue_);
    }
    arena_.reset();
    for (auto& [path, fd] : nocache_fds_)
        ::close(fd);
}

void EngineBase::init_runtime(ArenaInit init)
{
    const auto t_arena = Clock::now();
    load_.dense_bytes = dense_bytes_;
    uint64_t working_set = ctx_.info().recommended_working_set;
    uint64_t simulated_lock = 0;
    if (init.simulate_ram_gb > 0) {
        const double phys = query_memory(0).phys, ram = init.simulate_ram_gb * 1e9;
        if (ram >= phys)
            throw std::runtime_error("--simulate-ram must be below the machine's RAM");
        // Locking all of phys - ram would also leave less lockable memory than the smaller Mac has (macOS lets
        // processes lock RAM minus a reserve, ~85%): lock at most what keeps that much lockable.
        uint64_t wire_limit = 0;
        size_t len = sizeof(wire_limit);
        if (sysctlbyname("vm.global_user_wire_limit", &wire_limit, &len, nullptr, 0) != 0)
            wire_limit = uint64_t(0.85 * phys);
        const double lock = std::min(phys - ram, double(wire_limit) - 0.85 * ram);
        if (lock < phys - ram)
            std::fprintf(stderr, "mako: simulating %.0f GB with %.1f GB locked (not %.1f): %.1f GB more free than such a "
                                 "Mac would have\n",
                         init.simulate_ram_gb, lock / 1e9, (phys - ram) / 1e9, (phys - ram - lock) / 1e9);
        simulated_lock = uint64_t(std::max(0.0, lock)) / 16384 * 16384;
        simulated_ = std::make_unique<LockedMemory>(simulated_lock);
        working_set = uint64_t(double(working_set) * ram / phys);
    }
    arena_plan_.mem = query_memory(working_set);
    if (init.simulate_ram_gb > 0) {
        // The locked difference is not this process's own use on the simulated machine.
        arena_plan_.mem.footprint -= double(simulated_lock);
        arena_plan_.mem.phys = init.simulate_ram_gb * 1e9;
    }
    arena_plan_.lendable = init.lendable;
    arena_plan_.budget = init.arena_gb >= 0
                             ? init.arena_gb * 1e9
                             : std::max(1e9, arena_budget(arena_plan_.mem) - double(init.context_bytes));
    auto t_phase = Clock::now();
    routing_key_ = core::routing_key(model_, init.moe_layers);
    load_.key_ms = ms_since(t_phase);
    default_profile_ = default_profile_path(routing_key_);
    if (!init.expert_counts.empty())
        loaded_profile_ = init.expert_counts;
    else if (std::filesystem::exists(default_profile_))
        loaded_profile_ = default_profile_;
    const uint32_t rows = uint32_t(init.parts.size());
    Counts counts;
    if (!loaded_profile_.empty()) {
        counts = load_profile(loaded_profile_, rows, dims_.n_expert);
        counts.resize(rows); // a row for a draft block this engine did not load
        // A profile without a draft block's counts gives each of its experts the trunk's mean count, so the block
        // competes for slots like a typical layer (its live counts then decide which experts stay).
        uint64_t sum = 0, n = 0;
        for (uint32_t l = 0; l < std::min(rows, dims_.n_layer); ++l)
            for (uint64_t c : counts[l]) {
                sum += c;
                ++n;
            }
        // (Measured: half or a quarter of the mean saves the trunk a few slots but slows the drafts more.)
        for (uint32_t l = dims_.n_layer; l < rows; ++l)
            if (counts[l].empty() && n && init.parts[l][0].bytes)
                counts[l].assign(dims_.n_expert, std::max<uint64_t>(1, sum / n));
    }
    t_phase = Clock::now();
    arena_ = std::make_unique<ExpertArena>(ctx_.device(), ctx_.info().max_buffer, std::move(init.parts), dims_.n_expert,
                                           dims_.n_expert_used, arena_plan_.budget, counts, arena_plan_.lendable);
    load_.arena_ms = ms_since(t_phase);
    arena_->prefill();

    // The first compute pass that reads an arena buffer pays a driver cost that grows with its size (~0.5 s for
    // 75 GB); pay it here instead of in the first step.
    const auto t0 = Clock::now();
    MTL::Buffer* sink = scratch("warm", 16);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    const uint32_t one = 1;
    std::vector<MTL::Buffer*> arena_bufs;
    for (const auto& b : arena_->buffers())
        arena_bufs.push_back(b.get());
    for (MTL::Buffer* b : arena_->tail_buffers())
        arena_bufs.push_back(b);
    for (MTL::Buffer* b : arena_bufs) {
        enc->setComputePipelineState(pso("copy_f32"));
        enc->setBuffer(b, 0, 0);
        enc->setBuffer(sink, 0, 1);
        enc->setBytes(&one, sizeof(one), 2);
        for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    arena_plan_.warm_ms = ms_since(t0);

    load_.prefill_ms = arena_->stats().prefill_ms;
    load_.warm_ms = arena_plan_.warm_ms;
    load_.arena_bytes = arena_->bytes();
    load_.arena_budget = arena_plan_.budget;
    load_.slots_min = UINT32_MAX;
    for (uint32_t l = 0; l < dims_.n_layer; ++l)
        if (arena_->slots(l)) {
            load_.slots_min = std::min(load_.slots_min, arena_->slots(l));
            load_.slots_max = std::max(load_.slots_max, arena_->slots(l));
        }
    load_.profile = loaded_profile_.string();
    if (tel_ >= Telemetry::layers) {
        if (!ctx_.info().counters_stage || !ctx_.info().timestamp_set)
            throw std::runtime_error("this GPU has no timestamps at encoder boundaries; per-layer telemetry is "
                                     "unavailable");
        // Layers: a prefill group's MoE layers commit their misses a few experts at a time, each in a timed encoder.
        add_samples(tel_ == Telemetry::kernels ? uint64_t(dims_.n_layer) * 240 + 64 : uint64_t(dims_.n_layer) * 256 + 64);
    }
    pressure_queue_ = dispatch_queue_create("mako.memory-pressure", DISPATCH_QUEUE_SERIAL);
    pressure_src_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                           DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
                                               DISPATCH_MEMORYPRESSURE_CRITICAL,
                                           pressure_queue_);
    if (pressure_src_) {
        dispatch_set_context(pressure_src_, this);
        dispatch_source_set_event_handler_f(pressure_src_, [](void* ctx) {
            auto* e = static_cast<EngineBase*>(ctx);
            const uintptr_t d = dispatch_source_get_data(e->pressure_src_);
            e->pressure_.store(d & DISPATCH_MEMORYPRESSURE_CRITICAL ? 2 : d & DISPATCH_MEMORYPRESSURE_WARN ? 1 : 0,
                               std::memory_order_relaxed);
        });
        dispatch_resume(pressure_src_);
    }
    load_.total_ms = load_.dense_ms + ms_since(t_arena);
}

MTL::ComputeCommandEncoder* EngineBase::encoder(MTL::CommandBuffer* cb, uint32_t layer, int section,
                                                MTL::DispatchType type)
{
    if (!timing() || (tel_ == Telemetry::layers && section < 0))
        return cb->computeCommandEncoder(type);
    if (tel_ == Telemetry::kernels)
        section = -1;
    MTL::ComputePassDescriptor* d = MTL::ComputePassDescriptor::computePassDescriptor();
    d->setDispatchType(type);
    auto* att = d->sampleBufferAttachments()->object(0);
    if (next_sample_ + 2 > samples_.size() * kSamplesPerBuffer)
        throw std::runtime_error("telemetry: too many timed encoders in one step");
    att->setSampleBuffer(samples_[next_sample_ / kSamplesPerBuffer].get());
    att->setStartOfEncoderSampleIndex(next_sample_ % kSamplesPerBuffer);
    att->setEndOfEncoderSampleIndex(next_sample_ % kSamplesPerBuffer + 1);
    timed_.push_back({layer, section, next_sample_, next_sample_ + 1, {}});
    next_sample_ += 2;
    return cb->computeCommandEncoder(d);
}

void EngineBase::after(MTL::ComputeCommandEncoder*& enc, std::string_view kernel)
{
    if (tel_ != Telemetry::kernels || timed_.empty())
        return;
    timed_.back().kernel = kernel;
    timed_.back().layer = cur_layer_;
    enc->endEncoding();
    enc = encoder(cur_cb_, cur_layer_, -1);
}

void EngineBase::add_samples(uint64_t want)
{
    NS::Array* sets = ctx_.device()->counterSets();
    MTL::CounterSet* ts = nullptr;
    for (NS::UInteger i = 0; i < sets->count(); ++i)
        if (sets->object<MTL::CounterSet>(i)->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
            ts = sets->object<MTL::CounterSet>(i);
    mtl::Ref<MTL::CounterSampleBufferDescriptor> d(MTL::CounterSampleBufferDescriptor::alloc()->init());
    d->setCounterSet(ts);
    d->setSampleCount(kSamplesPerBuffer);
    d->setStorageMode(MTL::StorageModeShared);
    for (uint64_t n = samples_.size() * kSamplesPerBuffer; n < want; n += kSamplesPerBuffer) {
        NS::Error* err = nullptr;
        samples_.emplace_back(ctx_.device()->newCounterSampleBuffer(d.get(), &err));
        if (!samples_.back())
            throw std::runtime_error(std::format("newCounterSampleBuffer: {}",
                                                 err ? err->localizedDescription()->utf8String() : "?"));
    }
}

void EngineBase::set_part_timing(bool on)
{
    part_timing_ = on && tel_ == Telemetry::layers;
    if (part_timing_) // two parts per sparse-attention group, a few per layer otherwise
        add_samples(uint64_t(dims_.n_layer) * 256 + 64 + 32768);
}

void EngineBase::part(MTL::ComputeCommandEncoder*& enc, std::string_view name)
{
    if (!part_timing_ || tel_ != Telemetry::layers || timed_.empty() || timed_.back().section < 0)
        return;
    const Timed cur = timed_.back();
    enc->endEncoding();
    enc = encoder(cur_cb_, cur.layer, cur.section);
    timed_.back().kernel = name;
}

// Turns the step's encoder timestamps into per-layer section times. GPU ticks are converted with the CPU clock
// over the step, from timestamp pairs taken at its start and end.
void EngineBase::resolve_layer_times(uint64_t cpu0, uint64_t gpu0)
{
    MTL::Timestamp cpu1 = 0, gpu1 = 0;
    ctx_.device()->sampleTimestamps(&cpu1, &gpu1); // the CPU side is in nanoseconds
    const double ns_per_tick = gpu1 > gpu0 ? double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1.0;
    std::vector<uint64_t> ts(next_sample_, MTL::CounterErrorValue);
    for (NS::UInteger first = 0; first < next_sample_; first += kSamplesPerBuffer) {
        const NS::UInteger n = std::min<NS::UInteger>(kSamplesPerBuffer, next_sample_ - first);
        NS::Data* data = samples_[first / kSamplesPerBuffer]->resolveCounterRange(NS::Range(0, n));
        if (!data)
            continue;
        const auto* r = static_cast<const MTL::CounterResultTimestamp*>(data->mutableBytes());
        for (NS::UInteger i = 0; i < n; ++i)
            ts[first + i] = r[i].timestamp;
    }
    last_step_.parts.clear();
    if (tel_ == Telemetry::layers) {
        last_step_.layers.assign(dims_.n_layer, LayerTimes{});
        for (uint32_t l = 0; l < dims_.n_layer; ++l) {
            last_step_.layers[l].layer = l;
            last_step_.layers[l].mla = l < layer_mla_.size() && layer_mla_[l];
        }
    }
    for (const Timed& t : timed_) {
        const uint64_t a = ts[t.start], b = ts[t.end];
        if (a == MTL::CounterErrorValue || b == MTL::CounterErrorValue || b < a)
            continue;
        const double ms = double(b - a) * ns_per_tick / 1e6;
        if (t.section < 0) {
            if (t.kernel.empty())
                continue;
            auto it = std::ranges::find(last_step_.kernels, t.kernel, &KernelTime::kernel);
            if (it == last_step_.kernels.end())
                it = last_step_.kernels.insert(it, KernelTime{t.kernel, 0, 0});
            it->ms += ms;
            ++it->count;
            continue;
        }
        if (!t.kernel.empty()) {
            auto it = std::ranges::find(last_step_.parts, t.kernel, &KernelTime::kernel);
            if (it == last_step_.parts.end())
                it = last_step_.parts.insert(it, KernelTime{t.kernel, 0, 0});
            it->ms += ms;
            ++it->count;
        }
        LayerTimes& lt = last_step_.layers[t.layer];
        (t.section == 0 ? lt.attn_ms : t.section == 1 ? lt.ffn_ms : t.section == 2 ? lt.hit_ms : lt.miss_ms) += ms;
    }
    timed_.clear();
    next_sample_ = 0;
}

void EngineBase::apply_pressure()
{
    const int level = pressure_.load(std::memory_order_relaxed);
    if (level > 0 && !pressure_lent_ && !arena_->lent() && arena_->lendable_bytes()) {
        arena_->lend();
        pressure_lent_ = true;
        std::fprintf(stderr, "mako: memory pressure (%s): released %.1f GB of expert slots\n",
                     level > 1 ? "critical" : "warning", arena_->lendable_bytes() / 1e9);
    } else if (level == 0 && pressure_lent_ && arena_->reclaim()) {
        pressure_lent_ = false;
        std::fprintf(stderr, "mako: memory pressure over: took back %.1f GB of expert slots\n",
                     arena_->lendable_bytes() / 1e9);
    }
}

EngineBase::ScratchLoan::ScratchLoan(EngineBase& e, uint64_t need) : e_(e)
{
    constexpr uint64_t kLendBelow = 256ull << 20;
    lend_ = need > kLendBelow && e.arena_->lendable_bytes() > 0;
    if (!lend_)
        return;
    for (const auto& [name, b] : e.scratch_)
        kept_.emplace(name, b.get());
    e.arena_->lend(need - kLendBelow);
}

EngineBase::ScratchLoan::~ScratchLoan()
{
    if (!lend_)
        return;
    std::erase_if(e_.scratch_, [&](const auto& kv) {
        auto it = kept_.find(kv.first);
        return it == kept_.end() || it->second != kv.second.get();
    });
    if (!e_.pressure_lent_)
        e_.arena_->reclaim();
}

void EngineBase::save_profile(const std::filesystem::path& path, bool merge) const
{
    const auto& usage = arena_->usage();
    const uint32_t nl = uint32_t(usage.size()); // draft rows included
    Counts counts = merge && std::filesystem::exists(path) ? load_profile(path, nl, dims_.n_expert) : Counts(nl);
    counts.resize(std::max<size_t>(counts.size(), nl)); // rows of layers this engine did not load stay
    for (uint32_t l = 0; l < nl; ++l) {
        if (!arena_->slots(l))
            continue;
        counts[l].resize(dims_.n_expert, 0);
        for (uint32_t e = 0; e < dims_.n_expert; ++e)
            counts[l][e] += usage[l][e];
    }
    core::save_profile(path, counts);
}

EngineBase::DevTensor EngineBase::upload(const std::string& name)
{
    auto ref = model_.find(name);
    if (!ref)
        throw std::runtime_error("missing tensor " + name);
    const gguf::Tensor& t = *ref.tensor;
    if (dense_quant_ && t.type == gguf::Type::Q8_0 && t.dims[0] % 256 == 0)
        return upload_requant(ref.file->fd(), t, *dense_quant_, name);
    return upload_raw(ref.file->fd(), t.offset, t.bytes, t.type, t.dims, name);
}

// Q8_0 rows dequantized and quantized again to `to`, on all cores.
EngineBase::DevTensor EngineBase::upload_requant(int fd, const gguf::Tensor& t, gguf::Type to, const std::string& name)
{
    const uint64_t k = t.dims[0], src_rb = mtl::Gemv::row_bytes(t.type, uint32_t(k));
    const uint64_t dst_rb = mtl::Gemv::row_bytes(to, uint32_t(k)), rows = t.bytes / src_rb;
    std::vector<std::byte> raw(t.bytes);
    gguf::read_exact(fd, raw.data(), t.bytes, t.offset);
    DevTensor d = reserve(rows * dst_rb, to, t.dims);
    auto* dst = static_cast<std::byte*>(d.buf->contents()) + d.off;
    const float* imp = nullptr;
    uint64_t rows_per_group = rows;
    if (const auto it = importance_.find(name); it != importance_.end()) {
        const uint64_t groups = it->second.size() / k;
        if (groups && it->second.size() % k == 0 && rows % groups == 0) {
            imp = it->second.data();
            rows_per_group = rows / groups;
        }
    }
    const uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::jthread> pool;
    for (uint32_t w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            std::vector<float> row(k);
            for (uint64_t r = w; r < rows; r += threads) {
                quant::dequant_row(t.type, raw.data() + r * src_rb, row.data(), k);
                quant::quantize_row(to, row.data(), dst + r * dst_rb, k, imp ? imp + (r / rows_per_group) * k : nullptr);
            }
        });
    pool.clear();
    requant_bytes_ += t.bytes - rows * dst_rb;
    return d;
}

EngineBase::DevTensor EngineBase::reserve(uint64_t bytes, gguf::Type type, std::vector<uint64_t> dims)
{
    if (weight_chunks_.empty() || chunk_used_ + bytes > weight_chunks_.back()->length()) {
        const uint64_t size = std::max(kChunkBytes, (bytes + kAlign - 1) / kAlign * kAlign);
        weight_chunks_.emplace_back(ctx_.device()->newBuffer(size, MTL::ResourceStorageModeShared));
        if (!weight_chunks_.back())
            throw std::runtime_error("weight buffer allocation failed");
        chunk_used_ = 0;
    }
    DevTensor d;
    d.buf = weight_chunks_.back().get();
    d.off = chunk_used_;
    d.type = type;
    d.dims = std::move(dims);
    chunk_used_ += (bytes + kAlign - 1) / kAlign * kAlign;
    dense_bytes_ += bytes;
    return d;
}

EngineBase::DevTensor EngineBase::upload_raw(int fd, uint64_t offset, uint64_t bytes, gguf::Type type,
                                             std::vector<uint64_t> dims, const std::string& what)
{
    const bool bf16 = type == gguf::Type::BF16;
    if (type != gguf::Type::F32 && !bf16 && !mtl::Gemv::supports(type))
        throw std::runtime_error(std::format("{}: unsupported type {}", what, gguf::traits(type).name));
    DevTensor d = reserve(bf16 ? bytes * 2 : bytes, bf16 ? gguf::Type::F32 : type, std::move(dims));
    char* dst = static_cast<char*>(d.buf->contents()) + d.off;
    if (bf16) {
        std::vector<uint16_t> raw(bytes / 2);
        gguf::read_exact(fd, raw.data(), bytes, offset);
        auto* f = reinterpret_cast<uint32_t*>(dst);
        for (size_t i = 0; i < raw.size(); ++i)
            f[i] = uint32_t(raw[i]) << 16;
    } else {
        gguf::read_exact(fd, dst, bytes, offset);
    }
    return d;
}

ExpertArena::Part EngineBase::expert(const std::string& name, bool prefill)
{
    auto ref = model_.find(name);
    if (!ref)
        throw std::runtime_error("missing tensor " + name);
    const gguf::Tensor& t = *ref.tensor;
    if (t.dims.size() != 3 || t.dims[2] != dims_.n_expert)
        throw std::runtime_error(name + ": expected [k, n, n_expert]");
    if (!mtl::Gemv::supports(t.type))
        throw std::runtime_error(std::format("{}: unsupported type {}", name, gguf::traits(t.type).name));
    if (prefill && !mtl::Gemm::supports(t.type))
        throw std::runtime_error(std::format("{}: type {} has no prefill GEMM (experts: IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, "
                                             "IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32)",
                                             name, gguf::traits(t.type).name));
    auto [it, fresh] = nocache_fds_.try_emplace(ref.file->path(), -1);
    if (fresh) {
        it->second = ::open(ref.file->path().c_str(), O_RDONLY);
        if (it->second < 0)
            throw std::runtime_error(std::format("open {}: {}", ref.file->path(), std::strerror(errno)));
        ::fcntl(it->second, F_NOCACHE, 1);
    }
    ExpertArena::Part e;
    e.fd = it->second;
    e.file_off = t.offset;
    e.bytes = t.bytes / dims_.n_expert;
    e.type = t.type;
    e.k = uint32_t(t.dims[0]);
    e.n = uint32_t(t.dims[1]);
    return e;
}

MTL::Buffer* EngineBase::scratch(const std::string& name, uint64_t bytes)
{
    auto it = scratch_.find(name);
    if (it == scratch_.end() || it->second->length() < bytes) {
        mtl::Ref<MTL::Buffer> b(ctx_.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
        it = scratch_.insert_or_assign(name, std::move(b)).first;
    }
    return it->second.get();
}

MTL::ComputePipelineState* EngineBase::pso(const std::string& fn)
{
    if (repeat_n_ > 1)
        last_pso_ = fn;
    auto it = psos_.find(fn);
    if (it == psos_.end())
        it = psos_.emplace(fn, ctx_.pipeline(lib_.get(), fn)).first;
    return it->second.get();
}

void EngineBase::gemv(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off,
                      MTL::Buffer* y, uint64_t y_off, uint32_t batch, uint32_t t)
{
    mtl::Gemv::Args a;
    a.type = w.type;
    a.k = uint32_t(w.dims[0]);
    a.n = uint32_t(w.dims[1]);
    a.t = t;
    a.batch = batch;
    const bool labelled = tel_ == Telemetry::kernels || repeat_n_ > 1;
    const std::string label = labelled ? std::format("gemv {} {}x{}", gguf::traits(a.type).name, a.k, a.n) : "";
    for (int r = repeats(label); r > 0; --r)
        gemv_->encode(enc, a, w.buf, w.off, x, x_off, y, y_off);
    if (tel_ == Telemetry::kernels)
        after(enc, label);
}

// y[t] = W x[t] for T tokens: the GEMV for one token, the GEMM for wide matrices, else the GEMV over all tokens.
// y rows are ldy elements apart (0: n).
void EngineBase::proj(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off,
                      MTL::Buffer* y, uint64_t y_off, uint32_t T, uint32_t ldy)
{
    const uint32_t k = uint32_t(w.dims[0]), n = uint32_t(w.dims[1]);
    // A few tokens (decode, speculative verification) stay on the GEMV, which reads each weight once for them.
    if (T >= gemm_min_tokens_ && (n >= 256 || (ldy && ldy != n)) && k % 32 == 0 && mtl::Gemm::supports(w.type)) {
        gemm_->encode(enc, w.type, k, n, T, w.buf, w.off, halves(enc, x, x_off, T, k, 0), 0, y, y_off, ldy);
        if (tel_ == Telemetry::kernels)
            after(enc, std::format("gemm {} {}x{}", gguf::traits(w.type).name, k, n));
        return;
    }
    if (T == 1 || !ldy || ldy == n) {
        gemv(enc, w, x, x_off, y, y_off, 1, T);
        return;
    }
    // Strided rows: one multi-token GEMV (weights read once) into contiguous rows, then copied into place.
    MTL::Buffer* tmp = scratch("proj_rows", uint64_t(T) * n * 4);
    gemv(enc, w, x, x_off, tmp, 0, 1, T);
    const uint32_t a[2] = {n, ldy};
    enc->setComputePipelineState(pso("copy_rows"));
    enc->setBuffer(tmp, 0, 0);
    enc->setBuffer(y, y_off, 1);
    enc->setBytes(a, sizeof(a), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(n, T, 1), MTL::Size(256, 1, 1));
    after(enc, "copy_rows");
}

MTL::Buffer* EngineBase::halves(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                                uint32_t cols, uint32_t ld)
{
    MTL::Buffer* h = scratch("xh", uint64_t(rows) * cols * 2);
    gemm_->to_half(enc, x, x_off, rows, cols, ld, h, 0);
    return h;
}

void EngineBase::dispatch(MTL::ComputeCommandEncoder*& enc, const std::string& fn,
                          std::initializer_list<MTL::Buffer*> bufs, const void* args, size_t args_size, MTL::Size grid,
                          MTL::Size tg, bool threads)
{
    enc->setComputePipelineState(pso(fn));
    NS::UInteger i = 0;
    for (MTL::Buffer* b : bufs)
        enc->setBuffer(b, 0, i++);
    if (args)
        enc->setBytes(args, args_size, i);
    for (int r = repeats(fn); r > 0; --r) {
        if (threads)
            enc->dispatchThreads(grid, tg);
        else
            enc->dispatchThreadgroups(grid, tg);
    }
    after(enc, fn);
}

void EngineBase::norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const DevTensor* w, MTL::Buffer* y,
                      uint32_t rows, uint32_t n, uint32_t wn, float eps, uint64_t x_off, uint64_t y_off,
                      uint32_t ystride)
{
    const NormArgs a{n, w ? wn : 0, eps, ystride};
    enc->setComputePipelineState(pso("rms_norm"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(w ? w->buf : x, w ? w->off : 0, 1);
    enc->setBuffer(y, y_off, 2);
    enc->setBytes(&a, sizeof(a), 3);
    for (int r = repeats("rms_norm"); r > 0; --r)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "rms_norm");
}

void EngineBase::hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale,
                        const DevTensor& base, MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T,
                        uint64_t r_off)
{
    const uint32_t E = dims_.n_embd, hd = dims_.hc * E;
    MTL::Buffer* flat = scratch("flat", uint64_t(T) * hd * 4);
    MTL::Buffer* mixes = scratch("mixes", uint64_t(T) * 24 * 4);
    norm(enc, R, nullptr, flat, T, hd, 0, dims_.rms_eps, r_off);
    proj(enc, fn, flat, 0, mixes, 0, T);
    const HcArgs ha{E, dims_.hc_iters, dims_.hc_eps, T};
    enc->setComputePipelineState(pso("hc_mix"));
    enc->setBuffer(mixes, 0, 0);
    enc->setBuffer(scale.buf, scale.off, 1);
    enc->setBuffer(base.buf, base.off, 2);
    enc->setBuffer(pc, 0, 3);
    enc->setBytes(&ha, sizeof(ha), 4);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(T, 1, 1), MTL::Size(std::min<uint32_t>(T, 64), 1, 1));
    after(enc, "hc_mix");
    enc->setComputePipelineState(pso("hc_collapse"));
    enc->setBuffer(R, r_off, 0);
    enc->setBuffer(pc, 0, 1);
    enc->setBuffer(x, 0, 2);
    enc->setBytes(&ha, sizeof(ha), 3);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    after(enc, "hc_collapse");
}

void EngineBase::hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* R, MTL::Buffer* out,
                         MTL::Buffer* pc, uint32_t T, uint64_t r_off, uint64_t x_off, uint64_t pc_off)
{
    const HcArgs ha{dims_.n_embd, dims_.hc_iters, dims_.hc_eps, T};
    enc->setComputePipelineState(pso("hc_expand"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(R, r_off, 1);
    enc->setBuffer(pc, pc_off, 2);
    enc->setBuffer(out, r_off, 3);
    enc->setBytes(&ha, sizeof(ha), 4);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(dims_.n_embd, T, 1), MTL::Size(256, 1, 1));
    after(enc, "hc_expand");
}

void EngineBase::hc_step(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* post_x, MTL::Buffer* R, MTL::Buffer* R_out,
                         const DevTensor& fn, const DevTensor& scale, const DevTensor& base, MTL::Buffer* x,
                         MTL::Buffer* pc, uint32_t T)
{
    constexpr uint32_t kParts = 32; // at most 64 (hc_step_b)
    const HcStepArgs a{dims_.n_embd, post_x ? 1u : 0u, kParts, dims_.hc_iters, dims_.hc_eps, dims_.rms_eps};
    MTL::Buffer* part = scratch("hc_part", uint64_t(kMaxDecodeTokens) * kParts * 25 * 4);
    enc->setComputePipelineState(pso("hc_step_a"));
    enc->setBuffer(post_x ? post_x : R, 0, 0);
    enc->setBuffer(R, 0, 1);
    enc->setBuffer(pc, 0, 2);
    enc->setBuffer(post_x ? R_out : part, 0, 3);
    enc->setBuffer(fn.buf, fn.off, 4);
    enc->setBuffer(part, 0, 5);
    enc->setBytes(&a, sizeof(a), 6);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(kParts, T, 1), MTL::Size(256, 1, 1));
    after(enc, "hc_step_a");
    enc->setComputePipelineState(pso("hc_step_b"));
    enc->setBuffer(part, 0, 0);
    enc->setBuffer(post_x ? R_out : R, 0, 1);
    enc->setBuffer(scale.buf, scale.off, 2);
    enc->setBuffer(base.buf, base.off, 3);
    enc->setBuffer(pc, 0, 4);
    enc->setBuffer(x, 0, 5);
    enc->setBytes(&a, sizeof(a), 6);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(1024, 1, 1));
    after(enc, "hc_step_b");
}

void EngineBase::ffn_dense(MTL::ComputeCommandEncoder*& enc, const DevTensor& gate, const DevTensor& up,
                           const DevTensor& down, MTL::Buffer* xn, MTL::Buffer* out, float limit, uint32_t T,
                           uint64_t x_off, uint64_t out_off)
{
    const uint32_t nff = uint32_t(gate.dims[1]);
    MTL::Buffer* g = scratch("ffg", uint64_t(T) * nff * 4);
    MTL::Buffer* u = scratch("ffu", uint64_t(T) * nff * 4);
    MTL::Buffer* m = scratch("ffm", uint64_t(T) * nff * 4);
    proj(enc, gate, xn, x_off, g, 0, T);
    proj(enc, up, xn, x_off, u, 0, T);
    const SwigluArgs sa{T * nff, limit > 1e-6f ? limit : 0.0f};
    dispatch(enc, "swiglu", {g, u, m}, &sa, sizeof(sa), MTL::Size(T * nff, 1, 1), MTL::Size(256, 1, 1));
    if (T == 1)
        probe(enc, "imx_ffm", int(cur_layer_), m, nff);
    proj(enc, down, m, 0, out, out_off, T);
}

// Serves the MoE layers of a committed command buffer in order: waits until a layer's router has run, makes its
// selected experts resident and writes their slots, then lets the GPU continue.
void EngineBase::serve_moe(MTL::CommandBuffer* cb, bool block)
{
    const uint32_t k = dims_.n_expert_used;
    const uint64_t rows = kMaxDecodeTokens * 16 * 4;
    auto* sel = static_cast<int*>(scratch("msel", rows)->contents());
    auto* w = static_cast<float*>(scratch("ew", rows)->contents());
    auto* hit_ids = static_cast<const int32_t*>(scratch("eids_hit", rows)->contents());
    auto* miss_ids = static_cast<int32_t*>(scratch("eids_miss", rows)->contents());
    auto* miss_rank = static_cast<int32_t*>(scratch("mrank", rows)->contents());
    const uint32_t arena_rows = uint32_t(arena_->usage().size());
    auto* indirect = static_cast<uint32_t*>(scratch("mind", uint64_t(arena_rows) * kMaxDecodeTokens * 36)->contents());
    size_t served = 0;
    for (const auto [il, v, T, predicted] : pending_moe_) {
        if (!block && routed_->signaledValue() < v)
            break;
        ++served;
        const auto t_wait = Clock::now();
        for (uint32_t spins = 0; routed_->signaledValue() < v; ++spins)
            if (spins % 4096 == 4095 && cb->status() == MTL::CommandBufferStatusError)
                throw std::runtime_error(std::format("command buffer failed: {}",
                                                     cb->error() ? cb->error()->localizedDescription()->utf8String()
                                                                 : "?"));
        if (probe_) {
            probe_("ffn_moe_probs_biased", int(il),
                   static_cast<const float*>(scratch("mbiased", dims_.n_expert * 4)->contents()), dims_.n_expert);
            std::array<float, 16> self{};
            std::copy_n(sel, k, self.begin());
            probe_("moe_sel", int(il), self.data(), k);
            // Which experts are resident before this layer's misses are read (1 / 0 per expert).
            const auto* table = static_cast<const int32_t*>(arena_->slot_table()->contents()) +
                                uint64_t(il) * dims_.n_expert;
            std::vector<float> resident(dims_.n_expert);
            for (uint32_t e = 0; e < dims_.n_expert; ++e)
                resident[e] = table[e] >= 0 ? 1.0f : 0.0f;
            probe_("moe_resident", int(il), resident.data(), resident.size());
        }
        // An override changes the selection after the GPU looked it up: then every expert goes to the miss pass.
        const bool overridden = bool(route_override_);
        if (overridden) {
            if (T != 1)
                throw std::runtime_error("routing overrides need one token per step");
            route_override_(int(il), sel, w, k);
        }
        last_step_.route_wait_ms += ms_since(t_wait);
        // One ensure for the union of the tokens' experts: separate ones could evict a slot another token's
        // resident pass reads, or read an expert twice.
        const auto t_ensure = Clock::now();
        std::vector<int> want;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t j = 0; j < k; ++j)
                if (std::ranges::find(want, sel[t * 16 + j]) == want.end())
                    want.push_back(sel[t * 16 + j]);
        std::vector<int32_t> got(want.size());
        arena_->ensure(il, want, got.data());
        last_step_.ensure_ms += ms_since(t_ensure);
        auto slot_of = [&](int e) { return got[std::ranges::find(want, e) - want.begin()]; };
        uint32_t* ind = indirect + uint64_t(il) * kMaxDecodeTokens * 9;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t i = t * 16 + j;
                const int32_t s = slot_of(sel[i]);
                if (overridden) {
                    miss_rank[i] = int32_t(j);
                    miss_ids[i] = s;
                } else if (miss_rank[i] >= 0) {
                    miss_ids[t * 16 + miss_rank[i]] = s;
                } else if (hit_ids[i] != s) {
                    throw std::runtime_error(std::format("layer {}: GPU found expert {} in slot {}, the arena has it "
                                                         "in {}",
                                                         il, sel[i], hit_ids[i], s));
                }
            }
        if (overridden)
            for (int d = 0; d < 3; ++d)
                ind[d * 3 + 2] = k;
        ready_->setSignaledValue(v);
        if (predicted) { // arena row il + predicted
            const auto* pred = static_cast<const int32_t*>(scratch("mpred", uint64_t(arena_rows) * 64)->contents()) +
                               uint64_t(il) * 16;
            std::vector<int> ahead;
            for (int i = 0; i < 16 && pred[i] >= 0; ++i)
                ahead.push_back(pred[i]);
            if (!ahead.empty())
                arena_->prefetch(il + predicted, ahead);
        }
    }
    pending_moe_.erase(pending_moe_.begin(), pending_moe_.begin() + ptrdiff_t(served));
}

void EngineBase::finish(MTL::CommandBuffer* cb)
{
    auto t0 = Clock::now();
    if (encoding_) {
        stats_.encode_ms += std::chrono::duration<double, std::milli>(t0 - step_start_).count();
        encoding_ = false;
    }
    cb->commit();
    serve_moe(cb);
    cb->waitUntilCompleted();
    stats_.gpu_wait_ms += ms_since(t0);
    stats_.gpu_exec_ms += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
    if (cb->status() != MTL::CommandBufferStatusCompleted)
        throw std::runtime_error(std::format("command buffer failed: {}",
                                             cb->error() ? cb->error()->localizedDescription()->utf8String() : "?"));
    deliver_probes();
}

void EngineBase::submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc)
{
    enc->endEncoding();
    finish(cb);
    cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    enc = encoder(cb, cur_layer_, -1);
}

void EngineBase::probe(MTL::ComputeCommandEncoder*& enc, const std::string& name, int layer, MTL::Buffer* buf,
                       size_t n, uint64_t off)
{
    if (!probe_)
        return;
    PendingProbe p{name, layer, mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(n * 4, MTL::ResourceStorageModeShared)), n};
    const uint32_t count = uint32_t(n);
    enc->setComputePipelineState(pso("copy_f32"));
    enc->setBuffer(buf, off, 0);
    enc->setBuffer(p.copy.get(), 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "probe copy");
    pending_probes_.push_back(std::move(p));
}

void EngineBase::deliver_probes()
{
    for (const auto& p : pending_probes_)
        probe_(p.name, p.layer, static_cast<const float*>(p.copy->contents()), p.n);
    pending_probes_.clear();
}

// T tokens' routed FFN (plus the shared expert) of arena row il on xn into ffo: GPU routing, the resident experts
// at once and the misses after the CPU has read them (serve_moe).
void EngineBase::moe_decode(MTL::CommandBuffer* cb, MTL::ComputeCommandEncoder*& enc, const MoeWeights& mw,
                            uint32_t il, MTL::Buffer* xn, MTL::Buffer* ffo, uint32_t T,
                            const std::function<void(MTL::ComputeCommandEncoder*&, MTL::Buffer*)>& between,
                            const MoeWeights* next, uint32_t predict_k, uint32_t ahead)
{
    if (T > kMaxDecodeTokens)
        throw std::runtime_error("too many tokens for a decode step");
    const uint32_t E = dims_.n_embd, k = dims_.n_expert_used, nff = dims_.n_ff_exp;
    const uint32_t tl = std::min(il, dims_.n_layer); // telemetry: a draft block counts as the rest
    MTL::Buffer* sho = scratch("sho", uint64_t(T) * E * 4);
    MTL::Buffer* rlog = scratch("rlog", uint64_t(T) * dims_.n_expert * 4);
    // Sized for kMaxDecodeTokens whatever T is: serve_moe reads them (a larger request would reallocate).
    const uint64_t id_rows = kMaxDecodeTokens * 16 * 4;
    MTL::Buffer* ew = scratch("ew", id_rows);
    MTL::Buffer* hit_ids = scratch("eids_hit", id_rows);
    MTL::Buffer* miss_ids = scratch("eids_miss", id_rows);
    MTL::Buffer* miss_rank = scratch("mrank", id_rows);
    // Per layer and token: x, y, z of the gate, up and down miss dispatches (moe_route writes z, the misses).
    const uint32_t arena_rows = uint32_t(arena_->usage().size());
    MTL::Buffer* indirect = scratch("mind", uint64_t(arena_rows) * kMaxDecodeTokens * 36);
    const uint64_t ind_off = uint64_t(il) * kMaxDecodeTokens * 36;
    auto part_args = [&](int part, uint32_t tokens) {
        const ExpertArena::Part& e = arena_->part(il, part);
        mtl::Gemv::Args a;
        a.type = e.type;
        a.k = e.k;
        a.n = e.n;
        a.t = tokens;
        a.batch = k;
        a.x_shared = part < 2;
        a.ids_stride = 16 * 4;
        return a;
    };
    auto* ind = static_cast<uint32_t*>(indirect->contents()) + ind_off / 4;
    for (uint32_t t = 0; t < T; ++t)
        for (int part = 0; part < 3; ++part) {
            const MTL::Size g = gemv_->indexed_grid(part_args(part, 1));
            ind[t * 9 + part * 3] = uint32_t(g.width);
            ind[t * 9 + part * 3 + 1] = uint32_t(g.height);
        }

    part(enc, "moe router");
    gemv(enc, mw.router, xn, 0, rlog, 0, 1, T);
    const RouteArgs ra{dims_.n_expert, k, routing_.norm ? 1u : 0u, routing_.scale, 1, uint32_t(routing_.score),
                       routing_.miss_penalty};
    enc->setComputePipelineState(pso("moe_route"));
    enc->setBuffer(rlog, 0, 0);
    enc->setBuffer(mw.bias.buf, mw.bias.off, 1);
    enc->setBuffer(scratch("msel", id_rows), 0, 2);
    enc->setBuffer(ew, 0, 3);
    enc->setBuffer(scratch("mbiased", uint64_t(kMaxDecodeTokens) * dims_.n_expert * 4), 0, 4);
    enc->setBytes(&ra, sizeof(ra), 5);
    enc->setBuffer(arena_->slot_table(), uint64_t(il) * dims_.n_expert * 4, 6);
    enc->setBuffer(hit_ids, 0, 7);
    enc->setBuffer(miss_rank, 0, 8);
    enc->setBuffer(indirect, ind_off, 9);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(32, 1, 1));
    after(enc, "moe_route");
    // The prediction costs a router GEMV: skipped while reads ahead are not paying off (ExpertArena decides).
    const bool predict = next && predict_k && T == 1 && il + ahead < arena_rows && arena_->prefetch_worthwhile();
    if (predict) {
        MTL::Buffer* rnext = scratch("rlog_next", uint64_t(dims_.n_expert) * 4);
        gemv(enc, next->router, xn, 0, rnext, 0, 1, 1);
        const uint32_t pa[3] = {dims_.n_expert, std::min<uint32_t>(predict_k, 16), uint32_t(routing_.score)};
        enc->setComputePipelineState(pso("moe_predict"));
        enc->setBuffer(rnext, 0, 0);
        enc->setBuffer(next->bias.buf, next->bias.off, 1);
        enc->setBuffer(arena_->slot_table(), uint64_t(il + ahead) * dims_.n_expert * 4, 2);
        enc->setBuffer(scratch("mpred", uint64_t(arena_rows) * 64), uint64_t(il) * 64, 3);
        enc->setBytes(pa, sizeof(pa), 4);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
        after(enc, "moe_predict");
    }
    enc->endEncoding();
    const uint64_t v = ++event_value_;
    cb->encodeSignalEvent(routed_.get(), v);
    pending_moe_.push_back({il, v, T, predict ? ahead : 0u});
    enc = encoder(cb, tl, il < dims_.n_layer ? 2 : -1);
    part(enc, "moe shared expert");
    // The CPU starts on the misses as soon as the route is known; the shared expert runs meanwhile.
    ffn_dense(enc, mw.sh_gate, mw.sh_up, mw.sh_down, xn, sho, mw.clamp_sh, T);
    if (between)
        between(enc, sho);

    // Resident experts of every token run while the CPU reads the misses; the miss passes (one per token) wait.
    const SwigluArgs sa{T * k * nff, mw.clamp_exp > 1e-6f ? mw.clamp_exp : 0.0f};
    auto run = [&](int part, uint32_t tokens, bool miss, MTL::Buffer* ids, uint64_t ids_off, MTL::Buffer* in,
                   uint64_t in_off, MTL::Buffer* out, uint64_t out_off, uint64_t ind_at) {
        const ExpertArena::Region& r = arena_->region(il, part);
        const mtl::Gemv::Args a = part_args(part, tokens);
        const bool labelled = tel_ == Telemetry::kernels || repeat_n_ > 1;
        const std::string label = labelled ? std::format("expert {} {}x{} {}", gguf::traits(a.type).name, a.k, a.n,
                                                         miss ? "miss" : "hit")
                                           : "";
        for (int rep = repeats(label); rep > 0; --rep)
            gemv_->encode_indexed(enc, a, r.buf, r.off, r.stride, ids, ids_off, in, in_off, out, out_off,
                                  miss ? indirect : nullptr, ind_at + uint64_t(part) * 12, {r.tail, r.tail_off, r.split});
        if (tel_ == Telemetry::kernels)
            after(enc, label);
    };
    auto bufs = [&](const char* tag) {
        return std::array<MTL::Buffer*, 4>{scratch(std::string("eg") + tag, uint64_t(T) * k * nff * 4),
                                           scratch(std::string("eu") + tag, uint64_t(T) * k * nff * 4),
                                           scratch(std::string("em") + tag, uint64_t(T) * k * nff * 4),
                                           scratch(std::string("ed") + tag, uint64_t(T) * k * E * 4)};
    };
    const auto h = bufs("h");
    part(enc, "moe resident experts");
    run(0, T, false, hit_ids, 0, xn, 0, h[0], 0, 0);
    run(1, T, false, hit_ids, 0, xn, 0, h[1], 0, 0);
    dispatch(enc, "swiglu", {h[0], h[1], h[2]}, &sa, sizeof(sa), MTL::Size(T * k * nff, 1, 1), MTL::Size(256, 1, 1));
    run(2, T, false, hit_ids, 0, h[2], 0, h[3], 0, 0);
    enc->endEncoding();
    cb->encodeWait(ready_.get(), v);
    enc = encoder(cb, tl, il < dims_.n_layer ? 3 : -1);
    part(enc, "moe miss experts");
    const auto m = bufs("m");
    const uint64_t rows = uint64_t(k) * nff * 4;
    for (uint32_t t = 0; t < T; ++t) {
        run(0, 1, true, miss_ids, t * 64, xn, uint64_t(t) * E * 4, m[0], t * rows, ind_off + t * 36);
        run(1, 1, true, miss_ids, t * 64, xn, uint64_t(t) * E * 4, m[1], t * rows, ind_off + t * 36);
    }
    dispatch(enc, "swiglu", {m[0], m[1], m[2]}, &sa, sizeof(sa), MTL::Size(T * k * nff, 1, 1), MTL::Size(256, 1, 1));
    for (uint32_t t = 0; t < T; ++t)
        run(2, 1, true, miss_ids, t * 64, m[2], t * rows, m[3], uint64_t(t) * k * E * 4, ind_off + t * 36);
    const CombineArgs ca{E, k};
    part(enc, "moe combine");
    dispatch(enc, "moe_combine_split", {h[3], m[3], miss_rank, ew, sho, ffo}, &ca, sizeof(ca), MTL::Size(E, T, 1),
             MTL::Size(256, 1, 1));
}

// G tokens' routed experts of arena row il (rows of xn_g) added into ffo, which holds the shared expert's output:
// GPU routing, then the (token, expert) pairs grouped by expert, each expert run once over its tokens by GEMM:
// resident experts first, the misses read a few at a time while the GPU runs the batches before them. Command
// buffers committed here go to `committed` (wait for them before this layer's slots are touched again); cb and enc
// continue after.
void EngineBase::moe_prefill(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc, const MoeWeights& mw,
                             uint32_t il, MTL::Buffer* xn_g, MTL::Buffer* ffo, uint32_t G,
                             std::vector<MTL::CommandBuffer*>& committed)
{
    const uint32_t E = dims_.n_embd, k = dims_.n_expert_used, nff = dims_.n_ff_exp;
    MTL::Buffer* rlog = scratch("p_rlog", uint64_t(G) * dims_.n_expert * 4);
    MTL::Buffer* sel = scratch("p_sel", uint64_t(G) * 16 * 4);
    MTL::Buffer* ew = scratch("p_ew", uint64_t(G) * 16 * 4);
    proj(enc, mw.router, xn_g, 0, rlog, 0, G);
    // Routing only; the CPU assigns slots after this sync.
    // A group too small to stage its reads routes with the decode steps' miss penalty (a large one reads every
    // missing expert ahead anyway).
    const bool penalize = routing_.miss_penalty > 0.0f && G < kStageMinTokens;
    const RouteArgs ra{dims_.n_expert, k, routing_.norm ? 1u : 0u, routing_.scale, 0, uint32_t(routing_.score),
                       penalize ? routing_.miss_penalty : 0.0f};
    MTL::Buffer* unused = scratch("p_unused", 64);
    enc->setComputePipelineState(pso("moe_route"));
    enc->setBuffer(rlog, 0, 0);
    enc->setBuffer(mw.bias.buf, mw.bias.off, 1);
    enc->setBuffer(sel, 0, 2);
    enc->setBuffer(ew, 0, 3);
    enc->setBuffer(scratch("p_biased", uint64_t(G) * dims_.n_expert * 4), 0, 4);
    enc->setBytes(&ra, sizeof(ra), 5);
    if (penalize)
        enc->setBuffer(arena_->slot_table(), uint64_t(il) * dims_.n_expert * 4, 6);
    else
        enc->setBuffer(unused, 0, 6);
    enc->setBuffer(unused, 0, 7);
    enc->setBuffer(unused, 0, 8);
    enc->setBuffer(unused, 0, 9);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(G, 1, 1), MTL::Size(32, 1, 1));
    after(enc, "moe_route");
    submit(cb, enc);
    cur_cb_ = cb;

    // Group the (token, expert) pairs by expert, resident experts first, and run each expert once over
    // all its tokens.
    const auto* selp = static_cast<const int*>(sel->contents());
    const auto* table = static_cast<const int32_t*>(arena_->slot_table()->contents()) +
                        uint64_t(il) * dims_.n_expert;
    std::vector<std::vector<uint32_t>> users(dims_.n_expert); // per expert: t * 16 + j
    for (uint32_t t = 0; t < G; ++t)
        for (uint32_t j = 0; j < k; ++j)
            users[selp[t * 16 + j]].push_back(t * 16 + j);
    std::vector<int> order;
    for (int e = 0; e < int(dims_.n_expert); ++e)
        if (!users[e].empty() && table[e] >= 0)
            order.push_back(e);
    const size_t n_resident = order.size();
    {
        std::vector<uint32_t> counts(dims_.n_expert);
        for (uint32_t e = 0; e < dims_.n_expert; ++e)
            counts[e] = uint32_t(users[e].size());
        arena_->note_prefill(il, counts);
    }
    for (int e = 0; e < int(dims_.n_expert); ++e)
        if (!users[e].empty() && table[e] < 0)
            order.push_back(e);
    const uint32_t rows = G * k;
    MTL::Buffer* tok = scratch("p_tok", uint64_t(rows) * 4);
    MTL::Buffer* pos = scratch("p_pos", uint64_t(G) * 16 * 4);
    auto* tokp = static_cast<int32_t*>(tok->contents());
    auto* posp = static_cast<int32_t*>(pos->contents());
    std::vector<uint32_t> start(dims_.n_expert + 1);
    uint32_t row = 0;
    for (int e : order) {
        start[e] = row;
        for (uint32_t u : users[e]) {
            tokp[row] = int32_t(u / 16);
            posp[u] = int32_t(row++);
        }
    }
    const uint32_t brows = std::min(rows, kBatchRows);
    MTL::Buffer* xb = scratch("p_xb", uint64_t(brows) * E * 2); // half, for the GEMMs
    MTL::Buffer* gb = scratch("p_gb", uint64_t(brows) * nff * 4);
    MTL::Buffer* ub = scratch("p_ub", uint64_t(brows) * nff * 4);
    MTL::Buffer* mb = scratch("p_mb", uint64_t(brows) * nff * 2); // half
    // Expert outputs for all rows, combined once: an order of the sum that does not depend on the batches
    // (and so on which experts happened to be resident).
    MTL::Buffer* yg = scratch("p_yg", uint64_t(rows) * E * 4);
    const SwigluArgs sa0{0, mw.clamp_exp > 1e-6f ? mw.clamp_exp : 0.0f};
    auto rows_of = [&](size_t b0, size_t b1) {
        return b0 == b1 ? 0u : start[order[b1 - 1]] + uint32_t(users[order[b1 - 1]].size()) - start[order[b0]];
    };

    // Where each part of expert e is: its arena slot, or (staged) its staging slot.
    std::vector<int32_t> slot(dims_.n_expert, -1);
    std::vector<Stager::Where> staged(dims_.n_expert);
    auto weights = [&](int part, int e) -> std::pair<MTL::Buffer*, uint64_t> {
        if (staged[e].job >= 0)
            return {staged[e].buf, staged[e].off[part]};
        return arena_->region(il, part).at(uint32_t(slot[e]));
    };
    // Experts [b0, b1) of order (rows within kBatchRows), pinned resident or staged: gather their rows, then
    // in a concurrent encoder gate and up of every expert, the SwiGLU, every down into yg.
    auto run = [&](size_t b0, size_t b1) {
        if (b0 == b1)
            return;
        const uint32_t r0 = start[order[b0]], nr = rows_of(b0, b1);
        const uint32_t ne = E;
        enc->setComputePipelineState(pso("gather_rows_half"));
        enc->setBuffer(xn_g, 0, 0);
        enc->setBuffer(tok, uint64_t(r0) * 4, 1);
        enc->setBuffer(xb, 0, 2);
        enc->setBytes(&ne, sizeof(ne), 3);
        for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(E, nr, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        enc = encoder(cb, il, 2, MTL::DispatchTypeConcurrent);
        auto gemm_part = [&](int part, int e, MTL::Buffer* xin, uint32_t xw, MTL::Buffer* y, uint32_t yw) {
            const ExpertArena::Part& p = arena_->part(il, part);
            const auto [wb, wo] = weights(part, e);
            gemm_->encode(enc, p.type, p.k, p.n, uint32_t(users[e].size()), wb, wo, xin,
                          uint64_t(start[e] - r0) * xw * 2, y, uint64_t(start[e] - r0) * yw * 4);
        };
        for (size_t i = b0; i < b1; ++i) {
            gemm_part(0, order[i], xb, E, gb, nff);
            gemm_part(1, order[i], xb, E, ub, nff);
        }
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        SwigluArgs sa = sa0;
        sa.n = nr * nff;
        enc->setComputePipelineState(pso("swiglu_half"));
        enc->setBuffer(gb, 0, 0);
        enc->setBuffer(ub, 0, 1);
        enc->setBuffer(mb, 0, 2);
        enc->setBytes(&sa, sizeof(sa), 3);
        for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(sa.n, 1, 1), MTL::Size(256, 1, 1));
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        for (size_t i = b0; i < b1; ++i) {
            const int e = order[i];
            const ExpertArena::Part& p = arena_->part(il, 2);
            const auto [wb, wo] = weights(2, e);
            gemm_->encode(enc, p.type, p.k, p.n, uint32_t(users[e].size()), wb, wo, mb,
                          uint64_t(start[e] - r0) * nff * 2, yg, uint64_t(start[e]) * E * 4);
        }
        enc->endEncoding();
        enc = encoder(cb, il, 3);
    };

    const auto t_ensure = Clock::now();
    // Batches go to the GPU in command buffers of their own, so that when the misses run out of
    // unpinned slots only the oldest needs to finish: its experts' slots are then unpinned and reused.
    // Resident experts come first, so the GPU runs them while the misses are read.
    struct InFlight {
        MTL::CommandBuffer* cb;
        std::vector<int32_t> slots;
    };
    std::deque<InFlight> inflight;
    auto commit = [&](size_t b0, size_t b1) {
        if (b0 == b1)
            return;
        run(b0, b1);
        enc->endEncoding();
        cb->commit();
        InFlight f{cb->retain(), {}};
        for (size_t i = b0; i < b1; ++i)
            f.slots.push_back(slot[order[i]]);
        inflight.push_back(std::move(f));
        cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        enc = encoder(cb, cur_layer_, -1);
    };
    auto retire = [&] {
        InFlight& f = inflight.front();
        f.cb->waitUntilCompleted();
        if (f.cb->status() != MTL::CommandBufferStatusCompleted)
            throw std::runtime_error("command buffer failed");
        stats_.gpu_exec_ms += (f.cb->GPUEndTime() - f.cb->GPUStartTime()) * 1e3;
        f.cb->release();
        arena_->unpin(il, f.slots);
        inflight.pop_front();
    };
    constexpr size_t kExpertsPerCommit = 32;
    size_t b0 = 0;
    for (size_t i = 0; i < n_resident; ++i) {
        if (rows_of(b0, i + 1) > kBatchRows || i - b0 == kExpertsPerCommit) {
            commit(b0, i);
            b0 = i;
        }
        arena_->ensure(il, std::span<const int>(&order[i], 1), &slot[order[i]], true, false);
    }
    commit(b0, n_resident);
    if (stager_) {
        // Misses from staging, read ahead of this layer: batches as their experts arrive, each slot released when
        // the GPU work that reads it completes.
        Stager* st = stager_.get();
        for (size_t i = n_resident; i < order.size();) {
            size_t j = i + 1;
            while (j < order.size() && j - i < kExpertsPerCommit && rows_of(i, j + 1) <= kBatchRows)
                ++j;
            std::vector<int64_t> jobs;
            for (size_t m = i; m < j; ++m) {
                staged[order[m]] = st->take(il, order[m]);
                if (staged[order[m]].job < 0)
                    throw std::logic_error("routed expert neither resident nor staged");
                jobs.push_back(staged[order[m]].job);
            }
            arena_->touch(il, std::span<const int>(&order[i], j - i));
            run(i, j);
            enc->endEncoding();
            cb->addCompletedHandler([st, jobs](MTL::CommandBuffer*) {
                for (int64_t job : jobs)
                    st->release(job);
            });
            cb->commit();
            committed.push_back(cb->retain());
            cb = ctx_.queue()->commandBuffer();
            cur_cb_ = cb;
            enc = encoder(cb, cur_layer_, -1);
            i = j;
        }
        st->release_untaken(il);
    }
    // Misses are read a few at a time (their chunks in flight together) and each batch goes to the GPU
    // as soon as it is resident, so reading the next overlaps running this one.
    constexpr size_t kMissesPerRead = 4;
    std::vector<int32_t> got(kMissesPerRead);
    for (size_t i = stager_ ? order.size() : n_resident; i < order.size();) {
        size_t j = i + 1;
        while (j < order.size() && j - i < kMissesPerRead && rows_of(i, j + 1) <= kBatchRows)
            ++j;
        while (!arena_->ensure(il, std::span<const int>(&order[i], j - i), got.data(), true, false)) {
            // Not enough unpinned slots: free the oldest batch's, or read fewer.
            if (!inflight.empty())
                retire();
            else if (j - i > 1)
                j = i + (j - i) / 2;
            else
                throw std::runtime_error("expert arena has no slot for one expert");
        }
        for (size_t m = i; m < j; ++m)
            slot[order[m]] = got[m - i];
        commit(i, j);
        i = j;
    }
    last_step_.ensure_ms += ms_since(t_ensure);
    // This layer's slots are only touched again after the group's work is drained.
    for (InFlight& f : inflight)
        committed.push_back(f.cb);
    arena_->unpin(il);

    const CombineArgs ca{E, k};
    enc->setComputePipelineState(pso("moe_combine_rows"));
    enc->setBuffer(yg, 0, 0);
    enc->setBuffer(pos, 0, 1);
    enc->setBuffer(ew, 0, 2);
    enc->setBuffer(ffo, 0, 3);
    enc->setBuffer(ffo, 0, 4);
    enc->setBytes(&ca, sizeof(ca), 5);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(E, G, 1), MTL::Size(256, 1, 1));
}

} // namespace mako::core
