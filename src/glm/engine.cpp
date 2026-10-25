#include "glm/engine.hpp"

#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <sys/sysctl.h>

#include <algorithm>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <numeric>
#include <span>
#include <stdexcept>

namespace mako::glm {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr uint64_t kAlign = 256;
constexpr uint64_t kChunkBytes = 1ull << 30;
constexpr NS::UInteger kSamplesPerBuffer = 4096; // the largest counter sample buffer is 32 KiB
constexpr uint32_t kBatchRows = 8192;            // prefill expert rows (token, expert pairs) run and combined together

struct NormArgs {
    uint32_t n, wn;
    float eps;
    uint32_t ystride;
};
struct AblateArgs {
    uint32_t n_embd, k;
    float scale;
};

struct HcArgs {
    uint32_t n_embd, iters;
    float eps;
    uint32_t tokens;
};
struct ConvArgs {
    uint32_t channels, kernel_len, tokens, snap;
};
struct GateArgs {
    uint32_t head_dim, heads;
    float lower_bound;
};
struct L2Args {
    uint32_t n;
    float eps;
    uint32_t token_stride;
};
struct ScanArgs {
    uint32_t head_dim, tokens, qkv_stride, inner, heads, snap;
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
    uint32_t lookup;
};
struct PoolArgs {
    uint32_t dim, kpool, pool;
};
struct ScoreArgs {
    uint32_t dim, heads, first, kpool;
    float w_scale;
    uint32_t stride;
};
struct SelectArgs {
    uint32_t first, n_top, kpool, score_stride, list_stride;
};
struct GatherArgs {
    uint32_t heads, dim, stride, splits, tokens;
};
struct ListSoftmaxArgs {
    uint32_t stride;
    float scale;
    uint32_t heads;
};
struct LayerNormArgs {
    uint32_t n;
    float eps;
    uint32_t ystride;
};
struct SoftmaxArgs {
    uint32_t n_kv;
    float scale;
    uint32_t first_kv, heads;
};
struct PvArgs {
    uint32_t heads, dim, n_kv, keys_per_split;
};

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

struct Engine::Layer {
    bool kda = false, dense = false;
    bool mtp = false; // the next-token block: plain residuals instead of hyper-connections
    DevTensor eh_proj, enorm, hnorm, head_norm;
    DevTensor attn_norm, ffn_norm, hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    // KDA
    DevTensor wq, wk, wv, conv_q, conv_k, conv_v, f_a, f_b, beta, ssm_a, dt_b, g_a, g_b, o_norm;
    // MLA
    DevTensor q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b;
    DevTensor idx_q_b, idx_k, idx_k_norm, idx_k_norm_b, idx_proj, idx_gate, idx_ape;
    DevTensor wo;
    // FFN
    DevTensor ffn_gate, ffn_up, ffn_down;
    DevTensor router, sh_gate, sh_up, sh_down;
    DevTensor router_bias;
    ExpertArena::LayerParts experts{};
};

Engine::Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx)
    : model_(model), cfg_(cfg), opts_(opts), ctx_(ctx)
{
    tel_ = opts_.telemetry;
    const auto t_load = Clock::now();
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

    if (cfg_.n_expert_used > 16)
        throw std::runtime_error("more than 16 experts per token");
    if (opts_.indexer_top_k)
        cfg_.indexer_top_k = opts_.indexer_top_k;
    if (!cfg_.indexer_select_tail || cfg_.indexer_top_k % cfg_.indexer_kpool != 0)
        throw std::runtime_error("unsupported indexer configuration");

    token_embd_ = model_.find("token_embd.weight");
    if (!token_embd_ || !quant::can_dequant(token_embd_.tensor->type))
        throw std::runtime_error("token_embd.weight missing or of an unsupported type");
    output_norm_ = upload("output_norm.weight");
    output_ = upload("output.weight");

    const uint32_t D = cfg_.kda_head_dim, inner = cfg_.kda_inner();
    auto load_mla = [&](Layer& L, const std::string& p) {
        L.q_a = upload(p + "attn_q_a.weight");
        L.q_a_norm = upload(p + "attn_q_a_norm.weight");
        L.q_b = upload(p + "attn_q_b.weight");
        L.kv_a = upload(p + "attn_kv_a_mqa.weight");
        L.kv_a_norm = upload(p + "attn_kv_a_norm.weight");
        L.k_b = upload(p + "attn_k_b.weight");
        L.v_b = upload(p + "attn_v_b.weight");
        L.idx_q_b = upload(p + "indexer.attn_q_b.weight");
        L.idx_k = upload(p + "indexer.attn_k.weight");
        L.idx_k_norm = upload(p + "indexer.k_norm.weight");
        L.idx_k_norm_b = upload(p + "indexer.k_norm.bias");
        L.idx_proj = upload(p + "indexer.proj.weight");
        L.idx_gate = upload(p + "indexer_compressor_gate.weight");
        L.idx_ape = upload(p + "indexer_compressor_ape.weight");
        kda_state_.emplace_back();
        conv_state_.emplace_back();
        kda_state_alt_.emplace_back();
        conv_state_alt_.emplace_back();
        kv_cache_.emplace_back(ctx_.device()->newBuffer(uint64_t(opts_.max_ctx) * cfg_.kv_lora_rank * 2,
                                                        MTL::ResourceStorageModeShared));
        idx_cache_.emplace_back(ctx_.device()->newBuffer(uint64_t(opts_.max_ctx) * 2 * cfg_.indexer_dim * 4,
                                                         MTL::ResourceStorageModeShared));
        idx_pooled_.emplace_back(ctx_.device()->newBuffer(
            uint64_t(opts_.max_ctx / cfg_.indexer_kpool + 1) * cfg_.indexer_dim * 4, MTL::ResourceStorageModeShared));
    };
    // Trunk experts also run in prefill (the GEMM); the next-token block's only in steps (the GEMV).
    auto load_moe = [&](Layer& L, const std::string& p, bool prefill) {
        L.router = upload(p + "ffn_gate_inp.weight");
        L.sh_gate = upload(p + "ffn_gate_shexp.weight");
        L.sh_up = upload(p + "ffn_up_shexp.weight");
        L.sh_down = upload(p + "ffn_down_shexp.weight");
        L.router_bias = upload(p + "exp_probs_b.bias");
        if (L.router_bias.type != gguf::Type::F32 || L.router_bias.dims[0] != cfg_.n_expert)
            throw std::runtime_error(p + "exp_probs_b.bias is not F32 [n_expert]");
        L.experts = {expert(p + "ffn_gate_exps.weight", prefill), expert(p + "ffn_up_exps.weight", prefill),
                     expert(p + "ffn_down_exps.weight", prefill)};
    };
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        auto L = std::make_unique<Layer>();
        const std::string p = std::format("blk.{}.", il);
        L->kda = cfg_.is_kda[il];
        L->dense = il < cfg_.n_dense_lead;
        L->attn_norm = upload(p + "attn_norm.weight");
        L->ffn_norm = upload(p + "ffn_norm.weight");
        L->hc_attn_fn = upload(p + "hc_attn_fn.weight");
        L->hc_attn_base = upload(p + "hc_attn_base.weight");
        L->hc_attn_scale = upload(p + "hc_attn_scale.weight");
        L->hc_ffn_fn = upload(p + "hc_ffn_fn.weight");
        L->hc_ffn_base = upload(p + "hc_ffn_base.weight");
        L->hc_ffn_scale = upload(p + "hc_ffn_scale.weight");
        L->wo = upload(p + "attn_output.weight");
        if (L->kda) {
            L->wq = upload(p + "attn_q.weight");
            L->wk = upload(p + "attn_k.weight");
            L->wv = upload(p + "attn_v.weight");
            L->conv_q = upload(p + "ssm_conv1d_q.weight");
            L->conv_k = upload(p + "ssm_conv1d_k.weight");
            L->conv_v = upload(p + "ssm_conv1d_v.weight");
            L->f_a = upload(p + "ssm_f_a.weight");
            L->f_b = upload(p + "ssm_f_b.weight");
            L->beta = upload(p + "ssm_beta.weight");
            L->ssm_a = upload(p + "ssm_a");
            L->dt_b = upload(p + "ssm_dt.bias");
            L->g_a = upload(p + "ssm_g_a.weight");
            L->g_b = upload(p + "ssm_g_b.weight");
            L->o_norm = upload(p + "ssm_norm.weight");
            kda_state_.emplace_back(ctx_.device()->newBuffer(uint64_t(cfg_.n_head) * D * D * 4,
                                                             MTL::ResourceStorageModeShared));
            conv_state_.emplace_back(ctx_.device()->newBuffer(uint64_t(3) * inner * (cfg_.conv_kernel - 1) * 4,
                                                              MTL::ResourceStorageModeShared));
            std::memset(kda_state_.back()->contents(), 0, kda_state_.back()->length());
            std::memset(conv_state_.back()->contents(), 0, conv_state_.back()->length());
            kda_state_alt_.emplace_back(ctx_.device()->newBuffer(uint64_t(cfg_.n_head) * D * D * 4,
                                                                 MTL::ResourceStorageModeShared));
            conv_state_alt_.emplace_back(ctx_.device()->newBuffer(uint64_t(3) * inner * (cfg_.conv_kernel - 1) * 4,
                                                                  MTL::ResourceStorageModeShared));
            kv_cache_.emplace_back();
            idx_cache_.emplace_back();
            idx_pooled_.emplace_back();
        } else {
            load_mla(*L, p);
        }
        if (L->dense) {
            L->ffn_gate = upload(p + "ffn_gate.weight");
            L->ffn_up = upload(p + "ffn_up.weight");
            L->ffn_down = upload(p + "ffn_down.weight");
        } else {
            load_moe(*L, p, true);
        }
        layers_.push_back(std::move(L));
    }
    // The next-token (MTP) block after the trunk, as layers_[n_layer]: an MLA attention and a MoE FFN of its own.
    if (opts_.mtp && cfg_.n_layer_mtp > 0) {
        auto L = std::make_unique<Layer>();
        const std::string p = std::format("blk.{}.", cfg_.n_layer);
        L->mtp = true;
        L->attn_norm = upload(p + "attn_norm.weight");
        L->ffn_norm = upload(p + "ffn_norm.weight");
        L->wo = upload(p + "attn_output.weight");
        load_mla(*L, p);
        load_moe(*L, p, false);
        L->eh_proj = upload(p + "nextn.eh_proj.weight");
        L->enorm = upload(p + "nextn.enorm.weight");
        L->hnorm = upload(p + "nextn.hnorm.weight");
        L->head_norm = upload(p + "nextn.shared_head_norm.weight");
        layers_.push_back(std::move(L));
    }
    logits_.resize(cfg_.n_vocab);
    routed_ = mtl::Ref<MTL::SharedEvent>(ctx_.device()->newSharedEvent());
    ready_ = mtl::Ref<MTL::SharedEvent>(ctx_.device()->newSharedEvent());

    load_.dense_ms = ms_since(t_load);
    load_.dense_bytes = dense_bytes_;
    std::vector<ExpertArena::LayerParts> parts;
    for (const auto& L : layers_)
        parts.push_back(L->experts);
    uint64_t working_set = ctx_.info().recommended_working_set;
    uint64_t simulated_lock = 0;
    if (opts_.simulate_ram_gb > 0) {
        const double phys = query_memory(0).phys, ram = opts_.simulate_ram_gb * 1e9;
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
                         opts_.simulate_ram_gb, lock / 1e9, (phys - ram) / 1e9, (phys - ram - lock) / 1e9);
        simulated_lock = uint64_t(std::max(0.0, lock)) / 16384 * 16384;
        simulated_ = std::make_unique<LockedMemory>(simulated_lock);
        working_set = uint64_t(double(working_set) * ram / phys);
    }
    arena_plan_.mem = query_memory(working_set);
    if (opts_.simulate_ram_gb > 0) {
        // The locked difference is not this process's own use on the simulated machine.
        arena_plan_.mem.footprint -= double(simulated_lock);
        arena_plan_.mem.phys = opts_.simulate_ram_gb * 1e9;
    }
    built_chunk_ = opts_.prefill_chunk;
    built_group_ = std::max(opts_.prefill_chunk, opts_.prefill_group);
    arena_plan_.lendable = double(prefill_scratch(opts_.prefill_chunk, std::max(opts_.prefill_chunk, opts_.prefill_group)));
    // The context caches are allocated for max_ctx but only fill (and count in the footprint) as the context grows;
    // the checkpoints a Session keeps add to them.
    uint64_t context_bytes = 0;
    for (const auto* caches : {&kv_cache_, &idx_cache_, &idx_pooled_})
        for (const auto& b : *caches)
            if (b)
                context_bytes += b->length();
    context_bytes += uint64_t(opts_.checkpoints) * checkpoint_bytes();
    arena_plan_.budget = opts_.arena_gb >= 0 ? opts_.arena_gb * 1e9
                                             : std::max(1e9, arena_budget(arena_plan_.mem) - double(context_bytes));
    auto t_phase = Clock::now();
    routing_key_ = glm::routing_key(model_, cfg_);
    load_.key_ms = ms_since(t_phase);
    default_profile_ = default_profile_path(routing_key_);
    if (!opts_.expert_counts.empty())
        loaded_profile_ = opts_.expert_counts;
    else if (std::filesystem::exists(default_profile_))
        loaded_profile_ = default_profile_;
    Counts counts;
    if (!loaded_profile_.empty()) {
        counts = load_profile(loaded_profile_, uint32_t(layers_.size()), cfg_.n_expert);
        counts.resize(layers_.size()); // a row for a next-token block this engine did not load
        // A profile without the next-token block's counts gives each of its experts the trunk's mean count, so
        // the block competes for slots like a typical layer (its live counts then decide which experts stay).
        if (layers_.size() > cfg_.n_layer && counts[cfg_.n_layer].empty()) {
            uint64_t sum = 0, n = 0;
            for (uint32_t l = 0; l < cfg_.n_layer; ++l)
                for (uint64_t c : counts[l]) {
                    sum += c;
                    ++n;
                }
            // (Measured: half or a quarter of the mean saves the trunk a few slots but slows the drafts more.)
            if (n)
                counts[cfg_.n_layer].assign(cfg_.n_expert, std::max<uint64_t>(1, sum / n));
        }
    }
    t_phase = Clock::now();
    arena_ = std::make_unique<ExpertArena>(ctx_.device(), ctx_.info().max_buffer, std::move(parts), cfg_.n_expert,
                                           cfg_.n_expert_used, arena_plan_.budget, counts, arena_plan_.lendable);
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
    for (uint32_t l = 0; l < cfg_.n_layer; ++l)
        if (arena_->slots(l)) {
            load_.slots_min = std::min(load_.slots_min, arena_->slots(l));
            load_.slots_max = std::max(load_.slots_max, arena_->slots(l));
        }
    load_.profile = loaded_profile_.string();
    if (opts_.telemetry >= Telemetry::layers) {
        if (!ctx_.info().counters_stage || !ctx_.info().timestamp_set)
            throw std::runtime_error("this GPU has no timestamps at encoder boundaries; per-layer telemetry is "
                                     "unavailable");
        NS::Array* sets = ctx_.device()->counterSets();
        MTL::CounterSet* ts = nullptr;
        for (NS::UInteger i = 0; i < sets->count(); ++i)
            if (sets->object<MTL::CounterSet>(i)->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
                ts = sets->object<MTL::CounterSet>(i);
        // Layers: a prefill group's MoE layers commit their misses a few experts at a time, each in a timed encoder.
        const uint64_t want = opts_.telemetry == Telemetry::kernels ? uint64_t(cfg_.n_layer) * 240 + 64
                                                                    : uint64_t(cfg_.n_layer) * 256 + 64;
        mtl::Ref<MTL::CounterSampleBufferDescriptor> d(MTL::CounterSampleBufferDescriptor::alloc()->init());
        d->setCounterSet(ts);
        d->setSampleCount(kSamplesPerBuffer);
        d->setStorageMode(MTL::StorageModeShared);
        for (uint64_t n = 0; n < want; n += kSamplesPerBuffer) {
            NS::Error* err = nullptr;
            samples_.emplace_back(ctx_.device()->newCounterSampleBuffer(d.get(), &err));
            if (!samples_.back())
                throw std::runtime_error(std::format("newCounterSampleBuffer: {}",
                                                     err ? err->localizedDescription()->utf8String() : "?"));
        }
    }
    pressure_queue_ = dispatch_queue_create("mako.memory-pressure", DISPATCH_QUEUE_SERIAL);
    pressure_src_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                           DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
                                               DISPATCH_MEMORYPRESSURE_CRITICAL,
                                           pressure_queue_);
    if (pressure_src_) {
        dispatch_set_context(pressure_src_, this);
        dispatch_source_set_event_handler_f(pressure_src_, [](void* ctx) {
            auto* e = static_cast<Engine*>(ctx);
            const uintptr_t d = dispatch_source_get_data(e->pressure_src_);
            e->pressure_.store(d & DISPATCH_MEMORYPRESSURE_CRITICAL ? 2 : d & DISPATCH_MEMORYPRESSURE_WARN ? 1 : 0,
                               std::memory_order_relaxed);
        });
        dispatch_resume(pressure_src_);
    }
    load_.total_ms = ms_since(t_load);
}

MTL::ComputeCommandEncoder* Engine::encoder(MTL::CommandBuffer* cb, uint32_t layer, int section, MTL::DispatchType type)
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

void Engine::after(MTL::ComputeCommandEncoder*& enc, std::string_view kernel)
{
    if (tel_ != Telemetry::kernels || timed_.empty())
        return;
    timed_.back().kernel = kernel;
    timed_.back().layer = cur_layer_;
    enc->endEncoding();
    enc = encoder(cur_cb_, cur_layer_, -1);
}

// Turns the step's encoder timestamps into per-layer section times. GPU ticks are converted with the CPU clock
// over the step, from timestamp pairs taken at its start and end.
void Engine::resolve_layer_times(uint64_t cpu0, uint64_t gpu0)
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
    if (tel_ == Telemetry::layers) {
        last_step_.layers.assign(cfg_.n_layer, LayerTimes{});
        for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
            last_step_.layers[l].layer = l;
            last_step_.layers[l].mla = !layers_[l]->kda;
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
        LayerTimes& lt = last_step_.layers[t.layer];
        (t.section == 0 ? lt.attn_ms : t.section == 1 ? lt.ffn_ms : t.section == 2 ? lt.hit_ms : lt.miss_ms) += ms;
    }
    timed_.clear();
    next_sample_ = 0;
}

// Scratch bytes a prefill allocates: per token of a chunk (the attention stages and the dense FFNs), per token of a
// group (the residual streams and the routed FFN's inputs and outputs), the expert batch buffers, and the attention
// scores and partial sums (bounded by the query group size, not the chunk) plus the decode scratch.
uint32_t Engine::sparse_group() const
{
    // The indexer's dot products (heads x pools per token) dominate: at most 64 MB of them, at most 32 tokens.
    const uint64_t dots = uint64_t(cfg_.indexer_heads) * (opts_.max_ctx / cfg_.indexer_kpool + 1) * 4;
    return uint32_t(std::clamp<uint64_t>((64ull << 20) / dots, 1, 32));
}

uint64_t Engine::sparse_scratch() const
{
    const uint64_t pools = opts_.max_ctx / cfg_.indexer_kpool + 1, L = sparse_list_stride(), H = cfg_.n_head;
    const uint64_t per_token = (cfg_.indexer_heads + 1) * pools + L + 1 + H * L + 8 * H * cfg_.kv_lora_rank;
    return uint64_t(sparse_group()) * per_token * 4;
}

uint64_t Engine::prefill_scratch(uint32_t chunk, uint32_t group) const
{
    const uint64_t E = cfg_.n_embd, hd = uint64_t(cfg_.hc) * E, inner = cfg_.kda_inner(), D = cfg_.kda_head_dim;
    const uint64_t H = cfg_.n_head, R = cfg_.kv_lora_rank, k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
    const uint64_t per_chunk_token = 3 * E + 24 + hd                                         // streams, hc
                                     + 11 * inner + 2 * D + 2 * H                                // KDA
                                     + 2 * cfg_.q_lora_rank + H * cfg_.head_k + 2 * R + 2 * H * R + H * cfg_.head_v
                                     + uint64_t(cfg_.indexer_heads) * (cfg_.indexer_dim + 1) + cfg_.indexer_dim // MLA
                                     + 3 * uint64_t(cfg_.n_ff)                                   // dense FFNs
                                     + std::max<uint64_t>({cfg_.n_ff, H * R, H * cfg_.head_k, inner}) / 2; // GEMM input
    const uint64_t per_group_token = 3 * E + 3 * hd + 48 + 2 * cfg_.n_expert + 32 + k + 16 + E / 2 + k * E;
    const uint64_t batch = std::min<uint64_t>(kBatchRows, uint64_t(group) * k) * (E / 2 + 2 * nff + nff / 2);
    return (uint64_t(chunk) * per_chunk_token + uint64_t(group) * per_group_token + batch) * 4 + sparse_scratch() +
           (96ull << 20);
}

void Engine::apply_pressure()
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

void Engine::reset()
{
    for (auto& b : kda_state_)
        if (b)
            std::memset(b->contents(), 0, b->length());
    for (auto& b : conv_state_)
        if (b)
            std::memset(b->contents(), 0, b->length());
    pos_ = 0;
}

void Engine::set_prefill_chunk(uint32_t chunk)
{
    if (chunk > built_chunk_)
        throw std::runtime_error(std::format("prefill chunk {} above the {} the engine was built with", chunk, built_chunk_));
    opts_.prefill_chunk = chunk;
}

void Engine::set_prefill_group(uint32_t group)
{
    if (group > built_group_)
        throw std::runtime_error(std::format("prefill group {} above the {} the engine was built with", group, built_group_));
    opts_.prefill_group = group;
}

void Engine::save_profile(const std::filesystem::path& path, bool merge) const
{
    const uint32_t nl = uint32_t(layers_.size()); // the next-token block included
    Counts counts = merge && std::filesystem::exists(path) ? load_profile(path, nl, cfg_.n_expert) : Counts(nl);
    const auto& usage = arena_->usage();
    counts.resize(std::max<size_t>(counts.size(), nl)); // rows of layers this engine did not load stay
    for (uint32_t l = 0; l < nl; ++l) {
        if (!arena_->slots(l))
            continue;
        counts[l].resize(cfg_.n_expert, 0);
        for (uint32_t e = 0; e < cfg_.n_expert; ++e)
            counts[l][e] += usage[l][e];
    }
    glm::save_profile(path, counts);
}

Engine::~Engine()
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

Engine::DevTensor Engine::upload(const std::string& name)
{
    auto ref = model_.find(name);
    if (!ref)
        throw std::runtime_error("missing tensor " + name);
    const gguf::Tensor& t = *ref.tensor;
    if (t.type != gguf::Type::F32 && !mtl::Gemv::supports(t.type))
        throw std::runtime_error(std::format("{}: unsupported type {}", name, gguf::traits(t.type).name));
    if (weight_chunks_.empty() || chunk_used_ + t.bytes > weight_chunks_.back()->length()) {
        const uint64_t size = std::max(kChunkBytes, (t.bytes + kAlign - 1) / kAlign * kAlign);
        weight_chunks_.emplace_back(ctx_.device()->newBuffer(size, MTL::ResourceStorageModeShared));
        if (!weight_chunks_.back())
            throw std::runtime_error("weight buffer allocation failed");
        chunk_used_ = 0;
    }
    DevTensor d;
    d.buf = weight_chunks_.back().get();
    d.off = chunk_used_;
    d.type = t.type;
    d.dims = t.dims;
    read_exact(ref.file->fd(), static_cast<char*>(d.buf->contents()) + d.off, t.bytes, t.offset);
    chunk_used_ += (t.bytes + kAlign - 1) / kAlign * kAlign;
    dense_bytes_ += t.bytes;
    return d;
}

ExpertArena::Part Engine::expert(const std::string& name, bool prefill)
{
    auto ref = model_.find(name);
    if (!ref)
        throw std::runtime_error("missing tensor " + name);
    const gguf::Tensor& t = *ref.tensor;
    if (t.dims.size() != 3 || t.dims[2] != cfg_.n_expert)
        throw std::runtime_error(name + ": expected [k, n, n_expert]");
    if (!mtl::Gemv::supports(t.type))
        throw std::runtime_error(std::format("{}: unsupported type {}", name, gguf::traits(t.type).name));
    if (prefill && !mtl::Gemm::supports(t.type))
        throw std::runtime_error(std::format("{}: type {} has no prefill GEMM (experts: IQ2_XXS, IQ2_S, IQ3_XXS, Q6_K, "
                                             "Q8_0, F16, F32)",
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
    e.bytes = t.bytes / cfg_.n_expert;
    e.type = t.type;
    e.k = uint32_t(t.dims[0]);
    e.n = uint32_t(t.dims[1]);
    return e;
}

MTL::Buffer* Engine::scratch(const std::string& name, uint64_t bytes)
{
    auto it = scratch_.find(name);
    if (it == scratch_.end() || it->second->length() < bytes) {
        mtl::Ref<MTL::Buffer> b(ctx_.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
        it = scratch_.insert_or_assign(name, std::move(b)).first;
    }
    return it->second.get();
}

MTL::ComputePipelineState* Engine::pso(const std::string& fn)
{
    if (repeat_n_ > 1)
        last_pso_ = fn;
    auto it = psos_.find(fn);
    if (it == psos_.end())
        it = psos_.emplace(fn, ctx_.pipeline(lib_.get(), fn)).first;
    return it->second.get();
}

void Engine::gemv(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
                  uint64_t y_off, uint32_t batch, uint32_t t)
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
void Engine::proj(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
                  uint64_t y_off, uint32_t T, uint32_t ldy)
{
    const uint32_t k = uint32_t(w.dims[0]), n = uint32_t(w.dims[1]);
    // A few tokens (decode, speculative verification) stay on the GEMV, which reads each weight once for them.
    if (T > kMaxDecodeTokens && (n >= 256 || (ldy && ldy != n)) && k % 32 == 0 && mtl::Gemm::supports(w.type)) {
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

MTL::Buffer* Engine::halves(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                            uint32_t ld)
{
    MTL::Buffer* h = scratch("xh", uint64_t(rows) * cols * 2);
    gemm_->to_half(enc, x, x_off, rows, cols, ld, h, 0);
    return h;
}

void Engine::set_ablation(std::vector<float> dirs, float scale)
{
    const uint64_t per_dir = uint64_t(cfg_.n_layer) * cfg_.n_embd;
    if (dirs.empty()) {
        ablate_k_ = 0;
        ablate_buf_ = {};
        ablate_layer_.clear();
        return;
    }
    if (dirs.size() % per_dir)
        throw std::invalid_argument(std::format("ablation directions: {} floats are not {} layers x k x {}",
                                                dirs.size(), cfg_.n_layer, cfg_.n_embd));
    const uint32_t k = uint32_t(dirs.size() / per_dir);
    ablate_layer_.assign(cfg_.n_layer, 0);
    for (uint32_t l = 0; l < cfg_.n_layer; ++l)
        ablate_layer_[l] = std::ranges::any_of(std::span(dirs).subspan(uint64_t(l) * k * cfg_.n_embd,
                                                                       uint64_t(k) * cfg_.n_embd),
                                               [](float x) { return x != 0.0f; });
    ablate_buf_ = mtl::Ref<MTL::Buffer>(
        ctx_.device()->newBuffer(dirs.data(), dirs.size() * 4, MTL::ResourceStorageModeShared));
    ablate_k_ = k;
    ablate_scale_ = scale;
}

std::vector<float> Engine::read_directions(const std::string& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error(std::format("cannot open {}", path));
    const auto bytes = uint64_t(in.tellg());
    if (bytes == 0 || bytes % 4)
        throw std::runtime_error(std::format("{} is not float32 values ({} bytes)", path, bytes));
    std::vector<float> v(bytes / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes)))
        throw std::runtime_error(std::format("reading {} failed", path));
    return v;
}

void Engine::ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, MTL::Buffer* x, uint32_t rows)
{
    if (!ablate_k_ || il >= ablate_layer_.size() || !ablate_layer_[il])
        return;
    const AblateArgs a{cfg_.n_embd, ablate_k_, ablate_scale_};
    enc->setComputePipelineState(pso("ablate_rows"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(ablate_buf_.get(), uint64_t(il) * ablate_k_ * cfg_.n_embd * 4, 1);
    enc->setBytes(&a, sizeof(a), 2);
    for (int r = repeats(last_pso_); r > 0; --r)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "ablate_rows");
}

void Engine::dispatch(MTL::ComputeCommandEncoder*& enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                      const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads)
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

void Engine::norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const DevTensor* w, MTL::Buffer* y, uint32_t rows,
                  uint32_t n, uint32_t wn, float eps, uint64_t x_off, uint64_t y_off, uint32_t ystride)
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

// x = mHC pre-mix of the streams R for T tokens; pc keeps each token's pre/post/comb for hc_post.
void Engine::hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                    MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T, uint64_t r_off)
{
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    MTL::Buffer* flat = scratch("flat", uint64_t(T) * hd * 4);
    MTL::Buffer* mixes = scratch("mixes", uint64_t(T) * 24 * 4);
    norm(enc, R, nullptr, flat, T, hd, 0, cfg_.rms_eps, r_off);
    proj(enc, fn, flat, 0, mixes, 0, T);
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
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

// out = mHC post-mix of x into the streams R for T tokens, with hc_pre's pc.
void Engine::hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* R, MTL::Buffer* out, MTL::Buffer* pc,
                     uint32_t T, uint64_t r_off)
{
    const HcArgs ha{cfg_.n_embd, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
    enc->setComputePipelineState(pso("hc_expand"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(R, r_off, 1);
    enc->setBuffer(pc, 0, 2);
    enc->setBuffer(out, r_off, 3);
    enc->setBytes(&ha, sizeof(ha), 4);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(cfg_.n_embd, T, 1), MTL::Size(256, 1, 1));
    after(enc, "hc_expand");
}

void Engine::kda(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
                 uint32_t T)
{
    const uint32_t inner = cfg_.kda_inner(), D = cfg_.kda_head_dim, H = cfg_.n_head;
    const bool probes = T == 1;
    if (D != 128)
        throw std::runtime_error("the KDA scan kernel needs head_dim 128");
    MTL::Buffer* qkv = scratch("qkv", uint64_t(T) * 3 * inner * 4);
    MTL::Buffer* qkvc = scratch("qkvc", uint64_t(T) * 3 * inner * 4);
    proj(enc, L.wq, xn, 0, qkv, 0, T, 3 * inner);
    proj(enc, L.wk, xn, 0, qkv, uint64_t(inner) * 4, T, 3 * inner);
    proj(enc, L.wv, xn, 0, qkv, uint64_t(inner) * 8, T, 3 * inner);
    // With snap_ (a speculative step), the state after the first snap_ tokens also goes to the alternate buffers.
    const uint32_t snap = snap_ && T > snap_ ? snap_ : 0;
    const ConvArgs ca{inner, cfg_.conv_kernel, T, snap};
    enc->setComputePipelineState(pso("kda_conv"));
    enc->setBuffer(qkv, 0, 0);
    enc->setBuffer(L.conv_q.buf, L.conv_q.off, 1);
    enc->setBuffer(L.conv_k.buf, L.conv_k.off, 2);
    enc->setBuffer(L.conv_v.buf, L.conv_v.off, 3);
    enc->setBuffer(conv_state_[il].get(), 0, 4);
    enc->setBuffer(qkvc, 0, 5);
    enc->setBytes(&ca, sizeof(ca), 6);
    enc->setBuffer(snap ? conv_state_alt_[il].get() : conv_state_[il].get(), 0, 7);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(3 * inner, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "kda_conv");
    if (probes) {
        probe(enc, "kda_q_conv", int(il), qkvc, inner);
        probe(enc, "kda_v_conv", int(il), qkvc, inner, uint64_t(inner) * 8);
    }

    MTL::Buffer* fa = scratch("fa", uint64_t(T) * D * 4);
    MTL::Buffer* graw = scratch("graw", uint64_t(T) * inner * 4);
    MTL::Buffer* braw = scratch("braw", uint64_t(T) * H * 4);
    MTL::Buffer* g = scratch("g", uint64_t(T) * inner * 4);
    MTL::Buffer* beta = scratch("beta", uint64_t(T) * H * 4);
    proj(enc, L.f_a, xn, 0, fa, 0, T);
    proj(enc, L.f_b, fa, 0, graw, 0, T);
    proj(enc, L.beta, xn, 0, braw, 0, T);
    const GateArgs ga{D, H, cfg_.kda_gate_lower_bound};
    enc->setComputePipelineState(pso("kda_gate"));
    enc->setBuffer(graw, 0, 0);
    enc->setBuffer(L.dt_b.buf, L.dt_b.off, 1);
    enc->setBuffer(L.ssm_a.buf, L.ssm_a.off, 2);
    enc->setBuffer(braw, 0, 3);
    enc->setBuffer(g, 0, 4);
    enc->setBuffer(beta, 0, 5);
    enc->setBytes(&ga, sizeof(ga), 6);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(inner, T, 1), MTL::Size(256, 1, 1));
    after(enc, "kda_gate");

    const L2Args la{D, 1e-6f, 3 * inner};
    enc->setComputePipelineState(pso("l2_norm"));
    enc->setBuffer(qkvc, 0, 0);
    enc->setBytes(&la, sizeof(la), 1);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(2 * H, T, 1), MTL::Size(256, 1, 1));
    after(enc, "l2_norm");

    MTL::Buffer* o = scratch("o", uint64_t(T) * inner * 4);
    const ScanArgs sa{D, T, 3 * inner, inner, H, snap};
    enc->setComputePipelineState(pso("kda_scan"));
    enc->setBuffer(qkvc, 0, 0);
    enc->setBuffer(qkvc, uint64_t(inner) * 4, 1);
    enc->setBuffer(qkvc, uint64_t(inner) * 8, 2);
    enc->setBuffer(g, 0, 3);
    enc->setBuffer(beta, 0, 4);
    enc->setBuffer(kda_state_[il].get(), 0, 5);
    enc->setBuffer(o, 0, 6);
    enc->setBytes(&sa, sizeof(sa), 7);
    enc->setBuffer(snap ? kda_state_alt_[il].get() : kda_state_[il].get(), 0, 8);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(H, D / 32, 1), MTL::Size(128, 1, 1));
    after(enc, "kda_scan");
    if (probes) {
        probe(enc, "kda_g1", int(il), g, inner);
        probe(enc, "kda_beta", int(il), beta, H);
        probe(enc, "kda_scan_out", int(il), o, inner);
    }

    MTL::Buffer* gab = scratch("ga", uint64_t(T) * D * 4);
    MTL::Buffer* g2 = scratch("g2", uint64_t(T) * inner * 4);
    MTL::Buffer* gated = scratch("gated", uint64_t(T) * inner * 4);
    proj(enc, L.g_a, xn, 0, gab, 0, T);
    proj(enc, L.g_b, gab, 0, g2, 0, T);
    if (probes)
        probe(enc, "kda_g2", int(il), g2, inner);
    const NormArgs na{D, D, cfg_.rms_eps, 0};
    enc->setComputePipelineState(pso("rms_norm_gate"));
    enc->setBuffer(o, 0, 0);
    enc->setBuffer(L.o_norm.buf, L.o_norm.off, 1);
    enc->setBuffer(g2, 0, 2);
    enc->setBuffer(gated, 0, 3);
    enc->setBytes(&na, sizeof(na), 4);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(T * H, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "rms_norm_gate");
    if (probes)
        probe(enc, "attn_output", int(il), gated, inner);
    proj(enc, L.wo, gated, 0, att, 0, T);
}

// Attention of `tokens` consecutive query positions (rows of H heads of q, latent width R) over a half latent cache
// read once for all heads: scores by GEMM, softmax over each row's visible keys (row r of token i sees
// first_kv + i keys), then the output split over key ranges and summed.
void Engine::attend(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off,
                    MTL::Buffer* out, uint64_t out_off, uint32_t tokens, uint32_t first_kv)
{
    const uint32_t H = cfg_.n_head, R = cfg_.kv_lora_rank, rows = tokens * H;
    const uint32_t n_kv = first_kv + tokens - 1;
    // At most 16 splits of at least 256 keys, whole 64-key tiles.
    const uint32_t keys_per_split = std::max<uint32_t>(256, ((n_kv + 15) / 16 + 63) / 64 * 64);
    const uint32_t splits = (n_kv + keys_per_split - 1) / keys_per_split;
    MTL::Buffer* scores = scratch("scores", uint64_t(rows) * n_kv * 4);
    MTL::Buffer* partial = scratch("pv_part", uint64_t(splits) * rows * R * 4);
    gemm_->encode(enc, gguf::Type::F16, R, n_kv, rows, cache, 0, halves(enc, q, q_off, rows, R, 0), 0, scores, 0);
    after(enc, "attn_scores");
    const SoftmaxArgs sa{n_kv, 1.0f / std::sqrt(float(cfg_.head_k)), first_kv, H};
    enc->setComputePipelineState(pso("attn_softmax"));
    enc->setBuffer(scores, 0, 0);
    enc->setBytes(&sa, sizeof(sa), 1);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "attn_softmax");
    const PvArgs pa{rows, R, n_kv, keys_per_split};
    enc->setComputePipelineState(pso("attn_pv"));
    enc->setBuffer(scores, 0, 0);
    enc->setBuffer(cache, 0, 1);
    enc->setBuffer(partial, 0, 2);
    enc->setBytes(&pa, sizeof(pa), 3);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(R / 64, (rows + 63) / 64, splits), MTL::Size(128, 1, 1));
    after(enc, "attn_pv");
    const uint32_t sum[2] = {rows * R, splits};
    enc->setComputePipelineState(pso("sum_splits"));
    enc->setBuffer(partial, 0, 0);
    enc->setBuffer(out, out_off, 1);
    enc->setBytes(sum, sizeof(sum), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(rows * R, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "sum_splits");
}

// MLA for T tokens at positions pos0..: projections over all of them; attention in groups of query tokens while
// the indexer keeps every key, token by token once it selects.
void Engine::mla(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
                 uint32_t T, uint32_t pos0)
{
    const uint32_t H = cfg_.n_head, R = cfg_.kv_lora_rank, QR = cfg_.q_lora_rank;
    const uint64_t qrow = uint64_t(H) * cfg_.head_k, vrow = uint64_t(H) * cfg_.head_v, lrow = uint64_t(H) * R;
    MTL::Buffer* qr = scratch("qr", uint64_t(T) * QR * 4);
    MTL::Buffer* qrn = scratch("qrn", uint64_t(T) * QR * 4);
    MTL::Buffer* q = scratch("q", T * qrow * 4);
    MTL::Buffer* kv = scratch("kv", uint64_t(T) * R * 4);
    MTL::Buffer* kvn = scratch("kvn", uint64_t(T) * R * 4);
    MTL::Buffer* qabs = scratch("qabs", T * lrow * 4);
    MTL::Buffer* lat = scratch("lat", T * lrow * 4);
    MTL::Buffer* vout = scratch("vout", T * vrow * 4);

    proj(enc, L.q_a, xn, 0, qr, 0, T);
    norm(enc, qr, &L.q_a_norm, qrn, T, QR, QR, cfg_.rms_eps);
    proj(enc, L.q_b, qrn, 0, q, 0, T);
    proj(enc, L.kv_a, xn, 0, kv, 0, T);
    norm(enc, kv, &L.kv_a_norm, kvn, T, R, R, cfg_.rms_eps);

    const uint32_t n = T * R;
    enc->setComputePipelineState(pso("store_half"));
    enc->setBuffer(kvn, 0, 0);
    enc->setBuffer(kv_cache_[il].get(), uint64_t(pos0) * R * 2, 1);
    enc->setBytes(&n, sizeof(n), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "store_half");

    // Per-head matrices (k_b: head_k -> R, v_b: R -> head_v): one GEMM per head over all tokens, or the batched
    // GEMV for one token.
    auto per_head = [&](const DevTensor& w, MTL::Buffer* x, uint32_t xw, MTL::Buffer* y, uint32_t yw) {
        if (T <= kMaxDecodeTokens) {
            for (uint32_t t = 0; t < T; ++t)
                gemv(enc, w, x, uint64_t(t) * H * xw * 4, y, uint64_t(t) * H * yw * 4, H);
            return;
        }
        const uint32_t k = uint32_t(w.dims[0]), nr = uint32_t(w.dims[1]);
        const uint64_t mat = mtl::Gemv::row_bytes(w.type, k) * nr;
        MTL::Buffer* xh = halves(enc, x, 0, T, H * xw, 0);
        for (uint32_t h = 0; h < H; ++h)
            gemm_->encode(enc, w.type, k, nr, T, w.buf, w.off + h * mat, xh, uint64_t(h) * xw * 2, y,
                          uint64_t(h) * yw * 4, H * yw, H * xw);
        after(enc, "per_head_gemm");
    };
    per_head(L.k_b, q, cfg_.head_k, qabs, R);
    indexer_project(enc, L, il, xn, qrn, T, pos0);
    const uint32_t kp = cfg_.indexer_kpool, n_top = cfg_.indexer_top_k / kp;
    // Query tokens per attention group: up to 16, fewer when the scores would pass 4M floats.
    const uint32_t group = std::clamp<uint32_t>((4u << 20) / (H * (pos0 + T)), 1, 16);
    uint32_t t = 0;
    while (t < T && (pos0 + t + 1) / kp <= n_top) {
        uint32_t g = 1;
        while (g < group && t + g < T && (pos0 + t + g + 1) / kp <= n_top)
            ++g;
        attend(enc, kv_cache_[il].get(), qabs, t * lrow * 4, lat, t * lrow * 4, g, pos0 + t + 1);
        t += g;
    }
    // Past the dense window, groups of a size independent of the context: the selection bounds their scratch.
    const uint32_t sgroup = sparse_group();
    while (t < T) {
        const uint32_t g = std::min(sgroup, T - t);
        if (!indexer_select(enc, il, t, pos0 + t, g))
            throw std::logic_error("sparse attention group inside the dense window");
        attend_sparse(enc, kv_cache_[il].get(), qabs, t * lrow * 4, lat, t * lrow * 4, g);
        t += g;
    }
    per_head(L.v_b, lat, R, vout, cfg_.head_v);
    proj(enc, L.wo, vout, 0, att, 0, T);
}

// Indexer inputs for T tokens at positions pos0..: their query and weight projections, their key|gate rows of the
// layer's indexer cache, and the pools those rows complete.
void Engine::indexer_project(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                             MTL::Buffer* qrn, uint32_t T, uint32_t pos0)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    MTL::Buffer* iq = scratch("iq", uint64_t(T) * Hi * Di * 4);
    MTL::Buffer* ikr = scratch("ikr", uint64_t(T) * Di * 4);
    MTL::Buffer* iw = scratch("iw", uint64_t(T) * Hi * 4);
    MTL::Buffer* cache = idx_cache_[il].get();
    const uint64_t row = uint64_t(pos0) * 2 * Di * 4;

    if (qrn) // else only the cache rows and pools (no queries)
        proj(enc, L.idx_q_b, qrn, 0, iq, 0, T);
    proj(enc, L.idx_k, xn, 0, ikr, 0, T);
    const LayerNormArgs na{Di, cfg_.norm_eps, 2 * Di};
    enc->setComputePipelineState(pso("layer_norm"));
    enc->setBuffer(ikr, 0, 0);
    enc->setBuffer(L.idx_k_norm.buf, L.idx_k_norm.off, 1);
    enc->setBuffer(L.idx_k_norm_b.buf, L.idx_k_norm_b.off, 2);
    enc->setBuffer(cache, row, 3);
    enc->setBytes(&na, sizeof(na), 4);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "layer_norm");
    proj(enc, L.idx_gate, xn, 0, cache, row + Di * 4, T, 2 * Di);
    if (qrn)
        proj(enc, L.idx_proj, xn, 0, iw, 0, T);

    const uint32_t first = pos0 / kp, last = (pos0 + T) / kp;
    if (last > first) {
        const PoolArgs pa{Di, kp, first};
        enc->setComputePipelineState(pso("idx_pool"));
        enc->setBuffer(cache, 0, 0);
        enc->setBuffer(L.idx_ape.buf, L.idx_ape.off, 1);
        enc->setBuffer(idx_pooled_[il].get(), 0, 2);
        enc->setBytes(&pa, sizeof(pa), 3);
        for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(Di, last - first, 1), MTL::Size(Di, 1, 1));
        after(enc, "idx_pool");
    }
}

// For tokens t..t+count-1 (at positions pos..) of the last indexer_project: once more pools are complete than the
// selection keeps, writes each token's cell mask (rows of max_ctx bytes) of its top pools plus the incomplete tail.
// Returns whether the masks apply (whether token t has more pools; later tokens then do too).
bool Engine::indexer_select(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t t, uint32_t pos, uint32_t count)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    const uint32_t n_pool = (pos + 1) / kp, n_top = cfg_.indexer_top_k / kp;
    if (n_pool <= n_top)
        return false;
    const uint32_t stride = opts_.max_ctx / kp + 1, last_pools = (pos + count) / kp;
    MTL::Buffer* score = scratch("idx_score", uint64_t(count) * stride * 4);
    const ScoreArgs sa{Di, Hi, pos + 1, kp, 1.0f / std::sqrt(float(Di * Hi)), stride};
    // Dot products of every head with every pool as a float matmul, then the relu-weighted sum over heads.
    if (Hi != 32)
        throw std::runtime_error("the indexer's matmul expects 32 heads");
    MTL::Buffer* dots = scratch("idx_dots", uint64_t(count) * Hi * stride * 4);
    const uint32_t da[4] = {Hi, Di, last_pools, stride};
    enc->setComputePipelineState(pso("idx_dots"));
    enc->setBuffer(scratch("iq", 0), uint64_t(t) * Hi * Di * 4, 0);
    enc->setBuffer(idx_pooled_[il].get(), 0, 1);
    enc->setBuffer(dots, 0, 2);
    enc->setBytes(da, sizeof(da), 3);
    for (int rep_ = repeats("idx_dots"); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size((last_pools + 63) / 64, 1, count), MTL::Size(128, 1, 1));
    after(enc, "idx_dots");
    enc->setComputePipelineState(pso("idx_reduce"));
    enc->setBuffer(dots, 0, 0);
    enc->setBuffer(scratch("iw", 0), uint64_t(t) * Hi * 4, 1);
    enc->setBuffer(score, 0, 2);
    enc->setBytes(&sa, sizeof(sa), 3);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(last_pools, count, 1), MTL::Size(256, 1, 1));
    after(enc, "idx_reduce");
    if (pos == pos_ && count == 1)
        probe(enc, "indexer_score", int(il), score, n_pool);
    const SelectArgs sel{pos + 1, n_top, kp, stride, sparse_list_stride()};
    enc->setComputePipelineState(pso("idx_select"));
    enc->setBuffer(score, 0, 0);
    enc->setBytes(&sel, sizeof(sel), 1);
    enc->setBuffer(scratch("idx_list", uint64_t(count) * sparse_list_stride() * 4), 0, 2);
    enc->setBuffer(scratch("idx_count", uint64_t(count) * 4), 0, 3);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "idx_select");
    return true;
}

// Attention of `tokens` consecutive query positions (rows of H heads of q) over the cache rows the indexer listed for
// each (indexer_select): gathered scores, softmax over each token's list, gathered sum (split over the list for
// few tokens, so a single token still fills the GPU).
void Engine::attend_sparse(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off,
                           MTL::Buffer* out, uint64_t out_off, uint32_t tokens)
{
    const uint32_t H = cfg_.n_head, R = cfg_.kv_lora_rank, rows = tokens * H, L = sparse_list_stride();
    const uint32_t splits = tokens < 8 ? 8 : 1;
    MTL::Buffer* list = scratch("idx_list", uint64_t(tokens) * L * 4);
    MTL::Buffer* count = scratch("idx_count", uint64_t(tokens) * 4);
    MTL::Buffer* scores = scratch("sp_scores", uint64_t(rows) * L * 4);
    MTL::Buffer* partial = scratch("sp_part", uint64_t(splits) * rows * R * 4);
    MTL::Buffer* qh = halves(enc, q, q_off, rows, R, 0);
    const GatherArgs ga{H, R, L, splits, tokens};
    enc->setComputePipelineState(pso("attn_gather_scores"));
    enc->setBuffer(qh, 0, 0);
    enc->setBuffer(cache, 0, 1);
    enc->setBuffer(list, 0, 2);
    enc->setBuffer(count, 0, 3);
    enc->setBuffer(scores, 0, 4);
    enc->setBytes(&ga, sizeof(ga), 5);
    for (int rep_ = repeats("attn_gather_scores"); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size((L + 31) / 32, 1, tokens), MTL::Size(128, 1, 1));
    after(enc, "attn_gather_scores");
    const ListSoftmaxArgs sa{L, 1.0f / std::sqrt(float(cfg_.head_k)), H};
    enc->setComputePipelineState(pso("attn_softmax_list"));
    enc->setBuffer(scores, 0, 0);
    enc->setBuffer(count, 0, 1);
    enc->setBytes(&sa, sizeof(sa), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "attn_softmax_list");
    enc->setComputePipelineState(pso("attn_gather_pv"));
    enc->setBuffer(scores, 0, 0);
    enc->setBuffer(cache, 0, 1);
    enc->setBuffer(list, 0, 2);
    enc->setBuffer(count, 0, 3);
    enc->setBuffer(partial, 0, 4);
    enc->setBytes(&ga, sizeof(ga), 5);
    for (int rep_ = repeats("attn_gather_pv"); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(R / 128, 1, tokens * splits), MTL::Size(256, 1, 1));
    after(enc, "attn_gather_pv");
    const uint32_t sum[2] = {rows * R, splits};
    enc->setComputePipelineState(pso("sum_splits"));
    enc->setBuffer(partial, 0, 0);
    enc->setBuffer(out, out_off, 1);
    enc->setBytes(sum, sizeof(sum), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(rows * R, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "sum_splits");
}

struct HcStepArgs {
    uint32_t n_embd, expand, parts, iters;
    float eps, rms_eps;
};

void Engine::hc_step(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* post_x, MTL::Buffer* R, MTL::Buffer* R_out,
                     const DevTensor& fn, const DevTensor& scale, const DevTensor& base, MTL::Buffer* x, MTL::Buffer* pc,
                     uint32_t T)
{
    constexpr uint32_t kParts = 32; // at most 64 (hc_step_b)
    const HcStepArgs a{cfg_.n_embd, post_x ? 1u : 0u, kParts, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps};
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

void Engine::ffn_dense(MTL::ComputeCommandEncoder*& enc, const DevTensor& gate, const DevTensor& up,
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
    proj(enc, down, m, 0, out, out_off, T);
}

// Serves the MoE layers of a committed command buffer in order: waits until a layer's router has run, makes its
// selected experts resident and writes their slots, then lets the GPU continue.
void Engine::serve_moe(MTL::CommandBuffer* cb)
{
    const uint32_t k = cfg_.n_expert_used;
    const uint64_t rows = kMaxDecodeTokens * 16 * 4;
    auto* sel = static_cast<int*>(scratch("msel", rows)->contents());
    auto* w = static_cast<float*>(scratch("ew", rows)->contents());
    auto* hit_ids = static_cast<const int32_t*>(scratch("eids_hit", rows)->contents());
    auto* miss_ids = static_cast<int32_t*>(scratch("eids_miss", rows)->contents());
    auto* miss_rank = static_cast<int32_t*>(scratch("mrank", rows)->contents());
    auto* indirect = static_cast<uint32_t*>(
        scratch("mind", uint64_t(cfg_.n_layer + 1) * kMaxDecodeTokens * 36)->contents());
    for (const auto [il, v, T] : pending_moe_) {
        const auto t_wait = Clock::now();
        for (uint32_t spins = 0; routed_->signaledValue() < v; ++spins)
            if (spins % 4096 == 4095 && cb->status() == MTL::CommandBufferStatusError)
                throw std::runtime_error(std::format("command buffer failed: {}",
                                                     cb->error() ? cb->error()->localizedDescription()->utf8String()
                                                                 : "?"));
        if (probe_) {
            probe_("ffn_moe_probs_biased", int(il),
                   static_cast<const float*>(scratch("mbiased", cfg_.n_expert * 4)->contents()), cfg_.n_expert);
            std::array<float, 16> self{};
            std::copy_n(sel, k, self.begin());
            probe_("moe_sel", int(il), self.data(), k);
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
    }
    pending_moe_.clear();
}

void Engine::finish(MTL::CommandBuffer* cb)
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

void Engine::submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc)
{
    enc->endEncoding();
    finish(cb);
    cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    enc = encoder(cb, cur_layer_, -1);
}

void Engine::probe(MTL::ComputeCommandEncoder*& enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
                   uint64_t off)
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

void Engine::deliver_probes()
{
    for (const auto& p : pending_probes_)
        probe_(p.name, p.layer, static_cast<const float*>(p.copy->contents()), p.n);
    pending_probes_.clear();
}

const std::vector<float>& Engine::step(int32_t token)
{
    const int32_t t[1] = {token};
    decode_tokens(t);
    return logits_;
}

const std::vector<float>& Engine::step2(int32_t a, int32_t b)
{
    const int32_t t[2] = {a, b};
    decode_tokens(t);
    return logits_all_;
}

void Engine::rollback()
{
    if (last_tokens_ != 2)
        throw std::runtime_error("rollback only follows a two-token step");
    for (uint32_t il = 0; il < cfg_.n_layer; ++il)
        if (layers_[il]->kda) {
            std::swap(kda_state_[il], kda_state_alt_[il]);
            std::swap(conv_state_[il], conv_state_alt_[il]);
        }
    --pos_;
    last_tokens_ = 1;
}

// One decode step over T (<= kMaxDecodeTokens) consecutive tokens; with two, the KDA states after the first are
// kept for rollback().
void Engine::decode_tokens(std::span<const int32_t> tokens)
{
    const uint32_t T = uint32_t(tokens.size());
    if (T == 0 || T > kMaxDecodeTokens || (T > 1 && (probe_ || stream_override_ || route_override_)))
        throw std::runtime_error("bad decode step");
    apply_pressure();
    if (pos_ + T > opts_.max_ctx)
        throw std::runtime_error("context full");
    for (int32_t token : tokens)
        if (token < 0 || uint32_t(token) >= cfg_.n_vocab)
            throw std::runtime_error("token out of range");
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
    snap_ = T > 1 ? 1 : 0;
    last_pos0_ = pos_;

    step_start_ = Clock::now();
    encoding_ = true;
    const ExpertArena::Stats before = arena_->stats();
    const Stats stats_before = stats_;
    last_step_ = StepStats{};
    last_step_.pos = pos_;
    last_step_.token = tokens[0];
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const uint64_t rows = kMaxDecodeTokens; // fixed sizes: the next-token block reads "xn" after the step
    MTL::Buffer* x = scratch("x", rows * E * 4);
    MTL::Buffer* xn = scratch("xn", rows * E * 4);
    MTL::Buffer* att = scratch("att", rows * E * 4);
    MTL::Buffer* ffo = scratch("ffo", rows * E * 4);
    MTL::Buffer* pc = scratch("pc", rows * 24 * 4);
    scratch("R", rows * hd * 4);
    scratch("R2", rows * hd * 4);

    {
        const gguf::Tensor& t = *token_embd_.tensor;
        const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
        std::vector<std::byte> row(rb);
        for (uint32_t i = 0; i < T; ++i) {
            read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(tokens[i]) * rb);
            quant::dequant_row(t.type, row.data(), static_cast<float*>(x->contents()) + uint64_t(i) * E, E);
        }
    }

    if (probe_)
        probe_("inp_embd", -1, static_cast<const float*>(x->contents()), E);
    MTL::Timestamp cpu0 = 0, gpu0 = 0;
    if (timing())
        ctx_.device()->sampleTimestamps(&cpu0, &gpu0);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    cur_layer_ = cfg_.n_layer;
    MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
    dispatch(enc, "hc_init", {x, scratch("R", rows * hd * 4)}, &ha, sizeof(ha), MTL::Size(E, T, 1), MTL::Size(256, 1, 1));

    // Stream steps fused (hc_step) unless something observes the stream between them.
    const bool fused = !probe_ && !stream_override_;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        cur_layer_ = il;
        if (stream_override_) {
            submit(cb, enc);
            stream_override_(int(il), static_cast<float*>(scratch("R", rows * hd * 4)->contents()), hd);
        }
        if (timing()) {
            enc->endEncoding();
            enc = encoder(cb, il, 0);
        }
        if (!fused)
            hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, scratch("R", rows * hd * 4), x, pc, T);
        else if (il == 0)
            hc_step(enc, nullptr, scratch("R", rows * hd * 4), nullptr, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, x, pc,
                    T);
        probe(enc, "hc_attn_pre", int(il), x, E);
        norm(enc, x, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
        probe(enc, "attn_norm", int(il), xn, E);
        if (L.kda)
            kda(enc, L, il, xn, att, T);
        else
            mla(enc, L, il, xn, att, T, pos_);
        probe(enc, L.kda ? "kda_out" : "attn_out", int(il), att, E);
        if (timing()) {
            enc->endEncoding();
            enc = encoder(cb, il, 1);
        }
        if (fused) {
            hc_step(enc, att, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), L.hc_ffn_fn, L.hc_ffn_scale,
                    L.hc_ffn_base, x, pc, T);
            std::swap(scratch_["R"], scratch_["R2"]);
        } else {
            hc_post(enc, att, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), pc, T);
            std::swap(scratch_["R"], scratch_["R2"]);
            hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, scratch("R", rows * hd * 4), x, pc, T);
        }
        probe(enc, "hc_ffn_pre", int(il), x, E);
        probe(enc, "hc_mixes", int(il), scratch("mixes", 32 * 4), 24);
        norm(enc, x, &L.ffn_norm, xn, T, E, E, cfg_.rms_eps);
        probe(enc, "ffn_norm", int(il), xn, E);
        if (L.dense) {
            ffn_dense(enc, L.ffn_gate, L.ffn_up, L.ffn_down, xn, ffo, cfg_.swiglu_clamp_shexp[il], T);
        } else {
            moe_decode(cb, enc, L, il, xn, ffo, T);
        }
        probe(enc, "ffn_out", int(il), ffo, E);
        ablate(enc, il, ffo, T);
        ablate(enc, il, scratch("R", rows * hd * 4), T * cfg_.hc);
        if (fused && il + 1 < cfg_.n_layer) {
            const Layer& N = *layers_[il + 1];
            hc_step(enc, ffo, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), N.hc_attn_fn, N.hc_attn_scale,
                    N.hc_attn_base, x, pc, T);
        } else {
            hc_post(enc, ffo, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), pc, T);
        }
        std::swap(scratch_["R"], scratch_["R2"]);
        probe(enc, "l_out", int(il), scratch("R", rows * hd * 4), hd);
    }

    cur_layer_ = cfg_.n_layer;
    if (timing()) {
        enc->endEncoding();
        enc = encoder(cb, cur_layer_, -1);
    }
    dispatch(enc, "hc_mean", {scratch("R", rows * hd * 4), x}, &ha, sizeof(ha), MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    norm(enc, x, &output_norm_, xn, T, E, E, cfg_.rms_eps);
    MTL::Buffer* lg = scratch("logits", rows * cfg_.n_vocab * 4);
    gemv(enc, output_, xn, 0, lg, 0, 1, T);
    enc->endEncoding();
    finish(cb);
    snap_ = 0;
    const auto* lgp = static_cast<const float*>(lg->contents());
    logits_all_.assign(lgp, lgp + uint64_t(T) * cfg_.n_vocab);
    std::memcpy(logits_.data(), lgp + uint64_t(T - 1) * cfg_.n_vocab, uint64_t(cfg_.n_vocab) * 4);
    stats_.expert_read_ms = arena_->stats().read_ms;
    stats_.expert_bytes = arena_->stats().read_bytes;
    pos_ += T;
    last_tokens_ = T;
    pool->release();
    stats_.step_ms += ms_since(step_start_);
    const ExpertArena::Stats& after = arena_->stats();
    last_step_.wall_ms = ms_since(step_start_);
    last_step_.encode_ms = stats_.encode_ms - stats_before.encode_ms;
    last_step_.gpu_ms = stats_.gpu_exec_ms - stats_before.gpu_exec_ms;
    last_step_.read_ms = after.read_ms - before.read_ms;
    last_step_.hits = after.hits - before.hits;
    last_step_.misses = after.misses - before.misses;
    last_step_.read_bytes = after.read_bytes - before.read_bytes;
    if (timing())
        resolve_layer_times(cpu0, gpu0);
}


// The next-token block's attention cache rows for T positions from pos0: position i's input is its hidden state
// (row i of h) with the following token's embedding (row i of e). Only what later attention reads: the latent row
// and the indexer's key row and pools; no attention output, no FFN.
void Engine::mtp_fill(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* h, uint64_t h_off, MTL::Buffer* e, uint64_t e_off,
                      uint32_t T, uint32_t pos0)
{
    const Layer& L = *layers_[cfg_.n_layer];
    const uint32_t E = cfg_.n_embd, R = cfg_.kv_lora_rank, il = cfg_.n_layer;
    MTL::Buffer* cat = scratch("mf_cat", uint64_t(T) * 2 * E * 4);
    MTL::Buffer* r = scratch("mf_r", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("mf_xn", uint64_t(T) * E * 4);
    MTL::Buffer* kv = scratch("mf_kv", uint64_t(T) * R * 4);
    MTL::Buffer* kvn = scratch("mf_kvn", uint64_t(T) * R * 4);
    norm(enc, e, &L.enorm, cat, T, E, E, cfg_.rms_eps, e_off, 0, 2 * E);
    norm(enc, h, &L.hnorm, cat, T, E, E, cfg_.rms_eps, h_off, uint64_t(E) * 4, 2 * E);
    proj(enc, L.eh_proj, cat, 0, r, 0, T);
    norm(enc, r, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
    proj(enc, L.kv_a, xn, 0, kv, 0, T);
    norm(enc, kv, &L.kv_a_norm, kvn, T, R, R, cfg_.rms_eps);
    const uint32_t n = T * R;
    enc->setComputePipelineState(pso("store_half"));
    enc->setBuffer(kvn, 0, 0);
    enc->setBuffer(kv_cache_[il].get(), uint64_t(pos0) * R * 2, 1);
    enc->setBytes(&n, sizeof(n), 2);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    indexer_project(enc, L, il, xn, nullptr, T, pos0);
}

int32_t Engine::mtp_draft(int32_t next)
{
    const int32_t t[1] = {next};
    return mtp_run(t, 0);
}

int32_t Engine::mtp_draft(std::span<const int32_t> next)
{
    return mtp_run(next, 0);
}

std::span<const float> Engine::mtp_logits()
{
    return {static_cast<const float*>(scratch("mtp_logits", uint64_t(cfg_.n_vocab) * 4)->contents()), cfg_.n_vocab};
}

void Engine::mtp_seed(int32_t next)
{
    if (!has_mtp() || pos_ == 0)
        return;
    const int32_t t[1] = {next};
    mtp_run(t, pos_ - 1 - last_pos0_);
}

uint64_t Engine::checkpoint_bytes() const
{
    uint64_t n = has_mtp() ? uint64_t(cfg_.n_embd) * 4 : 0;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il)
        for (const auto* b : {kda_state_[il].get(), conv_state_[il].get()})
            if (b)
                n += b->length();
    return n;
}

Engine::Checkpoint Engine::checkpoint()
{
    Checkpoint cp;
    cp.pos = pos_;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il)
        for (auto* b : {kda_state_[il].get(), conv_state_[il].get()})
            if (b) {
                const auto* p = static_cast<const std::byte*>(b->contents());
                cp.state.insert(cp.state.end(), p, p + b->length());
            }
    if (has_mtp() && pos_ > 0) {
        const uint32_t E = cfg_.n_embd;
        const auto* h = static_cast<const float*>(scratch("xn", uint64_t(kMaxDecodeTokens) * E * 4)->contents());
        cp.hidden.assign(h + uint64_t(pos_ - 1 - last_pos0_) * E, h + uint64_t(pos_ - last_pos0_) * E);
    }
    return cp;
}

void Engine::restore(const Checkpoint& cp)
{
    if (cp.pos > opts_.max_ctx)
        throw std::runtime_error("checkpoint past the context size");
    size_t off = 0;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il)
        for (auto* b : {kda_state_[il].get(), conv_state_[il].get()})
            if (b) {
                if (off + b->length() > cp.state.size())
                    throw std::runtime_error("checkpoint from another engine");
                std::memcpy(b->contents(), cp.state.data() + off, b->length());
                off += b->length();
            }
    if (off != cp.state.size())
        throw std::runtime_error("checkpoint from another engine");
    pos_ = cp.pos;
    last_pos0_ = pos_ > 0 ? pos_ - 1 : 0;
    last_tokens_ = 1;
    if (!cp.hidden.empty()) {
        auto* h = static_cast<float*>(scratch("xn", uint64_t(kMaxDecodeTokens) * cfg_.n_embd * 4)->contents());
        std::memcpy(h, cp.hidden.data(), cp.hidden.size() * 4);
    }
}

// The next-token block over the last step's hidden rows row0.. (positions last_pos0_ + row0..), each followed by
// next[i]; returns the guess after the last.
int32_t Engine::mtp_run(std::span<const int32_t> next, uint32_t row0)
{
    const uint32_t T = uint32_t(next.size());
    if (!has_mtp() || pos_ == 0)
        return -1;
    if (T == 0 || row0 + T > last_tokens_)
        throw std::runtime_error("next-token block: more tokens than the last step had");
    for (int32_t token : next)
        if (token < 0 || uint32_t(token) >= cfg_.n_vocab)
            throw std::runtime_error("token out of range");
    const Layer& L = *layers_[cfg_.n_layer];
    const uint32_t E = cfg_.n_embd, il = cfg_.n_layer, pos = last_pos0_ + row0;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const uint64_t rows = kMaxDecodeTokens;
    MTL::Buffer* e = scratch("mtp_e", rows * E * 4);
    MTL::Buffer* cat = scratch("mtp_cat", rows * 2 * E * 4);
    MTL::Buffer* r = scratch("mtp_r", rows * E * 4);
    MTL::Buffer* r2 = scratch("mtp_r2", rows * E * 4);
    MTL::Buffer* xn = scratch("mtp_xn", rows * E * 4);
    MTL::Buffer* att = scratch("mtp_att", rows * E * 4);
    MTL::Buffer* ffo = scratch("mtp_ffo", rows * E * 4);
    MTL::Buffer* lg = scratch("mtp_logits", uint64_t(cfg_.n_vocab) * 4);
    {
        const gguf::Tensor& t = *token_embd_.tensor;
        const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
        std::vector<std::byte> row(rb);
        for (uint32_t i = 0; i < T; ++i) {
            read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(next[i]) * rb);
            quant::dequant_row(t.type, row.data(), static_cast<float*>(e->contents()) + uint64_t(i) * E, E);
        }
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    cur_layer_ = cfg_.n_layer;
    MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
    // Row i: [enorm(embedding of next[i]); hnorm(the step's post-norm hidden state row row0 + i, still in "xn")]
    // -> eh_proj, at the step's position row0 + i.
    MTL::Buffer* h = scratch("xn", rows * E * 4);
    for (uint32_t i = 0; i < T; ++i) {
        norm(enc, e, &L.enorm, cat, 1, E, E, cfg_.rms_eps, uint64_t(i) * E * 4, uint64_t(i) * 2 * E * 4);
        norm(enc, h, &L.hnorm, cat, 1, E, E, cfg_.rms_eps, uint64_t(row0 + i) * E * 4, uint64_t(2 * i + 1) * E * 4);
    }
    gemv(enc, L.eh_proj, cat, 0, r, 0, 1, T);
    norm(enc, r, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
    mla(enc, L, il, xn, att, T, pos);
    const uint32_t n = T * E;
    dispatch(enc, "add", {r, att, r2}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    norm(enc, r2, &L.ffn_norm, xn, T, E, E, cfg_.rms_eps);
    moe_decode(cb, enc, L, il, xn, ffo, T);
    dispatch(enc, "add", {r2, ffo, r}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    // Only the last row's guess is needed.
    norm(enc, r, &L.head_norm, xn, 1, E, E, cfg_.rms_eps, uint64_t(T - 1) * E * 4);
    gemv(enc, output_, xn, 0, lg, 0);
    enc->endEncoding();
    finish(cb);
    const auto* l = static_cast<const float*>(lg->contents());
    const int32_t draft = int32_t(std::max_element(l, l + cfg_.n_vocab) - l);
    pool->release();
    return draft;
}

// One token's routed FFN (plus the shared expert) of layer il on xn into ffo: GPU routing, the resident experts at
// once and the misses after the CPU has read them (serve_moe).
void Engine::moe_decode(MTL::CommandBuffer* cb, MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il,
                        MTL::Buffer* xn, MTL::Buffer* ffo, uint32_t T)
{
    if (T > kMaxDecodeTokens)
        throw std::runtime_error("too many tokens for a decode step");
    const uint32_t E = cfg_.n_embd, k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
    const uint32_t tl = std::min(il, cfg_.n_layer); // telemetry: the next-token block counts as the rest
    MTL::Buffer* sho = scratch("sho", uint64_t(T) * E * 4);
    MTL::Buffer* rlog = scratch("rlog", uint64_t(T) * cfg_.n_expert * 4);
    // Sized for kMaxDecodeTokens whatever T is: serve_moe reads them (a larger request would reallocate).
    const uint64_t id_rows = kMaxDecodeTokens * 16 * 4;
    MTL::Buffer* ew = scratch("ew", id_rows);
    MTL::Buffer* hit_ids = scratch("eids_hit", id_rows);
    MTL::Buffer* miss_ids = scratch("eids_miss", id_rows);
    MTL::Buffer* miss_rank = scratch("mrank", id_rows);
    // Per layer and token: x, y, z of the gate, up and down miss dispatches (moe_route writes z, the misses).
    MTL::Buffer* indirect = scratch("mind", uint64_t(cfg_.n_layer + 1) * kMaxDecodeTokens * 36);
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

    gemv(enc, L.router, xn, 0, rlog, 0, 1, T);
    const RouteArgs ra{cfg_.n_expert, k, cfg_.expert_weights_norm ? 1u : 0u, cfg_.expert_weights_scale, 1};
    enc->setComputePipelineState(pso("moe_route"));
    enc->setBuffer(rlog, 0, 0);
    enc->setBuffer(L.router_bias.buf, L.router_bias.off, 1);
    enc->setBuffer(scratch("msel", id_rows), 0, 2);
    enc->setBuffer(ew, 0, 3);
    enc->setBuffer(scratch("mbiased", uint64_t(kMaxDecodeTokens) * cfg_.n_expert * 4), 0, 4);
    enc->setBytes(&ra, sizeof(ra), 5);
    enc->setBuffer(arena_->slot_table(), uint64_t(il) * cfg_.n_expert * 4, 6);
    enc->setBuffer(hit_ids, 0, 7);
    enc->setBuffer(miss_rank, 0, 8);
    enc->setBuffer(indirect, ind_off, 9);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(32, 1, 1));
    after(enc, "moe_route");
    enc->endEncoding();
    const uint64_t v = ++event_value_;
    cb->encodeSignalEvent(routed_.get(), v);
    pending_moe_.push_back({il, v, T});
    enc = encoder(cb, tl, il < cfg_.n_layer ? 2 : -1);
    // The CPU starts on the misses as soon as the route is known; the shared expert runs meanwhile.
    ffn_dense(enc, L.sh_gate, L.sh_up, L.sh_down, xn, sho, cfg_.swiglu_clamp_shexp[il], T);

    // Resident experts of every token run while the CPU reads the misses; the miss passes (one per token) wait.
    const SwigluArgs sa{T * k * nff, cfg_.swiglu_clamp_exp[il] > 1e-6f ? cfg_.swiglu_clamp_exp[il] : 0.0f};
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
    run(0, T, false, hit_ids, 0, xn, 0, h[0], 0, 0);
    run(1, T, false, hit_ids, 0, xn, 0, h[1], 0, 0);
    dispatch(enc, "swiglu", {h[0], h[1], h[2]}, &sa, sizeof(sa), MTL::Size(T * k * nff, 1, 1), MTL::Size(256, 1, 1));
    run(2, T, false, hit_ids, 0, h[2], 0, h[3], 0, 0);
    enc->endEncoding();
    cb->encodeWait(ready_.get(), v);
    enc = encoder(cb, tl, il < cfg_.n_layer ? 3 : -1);
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
    dispatch(enc, "moe_combine_split", {h[3], m[3], miss_rank, ew, sho, ffo}, &ca, sizeof(ca), MTL::Size(E, T, 1),
             MTL::Size(256, 1, 1));
}

// Prompt processing: chunks of tokens go through each layer together (projections as GEMMs, the sequential parts
// token by token inside the kernels), with one CPU synchronization per MoE layer to make the chunk's experts
// resident. Leaves the same caches and state as step() would token by token.
const std::vector<float>& Engine::prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled)
{
    if (tokens.size() < 2 || probe_ || stream_override_ || route_override_) {
        for (int32_t t : tokens)
            step(t);
        return logits_;
    }
    for (int32_t t : tokens)
        if (t < 0 || uint32_t(t) >= cfg_.n_vocab)
            throw std::runtime_error("token out of range");
    if (pos_ + tokens.size() > opts_.max_ctx)
        throw std::runtime_error("context full");
    apply_pressure();
    // Kernel timing is for steps: a prefill's dispatches would overflow the sample buffers.
    struct Level {
        Telemetry& tel;
        Telemetry saved;
        ~Level() { tel = saved; }
    } level{tel_, tel_};
    if (tel_ == Telemetry::kernels)
        tel_ = Telemetry::summary;
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E, k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
    const uint32_t chunk = std::max<uint32_t>(1, opts_.prefill_chunk);
    const auto t_start = Clock::now();
    const ExpertArena::Stats arena_before = arena_->stats();
    const double gpu_before = stats_.gpu_exec_ms;
    StepStats total;
    total.pos = pos_;
    if (timing()) {
        total.layers.assign(cfg_.n_layer, LayerTimes{});
        for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
            total.layers[l].layer = l;
            total.layers[l].mla = !layers_[l]->kda;
        }
    }

    // A longer prompt's scratch beyond kLendBelow (left to the budget's headroom) takes the memory of lendable
    // slots, whole layers' at a time; afterwards the buffers go and the slots come back (read again).
    constexpr uint64_t kLendBelow = 256ull << 20;
    const uint32_t group = std::max(chunk, opts_.prefill_group);
    const uint64_t need = prefill_scratch(uint32_t(std::min<size_t>(chunk, tokens.size())),
                                          uint32_t(std::min<size_t>(group, tokens.size())));
    const bool lend = need > kLendBelow;
    // Kept past a lending prefill: the next-token block's first draft reads the last hidden state from it.
    if (has_mtp())
        scratch("xn", uint64_t(kMaxDecodeTokens) * cfg_.n_embd * 4);
    std::map<std::string, MTL::Buffer*> kept;
    if (lend) {
        for (const auto& [name, b] : scratch_)
            kept.emplace(name, b.get());
        arena_->lend(need - kLendBelow);
    }
    struct Restore {
        Engine& e;
        bool lend;
        std::map<std::string, MTL::Buffer*>& kept;
        ~Restore()
        {
            if (!lend)
                return;
            std::erase_if(e.scratch_, [&](const auto& kv) {
                auto it = kept.find(kv.first);
                return it == kept.end() || it->second != kv.second.get();
            });
            if (!e.pressure_lent_) // under memory pressure the slots stay lent
                e.arena_->reclaim();
        }
    } restore{*this, lend, kept};

    // Groups of up to `group` tokens go through the layers together: each layer's attention in chunks of `chunk`
    // tokens, then its FFN for the whole group, so a routed expert is loaded at most once per layer and group.
    size_t done = tokens.size();
    for (size_t g0 = 0; g0 < tokens.size(); g0 += group) {
        const uint32_t G = uint32_t(std::min<size_t>(group, tokens.size() - g0));
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Buffer* xg_in = scratch("p_xin", uint64_t(G) * E * 4);
        MTL::Buffer* xn_g = scratch("p_xng", uint64_t(G) * E * 4);
        MTL::Buffer* ffo = scratch("p_ffo", uint64_t(G) * E * 4);
        MTL::Buffer* R = scratch("p_R", uint64_t(G) * hd * 4);
        MTL::Buffer* R2 = scratch("p_R2", uint64_t(G) * hd * 4);
        MTL::Buffer* pc_g = scratch("p_pcg", uint64_t(G) * 24 * 4);
        MTL::Buffer* x = scratch("p_x", uint64_t(std::min(chunk, G)) * E * 4);
        MTL::Buffer* xn = scratch("p_xn", uint64_t(std::min(chunk, G)) * E * 4);
        MTL::Buffer* att = scratch("p_att", uint64_t(std::min(chunk, G)) * E * 4);
        MTL::Buffer* pc = scratch("p_pc", uint64_t(std::min(chunk, G)) * 24 * 4);

        {
            const gguf::Tensor& t = *token_embd_.tensor;
            const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
            std::vector<std::byte> row(rb);
            for (uint32_t i = 0; i < G; ++i) {
                read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(tokens[g0 + i]) * rb);
                quant::dequant_row(t.type, row.data(), static_cast<float*>(xg_in->contents()) + uint64_t(i) * E, E);
            }
            // The next-token block's inputs: each position's following prompt token (written before any of the
            // group's GPU work, which reads them).
            if (has_mtp())
                for (uint32_t i = 0; i < G && g0 + i + 1 < tokens.size(); ++i) {
                    read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(tokens[g0 + i + 1]) * rb);
                    quant::dequant_row(t.type, row.data(),
                                       static_cast<float*>(scratch("mf_e", uint64_t(group) * E * 4)->contents()) +
                                           uint64_t(i) * E,
                                       E);
                }
        }

        MTL::Timestamp cpu0 = 0, gpu0 = 0;
        if (timing())
            ctx_.device()->sampleTimestamps(&cpu0, &gpu0);
        last_step_ = StepStats{};
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        cur_layer_ = cfg_.n_layer;
        MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
        const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, G};
        // Command buffers committed without waiting (the resident experts' work); drain() waits for them.
        std::vector<MTL::CommandBuffer*> committed;
        auto drain = [&] {
            for (MTL::CommandBuffer* c : committed) {
                c->waitUntilCompleted();
                if (c->status() != MTL::CommandBufferStatusCompleted)
                    throw std::runtime_error("command buffer failed");
                stats_.gpu_exec_ms += (c->GPUEndTime() - c->GPUStartTime()) * 1e3;
                c->release();
            }
            committed.clear();
        };
        dispatch(enc, "hc_init", {xg_in, R}, &ha, sizeof(ha), MTL::Size(E, G, 1), MTL::Size(256, 1, 1));

        for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
            const Layer& L = *layers_[il];
            cur_layer_ = il;
            for (uint32_t c0 = 0; c0 < G; c0 += chunk) {
                const uint32_t T = std::min(chunk, G - c0);
                const uint64_t r_off = uint64_t(c0) * hd * 4;
                if (timing()) {
                    enc->endEncoding();
                    enc = encoder(cb, il, 0);
                }
                hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R, x, pc, T, r_off);
                norm(enc, x, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
                if (L.kda)
                    kda(enc, L, il, xn, att, T);
                else
                    mla(enc, L, il, xn, att, T, pos_ + uint32_t(g0) + c0);
                hc_post(enc, att, R, R2, pc, T, r_off);
            }
            std::swap(R, R2);
            if (timing()) {
                enc->endEncoding();
                enc = encoder(cb, il, 1);
            }

            hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, R, xg_in, pc_g, G);
            norm(enc, xg_in, &L.ffn_norm, xn_g, G, E, E, cfg_.rms_eps);
            // Dense FFNs (the leading layers' and the shared expert) in chunks: their intermediates are wide.
            const DevTensor& fg = L.dense ? L.ffn_gate : L.sh_gate;
            const DevTensor& fu = L.dense ? L.ffn_up : L.sh_up;
            const DevTensor& fd = L.dense ? L.ffn_down : L.sh_down;
            for (uint32_t c0 = 0; c0 < G; c0 += chunk)
                ffn_dense(enc, fg, fu, fd, xn_g, ffo, cfg_.swiglu_clamp_shexp[il], std::min(chunk, G - c0),
                          uint64_t(c0) * E * 4, uint64_t(c0) * E * 4);
            if (!L.dense) {
                MTL::Buffer* rlog = scratch("p_rlog", uint64_t(G) * cfg_.n_expert * 4);
                MTL::Buffer* sel = scratch("p_sel", uint64_t(G) * 16 * 4);
                MTL::Buffer* ew = scratch("p_ew", uint64_t(G) * 16 * 4);
                proj(enc, L.router, xn_g, 0, rlog, 0, G);
                // Routing only; the CPU assigns slots after this sync.
                const RouteArgs ra{cfg_.n_expert, k, cfg_.expert_weights_norm ? 1u : 0u, cfg_.expert_weights_scale, 0};
                MTL::Buffer* unused = scratch("p_unused", 64);
                enc->setComputePipelineState(pso("moe_route"));
                enc->setBuffer(rlog, 0, 0);
                enc->setBuffer(L.router_bias.buf, L.router_bias.off, 1);
                enc->setBuffer(sel, 0, 2);
                enc->setBuffer(ew, 0, 3);
                enc->setBuffer(scratch("p_biased", uint64_t(G) * cfg_.n_expert * 4), 0, 4);
                enc->setBytes(&ra, sizeof(ra), 5);
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
                                    uint64_t(il) * cfg_.n_expert;
                std::vector<std::vector<uint32_t>> users(cfg_.n_expert); // per expert: t * 16 + j
                for (uint32_t t = 0; t < G; ++t)
                    for (uint32_t j = 0; j < k; ++j)
                        users[selp[t * 16 + j]].push_back(t * 16 + j);
                std::vector<int> order;
                for (int e = 0; e < int(cfg_.n_expert); ++e)
                    if (!users[e].empty() && table[e] >= 0)
                        order.push_back(e);
                const size_t n_resident = order.size();
                for (int e = 0; e < int(cfg_.n_expert); ++e)
                    if (!users[e].empty() && table[e] < 0)
                        order.push_back(e);
                const uint32_t rows = G * k;
                MTL::Buffer* tok = scratch("p_tok", uint64_t(rows) * 4);
                MTL::Buffer* pos = scratch("p_pos", uint64_t(G) * 16 * 4);
                auto* tokp = static_cast<int32_t*>(tok->contents());
                auto* posp = static_cast<int32_t*>(pos->contents());
                std::vector<uint32_t> start(cfg_.n_expert + 1);
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
                const SwigluArgs sa0{0, cfg_.swiglu_clamp_exp[il] > 1e-6f ? cfg_.swiglu_clamp_exp[il] : 0.0f};
                auto rows_of = [&](size_t b0, size_t b1) {
                    return b0 == b1 ? 0u : start[order[b1 - 1]] + uint32_t(users[order[b1 - 1]].size()) - start[order[b0]];
                };

                // Experts [b0, b1) of order (rows within kBatchRows), all pinned resident: gather their rows, then
                // in a concurrent encoder gate and up of every expert, the SwiGLU, every down into yg.
                auto run = [&](size_t b0, size_t b1, const std::vector<int32_t>& slot) {
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
                        const auto [wb, wo] = arena_->region(il, part).at(uint32_t(slot[e]));
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
                        const auto [wb, wo] = arena_->region(il, 2).at(uint32_t(slot[e]));
                        gemm_->encode(enc, p.type, p.k, p.n, uint32_t(users[e].size()), wb, wo, mb,
                                      uint64_t(start[e] - r0) * nff * 2, yg, uint64_t(start[e]) * E * 4);
                    }
                    enc->endEncoding();
                    enc = encoder(cb, il, 3);
                };

                const auto t_ensure = Clock::now();
                std::vector<int32_t> slot(cfg_.n_expert, -1);
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
                    run(b0, b1, slot);
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
                    arena_->ensure(il, std::span<const int>(&order[i], 1), &slot[order[i]], true);
                }
                commit(b0, n_resident);
                // Misses are read a few at a time (their chunks in flight together) and each batch goes to the GPU
                // as soon as it is resident, so reading the next overlaps running this one.
                constexpr size_t kMissesPerRead = 4;
                std::vector<int32_t> got(kMissesPerRead);
                for (size_t i = n_resident; i < order.size();) {
                    size_t j = i + 1;
                    while (j < order.size() && j - i < kMissesPerRead && rows_of(i, j + 1) <= kBatchRows)
                        ++j;
                    while (!arena_->ensure(il, std::span<const int>(&order[i], j - i), got.data(), true)) {
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
            ablate(enc, il, ffo, G);
            ablate(enc, il, R, G * cfg_.hc);
            hc_post(enc, ffo, R, R2, pc_g, G);
            std::swap(R, R2);
        }

        // The next-token block's cache for the group's positions whose following token is known; the hidden state of
        // the group's last position goes to "xn" for the first mtp_draft() (or a checkpoint if prefill stops here).
        cur_layer_ = cfg_.n_layer;
        if (has_mtp()) {
            MTL::Buffer* hg = scratch("p_h", uint64_t(G) * E * 4);
            const HcArgs hm{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, G};
            dispatch(enc, "hc_mean", {R, xg_in}, &hm, sizeof(hm), MTL::Size(E, G, 1), MTL::Size(256, 1, 1));
            norm(enc, xg_in, &output_norm_, hg, G, E, E, cfg_.rms_eps);
            for (uint32_t c0 = 0; c0 < G && g0 + c0 + 1 < tokens.size(); c0 += chunk) {
                const uint32_t T = std::min<uint32_t>({chunk, G - c0, uint32_t(tokens.size() - 1 - g0 - c0)});
                mtp_fill(enc, hg, uint64_t(c0) * E * 4, scratch("mf_e", uint64_t(group) * E * 4), uint64_t(c0) * E * 4,
                         T, pos_ + uint32_t(g0) + c0);
            }
            const uint32_t n = E;
            enc->setComputePipelineState(pso("copy_f32"));
            enc->setBuffer(hg, uint64_t(G - 1) * E * 4, 0);
            enc->setBuffer(scratch("xn", uint64_t(kMaxDecodeTokens) * E * 4), 0, 1);
            enc->setBytes(&n, sizeof(n), 2);
            enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        }

        // Logits of the group's last token, or of every token (in chunks) for a logits sink.
        const uint32_t nl = logits_sink_ ? std::min(chunk, G) : 1;
        MTL::Buffer* xl = scratch("p_xl", uint64_t(nl) * E * 4);
        MTL::Buffer* xnl = scratch("p_xnl", uint64_t(nl) * E * 4);
        MTL::Buffer* lg = scratch("p_logits", uint64_t(nl) * cfg_.n_vocab * 4);
        for (uint32_t l0 = logits_sink_ ? 0 : G - 1; l0 < G; l0 += nl) {
            const uint32_t n = std::min(nl, G - l0);
            const HcArgs h1{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, n};
            enc->setComputePipelineState(pso("hc_mean"));
            enc->setBuffer(R, uint64_t(l0) * hd * 4, 0);
            enc->setBuffer(xl, 0, 1);
            enc->setBytes(&h1, sizeof(h1), 2);
            for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
                enc->dispatchThreads(MTL::Size(E, n, 1), MTL::Size(256, 1, 1));
            norm(enc, xl, &output_norm_, xnl, n, E, E, cfg_.rms_eps);
            proj(enc, output_, xnl, 0, lg, 0, n);
            enc->endEncoding();
            finish(cb);
            drain();
            const auto* lgp = static_cast<const float*>(lg->contents());
            if (logits_sink_)
                for (uint32_t i = 0; i < n; ++i)
                    logits_sink_(pos_ + uint32_t(g0) + l0 + i, lgp + uint64_t(i) * cfg_.n_vocab);
            if (l0 + n == G)
                std::memcpy(logits_.data(), lgp + uint64_t(n - 1) * cfg_.n_vocab, uint64_t(cfg_.n_vocab) * 4);
            cb = ctx_.queue()->commandBuffer();
            cur_cb_ = cb;
            enc = encoder(cb, cur_layer_, -1);
        }
        enc->endEncoding();
        finish(cb);
        total.ensure_ms += last_step_.ensure_ms;
        if (timing()) {
            resolve_layer_times(cpu0, gpu0);
            for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
                total.layers[l].attn_ms += last_step_.layers[l].attn_ms;
                total.layers[l].ffn_ms += last_step_.layers[l].ffn_ms;
                total.layers[l].hit_ms += last_step_.layers[l].hit_ms;
                total.layers[l].miss_ms += last_step_.layers[l].miss_ms;
            }
        }
        pool->release();
        if (cancelled && g0 + G < tokens.size() && cancelled()) {
            done = g0 + G;
            break;
        }
    }
    pos_ += uint32_t(done);
    last_pos0_ = pos_ - 1; // mtp_draft() continues from the last position
    last_tokens_ = 1;
    stats_.expert_read_ms = arena_->stats().read_ms;
    stats_.expert_bytes = arena_->stats().read_bytes;
    const ExpertArena::Stats& arena_after = arena_->stats();
    total.wall_ms = ms_since(t_start);
    total.gpu_ms = stats_.gpu_exec_ms - gpu_before;
    total.read_ms = arena_after.read_ms - arena_before.read_ms;
    total.hits = arena_after.hits - arena_before.hits;
    total.misses = arena_after.misses - arena_before.misses;
    total.read_bytes = arena_after.read_bytes - arena_before.read_bytes;
    last_prefill_ = std::move(total);
    return logits_;
}

} // namespace mako::glm
