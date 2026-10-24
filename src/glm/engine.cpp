#include "glm/engine.hpp"

#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
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

struct NormArgs {
    uint32_t n, wn;
    float eps;
};
struct HcArgs {
    uint32_t n_embd, iters;
    float eps;
    uint32_t tokens;
};
struct ConvArgs {
    uint32_t channels, kernel_len, tokens;
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
    uint32_t head_dim, tokens, qkv_stride, inner, heads;
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
    uint32_t dim, heads, n_pool;
    float w_scale;
};
struct SelectArgs {
    uint32_t n_pool, n_top, kpool, n_kv;
};
struct LayerNormArgs {
    uint32_t n;
    float eps;
    uint32_t ystride;
};
struct SoftmaxArgs {
    uint32_t n_kv;
    float scale;
    uint32_t masked, first_kv, heads;
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
            kv_cache_.emplace_back();
            idx_cache_.emplace_back();
            idx_pooled_.emplace_back();
        } else {
            L->q_a = upload(p + "attn_q_a.weight");
            L->q_a_norm = upload(p + "attn_q_a_norm.weight");
            L->q_b = upload(p + "attn_q_b.weight");
            L->kv_a = upload(p + "attn_kv_a_mqa.weight");
            L->kv_a_norm = upload(p + "attn_kv_a_norm.weight");
            L->k_b = upload(p + "attn_k_b.weight");
            L->v_b = upload(p + "attn_v_b.weight");
            L->idx_q_b = upload(p + "indexer.attn_q_b.weight");
            L->idx_k = upload(p + "indexer.attn_k.weight");
            L->idx_k_norm = upload(p + "indexer.k_norm.weight");
            L->idx_k_norm_b = upload(p + "indexer.k_norm.bias");
            L->idx_proj = upload(p + "indexer.proj.weight");
            L->idx_gate = upload(p + "indexer_compressor_gate.weight");
            L->idx_ape = upload(p + "indexer_compressor_ape.weight");
            kda_state_.emplace_back();
            conv_state_.emplace_back();
            kv_cache_.emplace_back(ctx_.device()->newBuffer(uint64_t(opts_.max_ctx) * cfg_.kv_lora_rank * 2,
                                                            MTL::ResourceStorageModeShared));
            idx_cache_.emplace_back(ctx_.device()->newBuffer(uint64_t(opts_.max_ctx) * 2 * cfg_.indexer_dim * 4,
                                                             MTL::ResourceStorageModeShared));
            idx_pooled_.emplace_back(ctx_.device()->newBuffer(
                uint64_t(opts_.max_ctx / cfg_.indexer_kpool + 1) * cfg_.indexer_dim * 4, MTL::ResourceStorageModeShared));
        }
        if (L->dense) {
            L->ffn_gate = upload(p + "ffn_gate.weight");
            L->ffn_up = upload(p + "ffn_up.weight");
            L->ffn_down = upload(p + "ffn_down.weight");
        } else {
            L->router = upload(p + "ffn_gate_inp.weight");
            L->sh_gate = upload(p + "ffn_gate_shexp.weight");
            L->sh_up = upload(p + "ffn_up_shexp.weight");
            L->sh_down = upload(p + "ffn_down_shexp.weight");
            L->router_bias = upload(p + "exp_probs_b.bias");
            if (L->router_bias.type != gguf::Type::F32 || L->router_bias.dims[0] != cfg_.n_expert)
                throw std::runtime_error(p + "exp_probs_b.bias is not F32 [n_expert]");
            L->experts = {expert(p + "ffn_gate_exps.weight"), expert(p + "ffn_up_exps.weight"),
                          expert(p + "ffn_down_exps.weight")};
        }
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
    arena_plan_.mem = query_memory(ctx_.info().recommended_working_set);
    built_chunk_ = opts_.prefill_chunk;
    arena_plan_.lendable = double(prefill_scratch(opts_.prefill_chunk));
    arena_plan_.budget = opts_.arena_gb >= 0 ? opts_.arena_gb * 1e9 : arena_budget(arena_plan_.mem);
    auto t_phase = Clock::now();
    routing_key_ = glm::routing_key(model_, cfg_);
    load_.key_ms = ms_since(t_phase);
    default_profile_ = default_profile_path(routing_key_);
    if (!opts_.expert_counts.empty())
        loaded_profile_ = opts_.expert_counts;
    else if (std::filesystem::exists(default_profile_))
        loaded_profile_ = default_profile_;
    Counts counts;
    if (!loaded_profile_.empty())
        counts = load_profile(loaded_profile_, cfg_.n_layer, cfg_.n_expert);
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
    if (arena_->tail_buffer())
        arena_bufs.push_back(arena_->tail_buffer());
    for (MTL::Buffer* b : arena_bufs) {
        enc->setComputePipelineState(pso("copy_f32"));
        enc->setBuffer(b, 0, 0);
        enc->setBuffer(sink, 0, 1);
        enc->setBytes(&one, sizeof(one), 2);
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
        const uint64_t want = opts_.telemetry == Telemetry::kernels ? uint64_t(cfg_.n_layer) * 240 + 64
                                                                    : uint64_t(cfg_.n_layer) * 24 + 64;
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
    load_.total_ms = ms_since(t_load);
}

MTL::ComputeCommandEncoder* Engine::encoder(MTL::CommandBuffer* cb, uint32_t layer, int section, MTL::DispatchType type)
{
    if (samples_.empty() || (opts_.telemetry == Telemetry::layers && section < 0))
        return cb->computeCommandEncoder(type);
    if (opts_.telemetry == Telemetry::kernels)
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
    if (opts_.telemetry != Telemetry::kernels || timed_.empty())
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
    if (opts_.telemetry == Telemetry::layers) {
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

// Scratch bytes a prefill chunk allocates: the per-token buffers of every stage it runs, plus the attention scores
// and partial sums (bounded by the query group size, not the chunk) and the decode scratch.
uint64_t Engine::prefill_scratch(uint32_t chunk) const
{
    const uint64_t E = cfg_.n_embd, hd = uint64_t(cfg_.hc) * E, inner = cfg_.kda_inner(), D = cfg_.kda_head_dim;
    const uint64_t H = cfg_.n_head, R = cfg_.kv_lora_rank, k = cfg_.n_expert_used;
    const uint64_t per_token = 5 * E + 3 * hd + 48                                            // streams, hc
                               + 11 * inner + 2 * D + 2 * H                                     // KDA
                               + 2 * cfg_.q_lora_rank + H * cfg_.head_k + 2 * R + 2 * H * R + H * cfg_.head_v
                               + uint64_t(cfg_.indexer_heads) * (cfg_.indexer_dim + 1) + cfg_.indexer_dim // MLA
                               + 3 * uint64_t(cfg_.n_ff)                                        // dense FFN
                               + 2 * cfg_.n_expert + 32 + 2 * k * E + 3 * k * cfg_.n_ff_exp;    // MoE
    return uint64_t(chunk) * per_token * 4 + (96ull << 20);
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

void Engine::save_profile(const std::filesystem::path& path, bool merge) const
{
    Counts counts = merge && std::filesystem::exists(path) ? load_profile(path, cfg_.n_layer, cfg_.n_expert)
                                                            : Counts(cfg_.n_layer);
    const auto& usage = arena_->usage();
    for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
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

ExpertArena::Part Engine::expert(const std::string& name)
{
    auto ref = model_.find(name);
    if (!ref)
        throw std::runtime_error("missing tensor " + name);
    const gguf::Tensor& t = *ref.tensor;
    if (t.dims.size() != 3 || t.dims[2] != cfg_.n_expert)
        throw std::runtime_error(name + ": expected [k, n, n_expert]");
    if (!mtl::Gemv::supports(t.type))
        throw std::runtime_error(std::format("{}: unsupported type {}", name, gguf::traits(t.type).name));
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
    const bool labelled = opts_.telemetry == Telemetry::kernels || repeat_n_ > 1;
    const std::string label = labelled ? std::format("gemv {} {}x{}", gguf::traits(a.type).name, a.k, a.n) : "";
    for (int r = repeats(label); r > 0; --r)
        gemv_->encode(enc, a, w.buf, w.off, x, x_off, y, y_off);
    if (opts_.telemetry == Telemetry::kernels)
        after(enc, label);
}

// y[t] = W x[t] for T tokens: the GEMV for one token, the GEMM for wide matrices, else the GEMV over all tokens.
// y rows are ldy elements apart (0: n).
void Engine::proj(MTL::ComputeCommandEncoder*& enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
                  uint64_t y_off, uint32_t T, uint32_t ldy)
{
    const uint32_t k = uint32_t(w.dims[0]), n = uint32_t(w.dims[1]);
    if (T > 1 && (n >= 256 || (ldy && ldy != n)) && k % 64 == 0 && mtl::Gemm::supports(w.type)) {
        gemm_->encode(enc, w.type, k, n, T, w.buf, w.off, x, x_off, y, y_off, ldy);
        if (opts_.telemetry == Telemetry::kernels)
            after(enc, std::format("gemm {} {}x{}", gguf::traits(w.type).name, k, n));
        return;
    }
    if (T == 1 || !ldy || ldy == n) {
        gemv(enc, w, x, x_off, y, y_off, 1, T);
        return;
    }
    for (uint32_t t = 0; t < T; ++t)
        gemv(enc, w, x, x_off + uint64_t(t) * k * 4, y, y_off + uint64_t(t) * ldy * 4);
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
                  uint32_t n, uint32_t wn, float eps)
{
    const NormArgs a{n, w ? wn : 0, eps};
    enc->setComputePipelineState(pso("rms_norm"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(w ? w->buf : x, w ? w->off : 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBytes(&a, sizeof(a), 3);
    for (int r = repeats("rms_norm"); r > 0; --r)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "rms_norm");
}

// x = mHC pre-mix of the streams R for T tokens; pc keeps each token's pre/post/comb for hc_post.
void Engine::hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base,
                    MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T)
{
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    MTL::Buffer* flat = scratch("flat", uint64_t(T) * hd * 4);
    MTL::Buffer* mixes = scratch("mixes", uint64_t(T) * 24 * 4);
    norm(enc, R, nullptr, flat, T, hd, 0, cfg_.rms_eps);
    proj(enc, fn, flat, 0, mixes, 0, T);
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
    enc->setComputePipelineState(pso("hc_mix"));
    enc->setBuffer(mixes, 0, 0);
    enc->setBuffer(scale.buf, scale.off, 1);
    enc->setBuffer(base.buf, base.off, 2);
    enc->setBuffer(pc, 0, 3);
    enc->setBytes(&ha, sizeof(ha), 4);
    enc->dispatchThreads(MTL::Size(T, 1, 1), MTL::Size(std::min<uint32_t>(T, 64), 1, 1));
    after(enc, "hc_mix");
    dispatch(enc, "hc_collapse", {R, pc, x}, &ha, sizeof(ha), MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
}

// out = mHC post-mix of x into the streams R for T tokens, with hc_pre's pc.
void Engine::hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* R, MTL::Buffer* out, MTL::Buffer* pc,
                     uint32_t T)
{
    const HcArgs ha{cfg_.n_embd, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
    dispatch(enc, "hc_expand", {x, R, pc, out}, &ha, sizeof(ha), MTL::Size(cfg_.n_embd, T, 1), MTL::Size(256, 1, 1));
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
    const ConvArgs ca{inner, cfg_.conv_kernel, T};
    enc->setComputePipelineState(pso("kda_conv"));
    enc->setBuffer(qkv, 0, 0);
    enc->setBuffer(L.conv_q.buf, L.conv_q.off, 1);
    enc->setBuffer(L.conv_k.buf, L.conv_k.off, 2);
    enc->setBuffer(L.conv_v.buf, L.conv_v.off, 3);
    enc->setBuffer(conv_state_[il].get(), 0, 4);
    enc->setBuffer(qkvc, 0, 5);
    enc->setBytes(&ca, sizeof(ca), 6);
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
    enc->dispatchThreads(MTL::Size(inner, T, 1), MTL::Size(256, 1, 1));
    after(enc, "kda_gate");

    const L2Args la{D, 1e-6f, 3 * inner};
    enc->setComputePipelineState(pso("l2_norm"));
    enc->setBuffer(qkvc, 0, 0);
    enc->setBytes(&la, sizeof(la), 1);
    enc->dispatchThreadgroups(MTL::Size(2 * H, T, 1), MTL::Size(256, 1, 1));
    after(enc, "l2_norm");

    MTL::Buffer* o = scratch("o", uint64_t(T) * inner * 4);
    const ScanArgs sa{D, T, 3 * inner, inner, H};
    enc->setComputePipelineState(pso("kda_scan"));
    enc->setBuffer(qkvc, 0, 0);
    enc->setBuffer(qkvc, uint64_t(inner) * 4, 1);
    enc->setBuffer(qkvc, uint64_t(inner) * 8, 2);
    enc->setBuffer(g, 0, 3);
    enc->setBuffer(beta, 0, 4);
    enc->setBuffer(kda_state_[il].get(), 0, 5);
    enc->setBuffer(o, 0, 6);
    enc->setBytes(&sa, sizeof(sa), 7);
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
    const NormArgs na{D, D, cfg_.rms_eps};
    enc->setComputePipelineState(pso("rms_norm_gate"));
    enc->setBuffer(o, 0, 0);
    enc->setBuffer(L.o_norm.buf, L.o_norm.off, 1);
    enc->setBuffer(g2, 0, 2);
    enc->setBuffer(gated, 0, 3);
    enc->setBytes(&na, sizeof(na), 4);
    enc->dispatchThreadgroups(MTL::Size(T * H, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "rms_norm_gate");
    if (probes)
        probe(enc, "attn_output", int(il), gated, inner);
    proj(enc, L.wo, gated, 0, att, 0, T);
}

// Attention of `tokens` consecutive query positions (rows of H heads of q, latent width R) over a half latent cache
// read once for all heads: scores by GEMM, softmax over each row's visible keys (row r of token i sees
// first_kv + i keys; only the indexer's selected cells when sparse), then the output split over key ranges and summed.
void Engine::attend(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* cache, MTL::Buffer* q, uint64_t q_off,
                    MTL::Buffer* out, uint64_t out_off, uint32_t tokens, uint32_t first_kv, bool sparse)
{
    const uint32_t H = cfg_.n_head, R = cfg_.kv_lora_rank, rows = tokens * H;
    const uint32_t n_kv = first_kv + tokens - 1;
    // At most 16 splits of at least 256 keys, whole 64-key tiles.
    const uint32_t keys_per_split = std::max<uint32_t>(256, ((n_kv + 15) / 16 + 63) / 64 * 64);
    const uint32_t splits = (n_kv + keys_per_split - 1) / keys_per_split;
    MTL::Buffer* scores = scratch("scores", uint64_t(rows) * n_kv * 4);
    MTL::Buffer* partial = scratch("pv_part", uint64_t(splits) * rows * R * 4);
    gemm_->encode(enc, gguf::Type::F16, R, n_kv, rows, cache, 0, q, q_off, scores, 0);
    after(enc, "attn_scores");
    const SoftmaxArgs sa{n_kv, 1.0f / std::sqrt(float(cfg_.head_k)), sparse ? 1u : 0u, first_kv, H};
    enc->setComputePipelineState(pso("attn_softmax"));
    enc->setBuffer(scores, 0, 0);
    enc->setBuffer(scratch("idx_mask", opts_.max_ctx), 0, 1);
    enc->setBytes(&sa, sizeof(sa), 2);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "attn_softmax");
    const PvArgs pa{rows, R, n_kv, keys_per_split};
    enc->setComputePipelineState(pso("attn_pv"));
    enc->setBuffer(scores, 0, 0);
    enc->setBuffer(cache, 0, 1);
    enc->setBuffer(partial, 0, 2);
    enc->setBytes(&pa, sizeof(pa), 3);
    enc->dispatchThreadgroups(MTL::Size(R / 64, (rows + 63) / 64, splits), MTL::Size(128, 1, 1));
    after(enc, "attn_pv");
    const uint32_t sum[2] = {rows * R, splits};
    enc->setComputePipelineState(pso("sum_splits"));
    enc->setBuffer(partial, 0, 0);
    enc->setBuffer(out, out_off, 1);
    enc->setBytes(sum, sizeof(sum), 2);
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
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "store_half");

    // Per-head matrices (k_b: head_k -> R, v_b: R -> head_v): one GEMM per head over all tokens, or the batched
    // GEMV for one token.
    auto per_head = [&](const DevTensor& w, MTL::Buffer* x, uint32_t xw, MTL::Buffer* y, uint32_t yw) {
        if (T == 1) {
            gemv(enc, w, x, 0, y, 0, H);
            return;
        }
        const uint32_t k = uint32_t(w.dims[0]), nr = uint32_t(w.dims[1]);
        const uint64_t mat = mtl::Gemv::row_bytes(w.type, k) * nr;
        for (uint32_t h = 0; h < H; ++h)
            gemm_->encode(enc, w.type, k, nr, T, w.buf, w.off + h * mat, x, uint64_t(h) * xw * 4, y,
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
        attend(enc, kv_cache_[il].get(), qabs, t * lrow * 4, lat, t * lrow * 4, g, pos0 + t + 1, false);
        t += g;
    }
    for (; t < T; ++t) {
        const bool sparse = indexer_select(enc, il, t, pos0 + t);
        attend(enc, kv_cache_[il].get(), qabs, t * lrow * 4, lat, t * lrow * 4, 1, pos0 + t + 1, sparse);
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

    proj(enc, L.idx_q_b, qrn, 0, iq, 0, T);
    proj(enc, L.idx_k, xn, 0, ikr, 0, T);
    const LayerNormArgs na{Di, cfg_.norm_eps, 2 * Di};
    enc->setComputePipelineState(pso("layer_norm"));
    enc->setBuffer(ikr, 0, 0);
    enc->setBuffer(L.idx_k_norm.buf, L.idx_k_norm.off, 1);
    enc->setBuffer(L.idx_k_norm_b.buf, L.idx_k_norm_b.off, 2);
    enc->setBuffer(cache, row, 3);
    enc->setBytes(&na, sizeof(na), 4);
    enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "layer_norm");
    proj(enc, L.idx_gate, xn, 0, cache, row + Di * 4, T, 2 * Di);
    proj(enc, L.idx_proj, xn, 0, iw, 0, T);

    const uint32_t first = pos0 / kp, last = (pos0 + T) / kp;
    if (last > first) {
        const PoolArgs pa{Di, kp, first};
        enc->setComputePipelineState(pso("idx_pool"));
        enc->setBuffer(cache, 0, 0);
        enc->setBuffer(L.idx_ape.buf, L.idx_ape.off, 1);
        enc->setBuffer(idx_pooled_[il].get(), 0, 2);
        enc->setBytes(&pa, sizeof(pa), 3);
        enc->dispatchThreads(MTL::Size(Di, last - first, 1), MTL::Size(Di, 1, 1));
        after(enc, "idx_pool");
    }
}

// For token t (at position pos) of the last indexer_project: once more pools are complete than the selection keeps,
// writes the cell mask of its top pools plus the incomplete tail. Returns whether the mask applies.
bool Engine::indexer_select(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t t, uint32_t pos)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    const uint32_t n_pool = (pos + 1) / kp, n_top = cfg_.indexer_top_k / kp;
    if (n_pool <= n_top)
        return false;
    MTL::Buffer* score = scratch("idx_score", uint64_t(opts_.max_ctx / kp + 1) * 4);
    const ScoreArgs sa{Di, Hi, n_pool, 1.0f / std::sqrt(float(Di * Hi))};
    enc->setComputePipelineState(pso("idx_score"));
    enc->setBuffer(scratch("iq", 0), uint64_t(t) * Hi * Di * 4, 0);
    enc->setBuffer(scratch("iw", 0), uint64_t(t) * Hi * 4, 1);
    enc->setBuffer(idx_pooled_[il].get(), 0, 2);
    enc->setBuffer(score, 0, 3);
    enc->setBytes(&sa, sizeof(sa), 4);
    enc->dispatchThreads(MTL::Size(n_pool, 1, 1), MTL::Size(64, 1, 1));
    after(enc, "idx_score");
    if (pos == pos_)
        probe(enc, "indexer_score", int(il), score, n_pool);
    const SelectArgs sel{n_pool, n_top, kp, pos + 1};
    dispatch(enc, "idx_select", {score, scratch("idx_mask", opts_.max_ctx)}, &sel, sizeof(sel), MTL::Size(1, 1, 1),
             MTL::Size(256, 1, 1), false);
    return true;
}

void Engine::ffn_dense(MTL::ComputeCommandEncoder*& enc, const DevTensor& gate, const DevTensor& up,
                       const DevTensor& down, MTL::Buffer* xn, MTL::Buffer* out, float limit, uint32_t T)
{
    const uint32_t nff = uint32_t(gate.dims[1]);
    MTL::Buffer* g = scratch("ffg", uint64_t(T) * nff * 4);
    MTL::Buffer* u = scratch("ffu", uint64_t(T) * nff * 4);
    MTL::Buffer* m = scratch("ffm", uint64_t(T) * nff * 4);
    proj(enc, gate, xn, 0, g, 0, T);
    proj(enc, up, xn, 0, u, 0, T);
    const SwigluArgs sa{T * nff, limit > 1e-6f ? limit : 0.0f};
    dispatch(enc, "swiglu", {g, u, m}, &sa, sizeof(sa), MTL::Size(T * nff, 1, 1), MTL::Size(256, 1, 1));
    proj(enc, down, m, 0, out, 0, T);
}

// Serves the MoE layers of a committed command buffer in order: waits until a layer's router has run, makes its
// selected experts resident and writes their slots, then lets the GPU continue.
void Engine::serve_moe(MTL::CommandBuffer* cb)
{
    const uint32_t k = cfg_.n_expert_used;
    auto* sel = static_cast<int*>(scratch("msel", 16 * 4)->contents());
    auto* w = static_cast<float*>(scratch("ew", 16 * 4)->contents());
    auto* hit_ids = static_cast<const int32_t*>(scratch("eids_hit", 16 * 4)->contents());
    auto* miss_ids = static_cast<int32_t*>(scratch("eids_miss", 16 * 4)->contents());
    auto* miss_rank = static_cast<int32_t*>(scratch("mrank", 16 * 4)->contents());
    auto* indirect = static_cast<uint32_t*>(scratch("mind", uint64_t(cfg_.n_layer) * 36)->contents());
    for (const auto [il, v] : pending_moe_) {
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
        if (overridden)
            route_override_(int(il), sel, w, k);
        last_step_.route_wait_ms += ms_since(t_wait);
        const auto t_ensure = Clock::now();
        std::array<int32_t, 16> slots{};
        arena_->ensure(il, std::span<const int>(sel, k), slots.data());
        last_step_.ensure_ms += ms_since(t_ensure);
        for (uint32_t j = 0; j < k; ++j) {
            if (overridden) {
                miss_rank[j] = int32_t(j);
                miss_ids[j] = slots[j];
            } else if (miss_rank[j] >= 0) {
                miss_ids[miss_rank[j]] = slots[j];
            } else if (hit_ids[j] != slots[j]) {
                throw std::runtime_error(std::format("layer {}: GPU found expert {} in slot {}, the arena has it in {}",
                                                     il, sel[j], hit_ids[j], slots[j]));
            }
        }
        if (overridden)
            for (int d = 0; d < 3; ++d)
                indirect[il * 9 + d * 3 + 2] = k;
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
    if (pos_ >= opts_.max_ctx)
        throw std::runtime_error("context full");
    if (token < 0 || uint32_t(token) >= cfg_.n_vocab)
        throw std::runtime_error("token out of range");
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, 1};

    step_start_ = Clock::now();
    encoding_ = true;
    const ExpertArena::Stats before = arena_->stats();
    const Stats stats_before = stats_;
    last_step_ = StepStats{};
    last_step_.pos = pos_;
    last_step_.token = token;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Buffer* x = scratch("x", E * 4);
    MTL::Buffer* xn = scratch("xn", E * 4);
    MTL::Buffer* att = scratch("att", E * 4);
    MTL::Buffer* ffo = scratch("ffo", E * 4);
    MTL::Buffer* pc = scratch("pc", 24 * 4);
    scratch("R", hd * 4);
    scratch("R2", hd * 4);

    {
        const gguf::Tensor& t = *token_embd_.tensor;
        const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
        std::vector<std::byte> row(rb);
        read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(token) * rb);
        quant::dequant_row(t.type, row.data(), static_cast<float*>(x->contents()), E);
    }

    if (probe_)
        probe_("inp_embd", -1, static_cast<const float*>(x->contents()), E);
    MTL::Timestamp cpu0 = 0, gpu0 = 0;
    if (!samples_.empty())
        ctx_.device()->sampleTimestamps(&cpu0, &gpu0);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    cur_layer_ = cfg_.n_layer;
    MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
    dispatch(enc, "hc_init", {x, scratch("R", hd * 4)}, &ha, sizeof(ha), MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));

    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        cur_layer_ = il;
        if (stream_override_) {
            submit(cb, enc);
            stream_override_(int(il), static_cast<float*>(scratch("R", hd * 4)->contents()), hd);
        }
        if (!samples_.empty()) {
            enc->endEncoding();
            enc = encoder(cb, il, 0);
        }
        hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, scratch("R", hd * 4), x, pc, 1);
        probe(enc, "hc_attn_pre", int(il), x, E);
        norm(enc, x, &L.attn_norm, xn, 1, E, E, cfg_.rms_eps);
        probe(enc, "attn_norm", int(il), xn, E);
        if (L.kda)
            kda(enc, L, il, xn, att, 1);
        else
            mla(enc, L, il, xn, att, 1, pos_);
        probe(enc, L.kda ? "kda_out" : "attn_out", int(il), att, E);
        hc_post(enc, att, scratch("R", hd * 4), scratch("R2", hd * 4), pc, 1);
        std::swap(scratch_["R"], scratch_["R2"]);
        if (!samples_.empty()) {
            enc->endEncoding();
            enc = encoder(cb, il, 1);
        }

        hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, scratch("R", hd * 4), x, pc, 1);
        probe(enc, "hc_ffn_pre", int(il), x, E);
        probe(enc, "hc_mixes", int(il), scratch("mixes", 32 * 4), 24);
        norm(enc, x, &L.ffn_norm, xn, 1, E, E, cfg_.rms_eps);
        probe(enc, "ffn_norm", int(il), xn, E);
        if (L.dense) {
            ffn_dense(enc, L.ffn_gate, L.ffn_up, L.ffn_down, xn, ffo, cfg_.swiglu_clamp_shexp[il], 1);
        } else {
            const uint32_t k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
            MTL::Buffer* sho = scratch("sho", E * 4);
            MTL::Buffer* rlog = scratch("rlog", cfg_.n_expert * 4);
            MTL::Buffer* ew = scratch("ew", 16 * 4);
            MTL::Buffer* hit_ids = scratch("eids_hit", 16 * 4);
            MTL::Buffer* miss_ids = scratch("eids_miss", 16 * 4);
            MTL::Buffer* miss_rank = scratch("mrank", 16 * 4);
            MTL::Buffer* indirect = scratch("mind", uint64_t(cfg_.n_layer) * 36);
            const uint64_t ind_off = uint64_t(il) * 36;
            auto part_args = [&](int part) {
                const ExpertArena::Part& e = arena_->part(il, part);
                mtl::Gemv::Args a;
                a.type = e.type;
                a.k = e.k;
                a.n = e.n;
                a.batch = k;
                a.x_shared = part < 2;
                return a;
            };
            // x, y of the miss pass's dispatches; moe_route writes z, the number of misses.
            auto* ind = static_cast<uint32_t*>(indirect->contents()) + il * 9;
            for (int part = 0; part < 3; ++part) {
                const MTL::Size g = gemv_->indexed_grid(part_args(part));
                ind[part * 3] = uint32_t(g.width);
                ind[part * 3 + 1] = uint32_t(g.height);
            }

            gemv(enc, L.router, xn, 0, rlog, 0);
            ffn_dense(enc, L.sh_gate, L.sh_up, L.sh_down, xn, sho, cfg_.swiglu_clamp_shexp[il], 1);
            const RouteArgs ra{cfg_.n_expert, k, cfg_.expert_weights_norm ? 1u : 0u, cfg_.expert_weights_scale, 1};
            enc->setComputePipelineState(pso("moe_route"));
            enc->setBuffer(rlog, 0, 0);
            enc->setBuffer(L.router_bias.buf, L.router_bias.off, 1);
            enc->setBuffer(scratch("msel", 16 * 4), 0, 2);
            enc->setBuffer(ew, 0, 3);
            enc->setBuffer(scratch("mbiased", cfg_.n_expert * 4), 0, 4);
            enc->setBytes(&ra, sizeof(ra), 5);
            enc->setBuffer(arena_->slot_table(), uint64_t(il) * cfg_.n_expert * 4, 6);
            enc->setBuffer(hit_ids, 0, 7);
            enc->setBuffer(miss_rank, 0, 8);
            enc->setBuffer(indirect, ind_off, 9);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
            after(enc, "moe_route");
            enc->endEncoding();
            const uint64_t v = ++event_value_;
            cb->encodeSignalEvent(routed_.get(), v);
            pending_moe_.emplace_back(il, v);
            enc = encoder(cb, il, 2);

            // Resident experts run while the CPU reads the misses; the miss pass waits for it.
            const SwigluArgs sa{k * nff, cfg_.swiglu_clamp_exp[il] > 1e-6f ? cfg_.swiglu_clamp_exp[il] : 0.0f};
            auto pass = [&](MTL::Buffer* ids, bool miss, const char* tag) {
                MTL::Buffer* eg = scratch(std::string("eg") + tag, k * nff * 4);
                MTL::Buffer* eu = scratch(std::string("eu") + tag, k * nff * 4);
                MTL::Buffer* em = scratch(std::string("em") + tag, k * nff * 4);
                MTL::Buffer* ed = scratch(std::string("ed") + tag, k * E * 4);
                auto run = [&](int part, MTL::Buffer* in, MTL::Buffer* out) {
                    const ExpertArena::Region& r = arena_->region(il, part);
                    const mtl::Gemv::Args a = part_args(part);
                    const bool labelled = opts_.telemetry == Telemetry::kernels || repeat_n_ > 1;
                    const std::string label = labelled ? std::format("expert {} {}x{} {}", gguf::traits(a.type).name,
                                                                     a.k, a.n, miss ? "miss" : "hit")
                                                       : "";
                    for (int rep = repeats(label); rep > 0; --rep)
                        gemv_->encode_indexed(enc, a, r.buf, r.off, r.stride, ids, 0, in, 0, out, 0,
                                              miss ? indirect : nullptr, ind_off + uint64_t(part) * 12,
                                              {r.tail, r.tail_off, r.split});
                    if (opts_.telemetry == Telemetry::kernels)
                        after(enc, label);
                };
                run(0, xn, eg);
                run(1, xn, eu);
                dispatch(enc, "swiglu", {eg, eu, em}, &sa, sizeof(sa), MTL::Size(k * nff, 1, 1),
                         MTL::Size(256, 1, 1));
                run(2, em, ed);
                return ed;
            };
            MTL::Buffer* ed_hit = pass(hit_ids, false, "h");
            enc->endEncoding();
            cb->encodeWait(ready_.get(), v);
            enc = encoder(cb, il, 3);
            MTL::Buffer* ed_miss = pass(miss_ids, true, "m");
            const CombineArgs ca{E, k};
            dispatch(enc, "moe_combine_split", {ed_hit, ed_miss, miss_rank, ew, sho, ffo}, &ca, sizeof(ca),
                     MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));
        }
        probe(enc, "ffn_out", int(il), ffo, E);
        hc_post(enc, ffo, scratch("R", hd * 4), scratch("R2", hd * 4), pc, 1);
        std::swap(scratch_["R"], scratch_["R2"]);
        probe(enc, "l_out", int(il), scratch("R", hd * 4), hd);
    }

    cur_layer_ = cfg_.n_layer;
    if (!samples_.empty()) {
        enc->endEncoding();
        enc = encoder(cb, cur_layer_, -1);
    }
    dispatch(enc, "hc_mean", {scratch("R", hd * 4), x}, &ha, sizeof(ha), MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));
    norm(enc, x, &output_norm_, xn, 1, E, E, cfg_.rms_eps);
    MTL::Buffer* lg = scratch("logits", uint64_t(cfg_.n_vocab) * 4);
    gemv(enc, output_, xn, 0, lg, 0);
    enc->endEncoding();
    finish(cb);
    std::memcpy(logits_.data(), lg->contents(), uint64_t(cfg_.n_vocab) * 4);
    stats_.expert_read_ms = arena_->stats().read_ms;
    stats_.expert_bytes = arena_->stats().read_bytes;
    ++pos_;
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
    if (!samples_.empty())
        resolve_layer_times(cpu0, gpu0);
    return logits_;
}


// Prompt processing: chunks of tokens go through each layer together (projections as GEMMs, the sequential parts
// token by token inside the kernels), with one CPU synchronization per MoE layer to make the chunk's experts
// resident. Leaves the same caches and state as step() would token by token.
const std::vector<float>& Engine::prefill(std::span<const int32_t> tokens)
{
    if (tokens.size() < 2 || probe_ || stream_override_ || route_override_ || opts_.telemetry == Telemetry::kernels) {
        for (int32_t t : tokens)
            step(t);
        return logits_;
    }
    for (int32_t t : tokens)
        if (t < 0 || uint32_t(t) >= cfg_.n_vocab)
            throw std::runtime_error("token out of range");
    if (pos_ + tokens.size() > opts_.max_ctx)
        throw std::runtime_error("context full");
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E, k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
    const uint32_t chunk = std::max<uint32_t>(1, opts_.prefill_chunk);
    const auto t_start = Clock::now();
    const ExpertArena::Stats arena_before = arena_->stats();
    const double gpu_before = stats_.gpu_exec_ms;
    StepStats total;
    total.pos = pos_;
    if (!samples_.empty()) {
        total.layers.assign(cfg_.n_layer, LayerTimes{});
        for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
            total.layers[l].layer = l;
            total.layers[l].mla = !layers_[l]->kda;
        }
    }

    // A long prompt's scratch buffers take the memory of the arena's lendable slots; afterwards the buffers go and
    // the slots come back. Shorter prompts' scratch (under kLendBelow) is left to the budget's headroom.
    constexpr uint64_t kLendBelow = 256ull << 20;
    const bool lend = prefill_scratch(uint32_t(std::min<size_t>(chunk, tokens.size()))) > kLendBelow;
    std::map<std::string, MTL::Buffer*> kept;
    if (lend) {
        for (const auto& [name, b] : scratch_)
            kept.emplace(name, b.get());
        arena_->lend();
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
            e.arena_->reclaim();
        }
    } restore{*this, lend, kept};

    for (size_t c0 = 0; c0 < tokens.size(); c0 += chunk) {
        const uint32_t T = uint32_t(std::min<size_t>(chunk, tokens.size() - c0));
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Buffer* x = scratch("p_x", uint64_t(T) * E * 4);
        MTL::Buffer* xn = scratch("p_xn", uint64_t(T) * E * 4);
        MTL::Buffer* att = scratch("p_att", uint64_t(T) * E * 4);
        MTL::Buffer* ffo = scratch("p_ffo", uint64_t(T) * E * 4);
        MTL::Buffer* sho = scratch("p_sho", uint64_t(T) * E * 4);
        MTL::Buffer* R = scratch("p_R", uint64_t(T) * hd * 4);
        MTL::Buffer* R2 = scratch("p_R2", uint64_t(T) * hd * 4);
        MTL::Buffer* pc = scratch("p_pc", uint64_t(T) * 24 * 4);

        {
            const gguf::Tensor& t = *token_embd_.tensor;
            const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
            std::vector<std::byte> row(rb);
            for (uint32_t i = 0; i < T; ++i) {
                read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(tokens[c0 + i]) * rb);
                quant::dequant_row(t.type, row.data(), static_cast<float*>(x->contents()) + uint64_t(i) * E, E);
            }
        }

        MTL::Timestamp cpu0 = 0, gpu0 = 0;
        if (!samples_.empty())
            ctx_.device()->sampleTimestamps(&cpu0, &gpu0);
        last_step_ = StepStats{};
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        cur_layer_ = cfg_.n_layer;
        MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
        const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
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
        dispatch(enc, "hc_init", {x, R}, &ha, sizeof(ha), MTL::Size(E, T, 1), MTL::Size(256, 1, 1));

        for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
            const Layer& L = *layers_[il];
            cur_layer_ = il;
            if (!samples_.empty()) {
                enc->endEncoding();
                enc = encoder(cb, il, 0);
            }
            hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R, x, pc, T);
            norm(enc, x, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
            if (L.kda)
                kda(enc, L, il, xn, att, T);
            else
                mla(enc, L, il, xn, att, T, pos_);
            hc_post(enc, att, R, R2, pc, T);
            std::swap(R, R2);
            if (!samples_.empty()) {
                enc->endEncoding();
                enc = encoder(cb, il, 1);
            }

            hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, R, x, pc, T);
            norm(enc, x, &L.ffn_norm, xn, T, E, E, cfg_.rms_eps);
            if (L.dense) {
                ffn_dense(enc, L.ffn_gate, L.ffn_up, L.ffn_down, xn, ffo, cfg_.swiglu_clamp_shexp[il], T);
            } else {
                MTL::Buffer* rlog = scratch("p_rlog", uint64_t(T) * cfg_.n_expert * 4);
                MTL::Buffer* sel = scratch("p_sel", uint64_t(T) * 16 * 4);
                MTL::Buffer* ew = scratch("p_ew", uint64_t(T) * 16 * 4);
                proj(enc, L.router, xn, 0, rlog, 0, T);
                ffn_dense(enc, L.sh_gate, L.sh_up, L.sh_down, xn, sho, cfg_.swiglu_clamp_shexp[il], T);
                // Routing only; the CPU assigns slots after this sync.
                const RouteArgs ra{cfg_.n_expert, k, cfg_.expert_weights_norm ? 1u : 0u, cfg_.expert_weights_scale, 0};
                MTL::Buffer* unused = scratch("p_unused", 64);
                enc->setComputePipelineState(pso("moe_route"));
                enc->setBuffer(rlog, 0, 0);
                enc->setBuffer(L.router_bias.buf, L.router_bias.off, 1);
                enc->setBuffer(sel, 0, 2);
                enc->setBuffer(ew, 0, 3);
                enc->setBuffer(scratch("p_biased", uint64_t(T) * cfg_.n_expert * 4), 0, 4);
                enc->setBytes(&ra, sizeof(ra), 5);
                enc->setBuffer(unused, 0, 6);
                enc->setBuffer(unused, 0, 7);
                enc->setBuffer(unused, 0, 8);
                enc->setBuffer(unused, 0, 9);
                enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
                after(enc, "moe_route");
                submit(cb, enc);
                cur_cb_ = cb;

                // Group the chunk's (token, expert) pairs by expert, resident experts first, and run each expert
                // once over all its tokens.
                const auto* selp = static_cast<const int*>(sel->contents());
                const auto* table = static_cast<const int32_t*>(arena_->slot_table()->contents()) +
                                    uint64_t(il) * cfg_.n_expert;
                std::vector<std::vector<uint32_t>> users(cfg_.n_expert); // per expert: t * 16 + j
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t j = 0; j < k; ++j)
                        users[selp[t * 16 + j]].push_back(t * 16 + j);
                std::vector<int> order;
                for (int e = 0; e < int(cfg_.n_expert); ++e)
                    if (!users[e].empty() && table[e] >= 0)
                        order.push_back(e);
                for (int e = 0; e < int(cfg_.n_expert); ++e)
                    if (!users[e].empty() && table[e] < 0)
                        order.push_back(e);
                const uint32_t rows = T * k;
                MTL::Buffer* tok = scratch("p_tok", uint64_t(rows) * 4);
                MTL::Buffer* pos = scratch("p_pos", uint64_t(T) * 16 * 4);
                auto* tokp = static_cast<int32_t*>(tok->contents());
                auto* posp = static_cast<int32_t*>(pos->contents());
                std::vector<uint32_t> start(cfg_.n_expert);
                uint32_t row = 0;
                for (int e : order) {
                    start[e] = row;
                    for (uint32_t u : users[e]) {
                        tokp[row] = int32_t(u / 16);
                        posp[u] = int32_t(row++);
                    }
                }
                MTL::Buffer* xg = scratch("p_xg", uint64_t(rows) * E * 4);
                MTL::Buffer* gg = scratch("p_gg", uint64_t(rows) * nff * 4);
                MTL::Buffer* ug = scratch("p_ug", uint64_t(rows) * nff * 4);
                MTL::Buffer* mg = scratch("p_mg", uint64_t(rows) * nff * 4);
                MTL::Buffer* yg = scratch("p_yg", uint64_t(rows) * E * 4);
                const SwigluArgs sa0{0, cfg_.swiglu_clamp_exp[il] > 1e-6f ? cfg_.swiglu_clamp_exp[il] : 0.0f};

                // Experts [b0, b1) of order, all pinned resident, in a concurrent encoder: gate and up of every
                // expert, then the SwiGLU over their rows, then every down.
                auto run = [&](size_t b0, size_t b1, const std::vector<int32_t>& slot) {
                    if (b0 == b1)
                        return;
                    enc->endEncoding();
                    enc = encoder(cb, il, 2, MTL::DispatchTypeConcurrent);
                    auto gemm_part = [&](int part, int e, MTL::Buffer* x, uint32_t xw, MTL::Buffer* y, uint32_t yw) {
                        const ExpertArena::Part& p = arena_->part(il, part);
                        const auto [wb, wo] = arena_->region(il, part).at(uint32_t(slot[e]));
                        gemm_->encode(enc, p.type, p.k, p.n, uint32_t(users[e].size()), wb, wo, x,
                                      uint64_t(start[e]) * xw * 4, y, uint64_t(start[e]) * yw * 4);
                    };
                    for (size_t i = b0; i < b1; ++i) {
                        gemm_part(0, order[i], xg, E, gg, nff);
                        gemm_part(1, order[i], xg, E, ug, nff);
                    }
                    enc->memoryBarrier(MTL::BarrierScopeBuffers);
                    const uint32_t r0 = start[order[b0]], r1 = start[order[b1 - 1]] + uint32_t(users[order[b1 - 1]].size());
                    SwigluArgs sa = sa0;
                    sa.n = (r1 - r0) * nff;
                    enc->setComputePipelineState(pso("swiglu"));
                    enc->setBuffer(gg, uint64_t(r0) * nff * 4, 0);
                    enc->setBuffer(ug, uint64_t(r0) * nff * 4, 1);
                    enc->setBuffer(mg, uint64_t(r0) * nff * 4, 2);
                    enc->setBytes(&sa, sizeof(sa), 3);
                    enc->dispatchThreads(MTL::Size(sa.n, 1, 1), MTL::Size(256, 1, 1));
                    enc->memoryBarrier(MTL::BarrierScopeBuffers);
                    for (size_t i = b0; i < b1; ++i)
                        gemm_part(2, order[i], mg, nff, yg, E);
                    enc->endEncoding();
                    enc = encoder(cb, il, 3);
                };

                const uint32_t ne = E;
                enc->setComputePipelineState(pso("gather_rows"));
                enc->setBuffer(xn, 0, 0);
                enc->setBuffer(tok, 0, 1);
                enc->setBuffer(xg, 0, 2);
                enc->setBytes(&ne, sizeof(ne), 3);
                enc->dispatchThreads(MTL::Size(E, rows, 1), MTL::Size(256, 1, 1));

                const auto t_ensure = Clock::now();
                std::vector<int32_t> slot(cfg_.n_expert, -1);
                // Resident experts need no reads: commit their work now so the GPU runs it while the misses are
                // read (pinned slots are never evicted).
                size_t b0 = 0;
                while (b0 < order.size() && table[order[b0]] >= 0) {
                    arena_->ensure(il, std::span<const int>(&order[b0], 1), &slot[order[b0]], true);
                    ++b0;
                }
                run(0, b0, slot);
                enc->endEncoding();
                cb->commit();
                committed.push_back(cb->retain());
                cb = ctx_.queue()->commandBuffer();
                cur_cb_ = cb;
                enc = encoder(cb, cur_layer_, -1);
                for (size_t i = b0; i < order.size(); ++i) {
                    const int e = order[i];
                    if (arena_->ensure(il, std::span<const int>(&e, 1), &slot[e], true))
                        continue;
                    // No unpinned slot left: run the experts so far, then their slots are free again.
                    run(b0, i, slot);
                    submit(cb, enc);
                    cur_cb_ = cb;
                    drain();
                    arena_->unpin(il);
                    b0 = i;
                    if (!arena_->ensure(il, std::span<const int>(&e, 1), &slot[e], true))
                        throw std::runtime_error("expert arena has no slot for one expert");
                }
                last_step_.ensure_ms += ms_since(t_ensure);
                run(b0, order.size(), slot);
                arena_->unpin(il);

                const CombineArgs ca{E, k};
                enc->setComputePipelineState(pso("moe_combine_rows"));
                enc->setBuffer(yg, 0, 0);
                enc->setBuffer(pos, 0, 1);
                enc->setBuffer(ew, 0, 2);
                enc->setBuffer(sho, 0, 3);
                enc->setBuffer(ffo, 0, 4);
                enc->setBytes(&ca, sizeof(ca), 5);
                enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
            }
            hc_post(enc, ffo, R, R2, pc, T);
            std::swap(R, R2);
        }

        // Logits of the chunk's last token, or of every token for a logits sink.
        cur_layer_ = cfg_.n_layer;
        const uint32_t nl = logits_sink_ ? T : 1;
        MTL::Buffer* xl = scratch("p_xl", uint64_t(nl) * E * 4);
        MTL::Buffer* xnl = scratch("p_xnl", uint64_t(nl) * E * 4);
        const HcArgs h1{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, nl};
        enc->setComputePipelineState(pso("hc_mean"));
        enc->setBuffer(R, uint64_t(T - nl) * hd * 4, 0);
        enc->setBuffer(xl, 0, 1);
        enc->setBytes(&h1, sizeof(h1), 2);
        enc->dispatchThreads(MTL::Size(E, nl, 1), MTL::Size(256, 1, 1));
        norm(enc, xl, &output_norm_, xnl, nl, E, E, cfg_.rms_eps);
        MTL::Buffer* lg = scratch("p_logits", uint64_t(nl) * cfg_.n_vocab * 4);
        proj(enc, output_, xnl, 0, lg, 0, nl);
        enc->endEncoding();
        finish(cb);
        drain();
        const auto* lgp = static_cast<const float*>(lg->contents());
        if (logits_sink_)
            for (uint32_t i = 0; i < T; ++i)
                logits_sink_(pos_ + i, lgp + uint64_t(i) * cfg_.n_vocab);
        std::memcpy(logits_.data(), lgp + uint64_t(nl - 1) * cfg_.n_vocab, uint64_t(cfg_.n_vocab) * 4);
        pos_ += T;
        total.ensure_ms += last_step_.ensure_ms;
        if (!samples_.empty()) {
            resolve_layer_times(cpu0, gpu0);
            for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
                total.layers[l].attn_ms += last_step_.layers[l].attn_ms;
                total.layers[l].ffn_ms += last_step_.layers[l].ffn_ms;
                total.layers[l].hit_ms += last_step_.layers[l].hit_ms;
                total.layers[l].miss_ms += last_step_.layers[l].miss_ms;
            }
        }
        pool->release();
    }
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
