#include "dsv41/engine.hpp"

#include "gguf/read_pool.hpp"
#include "quant/dequant.hpp"

#include <glaze/glaze.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <numbers>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <tuple>

namespace mako::dsv41 {

using core::ExpertArena;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct HcArgs {
    uint32_t n_embd, iters;
    float eps;
    uint32_t tokens;
};
struct RopeArgs {
    uint32_t head_dim, heads, n_rot, stride, pos0, pos_step, inverse;
};
struct RoundArgs {
    uint32_t n, kind, out_half;
};
struct PoolArgs {
    uint32_t dim, tokens, pos0, hist;
};
struct AttnArgs {
    uint32_t dim, heads, stride, ring, pos0, t0, ratio, top;
    float scale;
    uint32_t block;
};
struct RouteArgs {
    uint32_t n_expert, n_used, norm;
    float scale;
    uint32_t lookup, score;
    float miss_penalty;
};
struct SwigluArgs {
    uint32_t n;
    float limit;
};
struct CombineArgs {
    uint32_t n, n_used;
};
struct MarkovArgs {
    uint32_t vocab, rank, step;
    int32_t y;
};
struct IdxArgs {
    uint32_t dim, heads, stride, pos0, t0, ratio;
    float w_scale;
};
struct BlockArgs {
    uint32_t stride, blk_stride, pos0, t0, ratio, block, kblocks;
};
struct TopkArgs {
    uint32_t stride, k, ids_stride, flags_stride, pos0, t0, ratio, block;
};
struct EngramArgs {
    uint32_t dim;
    float eps;
};
struct EngramDecodeArgs {
    uint32_t dim, tokens, layers, cols;
};

constexpr uint32_t kFp8 = 0, kFp4E4m3 = 1, kFp4Pow2 = 2; // ds_round kinds
constexpr uint32_t kHistTokens = 8; // verify() runs at most this many tokens (the compressors keep a state per token)
constexpr uint32_t kAttnGroup = 64; // query tokens per attention dispatch (bounds its key copies and scores)

// RoPE frequencies for n_rot dims: base^(-2i/n_rot), with YaRN's ramp toward freq / factor when orig_ctx > 0.
std::vector<float> rope_freqs(uint32_t n_rot, float base, uint32_t orig_ctx, float factor, float beta_fast,
                              float beta_slow)
{
    std::vector<float> f(n_rot / 2);
    for (uint32_t i = 0; i < n_rot / 2; ++i)
        f[i] = 1.0f / std::pow(base, float(2 * i) / float(n_rot));
    if (orig_ctx > 0) {
        auto corr = [&](float rot) {
            return float(n_rot) * std::log(float(orig_ctx) / (rot * 2.0f * std::numbers::pi_v<float>)) /
                   (2.0f * std::log(base));
        };
        const float low = std::max(std::floor(corr(beta_fast)), 0.0f);
        const float high = std::min(std::ceil(corr(beta_slow)), float(n_rot - 1));
        for (uint32_t i = 0; i < n_rot / 2; ++i) {
            const float ramp = std::clamp((float(i) - low) / std::max(high - low, 1e-3f), 0.0f, 1.0f);
            f[i] = f[i] / factor * ramp + f[i] * (1.0f - ramp);
        }
    }
    return f;
}

} // namespace

struct Engine::Layer {
    int ratio = 0;
    int owner = -1; // the KV source (index into kv_sources_) whose compressed rows the layer reads
    bool kv_src = false, idx_src = false;
    int engram = -1; // Engram table
    DevTensor attn_norm, ffn_norm, hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    DevTensor q_a, q_a_norm, q_b, kv, kv_norm, out_a, out_b, sinks;
    DevTensor comp_kv, comp_gate, comp_norm, idx_k, idx_k_norm, idx_q_b, idx_proj;
    DevTensor engram_kv, engram_gate;
    MoeWeights moe;
    ExpertArena::LayerParts experts{};
};

struct Engine::Drafter {
    static constexpr uint32_t kBlocks = 3, kRows = 5, kExperts = 128, kUsed = 3, kRank = 256;
    static constexpr int32_t kNoise = 128799; // the block's rows after the first
    static constexpr std::array<uint32_t, 3> kTargets{37, 38, 39};
    std::array<Layer, kBlocks> blocks;
    std::array<std::array<DevTensor, 3>, kBlocks> experts; // gate, up, down: kExperts matrices each, contiguous
    std::array<uint64_t, 3> part_bytes{};
    DevTensor main_proj, main_norm, norm, markov_embed, markov_head, conf_w;
    std::array<mtl::Ref<MTL::Buffer>, kBlocks> rings;
    int fd = -1;
    ~Drafter()
    {
        if (fd >= 0)
            ::close(fd);
    }
};

Engine::Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx)
    : EngineBase(model, ctx, opts.telemetry), cfg_(cfg), opts_(opts)
{
    dense_quant_ = opts_.dense_quant;
    const auto t_load = Clock::now();
    if (cfg_.head_dim != 512 || cfg_.n_head != 64 || cfg_.n_rot != 64 || cfg_.hc != 4 || cfg_.n_head * cfg_.head_dim != cfg_.o_groups * 4096 ||
        cfg_.indexer_dim != 128 || cfg_.n_expert_used > 16)
        throw std::runtime_error("unsupported DeepSeek-V4.1 geometry");
    if (cfg_.gating != 4)
        throw std::runtime_error(std::format("expert gating function {} (only sqrt(softplus), 4)", cfg_.gating));
    dims_ = {cfg_.n_embd, cfg_.n_layer, cfg_.hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
             cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp};
    routing_ = {core::Routing::Score::sqrt_softplus, cfg_.expert_weights_norm, cfg_.expert_weights_scale,
                opts_.miss_penalty};
    // A verify of up to kMaxDecodeTokens tokens on the GEMVs and decode's MoE: each row as a step would compute it.
    gemm_min_tokens_ = kMaxDecodeTokens + 1;
    if (opts_.indexer_top_k)
        cfg_.indexer_top_k = opts_.indexer_top_k;
    opts_.prefill_chunk = std::max(1u, opts_.prefill_chunk);
    opts_.prefill_group = std::max(opts_.prefill_chunk, opts_.prefill_group);

    token_embd_ = model_.find("token_embd.weight");
    if (!token_embd_ || !quant::can_dequant(token_embd_.tensor->type))
        throw std::runtime_error("token_embd.weight missing or of an unsupported type");
    output_norm_ = upload("output_norm.weight");
    output_ = upload("output.weight");

    auto freqs = [&](bool compressed) {
        const auto f = compressed ? rope_freqs(cfg_.n_rot, cfg_.compress_rope_base, cfg_.yarn_orig_ctx, cfg_.yarn_factor,
                                               cfg_.yarn_beta_fast, cfg_.yarn_beta_slow)
                                  : rope_freqs(cfg_.n_rot, cfg_.rope_base, 0, 1.0f, 0, 0);
        return mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(f.data(), f.size() * 4, MTL::ResourceStorageModeShared));
    };
    freq_window_ = freqs(false);
    freq_comp_ = freqs(true);

    for (int32_t s : cfg_.kv_source_layers)
        kv_sources_.push_back(s);
    std::ranges::sort(kv_sources_);
    const auto& eg_layers = cfg_.engram.layers;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        auto L = std::make_unique<Layer>();
        const std::string p = std::format("blk.{}.", il);
        L->ratio = cfg_.compress_ratios[il];
        L->kv_src = std::ranges::find(kv_sources_, int32_t(il)) != kv_sources_.end();
        L->idx_src = std::ranges::find(cfg_.indexer_source_layers, int32_t(il)) != cfg_.indexer_source_layers.end();
        for (size_t s = 0; s < kv_sources_.size(); ++s)
            if (kv_sources_[s] <= int32_t(il))
                L->owner = int(s);
        if (L->ratio && L->owner < 0)
            throw std::runtime_error(std::format("layer {} compresses but has no KV source", il));
        if (L->ratio && cfg_.compress_ratios[kv_sources_[L->owner]] != L->ratio)
            throw std::runtime_error(std::format("layer {}: compress ratio differs from its KV source's", il));
        if (L->kv_src && L->ratio != 1 && L->ratio != 2)
            throw std::runtime_error(std::format("layer {}: compress ratio {} unsupported", il, L->ratio));
        L->attn_norm = upload(p + "attn_norm.weight");
        L->ffn_norm = upload(p + "ffn_norm.weight");
        L->hc_attn_fn = upload(p + "hc_attn_fn.weight");
        L->hc_attn_base = upload(p + "hc_attn_base.weight");
        L->hc_attn_scale = upload(p + "hc_attn_scale.weight");
        L->hc_ffn_fn = upload(p + "hc_ffn_fn.weight");
        L->hc_ffn_base = upload(p + "hc_ffn_base.weight");
        L->hc_ffn_scale = upload(p + "hc_ffn_scale.weight");
        L->q_a = upload(p + "attn_q_a.weight");
        L->q_a_norm = upload(p + "attn_q_a_norm.weight");
        L->q_b = upload(p + "attn_q_b.weight");
        L->kv = upload(p + "attn_kv.weight");
        L->kv_norm = upload(p + "attn_kv_a_norm.weight");
        L->out_a = upload(p + "attn_output_a.weight");
        // o_groups matrices of o_lora_rank rows, each over its group's heads
        L->out_a.dims = {L->out_a.dims[0], cfg_.o_lora_rank, cfg_.o_groups};
        L->out_b = upload(p + "attn_output_b.weight");
        L->sinks = upload(p + "attn_sinks.weight");
        if (L->kv_src) {
            L->comp_kv = upload(p + "attn_compressor_kv.weight");
            L->comp_norm = upload(p + "attn_compressor_norm.weight");
            if (L->ratio == 2)
                L->comp_gate = upload(p + "attn_compressor_gate.weight");
            L->idx_k = upload(p + "indexer.attn_k.weight");
            L->idx_k_norm = upload(p + "indexer.k_norm.weight");
        }
        if (L->idx_src) {
            L->idx_q_b = upload(p + "indexer.attn_q_b.weight");
            L->idx_proj = upload(p + "indexer.proj.weight");
        }
        if (auto it = std::ranges::find(eg_layers, int32_t(il)); it != eg_layers.end()) {
            L->engram = int(it - eg_layers.begin());
            L->engram_kv = upload(p + "engram_kv.weight");
            L->engram_gate = upload(p + "engram_gate.weight");
        }
        L->moe.router = upload(p + "ffn_gate_inp.weight");
        L->moe.bias = upload(p + "exp_probs_b.bias");
        L->moe.sh_gate = upload(p + "ffn_gate_shexp.weight");
        L->moe.sh_up = upload(p + "ffn_up_shexp.weight");
        L->moe.sh_down = upload(p + "ffn_down_shexp.weight");
        L->moe.clamp_exp = cfg_.swiglu_clamp_exp[il];
        L->moe.clamp_sh = cfg_.swiglu_clamp_shexp[il];
        L->experts = {expert(p + "ffn_gate_exps.weight", true), expert(p + "ffn_up_exps.weight", true),
                      expert(p + "ffn_down_exps.weight", true)};
        layers_.push_back(std::move(L));
        layer_mla_.push_back(1);
    }

    auto eg = Engram::open(cfg_, model_);
    if (!eg)
        throw std::runtime_error(eg.error());
    engram_.emplace(std::move(*eg));
    engram_pool_ = std::make_unique<gguf::ReadPool>(16);
    engram_history_.assign(engram_->history(), -1);

    const uint32_t D = cfg_.head_dim, W = cfg_.sliding_window;
    auto buffer = [&](uint64_t bytes) {
        mtl::Ref<MTL::Buffer> b(ctx_.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
        std::memset(b->contents(), 0, b->length());
        return b;
    };
    uint64_t context_bytes = 0;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        ring_.push_back(buffer(uint64_t(W) * D * 4));
        context_bytes += ring_.back()->length();
    }
    for (int32_t s : kv_sources_) {
        const uint32_t ratio = uint32_t(cfg_.compress_ratios[s]);
        const uint64_t rows = opts_.max_ctx / ratio + 1;
        ckv_.push_back(buffer(rows * D * 2));
        ik_.push_back(buffer(rows * cfg_.indexer_dim * 4));
        comp_state_.push_back(buffer(2 * D * 4));
        comp_hist_.push_back(buffer(uint64_t(kHistTokens) * 2 * D * 4));
        context_bytes += ckv_.back()->length() + ik_.back()->length();
    }
    logits_.resize(cfg_.n_vocab);
    if (!opts_.dspark.empty())
        load_drafter(opts_.dspark);

    load_.dense_ms = ms_since(t_load);
    set_staging_limit(opts_.stage_gb);
    ArenaInit init;
    for (const auto& L : layers_)
        init.parts.push_back(L->experts);
    for (uint32_t il = 0; il < cfg_.n_layer; ++il)
        init.moe_layers.push_back(il);
    init.context_bytes = context_bytes + uint64_t(opts_.checkpoints) * checkpoint_bytes();
    init.lendable = double(prefill_scratch(opts_.prefill_chunk, opts_.prefill_group));
    init.arena_gb = opts_.arena_gb;
    init.simulate_ram_gb = opts_.simulate_ram_gb;
    init.expert_counts = opts_.expert_counts;
    init_runtime(std::move(init));
}

Engine::~Engine() = default;

// Scratch bytes of a prefill: per group token (streams, mixes, FFN rows, Engram rows, routed outputs, selections),
// per chunk token (the attention's projections and outputs), the attention and indexer score buffers, the expert
// batch buffers.
uint64_t Engine::prefill_scratch(uint32_t chunk, uint32_t group) const
{
    const uint64_t E = cfg_.n_embd, hd = uint64_t(cfg_.hc) * E, k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
    const uint64_t HD = uint64_t(cfg_.n_head) * cfg_.head_dim, top = cfg_.indexer_top_k;
    const uint64_t eg = uint64_t(cfg_.engram.layers.size()) * cfg_.engram.columns() * cfg_.engram.head_dim;
    const uint64_t blocks = opts_.max_ctx / std::max(1u, cfg_.candidate_block_size) + 2;
    const uint64_t per_group = 3 * hd + 3 * 24 + 4 * E + 5 * E + eg + k * E + 2 * cfg_.n_expert + 64 + top + blocks / 4;
    const uint64_t per_chunk = 3 * HD + 2 * cfg_.q_lora_rank + 6 * uint64_t(cfg_.head_dim) + 2 * uint64_t(cfg_.o_groups) *
                               cfg_.o_lora_rank + uint64_t(cfg_.indexer_heads) * (cfg_.indexer_dim + 1) + 3 * nff + HD / 2;
    const uint64_t scores = uint64_t(kAttnGroup) * cfg_.n_head * (cfg_.sliding_window + top) +
                            (64ull << 20) / 4; // the indexer's
    const uint64_t batch = std::min<uint64_t>(kBatchRows, uint64_t(group) * k) * (E / 2 + 2 * nff + nff / 2);
    const uint64_t staging = group >= kStageMinTokens ? staging_bytes() : 0;
    return (uint64_t(group) * per_group + uint64_t(chunk) * per_chunk + scores + batch) * 4 + staging + (96ull << 20);
}

void Engine::reset()
{
    pos_ = 0;
    engram_history_.assign(engram_->history(), -1);
    for (auto& b : comp_state_)
        std::memset(b->contents(), 0, b->length());
}

uint64_t Engine::checkpoint_bytes() const
{
    uint64_t n = engram_history_.size() * 4;
    for (const auto* v : {&ring_, &comp_state_})
        for (const auto& b : *v)
            n += b->length();
    if (drafter_)
        for (const auto& b : drafter_->rings)
            n += b->length();
    return n;
}

Engine::Checkpoint Engine::checkpoint()
{
    Checkpoint cp;
    cp.pos = pos_;
    cp.state.reserve(checkpoint_bytes());
    for (const auto* v : {&ring_, &comp_state_})
        for (const auto& b : *v) {
            const auto* p = static_cast<const std::byte*>(b->contents());
            cp.state.insert(cp.state.end(), p, p + b->length());
        }
    if (drafter_)
        for (const auto& b : drafter_->rings) {
            const auto* p = static_cast<const std::byte*>(b->contents());
            cp.state.insert(cp.state.end(), p, p + b->length());
        }
    const auto* h = reinterpret_cast<const std::byte*>(engram_history_.data());
    cp.state.insert(cp.state.end(), h, h + engram_history_.size() * 4);
    return cp;
}

void Engine::restore(const Checkpoint& cp)
{
    if (cp.pos > opts_.max_ctx || cp.state.size() != checkpoint_bytes())
        throw std::runtime_error("checkpoint from another engine");
    size_t off = 0;
    for (const auto* v : {&ring_, &comp_state_})
        for (const auto& b : *v) {
            std::memcpy(b->contents(), cp.state.data() + off, b->length());
            off += b->length();
        }
    if (drafter_)
        for (const auto& b : drafter_->rings) {
            std::memcpy(b->contents(), cp.state.data() + off, b->length());
            off += b->length();
        }
    std::memcpy(engram_history_.data(), cp.state.data() + off, engram_history_.size() * 4);
    pos_ = cp.pos;
}

void Engine::set_ablation(std::optional<mtl::Ablation> ablation)
{
    ablation_ = std::move(ablation);
    ablate_attn_ = ablation_ ? ablation_->site("attn_out") : -1;
    ablate_shexp_ = ablation_ ? ablation_->site("shexp_out") : -1;
}

void Engine::ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* x, uint64_t off,
                    uint32_t rows)
{
    if (!ablation_ || !ablation_->active(il, site))
        return;
    ablation_->apply(enc, pso("ablate_rows"), il, site, x, off, rows);
    after(enc, "ablate_rows");
}

void Engine::rope(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, bool compressed, uint32_t head_dim,
                  uint32_t heads, uint32_t pos0, uint32_t T, uint32_t step, bool inverse)
{
    const RopeArgs a{head_dim, heads, cfg_.n_rot, heads * head_dim, pos0, step, inverse ? 1u : 0u};
    enc->setComputePipelineState(pso("ds_rope"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(compressed ? freq_comp_.get() : freq_window_.get(), 0, 1);
    enc->setBytes(&a, sizeof(a), 2);
    for (int r = repeats("ds_rope"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(cfg_.n_rot / 2, heads, T), MTL::Size(32, 1, 1));
    after(enc, "ds_rope");
}

void Engine::round(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y, uint64_t y_off,
                   uint32_t n, uint32_t kind, bool out_half)
{
    const RoundArgs a{n, kind, out_half ? 1u : 0u};
    enc->setComputePipelineState(pso("ds_round"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(y, y_off, 1);
    enc->setBytes(&a, sizeof(a), 2);
    for (int r = repeats("ds_round"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "ds_round");
}

void Engine::hc_lagged(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale,
                       const DevTensor& base, MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, MTL::Buffer* pc_prev,
                       uint32_t T, uint64_t r_off, uint64_t pc_off)
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
    enc->setBuffer(pc, pc_off, 3);
    enc->setBytes(&ha, sizeof(ha), 4);
    enc->dispatchThreads(MTL::Size(T, 1, 1), MTL::Size(std::min<uint32_t>(T, 64), 1, 1));
    after(enc, "hc_mix");
    enc->setComputePipelineState(pso("hc_collapse"));
    enc->setBuffer(R, r_off, 0);
    enc->setBuffer(pc_prev, pc_off, 1);
    enc->setBuffer(x, 0, 2);
    enc->setBytes(&ha, sizeof(ha), 3);
    enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    after(enc, "hc_collapse");
}

// T tokens' attention at positions pos0..: each token's window rows plus the compressed rows its KV source keeps
// (all of them up to indexer_top_k, else the indexer's selection), a sink per head, then the grouped output
// projection.
void Engine::attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att,
                       uint32_t T, uint32_t pos0, uint32_t g0, MTL::Buffer* ring, bool block)
{
    MTL::Buffer* ringb = ring ? ring : ring_[il].get();
    const uint32_t H = cfg_.n_head, D = cfg_.head_dim, QR = cfg_.q_lora_rank, W = cfg_.sliding_window;
    const uint32_t Hi = cfg_.indexer_heads, Di = cfg_.indexer_dim, top = cfg_.indexer_top_k, HD = H * D;
    const uint32_t ratio = uint32_t(L.ratio);
    const bool comp = ratio != 0;
    MTL::Buffer* qr = scratch("ds_qr", uint64_t(T) * QR * 4);
    MTL::Buffer* qrn = scratch("ds_qrn", uint64_t(T) * QR * 4);
    MTL::Buffer* q = scratch("ds_q", uint64_t(T) * HD * 4);
    MTL::Buffer* kv = scratch("ds_kv", uint64_t(T) * D * 4);
    MTL::Buffer* kvn = scratch("ds_kvn", uint64_t(T) * D * 4);
    MTL::Buffer* kvc = scratch("ds_kvc", uint64_t(T) * D * 4);
    part(enc, "attn q kv");
    proj(enc, L.q_a, xn, 0, qr, 0, T);
    norm(enc, qr, &L.q_a_norm, qrn, T, QR, QR, cfg_.rms_eps);
    proj(enc, L.q_b, qrn, 0, q, 0, T);
    proj(enc, L.kv, xn, 0, kv, 0, T);
    norm(enc, kv, &L.kv_norm, kvn, T, D, D, cfg_.rms_eps);
    rope(enc, q, 0, comp, D, H, pos0, T);
    rope(enc, kvn, 0, comp, D, 1, pos0, T);
    round(enc, kvn, 0, kvc, 0, T * D, kFp8);
    if (T == 1)
        probe(enc, "attn_q", int(il), q, HD);

    // A KV source's compressor: a new compressed row (and index key) per completed group of `ratio` tokens.
    if (L.kv_src) {
        part(enc, "attn compressor");
        const uint32_t src = uint32_t(L.owner);
        MTL::Buffer* lat = scratch("ds_lat", uint64_t(T) * D * 4);
        MTL::Buffer* latn = scratch("ds_latn", uint64_t(T) * D * 4);
        // rows [row0, row0 + n_emit): those whose group ends within the tokens
        const uint32_t row0 = pos0 / ratio, n_emit = (pos0 + T) / ratio - row0;
        if (ratio == 1) {
            proj(enc, L.comp_kv, xn, 0, lat, 0, T);
        } else {
            MTL::Buffer* ck = scratch("ds_ck", uint64_t(T) * D * 4);
            MTL::Buffer* cg = scratch("ds_cg", uint64_t(T) * D * 4);
            proj(enc, L.comp_kv, xn, 0, ck, 0, T);
            proj(enc, L.comp_gate, xn, 0, cg, 0, T);
            const PoolArgs pa{D, T, pos0, verifying_ ? std::min<uint32_t>(T, kHistTokens) : 0u};
            enc->setComputePipelineState(pso("ds_pool2"));
            enc->setBuffer(ck, 0, 0);
            enc->setBuffer(cg, 0, 1);
            enc->setBuffer(comp_state_[src].get(), 0, 2);
            enc->setBuffer(lat, 0, 3);
            enc->setBytes(&pa, sizeof(pa), 4);
            enc->setBuffer(comp_hist_[src].get(), 0, 5);
            enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(256, 1, 1));
            after(enc, "ds_pool2");
        }
        if (n_emit) {
            norm(enc, lat, &L.comp_norm, latn, n_emit, D, D, cfg_.rms_eps);
            MTL::Buffer* ikr = scratch("ds_ikr", uint64_t(n_emit) * Di * 4);
            MTL::Buffer* ikn = scratch("ds_ikn", uint64_t(n_emit) * Di * 4);
            proj(enc, L.idx_k, latn, 0, ikr, 0, n_emit);
            norm(enc, ikr, &L.idx_k_norm, ikn, n_emit, Di, Di, cfg_.rms_eps);
            rope(enc, ikn, 0, true, Di, 1, row0 * ratio, n_emit, ratio);
            round(enc, ikn, 0, ik_[src].get(), uint64_t(row0) * Di * 4, n_emit * Di, kFp4Pow2);
            rope(enc, latn, 0, true, D, 1, row0 * ratio, n_emit, ratio);
            round(enc, latn, 0, ckv_[src].get(), uint64_t(row0) * D * 2, n_emit * D, kFp4E4m3, true);
        }
    }

    // The indexer (index sources) for the tokens with more visible compressed rows than it keeps.
    MTL::Buffer* sel = scratch("ds_sel", uint64_t(opts_.prefill_group) * top * 4);
    const uint32_t bs = cfg_.candidate_block_size, kb = cfg_.candidate_top_k_blocks;
    const uint32_t blk_stride = opts_.max_ctx / bs + 2;
    MTL::Buffer* keep = scratch("ds_keep", uint64_t(opts_.prefill_group) * blk_stride);
    if (L.idx_src && comp && (pos0 + T) / ratio > top) {
        part(enc, "attn indexer");
        // first token with more than `top` rows
        const uint32_t t_s = (top + 1) * ratio > pos0 + 1 ? (top + 1) * ratio - 1 - pos0 : 0;
        MTL::Buffer* iq = scratch("ds_iq", uint64_t(T) * Hi * Di * 4);
        MTL::Buffer* iw = scratch("ds_iw", uint64_t(T) * Hi * 4);
        proj(enc, L.idx_q_b, qrn, 0, iq, 0, T);
        rope(enc, iq, 0, true, Di, Hi, pos0, T);
        round(enc, iq, 0, iq, 0, T * Hi * Di, kFp4Pow2);
        proj(enc, L.idx_proj, xn, 0, iw, 0, T);
        const uint32_t stride = (pos0 + T) / ratio;
        const uint32_t S = std::clamp<uint32_t>(uint32_t((64ull << 20) / (uint64_t(stride) * 4)), 1, 64);
        MTL::Buffer* score = scratch("ds_score", uint64_t(S) * stride * 4);
        MTL::Buffer* blk = scratch("ds_blk", uint64_t(S) * blk_stride * 4);
        MTL::Buffer* blk_ids = scratch("ds_blk_ids", uint64_t(S) * kb * 4);
        const int cand = cfg_.candidate_source_layer;
        for (uint32_t t0 = t_s; t0 < T; t0 += S) {
            const uint32_t n = std::min(S, T - t0);
            const IdxArgs ia{Di, Hi, stride, pos0, t0, ratio, 1.0f / std::sqrt(float(Di * Hi))};
            dispatch(enc, "ds_idx_score", {iq, iw, ik_[L.owner].get(), score}, &ia, sizeof(ia),
                     MTL::Size(uint64_t(stride) * 32, n, 1), MTL::Size(256, 1, 1));
            // The candidate stage: the source layer keeps each token's best blocks of rows; later index sources
            // score only those (no restriction while there are at most kb blocks).
            const BlockArgs ba{stride, blk_stride, pos0, t0, ratio, bs, kb};
            if (int(il) == cand && (stride + bs - 1) / bs > kb) {
                dispatch(enc, "ds_block_max", {score, blk}, &ba, sizeof(ba), MTL::Size(blk_stride, n, 1),
                         MTL::Size(256, 1, 1));
                const TopkArgs ta{blk_stride, kb, kb, blk_stride, pos0, t0, ratio, bs};
                enc->setComputePipelineState(pso("ds_topk"));
                enc->setBuffer(blk, 0, 0);
                enc->setBuffer(blk_ids, 0, 1);
                enc->setBuffer(keep, uint64_t(g0 + t0) * blk_stride, 2);
                enc->setBytes(&ta, sizeof(ta), 3);
                for (int r = repeats("ds_topk"); r > 0; --r)
                    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(1024, 1, 1));
                after(enc, "ds_topk");
            } else if (cand >= 0 && cand < int(il) && (stride + bs - 1) / bs > kb) {
                enc->setComputePipelineState(pso("ds_block_mask"));
                enc->setBuffer(score, 0, 0);
                enc->setBuffer(keep, uint64_t(g0 + t0) * blk_stride, 1);
                enc->setBytes(&ba, sizeof(ba), 2);
                enc->dispatchThreads(MTL::Size(stride, n, 1), MTL::Size(256, 1, 1));
                after(enc, "ds_block_mask");
            }
            const TopkArgs ta{stride, top, top, 0, pos0, t0, ratio, 1};
            enc->setComputePipelineState(pso("ds_topk"));
            enc->setBuffer(score, 0, 0);
            enc->setBuffer(sel, uint64_t(g0 + t0) * top * 4, 1);
            enc->setBuffer(keep, 0, 2);
            enc->setBytes(&ta, sizeof(ta), 3);
            for (int r = repeats("ds_topk"); r > 0; --r)
                enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(1024, 1, 1));
            after(enc, "ds_topk");
        }
    }

    MTL::Buffer* o = scratch("ds_o", uint64_t(T) * HD * 4);
    const uint32_t stride = (W + std::max(top, T) + 63) / 64 * 64; // whole tiles of 64 keys for the matmuls
    const uint32_t ng = std::min(T, kAttnGroup);
    MTL::Buffer* qh = scratch("ds_qh", uint64_t(T) * HD * 2);
    MTL::Buffer* sc = scratch("ds_sc", uint64_t(ng) * H * stride * 4);
    MTL::Buffer* ph = scratch("ds_ph", uint64_t(ng) * H * stride * 2);
    MTL::Buffer* cbuf = comp ? ckv_[L.owner].get() : ringb;
    part(enc, "attn core");
    gemm_->to_half(enc, q, 0, T, HD, 0, qh, 0);
    after(enc, "to_half");
    for (uint32_t t0 = 0; t0 < T; t0 += kAttnGroup) {
        const uint32_t n = std::min(kAttnGroup, T - t0);
        const AttnArgs aa{D, H, stride, W, pos0, t0, ratio, top, 1.0f / std::sqrt(float(D)), block ? T : 0u};
        auto keys = [&](const char* fn) {
            enc->setComputePipelineState(pso(fn));
            enc->setBuffer(kvc, 0, 1);
            enc->setBuffer(ringb, 0, 2);
            enc->setBuffer(cbuf, 0, 3);
            enc->setBuffer(sel, uint64_t(g0) * top * 4, 4);
            enc->setBytes(&aa, sizeof(aa), 6);
        };
        keys("ds_attn_scores");
        enc->setBuffer(qh, 0, 0);
        enc->setBuffer(sc, 0, 5);
        for (int r = repeats("ds_attn_scores"); r > 0; --r)
            enc->dispatchThreadgroups(MTL::Size(stride / 32, n, 1), MTL::Size(128, 1, 1));
        after(enc, "ds_attn_scores");
        enc->setComputePipelineState(pso("ds_attn_softmax"));
        enc->setBuffer(sc, 0, 0);
        enc->setBuffer(L.sinks.buf, L.sinks.off, 1);
        enc->setBytes(&aa, sizeof(aa), 2);
        enc->setBuffer(ph, 0, 3);
        for (int r = repeats("ds_attn_softmax"); r > 0; --r)
            enc->dispatchThreadgroups(MTL::Size(H, n, 1), MTL::Size(256, 1, 1));
        after(enc, "ds_attn_softmax");
        keys("ds_attn_out");
        enc->setBuffer(ph, 0, 0);
        enc->setBuffer(o, 0, 5);
        for (int r = repeats("ds_attn_out"); r > 0; --r)
            enc->dispatchThreadgroups(MTL::Size(D / 64, n, 1), MTL::Size(128, 1, 1));
        after(enc, "ds_attn_out");
    }
    if (!block) {
        const uint32_t ra[4] = {D, T, pos0, W};
        dispatch(enc, "ds_ring_store", {kvc, ringb}, ra, sizeof(ra), MTL::Size(D, std::min(T, W), 1),
                 MTL::Size(256, 1, 1));
    }

    part(enc, "attn output");
    rope(enc, o, 0, comp, D, H, pos0, T, 1, true);
    round(enc, o, 0, o, 0, T * HD, kFp8);
    // Grouped output projection: group g's heads (rows g * gw of each token) through its o_lora_rank rows.
    const uint32_t G = cfg_.o_groups, R = cfg_.o_lora_rank, gw = HD / G;
    MTL::Buffer* low = scratch("ds_low", uint64_t(T) * G * R * 4);
    if (T <= kMaxDecodeTokens) {
        for (uint32_t t = 0; t < T; ++t)
            gemv(enc, L.out_a, o, uint64_t(t) * HD * 4, low, uint64_t(t) * G * R * 4, G);
    } else {
        const uint64_t mat = mtl::Gemv::row_bytes(L.out_a.type, gw) * R;
        for (uint32_t g = 0; g < G; ++g) {
            MTL::Buffer* xh = halves(enc, o, uint64_t(g) * gw * 4, T, gw, HD);
            gemm_->encode(enc, L.out_a.type, gw, R, T, L.out_a.buf, L.out_a.off + g * mat, xh, 0, low,
                          uint64_t(g) * R * 4, G * R);
            after(enc, "out_a gemm");
        }
    }
    if (T == 1)
        probe(enc, "attn_low", int(il), low, uint64_t(G) * R);
    proj(enc, L.out_b, low, 0, att, 0, T);
}

const std::vector<float>& Engine::step(int32_t token)
{
    const int32_t t[1] = {token};
    const auto t0 = Clock::now();
    run(t, false);
    ++spec_stats_.steps;
    spec_stats_.step_ms += ms_since(t0);
    spec_stats_.step_misses += last_step_.misses;
    spec_stats_.step_read_ms += last_step_.ensure_ms;
    spec_stats_.step_gpu_ms += last_step_.gpu_ms;
    spec_stats_.step_wait_ms += last_step_.route_wait_ms;
    spec_stats_.step_encode_ms += last_step_.encode_ms;
    spec_stats_.step_prep_ms += last_step_.prep_ms;
    spec_stats_.step_wall_ms += last_step_.wall_ms;
    for (const auto& l : last_step_.layers) {
        spec_stats_.step_attn_ms += l.attn_ms;
        spec_stats_.step_ffn_ms += l.ffn_ms;
        spec_stats_.step_hit_ms += l.hit_ms;
        spec_stats_.step_miss_ms += l.miss_ms;
    }
    for (const auto* list : {&last_step_.parts, &last_step_.kernels})
    for (const core::KernelTime& k : *list) {
        auto it = std::ranges::find(spec_stats_.parts, k.kernel, &core::KernelTime::kernel);
        if (it == spec_stats_.parts.end())
            it = spec_stats_.parts.insert(it, core::KernelTime{k.kernel, 0, 0});
        it->ms += k.ms;
        it->count += k.count;
    }
    return logits_;
}

const std::vector<float>& Engine::prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled)
{
    if (tokens.empty())
        return logits_;
    if (tokens.size() == 1 && !logits_sink_)
        return step(tokens[0]);
    const auto t_start = Clock::now();
    const ExpertArena::Stats arena_before = arena_->stats();
    const double gpu_before = stats_.gpu_exec_ms;
    core::StepStats total;
    total.pos = pos_;
    // Kernel timing is for steps: a prefill's dispatches would overflow the sample buffers.
    struct Level {
        core::Telemetry& tel;
        core::Telemetry saved;
        ~Level() { tel = saved; }
    } level{tel_, tel_};
    if (tel_ == core::Telemetry::kernels)
        tel_ = core::Telemetry::summary;
    if (timing()) {
        total.layers.assign(cfg_.n_layer, core::LayerTimes{});
        for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
            total.layers[l].layer = l;
            total.layers[l].mla = true;
        }
    }
    const ScratchLoan loan(*this, prefill_scratch(std::min<uint32_t>(opts_.prefill_chunk, uint32_t(tokens.size())),
                                                  std::min<uint32_t>(opts_.prefill_group, uint32_t(tokens.size()))));
    for (size_t g0 = 0; g0 < tokens.size(); g0 += opts_.prefill_group) {
        const size_t n = std::min<size_t>(opts_.prefill_group, tokens.size() - g0);
        run(tokens.subspan(g0, n), bool(logits_sink_));
        total.ensure_ms += last_step_.ensure_ms;
        for (size_t l = 0; l < total.layers.size() && l < last_step_.layers.size(); ++l) {
            total.layers[l].attn_ms += last_step_.layers[l].attn_ms;
            total.layers[l].ffn_ms += last_step_.layers[l].ffn_ms;
            total.layers[l].hit_ms += last_step_.layers[l].hit_ms;
            total.layers[l].miss_ms += last_step_.layers[l].miss_ms;
        }
        for (const core::KernelTime& k : last_step_.parts) {
            auto it = std::ranges::find(total.parts, k.kernel, &core::KernelTime::kernel);
            if (it == total.parts.end())
                it = total.parts.insert(it, core::KernelTime{k.kernel, 0, 0});
            it->ms += k.ms;
            it->count += k.count;
        }
        if (cancelled && g0 + n < tokens.size() && cancelled())
            break;
    }
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

void Engine::run(std::span<const int32_t> tokens, bool all_logits)
{
    const uint32_t G = uint32_t(tokens.size());
    if (G == 0 || G > opts_.prefill_group)
        throw std::runtime_error("bad token count");
    for (int32_t t : tokens)
        if (t < 0 || uint32_t(t) >= cfg_.n_vocab)
            throw std::runtime_error("token out of range");
    if (pos_ + G > opts_.max_ctx)
        throw std::runtime_error("context full");
    apply_pressure();
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E, chunk = std::min(opts_.prefill_chunk, G);
    const bool decode = G <= kMaxDecodeTokens; // MoE with GPU routing and served misses, no expert GEMMs
    step_start_ = Clock::now();
    encoding_ = true;
    const ExpertArena::Stats before = arena_->stats();
    const Stats stats_before = stats_;
    last_step_ = core::StepStats{};
    last_step_.pos = pos_;
    last_step_.token = tokens[0];
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // Engram rows of these positions (both tables), read before the GPU work that needs them.
    const uint32_t eg_layers = engram_->layers(), eg_cols = engram_->columns(), eg_dim = engram_->dim();
    const uint64_t eg_rows = uint64_t(G) * eg_layers * eg_cols;
    MTL::Buffer* eg_raw = scratch("ds_eg_raw", eg_rows * engram_->row_bytes());
    MTL::Buffer* eg = scratch("ds_eg", eg_rows * eg_dim * 4);
    {
        std::vector<int64_t> rows(eg_rows);
        engram_->hash(tokens, engram_history_, rows);
        engram_->read_raw(rows, static_cast<uint8_t*>(eg_raw->contents()), *engram_pool_);
        engram_history_ = engram_->advance(engram_history_, tokens);
    }

    MTL::Buffer* xg = scratch("ds_xg", uint64_t(G) * E * 4);
    MTL::Buffer* xn_g = scratch("ds_xng", uint64_t(G) * E * 4);
    MTL::Buffer* ffo = scratch("ds_ffo", uint64_t(G) * E * 4);
    MTL::Buffer* x = scratch("ds_x", uint64_t(chunk) * E * 4);
    MTL::Buffer* xn = scratch("ds_xn", uint64_t(chunk) * E * 4);
    MTL::Buffer* att = scratch("ds_att", uint64_t(chunk) * E * 4);
    MTL::Buffer* ekv = scratch("ds_ekv", uint64_t(G) * 5 * E * 4);
    MTL::Buffer* R = scratch("ds_R", uint64_t(G) * hd * 4);
    MTL::Buffer* R2 = scratch("ds_R2", uint64_t(G) * hd * 4);
    MTL::Buffer* pcs[2] = {scratch("ds_pcA", uint64_t(G) * 24 * 4), scratch("ds_pcB", uint64_t(G) * 24 * 4)};
    MTL::Buffer* pc0 = scratch("ds_pc0", uint64_t(G) * 24 * 4);
    {
        auto* p0 = static_cast<float*>(pc0->contents());
        std::fill_n(p0, uint64_t(G) * 24, 0.0f);
        for (uint32_t t = 0; t < G; ++t)
            p0[t * 24] = 1.0f; // the first sublayer's input is stream 0
        const gguf::Tensor& t = *token_embd_.tensor;
        const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
        std::vector<std::byte> row(rb);
        for (uint32_t i = 0; i < G; ++i) {
            gguf::read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(tokens[i]) * rb);
            quant::dequant_row(t.type, row.data(), static_cast<float*>(xg->contents()) + uint64_t(i) * E, E);
        }
    }

    const double prep_ms = ms_since(step_start_);
    if (!decode)
        begin_staging(G);
    struct StagingEnd {
        Engine& e;
        ~StagingEnd() { e.end_staging(); }
    } staging_end{*this};
    MTL::Timestamp cpu0 = 0, gpu0 = 0;
    if (timing())
        ctx_.device()->sampleTimestamps(&cpu0, &gpu0);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    cur_layer_ = cfg_.n_layer;
    MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
    // Command buffers committed without waiting (the routed experts' batches); drained before the next group.
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
    const EngramDecodeArgs eda{eg_dim, G, eg_layers, eg_cols};
    dispatch(enc, "ds_engram_decode", {eg_raw, eg}, &eda, sizeof(eda), MTL::Size(eg_dim, eg_rows, 1),
             MTL::Size(256, 1, 1));
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, G};
    dispatch(enc, "hc_init", {xg, R}, &ha, sizeof(ha), MTL::Size(E, G, 1), MTL::Size(256, 1, 1));
    MTL::Buffer* prev = pc0;
    int cur = 0;
    MTL::Buffer* taps = drafter_ ? scratch("ds_taps", uint64_t(G) * 3 * E * 4) : nullptr;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        cur_layer_ = il;
        if (timing()) {
            enc->endEncoding();
            enc = encoder(cb, il, 0);
        }
        if (drafter_)
            for (uint32_t j = 0; j < Drafter::kTargets.size(); ++j)
                if (Drafter::kTargets[j] == il) {
                    const uint32_t ta[4] = {E, 3 * E, j, 0};
                    dispatch(enc, "ds_tap", {R, taps}, ta, sizeof(ta), MTL::Size(E, G, 1), MTL::Size(256, 1, 1));
                }
        if (L.engram >= 0) {
            part(enc, "engram");
            proj(enc, L.engram_kv, eg, uint64_t(L.engram) * G * eg_cols * eg_dim * 4, ekv, 0, G);
            const EngramArgs ea{E, cfg_.rms_eps};
            enc->setComputePipelineState(pso("ds_engram_gate"));
            enc->setBuffer(R, 0, 0);
            enc->setBuffer(ekv, 0, 1);
            enc->setBuffer(L.engram_gate.buf, L.engram_gate.off, 2);
            enc->setBytes(&ea, sizeof(ea), 3);
            enc->dispatchThreadgroups(MTL::Size(4, G, 1), MTL::Size(256, 1, 1));
            after(enc, "ds_engram_gate");
        }
        for (uint32_t c0 = 0; c0 < G; c0 += chunk) {
            const uint32_t T = std::min(chunk, G - c0);
            const uint64_t r_off = uint64_t(c0) * hd * 4, pc_off = uint64_t(c0) * 24 * 4;
            part(enc, "attn hc and norm");
            hc_lagged(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R, x, pcs[cur], prev, T, r_off, pc_off);
            norm(enc, x, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
            attention(enc, L, il, xn, att, T, pos_ + c0, c0);
            part(enc, "attn hc post");
            ablate(enc, il, ablate_attn_, att, 0, T);
            if (G == 1)
                probe(enc, "attn_out", int(il), att, E);
            hc_post(enc, att, R, R2, pcs[cur], T, r_off, 0, pc_off);
        }
        std::swap(R, R2);
        prev = pcs[cur];
        cur ^= 1;
        if (timing()) {
            enc->endEncoding();
            enc = encoder(cb, il, 1);
        }

        part(enc, "ffn hc and norm");
        hc_lagged(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, R, xg, pcs[cur], prev, G, 0, 0);
        norm(enc, xg, &L.ffn_norm, xn_g, G, E, E, cfg_.rms_eps);
        if (decode) {
            moe_decode(cb, enc, L.moe, il, xn_g, ffo, G, [&](MTL::ComputeCommandEncoder*& e, MTL::Buffer* sho) {
                if (G == 1)
                    probe(e, "shexp_out", int(il), sho, E);
                ablate(e, il, ablate_shexp_, sho, 0, G);
            });
        } else {
            // The shared expert in chunks (its intermediates are wide), then the routed experts added in.
            for (uint32_t c0 = 0; c0 < G; c0 += chunk)
                ffn_dense(enc, L.moe.sh_gate, L.moe.sh_up, L.moe.sh_down, xn_g, ffo, L.moe.clamp_sh,
                          std::min(chunk, G - c0), uint64_t(c0) * E * 4, uint64_t(c0) * E * 4);
            ablate(enc, il, ablate_shexp_, ffo, 0, G);
            moe_prefill(cb, enc, L.moe, il, xn_g, ffo, G, committed);
        }
        part(enc, "ffn hc post");
        if (G == 1)
            probe(enc, "ffn_out", int(il), ffo, E);
        hc_post(enc, ffo, R, R2, pcs[cur], G);
        std::swap(R, R2);
        prev = pcs[cur];
        cur ^= 1;
    }
    // The drafter's window rows of these positions (the last ring-length of them).
    if (drafter_) {
        const uint32_t r0 = G > cfg_.sliding_window ? G - cfg_.sliding_window : 0;
        seed_drafter(enc, taps, uint64_t(r0) * 3 * E * 4, G - r0, pos_ + r0);
    }
    // Logits of the last position, or of every one (in chunks) for the sink.
    cur_layer_ = cfg_.n_layer;
    if (all_logits)
        logits_all_.resize(uint64_t(G) * cfg_.n_vocab);
    if (timing()) {
        enc->endEncoding();
        enc = encoder(cb, cur_layer_, -1);
    }
    const uint32_t nl = all_logits ? chunk : 1;
    MTL::Buffer* xl = scratch("ds_xl", uint64_t(nl) * E * 4);
    MTL::Buffer* xnl = scratch("ds_xnl", uint64_t(nl) * E * 4);
    MTL::Buffer* lg = scratch("ds_logits", uint64_t(nl) * cfg_.n_vocab * 4);
    for (uint32_t l0 = all_logits ? 0 : G - 1; l0 < G; l0 += nl) {
        const uint32_t n = std::min(nl, G - l0);
        const HcArgs h1{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, n};
        enc->setComputePipelineState(pso("hc_collapse"));
        enc->setBuffer(R, uint64_t(l0) * hd * 4, 0);
        enc->setBuffer(prev, uint64_t(l0) * 24 * 4, 1);
        enc->setBuffer(xl, 0, 2);
        enc->setBytes(&h1, sizeof(h1), 3);
        enc->dispatchThreads(MTL::Size(E, n, 1), MTL::Size(256, 1, 1));
        after(enc, "hc_collapse");
        norm(enc, xl, &output_norm_, xnl, n, E, E, cfg_.rms_eps);
        proj(enc, output_, xnl, 0, lg, 0, n);
        enc->endEncoding();
        finish(cb);
        drain();
        const auto* lgp = static_cast<const float*>(lg->contents());
        if (all_logits) {
            std::memcpy(logits_all_.data() + uint64_t(l0) * cfg_.n_vocab, lgp, uint64_t(n) * cfg_.n_vocab * 4);
            if (logits_sink_)
                for (uint32_t i = 0; i < n; ++i)
                    logits_sink_(pos_ + l0 + i, lgp + uint64_t(i) * cfg_.n_vocab);
        }
        if (l0 + n == G)
            std::memcpy(logits_.data(), lgp + uint64_t(n - 1) * cfg_.n_vocab, uint64_t(cfg_.n_vocab) * 4);
        cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        enc = encoder(cb, cur_layer_, -1);
    }
    enc->endEncoding();
    finish(cb);
    pos_ += G;
    pool->release();

    stats_.expert_read_ms = arena_->stats().read_ms;
    stats_.expert_bytes = arena_->stats().read_bytes;
    stats_.step_ms += ms_since(step_start_);
    const ExpertArena::Stats& after_stats = arena_->stats();
    last_step_.wall_ms = ms_since(step_start_);
    last_step_.prep_ms = prep_ms;
    last_step_.encode_ms = stats_.encode_ms - stats_before.encode_ms;
    last_step_.gpu_ms = stats_.gpu_exec_ms - stats_before.gpu_exec_ms;
    last_step_.read_ms = after_stats.read_ms - before.read_ms;
    last_step_.hits = after_stats.hits - before.hits;
    last_step_.misses = after_stats.misses - before.misses;
    last_step_.read_bytes = after_stats.read_bytes - before.read_bytes;
    if (timing())
        resolve_layer_times(cpu0, gpu0);
}

// dspark.json: [name, dtype, offset, bytes, shape (outermost first)] per tensor of dspark.bin.
struct DsparkIndex {
    std::vector<std::tuple<std::string, std::string, uint64_t, uint64_t, std::vector<uint64_t>>> tensors;
};

void Engine::load_drafter(const std::string& dir)
{
    namespace fs = std::filesystem;
    const fs::path index = fs::path(dir) / "dspark.json", bin = fs::path(dir) / "dspark.bin";
    std::string text;
    {
        std::ifstream in(index, std::ios::binary);
        if (!in)
            throw std::runtime_error(std::format("cannot open {}", index.string()));
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    DsparkIndex idx;
    if (auto ec = glz::read<glz::opts{.error_on_unknown_keys = false}>(idx, text))
        throw std::runtime_error(std::format("{}: {}", index.string(), glz::format_error(ec, text)));
    struct Entry {
        gguf::Type type;
        uint64_t off, bytes;
        std::vector<uint64_t> dims; // innermost first
    };
    std::map<std::string, Entry> t;
    for (const auto& [name, dtype, off, bytes, shape] : idx.tensors) {
        gguf::Type type;
        if (dtype == "F32")
            type = gguf::Type::F32;
        else if (dtype == "BF16")
            type = gguf::Type::BF16;
        else if (dtype == "Q8_0")
            type = gguf::Type::Q8_0;
        else if (dtype == "MXFP4")
            type = gguf::Type::MXFP4;
        else
            throw std::runtime_error(std::format("{}: {} has type {}", index.string(), name, dtype));
        t[name] = {type, off, bytes, {shape.rbegin(), shape.rend()}};
    }
    auto d = std::make_unique<Drafter>();
    d->fd = ::open(bin.c_str(), O_RDONLY);
    if (d->fd < 0)
        throw std::runtime_error(std::format("cannot open {}", bin.string()));
    auto entry = [&](const std::string& name) -> const Entry& {
        const auto it = t.find(name);
        if (it == t.end())
            throw std::runtime_error(std::format("{}: no tensor {}", index.string(), name));
        return it->second;
    };
    auto get = [&](const std::string& name) {
        const Entry& e = entry(name);
        return upload_raw(d->fd, e.off, e.bytes, e.type, e.dims, name);
    };
    for (uint32_t b = 0; b < Drafter::kBlocks; ++b) {
        Layer& L = d->blocks[b];
        const std::string p = std::format("mtp.{}.", b);
        L.attn_norm = get(p + "attn_norm.weight");
        L.ffn_norm = get(p + "ffn_norm.weight");
        L.hc_attn_fn = get(p + "hc_attn_fn");
        L.hc_attn_base = get(p + "hc_attn_base");
        L.hc_attn_scale = get(p + "hc_attn_scale");
        L.hc_ffn_fn = get(p + "hc_ffn_fn");
        L.hc_ffn_base = get(p + "hc_ffn_base");
        L.hc_ffn_scale = get(p + "hc_ffn_scale");
        L.q_a = get(p + "attn.wq_a.weight");
        L.q_a_norm = get(p + "attn.q_norm.weight");
        L.q_b = get(p + "attn.wq_b.weight");
        L.kv = get(p + "attn.wkv.weight");
        L.kv_norm = get(p + "attn.kv_norm.weight");
        L.out_a = get(p + "attn.wo_a.weight");
        L.out_a.dims = {L.out_a.dims[0], cfg_.o_lora_rank, cfg_.o_groups};
        L.out_b = get(p + "attn.wo_b.weight");
        L.sinks = get(p + "attn.attn_sink");
        L.moe.router = get(p + "ffn.gate.weight");
        L.moe.bias = get(p + "ffn.gate.bias");
        L.moe.sh_gate = get(p + "ffn.shared_experts.w1.weight");
        L.moe.sh_up = get(p + "ffn.shared_experts.w3.weight");
        L.moe.sh_down = get(p + "ffn.shared_experts.w2.weight");
        L.moe.clamp_exp = cfg_.swiglu_clamp_exp[0];
        L.moe.clamp_sh = cfg_.swiglu_clamp_shexp[0];
        if (L.moe.router.dims[1] != Drafter::kExperts)
            throw std::runtime_error(std::format("{}: {} experts, the drafter has {}", index.string(),
                                                 L.moe.router.dims[1], Drafter::kExperts));
        const char* parts[3] = {"w1", "w3", "w2"}; // gate, up, down
        for (int q = 0; q < 3; ++q) {
            const Entry& e0 = entry(std::format("{}ffn.experts.0.{}.weight", p, parts[q]));
            d->part_bytes[q] = e0.bytes;
            DevTensor ex = reserve(e0.bytes * Drafter::kExperts, e0.type, {e0.dims[0], e0.dims[1], Drafter::kExperts});
            for (uint32_t e = 0; e < Drafter::kExperts; ++e) {
                const Entry& ee = entry(std::format("{}ffn.experts.{}.{}.weight", p, e, parts[q]));
                if (ee.bytes != e0.bytes || ee.type != e0.type)
                    throw std::runtime_error(std::format("{}: experts of {} differ in size", index.string(), p));
                gguf::read_exact(d->fd, static_cast<char*>(ex.buf->contents()) + ex.off + uint64_t(e) * e0.bytes,
                                 e0.bytes, ee.off);
            }
            d->experts[b][q] = std::move(ex);
        }
        d->rings[b] = mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(uint64_t(cfg_.sliding_window) * cfg_.head_dim * 4,
                                                                     MTL::ResourceStorageModeShared));
        std::memset(d->rings[b]->contents(), 0, d->rings[b]->length());
    }
    d->main_proj = get("mtp.0.main_proj.weight");
    d->main_norm = get("mtp.0.main_norm.weight");
    d->norm = get("mtp.2.norm.weight");
    d->markov_embed = get("mtp.2.markov_head.embed.weight");
    d->markov_head = get("mtp.2.markov_head.head.weight");
    d->conf_w = get("mtp.2.confidence_head.proj.weight");
    if (d->main_proj.dims[0] != 3 * uint64_t(cfg_.n_embd) || d->markov_embed.dims[0] != Drafter::kRank ||
        d->markov_embed.dims[1] != cfg_.n_vocab)
        throw std::runtime_error(index.string() + ": not this model's drafter");
    drafter_ = std::move(d);
}

void Engine::seed_drafter(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* taps, uint64_t taps_off, uint32_t T,
                          uint32_t pos0)
{
    const uint32_t E = cfg_.n_embd, D = cfg_.head_dim, W = cfg_.sliding_window;
    MTL::Buffer* mx = scratch("dr_mx", uint64_t(T) * E * 4);
    MTL::Buffer* mxn = scratch("dr_mxn", uint64_t(T) * E * 4);
    MTL::Buffer* kv = scratch("dr_kv", uint64_t(T) * D * 4);
    MTL::Buffer* kvn = scratch("dr_kvn", uint64_t(T) * D * 4);
    proj(enc, drafter_->main_proj, taps, taps_off, mx, 0, T);
    norm(enc, mx, &drafter_->main_norm, mxn, T, E, E, cfg_.rms_eps);
    for (uint32_t b = 0; b < Drafter::kBlocks; ++b) {
        const Layer& L = drafter_->blocks[b];
        proj(enc, L.kv, mxn, 0, kv, 0, T);
        norm(enc, kv, &L.kv_norm, kvn, T, D, D, cfg_.rms_eps);
        rope(enc, kvn, 0, false, D, 1, pos0, T);
        round(enc, kvn, 0, kvn, 0, T * D, kFp8);
        const uint32_t ra[4] = {D, T, pos0, W};
        dispatch(enc, "ds_ring_store", {kvn, drafter_->rings[b].get()}, ra, sizeof(ra), MTL::Size(D, std::min(T, W), 1),
                 MTL::Size(256, 1, 1));
    }
}

void Engine::draft_moe(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t b, MTL::Buffer* xn,
                       MTL::Buffer* ffo, uint32_t T)
{
    const uint32_t E = cfg_.n_embd, nff = cfg_.n_ff_exp, NE = Drafter::kExperts, k = Drafter::kUsed;
    MTL::Buffer* rlog = scratch("dr_rlog", uint64_t(T) * NE * 4);
    MTL::Buffer* sel = scratch("dr_sel", uint64_t(T) * 16 * 4);
    MTL::Buffer* ew = scratch("dr_ew", uint64_t(T) * 16 * 4);
    MTL::Buffer* biased = scratch("dr_biased", uint64_t(T) * NE * 4);
    MTL::Buffer* unused = scratch("dr_unused", 64);
    MTL::Buffer* sho = scratch("dr_sho", uint64_t(T) * E * 4);
    proj(enc, L.moe.router, xn, 0, rlog, 0, T);
    const RouteArgs ra{NE, k, routing_.norm ? 1u : 0u, routing_.scale, 0, uint32_t(routing_.score), 0.0f};
    enc->setComputePipelineState(pso("moe_route"));
    enc->setBuffer(rlog, 0, 0);
    enc->setBuffer(L.moe.bias.buf, L.moe.bias.off, 1);
    enc->setBuffer(sel, 0, 2);
    enc->setBuffer(ew, 0, 3);
    enc->setBuffer(biased, 0, 4);
    enc->setBytes(&ra, sizeof(ra), 5);
    for (int i = 6; i < 10; ++i)
        enc->setBuffer(unused, 0, i);
    enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(32, 1, 1));
    after(enc, "moe_route");
    ffn_dense(enc, L.moe.sh_gate, L.moe.sh_up, L.moe.sh_down, xn, sho, L.moe.clamp_sh, T);
    MTL::Buffer* g = scratch("dr_g", uint64_t(k) * nff * 4);
    MTL::Buffer* u = scratch("dr_u", uint64_t(k) * nff * 4);
    MTL::Buffer* m = scratch("dr_m", uint64_t(k) * nff * 4);
    MTL::Buffer* y = scratch("dr_y", uint64_t(k) * E * 4);
    const auto& ex = drafter_->experts[b];
    const SwigluArgs sa{k * nff, L.moe.clamp_exp > 1e-6f ? L.moe.clamp_exp : 0.0f};
    const CombineArgs ca{E, k};
    for (uint32_t t = 0; t < T; ++t) {
        for (int q = 0; q < 3; ++q) {
            mtl::Gemv::Args a;
            a.type = ex[q].type;
            a.k = uint32_t(ex[q].dims[0]);
            a.n = uint32_t(ex[q].dims[1]);
            a.t = 1;
            a.batch = k;
            a.x_shared = q < 2;
            MTL::Buffer* in = q < 2 ? xn : m;
            const uint64_t in_off = q < 2 ? uint64_t(t) * E * 4 : 0;
            MTL::Buffer* out = q == 0 ? g : q == 1 ? u : y;
            gemv_->encode_indexed(enc, a, ex[q].buf, ex[q].off, drafter_->part_bytes[q], sel, uint64_t(t) * 64, in,
                                  in_off, out, 0);
            after(enc, "draft expert");
            if (q == 1)
                dispatch(enc, "swiglu", {g, u, m}, &sa, sizeof(sa), MTL::Size(k * nff, 1, 1), MTL::Size(256, 1, 1));
        }
        enc->setComputePipelineState(pso("moe_combine"));
        enc->setBuffer(y, 0, 0);
        enc->setBuffer(ew, uint64_t(t) * 64, 1);
        enc->setBuffer(sho, uint64_t(t) * E * 4, 2);
        enc->setBuffer(ffo, uint64_t(t) * E * 4, 3);
        enc->setBytes(&ca, sizeof(ca), 4);
        enc->dispatchThreads(MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "moe_combine");
    }
}

std::vector<int32_t> Engine::draft(int32_t next)
{
    draft_conf_.clear();
    if (!drafter_ || next < 0 || uint32_t(next) >= cfg_.n_vocab || pos_ == 0)
        return {};
    const auto t_draft = Clock::now();
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E, V = cfg_.n_vocab, T = Drafter::kRows, P = pos_;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Buffer* x = scratch("dr_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("dr_xn", uint64_t(T) * E * 4);
    MTL::Buffer* att = scratch("dr_att", uint64_t(T) * E * 4);
    MTL::Buffer* ffo = scratch("dr_ffo", uint64_t(T) * E * 4);
    MTL::Buffer* R = scratch("dr_R", uint64_t(T) * hd * 4);
    MTL::Buffer* R2 = scratch("dr_R2", uint64_t(T) * hd * 4);
    MTL::Buffer* pcs[2] = {scratch("dr_pcA", uint64_t(T) * 24 * 4), scratch("dr_pcB", uint64_t(T) * 24 * 4)};
    MTL::Buffer* pc0 = scratch("dr_pc0", uint64_t(T) * 24 * 4);
    MTL::Buffer* lg = scratch("dr_logits", uint64_t(T) * V * 4);
    MTL::Buffer* scores = scratch("dr_scores", uint64_t(V) * 4);
    MTL::Buffer* drafts = scratch("dr_drafts", 64);
    MTL::Buffer* conf = scratch("dr_conf", 64);
    {
        auto* p0 = static_cast<float*>(pc0->contents());
        std::fill_n(p0, uint64_t(T) * 24, 0.0f);
        for (uint32_t t = 0; t < T; ++t)
            p0[t * 24] = 1.0f;
        const gguf::Tensor& te = *token_embd_.tensor;
        const uint64_t rb = gguf::traits(te.type).block_bytes * (te.dims[0] / gguf::traits(te.type).block_elems);
        std::vector<std::byte> row(rb);
        for (uint32_t i = 0; i < T; ++i) {
            const int32_t id = i == 0 ? next : Drafter::kNoise;
            gguf::read_exact(token_embd_.file->fd(), row.data(), rb, te.offset + uint64_t(id) * rb);
            quant::dequant_row(te.type, row.data(), static_cast<float*>(x->contents()) + uint64_t(i) * E, E);
        }
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    cur_layer_ = cfg_.n_layer;
    MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, T};
    dispatch(enc, "hc_init", {x, R}, &ha, sizeof(ha), MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    MTL::Buffer* prev = pc0;
    int cur = 0;
    for (uint32_t b = 0; b < Drafter::kBlocks; ++b) {
        const Layer& L = drafter_->blocks[b];
        hc_lagged(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R, x, pcs[cur], prev, T, 0, 0);
        norm(enc, x, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
        attention(enc, L, 0, xn, att, T, P, 0, drafter_->rings[b].get(), true);
        hc_post(enc, att, R, R2, pcs[cur], T);
        std::swap(R, R2);
        prev = pcs[cur];
        cur ^= 1;
        hc_lagged(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, R, x, pcs[cur], prev, T, 0, 0);
        norm(enc, x, &L.ffn_norm, xn, T, E, E, cfg_.rms_eps);
        draft_moe(enc, L, b, xn, ffo, T);
        hc_post(enc, ffo, R, R2, pcs[cur], T);
        std::swap(R, R2);
        prev = pcs[cur];
        cur ^= 1;
    }
    // The drafts: the trunk's head on the collapsed rows, each plus the Markov bias of the draft before it.
    dispatch(enc, "hc_collapse", {R, prev, x}, &ha, sizeof(ha), MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    norm(enc, x, &drafter_->norm, xn, T, E, E, cfg_.rms_eps);
    proj(enc, output_, xn, 0, lg, 0, T);
    const uint32_t dim = E;
    for (uint32_t i = 0; i < T; ++i) {
        const MarkovArgs ma{V, Drafter::kRank, i, next};
        enc->setComputePipelineState(pso("ds_markov"));
        enc->setBuffer(lg, uint64_t(i) * V * 4, 0);
        enc->setBuffer(drafter_->markov_head.buf, drafter_->markov_head.off, 1);
        enc->setBuffer(drafter_->markov_embed.buf, drafter_->markov_embed.off, 2);
        enc->setBuffer(drafts, 0, 3);
        enc->setBuffer(scores, 0, 4);
        enc->setBytes(&ma, sizeof(ma), 5);
        enc->dispatchThreads(MTL::Size(uint64_t(V + 7) / 8 * 32, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "ds_markov");
        enc->setComputePipelineState(pso("ds_draft_pick"));
        enc->setBuffer(scores, 0, 0);
        enc->setBuffer(drafts, 0, 1);
        enc->setBuffer(conf, 0, 2);
        enc->setBuffer(x, 0, 3);
        enc->setBuffer(drafter_->markov_embed.buf, drafter_->markov_embed.off, 4);
        enc->setBuffer(drafter_->conf_w.buf, drafter_->conf_w.off, 5);
        enc->setBytes(&ma, sizeof(ma), 6);
        enc->setBytes(&dim, sizeof(dim), 7);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
        after(enc, "ds_draft_pick");
    }
    enc->endEncoding();
    finish(cb);
    const auto* dp = static_cast<const int32_t*>(drafts->contents());
    const auto* cp = static_cast<const float*>(conf->contents());
    std::vector<int32_t> out;
    float cum = 1.0f;
    for (uint32_t i = 0; i < T; ++i) {
        draft_conf_.push_back(cp[i]);
        cum *= cp[i];
        if (out.size() == i && i < opts_.max_drafts && cp[i] >= opts_.draft_conf && cum >= opts_.draft_cum)
            out.push_back(dp[i]);
    }
    pool->release();
    ++spec_stats_.drafts;
    spec_stats_.draft_ms += ms_since(t_draft);
    return out;
}

const std::vector<float>& Engine::verify(std::span<const int32_t> tokens)
{
    if (tokens.empty() || tokens.size() > kHistTokens)
        throw std::runtime_error("verify: 1 to 8 tokens");
    verify_pos0_ = pos_;
    verify_tokens_.assign(tokens.begin(), tokens.end());
    verify_history_ = engram_history_;
    verifying_ = true;
    struct Reset {
        bool& v;
        ~Reset() { v = false; }
    } reset{verifying_};
    const auto t0 = Clock::now();
    run(tokens, true);
    ++spec_stats_.verifies;
    spec_stats_.verified_tokens += tokens.size();
    spec_stats_.verify_ms += ms_since(t0);
    spec_stats_.verify_misses += last_step_.misses;
    spec_stats_.verify_read_ms += last_step_.read_ms + last_step_.ensure_ms;
    return logits_all_;
}

void Engine::keep(uint32_t n)
{
    const uint32_t T = uint32_t(verify_tokens_.size());
    if (n == 0 || n > T || pos_ != verify_pos0_ + T)
        throw std::runtime_error("keep: not after a verify of at least that many tokens");
    if (n == T)
        return;
    pos_ = verify_pos0_ + n;
    engram_history_ = engram_->advance(verify_history_, std::span<const int32_t>(verify_tokens_).first(n));
    // A ratio-2 compressor whose pair is open (the next position odd) holds the first token's (k, g): the state the
    // verify left after token n - 1.
    if (pos_ % 2 == 1)
        for (size_t s = 0; s < kv_sources_.size(); ++s)
            if (cfg_.compress_ratios[kv_sources_[s]] == 2)
                std::memcpy(comp_state_[s]->contents(),
                            static_cast<const char*>(comp_hist_[s]->contents()) + uint64_t(n - 1) * 2 * cfg_.head_dim * 4,
                            uint64_t(2) * cfg_.head_dim * 4);
    verify_tokens_.resize(n);
}

} // namespace mako::dsv41
