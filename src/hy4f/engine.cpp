#include "hy4f/engine.hpp"

#include "gguf/read_pool.hpp"
#include "quant/dequant.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>
#include <tuple>

namespace mako::hy4f {

using core::ExpertArena;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct IhcArgs {
    uint32_t n_embd;
    float eps, magnitude;
    uint32_t tokens;
};
struct RopeArgs {
    uint32_t head_dim, heads, n_rot, stride, pos0, pos_step, inverse;
};
struct EngramRowsArgs {
    uint32_t dim, cols, token_stride;
    float eps;
};
struct EngramArgs {
    uint32_t n_embd;
    float eps;
};
struct SplitArgs {
    uint32_t heads, nope, rope;
};
struct AttnSoftmaxArgs {
    uint32_t n_kv;
    float scale;
    uint32_t vis0, heads, window;
};
struct GatherRopeArgs {
    uint32_t heads, dim, stride;
};
struct ListSoftmaxArgs {
    uint32_t stride;
    float scale;
    uint32_t heads;
};
struct PvArgs {
    uint32_t heads, dim, n_kv, keys_per_split;
};
struct GatherArgs {
    uint32_t heads, dim, stride, splits, tokens;
};
struct PoolArgs {
    uint32_t dim, kpool, pool, row0;
};
struct ScoreArgs {
    uint32_t dim, heads, first, kpool;
    float w_scale;
    uint32_t stride;
};
struct SelectArgs {
    uint32_t first, n_top, kpool, score_stride, list_stride, full_tail;
};

constexpr uint32_t kCommitLayers = 4; // decode: layers per command buffer
// A window layer's keys for a token start at a multiple of this many positions (the rest of them masked): a score's
// rounding depends on its key's place in the GEMM's tiles, so a fixed grid gives every token the same results
// whatever chunk or step it runs in.
constexpr uint32_t kKeyAlign = 64;

// The first key position window layers attend for a token at pos (window W).
uint32_t window_start(uint32_t pos, uint32_t W) { return pos + 1 > W ? (pos + 1 - W) / kKeyAlign * kKeyAlign : 0; }
// The Engram norms leave eps unset in the reference: the fp32 machine epsilon, not rms_norm_eps.
constexpr float kEngramEps = 1.1920929e-07f;

std::vector<float> rope_freqs(uint32_t n_rot, float base)
{
    std::vector<float> f(n_rot / 2);
    for (uint32_t i = 0; i < n_rot / 2; ++i)
        f[i] = 1.0f / std::pow(base, float(2 * i) / float(n_rot));
    return f;
}

} // namespace

struct Engine::Layer {
    bool swa = false, indexer = false, dense = false;
    int engram = -1; // Engram module
    DevTensor attn_norm, ffn_norm, post_attn_norm, post_ffn_norm;
    DevTensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    DevTensor q_a, q_a_norm, q_b, kv_lat, kv_pe, kv_norm, k_b, v_b, wo, gate, sinks;
    DevTensor idx_k, idx_k_norm, idx_q_b, idx_proj, idx_ape, idx_gate, idx_value, idx_norm;
    DevTensor eg_kv, eg_kv_b, eg_gate, eg_lookup_norm, eg_gamma, eg_hc_scale;
    DevTensor ffn_gate, ffn_up, ffn_down;
    MoeWeights moe;
    ExpertArena::LayerParts experts{};
};

Engine::Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx)
    : EngineBase(model, ctx, opts.telemetry), cfg_(cfg), opts_(opts)
{
    set_page_cache(opts_.page_cache);
    dense_quant_ = opts_.dense_quant;
    const auto t_load = Clock::now();
    // What the kernels are written for.
    if (cfg_.hc != 4 || cfg_.n_rot != 64 || cfg_.head_k != 256 || cfg_.kv_lora_rank != 512 || cfg_.indexer_dim != 128 ||
        cfg_.indexer_heads > 32 || cfg_.n_expert > 512 || cfg_.n_expert % 2 || cfg_.n_expert_used > 16 ||
        cfg_.engram.head_dim > 128 || !cfg_.indexer_select_tail)
        throw std::runtime_error("unsupported Hy4-flash geometry");
    for (uint32_t h : cfg_.n_head)
        if (h > 64)
            throw std::runtime_error("more than 64 attention heads");
    if (cfg_.gating != 2)
        throw std::runtime_error(std::format("expert gating function {} (only sigmoid, 2)", cfg_.gating));
    dims_ = {cfg_.n_embd, cfg_.n_layer, cfg_.hc, 0, cfg_.hc_eps, cfg_.rms_eps,
             cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp};
    routing_ = {core::Routing::Score::sigmoid, cfg_.expert_weights_norm, cfg_.expert_weights_scale, opts_.miss_penalty};
    gemm_min_tokens_ = kMaxDecodeTokens + 1;
    n_top_ = cfg_.indexer_top_k / cfg_.indexer_kpool - 1;
    opts_.prefill_chunk = std::max(1u, opts_.prefill_chunk);
    opts_.prefill_group = std::max(opts_.prefill_chunk, opts_.prefill_group);

    token_embd_ = model_.find("token_embd.weight");
    if (!token_embd_ || !quant::can_dequant(token_embd_.tensor->type))
        throw std::runtime_error("token_embd.weight missing or of an unsupported type");
    output_norm_ = upload("output_norm.weight");
    output_ = upload("output.weight");
    head_fn_ = upload("output_hc_fn.weight");
    head_base_ = upload("output_hc_base.weight");
    head_scale_ = upload("output_hc_scale.weight");

    auto freqs = [&](float base) {
        const auto f = rope_freqs(cfg_.n_rot, base);
        return mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(f.data(), f.size() * 4, MTL::ResourceStorageModeShared));
    };
    freq_full_ = freqs(cfg_.rope_base);
    freq_swa_ = freqs(cfg_.rope_base_swa);

    auto buffer = [&](uint64_t bytes) {
        mtl::Ref<MTL::Buffer> b(ctx_.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
        std::memset(b->contents(), 0, b->length());
        return b;
    };
    const uint32_t R = cfg_.kv_lora_rank, Di = cfg_.indexer_dim;
    // Sliding buffers: the rows a chunk's first token needs, the chunk's, and as many again so moves are rare.
    built_chunk_ = opts_.prefill_chunk;
    const uint32_t most = std::max(opts_.prefill_chunk, kMaxDecodeTokens);
    swa_cap_ = std::min(2 * cfg_.sliding_window + most + kKeyAlign, opts_.max_ctx);
    idx_cap_ = most + 2 * cfg_.indexer_kpool;
    uint64_t context_bytes = 0;
    const auto& eg_layers = cfg_.engram.layers;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        auto L = std::make_unique<Layer>();
        const std::string p = std::format("blk.{}.", il);
        L->swa = cfg_.is_swa[il];
        L->indexer = cfg_.has_indexer[il];
        L->dense = il < cfg_.n_dense_lead;
        L->attn_norm = upload(p + "attn_norm.weight");
        L->ffn_norm = upload(p + "ffn_norm.weight");
        L->post_attn_norm = upload(p + "post_attention_norm.weight");
        L->post_ffn_norm = upload(p + "post_ffw_norm.weight");
        L->hc_attn_fn = upload(p + "hc_attn_fn.weight");
        L->hc_attn_base = upload(p + "hc_attn_base.weight");
        L->hc_attn_scale = upload(p + "hc_attn_scale.weight");
        L->hc_ffn_fn = upload(p + "hc_ffn_fn.weight");
        L->hc_ffn_base = upload(p + "hc_ffn_base.weight");
        L->hc_ffn_scale = upload(p + "hc_ffn_scale.weight");
        L->q_a = upload(p + "attn_q_a.weight");
        L->q_a_norm = upload(p + "attn_q_a_norm.weight");
        L->q_b = upload(p + "attn_q_b.weight");
        // kv_a_mqa's rows: the latent (kv_lora_rank) then the rope part (n_rot), as two matrices.
        const DevTensor kv = upload(p + "attn_kv_a_mqa.weight");
        if (kv.dims.size() != 2 || kv.dims[1] != R + cfg_.n_rot)
            throw std::runtime_error(p + "attn_kv_a_mqa.weight is not [n_embd, kv_lora_rank + n_rot]");
        const uint64_t kv_row = mtl::Gemv::row_bytes(kv.type, uint32_t(kv.dims[0]));
        L->kv_lat = {kv.buf, kv.off, kv.type, {kv.dims[0], R}};
        L->kv_pe = {kv.buf, kv.off + R * kv_row, kv.type, {kv.dims[0], cfg_.n_rot}};
        L->kv_norm = upload(p + "attn_kv_a_norm.weight");
        L->k_b = upload(p + "attn_k_b.weight");
        L->v_b = upload(p + "attn_v_b.weight");
        L->wo = upload(p + "attn_output.weight");
        L->gate = upload(p + "attn_gate.weight");
        L->sinks = upload(p + "attn_sinks.weight");
        if (L->indexer) {
            L->idx_k = upload(p + "indexer.attn_k.weight");
            L->idx_k_norm = upload(p + "indexer.k_norm.weight");
            L->idx_q_b = upload(p + "indexer.attn_q_b.weight");
            L->idx_proj = upload(p + "indexer.proj.weight");
            L->idx_ape = upload(p + "indexer_compressor_ape.weight");
            L->idx_gate = upload(p + "indexer_compressor_gate.weight");
            L->idx_value = upload(p + "indexer_compressor_value.weight");
            L->idx_norm = upload(p + "indexer_compressor_norm.weight");
        }
        if (auto it = std::ranges::find(eg_layers, int32_t(il)); it != eg_layers.end()) {
            L->engram = int(it - eg_layers.begin());
            L->eg_kv = upload(p + "engram_kv.weight");
            L->eg_kv_b = upload(p + "engram_kv_b.weight");
            L->eg_gate = upload(p + "engram_gate.weight");
            L->eg_lookup_norm = upload(p + "engram_lookup_norm.weight");
            L->eg_gamma = upload(p + "engram_pre_gamma.weight");
            L->eg_hc_scale = upload(p + "engram_hc_scale.weight");
            for (const DevTensor* t : {&L->eg_kv_b, &L->eg_gate, &L->eg_lookup_norm, &L->eg_gamma, &L->eg_hc_scale})
                if (t->type != gguf::Type::F32)
                    throw std::runtime_error(p + "Engram norms and gates must be F32");
        }
        if (L->dense) {
            L->ffn_gate = upload(p + "ffn_gate.weight");
            L->ffn_up = upload(p + "ffn_up.weight");
            L->ffn_down = upload(p + "ffn_down.weight");
        } else {
            L->moe.router = upload(p + "ffn_gate_inp.weight");
            L->moe.bias = upload(p + "exp_probs_b.bias");
            if (L->moe.bias.type != gguf::Type::F32 || L->moe.bias.dims[0] != cfg_.n_expert)
                throw std::runtime_error(p + "exp_probs_b.bias is not F32 [n_expert]");
            L->moe.sh_gate = upload(p + "ffn_gate_shexp.weight");
            L->moe.sh_up = upload(p + "ffn_up_shexp.weight");
            L->moe.sh_down = upload(p + "ffn_down_shexp.weight");
            if (cfg_.n_latent != cfg_.n_embd) {
                L->moe.lat_down = upload(p + "ffn_routed_down.weight");
                L->moe.lat_up = upload(p + "ffn_routed_up.weight");
            }
            L->moe.clamp_exp = cfg_.swiglu_clamp_exp[il];
            L->moe.clamp_sh = 0.0f; // the shared and dense FFNs are not clamped
            L->experts = {expert(p + "ffn_gate_exps.weight", true), expert(p + "ffn_up_exps.weight", true),
                          expert(p + "ffn_down_exps.weight", true)};
        }
        const uint64_t rows = L->swa ? swa_cap_ : opts_.max_ctx;
        lat_.push_back(buffer(rows * R * 2));
        pe_.push_back(buffer(rows * cfg_.n_rot * 2));
        context_bytes += lat_.back()->length() + pe_.back()->length();
        if (L->indexer) {
            idx_rows_.push_back(buffer(uint64_t(idx_cap_) * 2 * Di * 4));
            idx_pooled_.push_back(buffer(uint64_t(opts_.max_ctx / cfg_.indexer_kpool + 1) * Di * 4));
            context_bytes += idx_rows_.back()->length() + idx_pooled_.back()->length();
        } else {
            idx_rows_.emplace_back();
            idx_pooled_.emplace_back();
        }
        layers_.push_back(std::move(L));
        layer_mla_.push_back(1);
    }
    row_base_.assign(cfg_.n_layer, 0);
    idx_base_.assign(cfg_.n_layer, 0);

    if (!eg_layers.empty()) {
        auto eg = dsv41::Engram::open(cfg_.engram, model_);
        if (!eg)
            throw std::runtime_error(eg.error());
        engram_.emplace(std::move(*eg));
        engram_pool_ = std::make_unique<gguf::ReadPool>(16);
        engram_history_.assign(engram_->history(), -1);
    }
    logits_.resize(cfg_.n_vocab);

    load_.dense_ms = ms_since(t_load);
    set_staging_limit(opts_.stage_gb);
    set_spin_wait(opts_.spin_wait);
    ArenaInit init;
    for (const auto& L : layers_)
        init.parts.push_back(L->experts);
    for (uint32_t il = cfg_.n_dense_lead; il < cfg_.n_layer; ++il)
        init.moe_layers.push_back(il);
    init.context_bytes = context_bytes + uint64_t(opts_.checkpoints) * checkpoint_bytes();
    init.lendable = double(prefill_scratch(opts_.prefill_chunk, opts_.prefill_group));
    init.arena_gb = opts_.arena_gb;
    init.park_gb = opts_.park_gb;
    init.simulate_ram_gb = opts_.simulate_ram_gb;
    init.expert_counts = opts_.expert_counts;
    init_runtime(std::move(init));
    set_keepalive(opts_.keepalive);
}

Engine::~Engine() = default;

// Scratch bytes of a prefill: per group token (streams, FFN rows, Engram rows, routed outputs), per chunk token (the
// attention's projections and outputs), the attention and indexer scores, the expert batch buffers.
uint64_t Engine::prefill_scratch(uint32_t chunk, uint32_t group) const
{
    const uint64_t E = cfg_.n_embd, hd = uint64_t(cfg_.hc) * E, k = cfg_.n_expert_used, nff = cfg_.n_ff_exp;
    const uint64_t Ex = cfg_.n_latent, H = *std::ranges::max_element(cfg_.n_head), R = cfg_.kv_lora_rank;
    const uint64_t eg = uint64_t(cfg_.engram.layers.size()) * cfg_.engram.columns() * cfg_.engram.head_dim;
    const uint64_t per_group = 2 * hd + 8 + 6 * E + eg + 5 * E + 3 * Ex + k * Ex + 2 * cfg_.n_expert + 64;
    const uint64_t per_chunk = 3 * E + 2 * cfg_.q_lora_rank + H * (cfg_.head_k + 2 * R + 2 * cfg_.head_v + 64) +
                               R + cfg_.n_rot + uint64_t(cfg_.indexer_heads) * (cfg_.indexer_dim + 1) +
                               3 * cfg_.indexer_dim + 3 * uint64_t(cfg_.n_ff) + H * (R + 64) / 2 + 5 * E;
    const uint64_t scores = 3 * (4ull << 20) + 64 * H * 3 * list_stride() + 16 * 64 * H * R +
                            32ull * (opts_.max_ctx / cfg_.indexer_kpool + 1);
    const uint64_t batch = std::min<uint64_t>(kBatchRows, uint64_t(group) * k) * (Ex / 2 + 2 * nff + nff / 2);
    const uint64_t staging = group >= kStageMinTokens ? staging_bytes() : 0;
    return (uint64_t(group) * per_group + uint64_t(chunk) * per_chunk + scores + batch) * 4 + staging + (96ull << 20);
}

void Engine::reset()
{
    pos_ = 0;
    if (engram_)
        engram_history_.assign(engram_->history(), -1);
    row_base_.assign(cfg_.n_layer, 0);
    idx_base_.assign(cfg_.n_layer, 0);
}

void Engine::set_prefill_chunk(uint32_t chunk)
{
    if (chunk == 0 || chunk > built_chunk_)
        throw std::runtime_error(std::format("prefill chunk {} outside 1..{} (the engine's)", chunk, built_chunk_));
    opts_.prefill_chunk = chunk;
}

void Engine::move_rows(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* b, uint64_t row_bytes, uint32_t from, uint32_t n)
{
    if (!n)
        return;
    const uint32_t words = uint32_t(uint64_t(n) * row_bytes / 4);
    enc->setComputePipelineState(pso("hy4_copy_words"));
    enc->setBuffer(b, uint64_t(from) * row_bytes, 0);
    enc->setBuffer(b, 0, 1);
    enc->setBytes(&words, sizeof(words), 2);
    enc->dispatchThreads(MTL::Size(words, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "hy4_copy_words");
}

void Engine::slide(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t pos0, uint32_t T)
{
    const Layer& L = *layers_[il];
    if (L.swa && pos0 + T - row_base_[il] > swa_cap_) {
        // From the chunk's first token's first key on.
        const uint32_t nb = window_start(pos0, cfg_.sliding_window);
        move_rows(enc, lat_[il].get(), uint64_t(cfg_.kv_lora_rank) * 2, nb - row_base_[il], pos0 - nb);
        move_rows(enc, pe_[il].get(), uint64_t(cfg_.n_rot) * 2, nb - row_base_[il], pos0 - nb);
        row_base_[il] = nb;
    }
    if (L.indexer && pos0 + T - idx_base_[il] > idx_cap_) {
        // The rows of pos0's group before it, for that group's pool.
        const uint32_t nb = pos0 / cfg_.indexer_kpool * cfg_.indexer_kpool;
        move_rows(enc, idx_rows_[il].get(), uint64_t(cfg_.indexer_dim) * 8, nb - idx_base_[il], pos0 - nb);
        idx_base_[il] = nb;
    }
    if (pos0 + T - row_base_[il] > (L.swa ? swa_cap_ : opts_.max_ctx) || (L.indexer && pos0 + T - idx_base_[il] > idx_cap_))
        throw std::logic_error("more rows than a sliding buffer holds");
}

std::vector<std::pair<MTL::Buffer*, uint64_t>> Engine::positional(uint32_t pos) const
{
    if (pos > opts_.max_ctx)
        throw std::runtime_error("position past the context size");
    std::vector<std::pair<MTL::Buffer*, uint64_t>> out;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        if (layers_[il]->swa)
            continue;
        out.emplace_back(lat_[il].get(), uint64_t(pos) * cfg_.kv_lora_rank * 2);
        out.emplace_back(pe_[il].get(), uint64_t(pos) * cfg_.n_rot * 2);
        if (idx_pooled_[il])
            out.emplace_back(idx_pooled_[il].get(), uint64_t(pos / cfg_.indexer_kpool) * cfg_.indexer_dim * 4);
    }
    return out;
}

std::vector<std::byte> Engine::save_positions(uint32_t pos) const
{
    const auto parts = positional(pos);
    uint64_t n = 0;
    for (const auto& [b, bytes] : parts)
        n += bytes;
    std::vector<std::byte> out(n);
    std::byte* p = out.data();
    for (const auto& [b, bytes] : parts) {
        std::memcpy(p, b->contents(), bytes);
        p += bytes;
    }
    return out;
}

void Engine::load_positions(uint32_t pos, std::span<const std::byte> data)
{
    const auto parts = positional(pos);
    uint64_t off = 0;
    for (const auto& [b, bytes] : parts) {
        if (off + bytes > data.size())
            throw std::runtime_error("positions from another engine");
        std::memcpy(b->contents(), data.data() + off, bytes);
        off += bytes;
    }
    if (off != data.size())
        throw std::runtime_error("positions from another engine");
}

// The state's layout: the Engram history; per window layer sliding_window + kKeyAlign latent then rope rows (the
// positions from the next token's first key to pos, from the start); per indexer layer kpool value | gate rows (the
// positions of pos's group before it, from the start).
uint64_t Engine::checkpoint_bytes() const
{
    uint64_t n = engram_history_.size() * 4;
    for (const auto& L : layers_) {
        if (L->swa)
            n += uint64_t(cfg_.sliding_window + kKeyAlign) * (cfg_.kv_lora_rank + cfg_.n_rot) * 2;
        if (L->indexer)
            n += uint64_t(cfg_.indexer_kpool) * cfg_.indexer_dim * 8;
    }
    return n;
}

Engine::Checkpoint Engine::checkpoint()
{
    Checkpoint cp;
    cp.pos = pos_;
    cp.state.resize(checkpoint_bytes());
    std::byte* p = cp.state.data();
    std::memcpy(p, engram_history_.data(), engram_history_.size() * 4);
    p += engram_history_.size() * 4;
    const uint32_t W = cfg_.sliding_window, kp = cfg_.indexer_kpool;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        if (L.swa) {
            const uint32_t s = window_start(pos_, W), n = pos_ - s;
            for (auto [buf, rb] : {std::pair{lat_[il].get(), uint64_t(cfg_.kv_lora_rank) * 2},
                                   std::pair{pe_[il].get(), uint64_t(cfg_.n_rot) * 2}}) {
                std::memcpy(p, static_cast<const std::byte*>(buf->contents()) + (s - row_base_[il]) * rb, n * rb);
                p += (W + kKeyAlign) * rb;
            }
        }
        if (L.indexer) {
            const uint64_t rb = uint64_t(cfg_.indexer_dim) * 8;
            const uint32_t g0 = pos_ / kp * kp;
            std::memcpy(p, static_cast<const std::byte*>(idx_rows_[il]->contents()) + (g0 - idx_base_[il]) * rb,
                        (pos_ - g0) * rb);
            p += kp * rb;
        }
    }
    return cp;
}

void Engine::restore(const Checkpoint& cp)
{
    if (cp.pos > opts_.max_ctx || cp.state.size() != checkpoint_bytes())
        throw std::runtime_error("checkpoint from another engine");
    const std::byte* p = cp.state.data();
    std::memcpy(engram_history_.data(), p, engram_history_.size() * 4);
    p += engram_history_.size() * 4;
    const uint32_t W = cfg_.sliding_window, kp = cfg_.indexer_kpool;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        if (L.swa) {
            const uint32_t s = window_start(cp.pos, W);
            row_base_[il] = s;
            for (auto [buf, rb] : {std::pair{lat_[il].get(), uint64_t(cfg_.kv_lora_rank) * 2},
                                   std::pair{pe_[il].get(), uint64_t(cfg_.n_rot) * 2}}) {
                std::memcpy(buf->contents(), p, (cp.pos - s) * rb);
                p += (W + kKeyAlign) * rb;
            }
        }
        if (L.indexer) {
            const uint64_t rb = uint64_t(cfg_.indexer_dim) * 8;
            const uint32_t g0 = cp.pos / kp * kp;
            idx_base_[il] = g0;
            std::memcpy(idx_rows_[il]->contents(), p, (cp.pos - g0) * rb);
            p += kp * rb;
        }
    }
    pos_ = cp.pos;
}

void Engine::rope(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* freqs,
                  uint32_t head_dim, uint32_t heads, uint32_t pos0, uint32_t T, uint32_t step)
{
    const RopeArgs a{head_dim, heads, cfg_.n_rot, heads * head_dim, pos0, step, 0};
    enc->setComputePipelineState(pso("ds_rope"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(freqs, 0, 1);
    enc->setBytes(&a, sizeof(a), 2);
    for (int r = repeats("ds_rope"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(cfg_.n_rot / 2, heads, T), MTL::Size(32, 1, 1));
    after(enc, "ds_rope");
}

void Engine::hc_pre(MTL::ComputeCommandEncoder*& enc, const DevTensor& fn, const DevTensor& scale,
                    const DevTensor& base, MTL::Buffer* R, uint64_t r_off, MTL::Buffer* x, MTL::Buffer* pc, uint32_t T,
                    const DevTensor& norm_w, MTL::Buffer* xn)
{
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    MTL::Buffer* flat = scratch("hy_flat", uint64_t(T) * hd * 4);
    MTL::Buffer* mixes = scratch("hy_mixes", uint64_t(T) * 8 * 4);
    norm(enc, R, nullptr, flat, T, hd, 0, cfg_.rms_eps, r_off);
    proj(enc, fn, flat, 0, mixes, 0, T);
    const IhcArgs a{E, cfg_.hc_eps, cfg_.hc_magnitude, T};
    enc->setComputePipelineState(pso("ihc_pre"));
    enc->setBuffer(mixes, 0, 0);
    enc->setBuffer(scale.buf, scale.off, 1);
    enc->setBuffer(base.buf, base.off, 2);
    enc->setBuffer(R, r_off, 3);
    enc->setBuffer(x, 0, 4);
    enc->setBuffer(pc, 0, 5);
    enc->setBytes(&a, sizeof(a), 6);
    for (int r = repeats("ihc_pre"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    after(enc, "ihc_pre");
    norm(enc, x, &norm_w, xn, T, E, E, cfg_.rms_eps);
}

void Engine::hc_post(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* out, const DevTensor& post_norm, MTL::Buffer* R,
                     uint64_t r_off, MTL::Buffer* pc, uint32_t T, const char* probe_name, int layer)
{
    const uint32_t E = cfg_.n_embd;
    MTL::Buffer* on = scratch("hy_post", uint64_t(T) * E * 4);
    norm(enc, out, &post_norm, on, T, E, E, cfg_.rms_eps);
    if (probe_name && T == 1)
        probe(enc, probe_name, layer, on, E);
    const IhcArgs a{E, cfg_.hc_eps, cfg_.hc_magnitude, T};
    enc->setComputePipelineState(pso("ihc_post"));
    enc->setBuffer(on, 0, 0);
    enc->setBuffer(R, r_off, 1);
    enc->setBuffer(R, r_off, 2);
    enc->setBuffer(pc, 0, 3);
    enc->setBytes(&a, sizeof(a), 4);
    for (int r = repeats("ihc_post"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
    after(enc, "ihc_post");
}

// Indexer inputs for T tokens at positions pos0..: their value | gate rows of the layer's indexer cache (projections
// of the normed key), the pools those rows complete (normed, roped at their first position), and with qrn the
// tokens' roped queries and head weights.
void Engine::indexer_project(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                             MTL::Buffer* qrn, uint32_t T, uint32_t pos0)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    MTL::Buffer* ikr = scratch("hy_ikr", uint64_t(T) * Di * 4);
    MTL::Buffer* ikn = scratch("hy_ikn", uint64_t(T) * Di * 4);
    MTL::Buffer* rows = idx_rows_[il].get();
    const uint64_t row = uint64_t(pos0 - idx_base_[il]) * 2 * Di * 4;
    proj(enc, L.idx_k, xn, 0, ikr, 0, T);
    norm(enc, ikr, &L.idx_k_norm, ikn, T, Di, Di, cfg_.rms_eps);
    proj(enc, L.idx_value, ikn, 0, rows, row, T, 2 * Di);
    proj(enc, L.idx_gate, ikn, 0, rows, row + Di * 4, T, 2 * Di);

    const uint32_t first = pos0 / kp, last = (pos0 + T) / kp;
    if (last > first) {
        MTL::Buffer* pooled = idx_pooled_[il].get();
        const PoolArgs pa{Di, kp, first, idx_base_[il]};
        enc->setComputePipelineState(pso("idx_pool"));
        enc->setBuffer(rows, 0, 0);
        enc->setBuffer(L.idx_ape.buf, L.idx_ape.off, 1);
        enc->setBuffer(pooled, 0, 2);
        enc->setBytes(&pa, sizeof(pa), 3);
        for (int rep_ = repeats("idx_pool"); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(Di, last - first, 1), MTL::Size(Di, 1, 1));
        after(enc, "idx_pool");
        norm(enc, pooled, &L.idx_norm, pooled, last - first, Di, Di, cfg_.rms_eps, uint64_t(first) * Di * 4,
             uint64_t(first) * Di * 4);
        rope(enc, pooled, uint64_t(first) * Di * 4, freq_full_.get(), Di, 1, first * kp, last - first, kp);
    }
    if (qrn) {
        MTL::Buffer* iq = scratch("hy_iq", uint64_t(T) * Hi * Di * 4);
        MTL::Buffer* iw = scratch("hy_iw", uint64_t(T) * Hi * 4);
        proj(enc, L.idx_q_b, qrn, 0, iq, 0, T);
        rope(enc, iq, 0, freq_full_.get(), Di, Hi, pos0, T);
        proj(enc, L.idx_proj, xn, 0, iw, 0, T);
    }
}

// Tokens t..t+count-1 (at positions pos..) of the last indexer_project, past the dense window: each token's cell list
// of its n_top_ best pools among those before its own group, then that group up to itself.
void Engine::indexer_select(MTL::ComputeCommandEncoder*& enc, uint32_t il, uint32_t t, uint32_t pos, uint32_t count)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    const uint32_t stride = opts_.max_ctx / kp + 1, last_pools = (pos + count - 1) / kp;
    MTL::Buffer* score = scratch("hy_idx_score", uint64_t(count) * stride * 4);
    const ScoreArgs sa{Di, Hi, pos, kp, 1.0f / std::sqrt(float(Di * Hi)), stride};
    enc->setComputePipelineState(pso("idx_score"));
    enc->setBuffer(scratch("hy_iq", 0), uint64_t(t) * Hi * Di * 4, 0);
    enc->setBuffer(scratch("hy_iw", 0), uint64_t(t) * Hi * 4, 1);
    enc->setBuffer(idx_pooled_[il].get(), 0, 2);
    enc->setBuffer(score, 0, 3);
    enc->setBytes(&sa, sizeof(sa), 4);
    for (int rep_ = repeats("idx_score"); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size((last_pools + 255) / 256, count, 1), MTL::Size(256, 1, 1));
    after(enc, "idx_score");
    const SelectArgs sel{pos + 1, n_top_, kp, stride, list_stride(), 1};
    enc->setComputePipelineState(pso("idx_select"));
    enc->setBuffer(score, 0, 0);
    enc->setBytes(&sel, sizeof(sel), 1);
    enc->setBuffer(scratch("hy_idx_list", uint64_t(count) * list_stride() * 4), 0, 2);
    enc->setBuffer(scratch("hy_idx_count", uint64_t(count) * 4), 0, 3);
    for (int rep_ = repeats("idx_select"); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "idx_select");
}

// MLA for T tokens at positions pos0..: q is the k_b-absorbed latent query plus a roped part per head; keys are the
// cached latent rows (also the values) plus their roped part. Window layers see their last sliding_window keys,
// indexer layers all keys while a token has at most n_top_ pools before its group, then the indexer's selection.
void Engine::attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                       MTL::Buffer* att, uint32_t T, uint32_t pos0)
{
    const uint32_t H = cfg_.n_head[il], R = cfg_.kv_lora_rank, QR = cfg_.q_lora_rank, NR = cfg_.n_rot;
    const uint32_t HK = cfg_.head_k, NOPE = HK - NR, HV = cfg_.head_v;
    MTL::Buffer* freqs = L.swa ? freq_swa_.get() : freq_full_.get();
    MTL::Buffer* qr = scratch("hy_qr", uint64_t(T) * QR * 4);
    MTL::Buffer* qrn = scratch("hy_qrn", uint64_t(T) * QR * 4);
    MTL::Buffer* q = scratch("hy_q", uint64_t(T) * H * HK * 4);
    MTL::Buffer* qn = scratch("hy_qnope", uint64_t(T) * H * NOPE * 4);
    MTL::Buffer* qp = scratch("hy_qpe", uint64_t(T) * H * NR * 4);
    MTL::Buffer* kvl = scratch("hy_kvl", uint64_t(T) * R * 4);
    MTL::Buffer* kvn = scratch("hy_kvn", uint64_t(T) * R * 4);
    MTL::Buffer* kpe = scratch("hy_kpe", uint64_t(T) * NR * 4);
    MTL::Buffer* qabs = scratch("hy_qabs", uint64_t(T) * H * R * 4);
    MTL::Buffer* lat = scratch("hy_lat", uint64_t(T) * H * R * 4);
    MTL::Buffer* vout = scratch("hy_vout", uint64_t(T) * H * HV * 4);
    MTL::Buffer* g = scratch("hy_gate", uint64_t(T) * H * HV * 4);

    part(enc, "attn q kv");
    proj(enc, L.q_a, xn, 0, qr, 0, T);
    norm(enc, qr, &L.q_a_norm, qrn, T, QR, QR, cfg_.rms_eps);
    proj(enc, L.q_b, qrn, 0, q, 0, T);
    proj(enc, L.kv_lat, xn, 0, kvl, 0, T);
    proj(enc, L.kv_pe, xn, 0, kpe, 0, T);
    norm(enc, kvl, &L.kv_norm, kvn, T, R, R, cfg_.rms_eps);
    const SplitArgs spa{H, NOPE, NR};
    dispatch(enc, "hy4_split_q", {q, qn, qp}, &spa, sizeof(spa), MTL::Size(HK, H, T), MTL::Size(256, 1, 1));
    rope(enc, qp, 0, freqs, NR, H, pos0, T);
    rope(enc, kpe, 0, freqs, NR, 1, pos0, T);
    if (T == 1) {
        probe(enc, "q_resid", int(il), qrn, QR);
        probe(enc, "kv_cmpr", int(il), kvn, R);
        probe(enc, "q_pe", int(il), qp, uint64_t(H) * NR);
        probe(enc, "k_pe", int(il), kpe, NR);
    }
    slide(enc, il, pos0, T);
    const uint32_t base = row_base_[il]; // the position of the latent and rope caches' first row
    for (auto [src, dst, w] : {std::tuple{kvn, lat_[il].get(), R}, std::tuple{kpe, pe_[il].get(), NR}}) {
        const uint32_t n = T * w;
        enc->setComputePipelineState(pso("store_half"));
        enc->setBuffer(src, 0, 0);
        enc->setBuffer(dst, uint64_t(pos0 - base) * w * 2, 1);
        enc->setBytes(&n, sizeof(n), 2);
        for (int rep_ = repeats("store_half"); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "store_half");
    }

    // Per-head matrices (k_b: nope -> R, v_b: R -> head_v): the batched GEMV for a few tokens, else a GEMM per head.
    auto per_head = [&](const DevTensor& w, MTL::Buffer* x, uint32_t xw, MTL::Buffer* y, uint32_t yw) {
        if (T < gemm_min_tokens_) {
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
        after(enc, "per-head gemm");
    };
    part(enc, "attn k_b");
    per_head(L.k_b, qn, NOPE, qabs, R);
    if (L.indexer) {
        part(enc, "attn indexer project");
        indexer_project(enc, L, il, xn, qrn, T, pos0);
    }

    const float scale = 1.0f / std::sqrt(float(HK));
    const uint32_t W = L.swa ? cfg_.sliding_window : 0, kp = cfg_.indexer_kpool;
    MTL::Buffer* qlh = scratch("hy_qlh", uint64_t(T) * H * R * 2);
    MTL::Buffer* qph = scratch("hy_qph", uint64_t(T) * H * NR * 2);
    gemm_->to_half(enc, qabs, 0, T * H, R, 0, qlh, 0);
    gemm_->to_half(enc, qp, 0, T * H, NR, 0, qph, 0);
    after(enc, "to_half");
    // Dense: groups of query tokens over their keys from kbase on (scores of the latent and the rope part, a
    // softmax with the sinks over both, the latent rows weighted).
    const uint32_t maxkv = W ? W + kKeyAlign + T : pos0 + T;
    const uint32_t group = std::clamp<uint32_t>((4u << 20) / (H * maxkv), 1, 16);
    // Each group writes its rows of `lat` (sum_splits at the group's offset).
    auto dense_at = [&](uint32_t t, uint32_t gt) {
        const uint32_t p0 = pos0 + t, kbase = W ? window_start(p0, W) : 0;
        const uint32_t n_kv = p0 + gt - kbase, rows = gt * H;
        MTL::Buffer* sl = scratch("hy_sc_lat", uint64_t(rows) * n_kv * 4);
        MTL::Buffer* sp = scratch("hy_sc_pe", uint64_t(rows) * n_kv * 4);
        gemm_->encode(enc, gguf::Type::F16, R, n_kv, rows, lat_[il].get(), uint64_t(kbase - base) * R * 2, qlh,
                      uint64_t(t) * H * R * 2, sl, 0);
        // The dense kernel steps 64 columns at a time: the staged one's 256 would read past these 64-wide rows.
        gemm_->encode(enc, gguf::Type::F16, NR, n_kv, rows, pe_[il].get(), uint64_t(kbase - base) * NR * 2, qph,
                      uint64_t(t) * H * NR * 2, sp, 0, 0, 0, true);
        after(enc, "attn scores");
        const AttnSoftmaxArgs sa{n_kv, scale, p0 + 1 - kbase, H, W};
        enc->setComputePipelineState(pso("hy4_attn_softmax"));
        enc->setBuffer(sl, 0, 0);
        enc->setBuffer(sp, 0, 1);
        enc->setBuffer(L.sinks.buf, L.sinks.off, 2);
        enc->setBytes(&sa, sizeof(sa), 3);
        for (int rep_ = repeats("hy4_attn_softmax"); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "hy4_attn_softmax");
        const uint32_t keys_per_split = std::max<uint32_t>(256, ((n_kv + 15) / 16 + 63) / 64 * 64);
        const uint32_t splits = (n_kv + keys_per_split - 1) / keys_per_split;
        MTL::Buffer* partial = scratch("hy_pv_part", uint64_t(splits) * rows * R * 4);
        const PvArgs pa{rows, R, n_kv, keys_per_split};
        enc->setComputePipelineState(pso("attn_pv"));
        enc->setBuffer(sl, 0, 0);
        enc->setBuffer(lat_[il].get(), uint64_t(kbase - base) * R * 2, 1);
        enc->setBuffer(partial, 0, 2);
        enc->setBytes(&pa, sizeof(pa), 3);
        for (int rep_ = repeats("attn_pv"); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size(R / 64, (rows + 63) / 64, splits), MTL::Size(128, 1, 1));
        after(enc, "attn_pv");
        const uint32_t sum[2] = {rows * R, splits};
        enc->setComputePipelineState(pso("sum_splits"));
        enc->setBuffer(partial, 0, 0);
        enc->setBuffer(lat, uint64_t(t) * H * R * 4, 1);
        enc->setBytes(sum, sizeof(sum), 2);
        for (int rep_ = repeats("sum_splits"); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(rows * R, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "sum_splits");
    };
    auto sparse = [&](uint32_t t, uint32_t gt) {
        // A fixed split of each token's list (a single token still fills the GPU): its sums in the same order in any
        // group.
        const uint32_t Ls = list_stride(), rows = gt * H, splits = 8;
        part(enc, "attn indexer select");
        indexer_select(enc, il, t, pos0 + t, gt);
        part(enc, "attn sparse");
        MTL::Buffer* list = scratch("hy_idx_list", uint64_t(gt) * Ls * 4);
        MTL::Buffer* count = scratch("hy_idx_count", uint64_t(gt) * 4);
        MTL::Buffer* sl = scratch("hy_sp_lat", uint64_t(rows) * Ls * 4);
        MTL::Buffer* sp = scratch("hy_sp_pe", uint64_t(rows) * Ls * 4);
        MTL::Buffer* partial = scratch("hy_sp_part", uint64_t(splits) * rows * R * 4);
        const GatherArgs ga{H, R, Ls, splits, gt};
        enc->setComputePipelineState(pso("attn_gather_scores"));
        enc->setBuffer(qlh, uint64_t(t) * H * R * 2, 0);
        enc->setBuffer(lat_[il].get(), 0, 1);
        enc->setBuffer(list, 0, 2);
        enc->setBuffer(count, 0, 3);
        enc->setBuffer(sl, 0, 4);
        enc->setBytes(&ga, sizeof(ga), 5);
        for (int rep_ = repeats("attn_gather_scores"); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size((Ls + 31) / 32, 1, gt), MTL::Size(128, 1, 1));
        after(enc, "attn_gather_scores");
        const GatherRopeArgs gr{H, NR, Ls};
        enc->setComputePipelineState(pso("hy4_gather_rope"));
        enc->setBuffer(qp, uint64_t(t) * H * NR * 4, 0);
        enc->setBuffer(pe_[il].get(), 0, 1);
        enc->setBuffer(list, 0, 2);
        enc->setBuffer(count, 0, 3);
        enc->setBuffer(sp, 0, 4);
        enc->setBytes(&gr, sizeof(gr), 5);
        for (int rep_ = repeats("hy4_gather_rope"); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size((Ls + 63) / 64, H, gt), MTL::Size(64, 1, 1));
        after(enc, "hy4_gather_rope");
        const ListSoftmaxArgs la{Ls, scale, H};
        enc->setComputePipelineState(pso("hy4_softmax_list"));
        enc->setBuffer(sl, 0, 0);
        enc->setBuffer(sp, 0, 1);
        enc->setBuffer(count, 0, 2);
        enc->setBuffer(L.sinks.buf, L.sinks.off, 3);
        enc->setBytes(&la, sizeof(la), 4);
        for (int rep_ = repeats("hy4_softmax_list"); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "hy4_softmax_list");
        enc->setComputePipelineState(pso("attn_gather_pv"));
        enc->setBuffer(sl, 0, 0);
        enc->setBuffer(lat_[il].get(), 0, 1);
        enc->setBuffer(list, 0, 2);
        enc->setBuffer(count, 0, 3);
        enc->setBuffer(partial, 0, 4);
        enc->setBytes(&ga, sizeof(ga), 5);
        for (int rep_ = repeats("attn_gather_pv"); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size(R / 128, 1, gt * splits), MTL::Size(256, 1, 1));
        after(enc, "attn_gather_pv");
        const uint32_t sum[2] = {rows * R, splits};
        enc->setComputePipelineState(pso("sum_splits"));
        enc->setBuffer(partial, 0, 0);
        enc->setBuffer(lat, uint64_t(t) * H * R * 4, 1);
        enc->setBytes(sum, sizeof(sum), 2);
        for (int rep_ = repeats("sum_splits"); rep_ > 0; --rep_)
            enc->dispatchThreads(MTL::Size(rows * R, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "sum_splits");
    };
    part(enc, "attn core");
    // A token selects once it has more than n_top_ pools before its group: from position (n_top_ + 1) * kpool on.
    const uint32_t select_from = L.indexer ? (n_top_ + 1) * kp : UINT32_MAX;
    uint32_t t = 0;
    while (t < T && pos0 + t < select_from) {
        uint32_t gt = std::min({group, T - t, select_from - (pos0 + t)});
        // A window layer's group ends where its tokens' first key would change: each token's keys then sit at the
        // same places in the score GEMM in any chunk.
        while (W && gt > 1 && window_start(pos0 + t + gt - 1, W) != window_start(pos0 + t, W))
            --gt;
        dense_at(t, gt);
        t += gt;
    }
    while (t < T) {
        const uint32_t gt = std::min<uint32_t>(16, T - t);
        sparse(t, gt);
        t += gt;
    }

    part(enc, "attn v_b");
    per_head(L.v_b, lat, R, vout, HV);
    part(enc, "attn output");
    proj(enc, L.gate, xn, 0, g, 0, T);
    if (T == 1)
        probe(enc, "attn_kqv", int(il), vout, uint64_t(H) * HV);
    const uint32_t n = T * H * HV;
    dispatch(enc, "hy4_gate_mul", {vout, g}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    if (T == 1)
        probe(enc, "attn_gated", int(il), vout, uint64_t(H) * HV);
    proj(enc, L.wo, vout, 0, att, 0, T);
}

const std::vector<float>& Engine::step(int32_t token)
{
    const int32_t t[1] = {token};
    run(t, false);
    return logits_;
}

const std::vector<float>& Engine::prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled,
                                          const std::function<void(size_t)>& progress)
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
    struct Level {
        core::Telemetry& tel;
        core::Telemetry saved;
        ~Level() { tel = saved; }
    } level{tel_, tel_};
    if (tel_ == core::Telemetry::kernels)
        tel_ = core::Telemetry::summary;
    const ScratchLoan loan(*this, prefill_scratch(std::min<uint32_t>(opts_.prefill_chunk, uint32_t(tokens.size())),
                                                  std::min<uint32_t>(opts_.prefill_group, uint32_t(tokens.size()))));
    for (size_t g0 = 0; g0 < tokens.size(); g0 += opts_.prefill_group) {
        const size_t n = std::min<size_t>(opts_.prefill_group, tokens.size() - g0);
        if (progress)
            layer_done_ = [&](uint32_t layers) { progress(g0 + n * layers / cfg_.n_layer); };
        run(tokens.subspan(g0, n), bool(logits_sink_));
        layer_done_ = {};
        total.ensure_ms += last_step_.ensure_ms;
        total.prep_ms += last_step_.prep_ms;
        total.engram_ms += last_step_.engram_ms;
        total.engram_rows += last_step_.engram_rows;
        if (progress && g0 + n < tokens.size())
            progress(g0 + n);
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
    const bool decode = G <= kMaxDecodeTokens;
    step_start_ = Clock::now();
    encoding_ = true;
    const ExpertArena::Stats before = arena_->stats();
    const Stats stats_before = stats_;
    last_step_ = core::StepStats{};
    last_step_.pos = pos_;
    last_step_.token = tokens[0];
    last_step_.tokens = G;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // Engram rows of these positions (every module's), read and decoded before the GPU work that needs them.
    const uint32_t eg_layers = engram_ ? engram_->layers() : 0, eg_cols = engram_ ? engram_->columns() : 0;
    const uint32_t eg_dim = engram_ ? engram_->dim() : 0;
    const uint64_t eg_rows = uint64_t(G) * eg_layers * eg_cols;
    MTL::Buffer* eg = scratch("hy_eg", std::max<uint64_t>(eg_rows * eg_dim * 4, 16));
    MTL::Buffer* eg_valid = scratch("hy_eg_valid", uint64_t(G) * 4);
    if (engram_) {
        const auto t_eg = Clock::now();
        std::vector<int64_t> rows(eg_rows);
        engram_->hash(tokens, engram_history_, rows);
        engram_->read(rows, static_cast<float*>(eg->contents()), *engram_pool_);
        engram_history_ = engram_->advance(engram_history_, tokens);
        auto* v = static_cast<uint32_t*>(eg_valid->contents());
        for (uint32_t t = 0; t < G; ++t)
            v[t] = rows[uint64_t(t) * eg_layers * eg_cols] >= 0 ? 1u : 0u;
        last_step_.engram_ms = ms_since(t_eg);
        last_step_.engram_rows = eg_rows;
    }

    MTL::Buffer* xg = scratch("hy_xg", uint64_t(G) * E * 4);
    MTL::Buffer* xn_g = scratch("hy_xng", uint64_t(G) * E * 4);
    MTL::Buffer* ffo = scratch("hy_ffo", uint64_t(G) * E * 4);
    MTL::Buffer* x = scratch("hy_x", uint64_t(chunk) * E * 4);
    MTL::Buffer* xn = scratch("hy_xn", uint64_t(chunk) * E * 4);
    MTL::Buffer* att = scratch("hy_att", uint64_t(chunk) * E * 4);
    MTL::Buffer* pc = scratch("hy_pc", uint64_t(G) * 8 * 4);
    MTL::Buffer* R = scratch("hy_R", uint64_t(G) * hd * 4);
    MTL::Buffer* er = scratch("hy_er", std::max<uint64_t>(uint64_t(G) * eg_cols * eg_dim * 4, 16));
    MTL::Buffer* ekv = scratch("hy_ekv", uint64_t(G) * 5 * E * 4);
    embed(token_embd_, tokens.first(G), pos_, static_cast<float*>(xg->contents()));

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
    const IhcArgs ia{E, 0.0f, 0.0f, G}; // hc_init reads only the width
    dispatch(enc, "hc_init", {xg, R}, &ia, sizeof(ia), MTL::Size(E, G, 1), MTL::Size(256, 1, 1));
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        cur_layer_ = il;
        if (timing()) {
            enc->endEncoding();
            enc = encoder(cb, il, 0);
        }
        if (L.engram >= 0) {
            part(enc, "engram");
            const EngramRowsArgs ra{eg_dim, eg_cols, eg_layers * eg_cols * eg_dim, kEngramEps};
            enc->setComputePipelineState(pso("hy4_engram_rows"));
            enc->setBuffer(eg, uint64_t(L.engram) * eg_cols * eg_dim * 4, 0);
            enc->setBuffer(L.eg_lookup_norm.buf, L.eg_lookup_norm.off, 1);
            enc->setBuffer(er, 0, 2);
            enc->setBytes(&ra, sizeof(ra), 3);
            for (int rep_ = repeats("hy4_engram_rows"); rep_ > 0; --rep_)
                enc->dispatchThreadgroups(MTL::Size(eg_cols, G, 1), MTL::Size(32, 1, 1));
            after(enc, "hy4_engram_rows");
            proj(enc, L.eg_kv, er, 0, ekv, 0, G);
            const EngramArgs ea{E, kEngramEps};
            enc->setComputePipelineState(pso("hy4_engram_apply"));
            enc->setBuffer(R, 0, 0);
            enc->setBuffer(ekv, 0, 1);
            enc->setBuffer(L.eg_kv_b.buf, L.eg_kv_b.off, 2);
            enc->setBuffer(L.eg_gate.buf, L.eg_gate.off, 3);
            enc->setBuffer(L.eg_hc_scale.buf, L.eg_hc_scale.off, 4);
            enc->setBuffer(L.eg_gamma.buf, L.eg_gamma.off, 5);
            enc->setBuffer(eg_valid, 0, 6);
            enc->setBytes(&ea, sizeof(ea), 7);
            for (int rep_ = repeats("hy4_engram_apply"); rep_ > 0; --rep_)
                enc->dispatchThreadgroups(MTL::Size(G, 1, 1), MTL::Size(256, 1, 1));
            after(enc, "hy4_engram_apply");
            if (G == 1)
                probe(enc, "engram_out", int(il), R, hd);
        }
        for (uint32_t c0 = 0; c0 < G; c0 += chunk) {
            const uint32_t T = std::min(chunk, G - c0);
            const uint64_t r_off = uint64_t(c0) * hd * 4;
            part(enc, "attn hc and norm");
            hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R, r_off, x, pc, T, L.attn_norm, xn);
            attention(enc, L, il, xn, att, T, pos_ + c0);
            part(enc, "attn hc post");
            hc_post(enc, att, L.post_attn_norm, R, r_off, pc, T, "attn_out", int(il));
        }
        if (decode && !timing() && il == 0)
            commit_ahead(cb, enc);
        if (timing()) {
            enc->endEncoding();
            enc = encoder(cb, il, 1);
        }

        part(enc, "ffn hc and norm");
        hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, R, 0, xg, pc, G, L.ffn_norm, xn_g);
        if (L.dense) {
            for (uint32_t c0 = 0; c0 < G; c0 += chunk)
                ffn_dense(enc, L.ffn_gate, L.ffn_up, L.ffn_down, xn_g, ffo, 0.0f, std::min(chunk, G - c0),
                          uint64_t(c0) * E * 4, uint64_t(c0) * E * 4);
        } else if (decode) {
            const uint32_t nx = il + opts_.prefetch_ahead;
            moe_decode(cb, enc, L.moe, il, xn_g, ffo, G, {},
                       nx < cfg_.n_layer && !layers_[nx]->dense ? &layers_[nx]->moe : nullptr, opts_.prefetch_k,
                       opts_.prefetch_ahead);
        } else {
            for (uint32_t c0 = 0; c0 < G; c0 += chunk)
                ffn_dense(enc, L.moe.sh_gate, L.moe.sh_up, L.moe.sh_down, xn_g, ffo, 0.0f, std::min(chunk, G - c0),
                          uint64_t(c0) * E * 4, uint64_t(c0) * E * 4);
            moe_prefill(cb, enc, L.moe, il, xn_g, ffo, G, pos_, committed);
        }
        part(enc, "ffn hc post");
        hc_post(enc, ffo, L.post_ffn_norm, R, 0, pc, G, "ffn_out", int(il));
        if (G == 1)
            probe(enc, "l_out", int(il), R, hd);
        if (layer_done_ && il + 1 < cfg_.n_layer)
            layer_done_(il + 1);
        if (decode && !timing() && (il + 1) % kCommitLayers == 0 && il + 1 < cfg_.n_layer) {
            enc->endEncoding();
            cb->commit();
            committed.push_back(cb->retain());
            cb = ctx_.queue()->commandBuffer();
            cur_cb_ = cb;
            enc = encoder(cb, cur_layer_, -1);
        }
        if (decode && !committed.empty())
            serve_moe(committed.back(), false);
    }

    // Logits of the last position, or of every one (in chunks) for the sink.
    cur_layer_ = cfg_.n_layer;
    if (timing()) {
        enc->endEncoding();
        enc = encoder(cb, cur_layer_, -1);
    }
    const uint32_t nl = all_logits ? chunk : 1;
    MTL::Buffer* flat = scratch("hy_lflat", uint64_t(nl) * hd * 4);
    MTL::Buffer* mixes = scratch("hy_lmix", uint64_t(nl) * 4 * 4);
    MTL::Buffer* xl = scratch("hy_xl", uint64_t(nl) * E * 4);
    MTL::Buffer* xnl = scratch("hy_xnl", uint64_t(nl) * E * 4);
    MTL::Buffer* lg = scratch("hy_logits", uint64_t(nl) * cfg_.n_vocab * 4);
    for (uint32_t l0 = all_logits ? 0 : G - 1; l0 < G; l0 += nl) {
        const uint32_t n = std::min(nl, G - l0);
        norm(enc, R, nullptr, flat, n, hd, 0, cfg_.rms_eps, uint64_t(l0) * hd * 4);
        proj(enc, head_fn_, flat, 0, mixes, 0, n);
        const IhcArgs ha{E, cfg_.hc_eps, cfg_.hc_magnitude, n};
        enc->setComputePipelineState(pso("ihc_head"));
        enc->setBuffer(mixes, 0, 0);
        enc->setBuffer(head_scale_.buf, head_scale_.off, 1);
        enc->setBuffer(head_base_.buf, head_base_.off, 2);
        enc->setBuffer(R, uint64_t(l0) * hd * 4, 3);
        enc->setBuffer(xl, 0, 4);
        enc->setBytes(&ha, sizeof(ha), 5);
        enc->dispatchThreads(MTL::Size(E, n, 1), MTL::Size(256, 1, 1));
        after(enc, "ihc_head");
        norm(enc, xl, &output_norm_, xnl, n, E, E, cfg_.rms_eps);
        if (G == 1)
            probe(enc, "result_norm", -1, xnl, E);
        proj(enc, output_, xnl, 0, lg, 0, n);
        enc->endEncoding();
        finish(cb);
        drain();
        const auto* lgp = static_cast<const float*>(lg->contents());
        if (all_logits && logits_sink_)
            for (uint32_t i = 0; i < n; ++i)
                logits_sink_(pos_ + l0 + i, lgp + uint64_t(i) * cfg_.n_vocab);
        if (l0 + n == G) {
            std::memcpy(logits_.data(), lgp + uint64_t(n - 1) * cfg_.n_vocab, uint64_t(cfg_.n_vocab) * 4);
            break;
        }
        cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        enc = encoder(cb, cur_layer_, -1);
    }
    pos_ += G;
    pool->release();

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

} // namespace mako::hy4f
