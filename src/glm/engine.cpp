#include "glm/engine.hpp"

#include "gguf/read_pool.hpp"

#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"

#include <array>
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
struct PrepArgs {
    uint32_t channels, kernel_len, tokens, snap, head_dim, heads;
    float lower_bound, l2_eps;
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
    MoeWeights moe;
    ExpertArena::LayerParts experts{};
};

Engine::Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx)
    : EngineBase(model, ctx, opts.telemetry), cfg_(cfg), opts_(opts)
{
    const auto t_load = Clock::now();
    dims_ = {cfg_.n_embd, cfg_.n_layer, cfg_.hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
             cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp};
    dense_quant_ = opts.dense_quant;
    if (!dense_quant_)
        if (const char* e = std::getenv("MAKO_DENSE_QUANT")) {
            const std::string_view v(e);
            dense_quant_ = v == "q4_k" ? gguf::Type::Q4_K : v == "q5_k" ? gguf::Type::Q5_K : gguf::Type::Q6_K;
        }
    routing_ = {core::Routing::Score::sigmoid, cfg_.expert_weights_norm, cfg_.expert_weights_scale};
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
    auto load_moe = [&](Layer& L, const std::string& p, bool prefill, uint32_t il) {
        L.moe.clamp_exp = cfg_.swiglu_clamp_exp[il];
        L.moe.clamp_sh = cfg_.swiglu_clamp_shexp[il];
        L.moe.router = upload(p + "ffn_gate_inp.weight");
        L.moe.sh_gate = upload(p + "ffn_gate_shexp.weight");
        L.moe.sh_up = upload(p + "ffn_up_shexp.weight");
        L.moe.sh_down = upload(p + "ffn_down_shexp.weight");
        L.moe.bias = upload(p + "exp_probs_b.bias");
        if (L.moe.bias.type != gguf::Type::F32 || L.moe.bias.dims[0] != cfg_.n_expert)
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
            L->g_a = upload(p + "ssm_g_a.weight"); // next to f_a: one GEMV runs both
            L->f_b = upload(p + "ssm_f_b.weight");
            L->beta = upload(p + "ssm_beta.weight");
            L->ssm_a = upload(p + "ssm_a");
            L->dt_b = upload(p + "ssm_dt.bias");
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
            load_moe(*L, p, true, il);
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
        load_moe(*L, p, false, cfg_.n_layer);
        L->eh_proj = upload(p + "nextn.eh_proj.weight");
        L->enorm = upload(p + "nextn.enorm.weight");
        L->hnorm = upload(p + "nextn.hnorm.weight");
        L->head_norm = upload(p + "nextn.shared_head_norm.weight");
        layers_.push_back(std::move(L));
    }
    logits_.resize(cfg_.n_vocab);
    for (const auto& L : layers_)
        layer_mla_.push_back(!L->kda);

    load_.dense_ms = ms_since(t_load);
    set_staging_limit(opts_.stage_gb);
    ArenaInit init;
    for (const auto& L : layers_)
        init.parts.push_back(L->experts);
    for (uint32_t il = cfg_.n_dense_lead; il < cfg_.n_layer; ++il)
        init.moe_layers.push_back(il);
    built_chunk_ = opts_.prefill_chunk;
    built_group_ = std::max(opts_.prefill_chunk, opts_.prefill_group);
    init.lendable = double(prefill_scratch(opts_.prefill_chunk, std::max(opts_.prefill_chunk, opts_.prefill_group)));
    // The context caches are allocated for max_ctx but only fill (and count in the footprint) as the context grows;
    // the checkpoints a Session keeps add to them.
    for (const auto* caches : {&kv_cache_, &idx_cache_, &idx_pooled_})
        for (const auto& b : *caches)
            if (b)
                init.context_bytes += b->length();
    init.context_bytes += uint64_t(opts_.checkpoints) * checkpoint_bytes();
    init.arena_gb = opts_.arena_gb;
    init.park_gb = opts_.park_gb;
    init.simulate_ram_gb = opts_.simulate_ram_gb;
    init.expert_counts = opts_.expert_counts;
    init_runtime(std::move(init));
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
    const uint64_t staging = group >= kStageMinTokens ? staging_bytes() : 0;
    return (uint64_t(chunk) * per_chunk_token + uint64_t(group) * per_group_token + batch) * 4 + sparse_scratch() +
           staging + (96ull << 20);
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

Engine::~Engine() = default;

void Engine::set_ablation(std::optional<mtl::Ablation> ablation)
{
    ablation_ = std::move(ablation);
    ablate_ffn_ = ablation_ ? ablation_->site("ffn_out") : -1;
    ablate_stream_ = ablation_ ? ablation_->site("stream") : -1;
}

void Engine::ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* x, uint32_t rows)
{
    if (!ablation_ || !ablation_->active(il, site))
        return;
    ablation_->apply(enc, pso("ablate_rows"), il, site, x, 0, rows);
    after(enc, "ablate_rows");
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
    // With snap_ (a speculative step), the state after the first snap_ tokens also goes to the alternate buffers.
    const uint32_t snap = snap_ && T > snap_ ? snap_ : 0;
    // f_a's rows, then g_a's (one GEMV when their weights are adjacent); f_b, beta and g_b from those: all before
    // the conv, which needs none of them.
    MTL::Buffer* fga = scratch("fga", uint64_t(T) * D * 8);
    const uint64_t ga_off = uint64_t(T) * D * 4;
    MTL::Buffer* graw = scratch("graw", uint64_t(T) * inner * 4);
    MTL::Buffer* braw = scratch("braw", uint64_t(T) * H * 4);
    MTL::Buffer* g = scratch("g", uint64_t(T) * inner * 4);
    MTL::Buffer* beta = scratch("beta", uint64_t(T) * H * 4);
    MTL::Buffer* g2 = scratch("g2", uint64_t(T) * inner * 4);
    const bool qkv_adjacent = adjacent(L.wq, L.wk) && adjacent(L.wk, L.wv);
    const bool gates_adjacent = T < gemm_min_tokens_ && adjacent(L.f_a, L.g_a);
    part(enc, "kda q k v");
    bool done = false;
    if (T == 1 && qkv_adjacent && gates_adjacent) {
        // q/k/v, f_a/g_a and beta (all of xn) in one dispatch: the small ones hide under the large.
        const std::array<mtl::Gemv::Job, 3> jobs{gemv_job(L.wq, xn, 0, qkv, 0, 3, 1, true),
                                                  gemv_job(L.f_a, xn, 0, fga, 0, 2, 1, true),
                                                  gemv_job(L.beta, xn, 0, braw, 0, 1, 1, false)};
        if (gemv_jobs_ok(jobs)) {
            gemv_jobs(enc, jobs);
            done = true;
        }
    }
    if (!done) {
        if (T == 1 && qkv_adjacent) {
            gemv(enc, L.wq, xn, 0, qkv, 0, 3, 1, true);
        } else {
            proj(enc, L.wq, xn, 0, qkv, 0, T, 3 * inner);
            proj(enc, L.wk, xn, 0, qkv, uint64_t(inner) * 4, T, 3 * inner);
            proj(enc, L.wv, xn, 0, qkv, uint64_t(inner) * 8, T, 3 * inner);
        }
    }
    part(enc, "kda gates");
    if (!done) {
        if (gates_adjacent) {
            gemv(enc, L.f_a, xn, 0, fga, 0, 2, T, true);
        } else {
            proj(enc, L.f_a, xn, 0, fga, 0, T);
            proj(enc, L.g_a, xn, 0, fga, ga_off, T);
        }
        proj(enc, L.beta, xn, 0, braw, 0, T);
    }
    proj(enc, L.f_b, fga, 0, graw, 0, T);
    proj(enc, L.g_b, fga, ga_off, g2, 0, T);
    const bool fused = !probe_; // the probes read the conv output before its norm
    if (fused) {
        // Conv, gate and the q/k norm in one dispatch: a threadgroup per head walking the tokens, or past a decode
        // step's rows one per (head, token) and the state moved on after.
        part(enc, "kda conv");
        const bool rows = T > 8;
        const PrepArgs pa{inner, cfg_.conv_kernel, T, snap, D, H, cfg_.kda_gate_lower_bound, 1e-6f};
        enc->setComputePipelineState(pso(rows ? "kda_prep_rows" : "kda_prep"));
        enc->setBuffer(qkv, 0, 0);
        enc->setBuffer(L.conv_q.buf, L.conv_q.off, 1);
        enc->setBuffer(L.conv_k.buf, L.conv_k.off, 2);
        enc->setBuffer(L.conv_v.buf, L.conv_v.off, 3);
        enc->setBuffer(conv_state_[il].get(), 0, 4);
        enc->setBuffer(qkvc, 0, 5);
        enc->setBytes(&pa, sizeof(pa), 6);
        enc->setBuffer(snap ? conv_state_alt_[il].get() : conv_state_[il].get(), 0, 7);
        enc->setBuffer(graw, 0, 8);
        enc->setBuffer(L.dt_b.buf, L.dt_b.off, 9);
        enc->setBuffer(L.ssm_a.buf, L.ssm_a.off, 10);
        enc->setBuffer(braw, 0, 11);
        enc->setBuffer(g, 0, 12);
        enc->setBuffer(beta, 0, 13);
        for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size(H, rows ? T : 1, 1), MTL::Size(D, 1, 1));
        after(enc, rows ? "kda_prep_rows" : "kda_prep");
        if (rows) {
            const ConvArgs ca{inner, cfg_.conv_kernel, T, 0};
            enc->setComputePipelineState(pso("kda_conv_shift"));
            enc->setBuffer(qkv, 0, 0);
            enc->setBuffer(conv_state_[il].get(), 0, 1);
            enc->setBytes(&ca, sizeof(ca), 2);
            for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
                enc->dispatchThreads(MTL::Size(3 * inner, 1, 1), MTL::Size(256, 1, 1));
            after(enc, "kda_conv_shift");
        }
    } else {
        part(enc, "kda conv");
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

        part(enc, "kda l2 norm");
        const L2Args la{D, 1e-6f, 3 * inner};
        enc->setComputePipelineState(pso("l2_norm"));
        enc->setBuffer(qkvc, 0, 0);
        enc->setBytes(&la, sizeof(la), 1);
        const uint32_t l2_rows = 2 * H; // q and k heads
        enc->setBytes(&l2_rows, sizeof(l2_rows), 2);
        for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
            enc->dispatchThreadgroups(MTL::Size((l2_rows + 3) / 4, T, 1), MTL::Size(128, 1, 1));
        after(enc, "l2_norm");
    }

    MTL::Buffer* o = scratch("o", uint64_t(T) * inner * 4);
    part(enc, "kda scan");
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

    MTL::Buffer* gated = scratch("gated", uint64_t(T) * inner * 4);
    part(enc, "kda output gate");
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
    part(enc, "kda output");
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

    part(enc, "mla q kv");
    // A few tokens: the projections of xn (q_a, kv_a, the indexer's key, gate and weights) in one dispatch, then
    // those of the normed q latent (q_b, the indexer's query) in one.
    bool projected = false;
    if (T < gemm_min_tokens_) {
        const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads;
        MTL::Buffer* iq = scratch("iq", uint64_t(T) * Hi * Di * 4);
        MTL::Buffer* ikr = scratch("ikr", uint64_t(T) * Di * 4);
        MTL::Buffer* iw = scratch("iw", uint64_t(T) * Hi * 4);
        mtl::Gemv::Job gate = gemv_job(L.idx_gate, xn, 0, idx_cache_[il].get(), uint64_t(pos0) * 2 * Di * 4 + Di * 4,
                                       1, T, false);
        gate.ldy = 2 * Di;
        const std::array<mtl::Gemv::Job, 5> first{gemv_job(L.q_a, xn, 0, qr, 0, 1, T, false),
                                                  gemv_job(L.kv_a, xn, 0, kv, 0, 1, T, false),
                                                  gemv_job(L.idx_k, xn, 0, ikr, 0, 1, T, false), gate,
                                                  gemv_job(L.idx_proj, xn, 0, iw, 0, 1, T, false)};
        const std::array<mtl::Gemv::Job, 2> second{gemv_job(L.q_b, qrn, 0, q, 0, 1, T, false),
                                                   gemv_job(L.idx_q_b, qrn, 0, iq, 0, 1, T, false)};
        if (gemv_jobs_ok(first) && gemv_jobs_ok(second)) {
            gemv_jobs(enc, first);
            norm(enc, qr, &L.q_a_norm, qrn, T, QR, QR, cfg_.rms_eps);
            gemv_jobs(enc, second);
            norm(enc, kv, &L.kv_a_norm, kvn, T, R, R, cfg_.rms_eps);
            projected = true;
        }
    }
    if (!projected) {
        proj(enc, L.q_a, xn, 0, qr, 0, T);
        norm(enc, qr, &L.q_a_norm, qrn, T, QR, QR, cfg_.rms_eps);
        proj(enc, L.q_b, qrn, 0, q, 0, T);
        proj(enc, L.kv_a, xn, 0, kv, 0, T);
        norm(enc, kv, &L.kv_a_norm, kvn, T, R, R, cfg_.rms_eps);
    }

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
        if (T < gemm_min_tokens_) {
            for (uint32_t t = 0; t < T; ++t)
                gemv(enc, w, x, uint64_t(t) * H * xw * 4, y, uint64_t(t) * H * yw * 4, H);
            return;
        }
        const uint32_t k = uint32_t(w.dims[0]), nr = uint32_t(w.dims[1]);
        const uint64_t mat = mtl::Gemv::row_bytes(w.type, k) * nr;
        part(enc, "mla k_b v_b to half");
        MTL::Buffer* xh = halves(enc, x, 0, T, H * xw, 0);
        // The heads write disjoint columns of y: one concurrent encoder, so their GEMMs overlap.
        enc->endEncoding();
        enc = encoder(cur_cb_, cur_layer_, 0, MTL::DispatchTypeConcurrent);
        name_part("mla k_b v_b");
        for (uint32_t h = 0; h < H; ++h)
            gemm_->encode(enc, w.type, k, nr, T, w.buf, w.off + h * mat, xh, uint64_t(h) * xw * 2, y,
                          uint64_t(h) * yw * 4, H * yw, H * xw);
        enc->endEncoding();
        enc = encoder(cur_cb_, cur_layer_, 0);
    };
    part(enc, "mla k_b v_b");
    per_head(L.k_b, q, cfg_.head_k, qabs, R);
    part(enc, "mla indexer project");
    indexer_project(enc, L, il, xn, qrn, T, pos0, projected);
    part(enc, "mla dense attention");
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
        part(enc, "mla indexer select");
        if (!indexer_select(enc, il, t, pos0 + t, g))
            throw std::logic_error("sparse attention group inside the dense window");
        part(enc, "mla sparse attention");
        attend_sparse(enc, kv_cache_[il].get(), qabs, t * lrow * 4, lat, t * lrow * 4, g);
        t += g;
    }
    part(enc, "mla k_b v_b");
    per_head(L.v_b, lat, R, vout, cfg_.head_v);
    part(enc, "mla output");
    proj(enc, L.wo, vout, 0, att, 0, T);
}

// Indexer inputs for T tokens at positions pos0..: their query and weight projections, their key|gate rows of the
// layer's indexer cache, and the pools those rows complete.
void Engine::indexer_project(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn,
                             MTL::Buffer* qrn, uint32_t T, uint32_t pos0, bool projected)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    MTL::Buffer* iq = scratch("iq", uint64_t(T) * Hi * Di * 4);
    MTL::Buffer* ikr = scratch("ikr", uint64_t(T) * Di * 4);
    MTL::Buffer* iw = scratch("iw", uint64_t(T) * Hi * 4);
    MTL::Buffer* cache = idx_cache_[il].get();
    const uint64_t row = uint64_t(pos0) * 2 * Di * 4;

    if (qrn && !projected) // else only the cache rows and pools (no queries)
        proj(enc, L.idx_q_b, qrn, 0, iq, 0, T);
    if (!projected)
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
    if (!projected) {
        proj(enc, L.idx_gate, xn, 0, cache, row + Di * 4, T, 2 * Di);
        if (qrn)
            proj(enc, L.idx_proj, xn, 0, iw, 0, T);
    }

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
    part(enc, "mla sparse scores");
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
    part(enc, "mla sparse softmax");
    const ListSoftmaxArgs sa{L, 1.0f / std::sqrt(float(cfg_.head_k)), H};
    enc->setComputePipelineState(pso("attn_softmax_list"));
    enc->setBuffer(scores, 0, 0);
    enc->setBuffer(count, 0, 1);
    enc->setBytes(&sa, sizeof(sa), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "attn_softmax_list");
    part(enc, "mla sparse output");
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
    last_step_.tokens = T;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const uint64_t rows = kMaxDecodeTokens; // fixed sizes: the next-token block reads "xn" after the step
    MTL::Buffer* x = scratch("x", rows * E * 4);
    MTL::Buffer* xn = scratch("xn", rows * E * 4);
    MTL::Buffer* att = scratch("att", rows * E * 4);
    MTL::Buffer* ffo = scratch("ffo", rows * E * 4);
    MTL::Buffer* pc = scratch("pc", rows * 24 * 4);
    scratch("R", rows * hd * 4);
    scratch("R2", rows * hd * 4);

    embed(token_embd_, tokens.first(T), pos_, static_cast<float*>(x->contents()));

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
                    T, &L.attn_norm, xn);
        probe(enc, "hc_attn_pre", int(il), x, E);
        if (!fused) // fused: the stream step wrote xn
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
                    L.hc_ffn_base, x, pc, T, &L.ffn_norm, xn);
            std::swap(scratch_["R"], scratch_["R2"]);
        } else {
            hc_post(enc, att, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), pc, T);
            std::swap(scratch_["R"], scratch_["R2"]);
            hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, scratch("R", rows * hd * 4), x, pc, T);
        }
        probe(enc, "hc_ffn_pre", int(il), x, E);
        probe(enc, "hc_mixes", int(il), scratch("mixes", 32 * 4), 24);
        if (!fused)
            norm(enc, x, &L.ffn_norm, xn, T, E, E, cfg_.rms_eps);
        probe(enc, "ffn_norm", int(il), xn, E);
        if (L.dense) {
            ffn_dense(enc, L.ffn_gate, L.ffn_up, L.ffn_down, xn, ffo, cfg_.swiglu_clamp_shexp[il], T);
        } else {
            moe_decode(cb, enc, L.moe, il, xn, ffo, T);
        }
        probe(enc, "ffn_out", int(il), ffo, E);
        ablate(enc, il, ablate_ffn_, ffo, T);
        ablate(enc, il, ablate_stream_, scratch("R", rows * hd * 4), T * cfg_.hc);
        if (fused && il + 1 < cfg_.n_layer) {
            const Layer& N = *layers_[il + 1];
            hc_step(enc, ffo, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), N.hc_attn_fn, N.hc_attn_scale,
                    N.hc_attn_base, x, pc, T, &N.attn_norm, xn);
        } else {
            hc_post(enc, ffo, scratch("R", rows * hd * 4), scratch("R2", rows * hd * 4), pc, T);
        }
        std::swap(scratch_["R"], scratch_["R2"]);
        probe(enc, "l_out", int(il), scratch("R", rows * hd * 4), hd);
        if (il == 0) // the GPU starts on the first layer while the rest is encoded (~0.3 ms)
            commit_ahead(cb, enc);
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
    cp.state.reserve(checkpoint_bytes());
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

std::vector<std::pair<MTL::Buffer*, uint64_t>> Engine::positional(uint32_t pos) const
{
    if (pos > opts_.max_ctx)
        throw std::runtime_error("position past the context size");
    const uint64_t Di = cfg_.indexer_dim, pools = pos / cfg_.indexer_kpool;
    std::vector<std::pair<MTL::Buffer*, uint64_t>> out;
    for (size_t il = 0; il < kv_cache_.size(); ++il)
        if (kv_cache_[il]) {
            out.emplace_back(kv_cache_[il].get(), uint64_t(pos) * cfg_.kv_lora_rank * 2);
            out.emplace_back(idx_cache_[il].get(), uint64_t(pos) * 2 * Di * 4);
            out.emplace_back(idx_pooled_[il].get(), pools * Di * 4);
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
            gguf::read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(next[i]) * rb);
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
    moe_decode(cb, enc, L.moe, il, xn, ffo, T);
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

// Prompt processing: chunks of tokens go through each layer together (projections as GEMMs, the sequential parts
// token by token inside the kernels), with one CPU synchronization per MoE layer to make the chunk's experts
// resident. Leaves the same caches and state as step() would token by token.
const std::vector<float>& Engine::prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled,
                                          const std::function<void(size_t)>& progress)
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
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
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

        embed(token_embd_, tokens.subspan(g0, G), pos_ + uint32_t(g0), static_cast<float*>(xg_in->contents()));
        // The next-token block's inputs: each position's following prompt token (written before any of the group's
        // GPU work, which reads them).
        if (has_mtp() && g0 + 1 < tokens.size())
            embed(token_embd_, tokens.subspan(g0 + 1, std::min<size_t>(G, tokens.size() - g0 - 1)),
                  pos_ + uint32_t(g0) + 1, static_cast<float*>(scratch("mf_e", uint64_t(group) * E * 4)->contents()));

        begin_staging(G);
        struct StagingEnd {
            Engine& e;
            ~StagingEnd() { e.end_staging(); }
        } staging_end{*this};
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
                part(enc, "hc and norm");
                hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R, x, pc, T, r_off);
                norm(enc, x, &L.attn_norm, xn, T, E, E, cfg_.rms_eps);
                if (L.kda)
                    kda(enc, L, il, xn, att, T);
                else
                    mla(enc, L, il, xn, att, T, pos_ + uint32_t(g0) + c0);
                part(enc, "hc and norm");
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
            const DevTensor& fg = L.dense ? L.ffn_gate : L.moe.sh_gate;
            const DevTensor& fu = L.dense ? L.ffn_up : L.moe.sh_up;
            const DevTensor& fd = L.dense ? L.ffn_down : L.moe.sh_down;
            for (uint32_t c0 = 0; c0 < G; c0 += chunk)
                ffn_dense(enc, fg, fu, fd, xn_g, ffo, cfg_.swiglu_clamp_shexp[il], std::min(chunk, G - c0),
                          uint64_t(c0) * E * 4, uint64_t(c0) * E * 4);
            if (!L.dense)
                moe_prefill(cb, enc, L.moe, il, xn_g, ffo, G, pos_, committed);
            ablate(enc, il, ablate_ffn_, ffo, G);
            ablate(enc, il, ablate_stream_, R, G * cfg_.hc);
            hc_post(enc, ffo, R, R2, pc_g, G);
            std::swap(R, R2);
            // Within a group, its share of the layers done: the GPU has a MoE layer up to routing once it returns.
            if (progress && il + 1 < cfg_.n_layer)
                progress(g0 + size_t(G) * (il + 1) / cfg_.n_layer);
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
            for (const core::KernelTime& k : last_step_.parts) {
                auto it = std::ranges::find(total.parts, k.kernel, &core::KernelTime::kernel);
                if (it == total.parts.end())
                    it = total.parts.insert(it, core::KernelTime{k.kernel, 0, 0});
                it->ms += k.ms;
                it->count += k.count;
            }
        }
        pool->release();
        if (progress && g0 + G < tokens.size())
            progress(g0 + G);
        if (cancelled && g0 + G < tokens.size() && cancelled()) {
            done = g0 + G;
            break;
        }
    }
    pos_ += uint32_t(done);
    last_pos0_ = pos_ - 1; // mtp_draft() continues from the last position
    last_tokens_ = 1;
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
