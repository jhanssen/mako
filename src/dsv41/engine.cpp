#include "dsv41/engine.hpp"

#include "gguf/read_pool.hpp"
#include "quant/dequant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <numbers>
#include <stdexcept>

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
    uint32_t dim, emit;
};
struct GatherArgs {
    uint32_t dim, n_window, first_pos, ring, n_comp, ids_identity;
};
struct ScoreArgs {
    uint32_t dim, heads, n;
    float scale;
};
struct IdxArgs {
    uint32_t dim, heads, n;
    float w_scale;
};
struct BlockArgs {
    uint32_t n, block;
};
struct TopkArgs {
    uint32_t n, k, with_flags;
};
struct EngramArgs {
    uint32_t dim;
    float eps;
};

constexpr uint32_t kFp8 = 0, kFp4E4m3 = 1, kFp4Pow2 = 2; // ds_round kinds

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

Engine::Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx)
    : EngineBase(model, ctx, opts.telemetry), cfg_(cfg), opts_(opts)
{
    const auto t_load = Clock::now();
    if (cfg_.head_dim != 512 || cfg_.n_rot != 64 || cfg_.hc != 4 || cfg_.n_head * cfg_.head_dim != cfg_.o_groups * 4096 ||
        cfg_.indexer_dim != 128 || cfg_.n_expert_used > 16)
        throw std::runtime_error("unsupported DeepSeek-V4.1 geometry");
    if (cfg_.gating != 4)
        throw std::runtime_error(std::format("expert gating function {} (only sqrt(softplus), 4)", cfg_.gating));
    dims_ = {cfg_.n_embd, cfg_.n_layer, cfg_.hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
             cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp};
    routing_ = {core::Routing::Score::sqrt_softplus, cfg_.expert_weights_norm, cfg_.expert_weights_scale};

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
        // Prefill (expert GEMMs) is not built yet: decode only needs the GEMV.
        L->experts = {expert(p + "ffn_gate_exps.weight", false), expert(p + "ffn_up_exps.weight", false),
                      expert(p + "ffn_down_exps.weight", false)};
        layers_.push_back(std::move(L));
        layer_mla_.push_back(1);
    }

    auto eg = Engram::open(cfg_, model_);
    if (!eg)
        throw std::runtime_error(eg.error());
    engram_.emplace(std::move(*eg));
    engram_pool_ = std::make_unique<gguf::ReadPool>(8);
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
        context_bytes += ckv_.back()->length() + ik_.back()->length();
    }
    logits_.resize(cfg_.n_vocab);

    load_.dense_ms = ms_since(t_load);
    ArenaInit init;
    for (const auto& L : layers_)
        init.parts.push_back(L->experts);
    for (uint32_t il = 0; il < cfg_.n_layer; ++il)
        init.moe_layers.push_back(il);
    init.context_bytes = context_bytes;
    init.arena_gb = opts_.arena_gb;
    init.simulate_ram_gb = opts_.simulate_ram_gb;
    init.expert_counts = opts_.expert_counts;
    init_runtime(std::move(init));
}

Engine::~Engine() = default;

void Engine::reset()
{
    pos_ = 0;
    engram_history_.assign(engram_->history(), -1);
    for (auto& b : comp_state_)
        std::memset(b->contents(), 0, b->length());
}

void Engine::set_ablation(std::optional<mtl::Ablation> ablation)
{
    ablation_ = std::move(ablation);
    ablate_attn_ = ablation_ ? ablation_->site("attn_out") : -1;
    ablate_shexp_ = ablation_ ? ablation_->site("shexp_out") : -1;
}

void Engine::ablate(MTL::ComputeCommandEncoder*& enc, uint32_t il, int site, MTL::Buffer* x)
{
    if (!ablation_ || !ablation_->active(il, site))
        return;
    ablation_->apply(enc, pso("ablate_rows"), il, site, x, 0, 1);
    after(enc, "ablate_rows");
}

void Engine::rope(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint64_t x_off, bool compressed, uint32_t head_dim,
                  uint32_t heads, uint32_t pos, bool inverse)
{
    const RopeArgs a{head_dim, heads, cfg_.n_rot, heads * head_dim, pos, 1, inverse ? 1u : 0u};
    enc->setComputePipelineState(pso("ds_rope"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(compressed ? freq_comp_.get() : freq_window_.get(), 0, 1);
    enc->setBytes(&a, sizeof(a), 2);
    for (int r = repeats("ds_rope"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(cfg_.n_rot / 2, heads, 1), MTL::Size(32, 1, 1));
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
                       const DevTensor& base, MTL::Buffer* R, MTL::Buffer* x, MTL::Buffer* pc, MTL::Buffer* pc_prev)
{
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    MTL::Buffer* flat = scratch("flat", uint64_t(hd) * 4);
    MTL::Buffer* mixes = scratch("mixes", 24 * 4);
    norm(enc, R, nullptr, flat, 1, hd, 0, cfg_.rms_eps);
    proj(enc, fn, flat, 0, mixes, 0, 1);
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, 1};
    enc->setComputePipelineState(pso("hc_mix"));
    enc->setBuffer(mixes, 0, 0);
    enc->setBuffer(scale.buf, scale.off, 1);
    enc->setBuffer(base.buf, base.off, 2);
    enc->setBuffer(pc, 0, 3);
    enc->setBytes(&ha, sizeof(ha), 4);
    enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    after(enc, "hc_mix");
    dispatch(enc, "hc_collapse", {R, pc_prev, x}, &ha, sizeof(ha), MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));
}

// One token's attention at position pos_: window rows of this layer plus the compressed rows its KV source keeps
// (all of them up to indexer_top_k, else the indexer's selection), a sink per head, then the grouped output
// projection.
void Engine::attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, uint32_t il, MTL::Buffer* xn, MTL::Buffer* att)
{
    const uint32_t H = cfg_.n_head, D = cfg_.head_dim, QR = cfg_.q_lora_rank, W = cfg_.sliding_window;
    const uint32_t p = pos_, Hi = cfg_.indexer_heads, Di = cfg_.indexer_dim, top = cfg_.indexer_top_k;
    const bool comp = L.ratio != 0;
    MTL::Buffer* qr = scratch("ds_qr", QR * 4);
    MTL::Buffer* qrn = scratch("ds_qrn", QR * 4);
    MTL::Buffer* q = scratch("ds_q", uint64_t(H) * D * 4);
    MTL::Buffer* kv = scratch("ds_kv", D * 4);
    MTL::Buffer* kvn = scratch("ds_kvn", D * 4);
    proj(enc, L.q_a, xn, 0, qr, 0, 1);
    norm(enc, qr, &L.q_a_norm, qrn, 1, QR, QR, cfg_.rms_eps);
    proj(enc, L.q_b, qrn, 0, q, 0, 1);
    proj(enc, L.kv, xn, 0, kv, 0, 1);
    norm(enc, kv, &L.kv_norm, kvn, 1, D, D, cfg_.rms_eps);
    rope(enc, q, 0, comp, D, H, p);
    rope(enc, kvn, 0, comp, D, 1, p);
    round(enc, kvn, 0, ring_[il].get(), uint64_t(p % W) * D * 4, D, kFp8);
    probe(enc, "attn_q", int(il), q, uint64_t(H) * D);

    // A KV source's compressor: a new compressed row (and index key) once its group of `ratio` tokens is complete.
    if (L.kv_src) {
        const uint32_t src = uint32_t(L.owner);
        MTL::Buffer* lat = scratch("ds_lat", D * 4);
        MTL::Buffer* latn = scratch("ds_latn", D * 4);
        bool emit = true;
        uint32_t row = p, rpos = p;
        if (L.ratio == 1) {
            proj(enc, L.comp_kv, xn, 0, lat, 0, 1);
        } else {
            MTL::Buffer* ck = scratch("ds_ck", D * 4);
            MTL::Buffer* cg = scratch("ds_cg", D * 4);
            proj(enc, L.comp_kv, xn, 0, ck, 0, 1);
            proj(enc, L.comp_gate, xn, 0, cg, 0, 1);
            emit = p % 2 == 1;
            row = p / 2;
            rpos = p - 1;
            const PoolArgs pa{D, emit ? 1u : 0u};
            dispatch(enc, "ds_pool2", {ck, cg, comp_state_[src].get(), lat}, &pa, sizeof(pa), MTL::Size(D, 1, 1),
                     MTL::Size(256, 1, 1));
        }
        if (emit) {
            norm(enc, lat, &L.comp_norm, latn, 1, D, D, cfg_.rms_eps);
            MTL::Buffer* ikr = scratch("ds_ikr", Di * 4);
            MTL::Buffer* ikn = scratch("ds_ikn", Di * 4);
            proj(enc, L.idx_k, latn, 0, ikr, 0, 1);
            norm(enc, ikr, &L.idx_k_norm, ikn, 1, Di, Di, cfg_.rms_eps);
            rope(enc, ikn, 0, true, Di, 1, rpos);
            round(enc, ikn, 0, ik_[src].get(), uint64_t(row) * Di * 4, Di, kFp4Pow2);
            rope(enc, latn, 0, true, D, 1, rpos);
            round(enc, latn, 0, ckv_[src].get(), uint64_t(row) * D * 2, D, kFp4E4m3, true);
        }
    }

    const uint32_t n_comp = comp ? (p + 1) / uint32_t(L.ratio) : 0;
    const bool sparse = n_comp > top;
    MTL::Buffer* sel = scratch("ds_sel", top * 4);
    if (L.idx_src && sparse) {
        MTL::Buffer* iq = scratch("ds_iq", uint64_t(Hi) * Di * 4);
        MTL::Buffer* iw = scratch("ds_iw", Hi * 4);
        MTL::Buffer* score = scratch("ds_score", uint64_t(opts_.max_ctx) * 4);
        proj(enc, L.idx_q_b, qrn, 0, iq, 0, 1);
        rope(enc, iq, 0, true, Di, Hi, p);
        round(enc, iq, 0, iq, 0, Hi * Di, kFp4Pow2);
        proj(enc, L.idx_proj, xn, 0, iw, 0, 1);
        const IdxArgs ia{Di, Hi, n_comp, 1.0f / std::sqrt(float(Di * Hi))};
        dispatch(enc, "ds_idx_score", {iq, iw, ik_[L.owner].get(), score}, &ia, sizeof(ia),
                 MTL::Size(uint64_t(n_comp) * 32, 1, 1), MTL::Size(256, 1, 1));
        // The candidate stage: the source layer keeps its best blocks of rows; later index sources score only
        // those (all blocks survive while there are at most candidate_top_k_blocks).
        const uint32_t bs = cfg_.candidate_block_size, nb = (n_comp + bs - 1) / bs;
        const int cand = cfg_.candidate_source_layer;
        MTL::Buffer* keep = scratch("ds_keep", uint64_t(opts_.max_ctx) / bs + 2);
        const BlockArgs ba{n_comp, bs};
        if (int(il) == cand && nb > cfg_.candidate_top_k_blocks) {
            MTL::Buffer* blk = scratch("ds_blk", (uint64_t(opts_.max_ctx) / bs + 2) * 4);
            MTL::Buffer* blk_ids = scratch("ds_blk_ids", uint64_t(cfg_.candidate_top_k_blocks) * 4);
            dispatch(enc, "ds_block_max", {score, blk}, &ba, sizeof(ba), MTL::Size(nb, 1, 1), MTL::Size(256, 1, 1));
            const TopkArgs ta{nb, cfg_.candidate_top_k_blocks, 1};
            dispatch(enc, "ds_topk", {blk, blk_ids, keep}, &ta, sizeof(ta), MTL::Size(1, 1, 1),
                     MTL::Size(1024, 1, 1), false);
        } else if (cand >= 0 && cand < int(il) && nb > cfg_.candidate_top_k_blocks) {
            dispatch(enc, "ds_block_mask", {score, keep}, &ba, sizeof(ba), MTL::Size(n_comp, 1, 1),
                     MTL::Size(256, 1, 1));
        }
        const TopkArgs ta{n_comp, top, 0};
        dispatch(enc, "ds_topk", {score, sel, keep}, &ta, sizeof(ta), MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1),
                 false);
    }

    const uint32_t nw = std::min(p + 1, W), nc = std::min(n_comp, top), N = nw + nc;
    MTL::Buffer* K = scratch("ds_K", uint64_t(W + top) * D * 4);
    MTL::Buffer* sc = scratch("ds_sc", uint64_t(H) * (W + top) * 4);
    MTL::Buffer* o = scratch("ds_o", uint64_t(H) * D * 4);
    const GatherArgs ga{D, nw, p + 1 - nw, W, nc, sparse ? 0u : 1u};
    dispatch(enc, "ds_gather_keys", {ring_[il].get(), comp ? ckv_[L.owner].get() : ring_[il].get(), sel, K}, &ga,
             sizeof(ga), MTL::Size(D, N, 1), MTL::Size(256, 1, 1));
    const ScoreArgs sa{D, H, N, 1.0f / std::sqrt(float(D))};
    dispatch(enc, "ds_attn_scores", {q, K, sc}, &sa, sizeof(sa), MTL::Size((N + 7) / 8, 1, 1), MTL::Size(256, 1, 1),
             false);
    enc->setComputePipelineState(pso("ds_attn_softmax"));
    enc->setBuffer(sc, 0, 0);
    enc->setBuffer(L.sinks.buf, L.sinks.off, 1);
    enc->setBytes(&sa, sizeof(sa), 2);
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "ds_attn_softmax");
    dispatch(enc, "ds_attn_out", {sc, K, o}, &sa, sizeof(sa), MTL::Size(D / 64, 1, 1), MTL::Size(256, 1, 1), false);
    rope(enc, o, 0, comp, D, H, p, true);
    round(enc, o, 0, o, 0, H * D, kFp8);
    MTL::Buffer* low = scratch("ds_low", uint64_t(cfg_.o_groups) * cfg_.o_lora_rank * 4);
    gemv(enc, L.out_a, o, 0, low, 0, cfg_.o_groups);
    proj(enc, L.out_b, low, 0, att, 0, 1);
}

const std::vector<float>& Engine::step(int32_t token)
{
    if (token < 0 || uint32_t(token) >= cfg_.n_vocab)
        throw std::runtime_error("token out of range");
    if (pos_ >= opts_.max_ctx)
        throw std::runtime_error("context full");
    apply_pressure();
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    step_start_ = Clock::now();
    encoding_ = true;
    const ExpertArena::Stats before = arena_->stats();
    const Stats stats_before = stats_;
    last_step_ = core::StepStats{};
    last_step_.pos = pos_;
    last_step_.token = token;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // Engram rows of this position (both tables), read before the GPU work that needs them.
    const uint32_t eg_per = engram_->layers() * engram_->columns();
    MTL::Buffer* eg = scratch("ds_eg", uint64_t(eg_per) * engram_->dim() * 4);
    {
        std::vector<int64_t> rows(eg_per);
        const int32_t t[1] = {token};
        engram_->hash(t, engram_history_, rows);
        engram_->read(rows, static_cast<float*>(eg->contents()), *engram_pool_);
        engram_history_ = engram_->advance(engram_history_, t);
    }

    MTL::Buffer* x = scratch("x", uint64_t(E) * 4);
    MTL::Buffer* xn = scratch("xn", uint64_t(E) * 4);
    MTL::Buffer* att = scratch("att", uint64_t(E) * 4);
    MTL::Buffer* ffo = scratch("ffo", uint64_t(E) * 4);
    MTL::Buffer* ekv = scratch("ds_ekv", uint64_t(5) * E * 4);
    scratch("R", uint64_t(hd) * 4);
    scratch("R2", uint64_t(hd) * 4);
    MTL::Buffer* pcs[2] = {scratch("pcA", 24 * 4), scratch("pcB", 24 * 4)};
    MTL::Buffer* pc0 = scratch("pc0", 24 * 4);
    {
        auto* p0 = static_cast<float*>(pc0->contents());
        std::fill_n(p0, 24, 0.0f);
        p0[0] = 1.0f; // the first sublayer's input is stream 0
        const gguf::Tensor& t = *token_embd_.tensor;
        const uint64_t rb = gguf::traits(t.type).block_bytes * (t.dims[0] / gguf::traits(t.type).block_elems);
        std::vector<std::byte> row(rb);
        gguf::read_exact(token_embd_.file->fd(), row.data(), rb, t.offset + uint64_t(token) * rb);
        quant::dequant_row(t.type, row.data(), static_cast<float*>(x->contents()), E);
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    cur_layer_ = cfg_.n_layer;
    MTL::ComputeCommandEncoder* enc = encoder(cb, cur_layer_, -1);
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, 1};
    dispatch(enc, "hc_init", {x, scratch("R", uint64_t(hd) * 4)}, &ha, sizeof(ha), MTL::Size(E, 1, 1),
             MTL::Size(256, 1, 1));
    MTL::Buffer* prev = pc0;
    int cur = 0;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        cur_layer_ = il;
        auto R = [&] { return scratch("R", uint64_t(hd) * 4); };
        auto R2 = [&] { return scratch("R2", uint64_t(hd) * 4); };
        if (L.engram >= 0) {
            gemv(enc, L.engram_kv, eg, uint64_t(L.engram) * engram_->columns() * engram_->dim() * 4, ekv, 0);
            const EngramArgs ea{E, cfg_.rms_eps};
            enc->setComputePipelineState(pso("ds_engram_gate"));
            enc->setBuffer(R(), 0, 0);
            enc->setBuffer(ekv, 0, 1);
            enc->setBuffer(L.engram_gate.buf, L.engram_gate.off, 2);
            enc->setBytes(&ea, sizeof(ea), 3);
            enc->dispatchThreadgroups(MTL::Size(4, 1, 1), MTL::Size(256, 1, 1));
            after(enc, "ds_engram_gate");
        }
        hc_lagged(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, R(), x, pcs[cur], prev);
        norm(enc, x, &L.attn_norm, xn, 1, E, E, cfg_.rms_eps);
        probe(enc, "attn_norm", int(il), xn, E);
        attention(enc, L, il, xn, att);
        probe(enc, "attn_out", int(il), att, E);
        ablate(enc, il, ablate_attn_, att);
        hc_post(enc, att, R(), R2(), pcs[cur], 1);
        std::swap(scratch_["R"], scratch_["R2"]);
        prev = pcs[cur];
        cur ^= 1;

        hc_lagged(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, R(), x, pcs[cur], prev);
        norm(enc, x, &L.ffn_norm, xn, 1, E, E, cfg_.rms_eps);
        probe(enc, "ffn_norm", int(il), xn, E);
        moe_decode(cb, enc, L.moe, il, xn, ffo, 1, [&](MTL::ComputeCommandEncoder*& e, MTL::Buffer* sho) {
            ablate(e, il, ablate_shexp_, sho);
        });
        probe(enc, "ffn_out", int(il), ffo, E);
        hc_post(enc, ffo, R(), R2(), pcs[cur], 1);
        std::swap(scratch_["R"], scratch_["R2"]);
        prev = pcs[cur];
        cur ^= 1;
        probe(enc, "l_out", int(il), R(), hd);
    }
    cur_layer_ = cfg_.n_layer;
    dispatch(enc, "hc_collapse", {scratch("R", uint64_t(hd) * 4), prev, x}, &ha, sizeof(ha), MTL::Size(E, 1, 1),
             MTL::Size(256, 1, 1));
    norm(enc, x, &output_norm_, xn, 1, E, E, cfg_.rms_eps);
    MTL::Buffer* lg = scratch("logits", uint64_t(cfg_.n_vocab) * 4);
    gemv(enc, output_, xn, 0, lg, 0);
    enc->endEncoding();
    finish(cb);
    std::memcpy(logits_.data(), lg->contents(), uint64_t(cfg_.n_vocab) * 4);
    ++pos_;
    pool->release();

    stats_.expert_read_ms = arena_->stats().read_ms;
    stats_.expert_bytes = arena_->stats().read_bytes;
    stats_.step_ms += ms_since(step_start_);
    const ExpertArena::Stats& after = arena_->stats();
    last_step_.wall_ms = ms_since(step_start_);
    last_step_.encode_ms = stats_.encode_ms - stats_before.encode_ms;
    last_step_.gpu_ms = stats_.gpu_exec_ms - stats_before.gpu_exec_ms;
    last_step_.read_ms = after.read_ms - before.read_ms;
    last_step_.hits = after.hits - before.hits;
    last_step_.misses = after.misses - before.misses;
    last_step_.read_bytes = after.read_bytes - before.read_bytes;
    return logits_;
}

} // namespace mako::dsv41
