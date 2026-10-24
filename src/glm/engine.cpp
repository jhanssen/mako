#include "glm/engine.hpp"

#include "mtl/embedded.hpp"
#include "quant/dequant.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>

namespace mako::glm {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr uint64_t kAlign = 256;
constexpr uint64_t kChunkBytes = 1ull << 30;

struct NormArgs {
    uint32_t n, wn;
    float eps;
};
struct HcArgs {
    uint32_t n_embd, iters;
    float eps;
};
struct ConvArgs {
    uint32_t channels, kernel_len;
};
struct GateArgs {
    uint32_t head_dim, heads;
    float lower_bound;
};
struct L2Args {
    uint32_t n;
    float eps;
};
struct ScanArgs {
    uint32_t head_dim;
};
struct AttnArgs {
    uint32_t dim, n_kv;
    float scale;
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
    gemv_ = std::make_unique<mtl::Gemv>(ctx_);
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

    std::vector<ExpertArena::LayerParts> parts;
    for (const auto& L : layers_)
        parts.push_back(L->experts);
    arena_plan_.mem = query_memory(ctx_.info().recommended_working_set);
    arena_plan_.budget = opts_.arena_gb >= 0 ? opts_.arena_gb * 1e9 : arena_budget(arena_plan_.mem);
    std::vector<std::vector<uint64_t>> counts;
    if (!opts_.expert_counts.empty()) {
        std::ifstream in(opts_.expert_counts);
        if (!in)
            throw std::runtime_error("cannot open " + opts_.expert_counts);
        counts.resize(cfg_.n_layer);
        for (std::string line; std::getline(in, line);) {
            std::istringstream ls(line);
            uint32_t layer;
            if (!(ls >> layer) || layer >= cfg_.n_layer)
                continue;
            for (uint64_t c; ls >> c;)
                counts[layer].push_back(c);
            if (counts[layer].size() != cfg_.n_expert)
                throw std::runtime_error(std::format("{}: layer {} has {} counts, expected {}", opts_.expert_counts,
                                                     layer, counts[layer].size(), cfg_.n_expert));
        }
    }
    arena_ = std::make_unique<ExpertArena>(ctx_.device(), ctx_.info().max_buffer, std::move(parts), cfg_.n_expert,
                                           cfg_.n_expert_used, arena_plan_.budget, counts);
    arena_->prefill();

    // The first compute pass that reads an arena buffer pays a driver cost that grows with its size (~0.5 s for
    // 75 GB); pay it here instead of in the first step.
    const auto t0 = Clock::now();
    MTL::Buffer* sink = scratch("warm", 16);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    const uint32_t one = 1;
    for (const auto& b : arena_->buffers()) {
        enc->setComputePipelineState(pso("copy_f32"));
        enc->setBuffer(b.get(), 0, 0);
        enc->setBuffer(sink, 0, 1);
        enc->setBytes(&one, sizeof(one), 2);
        enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    arena_plan_.warm_ms = ms_since(t0);
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

void Engine::gemv(MTL::ComputeCommandEncoder* enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
                  uint64_t y_off, uint32_t batch)
{
    mtl::Gemv::Args a;
    a.type = w.type;
    a.k = uint32_t(w.dims[0]);
    a.n = uint32_t(w.dims[1]);
    a.batch = batch;
    gemv_->encode(enc, a, w.buf, w.off, x, x_off, y, y_off);
}

void Engine::dispatch(MTL::ComputeCommandEncoder* enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                      const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads)
{
    enc->setComputePipelineState(pso(fn));
    NS::UInteger i = 0;
    for (MTL::Buffer* b : bufs)
        enc->setBuffer(b, 0, i++);
    if (args)
        enc->setBytes(args, args_size, i);
    if (threads)
        enc->dispatchThreads(grid, tg);
    else
        enc->dispatchThreadgroups(grid, tg);
}

void Engine::norm(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, const DevTensor* w, MTL::Buffer* y, uint32_t rows,
                  uint32_t n, uint32_t wn, float eps)
{
    const NormArgs a{n, w ? wn : 0, eps};
    enc->setComputePipelineState(pso("rms_norm"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(w ? w->buf : x, w ? w->off : 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBytes(&a, sizeof(a), 3);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
}

// x = mHC pre-mix of the streams R, with post/comb kept for hc_post.
void Engine::hc_pre(MTL::ComputeCommandEncoder* enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base)
{
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    MTL::Buffer* R = scratch("R", hd * 4);
    MTL::Buffer* flat = scratch("flat", hd * 4);
    MTL::Buffer* mixes = scratch("mixes", 32 * 4);
    norm(enc, R, nullptr, flat, 1, hd, 0, cfg_.rms_eps);
    gemv(enc, fn, flat, 0, mixes, 0);
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps};
    enc->setComputePipelineState(pso("hc_mix"));
    enc->setBuffer(mixes, 0, 0);
    enc->setBuffer(scale.buf, scale.off, 1);
    enc->setBuffer(base.buf, base.off, 2);
    enc->setBuffer(scratch("pre", 16), 0, 3);
    enc->setBuffer(scratch("post", 16), 0, 4);
    enc->setBuffer(scratch("comb", 64), 0, 5);
    enc->setBytes(&ha, sizeof(ha), 6);
    enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    dispatch(enc, "hc_collapse", {R, scratch("pre", 16), scratch("x", E * 4)}, &ha, sizeof(ha), MTL::Size(E, 1, 1),
             MTL::Size(256, 1, 1));
}

// R = mHC post-mix of x into the streams R.
void Engine::hc_post(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x)
{
    const uint32_t E = cfg_.n_embd, hd = cfg_.hc * E;
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps};
    MTL::Buffer* R = scratch("R", hd * 4);
    MTL::Buffer* R2 = scratch("R2", hd * 4);
    dispatch(enc, "hc_expand", {x, R, scratch("post", 16), scratch("comb", 64), R2}, &ha, sizeof(ha), MTL::Size(E, 1, 1),
             MTL::Size(256, 1, 1));
    std::swap(scratch_["R"], scratch_["R2"]);
}

void Engine::kda(MTL::ComputeCommandEncoder* enc, const Layer& L, uint32_t il)
{
    const uint32_t inner = cfg_.kda_inner(), D = cfg_.kda_head_dim, H = cfg_.n_head;
    MTL::Buffer* xn = scratch("xn", cfg_.n_embd * 4);
    MTL::Buffer* qkv = scratch("qkv", 3 * inner * 4);
    MTL::Buffer* qkvc = scratch("qkvc", 3 * inner * 4);
    gemv(enc, L.wq, xn, 0, qkv, 0);
    gemv(enc, L.wk, xn, 0, qkv, uint64_t(inner) * 4);
    gemv(enc, L.wv, xn, 0, qkv, uint64_t(inner) * 8);
    const ConvArgs ca{inner, cfg_.conv_kernel};
    enc->setComputePipelineState(pso("kda_conv"));
    enc->setBuffer(qkv, 0, 0);
    enc->setBuffer(L.conv_q.buf, L.conv_q.off, 1);
    enc->setBuffer(L.conv_k.buf, L.conv_k.off, 2);
    enc->setBuffer(L.conv_v.buf, L.conv_v.off, 3);
    enc->setBuffer(conv_state_[il].get(), 0, 4);
    enc->setBuffer(qkvc, 0, 5);
    enc->setBytes(&ca, sizeof(ca), 6);
    enc->dispatchThreads(MTL::Size(3 * inner, 1, 1), MTL::Size(256, 1, 1));
    probe(enc, "kda_q_conv", int(il), qkvc, inner);
    probe(enc, "kda_v_conv", int(il), qkvc, inner, uint64_t(inner) * 8);

    MTL::Buffer* fa = scratch("fa", D * 4);
    MTL::Buffer* graw = scratch("graw", inner * 4);
    MTL::Buffer* braw = scratch("braw", H * 4);
    MTL::Buffer* g = scratch("g", inner * 4);
    MTL::Buffer* beta = scratch("beta", H * 4);
    gemv(enc, L.f_a, xn, 0, fa, 0);
    gemv(enc, L.f_b, fa, 0, graw, 0);
    gemv(enc, L.beta, xn, 0, braw, 0);
    const GateArgs ga{D, H, cfg_.kda_gate_lower_bound};
    enc->setComputePipelineState(pso("kda_gate"));
    enc->setBuffer(graw, 0, 0);
    enc->setBuffer(L.dt_b.buf, L.dt_b.off, 1);
    enc->setBuffer(L.ssm_a.buf, L.ssm_a.off, 2);
    enc->setBuffer(braw, 0, 3);
    enc->setBuffer(g, 0, 4);
    enc->setBuffer(beta, 0, 5);
    enc->setBytes(&ga, sizeof(ga), 6);
    enc->dispatchThreads(MTL::Size(inner, 1, 1), MTL::Size(256, 1, 1));

    const L2Args la{D, 1e-6f};
    enc->setComputePipelineState(pso("l2_norm"));
    enc->setBuffer(qkvc, 0, 0);
    enc->setBytes(&la, sizeof(la), 1);
    enc->dispatchThreadgroups(MTL::Size(2 * H, 1, 1), MTL::Size(256, 1, 1));

    MTL::Buffer* o = scratch("o", inner * 4);
    const ScanArgs sa{D};
    enc->setComputePipelineState(pso("kda_scan"));
    enc->setBuffer(qkvc, 0, 0);
    enc->setBuffer(qkvc, uint64_t(inner) * 4, 1);
    enc->setBuffer(qkvc, uint64_t(inner) * 8, 2);
    enc->setBuffer(g, 0, 3);
    enc->setBuffer(beta, 0, 4);
    enc->setBuffer(kda_state_[il].get(), 0, 5);
    enc->setBuffer(o, 0, 6);
    enc->setBytes(&sa, sizeof(sa), 7);
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(D, 1, 1));
    probe(enc, "kda_g1", int(il), g, inner);
    probe(enc, "kda_beta", int(il), beta, H);
    probe(enc, "kda_scan_out", int(il), o, inner);

    MTL::Buffer* gab = scratch("ga", D * 4);
    MTL::Buffer* g2 = scratch("g2", inner * 4);
    MTL::Buffer* gated = scratch("gated", inner * 4);
    gemv(enc, L.g_a, xn, 0, gab, 0);
    gemv(enc, L.g_b, gab, 0, g2, 0);
    probe(enc, "kda_g2", int(il), g2, inner);
    const NormArgs na{D, D, cfg_.rms_eps};
    enc->setComputePipelineState(pso("rms_norm_gate"));
    enc->setBuffer(o, 0, 0);
    enc->setBuffer(L.o_norm.buf, L.o_norm.off, 1);
    enc->setBuffer(g2, 0, 2);
    enc->setBuffer(gated, 0, 3);
    enc->setBytes(&na, sizeof(na), 4);
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(256, 1, 1));
    probe(enc, "attn_output", int(il), gated, inner);
    gemv(enc, L.wo, gated, 0, scratch("att", cfg_.n_embd * 4), 0);
}

void Engine::mla(MTL::ComputeCommandEncoder* enc, const Layer& L, uint32_t il)
{
    const uint32_t H = cfg_.n_head, R = cfg_.kv_lora_rank;
    MTL::Buffer* xn = scratch("xn", cfg_.n_embd * 4);
    MTL::Buffer* qr = scratch("qr", cfg_.q_lora_rank * 4);
    MTL::Buffer* qrn = scratch("qrn", cfg_.q_lora_rank * 4);
    MTL::Buffer* q = scratch("q", H * cfg_.head_k * 4);
    MTL::Buffer* kv = scratch("kv", R * 4);
    MTL::Buffer* kvn = scratch("kvn", R * 4);
    MTL::Buffer* qabs = scratch("qabs", H * R * 4);
    MTL::Buffer* lat = scratch("lat", H * R * 4);
    MTL::Buffer* vout = scratch("vout", H * cfg_.head_v * 4);
    MTL::Buffer* scores = scratch("scores", uint64_t(H) * opts_.max_ctx * 4);

    gemv(enc, L.q_a, xn, 0, qr, 0);
    norm(enc, qr, &L.q_a_norm, qrn, 1, cfg_.q_lora_rank, cfg_.q_lora_rank, cfg_.rms_eps);
    gemv(enc, L.q_b, qrn, 0, q, 0);
    gemv(enc, L.kv_a, xn, 0, kv, 0);
    norm(enc, kv, &L.kv_a_norm, kvn, 1, R, R, cfg_.rms_eps);

    const uint32_t n = R;
    enc->setComputePipelineState(pso("store_half"));
    enc->setBuffer(kvn, 0, 0);
    enc->setBuffer(kv_cache_[il].get(), uint64_t(pos_) * R * 2, 1);
    enc->setBytes(&n, sizeof(n), 2);
    enc->dispatchThreads(MTL::Size(R, 1, 1), MTL::Size(256, 1, 1));

    gemv(enc, L.k_b, q, 0, qabs, 0, H);
    const bool sparse = indexer(enc, L, il);
    const AttnArgs aa{R, pos_ + 1, 1.0f / std::sqrt(float(cfg_.head_k))};
    enc->setComputePipelineState(pso(sparse ? "mla_attn_masked" : "mla_attn"));
    enc->setBuffer(qabs, 0, 0);
    enc->setBuffer(kv_cache_[il].get(), 0, 1);
    enc->setBuffer(lat, 0, 2);
    enc->setBuffer(scores, 0, 3);
    if (sparse)
        enc->setBuffer(scratch("idx_mask", opts_.max_ctx), 0, 4);
    enc->setBytes(&aa, sizeof(aa), sparse ? 5 : 4);
    enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(256, 1, 1));
    gemv(enc, L.v_b, lat, 0, vout, 0, H);
    gemv(enc, L.wo, vout, 0, scratch("att", cfg_.n_embd * 4), 0);
}

// Updates the layer's indexer cache with this token and, once more pools are complete than the selection
// keeps, writes the cell mask of the top pools plus the incomplete tail. Returns whether the mask applies.
bool Engine::indexer(MTL::ComputeCommandEncoder* enc, const Layer& L, uint32_t il)
{
    const uint32_t Di = cfg_.indexer_dim, Hi = cfg_.indexer_heads, kp = cfg_.indexer_kpool;
    MTL::Buffer* xn = scratch("xn", cfg_.n_embd * 4);
    MTL::Buffer* qrn = scratch("qrn", cfg_.q_lora_rank * 4);
    MTL::Buffer* iq = scratch("iq", Hi * Di * 4);
    MTL::Buffer* ikr = scratch("ikr", Di * 4);
    MTL::Buffer* iw = scratch("iw", Hi * 4);
    MTL::Buffer* cache = idx_cache_[il].get();
    MTL::Buffer* pooled = idx_pooled_[il].get();
    const uint64_t row = uint64_t(pos_) * 2 * Di * 4;

    gemv(enc, L.idx_q_b, qrn, 0, iq, 0);
    gemv(enc, L.idx_k, xn, 0, ikr, 0);
    const NormArgs na{Di, Di, cfg_.norm_eps};
    enc->setComputePipelineState(pso("layer_norm"));
    enc->setBuffer(ikr, 0, 0);
    enc->setBuffer(L.idx_k_norm.buf, L.idx_k_norm.off, 1);
    enc->setBuffer(L.idx_k_norm_b.buf, L.idx_k_norm_b.off, 2);
    enc->setBuffer(cache, row, 3);
    enc->setBytes(&na, sizeof(na), 4);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
    gemv(enc, L.idx_gate, xn, 0, cache, row + Di * 4);
    gemv(enc, L.idx_proj, xn, 0, iw, 0);

    if ((pos_ + 1) % kp == 0) {
        const PoolArgs pa{Di, kp, pos_ / kp};
        enc->setComputePipelineState(pso("idx_pool"));
        enc->setBuffer(cache, 0, 0);
        enc->setBuffer(L.idx_ape.buf, L.idx_ape.off, 1);
        enc->setBuffer(pooled, 0, 2);
        enc->setBytes(&pa, sizeof(pa), 3);
        enc->dispatchThreads(MTL::Size(Di, 1, 1), MTL::Size(Di, 1, 1));
    }

    const uint32_t n_pool = (pos_ + 1) / kp, n_top = cfg_.indexer_top_k / kp;
    if (n_pool <= n_top)
        return false;
    MTL::Buffer* score = scratch("idx_score", uint64_t(opts_.max_ctx / kp + 1) * 4);
    const ScoreArgs sa{Di, Hi, n_pool, 1.0f / std::sqrt(float(Di * Hi))};
    dispatch(enc, "idx_score", {iq, iw, pooled, score}, &sa, sizeof(sa), MTL::Size(n_pool, 1, 1), MTL::Size(64, 1, 1));
    probe(enc, "indexer_score", int(il), score, n_pool);
    const SelectArgs sel{n_pool, n_top, kp, pos_ + 1};
    dispatch(enc, "idx_select", {score, scratch("idx_mask", opts_.max_ctx)}, &sel, sizeof(sel), MTL::Size(1, 1, 1),
             MTL::Size(256, 1, 1), false);
    return true;
}

void Engine::ffn_dense(MTL::ComputeCommandEncoder* enc, const DevTensor& gate, const DevTensor& up,
                       const DevTensor& down, MTL::Buffer* out, float limit)
{
    const uint32_t nff = uint32_t(gate.dims[1]);
    MTL::Buffer* xn = scratch("xn", cfg_.n_embd * 4);
    MTL::Buffer* g = scratch("ffg", nff * 4);
    MTL::Buffer* u = scratch("ffu", nff * 4);
    MTL::Buffer* m = scratch("ffm", nff * 4);
    gemv(enc, gate, xn, 0, g, 0);
    gemv(enc, up, xn, 0, u, 0);
    const SwigluArgs sa{nff, limit > 1e-6f ? limit : 0.0f};
    dispatch(enc, "swiglu", {g, u, m}, &sa, sizeof(sa), MTL::Size(nff, 1, 1), MTL::Size(256, 1, 1));
    gemv(enc, down, m, 0, out, 0);
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
        std::array<int32_t, 16> slots{};
        arena_->ensure(il, std::span<const int>(sel, k), slots.data());
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
    enc = cb->computeCommandEncoder();
}

void Engine::probe(MTL::ComputeCommandEncoder* enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
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
    const HcArgs ha{E, cfg_.hc_sinkhorn_iters, cfg_.hc_eps};

    step_start_ = Clock::now();
    encoding_ = true;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Buffer* x = scratch("x", E * 4);
    MTL::Buffer* xn = scratch("xn", E * 4);
    MTL::Buffer* att = scratch("att", E * 4);
    MTL::Buffer* ffo = scratch("ffo", E * 4);
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
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    dispatch(enc, "hc_init", {x, scratch("R", hd * 4)}, &ha, sizeof(ha), MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));

    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const Layer& L = *layers_[il];
        if (stream_override_) {
            submit(cb, enc);
            stream_override_(int(il), static_cast<float*>(scratch("R", hd * 4)->contents()), hd);
        }
        hc_pre(enc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base);
        probe(enc, "hc_attn_pre", int(il), x, E);
        norm(enc, x, &L.attn_norm, xn, 1, E, E, cfg_.rms_eps);
        probe(enc, "attn_norm", int(il), xn, E);
        if (L.kda)
            kda(enc, L, il);
        else
            mla(enc, L, il);
        probe(enc, L.kda ? "kda_out" : "attn_out", int(il), att, E);
        hc_post(enc, att);

        hc_pre(enc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base);
        probe(enc, "hc_ffn_pre", int(il), x, E);
        probe(enc, "hc_mixes", int(il), scratch("mixes", 32 * 4), 24);
        norm(enc, x, &L.ffn_norm, xn, 1, E, E, cfg_.rms_eps);
        probe(enc, "ffn_norm", int(il), xn, E);
        if (L.dense) {
            ffn_dense(enc, L.ffn_gate, L.ffn_up, L.ffn_down, ffo, cfg_.swiglu_clamp_shexp[il]);
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
            ffn_dense(enc, L.sh_gate, L.sh_up, L.sh_down, sho, cfg_.swiglu_clamp_shexp[il]);
            const RouteArgs ra{cfg_.n_expert, k, cfg_.expert_weights_norm ? 1u : 0u, cfg_.expert_weights_scale};
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
            enc->endEncoding();
            const uint64_t v = ++event_value_;
            cb->encodeSignalEvent(routed_.get(), v);
            pending_moe_.emplace_back(il, v);
            enc = cb->computeCommandEncoder();

            // Resident experts run while the CPU reads the misses; the miss pass waits for it.
            const SwigluArgs sa{k * nff, cfg_.swiglu_clamp_exp[il] > 1e-6f ? cfg_.swiglu_clamp_exp[il] : 0.0f};
            auto pass = [&](MTL::Buffer* ids, bool miss, const char* tag) {
                MTL::Buffer* eg = scratch(std::string("eg") + tag, k * nff * 4);
                MTL::Buffer* eu = scratch(std::string("eu") + tag, k * nff * 4);
                MTL::Buffer* em = scratch(std::string("em") + tag, k * nff * 4);
                MTL::Buffer* ed = scratch(std::string("ed") + tag, k * E * 4);
                auto run = [&](int part, MTL::Buffer* in, MTL::Buffer* out) {
                    const ExpertArena::Region& r = arena_->region(il, part);
                    gemv_->encode_indexed(enc, part_args(part), r.buf, r.off, r.stride, ids, 0, in, 0, out, 0,
                                          miss ? indirect : nullptr, ind_off + uint64_t(part) * 12);
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
            enc = cb->computeCommandEncoder();
            MTL::Buffer* ed_miss = pass(miss_ids, true, "m");
            const CombineArgs ca{E, k};
            dispatch(enc, "moe_combine_split", {ed_hit, ed_miss, miss_rank, ew, sho, ffo}, &ca, sizeof(ca),
                     MTL::Size(E, 1, 1), MTL::Size(256, 1, 1));
        }
        probe(enc, "ffn_out", int(il), ffo, E);
        hc_post(enc, ffo);
        probe(enc, "l_out", int(il), scratch("R", hd * 4), hd);
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
    return logits_;
}

} // namespace mako::glm
