#include "image/anima.hpp"

#include "core/log.hpp"
#include "gguf/read_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>

namespace mako::image {

namespace {

namespace fs = std::filesystem;

struct LnAffineArgs {
    uint32_t n;
    float eps;
};
struct HeadsPartialArgs {
    uint32_t ld, col0, stride, dim, rot, n_pad;
    float eps;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

// The files' directory: dir, or its split_files/ as the model repository lays them out.
fs::path root_of(const std::string& dir)
{
    return fs::is_directory(fs::path(dir) / "split_files") ? fs::path(dir) / "split_files" : fs::path(dir);
}

// The one .safetensors file in d (a file named given wins).
std::string only_file(const fs::path& d, const std::string& given, const char* what)
{
    if (!given.empty())
        return given;
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(d, ec))
        if (e.path().extension() == ".safetensors")
            files.push_back(e.path());
    if (files.size() != 1) {
        std::string names;
        for (const auto& p : files)
            names += " " + p.filename().string();
        throw std::runtime_error(std::format("{}: {} {} files, name one{}", d.string(), files.size(), what,
                                             files.empty() ? "" : ":" + names));
    }
    return files[0].string();
}

// A tokenizer.json under tokenizers/<name> beside the files (or beside split_files/).
std::string tokenizer_file(const std::string& dir, const char* name, const char* repo)
{
    for (const fs::path& base : {fs::path(dir), root_of(dir), fs::path(dir).parent_path()}) {
        const fs::path p = base / "tokenizers" / name / "tokenizer.json";
        if (fs::exists(p))
            return p.string();
    }
    throw std::runtime_error(std::format("{}: no tokenizers/{}/tokenizer.json (hf download {} tokenizer.json "
                                         "--local-dir {}/tokenizers/{})",
                                         dir, name, repo, dir, name));
}

Engine::Parts anima_parts(const std::string& dir, const Options& opts)
{
    const fs::path root = root_of(dir);
    Engine::Parts p;
    p.text_encoder = only_file(root / "text_encoders", opts.text_encoder, "text encoder");
    p.vae = only_file(root / "vae", "", "VAE");
    p.tokenizer = tokenizer_file(dir, "qwen3", "Qwen/Qwen3-0.6B-Base");
    p.text_theta = 1e6f;
    return p;
}

} // namespace

bool Anima::detect(const std::string& dir)
{
    const fs::path root = root_of(dir);
    return fs::is_directory(root / "diffusion_models") && fs::is_directory(root / "text_encoders") &&
           fs::is_directory(root / "vae");
}

Anima::Anima(const std::string& dir, mtl::Context& ctx, const Options& opts)
    : Engine(anima_parts(dir, opts), ctx, opts),
      tfiles_(open_files(only_file(root_of(dir) / "diffusion_models", opts.transformer, "transformer")))
{
    dir_ = dir;
    const std::string tpath = only_file(root_of(dir) / "diffusion_models", opts.transformer, "transformer");
    // The distilled checkpoint by its file name, as it is published.
    std::string base = fs::path(tpath).filename().string();
    std::transform(base.begin(), base.end(), base.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    turbo_ = base.find("turbo") != std::string::npos;
    auto t5 = text::Unigram::load_json(tokenizer_file(dir, "t5", "google/flan-t5-xxl"));
    if (!t5)
        throw std::runtime_error(t5.error());
    t5_ = std::move(*t5);
    if (text_width() != kCtx)
        throw std::runtime_error("text encoder: not 1024 wide");
    load_transformer(tfiles_);
}

std::optional<std::string> Anima::default_negative() const
{
    // The model card's recommended negative prompt, without the score tags it advises against for Aesthetic.
    if (turbo_)
        return std::nullopt;
    return "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration";
}

void Anima::load_transformer(const safetensors::Files& f)
{
    const std::string p = f.find("model.diffusion_model.x_embedder.proj.1.weight") ? "model.diffusion_model." : "";
    if (!f.find(p + "x_embedder.proj.1.weight") || !f.find(p + "llm_adapter.embed.weight"))
        throw std::runtime_error("transformer: not Anima's weights");
    auto mat = [&](const std::string& n) { return upload(f, p + n, true); };
    auto vec = [&](const std::string& n) { return upload(f, p + n, false); };
    auto rows = [&](std::initializer_list<std::string> names) {
        std::vector<std::string> full;
        for (const auto& n : names)
            full.push_back(p + n);
        return upload_rows(f, full);
    };
    // The patch embedding without the padding mask's columns (always zero): [D][16 x 2 x 2] of [D][17 x 2 x 2].
    {
        const auto r = f.find(p + "x_embedder.proj.1.weight");
        const uint32_t cols = uint32_t(r.tensor->dims[0]), out = uint32_t(r.tensor->dims[1]);
        if (r.tensor->type != gguf::Type::BF16 || cols != 68 || out != kDim)
            throw std::runtime_error("transformer: the patch embedding is not BF16 [2048][68]");
        x_in_ = reserve(uint64_t(out) * 64 * 2, gguf::Type::BF16, {64, out});
        std::vector<uint16_t> row(cols);
        auto* dst = reinterpret_cast<uint16_t*>(static_cast<char*>(x_in_.buf->contents()) + x_in_.off);
        for (uint32_t o = 0; o < out; ++o) {
            gguf::read_exact(r.file->fd(), row.data(), cols * 2, r.tensor->offset + uint64_t(o) * cols * 2);
            std::memcpy(dst + uint64_t(o) * 64, row.data(), 64 * 2);
        }
    }
    t1_ = mat("t_embedder.1.linear_1.weight");
    t2_ = mat("t_embedder.1.linear_2.weight");
    t_norm_ = vec("t_embedding_norm.weight");
    std::vector<std::string> m1, m2;
    for (uint32_t i = 0; f.find(std::format("{}blocks.{}.mlp.layer1.weight", p, i)); ++i) {
        const std::string b = std::format("blocks.{}.", i);
        Block B;
        B.self.q = rows({b + "self_attn.q_proj.weight", b + "self_attn.k_proj.weight", b + "self_attn.v_proj.weight"});
        B.self.o = mat(b + "self_attn.output_proj.weight");
        B.self.norm_q = vec(b + "self_attn.q_norm.weight");
        B.self.norm_k = vec(b + "self_attn.k_norm.weight");
        B.cross.q = mat(b + "cross_attn.q_proj.weight");
        B.cross.k = rows({b + "cross_attn.k_proj.weight", b + "cross_attn.v_proj.weight"});
        B.cross.o = mat(b + "cross_attn.output_proj.weight");
        B.cross.norm_q = vec(b + "cross_attn.q_norm.weight");
        B.cross.norm_k = vec(b + "cross_attn.k_norm.weight");
        B.mlp1 = mat(b + "mlp.layer1.weight");
        B.mlp2 = mat(b + "mlp.layer2.weight");
        blocks_.push_back(std::move(B));
        for (const char* k : {"self_attn", "cross_attn", "mlp"}) {
            m1.push_back(std::format("{}{}adaln_modulation_{}.1.weight", p, b, k));
            m2.push_back(std::format("{}{}adaln_modulation_{}.2.weight", p, b, k));
        }
    }
    if (blocks_.empty())
        throw std::runtime_error("transformer: no blocks");
    m1.push_back(p + "final_layer.adaln_modulation.1.weight");
    m2.push_back(p + "final_layer.adaln_modulation.2.weight");
    mod1_ = upload_rows(f, m1);
    mod2_ = upload_rows(f, m2);
    out_ = mat("final_layer.linear.weight");
    embed_ref_ = f.find(p + "llm_adapter.embed.weight");
    if (embed_ref_.tensor->type != gguf::Type::BF16 || embed_ref_.tensor->dims[0] != kCtx)
        throw std::runtime_error("transformer: the adapter's embeddings are not BF16 x 1024");
    for (uint32_t i = 0; f.find(std::format("{}llm_adapter.blocks.{}.mlp.0.weight", p, i)); ++i) {
        const std::string b = std::format("llm_adapter.blocks.{}.", i);
        AdapterBlock A;
        A.norm_self = vec(b + "norm_self_attn.weight");
        A.norm_cross = vec(b + "norm_cross_attn.weight");
        A.norm_mlp = vec(b + "norm_mlp.weight");
        A.self.q = rows({b + "self_attn.q_proj.weight", b + "self_attn.k_proj.weight", b + "self_attn.v_proj.weight"});
        A.self.o = mat(b + "self_attn.o_proj.weight");
        A.self.norm_q = vec(b + "self_attn.q_norm.weight");
        A.self.norm_k = vec(b + "self_attn.k_norm.weight");
        A.cross.q = mat(b + "cross_attn.q_proj.weight");
        A.cross.k = rows({b + "cross_attn.k_proj.weight", b + "cross_attn.v_proj.weight"});
        A.cross.o = mat(b + "cross_attn.o_proj.weight");
        A.cross.norm_q = vec(b + "cross_attn.q_norm.weight");
        A.cross.norm_k = vec(b + "cross_attn.k_norm.weight");
        A.mlp1 = mat(b + "mlp.0.weight");
        A.mlp1_b = vec(b + "mlp.0.bias");
        A.mlp2 = mat(b + "mlp.2.weight");
        A.mlp2_b = vec(b + "mlp.2.bias");
        adapter_.push_back(std::move(A));
    }
    a_out_ = mat("llm_adapter.out_proj.weight");
    a_out_b_ = vec("llm_adapter.out_proj.bias");
    a_norm_ = vec("llm_adapter.norm.weight");
    log::info("Anima transformer: {} blocks, {} adapter blocks{}", blocks_.size(), adapter_.size(),
              turbo_ ? ", turbo" : "");
}

std::vector<std::pair<std::string, float>> Anima::weighted(const std::string& prompt)
{
    // Escaped parentheses out of the way, as two unused bytes each.
    std::string text;
    for (size_t i = 0; i < prompt.size(); ++i)
        if (prompt[i] == '\\' && i + 1 < prompt.size() && (prompt[i + 1] == '(' || prompt[i + 1] == ')')) {
            text += prompt[i + 1] == ')' ? std::string("\0\1", 2) : std::string("\0\2", 2);
            ++i;
        } else {
            text += prompt[i];
        }
    // Top-level groups: text, and parenthesized runs (an unclosed one runs to the end, unmatched ")" stay in text).
    auto groups = [](const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        int level = 0;
        for (char c : s) {
            if (c == '(') {
                if (level == 0) {
                    if (!cur.empty())
                        out.push_back(cur);
                    cur = "(";
                } else {
                    cur += c;
                }
                ++level;
            } else if (c == ')') {
                --level;
                if (level == 0) {
                    out.push_back(cur + ")");
                    cur.clear();
                } else {
                    cur += c;
                }
            } else {
                cur += c;
            }
        }
        if (!cur.empty())
            out.push_back(cur);
        return out;
    };
    std::vector<std::pair<std::string, float>> out;
    // Nesting deeper than kMaxDepth is text (each level copies the string, and the weights compound).
    constexpr int kMaxDepth = 16;
    auto parse = [&](auto& self, const std::string& s, float weight, int depth) -> void {
        for (std::string x : groups(s)) {
            if (x.size() >= 2 && x.front() == '(' && x.back() == ')' && depth < kMaxDepth) {
                x = x.substr(1, x.size() - 2);
                float w = weight * 1.1f;
                if (const size_t colon = x.rfind(':'); colon != std::string::npos && colon > 0) {
                    std::string num = x.substr(colon + 1);
                    const size_t a = num.find_first_not_of(" \t\n\r"), b = num.find_last_not_of(" \t\n\r");
                    num = a == std::string::npos ? "" : num.substr(a, b - a + 1);
                    char* end = nullptr;
                    const double v = num.empty() ? 0 : std::strtod(num.c_str(), &end);
                    if (!num.empty() && end && *end == '\0' && std::isfinite(v)) {
                        w = float(std::clamp(v, -100.0, 100.0));
                        x = x.substr(0, colon);
                    }
                }
                self(self, x, w, depth + 1);
            } else {
                out.emplace_back(x, weight);
            }
        }
    };
    parse(parse, text, 1.0f, 0);
    for (auto& [s, w] : out) {
        std::string u;
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i] == '\0' && i + 1 < s.size() && (s[i + 1] == '\1' || s[i + 1] == '\2')) {
                u += s[i + 1] == '\1' ? ')' : '(';
                ++i;
            } else {
                u += s[i];
            }
        s = u;
    }
    return out;
}

std::vector<int32_t> Anima::qwen_ids(const std::string& prompt) const
{
    std::vector<int32_t> ids;
    for (const auto& [s, w] : weighted(prompt)) {
        if (s.empty())
            continue;
        const auto part = tok_->encode(s, true);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    if (ids.empty())
        ids.push_back(kQwenPad);
    return ids;
}

std::pair<std::vector<int32_t>, std::vector<float>> Anima::t5_ids(const std::string& prompt) const
{
    std::vector<int32_t> ids;
    std::vector<float> weights;
    for (const auto& [s, w] : weighted(prompt)) {
        if (s.empty())
            continue;
        const auto part = t5_.encode(s);
        ids.insert(ids.end(), part.begin(), part.end());
        weights.insert(weights.end(), part.size(), w);
    }
    ids.push_back(t5_.eos());
    weights.push_back(1.0f);
    return {ids, weights};
}

std::vector<float> Anima::prompt_embeddings(const std::string& prompt)
{
    const auto q = qwen_ids(prompt);
    const auto states = encode_text(q, 0, {}, true);
    const auto [ids, weights] = t5_ids(prompt);
    std::vector<float> out{float(q.size()), float(ids.size())};
    out.insert(out.end(), states.begin(), states.end());
    for (int32_t id : ids)
        out.push_back(float(id));
    out.insert(out.end(), weights.begin(), weights.end());
    return out;
}

std::vector<float> Anima::sigmas(uint32_t steps, uint32_t, std::optional<float>) const
{
    // sigma(t) = 3 t / (1 + 2 t) at t = 1 / 1000 ... 1, taken every 1000 / steps from the top.
    constexpr uint32_t kLevels = 1000;
    std::vector<float> levels(kLevels);
    for (uint32_t i = 0; i < kLevels; ++i) {
        const float t = float(i + 1) / float(kLevels);
        levels[i] = 3.0f * t / (1.0f + 2.0f * t);
    }
    std::vector<float> s;
    const double every = double(kLevels) / double(steps);
    for (uint32_t x = 0; x < steps; ++x)
        s.push_back(levels[kLevels - 1 - uint32_t(double(x) * every)]);
    s.push_back(0.0f);
    return s;
}

void Anima::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    if (prompt.size() < 2)
        throw std::runtime_error("prompt embeddings: no header");
    const uint32_t S = uint32_t(prompt[0]), M = uint32_t(prompt[1]), E = kCtx;
    if (prompt.size() != 2 + uint64_t(S) * E + 2ull * M || !S || !M)
        throw std::runtime_error("prompt embeddings: sizes do not add up");
    gh_ = h;
    gw_ = w;
    const float* states = prompt.data() + 2;
    const float* idsf = states + uint64_t(S) * E;
    const float* weights = idsf + M;
    const uint32_t D = kAdapterHead, H = kAdapterHeads;
    const uint32_t mq_pad = round_up(M, kAttnRows), mk_pad = round_up(M, 128), sk_pad = round_up(S, 128);
    const uint32_t nff = uint32_t(adapter_.front().mlp1.dims[1]);
    MTL::Buffer* x = scratch("n_x", uint64_t(M) * E * 4);
    MTL::Buffer* xn = scratch("n_xn", uint64_t(M) * E * 4);
    MTL::Buffer* ctxb = scratch("n_ctx", uint64_t(S) * E * 4);
    MTL::Buffer* qkv = scratch("n_qkv", uint64_t(M) * 3 * E * 4);
    MTL::Buffer* kv = scratch("n_kv", uint64_t(S) * 2 * E * 4);
    MTL::Buffer* qh = scratch("n_qh", uint64_t(H) * mq_pad * D * 2);
    MTL::Buffer* kh = scratch("n_kh", uint64_t(H) * std::max(mk_pad, sk_pad) * D * 2);
    MTL::Buffer* vh = scratch("n_vh", uint64_t(H) * std::max(mk_pad, sk_pad) * D * 2);
    MTL::Buffer* att = scratch("n_att", uint64_t(M) * E * 4);
    MTL::Buffer* y = scratch("n_y", uint64_t(M) * E * 4);
    MTL::Buffer* ff = scratch("n_ff", uint64_t(M) * nff * 4);
    MTL::Buffer* cs_t = scratch("n_cst", uint64_t(M) * D / 2 * 8);
    MTL::Buffer* cs_c = scratch("n_csc", uint64_t(S) * D / 2 * 8);
    // The adapter's queries: the T5 tokens' embeddings (BF16 rows read from the file).
    {
        auto* xf = static_cast<float*>(x->contents());
        std::vector<uint16_t> row(E);
        for (uint32_t t = 0; t < M; ++t) {
            const uint64_t id = uint64_t(idsf[t]);
            if (id >= embed_ref_.tensor->dims[1])
                throw std::runtime_error("T5 token id out of range");
            gguf::read_exact(embed_ref_.file->fd(), row.data(), E * 2, embed_ref_.tensor->offset + id * E * 2);
            for (uint32_t i = 0; i < E; ++i) {
                const uint32_t bits = uint32_t(row[i]) << 16;
                std::memcpy(xf + uint64_t(t) * E + i, &bits, 4);
            }
        }
    }
    std::memcpy(ctxb->contents(), states, uint64_t(S) * E * 4);
    // Rotary angles of the queries' and the context's own positions (64-wide heads, base 10000, halves).
    auto angles = [&](MTL::Buffer* cs, uint32_t n) {
        auto* c = static_cast<float*>(cs->contents());
        for (uint32_t t = 0; t < n; ++t)
            for (uint32_t i = 0; i < D / 2; ++i) {
                const float inv = 1.0f / std::pow(10000.0f, float(2 * i) / float(D));
                const float a = float(t) * inv;
                c[(uint64_t(t) * D / 2 + i) * 2] = std::cos(a);
                c[(uint64_t(t) * D / 2 + i) * 2 + 1] = std::sin(a);
            }
    };
    angles(cs_t, M);
    angles(cs_c, S);

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    auto heads64 = [&](MTL::Buffer* src, uint32_t ld, uint32_t col0, uint32_t n, uint32_t pad, MTL::Buffer* cs,
                       const Tensor* norm, MTL::Buffer* dst) {
        const HeadsPartialArgs a{ld, col0, D, D, cs ? D : 0, pad, norm ? kEps : 0.0f};
        enc->setComputePipelineState(pso("heads_partial"));
        enc->setBuffer(src, 0, 0);
        enc->setBuffer(cs ? cs : src, 0, 1);
        enc->setBuffer(dst, 0, 2);
        enc->setBuffer(norm ? norm->buf : src, norm ? norm->off : 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(n, H, 1), MTL::Size(D, 1, 1));
        after(enc, "heads_partial");
        if (pad > n) {
            const uint32_t c[4] = {D, pad, n, 0};
            dispatch(enc, "heads_clear", {dst}, c, sizeof(c), MTL::Size(D, pad - n, H), MTL::Size(64, 1, 1));
        }
    };
    auto attend = [&](uint32_t nk, uint32_t k_pad) {
        const AttnArgs a{M, mq_pad, nk, k_pad, H, H, 0, 0, H * D, 1.0f / std::sqrt(float(D))};
        dispatch(enc, "flash_attn_sg_64_16x32_4", {qh, kh, vh, att}, &a, sizeof(a), MTL::Size(mq_pad / kAttnRows, H, 1),
                 MTL::Size(128, 1, 1), false);
    };
    for (const AdapterBlock& A : adapter_) {
        norm(enc, x, &A.norm_self, xn, M, E, E, kEps);
        proj(enc, A.self.q, xn, 0, qkv, 0, M);
        heads64(qkv, 3 * E, 0, M, mq_pad, cs_t, &A.self.norm_q, qh);
        heads64(qkv, 3 * E, E, M, mk_pad, cs_t, &A.self.norm_k, kh);
        heads64(qkv, 3 * E, 2 * E, M, mk_pad, nullptr, nullptr, vh);
        attend(M, mk_pad);
        proj(enc, A.self.o, att, 0, y, 0, M);
        add(enc, x, y, x, M * E);
        norm(enc, x, &A.norm_cross, xn, M, E, E, kEps);
        proj(enc, A.cross.q, xn, 0, qkv, 0, M);
        heads64(qkv, E, 0, M, mq_pad, cs_t, &A.cross.norm_q, qh);
        proj(enc, A.cross.k, ctxb, 0, kv, 0, S);
        heads64(kv, 2 * E, 0, S, sk_pad, cs_c, &A.cross.norm_k, kh);
        heads64(kv, 2 * E, E, S, sk_pad, nullptr, nullptr, vh);
        attend(S, sk_pad);
        proj(enc, A.cross.o, att, 0, y, 0, M);
        add(enc, x, y, x, M * E);
        norm(enc, x, &A.norm_mlp, xn, M, E, E, kEps);
        proj(enc, A.mlp1, xn, 0, ff, 0, M);
        bias(enc, ff, A.mlp1_b, nff, M);
        uint32_t n = M * nff;
        dispatch(enc, "gelu_erf", {ff}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        proj(enc, A.mlp2, ff, 0, y, 0, M);
        bias(enc, y, A.mlp2_b, E, M);
        add(enc, x, y, x, M * E);
    }
    proj(enc, a_out_, x, 0, y, 0, M);
    bias(enc, y, a_out_b_, E, M);
    norm(enc, y, &a_norm_, y, M, E, E, kEps);
    enc->endEncoding();
    finish(cb);

    // The context: the adapter's rows times their weights, zero rows up to 512.
    const uint32_t R = std::max(M, kMinRows), r_pad = round_up(R, 128);
    rows_[slot] = R;
    MTL::Buffer* ctx = scratch("n_full", uint64_t(R) * E * 4);
    {
        auto* c = static_cast<float*>(ctx->contents());
        const auto* o = static_cast<const float*>(y->contents());
        std::memset(c, 0, uint64_t(R) * E * 4);
        for (uint32_t t = 0; t < M; ++t)
            for (uint32_t i = 0; i < E; ++i)
                c[uint64_t(t) * E + i] = o[uint64_t(t) * E + i] * weights[t];
    }
    if (probe_)
        probe_("context", -1, static_cast<const float*>(ctx->contents()), uint64_t(R) * E);
    // Each block's keys and values of the context.
    MTL::Buffer* bkv = scratch("n_bkv", uint64_t(R) * 2 * kDim * 4);
    cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    enc = encoder(cb, 0, -1);
    for (uint32_t b = 0; b < blocks_.size(); ++b) {
        const Block& B = blocks_[b];
        MTL::Buffer* k = scratch(std::format("n_k{}_{}", slot, b), uint64_t(kHeads) * r_pad * kHeadDim * 2);
        MTL::Buffer* v = scratch(std::format("n_v{}_{}", slot, b), uint64_t(kHeads) * r_pad * kHeadDim * 2);
        proj(enc, B.cross.k, ctx, 0, bkv, 0, R);
        heads_norm(enc, bkv, 2 * kDim, 0, kHeads, R, r_pad, nullptr, 0, 0, B.cross.norm_k, kEps, 0, k);
        if (r_pad > R) {
            const uint32_t c[4] = {kHeadDim, r_pad, R, 0};
            dispatch(enc, "heads_clear", {k}, c, sizeof(c), MTL::Size(kHeadDim, r_pad - R, kHeads),
                     MTL::Size(128, 1, 1));
        }
        heads(enc, bkv, 2 * kDim, kDim, kHeads, R, r_pad, nullptr, 0, v);
    }
    enc->endEncoding();
    finish(cb);
}

std::vector<float> Anima::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    const uint32_t N = gh_ * gw_, D = kDim, R = rows_[slot], r_pad = round_up(R, 128), nb = uint32_t(blocks_.size());
    if (latents.size() != uint64_t(N) * kLatent)
        throw std::runtime_error("latents must be [h * w][64]");
    const uint32_t q_pad = round_up(N, kAttnRows), k_pad = round_up(N, 128);
    // The GEMMs run a multiple of 256 rows (the rest ignored), which keeps them on their largest tiles.
    const uint32_t Ng = N > 256 ? round_up(N, 256) : N;
    const uint32_t nff = uint32_t(blocks_.front().mlp1.dims[1]), nm = 3 * nb + 1;
    MTL::Buffer* x = scratch("d_x", uint64_t(Ng) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Ng) * D * 4);
    MTL::Buffer* qkv = scratch("d_qkv", uint64_t(Ng) * 3 * D * 4);
    MTL::Buffer* ff = scratch("d_ff", uint64_t(Ng) * nff * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(Ng) * D * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(Ng) * D * 4);
    MTL::Buffer* lat = scratch("d_lat", uint64_t(N) * kLatent * 4);
    MTL::Buffer* out = scratch("d_out", uint64_t(N) * kLatent * 4);
    MTL::Buffer* tp = scratch("d_tproj", D * 4);
    MTL::Buffer* t1 = scratch("d_t1", D * 4);
    MTL::Buffer* lora = scratch("d_lora", 3 * D * 4);
    MTL::Buffer* temb = scratch("d_temb", D * 4);
    MTL::Buffer* m1 = scratch("d_m1", uint64_t(nm) * kLora * 4);
    MTL::Buffer* mods = scratch("d_mods", uint64_t(nm) * 3 * D * 4);
    MTL::Buffer* cs = scratch("d_cs", uint64_t(N) * kHeadDim / 2 * 8);
    std::memcpy(lat->contents(), latents.data(), latents.size() * 4);
    // Sinusoidal embedding (cos half first) of the noise level itself, in float as the reference computes it.
    {
        auto* t = static_cast<float*>(tp->contents());
        for (uint32_t i = 0; i < D / 2; ++i) {
            const float e = std::exp(-std::log(10000.0f) * float(i) / float(D / 2)) * sigma;
            t[i] = std::cos(e);
            t[D / 2 + i] = std::sin(e);
        }
    }
    // Rotary angles of the image: of each head's 64 pairs, 22 for time (0 here), 21 for the row, 21 for the column;
    // bases 10000 x (4 for the image's axes)^(42 / 40).
    {
        auto* c = static_cast<float*>(cs->contents());
        const float hbase = float(10000.0 * std::pow(4.0, 42.0 / 40.0));
        float freq[21];
        for (uint32_t j = 0; j < 21; ++j)
            freq[j] = 1.0f / std::pow(hbase, float(2 * j) / 42.0f);
        for (uint32_t t = 0; t < N; ++t)
            for (uint32_t i = 0; i < kHeadDim / 2; ++i) {
                const float a = i < 22 ? 0.0f : i < 43 ? float(t / gw_) * freq[i - 22] : float(t % gw_) * freq[i - 43];
                c[(uint64_t(t) * kHeadDim / 2 + i) * 2] = std::cos(a);
                c[(uint64_t(t) * kHeadDim / 2 + i) * 2 + 1] = std::sin(a);
            }
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // The shared low-rank term from the embedding; the modulations from the normed embedding.
    proj(enc, t1_, tp, 0, t1, 0, 1);
    unary(enc, "silu", t1, D);
    proj(enc, t2_, t1, 0, lora, 0, 1);
    norm(enc, tp, &t_norm_, temb, 1, D, D, kEps);
    probe(enc, "t_emb", -1, temb, D);
    unary(enc, "silu", temb, D);
    proj(enc, mod1_, temb, 0, m1, 0, 1);
    for (uint32_t i = 0; i < nm; ++i) {
        const uint32_t n = i + 1 < nm ? 3 * D : 2 * D;
        const Tensor w{mod2_.buf, mod2_.off + uint64_t(i) * 3 * D * kLora * 2, mod2_.type, {kLora, n}};
        proj(enc, w, m1, uint64_t(i) * kLora * 4, mods, uint64_t(i) * 3 * D * 4, 1);
    }
    {
        const Tensor l{lora, 0, gguf::Type::F32, {3 * D}};
        bias(enc, mods, l, 3 * D, nm - 1);
        bias(enc, mods, l, 2 * D, 1, uint64_t(nm - 1) * 3 * D * 4);
    }
    proj(enc, x_in_, lat, 0, x, 0, N);
    probe(enc, "x_embed", -1, x, uint64_t(N) * D);

    // y = layer_norm(x) * (1 + scale) + shift with modulation m's (shift, scale).
    auto ln = [&](uint64_t m) {
        const LnAffineArgs a{D, kEps};
        enc->setComputePipelineState(pso("ln_affine"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(mods, (m + D) * 4, 1);
        enc->setBuffer(mods, m * 4, 2);
        enc->setBuffer(xn, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(N, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "ln_affine");
    };
    auto gated = [&](uint64_t m) {
        enc->setComputePipelineState(pso("scale_add"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(y, 0, 1);
        enc->setBuffer(mods, (m + 2 * D) * 4, 2);
        enc->setBytes(&D, sizeof(D), 3);
        enc->dispatchThreadgroups(MTL::Size((D + 255) / 256, N, 1), MTL::Size(256, 1, 1));
        after(enc, "scale_add");
    };
    for (uint32_t il = 0; il < nb; ++il) {
        const Block& B = blocks_[il];
        cur_layer_ = il;
        const uint64_t mo = uint64_t(il) * 3 * 3 * D;
        ln(mo);
        proj(enc, B.self.q, xn, 0, qkv, 0, Ng);
        heads_norm(enc, qkv, 3 * D, 0, kHeads, N, q_pad, cs, 0, 1, B.self.norm_q, kEps, 0, qh);
        heads_norm(enc, qkv, 3 * D, D, kHeads, N, k_pad, cs, 0, 1, B.self.norm_k, kEps, 0, kh);
        if (k_pad > N) {
            const uint32_t c[4] = {kHeadDim, k_pad, N, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - N, kHeads),
                     MTL::Size(128, 1, 1));
        }
        heads(enc, qkv, 3 * D, 2 * D, kHeads, N, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, N, q_pad, N, k_pad, kHeads, kHeads, 0, 0);
        proj(enc, B.self.o, att, 0, y, 0, Ng);
        gated(mo);
        ln(mo + 3 * D);
        proj(enc, B.cross.q, xn, 0, qkv, 0, Ng);
        heads_norm(enc, qkv, D, 0, kHeads, N, q_pad, nullptr, 0, 0, B.cross.norm_q, kEps, 0, qh);
        attention(enc, qh, scratch(std::format("n_k{}_{}", slot, il), uint64_t(kHeads) * r_pad * kHeadDim * 2),
                  scratch(std::format("n_v{}_{}", slot, il), uint64_t(kHeads) * r_pad * kHeadDim * 2), att, N, q_pad, R,
                  r_pad, kHeads, kHeads, 0, 0);
        proj(enc, B.cross.o, att, 0, y, 0, Ng);
        gated(mo + 3 * D);
        ln(mo + 6 * D);
        proj(enc, B.mlp1, xn, 0, ff, 0, Ng);
        uint32_t n = N * nff;
        dispatch(enc, "gelu_erf", {ff}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        proj(enc, B.mlp2, ff, 0, y, 0, Ng);
        gated(mo + 6 * D);
        probe(enc, "block", int(il), x, uint64_t(N) * D);
    }
    ln(uint64_t(nm - 1) * 3 * D);
    proj(enc, out_, xn, 0, out, 0, N);
    enc->endEncoding();
    finish(cb);
    // A token's outputs are (row, column, channel) of its 2x2 patch; the latents (channel, row, column).
    const auto* o = static_cast<const float*>(out->contents());
    std::vector<float> v(uint64_t(N) * kLatent);
    for (uint32_t t = 0; t < N; ++t)
        for (uint32_t c = 0; c < 16; ++c)
            for (uint32_t p = 0; p < 4; ++p)
                v[uint64_t(t) * kLatent + c * 4 + p] = o[uint64_t(t) * kLatent + p * 16 + c];
    return v;
}

Engine::LoraTarget Anima::lora_target(const std::string& module)
{
    unsigned i = 0;
    int used = 0;
    if (std::sscanf(module.c_str(), "blocks.%u.%n", &i, &used) != 1 || used <= 0 || i >= blocks_.size())
        return {};
    Block& B = blocks_[i];
    const std::string_view rest = std::string_view(module).substr(size_t(used));
    if (rest == "self_attn.q_proj")
        return {&B.self.q, 0, kDim};
    if (rest == "self_attn.k_proj")
        return {&B.self.q, kDim, kDim};
    if (rest == "self_attn.v_proj")
        return {&B.self.q, 2 * kDim, kDim};
    if (rest == "self_attn.output_proj")
        return {&B.self.o, 0, kDim};
    if (rest == "cross_attn.q_proj")
        return {&B.cross.q, 0, kDim};
    if (rest == "cross_attn.k_proj")
        return {&B.cross.k, 0, kDim};
    if (rest == "cross_attn.v_proj")
        return {&B.cross.k, kDim, kDim};
    if (rest == "cross_attn.output_proj")
        return {&B.cross.o, 0, kDim};
    if (rest == "mlp.layer1")
        return {&B.mlp1, 0, uint32_t(B.mlp1.dims[1])};
    if (rest == "mlp.layer2")
        return {&B.mlp2, 0, kDim};
    return {};
}

} // namespace mako::image
