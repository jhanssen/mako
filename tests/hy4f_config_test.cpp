// Hy4-flash's config from a Maya GGUF against its metadata (the values every Maya size shares). No GPU.
//   hy4f_config_test <model-00001.gguf>
#include "hy4f/config.hpp"

#include <filesystem>
#include <format>
#include <print>
#include <string>

namespace {

int failures = 0;

template <class A, class B>
void eq(const char* what, const A& got, const B& want)
{
    const bool ok = got == want;
    std::println("  {}  {} = {}", ok ? "ok  " : "FAIL", what, got);
    failures += !ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: hy4f_config_test <model-00001.gguf>");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m) {
        std::println(stderr, "cannot open the model files");
        return 1;
    }
    auto c = mako::hy4f::Config::load(*m);
    if (!c) {
        std::println(stderr, "{}", c.error());
        return 1;
    }
    eq("n_layer", c->n_layer, 42u);
    eq("n_embd", c->n_embd, 3072u);
    eq("n_vocab", c->n_vocab, 120832u);
    eq("n_ff", c->n_ff, 13824u);
    eq("n_dense_lead", c->n_dense_lead, 1u);
    eq("q_lora_rank", c->q_lora_rank, 1024u);
    eq("kv_lora_rank", c->kv_lora_rank, 512u);
    eq("head_k", c->head_k, 256u);
    eq("head_v", c->head_v, 256u);
    eq("n_rot", c->n_rot, 64u);
    eq("rope_base", c->rope_base, 1e7f);
    eq("rope_base_swa", c->rope_base_swa, 1e4f);
    eq("sliding_window", c->sliding_window, 512u);
    eq("n_expert", c->n_expert, 512u);
    eq("n_expert_used", c->n_expert_used, 16u);
    eq("n_expert_shared", c->n_expert_shared, 1u);
    eq("n_ff_exp", c->n_ff_exp, 1536u);
    eq("n_latent", c->n_latent, 1536u);
    eq("gating (2: sigmoid)", c->gating, 2u);
    eq("expert_weights_norm", c->expert_weights_norm, true);
    eq("hc", c->hc, 4u);
    eq("hc_magnitude", c->hc_magnitude, 2.0f);
    eq("indexer_heads", c->indexer_heads, 16u);
    eq("indexer_dim", c->indexer_dim, 128u);
    eq("indexer_top_k", c->indexer_top_k, 2048u);
    eq("indexer_kpool", c->indexer_kpool, 4u);
    eq("indexer_select_tail", c->indexer_select_tail, true);

    // Layers 0, 1 and every fourth from 5 are sparse (32 heads, an indexer each); the rest sliding-window (64 heads).
    std::string layout;
    bool consistent = true;
    for (uint32_t il = 0; il < c->n_layer; ++il) {
        const bool sparse = il < 2 || (il >= 5 && (il - 5) % 4 == 0);
        consistent = consistent && c->is_swa[il] == !sparse && c->has_indexer[il] == sparse &&
                     c->n_head[il] == (sparse ? 32u : 64u) && c->swiglu_clamp_exp[il] == 10.0f;
        layout += sparse ? 'S' : 'w';
    }
    std::println("  {}  layers {}", consistent ? "ok  " : "FAIL", layout);
    failures += !consistent;

    const auto& e = c->engram;
    eq("engram layers", std::format("{}", e.layers), std::string("[1, 11, 21, 31]"));
    eq("engram max_ngram", e.max_ngram, 3u);
    eq("engram heads", e.heads, 8u);
    eq("engram head_dim", e.head_dim, 96u);
    eq("engram pad_token", e.pad_token, 120002);
    eq("engram token_map", e.token_map.size(), size_t(120832));
    int64_t rows = 0;
    for (int64_t r : e.num_embeddings)
        rows += r;
    eq("engram rows (per_layer_token_embd)", rows, int64_t(660006416));

    std::println("{}", failures ? std::format("{} failures", failures) : std::string("all passed"));
    return failures ? 1 : 0;
}
