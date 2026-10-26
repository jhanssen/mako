#include "dsv41/config.hpp"

#include "gguf/keys.hpp"

#include <cstring>
#include <format>

namespace mako::dsv41 {

namespace {

std::vector<int32_t> narrow(const std::vector<int64_t>& v)
{
    return {v.begin(), v.end()};
}

// An integer tensor (I32 or I64) of the model's files as int64 values; empty when absent.
std::vector<int64_t> int_tensor(const gguf::ModelFiles& m, const std::string& name)
{
    const auto ref = m.find(name);
    if (!ref || (ref.tensor->type != gguf::Type::I32 && ref.tensor->type != gguf::Type::I64))
        return {};
    const auto data = ref.file->data(*ref.tensor);
    std::vector<int64_t> out(ref.tensor->elements());
    for (size_t i = 0; i < out.size(); ++i) {
        if (ref.tensor->type == gguf::Type::I32) {
            int32_t v;
            std::memcpy(&v, data.data() + i * 4, 4);
            out[i] = v;
        } else {
            std::memcpy(&out[i], data.data() + i * 8, 8);
        }
    }
    return out;
}

} // namespace

std::expected<Config, std::string> Config::load(const gguf::ModelFiles& m)
{
    const gguf::File& f = m.meta();
    if (f.get_string("general.architecture").value_or("") != "deepseek41")
        return std::unexpected("not a deepseek41 model");
    gguf::Keys k(f, "deepseek41");
    Config c;
    c.n_layer = k.u32("block_count");
    c.n_embd = k.u32("embedding_length");
    c.n_head = k.u32("attention.head_count");
    c.head_dim = k.u32("attention.key_length");
    if (k.u32("attention.value_length") != c.head_dim)
        k.missing += " attention.value_length(!= key_length)";
    c.n_rot = k.u32("rope.dimension_count");
    c.q_lora_rank = k.u32("attention.q_lora_rank");
    c.o_lora_rank = k.u32("attention.output_lora_rank");
    c.o_groups = k.u32("attention.output_group_count");
    c.sliding_window = k.u32("attention.sliding_window");
    c.rms_eps = k.f32("attention.layer_norm_rms_epsilon");

    c.n_expert = k.u32("expert_count");
    c.n_expert_used = k.u32("expert_used_count");
    c.n_expert_shared = k.u32("expert_shared_count");
    c.n_ff_exp = k.u32("expert_feed_forward_length");
    c.gating = k.u32("expert_gating_func");
    c.expert_weights_scale = k.f32("expert_weights_scale", 1.0f);
    c.expert_weights_norm = k.flag("expert_weights_norm", false);
    for (double v : k.per_layer("swiglu_clamp_exp", c.n_layer))
        c.swiglu_clamp_exp.push_back(float(v));
    for (double v : k.per_layer("swiglu_clamp_shexp", c.n_layer))
        c.swiglu_clamp_shexp.push_back(float(v));

    c.rope_base = k.f32("rope.freq_base");
    c.compress_rope_base = k.f32("attention.compress_rope_freq_base");
    c.yarn_factor = k.f32("rope.scaling.factor", 1.0f);
    c.yarn_orig_ctx = k.u32("rope.scaling.original_context_length");
    c.yarn_beta_fast = k.f32("rope.scaling.yarn_beta_fast");
    c.yarn_beta_slow = k.f32("rope.scaling.yarn_beta_slow");

    c.compress_ratios = narrow(k.ints("attention.compress_ratios"));
    if (c.compress_ratios.size() < c.n_layer)
        k.missing += " attention.compress_ratios(shorter than block_count)";
    // Some files name these differently (the second name of each pair).
    c.kv_source_layers = narrow(k.ints(k.either("attention.kv_source_layers", "kv_source_layer_ids")));
    c.indexer_source_layers = narrow(k.ints(k.either("attention.indexer.source_layers", "index_source_layer_ids")));
    c.indexer_heads = k.u32("attention.indexer.head_count");
    c.indexer_dim = k.u32("attention.indexer.key_length");
    c.indexer_top_k = k.u32("attention.indexer.top_k");
    c.candidate_source_layer =
        int32_t(k.u32(k.either("attention.indexer.candidate_source_layer", "attention.candidate_source_layer_id")));
    c.candidate_top_k_blocks =
        k.u32(k.either("attention.indexer.candidate_top_k_blocks", "attention.candidate_topk_blocks"));
    c.candidate_block_size = k.u32(k.either("attention.indexer.candidate_block_size", "attention.candidate_block_size"));

    c.hc = k.u32("hyper_connection.count");
    c.hc_sinkhorn_iters = k.u32("hyper_connection.sinkhorn_iterations");
    c.hc_eps = k.f32("hyper_connection.epsilon");

    auto& e = c.engram;
    e.layers = narrow(k.ints(k.either("engram.layers", "engram.layer_ids")));
    e.num_embeddings = k.ints("engram.num_embeddings");
    e.max_ngram = k.u32(k.either("engram.max_ngram_size", "engram_max_ngram_size"));
    e.heads = k.u32(k.either("engram.head_count", "engram_n_heads"));
    e.head_dim = k.u32(k.either("engram.head_dim", "engram_head_dim"));
    e.pad_token = int32_t(k.u32(k.either("engram.pad_token_id", "engram_pad_token_id")));
    // The hash parameters: metadata arrays, or integer tensors of the same values.
    auto params = [&](const char* key, const char* tensor) {
        return k.has(key) ? k.ints(key) : int_tensor(m, tensor);
    };
    e.token_map = narrow(params("engram.token_map", "engram_token_map.weight"));
    e.multipliers = params("engram.multipliers", "engram_multipliers.weight");
    e.primes = params("engram.primes", "engram_primes.weight");
    e.offsets = params("engram.offsets", "engram_offsets.weight");
    if (e.token_map.empty())
        k.missing += " engram.token_map";
    const size_t nl = e.layers.size();
    if (e.multipliers.size() != nl * e.max_ngram || e.primes.size() != nl * e.columns() ||
        e.offsets.size() != nl * e.columns() || e.num_embeddings.size() != nl)
        k.missing += std::format(" engram(sizes: {} multipliers, {} primes, {} offsets for {} layers)",
                                 e.multipliers.size(), e.primes.size(), e.offsets.size(), nl);

    if (const auto* tokens = f.find("tokenizer.ggml.tokens"))
        if (const auto* a = std::get_if<gguf::Array>(tokens))
            c.n_vocab = uint32_t(a->items.size());
    if (!k.missing.empty())
        return std::unexpected("deepseek41 metadata missing or invalid:" + k.missing);
    return c;
}

} // namespace mako::dsv41
