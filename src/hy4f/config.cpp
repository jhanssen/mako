#include "hy4f/config.hpp"

#include "gguf/keys.hpp"

#include <format>

namespace mako::hy4f {

std::expected<Config, std::string> Config::load(const gguf::ModelFiles& m)
{
    const gguf::File& f = m.meta();
    if (f.get_string("general.architecture").value_or("") != "hy_v4_flash")
        return std::unexpected("not a hy_v4_flash model");
    gguf::Keys k(f, "hy_v4_flash");
    Config c;
    c.n_layer = k.u32("block_count");
    c.n_embd = k.u32("embedding_length");
    c.n_ff = k.u32("feed_forward_length");
    c.n_dense_lead = k.u32("leading_dense_block_count");
    c.rms_eps = k.f32("attention.layer_norm_rms_epsilon");

    for (double v : k.per_layer("attention.head_count", c.n_layer))
        c.n_head.push_back(uint32_t(v));
    for (double v : k.per_layer("attention.head_count_kv", c.n_layer))
        if (v != 1.0)
            k.missing += " attention.head_count_kv(!= 1)";
    c.q_lora_rank = k.u32("attention.q_lora_rank");
    c.kv_lora_rank = k.u32("attention.kv_lora_rank");
    c.head_k = k.u32("attention.key_length_mla");
    c.head_v = k.u32("attention.value_length_mla");
    c.n_rot = k.u32("rope.dimension_count");
    c.rope_base = k.f32("rope.freq_base");
    c.rope_base_swa = k.f32("rope.freq_base_swa");
    if (c.n_rot == 0 || c.n_rot >= c.head_k)
        k.missing += std::format(" rope.dimension_count({} for a {}-wide head)", c.n_rot, c.head_k);
    // The sliding-window layers must not declare another latent geometry.
    const std::pair<const char*, uint32_t> swa_dims[] = {{"attention.key_length_mla_swa", c.head_k},
                                                          {"attention.value_length_mla_swa", c.head_v}};
    for (const auto& [key, dim] : swa_dims)
        if (k.has(key) && k.u32(key) != dim)
            k.missing += std::format(" {}(differs from the full-attention layers')", key);

    c.sliding_window = k.u32("attention.sliding_window");
    c.is_swa = k.bools("attention.sliding_window_pattern", c.n_layer);
    c.indexer_top_k = k.u32("attention.indexer.top_k");
    c.indexer_heads = k.u32("attention.indexer.head_count");
    c.indexer_dim = k.u32("attention.indexer.key_length");
    c.indexer_kpool = k.u32("attention.indexer.kpool");
    c.indexer_select_tail = k.flag("attention.indexer.kpool_select_tail", false);
    c.has_indexer = k.bools("attention.indexer.types", c.n_layer);
    if (c.indexer_kpool < 2 || c.indexer_top_k % c.indexer_kpool)
        k.missing += std::format(" attention.indexer.kpool({} for top_k {})", c.indexer_kpool, c.indexer_top_k);
    if (c.n_rot >= c.indexer_dim)
        k.missing += std::format(" attention.indexer.key_length({} with {} roped)", c.indexer_dim, c.n_rot);
    for (uint32_t il = 0; il < c.n_layer && il < c.is_swa.size(); ++il)
        if (c.is_swa[il] && c.has_indexer[il])
            k.missing += std::format(" attention.indexer.types(layer {} is a sliding-window layer)", il);

    c.n_expert = k.u32("expert_count");
    c.n_expert_used = k.u32("expert_used_count");
    c.n_expert_shared = k.u32("expert_shared_count");
    c.n_ff_exp = k.u32("expert_feed_forward_length");
    c.n_latent = k.has("expert_latent_length") ? k.u32("expert_latent_length") : c.n_embd;
    c.gating = k.u32("expert_gating_func");
    c.expert_weights_scale = k.f32("expert_weights_scale", 1.0f);
    c.expert_weights_norm = k.flag("expert_weights_norm", false);
    for (double v : k.per_layer("swiglu_clamp_exp", c.n_layer))
        c.swiglu_clamp_exp.push_back(float(v));

    c.hc = k.u32("hyper_connection.count");
    c.hc_eps = k.f32("hyper_connection.epsilon");
    c.hc_magnitude = k.f32("hyper_connection.magnitude");

    if (k.has("ple.layers"))
        c.engram = dsv41::EngramConfig::load(k, m);

    if (const auto* tokens = f.find("tokenizer.ggml.tokens"))
        if (const auto* a = std::get_if<gguf::Array>(tokens))
            c.n_vocab = uint32_t(a->items.size());
    if (!k.missing.empty())
        return std::unexpected("hy_v4_flash metadata missing or invalid:" + k.missing);
    return c;
}

} // namespace mako::hy4f
