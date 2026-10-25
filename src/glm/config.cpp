#include "glm/config.hpp"

#include "gguf/keys.hpp"

#include <format>
#include <variant>

namespace mako::glm {

std::expected<Config, std::string> Config::load(const gguf::ModelFiles& m)
{
    const gguf::File& f = m.meta();
    if (f.get_string("general.architecture").value_or("") != "glm5-next")
        return std::unexpected("not a glm5-next model");

    gguf::Keys k(f, "glm5-next");
    Config c;
    const uint32_t blocks = k.u32("block_count");
    c.n_layer_mtp = static_cast<uint32_t>(f.get_int("glm5-next.nextn_predict_layers").value_or(0));
    c.n_layer = blocks - c.n_layer_mtp;
    c.n_embd = k.u32("embedding_length");
    c.n_vocab = k.u32("vocab_size");
    c.n_head = k.u32("attention.head_count");
    c.n_ff = k.u32("feed_forward_length");
    c.n_ff_exp = k.u32("expert_feed_forward_length");
    c.n_expert = k.u32("expert_count");
    c.n_expert_used = k.u32("expert_used_count");
    c.n_expert_shared = k.u32("expert_shared_count");
    c.n_dense_lead = k.u32("leading_dense_block_count");
    c.expert_weights_scale = k.f32("expert_weights_scale", 1.0f);
    c.expert_weights_norm = k.flag("expert_weights_norm", false);
    for (double v : k.per_layer("swiglu_clamp_exp", blocks))
        c.swiglu_clamp_exp.push_back(float(v));
    if (f.find("glm5-next.swiglu_clamp_shexp")) {
        for (double v : k.per_layer("swiglu_clamp_shexp", blocks))
            c.swiglu_clamp_shexp.push_back(float(v));
    } else {
        c.swiglu_clamp_shexp = c.swiglu_clamp_exp;
    }

    c.rms_eps = k.f32("attention.layer_norm_rms_epsilon");
    c.norm_eps = k.f32("attention.layer_norm_epsilon");

    c.q_lora_rank = k.u32("attention.q_lora_rank");
    c.kv_lora_rank = k.u32("attention.kv_lora_rank");
    c.head_k = k.u32("attention.key_length_mla");
    c.head_v = k.u32("attention.value_length_mla");
    if (k.u32("rope.dimension_count") != 0)
        k.missing += " rope.dimension_count(must be 0)";

    c.kda_head_dim = k.u32("kda.head_dim");
    c.conv_kernel = k.u32("ssm.conv_kernel");
    c.kda_gate_lower_bound = k.f32("kda.gate_lower_bound");

    c.indexer_heads = k.u32("attention.indexer.head_count");
    c.indexer_dim = k.u32("attention.indexer.key_length");
    c.indexer_top_k = k.u32("attention.indexer.top_k");
    c.indexer_kpool = k.u32("attention.indexer.kpool");
    c.indexer_select_tail = k.flag("attention.indexer.kpool_select_tail", true);

    c.hc = k.u32("hyper_connection.count");
    c.hc_sinkhorn_iters = k.u32("hyper_connection.sinkhorn_iterations");
    c.hc_eps = k.f32("hyper_connection.epsilon");

    for (double kv : k.per_layer("attention.head_count_kv", blocks))
        c.is_kda.push_back(kv == 0.0);

    if (!k.missing.empty())
        return std::unexpected("glm5-next metadata missing or invalid:" + k.missing);
    if (c.hc != 4)
        return std::unexpected(std::format("hyper_connection.count {} is not supported", c.hc));
    if (f.find("glm5-next.attention.indexer.types"))
        return std::unexpected("per-layer indexer types are not supported yet");
    return c;
}

} // namespace mako::glm
