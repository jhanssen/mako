#pragma once

#include "gguf/model_files.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace mako::glm {

struct Config {
    uint32_t n_layer = 0;      // trunk layers, the MTP block excluded
    uint32_t n_layer_mtp = 0;
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    uint32_t n_head = 0;
    uint32_t n_ff = 0;         // dense FFN of the leading layers
    uint32_t n_ff_exp = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_shared = 0;
    uint32_t n_dense_lead = 0;
    float expert_weights_scale = 1.0f;
    bool expert_weights_norm = false;
    std::vector<float> swiglu_clamp_exp;   // per layer, MTP included
    std::vector<float> swiglu_clamp_shexp;

    float rms_eps = 0.0f;
    float norm_eps = 0.0f;

    // MLA (full attention) layers
    uint32_t q_lora_rank = 0;
    uint32_t kv_lora_rank = 0;
    uint32_t head_k = 0;
    uint32_t head_v = 0;

    // KDA (linear attention) layers
    uint32_t kda_head_dim = 0;
    uint32_t conv_kernel = 0;
    float kda_gate_lower_bound = 0.0f;

    // DSA indexer
    uint32_t indexer_heads = 0;
    uint32_t indexer_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t indexer_kpool = 0;
    bool indexer_select_tail = true;

    // mHC
    uint32_t hc = 0;
    uint32_t hc_sinkhorn_iters = 0;
    float hc_eps = 0.0f;

    std::vector<bool> is_kda; // per layer, MTP included

    uint32_t kda_inner() const { return kda_head_dim * n_head; }

    static std::expected<Config, std::string> load(const gguf::ModelFiles& m);
};

} // namespace mako::glm
