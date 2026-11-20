#pragma once

#include "dsv41/config.hpp"
#include "gguf/model_files.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace mako::hy4f {

// Hy4-flash's geometry from a GGUF's hy_v4_flash.* metadata.
struct Config {
    uint32_t n_layer = 0; // trunk layers
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ff = 0;    // dense FFN of the leading layers
    uint32_t n_dense_lead = 0;
    float rms_eps = 0.0f;

    // MLA on every layer, one latent row (kv_lora_rank + n_rot) per token; only the head count differs by layer type.
    std::vector<uint32_t> n_head; // per layer
    uint32_t q_lora_rank = 0, kv_lora_rank = 0;
    uint32_t head_k = 0; // qk head: (head_k - n_rot) nope dims, then n_rot roped
    uint32_t head_v = 0;
    uint32_t n_rot = 0;
    float rope_base = 0.0f, rope_base_swa = 0.0f;

    uint32_t sliding_window = 0;
    std::vector<bool> is_swa;      // per layer
    std::vector<bool> has_indexer; // per layer; the other full-attention layers attend everything
    uint32_t indexer_heads = 0, indexer_dim = 0, indexer_top_k = 0;
    uint32_t indexer_kpool = 0; // keys pooled over groups of this many tokens
    bool indexer_select_tail = false;

    // Routed experts run at n_latent (projected from and back to n_embd); the shared expert at n_embd.
    uint32_t n_expert = 0, n_expert_used = 0, n_expert_shared = 0;
    uint32_t n_ff_exp = 0;
    uint32_t n_latent = 0;
    uint32_t gating = 0; // expert_gating_func
    float expert_weights_scale = 1.0f;
    bool expert_weights_norm = false;
    std::vector<float> swiglu_clamp_exp; // per layer, routed experts only

    uint32_t hc = 0;
    float hc_eps = 0.0f, hc_magnitude = 0.0f;

    dsv41::EngramConfig engram;

    float rope_base_of(uint32_t il) const { return is_swa[il] ? rope_base_swa : rope_base; }

    static std::expected<Config, std::string> load(const gguf::ModelFiles& m);
};

} // namespace mako::hy4f
