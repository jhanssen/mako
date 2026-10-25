#pragma once

#include "gguf/model_files.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace mako::dsv41 {

// DeepSeek-V4.1's geometry from a GGUF's deepseek41.* metadata.
struct Config {
    uint32_t n_layer = 0; // trunk layers
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    uint32_t n_head = 0;
    uint32_t head_dim = 0; // K = V
    uint32_t n_rot = 0;    // the head's last n_rot dims are roped
    uint32_t q_lora_rank = 0;
    uint32_t o_lora_rank = 0;
    uint32_t o_groups = 0;
    uint32_t sliding_window = 0;
    float rms_eps = 0.0f;

    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_shared = 0;
    uint32_t n_ff_exp = 0;
    uint32_t gating = 0; // expert_gating_func
    float expert_weights_scale = 1.0f;
    bool expert_weights_norm = false;
    std::vector<float> swiglu_clamp_exp, swiglu_clamp_shexp; // per layer

    // RoPE (YaRN) for uncompressed layers, and the base of compressed ones.
    float rope_base = 0.0f, compress_rope_base = 0.0f;
    float yarn_factor = 1.0f, yarn_beta_fast = 0.0f, yarn_beta_slow = 0.0f;
    uint32_t yarn_orig_ctx = 0;

    std::vector<int32_t> compress_ratios; // per block (trunk, then the draft blocks)
    std::vector<int32_t> kv_source_layers, indexer_source_layers;
    uint32_t indexer_heads = 0, indexer_dim = 0, indexer_top_k = 0;
    int32_t candidate_source_layer = -1;
    uint32_t candidate_top_k_blocks = 0, candidate_block_size = 0;

    uint32_t hc = 0, hc_sinkhorn_iters = 0;
    float hc_eps = 0.0f;

    // Engram: hashed n-gram tables read per token.
    struct Engram {
        std::vector<int32_t> layers;
        std::vector<int64_t> num_embeddings; // rows per layer's table
        uint32_t max_ngram = 0, heads = 0, head_dim = 0;
        int32_t pad_token = 0;
        std::vector<int32_t> token_map;    // vocabulary -> compressed id
        std::vector<int64_t> multipliers;  // per layer, max_ngram
        std::vector<int64_t> primes, offsets; // per layer, (max_ngram - 1) * heads columns
        uint32_t columns() const { return (max_ngram - 1) * heads; }
    } engram;

    static std::expected<Config, std::string> load(const gguf::ModelFiles& m);
};

} // namespace mako::dsv41
