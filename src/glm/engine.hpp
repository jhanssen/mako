#pragma once

#include "glm/arena.hpp"
#include "glm/config.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mako::glm {

struct Options {
    uint32_t max_ctx = 4096;
    uint32_t indexer_top_k = 0; // nonzero: override the model's (for testing the sparse selection)
    double arena_gb = -1;       // expert arena size; < 0: from the memory budget
};

// Decode, one token per step. Dense weights are resident; routed experts live in an expert arena, and the ones
// a token selects but the arena lacks are read from the model files after each MoE layer's router runs.
class Engine {
public:
    Engine(const gguf::ModelFiles& model, const Config& cfg, const Options& opts, mtl::Context& ctx);
    ~Engine();

    // Runs one token at the next position and returns its logits.
    const std::vector<float>& step(int32_t token);

    uint32_t position() const { return pos_; }

    // Called after each named stage of a step with the fp32 values it produced (for parity checks).
    using Probe = std::function<void(const std::string& name, int layer, const float* data, size_t n)>;
    void set_probe(Probe probe) { probe_ = std::move(probe); }

    // Teacher forcing for parity checks. Called before each layer with its input residual streams
    // (hc * n_embd values), which it may overwrite. Costs a GPU sync per layer.
    using StreamOverride = std::function<void(int layer, float* streams, size_t n)>;
    void set_stream_override(StreamOverride f) { stream_override_ = std::move(f); }
    // Called after routing a MoE layer; may overwrite the selected experts and their final weights (k each).
    using RouteOverride = std::function<void(int layer, int* sel, float* w, size_t k)>;
    void set_route_override(RouteOverride f) { route_override_ = std::move(f); }

    struct Stats {
        double expert_read_ms = 0, gpu_wait_ms = 0;
        uint64_t expert_bytes = 0;
    };
    const Stats& stats() const { return stats_; }

    // The memory figures the arena was sized from and its size (bytes).
    struct ArenaPlan {
        MemInfo mem;
        double budget = 0;
        double warm_ms = 0; // first GPU pass over the arena, at load
    };
    const ArenaPlan& arena_plan() const { return arena_plan_; }
    const ExpertArena& arena() const { return *arena_; }

private:
    struct DevTensor {
        MTL::Buffer* buf = nullptr;
        uint64_t off = 0;
        gguf::Type type = gguf::Type::F32;
        std::vector<uint64_t> dims;
    };
    struct Layer;

    DevTensor upload(const std::string& name);
    DevTensor upload_optional(const std::string& name);
    ExpertArena::Part expert(const std::string& name);
    MTL::Buffer* scratch(const std::string& name, uint64_t bytes);
    MTL::ComputePipelineState* pso(const std::string& fn);

    void gemv(MTL::ComputeCommandEncoder* enc, const DevTensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t batch = 1);
    void dispatch(MTL::ComputeCommandEncoder* enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                  const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads = true);
    void norm(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, const DevTensor* w, MTL::Buffer* y, uint32_t rows,
              uint32_t n, uint32_t wn, float eps);
    void hc_pre(MTL::ComputeCommandEncoder* enc, const DevTensor& fn, const DevTensor& scale, const DevTensor& base);
    void hc_post(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x);
    void kda(MTL::ComputeCommandEncoder* enc, const Layer& L, uint32_t il);
    void mla(MTL::ComputeCommandEncoder* enc, const Layer& L, uint32_t il);
    bool indexer(MTL::ComputeCommandEncoder* enc, const Layer& L, uint32_t il);
    void ffn_dense(MTL::ComputeCommandEncoder* enc, const DevTensor& gate, const DevTensor& up,
                   const DevTensor& down, MTL::Buffer* out, float limit);
    void route(const Layer& L, uint32_t il, std::array<int, 16>& sel, std::array<float, 16>& w);
    void probe(MTL::ComputeCommandEncoder* enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
               uint64_t off = 0);
    void deliver_probes();
    void submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc);

    const gguf::ModelFiles& model_;
    Config cfg_;
    Options opts_;
    mtl::Context& ctx_;
    std::unique_ptr<mtl::Gemv> gemv_;
    mtl::Ref<MTL::Library> lib_;
    std::map<std::string, mtl::Ref<MTL::ComputePipelineState>> psos_;

    std::vector<mtl::Ref<MTL::Buffer>> weight_chunks_;
    uint64_t chunk_used_ = 0;
    std::map<std::string, mtl::Ref<MTL::Buffer>> scratch_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::map<std::string, int> nocache_fds_; // shard path -> descriptor for expert reads
    std::unique_ptr<ExpertArena> arena_;
    ArenaPlan arena_plan_;

    gguf::ModelFiles::Ref token_embd_;
    DevTensor output_norm_, output_;
    std::vector<mtl::Ref<MTL::Buffer>> kda_state_, conv_state_, kv_cache_, idx_cache_, idx_pooled_;

    uint32_t pos_ = 0;
    std::vector<float> logits_;
    Probe probe_;
    StreamOverride stream_override_;
    RouteOverride route_override_;
    struct PendingProbe {
        std::string name;
        int layer;
        mtl::Ref<MTL::Buffer> copy;
        size_t n;
    };
    std::vector<PendingProbe> pending_probes_;
    Stats stats_;
};

} // namespace mako::glm
