#pragma once

#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "vision/mmproj.hpp"
#include "vision/preprocess.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace mako::vision {

// The vision encoder of a projector file on the GPU: patch embedding, ViT, merge and projector. Linear weights are
// held as fp16 (the files' bf16 converted at load: no file weight is beyond fp16's range; ~0.2% are below its
// normal range, under 6e-5); activations are fp32 between kernels and half into each matrix product.
class Encoder {
public:
    Encoder(const mtl::Context& ctx, const Mmproj& mm);

    // An image's rows for the text model, [rows][out_embd]: deepseek41v the whole block (start, each row of cells
    // followed by a newline, end); glm5v the cells in raster order (the text model's begin / end tokens go around
    // them).
    struct Output {
        uint32_t rows = 0, cells_x = 0, cells_y = 0;
        std::vector<float> embd;
    };
    Output encode(const Prepared& img);
    // The rows encode() gives for an image prepared for this projector.
    static uint32_t rows_for(const Mmproj& mm, const Prepared& img);

    // Called with each stage's output once the encode completes (patch_bias, layer_out-N, vit_out, merged,
    // projected), as fp32 rows.
    std::function<void(std::string_view name, const float* data, size_t n)> probe;

    uint32_t out_embd() const { return out_embd_; }

    // With timing, encode() commits each stage on its own and adds its GPU time to stage_ms (patch, qkv,
    // attention, out_ffn, merge).
    bool timing = false;
    std::map<std::string, double> stage_ms;

private:
    // A weight on the GPU: fp16 for linears, fp32 otherwise; null when the file has no such tensor.
    using Tensor = MTL::Buffer*;
    struct Layer {
        Tensor ln1, qkv, qkv_b, qn, kn, out, out_b, ln2, gate, gate_b, up, up_b, down, down_b;
    };
    Tensor upload(const gguf::Tensor& t, bool to_f16);
    Tensor weight(std::string_view name, bool to_f16 = true, bool required = true);
    // Each output stage when probing: the GPU work so far completes first.
    void sync(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc, std::string_view name, MTL::Buffer* buf,
              size_t n);
    void linear(MTL::ComputeCommandEncoder* enc, Tensor w, uint32_t k, uint32_t n, uint32_t t, MTL::Buffer* xh,
                uint64_t xh_off, MTL::Buffer* y, uint64_t y_off = 0, uint32_t ldy = 0);
    void rows_op(MTL::ComputeCommandEncoder* enc, const char* fn, std::initializer_list<MTL::Buffer*> bufs,
                 const void* args, size_t size, uint32_t n, uint32_t rows, int flag = -1);
    MTL::Buffer* scratch(const std::string& name, uint64_t bytes);
    MTL::ComputePipelineState* pso(const std::string& name);

    const mtl::Context& ctx_;
    const Mmproj& mm_;
    mtl::Gemm gemm_;
    mtl::Ref<MTL::Library> lib_;
    std::map<std::string, mtl::Ref<MTL::ComputePipelineState>> psos_;
    std::vector<mtl::Ref<MTL::Buffer>> weights_;
    Tensor patch_w_ = nullptr, patch_b_ = nullptr, post_ln_ = nullptr;
    std::vector<Layer> layers_;
    // deepseek41v: the aligner and the block's marker rows (fp32 on the CPU); glm5v: merger (rearranged to rows of
    // the 4 consecutive tokens' features), fc, LayerNorm, gate / up / down.
    Tensor mm1_ = nullptr, mm1_b_ = nullptr, mm2_ = nullptr, mm2_b_ = nullptr;
    std::vector<float> start_, end_, newline_;
    Tensor merger_ = nullptr, merger_b_ = nullptr, fc_ = nullptr, norm_w_ = nullptr, norm_b_ = nullptr;
    Tensor pgate_ = nullptr, pup_ = nullptr, pdown_ = nullptr;
    uint32_t proj_ff_ = 0;
    std::map<std::string, mtl::Ref<MTL::Buffer>> scratch_;
    uint32_t out_embd_ = 0;
};

} // namespace mako::vision
