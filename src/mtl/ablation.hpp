#pragma once

#include "mtl/context.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mako::mtl {

// Per-layer directions removed from chosen rows of a model's computation (abliteration). Each row x at a site loses
// x -= scale (x . v) v for each of the site's k directions v in turn: the vectors' norms carry the strength (they
// need only be mutually orthogonal), scale is a global multiplier. File layout:
//   "MAKOABL1", uint32 n_embd, n_layer, n_site, k; n_site names of 16 bytes (NUL padded);
//   float32 vectors [n_layer][n_site][k][n_embd].
class Ablation {
public:
    // `sites`: the names the engine knows; a file naming another, or of another width or depth, is refused
    // (std::runtime_error).
    static Ablation load(const std::string& path, const Context& ctx, uint32_t n_embd, uint32_t n_layer,
                         std::span<const std::string_view> sites);
    // The same from vectors in the file's layout.
    static Ablation from_vectors(const Context& ctx, uint32_t n_embd, uint32_t n_layer, std::vector<std::string> names,
                                 uint32_t k, std::span<const float> vectors, std::span<const std::string_view> sites);

    // The site's index, -1 when the file has none.
    int site(std::string_view name) const;
    bool active(uint32_t layer, int site) const
    {
        return site >= 0 && layer < n_layer_ && nonzero_[uint64_t(layer) * names_.size() + uint64_t(site)];
    }
    void set_scale(float scale) { scale_ = scale; }
    float scale() const { return scale_; }

    // Encodes the projection of `rows` rows of n_embd floats at x + x_off with `pso` (the ablate_rows kernel);
    // nothing when the layer's vectors at the site are all zero.
    void apply(MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso, uint32_t layer, int site,
               MTL::Buffer* x, uint64_t x_off, uint32_t rows) const;

private:
    uint32_t n_embd_ = 0, n_layer_ = 0, k_ = 0;
    float scale_ = 1.0f;
    std::vector<std::string> names_;
    std::vector<char> nonzero_; // [layer][site]
    Ref<MTL::Buffer> buf_;
};

} // namespace mako::mtl
