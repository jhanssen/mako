#include "mtl/ablation.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>

namespace mako::mtl {

namespace {

struct AblateArgs {
    uint32_t n_embd, k;
    float scale;
};

} // namespace

Ablation Ablation::from_vectors(const Context& ctx, uint32_t n_embd, uint32_t n_layer, std::vector<std::string> names,
                                uint32_t k, std::span<const float> vectors, std::span<const std::string_view> sites)
{
    for (const auto& n : names)
        if (std::ranges::find(sites, std::string_view(n)) == sites.end())
            throw std::runtime_error(std::format("ablation site '{}' is not one this model has", n));
    const uint64_t per = uint64_t(k) * n_embd;
    if (k == 0 || vectors.size() != uint64_t(n_layer) * names.size() * per)
        throw std::runtime_error(std::format("ablation: {} floats are not {} layers x {} sites x {} x {}",
                                             vectors.size(), n_layer, names.size(), k, n_embd));
    Ablation a;
    a.n_embd_ = n_embd;
    a.n_layer_ = n_layer;
    a.k_ = k;
    a.names_ = std::move(names);
    a.nonzero_.resize(uint64_t(n_layer) * a.names_.size());
    for (size_t i = 0; i < a.nonzero_.size(); ++i)
        a.nonzero_[i] = std::ranges::any_of(vectors.subspan(i * per, per), [](float x) { return x != 0.0f; });
    a.buf_ = Ref<MTL::Buffer>(
        ctx.device()->newBuffer(vectors.data(), vectors.size() * 4, MTL::ResourceStorageModeShared));
    return a;
}

Ablation Ablation::load(const std::string& path, const Context& ctx, uint32_t n_embd, uint32_t n_layer,
                        std::span<const std::string_view> sites)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error(std::format("cannot open {}", path));
    char magic[8];
    uint32_t head[4];
    if (!in.read(magic, 8) || std::memcmp(magic, "MAKOABL1", 8) || !in.read(reinterpret_cast<char*>(head), 16))
        throw std::runtime_error(std::format("{} is not an ablation file (MAKOABL1)", path));
    const auto [embd, layers, n_site, k] = head;
    if (embd != n_embd || layers != n_layer)
        throw std::runtime_error(std::format("{}: {} layers x {} wide, the model has {} x {}", path, layers, embd,
                                             n_layer, n_embd));
    std::vector<std::string> names;
    for (uint32_t s = 0; s < n_site; ++s) {
        char name[17] = {};
        if (!in.read(name, 16))
            throw std::runtime_error(std::format("{}: truncated", path));
        names.emplace_back(name);
    }
    std::vector<float> v(uint64_t(layers) * n_site * k * embd);
    if (!in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
        throw std::runtime_error(std::format("{}: truncated", path));
    return from_vectors(ctx, n_embd, n_layer, std::move(names), k, v, sites);
}

int Ablation::site(std::string_view name) const
{
    const auto it = std::ranges::find(names_, name);
    return it == names_.end() ? -1 : int(it - names_.begin());
}

void Ablation::apply(MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso, uint32_t layer, int site,
                     MTL::Buffer* x, uint64_t x_off, uint32_t rows) const
{
    if (!active(layer, site) || rows == 0)
        return;
    const AblateArgs a{n_embd_, k_, scale_};
    enc->setComputePipelineState(pso);
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(buf_.get(), (uint64_t(layer) * names_.size() + uint64_t(site)) * k_ * n_embd_ * 4, 1);
    enc->setBytes(&a, sizeof(a), 2);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
}

} // namespace mako::mtl
