#include "vision/mmproj.hpp"

#include <array>
#include <format>
#include <optional>

namespace mako::vision {

namespace {

std::optional<std::array<float, 3>> triple(const gguf::File& f, std::string_view key)
{
    const gguf::Value* v = f.find(key);
    const auto* a = v ? std::get_if<gguf::Array>(v) : nullptr;
    if (!a || a->items.size() != 3)
        return std::nullopt;
    std::array<float, 3> out{};
    for (size_t i = 0; i < 3; ++i) {
        const auto* d = std::get_if<double>(&a->items[i]);
        if (!d)
            return std::nullopt;
        out[i] = float(*d);
    }
    return out;
}

} // namespace

std::expected<Mmproj, std::string> open_mmproj(const std::string& path)
{
    auto f = gguf::File::open(path);
    if (!f)
        return std::unexpected(f.error());
    if (f->get_string("general.architecture") != "clip")
        return std::unexpected(std::format("{}: not a vision projector (general.architecture is not clip)", path));
    const std::string type(f->get_string("clip.projector_type").value_or(""));
    Mmproj m{.file = std::move(*f)};
    const gguf::File& g = m.file;
    if (type == "glm5v") {
        m.prep.kind = Projector::glm5v;
    } else if (type == "deepseek41v" || (type == "deepseek4v" && !g.tensor("v.token_embd.img_pad"))) {
        m.prep.kind = Projector::deepseek41v;
    } else {
        return std::unexpected(std::format("{}: projector type {} is not supported", path, type));
    }
    const auto u32 = [&](std::string_view key) { return uint32_t(g.get_int(key).value_or(0)); };
    m.n_embd = u32("clip.vision.embedding_length");
    m.n_head = u32("clip.vision.attention.head_count");
    m.n_ff = u32("clip.vision.feed_forward_length");
    m.n_layer = u32("clip.vision.block_count");
    m.out_embd = u32("clip.vision.projection_dim");
    m.eps = float(g.get_float("clip.vision.attention.layer_norm_epsilon").value_or(0.0));
    m.swiglu_clamp = float(g.get_float("clip.vision.swiglu_clamp").value_or(0.0));
    m.prep.patch = u32("clip.vision.patch_size");
    m.prep.merge = m.prep.kind == Projector::glm5v ? u32("clip.vision.spatial_merge_size")
                                                   : u32("clip.vision.projector.scale_factor");
    m.prep.min_pixels = uint64_t(g.get_int("clip.vision.image_min_pixels").value_or(0));
    m.prep.max_pixels = uint64_t(g.get_int("clip.vision.image_max_pixels").value_or(0));
    const auto mean = triple(g, "clip.vision.image_mean"), sd = triple(g, "clip.vision.image_std");
    if (!mean || !sd)
        return std::unexpected(std::format("{}: no image mean / std", path));
    m.prep.mean = *mean;
    m.prep.std = *sd;
    if (!m.n_embd || !m.n_head || m.n_embd % m.n_head || !m.n_ff || !m.n_layer || !m.out_embd || !m.prep.patch ||
        !m.prep.merge || m.eps <= 0.0f)
        return std::unexpected(std::format("{}: incomplete vision metadata", path));
    if (m.prep.kind == Projector::glm5v && (!m.prep.min_pixels || !m.prep.max_pixels))
        return std::unexpected(std::format("{}: no image pixel range", path));
    return m;
}

} // namespace mako::vision
