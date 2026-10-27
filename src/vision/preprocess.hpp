#pragma once

#include "vision/image.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace mako::vision {

enum class Projector {
    deepseek41v, // DeepSeek-V4.1-Flash
    glm5v,       // GLM-5.3-Flash
};

// What sizing and normalising an image takes from the projector file.
struct PrepParams {
    Projector kind = Projector::glm5v;
    uint32_t patch = 14, merge = 2;
    std::array<float, 3> mean{}, std{};
    uint64_t min_pixels = 0, max_pixels = 0; // max_pixels: glm5v only
};

// The encoder's input: normalised samples, planar ([3][height][width]), height and width multiples of the patch
// size (glm5v: of patch * merge).
struct Prepared {
    uint32_t width = 0, height = 0;
    std::vector<float> chw;
};

// deepseek41v: the largest size, rounded up to whole patches, whose token block (h * (w + 1) + 2 of the merged
// cells) stays within 1024, at least min_pixels; the image is fitted to it with its aspect kept and centred on grey
// (127), and the normalised samples rounded to bf16.
// glm5v: the image rounded up to a multiple of patch * merge, resized only to bring its pixels within [min_pixels,
// max_pixels], top left on black.
Prepared prepare(const PrepParams& p, const Rgb8& img);

} // namespace mako::vision
