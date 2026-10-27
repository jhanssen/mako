#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mako::vision {

// An 8-bit RGB image: rows top to bottom, pixels interleaved R, G, B.
struct Rgb8 {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> px;
};

// Decodes an image file (PNG, JPEG, and the other formats ImageIO reads) to its stored samples, without colour
// management: alpha dropped (not composited), grey replicated, 16-bit samples reduced to their high byte, palette
// entries looked up. Formats outside that (float or packed samples, CMYK, ...) are drawn into device RGB instead.
// Throws std::runtime_error.
Rgb8 decode_image(std::span<const std::byte> data);

// Bicubic resampling (a = -0.5) in two separable passes, horizontal first, each with 22-bit fixed-point weights
// normalised to sum 1 and the intermediate image clamped to 8 bits; a dimension of the same size is not resampled.
Rgb8 resize_bicubic(const Rgb8& in, uint32_t width, uint32_t height);

} // namespace mako::vision
