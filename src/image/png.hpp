#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

// RGBA [height][width][4] in [-1, 1] to 8-bit samples as the reference pipeline makes them (x / 2 + 1/2, clamped,
// times 255, rounded).
std::vector<uint8_t> to_rgba8(std::span<const float> rgba);
// A PNG of 8-bit RGBA; with `rgb`, the alpha channel left out. Throws std::runtime_error.
std::vector<uint8_t> encode_png(std::span<const uint8_t> rgba8, uint32_t width, uint32_t height, bool rgb = false);
void write_file(const std::string& path, std::span<const uint8_t> data);

} // namespace mako::image
