#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mako::image {

// RGBA [height][width][4] in [-1, 1] to 8-bit samples as the reference pipeline makes them (x / 2 + 1/2, clamped,
// times 255, rounded).
std::vector<uint8_t> to_rgba8(std::span<const float> rgba);
// A PNG of 8-bit RGBA (with `rgb`, the alpha channel left out), or a JPEG (no alpha; quality 0.92). Throws
// std::runtime_error.
std::vector<uint8_t> encode_png(std::span<const uint8_t> rgba8, uint32_t width, uint32_t height, bool rgb = false,
                                bool jpeg = false);
void write_file(const std::string& path, std::span<const uint8_t> data);

// How an image was made, for the PNG's text chunks.
struct Provenance {
    std::string prompt, model;
    std::optional<std::string> negative; // when guided
    float guidance = 1;
    uint32_t steps = 0, width = 0, height = 0;
    uint64_t seed = 0;
    std::vector<std::pair<std::string, float>> loras; // names and scales
};
// The PNG with UTF-8 text chunks before its end: "parameters" (the prompt, then "Negative prompt: ...", then
// "Steps: ..., Sampler: Euler, CFG scale: ..., Seed: ..., Size: WxH, Model: ...", as image tools commonly read it)
// and "mako" (the same as JSON, with the LoRAs). Throws std::runtime_error on data that is not a PNG.
std::vector<uint8_t> with_provenance(std::vector<uint8_t> png, const Provenance& p);

} // namespace mako::image
