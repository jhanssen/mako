#pragma once

#include "gen/session.hpp"
#include "vision/preprocess.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace mako::vision {

// The chat template's image placeholder token text: deepseek41v's stands for the whole block, glm5v's for the cells
// (the template writes the begin / end tokens around it).
std::string_view placeholder(Projector kind);

// A prompt with its images in place: each placeholder token (in order) replaced by as many placeholder tokens as its
// image has rows (counts[i]), which take those rows. keys[i] identifies image i's content. The spans' row pointers
// are left null for the caller to set once the images are encoded.
struct ImagePrompt {
    std::vector<int32_t> tokens;
    std::vector<gen::GenerateParams::PromptRows> rows;
};
// Throws std::runtime_error when the number of placeholders and images differ.
ImagePrompt expand_images(std::span<const int32_t> tokens, int32_t placeholder, std::span<const uint32_t> counts,
                          std::span<const uint64_t> keys);

} // namespace mako::vision
