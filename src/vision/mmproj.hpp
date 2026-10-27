#pragma once

#include "gguf/gguf.hpp"
#include "vision/preprocess.hpp"

#include <cstdint>
#include <expected>
#include <string>

namespace mako::vision {

// A vision projector file (GGUF, general.architecture "clip"): the encoder's shape and the image sizing.
struct Mmproj {
    gguf::File file;
    PrepParams prep;
    uint32_t n_embd = 0, n_head = 0, n_ff = 0, n_layer = 0;
    uint32_t out_embd = 0; // the text model's width
    float eps = 0.0f;
    float swiglu_clamp = 0.0f; // 0: none
};

// Reads the metadata and checks the tensors the projector type needs. The DeepSeek-V4.1 file may name its type
// "deepseek4v" (the V4 type); without V4's padding vector it is taken as V4.1.
std::expected<Mmproj, std::string> open_mmproj(const std::string& path);

} // namespace mako::vision
