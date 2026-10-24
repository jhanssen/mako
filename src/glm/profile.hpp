#pragma once

#include "glm/config.hpp"
#include "gguf/model_files.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mako::glm {

// A routing profile: how often each routed expert was selected over some text, [layer][expert]; rows of non-MoE
// layers are empty. Stored as text, one "layer c0 c1 ..." line per MoE layer.
using Counts = std::vector<std::vector<uint64_t>>;

// Hex SHA-256 over every MoE layer's router weights and bias, which alone decide routing.
std::string routing_key(const gguf::ModelFiles& model, const Config& cfg);
// ~/Library/Application Support/mako/profiles/<key>.counts
std::filesystem::path default_profile_path(const std::string& key);

Counts load_profile(const std::filesystem::path& path, uint32_t n_layer, uint32_t n_expert);
// Writes through a temporary file and a rename, creating the directory.
void save_profile(const std::filesystem::path& path, const Counts& counts);

} // namespace mako::glm
