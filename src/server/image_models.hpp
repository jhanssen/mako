#pragma once

#include "image/engine.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mako::server {

// An image model as a directory: its files (the directory, or where its mako.json points) and the settings mako.json
// gives. mako.json (every field optional, paths relative to its directory):
//   {"model": DIR, "transformer": PATH, "text_encoder": PATH, "lora_dir": DIR, "loras": {NAME: SCALE, ...},
//    "steps": N, "terminal": F, "guidance": F, "description": TEXT, "prompting": TEXT}
// LoRAs are named by file name without .safetensors, from lora_dir (default: the directory's loras/); "loras" are the
// ones merged for requests that name none.
struct ImageModel {
    std::string name, dir;
    std::string family, description; // which model it is (image::Engine::family); what it is for, from mako.json
    std::string prompting;           // how to write its prompts, from mako.json
    image::Options options;
    std::string lora_dir; // empty: none
    image::Loras loras;   // files and scales
    uint32_t steps = 0;   // 0: the model's
    std::optional<float> terminal, guidance;
    // What decides the loaded weights besides LoRAs: two models with the same key load the same.
    std::string key() const { return dir + "|" + options.text_encoder + "|" + options.transformer; }
};

// The model in `dir` under `name`. Throws std::runtime_error naming the problem.
ImageModel load_image_model(const std::string& dir, const std::string& name);
// The subdirectories of `root` that hold a model or a mako.json, by name (read now); ones that fail to load are left
// out with a warning.
std::vector<ImageModel> image_models(const std::string& root);

} // namespace mako::server
