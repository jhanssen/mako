#include "server/image_models.hpp"

#include "core/log.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace mako::server {

namespace {

namespace fs = std::filesystem;

std::string resolve(const fs::path& base, const std::string& p)
{
    if (p.starts_with("~/"))
        if (const char* home = std::getenv("HOME"))
            return (fs::path(home) / p.substr(2)).lexically_normal().string();
    const fs::path q(p);
    return (q.is_absolute() ? q : base / q).lexically_normal().string();
}

bool holds_model(const fs::path& d)
{
    return fs::exists(d / "model_index.json") || fs::exists(d / "mako.json") || fs::is_directory(d / "split_files") ||
           fs::is_directory(d / "diffusion_models");
}

} // namespace

ImageModel load_image_model(const std::string& dir, const std::string& name)
{
    const fs::path base = fs::absolute(fs::path(dir)).lexically_normal();
    ImageModel m;
    m.name = name;
    m.dir = base.string();
    std::string lora_dir = (base / "loras").string();
    const fs::path cfg = base / "mako.json";
    if (fs::exists(cfg)) {
        std::ifstream in(cfg, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (glz::read_json(j, text) || !j.is_object())
            throw std::runtime_error(cfg.string() + ": not a JSON object");
        auto str = [&](const char* key) -> std::optional<std::string> {
            if (!j.contains(key))
                return std::nullopt;
            if (!j[key].is_string())
                throw std::runtime_error(std::format("{}: {} must be a string", cfg.string(), key));
            return j[key].get_string();
        };
        auto num = [&](const char* key) -> std::optional<double> {
            if (!j.contains(key))
                return std::nullopt;
            if (!j[key].is_number())
                throw std::runtime_error(std::format("{}: {} must be a number", cfg.string(), key));
            return j[key].get_number();
        };
        if (auto v = str("model"))
            m.dir = resolve(base, *v);
        if (auto v = str("transformer"))
            m.options.transformer = resolve(base, *v);
        if (auto v = str("text_encoder"))
            m.options.text_encoder = resolve(base, *v);
        if (auto v = str("description"))
            m.description = *v;
        if (auto v = str("lora_dir"))
            lora_dir = resolve(base, *v);
        if (auto v = num("steps")) {
            if (*v < 0 || *v > 100 || *v != double(uint32_t(*v)))
                throw std::runtime_error(cfg.string() + ": steps must be a whole number from 0 to 100");
            m.steps = uint32_t(*v);
        }
        if (auto v = num("terminal"))
            m.terminal = float(*v);
        if (auto v = num("guidance"))
            m.guidance = float(*v);
        if (j.contains("loras")) {
            if (!j["loras"].is_object())
                throw std::runtime_error(cfg.string() + ": loras must be an object of name: scale");
            for (const auto& [n, s] : j["loras"].get_object()) {
                if (!s.is_number())
                    throw std::runtime_error(std::format("{}: the scale of LoRA {} must be a number", cfg.string(), n));
                const fs::path f = fs::path(lora_dir) / (n.ends_with(".safetensors") ? n : n + ".safetensors");
                if (!fs::exists(f))
                    throw std::runtime_error(std::format("{}: no LoRA {}", cfg.string(), f.string()));
                m.loras.emplace_back(f.string(), float(s.get_number()));
            }
        }
    }
    for (const std::string* p : {&m.options.transformer, &m.options.text_encoder})
        if (!p->empty() && !fs::exists(*p))
            throw std::runtime_error(std::format("{}: {} does not exist", name, *p));
    if (!fs::is_directory(m.dir))
        throw std::runtime_error(std::format("{}: {} is not a directory", name, m.dir));
    if (fs::is_directory(lora_dir))
        m.lora_dir = lora_dir;
    m.family = image::Engine::family(m.dir);
    if (m.family.empty())
        throw std::runtime_error(std::format("{}: {} holds no image model mako knows", name, m.dir));
    return m;
}

std::vector<ImageModel> image_models(const std::string& root)
{
    std::vector<ImageModel> out;
    std::vector<fs::path> dirs;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(root, ec))
        if (e.is_directory(ec) && holds_model(e.path()))
            dirs.push_back(e.path());
    std::ranges::sort(dirs);
    for (const auto& d : dirs) {
        try {
            out.push_back(load_image_model(d.string(), d.filename().string()));
        } catch (const std::exception& e) {
            log::warn("image model {}: {}", d.filename().string(), e.what());
        }
    }
    return out;
}

} // namespace mako::server
