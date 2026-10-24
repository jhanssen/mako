#include "glm/profile.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cstdlib>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mako::glm {

std::string routing_key(const gguf::ModelFiles& model, const Config& cfg)
{
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    for (uint32_t il = cfg.n_dense_lead; il < cfg.n_layer; ++il)
        for (const char* name : {"ffn_gate_inp.weight", "exp_probs_b.bias"}) {
            const std::string full = std::format("blk.{}.{}", il, name);
            auto ref = model.find(full);
            if (!ref)
                throw std::runtime_error("missing tensor " + full);
            auto data = ref.file->data(*ref.tensor);
            if (data.size() != ref.tensor->bytes)
                throw std::runtime_error(full + " extends past the end of its file");
            CC_SHA256_Update(&ctx, data.data(), CC_LONG(data.size()));
        }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);
    std::string hex;
    for (unsigned char b : digest)
        hex += std::format("{:02x}", b);
    return hex;
}

std::filesystem::path default_profile_path(const std::string& key)
{
    const char* home = std::getenv("HOME");
    if (!home)
        throw std::runtime_error("HOME is not set");
    return std::filesystem::path(home) / "Library/Application Support/mako/profiles" / (key + ".counts");
}

Counts load_profile(const std::filesystem::path& path, uint32_t n_layer, uint32_t n_expert)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path.string());
    Counts counts(n_layer);
    for (std::string line; std::getline(in, line);) {
        std::istringstream ls(line);
        uint32_t layer;
        if (!(ls >> layer))
            continue;
        if (layer >= n_layer)
            throw std::runtime_error(std::format("{}: layer {} out of range", path.string(), layer));
        counts[layer].clear();
        for (uint64_t c; ls >> c;)
            counts[layer].push_back(c);
        if (counts[layer].size() != n_expert)
            throw std::runtime_error(std::format("{}: layer {} has {} counts, expected {}", path.string(), layer,
                                                 counts[layer].size(), n_expert));
    }
    return counts;
}

void save_profile(const std::filesystem::path& path, const Counts& counts)
{
    std::filesystem::create_directories(path.parent_path());
    const auto tmp = std::filesystem::path(path).concat(".tmp");
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot write " + tmp.string());
        for (size_t l = 0; l < counts.size(); ++l) {
            if (counts[l].empty())
                continue;
            out << l;
            for (uint64_t c : counts[l])
                out << ' ' << c;
            out << '\n';
        }
        if (!out)
            throw std::runtime_error("writing " + tmp.string() + " failed");
    }
    std::filesystem::rename(tmp, path);
}

} // namespace mako::glm
