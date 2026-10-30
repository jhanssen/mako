// The safetensors reader on Qwen-Image 2.1's files: tensor counts and types per component, shapes as GGUF orders them
// (dims[0] contiguous), and values read at the offsets it gives (expected values from the files' own Python reader).
//   safetensors_test <model directory>
#include "gguf/read_pool.hpp"
#include "safetensors/safetensors.hpp"

#include <cstring>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

float bf16(uint16_t b)
{
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

} // namespace

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;
    if (argc < 2 || !fs::exists(fs::path(argv[1]) / "transformer")) {
        std::println("model {} not found: skipped", argc < 2 ? "" : argv[1]);
        return 77;
    }
    const fs::path root = argv[1];
    using T = mako::gguf::Type;
    for (const auto& [dir, count, type] : {std::tuple{"transformer", size_t(297), T::BF16},
                                           std::tuple{"text_encoder", size_t(750), T::BF16},
                                           std::tuple{"vae", size_t(238), T::F32}}) {
        auto m = mako::safetensors::Files::open((root / dir).string());
        if (!m) {
            check(false, m.error());
            continue;
        }
        size_t n = 0, typed = 0;
        for (const auto& f : m->files())
            for (const auto& t : f.tensors()) {
                ++n;
                typed += t.type == type;
            }
        check(n == count && typed == count, std::format("{}: {} tensors (want {}), {} of type {}", dir, n, count,
                                                        typed, mako::gguf::traits(type).name));
    }

    auto m = mako::safetensors::Files::open((root / "transformer").string());
    if (!m)
        return 1;
    struct Want {
        const char* name;
        std::vector<uint64_t> dims;
        float first[4];
    };
    for (const Want& w : {Want{"img_in.weight", {64, 4096}, {-0.021240234375f, -0.01031494140625f,
                                                              -0.0019378662109375f, -0.00823974609375f}},
                          Want{"transformer_blocks.0.attn.to_q.weight", {4096, 4096},
                               {-0.07470703125f, 0.03271484375f, 0.06298828125f, 0.1650390625f}}}) {
        const auto r = m->find(w.name);
        if (!r) {
            check(false, std::format("{} missing", w.name));
            continue;
        }
        uint16_t raw[4];
        mako::gguf::read_exact(r.file->fd(), raw, sizeof(raw), r.tensor->offset);
        bool same = true;
        for (int i = 0; i < 4; ++i)
            same &= bf16(raw[i]) == w.first[i];
        check(r.tensor->dims == w.dims && same,
              std::format("{}: dims [{}, {}], first values {}", w.name, r.tensor->dims[0], r.tensor->dims[1],
                          same ? "as the Python reader gives" : "differ"));
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
