// Qwen-Image 2.1 against the reference pipeline's tensors (fp32 on the CPU, 512x512, 3 steps, seed 42; written by a
// script beside the model, manifest.json listing them):
//   tokens   the prompt template's token ids
//   image_ref_test <model dir> <reference dir> <mode>
#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

std::string read_file(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The pipeline's text-to-image prompt template.
std::string templated(const std::string& prompt)
{
    return "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n" + prompt +
           "<|im_end|>\n<|im_start|>assistant\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || !fs::exists(fs::path(argv[1]) / "transformer") || !fs::exists(fs::path(argv[2]) / "manifest.json")) {
        std::println("model {} or reference {} not found: skipped", argc > 1 ? argv[1] : "", argc > 2 ? argv[2] : "");
        return 77;
    }
    const fs::path model = argv[1], ref = argv[2];
    const std::string mode = argv[3];
    glz::generic manifest;
    if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    auto tok = mako::text::Tokenizer::load_json((model / "processor" / "tokenizer.json").string());
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    const std::string prompt = manifest["prompt"].get_string();
    std::vector<int32_t> want;
    for (const auto& v : manifest["ids"].get_array())
        want.push_back(int32_t(v.get_number()));

    if (mode == "tokens") {
        const auto ids = tok->encode(templated(prompt), true);
        check(ids == want, std::format("the template's {} token ids (the reference's {})", ids.size(), want.size()));
        check(tok->decode(ids) == templated(prompt), "they decode to the template");
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
