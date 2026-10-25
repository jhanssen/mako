// A model's chat template against reference renderings: each <dir>/case-N.json (the template's variables) must
// render to <dir>/case-N.txt byte for byte.
//   template_cases_test <model.gguf> <dir>
#include "chat/template.hpp"
#include "gguf/gguf.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <string>

namespace {

std::string read(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: template_cases_test <model.gguf> <dir>");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    auto f = mako::gguf::File::open(argv[1]);
    const auto source = f ? f->get_string("tokenizer.chat_template") : std::nullopt;
    if (!source) {
        std::println(stderr, "no chat template in {}", argv[1]);
        return 1;
    }
    const mako::chat::Template tmpl{std::string(*source)};
    int cases = 0, failures = 0;
    for (int n = 1; std::filesystem::exists(std::filesystem::path(argv[2]) / std::format("case-{}.json", n)); ++n, ++cases) {
        const auto dir = std::filesystem::path(argv[2]);
        const std::string want = read(dir / std::format("case-{}.txt", n));
        std::string got;
        try {
            got = tmpl.render(read(dir / std::format("case-{}.json", n))).text;
        } catch (const std::exception& e) {
            got = std::string("error: ") + e.what();
        }
        if (got != want) {
            size_t i = 0;
            while (i < got.size() && i < want.size() && got[i] == want[i])
                ++i;
            std::println("FAIL case {}: differs at byte {} of {}", n, i, want.size());
            ++failures;
        }
    }
    std::println("{} cases, {} failures", cases, failures);
    return failures || cases == 0 ? 1 : 0;
}
