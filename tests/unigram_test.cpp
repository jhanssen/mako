// The unigram tokenizer (T5's) against the reference tokenizer's ids for tests/data/t5_unigram_cases.json.
//   unigram_test <T5 tokenizer.json>
#include "text/unigram.hpp"

#include <glaze/glaze.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2 || !std::filesystem::exists(argv[1])) {
        std::println("tokenizer {} not found: skipped", argc > 1 ? argv[1] : "");
        return 77;
    }
    auto t = mako::text::Unigram::load_json(argv[1]);
    if (!t) {
        std::println(stderr, "{}", t.error());
        return 1;
    }
    std::ifstream in(MAKO_TEST_DATA "/t5_unigram_cases.json", std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic cases;
    if (glz::read_json(cases, text)) {
        std::println(stderr, "bad cases file");
        return 1;
    }
    int failures = 0;
    for (const auto& c : cases.get_array()) {
        const std::string s = c["text"].get_string();
        std::vector<int32_t> want;
        for (const auto& v : c["ids"].get_array())
            want.push_back(int32_t(v.get_number()));
        auto got = t->encode(s);
        got.push_back(t->eos());
        const bool ok = got == want;
        failures += !ok;
        std::println("  {:4} {:?} ({} tokens)", ok ? "ok" : "FAIL", s.substr(0, 40), want.size());
        if (!ok) {
            std::string g, w;
            for (int32_t x : got)
                g += std::to_string(x) + " ";
            for (int32_t x : want)
                w += std::to_string(x) + " ";
            std::println("       got  {}\n       want {}", g, w);
        }
    }
    return failures ? 1 : 0;
}
