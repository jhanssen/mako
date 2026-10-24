// Tokenizer against golden ids recorded from the reference tokenizer, plus decode round trips.
//   tokenizer_test <model.gguf> <cases file> <expected ids file>
// Cases are separated by 0x1E; the expected file has one comma-separated line of ids per case.
#include "gguf/gguf.hpp"
#include "text/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <print>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::println(stderr, "usage: tokenizer_test <model.gguf> <cases> <expected>");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(*f);
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    std::ifstream cin_(argv[2], std::ios::binary), ein(argv[3]);
    std::string all((std::istreambuf_iterator<char>(cin_)), std::istreambuf_iterator<char>());
    std::stringstream cases(all);
    int n = 0, wrong = 0, trip = 0;
    std::string c, expect;
    while (std::getline(cases, c, '\x1e')) {
        if (!std::getline(ein, expect)) {
            std::println("expected file is shorter than the cases");
            return 1;
        }
        const auto ids = tok->encode(c, true);
        std::string got;
        for (size_t i = 0; i < ids.size(); ++i)
            got += (i ? "," : "") + std::to_string(ids[i]);
        if (got != expect) {
            ++wrong;
            if (wrong <= 5)
                std::println("case {}: got {} expected {}", n, got, expect);
        }
        trip += tok->decode(ids) != c;
        ++n;
    }
    std::println("{} cases: {} id mismatches, {} round-trip failures", n, wrong, trip);
    return wrong || trip ? 1 : 0;
}
