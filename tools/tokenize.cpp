// Tokenizes text with a model's tokenizer.
//   tokenize <model.gguf> <text>              prints the ids, then the decoded text
//   tokenize <model.gguf> --cases <file>      one line of ids per 0x1E-separated case of <file>
//   tokenize <model.gguf> --roundtrip <file>  checks decode(encode(case)) == case for every case
#include "gguf/gguf.hpp"
#include "text/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <print>
#include <sstream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: tokenize <model.gguf> <text> | --cases <file>");
        return 2;
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
    auto line = [](const std::vector<int32_t>& ids) {
        std::string s;
        for (size_t i = 0; i < ids.size(); ++i)
            s += (i ? "," : "") + std::to_string(ids[i]);
        return s;
    };
    if (std::string(argv[2]) == "--roundtrip" && argc > 3) {
        std::ifstream in(argv[3], std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::stringstream cases(all);
        int n = 0, bad = 0;
        for (std::string c; std::getline(cases, c, '\x1e'); ++n)
            if (tok->decode(tok->encode(c, true)) != c) {
                ++bad;
                std::println("mismatch in case {}", n);
            }
        std::println("{} of {} cases round-trip", n - bad, n);
        return bad ? 1 : 0;
    }
    if (std::string(argv[2]) == "--cases" && argc > 3) {
        std::ifstream in(argv[3], std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::stringstream cases(all);
        for (std::string c; std::getline(cases, c, '\x1e');)
            std::println("{}", line(tok->encode(c, true)));
        return 0;
    }
    const auto ids = tok->encode(argv[2], true);
    std::println("{}", line(ids));
    std::println("{}", tok->decode(ids));
    return 0;
}
