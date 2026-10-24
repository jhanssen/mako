// Tokenizer against golden ids recorded from the reference tokenizer, plus decode round trips, token by token
// through Utf8Stream too (whole characters only, the same text).
//   tokenizer_test <model.gguf> <cases file> <expected ids file>
// Cases are separated by 0x1E; the expected file has one comma-separated line of ids per case.
#include "gguf/gguf.hpp"
#include "text/stream.hpp"
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
    // Whether s is whole UTF-8 sequences (lead and continuation bytes only; enough to catch a split character).
    auto whole = [](const std::string& s) {
        for (size_t i = 0; i < s.size();) {
            const auto c = static_cast<unsigned char>(s[i]);
            const size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
            if (len == 0 || i + len > s.size())
                return false;
            for (size_t j = 1; j < len; ++j)
                if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80)
                    return false;
            i += len;
        }
        return true;
    };
    int n = 0, wrong = 0, trip = 0, stream = 0;
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
        mako::text::Utf8Stream us(*tok);
        std::string streamed;
        bool pieces_whole = true;
        for (int32_t id : ids) {
            const std::string piece = us.push(id);
            pieces_whole = pieces_whole && whole(piece);
            streamed += piece;
        }
        streamed += us.flush();
        stream += streamed != c || (whole(c) && !pieces_whole);
        ++n;
    }
    std::println("{} cases: {} id mismatches, {} round-trip failures, {} streaming failures", n, wrong, trip, stream);
    return wrong || trip || stream ? 1 : 0;
}
