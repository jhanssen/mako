#pragma once

#include "text/tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mako::text {

// Detokenizes a token stream into whole UTF-8 characters: a character split across tokens is held back until
// its last byte arrives.
class Utf8Stream {
public:
    explicit Utf8Stream(const Tokenizer& tok, bool special = true) : tok_(tok), special_(special) {}

    // The text that token completes (possibly empty).
    std::string push(int32_t token);
    // Whatever is held back, invalid sequences included (at the end of the stream).
    std::string flush();

private:
    const Tokenizer& tok_;
    bool special_;
    std::string held_;
};

} // namespace mako::text
