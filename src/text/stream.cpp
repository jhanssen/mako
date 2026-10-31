#include "text/stream.hpp"

#include <utility>

namespace mako::text {

namespace {

// Bytes at the end of s that start a UTF-8 sequence not yet complete.
size_t incomplete_tail(const std::string& s)
{
    const size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const auto c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0) == 0x80)
            continue; // continuation byte
        const size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return len > back ? back : 0;
    }
    return 0;
}

// s with each byte that is not part of a well-formed UTF-8 sequence replaced by U+FFFD.
std::string valid(std::string s)
{
    const auto at = [&](size_t i) { return static_cast<unsigned char>(s[i]); };
    const auto cont = [&](size_t i) { return i < s.size() && (at(i) & 0xC0) == 0x80; };
    std::string out;
    size_t i = 0, good = 0; // s[good, i) is valid and not yet copied
    while (i < s.size()) {
        const unsigned char c = at(i);
        size_t len = 0;
        if (c < 0x80)
            len = 1;
        else if (c >= 0xC2 && c <= 0xDF && cont(i + 1))
            len = 2;
        else if (c >= 0xE0 && c <= 0xEF && cont(i + 1) && cont(i + 2) && !(c == 0xE0 && at(i + 1) < 0xA0) &&
                 !(c == 0xED && at(i + 1) >= 0xA0))
            len = 3;
        else if (c >= 0xF0 && c <= 0xF4 && cont(i + 1) && cont(i + 2) && cont(i + 3) &&
                 !(c == 0xF0 && at(i + 1) < 0x90) && !(c == 0xF4 && at(i + 1) >= 0x90))
            len = 4;
        if (len) {
            i += len;
            continue;
        }
        out.append(s, good, i - good);
        out += "\xEF\xBF\xBD";
        good = ++i;
    }
    if (good == 0)
        return s;
    out.append(s, good, s.size() - good);
    return out;
}

} // namespace

std::string Utf8Stream::push(int32_t token)
{
    const int32_t id[1] = {token};
    held_ += tok_.decode(id, special_);
    const size_t keep = incomplete_tail(held_);
    std::string out = held_.substr(0, held_.size() - keep);
    held_.erase(0, held_.size() - keep);
    return valid(std::move(out));
}

std::string Utf8Stream::flush()
{
    return valid(std::exchange(held_, {}));
}

} // namespace mako::text
