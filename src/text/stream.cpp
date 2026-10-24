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

} // namespace

std::string Utf8Stream::push(int32_t token)
{
    const int32_t id[1] = {token};
    held_ += tok_.decode(id, special_);
    const size_t keep = incomplete_tail(held_);
    std::string out = held_.substr(0, held_.size() - keep);
    held_.erase(0, held_.size() - keep);
    return out;
}

std::string Utf8Stream::flush()
{
    return std::exchange(held_, {});
}

} // namespace mako::text
