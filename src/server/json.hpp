#pragma once

// JSON helpers shared by the protocol translations (glaze; include in .cpp files only).
#include <glaze/glaze.hpp>

#include <string>
#include <string_view>

namespace mako::server::json {

template <class T>
std::string write(const T& v)
{
    std::string s;
    (void)glz::write_json(v, s);
    return s;
}

inline std::string quoted(std::string_view s)
{
    return write(s);
}

inline bool is_space(char c)
{
    return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

inline std::string_view trim(std::string_view s)
{
    while (!s.empty() && is_space(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && is_space(s.back()))
        s.remove_suffix(1);
    return s;
}

// A JSON object's text (as tool call arguments must be).
inline bool is_object_text(std::string_view s)
{
    s = trim(s);
    return !s.empty() && s.front() == '{' && !glz::validate_json(s);
}

} // namespace mako::server::json
