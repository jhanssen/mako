#include "json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <format>
#include <string_view>
#include <unordered_map>

class common_json_parser {
public:
    explicit common_json_parser(std::string_view s) : s_(s) {}

    common_json document()
    {
        common_json j = value(0);
        space();
        if (p_ != s_.size())
            fail("text after the value");
        return j;
    }

private:
    static constexpr int max_depth = 512;

    [[noreturn]] void fail(std::string_view what) const
    {
        throw common_json_error(std::format("JSON: {} at byte {}", what, p_));
    }

    void space()
    {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r'))
            ++p_;
    }

    bool take(char c)
    {
        if (p_ < s_.size() && s_[p_] == c) {
            ++p_;
            return true;
        }
        return false;
    }

    common_json value(int depth)
    {
        if (depth > max_depth)
            fail("nested too deep");
        space();
        if (p_ == s_.size())
            fail("missing value");
        common_json j;
        const char c = s_[p_];
        if (c == '{') {
            ++p_;
            j.kind_ = common_json::Kind::object;
            std::unordered_map<std::string, size_t> index;
            space();
            if (take('}'))
                return j;
            do {
                space();
                if (p_ == s_.size() || s_[p_] != '"')
                    fail("expected a key");
                std::string key = string();
                space();
                if (!take(':'))
                    fail("expected ':'");
                common_json v = value(depth + 1);
                if (auto it = index.find(key); it != index.end()) {
                    j.items_[it->second] = std::move(v);
                } else {
                    index.emplace(key, j.items_.size());
                    j.keys_.push_back(std::move(key));
                    j.items_.push_back(std::move(v));
                }
                space();
            } while (take(','));
            if (!take('}'))
                fail("expected ',' or '}'");
        } else if (c == '[') {
            ++p_;
            j.kind_ = common_json::Kind::array;
            space();
            if (take(']'))
                return j;
            do {
                j.items_.push_back(value(depth + 1));
                j.keys_.emplace_back();
                space();
            } while (take(','));
            if (!take(']'))
                fail("expected ',' or ']'");
        } else if (c == '"') {
            j.kind_ = common_json::Kind::string;
            j.string_ = string();
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            number(j);
        } else if (literal("true")) {
            j.kind_ = common_json::Kind::boolean;
            j.bool_ = true;
        } else if (literal("false")) {
            j.kind_ = common_json::Kind::boolean;
        } else if (!literal("null")) {
            fail("invalid value");
        }
        return j;
    }

    bool literal(std::string_view word)
    {
        if (s_.substr(p_, word.size()) != word)
            return false;
        p_ += word.size();
        return true;
    }

    bool digits()
    {
        const size_t start = p_;
        while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9')
            ++p_;
        return p_ > start;
    }

    void number(common_json& j)
    {
        const size_t start = p_;
        const bool negative = take('-');
        if (take('0')) {
            if (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9')
                fail("leading zero");
        } else if (!digits()) {
            fail("invalid number");
        }
        bool integer = true;
        if (take('.')) {
            integer = false;
            if (!digits())
                fail("invalid number");
        }
        if (take('e') || take('E')) {
            integer = false;
            if (!take('+'))
                take('-');
            if (!digits())
                fail("invalid number");
        }
        const std::string text(s_.substr(start, p_ - start));
        if (integer) {
            errno = 0;
            char* end = nullptr;
            if (negative) {
                const long long v = std::strtoll(text.c_str(), &end, 10);
                if (errno == 0) {
                    j.kind_ = common_json::Kind::integer;
                    j.int_ = v;
                    return;
                }
            } else {
                const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
                if (errno == 0) {
                    j.kind_ = common_json::Kind::integer;
                    j.int_ = static_cast<int64_t>(v);
                    return;
                }
            }
        }
        // A fraction, an exponent or an integer beyond 64 bits: a double, which must be finite.
        const double v = std::strtod(text.c_str(), nullptr);
        if (!std::isfinite(v))
            fail("number out of range");
        j.kind_ = common_json::Kind::floating;
        j.double_ = v;
    }

    unsigned hex4()
    {
        if (s_.size() - p_ < 4)
            fail("short \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[p_++];
            v <<= 4;
            if (c >= '0' && c <= '9')
                v |= unsigned(c - '0');
            else if (c >= 'a' && c <= 'f')
                v |= unsigned(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                v |= unsigned(c - 'A' + 10);
            else
                fail("invalid \\u escape");
        }
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp)
    {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    // One UTF-8 sequence starting at a byte >= 0x80, copied after checking it (no overlong forms, surrogates or
    // code points above U+10FFFF).
    void utf8(std::string& out)
    {
        const auto b = [&](size_t i) { return static_cast<unsigned char>(s_[i]); };
        const unsigned lead = b(p_);
        size_t n;
        unsigned cp, min;
        if (lead >= 0xC2 && lead <= 0xDF) {
            n = 1, cp = lead & 0x1F, min = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            n = 2, cp = lead & 0x0F, min = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            n = 3, cp = lead & 0x07, min = 0x10000;
        } else {
            fail("invalid UTF-8");
        }
        if (s_.size() - p_ <= n)
            fail("invalid UTF-8");
        for (size_t i = 1; i <= n; ++i) {
            if ((b(p_ + i) & 0xC0) != 0x80)
                fail("invalid UTF-8");
            cp = (cp << 6) | (b(p_ + i) & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            fail("invalid UTF-8");
        out.append(s_.substr(p_, n + 1));
        p_ += n + 1;
    }

    std::string string()
    {
        ++p_; // the opening quote
        std::string out;
        for (;;) {
            if (p_ == s_.size())
                fail("unterminated string");
            const auto c = static_cast<unsigned char>(s_[p_]);
            if (c == '"') {
                ++p_;
                return out;
            }
            if (c < 0x20)
                fail("control character in a string");
            if (c >= 0x80) {
                utf8(out);
                continue;
            }
            ++p_;
            if (c != '\\') {
                out += char(c);
                continue;
            }
            if (p_ == s_.size())
                fail("unterminated string");
            switch (s_[p_++]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xDC00 && cp <= 0xDFFF)
                    fail("lone low surrogate");
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (!literal("\\u"))
                        fail("lone high surrogate");
                    const unsigned low = hex4();
                    if (low < 0xDC00 || low > 0xDFFF)
                        fail("lone high surrogate");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                }
                append_utf8(out, cp);
                break;
            }
            default: fail("invalid escape");
            }
        }
    }

    std::string_view s_;
    size_t p_ = 0;
};

common_json common_json::parse(const std::string& text)
{
    return common_json_parser(text).document();
}

template <> bool common_json::get<bool>() const
{
    if (kind_ != Kind::boolean)
        throw common_json_error("JSON: not a boolean");
    return bool_;
}

template <> int64_t common_json::get<int64_t>() const
{
    if (kind_ != Kind::integer)
        throw common_json_error("JSON: not an integer");
    return int_;
}

template <> double common_json::get<double>() const
{
    if (kind_ == Kind::integer)
        return double(int_);
    if (kind_ != Kind::floating)
        throw common_json_error("JSON: not a number");
    return double_;
}

template <> std::string common_json::get<std::string>() const
{
    if (kind_ != Kind::string)
        throw common_json_error("JSON: not a string");
    return string_;
}
