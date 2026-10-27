#pragma once

// The JSON value the Jinja engine reads its context from (jinja::global_from_json in value.cpp), in place of llama.cpp's
// wrapper around nlohmann::json: only what value.cpp uses. Objects keep their keys in the order given (a repeated key
// keeps its first place and takes the last value); numbers without a fraction or exponent that fit 64 bits are
// integers (above INT64_MAX as their two's-complement int64), others doubles.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct common_json_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class common_json {
public:
    // Strict RFC 8259 JSON, valid UTF-8 and at most 512 levels deep; throws common_json_error otherwise.
    static common_json parse(const std::string& text);

    bool is_null() const { return kind_ == Kind::null; }
    bool is_boolean() const { return kind_ == Kind::boolean; }
    bool is_number_integer() const { return kind_ == Kind::integer; }
    bool is_number_float() const { return kind_ == Kind::floating; }
    bool is_string() const { return kind_ == Kind::string; }
    bool is_array() const { return kind_ == Kind::array; }
    bool is_object() const { return kind_ == Kind::object; }

    // bool, int64_t, double or std::string, of a value of that type.
    template <typename T> T get() const;

    // The elements of an array or object in order; key() is an object's key (empty in an array).
    class iterator {
    public:
        iterator(const common_json* j, size_t i) : j_(j), i_(i) {}
        const common_json& operator*() const { return j_->items_[i_]; }
        const common_json* operator->() const { return &j_->items_[i_]; }
        iterator& operator++()
        {
            ++i_;
            return *this;
        }
        bool operator==(const iterator& o) const { return i_ == o.i_; }
        const std::string& key() const { return j_->keys_[i_]; }
        const common_json& value() const { return j_->items_[i_]; }

    private:
        const common_json* j_;
        size_t i_;
    };
    iterator begin() const { return {this, 0}; }
    iterator end() const { return {this, items_.size()}; }

private:
    friend class common_json_parser;
    enum class Kind { null, boolean, integer, floating, string, array, object };
    Kind kind_ = Kind::null;
    bool bool_ = false;
    int64_t int_ = 0;
    double double_ = 0.0;
    std::string string_;
    std::vector<common_json> items_;
    std::vector<std::string> keys_; // an object's, beside items_
};

template <> bool common_json::get<bool>() const;
template <> int64_t common_json::get<int64_t>() const;
template <> double common_json::get<double>() const;
template <> std::string common_json::get<std::string>() const;
