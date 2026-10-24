#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mako::chat {

// A rendered prompt; `input` holds the sorted byte ranges that came from the context (message text, tool
// definitions) rather than the template, where control tokens must not be recognized (Tokenizer::encode).
struct Rendered {
    std::string text;
    std::vector<std::pair<size_t, size_t>> input;
};

// A Jinja chat template (a GGUF's tokenizer.chat_template), parsed once.
class Template {
public:
    // Throws std::runtime_error on a syntax error.
    explicit Template(std::string source);
    ~Template();
    Template(Template&&) noexcept;
    Template& operator=(Template&&) noexcept;

    // context: a JSON object of the template's variables (messages, tools, add_generation_prompt, ...).
    // Throws std::runtime_error on invalid JSON or a template error (raise_exception included).
    Rendered render(std::string_view context) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mako::chat
