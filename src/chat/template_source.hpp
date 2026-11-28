#pragma once

#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace mako::chat {

// Where a tool's chat template comes from: `file` (a Jinja file) instead of the model's, then `edits` applied.
struct TemplateSource {
    std::string file, edits;
};

// edits: a JSON array of {"find": "...", "replace": "..."}, applied in order; each find must occur exactly once in the
// text it is applied to, so an edit that no longer matches (another model, a changed template) is an error.
std::expected<std::string, std::string> apply_edits(std::string source, std::string_view edits);

// The template to use: `model`'s (absent: the model has none), or s.file's text, with s.edits' file applied.
std::expected<std::string, std::string> template_source(std::optional<std::string_view> model, const TemplateSource& s);

} // namespace mako::chat
