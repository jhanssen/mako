#include "chat/template_source.hpp"

#include <glaze/glaze.hpp>

#include <format>
#include <fstream>
#include <iterator>
#include <vector>

namespace mako::chat {

// Outside the anonymous namespace: glaze's reflection needs a type with linkage.
struct TemplateEdit {
    std::string find, replace;
};

namespace {

std::expected<std::string, std::string> read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("cannot open {}", path));
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

std::expected<std::string, std::string> apply_edits(std::string source, std::string_view edits)
{
    std::vector<TemplateEdit> list;
    if (auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(list, edits))
        return std::unexpected(std::format("template edits: {}", glz::format_error(ec, edits)));
    for (size_t i = 0; i < list.size(); ++i) {
        const auto& e = list[i];
        if (e.find.empty())
            return std::unexpected(std::format("template edit {}: an empty find", i + 1));
        const size_t at = source.find(e.find);
        size_t n = 0;
        for (size_t p = at; p != std::string::npos; p = source.find(e.find, p + e.find.size()))
            ++n;
        if (n != 1)
            return std::unexpected(std::format("template edit {}: its find occurs {} times in the template (needs "
                                               "exactly one): {:.60}",
                                               i + 1, n, e.find));
        source.replace(at, e.find.size(), e.replace);
    }
    return source;
}

std::expected<std::string, std::string> template_source(std::optional<std::string_view> model, const TemplateSource& s)
{
    std::string source;
    if (!s.file.empty()) {
        auto f = read_file(s.file);
        if (!f)
            return std::unexpected(f.error());
        source = std::move(*f);
    } else if (model) {
        source = std::string(*model);
    } else {
        return std::unexpected("the model has no chat template");
    }
    if (s.edits.empty())
        return source;
    auto e = read_file(s.edits);
    if (!e)
        return std::unexpected(e.error());
    auto r = apply_edits(std::move(source), *e);
    if (!r)
        return std::unexpected(std::format("{}: {}", s.edits, r.error()));
    return r;
}

} // namespace mako::chat
