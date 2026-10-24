#include "chat/template.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"

#include <format>
#include <stdexcept>

namespace mako::chat {

struct Template::Impl {
    std::string source;
    jinja::program program;
};

Template::Template(std::string source) : impl_(std::make_unique<Impl>())
{
    try {
        jinja::lexer lexer;
        auto tokens = lexer.tokenize(source);
        impl_->program = jinja::parse_from_tokens(tokens);
        impl_->source = std::move(tokens.source);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::format("chat template: {}", e.what()));
    }
}

Template::~Template() = default;
Template::Template(Template&&) noexcept = default;
Template& Template::operator=(Template&&) noexcept = default;

Rendered Template::render(std::string_view context) const
{
    Rendered r;
    try {
        jinja::context ctx(impl_->source);
        jinja::global_from_json(ctx, common_json::parse(std::string(context)), true);
        jinja::runtime runtime(ctx);
        const auto parts = jinja::runtime::gather_string_parts(runtime.execute(impl_->program));
        for (const auto& p : parts->val_str.parts) {
            if (p.is_input && !p.val.empty()) {
                if (!r.input.empty() && r.input.back().second == r.text.size())
                    r.input.back().second += p.val.size();
                else
                    r.input.emplace_back(r.text.size(), r.text.size() + p.val.size());
            }
            r.text += p.val;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::format("chat template: {}", e.what()));
    }
    return r;
}

} // namespace mako::chat
