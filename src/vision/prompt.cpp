#include "vision/prompt.hpp"

#include <format>
#include <stdexcept>

namespace mako::vision {

std::string_view placeholder(Projector kind)
{
    return kind == Projector::deepseek41v ? "<｜deepseek_image｜>" : "<|image|>";
}

ImagePrompt expand_images(std::span<const int32_t> tokens, int32_t placeholder, std::span<const uint32_t> counts,
                          std::span<const uint64_t> keys)
{
    ImagePrompt out;
    size_t next = 0;
    for (const int32_t t : tokens) {
        if (t != placeholder) {
            out.tokens.push_back(t);
            continue;
        }
        if (next >= counts.size())
            throw std::runtime_error(std::format("the prompt has more image placeholders than the {} images", counts.size()));
        out.rows.push_back({{uint32_t(out.tokens.size()), counts[next], nullptr}, keys[next]});
        out.tokens.insert(out.tokens.end(), counts[next], placeholder);
        ++next;
    }
    if (next != counts.size())
        throw std::runtime_error(std::format("{} images but {} placeholders in the prompt", counts.size(), next));
    return out;
}

} // namespace mako::vision
