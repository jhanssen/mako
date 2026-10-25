// Renders a model's chat template (tokenizer.chat_template) with a JSON context and prints the text.
//   render_template <model.gguf> <context.json>
#include "chat/template.hpp"
#include "gguf/gguf.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <print>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: render_template <model.gguf> <context.json>");
        return 2;
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    const auto source = f->get_string("tokenizer.chat_template");
    if (!source) {
        std::println(stderr, "no chat template");
        return 1;
    }
    std::ifstream in(argv[2], std::ios::binary);
    const std::string ctx((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    try {
        const mako::chat::Template tmpl{std::string(*source)};
        const auto r = tmpl.render(ctx);
        std::fwrite(r.text.data(), 1, r.text.size(), stdout);
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
    return 0;
}
