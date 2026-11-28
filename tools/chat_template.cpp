// Prints the chat template each model uses: the GGUF's (tokenizer.chat_template, in the first shard), or what
// --chat-template and --template-edits make of it, as mako_serve would use it.
//   chat_template <model.gguf>... [--chat-template FILE] [--template-edits FILE] [--check]
// More than one model: each template follows a "==> path <==" line. --check also parses the result. The text is
// printed as is, so one model's output redirected to a file is a starting point for --chat-template.
#include "app/cli.hpp"
#include "chat/template.hpp"
#include "chat/template_source.hpp"
#include "gguf/gguf.hpp"

#include <cstdio>
#include <limits>
#include <print>
#include <string>

int main(int argc, char** argv)
{
    mako::chat::TemplateSource src;
    bool check = false;
    const auto models =
        mako::app::Cli("chat_template", "<model.gguf>...", 1, std::numeric_limits<size_t>::max())
            .option("--chat-template", "FILE", "a Jinja file to use instead of the model's template", src.file)
            .option("--template-edits", "FILE", "find/replace pairs (JSON) to apply to the template", src.edits)
            .flag("--check", "parse the resulting template", check)
            .parse(argc, argv);
    int status = 0;
    bool ended_line = true;
    for (const auto& path : models) {
        if (models.size() > 1)
            std::println("{}==> {} <==", ended_line ? "" : "\n", path);
        auto f = mako::gguf::File::open(path);
        if (!f) {
            std::println(stderr, "{}: {}", path, f.error());
            status = 1;
            continue;
        }
        auto t = mako::chat::template_source(f->get_string("tokenizer.chat_template"), src);
        if (!t) {
            std::println(stderr, "{}: {}", path, t.error());
            status = 1;
            continue;
        }
        if (check) {
            try {
                mako::chat::Template parsed{*t};
            } catch (const std::exception& e) {
                std::println(stderr, "{}: {}", path, e.what());
                status = 1;
            }
        }
        // As is: a newline added at the end would be part of the template written to a file.
        std::fwrite(t->data(), 1, t->size(), stdout);
        ended_line = t->empty() || t->back() == '\n';
    }
    return status;
}
