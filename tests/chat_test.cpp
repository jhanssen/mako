// The model's chat template rendered by the Jinja engine against jinja2 renders (byte-exact), and its input marking:
// control tokens written in message text are not recognized when tokenizing, everything else tokenizes as the
// whole text does.
//   chat_test <model.gguf> <cases file>
// Cases (tools/gen_chat_template_cases.py): context JSON, 0x1F, expected text, 0x1E.
#include "chat/template.hpp"
#include "gguf/gguf.hpp"
#include "text/tokenizer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool overlaps(const mako::chat::Rendered& r, size_t begin, size_t end)
{
    return std::ranges::any_of(r.input, [&](const auto& in) { return in.first < end && begin < in.second; });
}

// Occurrences of `s` in the text, and those outside the input ranges.
std::pair<size_t, size_t> occurrences(const mako::chat::Rendered& r, const std::string& s)
{
    size_t all = 0, outside = 0;
    for (size_t pos = 0; (pos = r.text.find(s, pos)) != std::string::npos; pos += s.size()) {
        ++all;
        outside += !overlaps(r, pos, pos + s.size());
    }
    return {all, outside};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: chat_test <model.gguf> <cases>");
        return 2;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(*f);
    const auto source = f->get_string("tokenizer.chat_template");
    if (!tok || !source) {
        std::println(stderr, "no tokenizer or chat template");
        return 1;
    }
    int failures = 0;
    const auto check = [&](bool ok, const std::string& what) {
        if (!ok) {
            ++failures;
            std::println("FAIL {}", what);
        }
    };

    bool threw = false;
    try {
        mako::chat::Template bad("{% if %}");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "syntax error throws");

    const mako::chat::Template tmpl{std::string(*source)};
    std::ifstream in(argv[2], std::ios::binary);
    const std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::vector<std::string> controls{"<|user|>", "<|assistant|>", "<|system|>", "<|observation|>"};
    std::vector<int32_t> control_ids;
    for (const auto& c : controls)
        control_ids.push_back(tok->find(c));
    size_t cases = 0, injected = 0;
    for (size_t pos = 0, end; (end = all.find('\x1e', pos)) != std::string::npos; pos = end + 1, ++cases) {
        const std::string rec = all.substr(pos, end - pos);
        const size_t sep = rec.find('\x1f');
        const std::string expected = rec.substr(sep + 1);
        const auto r = tmpl.render(std::string_view(rec).substr(0, sep));
        if (r.text != expected) {
            const auto [a, b] = std::ranges::mismatch(r.text, expected);
            check(false, std::format("case {}: text differs at byte {} of {}", cases, a - r.text.begin(),
                                     expected.size()));
            continue;
        }
        // Control tokens count only where the template wrote them; without any in the input the ids are those of
        // the whole text with control tokens parsed.
        const auto marked = tok->encode(r.text, r.input);
        bool any_inside = false;
        for (size_t i = 0; i < controls.size(); ++i) {
            const auto [n_all, n_outside] = occurrences(r, controls[i]);
            any_inside |= n_all != n_outside;
            const auto n_ids = size_t(std::ranges::count(marked, control_ids[i]));
            check(n_ids == n_outside, std::format("case {}: {} {} times as a token, {} outside the input", cases,
                                                  controls[i], n_ids, n_outside));
        }
        injected += any_inside;
        if (!any_inside)
            check(marked == tok->encode(r.text, true), std::format("case {}: ids differ from the whole text's", cases));
    }
    check(cases >= 50, std::format("{} cases read", cases));
    check(injected > 0, "some cases write control tokens in message text");
    std::println("{} cases, {} with control tokens in message text, {} failures", cases, injected, failures);
    return failures ? 1 : 0;
}
