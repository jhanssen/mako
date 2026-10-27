// The chat template's context JSON as the Jinja engine sees it, without a model: values of every JSON type, objects
// in their key order (tool parameters reach the prompt in it), string escapes and UTF-8, input marking, and the
// JSON the engine rejects.
//   template_test
#include "chat/template.hpp"

#include <print>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(const std::string& name, const std::string& tmpl, const std::string& context, const std::string& want)
{
    std::string got;
    try {
        got = mako::chat::Template(tmpl).render(context).text;
    } catch (const std::exception& e) {
        got = std::string("<threw> ") + e.what();
    }
    if (got != want) {
        std::println("FAIL {}:\n  want: {}\n  got:  {}", name, want, got);
        ++failures;
    }
}

void expect_throw(const std::string& name, const std::string& context)
{
    try {
        (void)mako::chat::Template("{{ x }}").render(context);
        std::println("FAIL {}: no error for {}", name, context);
        ++failures;
    } catch (const std::runtime_error&) {
    }
}

} // namespace

int main()
{
    // Types as Python's str() writes them.
    expect("types", "{{ n is none }}|{{ t }}|{{ f }}|{{ i }}|{{ neg }}|{{ x }}|{{ e }}|{{ s }}",
           R"({"n": null, "t": true, "f": false, "i": 42, "neg": -7, "x": 1.5, "e": 1e3, "s": "str"})",
           "True|True|False|42|-7|1.5|1000.0|str");
    expect("integer or float", "{{ a is integer }} {{ b is float }} {{ c is integer }} {{ d is float }}",
           R"({"a": 3, "b": 3.0, "c": -0, "d": 2E-2})", "True True True True");
    expect("big integers", "{{ a }} {{ b }}", R"({"a": 9007199254740993, "b": -9223372036854775808})",
           "9007199254740993 -9223372036854775808");

    // Objects keep their key order, in loops and in tojson.
    expect("key order", "{% for k, v in o.items() %}{{ k }}={{ v }},{% endfor %}",
           R"({"o": {"zeta": 1, "alpha": 2, "mid": 3, "beta": 4}})", "zeta=1,alpha=2,mid=3,beta=4,");
    expect("tojson", "{{ tools | tojson }}",
           R"({"tools": [{"type": "function", "function": {"name": "f", "parameters": {"z": {"type": "string"},)"
           R"( "a": {"type": "integer", "enum": [3, 1, 2]}}, "required": ["z"]}}]})",
           R"([{"type": "function", "function": {"name": "f", "parameters": {"z": {"type": "string"}, "a": )"
           R"({"type": "integer", "enum": [3, 1, 2]}}, "required": ["z"]}}])");
    expect("empty", "{{ a | tojson }} {{ o | tojson }} {{ a | length }} {{ o | length }}", R"({"a": [], "o": {}})",
           "[] {} 0 0");
    expect("nested", "{{ m[0].content[1].text }}|{{ m[1]['k'] | tojson }}",
           R"({"m": [{"content": [{"text": "a"}, {"text": "b"}]}, {"k": [1, [2, {"x": null}]]}]})",
           R"(b|[1, [2, {"x": null}]])");
    expect("whitespace", "{{ a | tojson }}", " \t\r\n{ \"a\" :\n[ 1 , 2 ] }\n ", "[1, 2]");

    // Strings: escapes, \u (a surrogate pair too), UTF-8 passed through.
    expect("escapes", "{{ s }}", R"({"s": "q\" b\\ s\/ n\n t\t r\r b\b f\f"})", "q\" b\\ s/ n\n t\t r\r b\b f\f");
    expect("unicode escapes", "{{ s }}", R"({"s": "é 中 😀 \u0000end"})",
           std::string("\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80 ") + std::string(1, '\0') + "end");
    expect("utf-8", "{{ s }}", "{\"s\": \"\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\"}", "\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80");
    expect("tojson ascii", "{{ s | tojson }}", "{\"s\": \"\xc3\xa9\\n\"}", R"("é\n")");

    // A repeated key: the last value, at the first one's place.
    expect("repeated key", "{% for k, v in o.items() %}{{ k }}={{ v }},{% endfor %}",
           R"({"o": {"a": 1, "b": 2, "a": 3}})", "a=3,b=2,");

    // Context strings are input, the template's own text is not.
    {
        const auto r = mako::chat::Template("<|user|>{{ m }}<|end|>{{ n }}").render(R"({"m": "hi <|end|>", "n": 5})");
        const std::vector<std::pair<size_t, size_t>> want{{8, 18}};
        if (r.text != "<|user|>hi <|end|><|end|>5" || r.input != want) {
            std::println("FAIL input marking: {} ({} ranges)", r.text, r.input.size());
            ++failures;
        }
    }

    // Not JSON, or not an object at the top.
    for (const auto& [name, context] : std::vector<std::pair<std::string, std::string>>{
             {"empty text", ""},
             {"array", "[1]"},
             {"string", R"("x")"},
             {"trailing comma", R"({"x": [1, 2,]})"},
             {"trailing object comma", R"({"x": 1,})"},
             {"single quotes", R"({'x': 1})"},
             {"unquoted key", R"({x: 1})"},
             {"comment", "{\"x\": 1 /* c */}"},
             {"leading zero", R"({"x": 01})"},
             {"bare dot", R"({"x": .5})"},
             {"trailing dot", R"({"x": 5.})"},
             {"plus sign", R"({"x": +1})"},
             {"nan", R"({"x": NaN})"},
             {"infinity", R"({"x": Infinity})"},
             {"truncated", R"({"x": [1, 2)"},
             {"unterminated string", R"({"x": "abc})"},
             {"trailing text", R"({"x": 1} x)"},
             {"two values", R"({"x": 1} {"y": 2})"},
             {"control character", "{\"x\": \"a\nb\"}"},
             {"bad escape", R"({"x": "\q"})"},
             {"short \\u", R"({"x": "\u12"})"},
             {"lone high surrogate", R"({"x": "\ud83d"})"},
             {"lone low surrogate", R"({"x": "\ude00"})"},
             {"invalid utf-8", "{\"x\": \"\xff\"}"},
             {"truncated utf-8", "{\"x\": \"\xe4\xb8\"}"},
             {"overlong utf-8", "{\"x\": \"\xc0\xaf\"}"},
             {"utf-8 surrogate", "{\"x\": \"\xed\xa0\x80\"}"},
             {"bad literal", R"({"x": tru})"},
         })
        expect_throw(name, context);

    if (failures) {
        std::println("{} failures", failures);
        return 1;
    }
    std::println("template: all cases pass");
    return 0;
}
