// The server's request translation and output parsing: requests render through the model's chat template as
// intended, and an assistant turn written by the template parses back into the same reasoning, answer and tool
// calls (streamed increments adding up to the whole); stop strings cut the answer.
//   server_test <model.gguf>
#include "chat/template.hpp"
#include "gguf/gguf.hpp"
#include "server/openai.hpp"
#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>

#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++failures;
        std::println("FAIL {}", what);
    }
}

// JSON text in a canonical spelling (key order kept), for comparing values.
std::string canonical(std::string_view json)
{
    glz::generic g;
    if (glz::read_json(g, json))
        return "<invalid: " + std::string(json) + ">";
    std::string s;
    (void)glz::write_json(g, s);
    return s;
}

const char* kTools = R"([
  {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{
    "path":{"type":"string"},"limit":{"type":"integer"},"label":{"type":"string"}}}}},
  {"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{
    "command":{"type":"string"},"env":{"type":"object"}}}}}])";

struct Parsed {
    std::string reasoning, content;
    std::vector<mako::server::ToolCall> calls;
    bool stopped = false;
};

// Feeds the tokens of `text` (as generated after the prompt's <think>), returning the parser's results and the
// concatenation of the streamed increments.
std::pair<Parsed, Parsed> parse(const mako::text::Tokenizer& tok, const std::string& text, const char* tools,
                                std::vector<std::string> stop = {})
{
    mako::server::OutputParser p(tok, tools, std::move(stop));
    Parsed streamed;
    const auto add = [&](const mako::server::Delta& d) {
        streamed.reasoning += d.reasoning;
        streamed.content += d.content;
        for (const auto& c : d.calls) {
            if (c.index >= streamed.calls.size())
                streamed.calls.resize(c.index + 1);
            auto& s = streamed.calls[c.index];
            if (!c.id.empty())
                s.id = c.id, s.name = c.name;
            s.arguments += c.arguments;
        }
    };
    for (int32_t t : tok.encode(text, true)) {
        mako::server::Delta d;
        const bool more = p.push(t, d);
        add(d);
        if (!more) {
            streamed.stopped = true;
            break;
        }
    }
    mako::server::Delta d;
    p.finish(d);
    add(d);
    return {Parsed{p.reasoning(), p.content(), p.calls(), streamed.stopped}, streamed};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: server_test <model.gguf>");
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
    const mako::chat::Template tmpl{std::string(*source)};
    const mako::gen::SamplingParams defaults{.temperature = 0.6f, .top_p = 0.9f};

    // Translation: developer becomes system, argument strings become objects, missing content becomes empty,
    // reasoning effort medium becomes high, unset sampling fields keep the defaults.
    const std::string body = std::format(R"({{"model":"m","messages":[
        {{"role":"developer","content":"Be terse."}},
        {{"role":"user","content":"List files"}},
        {{"role":"assistant","content":null,"reasoning":"I should call ls.",
          "tool_calls":[{{"id":"c1","type":"function","function":{{"name":"bash","arguments":"{{\"command\":\"ls -l\",\"env\":{{\"A\":1.0}}}}"}}}}]}},
        {{"role":"tool","tool_call_id":"c1","content":"a.txt"}}],
      "tools":{},"reasoning_effort":"medium","temperature":0.2,"max_tokens":77,"stop":"END","stream":true,
      "stream_options":{{"include_usage":true}},"unknown_field":[1,2]}})",
                                         kTools);
    try {
        const auto r = mako::server::parse_chat_request(body, defaults);
        check(r.gen.sampling.temperature == 0.2f && r.gen.sampling.top_p == 0.9f, "sampling fields and defaults");
        check(r.gen.max_tokens == 77 && r.stop == std::vector<std::string>{"END"} && r.stream && r.include_usage,
              "max_tokens, stop, stream options");
        const auto text = tmpl.render(r.context).text;
        check(text.contains("<|system|>Reasoning Effort: High"), "reasoning effort medium renders as High");
        check(text.contains("<|system|>Be terse.<|user|>List files"), "developer message renders as system");
        check(text.contains("<think>I should call ls.</think>"), "reasoning renders in the assistant turn");
        check(text.contains("<tool_call>bash<arg_key>command</arg_key><arg_value>ls -l</arg_value><arg_key>env"
                            "</arg_key><arg_value>{\"A\": 1.0}</arg_value></tool_call>"),
              "argument string renders as key/value pairs, numbers as sent");
        check(text.contains("<|observation|><tool_response>a.txt</tool_response>"), "tool result");
        check(text.contains("\"name\": \"read\""), "tools render");
        check(text.ends_with("<|assistant|><think>"), "generation prompt");
    } catch (const std::exception& e) {
        check(false, std::format("translation threw: {}", e.what()));
    }
    const auto no_tools = mako::server::parse_chat_request(
        std::format(R"({{"messages":[{{"role":"user","content":"hi"}}],"tools":{},"tool_choice":"none"}})", kTools),
        defaults);
    check(no_tools.tools == "[]" && !tmpl.render(no_tools.context).text.contains("# Tools"),
          "tool_choice none leaves the tools out");
    const auto plain = mako::server::parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})", defaults);
    check(plain.gen.sampling.temperature == 0.6f && plain.gen.max_tokens == 0 && !plain.stream,
          "defaults without fields");
    for (const char* bad : {R"({"messages":[]})", R"({"messages":[{"role":"robot","content":"x"}]})",
                            R"({"messages":[{"role":"user","content":"x"}],"n":2})",
                            R"({"messages":[{"role":"user","content":"x"}],"tools":{"a":1}})",
                            R"({"messages":[{"role":"user","content":"x"}],"tool_choice":"required"})",
                            R"({"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"[1]"}}]}]})",
                            R"({"messages":[{"role":"user","content":"x"}],"temperature":"hot"})", "not json"}) {
        bool threw = false;
        try {
            mako::server::parse_chat_request(bad, defaults);
        } catch (const mako::server::BadRequest&) {
            threw = true;
        }
        check(threw, std::format("rejected: {}", bad));
    }

    // Round trip: an assistant turn as the template writes it parses back into what produced it.
    const std::string args_read = R"({"path": "/tmp/a b.txt", "limit": 20, "label": "42"})";
    const std::string args_bash = R"({"command": "echo \"hi\"\nls", "env": {"A": "1", "B": [true, null]}})";
    const std::string turn = std::format(R"({{"messages":[{{"role":"user","content":"go"}},
        {{"role":"assistant","content":"Checking the file.","reasoning_content":"Line one.\nLine two ü.",
          "tool_calls":[{{"id":"a","type":"function","function":{{"name":"read","arguments":{}}}}},
                        {{"id":"b","type":"function","function":{{"name":"bash","arguments":{}}}}}]}}],
        "tools":{}}})",
                                         args_read, args_bash, kTools);
    const auto rendered = tmpl.render(turn).text;
    const std::string marker = "<|assistant|><think>";
    const auto at = rendered.rfind(marker);
    check(at != std::string::npos, "assistant turn rendered");
    const std::string generated = rendered.substr(at + marker.size()) + "<|observation|>";
    const auto [whole, streamed] = parse(*tok, generated, kTools);
    for (const auto* p : {&whole, &streamed}) {
        const char* which = p == &whole ? "whole" : "streamed";
        check(p->reasoning == "Line one.\nLine two ü.", std::format("{}: reasoning '{}'", which, p->reasoning));
        check(p->content == "Checking the file.", std::format("{}: content '{}'", which, p->content));
        check(p->calls.size() == 2, std::format("{}: {} calls", which, p->calls.size()));
        if (p->calls.size() == 2) {
            check(p->calls[0].name == "read" && p->calls[1].name == "bash", std::format("{}: call names", which));
            check(canonical(p->calls[0].arguments) == canonical(args_read),
                  std::format("{}: read arguments {}", which, p->calls[0].arguments));
            check(canonical(p->calls[1].arguments) == canonical(args_bash),
                  std::format("{}: bash arguments {}", which, p->calls[1].arguments));
            check(!p->calls[0].id.empty() && p->calls[0].id != p->calls[1].id, std::format("{}: call ids", which));
        }
    }

    // Stop strings end the answer, also when split across tokens, and are not part of it.
    for (const char* stop : {"STOP", "P mo", "Hello"}) {
        const auto [w, s] = parse(*tok, "thinking</think>Hello world STOP more text", "[]", {stop});
        const std::string expect = std::string("Hello world STOP more text").substr(
            0, std::string("Hello world STOP more text").find(stop));
        const std::string trimmed = expect.substr(0, expect.find_last_not_of(' ') + 1);
        check(w.content == trimmed && s.content == trimmed && s.stopped,
              std::format("stop '{}': '{}' / '{}'", stop, w.content, s.content));
    }
    const auto [held, held_s] = parse(*tok, "x</think>ends with ST", "[]", {"STOP"});
    check(held.content == "ends with ST" && held_s.content == "ends with ST" && !held_s.stopped,
          "a stop string's start held back is released at the end");
    const auto [cut, cut_s] = parse(*tok, "unfinished reasoning", "[]");
    check(cut.reasoning == "unfinished reasoning" && cut.content.empty(), "reasoning cut off");

    std::println("{} failures", failures);
    return failures ? 1 : 0;
}
