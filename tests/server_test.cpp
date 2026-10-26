// The server's request translation and output parsing: requests render through the model's chat template as
// intended, and an assistant turn written by the template parses back into the same reasoning, answer and tool
// calls (streamed increments adding up to the whole); stop strings cut the answer.
//   server_test <model.gguf>
#include "chat/template.hpp"
#include "gguf/gguf.hpp"
#include "server/anthropic.hpp"
#include "server/openai.hpp"
#include "server/server.hpp"
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
                                std::vector<std::string> stop = {}, bool reasoning_first = true)
{
    mako::server::OutputParser p(tok, tools, std::move(stop), reasoning_first);
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

// DeepSeek-V4.1's markup: reasoning only when the prompt opened it, DSML tool calls.
int deepseek(const mako::text::Tokenizer& tok, const mako::chat::Template& tmpl)
{
    {
        const auto [p, s] = parse(tok, "Weighing it.</think>The answer is 4.", kTools);
        check(p.reasoning == "Weighing it." && p.content == "The answer is 4." && p.calls.empty(), "reasoning, answer");
        check(s.reasoning == p.reasoning && s.content == p.content, "streamed reasoning and answer");
    }
    {
        const auto [p, s] = parse(tok, "Just the answer.", kTools, {}, false);
        check(p.reasoning.empty() && p.content == "Just the answer.", "no reasoning when the prompt did not open it");
    }
    const std::string call =
        "Let me look.</think>Reading it.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter "
        "name=\"path\" string=\"true\">/tmp/a \"b\".txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" "
        "string=\"false\">10</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ "
        "parameter name=\"env\" string=\"false\">{\"A\": 1}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    {
        const auto [p, s] = parse(tok, call, kTools);
        check(p.content == "Reading it." && p.calls.size() == 2, std::format("DSML calls ({} calls, content '{}')",
                                                                             p.calls.size(), p.content));
        if (p.calls.size() == 2) {
            check(p.calls[0].name == "read" && p.calls[0].arguments == R"({"path": "/tmp/a \"b\".txt", "limit": 10})",
                  "first call's arguments: " + p.calls[0].arguments);
            check(p.calls[1].name == "bash" && p.calls[1].arguments == R"({"env": {"A": 1}})",
                  "second call's arguments: " + p.calls[1].arguments);
        }
        check(s.content == p.content && s.calls.size() == p.calls.size() &&
                  (s.calls.empty() || (s.calls[0].arguments == p.calls[0].arguments && s.calls[0].name == "read")),
              "streamed calls match");
    }
    {
        // A call cut off inside its parameters closes what was complete.
        const auto [p, s] = parse(tok, call.substr(0, call.find("</｜DSML｜ invoke>")), kTools);
        check(p.calls.size() == 1 && p.calls[0].arguments == R"({"path": "/tmp/a \"b\".txt", "limit": 10})",
              "cut-off call closed");
    }
    {
        mako::server::OutputParser p(tok, kTools, {}, false);
        p.require_call("read");
        const auto want = tok.encode("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">", true);
        std::vector<int32_t> got;
        for (int32_t f; (f = p.forced()) >= 0;) {
            got.push_back(f);
            mako::server::Delta d;
            p.push(f, d);
        }
        check(got == want, std::format("required call forces its opening ({} of {} tokens)", got.size(), want.size()));
    }
    // The template's generation prompt and DSML for earlier calls.
    const std::string body = R"({"model":"m","tools":[{"type":"function","function":{"name":"read","parameters":{"type":
        "object","properties":{"path":{"type":"string"}}}}}],"messages":[{"role":"user","content":"Read a"},
        {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read",
        "arguments":"{\"path\":\"a\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"hello"}]})";
    try {
        const auto r = mako::server::parse_chat_request(body, {});
        const std::string text = tmpl.render(r.context).text;
        check(text.contains("<｜DSML｜ invoke name=\"read\">") && text.contains("<tool_result>hello</tool_result>"),
              "earlier call and its result render");
        check(text.ends_with("<｜Assistant｜><think>") || text.ends_with("<｜Assistant｜></think>"), "generation prompt");
    } catch (const std::exception& e) {
        check(false, std::format("translation threw: {}", e.what()));
    }
    // Thinking controls: the template takes numbers and medium and can skip the reasoning.
    {
        using mako::server::BadRequest;
        auto t = mako::server::thinking_support(tmpl);
        check(t.can_disable && t.numeric && t.medium, "template thinking support found by rendering");
        const auto chat = [&](const std::string& extra) {
            return tmpl
                .render(mako::server::parse_chat_request(
                            std::format(R"({{"messages":[{{"role":"user","content":"hi"}}]{}}})", extra), {}, t)
                            .context)
                .text;
        };
        const auto msgs = [&](const std::string& extra) {
            return tmpl
                .render(mako::server::parse_messages_request(
                            std::format(R"({{"max_tokens":10,"messages":[{{"role":"user","content":"hi"}}]{}}})",
                                        extra),
                            {}, false, t)
                            .context)
                .text;
        };
        const auto no_thinking = [](const std::string& text) {
            return text.ends_with("<｜Assistant｜></think>") && !text.contains("Reasoning Effort");
        };
        check(chat("").contains("Reasoning Effort: 75") && chat("").ends_with("<think>"), "no effort: the default");
        check(no_thinking(chat(R"(,"reasoning_effort":"none")")) && no_thinking(chat(R"(,"reasoning_effort":0)")),
              "effort none or 0 skips the reasoning");
        check(chat(R"(,"reasoning_effort":"medium")").contains("Reasoning Effort: 62"), "medium is its own level");
        check(chat(R"(,"reasoning_effort":20)").contains("Reasoning Effort: 20"), "numeric effort");
        check(chat(R"(,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true})")
                  .contains("Reasoning Effort: 75"),
              "thinking keys in chat_template_kwargs replace the effort");
        bool threw = false;
        try {
            chat(R"(,"reasoning_effort":101)");
        } catch (const BadRequest&) {
            threw = true;
        }
        check(threw, "effort above 100 rejected");
        check(no_thinking(msgs("")) && no_thinking(msgs(R"(,"thinking":{"type":"disabled"})")),
              "Anthropic: no thinking requested, none rendered");
        check(msgs(R"(,"thinking":{"type":"enabled","budget_tokens":2000})").contains("Reasoning Effort: 50"),
              "Anthropic: a small budget is low");
        t.default_effort = mako::server::Effort(int64_t(30));
        check(chat("").contains("Reasoning Effort: 30") &&
                  msgs(R"(,"thinking":{"type":"enabled","budget_tokens":32000})").contains("Reasoning Effort: 30"),
              "the server's default effort for requests naming none");
        check(chat(R"(,"reasoning_effort":"high")").contains("Reasoning Effort: 75"), "a request's effort wins");
    }
    std::println("{} failures", failures);
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: server_test <model.gguf>");
        return 2;
    }
    {
        using mako::server::key_matches;
        const std::vector<std::string> keys{"s3cret-one-0123456", "s3cret-two-0123456"};
        check(key_matches("Bearer s3cret-two-0123456", "", keys), "an API key as a bearer token (the second of two)");
        check(key_matches("", "s3cret-one-0123456", keys), "an API key as x-api-key");
        check(!key_matches("Bearer s3cret-one-012345", "", keys) &&
                  !key_matches("Bearer s3cret-one-01234567", "", keys),
              "a shorter or longer key is refused");
        check(!key_matches("s3cret-one-0123456", "", keys), "the Authorization header needs the Bearer scheme");
        check(!key_matches("Bearer wrong", "wrong", keys) && !key_matches("", "", keys),
              "a wrong or missing key is refused");
        check(!key_matches("Bearer ", "", {""}) && !key_matches("Bearer x", "x", {}), "no or empty keys match nothing");
        using mako::server::host_allowed, mako::server::origin_allowed;
        check(host_allowed("localhost:8080") && host_allowed("127.0.0.1") && host_allowed("[::1]:8080") &&
                  host_allowed(""),
              "loopback Host headers (and none)");
        check(!host_allowed("evil.example.com") && !host_allowed("localhost.evil.com:8080") &&
                  !host_allowed("192.168.1.5:8080"),
              "other Host headers refused without a key");
        check(origin_allowed("") && origin_allowed("http://localhost:3000") && origin_allowed("https://127.0.0.1") &&
                  origin_allowed("http://[::1]:8080"),
              "loopback Origins (and none)");
        check(!origin_allowed("https://evil.example.com") && !origin_allowed("http://localhost.evil.com") &&
                  !origin_allowed("null") && !origin_allowed("file://"),
              "other Origins refused without a key");
    }
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return failures ? 1 : 77;
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
    if (f->get_string("general.architecture").value_or("") == "deepseek41")
        return deepseek(*tok, tmpl);
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
    // Template variables from the request: clear_thinking drops the reasoning of turns before the last user message.
    const char* earlier = R"({"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b",
        "reasoning_content":"old thoughts"},{"role":"user","content":"c"}]%s})";
    auto with = [&](const char* extra) {
        std::string b = earlier;
        b.replace(b.find("%s"), 2, extra);
        return tmpl.render(mako::server::parse_chat_request(b, defaults).context).text;
    };
    check(with("").contains("<think>old thoughts</think>"), "earlier reasoning kept by default");
    check(!with(R"(,"chat_template_kwargs":{"clear_thinking": true})").contains("old thoughts"),
          "chat_template_kwargs clear_thinking drops it");
    for (const char* bad : {R"(,"chat_template_kwargs":{"messages":[]})", R"(,"chat_template_kwargs":[1])"}) {
        bool threw = false;
        try {
            with(bad);
        } catch (const mako::server::BadRequest&) {
            threw = true;
        }
        check(threw, std::format("rejected kwargs: {}", bad));
    }
    const auto plain = mako::server::parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})", defaults);
    check(plain.gen.sampling.temperature == 0.6f && plain.gen.max_tokens == 0 && !plain.stream,
          "defaults without fields");
    for (const char* bad : {R"({"messages":[]})", R"({"messages":[{"role":"robot","content":"x"}]})",
                            R"({"messages":[{"role":"user","content":"x"}],"n":2})",
                            R"({"messages":[{"role":"user","content":"x"}],"tools":{"a":1}})",
                            R"({"messages":[{"role":"user","content":"x"}],"tool_choice":"required"})",
                            R"({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"a"}}],"tool_choice":{"type":"function","function":{"name":"b"}}})",
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

    // A required call: <tool_call> forced right after the reasoning; a named one also forces the name, then the
    // start of the arguments.
    {
        const auto req = mako::server::parse_chat_request(
            std::format(R"({{"messages":[{{"role":"user","content":"x"}}],"tools":{},"tool_choice":{{"type":"function","function":{{"name":"bash"}}}}}})",
                        kTools),
            defaults);
        check(req.tool_required && req.tool_name == "bash", "named tool_choice parsed");
        mako::server::OutputParser p(*tok, req.tools, {});
        p.require_call(req.tool_name);
        std::vector<int32_t> forced;
        mako::server::Delta d;
        check(p.forced() < 0, "nothing forced while reasoning");
        for (int32_t t : tok->encode("some thought</think>", true))
            p.push(t, d);
        for (int32_t f; (f = p.forced()) >= 0;) {
            forced.push_back(f);
            p.push(f, d);
        }
        const auto want = tok->encode("<tool_call>bash<arg_key>", true);
        check(forced == want, std::format("forced {} tokens, want {}", forced.size(), want.size()));
        for (int32_t t : tok->encode("command</arg_key><arg_value>ls</arg_value></tool_call>", true))
            p.push(t, d);
        p.finish(d);
        check(p.calls().size() == 1 && p.calls()[0].name == "bash" && canonical(p.calls()[0].arguments) == canonical(R"({"command":"ls"})"),
              "the forced call parses");
        const auto any = mako::server::parse_chat_request(
            std::format(R"({{"messages":[{{"role":"user","content":"x"}}],"tools":{},"tool_choice":"required"}})", kTools), defaults);
        mako::server::OutputParser q(*tok, any.tools, {});
        q.require_call();
        for (int32_t t : tok->encode("x</think>", true))
            q.push(t, d);
        check(any.tool_required && any.tool_name.empty() && q.forced() == tok->find("<tool_call>"), "required forces <tool_call>");
        q.push(q.forced(), d);
        check(q.forced() < 0, "required leaves the name to the model");
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

    // Anthropic Messages: blocks translate to the template's turns the same way.
    {
        const std::string body = R"({"model":"m","max_tokens":100,"system":[{"type":"text","text":"Be terse."}],
          "tools":[{"name":"bash","description":"Run","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}],
          "messages":[{"role":"user","content":"List files"},
            {"role":"assistant","content":[{"type":"thinking","thinking":"I should call ls.","signature":"x"},
              {"type":"tool_use","id":"t1","name":"bash","input":{"command":"ls -l","n":1.0}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"a.txt"}]},
              {"type":"text","text":"and now?"}]}],
          "thinking":{"type":"enabled","budget_tokens":2000},"stop_sequences":["END"],"stream":true})";
        const auto r = mako::server::parse_messages_request(body, defaults);
        const auto text = tmpl.render(r.context).text;
        check(r.show_reasoning && r.stream && r.gen.max_tokens == 100 && r.stop == std::vector<std::string>{"END"},
              "anthropic: thinking, stream, max_tokens, stop_sequences");
        check(text.contains("<|system|>Reasoning Effort: Low"), "anthropic: thinking budget 2000 renders as Low");
        check(text.contains("<|system|>Be terse.<|user|>List files"), "anthropic: system blocks");
        check(text.contains("<think>I should call ls.</think>"), "anthropic: thinking block becomes reasoning");
        check(text.contains("<tool_call>bash<arg_key>command</arg_key><arg_value>ls -l</arg_value><arg_key>n</arg_key>"
                            "<arg_value>1.0</arg_value></tool_call>"),
              "anthropic: tool_use input renders as sent");
        check(text.contains("<|observation|><tool_response>a.txt</tool_response><|user|>and now?"),
              "anthropic: tool_result then the user's text");
        check(text.contains("\"name\": \"bash\"") && text.contains("\"description\": \"Run\""), "anthropic: tools");
        const auto any = mako::server::parse_messages_request(
            R"({"max_tokens":5,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"bash"}],"tool_choice":{"type":"tool","name":"bash"}})",
            defaults);
        check(any.tool_required && any.tool_name == "bash" && !any.show_reasoning, "anthropic: tool_choice tool");
        for (const char* bad : {R"({"messages":[{"role":"user","content":"x"}]})",
                                R"({"max_tokens":5,"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]})",
                                R"({"max_tokens":5,"messages":[{"role":"robot","content":"x"}]})",
                                R"({"max_tokens":5,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"any"}})"}) {
            bool threw = false;
            try {
                mako::server::parse_messages_request(bad, defaults);
            } catch (const mako::server::BadRequest&) {
                threw = true;
            }
            check(threw, std::format("anthropic rejected: {}", bad));
        }
        check(mako::server::parse_messages_request(R"({"messages":[{"role":"user","content":"x"}]})", defaults, true)
                  .gen.max_tokens == 0,
              "anthropic count_tokens needs no max_tokens");

        // The stream: blocks open and close in order, deltas add up to the whole message.
        mako::server::OutputParser p(*tok, r.tools, {});
        auto proto = mako::server::anthropic_protocol("msg_1", "m", true);
        std::string events = proto->begin(10);
        mako::server::Delta d;
        for (int32_t t : tok->encode("Thinking it over.</think>Here.<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call><|observation|>", true)) {
            mako::server::Delta x;
            p.push(t, x);
            events += proto->delta(x);
        }
        p.finish(d);
        events += proto->delta(d);
        mako::gen::GenerateResult gr;
        gr.finish = mako::gen::Finish::stop_token;
        gr.prompt_tokens = 10;
        events += proto->end(mako::server::End::tool_calls, p, gr);
        std::vector<std::string> names;
        for (size_t at = 0; (at = events.find("event: ", at)) != std::string::npos; at += 7)
            names.push_back(events.substr(at + 7, events.find('\n', at) - at - 7));
        auto count = [&](const char* n) { return std::ranges::count(names, std::string(n)); };
        check(!names.empty() && names.front() == "message_start" && names.back() == "message_stop" &&
                  count("content_block_start") == 3 && count("content_block_stop") == 3 && count("message_delta") == 1,
              std::format("anthropic stream events ({} starts, {} stops)", count("content_block_start"), count("content_block_stop")));
        check(events.contains(R"("type":"thinking")") && events.contains(R"("type":"text")") &&
                  events.contains(R"("type":"tool_use")") && events.contains(R"("stop_reason":"tool_use")"),
              "anthropic stream: thinking, text and tool_use blocks, stop_reason tool_use");
        const auto whole = proto->whole(mako::server::End::tool_calls, p, gr);
        glz::generic g;
        check(!glz::read_json(g, whole) && g["content"].get_array().size() == 3 &&
                  g["content"][2]["input"]["command"].get_string() == "pwd" && g["stop_reason"].get_string() == "tool_use",
              std::format("anthropic message body: {}", whole));
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

    // Thinking controls: this template knows low and high only and always reasons.
    {
        using mako::server::BadRequest;
        auto t = mako::server::thinking_support(tmpl);
        check(!t.can_disable && !t.numeric && !t.medium, "GLM template thinking support");
        const auto effort = [&](const std::string& e) {
            return tmpl
                .render(mako::server::parse_chat_request(
                            std::format(R"({{"messages":[{{"role":"user","content":"hi"}}],"reasoning_effort":{}}})", e),
                            {}, t)
                            .context)
                .text;
        };
        check(effort(R"("none")").contains("Reasoning Effort: Low"), "GLM: none asks for low");
        check(effort(R"("medium")").contains("Reasoning Effort: High"), "GLM: medium becomes high");
        bool threw = false;
        try {
            effort("40");
        } catch (const BadRequest&) {
            threw = true;
        }
        check(threw, "GLM: numeric effort rejected");
    }

    std::println("{} failures", failures);
    return failures ? 1 : 0;
}
