#include "server/openai.hpp"

#include "chat/template.hpp"
#include "server/json.hpp"
#include "server/protocol.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <random>
#include <variant>

namespace mako::server {

// Glaze reflects these, which needs types with linkage.
namespace wire {

// Request fields the server reads; values passed to the template unchanged stay raw JSON text.
struct FunctionIn {
    std::string name;
    glz::raw_json arguments{"{}"};
};
struct ToolCallIn {
    std::optional<std::string> id, type;
    FunctionIn function;
};
struct MessageIn {
    std::string role;
    std::optional<glz::raw_json> content;
    std::optional<std::string> reasoning_content, reasoning;
    std::optional<std::vector<ToolCallIn>> tool_calls;
    std::optional<std::string> tool_call_id, name;
};
struct StreamOptions {
    std::optional<bool> include_usage;
};
struct ChatBody {
    std::optional<std::string> model;
    std::vector<MessageIn> messages;
    std::optional<glz::raw_json> tools, tool_choice, response_format, chat_template_kwargs;
    std::optional<Effort> reasoning_effort;
    std::optional<double> temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty, repetition_penalty;
    std::optional<double> max_tokens, max_completion_tokens, n;
    std::optional<int64_t> seed;
    std::optional<bool> stream, logprobs;
    std::optional<StreamOptions> stream_options;
    std::optional<std::variant<std::string, std::vector<std::string>>> stop;
};
struct TemplateContext {
    std::vector<MessageIn> messages;
    std::optional<glz::raw_json> tools;
    bool add_generation_prompt = true;
    std::optional<Effort> reasoning_effort;
    std::optional<bool> enable_thinking;
};

struct FunctionOut {
    std::optional<std::string> name;
    std::string arguments;
};
struct CallOut {
    size_t index = 0;
    std::optional<std::string> id, type;
    FunctionOut function;
};
struct DeltaOut {
    std::optional<std::string> role, content, reasoning_content;
    std::optional<std::vector<CallOut>> tool_calls;
};
struct ChunkChoice {
    int index = 0;
    DeltaOut delta;
    glz::raw_json finish_reason{"null"};
};
struct TokenDetails {
    uint32_t cached_tokens = 0;
};
struct Usage {
    uint32_t prompt_tokens = 0, completion_tokens = 0, total_tokens = 0;
    TokenDetails prompt_tokens_details;
};
struct Chunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created = 0;
    std::string model;
    std::vector<ChunkChoice> choices;
    std::optional<Usage> usage;
};
struct MessageOut {
    std::string role = "assistant";
    glz::raw_json content{"null"};
    std::optional<std::string> reasoning_content;
    std::optional<std::vector<CallOut>> tool_calls;
};
struct CompletionChoice {
    int index = 0;
    MessageOut message;
    std::string finish_reason;
};
struct Completion {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    std::vector<CompletionChoice> choices;
    Usage usage;
};
struct ErrorInner {
    std::string message, type;
    glz::raw_json code{"null"};
};
struct ErrorBody {
    ErrorInner error;
};
struct ModelEntry {
    std::string id;
    std::string object = "model";
    int64_t created = 0;
    std::string owned_by = "mako";
};
struct ModelList {
    std::string object = "list";
    std::vector<ModelEntry> data;
};

} // namespace wire

namespace {

using namespace wire;

using json::is_object_text;
using json::is_space;
using json::trim;

template <class T>
std::string to_json(const T& v)
{
    return json::write(v);
}

std::string json_string(std::string_view s)
{
    return json::quoted(s);
}

// Tool call arguments as the template iterates them: an object (OpenAI sends its JSON text as a string).
void parse_arguments(glz::raw_json& args)
{
    std::string_view s = trim(args.str);
    if (s == "null") {
        args.str = "{}";
        return;
    }
    if (!s.empty() && s.front() == '"') {
        std::string text;
        if (glz::read_json(text, s))
            throw BadRequest("tool call arguments are not valid JSON");
        if (trim(text).empty())
            text = "{}";
        if (!is_object_text(text))
            throw BadRequest("tool call arguments must be a JSON object");
        args.str = std::move(text);
    } else if (!is_object_text(s))
        throw BadRequest("tool call arguments must be a JSON object");
}

} // namespace

Effort parse_effort(std::string_view text)
{
    int64_t n = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (ec == std::errc() && end == text.data() + text.size())
        return n;
    return std::string(text);
}

ThinkingOptions thinking_support(const chat::Template& tmpl)
{
    const auto render = [&](std::string_view extra) -> std::optional<std::string> {
        try {
            return tmpl
                .render(std::format(R"({{"messages":[{{"role":"user","content":"x"}}],"add_generation_prompt":true{}}})",
                                    extra))
                .text;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    };
    // An effort counts when it renders other than an unknown one does (or the template rejects unknown ones).
    const auto unknown = render(R"(,"reasoning_effort":"no-such-level")");
    const auto honours = [&](std::string_view extra) {
        const auto r = render(extra);
        return r && (!unknown || *r != *unknown);
    };
    ThinkingOptions t;
    const auto plain = render(""), off = render(R"(,"enable_thinking":false)");
    t.can_disable = plain && off && *plain != *off;
    t.numeric = honours(R"(,"reasoning_effort":25)");
    t.medium = honours(R"(,"reasoning_effort":"medium")");
    return t;
}

ThinkingChoice choose_thinking(const std::optional<Effort>& effort, const ThinkingOptions& t)
{
    const std::optional<Effort>& e = effort ? effort : t.default_effort;
    if (!e)
        return {};
    const auto none = [&]() -> ThinkingChoice {
        if (t.can_disable)
            return {.enable_thinking = false};
        return {.reasoning_effort = Effort("low")};
    };
    if (const auto* n = std::get_if<int64_t>(&*e)) {
        if (*n == 0)
            return none();
        if (*n < 1 || *n > 100)
            throw BadRequest(std::format("reasoning_effort {} is outside 0..100", *n));
        if (!t.numeric)
            throw BadRequest("this model's chat template takes reasoning_effort as a level name (low, high), not a number");
        return {.reasoning_effort = *e};
    }
    const auto& name = std::get<std::string>(*e);
    if (name == "none" || name == "minimal")
        return none();
    if (name == "medium" && !t.medium)
        return {.reasoning_effort = Effort("high")};
    return {.reasoning_effort = *e};
}

ChatRequest parse_chat_request(std::string_view body, const gen::SamplingParams& sampling,
                               const ThinkingOptions& thinking)
{
    ChatBody b{};
    if (auto ec = glz::read<glz::opts{.error_on_unknown_keys = false}>(b, body))
        throw BadRequest(glz::format_error(ec, body));
    if (b.messages.empty())
        throw BadRequest("messages must be a non-empty array");

    TemplateContext ctx;
    for (auto& m : b.messages) {
        if (m.role == "developer")
            m.role = "system";
        else if (m.role != "system" && m.role != "user" && m.role != "assistant" && m.role != "tool")
            throw BadRequest(std::format("unsupported role '{}'", m.role));
        if (!m.content || trim(m.content->str) == "null")
            m.content = glz::raw_json{"\"\""};
        if (!m.reasoning_content && m.reasoning)
            m.reasoning_content = std::move(m.reasoning);
        m.reasoning.reset();
        if (m.tool_calls) {
            if (m.role != "assistant")
                throw BadRequest("only assistant messages have tool calls");
            for (auto& tc : *m.tool_calls) {
                if (tc.function.name.empty())
                    throw BadRequest("each tool call needs a function name");
                parse_arguments(tc.function.arguments);
            }
        }
    }
    ctx.messages = std::move(b.messages);

    ChatRequest r;
    bool use_tools = b.tools.has_value();
    if (use_tools) {
        const std::string_view t = trim(b.tools->str);
        if (t == "null")
            use_tools = false;
        else if (t.empty() || t.front() != '[')
            throw BadRequest("tools must be an array");
        else
            use_tools = t != "[]";
    }
    if (b.tool_choice) {
        const std::string_view c = trim(b.tool_choice->str);
        if (c == "\"none\"")
            use_tools = false;
        else if (c == "\"required\"")
            r.tool_required = true;
        else if (c != "\"auto\"" && c != "null") {
            glz::generic g;
            if (glz::read_json(g, c) || !g.is_object() || !g.contains("function") || !g["function"].is_object() ||
                !g["function"].contains("name") || !g["function"]["name"].is_string())
                throw BadRequest("tool_choice must be auto, none, required or {\"type\": \"function\", \"function\": "
                                 "{\"name\": ...}}");
            r.tool_required = true;
            r.tool_name = g["function"]["name"].get_string();
        }
        if (r.tool_required && !use_tools)
            throw BadRequest("tool_choice requires tools");
    }
    r.tools = use_tools ? b.tools->str : "[]";
    if (!r.tool_name.empty()) {
        glz::generic defs;
        bool found = false;
        if (!glz::read_json(defs, r.tools) && defs.is_array())
            for (const auto& t : defs.get_array()) {
                const auto& fn = t.is_object() && t.contains("function") ? t["function"] : t;
                found |= fn.is_object() && fn.contains("name") && fn["name"].is_string() &&
                         fn["name"].get_string() == r.tool_name;
            }
        if (!found)
            throw BadRequest(std::format("tool_choice names {}, which is not among the tools", r.tool_name));
    }
    if (use_tools)
        ctx.tools = std::move(b.tools);
    // Thinking keys in chat_template_kwargs replace the request's effort.
    glz::generic kwargs;
    const bool has_kwargs = b.chat_template_kwargs && trim(b.chat_template_kwargs->str) != "null";
    if (has_kwargs && (glz::read_json(kwargs, trim(b.chat_template_kwargs->str)) || !kwargs.is_object()))
        throw BadRequest("chat_template_kwargs must be an object");
    bool own_thinking = true;
    for (const char* k : {"reasoning_effort", "enable_thinking", "thinking", "thinking_mode"})
        own_thinking &= !(has_kwargs && kwargs.contains(k));
    if (own_thinking) {
        auto c = choose_thinking(b.reasoning_effort, thinking);
        ctx.reasoning_effort = std::move(c.reasoning_effort);
        ctx.enable_thinking = c.enable_thinking;
    }
    r.context = to_json(ctx);
    if (has_kwargs) {
        // Extra template variables, spliced in as sent (the template's own keys stay the server's).
        const std::string_view kw = trim(b.chat_template_kwargs->str);
        const auto& g = kwargs;
        for (const char* own : {"messages", "tools", "add_generation_prompt"})
            if (g.contains(own))
                throw BadRequest(std::format("chat_template_kwargs may not set {}", own));
        const std::string_view inner = trim(kw.substr(1, kw.size() - 2));
        if (!inner.empty()) {
            r.context.pop_back();
            r.context += ",";
            r.context += inner;
            r.context += "}";
        }
    }

    auto& s = r.gen.sampling;
    s = sampling;
    if (b.temperature)
        s.temperature = float(*b.temperature);
    if (b.top_p)
        s.top_p = float(*b.top_p);
    if (b.top_k)
        s.top_k = uint32_t(std::max(0.0, *b.top_k));
    if (b.min_p)
        s.min_p = float(*b.min_p);
    if (b.presence_penalty)
        s.presence_penalty = float(*b.presence_penalty);
    if (b.frequency_penalty)
        s.frequency_penalty = float(*b.frequency_penalty);
    if (b.repetition_penalty)
        s.repetition_penalty = float(*b.repetition_penalty);
    r.gen.seed = b.seed ? uint64_t(*b.seed) : (uint64_t(std::random_device{}()) << 32) | std::random_device{}();
    r.gen.max_tokens = 0;
    if (const auto& mt = b.max_completion_tokens ? b.max_completion_tokens : b.max_tokens) {
        if (*mt < 1)
            throw BadRequest("max_tokens must be positive");
        r.gen.max_tokens = uint32_t(std::min(*mt, 4e9));
    }
    if (b.n && *b.n != 1)
        throw BadRequest("n other than 1 is not supported");
    if (b.logprobs && *b.logprobs)
        throw BadRequest("logprobs are not supported");
    if (b.response_format) {
        const std::string_view f = trim(b.response_format->str);
        glz::generic g;
        if (f != "null" && (glz::read_json(g, f) || !g.is_object() || !g.contains("type") || !g["type"].is_string() ||
                            g["type"].get_string() != "text"))
            throw BadRequest("response_format other than text is not supported");
    }
    if (b.stop) {
        if (const auto* one = std::get_if<std::string>(&*b.stop))
            r.stop.push_back(*one);
        else
            r.stop = std::get<std::vector<std::string>>(*b.stop);
        std::erase_if(r.stop, [](const std::string& x) { return x.empty(); });
    }
    r.stream = b.stream.value_or(false);
    r.include_usage = b.stream_options && b.stream_options->include_usage.value_or(false);
    return r;
}

struct OutputParser::Tools {
    glz::generic defs;
};

namespace {

constexpr std::string_view kDsmlCalls = "<｜DSML｜ calls>", kDsmlCallsEnd = "</｜DSML｜ calls>",
                           kDsmlInvoke = "<｜DSML｜ invoke name=\"", kDsmlInvokeEnd = "</｜DSML｜ invoke>",
                           kDsmlParam = "<｜DSML｜ parameter name=\"", kDsmlParamEnd = "</｜DSML｜ parameter>";

} // namespace

OutputParser::OutputParser(const text::Tokenizer& tok, std::string_view tools, std::vector<std::string> stop,
                           bool reasoning_first)
    : tools_(std::make_unique<Tools>()), stop_(std::move(stop)), stream_(tok, false),
      mode_(reasoning_first ? Mode::reasoning : Mode::content), tok_(tok)
{
    if (glz::read_json(tools_->defs, tools) || !tools_->defs.is_array())
        tools_->defs = glz::generic::array_t{};
    think_end_ = tok.find("</think>");
    call_begin_ = tok.find("<tool_call>");
    call_end_ = tok.find("</tool_call>");
    key_begin_ = tok.find("<arg_key>");
    key_end_ = tok.find("</arg_key>");
    value_begin_ = tok.find("<arg_value>");
    value_end_ = tok.find("</arg_value>");
    dsml_form_ = tok.find("｜DSML｜") >= 0 && call_begin_ < 0;
}

OutputParser::~OutputParser() = default;

void OutputParser::require_call(const std::string& name)
{
    require_ = true;
    if (dsml_form_)
        forced_ = tok_.encode(std::string(kDsmlCalls) + "\n" + std::string(kDsmlInvoke) + (name.empty() ? "" : name + "\">"),
                              true);
    name_tokens_ = name.empty() ? std::vector<int32_t>{} : tok_.encode(name, false);
    named_has_args_ = false;
    for (const auto& t : tools_->defs.get_array()) {
        const auto& fn = t.is_object() && t.contains("function") ? t["function"] : t;
        if (!fn.is_object() || !fn.contains("name") || !fn["name"].is_string() || fn["name"].get_string() != name)
            continue;
        named_has_args_ = fn.contains("parameters") && fn["parameters"].is_object() &&
                          fn["parameters"].contains("properties") && fn["parameters"]["properties"].is_object() &&
                          !fn["parameters"]["properties"].get_object().empty();
    }
}

int32_t OutputParser::forced() const
{
    if (!require_)
        return -1;
    if (dsml_form_)
        return mode_ != Mode::reasoning && forced_seen_ < forced_.size() && (forced_seen_ > 0 || !content_started_)
                   ? forced_[forced_seen_]
                   : -1;
    if (mode_ == Mode::content && calls_.empty() && !content_started_ && pending_.empty())
        return call_begin_;
    if (mode_ == Mode::name && calls_.size() == 1 && !name_tokens_.empty()) {
        if (name_seen_ < name_tokens_.size())
            return name_tokens_[name_seen_];
        if (name_seen_ == name_tokens_.size())
            return named_has_args_ ? key_begin_ : call_end_;
    }
    return -1;
}

bool OutputParser::push(int32_t token, Delta& out)
{
    if (mode_ == Mode::done)
        return false;
    const auto marker = [&](int32_t id, std::initializer_list<Mode> modes) {
        return token == id && std::ranges::find(modes, mode_) != modes.end();
    };
    if (dsml_form_ && forced_seen_ < forced_.size() && mode_ != Mode::reasoning && token == forced_[forced_seen_])
        ++forced_seen_;
    if (marker(think_end_, {Mode::reasoning})) {
        flush_text(out);
        mode_ = Mode::content;
    } else if (dsml_form_) {
        text(stream_.push(token), out);
    } else if (marker(call_begin_, {Mode::content})) {
        flush_text(out);
        if (mode_ == Mode::done)
            return false;
        calls_.push_back({});
        buf_.clear();
        call_open_ = false;
        first_arg_ = true;
        mode_ = Mode::name;
    } else if (marker(key_begin_, {Mode::name, Mode::between})) {
        flush_text(out);
        if (mode_ == Mode::name)
            open_call(out);
        buf_.clear();
        mode_ = Mode::key;
    } else if (marker(key_end_, {Mode::key})) {
        flush_text(out);
        key_ = trim(buf_);
        mode_ = Mode::between;
    } else if (marker(value_begin_, {Mode::between})) {
        flush_text(out);
        buf_.clear();
        mode_ = Mode::value;
    } else if (marker(value_end_, {Mode::value})) {
        flush_text(out);
        // Non-string values are written as JSON; text that does not parse stays a string.
        const std::string_view v = trim(buf_);
        const bool raw = !string_typed(calls_.back().name, key_) && !v.empty() && !glz::validate_json(v);
        const std::string frag = (first_arg_ ? "{" : ", ") + json_string(key_) + ": " + (raw ? std::string(v) : json_string(buf_));
        first_arg_ = false;
        calls_.back().arguments += frag;
        out.calls.push_back({calls_.size() - 1, "", "", frag});
        mode_ = Mode::between;
    } else if (marker(call_end_, {Mode::name, Mode::between})) {
        flush_text(out);
        if (mode_ == Mode::name)
            open_call(out);
        const std::string frag = first_arg_ ? "{}" : "}";
        calls_.back().arguments += frag;
        out.calls.push_back({calls_.size() - 1, "", "", frag});
        mode_ = Mode::content;
    } else {
        if (mode_ == Mode::name && calls_.size() == 1)
            ++name_seen_;
        text(stream_.push(token), out);
    }
    return mode_ != Mode::done;
}

void OutputParser::finish(Delta& out)
{
    flush_text(out);
    switch (mode_) {
    case Mode::content:
        emit_content(out, true);
        break;
    case Mode::name: // cut off before the call had a name
        calls_.pop_back();
        break;
    case Mode::dsml:
        parse_dsml(out);
        if (call_open_)
            close_call(out);
        break;
    case Mode::key:
    case Mode::between:
    case Mode::value: { // cut off inside the arguments: close what was complete
        const std::string frag = first_arg_ ? "{}" : "}";
        calls_.back().arguments += frag;
        out.calls.push_back({calls_.size() - 1, "", "", frag});
        break;
    }
    default:
        break;
    }
    mode_ = Mode::done;
}

void OutputParser::text(const std::string& s, Delta& out)
{
    if (s.empty())
        return;
    switch (mode_) {
    case Mode::reasoning:
        reasoning_ += s;
        out.reasoning += s;
        break;
    case Mode::content:
        pending_ += s;
        if (dsml_form_) {
            if (const size_t at = pending_.find(kDsmlCalls); at != std::string::npos) {
                dsml_ = pending_.substr(at + kDsmlCalls.size());
                pending_.resize(at);
                emit_content(out, true);
                if (mode_ == Mode::done)
                    break;
                mode_ = Mode::dsml;
                parse_dsml(out);
                break;
            }
        }
        emit_content(out, false);
        break;
    case Mode::dsml:
        dsml_ += s;
        parse_dsml(out);
        break;
    case Mode::name:
    case Mode::key:
    case Mode::value:
        buf_ += s;
        break;
    default:
        break;
    }
}

void OutputParser::flush_text(Delta& out)
{
    text(stream_.flush(), out);
}

void OutputParser::emit_content(Delta& out, bool final)
{
    if (!content_started_) {
        const auto first = std::ranges::find_if_not(pending_, is_space);
        pending_.erase(pending_.begin(), first);
        if (pending_.empty())
            return;
        content_started_ = true;
    }
    size_t cut = std::string::npos;
    for (const auto& s : stop_)
        if (const size_t at = pending_.find(s); at < cut) {
            cut = at;
            stop_matched_ = s;
        }
    if (cut != std::string::npos) {
        pending_.resize(cut);
        final = true;
        mode_ = Mode::done;
    }
    // Held back: trailing whitespace (the answer is trimmed) and the start of a stop string.
    size_t keep = 0;
    while (keep < pending_.size() && is_space(pending_[pending_.size() - 1 - keep]))
        ++keep;
    if (!final) {
        for (const auto& s : stop_)
            for (size_t k = std::min(s.size() - 1, pending_.size()); k > keep; --k)
                if (pending_.ends_with(std::string_view(s).substr(0, k))) {
                    keep = k;
                    break;
                }
        if (dsml_form_)
            for (size_t k = std::min(kDsmlCalls.size() - 1, pending_.size()); k > keep; --k)
                if (pending_.ends_with(kDsmlCalls.substr(0, k))) {
                    keep = k;
                    break;
                }
        // and the whitespace before what is held back
        while (keep < pending_.size() && is_space(pending_[pending_.size() - 1 - keep]))
            ++keep;
    }
    const std::string emit = pending_.substr(0, pending_.size() - keep);
    pending_.erase(0, emit.size());
    if (final)
        pending_.clear();
    content_ += emit;
    out.content += emit;
}

void OutputParser::close_call(Delta& out)
{
    const std::string frag = first_arg_ ? "{}" : "}";
    calls_.back().arguments += frag;
    out.calls.push_back({calls_.size() - 1, "", "", frag});
    call_open_ = false;
}

void OutputParser::parse_dsml(Delta& out)
{
    for (;;) {
        const size_t start = dsml_.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            dsml_.clear();
            return;
        }
        const std::string_view rest = std::string_view(dsml_).substr(start);
        size_t used = 0;
        if (rest.starts_with(kDsmlInvoke)) {
            const size_t q = rest.find("\">", kDsmlInvoke.size());
            if (q == std::string_view::npos)
                return;
            if (call_open_)
                close_call(out);
            calls_.push_back({});
            buf_ = std::string(rest.substr(kDsmlInvoke.size(), q - kDsmlInvoke.size()));
            first_arg_ = true;
            open_call(out);
            used = q + 2;
        } else if (rest.starts_with(kDsmlParam)) {
            const size_t q = rest.find('"', kDsmlParam.size());
            const size_t open_end = q == std::string_view::npos ? q : rest.find('>', q);
            const size_t close = open_end == std::string_view::npos ? open_end : rest.find(kDsmlParamEnd, open_end);
            if (close == std::string_view::npos)
                return;
            const std::string key(rest.substr(kDsmlParam.size(), q - kDsmlParam.size()));
            const std::string_view attrs = rest.substr(q + 1, open_end - q - 1);
            const std::string value(rest.substr(open_end + 1, close - open_end - 1));
            const bool is_string = attrs.find("string=\"false\"") == std::string_view::npos;
            const bool raw = !is_string && !value.empty() && !glz::validate_json(value);
            if (call_open_) {
                const std::string frag = (first_arg_ ? "{" : ", ") + json_string(key) + ": " +
                                         (raw ? value : json_string(value));
                first_arg_ = false;
                calls_.back().arguments += frag;
                out.calls.push_back({calls_.size() - 1, "", "", frag});
            }
            used = close + kDsmlParamEnd.size();
        } else if (rest.starts_with(kDsmlInvokeEnd)) {
            if (call_open_)
                close_call(out);
            used = kDsmlInvokeEnd.size();
        } else if (rest.starts_with(kDsmlCallsEnd)) {
            if (call_open_)
                close_call(out);
            dsml_.erase(0, start + kDsmlCallsEnd.size());
            mode_ = Mode::content;
            content_started_ = false;
            if (!dsml_.empty()) {
                const std::string tail = std::move(dsml_);
                dsml_.clear();
                text(tail, out);
            }
            return;
        } else {
            // An element still arriving, or text the block does not allow (left until more makes sense of it).
            const std::string_view all[] = {kDsmlInvoke, kDsmlParam, kDsmlInvokeEnd, kDsmlCallsEnd};
            for (const auto m : all)
                if (m.starts_with(rest.substr(0, std::min(rest.size(), m.size()))))
                    return;
            const size_t lt = rest.find('<', 1);
            used = lt == std::string_view::npos ? rest.size() : lt;
        }
        dsml_.erase(0, start + used);
    }
}

void OutputParser::open_call(Delta& out)
{
    static std::mt19937_64 rng{std::random_device{}()};
    auto& c = calls_.back();
    c.name = trim(buf_);
    c.id = std::format("call_{:016x}", rng());
    call_open_ = true;
    out.calls.push_back({calls_.size() - 1, c.id, c.name, ""});
}

bool OutputParser::string_typed(const std::string& tool, const std::string& key) const
{
    for (const auto& t : tools_->defs.get_array()) {
        if (!t.is_object())
            continue;
        const auto& fn = t.contains("function") ? t["function"] : t;
        if (!fn.is_object() || !fn.contains("name") || !fn["name"].is_string() || fn["name"].get_string() != tool)
            continue;
        if (!fn.contains("parameters") || !fn["parameters"].is_object())
            return false;
        const auto& params = fn["parameters"];
        if (!params.contains("properties") || !params["properties"].is_object() || !params["properties"].contains(key))
            return false;
        const auto& prop = params["properties"][key];
        if (!prop.is_object() || !prop.contains("type"))
            return false;
        const auto& type = prop["type"];
        if (type.is_string())
            return type.get_string() == "string";
        return type.is_array() && type.get_array().size() == 1 && type[0].is_string() && type[0].get_string() == "string";
    }
    return false;
}

std::string chunk(const std::string& id, int64_t created, const std::string& model, const Delta& d, bool role,
                  const char* finish_reason)
{
    ChunkChoice choice;
    if (role)
        choice.delta.role = "assistant";
    if (!d.reasoning.empty())
        choice.delta.reasoning_content = d.reasoning;
    if (!d.content.empty() || role)
        choice.delta.content = d.content;
    if (!d.calls.empty()) {
        auto& calls = choice.delta.tool_calls.emplace();
        for (const auto& c : d.calls) {
            CallOut o{.index = c.index, .function = {.arguments = c.arguments}};
            if (!c.id.empty()) {
                o.id = c.id;
                o.type = "function";
                o.function.name = c.name;
            }
            calls.push_back(std::move(o));
        }
    }
    if (finish_reason)
        choice.finish_reason.str = json_string(finish_reason);
    return to_json(Chunk{.id = id, .created = created, .model = model, .choices = {std::move(choice)}});
}

namespace {

Usage usage_of(const gen::GenerateResult& r)
{
    const auto n = uint32_t(r.tokens.size());
    return {.prompt_tokens = r.prompt_tokens,
            .completion_tokens = n,
            .total_tokens = r.prompt_tokens + n,
            .prompt_tokens_details = {.cached_tokens = r.reused_tokens}};
}

} // namespace

std::string usage_chunk(const std::string& id, int64_t created, const std::string& model, const gen::GenerateResult& r)
{
    return to_json(Chunk{.id = id, .created = created, .model = model, .usage = usage_of(r)});
}

std::string completion(const std::string& id, int64_t created, const std::string& model, const OutputParser& p,
                       const char* finish_reason, const gen::GenerateResult& r)
{
    MessageOut msg;
    if (!p.content().empty() || p.calls().empty())
        msg.content.str = json_string(p.content());
    if (!p.reasoning().empty())
        msg.reasoning_content = p.reasoning();
    if (!p.calls().empty()) {
        auto& calls = msg.tool_calls.emplace();
        for (size_t i = 0; i < p.calls().size(); ++i) {
            const auto& c = p.calls()[i];
            calls.push_back({.index = i, .id = c.id, .type = "function",
                             .function = {.name = c.name, .arguments = c.arguments}});
        }
    }
    return to_json(Completion{.id = id,
                              .created = created,
                              .model = model,
                              .choices = {{.message = std::move(msg), .finish_reason = finish_reason}},
                              .usage = usage_of(r)});
}

std::string error_body(const std::string& message, const char* type)
{
    return to_json(ErrorBody{{.message = message, .type = type}});
}

End end_of(const gen::GenerateResult& r, const OutputParser& p, bool stopped)
{
    switch (r.finish) {
    case gen::Finish::stop_token:
        return p.calls().empty() ? End::stop : End::tool_calls;
    case gen::Finish::max_tokens:
    case gen::Finish::context_full:
        return End::length;
    case gen::Finish::cancelled:
        return stopped ? End::stop_sequence : End::cancelled;
    }
    return End::stop;
}

namespace {

const char* finish_reason(End e)
{
    switch (e) {
    case End::tool_calls: return "tool_calls";
    case End::length: return "length";
    case End::cancelled: return "cancelled";
    default: return "stop";
    }
}

class OpenAIProtocol : public Protocol {
public:
    OpenAIProtocol(std::string id, int64_t created, std::string model, bool include_usage)
        : id_(std::move(id)), created_(created), model_(std::move(model)), include_usage_(include_usage)
    {
    }
    std::string begin(uint32_t) override { return event(chunk(id_, created_, model_, {}, true)); }
    std::string delta(const Delta& d) override { return event(chunk(id_, created_, model_, d)); }
    std::string end(End e, const OutputParser&, const gen::GenerateResult& r) override
    {
        std::string s = event(chunk(id_, created_, model_, {}, false, finish_reason(e)));
        if (include_usage_)
            s += event(usage_chunk(id_, created_, model_, r));
        return s + "data: [DONE]\n\n";
    }
    std::string stream_error(const std::string& message) override
    {
        return event(error_body(message, "server_error")) + "data: [DONE]\n\n";
    }
    std::string whole(End e, const OutputParser& p, const gen::GenerateResult& r) override
    {
        return completion(id_, created_, model_, p, finish_reason(e), r);
    }

private:
    static std::string event(const std::string& json) { return "data: " + json + "\n\n"; }
    std::string id_;
    int64_t created_;
    std::string model_;
    bool include_usage_;
};

} // namespace

std::unique_ptr<Protocol> openai_protocol(std::string id, int64_t created, std::string model, bool include_usage)
{
    return std::make_unique<OpenAIProtocol>(std::move(id), created, std::move(model), include_usage);
}

std::string models_body(const std::string& model)
{
    return to_json(ModelList{.data = {{.id = model}}});
}

} // namespace mako::server
