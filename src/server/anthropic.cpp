#include "server/anthropic.hpp"

#include "server/json.hpp"

#include <format>
#include <random>
#include <variant>

namespace mako::server {

// Glaze reflects these, which needs types with linkage.
namespace wire_anthropic {

struct Block {
    std::string type;
    std::optional<std::string> text, thinking, signature, id, name, tool_use_id;
    std::optional<glz::raw_json> input, content; // tool_use input; tool_result content (text or text blocks)
    std::optional<glz::raw_json> source;         // image: {type: base64, media_type, data} or {type: url, url}
    std::optional<bool> is_error;
};
struct MessageIn {
    std::string role;
    glz::raw_json content{"\"\""}; // text or blocks
};
struct ImageSource {
    std::string type;
    std::optional<std::string> data, url;
};
struct ToolIn {
    std::optional<std::string> type, description;
    std::string name;
    std::optional<glz::raw_json> input_schema;
};
struct ToolChoiceIn {
    std::string type;
    std::optional<std::string> name;
};
struct ThinkingIn {
    std::string type;
    std::optional<int64_t> budget_tokens;
};
struct Body {
    std::optional<std::string> model;
    std::optional<double> max_tokens;
    std::optional<glz::raw_json> system;
    std::vector<MessageIn> messages;
    std::optional<std::vector<ToolIn>> tools;
    std::optional<ToolChoiceIn> tool_choice;
    std::optional<std::vector<std::string>> stop_sequences;
    std::optional<double> temperature, top_p, top_k;
    std::optional<bool> stream;
    std::optional<ThinkingIn> thinking;
};

// The template's form (as Chat Completions requests become).
struct FunctionOut {
    std::string name;
    glz::raw_json arguments{"{}"};
};
struct CallOut {
    std::string id, type = "function";
    FunctionOut function;
};
struct TMessage {
    std::string role;
    glz::raw_json content{"\"\""};
    std::optional<std::string> reasoning_content;
    std::optional<std::vector<CallOut>> tool_calls;
    std::optional<std::string> tool_call_id;
};
struct ToolFunction {
    std::string name;
    std::optional<std::string> description;
    glz::raw_json parameters{"{\"type\":\"object\",\"properties\":{}}"};
};
struct ToolDef {
    std::string type = "function";
    ToolFunction function;
};
struct TemplateContext {
    std::vector<TMessage> messages;
    std::optional<std::vector<ToolDef>> tools;
    bool add_generation_prompt = true;
    std::optional<Effort> reasoning_effort;
    std::optional<bool> enable_thinking;
};

// Responses.
struct BlockOut {
    std::string type;
    std::optional<std::string> text, thinking, signature, id, name;
    std::optional<glz::raw_json> input;
};
struct Usage {
    uint32_t input_tokens = 0, output_tokens = 0, cache_read_input_tokens = 0;
};
struct MessageOut {
    std::string id, type = "message", role = "assistant", model;
    std::vector<BlockOut> content;
    glz::raw_json stop_reason{"null"}, stop_sequence{"null"};
    Usage usage;
};
struct ErrorInner {
    std::string type, message;
};
struct ErrorOut {
    std::string type = "error";
    ErrorInner error;
};
struct CountOut {
    uint32_t input_tokens = 0;
};

} // namespace wire_anthropic

namespace {

using namespace wire_anthropic;
using json::trim;

// An image block's file bytes.
std::string image_bytes(const Block& b)
{
    if (!b.source)
        throw BadRequest("an image block needs a source");
    ImageSource s;
    if (glz::read<glz::opts{.error_on_unknown_keys = false}>(s, trim(b.source->str)))
        throw BadRequest("an image source must be an object");
    if (s.type == "base64" && s.data)
        return base64_decode(*s.data);
    if (s.type == "url" && s.url)
        return image_from_url(*s.url);
    throw BadRequest("image sources are base64 data or data: URLs");
}

// Template content for blocks with images among them: an array of text and image parts in order, the images'
// bytes appended to `images`.
std::string parts_with_images(std::span<const Block> blocks, std::vector<std::string>& images)
{
    std::string out = "[";
    for (const Block& b : blocks) {
        if (b.type == "text" && b.text) {
            out += std::format(R"({}{{"type":"text","text":{}}})", out.size() > 1 ? "," : "", json::quoted(*b.text));
        } else if (b.type == "image") {
            images.push_back(image_bytes(b));
            out += std::format(R"({}{{"type":"image"}})", out.size() > 1 ? "," : "");
        } else if (b.type == "document") {
            throw BadRequest("document blocks are not supported");
        }
    }
    return out + "]";
}

bool has_image(std::span<const Block> blocks)
{
    return std::ranges::any_of(blocks, [](const Block& b) { return b.type == "image"; });
}

// The text of a string or of an array of text blocks.
std::string text_of(std::string_view raw, const char* what)
{
    raw = trim(raw);
    if (raw.empty() || raw == "null")
        return {};
    if (raw.front() == '"') {
        std::string s;
        if (glz::read_json(s, raw))
            throw BadRequest(std::format("{} is not valid JSON", what));
        return s;
    }
    std::vector<Block> blocks;
    if (glz::read<glz::opts{.error_on_unknown_keys = false}>(blocks, raw))
        throw BadRequest(std::format("{} must be text or an array of blocks", what));
    std::string s;
    for (const auto& b : blocks) {
        if (b.type == "text" && b.text) {
            if (!s.empty())
                s += "\n";
            s += *b.text;
        } else if (b.type == "image" || b.type == "document") {
            throw BadRequest(std::format("{} blocks are not supported", b.type));
        }
    }
    return s;
}

const char* stop_reason(End e)
{
    switch (e) {
    case End::tool_calls: return "tool_use";
    case End::length: return "max_tokens";
    case End::stop_sequence: return "stop_sequence";
    default: return "end_turn";
    }
}

} // namespace

ChatRequest parse_messages_request(std::string_view body, const gen::SamplingParams& sampling, bool count_only,
                                   const ThinkingOptions& thinking)
{
    Body b{};
    if (auto ec = glz::read<glz::opts{.error_on_unknown_keys = false}>(b, body))
        throw BadRequest(glz::format_error(ec, body));
    if (b.messages.empty())
        throw BadRequest("messages must be a non-empty array");
    if (!count_only && !b.max_tokens)
        throw BadRequest("max_tokens is required");

    TemplateContext ctx;
    std::vector<std::string> images;
    if (b.system) {
        if (std::string s = text_of(b.system->str, "system"); !s.empty())
            ctx.messages.push_back({.role = "system", .content = {json::quoted(s)}});
    }
    for (const auto& m : b.messages) {
        if (m.role != "user" && m.role != "assistant" && m.role != "system")
            throw BadRequest(std::format("unsupported role '{}'", m.role));
        if (m.role == "system") { // mid-conversation instructions; the template renders them in place
            if (std::string t = text_of(m.content.str, "system message"); !t.empty())
                ctx.messages.push_back({.role = "system", .content = {json::quoted(t)}});
            continue;
        }
        const std::string_view raw = trim(m.content.str);
        if (!raw.empty() && raw.front() == '"') {
            ctx.messages.push_back({.role = m.role, .content = {std::string(raw)}});
            continue;
        }
        std::vector<Block> blocks;
        if (glz::read<glz::opts{.error_on_unknown_keys = false}>(blocks, raw))
            throw BadRequest("message content must be text or an array of blocks");
        if (m.role == "user") {
            // Tool results become tool messages (after the call), the text a user message after them.
            std::string text;
            std::vector<Block> parts; // the user's own blocks, when they hold images
            const bool user_images = has_image(blocks);
            for (const auto& bl : blocks) {
                if (bl.type == "tool_result") {
                    if (!bl.tool_use_id)
                        throw BadRequest("tool_result needs tool_use_id");
                    const std::string_view inner_raw = bl.content ? trim(bl.content->str) : std::string_view();
                    std::vector<Block> inner;
                    std::string content;
                    if (!inner_raw.empty() && inner_raw.front() == '[' &&
                        !glz::read<glz::opts{.error_on_unknown_keys = false}>(inner, inner_raw) && has_image(inner)) {
                        if (bl.is_error.value_or(false))
                            inner.insert(inner.begin(), Block{.type = "text", .text = "Error:"});
                        content = parts_with_images(inner, images);
                    } else {
                        std::string out = bl.content ? text_of(bl.content->str, "tool_result content") : "";
                        if (bl.is_error.value_or(false))
                            out = "Error: " + out;
                        content = json::quoted(out);
                    }
                    ctx.messages.push_back({.role = "tool", .content = {content}, .tool_call_id = *bl.tool_use_id});
                } else if (user_images && (bl.type == "text" || bl.type == "image")) {
                    parts.push_back(bl);
                } else if (bl.type == "text" && bl.text) {
                    if (!text.empty())
                        text += "\n";
                    text += *bl.text;
                } else if (bl.type == "document") {
                    throw BadRequest("document blocks are not supported");
                }
            }
            if (user_images)
                ctx.messages.push_back({.role = "user", .content = {parts_with_images(parts, images)}});
            else if (!text.empty() || std::ranges::none_of(blocks, [](const Block& x) { return x.type == "tool_result"; }))
                ctx.messages.push_back({.role = "user", .content = {json::quoted(text)}});
        } else {
            TMessage a{.role = "assistant"};
            std::string text, thinking;
            for (const auto& bl : blocks) {
                if (bl.type == "text" && bl.text)
                    text += *bl.text;
                else if (bl.type == "thinking" && bl.thinking)
                    thinking += *bl.thinking;
                else if (bl.type == "tool_use") {
                    if (!bl.id || !bl.name)
                        throw BadRequest("tool_use needs id and name");
                    const std::string input = bl.input ? std::string(trim(bl.input->str)) : "{}";
                    if (!json::is_object_text(input))
                        throw BadRequest("tool_use input must be an object");
                    if (!a.tool_calls)
                        a.tool_calls.emplace();
                    a.tool_calls->push_back({.id = *bl.id, .function = {.name = *bl.name, .arguments = {input}}});
                }
            }
            a.content.str = json::quoted(text);
            if (!thinking.empty())
                a.reasoning_content = thinking;
            ctx.messages.push_back(std::move(a));
        }
    }

    ChatRequest r;
    r.images = std::move(images);
    bool use_tools = b.tools && !b.tools->empty();
    if (b.tool_choice) {
        const auto& c = *b.tool_choice;
        if (c.type == "none")
            use_tools = false;
        else if (c.type == "any")
            r.tool_required = true;
        else if (c.type == "tool") {
            if (!c.name)
                throw BadRequest("tool_choice tool needs a name");
            r.tool_required = true;
            r.tool_name = *c.name;
        } else if (c.type != "auto")
            throw BadRequest(std::format("unsupported tool_choice type '{}'", c.type));
        if (r.tool_required && !use_tools)
            throw BadRequest("tool_choice requires tools");
    }
    if (use_tools) {
        auto& defs = ctx.tools.emplace();
        bool found = r.tool_name.empty();
        for (const auto& t : *b.tools) {
            if (t.type && *t.type != "custom")
                throw BadRequest(std::format("server tool '{}' is not supported", *t.type));
            ToolDef d{.function = {.name = t.name, .description = t.description}};
            if (t.input_schema)
                d.function.parameters = *t.input_schema;
            defs.push_back(std::move(d));
            found |= t.name == r.tool_name;
        }
        if (!found)
            throw BadRequest(std::format("tool_choice names {}, which is not among the tools", r.tool_name));
        r.tools = json::write(defs);
    } else {
        r.tools = "[]";
    }

    // Thinking: shown only when enabled.
    r.show_reasoning = b.thinking && b.thinking->type == "enabled";
    std::optional<Effort> effort = Effort("none");
    if (r.show_reasoning) {
        const int64_t budget = b.thinking->budget_tokens.value_or(16384);
        effort = budget <= 4096 ? std::optional<Effort>("low")
                 : budget <= 16384 ? std::optional<Effort>("high")
                                   : std::nullopt;
    }
    auto c = choose_thinking(effort, thinking);
    ctx.reasoning_effort = std::move(c.reasoning_effort);
    ctx.enable_thinking = c.enable_thinking;
    r.context = json::write(ctx);

    auto& s = r.gen.sampling;
    s = sampling;
    if (b.temperature)
        s.temperature = float(*b.temperature);
    if (b.top_p)
        s.top_p = float(*b.top_p);
    if (b.top_k)
        s.top_k = uint32_t(std::max(0.0, *b.top_k));
    r.gen.seed = (uint64_t(std::random_device{}()) << 32) | std::random_device{}();
    r.gen.max_tokens = b.max_tokens ? uint32_t(std::clamp(*b.max_tokens, 1.0, 4e9)) : 0;
    r.max_tokens_given = r.gen.max_tokens > 0;
    if (r.show_reasoning && b.thinking->budget_tokens)
        r.reasoning_budget = uint32_t(std::clamp<int64_t>(*b.thinking->budget_tokens, 0, UINT32_MAX));
    if (b.stop_sequences)
        for (const auto& x : *b.stop_sequences)
            if (!x.empty())
                r.stop.push_back(x);
    r.stream = b.stream.value_or(false);
    return r;
}

namespace {

class AnthropicProtocol : public Protocol {
public:
    AnthropicProtocol(std::string id, std::string model, bool show_reasoning)
        : id_(std::move(id)), model_(std::move(model)), show_reasoning_(show_reasoning)
    {
    }

    std::string begin(uint32_t prompt_tokens) override
    {
        MessageOut m{.id = id_, .model = model_};
        m.usage.input_tokens = prompt_tokens;
        return event("message_start", std::format(R"({{"type":"message_start","message":{}}})", json::write(m)));
    }

    std::string delta(const Delta& d) override
    {
        std::string s;
        if (show_reasoning_ && !d.reasoning.empty())
            s += append("thinking", "thinking_delta", "thinking", d.reasoning);
        if (!d.content.empty())
            s += append("text", "text_delta", "text", d.content);
        for (const auto& c : d.calls) {
            if (!c.id.empty()) {
                s += close();
                BlockOut b{.type = "tool_use", .id = c.id, .name = c.name, .input = glz::raw_json{"{}"}};
                s += open("tool_use", json::write(b));
            }
            if (!c.arguments.empty())
                s += event("content_block_delta",
                           std::format(R"({{"type":"content_block_delta","index":{},"delta":{{"type":"input_json_delta","partial_json":{}}}}})",
                                       index_, json::quoted(c.arguments)));
        }
        return s;
    }

    std::string end(End e, const OutputParser& p, const gen::GenerateResult& r) override
    {
        std::string s = close();
        s += event("message_delta",
                   std::format(R"({{"type":"message_delta","delta":{{"stop_reason":"{}","stop_sequence":{}}},"usage":{{"output_tokens":{}}}}})",
                               stop_reason(e), e == End::stop_sequence ? json::quoted(p.stop_matched()) : "null",
                               r.tokens.size()));
        return s + event("message_stop", R"({"type":"message_stop"})");
    }

    std::string stream_error(const std::string& message) override
    {
        return event("error", anthropic_error(message, "api_error"));
    }

    std::string whole(End e, const OutputParser& p, const gen::GenerateResult& r) override
    {
        MessageOut m{.id = id_, .model = model_};
        if (show_reasoning_ && !p.reasoning().empty())
            m.content.push_back({.type = "thinking", .thinking = p.reasoning(), .signature = ""});
        if (!p.content().empty())
            m.content.push_back({.type = "text", .text = p.content()});
        for (const auto& c : p.calls())
            m.content.push_back({.type = "tool_use", .id = c.id, .name = c.name, .input = glz::raw_json{c.arguments}});
        m.stop_reason.str = json::quoted(stop_reason(e));
        if (e == End::stop_sequence)
            m.stop_sequence.str = json::quoted(p.stop_matched());
        m.usage = {.input_tokens = r.prompt_tokens - r.reused_tokens,
                   .output_tokens = uint32_t(r.tokens.size()),
                   .cache_read_input_tokens = r.reused_tokens};
        return json::write(m);
    }

private:
    static std::string event(const char* name, const std::string& data)
    {
        return std::format("event: {}\ndata: {}\n\n", name, data);
    }
    std::string open(const char* kind, const std::string& block)
    {
        index_ = next_++;
        open_ = kind;
        return event("content_block_start",
                     std::format(R"({{"type":"content_block_start","index":{},"content_block":{}}})", index_, block));
    }
    std::string close()
    {
        if (open_.empty())
            return {};
        open_.clear();
        return event("content_block_stop", std::format(R"({{"type":"content_block_stop","index":{}}})", index_));
    }
    // Text into the open block of `kind`, opening one (and closing another) first.
    std::string append(const char* kind, const char* delta_type, const char* field, const std::string& text)
    {
        std::string s;
        if (open_ != kind) {
            s += close();
            BlockOut b{.type = kind};
            if (std::string_view(kind) == "thinking") {
                b.thinking = "";
                b.signature = "";
            } else {
                b.text = "";
            }
            s += open(kind, json::write(b));
        }
        return s + event("content_block_delta",
                         std::format(R"({{"type":"content_block_delta","index":{},"delta":{{"type":"{}","{}":{}}}}})",
                                     index_, delta_type, field, json::quoted(text)));
    }

    std::string id_, model_;
    bool show_reasoning_;
    std::string open_; // the open block's type
    int index_ = -1, next_ = 0;
};

} // namespace

std::unique_ptr<Protocol> anthropic_protocol(std::string id, std::string model, bool show_reasoning)
{
    return std::make_unique<AnthropicProtocol>(std::move(id), std::move(model), show_reasoning);
}

std::string anthropic_error(const std::string& message, const char* type)
{
    return json::write(ErrorOut{.error = {.type = type, .message = message}});
}

std::string count_tokens_body(uint32_t tokens)
{
    return json::write(CountOut{.input_tokens = tokens});
}

} // namespace mako::server
