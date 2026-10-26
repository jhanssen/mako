#pragma once

#include "gen/sampler.hpp"
#include "gen/session.hpp"
#include "text/stream.hpp"
#include "text/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mako::chat {
class Template;
}

namespace mako::server {

// A request the server rejects with HTTP 400.
struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An OpenAI Chat Completions request translated for the chat template and the session.
struct ChatRequest {
    std::string context;           // the template's variables, JSON
    std::string tools;             // the request's tool definitions, JSON ("[]" without)
    gen::GenerateParams gen;       // max_tokens 0: up to the context size
    std::vector<std::string> stop; // stop strings (in the answer, not the reasoning)
    bool stream = false, include_usage = false;
    bool show_reasoning = true; // reasoning goes to the client
    bool tool_required = false; // tool_choice "required" or a named function
    std::string tool_name;      // a named function's
    bool reasoning_first = true; // the rendered prompt opened the reasoning (ends in <think>)
    std::optional<uint32_t> reasoning_budget; // the request's (0: none); unset: the server's
    bool max_tokens_given = false;            // the client capped the reply
};

// A reasoning effort: a level name, or a number for templates that take one (1..100; 0 as "none").
using Effort = std::variant<std::string, int64_t>;
// Parses "none", a level name or a whole number.
Effort parse_effort(std::string_view text);

// What the chat template does with the thinking controls (thinking_support() finds out by rendering), and the
// server's effort for requests that name none (unset: the template's default).
struct ThinkingOptions {
    bool can_disable = false; // enable_thinking: false renders a prompt that skips the reasoning
    bool numeric = false;     // reasoning_effort takes a number
    bool medium = false;      // reasoning_effort "medium" is a level of its own
    std::optional<Effort> default_effort;
};
ThinkingOptions thinking_support(const chat::Template& tmpl);

// The template variables for an effort (unset: the default's): "none" (or "minimal", 0) ends the reasoning before it
// starts where the template can, else asks for "low"; "medium" becomes "high" where the template has no medium; a
// number needs a template that takes one. Throws BadRequest.
struct ThinkingChoice {
    std::optional<Effort> reasoning_effort;
    std::optional<bool> enable_thinking;
};
ThinkingChoice choose_thinking(const std::optional<Effort>& effort, const ThinkingOptions& t);

// Throws BadRequest. `sampling` holds the model's defaults for the fields the request leaves out.
ChatRequest parse_chat_request(std::string_view body, const gen::SamplingParams& sampling,
                               const ThinkingOptions& thinking = {});

struct ToolCall {
    std::string id, name, arguments; // arguments: a JSON object's text
};

// Incremental output for a streamed chunk.
struct Delta {
    std::string reasoning, content;
    struct Call {
        size_t index;
        std::string id, name, arguments; // id and name only in a call's first delta
    };
    std::vector<Call> calls;
    bool empty() const { return reasoning.empty() && content.empty() && calls.empty(); }
};

// Splits generated tokens into reasoning, answer and tool calls, following the template's markup: when the prompt
// ends in <think>, reasoning runs to </think>; then the answer with tool calls, in one of two forms (by the
// vocabulary): <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call> blocks (special tokens;
// values are JSON unless the tool's schema types them as strings), or DSML text blocks, <｜DSML｜ calls> of
// <｜DSML｜ invoke name="..."> with <｜DSML｜ parameter name="k" string="true|false">v</｜DSML｜ parameter> each
// (string="true": v is text, else JSON).
class OutputParser {
public:
    // tools: the request's tool definitions (JSON); reasoning_first: the prompt opened the reasoning.
    OutputParser(const text::Tokenizer& tok, std::string_view tools, std::vector<std::string> stop,
                 bool reasoning_first = true);
    ~OutputParser();

    // The answer must start with a call (of `name`, when given; its tokens are forced, then the arguments' start).
    void require_call(const std::string& name = {});
    // The token the next one must be, else -1 (GenerateParams::forced).
    int32_t forced() const;
    bool in_reasoning() const { return mode_ == Mode::reasoning; }
    // Returns false once a stop string ended the answer (later tokens are ignored).
    bool push(int32_t token, Delta& out);
    // End of generation: whatever is held back.
    void finish(Delta& out);

    const std::string& reasoning() const { return reasoning_; }
    const std::string& content() const { return content_; }
    const std::vector<ToolCall>& calls() const { return calls_; }
    // The stop string that ended the answer (empty if none did).
    const std::string& stop_matched() const { return stop_matched_; }

private:
    enum class Mode { reasoning, content, name, key, between, value, dsml, done };
    void text(const std::string& s, Delta& out);
    void flush_text(Delta& out);
    void emit_content(Delta& out, bool final);
    void open_call(Delta& out);
    // DSML: complete elements at the start of dsml_ become calls and arguments; false while one is incomplete.
    void parse_dsml(Delta& out);
    void close_call(Delta& out);
    bool string_typed(const std::string& tool, const std::string& key) const;

    struct Tools;
    std::unique_ptr<Tools> tools_;
    std::vector<std::string> stop_;
    text::Utf8Stream stream_;
    Mode mode_ = Mode::reasoning;
    std::string buf_;          // name, key or value being read
    std::string pending_;      // answer text held back (trailing whitespace, a possible stop string's start)
    bool content_started_ = false;
    std::string key_;
    bool call_open_ = false, first_arg_ = true;
    std::string reasoning_, content_, stop_matched_;
    std::vector<ToolCall> calls_;
    const text::Tokenizer& tok_;
    bool require_ = false, named_has_args_ = false;
    std::vector<int32_t> name_tokens_; // a required call's name
    size_t name_seen_ = 0;             // tokens of the first call's name so far
    int32_t think_end_, call_begin_, call_end_, key_begin_, key_end_, value_begin_, value_end_;
    bool dsml_form_ = false;       // tool calls as DSML text
    std::string dsml_;             // DSML text not yet parsed
    std::vector<int32_t> forced_;  // DSML: a required call's opening tokens
    size_t forced_seen_ = 0;
};

// Ends the reasoning after `budget` tokens of it: in the last stretch before (a quarter of the budget, at most 1024
// tokens) as soon as </think> is among the model's three likeliest tokens, at the budget by forcing a short closing
// sentence and </think>.
class ReasoningBudget {
public:
    ReasoningBudget(const text::Tokenizer& tok, uint32_t budget);
    // Each generated token, and whether the reasoning was still open when it came.
    void push(int32_t token, bool reasoning);
    // GenerateParams::forced while the reasoning is open: the token the next must be, else -1.
    int32_t forced(const float* logits, uint32_t n_vocab) const;
    uint32_t thought() const { return thought_; }
    // How the reasoning ended: 0 by itself (or not yet), 1 closed early, 2 cut at the budget.
    int how() const { return how_; }

private:
    std::vector<int32_t> wrap_; // the closing sentence, then </think>
    int32_t think_end_;
    uint32_t budget_, soft_from_, thought_ = 0;
    size_t wrapped_ = 0;
    int how_ = 0;
    mutable int64_t soft_at_ = -1; // thought_ when forced() closed early
};

// Chat Completions response bodies (JSON). A chunk with `role` opens the stream; one with `usage` closes it.
std::string chunk(const std::string& id, int64_t created, const std::string& model, const Delta& d, bool role = false,
                  const char* finish_reason = nullptr);
std::string usage_chunk(const std::string& id, int64_t created, const std::string& model,
                        const gen::GenerateResult& r);
std::string completion(const std::string& id, int64_t created, const std::string& model, const OutputParser& p,
                       const char* finish_reason, const gen::GenerateResult& r);
std::string error_body(const std::string& message, const char* type = "invalid_request_error");
std::string models_body(const std::string& model);

} // namespace mako::server
