#pragma once

#include "gen/sampler.hpp"
#include "glm/session.hpp"
#include "text/stream.hpp"
#include "text/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mako::server {

// A request the server rejects with HTTP 400.
struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An OpenAI Chat Completions request translated for the chat template and the session.
struct ChatRequest {
    std::string context;           // the template's variables, JSON
    std::string tools;             // the request's tool definitions, JSON ("[]" without)
    glm::GenerateParams gen;       // max_tokens 0: up to the context size
    std::vector<std::string> stop; // stop strings (in the answer, not the reasoning)
    bool stream = false, include_usage = false;
    bool show_reasoning = true; // reasoning goes to the client
    bool tool_required = false; // tool_choice "required" or a named function
    std::string tool_name;      // a named function's
};

// Throws BadRequest. `sampling` holds the model's defaults for the fields the request leaves out.
ChatRequest parse_chat_request(std::string_view body, const gen::SamplingParams& sampling);

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

// Splits generated tokens into reasoning, answer and tool calls, following the template's markup: the prompt ends in
// <think>, reasoning runs to </think>, then the answer with <tool_call>name<arg_key>k</arg_key><arg_value>v
// </arg_value>...</tool_call> blocks. Argument values are JSON unless the tool's schema types them as strings.
class OutputParser {
public:
    // tools: the request's tool definitions (JSON).
    OutputParser(const text::Tokenizer& tok, std::string_view tools, std::vector<std::string> stop);
    ~OutputParser();

    // The answer must start with a call (of `name`, when given; its tokens are forced, then the arguments' start).
    void require_call(const std::string& name = {});
    // The token the next one must be, else -1 (GenerateParams::forced).
    int32_t forced() const;
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
    enum class Mode { reasoning, content, name, key, between, value, done };
    void text(const std::string& s, Delta& out);
    void flush_text(Delta& out);
    void emit_content(Delta& out, bool final);
    void open_call(Delta& out);
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
};

// Chat Completions response bodies (JSON). A chunk with `role` opens the stream; one with `usage` closes it.
std::string chunk(const std::string& id, int64_t created, const std::string& model, const Delta& d, bool role = false,
                  const char* finish_reason = nullptr);
std::string usage_chunk(const std::string& id, int64_t created, const std::string& model,
                        const glm::GenerateResult& r);
std::string completion(const std::string& id, int64_t created, const std::string& model, const OutputParser& p,
                       const char* finish_reason, const glm::GenerateResult& r);
std::string error_body(const std::string& message, const char* type = "invalid_request_error");
std::string models_body(const std::string& model);

} // namespace mako::server
