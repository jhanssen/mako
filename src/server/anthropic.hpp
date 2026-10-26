#pragma once

#include "gen/sampler.hpp"
#include "server/openai.hpp"
#include "server/protocol.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace mako::server {

// An Anthropic Messages request translated for the chat template and the session (the same form as a Chat
// Completions request): system, text, thinking, tool_use and tool_result blocks; tools with input_schema;
// tool_choice auto, any, tool and none; thinking enabled shows the reasoning as thinking blocks. Throws BadRequest.
// count_only: for count_tokens, where max_tokens is not required.
// Thinking disabled or absent: no reasoning where the template can skip it (choose_thinking's "none"); enabled: by
// budget_tokens, low up to 4096, high up to 16384, else the server's default effort.
ChatRequest parse_messages_request(std::string_view body, const gen::SamplingParams& sampling, bool count_only = false,
                                   const ThinkingOptions& thinking = {});

std::unique_ptr<Protocol> anthropic_protocol(std::string id, std::string model, bool show_reasoning);
std::string anthropic_error(const std::string& message, const char* type);
std::string count_tokens_body(uint32_t tokens);

} // namespace mako::server
