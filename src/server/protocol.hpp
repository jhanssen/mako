#pragma once

#include "glm/session.hpp"
#include "server/openai.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace mako::server {

// Why an answer ended.
enum class End { stop, tool_calls, length, stop_sequence, cancelled };
End end_of(const glm::GenerateResult& r, const OutputParser& p, bool stopped);

// How one request's generation becomes response bodies, in a client protocol. Streaming writes begin(), then
// delta() for each increment, then end(); otherwise whole() is the body.
class Protocol {
public:
    virtual ~Protocol() = default;
    virtual std::string begin(uint32_t prompt_tokens) = 0;
    virtual std::string delta(const Delta& d) = 0;
    virtual std::string end(End e, const OutputParser& p, const glm::GenerateResult& r) = 0;
    // An error during a stream (after begin()).
    virtual std::string stream_error(const std::string& message) = 0;
    virtual std::string whole(End e, const OutputParser& p, const glm::GenerateResult& r) = 0;
};

std::unique_ptr<Protocol> openai_protocol(std::string id, int64_t created, std::string model, bool include_usage);

} // namespace mako::server
