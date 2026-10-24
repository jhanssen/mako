#pragma once

#include "chat/template.hpp"
#include "gen/sampler.hpp"
#include "glm/engine.hpp"
#include "glm/session.hpp"
#include "text/tokenizer.hpp"

#include <memory>
#include <string>

namespace mako::server {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_id;        // reported by /v1/models and in responses
    gen::SamplingParams sampling; // defaults for fields a request leaves out
    bool speculative = true;
};

// OpenAI-compatible HTTP server: /v1/chat/completions (streaming or not, tool calls), /v1/models, /health.
// Generation runs one request at a time.
class Server {
public:
    Server(glm::Engine& engine, const text::Tokenizer& tok, const chat::Template& tmpl, ServerOptions opts);
    ~Server();

    // Serves until stop(); false if the address could not be bound.
    bool run();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mako::server
