#pragma once

#include "chat/template.hpp"
#include "gen/sampler.hpp"
#include "gen/model.hpp"
#include "gen/session.hpp"
#include "server/openai.hpp"
#include "server/status.hpp"
#include "text/tokenizer.hpp"
#include "vision/encoder.hpp"
#include "vision/mmproj.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mako::server {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_id;        // reported by /v1/models and in responses
    gen::SamplingParams sampling; // defaults for fields a request leaves out
    bool speculative = true;
    uint32_t max_waiting = 4; // requests waiting behind the one generating; more get HTTP 503
    // Conversations kept besides the one generating (gen::SessionPool): a request continues the one that holds the
    // most of its prompt; the others are parked within park_bytes, the least recently used given up first.
    uint64_t park_bytes = 1ull << 30;
    uint32_t max_conversations = 16;
    // While other requests wait, the one generating gives the engine up every slice_ms and queues again. Prompts are
    // read in pieces cut at multiples of prefill_piece (0: whole), where turns can change.
    uint32_t slice_ms = 1000;
    uint32_t prefill_piece = 0;
    std::string dump_dir;     // non-empty: each generation request's body is written there (client debugging)
    // Non-empty: every request but /health needs one of them, as "Authorization: Bearer <key>" (OpenAI clients) or
    // "x-api-key: <key>" (Anthropic clients); others get HTTP 401. Empty: requests from web pages are refused instead
    // (a Host header other than a loopback name or address, or an Origin other than a loopback one: HTTP 403), so a
    // site the user visits cannot reach the server through the browser.
    std::vector<std::string> api_keys;
    std::optional<Effort> reasoning_effort; // requests that name none (unset: the template's default)
    // The most reasoning tokens for requests that name no budget (0: none). Either way, a reply capped at 1024
    // tokens or more keeps half of the cap for the answer.
    uint32_t reasoning_budget = 0;
    // Repetition loops (gen::LoopGuard, this many tokens; 0: off) in the reasoning or the answer's text outside code
    // blocks: one in the reasoning closes it, one in the answer ends the reply ("stop"; a whole reply is cut back to
    // the loop's first time through).
    uint32_t loop_guard = 256;
    // For GET /v1/status: the expert cache's counters (called on a generating thread, between steps), its size and
    // the share of the experts it holds.
    std::function<CacheCounters()> cache_counters;
    double cache_gb = 0, resident = 0;
    // Image input (both or neither): the projector file and its encoder; requests with images are refused without.
    const vision::Mmproj* mmproj = nullptr;
    vision::Encoder* vision = nullptr;
};

// Whether a request's credentials (the Authorization or x-api-key header value) carry one of `keys`; each compared in
// constant time.
bool key_matches(const std::string& authorization, const std::string& x_api_key, const std::vector<std::string>& keys);
// Without keys: whether a Host header names this machine's loopback (localhost, 127.0.0.1, [::1], with any port) or is
// empty (a client that sends none).
bool host_allowed(const std::string& host);
// Without keys: whether an Origin header is absent or a loopback page (http(s)://localhost, 127.0.0.1 or [::1]).
bool origin_allowed(const std::string& origin);

// OpenAI-compatible HTTP server: /v1/chat/completions (streaming or not, tool calls), /v1/models, /health, and
// /v1/status (JSON) with /status (a page showing it).
// Requests take turns generating (see ServerOptions::slice_ms), up to max_waiting waiting; each conversation keeps
// its context between requests (see ServerOptions::park_bytes).
class Server {
public:
    Server(gen::Model& model, const text::Tokenizer& tok, const chat::Template& tmpl, ServerOptions opts);
    ~Server();

    // Serves until stop(); false if the address could not be bound.
    bool run();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mako::server
