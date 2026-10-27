#include "server/server.hpp"

#include "server/anthropic.hpp"
#include "server/openai.hpp"
#include "server/protocol.hpp"
#include "core/log.hpp"
#include "gen/pool.hpp"
#include "vision/image.hpp"
#include "vision/prompt.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <list>
#include <memory>
#include <format>
#include <mutex>
#include <print>

namespace mako::server {

namespace {

const char* end_name(End e)
{
    switch (e) {
    case End::stop: return "stop";
    case End::tool_calls: return "tool calls";
    case End::length: return "length";
    case End::stop_sequence: return "stop sequence";
    case End::cancelled: return "cancelled";
    }
    return "?";
}

void log_result(const std::string& id, const gen::GenerateResult& r, const char* finish, const std::string& conv)
{
    const auto rate = [](size_t n, double ms) { return ms > 0 ? n / ms * 1e3 : 0.0; };
    const size_t read = r.prompt_tokens - r.reused_tokens, n = r.tokens.size();
    log::info("{}: prompt {} tokens ({} reused), prefill {} in {:.0f} ms ({:.0f} tok/s), generated {} in {:.0f} ms "
              "({:.1f} tok/s, {:.1f} ms/token){}, {}; {}",
                 id, r.prompt_tokens, r.reused_tokens, read, r.prefill_ms, rate(read, r.prefill_ms), n, r.decode_ms,
                 rate(n, r.decode_ms), r.decode_ms / double(std::max<size_t>(n, 1)),
                 r.drafts ? std::format(", {:.0f}% of drafts kept", 100.0 * r.accepted / r.drafts) : "", finish, conv);
}

} // namespace

struct Server::Impl {
    gen::Model& engine;
    const text::Tokenizer& tok;
    const chat::Template& tmpl;
    ServerOptions opts;
    ThinkingOptions thinking;
    gen::SessionPool pool;
    int32_t user_turn = -1; // the template's user-turn token: the system prompt and tools come before the first
    std::atomic<uint32_t> in_flight = 0; // accepted chat requests, generating or waiting
    httplib::Server http;
    std::atomic<uint64_t> next_id = 0; // handlers run on several threads

    Impl(gen::Model& e, const text::Tokenizer& t, const chat::Template& tm, ServerOptions o)
        : engine(e), tok(t), tmpl(tm), opts(std::move(o)), thinking(thinking_support(tm)),
          pool(e, opts.park_bytes, opts.max_conversations, std::chrono::milliseconds(opts.slice_ms))
    {
        thinking.default_effort = opts.reasoning_effort;
        user_turn = tok.find("<|user|>");
        if (user_turn < 0)
            user_turn = tok.find("<｜User｜>");
    }

    using ErrorBody = std::function<std::string(const std::string&, const char*)>;
    using MakeProtocol = std::function<std::unique_ptr<Protocol>(const ChatRequest&, const std::string& id)>;
    // A generation request in any protocol: parse makes the request, make its response encoder.
    void serve(const httplib::Request& req, httplib::Response& res, const std::function<ChatRequest()>& parse,
               const ErrorBody& error, const char* id_prefix, const MakeProtocol& make);
    void count_tokens(const httplib::Request& req, httplib::Response& res);
    // A request's images, sized when it is parsed and encoded when it runs.
    struct Images {
        std::vector<vision::Prepared> prepared;
        std::vector<uint64_t> keys;
        std::vector<gen::GenerateParams::PromptRows> rows;
    };
    // The rendered prompt with each image placeholder expanded to its image's rows (the images decoded and sized,
    // not yet encoded). Throws BadRequest without a projector.
    std::vector<int32_t> expand(ChatRequest& cr, std::vector<int32_t> prompt, Images& images) const;
    // Encodes the images (or takes them from the cache) and points the request's input rows at them; the outputs
    // are kept alive in `held`. In the request's turn.
    void encode_images(Images& images, ChatRequest& cr,
                       std::vector<std::shared_ptr<const vision::Encoder::Output>>& held);
    std::list<std::pair<uint64_t, std::shared_ptr<const vision::Encoder::Output>>> image_cache; // recent first
    // Runs the request in its turns, on its conversation; on_delta gets each increment (false: the client went away),
    // `begin` runs when its first turn comes (false: give up). False when it gave up before generating; `conv`
    // describes its conversation for the log.
    bool generate(const std::vector<int32_t>& prompt, ChatRequest& cr, OutputParser& parser,
                  const std::function<bool(const Delta&)>& on_delta, bool& stopped, Images* images,
                  const std::function<bool()>& begin, gen::GenerateResult& r, std::string& conv);
};

std::vector<int32_t> Server::Impl::expand(ChatRequest& cr, std::vector<int32_t> prompt, Images& images) const
{
    if (cr.images.empty())
        return prompt;
    if (!opts.vision || !opts.mmproj)
        throw BadRequest("this server takes no images (start it with --mmproj)");
    std::vector<uint32_t> counts;
    for (const std::string& bytes : cr.images) {
        vision::Prepared p;
        try {
            p = vision::prepare(opts.mmproj->prep, vision::decode_image(std::as_bytes(std::span(bytes))));
        } catch (const std::exception& e) {
            throw BadRequest(std::format("image: {}", e.what()));
        }
        unsigned char d[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(bytes.data(), CC_LONG(bytes.size()), d);
        uint64_t key = 0;
        std::memcpy(&key, d, sizeof(key));
        counts.push_back(vision::Encoder::rows_for(*opts.mmproj, p));
        images.keys.push_back(key);
        images.prepared.push_back(std::move(p));
    }
    cr.images.clear();
    const int32_t ph = tok.find(vision::placeholder(opts.mmproj->prep.kind));
    if (ph < 0)
        throw BadRequest("the model's vocabulary has no image placeholder");
    auto out = vision::expand_images(prompt, ph, counts, images.keys);
    images.rows = std::move(out.rows);
    return std::move(out.tokens);
}

void Server::Impl::encode_images(Images& images, ChatRequest& cr,
                                 std::vector<std::shared_ptr<const vision::Encoder::Output>>& held)
{
    constexpr size_t kCached = 8;
    for (size_t i = 0; i < images.prepared.size(); ++i) {
        const uint64_t key = images.keys[i];
        auto it = std::ranges::find(image_cache, key, &decltype(image_cache)::value_type::first);
        std::shared_ptr<const vision::Encoder::Output> out;
        if (it != image_cache.end()) {
            out = it->second;
            image_cache.erase(it);
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            out = std::make_shared<const vision::Encoder::Output>(opts.vision->encode(images.prepared[i]));
            log::info("image {}x{}: {} rows in {:.0f} ms", images.prepared[i].width, images.prepared[i].height,
                      out->rows, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        image_cache.emplace_front(key, out);
        if (image_cache.size() > kCached)
            image_cache.pop_back();
        if (out->rows != images.rows[i].rows.count)
            throw std::runtime_error("image rows differ from their placeholders");
        images.rows[i].rows.rows = out->embd.data();
        held.push_back(std::move(out));
    }
    cr.gen.input_rows = images.rows;
}

bool Server::Impl::generate(const std::vector<int32_t>& prompt, ChatRequest& cr, OutputParser& parser,
                            const std::function<bool(const Delta&)>& on_delta, bool& stopped, Images* images,
                            const std::function<bool()>& begin, gen::GenerateResult& r, std::string& conv)
{
    std::vector<std::shared_ptr<const vision::Encoder::Output>> held;
    cr.gen.prefill_piece = opts.prefill_piece;
    // Conversations with the same system prompt and tools share everything before the first user turn.
    if (const auto it = std::ranges::find(prompt, user_turn); user_turn >= 0 && it != prompt.end())
        cr.gen.fork_point = size_t(it - prompt.begin());
    cr.gen.stop_tokens = {tok.eos(), tok.eot(), tok.find("<|observation|>")};
    const uint32_t room = engine.max_context() - uint32_t(prompt.size());
    cr.gen.max_tokens = cr.gen.max_tokens ? std::min(cr.gen.max_tokens, room) : room;
    cr.gen.speculative = opts.speculative;
    // The reasoning budget, cut to leave a capped reply room to answer.
    uint32_t budget = cr.reasoning_budget.value_or(opts.reasoning_budget);
    if (cr.max_tokens_given && cr.gen.max_tokens >= 1024) {
        const uint32_t m = cr.gen.max_tokens, cap = m / 2;
        budget = budget ? std::min(budget, cap) : cap;
    }
    std::optional<ReasoningBudget> rb;
    if (budget && cr.reasoning_first)
        rb.emplace(tok, budget);
    if (cr.tool_required)
        parser.require_call(cr.tool_name);
    if (rb || cr.tool_required)
        cr.gen.forced = [&, nv = engine.n_vocab()](const float* logits) {
            if (rb && parser.in_reasoning())
                return rb->forced(logits, nv);
            return parser.forced();
        };
    bool client_gone = false;
    gen::SessionPool::Pick pick;
    const auto start = [&](gen::GenerateParams& p) {
        if (begin && !begin())
            return false;
        if (images && !images->prepared.empty()) {
            encode_images(*images, cr, held);
            p.input_rows = cr.gen.input_rows;
        }
        return true;
    };
    const bool ran = pool.generate(prompt, cr.gen, [&](int32_t token) {
        Delta d;
        if (rb)
            rb->push(token, parser.in_reasoning());
        const bool more = parser.push(token, d);
        if (!d.empty() && !on_delta(d)) {
            client_gone = true;
            return false;
        }
        stopped = !more;
        return more;
    }, r, pick, start);
    if (!ran)
        return false;
    conv = std::format("conversation {}{}{}, {} kept ({:.2f} of {:.1f} GB parked){}", pick.index,
                       pick.fresh     ? " (new)"
                       : pick.forked ? std::format(" (new, from {}'s first {} tokens)", pick.forked_from, pick.usable)
                                     : "", pick.switches ? std::format(", {} turns ({:.0f} ms switching)", pick.switches + 1, pick.switch_ms) : "",
                       pick.conversations, pick.parked / 1e9, pool.budget() / 1e9,
                       pick.evicted ? std::format(", {} given up", pick.evicted) : "");
    Delta d;
    parser.finish(d);
    if (!d.empty() && !client_gone)
        on_delta(d);
    if (rb && rb->how())
        log::info("reasoning {} after {} tokens (budget {})", rb->how() == 1 ? "closed early" : "cut", rb->thought(),
                  budget);
    return true;
}

void Server::Impl::serve(const httplib::Request& req, httplib::Response& res, const std::function<ChatRequest()>& parse,
                         const ErrorBody& error, const char* id_prefix, const MakeProtocol& make)
{
    const auto fail = [&](int status, const std::string& msg, const char* type = "invalid_request_error") {
        res.status = status;
        res.set_content(error(msg, type), "application/json");
    };
    ChatRequest cr;
    std::vector<int32_t> prompt;
    auto images = std::make_shared<Images>();
    if (!opts.dump_dir.empty()) {
        static std::atomic<uint64_t> dumped = 0;
        if (std::FILE* f = std::fopen(std::format("{}/request-{}.json", opts.dump_dir, ++dumped).c_str(), "w")) {
            std::fwrite(req.body.data(), 1, req.body.size(), f);
            std::fclose(f);
        }
    }
    try {
        cr = parse();
        const auto rendered = tmpl.render(cr.context);
        prompt = expand(cr, tok.encode(rendered.text, rendered.input), *images);
        cr.reasoning_first = !prompt.empty() && prompt.back() == tok.find("<think>");
    } catch (const std::exception& e) {
        return fail(400, e.what());
    }
    if (prompt.size() >= engine.max_context())
        return fail(400, std::format("the prompt is {} tokens; the context holds {}", prompt.size(),
                                     engine.max_context()));
    if (in_flight.fetch_add(1) > opts.max_waiting) {
        --in_flight;
        res.set_header("Retry-After", "2");
        return fail(503, std::format("{} requests are already waiting", opts.max_waiting), "server_busy");
    }
    // Held until the response is done (the streaming provider keeps a copy).
    const auto slot = std::shared_ptr<void>(nullptr, [this](void*) { --in_flight; });
    const std::string id = std::format("{}{}", id_prefix, ++next_id);
    std::shared_ptr<Protocol> proto = make(cr, id);
    if (!cr.stream) {
        const auto closed = req.is_connection_closed;
        try {
            OutputParser parser(tok, cr.tools, cr.stop, cr.reasoning_first);
            bool stopped = false;
            cr.gen.cancelled = closed;
            gen::GenerateResult r;
            std::string conv;
            if (!generate(prompt, cr, parser, [&](const Delta&) { return !closed(); }, stopped, images.get(),
                          [&] { return !closed(); }, r, conv))
                return; // gave up while waiting
            const End e = end_of(r, parser, stopped);
            log_result(id, r, end_name(e), conv);
            res.set_content(proto->whole(e, parser, r), "application/json");
        } catch (const std::exception& e) {
            fail(500, e.what(), "server_error");
        }
        return;
    }

    // The provider runs after this handler returns; the check only holds the socket.
    cr.gen.cancelled = req.is_connection_closed;
    auto shared = std::make_shared<ChatRequest>(std::move(cr));
    res.set_chunked_content_provider("text/event-stream", [this, shared, proto, prompt = std::move(prompt), id, slot,
                                                           images](size_t, httplib::DataSink& sink) {
        const auto write = [&](const std::string& s) { return s.empty() || sink.write(s.data(), s.size()); };
        try {
            OutputParser parser(tok, shared->tools, shared->stop, shared->reasoning_first);
            bool stopped = false;
            gen::GenerateResult r;
            std::string conv;
            const auto begin = [&] { return !shared->gen.cancelled() && write(proto->begin(uint32_t(prompt.size()))); };
            if (!generate(prompt, *shared, parser, [&](const Delta& d) { return write(proto->delta(d)); }, stopped,
                          images.get(), begin, r, conv))
                return false; // gave up while waiting
            const End e = end_of(r, parser, stopped);
            log_result(id, r, end_name(e), conv);
            if (e == End::cancelled)
                return false;
            write(proto->end(e, parser, r));
        } catch (const std::exception& e) {
            log::error("{}: {}", id, e.what());
            write(proto->stream_error(e.what()));
        }
        sink.done();
        return true;
    });
}

void Server::Impl::count_tokens(const httplib::Request& req, httplib::Response& res)
{
    try {
        auto cr = parse_messages_request(req.body, opts.sampling, true, thinking);
        const auto rendered = tmpl.render(cr.context);
        Images images;
        res.set_content(count_tokens_body(uint32_t(expand(cr, tok.encode(rendered.text, rendered.input), images).size())),
                        "application/json");
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(anthropic_error(e.what(), "invalid_request_error"), "application/json");
    }
}

bool key_matches(const std::string& authorization, const std::string& x_api_key, const std::vector<std::string>& keys)
{
    const auto same = [](std::string_view got, const std::string& key) {
        if (key.empty() || got.size() != key.size())
            return false;
        unsigned char diff = 0;
        for (size_t i = 0; i < key.size(); ++i)
            diff |= static_cast<unsigned char>(got[i] ^ key[i]);
        return diff == 0;
    };
    constexpr std::string_view bearer = "Bearer ";
    const std::string_view b = authorization.starts_with(bearer) ? std::string_view(authorization).substr(bearer.size())
                                                                  : std::string_view();
    bool ok = false;
    for (const auto& key : keys) { // every key compared, whichever matches
        ok |= same(b, key);
        ok |= same(x_api_key, key);
    }
    return ok;
}

namespace {

// The host part of "name[:port]" or "[v6]:port".
std::string_view host_part(std::string_view h)
{
    if (h.starts_with('['))
        return h.substr(0, h.find(']') + 1);
    return h.substr(0, h.find(':'));
}

bool loopback_name(std::string_view h)
{
    return h == "localhost" || h == "127.0.0.1" || h == "[::1]";
}

} // namespace

bool host_allowed(const std::string& host)
{
    return host.empty() || loopback_name(host_part(host));
}

bool origin_allowed(const std::string& origin)
{
    if (origin.empty())
        return true;
    std::string_view o(origin);
    for (const std::string_view scheme : {"http://", "https://"})
        if (o.starts_with(scheme))
            return loopback_name(host_part(o.substr(scheme.size())));
    return false;
}

Server::Server(gen::Model& engine, const text::Tokenizer& tok, const chat::Template& tmpl, ServerOptions opts)
    : impl_(std::make_unique<Impl>(engine, tok, tmpl, std::move(opts)))
{
    auto& h = impl_->http;
    h.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        const auto& keys = impl_->opts.api_keys;
        const bool ok = req.path == "/health" ||
                        (keys.empty() ? host_allowed(req.get_header_value("Host")) &&
                                            origin_allowed(req.get_header_value("Origin"))
                                      : key_matches(req.get_header_value("Authorization"),
                                                    req.get_header_value("x-api-key"), keys));
        if (ok)
            return httplib::Server::HandlerResponse::Unhandled;
        const char* message = keys.empty() ? "without an API key, requests from web pages or other hosts are refused"
                                           : "invalid API key";
        res.status = keys.empty() ? 403 : 401;
        res.set_content(req.path.starts_with("/v1/messages")
                            ? anthropic_error(message, keys.empty() ? "permission_error" : "authentication_error")
                            : error_body(message, keys.empty() ? "permission_denied" : "invalid_api_key"),
                        "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    const auto chat = [this](const httplib::Request& req, httplib::Response& res) {
        auto& im = *impl_;
        const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
        im.serve(
            req, res, [&] { return parse_chat_request(req.body, im.opts.sampling, im.thinking); },
            [](const std::string& m, const char* t) { return error_body(m, t); }, "chatcmpl-",
            [&](const ChatRequest& cr, const std::string& id) {
                return openai_protocol(id, created, im.opts.model_id, cr.include_usage);
            });
    };
    h.Post("/v1/chat/completions", chat);
    h.Post("/chat/completions", chat);
    const auto messages = [this](const httplib::Request& req, httplib::Response& res) {
        auto& im = *impl_;
        im.serve(
            req, res, [&] { return parse_messages_request(req.body, im.opts.sampling, false, im.thinking); },
            [](const std::string& m, const char* t) {
                const std::string_view type(t);
                return anthropic_error(m, type == "server_busy" ? "overloaded_error"
                                          : type == "server_error" ? "api_error"
                                                                   : t);
            },
            "msg_", [&](const ChatRequest& cr, const std::string& id) {
                return anthropic_protocol(id, im.opts.model_id, cr.show_reasoning);
            });
    };
    h.Post("/v1/messages", messages);
    h.Post("/v1/messages/count_tokens",
           [this](const httplib::Request& req, httplib::Response& res) { impl_->count_tokens(req, res); });
    const auto models = [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(models_body(impl_->opts.model_id), "application/json");
    };
    h.Get("/v1/models", models);
    h.Get("/models", models);
    h.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
}

Server::~Server() = default;

bool Server::run()
{
    return impl_->http.listen(impl_->opts.host, impl_->opts.port);
}

void Server::stop()
{
    impl_->http.stop();
}

} // namespace mako::server
