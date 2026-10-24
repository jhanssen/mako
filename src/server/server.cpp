#include "server/server.hpp"

#include "server/openai.hpp"

#include <httplib.h>

#include <chrono>
#include <format>
#include <mutex>
#include <print>

namespace mako::server {

namespace {

const char* finish_reason(const glm::GenerateResult& r, const OutputParser& p, bool stopped)
{
    switch (r.finish) {
    case glm::Finish::stop_token:
        return p.calls().empty() ? "stop" : "tool_calls";
    case glm::Finish::max_tokens:
    case glm::Finish::context_full:
        return "length";
    case glm::Finish::cancelled:
        return stopped ? "stop" : "cancelled";
    }
    return "stop";
}

void log_result(const std::string& id, const glm::GenerateResult& r, const char* finish)
{
    const size_t n = r.tokens.size();
    std::println(stderr, "{}: prompt {} tokens ({} reused), prefill {:.0f} ms, {} tokens in {:.0f} ms ({:.1f} ms/token){}, {}",
                 id, r.prompt_tokens, r.reused_tokens, r.prefill_ms, n, r.decode_ms,
                 r.decode_ms / double(std::max<size_t>(n, 1)),
                 r.drafts ? std::format(", {:.0f}% of drafts kept", 100.0 * r.accepted / r.drafts) : "", finish);
}

} // namespace

struct Server::Impl {
    glm::Engine& engine;
    const text::Tokenizer& tok;
    const chat::Template& tmpl;
    ServerOptions opts;
    glm::Session session;
    std::mutex gen_mutex; // one generation at a time
    httplib::Server http;
    uint64_t next_id = 0;

    Impl(glm::Engine& e, const text::Tokenizer& t, const chat::Template& tm, ServerOptions o)
        : engine(e), tok(t), tmpl(tm), opts(std::move(o)), session(e)
    {
    }

    void chat(const httplib::Request& req, httplib::Response& res);
    // Runs the request on the session; on_delta gets each increment (false: the client went away).
    glm::GenerateResult generate(const std::vector<int32_t>& prompt, ChatRequest& cr, OutputParser& parser,
                                 const std::function<bool(const Delta&)>& on_delta, bool& stopped);
};

glm::GenerateResult Server::Impl::generate(const std::vector<int32_t>& prompt, ChatRequest& cr, OutputParser& parser,
                                           const std::function<bool(const Delta&)>& on_delta, bool& stopped)
{
    cr.gen.stop_tokens = {tok.eos(), tok.eot(), tok.find("<|observation|>")};
    const uint32_t room = engine.max_context() - uint32_t(prompt.size());
    cr.gen.max_tokens = cr.gen.max_tokens ? std::min(cr.gen.max_tokens, room) : room;
    cr.gen.speculative = opts.speculative;
    bool client_gone = false;
    auto r = session.generate(prompt, cr.gen, [&](int32_t token) {
        Delta d;
        const bool more = parser.push(token, d);
        if (!d.empty() && !on_delta(d)) {
            client_gone = true;
            return false;
        }
        stopped = !more;
        return more;
    });
    Delta d;
    parser.finish(d);
    if (!d.empty() && !client_gone)
        on_delta(d);
    return r;
}

void Server::Impl::chat(const httplib::Request& req, httplib::Response& res)
{
    const auto fail = [&](int status, const std::string& msg, const char* type = "invalid_request_error") {
        res.status = status;
        res.set_content(error_body(msg, type), "application/json");
    };
    ChatRequest cr;
    std::vector<int32_t> prompt;
    try {
        cr = parse_chat_request(req.body, opts.sampling);
        const auto rendered = tmpl.render(cr.context);
        prompt = tok.encode(rendered.text, rendered.input);
    } catch (const std::exception& e) {
        return fail(400, e.what());
    }
    if (prompt.size() >= engine.max_context())
        return fail(400, std::format("the prompt is {} tokens; the context holds {}", prompt.size(),
                                     engine.max_context()));
    const std::string id = std::format("chatcmpl-{}", ++next_id);
    const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
    const std::string& model = opts.model_id;
    if (!cr.stream) {
        std::lock_guard lock(gen_mutex);
        try {
            OutputParser parser(tok, cr.tools, cr.stop);
            bool stopped = false;
            const auto r = generate(prompt, cr, parser, [](const Delta&) { return true; }, stopped);
            const char* finish = finish_reason(r, parser, stopped);
            log_result(id, r, finish);
            res.set_content(completion(id, created, model, parser, finish, r), "application/json");
        } catch (const std::exception& e) {
            fail(500, e.what(), "server_error");
        }
        return;
    }

    auto shared = std::make_shared<ChatRequest>(std::move(cr));
    res.set_chunked_content_provider("text/event-stream", [this, shared, prompt = std::move(prompt), id, created,
                                                           model](size_t, httplib::DataSink& sink) {
        const auto send = [&](const std::string& json) {
            const std::string event = "data: " + json + "\n\n";
            return sink.write(event.data(), event.size());
        };
        std::lock_guard lock(gen_mutex);
        if (!send(chunk(id, created, model, {}, true)))
            return false;
        try {
            OutputParser parser(tok, shared->tools, shared->stop);
            bool stopped = false;
            const auto r = generate(prompt, *shared, parser,
                                    [&](const Delta& d) { return send(chunk(id, created, model, d)); }, stopped);
            const char* finish = finish_reason(r, parser, stopped);
            log_result(id, r, finish);
            if (std::string_view(finish) == "cancelled")
                return false;
            send(chunk(id, created, model, {}, false, finish));
            if (shared->include_usage)
                send(usage_chunk(id, created, model, r));
        } catch (const std::exception& e) {
            std::println(stderr, "{}: {}", id, e.what());
            send(error_body(e.what(), "server_error"));
        }
        const std::string done = "data: [DONE]\n\n";
        sink.write(done.data(), done.size());
        sink.done();
        return true;
    });
}

Server::Server(glm::Engine& engine, const text::Tokenizer& tok, const chat::Template& tmpl, ServerOptions opts)
    : impl_(std::make_unique<Impl>(engine, tok, tmpl, std::move(opts)))
{
    auto& h = impl_->http;
    const auto chat = [this](const httplib::Request& req, httplib::Response& res) { impl_->chat(req, res); };
    h.Post("/v1/chat/completions", chat);
    h.Post("/chat/completions", chat);
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
