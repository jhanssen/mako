#include "server/server.hpp"

#include "server/anthropic.hpp"
#include "server/json.hpp"
#include "server/openai.hpp"
#include "server/protocol.hpp"
#include "server/status.hpp"
#include "server/status_page.hpp"
#include "image/png.hpp"
#include "safetensors/safetensors.hpp"
#include "core/log.hpp"
#include "gen/loop_guard.hpp"
#include "gen/pool.hpp"
#include "vision/image.hpp"
#include "vision/prompt.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <glaze/glaze.hpp>
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <memory>
#include <format>
#include <mutex>
#include <print>
#include <random>
#include <span>

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

constexpr auto kKeepalive = std::chrono::seconds(5);
constexpr auto kProgressEvery = std::chrono::seconds(5);

struct Server::Impl {
    Models& models;
    uint32_t max_context = 0, n_vocab = 0; // the LLM's (the same after a reload)
    const text::Tokenizer& tok;
    const chat::Template& tmpl;
    ServerOptions opts;
    ThinkingOptions thinking;
    gen::SessionPool pool;
    int32_t user_turn = -1; // the template's user-turn token: the system prompt and tools come before the first
    std::atomic<uint32_t> in_flight = 0; // accepted chat requests, generating or waiting
    httplib::Server http;
    std::atomic<uint64_t> next_id = 0; // handlers run on several threads
    StatusBoard board;

    Impl(Models& m, const text::Tokenizer& t, const chat::Template& tm, ServerOptions o)
        : models(m), max_context(m.first_llm().max_context()), n_vocab(m.first_llm().n_vocab()), tok(t), tmpl(tm),
          opts(std::move(o)), thinking(thinking_support(tm)),
          pool(m.first_llm(), opts.park_bytes, opts.max_conversations, std::chrono::milliseconds(opts.slice_ms),
               opts.model_id)
    {
        models.attach(pool, opts.model_id);
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
    // POST /v1/images/generations: OpenAI's fields (prompt, n, size, response_format b64_json, output_format png or
    // jpeg; model, quality, style, background, moderation and user accepted and left alone); extensions
    // negative_prompt, guidance_scale, loras (in place of the start-up LoRAs) and steps; the schedule's end from the
    // options.
    void images(const httplib::Request& req, httplib::Response& res);
    // GET /v1/images/loras[?model=NAME]: the LoRAs requests can name for a model, its default ones with their scales.
    void loras(const httplib::Request& req, httplib::Response& res);
    // GET /v1/images/models: the image models requests can name.
    void image_model_list(httplib::Response& res);
    // The model named (empty: the default). Throws BadRequest for a name it does not know.
    ImageModel image_model(const std::string& name) const;
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
    struct Outcome {
        std::string conv;     // its conversation, for the log
        size_t conversation = 0;
        size_t loop_keep = 0; // a loop ended the answer: the generated tokens up to its first time through
    };
    // Runs request `id` in its turns, on its conversation; on_delta gets each increment (false: the client went
    // away), `idle` each token that adds none, `begin` runs when its first turn comes (false: give up). False when it
    // gave up before generating.
    bool generate(const std::string& id, const std::vector<int32_t>& prompt, ChatRequest& cr, OutputParser& parser,
                  const std::function<bool(const Delta&)>& on_delta, bool& stopped, Images* images,
                  const std::function<bool()>& begin, gen::GenerateResult& r, Outcome& out,
                  const std::function<void()>& idle = {});
    CacheCounters cache() const { return opts.cache_counters ? opts.cache_counters() : CacheCounters{}; }
    // Takes request `id` off the status board when it ends (with its result once finished() is called).
    struct OnBoard {
        StatusBoard& board;
        std::string id;
        ~OnBoard() { board.finished(id, {}, nullptr, 0, {}); }
    };
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

bool Server::Impl::generate(const std::string& id, const std::vector<int32_t>& prompt, ChatRequest& cr,
                            OutputParser& parser, const std::function<bool(const Delta&)>& on_delta, bool& stopped,
                            Images* images, const std::function<bool()>& begin, gen::GenerateResult& r, Outcome& out,
                            const std::function<void()>& idle)
{
    std::vector<std::shared_ptr<const vision::Encoder::Output>> held;
    cr.gen.prefill_piece = opts.prefill_piece;
    // Conversations with the same system prompt and tools share everything before the first user turn.
    if (const auto it = std::ranges::find(prompt, user_turn); user_turn >= 0 && it != prompt.end())
        cr.gen.fork_point = size_t(it - prompt.begin());
    cr.gen.stop_tokens = {tok.eos(), tok.eot(), tok.find("<|observation|>")};
    const uint32_t room = max_context - uint32_t(prompt.size());
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
    // The loop guard sees the reasoning and the answer's text outside code blocks; `base` is the generated tokens
    // before what it holds.
    std::optional<gen::LoopGuard> guard;
    if (opts.loop_guard)
        guard.emplace(opts.loop_guard);
    const int32_t think_end = tok.find("</think>");
    size_t generated = 0, base = 0;
    bool close_reasoning = false, in_code = false;
    std::string fence_tail; // the answer's last two characters, for a ``` split across deltas
    size_t& loop_keep = out.loop_keep;
    loop_keep = 0;
    if (rb || cr.tool_required || guard)
        cr.gen.forced = [&, nv = n_vocab](const float* logits) {
            if (close_reasoning && parser.in_reasoning())
                return think_end;
            if (rb && parser.in_reasoning())
                return rb->forced(logits, nv);
            return parser.forced();
        };
    // A line every kProgressEvery or more while a long prompt is read.
    struct Reading {
        std::chrono::steady_clock::time_point start, last;
        size_t from = 0;
        bool started = false;
    } reading;
    cr.gen.progress = [&](size_t read, size_t total) {
        const auto now = std::chrono::steady_clock::now();
        if (!reading.started) {
            reading = {now, now, read, true};
            return;
        }
        if (now - reading.last < kProgressEvery || read <= reading.from)
            return;
        reading.last = now;
        const double s = std::chrono::duration<double>(now - reading.start).count();
        const double rate = double(read - reading.from) / s;
        log::info("{}: prompt read {} of {} tokens ({:.0f}%), {:.0f} tok/s, ~{:.0f} s left", id, read, total,
                  100.0 * read / total, rate, double(total - read) / rate);
    };
    bool client_gone = false;
    gen::SessionPool::Pick pick;
    const auto start = [&](gen::GenerateParams& p) {
        if (begin && !begin())
            return false;
        board.started(id);
        if (images && !images->prepared.empty()) {
            encode_images(*images, cr, held);
            p.input_rows = cr.gen.input_rows;
        }
        return true;
    };
    // The LLM in memory for the whole request (an image request waits for it to end, then swaps).
    const auto lease = models.llm();
    const bool ran = pool.generate(prompt, cr.gen, [&](int32_t token) {
        Delta d;
        if (rb)
            rb->push(token, parser.in_reasoning());
        const bool reasoning = parser.in_reasoning();
        const bool more = parser.push(token, d);
        if (!d.empty() && !on_delta(d)) {
            client_gone = true;
            return false;
        }
        if (d.empty() && idle)
            idle();
        stopped = !more;
        ++generated;
        if (generated == 1 || generated % 64 == 0) {
            pool.snapshot();
            if (generated == 1)
                board.decoding(id, cache());
            else
                board.cache(cache());
        }
        board.generated(id, generated);
        if (!guard || !more)
            return more;
        const std::string seen = fence_tail + d.content;
        for (size_t at = 0; (at = seen.find("```", at)) != std::string::npos; at += 3)
            in_code = !in_code;
        fence_tail = seen.substr(seen.size() - std::min<size_t>(seen.size(), 2));
        if (!parser.in_text() || in_code || token == think_end) {
            guard->reset();
            base = generated;
            if (token == think_end)
                close_reasoning = false;
            return true;
        }
        if (!guard->push(token))
            return true;
        log::info("repetition loop in the {} ({} tokens a time through, from generated token {})",
                  reasoning ? "reasoning" : "answer", guard->period(), base + guard->start());
        if (reasoning && think_end >= 0) {
            close_reasoning = true;
            guard->reset();
            base = generated;
            return true;
        }
        loop_keep = base + guard->start() + guard->period();
        return false;
    }, r, pick, start);
    if (!ran)
        return false;
    out.conversation = pick.index;
    out.conv = std::format("conversation {}{}{}, {} kept ({:.2f} of {:.1f} GB parked){}", pick.index,
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
    if (prompt.size() >= max_context)
        return fail(400, std::format("the prompt is {} tokens; the context holds {}", prompt.size(),
                                     max_context));
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
            Outcome out;
            board.arrived(id, prompt.size());
            const OnBoard on_board{board, id};
            if (!generate(id, prompt, cr, parser, [&](const Delta&) { return !closed(); }, stopped, images.get(),
                          [&] { return !closed(); }, r, out))
                return; // gave up while waiting
            const size_t loop_keep = out.loop_keep;
            const End e = loop_keep ? End::stop : end_of(r, parser, stopped);
            log_result(id, r, end_name(e), out.conv);
            board.finished(id, r, end_name(e), out.conversation, cache());
            if (loop_keep) { // the reply up to the loop's first time through
                OutputParser kept(tok, cr.tools, cr.stop, cr.reasoning_first);
                Delta d;
                for (size_t i = 0; i < loop_keep; ++i)
                    kept.push(r.tokens[i], d);
                kept.finish(d);
                res.set_content(proto->whole(e, kept, r), "application/json");
                return;
            }
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
        auto last = std::chrono::steady_clock::now();
        const auto write = [&](const std::string& s) {
            if (s.empty())
                return true;
            last = std::chrono::steady_clock::now();
            return sink.write(s.data(), s.size());
        };
        // An SSE comment after a few seconds without output (waiting for a turn, a long prefill, hidden reasoning):
        // clients drop streams idle for their body timeout. Polled on this thread.
        const auto keepalive = [&] {
            return std::chrono::steady_clock::now() - last < kKeepalive || write(": keepalive\n\n");
        };
        struct Restore {
            std::function<bool()>& f;
            std::function<bool()> saved;
            ~Restore() { f = std::move(saved); }
        } restore{shared->gen.cancelled, shared->gen.cancelled};
        shared->gen.cancelled = [&] { return !keepalive() || restore.saved(); };
        try {
            OutputParser parser(tok, shared->tools, shared->stop, shared->reasoning_first);
            bool stopped = false;
            gen::GenerateResult r;
            Outcome out; // a loop ends a stream where it was found: what it sent cannot be taken back
            board.arrived(id, prompt.size());
            const OnBoard on_board{board, id};
            const auto begin = [&] { return !shared->gen.cancelled() && write(proto->begin(uint32_t(prompt.size()))); };
            // A delta the protocol does not send (hidden reasoning) still keeps the stream alive and notices a close.
            const auto on_delta = [&](const Delta& d) {
                const std::string s = proto->delta(d);
                return s.empty() ? keepalive() : write(s);
            };
            if (!generate(id, prompt, *shared, parser, on_delta, stopped, images.get(), begin, r, out,
                          [&] { keepalive(); }))
                return false; // gave up while waiting
            const End e = out.loop_keep ? End::stop : end_of(r, parser, stopped);
            log_result(id, r, end_name(e), out.conv);
            board.finished(id, r, end_name(e), out.conversation, cache());
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

namespace {

std::string base64(std::span<const uint8_t> in)
{
    static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        const uint32_t v = uint32_t(in[i]) << 16 | (i + 1 < in.size() ? uint32_t(in[i + 1]) << 8 : 0) |
                           (i + 2 < in.size() ? uint32_t(in[i + 2]) : 0);
        out += kTable[v >> 18];
        out += kTable[(v >> 12) & 63];
        out += i + 1 < in.size() ? kTable[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? kTable[v & 63] : '=';
    }
    return out;
}

} // namespace

// Image prompts' text encoders and caches grow with the prompt.
constexpr size_t kMaxImagePrompt = 16384;

void Server::Impl::images(const httplib::Request& req, httplib::Response& res)
{
    auto fail = [&](int status, const std::string& message, const char* type) {
        res.status = status;
        res.set_content(error_body(message, type), "application/json");
    };
    if (!models.has_image())
        return fail(404, "this server has no image model (start it with --image-model)", "not_found_error");
    glz::generic body;
    if (glz::read_json(body, req.body) || !body.is_object())
        return fail(400, "the body is not a JSON object", "invalid_request_error");
    if (!body.contains("prompt") || !body["prompt"].is_string() || body["prompt"].get_string().empty())
        return fail(400, "prompt is required", "invalid_request_error");
    const std::string prompt = body["prompt"].get_string();
    if (prompt.size() > kMaxImagePrompt)
        return fail(400, std::format("prompt is longer than {} bytes", kMaxImagePrompt), "invalid_request_error");
    uint32_t n = 1, width = 1024, height = 1024;
    if (body.contains("n")) {
        if (!body["n"].is_number() || body["n"].get_number() < 1 || body["n"].get_number() > 4)
            return fail(400, "n must be 1 to 4", "invalid_request_error");
        n = uint32_t(body["n"].get_number());
    }
    if (body.contains("size") && body["size"].is_string() && body["size"].get_string() != "auto") {
        const std::string sz = body["size"].get_string();
        unsigned w = 0, h = 0;
        char x = 0;
        if (std::sscanf(sz.c_str(), "%u%c%u", &w, &x, &h) != 3 || x != 'x' || w % 32 || h % 32 || w < 256 ||
            h < 256 || uint64_t(w) * h > 2048ull * 2048)
            return fail(400, "size must be WIDTHxHEIGHT, multiples of 32 from 256, at most 2048x2048 pixels",
                        "invalid_request_error");
        width = w;
        height = h;
    }
    if (body.contains("response_format") && body["response_format"].is_string() &&
        body["response_format"].get_string() != "b64_json")
        return fail(400, "response_format: only b64_json (images are not stored)", "invalid_request_error");
    // Extensions: a negative prompt (classifier-free guidance) and its scale.
    std::optional<std::string> negative;
    ImageModel model;
    try {
        model = image_model(body.contains("model") && body["model"].is_string() ? body["model"].get_string() : "");
    } catch (const std::exception& e) {
        return fail(400, e.what(), "invalid_request_error");
    }
    std::optional<float> guidance = model.guidance;
    if (body.contains("negative_prompt") && !body["negative_prompt"].is_null()) {
        if (!body["negative_prompt"].is_string())
            return fail(400, "negative_prompt must be a string", "invalid_request_error");
        negative = body["negative_prompt"].get_string();
        if (negative->size() > kMaxImagePrompt)
            return fail(400, std::format("negative_prompt is longer than {} bytes", kMaxImagePrompt),
                        "invalid_request_error");
    }
    if (body.contains("guidance_scale") && !body["guidance_scale"].is_null()) {
        if (!body["guidance_scale"].is_number() || body["guidance_scale"].get_number() < 0 ||
            body["guidance_scale"].get_number() > 30)
            return fail(400, "guidance_scale must be a number from 0 to 30", "invalid_request_error");
        guidance = float(body["guidance_scale"].get_number());
    }
    uint32_t steps = model.steps;
    if (body.contains("steps") && !body["steps"].is_null()) {
        const auto& v = body["steps"];
        if (!v.is_number() || v.get_number() < 0 || v.get_number() > 100 || v.get_number() != std::floor(v.get_number()))
            return fail(400, "steps must be a whole number from 0 (the model's) to 100", "invalid_request_error");
        steps = uint32_t(v.get_number());
    }
    image::Loras loras = model.loras;
    if (body.contains("loras") && !body["loras"].is_null()) {
        try {
            std::string text;
            (void)glz::write_json(body["loras"], text);
            loras = resolve_loras(text, lora_choices(model.loras, model.lora_dir));
        } catch (const std::exception& e) {
            return fail(400, std::format("loras: {}", e.what()), "invalid_request_error");
        }
    }
    bool jpeg = false;
    if (body.contains("output_format") && body["output_format"].is_string()) {
        const std::string f = body["output_format"].get_string();
        if (f != "png" && f != "jpeg")
            return fail(400, "output_format: png or jpeg", "invalid_request_error");
        jpeg = f == "jpeg";
    }
    if (in_flight.fetch_add(1) > opts.max_waiting) {
        --in_flight;
        return fail(503, std::format("{} requests are already waiting", opts.max_waiting), "server_busy");
    }
    const auto slot = std::shared_ptr<void>(nullptr, [this](void*) { --in_flight; });
    const std::string id = std::format("img-{}", next_id.fetch_add(1) + 1);
    std::string lora_names;
    for (const auto& [path, scale] : loras)
        lora_names += std::format("{}{} {:g}", lora_names.empty() ? ", LoRAs " : ", ",
                                  std::filesystem::path(path).stem().string(), scale);
    log::info("{}: {} image{} {}x{} with {}{}{}, waiting for the image model", id, n, n > 1 ? "s" : "", width, height,
              model.name, negative ? ", a negative prompt" : "", lora_names);
    std::string data = "[";
    try {
        auto engine = models.image(model, loras);
        std::random_device rd;
        for (uint32_t i = 0; i < n; ++i) {
            image::Engine::Request r;
            r.prompt = prompt;
            r.width = width;
            r.height = height;
            r.seed = (uint64_t(rd()) << 32) | rd();
            r.steps = steps;
            r.terminal = model.terminal;
            r.negative = negative;
            r.guidance = guidance;
            auto last = std::chrono::steady_clock::now();
            const auto t0 = last;
            r.progress = [&](uint32_t step, uint32_t steps) {
                const auto now = std::chrono::steady_clock::now();
                if (now - last < kProgressEvery && step < steps)
                    return;
                last = now;
                const double s = std::chrono::duration<double>(now - t0).count();
                log::info("{}: image {} of {}, step {} of {} ({:.1f} s a step, ~{:.0f} s left)", id, i + 1, n, step,
                          steps, s / step, s / step * (steps - step));
            };
            const auto img = engine->generate(r);
            auto bytes = image::encode_png(image::to_rgba8(img), width, height, false, jpeg);
            if (!jpeg) {
                image::Provenance prov = engine->provenance(r);
                prov.model = model.name;
                for (const auto& [path, scale] : loras)
                    prov.loras.emplace_back(std::filesystem::path(path).stem().string(), scale);
                bytes = image::with_provenance(std::move(bytes), prov);
            }
            log::info("{}: image {} of {} in {:.1f} s (seed {})", id, i + 1, n,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), r.seed);
            data += std::format(R"({}{{"b64_json":"{}"}})", i ? "," : "", base64(bytes));
        }
    } catch (const std::exception& e) {
        log::error("{}: {}", id, e.what());
        return fail(500, e.what(), "server_error");
    }
    data += "]";
    const int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    res.set_content(std::format(R"({{"created":{},"data":{},"output_format":"{}","size":"{}x{}"}})", created, data,
                                jpeg ? "jpeg" : "png", width, height),
                    "application/json");
}

ImageModel Server::Impl::image_model(const std::string& name) const
{
    if (!opts.image_models.empty() && (!name.empty() || !std::filesystem::is_directory(opts.image_model))) {
        const std::string want = name.empty() ? opts.image_model : name;
        auto list = image_models(opts.image_models);
        if (want.empty() && !list.empty()) // no default named: the first
            return list.front();
        for (auto& m : list)
            if (m.name == want)
                return m;
        throw BadRequest(std::format("no image model named \"{}\" (GET /v1/images/models lists them)", want));
    }
    const std::string dir = opts.image_model;
    const std::string own = std::filesystem::path(dir).lexically_normal().filename().string();
    if (!name.empty() && name != own)
        throw BadRequest(std::format("no image model named \"{}\" (this server has only {})", name, own));
    return load_image_model(dir, own);
}

void Server::Impl::image_model_list(httplib::Response& res)
{
    if (!models.has_image()) {
        res.status = 404;
        res.set_content(error_body("this server has no image models (start it with --image-models)", "not_found_error"),
                        "application/json");
        return;
    }
    std::vector<ImageModel> list;
    if (!opts.image_models.empty())
        list = image_models(opts.image_models);
    std::string def, loaded = models.image_name();
    try {
        const ImageModel d = image_model("");
        def = d.name;
        if (!std::ranges::contains(list, d.name, &ImageModel::name))
            list.insert(list.begin(), d);
    } catch (const std::exception& e) {
        log::warn("the default image model: {}", e.what());
    }
    std::string data;
    for (const ImageModel& m : list) {
        data += std::format(R"({}{{"id":{},"object":"model","default":{},"loaded":{})", data.empty() ? "" : ",",
                            json::quoted(m.name), m.name == def, m.name == loaded);
        data += std::format(R"(,"family":{})", json::quoted(m.family));
        if (!m.description.empty())
            data += std::format(R"(,"description":{})", json::quoted(m.description));
        if (!m.prompting.empty())
            data += std::format(R"(,"prompting":{})", json::quoted(m.prompting));
        if (m.steps)
            data += std::format(R"(,"steps":{})", m.steps);
        if (m.guidance)
            data += std::format(R"(,"guidance":{:g})", *m.guidance);
        std::string l;
        for (const auto& [path, scale] : m.loras)
            l += std::format(R"({}{{"name":{},"scale":{:g}}})", l.empty() ? "" : ",",
                             json::quoted(std::filesystem::path(path).stem().string()), scale);
        data += std::format(R"(,"loras":[{}]}})", l);
    }
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
}

void Server::Impl::loras(const httplib::Request& req, httplib::Response& res)
{
    if (!models.has_image()) {
        res.status = 404;
        res.set_content(error_body("this server has no image models (start it with --image-models)", "not_found_error"),
                        "application/json");
        return;
    }
    ImageModel model;
    try {
        model = image_model(req.has_param("model") ? req.get_param_value("model") : "");
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(error_body(e.what(), "invalid_request_error"), "application/json");
        return;
    }
    std::string data;
    for (const auto& c : lora_choices(model.loras, model.lora_dir)) {
        data += std::format(R"({}{{"name":{})", data.empty() ? "" : ",", json::quoted(c.name));
        if (c.scale)
            data += std::format(R"(,"default_scale":{:g})", *c.scale);
        const auto notes = lora_notes(c.path);
        if (!notes.trigger.empty())
            data += std::format(R"(,"trigger":{})", json::quoted(notes.trigger));
        if (!notes.description.empty())
            data += std::format(R"(,"description":{})", json::quoted(notes.description));
        if (notes.scale_range)
            data += std::format(R"(,"scale_range":[{:g},{:g}])", notes.scale_range->first, notes.scale_range->second);
        data += "}";
    }
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
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

std::vector<LoraChoice> lora_choices(const image::Loras& defaults, const std::string& dir)
{
    namespace fs = std::filesystem;
    std::vector<LoraChoice> out;
    auto known = [&](const std::string& name) { return std::ranges::contains(out, name, &LoraChoice::name); };
    for (const auto& [path, scale] : defaults) {
        const std::string name = fs::path(path).stem().string();
        if (!known(name))
            out.push_back({name, path, scale});
    }
    if (dir.empty())
        return out;
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".safetensors" && e.is_regular_file(ec))
            files.push_back(e.path());
    std::ranges::sort(files);
    for (const auto& f : files)
        if (!known(f.stem().string()))
            out.push_back({f.stem().string(), f.string(), std::nullopt});
    return out;
}

LoraNotes lora_notes(const std::string& path)
{
    LoraNotes n;
    const auto sidecar = std::filesystem::path(path).replace_extension(".json");
    if (std::ifstream in(sidecar); in) {
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        glz::generic v;
        if (glz::read_json(v, text) || !v.is_object()) {
            log::warn("{}: not a JSON object; its notes left out", sidecar.string());
        } else {
            if (v.contains("trigger") && v["trigger"].is_string())
                n.trigger = v["trigger"].get_string();
            if (v.contains("description") && v["description"].is_string())
                n.description = v["description"].get_string();
            if (v.contains("scale_range")) {
                auto& r = v["scale_range"];
                if (r.is_array() && r.get_array().size() == 2 && r[0].is_number() && r[1].is_number() &&
                    r[0].get_number() <= r[1].get_number())
                    n.scale_range = {float(r[0].get_number()), float(r[1].get_number())};
                else
                    log::warn("{}: scale_range is not [low, high]; left out", sidecar.string());
            }
        }
    }
    if (!n.trigger.empty() && !n.description.empty())
        return n;
    auto f = safetensors::File::open(path);
    if (!f)
        return n;
    const auto& m = f->metadata();
    auto first = [&](std::initializer_list<const char*> keys) -> std::string {
        for (const char* k : keys)
            if (auto it = m.find(k); it != m.end() && !it->second.empty())
                return it->second;
        return {};
    };
    if (n.trigger.empty())
        n.trigger = first({"trigger_word", "modelspec.trigger_phrase"});
    if (n.description.empty())
        n.description = first({"modelspec.description"});
    return n;
}

image::Loras resolve_loras(std::string_view json, const std::vector<LoraChoice>& choices)
{
    glz::generic v;
    if (glz::read_json(v, json) || !v.is_array())
        throw std::invalid_argument("an array of {\"name\", \"scale\"} objects");
    image::Loras out;
    std::vector<std::string> names;
    for (auto& e : v.get_array()) {
        if (!e.is_object() || !e.contains("name") || !e["name"].is_string())
            throw std::invalid_argument("each entry needs a \"name\"");
        std::string name = e["name"].get_string();
        if (name.ends_with(".safetensors"))
            name.resize(name.size() - std::string_view(".safetensors").size());
        const auto it = std::ranges::find(choices, name, &LoraChoice::name);
        if (it == choices.end())
            throw std::invalid_argument(std::format("no LoRA named \"{}\" (GET /v1/images/loras lists them)", name));
        if (std::ranges::contains(names, name))
            throw std::invalid_argument(std::format("\"{}\" named twice", name));
        float scale = 1.0f;
        if (e.contains("scale") && !e["scale"].is_null()) {
            if (!e["scale"].is_number() || !std::isfinite(e["scale"].get_number()) ||
                std::abs(e["scale"].get_number()) > 10)
                throw std::invalid_argument(std::format("\"{}\": scale must be a number from -10 to 10", name));
            scale = float(e["scale"].get_number());
        }
        names.push_back(name);
        out.emplace_back(it->path, scale);
    }
    return out;
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

Server::Server(Models& models, const text::Tokenizer& tok, const chat::Template& tmpl, ServerOptions opts)
    : impl_(std::make_unique<Impl>(models, tok, tmpl, std::move(opts)))
{
    auto& h = impl_->http;
    h.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        const auto& keys = impl_->opts.api_keys;
        const bool ok = req.path == "/health" || req.path == "/status" || // the page holds no data
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
    const auto list_models = [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(models_body(impl_->opts.model_id), "application/json");
    };
    const auto images = [this](const httplib::Request& req, httplib::Response& res) { impl_->images(req, res); };
    h.Post("/v1/images/generations", images);
    h.Post("/images/generations", images);
    const auto loras = [this](const httplib::Request& req, httplib::Response& res) { impl_->loras(req, res); };
    h.Get("/v1/images/loras", loras);
    h.Get("/images/loras", loras);
    const auto image_list = [this](const httplib::Request&, httplib::Response& res) { impl_->image_model_list(res); };
    h.Get("/v1/images/models", image_list);
    h.Get("/images/models", image_list);
    h.Get("/v1/models", list_models);
    h.Get("/models", list_models);
    h.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    h.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        auto& im = *impl_;
        const StatusHeader head{im.opts.model_id, im.max_context,      im.opts.cache_gb,       im.opts.resident,
                                im.pool.budget(),  im.in_flight.load(), im.models.resident()};
        res.set_content(im.board.json(head, im.pool.info()), "application/json");
    });
    h.Get("/status", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(status_page, "text/html; charset=utf-8");
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
