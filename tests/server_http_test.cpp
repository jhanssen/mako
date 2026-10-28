// The server over HTTP with a scripted stand-in for the engine (the model file gives only the tokenizer and chat
// template): streams carry SSE comments while a request waits for its turn or its prompt is read; repetition loops
// close the reasoning or end the answer, and are left alone in code blocks; /v1/status shows requests as they run and
// after, and the /status page needs no key while /v1/status does.
//   server_http_test <model.gguf>
#include "gguf/gguf.hpp"
#include "server/server.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "ok  " : "FAIL", what);
    if (!ok)
        ++failures;
}

// Generates `script` after any prompt (greedy: one-hot logits), its last token repeated past its end; once fed
// `switch_on`, `after` instead. A prefill takes prefill_ms, polling `cancelled` every 500 ms as an engine does between
// groups; a step takes step_ms.
class Scripted : public mako::gen::Model {
public:
    explicit Scripted(uint32_t n_vocab) : logits_(n_vocab, 0.0f) {}

    std::vector<int32_t> script, after;
    int32_t switch_on = -1;
    std::atomic<int> prefill_ms = 0, step_ms = 0;
    std::atomic<size_t> steps = 0; // since the last prefill

    const std::vector<float>& step(int32_t token) override
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms.load()));
        ++pos_;
        ++steps;
        if (token == switch_on && !switched_) {
            switched_ = true;
            steps_ = 0;
        }
        return next();
    }
    const std::vector<float>& prefill(std::span<const int32_t> tokens, const std::function<bool()>& cancelled) override
    {
        for (int ms = prefill_ms; ms > 0; ms -= 500) {
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min(ms, 500)));
            if (ms > 500 && cancelled && cancelled())
                break;
        }
        pos_ += uint32_t(tokens.size());
        steps_ = 0;
        steps = 0;
        switched_ = false;
        return logits_;
    }
    uint32_t position() const override { return pos_; }
    uint32_t max_context() const override { return 8192; }
    uint32_t n_vocab() const override { return uint32_t(logits_.size()); }
    void reset() override { pos_ = 0, steps_ = 0; }
    Checkpoint checkpoint() override { return {pos_, {}, {}}; }
    void restore(const Checkpoint& cp) override { pos_ = cp.pos, steps_ = 0; }
    std::vector<std::byte> save_positions(uint32_t) const override { return {}; }
    void load_positions(uint32_t, std::span<const std::byte>) override {}
    uint32_t checkpoint_slots() const override { return 4; }

private:
    const std::vector<float>& next()
    {
        std::ranges::fill(logits_, 0.0f);
        const auto& s = switched_ ? after : script;
        if (!s.empty())
            logits_[size_t(s[std::min<size_t>(steps_, s.size() - 1)])] = 1.0f;
        ++steps_;
        return logits_;
    }
    std::vector<float> logits_;
    uint32_t pos_ = 0;
    size_t steps_ = 0;
    bool switched_ = false;
};

// A chat completion's raw body (streamed or whole).
std::string chat(int port, const std::string& text, bool streamed = true)
{
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(std::chrono::seconds(60));
    std::string body;
    const std::string req =
        std::format(R"({{"model":"m","stream":{},"temperature":0,"messages":[{{"role":"user","content":"{}"}}]}})",
                    streamed, text);
    cli.Post("/v1/chat/completions", {}, req, "application/json", [&](const char* d, size_t n) {
        body.append(d, n);
        return true;
    });
    return body;
}

std::string stream(int port, const std::string& text)
{
    return chat(port, text);
}

// A field's string values joined: a streamed answer's deltas, or a whole one (plain text without escapes).
std::string joined(const std::string& body, const std::string& field)
{
    std::string out;
    const std::string key = std::format(R"("{}":")", field);
    for (size_t at = 0; (at = body.find(key, at)) != std::string::npos;) {
        at += key.size();
        const size_t end = body.find('"', at);
        out += body.substr(at, end - at);
    }
    return out;
}

std::string content(const std::string& body)
{
    return joined(body, "content");
}

size_t count(const std::string& s, const std::string& what)
{
    size_t n = 0;
    for (size_t at = 0; (at = s.find(what, at)) != std::string::npos; at += what.size())
        ++n;
    return n;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || !std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argc < 2 ? "" : argv[1]);
        return 77;
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(*f);
    const auto source = f->get_string("tokenizer.chat_template");
    if (!tok || !source) {
        std::println(stderr, "no tokenizer or chat template");
        return 1;
    }
    const mako::chat::Template tmpl{std::string(*source)};
    Scripted model(uint32_t(tok->size()));
    model.script = tok->encode("</think>Hello there.", true);
    model.script.push_back(tok->eot());

    mako::server::ServerOptions opts;
    opts.port = 18734;
    opts.model_id = "m";
    opts.slice_ms = 60000; // a waiting request waits for the whole of the one generating
    mako::server::Server server(model, *tok, tmpl, opts);
    std::thread t([&] { server.run(); });
    for (int i = 0; i < 100 && !httplib::Client("127.0.0.1", opts.port).Get("/health"); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // A prompt read for 7 s: a comment after 5 s of silence, before the answer.
    model.prefill_ms = 7000;
    const std::string slow = stream(opts.port, "one");
    check(slow.find(": keepalive") != std::string::npos && slow.find(": keepalive") < slow.find("Hello"),
          "a comment while the prompt is read");
    check(content(slow) == "Hello there.", std::format("the answer after it ({:?})", content(slow)));

    // A request behind one generating for ~9 s gets comments while it waits.
    model.prefill_ms = 0;
    model.step_ms = 300;
    model.script.insert(model.script.end() - 1, 25, tok->encode(" x", false).front());
    std::string first, second;
    std::thread a([&] { first = stream(opts.port, "two"); });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    second = stream(opts.port, "three");
    a.join();
    check(first.find(": keepalive") == std::string::npos, "no comment while tokens stream");
    check(second.find(": keepalive") != std::string::npos && second.find(": keepalive") < second.find("data:"),
          "a comment while waiting for the turn");
    check(content(second) == content(first) && content(first).starts_with("Hello there. x"),
          "both answers whole");

    // Status while a request generates (~9 s at 300 ms a token) and after it.
    std::string during;
    std::thread b([&] { first = stream(opts.port, "status"); });
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    if (auto res = httplib::Client("127.0.0.1", opts.port).Get("/v1/status"))
        during = res->body;
    b.join();
    std::string after;
    if (auto res = httplib::Client("127.0.0.1", opts.port).Get("/v1/status"))
        after = res->body;
    check(during.find(R"("phase":"generating")") != std::string::npos && during.find(R"("in_flight":1)") != std::string::npos,
          "status during a request: generating, one in flight");
    check(after.find(R"("requests":[])") != std::string::npos &&
              after.find(R"("recent":[{"id":"chatcmpl-)") != std::string::npos &&
              after.find(R"("finish":"stop")") != std::string::npos && after.find(R"("state":"live")") != std::string::npos,
          "status after: no requests, the finished one recent, a live conversation");
    {
        const auto page = httplib::Client("127.0.0.1", opts.port).Get("/status");
        check(page && page->status == 200 && page->body.find("/v1/status") != std::string::npos, "the status page");
    }

    // Loops: a 60-times repeated sentence of ~15 tokens, ended by the guard after 256 + one time through.
    model.step_ms = 0;
    const std::string again = " The value is the same as before, so I will check it once more.";
    const auto loop = [&](const std::string& before, int times, const std::string& tail) {
        std::string text = before;
        for (int i = 0; i < times; ++i)
            text += again;
        auto v = tok->encode(text + tail, true);
        v.push_back(tok->eot());
        return v;
    };
    model.script = loop("</think>Start.", 60, "");
    const std::string looped = stream(opts.port, "four");
    check(model.steps < 400 && looped.find(R"("finish_reason":"stop")") != std::string::npos &&
              content(looped).starts_with("Start." + again) && count(content(looped), again) < 25,
          std::format("a loop in a streamed answer ends it with stop ({} steps, {} times through)", model.steps.load(),
                      count(content(looped), again)));
    const std::string whole = chat(opts.port, "five", false);
    check(content(whole).starts_with("Start. The value") && count(content(whole), " The value") == 1 &&
              whole.find(R"("finish_reason":"stop")") != std::string::npos,
          std::format("a whole answer is cut to the loop's first time through ({:?})", content(whole)));

    model.script = loop("", 60, "");
    model.switch_on = tok->find("</think>");
    model.after = tok->encode("The answer.", false);
    model.after.push_back(tok->eot());
    const std::string thought = stream(opts.port, "six");
    check(count(joined(thought, "reasoning_content"), again) < 25 && content(thought) == "The answer." &&
              thought.find(R"("finish_reason":"stop")") != std::string::npos,
          std::format("a loop in the reasoning closes it; the answer follows ({:?})", content(thought)));
    model.switch_on = -1;

    model.script = loop("</think>Code:\n```\n", 60, "\n```\nDone.");
    const std::string code = chat(opts.port, "seven", false);
    check(count(content(code), again) == 60 && content(code).ends_with("Done."),
          "a repeated line inside a code block is left alone");

    server.stop();
    t.join();

    // With an API key: the page needs none, its data does.
    mako::server::ServerOptions keyed = opts;
    keyed.port = opts.port + 1;
    keyed.api_keys = {"test-key-0123456789"};
    mako::server::Server locked(model, *tok, tmpl, keyed);
    std::thread u([&] { locked.run(); });
    httplib::Client lc("127.0.0.1", keyed.port);
    for (int i = 0; i < 100 && !lc.Get("/health"); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto page = lc.Get("/status");
    const auto bare = lc.Get("/v1/status");
    const auto keyed_res = lc.Get("/v1/status", httplib::Headers{{"Authorization", "Bearer test-key-0123456789"}});
    check(page && page->status == 200 && bare && bare->status == 401 && keyed_res && keyed_res->status == 200,
          "with a key: the page open, /v1/status 401 without the key and 200 with it");
    locked.stop();
    u.join();
    std::println("{}", failures ? std::format("{} failed", failures) : "all passed");
    return failures ? 1 : 0;
}
