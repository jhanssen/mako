// The server over HTTP with a scripted stand-in for the engine (the model file gives only the tokenizer and chat
// template): streams carry SSE comments while a request waits for its turn or its prompt is read.
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

// Generates `script` after any prompt (greedy: one-hot logits), its last token repeated past its end. A prefill takes
// prefill_ms, polling `cancelled` every 500 ms as an engine does between groups; a step takes step_ms.
class Scripted : public mako::gen::Model {
public:
    explicit Scripted(uint32_t n_vocab) : logits_(n_vocab, 0.0f) {}

    std::vector<int32_t> script;
    std::atomic<int> prefill_ms = 0, step_ms = 0;

    const std::vector<float>& step(int32_t) override
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms.load()));
        ++pos_;
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
        if (!script.empty())
            logits_[size_t(script[std::min<size_t>(steps_, script.size() - 1)])] = 1.0f;
        ++steps_;
        return logits_;
    }
    std::vector<float> logits_;
    uint32_t pos_ = 0;
    size_t steps_ = 0;
};

// A streamed chat completion's raw body.
std::string stream(int port, const std::string& text)
{
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(std::chrono::seconds(60));
    std::string body;
    const std::string req = std::format(
        R"({{"model":"m","stream":true,"temperature":0,"messages":[{{"role":"user","content":"{}"}}]}})", text);
    cli.Post("/v1/chat/completions", {}, req, "application/json", [&](const char* d, size_t n) {
        body.append(d, n);
        return true;
    });
    return body;
}

// The streamed answer: the content deltas joined.
std::string content(const std::string& body)
{
    std::string out;
    for (size_t at = 0; (at = body.find(R"("content":")", at)) != std::string::npos;) {
        at += 11;
        const size_t end = body.find('"', at);
        out += body.substr(at, end - at);
    }
    return out;
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

    server.stop();
    t.join();
    std::println("{}", failures ? std::format("{} failed", failures) : "all passed");
    return failures ? 1 : 0;
}
