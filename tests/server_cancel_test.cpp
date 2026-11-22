// A client cancelling a reply mid-reasoning and going on without it, over HTTP as an agent client does: requests
// with tools and each earlier reply's reasoning_content and content as streamed; the third reply cancelled (the
// connection closed) while it reasons, then a request that leaves that reply out and adds a user message. The
// scripted stand-in replies with other tokens than the template's rendering encodes to (one a character), so the
// context holds every reply respelled. The request after the cancel must read only what follows the second reply
// (not the prompt again from the first reply).
//   server_cancel_test <model.gguf>
#include "chat/template.hpp"
#include "gguf/gguf.hpp"
#include "server/server.hpp"
#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>
#include <httplib.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

// Replies `script` after any prompt (one-hot logits), its last token repeated past its end; counts the tokens
// prefill reads.
class Scripted : public mako::gen::Model {
public:
    explicit Scripted(uint32_t n_vocab) : logits_(n_vocab, 0.0f) {}

    std::vector<int32_t> script;
    std::atomic<int> step_ms = 0;
    std::atomic<size_t> prefilled = 0;

    const std::vector<float>& step(int32_t) override
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms.load()));
        ++pos_;
        return next();
    }
    const std::vector<float>& prefill(std::span<const int32_t> tokens, const std::function<bool()>&,
                                      const std::function<void(size_t)>&) override
    {
        pos_ += uint32_t(tokens.size());
        prefilled += tokens.size();
        steps_ = 0;
        return logits_;
    }
    uint32_t position() const override { return pos_; }
    uint32_t max_context() const override { return 32768; }
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
            logits_[size_t(script[std::min(steps_, script.size() - 1)])] = 1.0f;
        ++steps_;
        return logits_;
    }
    std::vector<float> logits_;
    uint32_t pos_ = 0;
    size_t steps_ = 0;
};

int free_port()
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(a);
    if (fd < 0 || bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
        getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) != 0)
        throw std::runtime_error("no free port");
    close(fd);
    return ntohs(a.sin_port);
}

std::string quoted(const std::string& s)
{
    std::string out;
    (void)glz::write_json(s, out);
    return out;
}

const std::string kTools =
    R"([{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}])";

struct Message {
    std::string role, content, reasoning;
};

std::string messages_json(const std::vector<Message>& ms)
{
    std::string out = "[";
    for (const auto& m : ms) {
        if (out.size() > 1)
            out += ",";
        out += std::format(R"({{"role":"{}","content":{})", m.role, quoted(m.content));
        if (!m.reasoning.empty())
            out += std::format(R"(,"reasoning_content":{})", quoted(m.reasoning));
        out += "}";
    }
    return out + "]";
}

// What a model writes for an assistant message after the generation prompt (the template's rendering past it),
// each plain token's text a character a token (markup tokens whole): other tokens than the rendering encodes to.
std::vector<int32_t> reply_tokens(const mako::text::Tokenizer& tok, const mako::chat::Template& tmpl,
                                  const std::vector<Message>& before, const Message& reply)
{
    const auto ctx = [&](const std::vector<Message>& ms, bool gen) {
        return std::format(R"({{"messages":{},"tools":{},"add_generation_prompt":{}}})", messages_json(ms), kTools,
                           gen);
    };
    std::vector<Message> with = before;
    with.push_back(reply);
    const std::string head = tmpl.render(ctx(before, true)).text, whole = tmpl.render(ctx(with, false)).text;
    if (!whole.starts_with(head))
        throw std::runtime_error("the assistant message does not continue the generation prompt");
    std::vector<int32_t> out;
    for (const int32_t id : tok.encode(std::string_view(whole).substr(head.size()), true)) {
        const std::string text = tok.decode(std::span(&id, 1), false);
        if (text.empty() || (text.size() > 2 && text.front() == '<' && text.back() == '>')) { // markup tokens whole
            out.push_back(id);
            continue;
        }
        for (const char c : text) {
            const auto one = tok.encode(std::string(1, c), false);
            out.insert(out.end(), one.begin(), one.end());
        }
    }
    if (out.empty() || out.back() != tok.eot())
        out.push_back(tok.eot());
    return out;
}

struct Streamed {
    std::string reasoning, content;
    bool cut = false;
};

// A streamed chat completion; with cut_after, the connection is closed after that many reasoning bytes.
Streamed post(int port, const std::vector<Message>& ms, size_t cut_after = 0)
{
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(std::chrono::seconds(60));
    const std::string req = std::format(
        R"({{"model":"m","stream":true,"stream_options":{{"include_usage":true}},"temperature":0,"tools":{},"messages":{}}})",
        kTools, messages_json(ms));
    Streamed s;
    std::string pending;
    cli.Post("/v1/chat/completions", {}, req, "application/json", [&](const char* d, size_t n) {
        pending.append(d, n);
        for (size_t nl; (nl = pending.find('\n')) != std::string::npos; pending.erase(0, nl + 1)) {
            const std::string line = pending.substr(0, nl);
            if (!line.starts_with("data: {"))
                continue;
            glz::generic g;
            if (glz::read_json(g, line.substr(6)) || !g.contains("choices") || g["choices"].get_array().empty())
                continue;
            auto& delta = g["choices"][0]["delta"];
            if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string())
                s.reasoning += delta["reasoning_content"].get_string();
            if (delta.contains("content") && delta["content"].is_string())
                s.content += delta["content"].get_string();
        }
        if (cut_after && s.reasoning.size() >= cut_after) {
            s.cut = true;
            return false;
        }
        return true;
    });
    return s;
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
    mako::server::ServerOptions opts;
    opts.port = free_port();
    opts.model_id = "m";
    mako::server::Models models(model);
    mako::server::Server server(models, *tok, tmpl, opts);
    std::thread t([&] { server.run(); });
    for (int i = 0; i < 100 && !httplib::Client("127.0.0.1", opts.port).Get("/health"); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::filesystem::path log_path = std::filesystem::temp_directory_path() /
                                   std::format("mako_server_cancel_test.{}.log", getpid());
    std::fflush(stderr);
    const int saved_err = dup(2), log_fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    dup2(log_fd, 2);

    std::string system = "You are a coding agent.";
    for (int i = 0; i < 200; ++i)
        system += std::format(" Rule {}: keep the build green.", i);
    std::vector<Message> ms = {{"system", system, ""}, {"user", "Look at the build please", ""}};
    const auto turn = [&](const std::string& reasoning, const std::string& answer) {
        model.script = reply_tokens(*tok, tmpl, ms, {"assistant", answer, reasoning});
        const Streamed s = post(opts.port, ms);
        ms.push_back({"assistant", s.content, s.reasoning});
        return s;
    };
    const Streamed r1 = turn("The user wants the build checked. I will look at it", "The build is green now.");
    ms.push_back({"user", "Now run the tests too", ""});
    const Streamed r2 = turn("They want the tests run as well. Running them", "All the tests pass.");
    // The third reply cancelled while it reasons; the client leaves it out and adds a message.
    ms.push_back({"user", "And check the docs", ""});
    std::string long_reasoning = "Checking the docs";
    for (int i = 0; i < 40; ++i)
        long_reasoning += std::format(" then page {}", i);
    model.script = reply_tokens(*tok, tmpl, ms, {"assistant", "Done.", long_reasoning});
    model.step_ms = 20;
    const Streamed r3 = post(opts.port, ms, 40);
    model.step_ms = 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); // the server sees the connection closed
    ms.push_back({"user", "Stop, do the changelog instead", ""});
    model.script = reply_tokens(*tok, tmpl, ms, {"assistant", "Changelog done.", "Writing the changelog"});
    model.prefilled = 0;
    const Streamed r4 = post(opts.port, ms);
    const size_t read = model.prefilled;

    std::fflush(stderr);
    dup2(saved_err, 2);
    close(log_fd);
    close(saved_err);
    std::string log;
    {
        std::ifstream in(log_path);
        log.assign(std::istreambuf_iterator<char>(in), {});
    }
    std::filesystem::remove(log_path);
    std::fputs(log.c_str(), stderr);

    // The prompt past the second reply (the checkpoint where it ended): the messages after it and the generation
    // prompt, and the second reply's end of turn the context left out (a stop token is not run).
    const auto tokens = [&](const std::vector<Message>& m, bool gen) {
        const auto r = tmpl.render(std::format(R"({{"messages":{},"tools":{},"add_generation_prompt":{}}})",
                                               messages_json(m), kTools, gen));
        return tok->encode(r.text, r.input).size();
    };
    const size_t whole = tokens(ms, true), before = tokens({ms.begin(), ms.begin() + 5}, false);
    check(r1.content == "The build is green now." && r2.content == "All the tests pass.",
          std::format("the replies as scripted ({:?}, {:?})", r1.content, r2.content));
    check(log.find("with other tokens; kept the context's") != std::string::npos,
          "the context keeps the replies' own tokens (respelled)");
    check(r3.cut && r3.content.empty() && log.find("cancelled") != std::string::npos,
          "the third reply cancelled while it reasoned");
    check(r4.content == "Changelog done.", std::format("the reply after it ({:?})", r4.content));
    check(read <= whole - before + 2,
          std::format("after the cancel only the messages after the second reply are read: {} of {} tokens (they are {})",
                      read, whole, whole - before));
    server.stop();
    t.join();
    std::println("{} failures", failures);
    return failures ? 1 : 0;
}
