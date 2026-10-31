// Lending the LLM's memory (server::Models with memory::Lender) to a borrower on loopback, with a stand-in LLM: a
// loan waits for the leases out and unloads the LLM; an LLM lease asks for the memory back, waits for it and loads the
// LLM again; a fixed LLM is never lent. No GPU or model.
//   models_test
#include "memory/borrower.hpp"
#include "memory/lender.hpp"
#include "server/models.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <format>
#include <functional>
#include <print>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

std::atomic<int> alive = 0;

class Stub : public mako::gen::Model {
public:
    Stub() { ++alive; }
    ~Stub() override { --alive; }
    const std::vector<float>& step(int32_t) override { return logits_; }
    const std::vector<float>& prefill(std::span<const int32_t>, const std::function<bool()>&,
                                      const std::function<void(size_t)>&) override
    {
        return logits_;
    }
    uint32_t position() const override { return 0; }
    uint32_t max_context() const override { return 16; }
    uint32_t n_vocab() const override { return 1; }
    void reset() override {}
    Checkpoint checkpoint() override { return {}; }
    void restore(const Checkpoint&) override {}
    std::vector<std::byte> save_positions(uint32_t) const override { return {}; }
    void load_positions(uint32_t, std::span<const std::byte>) override {}
    uint32_t checkpoint_slots() const override { return 1; }

private:
    std::vector<float> logits_ = std::vector<float>(1);
};

bool eventually(const std::function<bool()>& f)
{
    for (int i = 0; i < 200; ++i) {
        if (f())
            return true;
        std::this_thread::sleep_for(25ms);
    }
    return f();
}

} // namespace

int main()
{
    std::atomic<int> loads = 0;
    mako::server::Models models([&] {
        ++loads;
        return mako::server::Models::Llm{.model = std::make_unique<Stub>(), .base = nullptr};
    });
    httplib::Server server;
    models.lender().install(server);
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread t([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string url = std::format("http://127.0.0.1:{}", port);
    check(loads == 1 && alive == 1 && std::string(models.resident()) == "llm", "the LLM loads at the start");

    mako::memory::Borrower b(url, "", "images");
    std::atomic<bool> asked = false;
    b.on_release = [&] { asked = true; };
    {
        auto lease = models.llm();
        std::atomic<bool> got = false;
        std::thread tb([&] {
            b.acquire();
            got = true;
        });
        std::this_thread::sleep_for(300ms);
        check(!got && alive == 1, "a loan waits while an LLM lease is out");
        { auto drop = std::move(lease); }
        tb.join();
        check(got && alive == 0 && std::string(models.resident()) == "lent", "then the LLM is unloaded and lent");
    }
    std::atomic<bool> leased = false;
    std::thread tl([&] {
        auto lease = models.llm();
        leased = true;
    });
    check(eventually([&] { return asked.load(); }) && !leased, "an LLM lease asks for the memory back and waits");
    b.release();
    tl.join();
    check(leased && loads == 2 && alive == 1 && std::string(models.resident()) == "llm",
          "once returned, the LLM loads again");

    asked = false;
    b.acquire();
    check(alive == 0 && std::string(models.resident()) == "lent", "lent again");
    b.release();
    check(eventually([&] { return std::string(models.resident()) == "none"; }) && alive == 0,
          "returned without a request: the LLM loads with the next one");
    { auto lease = models.llm(); }
    check(loads == 3 && alive == 1, "the next lease loads it");

    models.lender().stop();
    server.stop();
    t.join();

    Stub fixed;
    mako::server::Models f(fixed);
    httplib::Server s2;
    f.lender().install(s2);
    const int p2 = s2.bind_to_any_port("127.0.0.1");
    std::thread t2([&] { s2.listen_after_bind(); });
    s2.wait_until_ready();
    bool threw = false;
    try {
        mako::memory::Borrower b2(std::format("http://127.0.0.1:{}", p2), "", "x");
        b2.acquire();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw && std::string(f.resident()) == "llm", "a fixed LLM refuses a loan");
    f.lender().stop();
    s2.stop();
    t2.join();
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
