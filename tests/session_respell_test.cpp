// gen::Session keeping the context's tokens where the next prompt spells the same text with other tokens, on a
// stand-in model that records what it reads (no GPU).
// - a reply sampled as "." "\n" and rendered again as ".\n": the whole context is reused, only what follows is read;
// - the context ending inside one of the prompt's tokens: the rest of that token's text is encoded again;
// - other text, or a control token where the context has plain text: the prompt diverges as without respelling;
// - input rows after the respelled part move with it;
// - a third turn that spells both earlier replies otherwise: the pool keeps the conversation instead of forking it
//   from before the first reply;
// - a reply cut off and left out of the next prompt, which adds a message instead (a client's cancel): the
//   respelled context is kept up to that prompt's last token, not read again from the first respelled token.
#include "gen/pool.hpp"
#include "gen/session.hpp"

#include <cstring>
#include <format>
#include <memory>
#include <print>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("{} {}", ok ? "  ok  " : "  FAIL", what);
    failures += !ok;
}

// Tokens: 0 "a", 1 "b", 2 "ab", 3 <c> (control), 4 "x", 5 ".", 6 "\n", 7 ".\n", 8 <end> (control), 9 "<c>" (plain).
const std::vector<std::string> kText = {"a", "b", "ab", "<c>", "x", ".", "\n", ".\n", "<end>", "<c>"};
bool control(int32_t id) { return id == 3 || id == 8; }

// Records every token it reads; a checkpoint holds the tokens read so far.
class Recorder : public mako::gen::Model {
public:
    std::vector<int32_t> fed;
    size_t read = 0; // tokens run since the last clear_read()
    std::vector<mako::gen::InputRows> rows;

    void set_input_rows(std::span<const mako::gen::InputRows> r) override
    {
        if (!r.empty())
            rows.assign(r.begin(), r.end());
    }
    const std::vector<float>& step(int32_t token) override
    {
        fed.push_back(token);
        ++read;
        return logits_;
    }
    const std::vector<float>& prefill(std::span<const int32_t> tokens, const std::function<bool()>&,
                                      const std::function<void(size_t)>&) override
    {
        fed.insert(fed.end(), tokens.begin(), tokens.end());
        read += tokens.size();
        return logits_;
    }
    uint32_t position() const override { return uint32_t(fed.size()); }
    uint32_t max_context() const override { return 256; }
    uint32_t n_vocab() const override { return uint32_t(kText.size()); }
    void reset() override { fed.clear(); }
    Checkpoint checkpoint() override
    {
        Checkpoint c;
        c.pos = position();
        c.state.resize(fed.size() * 4);
        std::memcpy(c.state.data(), fed.data(), c.state.size());
        return c;
    }
    void restore(const Checkpoint& c) override
    {
        fed.resize(c.pos);
        std::memcpy(fed.data(), c.state.data(), c.state.size());
    }
    std::vector<std::byte> save_positions(uint32_t) const override { return {}; }
    void load_positions(uint32_t, std::span<const std::byte>) override {}
    uint32_t checkpoint_slots() const override { return 4; }

private:
    std::vector<float> logits_ = std::vector<float>(kText.size());
};

mako::gen::GenerateParams params(std::vector<int32_t> reply)
{
    mako::gen::GenerateParams p;
    p.max_tokens = 64;
    p.stop_tokens = {8};
    p.speculative = false;
    auto at = std::make_shared<size_t>(0);
    p.forced = [reply, at](const float*) { return *at < reply.size() ? reply[(*at)++] : 8; };
    p.text = mako::gen::GenerateParams::Text{
        .piece = [](int32_t id) { return kText[size_t(id)]; },
        .control = control,
        .encode =
            [](std::string_view s) {
                std::vector<int32_t> out; // one character a token (enough for the test's texts)
                for (char c : s)
                    out.push_back(c == 'a' ? 0 : c == 'b' ? 1 : c == 'x' ? 4 : c == '.' ? 5 : 6);
                return out;
            },
    };
    return p;
}

} // namespace

int main()
{
    using V = std::vector<int32_t>;
    {
        Recorder m;
        mako::gen::Session s(m);
        s.generate(V{0, 1, 3}, params({5, 6, 4})); // context: a b <c> . \n x (the stop token is not run)
        m.read = 0;
        const auto r = s.generate(V{0, 1, 3, 7, 4, 8, 0, 3}, params({}));
        check(r.reused_tokens == 6 && m.read == 3 && r.respelled == 3 && r.respelled_at == 3 &&
                  m.fed == V({0, 1, 3, 5, 6, 4, 8, 0, 3}),
              std::format("\".\" \"\\n\" again as \".\\n\": the context reused ({} tokens, {} read, {} respelled)",
                          r.reused_tokens, m.read, r.respelled));
    }
    {
        Recorder m;
        mako::gen::Session s(m);
        s.generate(V{4, 3}, params({0})); // context: x <c> a
        m.read = 0;
        const auto r = s.generate(V{4, 3, 2, 8, 4}, params({}));
        check(r.reused_tokens == 3 && m.fed == V({4, 3, 0, 1, 8, 4}) && m.read == 3,
              std::format("the context ending inside \"ab\": \"b\" encoded again (reused {}, read {})",
                          r.reused_tokens, m.read));
    }
    {
        Recorder m;
        mako::gen::Session s(m);
        s.generate(V{0, 1, 3}, params({5, 6, 4}));
        m.read = 0;
        const auto r = s.generate(V{0, 1, 3, 7, 0, 8, 0, 3}, params({})); // ".\na", not ".\nx"
        check(r.respelled == 0 && r.reused_tokens == 2 && r.diverged && r.diverged->at == 3,
              std::format("other text: diverges as before (reused {}, respelled {})", r.reused_tokens, r.respelled));
    }
    {
        Recorder m;
        mako::gen::Session s(m);
        s.generate(V{0, 1, 3}, params({9, 4})); // the model wrote "<c>" as plain text
        m.read = 0;
        const auto r = s.generate(V{0, 1, 3, 3, 4, 8, 0, 3}, params({}));
        check(r.respelled == 0 && r.reused_tokens == 2,
              std::format("a control token where the context has the same text in plain tokens: diverges (reused {})",
                          r.reused_tokens));
    }
    {
        Recorder m;
        mako::gen::Session s(m);
        s.generate(V{0, 1, 3}, params({5, 6, 4}));
        m.read = 0;
        auto p = params({});
        // Rows at prompt positions 6-7 (two image rows after the reply), which move to 7-8.
        static float rows[2 * 4] = {};
        p.input_rows = {{.rows = {.pos = 6, .count = 2, .rows = rows}, .key = 42}};
        const auto r = s.generate(V{0, 1, 3, 7, 4, 8, 1, 1, 3}, p);
        check(r.respelled == 3 && m.rows.size() == 1 && m.rows[0].pos == 7 && m.rows[0].count == 2,
              std::format("input rows after the respelled part move with it (at {})",
                          m.rows.empty() ? -1 : int(m.rows[0].pos)));
    }
    {
        Recorder m;
        mako::gen::SessionPool pool(m, 1 << 20);
        const auto run = [&](const V& prompt, V reply) {
            mako::gen::GenerateResult r;
            mako::gen::SessionPool::Pick pick;
            m.read = 0;
            pool.generate(prompt, params(std::move(reply)), {}, r, pick);
            return std::pair{r, pick};
        };
        run(V{0, 1, 3}, {5, 6, 4});                                 // context: a b <c> . \n x
        run(V{0, 1, 3, 7, 4, 8, 0, 3}, {5, 6});                     // . \n x again as .\n x; context ... <end> a <c> . \n
        const auto [r, pick] = run(V{0, 1, 3, 7, 4, 8, 0, 3, 7, 8, 1, 3}, {}); // both replies as .\n
        check(!pick.forked && pick.usable == 11 && r.reused_tokens == 11 && m.read == 3 && pool.size() == 1,
              std::format("both replies spelled otherwise: kept, not forked (forked {}, usable {}, read {})",
                          pick.forked, pick.usable, m.read));
    }
    {
        Recorder m;
        mako::gen::SessionPool pool(m, 1 << 20);
        const auto run = [&](const V& prompt, V reply, size_t cut = 0) {
            mako::gen::GenerateResult r;
            mako::gen::SessionPool::Pick pick;
            m.read = 0;
            size_t emitted = 0;
            pool.generate(prompt, params(std::move(reply)), [&](int32_t) { return !cut || ++emitted < cut; }, r, pick);
            return std::pair{r, pick};
        };
        run(V{0, 1, 3}, {5, 6, 4});
        run(V{0, 1, 3, 7, 4, 8, 0, 3}, {5, 6});
        const V turn = {0, 1, 3, 7, 4, 8, 0, 3, 7, 8, 1}; // both replies as .\n, then "b"
        V cut = turn, next = turn;
        cut.push_back(3);
        const auto [r3, p3] = run(cut, {4, 4, 4, 4, 4}, 2); // cancelled after two tokens
        next.insert(next.end(), {8, 0, 0, 3});              // the cut reply left out, another message after "b"
        const auto [r, pick] = run(next, {});
        check(r3.finish == mako::gen::Finish::cancelled && !pick.forked && r.reused_tokens == 13 && m.read == 4 &&
                  r.respelled == 10 && pool.size() == 1,
              std::format("a cut-off reply left out: the respelled context kept to before its prompt's last token "
                          "(reused {}, read {}, respelled {}, forked {})",
                          r.reused_tokens, m.read, r.respelled, pick.forked));
    }
    std::println("{} failures", failures);
    return failures ? 1 : 0;
}
