// Conversations across an engine destroyed and built again (a model swap): a pool suspended, its engine gone, resumed
// on a new engine of the same model gives the conversation's next turn the tokens and logits it gives without the
// swap, bit for bit, reading only the new tokens; resumed with another identity it starts over.
//   reload_test <model.gguf>                        GLM-5.3
//   reload_test <model-00001.gguf> <engram.gguf>    DeepSeek-V4.1
#include "core/arena.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"
#include "dsv41/engram.hpp"
#include "gen/pool.hpp"
#include "gen/session.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"

#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

std::vector<int32_t> parse_ids(const std::string& s)
{
    std::vector<int32_t> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty())
            v.push_back(std::stoi(t));
    return v;
}

struct Turn {
    std::vector<int32_t> tokens;
    uint64_t hash = 1469598103934665603ull;
    uint32_t reused = 0;
};

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || !fs::exists(argv[1])) {
        std::println("model {} not found: skipped", argc < 2 ? "" : argv[1]);
        return 77;
    }
    const bool ds = argc > 2;
    if (ds && !fs::exists(argv[2])) {
        std::println("Engram tables {} not found: skipped", argv[2]);
        return 77;
    }
    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model || (ds && (!mako::dsv41::Engram::add_files(*model, argv[2]) ||
                          !mako::dsv41::Engram::add_named(*model, argv[1])))) {
        std::println(stderr, "loading {} failed", argv[1]);
        return 1;
    }
    std::optional<mako::glm::Config> glm_cfg;
    std::optional<mako::dsv41::Config> ds_cfg;
    if (ds) {
        auto c = mako::dsv41::Config::load(*model);
        if (!c) {
            std::println(stderr, "{}", c.error());
            return 1;
        }
        ds_cfg = *c;
    } else {
        auto c = mako::glm::Config::load(*model);
        if (!c) {
            std::println(stderr, "{}", c.error());
            return 1;
        }
        glm_cfg = *c;
    }
    std::ifstream in(fs::path(MAKO_TEST_DATA) / (ds ? "ds41_long_ids.txt" : "long_ids.txt"), std::ios::binary);
    const auto corpus = parse_ids({std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()});
    const std::vector<int32_t> first(corpus.begin(), corpus.begin() + 2100);

    mako::mtl::Context ctx;
    auto make = [&]() -> std::unique_ptr<mako::gen::Model> {
        if (ds) {
            mako::dsv41::Options opts;
            opts.max_ctx = 4096;
            return std::make_unique<mako::dsv41::Engine>(*model, *ds_cfg, opts, ctx);
        }
        mako::glm::Options opts;
        opts.prefill_chunk = 1024;
        opts.mtp = true;
        return std::make_unique<mako::glm::Engine>(*model, *glm_cfg, opts, ctx);
    };
    std::unique_ptr<mako::gen::Model> engine;
    auto params = [&](Turn& t) {
        mako::gen::GenerateParams gp;
        gp.sampling.temperature = 0;
        gp.max_tokens = 32;
        gp.speculative = true;
        gp.prefill_piece = 512;
        gp.forced = [&t, nv = engine->n_vocab()](const float* l) {
            const auto* p = reinterpret_cast<const unsigned char*>(l);
            for (size_t i = 0; i < size_t(nv) * 4; ++i)
                t.hash = (t.hash ^ p[i]) * 1099511628211ull;
            return -1;
        };
        return gp;
    };
    auto next = [](std::vector<int32_t> p, const std::vector<int32_t>& out) {
        p.insert(p.end(), out.begin(), out.end());
        p.insert(p.end(), {13, 785, 6722, 315, 279});
        return p;
    };
    auto run = [&](mako::gen::SessionPool& pool, const std::vector<int32_t>& prompt) {
        Turn t;
        mako::gen::GenerateResult r;
        mako::gen::SessionPool::Pick pick;
        pool.generate(prompt, params(t), {}, r, pick);
        t.tokens = r.tokens;
        t.reused = r.reused_tokens;
        return t;
    };

    // Both turns on one engine; then the first, a swap, the second.
    std::array<Turn, 2> want, got;
    const double start = mako::core::query_memory(0).footprint;
    engine = make();
    {
        mako::gen::SessionPool pool(*engine, 64ull << 30, 16, std::chrono::milliseconds(1000), "a");
        want[0] = run(pool, first);
        want[1] = run(pool, next(first, want[0].tokens));
    }
    engine->reset();
    mako::gen::SessionPool pool(*engine, 64ull << 30, 16, std::chrono::milliseconds(1000), "a");
    got[0] = run(pool, first);
    pool.suspend();
    const double loaded = mako::core::query_memory(0).footprint;
    engine.reset();
    const double unloaded = mako::core::query_memory(0).footprint;
    std::println("footprint: {:.1f} GB at the start, {:.1f} GB loaded, {:.1f} GB after the engine is gone ({:.2f} GB "
                 "of conversations parked)",
                 start / 1e9, loaded / 1e9, unloaded / 1e9, pool.parked_bytes() / 1e9);
    check(unloaded < start + double(pool.parked_bytes()) + 2e9,
          "destroying the engine gives its memory back (within 2 GB, beside the parked conversations)");
    engine = make();
    pool.resume(*engine, "a");
    const auto second = next(first, got[0].tokens);
    got[1] = run(pool, second);
    for (int i = 0; i < 2; ++i)
        check(got[i].tokens == want[i].tokens && got[i].hash == want[i].hash && got[i].reused == want[i].reused,
              std::format("turn {} across the swap: tokens {}, logits {}, {} reused (without the swap {})", i + 1,
                          got[i].tokens == want[i].tokens ? "the same" : "differ",
                          got[i].hash == want[i].hash ? "bit for bit" : "differ", got[i].reused, want[i].reused));

    // Another identity: the conversation is given up and the turn reads its whole prompt.
    pool.suspend();
    pool.resume(*engine, "b");
    const Turn other = run(pool, second);
    check(other.reused == 0 && other.tokens == want[1].tokens,
          std::format("another identity: {} reused, tokens {}", other.reused,
                      other.tokens == want[1].tokens ? "the same" : "differ"));
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
