// Runs a calibration text through the model and writes its routing profile, which later runs read to decide which
// experts stay resident (GLM-5.3: step by step; DeepSeek-V4.1: through prefill, the same routing per token).
//   mako_calibrate <model.gguf> [--corpus FILE] [--out FILE] [--arena-gb N] [--engram FILE]
// Without --corpus it uses the built-in corpus; without --out it writes the model's default profile. --engram: the
// GGUF with DeepSeek-V4.1's Engram tables (default: found beside the model, Engram::find_tables).
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "text/tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <print>
#include <string>

namespace {

constexpr char kCorpus[] = {
#embed "calibration.txt"
    , '\0'};

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) {
        std::println(stderr, "usage: mako_calibrate <model.gguf> [--corpus FILE] [--out FILE] [--arena-gb N] [--engram FILE]");
        return 2;
    }
    std::string corpus_path, out, engram;
    double arena_gb = -1;
    for (int i = 2; i < argc; i += 2) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            std::println(stderr, "option {} needs a value", a);
            return 2;
        }
        if (a == "--corpus")
            corpus_path = argv[i + 1];
        else if (a == "--out")
            out = argv[i + 1];
        else if (a == "--arena-gb")
            arena_gb = std::stod(argv[i + 1]);
        else if (a == "--engram")
            engram = argv[i + 1];
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    std::string text = kCorpus;
    if (!corpus_path.empty()) {
        std::ifstream in(corpus_path, std::ios::binary);
        if (!in) {
            std::println(stderr, "cannot open {}", corpus_path);
            return 1;
        }
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model) {
        std::println(stderr, "{}", model.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(model->meta());
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    const auto ids = tok->encode(text, true);
    mako::mtl::Context ctx;
    std::println("calibrating on {} tokens ({})", ids.size(), corpus_path.empty() ? "built-in corpus" : corpus_path);
    const auto t0 = std::chrono::steady_clock::now();
    if (model->meta().get_string("general.architecture").value_or("") == "deepseek41") {
        if (engram.empty())
            engram = mako::dsv41::Engram::find_tables(argv[1]);
        if (!engram.empty())
            if (auto r = model->add(engram); !r) {
                std::println(stderr, "{}", r.error());
                return 1;
            }
        auto cfg = mako::dsv41::Config::load(*model);
        if (!cfg) {
            std::println(stderr, "{}", cfg.error());
            return 1;
        }
        mako::dsv41::Options opts;
        opts.arena_gb = arena_gb;
        opts.max_ctx = std::max<uint32_t>(opts.max_ctx, uint32_t(ids.size()) + 1);
        mako::dsv41::Engine engine(*model, *cfg, opts, ctx);
        const auto path = out.empty() ? engine.default_profile() : std::filesystem::path(out);
        engine.prefill(ids);
        std::println("  {} tokens, {:.0f} s", ids.size(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        engine.save_profile(path, false);
        std::println("wrote {}", path.string());
        return 0;
    }
    auto cfg = mako::glm::Config::load(*model);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::glm::Options opts;
    opts.arena_gb = arena_gb;
    opts.max_ctx = std::max<uint32_t>(opts.max_ctx, uint32_t(ids.size()) + 1);
    mako::glm::Engine engine(*model, *cfg, opts, ctx);
    const auto path = out.empty() ? engine.default_profile() : std::filesystem::path(out);
    for (size_t i = 0; i < ids.size(); ++i) {
        engine.step(ids[i]);
        if ((i + 1) % 256 == 0 || i + 1 == ids.size()) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::println("  {}/{} tokens, {:.0f} s", i + 1, ids.size(), s);
        }
    }
    engine.save_profile(path, false);
    std::println("wrote {}", path.string());
    return 0;
}
