// Runs a calibration text through the model and writes its routing profile, which later runs read to decide which
// experts stay resident (GLM-5.3: step by step; DeepSeek-V4.1: through prefill, the same routing per token).
//   mako_calibrate <model.gguf> [--corpus FILE] [--out FILE] [--arena-gb N] [--engram FILE]
// Without --corpus it uses the built-in corpus; without --out it writes the model's default profile. --engram: the
// GGUF with DeepSeek-V4.1's Engram tables (default: the model's own shards).
#include "app/cli.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"
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
    std::string corpus_path, out, engram;
    double arena_gb = -1;
    const auto args =
        mako::app::Cli("mako_calibrate", "<model.gguf>", 1, 1)
            .option("--corpus", "FILE", "the calibration text (default: the built-in one)", corpus_path)
            .option("--out", "FILE", "the profile to write (default: the model's, read by later runs)", out)
            .option("--arena-gb", "N", "the expert cache's size in GB (default: what memory allows)", arena_gb)
            .option("--engram", "FILE", "DeepSeek-V4.1: the Engram tables of another GGUF (default: the model's own)",
                    engram)
            .parse(argc, argv);
    const char* model_path = args[0].c_str();
    std::string text = kCorpus;
    if (!corpus_path.empty()) {
        std::ifstream in(corpus_path, std::ios::binary);
        if (!in) {
            std::println(stderr, "cannot open {}", corpus_path);
            return 1;
        }
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    auto model = mako::gguf::ModelFiles::open(model_path);
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
        if (auto r = engram.empty() ? mako::dsv41::Engram::add_named(*model, model_path)
                                    : mako::dsv41::Engram::add_files(*model, engram);
            !r) {
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
