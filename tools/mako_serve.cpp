// OpenAI-compatible HTTP server for a model.
//   mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] [--arena-gb N] [--no-spec]
// Sampling defaults come from the model's general.sampling.* metadata.
#include "chat/template.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "server/server.hpp"
#include "text/tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <print>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] "
                             "[--arena-gb N] [--no-spec]");
        return 2;
    }
    mako::server::ServerOptions so;
    mako::glm::Options opts;
    opts.max_ctx = 131072;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-spec") {
            so.speculative = false;
            continue;
        }
        if (i + 1 >= argc) {
            std::println(stderr, "option {} needs a value", a);
            return 2;
        }
        const char* v = argv[++i];
        if (a == "--host")
            so.host = v;
        else if (a == "--port")
            so.port = std::stoi(v);
        else if (a == "--ctx")
            opts.max_ctx = uint32_t(std::stoul(v));
        else if (a == "--model-id")
            so.model_id = v;
        else if (a == "--arena-gb")
            opts.arena_gb = std::stod(v);
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model) {
        std::println(stderr, "{}", model.error());
        return 1;
    }
    const auto& meta = model->meta();
    auto tok = mako::text::Tokenizer::load(meta);
    auto cfg = mako::glm::Config::load(*model);
    const auto source = meta.get_string("tokenizer.chat_template");
    if (!tok || !cfg || !source) {
        std::println(stderr, "{}", !tok ? tok.error() : !cfg ? cfg.error() : "the model has no chat template");
        return 1;
    }
    std::optional<mako::chat::Template> tmpl;
    try {
        tmpl.emplace(std::string(*source));
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
    if (auto t = meta.get_float("general.sampling.temp"))
        so.sampling.temperature = float(*t);
    if (auto p = meta.get_float("general.sampling.top_p"))
        so.sampling.top_p = float(*p);
    if (auto k = meta.get_int("general.sampling.top_k"))
        so.sampling.top_k = uint32_t(*k);
    if (so.model_id.empty())
        so.model_id = std::string(meta.get_string("general.basename").value_or(
            std::filesystem::path(argv[1]).stem().string()));

    mako::mtl::Context ctx;
    const auto t0 = std::chrono::steady_clock::now();
    opts.mtp = so.speculative;
    mako::glm::Engine engine(*model, *cfg, opts, ctx);
    std::println(stderr, "loaded in {:.1f} s; context {} tokens; temperature {}, top_p {}",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), opts.max_ctx,
                 so.sampling.temperature, so.sampling.top_p);
    mako::server::Server server(engine, *tok, *tmpl, so);
    std::println(stderr, "serving {} on http://{}:{}/v1", so.model_id, so.host, so.port);
    if (!server.run()) {
        std::println(stderr, "could not listen on {}:{}", so.host, so.port);
        return 1;
    }
    return 0;
}
