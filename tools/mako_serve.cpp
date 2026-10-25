// OpenAI-compatible HTTP server for a model.
//   mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] [--arena-gb N] [--no-spec]
//              [--ablate FILE] [--ablate-scale S] [--max-waiting N] [--learn]
// Sampling defaults come from the model's general.sampling.* metadata. SIGINT or SIGTERM stops it; with --learn the
// session's routing counts are then added to the model's profile (which experts start resident). --ablate removes per-layer directions
// (raw float32, layers x k x n_embd) from the residual streams (Engine::set_ablation).
#include "chat/template.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "server/server.hpp"
#include "text/tokenizer.hpp"

#include <csignal>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <print>
#include <string>
#include <thread>

#include <pthread.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] "
                             "[--arena-gb N] [--no-spec] [--ablate FILE] [--ablate-scale S] [--max-waiting N] [--learn]");
        return 2;
    }
    // SIGINT and SIGTERM go to one thread that stops the server (every thread started later inherits the mask).
    sigset_t stop_signals;
    sigemptyset(&stop_signals);
    sigaddset(&stop_signals, SIGINT);
    sigaddset(&stop_signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
    bool learn = false;
    mako::server::ServerOptions so;
    mako::glm::Options opts;
    opts.max_ctx = 131072;
    std::string ablate;
    float ablate_scale = 1.0f;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-spec") {
            so.speculative = false;
            continue;
        }
        if (a == "--learn") {
            learn = true;
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
        else if (a == "--ablate")
            ablate = v;
        else if (a == "--ablate-scale")
            ablate_scale = std::stof(v);
        else if (a == "--max-waiting")
            so.max_waiting = uint32_t(std::stoul(v));
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
    if (!ablate.empty()) {
        try {
            engine.set_ablation(mako::glm::Engine::read_directions(ablate), ablate_scale);
        } catch (const std::exception& e) {
            std::println(stderr, "--ablate: {}", e.what());
            return 1;
        }
        std::println(stderr, "ablating the directions in {} (scale {})", ablate, ablate_scale);
    }
    std::println(stderr, "loaded in {:.1f} s; context {} tokens; temperature {}, top_p {}",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), opts.max_ctx,
                 so.sampling.temperature, so.sampling.top_p);
    {
        mako::server::Server server(engine, *tok, *tmpl, so);
        std::jthread stopper([&] {
            int sig = 0;
            sigwait(&stop_signals, &sig);
            server.stop();
        });
        std::println(stderr, "serving {} on http://{}:{}/v1", so.model_id, so.host, so.port);
        const bool listened = server.run();
        if (!listened)
            std::println(stderr, "could not listen on {}:{}", so.host, so.port);
        ::kill(::getpid(), SIGTERM); // wakes the stopper if the server ended on its own
        if (!listened)
            return 1;
    } // the server's threads are done before the profile is written
    if (learn) {
        engine.save_profile(engine.default_profile(), true);
        std::println(stderr, "routing counts added to {}", engine.default_profile().string());
    }
    return 0;
}
