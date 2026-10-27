// OpenAI- and Anthropic-compatible HTTP server for a model (GLM-5.3 or DeepSeek-V4.1).
//   mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] [--arena-gb N] [--stage-gb N] [--no-spec]
//              [--ablate FILE] [--ablate-scale S] [--max-waiting N] [--learn] [--dump-requests DIR] [--engram FILE]
//              [--dspark DIR] [--miss-penalty F] [--dense-quant q4_k|q5_k|q6_k] [--prefetch K]
//              [--prefetch-ahead D] [--imatrix FILE] [--prefill-skip 0|1] [--api-key KEY | --api-key-file FILE]
//              [--reasoning-effort E] [--reasoning-budget N] [--mmproj FILE] [--image-max-tokens N]
//              [--park-gb N] [--conversations N] [--slice-ms N]
// --api-key KEY[,KEY...] / --api-key-file FILE (a key per line) / MAKO_API_KEY (KEY[,KEY...], when neither is given):
// requests need one of the keys ("Authorization: Bearer KEY" or "x-api-key: KEY"); a --host other than the loopback
// address requires one. Without keys, requests from web pages (an Origin or Host header that is not loopback) get 403.
// Sampling defaults come from the model's general.sampling.* metadata. SIGINT or SIGTERM stops it; with --learn the
// session's routing counts are then added to the model's profile (which experts start resident). --ablate removes
// per-layer directions (an ablation file, mtl::Ablation, with the model's sites: GLM-5.3 ffn_out and stream,
// DeepSeek-V4.1 attn_out and shexp_out); --ablate-scale multiplies their strength. --engram: the GGUF with
// DeepSeek-V4.1's Engram tables (default: the model's own shards). --stage-gb limits the ring that prompts'
// missing experts are read ahead into (0: off; default: two layers' worth, at most 3 GB). --dspark:
// DeepSeek-V4.1's drafter (dspark.json and dspark.bin) for speculative decoding; its 8 GB come out of the expert
// cache (measured: +13% on code, -3% on prose). DeepSeek-V4.1 trade-offs (accuracy for speed, docs/performance.md):
// --miss-penalty lowers the routing score of experts not in memory (fewer SSD reads; output then depends on what is
// cached); --dense-quant re-quantizes the dense Q8_0 weights at load (--imatrix: activation statistics from
// ds_run --imatrix-out weighting its rounding). --prefetch K (default 4; 0: off) reads the top K experts that layer
// l + D's router (--prefetch-ahead, default 2) predicts from layer l's input ahead (results unchanged).
// --prefill-skip 0 runs every prompt row through every layer (default 1: a long prompt's earlier rows skip the layers
// after the last KV source, which only the last few thousand rows need; results unchanged). --mmproj: the model's
// vision projector, for images in requests (docs/vision.md); --image-max-tokens scales larger images down to at
// most N prompt rows (default: the model's limit, 1024 for DeepSeek-V4.1, 8000 for GLM-5.3). Each conversation keeps
// its context between requests: up to --conversations (default 16) besides the one generating are parked in memory,
// within --park-gb (default: what the expert cache leaves when every expert is resident, 1-8 GB; taken from the
// expert cache beyond that), the least recently used given up first. Requests take turns: while others wait, the
// one generating gives the engine up every --slice-ms (default 1000; at a decode step, or between prompt pieces of the
// engine's prefill group) and queues again.
#include "app/presets.hpp"
#include "chat/template.hpp"
#include "core/log.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"
#include "dsv41/engine.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "mtl/context.hpp"
#include "server/server.hpp"
#include "text/tokenizer.hpp"

#include <cctype>
#include <cstdlib>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <print>
#include <string>
#include <thread>

#include <pthread.h>

int main(int argc, char** argv)
{
    mako::app::Args expanded = [&] {
        try {
            return mako::app::Args(argc, argv);
        } catch (const std::exception& e) {
            std::println(stderr, "{}", e.what());
            std::exit(2);
        }
    }();
    argc = expanded.argc();
    argv = expanded.argv();
    if (argc < 2) {
        std::println(stderr, "usage: mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] "
                             "[--arena-gb N] [--stage-gb N] [--no-spec] [--ablate FILE] [--ablate-scale S] [--max-waiting N] [--learn] "
                             "[--dump-requests DIR] [--engram FILE] [--dspark DIR] [--miss-penalty F] "
                             "[--dense-quant q4_k|q5_k|q6_k] [--prefetch K] [--prefetch-ahead D] [--imatrix FILE] [--prefill-skip 0|1] "
                             "[--api-key KEY | --api-key-file FILE] [--reasoning-effort none|low|medium|high|max|N] [--reasoning-budget N] [--mmproj FILE] [--image-max-tokens N] "
                             "[--park-gb N] [--conversations N] [--slice-ms N]");
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
    // Keys separated by `sep`, surrounding whitespace dropped; false when none.
    auto add_keys = [&](std::string_view list, char sep) {
        const size_t before = so.api_keys.size();
        for (size_t at = 0; at <= list.size();) {
            const size_t end = std::min(list.find(sep, at), list.size());
            std::string_view k = list.substr(at, end - at);
            while (!k.empty() && std::isspace(static_cast<unsigned char>(k.front())))
                k.remove_prefix(1);
            while (!k.empty() && std::isspace(static_cast<unsigned char>(k.back())))
                k.remove_suffix(1);
            if (!k.empty())
                so.api_keys.emplace_back(k);
            at = end + 1;
        }
        return so.api_keys.size() > before;
    };
    uint32_t max_ctx = 131072;
    double arena_gb = -1, stage_gb = -1, park_gb = -1;
    std::string ablate, engram, dspark, imatrix, mmproj_path;
    uint32_t image_max_tokens = 0;
    float ablate_scale = 1.0f, miss_penalty = 0.0f;
    uint32_t prefetch_k = 4, prefetch_ahead = 2;
    bool prefill_skip = true;
    std::optional<mako::gguf::Type> dense_quant;
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
            max_ctx = uint32_t(std::stoul(v));
        else if (a == "--model-id")
            so.model_id = v;
        else if (a == "--arena-gb")
            arena_gb = std::stod(v);
        else if (a == "--park-gb")
            park_gb = std::stod(v);
        else if (a == "--conversations")
            so.max_conversations = uint32_t(std::stoul(v));
        else if (a == "--slice-ms")
            so.slice_ms = uint32_t(std::stoul(v));
        else if (a == "--stage-gb")
            stage_gb = std::stod(v);
        else if (a == "--engram")
            engram = v;
        else if (a == "--dspark")
            dspark = v;
        else if (a == "--miss-penalty")
            miss_penalty = std::stof(v);
        else if (a == "--prefetch")
            prefetch_k = uint32_t(std::stoul(v));
        else if (a == "--prefill-skip")
            prefill_skip = std::string_view(v) != "0";
        else if (a == "--imatrix")
            imatrix = v;
        else if (a == "--prefetch-ahead")
            prefetch_ahead = uint32_t(std::stoul(v));
        else if (a == "--dense-quant") {
            const std::string q = v;
            if (q == "q4_k")
                dense_quant = mako::gguf::Type::Q4_K;
            else if (q == "q5_k")
                dense_quant = mako::gguf::Type::Q5_K;
            else if (q == "q6_k")
                dense_quant = mako::gguf::Type::Q6_K;
            else {
                std::println(stderr, "--dense-quant takes q4_k, q5_k or q6_k");
                return 2;
            }
        }
        else if (a == "--ablate")
            ablate = v;
        else if (a == "--ablate-scale")
            ablate_scale = std::stof(v);
        else if (a == "--max-waiting")
            so.max_waiting = uint32_t(std::stoul(v));
        else if (a == "--dump-requests")
            so.dump_dir = v;
        else if (a == "--reasoning-effort")
            so.reasoning_effort = mako::server::parse_effort(v);
        else if (a == "--reasoning-budget")
            so.reasoning_budget = uint32_t(std::stoul(v));
        else if (a == "--mmproj")
            mmproj_path = v;
        else if (a == "--image-max-tokens") {
            image_max_tokens = uint32_t(std::stoul(v));
            if (image_max_tokens < 16) {
                std::println(stderr, "--image-max-tokens takes 16 or more");
                return 2;
            }
        }
        else if (a == "--api-key")
            add_keys(v, ',');
        else if (a == "--api-key-file") {
            std::ifstream in(v);
            std::stringstream text;
            text << in.rdbuf();
            if (!in || !add_keys(text.str(), '\n')) {
                std::println(stderr, "cannot read an API key from {}", v);
                return 2;
            }
        }
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    if (const char* env = std::getenv("MAKO_API_KEY"); env && so.api_keys.empty())
        add_keys(env, ',');
    for (const auto& k : so.api_keys)
        if (k.size() < 16)
            mako::log::warn("an API key of {} characters is easy to guess (16 or more; e.g. openssl rand -hex 24)",
                            k.size());
    if (so.host != "127.0.0.1" && so.host != "localhost" && so.host != "::1" && so.api_keys.empty()) {
        std::println(stderr, "--host {} serves beyond this machine: give an --api-key or --api-key-file", so.host);
        return 2;
    }
    const auto t_start = std::chrono::steady_clock::now();
    mako::log::info("reading {}", argv[1]);
    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model) {
        mako::log::error("{}", model.error());
        return 1;
    }
    const bool deepseek = model->meta().get_string("general.architecture").value_or("") == "deepseek41";
    if (deepseek) {
        if (auto r = engram.empty() ? mako::dsv41::Engram::add_named(*model, argv[1])
                                    : mako::dsv41::Engram::add_files(*model, engram);
            !r) {
            mako::log::error("{}", r.error());
            return 1;
        }
    }
    const auto& meta = model->meta(); // (after add(), which may move the files)
    auto tok = mako::text::Tokenizer::load(meta);
    const auto source = meta.get_string("tokenizer.chat_template");
    if (!tok || !source) {
        mako::log::error("{}", !tok ? tok.error() : "the model has no chat template");
        return 1;
    }
    std::optional<mako::chat::Template> tmpl;
    try {
        tmpl.emplace(std::string(*source));
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    {
        auto t = mako::server::thinking_support(*tmpl);
        mako::log::info("chat template: reasoning {}; reasoning_effort by name{}{}",
                        t.can_disable ? "can be skipped" : "always on", t.medium ? " (medium included)" : "",
                        t.numeric ? " or 1..100" : "");
        t.default_effort = so.reasoning_effort;
        try {
            mako::server::choose_thinking(std::nullopt, t);
        } catch (const std::exception& e) {
            mako::log::error("--reasoning-effort: {}", e.what());
            return 2;
        }
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

    {
        uint64_t bytes = 0;
        for (const auto& f : model->files())
            bytes += f.size();
        mako::log::info("{} files, {:.1f} GB; loading the resident weights and setting up the expert cache",
                        model->files().size(), bytes / 1e9);
    }
    mako::mtl::Context ctx;
    // The vision projector goes first: its weights then count against what the expert cache may take.
    std::optional<mako::vision::Mmproj> mmproj;
    std::unique_ptr<mako::vision::Encoder> vision;
    if (!mmproj_path.empty()) {
        auto m = mako::vision::open_mmproj(mmproj_path);
        if (!m) {
            mako::log::error("{}", m.error());
            return 1;
        }
        if ((m->prep.kind == mako::vision::Projector::deepseek41v) != deepseek) {
            mako::log::error("{} is a projector for the other model", mmproj_path);
            return 1;
        }
        mmproj.emplace(std::move(*m));
        mmproj->prep.max_tokens = image_max_tokens;
        try {
            vision = std::make_unique<mako::vision::Encoder>(ctx, *mmproj);
        } catch (const std::exception& e) {
            mako::log::error("{}", e.what());
            return 1;
        }
        so.mmproj = &*mmproj;
        so.vision = vision.get();
        mako::log::info("vision projector {}: {} blocks, rows of {}", mmproj_path, mmproj->n_layer, mmproj->out_embd);
    }
    std::unique_ptr<mako::glm::Engine> glm_engine;
    std::unique_ptr<mako::dsv41::Engine> ds_engine;
    mako::gen::Model* model_if = nullptr;
    mako::core::EngineBase* base = nullptr;
    double expected_resident = 50; // below this share of the experts, the cache is smaller than this Mac allows
    try {
        uint32_t n_embd = 0, n_layer = 0;
        std::span<const std::string_view> sites;
        std::function<void(mako::mtl::Ablation)> set_ablation;
        if (deepseek) {
            auto cfg = mako::dsv41::Config::load(*model);
            if (!cfg) {
                mako::log::error("{}", cfg.error());
                return 1;
            }
            mako::dsv41::Options opts;
            opts.max_ctx = max_ctx;
            opts.arena_gb = arena_gb;
            opts.park_gb = park_gb;
            opts.stage_gb = stage_gb;
            if (so.speculative)
                opts.dspark = dspark;
            opts.miss_penalty = miss_penalty;
            opts.prefetch_k = prefetch_k;
            opts.prefill_skip = prefill_skip;
            opts.prefetch_ahead = prefetch_ahead;
            opts.dense_imatrix = imatrix;
            opts.dense_quant = dense_quant;
            ds_engine = std::make_unique<mako::dsv41::Engine>(*model, *cfg, opts, ctx);
            so.prefill_piece = opts.prefill_group;
            model_if = ds_engine.get();
            base = ds_engine.get();
            n_embd = cfg->n_embd, n_layer = cfg->n_layer;
            sites = mako::dsv41::Engine::kAblationSites;
            set_ablation = [&](mako::mtl::Ablation a) { ds_engine->set_ablation(std::move(a)); };
            expected_resident = 25;
            so.speculative = ds_engine->has_drafter();
        } else {
            auto cfg = mako::glm::Config::load(*model);
            if (!cfg) {
                mako::log::error("{}", cfg.error());
                return 1;
            }
            mako::glm::Options opts;
            opts.max_ctx = max_ctx;
            opts.arena_gb = arena_gb;
            opts.park_gb = park_gb;
            opts.stage_gb = stage_gb;
            opts.mtp = so.speculative;
            glm_engine = std::make_unique<mako::glm::Engine>(*model, *cfg, opts, ctx);
            so.prefill_piece = opts.prefill_group;
            model_if = glm_engine.get();
            base = glm_engine.get();
            n_embd = cfg->n_embd, n_layer = cfg->n_layer;
            sites = mako::glm::Engine::kAblationSites;
            set_ablation = [&](mako::mtl::Ablation a) { glm_engine->set_ablation(std::move(a)); };
        }
        if (mmproj && mmproj->out_embd != n_embd) {
            mako::log::error("the projector's rows are {} wide; the model's embeddings {}", mmproj->out_embd, n_embd);
            return 1;
        }
        if (!ablate.empty()) {
            auto a = mako::mtl::Ablation::load(ablate, ctx, n_embd, n_layer, sites);
            a.set_scale(ablate_scale);
            set_ablation(std::move(a));
            mako::log::info("ablating the directions in {} (scale {})", ablate, ablate_scale);
        }
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    uint64_t slots = 0, experts = 0;
    const auto& usage = base->arena().usage();
    for (uint32_t l = 0; l < usage.size(); ++l)
        if (const uint32_t n = base->arena().slots(l)) {
            slots += n;
            experts += usage[l].size();
        }
    const double resident = experts ? 100.0 * double(slots) / double(experts) : 100.0;
    so.park_bytes = uint64_t(base->arena_plan().park);
    mako::log::info("ready in {:.1f} s; context {} tokens; expert cache {:.1f} GB ({:.0f}% of the experts); "
                    "{:.1f} GB for parked conversations; temperature {}, top_p {}",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(), max_ctx,
                    base->arena().bytes() / 1e9, resident, so.park_bytes / 1e9, so.sampling.temperature,
                    so.sampling.top_p);
    if (resident < expected_resident)
        mako::log::warn("little memory was free for the expert cache (a process that just exited may still be "
                        "returning its memory); decoding will read experts from the SSD often");
    {
        mako::server::Server server(*model_if, *tok, *tmpl, so);
        std::jthread stopper([&] {
            int sig = 0;
            sigwait(&stop_signals, &sig);
            server.stop();
        });
        mako::log::info("serving {} on http://{}:{}/v1", so.model_id, so.host, so.port);
        const bool listened = server.run();
        if (!listened)
            mako::log::error("could not listen on {}:{}", so.host, so.port);
        ::kill(::getpid(), SIGTERM); // wakes the stopper if the server ended on its own
        if (!listened)
            return 1;
    } // the server's threads are done before the profile is written
    if (learn) {
        base->save_profile(base->default_profile(), true);
        mako::log::info("routing counts added to {}", base->default_profile().string());
    }
    return 0;
}
