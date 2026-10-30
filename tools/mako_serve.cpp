// OpenAI- and Anthropic-compatible HTTP server for a model (GLM-5.3 or DeepSeek-V4.1).
//   mako_serve <model.gguf> [--host H] [--port N] [--ctx N] [--model-id NAME] [--arena-gb N] [--stage-gb N] [--no-spec]
//              [--ablate FILE] [--ablate-scale S] [--max-waiting N] [--learn] [--dump-requests DIR] [--engram FILE]
//              [--dspark DIR] [--miss-penalty F] [--dense-quant q4_k|q5_k|q6_k] [--prefetch K]
//              [--prefetch-ahead D] [--imatrix FILE] [--prefill-skip 0|1] [--api-key KEY | --api-key-file FILE]
//              [--reasoning-effort E] [--reasoning-budget N] [--mmproj FILE] [--image-max-tokens N]
//              [--park-gb N] [--conversations N] [--slice-ms N] [--loop-guard N] [--image-model DIR]
//              [--image-lora FILE[:SCALE] ...] [--image-steps N] [--image-terminal F] [--image-text-encoder DIR]
//              [--image-transformer PATH] [--image-guidance F]
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
// engine's prefill group) and queues again. --image-model: Qwen-Image 2.1's or Krea 2's directory (its files as
// published), for /v1/images/generations: an image request waits for the chat requests in progress, then the LLM is
// unloaded (its conversations kept in memory) and the image model loaded; the next chat request loads the LLM again.
// --image-lora merges LoRAs into the image model (repeatable, FILE:SCALE); --image-steps and --image-terminal set the
// schedule (a 4-step distilled LoRA: --image-steps 4 --image-terminal 0); --image-text-encoder loads the text encoder
// from another directory (the same model's weights, changed: e.g. refusal-ablated), --image-transformer the transformer
// from another file or directory (e.g. a checkpoint with LoRAs merged); --image-guidance the guidance scale of requests
// with a negative prompt that name none.
#include "app/cli.hpp"
#include "app/dense_quant.hpp"
#include "chat/template.hpp"
#include "core/log.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"
#include "dsv41/engine.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "image/engine.hpp"
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
    std::string ablate, engram, dspark, imatrix, mmproj_path, image_model;
    std::vector<std::pair<std::string, float>> image_loras;
    std::string image_text_encoder, image_transformer;
    uint32_t image_max_tokens = 0;
    float ablate_scale = 1.0f, miss_penalty = 0.0f;
    uint32_t prefetch_k = 4, prefetch_ahead = 2;
    bool prefill_skip = true;
    std::optional<mako::gguf::Type> dense_quant;
    const auto args =
        mako::app::Cli("mako_serve", "<model.gguf>", 1, 1)
            .option("--host", "H", "address to listen on (default 127.0.0.1; others need an API key)", so.host)
            .option("--port", "N", "port to listen on", so.port)
            .option("--ctx", "N", "the context length in tokens", max_ctx)
            .option("--model-id", "NAME", "the model name the API reports", so.model_id)
            .option("--api-key", "KEY[,KEY...]", "keys requests must carry (Bearer or x-api-key)",
                    [&](const std::string& v) { add_keys(v, ','); })
            .secret()
            .option("--api-key-file", "FILE", "keys requests must carry, one a line",
                    [&](const std::string& v) {
                        std::ifstream in(v);
                        std::stringstream text;
                        text << in.rdbuf();
                        if (!in || !add_keys(text.str(), '\n'))
                            throw std::invalid_argument("names no readable key file");
                    })
            .shared_env("--api-key")
            .option("--arena-gb", "N", "the expert cache's size in GB (default: what memory allows)", arena_gb)
            .option("--stage-gb", "N", "the ring prompts' missing experts are read ahead into (0: off)", stage_gb)
            .option("--park-gb", "N", "memory for parked conversations (default: what the expert cache leaves)",
                    park_gb)
            .option("--conversations", "N", "conversations kept parked besides the one generating (default 16)",
                    so.max_conversations)
            .option("--slice-ms", "N", "how long a request generates before waiting ones get a turn (default 1000)",
                    so.slice_ms)
            .option("--max-waiting", "N", "requests that may wait before new ones are refused", so.max_waiting)
            .flag("--spec", "speculative decoding (GLM-5.3's MTP block, DeepSeek-V4.1 with --dspark; default on)",
                  so.speculative)
            .option("--dspark", "DIR", "DeepSeek-V4.1's drafter (dspark.json and dspark.bin)", dspark)
            .option("--engram", "FILE", "DeepSeek-V4.1: the Engram tables of another GGUF (default: the model's own)",
                    engram)
            .option("--miss-penalty", "F", "DeepSeek-V4.1: lower routing scores of experts not in memory (0.05-0.2)",
                    miss_penalty)
            .option("--dense-quant", "q4_k|q5_k|q6_k", "re-quantize the dense Q8_0 weights at load",
                    [&](const std::string& v) { dense_quant = mako::app::parse_dense_quant(v); })
            .option("--imatrix", "FILE", "activation statistics (ds_run --imatrix-out) weighting --dense-quant",
                    imatrix)
            .option("--prefetch", "K", "read ahead the K experts a later layer's router predicts (default 4; 0: off)",
                    prefetch_k)
            .option("--prefetch-ahead", "D", "how many layers ahead --prefetch predicts (default 2)", prefetch_ahead)
            .option("--prefill-skip", "0|1", "a long prompt's earlier rows skip the layers only later rows need",
                    prefill_skip)
            .option("--reasoning-effort", "none|low|medium|high|max|N", "the default reasoning effort",
                    [&](const std::string& v) { so.reasoning_effort = mako::server::parse_effort(v); })
            .option("--reasoning-budget", "N", "the default reasoning budget in tokens", so.reasoning_budget)
            .option("--loop-guard", "N",
                    "end repetition loops of N tokens or more in reasoning or answers (default 256; 0: off)",
                    so.loop_guard)
            .option("--mmproj", "FILE", "the model's vision projector, for images in requests", mmproj_path)
            .option("--image-model", "DIR", "Qwen-Image 2.1's or Krea 2's files, for /v1/images/generations (swapped in)",
                    image_model)
            .option("--image-lora", "FILE[:SCALE]", "merge a LoRA into the image model (repeatable)",
                    [&](const std::string& v) { image_loras.push_back(mako::image::parse_lora(v)); })
            .option("--image-text-encoder", "DIR", "another directory of the image model's text encoder weights",
                    image_text_encoder)
            .option("--image-transformer", "PATH", "another file or directory of the image model's transformer weights",
                    image_transformer)
            .option("--image-steps", "N", "denoising steps of an image (default: the model's; 4 with a 4-step LoRA)",
                    so.image_steps)
            .option("--image-guidance", "F", "guidance scale for negative prompts (default: the model's)",
                    [&](const std::string& v) { so.image_guidance = mako::app::Cli::parse<float>(v); })
            .option("--image-terminal", "F", "the schedule's last noise level (0: none, as few-step LoRAs want)",
                    [&](const std::string& v) { so.image_terminal = mako::app::Cli::parse<float>(v); })
            .option("--image-max-tokens", "N", "scale larger images down to at most N prompt rows (16 or more)",
                    [&](const std::string& v) {
                        image_max_tokens = mako::app::Cli::parse<uint32_t>(v);
                        if (image_max_tokens < 16)
                            throw std::invalid_argument("takes 16 or more");
                    })
            .option("--ablate", "FILE", "remove per-layer directions (an ablation file)", ablate)
            .option("--ablate-scale", "S", "multiply the ablation's strength", ablate_scale)
            .flag("--learn", "add the session's routing counts to the model's profile on exit", learn)
            .option("--dump-requests", "DIR", "write each request body to DIR", so.dump_dir)
            .parse(argc, argv);
    const char* model_path = args[0].c_str();
    // SIGINT and SIGTERM go to one thread that stops the server (every thread started later inherits the mask).
    sigset_t stop_signals;
    sigemptyset(&stop_signals);
    sigaddset(&stop_signals, SIGINT);
    sigaddset(&stop_signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
    for (const auto& k : so.api_keys)
        if (k.size() < 16)
            mako::log::warn("an API key of {} characters is easy to guess (16 or more; e.g. openssl rand -hex 24)",
                            k.size());
    if (so.host != "127.0.0.1" && so.host != "localhost" && so.host != "::1" && so.api_keys.empty()) {
        std::println(stderr, "--host {} serves beyond this machine: give an --api-key or --api-key-file", so.host);
        return 2;
    }
    const auto t_start = std::chrono::steady_clock::now();
    mako::log::info("reading {}", model_path);
    auto model = mako::gguf::ModelFiles::open(model_path);
    if (!model) {
        mako::log::error("{}", model.error());
        return 1;
    }
    const bool deepseek = model->meta().get_string("general.architecture").value_or("") == "deepseek41";
    if (deepseek) {
        if (auto r = engram.empty() ? mako::dsv41::Engram::add_named(*model, model_path)
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
            std::filesystem::path(model_path).stem().string()));

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
    // Below this share of the experts resident, the cache is smaller than this Mac allows.
    const double expected_resident = deepseek ? 25 : 50;
    // The LLM: loaded now, and again after an image request had it unloaded.
    auto load_llm = [&]() -> mako::server::Models::Llm {
        mako::server::Models::Llm l;
        uint32_t n_embd = 0, n_layer = 0;
        std::span<const std::string_view> sites;
        std::function<void(mako::mtl::Ablation)> set_ablation;
        if (deepseek) {
            auto cfg = mako::dsv41::Config::load(*model);
            if (!cfg)
                throw std::runtime_error(cfg.error());
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
            auto e = std::make_unique<mako::dsv41::Engine>(*model, *cfg, opts, ctx);
            so.prefill_piece = opts.prefill_group;
            so.speculative = e->has_drafter();
            n_embd = cfg->n_embd, n_layer = cfg->n_layer;
            sites = mako::dsv41::Engine::kAblationSites;
            set_ablation = [p = e.get()](mako::mtl::Ablation a) { p->set_ablation(std::move(a)); };
            l.base = e.get();
            l.model = std::move(e);
        } else {
            auto cfg = mako::glm::Config::load(*model);
            if (!cfg)
                throw std::runtime_error(cfg.error());
            mako::glm::Options opts;
            opts.max_ctx = max_ctx;
            opts.arena_gb = arena_gb;
            opts.park_gb = park_gb;
            opts.stage_gb = stage_gb;
            opts.mtp = so.speculative;
            auto e = std::make_unique<mako::glm::Engine>(*model, *cfg, opts, ctx);
            so.prefill_piece = opts.prefill_group;
            n_embd = cfg->n_embd, n_layer = cfg->n_layer;
            sites = mako::glm::Engine::kAblationSites;
            set_ablation = [p = e.get()](mako::mtl::Ablation a) { p->set_ablation(std::move(a)); };
            l.base = e.get();
            l.model = std::move(e);
        }
        if (mmproj && mmproj->out_embd != n_embd)
            throw std::runtime_error(std::format("the projector's rows are {} wide; the model's embeddings {}",
                                                 mmproj->out_embd, n_embd));
        if (!ablate.empty()) {
            auto a = mako::mtl::Ablation::load(ablate, ctx, n_embd, n_layer, sites);
            a.set_scale(ablate_scale);
            set_ablation(std::move(a));
            mako::log::info("ablating the directions in {} (scale {})", ablate, ablate_scale);
        }
        return l;
    };
    mako::server::Models::LoadImage load_image;
    if (!image_model.empty())
        load_image = [&] {
            auto e = mako::image::Engine::open(image_model, ctx, {image_text_encoder, image_transformer});
            for (const auto& [path, scale] : image_loras)
                e->add_lora(path, scale);
            return e;
        };
    std::optional<mako::server::Models> models;
    try {
        models.emplace(load_llm, load_image);
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    mako::core::EngineBase* base = nullptr;
    models->with_llm([&](mako::core::EngineBase& b) { base = &b; });
    uint64_t slots = 0, experts = 0;
    const auto& usage = base->arena().usage();
    for (uint32_t l = 0; l < usage.size(); ++l)
        if (const uint32_t n = base->arena().slots(l)) {
            slots += n;
            experts += usage[l].size();
        }
    const double resident = experts ? 100.0 * double(slots) / double(experts) : 100.0;
    so.park_bytes = uint64_t(base->arena_plan().park);
    so.cache_gb = base->arena().bytes() / 1e9;
    so.resident = resident / 100.0;
    so.cache_counters = [&models] {
        mako::server::CacheCounters c;
        models->with_llm([&](mako::core::EngineBase& b) { c = {b.arena().stats().hits, b.arena().stats().misses}; });
        return c;
    };
    mako::log::info("ready in {:.1f} s; context {} tokens; expert cache {:.1f} GB ({:.0f}% of the experts); "
                    "{:.1f} GB for parked conversations; temperature {}, top_p {}",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(), max_ctx,
                    base->arena().bytes() / 1e9, resident, so.park_bytes / 1e9, so.sampling.temperature,
                    so.sampling.top_p);
    if (resident < expected_resident)
        mako::log::warn("little memory was free for the expert cache (a process that just exited may still be "
                        "returning its memory); decoding will read experts from the SSD often");
    {
        if (models->has_image())
            mako::log::info("image model {}: loaded for image requests, in place of the LLM", image_model);
        mako::server::Server server(*models, *tok, *tmpl, so);
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
    if (learn && !models->with_llm([](mako::core::EngineBase& b) {
            b.save_profile(b.default_profile(), true);
            mako::log::info("routing counts added to {}", b.default_profile().string());
        }))
        mako::log::warn("--learn: the LLM was not loaded at the end (the image model was); no routing counts saved");
    return 0;
}
