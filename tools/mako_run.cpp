// Runs token ids (or '@text') through the GLM engine and prints the top logits at each position and a greedy continuation.
//   mako_run <model.gguf> <id,id,...> [-n N] [--ctx N] [--top K] [--ref DIR] [--dump DIR] [--indexer-top-k N]
//            [--force DIR] [--force-route DIR] [--arena-gb N] [--expert-counts FILE] [--save-counts FILE] [--learn]
//            [--stats summary|steps|layers|kernels] [--stats-out FILE] [--prefill-chunk N] [--prefill-group N] [--logits-out DIR]
// The arena places experts by the model's routing profile (written by mako_calibrate) unless --expert-counts names
// another; --save-counts writes this run's routing counts, --learn adds them to the model's profile.
// --stats prints a Markdown telemetry summary at the end; --stats-out also writes its records as JSON lines.
// The prompt runs through prefill() in chunks of --prefill-chunk tokens (default 512; 0: token by token, as it
// always does with --ref or --force); --dump then writes every position's logits. --logits-out writes only the
// logits of every prompt position, in either mode.
// With --ref, every probed intermediate is compared against <DIR>/<name>-<layer>.<pos>.f32 and the
// logits against <DIR>/logits.<pos>.f32. --dump writes the probes and logits in the same layout.
// --force feeds each layer the reference's l_out of the layer before (teacher forcing), so --ref then
// measures each layer's own error; --force-route also takes the reference's expert selection and weights.
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "text/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> parse_ids(const std::string& s)
{
    std::vector<int32_t> ids;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty())
            ids.push_back(std::stoi(tok));
    return ids;
}

std::vector<int> top_k(const std::vector<float>& v, int k)
{
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

struct Diff {
    double rel = 0, cos = 0, max_abs = 0;
};

template <typename T>
std::vector<T> load(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in)
        return {};
    std::vector<T> v(size_t(in.tellg()) / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

// A reference longer than n is accepted when its tail is all -inf (padded scores).
std::optional<Diff> compare(const std::filesystem::path& file, const float* got, size_t n)
{
    const auto ref = load<float>(file);
    if (ref.size() < n || !std::all_of(ref.begin() + n, ref.end(), [](float v) { return std::isinf(v) && v < 0; }))
        return std::nullopt;
    double dd = 0, rr = 0, gg = 0, gr = 0;
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        const double e = double(got[i]) - ref[i];
        dd += e * e;
        rr += double(ref[i]) * ref[i];
        gg += double(got[i]) * got[i];
        gr += double(got[i]) * ref[i];
        d.max_abs = std::max(d.max_abs, std::abs(e));
    }
    d.rel = std::sqrt(dd / std::max(rr, 1e-30));
    d.cos = gr / std::sqrt(std::max(gg * rr, 1e-30));
    return d;
}

void dump(const std::filesystem::path& file, const float* data, size_t n)
{
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(data), std::streamsize(n * 4));
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3) {
        std::println(stderr, "usage: mako_run <model.gguf> <id,id,...> [-n N] [--ctx N] [--top K]");
        return 2;
    }
    int n_gen = 8, top = 5;
    std::string ref_dir, dump_dir, force_dir, route_dir, save_counts, stats_out, logits_dir;
    bool learn = false;
    mako::glm::Options opts;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--learn") {
            learn = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::println(stderr, "option {} needs a value", a);
            return 2;
        }
        const char* v = argv[++i];
        if (a == "-n")
            n_gen = std::stoi(v);
        else if (a == "--ctx")
            opts.max_ctx = uint32_t(std::stoul(v));
        else if (a == "--top")
            top = std::stoi(v);
        else if (a == "--indexer-top-k")
            opts.indexer_top_k = uint32_t(std::stoul(v));
        else if (a == "--ref")
            ref_dir = v;
        else if (a == "--expert-counts")
            opts.expert_counts = v;
        else if (a == "--save-counts")
            save_counts = v;
        else if (a == "--prefill-chunk")
            opts.prefill_chunk = uint32_t(std::stoul(v));
        else if (a == "--prefill-group")
            opts.prefill_group = uint32_t(std::stoul(v));
        else if (a == "--arena-gb")
            opts.arena_gb = std::stod(v);
        else if (a == "--dump")
            dump_dir = v;
        else if (a == "--logits-out")
            logits_dir = v;
        else if (a == "--force")
            force_dir = v;
        else if (a == "--force-route")
            route_dir = v;
        else if (a == "--stats") {
            const auto t = mako::glm::parse_telemetry(v);
            if (!t) {
                std::println(stderr, "--stats takes summary, steps, layers or kernels");
                return 2;
            }
            opts.telemetry = *t;
        } else if (a == "--stats-out")
            stats_out = v;
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    if (!dump_dir.empty())
        std::filesystem::create_directories(dump_dir);
    if (!logits_dir.empty())
        std::filesystem::create_directories(logits_dir);
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
    const std::string input = argv[2];
    auto ids = input.starts_with('@') ? tok->encode(std::string_view(input).substr(1), true) : parse_ids(input);
    if (ids.empty()) {
        std::println(stderr, "no input tokens");
        return 2;
    }
    auto cfg = mako::glm::Config::load(*model);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::mtl::Context ctx;
    auto t0 = std::chrono::steady_clock::now();
    mako::glm::Engine engine(*model, *cfg, opts, ctx);
    std::println("loaded in {:.1f} s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (!stats_out.empty() && opts.telemetry == mako::glm::Telemetry::off)
        opts.telemetry = mako::glm::Telemetry::summary;
    std::ofstream stats_file;
    if (!stats_out.empty())
        stats_file.open(stats_out, std::ios::trunc);
    mako::glm::Report telemetry(opts.telemetry, stats_out.empty() ? nullptr : &stats_file);
    telemetry.machine(mako::glm::query_machine(ctx.info()));
    telemetry.load(engine.load_stats());
    {
        const auto& p = engine.arena_plan();
        const auto& a = engine.arena();
        std::println("memory: RAM {:.1f} GB, working set {:.1f} GB, footprint {:.1f} GB, file-cache floor {:.1f} GB, "
                     "available {:.1f} GB, lendable to prefill {:.1f} GB",
                     p.mem.phys / 1e9, p.mem.working_set / 1e9, p.mem.footprint / 1e9, p.mem.filecache_min / 1e9,
                     p.mem.available / 1e9, engine.arena().lendable_bytes() / 1e9);
        uint32_t lo = UINT32_MAX, hi = 0, moe = 0;
        for (uint32_t l = 0; l < cfg->n_layer; ++l)
            if (a.slots(l)) {
                lo = std::min(lo, a.slots(l));
                hi = std::max(hi, a.slots(l));
                ++moe;
            }
        if (engine.loaded_profile().empty())
            std::println("no routing profile for this model; mako_calibrate makes one ({})",
                         engine.default_profile().string());
        else
            std::println("routing profile: {}", engine.loaded_profile().string());
        std::println("expert arena: budget {:.1f} GB, {:.1f} GB, {}-{} slots in each of {} layers, GPU warm-up {:.2f} s, "
                     "prefilled in {:.1f} s",
                     p.budget / 1e9, a.bytes() / 1e9, lo, hi, moe, p.warm_ms / 1e3, a.stats().prefill_ms / 1e3);
    }

    if (!force_dir.empty())
        engine.set_stream_override([&](int layer, float* streams, size_t n) {
            if (layer == 0)
                return;
            const auto file = std::filesystem::path(force_dir) /
                              std::format("l_out-{}.{}.f32", layer - 1, engine.position());
            const auto v = load<float>(file);
            if (v.size() != n)
                throw std::runtime_error("--force: missing or wrong-sized " + file.string());
            std::ranges::copy(v, streams);
        });
    if (!route_dir.empty())
        engine.set_route_override([&](int layer, int* sel, float* w, size_t k) {
            const auto dir = std::filesystem::path(route_dir);
            const auto ids = load<int32_t>(dir / std::format("ffn_moe_topk-{}.{}.i32", layer, engine.position()));
            const auto ws =
                load<float>(dir / std::format("ffn_moe_weights_scaled-{}.{}.f32", layer, engine.position()));
            if (ids.size() != k || ws.size() != k)
                throw std::runtime_error(std::format("--force-route: missing routing for layer {}", layer));
            std::ranges::copy(ids, sel);
            std::ranges::copy(ws, w);
        });

    double worst_rel = 0;
    auto report = [&](const std::string& name, const float* data, size_t n) {
        const auto base = name + "." + std::to_string(engine.position()) + ".f32";
        if (!dump_dir.empty())
            dump(std::filesystem::path(dump_dir) / base, data, n);
        if (ref_dir.empty())
            return;
        if (name.starts_with("moe_sel-")) {
            const auto file = std::format("ffn_moe_topk-{}.{}.i32", name.substr(8), engine.position());
            auto ref = load<int32_t>(std::filesystem::path(ref_dir) / file);
            std::vector<int32_t> got(data, data + n);
            if (ref.size() != n) {
                std::println("  {:16} (no reference)", name);
                return;
            }
            std::ranges::sort(ref);
            std::ranges::sort(got);
            if (ref == got)
                std::println("  {:16} same", name);
            else
                std::println("  {:16} DIFFERS mako {} ref {}", name, got, ref);
            return;
        }
        if (auto d = compare(std::filesystem::path(ref_dir) / base, data, n)) {
            worst_rel = std::max(worst_rel, d->rel);
            std::println("  {:16} rel {:9.2e}  cos {:.6f}  max {:9.2e}", name, d->rel, d->cos, d->max_abs);
        } else {
            std::println("  {:16} (no reference)", name);
        }
    };
    if (!ref_dir.empty() || !dump_dir.empty())
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            report(layer < 0 ? name : std::format("{}-{}", name, layer), data, n);
        });

    auto show = [&](const std::vector<float>& logits) {
        std::string line;
        for (int t : top_k(logits, top))
            line += std::format(" {}:{:.3f}", t, logits[t]);
        return line;
    };

    std::vector<float> logits;
    const bool batched = opts.prefill_chunk > 0 && ids.size() > 1 && ref_dir.empty() &&
                         force_dir.empty() && route_dir.empty();
    if (batched) {
        if (!dump_dir.empty() || !logits_dir.empty())
            engine.set_logits_sink([&](uint32_t pos, const float* lg) {
                dump(std::filesystem::path(logits_dir.empty() ? dump_dir : logits_dir) /
                         std::format("logits.{}.f32", pos),
                     lg, cfg->n_vocab);
            });
        auto t = std::chrono::steady_clock::now();
        logits = engine.prefill(ids);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        telemetry.prefill(engine.last_prefill(), uint32_t(ids.size()));
        std::println("prefill {} tokens in {:.0f} ms ({:.1f} tok/s), chunks of {}  top{}", ids.size(), ms,
                     ids.size() / ms * 1e3, opts.prefill_chunk, show(logits));
    }
    for (size_t i = 0; i < (batched ? 0 : ids.size()); ++i) {
        auto t = std::chrono::steady_clock::now();
        logits = engine.step(ids[i]);
        telemetry.step(engine.last_step());
        if (!dump_dir.empty())
            dump(std::filesystem::path(dump_dir) / std::format("logits.{}.f32", i), logits.data(), logits.size());
        if (!logits_dir.empty())
            dump(std::filesystem::path(logits_dir) / std::format("logits.{}.f32", i), logits.data(), logits.size());
        if (!ref_dir.empty()) {
            const auto file = std::filesystem::path(ref_dir) / std::format("logits.{}.f32", i);
            if (auto d = compare(file, logits.data(), logits.size()))
                std::println("  logits           rel {:9.2e}  cos {:.6f}  max {:9.2e}", d->rel, d->cos, d->max_abs);
        }
        std::println("pos {:4} in {:7} {:7.0f} ms  top{}", i, ids[i],
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(),
                     show(logits));
    }
    std::string gen;
    std::vector<int32_t> gen_ids;
    for (int g = 0; g < n_gen; ++g) {
        const int next = top_k(logits, 1)[0];
        gen += std::format("{},", next);
        gen_ids.push_back(next);
        auto t = std::chrono::steady_clock::now();
        logits = engine.step(next);
        telemetry.step(engine.last_step());
        std::println("gen {:4} in {:7} {:7.0f} ms  top{}", g, next,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(),
                     show(logits));
    }
    if (!ref_dir.empty())
        std::println("worst probe rel error {:.2e}", worst_rel);
    const auto& st = engine.stats();
    std::println("greedy: {}", gen);
    std::println("text: {}{}", tok->decode(ids), tok->decode(gen_ids));
    if (opts.telemetry != mako::glm::Telemetry::off)
        std::print("\n{}", telemetry.markdown());
    if (!save_counts.empty())
        engine.save_profile(save_counts, false);
    if (learn) {
        engine.save_profile(engine.default_profile(), true);
        std::println("routing counts added to {}", engine.default_profile().string());
    }
    const auto& as = engine.arena().stats();
    std::println("expert hits {} misses {} ({:.1f}% hit), reads {:.0f} ms ({:.2f} GB), gpu wait {:.0f} ms", as.hits,
                 as.misses, 100.0 * as.hits / std::max<uint64_t>(1, as.hits + as.misses), st.expert_read_ms,
                 st.expert_bytes / 1e9, st.gpu_wait_ms);
    const double steps = double(ids.size() + gen_ids.size());
    std::println("per step: {:.1f} ms in step(), {:.1f} ms encoding, {:.1f} ms commit to done, {:.1f} ms GPU start to end",
                 st.step_ms / steps, st.encode_ms / steps, st.gpu_wait_ms / steps, st.gpu_exec_ms / steps);
    return 0;
}
