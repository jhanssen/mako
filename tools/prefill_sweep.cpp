// Prefill time of reads of several sizes on top of a base context, as an agent turn reading a tool result.
//   prefill_sweep <model.gguf> [--base N] [--sizes a,b,...] [--repeats N] [--layers] [--steps N] [--prefill-group N]
// The text is this repository's sources, tokenized. Each size runs `repeats` times (default 2), each from a checkpoint
// of the base context; the experts left resident by the runs before stay. --layers adds per-layer GPU timestamps
// (GPU time by section; extra encoder boundaries slow it somewhat). --steps N also times sizes up to N as decode
// steps, one token at a time.
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"
#include "text/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string sources()
{
    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::recursive_directory_iterator(MAKO_SOURCE_DIR "/src"))
        if (e.is_regular_file() && (e.path().extension() == ".cpp" || e.path().extension() == ".hpp" ||
                                    e.path().extension() == ".metal"))
            files.push_back(e.path());
    std::ranges::sort(files);
    std::string all;
    for (const auto& f : files) {
        std::ifstream in(f);
        std::stringstream ss;
        ss << in.rdbuf();
        all += "// " + f.filename().string() + "\n" + ss.str() + "\n";
    }
    return all;
}

double ms_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: prefill_sweep <model.gguf> [--base N] [--sizes a,b,...] [--repeats N] [--layers] "
                             "[--steps N] [--prefill-group N]");
        return 2;
    }
    uint32_t base = 15000, repeats = 2, step_max = 0;
    std::vector<uint32_t> sizes{16, 64, 256, 512, 1024, 2048, 4096};
    mako::glm::Options opts;
    opts.telemetry = mako::glm::Telemetry::summary;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers") {
            opts.telemetry = mako::glm::Telemetry::layers;
            continue;
        }
        if (i + 1 >= argc) {
            std::println(stderr, "option {} needs a value", a);
            return 2;
        }
        const std::string v = argv[++i];
        if (a == "--base")
            base = uint32_t(std::stoul(v));
        else if (a == "--repeats")
            repeats = uint32_t(std::stoul(v));
        else if (a == "--steps")
            step_max = uint32_t(std::stoul(v));
        else if (a == "--prefill-group")
            opts.prefill_group = uint32_t(std::stoul(v));
        else if (a == "--sizes") {
            sizes.clear();
            std::stringstream ss(v);
            for (std::string x; std::getline(ss, x, ',');)
                sizes.push_back(uint32_t(std::stoul(x)));
        } else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model) {
        std::println(stderr, "{}", model.error());
        return 1;
    }
    auto tok = mako::text::Tokenizer::load(model->meta());
    auto cfg = mako::glm::Config::load(*model);
    if (!tok || !cfg) {
        std::println(stderr, "{}", !tok ? tok.error() : cfg.error());
        return 1;
    }
    const uint32_t most = *std::ranges::max_element(sizes);
    std::vector<int32_t> ids;
    const auto text = tok->encode(sources(), false);
    while (ids.size() < base + most)
        ids.insert(ids.end(), text.begin(), text.end());
    opts.max_ctx = base + most + 16;
    mako::mtl::Context ctx;
    mako::glm::Engine engine(*model, *cfg, opts, ctx);

    auto t = std::chrono::steady_clock::now();
    engine.prefill(std::span(ids).first(base));
    std::println("base context: {} tokens in {:.0f} ms", base, ms_since(t));
    const auto cp = engine.checkpoint();
    const std::span<const int32_t> more = std::span(ids).subspan(base);

    std::println("| tokens | run | wall ms | tok/s | GPU ms | resident ms | misses | GB read | read ms |{}",
                 opts.telemetry == mako::glm::Telemetry::layers ? " KDA | MLA | FFN | experts | rest |" : "");
    std::println("|---|---|---|---|---|---|---|---|---|{}",
                 opts.telemetry == mako::glm::Telemetry::layers ? "---|---|---|---|---|" : "");
    for (uint32_t n : sizes)
        for (uint32_t r = 0; r < repeats; ++r) {
            engine.restore(cp);
            t = std::chrono::steady_clock::now();
            engine.prefill(more.first(n));
            const double wall = ms_since(t);
            const auto& p = engine.last_prefill();
            std::string sections;
            if (!p.layers.empty()) {
                double kda = 0, mla = 0, ffn = 0, ex = 0, rest = 0;
                for (const auto& l : p.layers) {
                    (l.mla ? mla : kda) += l.attn_ms;
                    ffn += l.ffn_ms;
                    ex += l.hit_ms;
                    rest += l.miss_ms;
                }
                sections = std::format(" {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |", kda, mla, ffn, ex, rest);
            }
            std::println("| {} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {} | {:.2f} | {:.0f} |{}", n, r + 1, wall,
                         n / wall * 1e3, p.gpu_ms, p.ensure_ms, p.misses, p.read_bytes / 1e9, p.read_ms, sections);
        }
    for (uint32_t n : sizes) {
        if (n > step_max)
            continue;
        engine.restore(cp);
        t = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n; ++i)
            engine.step(more[i]);
        const double wall = ms_since(t);
        std::println("decode steps: {} tokens in {:.0f} ms ({:.0f} tok/s)", n, wall, n / wall * 1e3);
    }
    return 0;
}
