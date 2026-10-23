// Runs token ids through the GLM engine and prints the top logits at each position and a greedy continuation.
//   mako_run <model.gguf> <id,id,...> [-n N] [--ctx N] [--top K] [--ref DIR] [--indexer-top-k N]
// With --ref, every probed intermediate is compared against <DIR>/<name>-<layer>.<pos>.f32 and the
// logits against <DIR>/logits.<pos>.f32.
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "gguf/model_files.hpp"

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

std::optional<Diff> compare(const std::filesystem::path& file, const float* got, size_t n)
{
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in || size_t(in.tellg()) != n * 4)
        return std::nullopt;
    std::vector<float> ref(n);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(ref.data()), std::streamsize(n * 4));
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

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3) {
        std::println(stderr, "usage: mako_run <model.gguf> <id,id,...> [-n N] [--ctx N] [--top K]");
        return 2;
    }
    int n_gen = 8, top = 5;
    std::string ref_dir;
    mako::glm::Options opts;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "-n")
            n_gen = std::stoi(argv[i + 1]);
        else if (a == "--ctx")
            opts.max_ctx = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--top")
            top = std::stoi(argv[i + 1]);
        else if (a == "--indexer-top-k")
            opts.indexer_top_k = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--ref")
            ref_dir = argv[i + 1];
    }
    auto ids = parse_ids(argv[2]);
    if (ids.empty()) {
        std::println(stderr, "no token ids");
        return 2;
    }

    auto model = mako::gguf::ModelFiles::open(argv[1]);
    if (!model) {
        std::println(stderr, "{}", model.error());
        return 1;
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

    double worst_rel = 0;
    auto report = [&](const std::string& name, const float* data, size_t n) {
        const auto file = std::filesystem::path(ref_dir) / (name + "." + std::to_string(engine.position()) + ".f32");
        if (auto d = compare(file, data, n)) {
            worst_rel = std::max(worst_rel, d->rel);
            std::println("  {:16} rel {:9.2e}  cos {:.6f}  max {:9.2e}", name, d->rel, d->cos, d->max_abs);
        } else {
            std::println("  {:16} (no reference)", name);
        }
    };
    if (!ref_dir.empty())
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
    for (size_t i = 0; i < ids.size(); ++i) {
        auto t = std::chrono::steady_clock::now();
        logits = engine.step(ids[i]);
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
    for (int g = 0; g < n_gen; ++g) {
        const int next = top_k(logits, 1)[0];
        gen += std::format("{},", next);
        auto t = std::chrono::steady_clock::now();
        logits = engine.step(next);
        std::println("gen {:4} in {:7} {:7.0f} ms  top{}", g, next,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(),
                     show(logits));
    }
    if (!ref_dir.empty())
        std::println("worst probe rel error {:.2e}", worst_rel);
    const auto& st = engine.stats();
    std::println("greedy: {}", gen);
    std::println("expert reads {:.0f} ms ({:.2f} GB), gpu wait {:.0f} ms", st.expert_read_ms, st.expert_bytes / 1e9,
                 st.gpu_wait_ms);
    return 0;
}
