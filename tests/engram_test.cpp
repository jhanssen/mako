// DeepSeek-V4.1's Engram rows against the reference hasher (tests/data/ds41_engram_expected.txt: "t layer rows..." per
// position and layer, then "V t layer values..." for the first column's decoded row of the first positions), and
// hashing in chunks with the carried history against hashing in one go.
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"

#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::println(stderr, "usage: engram_test <model-00001.gguf> <engram.gguf> <expected.txt>");
        return 2;
    }
    if (!std::filesystem::exists(argv[1]) || !std::filesystem::exists(argv[2])) {
        std::println("model {} or Engram tables {} not found: skipped", argv[1], argv[2]);
        return 77;
    }
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m || !m->add(argv[2])) {
        std::println(stderr, "cannot open the model files");
        return 1;
    }
    auto cfg = mako::dsv41::Config::load(*m);
    auto e = cfg ? mako::dsv41::Engram::open(*cfg, *m) : std::unexpected(cfg.error());
    if (!e) {
        std::println(stderr, "{}", e.error());
        return 1;
    }

    std::ifstream in(argv[3]);
    std::string line;
    std::getline(in, line);
    std::vector<int32_t> ids;
    {
        std::stringstream ss(line.substr(line.find(':') + 1));
        for (std::string s; std::getline(ss, s, ',');)
            ids.push_back(std::stoi(s));
    }
    const uint32_t L = e->layers(), C = e->columns(), D = e->dim();
    std::vector<int64_t> want(ids.size() * L * C);
    std::vector<std::vector<float>> want_v;
    std::vector<std::pair<size_t, uint32_t>> v_at;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        if (line.starts_with("V ")) {
            std::string tag;
            size_t t;
            uint32_t l;
            ss >> tag >> t >> l;
            want_v.emplace_back(D);
            for (auto& x : want_v.back())
                ss >> x;
            v_at.emplace_back(t, l);
            continue;
        }
        size_t t;
        uint32_t l;
        ss >> t >> l;
        for (uint32_t c = 0; c < C; ++c)
            ss >> want[(t * L + l) * C + c];
    }

    int failures = 0;
    const std::vector<int32_t> start(e->history(), -1);
    std::vector<int64_t> got(want.size());
    e->hash(ids, start, got);
    if (got != want) {
        size_t i = 0;
        while (got[i] == want[i])
            ++i;
        std::println("FAIL rows: position {} differs", i / (L * C));
        ++failures;
    }
    // Chunks of uneven sizes, each with the history the previous ones leave.
    std::vector<int64_t> chunked(want.size());
    std::vector<int32_t> hist = start;
    for (size_t at = 0, n = 1; at < ids.size(); at += n, n = n * 2 + 1) {
        n = std::min(n, ids.size() - at);
        const std::span<const int32_t> part(ids.data() + at, n);
        e->hash(part, hist, std::span(chunked).subspan(at * L * C, n * L * C));
        hist = e->advance(hist, part);
    }
    if (chunked != want) {
        std::println("FAIL chunked hashing differs from hashing in one go");
        ++failures;
    }
    mako::gguf::ReadPool pool(4);
    std::vector<float> v(got.size() * D);
    e->read(got, v.data(), pool);
    for (size_t i = 0; i < v_at.size(); ++i) {
        const auto [t, l] = v_at[i];
        const float* row = v.data() + (t * L + l) * C * D;
        for (uint32_t j = 0; j < D; ++j)
            if (row[j] != want_v[i][j]) {
                std::println("FAIL values: position {} layer {} value {}: {} vs {}", t, l, j, row[j], want_v[i][j]);
                ++failures;
                break;
            }
    }
    std::println("{} positions x {} layers x {} rows, {} decoded rows: {} failures", ids.size(), L, C, v_at.size(),
                 failures);
    return failures ? 1 : 0;
}
