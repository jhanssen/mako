// Recomputes a dense layer's FFN in fp64 from a dumped input and compares it with the dumped output:
//   down(silu(min(gate x, L)) * clamp(up x, -L, L)), x = <dir>/ffn_norm-<layer>.<pos>.f32,
//   compared against <dir>/ffn_out-<layer>.<pos>.f32, for each dir given.
//   ffn_check <model.gguf> <layer> <pos> <dir> [dir ...]
#include "glm/config.hpp"
#include "gguf/model_files.hpp"
#include "quant/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <vector>

namespace {

std::vector<float> load(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in)
        return {};
    std::vector<float> v(size_t(in.tellg()) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
    return v;
}

struct Matrix {
    uint64_t k = 0, n = 0;
    std::vector<float> w; // [n][k]
};

Matrix dequant(const mako::gguf::ModelFiles& model, const std::string& name)
{
    const auto ref = model.find(name);
    if (!ref)
        throw std::runtime_error("missing tensor " + name);
    const auto* t = ref.tensor;
    Matrix m{t->dims[0], t->dims[1], {}};
    const auto tr = mako::gguf::traits(t->type);
    const uint64_t rb = m.k / tr.block_elems * tr.block_bytes;
    auto data = ref.file->data(*t);
    m.w.resize(m.k * m.n);
    for (uint64_t r = 0; r < m.n; ++r)
        mako::quant::dequant_row(t->type, data.data() + r * rb, m.w.data() + r * m.k, m.k);
    return m;
}

std::vector<double> gemv(const Matrix& m, const std::vector<double>& x)
{
    std::vector<double> y(m.n);
    for (uint64_t r = 0; r < m.n; ++r) {
        double acc = 0;
        for (uint64_t i = 0; i < m.k; ++i)
            acc += double(m.w[r * m.k + i]) * x[i];
        y[r] = acc;
    }
    return y;
}

void report(const char* label, const std::vector<double>& exact, const std::vector<float>& got)
{
    double dd = 0, rr = 0, gr = 0, gg = 0;
    for (size_t i = 0; i < exact.size(); ++i) {
        const double e = got[i] - exact[i];
        dd += e * e;
        rr += exact[i] * exact[i];
        gg += double(got[i]) * got[i];
        gr += double(got[i]) * exact[i];
    }
    std::println("    {:28} rel {:9.2e}  cos {:.6f}", label, std::sqrt(dd / rr), gr / std::sqrt(gg * rr));
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 5) {
        std::println(stderr, "usage: ffn_check <model.gguf> <layer> <pos> <dir> [dir ...]");
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
    const int layer = std::stoi(argv[2]), pos = std::stoi(argv[3]);
    if (layer >= int(cfg->n_dense_lead)) {
        std::println(stderr, "layer {} is not dense", layer);
        return 2;
    }
    const std::string p = std::format("blk.{}.", layer);
    const Matrix gate = dequant(*model, p + "ffn_gate.weight");
    const Matrix up = dequant(*model, p + "ffn_up.weight");
    const Matrix down = dequant(*model, p + "ffn_down.weight");
    const double limit = cfg->swiglu_clamp_shexp[layer];

    int failures = 0;
    for (int i = 4; i < argc; ++i) {
        const std::filesystem::path dir = argv[i];
        const auto x = load(dir / std::format("ffn_norm-{}.{}.f32", layer, pos));
        const auto out = load(dir / std::format("ffn_out-{}.{}.f32", layer, pos));
        if (x.size() != gate.k || out.size() != down.n) {
            std::println("{}: missing or wrong-sized ffn_norm/ffn_out for layer {} pos {}", dir.string(), layer, pos);
            ++failures;
            continue;
        }
        const auto g = gemv(gate, std::vector<double>(x.begin(), x.end()));
        const auto u = gemv(up, std::vector<double>(x.begin(), x.end()));
        std::vector<double> m(g.size());
        for (size_t j = 0; j < m.size(); ++j) {
            double gj = g[j], uj = u[j];
            if (limit > 1e-6) {
                gj = std::min(gj, limit);
                uj = std::clamp(uj, -limit, limit);
            }
            m[j] = gj / (1 + std::exp(-gj)) * uj;
        }
        const auto y = gemv(down, m);
        std::println("{} layer {} pos {}:", dir.string(), layer, pos);
        report("ffn_out vs fp64(own input)", y, out);
    }
    return failures ? 1 : 0;
}
