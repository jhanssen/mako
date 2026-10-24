// Compares two runs' logits position by position (files <dir>/logits.<pos>.f32): top-1 agreement and the KL
// divergence KL(P_a || P_b) of their softmax distributions.
//   logit_compare <dir_a> <dir_b> [first_pos] [last_pos]
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

std::vector<double> log_softmax(const std::vector<float>& x)
{
    const double m = *std::max_element(x.begin(), x.end());
    double s = 0;
    for (float v : x)
        s += std::exp(v - m);
    const double lse = m + std::log(s);
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = x[i] - lse;
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: logit_compare <dir_a> <dir_b> [first_pos] [last_pos]");
        return 2;
    }
    const std::filesystem::path a = argv[1], b = argv[2];
    const int first = argc > 3 ? std::stoi(argv[3]) : 0, last = argc > 4 ? std::stoi(argv[4]) : 1 << 30;
    std::vector<double> kls;
    int agree = 0, n = 0;
    for (int pos = first; pos <= last; ++pos) {
        const auto name = std::format("logits.{}.f32", pos);
        const auto la = load(a / name), lb = load(b / name);
        if (la.empty() || lb.empty())
            break;
        if (la.size() != lb.size()) {
            std::println(stderr, "position {}: sizes differ", pos);
            return 1;
        }
        const auto pa = log_softmax(la), pb = log_softmax(lb);
        double kl = 0;
        for (size_t i = 0; i < pa.size(); ++i)
            kl += std::exp(pa[i]) * (pa[i] - pb[i]);
        kls.push_back(kl);
        agree += std::max_element(la.begin(), la.end()) - la.begin() == std::max_element(lb.begin(), lb.end()) - lb.begin();
        ++n;
    }
    if (!n) {
        std::println(stderr, "no common positions");
        return 1;
    }
    std::vector<double> sorted = kls;
    std::ranges::sort(sorted);
    double mean = 0;
    for (double k : kls)
        mean += k;
    mean /= n;
    std::println("{} positions: top-1 agreement {:.2f}%, KL mean {:.5f}, median {:.5f}, p90 {:.5f}, p99 {:.5f}, max {:.5f}",
                 n, 100.0 * agree / n, mean, sorted[n / 2], sorted[n * 9 / 10], sorted[std::min(n - 1, n * 99 / 100)],
                 sorted.back());
    return 0;
}
