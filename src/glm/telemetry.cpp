#include "glm/telemetry.hpp"

#include "mako_version.hpp"

#include <sys/sysctl.h>

#include <algorithm>
#include <format>
#include <numeric>

namespace mako::glm {

namespace {

std::string sysctl_str(const char* name)
{
    char buf[256] = {};
    size_t len = sizeof(buf) - 1;
    if (::sysctlbyname(name, buf, &len, nullptr, 0) != 0)
        return "unknown";
    return buf;
}

uint64_t sysctl_u64(const char* name)
{
    uint64_t v = 0; // a 4-byte value fills the low half
    size_t len = sizeof(v);
    if (::sysctlbyname(name, &v, &len, nullptr, 0) != 0)
        return 0;
    return v;
}

std::string json_str(std::string_view s)
{
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        if (static_cast<unsigned char>(c) < 0x20)
            out += std::format("\\u{:04x}", int(c));
        else
            out += c;
    }
    return out + "\"";
}

double mean(const std::vector<double>& v) { return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }

double median(std::vector<double> v)
{
    if (v.empty())
        return 0;
    std::ranges::sort(v);
    return v[v.size() / 2];
}

} // namespace

std::optional<Telemetry> parse_telemetry(std::string_view s)
{
    if (s == "off")
        return Telemetry::off;
    if (s == "summary")
        return Telemetry::summary;
    if (s == "steps")
        return Telemetry::steps;
    if (s == "layers")
        return Telemetry::layers;
    if (s == "kernels")
        return Telemetry::kernels;
    return std::nullopt;
}

MachineInfo query_machine(const mtl::DeviceInfo& gpu)
{
    MachineInfo m;
    m.commit = MAKO_COMMIT;
    m.chip = sysctl_str("machdep.cpu.brand_string");
    m.macos = sysctl_str("kern.osproductversion");
    m.macos_build = sysctl_str("kern.osversion");
    m.perf_cores = uint32_t(sysctl_u64("hw.perflevel0.physicalcpu"));
    m.eff_cores = uint32_t(sysctl_u64("hw.perflevel1.physicalcpu"));
    m.ram = sysctl_u64("hw.memsize");
    m.wired_limit_mb = sysctl_u64("iogpu.wired_limit_mb");
    m.gpu = gpu;
    return m;
}

void Report::machine(const MachineInfo& m)
{
    machine_ = m;
    if (!out_)
        return;
    *out_ << std::format("{{\"type\":\"machine\",\"commit\":{},\"chip\":{},\"macos\":{},\"macos_build\":{},"
                         "\"perf_cores\":{},\"eff_cores\":{},\"ram\":{},\"wired_limit_mb\":{},\"gpu\":{},"
                         "\"apple9\":{},\"apple10\":{},\"metal4\":{},\"max_buffer\":{},\"working_set\":{},"
                         "\"counters_stage\":{},\"counters_dispatch\":{},\"timestamp_set\":{}}}\n",
                         json_str(m.commit), json_str(m.chip), json_str(m.macos), json_str(m.macos_build),
                         m.perf_cores, m.eff_cores, m.ram, m.wired_limit_mb, json_str(m.gpu.name), m.gpu.apple9,
                         m.gpu.apple10, m.gpu.metal4, m.gpu.max_buffer, m.gpu.recommended_working_set,
                         m.gpu.counters_stage, m.gpu.counters_dispatch, m.gpu.timestamp_set);
}

void Report::load(const LoadStats& l)
{
    load_ = l;
    if (!out_)
        return;
    *out_ << std::format("{{\"type\":\"load\",\"dense_ms\":{:.1f},\"dense_bytes\":{},\"key_ms\":{:.1f},"
                         "\"arena_ms\":{:.1f},\"arena_bytes\":{},\"arena_budget\":{:.0f},\"slots_min\":{},"
                         "\"slots_max\":{},\"prefill_ms\":{:.1f},\"warm_ms\":{:.1f},\"total_ms\":{:.1f},"
                         "\"profile\":{}}}\n",
                         l.dense_ms, l.dense_bytes, l.key_ms, l.arena_ms, l.arena_bytes, l.arena_budget, l.slots_min,
                         l.slots_max, l.prefill_ms, l.warm_ms, l.total_ms, json_str(l.profile));
}

void Report::step(const StepStats& s)
{
    steps_.push_back(s);
    if (!out_ || level_ < Telemetry::steps)
        return;
    std::string layers;
    for (const LayerTimes& t : s.layers)
        layers += std::format("{}{{\"layer\":{},\"attn_ms\":{:.3f},\"ffn_ms\":{:.3f},\"hit_ms\":{:.3f},\"miss_ms\":{:.3f}}}",
                              layers.empty() ? "" : ",", t.layer, t.attn_ms, t.ffn_ms, t.hit_ms, t.miss_ms);
    std::string kernels;
    for (const KernelTime& k : s.kernels)
        kernels += std::format("{}{{\"kernel\":{},\"ms\":{:.4f},\"count\":{}}}", kernels.empty() ? "" : ",",
                               json_str(k.kernel), k.ms, k.count);
    *out_ << std::format("{{\"type\":\"step\",\"pos\":{},\"token\":{},\"wall_ms\":{:.2f},\"encode_ms\":{:.2f},"
                         "\"gpu_ms\":{:.2f},\"route_wait_ms\":{:.2f},\"ensure_ms\":{:.2f},\"read_ms\":{:.2f},"
                         "\"hits\":{},\"misses\":{},\"read_bytes\":{}{}}}\n",
                         s.pos, s.token, s.wall_ms, s.encode_ms, s.gpu_ms, s.route_wait_ms, s.ensure_ms, s.read_ms,
                         s.hits, s.misses, s.read_bytes,
                         (s.layers.empty() ? std::string() : ",\"layers\":[" + layers + "]") +
                             (s.kernels.empty() ? std::string() : ",\"kernels\":[" + kernels + "]"));
}

void Report::prefill(const StepStats& s, uint32_t tokens)
{
    prefills_.emplace_back(s, tokens);
    if (!out_)
        return;
    double sums[2][4] = {};
    for (const LayerTimes& t : s.layers) {
        double* row = sums[t.mla ? 1 : 0];
        row[0] += t.attn_ms;
        row[1] += t.ffn_ms;
        row[2] += t.hit_ms;
        row[3] += t.miss_ms;
    }
    *out_ << std::format("{{\"type\":\"prefill\",\"pos\":{},\"tokens\":{},\"wall_ms\":{:.1f},\"gpu_ms\":{:.1f},"
                         "\"ensure_ms\":{:.1f},\"read_ms\":{:.1f},\"misses\":{},\"read_bytes\":{},"
                         "\"kda_attn_ms\":{:.1f},\"mla_attn_ms\":{:.1f},\"ffn_ms\":{:.1f},\"experts_ms\":{:.1f},"
                         "\"combine_ms\":{:.1f}}}\n",
                         s.pos, tokens, s.wall_ms, s.gpu_ms, s.ensure_ms, s.read_ms, s.misses, s.read_bytes, sums[0][0],
                         sums[1][0], sums[0][1] + sums[1][1], sums[0][2] + sums[1][2], sums[0][3] + sums[1][3]);
}

std::string Report::markdown(size_t skip) const
{
    const MachineInfo& m = machine_;
    std::string md = "### mako telemetry\n\n";
    md += std::format("- mako `{}`, macOS {} ({}), {}, {}P+{}E cores, {:.1f} GB RAM, GPU wired limit {}\n", m.commit,
                      m.macos, m.macos_build, m.chip, m.perf_cores, m.eff_cores, m.ram / 1e9,
                      m.wired_limit_mb ? std::format("{} MB", m.wired_limit_mb) : std::string("default"));
    md += std::format("- GPU {}: Apple9 {}, Apple10 {}, Metal 4 {}, max buffer {:.1f} GB, working set {:.1f} GB, "
                      "timestamps at stage boundaries {}, at dispatch boundaries {}\n",
                      m.gpu.name, m.gpu.apple9, m.gpu.apple10, m.gpu.metal4, m.gpu.max_buffer / 1e9,
                      m.gpu.recommended_working_set / 1e9, m.gpu.counters_stage, m.gpu.counters_dispatch);
    const LoadStats& l = load_;
    md += std::format("- load {:.1f} s: dense weights {:.1f} GB in {:.1f} s, routing key {:.2f} s, arena {:.1f} GB "
                      "(budget {:.1f} GB, {}-{} slots per layer) allocated in {:.1f} s, prefilled in {:.1f} s, GPU "
                      "warm-up {:.2f} s; routing profile: {}\n",
                      l.total_ms / 1e3, l.dense_bytes / 1e9, l.dense_ms / 1e3, l.key_ms / 1e3, l.arena_bytes / 1e9,
                      l.arena_budget / 1e9, l.slots_min, l.slots_max, l.arena_ms / 1e3, l.prefill_ms / 1e3,
                      l.warm_ms / 1e3, l.profile.empty() ? "none" : l.profile);
    for (const auto& [p, tokens] : prefills_) {
        md += std::format("- prefill, {} tokens at position {}: {:.0f} ms ({:.1f} tok/s), GPU {:.0f} ms, making experts "
                          "resident {:.0f} ms ({} misses, {:.1f} GB read in {:.0f} ms)\n",
                          tokens, p.pos, p.wall_ms, tokens / p.wall_ms * 1e3, p.gpu_ms, p.ensure_ms, p.misses,
                          p.read_bytes / 1e9, p.read_ms);
        if (!p.layers.empty()) {
            double kda = 0, mla = 0, ffn = 0, ex = 0, comb = 0;
            for (const LayerTimes& t : p.layers) {
                (t.mla ? mla : kda) += t.attn_ms;
                ffn += t.ffn_ms;
                ex += t.hit_ms;
                comb += t.miss_ms;
            }
            md += std::format("  - GPU by section: KDA attention {:.0f} ms, MLA attention {:.0f} ms, FFN to routing "
                              "{:.0f} ms, experts {:.0f} ms, combine and rest {:.0f} ms\n",
                              kda, mla, ffn, ex, comb);
        }
    }
    if (steps_.size() <= skip) {
        md += "- no decode steps measured\n";
        return md;
    }
    std::vector<double> wall, encode, gpu, wait, ensure, read;
    uint64_t hits = 0, misses = 0, bytes = 0;
    for (size_t i = skip; i < steps_.size(); ++i) {
        const StepStats& s = steps_[i];
        wall.push_back(s.wall_ms);
        encode.push_back(s.encode_ms);
        gpu.push_back(s.gpu_ms);
        wait.push_back(s.route_wait_ms);
        ensure.push_back(s.ensure_ms);
        read.push_back(s.read_ms);
        hits += s.hits;
        misses += s.misses;
        bytes += s.read_bytes;
    }
    const double n = double(wall.size());
    md += std::format("- decode, {} steps (first {} left out): {:.1f} ms/token median ({:.1f} tok/s), mean {:.1f}; per "
                      "token: encode {:.2f} ms, GPU {:.1f} ms, CPU waiting for routing {:.1f} ms, making experts "
                      "resident {:.1f} ms\n",
                      wall.size(), skip, median(wall), 1e3 / median(wall), mean(wall), mean(encode), mean(gpu),
                      mean(wait), mean(ensure));
    md += std::format("- experts: {:.1f}% resident, {:.1f} misses/token, {:.1f} MB/token read in {:.1f} ms\n",
                      100.0 * hits / std::max<uint64_t>(1, hits + misses), misses / n, bytes / n / 1e6, mean(read));
    if (level_ >= Telemetry::kernels) {
        std::vector<KernelTime> agg;
        for (size_t i = skip; i < steps_.size(); ++i)
            for (const KernelTime& k : steps_[i].kernels) {
                auto it = std::ranges::find(agg, k.kernel, &KernelTime::kernel);
                if (it == agg.end())
                    it = agg.insert(agg.end(), KernelTime{k.kernel, 0, 0});
                it->ms += k.ms;
                it->count += k.count;
            }
        std::ranges::sort(agg, [](const KernelTime& a, const KernelTime& b) { return a.ms > b.ms; });
        double total = 0;
        for (const KernelTime& k : agg)
            total += k.ms;
        md += std::format("\n| kernel | ms/token | share | dispatches/token | us/dispatch |\n|---|---|---|---|---|\n");
        for (size_t i = 0; i < std::min<size_t>(agg.size(), 30); ++i)
            md += std::format("| {} | {:.3f} | {:.1f}% | {:.0f} | {:.1f} |\n", agg[i].kernel, agg[i].ms / n,
                              100 * agg[i].ms / total, agg[i].count / n, 1e3 * agg[i].ms / agg[i].count);
        md += std::format("| all kernels | {:.2f} | | {:.0f} | |\n", total / n,
                          std::accumulate(agg.begin(), agg.end(), 0.0, [](double a, const KernelTime& k) { return a + k.count; }) / n);
    }
    if (level_ == Telemetry::layers && !steps_[skip].layers.empty()) {
        md += "\n| layer | attention ms | FFN ms | resident experts ms | misses + combine ms |\n|---|---|---|---|---|\n";
        for (size_t j = 0; j < steps_[skip].layers.size(); ++j) {
            double a = 0, f = 0, h = 0, mi = 0;
            for (size_t i = skip; i < steps_.size(); ++i) {
                const LayerTimes& t = steps_[i].layers[j];
                a += t.attn_ms;
                f += t.ffn_ms;
                h += t.hit_ms;
                mi += t.miss_ms;
            }
            md += std::format("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |\n", steps_[skip].layers[j].layer, a / n,
                              f / n, h / n, mi / n);
        }
    }
    return md;
}

} // namespace mako::glm
