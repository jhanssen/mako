// Replays an expert-arena routing trace (MAKO_ARENA_TRACE=file while an engine runs: every ensure, touch and
// prefetch with the initial residency and eviction scores) through the arena's eviction policy and variants of it,
// and prints each one's misses over the decode calls (ensure() calls of n_used experts). The baseline's count
// against the recorded misses says how faithful the replay is.
//   arena_replay <trace>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <print>
#include <string>
#include <vector>

namespace {

struct Record {
    char kind;
    uint16_t layer, misses;
    std::vector<uint16_t> experts;
};

struct Trace {
    uint32_t n_layer = 0, n_expert = 0, n_used = 0;
    std::vector<std::vector<int32_t>> owner; // [layer][slot]
    std::vector<std::vector<float>> seed;    // [layer][expert]
    std::vector<Record> records;
};

Trace read_trace(const char* path)
{
    std::FILE* f = std::fopen(path, "rb");
    if (!f)
        throw std::runtime_error(std::string("open ") + path);
    auto get = [&](void* p, size_t n) {
        if (std::fread(p, 1, n, f) != n)
            throw std::runtime_error("short trace");
    };
    char magic[8];
    get(magic, 8);
    if (std::memcmp(magic, "MAKOTRC1", 8) != 0)
        throw std::runtime_error("not a trace");
    Trace t;
    uint32_t hdr[3];
    get(hdr, sizeof(hdr));
    t.n_layer = hdr[0], t.n_expert = hdr[1], t.n_used = hdr[2];
    t.owner.resize(t.n_layer);
    t.seed.resize(t.n_layer);
    for (uint32_t l = 0; l < t.n_layer; ++l) {
        uint32_t ns = 0;
        get(&ns, 4);
        t.owner[l].resize(ns);
        get(t.owner[l].data(), size_t(ns) * 4);
        t.seed[l].resize(t.n_expert);
        get(t.seed[l].data(), size_t(t.n_expert) * 4);
    }
    for (;;) {
        char kind;
        if (std::fread(&kind, 1, 1, f) != 1)
            break;
        uint16_t h[3];
        get(h, sizeof(h));
        Record r{kind, h[0], h[2], std::vector<uint16_t>(h[1])};
        get(r.experts.data(), size_t(h[1]) * 2);
        t.records.push_back(std::move(r));
    }
    std::fclose(f);
    return t;
}

struct Policy {
    std::string name;
    double half_life; // ensure() calls of a layer; 0: no decay
    double seed_w;    // the recorded seed scores times this
    bool lru;         // evict the least recently used slot regardless of score
    bool prefetch;    // apply the reads ahead
};

struct Result {
    uint64_t decode_calls = 0, decode_misses = 0, all_misses = 0, recorded = 0;
};

Result replay(const Trace& t, const Policy& p)
{
    struct Layer {
        std::vector<int32_t> table, owner;
        std::vector<uint64_t> last;
        std::vector<double> score;
    };
    std::vector<Layer> L(t.n_layer);
    for (uint32_t l = 0; l < t.n_layer; ++l) {
        auto& x = L[l];
        x.table.assign(t.n_expert, -1);
        x.owner = t.owner[l];
        x.last.assign(x.owner.size(), 0);
        x.score.assign(t.n_expert, 0.0);
        for (uint32_t e = 0; e < t.n_expert; ++e)
            x.score[e] = double(t.seed[l][e]) * p.seed_w;
        for (size_t s = 0; s < x.owner.size(); ++s)
            if (x.owner[s] >= 0)
                x.table[x.owner[s]] = int32_t(s);
    }
    const double decay = p.half_life > 0 ? std::exp2(-1.0 / p.half_life) : 1.0;
    uint64_t clock = 0;
    Result r;
    auto victim_of = [&](Layer& x, const std::vector<char>& taken) {
        int32_t v = -1;
        for (size_t s = 0; s < x.owner.size(); ++s) {
            if (taken[s])
                continue;
            if (x.owner[s] < 0)
                return int32_t(s);
            if (v < 0) {
                v = int32_t(s);
                continue;
            }
            if (p.lru) {
                if (x.last[s] < x.last[v])
                    v = int32_t(s);
            } else {
                const double us = x.score[x.owner[s]], uv = x.score[x.owner[v]];
                if (us < uv || (us == uv && x.last[s] < x.last[v]))
                    v = int32_t(s);
            }
        }
        return v;
    };
    auto place = [&](Layer& x, int e, int32_t v) {
        if (x.owner[v] >= 0)
            x.table[x.owner[v]] = -1;
        x.owner[v] = e;
        x.table[e] = v;
        x.last[v] = clock;
    };
    for (const Record& rec : t.records) {
        Layer& x = L[rec.layer];
        if (rec.kind == 'T') {
            for (double& c : x.score)
                c *= decay;
            for (uint16_t e : rec.experts)
                x.score[e] += 1.0;
            continue;
        }
        if (rec.kind == 'P') {
            if (!p.prefetch)
                continue;
            std::vector<char> taken(x.owner.size(), 0);
            for (uint16_t e : rec.experts) {
                if (x.table[e] >= 0)
                    continue;
                const int32_t v = victim_of(x, taken);
                if (v < 0)
                    break;
                place(x, e, v);
                taken[v] = 1;
            }
            continue;
        }
        if (rec.kind != 'E')
            continue;
        std::vector<char> taken(x.owner.size(), 0);
        for (uint16_t e : rec.experts)
            if (x.table[e] >= 0)
                taken[x.table[e]] = 1;
        ++clock;
        for (double& c : x.score)
            c *= decay;
        for (uint16_t e : rec.experts)
            x.score[e] += 1.0;
        uint64_t misses = 0;
        for (uint16_t e : rec.experts) {
            if (x.table[e] >= 0) {
                x.last[x.table[e]] = clock;
                continue;
            }
            ++misses;
            const int32_t v = victim_of(x, taken);
            if (v < 0)
                throw std::runtime_error("no slot free");
            place(x, e, v);
            taken[v] = 1;
        }
        r.all_misses += misses;
        if (rec.experts.size() == t.n_used) {
            ++r.decode_calls;
            r.decode_misses += misses;
            r.recorded += rec.misses;
        }
    }
    return r;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: arena_replay <trace>");
        return 2;
    }
    const Trace t = read_trace(argv[1]);
    size_t ens = 0, touch = 0, pf = 0;
    for (const auto& r : t.records)
        (r.kind == 'E' ? ens : r.kind == 'T' ? touch : pf)++;
    std::println("{} layers, {} experts, top {}; {} ensure, {} touch, {} prefetch records", t.n_layer, t.n_expert, t.n_used,
                 ens, touch, pf);
    const std::vector<Policy> policies{
        {"as the arena (half-life 256, seed x1)", 256, 1, false, true},
        {"half-life 64", 64, 1, false, true},
        {"half-life 128", 128, 1, false, true},
        {"half-life 512", 512, 1, false, true},
        {"half-life 1024", 1024, 1, false, true},
        {"no decay", 0, 1, false, true},
        {"seed x0", 256, 0, false, true},
        {"seed x0.3", 256, 0.3, false, true},
        {"seed x3", 256, 3, false, true},
        {"seed x10", 256, 10, false, true},
        {"seed x10, half-life 1024", 1024, 10, false, true},
        {"least recently used", 256, 1, true, true},
        {"as the arena, reads ahead ignored", 256, 1, false, false},
    };
    uint32_t moe_layers = 0;
    for (const auto& o : t.owner)
        moe_layers += !o.empty();
    std::println("{:40} {:>13} {:>13} {:>10}", "policy", "decode misses", "a step", "all");
    for (const Policy& p : policies) {
        const Result r = replay(t, p);
        const double steps = moe_layers ? double(r.decode_calls) / moe_layers : 1.0;
        std::println("{:40} {:13} {:13.2f} {:10}", p.name, r.decode_misses, double(r.decode_misses) / steps, r.all_misses);
        if (&p == &policies.front())
            std::println("{:40} {:13} {:13.2f}", "  recorded by the engine", r.recorded, double(r.recorded) / steps);
    }
    return 0;
}
