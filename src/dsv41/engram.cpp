#include "dsv41/engram.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <format>

namespace mako::dsv41 {

namespace {

float e4m3(uint8_t b)
{
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(float(8 + m), e - 10) : std::ldexp(float(m), -9);
    return b & 128 ? -v : v;
}

// Code x scale -> value; E4M3's 3-bit mantissa makes every finite product exact in BF16.
struct Lut {
    float v[256][256]; // [scale][code]
    Lut()
    {
        for (int s = 0; s < 256; ++s)
            for (int c = 0; c < 256; ++c)
                v[s][c] = (c & 127) == 127 || s == 255 ? NAN : std::ldexp(e4m3(uint8_t(c)), s - 127);
    }
};

const Lut& lut()
{
    static const Lut l;
    return l;
}

} // namespace

std::string Engram::find_tables(const std::string& first_shard)
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(first_shard).parent_path();
    for (const fs::path& d : {dir / "Engram", dir.parent_path() / "Engram"}) {
        std::error_code ec;
        if (!fs::is_directory(d, ec))
            continue;
        std::vector<fs::path> files;
        for (const auto& e : fs::directory_iterator(d, ec))
            if (e.path().extension() == ".gguf")
                files.push_back(e.path());
        std::ranges::sort(files);
        for (const auto& f : files) {
            auto g = gguf::File::open(f.string());
            if (g && (g->tensor("per_layer_token_embd.weight") || g->tensor("blk.1.engram_embd.weight")))
                return f.string();
        }
    }
    return {};
}

std::expected<Engram, std::string> Engram::open(const Config& cfg, const gguf::ModelFiles& m)
{
    const auto& c = cfg.engram;
    Engram e;
    e.ngram_ = c.max_ngram;
    e.cols_ = c.columns();
    e.dim_ = c.head_dim;
    e.token_map_ = c.token_map;
    e.mult_ = c.multipliers;
    e.primes_ = c.primes;
    e.offsets_ = c.offsets;
    if (e.dim_ % 32 || e.ngram_ < 2 || e.ngram_ > 8)
        return std::unexpected(std::format("engram: head_dim {} / max_ngram {} unsupported", e.dim_, e.ngram_));
    e.row_bytes_ = e.dim_ / 32 * 33;
    const auto pad = c.pad_token;
    e.pad_ = pad >= 0 && size_t(pad) < c.token_map.size() ? c.token_map[size_t(pad)] : pad;

    auto table = [&](const gguf::ModelFiles::Ref& r, uint64_t first, uint64_t rows) -> std::expected<Table, std::string> {
        const auto& t = *r.tensor;
        if (t.type != gguf::Type::MXFP8 || t.dims.size() != 2 || t.dims[0] != e.dim_)
            return std::unexpected(std::format("engram: {} is not MXFP8 [{} x rows]", t.name, e.dim_));
        if (first + rows > t.dims[1])
            return std::unexpected(std::format("engram: {} has {} rows, the tables need {}", t.name, t.dims[1],
                                               first + rows));
        return Table{r.file->fd(), t.offset + first * e.row_bytes_, rows};
    };
    if (auto all = m.find("per_layer_token_embd.weight")) {
        uint64_t first = 0;
        for (const auto rows : c.num_embeddings) {
            auto t = table(all, first, uint64_t(rows));
            if (!t)
                return std::unexpected(t.error());
            e.tables_.push_back(*t);
            first += uint64_t(rows);
        }
    } else {
        for (size_t i = 0; i < c.layers.size(); ++i) {
            const auto name = std::format("blk.{}.engram_embd.weight", c.layers[i]);
            const auto r = m.find(name);
            if (!r)
                return std::unexpected(std::format("engram: no {} (and no per_layer_token_embd.weight); give the "
                                                   "GGUF holding the Engram tables",
                                                   name));
            auto t = table(r, 0, uint64_t(c.num_embeddings[i]));
            if (!t)
                return std::unexpected(t.error());
            e.tables_.push_back(*t);
        }
    }
    for (size_t l = 0; l < e.tables_.size(); ++l)
        for (uint32_t col = 0; col < e.cols_; ++col)
            if (e.offsets_[l * e.cols_ + col] + e.primes_[l * e.cols_ + col] > int64_t(e.tables_[l].rows))
                return std::unexpected(std::format("engram: layer {} column {} reaches past its table", l, col));
    return e;
}

int32_t Engram::compress(int32_t token) const
{
    return token >= 0 && size_t(token) < token_map_.size() ? token_map_[size_t(token)] : -1;
}

void Engram::hash(std::span<const int32_t> tokens, std::span<const int32_t> before, std::span<int64_t> rows) const
{
    const uint32_t L = layers(), H = history();
    const auto at = [&](int64_t t, uint32_t back) -> int32_t { // compressed id `back` tokens before position t
        const int64_t i = t - int64_t(back);
        if (i >= 0)
            return compress(tokens[size_t(i)]);
        const int64_t h = int64_t(H) + i;
        return h >= 0 && size_t(h) < before.size() ? before[size_t(h)] : -1;
    };
    for (size_t t = 0; t < tokens.size(); ++t) {
        int64_t* out = rows.data() + t * L * cols_;
        const int32_t self = at(int64_t(t), 0);
        if (self < 0) {
            std::fill_n(out, L * cols_, int64_t(-1));
            continue;
        }
        int64_t ids[8];
        bool blocked = false;
        for (uint32_t s = 0; s < ngram_; ++s) {
            const int32_t id = at(int64_t(t), s);
            blocked = blocked || id < 0;
            ids[s] = blocked ? pad_ : id;
        }
        for (uint32_t l = 0; l < L; ++l) {
            const int64_t* mult = mult_.data() + size_t(l) * ngram_;
            int64_t h = ids[0] * mult[0];
            for (uint32_t n = 1; n < ngram_; ++n) {
                h ^= ids[n] * mult[n];
                for (uint32_t head = 0; head < cols_ / (ngram_ - 1); ++head) {
                    const uint32_t col = (n - 1) * (cols_ / (ngram_ - 1)) + head;
                    const size_t k = size_t(l) * cols_ + col;
                    out[l * cols_ + col] = h % primes_[k] + offsets_[k];
                }
            }
        }
    }
}

std::vector<int32_t> Engram::advance(std::span<const int32_t> before, std::span<const int32_t> tokens) const
{
    std::vector<int32_t> all(before.begin(), before.end());
    for (const auto t : tokens)
        all.push_back(compress(t));
    if (all.size() < history())
        all.insert(all.begin(), history() - all.size(), -1);
    return {all.end() - history(), all.end()};
}

void Engram::read(std::span<const int64_t> rows, float* out, gguf::ReadPool& pool) const
{
    const uint32_t L = layers();
    const size_t per_pos = size_t(L) * cols_;
    const size_t n = rows.size();
    constexpr size_t kChunk = 32;
    const auto& v = lut().v;
    pool.run((n + kChunk - 1) / kChunk, [&](size_t chunk) {
        std::vector<uint8_t> raw(row_bytes_);
        for (size_t i = chunk * kChunk; i < std::min(n, (chunk + 1) * kChunk); ++i) {
            float* dst = out + i * dim_;
            if (rows[i] < 0) {
                std::fill_n(dst, dim_, 0.0f);
                continue;
            }
            const auto& tb = tables_[(i % per_pos) / cols_];
            gguf::read_exact(tb.fd, raw.data(), row_bytes_, tb.offset + uint64_t(rows[i]) * row_bytes_);
            for (uint32_t b = 0; b < dim_ / 32; ++b) {
                const uint8_t* blk = raw.data() + b * 33;
                for (uint32_t j = 0; j < 32; ++j)
                    dst[b * 32 + j] = v[blk[0]][blk[1 + j]];
            }
        }
    });
}

void Engram::read_raw(std::span<const int64_t> rows, uint8_t* out, gguf::ReadPool& pool) const
{
    const size_t per_pos = size_t(layers()) * cols_;
    const size_t n = rows.size();
    // A decode step's ~48 rows go one per task so every reader thread issues one at a time (32 per task left two
    // threads reading 2.3 ms of rows per token); a prefill's thousands still batch.
    const size_t kChunk = std::max<size_t>(1, n / 256);
    pool.run((n + kChunk - 1) / kChunk, [&](size_t chunk) {
        for (size_t i = chunk * kChunk; i < std::min(n, (chunk + 1) * kChunk); ++i) {
            uint8_t* dst = out + i * row_bytes_;
            if (rows[i] < 0) {
                std::fill_n(dst, row_bytes_, uint8_t(0));
                continue;
            }
            const auto& tb = tables_[(i % per_pos) / cols_];
            gguf::read_exact(tb.fd, dst, row_bytes_, tb.offset + uint64_t(rows[i]) * row_bytes_);
        }
    });
}

} // namespace mako::dsv41
