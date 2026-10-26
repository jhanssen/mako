#pragma once

#include "dsv41/config.hpp"
#include "gguf/model_files.hpp"
#include "gguf/read_pool.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace mako::dsv41 {

// The Engram tables: per Engram layer and position, (max_ngram - 1) * heads hashed n-gram rows of head_dim values,
// MXFP8 (blocks of an E8M0 scale and 32 E4M3 codes), Q6_K or IQ4_NL on disk.
class Engram {
public:

    // Adds the GGUF the model's metadata names as holding its tables (deepseek41.engram.file, beside the first shard)
    // when `m` has none of its own; an error when it names one that cannot be opened.
    static std::expected<void, std::string> add_named(gguf::ModelFiles& m, const std::string& first_shard);
    // Finds the tables in `m` (one tensor holding every layer's table in turn, or one per layer).
    static std::expected<Engram, std::string> open(const Config& cfg, const gguf::ModelFiles& m);

    uint32_t layers() const { return uint32_t(tables_.size()); }
    uint32_t columns() const { return cols_; }
    uint32_t dim() const { return dim_; }
    // max_ngram - 1: the earlier tokens a position's hashes look at.
    uint32_t history() const { return ngram_ - 1; }

    // The compressed id of a token (tokens outside the map: -1, no rows).
    int32_t compress(int32_t token) const;

    // rows[t][layer][column] for tokens[t], each an index into that layer's table. `before` holds the compressed
    // ids of the history() tokens before tokens[0], oldest first, -1 for none (the sequence's start, or a token
    // without rows). A position whose own token has no rows gets -1 in every column.
    void hash(std::span<const int32_t> tokens, std::span<const int32_t> before, std::span<int64_t> rows) const;
    // The history after `tokens` follow `before`.
    std::vector<int32_t> advance(std::span<const int32_t> before, std::span<const int32_t> tokens) const;

    // Reads and decodes rows (as hash() gives them, n positions) into out[t][layer][column][dim] floats; rows of -1
    // become zeros. Throws std::runtime_error on a read error.
    void read(std::span<const int64_t> rows, float* out, gguf::ReadPool& pool) const;
    // The same rows' bytes as stored (row_bytes() each; zeros for -1, which decode to zeros).
    void read_raw(std::span<const int64_t> rows, uint8_t* out, gguf::ReadPool& pool) const;
    uint32_t row_bytes() const { return row_bytes_; }
    gguf::Type type() const { return type_; }

private:
    struct Table {
        int fd = -1;
        uint64_t offset = 0, rows = 0;
    };
    std::vector<Table> tables_;
    std::vector<int32_t> token_map_;
    std::vector<int64_t> mult_, primes_, offsets_;
    int64_t pad_ = 0;
    gguf::Type type_ = gguf::Type::MXFP8;
    uint32_t ngram_ = 0, cols_ = 0, dim_ = 0, row_bytes_ = 0;
};

} // namespace mako::dsv41
