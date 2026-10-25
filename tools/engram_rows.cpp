// Prints the Engram rows of a token sequence (DeepSeek-V4.1): per position and Engram layer, the hashed table rows,
// and with --values the first values of the first column's row.
//
//   engram_rows <model-00001.gguf> <engram.gguf> <ids, comma-separated> [--values]
#include "dsv41/config.hpp"
#include "dsv41/engram.hpp"

#include <print>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::println(stderr, "usage: engram_rows <model-00001.gguf> <engram.gguf> <ids> [--values]");
        return 2;
    }
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m) {
        std::println(stderr, "{}", m.error());
        return 1;
    }
    if (auto r = m->add(argv[2]); !r) {
        std::println(stderr, "{}", r.error());
        return 1;
    }
    auto cfg = mako::dsv41::Config::load(*m);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    auto e = mako::dsv41::Engram::open(*cfg, *m);
    if (!e) {
        std::println(stderr, "{}", e.error());
        return 1;
    }
    std::vector<int32_t> ids;
    std::stringstream ss(argv[3]);
    for (std::string s; std::getline(ss, s, ',');)
        ids.push_back(std::stoi(s));
    const bool values = argc > 4 && std::string(argv[4]) == "--values";
    const uint32_t per = e->layers() * e->columns();
    std::vector<int64_t> rows(ids.size() * per);
    e->hash(ids, std::vector<int32_t>(e->history(), -1), rows);
    std::vector<float> v;
    if (values) {
        mako::gguf::ReadPool pool(4);
        v.resize(rows.size() * e->dim());
        e->read(rows, v.data(), pool);
    }
    for (size_t t = 0; t < ids.size(); ++t)
        for (uint32_t l = 0; l < e->layers(); ++l) {
            std::print("{} {}", t, l);
            for (uint32_t c = 0; c < e->columns(); ++c)
                std::print(" {}", rows[t * per + l * e->columns() + c]);
            if (values)
                for (uint32_t j = 0; j < 4; ++j)
                    std::print(" v{}", v[(t * per + l * e->columns()) * e->dim() + j]);
            std::println("");
        }
}
