// Prints a GGUF file's metadata and tensor table; works on a partial download once the header is in.
//   gguf_dump <file> [--tensors] [--filter <substring>]
#include "gguf/gguf.hpp"

#include <cstdio>
#include <map>
#include <print>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: gguf_dump <file> [--tensors] [--filter <substring>]");
        return 2;
    }
    bool tensors = false;
    std::string filter;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tensors")
            tensors = true;
        else if (a == "--filter" && i + 1 < argc)
            filter = argv[++i];
    }
    auto f = mako::gguf::File::open(argv[1]);
    if (!f) {
        std::println(stderr, "{}", f.error());
        return 1;
    }
    std::println("GGUF v{}  {} tensors  data at {}  file {} bytes", f->version(), f->tensors().size(), f->data_offset(),
                 f->size());
    for (const auto& [k, v] : f->metadata())
        if (filter.empty() || k.find(filter) != std::string::npos)
            std::println("  {} = {}", k, mako::gguf::to_string(v));

    std::map<std::string, std::pair<uint64_t, uint64_t>> by_type;
    uint64_t total = 0;
    for (const auto& t : f->tensors()) {
        auto& e = by_type[std::string(mako::gguf::traits(t.type).name)];
        ++e.first;
        e.second += t.bytes;
        total += t.bytes;
        if (tensors && (filter.empty() || t.name.find(filter) != std::string::npos)) {
            std::string dims;
            for (uint64_t d : t.dims)
                dims += (dims.empty() ? "" : " x ") + std::to_string(d);
            std::println("  {:48} {:8} [{}]  {:.2f} MiB", t.name, mako::gguf::traits(t.type).name, dims,
                         t.bytes / 1048576.0);
        }
    }
    for (const auto& [name, e] : by_type)
        std::println("  {:8} {:5} tensors  {:8.2f} GiB", name, e.first, e.second / 1073741824.0);
    std::println("  total {:.2f} GiB", total / 1073741824.0);
    return 0;
}
