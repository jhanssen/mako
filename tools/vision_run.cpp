// Runs an image through a vision projector file.
//   vision_run <mmproj.gguf> <image> [--prep-out FILE]
// --prep-out: the encoder's prepared input as raw fp32, planar [3][height][width].
#include "vision/image.hpp"
#include "vision/mmproj.hpp"
#include "vision/preprocess.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "usage: vision_run <mmproj.gguf> <image> [--prep-out FILE]");
        return 2;
    }
    std::string prep_out;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--prep-out")
            prep_out = argv[i + 1];
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    auto mm = mako::vision::open_mmproj(argv[1]);
    if (!mm) {
        std::println(stderr, "{}", mm.error());
        return 1;
    }
    std::ifstream in(argv[2], std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
    if (!in && !in.eof()) {
        std::println(stderr, "cannot read {}", argv[2]);
        return 1;
    }
    try {
        const auto img = mako::vision::decode_image(std::as_bytes(std::span(bytes)));
        const auto prep = mako::vision::prepare(mm->prep, img);
        std::println("image {} x {} -> {} x {}", img.width, img.height, prep.width, prep.height);
        if (!prep_out.empty()) {
            std::FILE* f = std::fopen(prep_out.c_str(), "wb");
            if (!f || std::fwrite(prep.chw.data(), 4, prep.chw.size(), f) != prep.chw.size()) {
                std::println(stderr, "cannot write {}", prep_out);
                return 1;
            }
            std::fclose(f);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
    return 0;
}
