// Vision preprocessing: each test image decoded, sized and normalised for both projector types, against the
// reference's SHA-256 of its prepared input (tests/data/vision/expected.txt).
//   vision_prep_test <tests/data/vision>
#include "vision/image.hpp"
#include "vision/preprocess.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace vision = mako::vision;

std::string sha256(const std::vector<float>& v)
{
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(v.data(), CC_LONG(v.size() * sizeof(float)), d);
    std::string hex;
    for (unsigned char c : d)
        hex += std::format("{:02x}", c);
    return hex;
}

// The sizing and normalisation values of the two projector files.
vision::PrepParams params(const std::string& kind)
{
    vision::PrepParams p;
    if (kind == "deepseek41v") {
        p.kind = vision::Projector::deepseek41v;
        p.merge = 3;
        p.mean = {0.5f, 0.5f, 0.5f};
        p.std = {0.5f, 0.5f, 0.5f};
        p.min_pixels = 295936;
        p.max_pixels = 1806336;
    } else {
        p.kind = vision::Projector::glm5v;
        p.merge = 2;
        p.mean = {0.48145467042922974f, 0.45782750844955444f, 0.40821072459220886f};
        p.std = {0.2686295509338379f, 0.2613025903701782f, 0.27577710151672363f};
        p.min_pixels = 12544;
        p.max_pixels = 6272000;
    }
    return p;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: vision_prep_test <tests/data/vision>");
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    std::ifstream list(dir / "expected.txt");
    int cases = 0, failures = 0;
    for (std::string line; std::getline(list, line);) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream in(line);
        std::string kind, file, dims, want;
        in >> kind >> file >> dims >> want;
        ++cases;
        try {
            std::ifstream f(dir / file, std::ios::binary);
            const std::vector<char> bytes{std::istreambuf_iterator<char>(f), {}};
            const auto prep = vision::prepare(params(kind), vision::decode_image(std::as_bytes(std::span(bytes))));
            const std::string got_dims = std::format("{}x{}", prep.width, prep.height), got = sha256(prep.chw);
            if (got_dims != dims || got != want) {
                ++failures;
                std::println("FAIL {} {}: {} {} (want {} {})", kind, file, got_dims, got, dims, want);
            }
        } catch (const std::exception& e) {
            ++failures;
            std::println("FAIL {} {}: {}", kind, file, e.what());
        }
    }
    std::println("{} cases, {} failures", cases, failures);
    return cases == 0 || failures ? 1 : 0;
}
