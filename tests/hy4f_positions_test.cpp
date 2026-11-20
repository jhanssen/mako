// Hy4-flash's RoPE positions with images: the engine's positions, computed a run at a time from the run's first pool
// group on (as run() does), against the whole sequence's positions as the model's own code assigns them (text one
// step at a time; an image of r rows of c tokens at (start, start + row, start + col), the text after it from its
// largest position + 1). No GPU, no model.
#include "hy4f/engine.hpp"

#include <array>
#include <format>
#include <print>
#include <vector>

namespace {

using Pos = std::array<int32_t, 3>;

// segments: text runs (cols 0, count tokens) and images (rows x cols tokens).
struct Segment {
    uint32_t count, cols;
};

std::vector<Pos> reference(const std::vector<Segment>& segs)
{
    std::vector<Pos> out;
    int32_t cur = 0;
    for (const auto& s : segs) {
        if (!s.cols) {
            for (uint32_t i = 0; i < s.count; ++i)
                out.push_back({cur + int32_t(i), cur + int32_t(i), cur + int32_t(i)});
            cur += int32_t(s.count);
            continue;
        }
        int32_t mx = cur;
        for (uint32_t i = 0; i < s.count; ++i) {
            const Pos p{cur, cur + int32_t(i / s.cols), cur + int32_t(i % s.cols)};
            mx = std::max({mx, p[1], p[2]});
            out.push_back(p);
        }
        cur = mx + 1;
    }
    return out;
}

} // namespace

int main()
{
    int failures = 0, cases = 0;
    const std::vector<std::vector<Segment>> prompts{
        {{5, 0}, {15, 5}, {4, 0}, {8, 2}, {3, 0}},  // a wide image (3 x 5), a tall one (4 x 2)
        {{1, 0}, {12, 4}, {12, 4}, {2, 0}},         // two images in a row (3 x 4 each)
        {{3, 0}, {9, 3}, {7, 0}},                   // a square one
    };
    for (const auto& segs : prompts) {
        std::vector<mako::gen::InputRows> rows;
        uint32_t total = 0;
        for (const auto& s : segs) {
            if (s.cols)
                rows.push_back({total, s.count, nullptr, s.cols});
            total += s.count;
        }
        const auto want = reference(segs);
        for (const uint32_t chunk : {1u, 2u, 3u, 5u, 7u, 64u}) {
            constexpr uint32_t kp = 4; // the indexer's pool size: a run's positions start at its group
            int32_t delta = 0;
            bool ok = true;
            for (uint32_t pos = 0; pos < total;) {
                const uint32_t end = std::min(total, pos + chunk), lo = pos / kp * kp;
                std::vector<int32_t> out((end - lo) * 4);
                delta = mako::hy4f::Engine::rope_positions(rows, delta, pos, lo, end, out.data());
                for (uint32_t p = lo; p < end; ++p) {
                    const Pos got{out[(p - lo) * 4], out[(p - lo) * 4 + 1], out[(p - lo) * 4 + 2]};
                    if (got != want[p]) {
                        if (ok)
                            std::println("  FAIL chunk {}: position {}: ({}, {}, {}), want ({}, {}, {})", chunk, p,
                                         got[0], got[1], got[2], want[p][0], want[p][1], want[p][2]);
                        ok = false;
                    }
                }
                pos = end;
            }
            ++cases;
            failures += !ok;
        }
    }
    std::println("{} cases, {} failures", cases, failures);
    return failures ? 1 : 0;
}
