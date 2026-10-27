#include "vision/preprocess.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace mako::vision {

namespace {

// Round to bf16 (nearest, ties to even) and back.
float bf16_round(float v)
{
    uint32_t u = std::bit_cast<uint32_t>(v);
    if ((u & 0x7fffffff) > 0x7f800000) // NaN: quiet, keep the sign
        return std::bit_cast<float>((u | 0x00400000) & 0xffff0000);
    u = (u + (0x7fff + ((u >> 16) & 1))) & 0xffff0000;
    return std::bit_cast<float>(u);
}

Rgb8 canvas(uint32_t width, uint32_t height, uint8_t fill)
{
    return Rgb8{width, height, std::vector<uint8_t>(size_t(width) * height * 3, fill)};
}

void paste(Rgb8& dst, const Rgb8& src, uint32_t x0, uint32_t y0)
{
    for (uint32_t y = 0; y < src.height && y0 + y < dst.height; ++y) {
        const uint32_t n = std::min(src.width, dst.width - std::min(dst.width, x0));
        std::copy_n(src.px.data() + size_t(y) * src.width * 3, size_t(n) * 3,
                    dst.px.data() + (size_t(y0 + y) * dst.width + x0) * 3);
    }
}

Prepared normalise(const PrepParams& p, const Rgb8& img, bool bf16)
{
    Prepared out{img.width, img.height, std::vector<float>(size_t(img.width) * img.height * 3)};
    const size_t n = size_t(img.width) * img.height;
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < 3; ++c) {
            float v = (float(img.px[i * 3 + c]) / 255.0f - p.mean[c]) / p.std[c];
            out.chw[c * n + i] = bf16 ? bf16_round(v) : v;
        }
    return out;
}

constexpr int kDeepSeekMaxTokens = 1024; // the block limit DeepSeek-V4.1 was trained with
constexpr uint8_t kDeepSeekPad = 127;

// The canvas size for a width x height image: multiples of the patch size whose block fits max_tokens.
void deepseek_plan(int width, int height, int p, int r, int min_pixels, int max_tokens, int& best_w, int& best_h)
{
    const auto tokens = [&](int bh, int bw) {
        const int rows = (bh / p + r - 1) / r, cols = (bw / p + r - 1) / r;
        return rows * (cols + 1) + 2;
    };
    if (min_pixels > 0 && int64_t(width) * height > 0 && int64_t(width) * height < min_pixels) {
        const double ratio = std::sqrt(double(min_pixels) / (double(width) * double(height)));
        width = int(width * ratio);
        height = int(height * ratio);
    }
    best_w = (width + p - 1) / p * p;
    best_h = (height + p - 1) / p * p;
    if (tokens(best_h, best_w) <= max_tokens)
        return;
    // The largest aspect-preserving size whose block fits.
    const double ratio = double(height) / width;
    const double max_w = std::sqrt((max_tokens - 2) / ratio + 0.25) - 0.5, max_h = max_w * ratio;
    const int cell = p * r;
    if (max_w < 1.0) {
        best_h = (max_tokens - 2) / 2 * cell;
        best_w = cell;
    } else if (max_h < 1.0) {
        best_h = cell;
        best_w = (max_tokens - 3) * cell;
    } else {
        const double beta = std::min(std::floor(max_w) * cell / width, std::floor(max_h) * cell / height);
        best_h = int(std::floor(height * beta / p)) * p;
        best_w = int(std::floor(width * beta / p)) * p;
    }
    if (tokens(best_h, best_w) > max_tokens)
        throw std::runtime_error("image block over the token limit");
}

Prepared prepare_deepseek(const PrepParams& p, const Rgb8& img)
{
    int best_w = 0, best_h = 0;
    deepseek_plan(int(img.width), int(img.height), int(p.patch), int(p.merge), int(p.min_pixels),
                  p.max_tokens ? int(p.max_tokens) : kDeepSeekMaxTokens, best_w, best_h);
    // Fitted with the aspect kept, then centred; halves round to even.
    const auto round_even = [](double x) { return int(std::nearbyint(x)); };
    int new_w = best_w, new_h = best_h;
    const double im = double(img.width) / img.height, dest = double(best_w) / best_h;
    if (im > dest)
        new_h = round_even(double(img.height) / img.width * best_w);
    else if (im < dest)
        new_w = round_even(double(img.width) / img.height * best_h);
    const Rgb8 fitted = resize_bicubic(img, uint32_t(new_w), uint32_t(new_h));
    if (new_w == best_w && new_h == best_h)
        return normalise(p, fitted, true);
    Rgb8 padded = canvas(uint32_t(best_w), uint32_t(best_h), kDeepSeekPad);
    paste(padded, fitted, uint32_t(round_even((best_w - new_w) * 0.5)), uint32_t(round_even((best_h - new_h) * 0.5)));
    return normalise(p, padded, true);
}

Prepared prepare_glm(const PrepParams& p, const Rgb8& img)
{
    const int64_t factor = p.patch * p.merge, height = img.height, width = img.width;
    const int64_t min_px = int64_t(p.min_pixels);
    const int64_t max_px = p.max_tokens ? std::min<int64_t>(int64_t(p.max_pixels), int64_t(p.max_tokens) * factor * factor)
                                        : int64_t(p.max_pixels);
    if (min_px <= 0 || max_px <= 0)
        throw std::runtime_error("projector without a pixel range");
    const auto align = [factor](int64_t v) { return (v + factor - 1) / factor * factor; };
    int64_t ch = align(height), cw = align(width);
    if (ch * cw < min_px) {
        const double scale = std::sqrt(double(min_px) / double(height * width));
        ch = align(std::max<int64_t>(1, int64_t(std::ceil(height * scale))));
        cw = align(std::max<int64_t>(1, int64_t(std::ceil(width * scale))));
    }
    if (ch * cw > max_px) {
        // The largest content height whose aligned canvas fits.
        int64_t lo = 1, hi = height, best_h = factor, best_w = factor;
        while (lo <= hi) {
            const int64_t h = (lo + hi) / 2, w = std::max<int64_t>(1, width * h / height);
            if (align(h) * align(w) <= max_px) {
                best_h = align(h);
                best_w = align(w);
                lo = h + 1;
            } else {
                hi = h - 1;
            }
        }
        ch = best_h;
        cw = best_w;
    }
    // Fitted to the canvas, upscaled only below the minimum.
    double scale = std::min(double(ch) / height, double(cw) / width);
    if (height * width >= min_px)
        scale = std::min(1.0, scale);
    const auto content_h = uint32_t(std::max<int64_t>(1, std::min<int64_t>(ch, int64_t(std::floor(height * scale)))));
    const auto content_w = uint32_t(std::max<int64_t>(1, std::min<int64_t>(cw, int64_t(std::floor(width * scale)))));
    Rgb8 padded = canvas(uint32_t(cw), uint32_t(ch), 0);
    paste(padded, resize_bicubic(img, content_w, content_h), 0, 0);
    return normalise(p, padded, false);
}

} // namespace

Prepared prepare(const PrepParams& p, const Rgb8& img)
{
    if (img.width == 0 || img.height == 0)
        throw std::runtime_error("empty image");
    return p.kind == Projector::deepseek41v ? prepare_deepseek(p, img) : prepare_glm(p, img);
}

} // namespace mako::vision
