#include "vision/image.hpp"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace mako::vision {

namespace {

struct CfRelease {
    void operator()(CFTypeRef r) const
    {
        if (r)
            CFRelease(r);
    }
};
template <class T>
using Cf = std::unique_ptr<std::remove_pointer_t<T>, CfRelease>;

// The stored samples when the layout is one of the common ones (8- or 16-bit integer, RGB or grey with optional
// alpha, or an 8-bit palette); false otherwise.
bool copy_samples(CGImageRef img, Rgb8& out)
{
    const size_t bpc = CGImageGetBitsPerComponent(img), bpp = CGImageGetBitsPerPixel(img);
    const CGBitmapInfo info = CGImageGetBitmapInfo(img);
    const CGImageAlphaInfo alpha = CGImageGetAlphaInfo(img);
    CGColorSpaceRef cs = CGImageGetColorSpace(img);
    if (!cs || (info & kCGBitmapFloatComponents) || (bpc != 8 && bpc != 16))
        return false;
    // Premultiplied samples are not the stored ones.
    if (alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaPremultipliedLast)
        return false;
    const CGColorSpaceModel model = CGColorSpaceGetModel(cs);
    size_t colors = 0;
    std::vector<uint8_t> palette;
    switch (model) {
    case kCGColorSpaceModelRGB: colors = 3; break;
    case kCGColorSpaceModelMonochrome: colors = 1; break;
    case kCGColorSpaceModelIndexed: {
        CGColorSpaceRef base = CGColorSpaceGetBaseColorSpace(cs);
        if (bpc != 8 || !base || CGColorSpaceGetModel(base) != kCGColorSpaceModelRGB)
            return false;
        const size_t n = CGColorSpaceGetColorTableCount(cs);
        palette.resize(n * 3);
        CGColorSpaceGetColorTable(cs, palette.data());
        colors = 1;
        break;
    }
    default: return false;
    }
    const bool has_alpha = alpha != kCGImageAlphaNone;
    const size_t channels = colors + (has_alpha ? 1 : 0);
    if (bpp != channels * bpc)
        return false;
    // Where the colour samples sit in a pixel: after a leading alpha or skipped channel, else first.
    const bool alpha_first = alpha == kCGImageAlphaFirst || alpha == kCGImageAlphaNoneSkipFirst;
    const size_t first = alpha_first ? 1 : 0;
    const uint32_t order = info & kCGBitmapByteOrderMask;
    const bool little16 = bpc == 16 && order == kCGBitmapByteOrder16Little;
    const bool little32 = bpc == 8 && bpp == 32 && order == kCGBitmapByteOrder32Little;
    if ((bpc == 16 && order != kCGBitmapByteOrderDefault && order != kCGBitmapByteOrder16Big && !little16) ||
        (bpc == 8 && order != kCGBitmapByteOrderDefault && order != kCGBitmapByteOrder32Big && !little32))
        return false;

    Cf<CFDataRef> data(CGDataProviderCopyData(CGImageGetDataProvider(img)));
    if (!data)
        return false;
    const auto* bytes = CFDataGetBytePtr(data.get());
    const size_t stride = CGImageGetBytesPerRow(img), w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    if (size_t(CFDataGetLength(data.get())) < stride * (h - 1) + w * bpp / 8)
        return false;
    // Sample c of a pixel, high byte for 16 bits.
    const auto sample = [&](const uint8_t* p, size_t c) -> uint8_t {
        if (bpc == 16)
            return little16 ? p[2 * c + 1] : p[2 * c];
        return little32 ? p[3 - c] : p[c];
    };
    out.width = uint32_t(w);
    out.height = uint32_t(h);
    out.px.resize(w * h * 3);
    for (size_t y = 0; y < h; ++y) {
        const uint8_t* row = bytes + y * stride;
        uint8_t* dst = out.px.data() + y * w * 3;
        for (size_t x = 0; x < w; ++x) {
            const uint8_t* p = row + x * (bpp / 8);
            if (!palette.empty()) {
                const size_t i = std::min<size_t>(sample(p, first), palette.size() / 3 - 1);
                std::copy_n(palette.data() + i * 3, 3, dst + x * 3);
            } else if (colors == 1) {
                std::fill_n(dst + x * 3, 3, sample(p, first));
            } else {
                for (size_t c = 0; c < 3; ++c)
                    dst[x * 3 + c] = sample(p, first + c);
            }
        }
    }
    return true;
}

// Any other layout: drawn into an 8-bit device RGB context.
void draw_rgb(CGImageRef img, Rgb8& out)
{
    const size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    Cf<CGColorSpaceRef> rgb(CGColorSpaceCreateDeviceRGB());
    std::vector<uint8_t> rgbx(w * h * 4);
    Cf<CGContextRef> ctx(CGBitmapContextCreate(rgbx.data(), w, h, 8, w * 4, rgb.get(),
                                               uint32_t(kCGImageAlphaNoneSkipLast) | uint32_t(kCGBitmapByteOrder32Big)));
    if (!ctx)
        throw std::runtime_error("cannot convert the image to RGB");
    CGContextDrawImage(ctx.get(), CGRectMake(0, 0, double(w), double(h)), img);
    out.width = uint32_t(w);
    out.height = uint32_t(h);
    out.px.resize(w * h * 3);
    for (size_t i = 0; i < w * h; ++i)
        std::copy_n(rgbx.data() + i * 4, 3, out.px.data() + i * 3);
}

} // namespace

Rgb8 decode_image(std::span<const std::byte> data)
{
    Cf<CFDataRef> cf(CFDataCreateWithBytesNoCopy(nullptr, reinterpret_cast<const UInt8*>(data.data()),
                                                 CFIndex(data.size()), kCFAllocatorNull));
    Cf<CGImageSourceRef> src(cf ? CGImageSourceCreateWithData(cf.get(), nullptr) : nullptr);
    if (!src || CGImageSourceGetCount(src.get()) < 1)
        throw std::runtime_error("not an image ImageIO can read");
    Cf<CGImageRef> img(CGImageSourceCreateImageAtIndex(src.get(), 0, nullptr));
    if (!img)
        throw std::runtime_error("cannot decode the image");
    if (CGImageGetWidth(img.get()) == 0 || CGImageGetHeight(img.get()) == 0)
        throw std::runtime_error("empty image");
    Rgb8 out;
    if (!copy_samples(img.get(), out))
        draw_rgb(img.get(), out);
    return out;
}

namespace {

constexpr int kPrecision = 22; // fractional bits of the fixed-point weights

double bicubic(double x)
{
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

// For each output index: the first input index, the input count, and ksize fixed-point weights.
struct Coefficients {
    int ksize = 0;
    std::vector<int> first, count;
    std::vector<int32_t> weights;
};

Coefficients coefficients(int in, int out)
{
    const double scale = double(in) / out, filterscale = std::max(scale, 1.0), support = 2.0 * filterscale;
    Coefficients c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.first.resize(out);
    c.count.resize(out);
    c.weights.assign(size_t(out) * c.ksize, 0);
    std::vector<double> w(c.ksize);
    for (int o = 0; o < out; ++o) {
        const double center = (o + 0.5) * scale, ss = 1.0 / filterscale;
        const int lo = std::max(0, int(center - support + 0.5));
        const int n = std::min(in, int(center + support + 0.5)) - lo;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            w[i] = bicubic((i + lo - center + 0.5) * ss);
            sum += w[i];
        }
        for (int i = 0; i < n; ++i) {
            const double v = sum != 0.0 ? w[i] / sum : w[i];
            // +/- 0.5, then truncation toward zero
            c.weights[size_t(o) * c.ksize + i] = int32_t(v * double(1 << kPrecision) + (v < 0 ? -0.5 : 0.5));
        }
        c.first[o] = lo;
        c.count[o] = n;
    }
    return c;
}

uint8_t clip8(int32_t v) { return uint8_t(std::clamp(v, 0, 255)); }

Rgb8 horizontal(const Rgb8& in, uint32_t width)
{
    const Coefficients c = coefficients(int(in.width), int(width));
    Rgb8 out{width, in.height, std::vector<uint8_t>(size_t(width) * in.height * 3)};
    for (uint32_t y = 0; y < in.height; ++y) {
        const uint8_t* row = in.px.data() + size_t(y) * in.width * 3;
        uint8_t* dst = out.px.data() + size_t(y) * width * 3;
        for (uint32_t x = 0; x < width; ++x) {
            const int32_t* k = c.weights.data() + size_t(x) * c.ksize;
            const uint8_t* p = row + size_t(c.first[x]) * 3;
            int32_t s0 = 1 << (kPrecision - 1), s1 = s0, s2 = s0;
            for (int i = 0; i < c.count[x]; ++i, p += 3) {
                s0 += p[0] * k[i];
                s1 += p[1] * k[i];
                s2 += p[2] * k[i];
            }
            dst[x * 3] = clip8(s0 >> kPrecision);
            dst[x * 3 + 1] = clip8(s1 >> kPrecision);
            dst[x * 3 + 2] = clip8(s2 >> kPrecision);
        }
    }
    return out;
}

Rgb8 vertical(const Rgb8& in, uint32_t height)
{
    const Coefficients c = coefficients(int(in.height), int(height));
    const size_t row = size_t(in.width) * 3;
    Rgb8 out{in.width, height, std::vector<uint8_t>(row * height)};
    std::vector<int32_t> acc(row);
    for (uint32_t y = 0; y < height; ++y) {
        const int32_t* k = c.weights.data() + size_t(y) * c.ksize;
        std::fill(acc.begin(), acc.end(), 1 << (kPrecision - 1));
        for (int i = 0; i < c.count[y]; ++i) {
            const uint8_t* src = in.px.data() + size_t(c.first[y] + i) * row;
            for (size_t j = 0; j < row; ++j)
                acc[j] += src[j] * k[i];
        }
        uint8_t* dst = out.px.data() + size_t(y) * row;
        for (size_t j = 0; j < row; ++j)
            dst[j] = clip8(acc[j] >> kPrecision);
    }
    return out;
}

} // namespace

Rgb8 resize_bicubic(const Rgb8& in, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0 || width > 65536 || height > 65536)
        throw std::runtime_error("resize target out of range");
    if (width == in.width && height == in.height)
        return in;
    if (width == in.width)
        return vertical(in, height);
    const Rgb8 h = horizontal(in, width);
    return height == in.height ? h : vertical(h, height);
}

} // namespace mako::vision
