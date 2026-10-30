#include "image/png.hpp"

#include <ImageIO/ImageIO.h>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mako::image {

namespace {

template <typename T> struct Cf {
    T ref = nullptr;
    explicit Cf(T r) : ref(r) {}
    ~Cf()
    {
        if (ref)
            CFRelease(ref);
    }
    Cf(const Cf&) = delete;
    Cf& operator=(const Cf&) = delete;
};

} // namespace

std::vector<uint8_t> to_rgba8(std::span<const float> rgba)
{
    std::vector<uint8_t> out(rgba.size());
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float v = std::fmin(std::fmax(rgba[i] * 0.5f + 0.5f, 0.0f), 1.0f);
        out[i] = uint8_t(std::nearbyint(v * 255.0f));
    }
    return out;
}

std::vector<uint8_t> encode_png(std::span<const uint8_t> rgba8, uint32_t width, uint32_t height, bool rgb)
{
    if (rgba8.size() != size_t(width) * height * 4)
        throw std::runtime_error("encode_png: not width x height RGBA");
    std::vector<uint8_t> px(rgba8.begin(), rgba8.end());
    if (rgb)
        for (size_t i = 3; i < px.size(); i += 4)
            px[i] = 255;
    Cf<CGColorSpaceRef> cs(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    Cf<CFDataRef> data(CFDataCreate(nullptr, px.data(), CFIndex(px.size())));
    Cf<CGDataProviderRef> prov(CGDataProviderCreateWithCFData(data.ref));
    const CGBitmapInfo info = rgb ? CGBitmapInfo(kCGImageAlphaNoneSkipLast) : CGBitmapInfo(kCGImageAlphaLast);
    Cf<CGImageRef> img(CGImageCreate(width, height, 8, 32, size_t(width) * 4, cs.ref, info, prov.ref, nullptr, false,
                                     kCGRenderingIntentDefault));
    Cf<CFMutableDataRef> out(CFDataCreateMutable(nullptr, 0));
    Cf<CGImageDestinationRef> dst(CGImageDestinationCreateWithData(out.ref, CFSTR("public.png"), 1, nullptr));
    if (!img.ref || !dst.ref)
        throw std::runtime_error("encode_png: ImageIO could not set up the image");
    CGImageDestinationAddImage(dst.ref, img.ref, nullptr);
    if (!CGImageDestinationFinalize(dst.ref))
        throw std::runtime_error("encode_png: ImageIO could not write the PNG");
    const auto* p = CFDataGetBytePtr(out.ref);
    return std::vector<uint8_t>(p, p + CFDataGetLength(out.ref));
}

void write_file(const std::string& path, std::span<const uint8_t> data)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    if (!f)
        throw std::runtime_error("cannot write " + path);
}

} // namespace mako::image
