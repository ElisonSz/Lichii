// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include "common/common_types.h"

namespace VideoCommon::OotWhiteout {
inline float UnsignedFloat(u32 bits, u32 mantissa_bits) {
    const u32 exponent = bits >> mantissa_bits;
    const u32 mantissa = bits & ((1U << mantissa_bits) - 1);
    if (exponent == 31)
        return mantissa ? std::numeric_limits<float>::quiet_NaN()
                        : std::numeric_limits<float>::infinity();
    return exponent == 0
               ? std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissa_bits))
               : std::ldexp(static_cast<float>(mantissa | (1U << mantissa_bits)),
                            static_cast<int>(exponent) - 15 - static_cast<int>(mantissa_bits));
}
enum class Encoding { Rgba8, R11G11B10, Rgba16 };
struct PixelStats {
    u32 pixels{}, nonfinite{}, white{}, bright{};
    float maximum{};
    bool Suspicious(bool tonemap) const {
        return pixels != 0 && (nonfinite != 0 || (tonemap ? white * 100 >= pixels * 35
                                                          : bright * 100 >= pixels * 10));
    }
};
inline PixelStats Inspect(std::span<const u8> bytes, Encoding encoding) {
    PixelStats result;
    const size_t stride = encoding == Encoding::Rgba16 ? 8 : 4;
    for (size_t offset = 0; offset + stride <= bytes.size(); offset += stride) {
        std::array<float, 3> rgb{};
        if (encoding == Encoding::Rgba8) {
            for (size_t i = 0; i < 3; ++i)
                rgb[i] = static_cast<float>(bytes[offset + i]) / 255.0f;
        } else if (encoding == Encoding::R11G11B10) {
            u32 word;
            std::memcpy(&word, bytes.data() + offset, sizeof(word));
            rgb = {UnsignedFloat(word & 0x7ff, 6), UnsignedFloat((word >> 11) & 0x7ff, 6),
                   UnsignedFloat(word >> 22, 5)};
        } else {
            for (size_t i = 0; i < 3; ++i) {
                u16 word;
                std::memcpy(&word, bytes.data() + offset + i * 2, sizeof(word));
                rgb[i] = UnsignedFloat(word & 0x7fff, 10) * (word & 0x8000 ? -1.0f : 1.0f);
            }
        }
        ++result.pixels;
        bool finite = true;
        bool white = true;
        bool bright = false;
        for (const float channel : rgb) {
            finite &= std::isfinite(channel);
            white &= channel >= 0.97f;
            bright |= channel >= 64.0f;
            if (std::isfinite(channel) && channel > result.maximum)
                result.maximum = channel;
        }
        result.nonfinite += !finite;
        result.white += white && finite;
        result.bright += bright && finite;
    }
    return result;
}
} // namespace VideoCommon::OotWhiteout
