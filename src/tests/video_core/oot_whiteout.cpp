// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>
#include "shader_recompiler/storage_write_extent.h"
#include "video_core/buffer_cache/compute_write_grid.h"
#include "video_core/buffer_cache/indirect_buffer.h"
#include "video_core/oot_whiteout_pixels.h"

#ifdef WHITEOUT_STANDALONE
#include <iostream>
#define WHITEOUT_CHECK(condition)                                                                  \
    do {                                                                                           \
        if (!(condition))                                                                          \
            throw std::runtime_error(#condition);                                                  \
    } while (false)
#else
#include <catch2/catch_test_macros.hpp>
#define WHITEOUT_CHECK(condition) REQUIRE(condition)
#endif

namespace {
using namespace VideoCommon::OotWhiteout;
void PixelCases() {
    std::array<u8, 8> bytes{};
    auto stats = Inspect(std::span{bytes}.first(4), Encoding::Rgba8);
    WHITEOUT_CHECK(stats.pixels == 1);
    WHITEOUT_CHECK(stats.nonfinite == 0);
    WHITEOUT_CHECK(!stats.Suspicious(true));
    bytes.fill(255);
    stats = Inspect(std::span{bytes}.first(4), Encoding::Rgba8);
    WHITEOUT_CHECK(stats.white == 1);
    WHITEOUT_CHECK(stats.Suspicious(true));
    // B10G11R11: 1, 2, 4. Channel order and exponent bias matter for finite HDR.
    u32 packed = (15U << 6) | ((16U << 6) << 11) | ((17U << 5) << 22);
    std::memcpy(bytes.data(), &packed, 4);
    stats = Inspect(std::span{bytes}.first(4), Encoding::R11G11B10);
    WHITEOUT_CHECK(stats.maximum == 4);
    WHITEOUT_CHECK(stats.nonfinite == 0);
    WHITEOUT_CHECK(!stats.Suspicious(false));
    packed = 31U << 6; // +Inf
    std::memcpy(bytes.data(), &packed, 4);
    WHITEOUT_CHECK(Inspect(std::span{bytes}.first(4), Encoding::R11G11B10).nonfinite == 1);
    packed |= 1; // NaN
    std::memcpy(bytes.data(), &packed, 4);
    WHITEOUT_CHECK(Inspect(std::span{bytes}.first(4), Encoding::R11G11B10).nonfinite == 1);
    const std::array<u16, 4> half{0xbc00, 0x4000, 0x7bff, 0x7e00};
    std::memcpy(bytes.data(), half.data(), 8);
    stats = Inspect(bytes, Encoding::Rgba16);
    WHITEOUT_CHECK(stats.nonfinite == 0);
    WHITEOUT_CHECK(stats.maximum == 65504);
    WHITEOUT_CHECK(stats.bright == 1);
    // Alpha NaN is irrelevant; RGB NaN is detected.
    bytes[1] = 0x7e;
    WHITEOUT_CHECK(Inspect(bytes, Encoding::Rgba16).nonfinite == 1);
    WHITEOUT_CHECK(UnsignedFloat(1, 10) == std::ldexp(1.0f, -24));
    WHITEOUT_CHECK(UnsignedFloat(0, 6) == 0);
    WHITEOUT_CHECK(!PixelStats{}.Suspicious(false));
}

struct Dimension {
    u32 value;
    u32 Value() const { return value; }
};
struct Engine {
    struct {
        Dimension grid_dim_x, grid_dim_y, grid_dim_z;
    } launch_description;
    bool indirect{}, slot{};
    std::optional<GPUVAddr> GetIndirectComputeAddress() const {
        return indirect ? std::optional<GPUVAddr>{0x1000} : std::nullopt;
    }
    std::optional<u32> GetIndirectListSlot() const {
        return slot ? std::optional<u32>{1} : std::nullopt;
    }
};
void GridCases() {
    Engine engine{{{1}, {20}, {1}}, false, false};
    const Shader::StorageWriteExtent extent{
        .bounded = true, .const_end = 4, .group_stride = {0, 4, 0}};
    const std::array<u32, 3> block{1, 1, 1};
    auto grid = VideoCommon::ComputeWriteGrid(engine);
    WHITEOUT_CHECK(grid.has_value());
    WHITEOUT_CHECK((*grid)[1] == 20);
    WHITEOUT_CHECK(Shader::ResolveStorageWriteExtent(extent, block, grid) == u64{80});
    engine.indirect = true;
    WHITEOUT_CHECK(!VideoCommon::ComputeWriteGrid(engine));
    engine.indirect = false;
    engine.slot = true;
    // Slot 1 placeholders: grid y becomes 5 after the QMD's low-16-bit extraction, but the
    // real GPU-written y can be 20. Treating 5 as the grid under-marks this 80-byte output.
    engine.launch_description = {{0xf1a90004}, {5}, {6}};
    grid = VideoCommon::ComputeWriteGrid(engine);
    WHITEOUT_CHECK(!grid);
    WHITEOUT_CHECK(!Shader::ResolveStorageWriteExtent(extent, block, grid));
    auto local_extent = extent;
    local_extent.group_stride = {};
    WHITEOUT_CHECK(Shader::ResolveStorageWriteExtent(local_extent, block, grid) == u64{4});
}

// The cache's overwrite contract is deliberate: DiscardWrite forgets pending GPU contents.
// Exercise the production caller helper with that contract, including upload reordering usage.
struct Arguments {
    std::array<u8, 64> bytes{};
    bool used{};
    void MarkUsage(u32 offset, u32 size) {
        WHITEOUT_CHECK(offset == 16);
        WHITEOUT_CHECK(size == 12);
        used = true;
    }
};
struct Cache {
    Arguments buffer;
    bool gpu_dirty{true};
    std::pair<Arguments *, u32> ObtainBuffer(GPUVAddr address, u32 size,
                                             VideoCommon::ObtainBufferSynchronize sync,
                                             VideoCommon::ObtainBufferOperation operation) {
        WHITEOUT_CHECK(address == 0x1010);
        WHITEOUT_CHECK(size == 12);
        WHITEOUT_CHECK(sync == VideoCommon::ObtainBufferSynchronize::FullSynchronize);
        if (operation == VideoCommon::ObtainBufferOperation::DiscardWrite)
            gpu_dirty = false;
        return {&buffer, 16};
    }
    void UploadNeighbor() {
        for (u32 i = 0; i < buffer.bytes.size(); ++i)
            if (!gpu_dirty || i < 16 || i >= 28)
                buffer.bytes[i] = 0;
    }
};
void IndirectCases() {
    Cache cache;
    cache.buffer.bytes.fill(0x73);
    const auto [buffer, offset] = VideoCommon::ObtainIndirectBuffer(cache, 0x1010);
    WHITEOUT_CHECK(cache.gpu_dirty);
    WHITEOUT_CHECK(buffer->used);
    cache.UploadNeighbor();
    WHITEOUT_CHECK(buffer->bytes[offset] == 0x73);
    WHITEOUT_CHECK(buffer->bytes[offset + 11] == 0x73);
    WHITEOUT_CHECK(buffer->bytes[offset + 12] == 0);
}
} // namespace

#ifdef WHITEOUT_STANDALONE
int main() {
    unsigned failures = 0;
    const auto run = [&](const char *name, auto function) {
        try {
            function();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception &error) {
            ++failures;
            std::cout << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    run("pixel-decoding", PixelCases);
    run("indirect-grid-write-bounds", GridCases);
    run("indirect-argument-retention", IndirectCases);
    return failures ? 1 : 0;
}
#else
TEST_CASE("OoT whiteout samples distinguish finite HDR and nonfinite RGB", "[whiteout]") {
    PixelCases();
}
TEST_CASE("List-slot group counts cannot narrow GPU write tracking", "[whiteout]") { GridCases(); }
TEST_CASE("Indirect arguments retain GPU writes and mark their read usage", "[whiteout]") {
    IndirectCases();
}
#endif
