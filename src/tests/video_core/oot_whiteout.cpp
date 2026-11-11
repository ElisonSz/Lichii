// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>
#include "shader_recompiler/storage_write_extent.h"
#include "video_core/buffer_cache/compute_write_grid.h"
#include "video_core/buffer_cache/indirect_buffer.h"

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
    run("indirect-grid-write-bounds", GridCases);
    run("indirect-argument-retention", IndirectCases);
    return failures ? 1 : 0;
}
#else
TEST_CASE("List-slot group counts cannot narrow GPU write tracking", "[whiteout]") { GridCases(); }
TEST_CASE("Indirect arguments retain GPU writes and mark their read usage", "[whiteout]") {
    IndirectCases();
}
#endif
