// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_join.h"

namespace {
using namespace VideoCommon;
using ImageJoin::Relation;
using VideoCore::Surface::PixelFormat;

ImageInfo MakeInfo(u32 width, u32 height, s32 levels, s32 layers, u32 block_height,
                   ImageType type = ImageType::e2D) {
    ImageInfo info;
    info.type = type;
    info.format = PixelFormat::A8B8G8R8_UNORM;
    info.size = {width, height, type == ImageType::e3D ? 8U : 1U};
    info.resources = {levels, layers};
    info.block = {0, block_height, 0};
    info.native_tiling = true;
    info.layer_stride = CalculateLayerStride(info);
    info.maybe_unaligned_layer_stride = info.layer_stride;
    return info;
}

Relation Original(const ImageInfo& info, GPUVAddr gpu_address, DAddr cpu_address,
                  const ImageBase& overlap) {
    constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
    const ImageBase candidate(info, gpu_address, cpu_address);
    if (IsSubresource(info, overlap, gpu_address, options, false, true)) {
        return Relation::LeftAlias;
    }
    if (IsSubresource(overlap.info, candidate, overlap.gpu_addr, options, false, true)) {
        return Relation::RightAlias;
    }
    return Relation::BadOverlap;
}
} // namespace

TEST_CASE("Join metadata follows resolves which change resources and addresses",
          "[video_core][texture_cache]") {
    ImageJoin::Metadata metadata;
    auto info = MakeInfo(128, 128, 1, 1, 2);
    constexpr GPUVAddr gpu_base = 0x100000;
    constexpr DAddr cpu_base = 0x200000;
    const auto& first = metadata.Get(info, gpu_base, cpu_base);
    REQUIRE(first.info.resources.layers == 1);
    REQUIRE(&first == &metadata.Get(info, gpu_base, cpu_base));
    info.resources = {4, 3};
    metadata.Invalidate();
    const auto& expanded = metadata.Get(info, gpu_base - 4096, cpu_base - 4096);
    REQUIRE(expanded.info.resources == info.resources);
    REQUIRE(expanded.gpu_addr == gpu_base - 4096);
    REQUIRE(expanded.cpu_addr == cpu_base - 4096);
    const ImageBase expected(info, gpu_base - 4096, cpu_base - 4096);
    REQUIRE(expanded.guest_size_bytes == expected.guest_size_bytes);
    REQUIRE(expanded.mip_level_offsets == expected.mip_level_offsets);
}

TEST_CASE("Join metadata preserves ordered overlap decisions across expanding joins",
          "[video_core][texture_cache]") {
    // Use the real ResolveOverlap and IsSubresource implementations. A successful resolve can
    // move the base left and grow layers or levels between failed resolves in the same scan.
    std::mt19937 random{0x1A6E2026};
    std::array<u64, 3> relations{};
    u64 resolves{};
    for (u32 scan = 0; scan < 600; ++scan) {
        const ImageType type = scan % 3 == 0 ? ImageType::e3D : ImageType::e2D;
        auto info = MakeInfo(64U << (scan % 3), 64U << (scan % 2), 1, 1, scan % 4, type);
        GPUVAddr gpu_address = 0x400000;
        DAddr cpu_address = 0x800000;
        ImageJoin::Metadata metadata;
        for (u32 index = 0; index < 32; ++index) {
            auto other_info = info;
            const u32 choice = static_cast<u32>(random() % 8);
            if (choice == 0) {
                other_info.resources.levels = 4;
            } else if (choice == 1 && type != ImageType::e3D) {
                other_info.resources.layers += 2;
            } else if (choice == 2) {
                other_info.format = PixelFormat::D32_FLOAT;
            } else if (choice == 3) {
                other_info.size.width *= 2;
            } else if (choice == 4) {
                other_info.block.height = (other_info.block.height + 1) % 5;
            }
            const u64 offset = choice >= 5 ? u64{1} << (choice + 4) : 0;
            const bool before = index % 2 == 0;
            const GPUVAddr other_gpu = before ? gpu_address - offset : gpu_address + offset;
            const DAddr other_cpu = before ? cpu_address - offset : cpu_address + offset;
            const ImageBase overlap(other_info, other_gpu, other_cpu);
            const auto solution =
                ResolveOverlap(info, gpu_address, cpu_address, overlap, true, false, true);
            if (solution) {
                gpu_address = solution->gpu_addr;
                cpu_address = solution->cpu_addr;
                info.resources = solution->resources;
                metadata.Invalidate();
                ++resolves;
                continue;
            }
            const auto actual =
                ImageJoin::Classify(metadata, info, gpu_address, cpu_address, overlap, false, true);
            REQUIRE(actual == Original(info, gpu_address, cpu_address, overlap));
            REQUIRE(ImageJoin::Verify(actual, info, gpu_address, cpu_address, overlap, false,
                                      true) == actual);
            ++relations[static_cast<size_t>(actual)];
        }
    }
    REQUIRE(resolves > 1000);
    REQUIRE(relations[static_cast<size_t>(Relation::BadOverlap)] > 1000);
    REQUIRE(ImageJoin::counters.mismatches == 0);
}

TEST_CASE("Join metadata handles both alias directions after a failed strict resolve",
          "[video_core][texture_cache]") {
    auto small = MakeInfo(120, 128, 1, 1, 2);
    auto large = MakeInfo(128, 128, 1, 2, 2);
    // Equal aligned extents and compatible strides can alias despite different raw widths.
    small.layer_stride = large.layer_stride;
    small.maybe_unaligned_layer_stride = large.maybe_unaligned_layer_stride;
    const ImageBase smaller(small, 0x100000, 0x200000);
    const ImageBase larger(large, 0x100000, 0x200000);
    ImageJoin::Metadata metadata;
    REQUIRE(ImageJoin::Classify(metadata, small, 0x100000, 0x200000, larger, false, true) ==
            Relation::LeftAlias);
    metadata.Invalidate();
    REQUIRE(ImageJoin::Classify(metadata, large, 0x100000, 0x200000, smaller, false, true) ==
            Relation::RightAlias);
}

TEST_CASE("Non-sparse creation skips only a discarded continuity query",
          "[video_core][texture_cache]") {
    for (const bool requested : {false, true}) {
        for (const bool continuous : {false, true}) {
            for (const bool enabled : {false, true}) {
                u32 queries{};
                const bool result = ImageJoin::IsSparse(requested, enabled, [&] {
                    ++queries;
                    return continuous;
                });
                REQUIRE(result == (requested && !continuous));
                REQUIRE(queries == (enabled && !requested ? 0U : 1U));
            }
        }
    }
}

TEST_CASE("Retirement diagnostics distinguish same address from same storage",
          "[video_core][texture_cache]") {
    const auto info = MakeInfo(128, 128, 1, 1, 2);
    ImageBase retired(info, 0x100000, 0x200000);
    ImageJoin::RetirementHistory history;
    history.Retire(retired, 100);
    REQUIRE_FALSE(history.Take(0x300000));
    const auto previous = history.Take(retired.gpu_addr);
    REQUIRE(previous);
    REQUIRE(previous->frame == 100);
    REQUIRE(ImageJoin::RetirementHistory::SameShape(*previous, info, retired.cpu_addr));
    REQUIRE_FALSE(history.Take(retired.gpu_addr));
    auto changed = info;
    changed.size.width += 1;
    REQUIRE_FALSE(ImageJoin::RetirementHistory::SameShape(*previous, changed, retired.cpu_addr));
    changed = info;
    changed.resources.layers += 1;
    REQUIRE_FALSE(ImageJoin::RetirementHistory::SameShape(*previous, changed, retired.cpu_addr));
    REQUIRE_FALSE(
        ImageJoin::RetirementHistory::SameShape(*previous, info, retired.cpu_addr + 4096));
    for (size_t index = 0; index <= ImageJoin::RetirementHistory::Capacity; ++index) {
        retired.gpu_addr = 0x100000 + index * 4096;
        history.Retire(retired, index);
    }
    REQUIRE_FALSE(history.Take(0x100000));
    REQUIRE(history.Take(retired.gpu_addr));
}
