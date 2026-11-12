// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/texture_cache/image_info.h"

namespace VideoCommon::StaleTargets {

// Incompatible targets are disposable only with all four overwrite conditions proved.
inline bool enabled = true;
inline bool verify = false;
/// Frames a bad-overlap render target stays unused before RetireStaleTargets drops it: the age
/// at which the collector deletes one at its critical threshold. 0 turns the pass off.
inline u64 retire_age = 10;
/// The Vulkan runtime keeps the host images of deleted cache images for the next image of their
/// shape (HostImagePool). EDEN_NX2_IMAGE_POOL=0 turns that off.
inline bool pool_images = true;
/// Large single-level images created for render-target binding use the target pool's blocks
/// (Vulkan::Device::SeparatePool). EDEN_NX2_TARGET_POOL=0 turns it off.
inline bool target_memory_pool = true;
inline void ReadSwitch() {
    const char* const keep = std::getenv("EDEN_NX2_KEEP_STALE_TARGETS");
    enabled = keep == nullptr || keep[0] != '1';
    const char* const check = std::getenv("EDEN_NX2_STALE_TARGET_VERIFY");
    verify = check != nullptr && check[0] == '1';
    const char* const age = std::getenv("EDEN_NX2_STALE_TARGET_AGE");
    retire_age = age != nullptr ? std::strtoull(age, nullptr, 10) : 10;
    const char* const pool = std::getenv("EDEN_NX2_IMAGE_POOL");
    pool_images = pool == nullptr || pool[0] != '0';
    const char* const target_pool = std::getenv("EDEN_NX2_TARGET_POOL");
    target_memory_pool = target_pool == nullptr || target_pool[0] != '0';
}

/// Why an older target a whole clear overlapped stays, for the memory trace.
enum class Kept : size_t {
    Origin,     ///< Not made as a render target.
    History,    ///< Has held guest data or been a copy destination.
    Flags,      ///< CPU-modified, sparse, rescaled, remapped, multisampled or not GPU-written.
    Marker,     ///< Doesn't cover whole pages, so its stale marker would be incomplete.
    Referenced, ///< Aliased, bound, or named by pending work.
    Layout,     ///< A layout the byte proof doesn't handle.
    Partial,    ///< Not every live byte overwritten yet.
    Count,
};
inline constexpr size_t NumKept = static_cast<size_t>(Kept::Count);

struct Run {
    u64 begin;
    u64 end;
};

struct ClearCoverage {
    s32 x{};
    s32 y{};
    u32 width{};
    u32 height{};
    u32 base_layer{};
    u32 layers{};
    bool all_channels_aspects{};
    bool rescaled{};

    [[nodiscard]] bool Whole(u32 image_width, u32 image_height) const noexcept {
        return x == 0 && y == 0 && width >= image_width && height >= image_height &&
               base_layer == 0 && layers == 1 && all_channels_aspects && !rescaled;
    }
};

// Only raw == effective block dimensions are accepted. Allocation extent is not
// evidence of which guest bytes a clear writes. ImageLayout checks production math.
struct Layout {
    u64 row_bytes{};
    u32 height{};
    u32 block_height{};

    [[nodiscard]] bool WholeGobs() const noexcept {
        return row_bytes % 64 == 0 && height % 8 == 0;
    }

    [[nodiscard]] std::vector<Run> Runs(u64 base) const {
        std::vector<Run> result;
        const u64 columns = (row_bytes + 63) / 64;
        const u64 full_columns = row_bytes / 64;
        const u64 rows = height / 8;
        const u64 block_gobs = u64{1} << block_height;
        for (u64 band = 0; band * block_gobs < rows; ++band) {
            const u64 count = (std::min)(block_gobs, rows - band * block_gobs);
            for (u64 column = 0; column < full_columns; ++column) {
                const u64 begin = base + (band * columns + column) * block_gobs * 512;
                const u64 end = begin + count * 512;
                if (!result.empty() && result.back().end == begin) {
                    result.back().end = end;
                } else {
                    result.push_back({begin, end});
                }
            }
        }
        return result;
    }
};

[[nodiscard]] std::optional<Layout> ImageLayout(const ImageInfo& info, u64 guest_bytes);

// Independent verification uses production swizzle, not Layout::Runs address math.
// Verification is bounded and treats an unsupported/oversize footprint as unproven.
[[nodiscard]] bool VerifyFootprint(const ImageInfo& info, u64 guest_bytes,
                                   std::span<const Run> runs);

/// What VerifyFootprint's answer depends on, for an image ImageLayout accepted (one level, layer
/// and sample, no sparse or spaced tiling, raw block size equal to the effective one).
struct Shape {
    u32 format;
    u32 width;
    u32 height;
    u32 block_height;
    u32 native_tiling;
    u64 guest_bytes;

    [[nodiscard]] bool operator==(const Shape&) const noexcept = default;
};

struct ShapeHash {
    [[nodiscard]] size_t operator()(const Shape& shape) const noexcept {
        u64 hash = shape.guest_bytes * 0x9E3779B97F4A7C15ULL;
        for (const u64 part : {u64{shape.format}, u64{shape.width}, u64{shape.height},
                               u64{shape.block_height}, u64{shape.native_tiling}}) {
            hash = (hash ^ part) * 0x100000001B3ULL;
        }
        return static_cast<size_t>(hash ^ (hash >> 32));
    }
};

inline std::mutex verified_shapes_mutex;
inline std::unordered_map<Shape, bool, ShapeHash> verified_shapes;

/// VerifyFootprint swizzles and scans the whole image (15 MB for a 2560x1440 target), and the
/// game makes the same few shapes at new addresses hundreds of times a second: keep the answer
/// per shape. `relative` must be the runs ImageLayout gives this shape from address zero.
[[nodiscard]] inline bool VerifyShape(const ImageInfo& info, u64 guest_bytes,
                                      std::span<const Run> relative) {
    const Shape shape{static_cast<u32>(info.format), info.size.width,
                      info.size.height,              info.block.height,
                      info.native_tiling ? 1U : 0U,  guest_bytes};
    {
        const std::scoped_lock lock{verified_shapes_mutex};
        if (const auto found = verified_shapes.find(shape); found != verified_shapes.end()) {
            return found->second;
        }
    }
    const bool valid = VerifyFootprint(info, guest_bytes, relative);
    const std::scoped_lock lock{verified_shapes_mutex};
    if (verified_shapes.size() >= 4096) {
        verified_shapes.clear();
    }
    verified_shapes.insert_or_assign(shape, valid);
    return valid;
}

inline u64 ElapsedNs(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

struct Evidence {
    ImageInfo info;
    u64 bytes;
    u64 address;
    u64 birth;
};

struct Proof {
    u64 birth{};
    u64 use_epoch{};
    u64 overwrite_epoch{};
    u64 transfer_tick{};
    bool pending_transfer{};
    bool render_target_origin{};
    bool forbidden_history{};
    bool contents_defined{};
    bool started{};
    bool footprint_checked{};
    bool footprint_valid{};
    std::vector<Run> footprint;
    std::vector<Run> missing;
    std::vector<Evidence> evidence;

    void Use() noexcept {
        ++use_epoch;
        started = false;
        missing.clear();
        evidence.clear();
    }
    void Forbid() noexcept {
        forbidden_history = true;
        Use();
    }
    [[nodiscard]] bool NoUseSinceOverwrite() const noexcept {
        return started && overwrite_epoch == use_epoch;
    }
    [[nodiscard]] bool EveryLiveByteOverwritten() const noexcept {
        return started && missing.empty();
    }
    void Cover(std::span<const Run> target, std::span<const Run> writer, Evidence source) {
        if (!started) {
            missing.assign(target.begin(), target.end());
            overwrite_epoch = use_epoch;
            started = true;
        }
        std::vector<Run> remainder;
        u64 removed{};
        size_t cursor{};
        for (const Run live : missing) {
            u64 position = live.begin;
            while (cursor < writer.size() && writer[cursor].end <= position) {
                ++cursor;
            }
            for (size_t index = cursor; index < writer.size() && writer[index].begin < live.end;
                 ++index) {
                const Run clear = writer[index];
                if (clear.begin > position) {
                    remainder.push_back({position, clear.begin});
                }
                const u64 end = (std::min)(live.end, clear.end);
                if (end > position) {
                    removed += end - (std::max)(position, clear.begin);
                    position = end;
                }
            }
            if (position < live.end) {
                remainder.push_back({position, live.end});
            }
        }
        missing = std::move(remainder);
        if (removed != 0) {
            evidence.push_back(std::move(source));
        }
    }
};

// Rebuilds the target and every contributing writer using swizzle byte masks,
// independently of Runs/Cover. Called for each proposed drop in verify mode.
[[nodiscard]] bool VerifyDrop(const ImageInfo& info, u64 bytes, u64 address, const Proof& proof);

} // namespace VideoCommon::StaleTargets
