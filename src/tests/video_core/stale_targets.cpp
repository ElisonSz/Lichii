// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/container/unordered_set.h"
#include "common/slot_vector.h"
#include "video_core/nx2.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/cache_operations.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/stale_pages.h"
#include "video_core/texture_cache/stale_targets.h"
#include "video_core/texture_cache/util.h"
#include "video_core/textures/decoders.h"

namespace {
using namespace VideoCommon;
using namespace VideoCommon::StaleTargets;
using VideoCore::Surface::PixelFormat;

constexpr GPUVAddr GpuBase = 0x100000;
constexpr DAddr CpuBase = 0x200000;
constexpr auto Options = RelaxedOptions::Size | RelaxedOptions::Format;

ImageInfo MakeInfo(u32 width, u32 height, u32 block_height,
                   PixelFormat format = PixelFormat::A8B8G8R8_UNORM, bool native = true) {
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = format;
    info.size = {width, height, 1};
    info.resources = {1, 1};
    info.block = {0, block_height, 0};
    info.native_tiling = native;
    info.layer_stride = CalculateLayerStride(info);
    info.maybe_unaligned_layer_stride = info.layer_stride;
    return info;
}

std::vector<u8> SwizzledPixels(const ImageInfo& info, u8 value) {
    const u32 bpp = VideoCore::Surface::BytesPerBlock(info.format);
    const size_t linear_size = size_t{info.size.width} * info.size.height * bpp;
    std::vector<u8> linear(linear_size, value);
    std::vector<u8> guest(CalculateGuestSizeInBytes(info));
    const auto effective = MipBlockSize(info, 0);
    Tegra::Texture::SwizzleTexture(guest, linear, bpp, info.size.width, info.size.height, 1,
                                   effective.height, effective.depth, 1, info.native_tiling);
    return guest;
}

// Restore the globals touched by these tests without disturbing other bitmap words.
struct Nx2State {
    Nx2State()
        : ampere{Tegra::g_nx2_ampere.exchange(true)}, audit_enabled{enabled}, audit_verify{verify},
          written_word{Tegra::g_nx2_gpu_written_pages[Word].exchange(0)} {
        enabled = true;
        verify = true;
    }
    ~Nx2State() {
        Tegra::g_nx2_gpu_written_pages[Word].store(written_word);
        Tegra::g_nx2_ampere.store(ampere);
        enabled = audit_enabled;
        verify = audit_verify;
    }
    static constexpr size_t Word = (CpuBase >> Tegra::Nx2WrittenPageBits) / 64;
    bool ampere;
    bool audit_enabled;
    bool audit_verify;
    u64 written_word;
};

struct PixelStaging {
    explicit PixelStaging(size_t bytes) : storage(bytes), mapped_span(storage) {}
    std::vector<u8> storage;
    std::span<u8> mapped_span;
};

struct PixelsImage : ImageBase {
    PixelsImage(const ImageInfo& image_info, GPUVAddr gpu_address, DAddr cpu_address, u8 sentinel,
                bool allocate_pixels = true)
        : ImageBase{image_info, gpu_address, cpu_address},
          pixels(allocate_pixels ? guest_size_bytes : 0, sentinel) {}
    template <class Copies>
    void UploadMemory(const PixelStaging& staging, const Copies&) {
        std::copy_n(staging.mapped_span.begin(), pixels.size(), pixels.begin());
    }
    std::vector<u8> pixels;
};

struct PixelRuntime {
    bool HasBrokenTextureViewFormats() const {
        return false;
    }
    bool HasNativeBgr() const {
        return true;
    }
    bool ClearImage(PixelsImage& image) {
        std::fill(image.pixels.begin(), image.pixels.end(), u8{0});
        ++clears;
        return true;
    }
    void TransitionImageLayout(PixelsImage&) {
        ++transitions;
    }
    PixelStaging UploadStagingBuffer(size_t bytes) {
        return PixelStaging{bytes};
    }
    void InsertUploadMemoryBarrier() {}
    u64 transitions{};
    u64 clears{};
};

struct AuditView {
    ImageId image_id;
    SubresourceRange range{};
    Extent3D size;
    PixelFormat format;
};

// Supply storage, a one-to-one mapping and a pixel backend to the production
// operations. Lookup, layout eligibility, auditing and stale-refresh decisions
// are the same functions called by TextureCache, not copies in this fixture.
struct CacheFixture {
    CacheFixture() {
        // Slot zero denotes a null descriptor in the real cache.
        [[maybe_unused]] const auto null_view = slot_image_views.insert(AuditView{});
    }

    std::optional<DAddr> ImageCpuAddress(GPUVAddr gpu_address, u64) const {
        if (!mapped || gpu_address < GpuBase) {
            return std::nullopt;
        }
        return CpuBase + gpu_address - GpuBase;
    }

    template <class Func>
    void ForEachImageInRegion(DAddr address, u64 bytes, Func&& func) {
        for (auto [id, image] : slot_images) {
            if (!image->Overlaps(address, static_cast<size_t>(bytes))) {
                continue;
            }
            if constexpr (std::is_same_v<std::invoke_result_t<Func, ImageId, PixelsImage&>, bool>) {
                if (func(id, *image)) {
                    break;
                }
            } else {
                func(id, *image);
            }
        }
    }

    ImageId Add(const ImageInfo& info, u8 sentinel, u64 offset = 0, bool render_target = true) {
        const ImageId id =
            slot_images.insert(info, GpuBase + offset, CpuBase + offset, sentinel, allocate_pixels);
        auto& image = slot_images[id];
        TextureCacheOperations::Created(*this, image, render_target);
        image.stale_target.contents_defined = true;
        image.flags = ImageFlagBits::GpuModified;
        return id;
    }

    void CollectStaleReferences() {}
    bool StaleTargetReferenced(ImageId id) const {
        return stale_references.contains(id);
    }
    void MarkStale(const ImageBase& image) {
        stale_pages.Mark(image.cpu_addr, image.guest_size_bytes, [](DAddr, u64) {});
    }
    void UntrackImage(ImageBase& image, ImageId) {
        image.flags &= ~ImageFlagBits::Tracked;
    }
    void UnregisterImage(ImageId id) {
        slot_images[id].flags &= ~ImageFlagBits::Registered;
    }
    void DeleteImage(ImageId id) {
        for (auto [other_id, image] : slot_images) {
            if (other_id != id) {
                std::erase(image->overlapping_images, id);
            }
        }
        slot_images.erase(id);
        ++retired;
    }

    ImageViewId View(ImageId id) {
        const auto& info = slot_images[id].info;
        return slot_image_views.insert(AuditView{id, {}, info.size, info.format});
    }

    PixelRuntime runtime;
    Common::SlotVector<PixelsImage> slot_images;
    Common::SlotVector<AuditView> slot_image_views;
    bool allocate_pixels{true};
    bool mapped{true};
    bool mem_trace_enabled{};
    u64 trace_lookups{};
    u64 trace_lookup_hits{};
    u64 trace_lookup_candidates{};
    u64 trace_lookup_remapped{};
    u64 trace_lookup_rejected{};
    u32 trace_rejection_lines{};
    const char* trace_origin = "test";
    u32 trace_descriptor{};
    u64 trace_stale_layout{};
    u64 trace_stale_contents{};
    u64 trace_stale_footprints{};
    u64 trace_stale_failures{};
    u64 trace_stale_verified{};
    u64 trace_stale_clears{};
    u64 trace_stale_clear_ns{};
    u64 trace_stale_reference_scans{};
    std::array<u64, StaleTargets::NumKept> trace_stale_kept{};
    u64 trace_stale_drops{};
    u64 trace_stale_drop_bytes{};
    u64 stale_birth{};
    Common::unordered_set<ImageId> stale_references;
    StalePages stale_pages;
    u64 retired{};
};

struct ReplayResult {
    size_t live{};
    u64 bytes{};
    u64 created{};
    u64 dropped{};
    u64 last_drops{};
    size_t peak{};
};

// The original address workload, now using the production clear/proof/drop operation.
// Pixels are unnecessary for this replay; independent footprint verification still runs.
ReplayResult Replay(u64 moving_frames) {
    CacheFixture cache;
    cache.allocate_pixels = false;
    ReplayResult result;
    constexpr u64 GuestBytes = 589824;
    const auto info = MakeInfo(384, 384, 4, PixelFormat::D32_FLOAT);
    for (u64 frame = 1; frame <= moving_frames; ++frame) {
        const u64 drops_before = cache.trace_stale_drops;
        if (frame % 7 == 0 && cache.slot_images.size() != 0) {
            for (auto [id, image] : cache.slot_images) {
                image->stale_target.Use();
                break;
            }
        }
        for (u64 index = 0; index < 50; ++index) {
            const u64 offset = ((frame % 2) * 50 + index) * GuestBytes + ((frame / 2) % 32) * 4096;
            const auto id = cache.Add(info, 0, offset);
            ++result.created;
            result.peak = (std::max)(result.peak, cache.slot_images.size());
            const auto view = cache.View(id);
            TextureCacheOperations::NotifyFullClear(cache, view);
            cache.slot_image_views.erase(view);
        }
        result.last_drops = cache.trace_stale_drops - drops_before;
    }
    result.live = cache.slot_images.size();
    result.bytes = static_cast<u64>(result.live) * GuestBytes;
    result.dropped = cache.trace_stale_drops;
    CHECK(cache.trace_stale_failures == 0);
    return result;
}
} // namespace

TEST_CASE("Stale target footprints use production swizzle and effective blocks",
          "[video_core][stale_targets]") {
    size_t accepted{};
    size_t rejected{};
    for (const u32 width : {16U, 32U, 48U, 65U}) {
        for (const u32 height : {8U, 16U, 24U, 40U, 43U, 256U}) {
            for (u32 block_height = 0; block_height <= 5; ++block_height) {
                for (const bool native : {false, true}) {
                    const auto info =
                        MakeInfo(width, height, block_height, PixelFormat::A8B8G8R8_UNORM, native);
                    const u64 bytes = CalculateGuestSizeInBytes(info);
                    const auto layout = ImageLayout(info, bytes);
                    if (MipBlockSize(info, 0) != info.block) {
                        CHECK_FALSE(layout.has_value());
                        ++rejected;
                        continue;
                    }
                    REQUIRE(layout.has_value());
                    ++accepted;
                    const auto runs = layout->Runs(0);
                    CHECK(VerifyFootprint(info, bytes, runs));
                    const auto pixels = SwizzledPixels(info, 0x5A);
                    std::vector<u8> credited(pixels.size());
                    for (const auto run : runs) {
                        REQUIRE(run.end <= credited.size());
                        std::fill(credited.begin() + static_cast<std::ptrdiff_t>(run.begin),
                                  credited.begin() + static_cast<std::ptrdiff_t>(run.end), 1);
                    }
                    bool matches = true;
                    for (size_t offset = 0; offset < pixels.size(); ++offset) {
                        if (credited[offset] != 0) {
                            matches &= pixels[offset] == 0x5A;
                        }
                        if (layout->WholeGobs()) {
                            matches &= (credited[offset] != 0) == (pixels[offset] == 0x5A);
                        }
                    }
                    CHECK(matches);
                }
            }
        }
    }
    CHECK(accepted > 0);
    CHECK(rejected > 0);
}

TEST_CASE("Short raw-height images cannot credit unwritten padding",
          "[video_core][stale_targets]") {
    const auto writer_info = MakeInfo(32, 8, 1);
    const auto old_info = MakeInfo(16, 8, 0);
    ImageBase writer{writer_info, GpuBase, CpuBase};
    ImageBase old{old_info, GpuBase + 1024, CpuBase + 1024};
    REQUIRE(writer.guest_size_bytes == 2048);
    REQUIRE(old.guest_size_bytes == 512);
    CHECK(MipBlockSize(writer_info, 0).height == 0);
    CHECK_FALSE(ImageLayout(writer_info, writer.guest_size_bytes).has_value());
    REQUIRE(ImageLayout(old_info, old.guest_size_bytes).has_value());
    CHECK_FALSE(IsSubresource(old_info, writer, old.gpu_addr, Options, false, true));
    CHECK_FALSE(IsSubresource(writer_info, old, writer.gpu_addr, Options, false, true));

    const auto pixels = SwizzledPixels(writer_info, 0x5A);
    CHECK(std::ranges::all_of(std::span{pixels}.first(1024), [](u8 v) { return v == 0x5A; }));
    CHECK(std::ranges::all_of(std::span{pixels}.subspan(1024, 512), [](u8 v) { return v == 0; }));
    // This is the rejected v1 footprint, deliberately using the raw descriptor.
    const auto wrong_runs = Layout{128, 8, 1}.Runs(0);
    CHECK_FALSE(VerifyFootprint(writer_info, writer.guest_size_bytes, wrong_runs));
    const auto effective_runs = Layout{128, 8, 0}.Runs(0);
    CHECK(VerifyFootprint(writer_info, writer.guest_size_bytes, effective_runs));

    Nx2State state;
    CacheFixture cache;
    const auto old_id = cache.Add(old_info, 0x31, 1024);
    const auto writer_id = cache.Add(writer_info, 0xA7);
    TextureCacheOperations::NotifyFullClear(cache, cache.View(writer_id));
    CHECK(cache.trace_stale_layout == 1);
    CHECK(cache.trace_stale_footprints == 0);
    CHECK(cache.slot_images.size() == 2);
    CHECK(TextureCacheOperations::FindImage(cache, old_info, old.gpu_addr, Options) == old_id);
    CHECK(std::ranges::all_of(cache.slot_images[old_id].pixels, [](u8 v) { return v == 0x31; }));
}

TEST_CASE("An incompatible whole clear drops a proven target and recreates defined contents",
          "[video_core][stale_targets]") {
    Nx2State state;
    CacheFixture cache;
    const auto a_info = MakeInfo(64, 16, 0);
    const auto b_info = MakeInfo(32, 16, 0, PixelFormat::R16G16B16A16_FLOAT);
    const auto a_id = cache.Add(a_info, 0x31);
    const auto b_id = cache.Add(b_info, 0xA7);
    cache.slot_images[a_id].flags = cache.slot_images[b_id].flags = ImageFlagBits::GpuModified;
    cache.slot_images[a_id].overlapping_images.push_back(b_id);
    cache.slot_images[b_id].overlapping_images.push_back(a_id);
    REQUIRE_FALSE(IsSubresource(a_info, cache.slot_images[b_id], GpuBase, Options, false, true));
    REQUIRE_FALSE(IsSubresource(b_info, cache.slot_images[a_id], GpuBase, Options, false, true));
    TextureCacheOperations::NotifyFullClear(cache, cache.View(b_id));
    CHECK(cache.trace_stale_failures == 0);
    CHECK(cache.trace_stale_drops == 1);
    CHECK(cache.trace_stale_verified == 1);
    CHECK(cache.retired == 1);
    REQUIRE(cache.slot_images.size() == 1);
    CHECK_FALSE(TextureCacheOperations::FindImage(cache, a_info, GpuBase, Options));
    const auto recreated = cache.Add(a_info, 0xCC, 0, false);
    cache.slot_images[recreated].stale_target.contents_defined = false;
    u64 skipped{}, skipped_bytes{}, gpu_written{}, zeroes{}, zero_bytes{}, zero_ns{};
    REQUIRE(TextureCacheOperations::RefreshStaleContents(
        cache.runtime, cache.slot_images[recreated], cache.stale_pages, skipped, skipped_bytes,
        gpu_written, zeroes, zero_bytes, zero_ns));
    CHECK(zeroes == 1);
    CHECK(zero_bytes == 4096);
    CHECK(std::ranges::all_of(cache.slot_images[recreated].pixels, [](u8 v) { return v == 0; }));
    std::fill_n(cache.slot_images[recreated].pixels.begin(), 16, u8{0xF2});
    CHECK(std::ranges::all_of(std::span{cache.slot_images[recreated].pixels}.subspan(16),
                              [](u8 v) { return v == 0; }));
}

TEST_CASE("Stale refresh defines fresh contents and keeps existing contents",
          "[video_core][stale_targets]") {
    Nx2State state;
    const auto info = MakeInfo(64, 16, 0);
    PixelsImage retained{info, GpuBase, CpuBase, 0x31};
    PixelsImage unrelated{info, GpuBase, CpuBase, 0xCC};
    retained.stale_target.contents_defined = true;
    StalePages stale;
    stale.Mark(CpuBase, retained.guest_size_bytes, [](DAddr, u64) {});
    PixelRuntime runtime;
    u64 skipped{};
    u64 skipped_bytes{};
    u64 gpu_written{};
    u64 zeroes{};
    u64 zero_bytes{};
    u64 zero_ns{};
    CHECK(TextureCacheOperations::RefreshStaleContents(runtime, retained, stale, skipped,
                                                       skipped_bytes, gpu_written, zeroes,
                                                       zero_bytes, zero_ns));
    CHECK(TextureCacheOperations::RefreshStaleContents(runtime, unrelated, stale, skipped,
                                                       skipped_bytes, gpu_written, zeroes,
                                                       zero_bytes, zero_ns));
    CHECK(runtime.transitions == 2);
    CHECK(skipped == 2);
    CHECK(skipped_bytes == 8192);
    CHECK(gpu_written == 0);
    CHECK(std::ranges::all_of(retained.pixels, [](u8 v) { return v == 0x31; }));
    CHECK(std::ranges::all_of(unrelated.pixels, [](u8 v) { return v == 0; }));
    CHECK(zeroes == 1);
    CHECK(zero_bytes == 4096);
    enabled = false;
    PixelsImage control{info, GpuBase, CpuBase, 0xCC};
    REQUIRE(TextureCacheOperations::RefreshStaleContents(
        runtime, control, stale, skipped, skipped_bytes, gpu_written, zeroes, zero_bytes, zero_ns));
    CHECK(std::ranges::all_of(control.pixels, [](u8 v) { return v == 0xCC; }));
    CHECK(zeroes == 1);
    enabled = true;

    retained.dirty_layers.push_back(1);
    CHECK_FALSE(TextureCacheOperations::RefreshStaleContents(runtime, retained, stale, skipped,
                                                             skipped_bytes, gpu_written, zeroes,
                                                             zero_bytes, zero_ns));
    retained.dirty_layers.clear();
    for (const auto special : {ImageFlagBits::Sparse, ImageFlagBits::SparseLayers,
                               ImageFlagBits::SparseTiles, ImageFlagBits::Unbacked}) {
        retained.flags = special;
        CHECK_FALSE(TextureCacheOperations::RefreshStaleContents(runtime, retained, stale, skipped,
                                                                 skipped_bytes, gpu_written, zeroes,
                                                                 zero_bytes, zero_ns));
    }
    retained.flags = ImageFlagBits::CpuModified;
    Tegra::Nx2MarkGpuWritten(CpuBase, retained.guest_size_bytes);
    CHECK_FALSE(TextureCacheOperations::RefreshStaleContents(runtime, retained, stale, skipped,
                                                             skipped_bytes, gpu_written, zeroes,
                                                             zero_bytes, zero_ns));
    CHECK(gpu_written == 1);
    CHECK(runtime.transitions == 3);
    stale.Clear(CpuBase, retained.guest_size_bytes, [](DAddr, u64) {});
    CHECK_FALSE(TextureCacheOperations::RefreshStaleContents(runtime, retained, stale, skipped,
                                                             skipped_bytes, gpu_written, zeroes,
                                                             zero_bytes, zero_ns));
    CHECK(gpu_written == 1);
    stale.Mark(CpuBase, retained.guest_size_bytes, [](DAddr, u64) {});
    Tegra::g_nx2_ampere.store(false);
    CHECK_FALSE(TextureCacheOperations::RefreshStaleContents(runtime, retained, stale, skipped,
                                                             skipped_bytes, gpu_written, zeroes,
                                                             zero_bytes, zero_ns));
    CHECK(gpu_written == 1);
}

TEST_CASE("Production lookup rejects remapped and unmapped images and selects newest candidates",
          "[video_core][stale_targets]") {
    CacheFixture cache;
    const auto info = MakeInfo(64, 16, 0);
    const auto first = cache.Add(info, 0x31);
    const auto newer = cache.Add(info, 0xA7);
    cache.slot_images[first].modification_tick = 1;
    cache.slot_images[newer].modification_tick = 2;
    CHECK(TextureCacheOperations::FindImage(cache, info, GpuBase, Options) == newer);
    cache.slot_images[newer].flags |= ImageFlagBits::Remapped;
    CHECK(TextureCacheOperations::FindImage(cache, info, GpuBase, Options) == first);
    cache.mapped = false;
    CHECK_FALSE(TextureCacheOperations::FindImage(cache, info, GpuBase, Options));
}

TEST_CASE("Production layout audit rejects unsupported interpretations",
          "[video_core][stale_targets]") {
    const auto info = MakeInfo(384, 384, 4, PixelFormat::D32_FLOAT);
    constexpr u64 Bytes = 589824;
    REQUIRE(ImageLayout(info, Bytes).has_value());
    for (const auto type : {ImageType::e1D, ImageType::e3D, ImageType::Linear, ImageType::Buffer}) {
        auto other = info;
        other.type = type;
        CHECK_FALSE(ImageLayout(other, Bytes).has_value());
    }
    for (const auto format :
         {PixelFormat::BC1_RGBA_UNORM, PixelFormat::ASTC_2D_4X4_UNORM, PixelFormat::X8_D24_UNORM,
          PixelFormat::R16G16B16X16_FLOAT, PixelFormat::D32_FLOAT_S8_UINT}) {
        auto other = info;
        other.format = format;
        CHECK_FALSE(ImageLayout(other, Bytes).has_value());
    }
    const auto reject = [&](const ImageInfo& other) {
        CHECK_FALSE(ImageLayout(other, Bytes).has_value());
    };
    auto other = info;
    other.resources.levels = 2;
    reject(other);
    other = info;
    other.resources.layers = 2;
    reject(other);
    other = info;
    other.size.depth = 2;
    reject(other);
    other = info;
    other.num_samples = 4;
    reject(other);
    other = info;
    other.block.width = 1;
    reject(other);
    other = info;
    other.block.depth = 1;
    reject(other);
    other = info;
    other.tile_width_spacing = 1;
    reject(other);
    other = info;
    other.is_sparse = true;
    reject(other);
    other = info;
    ++other.layer_stride;
    reject(other);
    CHECK_FALSE(ImageLayout(info, Bytes - 512).has_value());
    const std::array<Run, 1> outside{{{Bytes, Bytes + 1}}};
    CHECK_FALSE(VerifyFootprint(info, Bytes, outside));
    const std::array<Run, 1> reversed{{{512, 0}}};
    CHECK_FALSE(VerifyFootprint(info, Bytes, reversed));
    const std::array<Run, 2> unsorted{{{512, 1024}, {0, 512}}};
    CHECK_FALSE(VerifyFootprint(info, Bytes, unsorted));
}

TEST_CASE("Only whole attachment clears enter the optional audit", "[video_core][stale_targets]") {
    const ClearCoverage whole{0, 0, 384, 384, 0, 1, true, false};
    CHECK(whole.Whole(384, 384));
    auto changed = whole;
    changed.x = 1;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.y = -1;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.width = 383;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.height = 383;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.base_layer = 1;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.layers = 2;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.all_channels_aspects = false;
    CHECK_FALSE(changed.Whole(384, 384));
    changed = whole;
    changed.rescaled = true;
    CHECK_FALSE(changed.Whole(384, 384));

    Nx2State state;
    CacheFixture cache;
    const auto id = cache.Add(MakeInfo(64, 16, 0), 0x31);
    const auto view_id = cache.View(id);
    verify = false;
    TextureCacheOperations::NotifyFullClear(cache, view_id);
    CHECK(cache.trace_stale_footprints == 1);
    cache.trace_stale_footprints = 0;
    verify = true;
    enabled = false;
    TextureCacheOperations::NotifyFullClear(cache, view_id);
    CHECK(cache.trace_stale_footprints == 0);
    enabled = true;
    TextureCacheOperations::NotifyFullClear(cache, NULL_IMAGE_VIEW_ID);
    CHECK(cache.trace_stale_footprints == 0);
    cache.slot_images[id].stale_target.contents_defined = false;
    cache.slot_image_views[view_id].range.base.level = 1;
    TextureCacheOperations::NotifyFullClear(cache, view_id);
    CHECK(cache.trace_stale_footprints == 0);
    CHECK_FALSE(cache.slot_images[id].stale_target.contents_defined);
    cache.slot_image_views[view_id].range.base.level = 0;
    TextureCacheOperations::NotifyFullClear(cache, view_id);
    CHECK(cache.trace_stale_footprints == 1);
    CHECK(cache.slot_images[id].stale_target.contents_defined);
    CHECK(cache.slot_images.size() == 1);
}

TEST_CASE("Drifting clear replay reports creation disposal and the retained bound",
          "[video_core][stale_targets]") {
    Nx2State state;
    verify = false; // VerifyDrop is exercised separately; layout verification is always required.
    const auto shorter = Replay(120);
    const auto longer = Replay(240);
    CHECK(shorter.created == 6000);
    CHECK(longer.created == 12000);
    CHECK(shorter.live + shorter.dropped == shorter.created);
    CHECK(longer.live + longer.dropped == longer.created);
    CHECK(shorter.live <= 200);
    CHECK(longer.live <= 200);
    CHECK(longer.peak <= 201);
    std::printf("stale replay v3: 120 frames %zu live %llu guest bytes created %llu dropped %llu; "
                "240 frames %zu live %llu guest bytes created %llu dropped %llu; "
                "created/frame 50 last-drops/frame %llu/%llu peak %zu/%zu; pool false\n",
                shorter.live, static_cast<unsigned long long>(shorter.bytes),
                static_cast<unsigned long long>(shorter.created),
                static_cast<unsigned long long>(shorter.dropped), longer.live,
                static_cast<unsigned long long>(longer.bytes),
                static_cast<unsigned long long>(longer.created),
                static_cast<unsigned long long>(longer.dropped),
                static_cast<unsigned long long>(shorter.last_drops),
                static_cast<unsigned long long>(longer.last_drops), shorter.peak, longer.peak);
}

TEST_CASE("Each disqualifying history retains the stale target", "[video_core][stale_targets]") {
    Nx2State state;
    const auto old_info = MakeInfo(64, 16, 0);
    const auto writer_info = MakeInfo(32, 16, 0, PixelFormat::R16G16B16A16_FLOAT);
    for (const auto flag :
         {ImageFlagBits::CpuModified, ImageFlagBits::Sparse, ImageFlagBits::SparseLayers,
          ImageFlagBits::SparseTiles, ImageFlagBits::Unbacked, ImageFlagBits::Rescaled,
          ImageFlagBits::Remapped, ImageFlagBits::TrimmedLevels}) {
        CacheFixture cache;
        const auto old = cache.Add(old_info, 0x31);
        cache.slot_images[old].flags = ImageFlagBits::GpuModified | flag;
        const auto writer = cache.Add(writer_info, 0xA7);
        TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
        CHECK(cache.trace_stale_drops == 0);
        CHECK(cache.slot_images.size() == 2);
    }
    for (u32 reason = 0; reason < 5; ++reason) {
        CacheFixture cache;
        const auto old = cache.Add(old_info, 0x31);
        auto& image = cache.slot_images[old];
        switch (reason) {
        case 0:
            image.stale_target.render_target_origin = false;
            break;
        case 1:
            image.stale_target.Forbid();
            break; // Prior upload/CPU/copy, even after flags clear.
        case 2:
            image.stale_target.contents_defined = false;
            break;
        case 3:
            image.info.num_samples = 4;
            break;
        case 4:
            image.info.is_sparse = true;
            break;
        default:
            FAIL("invalid reason");
            break;
        }
        const auto writer = cache.Add(writer_info, 0xA7);
        TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
        CHECK(cache.trace_stale_drops == 0);
        CHECK(std::ranges::all_of(cache.slot_images[old].pixels, [](u8 v) { return v == 0x31; }));
    }
}

TEST_CASE("Aliases bindings and pending work protect a target", "[video_core][stale_targets]") {
    Nx2State state;
    for (u32 reason = 0; reason < 5; ++reason) {
        CacheFixture cache;
        const auto old = cache.Add(MakeInfo(64, 16, 0), 0x31);
        const auto writer = cache.Add(MakeInfo(32, 16, 0, PixelFormat::R16G16B16A16_FLOAT), 0xA7);
        auto& image = cache.slot_images[old];
        switch (reason) {
        case 0:
            image.flags |= ImageFlagBits::Alias;
            break;
        case 1:
            image.flags |= ImageFlagBits::IsDecoding;
            break;
        case 2:
            image.aliased_images.push_back(AliasedImage{{}, writer});
            break;
        case 3:
            cache.stale_references.insert(old);
            break;
        case 4: {
            const auto saved_info = image.info;
            const auto subresource = cache.Add(saved_info, 0x5A);
            // The cache records an overlap on both images.
            cache.slot_images[old].overlapping_images.push_back(subresource);
            cache.slot_images[subresource].overlapping_images.push_back(old);
            break;
        }
        default:
            FAIL("invalid reason");
            break;
        }
        TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
        CHECK(cache.trace_stale_drops == 0);
    }
}

TEST_CASE("Cumulative clears preserve gaps and reset after any intervening use",
          "[video_core][stale_targets]") {
    Nx2State state;
    for (const bool gap : {false, true}) {
        for (const bool used : {false, true}) {
            CacheFixture cache;
            // One whole page, so the stale marker a drop leaves covers the image.
            const auto old = cache.Add(MakeInfo(16, 64, 0), 0x31);
            const auto piece = MakeInfo(8, 16, 0, PixelFormat::R16G16B16A16_FLOAT);
            const auto first = cache.Add(piece, 0xA7);
            TextureCacheOperations::NotifyFullClear(cache, cache.View(first));
            REQUIRE(cache.trace_stale_drops == 0);
            if (used) {
                // Exercise the real lookup path, which invalidates the proof.
                REQUIRE(TextureCacheOperations::FindImage(cache, cache.slot_images[old].info,
                                                          GpuBase, Options) == old);
            }
            const auto last = cache.Add(MakeInfo(8, 32, 0, PixelFormat::R16G16B16A16_FLOAT),
                                        0xA7, 2048);
            TextureCacheOperations::NotifyFullClear(cache, cache.View(last));
            CHECK(cache.trace_stale_drops == 0);
            if (!gap) {
                const auto middle = cache.Add(piece, 0xA7, 1024);
                TextureCacheOperations::NotifyFullClear(cache, cache.View(middle));
            }
            CHECK(cache.trace_stale_drops == ((!gap && !used) ? 1 : 0));
            if (gap || used) {
                CHECK_FALSE(cache.slot_images[old].stale_target.EveryLiveByteOverwritten());
                CHECK_FALSE(VerifyDrop(cache.slot_images[old].info,
                                       cache.slot_images[old].guest_size_bytes, CpuBase,
                                       cache.slot_images[old].stale_target));
            }
        }
    }
}

TEST_CASE("A target the stale marker cannot cover stays", "[video_core][stale_targets]") {
    Nx2State state;
    // Each old image is wholly overwritten by one newer clear, and only its size or address
    // differs: 1,536 bytes inside a page, and one page of bytes starting 512 bytes into a page.
    for (const bool unaligned : {false, true}) {
        CacheFixture cache;
        const auto old = unaligned ? cache.Add(MakeInfo(16, 64, 0), 0x31, 512)
                                   : cache.Add(MakeInfo(16, 24, 0), 0x31);
        const auto writer =
            cache.Add(MakeInfo(8, 128, 0, PixelFormat::R16G16B16A16_FLOAT), 0xA7);
        TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
        CHECK(cache.trace_stale_drops == 0);
        CHECK(cache.slot_images.size() == 2);
        CHECK_FALSE(TextureCacheOperations::StaleMarkerCovers(cache.slot_images[old]));
    }
    // The same overwrite of a whole, aligned page drops the old image.
    CacheFixture cache;
    cache.Add(MakeInfo(16, 64, 0), 0x31);
    const auto writer = cache.Add(MakeInfo(8, 128, 0, PixelFormat::R16G16B16A16_FLOAT), 0xA7);
    TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
    CHECK(cache.trace_stale_drops == 1);
}

TEST_CASE("A whole clear of an older image counts as an overwrite", "[video_core][stale_targets]") {
    Nx2State state;
    // The writer was made before the stale target and is cleared again after it: the target's
    // bytes are overwritten just the same. A use of the target in between keeps it.
    for (const bool used : {false, true}) {
        CacheFixture cache;
        const auto writer = cache.Add(MakeInfo(8, 128, 0, PixelFormat::R16G16B16A16_FLOAT), 0xA7);
        const auto old = cache.Add(MakeInfo(16, 64, 0), 0x31);
        TextureCacheOperations::NotifyFullClear(cache, cache.View(old));
        REQUIRE(cache.trace_stale_drops == 0);
        TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
        REQUIRE(cache.trace_stale_drops == 1);
        // Made again at the same place, it is the newer image now and the same writer drops it
        // again only while nothing uses it.
        const auto again = cache.Add(MakeInfo(16, 64, 0), 0x31);
        if (used) {
            cache.stale_references.insert(again);
        }
        TextureCacheOperations::NotifyFullClear(cache, cache.View(writer));
        CHECK(cache.trace_stale_drops == (used ? 1U : 2U));
    }
}

TEST_CASE("Independent drop verification rejects forged coverage", "[video_core][stale_targets]") {
    const auto info = MakeInfo(16, 16, 0);
    const auto piece = MakeInfo(16, 8, 0);
    Proof proof;
    proof.birth = 1;
    proof.started = true;
    proof.evidence.push_back({piece, CalculateGuestSizeInBytes(piece), CpuBase, 2});
    CHECK(proof.EveryLiveByteOverwritten()); // Deliberately corrupted interval result.
    CHECK_FALSE(VerifyDrop(info, CalculateGuestSizeInBytes(info), CpuBase, proof));
}

TEST_CASE("Thousands of overlaps report clear creation and initialization costs",
          "[video_core][stale_targets]") {
    Nx2State state;
    verify = false;
    CacheFixture cache;
    cache.allocate_pixels = false;
    const auto old_info = MakeInfo(64, 16, 0);
    const auto piece = MakeInfo(8, 8, 0, PixelFormat::R16G16B16A16_FLOAT);
    const auto creation_start = std::chrono::steady_clock::now();
    for (u32 index = 0; index < 4096; ++index) {
        [[maybe_unused]] const auto id = cache.Add(old_info, 0);
    }
    const u64 creation_ns = ElapsedNs(creation_start);
    const auto writer = cache.Add(piece, 0);
    const auto view = cache.View(writer);
    // The first audit includes independently swizzling every previously unseen layout.
    TextureCacheOperations::NotifyFullClear(cache, view);
    const u64 cold_ns = cache.trace_stale_clear_ns;
    cache.trace_stale_clear_ns = 0;
    for (u32 iteration = 0; iteration < 100; ++iteration) {
        TextureCacheOperations::NotifyFullClear(cache, view);
    }
    CHECK(cache.trace_stale_drops == 0);
    CHECK(cache.slot_images.size() == 4097);
    CHECK(cache.trace_stale_contents == 4096 * 101);
    PixelRuntime runtime;
    StalePages stale;
    const auto zero_info = MakeInfo(384, 384, 4, PixelFormat::D32_FLOAT);
    const u64 bytes = CalculateGuestSizeInBytes(zero_info);
    stale.Mark(CpuBase, bytes, [](DAddr, u64) {});
    u64 skipped{}, skipped_bytes{}, gpu_written{}, zeroes{}, zero_bytes{}, zero_ns{};
    const auto zero_start = std::chrono::steady_clock::now();
    for (u32 iteration = 0; iteration < 100; ++iteration) {
        PixelsImage fresh{zero_info, GpuBase, CpuBase, 0xCC};
        REQUIRE(TextureCacheOperations::RefreshStaleContents(runtime, fresh, stale, skipped,
                                                             skipped_bytes, gpu_written, zeroes,
                                                             zero_bytes, zero_ns));
        CHECK(std::ranges::all_of(fresh.pixels, [](u8 v) { return v == 0; }));
    }
    const u64 total_zero_ns = ElapsedNs(zero_start);
    CHECK(zeroes == 100);
    CHECK(zero_bytes == bytes * 100);
    std::printf("stale costs: overlaps 4096 creation-ns/image %llu cold-clear-ns %llu "
                "warm-clear-ns/clear %llu; pixel-backend zero-ns/image %llu "
                "creation-refresh-check-ns/image %llu zeroes %llu bytes/image %llu\n",
                static_cast<unsigned long long>(creation_ns / 4096),
                static_cast<unsigned long long>(cold_ns),
                static_cast<unsigned long long>(cache.trace_stale_clear_ns / 100),
                static_cast<unsigned long long>(zero_ns / 100),
                static_cast<unsigned long long>(total_zero_ns / 100),
                static_cast<unsigned long long>(zeroes), static_cast<unsigned long long>(bytes));
}
