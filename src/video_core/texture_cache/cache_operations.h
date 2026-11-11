// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <optional>
#include <ranges>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "video_core/compatible_formats.h"
#include "video_core/nx2.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/stale_pages.h"
#include "video_core/texture_cache/stale_targets.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

// These are the cache's production lookup/refresh/audit operations. Keeping their
// storage and backend calls dependent lets tests supply deterministic pixels and
// mappings without constructing a graphics device or replacing the decisions.
struct TextureCacheOperations {
    template <class C>
    static ImageId FindImage(C& cache, const ImageInfo& info, GPUVAddr gpu_addr,
                             RelaxedOptions options) {
        const std::optional<DAddr> cpu_addr =
            cache.ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
        if (!cpu_addr) {
            return ImageId{};
        }
        const bool broken_views = cache.runtime.HasBrokenTextureViewFormats() ||
                                  True(options & RelaxedOptions::ForceBrokenViews);
        const bool native_bgr = cache.runtime.HasNativeBgr();
        const bool flexible_formats = True(options & RelaxedOptions::Format);
        ImageId image_id{};
        if (cache.mem_trace_enabled) {
            ++cache.trace_lookups;
        }
        SCOPE_EXIT {
            if (cache.mem_trace_enabled && image_id) {
                ++cache.trace_lookup_hits;
            }
        };
        boost::container::small_vector<ImageId, 8> image_ids;
        const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
            if (cache.mem_trace_enabled) {
                ++cache.trace_lookup_candidates;
            }
            if (True(existing_image.flags & ImageFlagBits::Remapped)) {
                if (cache.mem_trace_enabled) {
                    ++cache.trace_lookup_remapped;
                }
                return false;
            }
            if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
                [[unlikely]] {
                const bool strict_size = False(options & RelaxedOptions::Size) &&
                                         True(existing_image.flags & ImageFlagBits::Strong);
                const ImageInfo& existing = existing_image.info;
                if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                    existing.pitch == info.pitch &&
                    IsPitchLinearSameSize(existing, info, strict_size) &&
                    IsViewCompatible(existing.format, info.format, broken_views, native_bgr)) {
                    image_id = existing_image_id;
                    image_ids.push_back(existing_image_id);
                    return !flexible_formats && existing.format == info.format;
                }
            } else if (IsSubresource(info, existing_image, gpu_addr, options, broken_views,
                                     native_bgr)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return !flexible_formats && existing_image.info.format == info.format;
            }
            if (cache.mem_trace_enabled) {
                ++cache.trace_lookup_rejected;
                if (existing_image.gpu_addr == gpu_addr && cache.trace_rejection_lines < 8) {
                    ++cache.trace_rejection_lines;
                    const auto& existing = existing_image.info;
                    LOG_INFO(
                        HW_GPU,
                        "Memory trace: same-address lookup rejection GPU {:#x} origin {} TIC {} "
                        "options {:#x}: requested type {} format {} {}x{}x{} mips {} layers {} "
                        "samples {} stride {} pitch/block-raw {}/{}/{}; existing type {} format {} "
                        "{}x{}x{} mips {} layers {} samples {} stride {} pitch/block-raw {}/{}/{} "
                        "flags {:#x} address-space {}",
                        gpu_addr, cache.trace_origin, cache.trace_descriptor,
                        static_cast<u32>(options), static_cast<u32>(info.type),
                        static_cast<u32>(info.format), info.size.width, info.size.height,
                        info.size.depth, info.resources.levels, info.resources.layers,
                        info.num_samples, info.layer_stride, info.block.width, info.block.height,
                        info.block.depth, static_cast<u32>(existing.type),
                        static_cast<u32>(existing.format), existing.size.width,
                        existing.size.height, existing.size.depth, existing.resources.levels,
                        existing.resources.layers, existing.num_samples, existing.layer_stride,
                        existing.block.width, existing.block.height, existing.block.depth,
                        static_cast<u32>(existing_image.flags), existing_image.address_space);
                }
            }
            return false;
        };
        cache.ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
        for (const ImageId found : image_ids) {
            cache.slot_images[found].stale_target.Use();
        }
        if (image_ids.size() <= 1) [[likely]] {
            return image_id;
        }
        auto image_ids_compare = [&cache](ImageId a, ImageId b) {
            auto& image_a = cache.slot_images[a];
            auto& image_b = cache.slot_images[b];
            return image_a.modification_tick < image_b.modification_tick;
        };
        return *std::ranges::max_element(image_ids, image_ids_compare);
    }

    // Condition 1: history is monotonic; a later clear cannot erase a guest-data origin.
    static bool GpuOnlyRenderTarget(const ImageBase& image) {
        return image.stale_target.render_target_origin && !image.stale_target.forbidden_history &&
               image.stale_target.contents_defined && image.info.num_samples == 1 &&
               !image.info.is_sparse && !image.HasScaled() &&
               True(image.flags & ImageFlagBits::GpuModified) &&
               False(image.flags & (ImageFlagBits::CpuModified | ImageFlagBits::Sparse |
                                    ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
                                    ImageFlagBits::Unbacked | ImageFlagBits::Rescaled |
                                    ImageFlagBits::Remapped | ImageFlagBits::TrimmedLevels));
    }

    // The stale marker a drop leaves behind is kept per whole 4 KiB page inside the image.
    // An image that begins or ends inside a page would leave its edge pages unmarked, and its
    // recreation would upload obsolete guest bytes instead of getting defined contents: it stays.
    static bool StaleMarkerCovers(const ImageBase& image) {
        constexpr u64 page_mask = StalePages::PAGE_SIZE - 1;
        return image.guest_size_bytes != 0 && (image.cpu_addr & page_mask) == 0 &&
               (image.guest_size_bytes & page_mask) == 0;
    }

    static bool VerifiedLayout(ImageBase& image) {
        auto& proof = image.stale_target;
        if (!proof.footprint_checked) {
            proof.footprint_checked = true;
            const auto layout = StaleTargets::ImageLayout(image.info, image.guest_size_bytes);
            // Whole GOBs make Runs the complete live footprint, including all tails.
            if (layout && layout->WholeGobs()) {
                auto relative = layout->Runs(0);
                proof.footprint_valid =
                    !relative.empty() &&
                    StaleTargets::VerifyShape(image.info, image.guest_size_bytes, relative);
                if (proof.footprint_valid) {
                    proof.footprint = layout->Runs(image.cpu_addr);
                }
            }
        }
        return proof.footprint_valid;
    }

    template <class C>
    static void Created(C& cache, ImageBase& image, bool target) {
        image.stale_target.birth = ++cache.stale_birth;
        image.stale_target.render_target_origin = target;
        if (image.info.is_sparse || image.info.num_samples != 1 || image.HasScaled() ||
            True(image.flags &
                 (ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
                  ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked |
                  ImageFlagBits::Rescaled | ImageFlagBits::TrimmedLevels))) {
            image.stale_target.Forbid();
        }
    }

    template <class Runtime, class Image>
    static bool RefreshStaleContents(Runtime& runtime, Image& image, const StalePages& stale_pages,
                                     u64& skipped, u64& skipped_bytes, u64& gpu_written,
                                     u64& zeroes, u64& zero_bytes, u64& zero_ns) {
        if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || !image.dirty_layers.empty() ||
            True(image.flags & (ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                                ImageFlagBits::SparseTiles | ImageFlagBits::Unbacked)) ||
            !stale_pages.IsAll(image.cpu_addr, image.guest_size_bytes)) {
            return false;
        }
        if (!Tegra::Nx2MayBeGpuWritten(image.cpu_addr, image.guest_size_bytes)) {
            if (StaleTargets::enabled && !image.stale_target.contents_defined) {
                const auto start = std::chrono::steady_clock::now();
                bool cleared{};
                if constexpr (requires { runtime.ClearImage(image); }) {
                    cleared = runtime.ClearImage(image);
                }
                if (!cleared) {
                    auto copies = FullDownloadCopies(image.info);
                    if (True(image.flags & ImageFlagBits::Converted)) {
                        ConvertedLayout(image.info, copies);
                    }
                    size_t zero_size{};
                    for (const auto& copy : copies) {
                        zero_size = (std::max)(zero_size, copy.buffer_offset + copy.buffer_size);
                    }
                    auto staging = runtime.UploadStagingBuffer(zero_size);
                    std::memset(staging.mapped_span.data(), 0, staging.mapped_span.size_bytes());
                    image.UploadMemory(staging, copies);
                    runtime.InsertUploadMemoryBarrier();
                }
                image.stale_target.contents_defined = true;
                ++zeroes;
                zero_bytes += image.guest_size_bytes;
                zero_ns += StaleTargets::ElapsedNs(start);
            }
            ++skipped;
            skipped_bytes += image.guest_size_bytes;
            runtime.TransitionImageLayout(image);
            return true;
        }
        ++gpu_written;
        return false;
    }

    // Condition 4: no live alias/subresource, binding, or queued work may own this image.
    template <class C>
    static bool NoReferences(C& cache, ImageId id, const ImageBase& image) {
        if (!image.aliased_images.empty() ||
            True(image.flags & (ImageFlagBits::Alias | ImageFlagBits::IsDecoding)) ||
            cache.StaleTargetReferenced(id)) {
            return false;
        }
        // Bad overlaps are incompatible interpretations, not alias/subresource relationships.
        constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        for (const ImageId other_id : image.overlapping_images) {
            const auto& other = cache.slot_images[other_id];
            if (IsSubresource(image.info, other, image.gpu_addr, options, false, true) ||
                IsSubresource(other.info, image, other.gpu_addr, options, false, true)) {
                return false;
            }
        }
        return true;
    }

    template <class C>
    static void NotifyFullClear(C& cache, ImageViewId view_id) {
        if (!StaleTargets::enabled || !IsRenderTargetView(view_id) ||
            !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        ++cache.trace_stale_clears;
        SCOPE_EXIT {
            cache.trace_stale_clear_ns += StaleTargets::ElapsedNs(start);
        };
        const auto& view = cache.slot_image_views[view_id];
        auto& writer = cache.slot_images[view.image_id];
        writer.stale_target.Use();
        if (view.range.base.level != 0 || view.range.base.layer != 0 ||
            view.size != writer.info.size || view.format != writer.info.format ||
            True(writer.flags & (ImageFlagBits::Rescaled | ImageFlagBits::Unbacked |
                                 ImageFlagBits::Sparse | ImageFlagBits::Remapped)) ||
            writer.HasScaled()) {
            ++cache.trace_stale_layout;
            return;
        }
        // A complete base-image clear defines the host image even when its layout is not
        // eligible for stale-target disposal. A subresource clear must not do so.
        writer.stale_target.contents_defined = true;
        if (!VerifiedLayout(writer)) {
            ++cache.trace_stale_layout;
            return;
        }
        ++cache.trace_stale_footprints;
        const StaleTargets::Evidence source{writer.info, writer.guest_size_bytes, writer.cpu_addr,
                                            writer.stale_target.birth};
        // Scanning what is bound walks every live descriptor: do it only for a clear that
        // overlaps an older target the cheap conditions admit, not for every clear.
        bool references_collected = false;
        // Delete after the region iterator has released Picked/map references.
        boost::container::small_vector<ImageId, 32> drops;
        cache.ForEachImageInRegion(
            writer.cpu_addr, writer.guest_size_bytes, [&](ImageId id, ImageBase& old) {
                if (id == view.image_id) {
                    return;
                }
                ++cache.trace_stale_contents;
                if (!GpuOnlyRenderTarget(old) || old.stale_target.birth >= source.birth ||
                    old.address_space != writer.address_space || !StaleMarkerCovers(old)) {
                    old.stale_target.Use();
                    return;
                }
                if (!references_collected) {
                    cache.CollectStaleReferences();
                    references_collected = true;
                    ++cache.trace_stale_reference_scans;
                }
                if (!NoReferences(cache, id, old) || !VerifiedLayout(old)) {
                    old.stale_target.Use();
                    return;
                }
                old.stale_target.Cover(old.stale_target.footprint, writer.stale_target.footprint,
                                       source);
                // Conditions 2 and 3 are separate named checks of the accumulated proof.
                if (!old.stale_target.NoUseSinceOverwrite() ||
                    !old.stale_target.EveryLiveByteOverwritten()) {
                    return;
                }
                if (StaleTargets::verify) {
                    if (!StaleTargets::VerifyDrop(old.info, old.guest_size_bytes, old.cpu_addr,
                                                  old.stale_target)) {
                        ++cache.trace_stale_failures;
                        LOG_ERROR(
                            HW_GPU,
                            "Stale target verify: FAIL GPU {:#x} CPU {:#x} bytes {} format {} "
                            "{}x{} block {}/{}/{} birth {} use {} overwrite {} writers {}; "
                            "retained",
                            old.gpu_addr, old.cpu_addr, old.guest_size_bytes,
                            static_cast<u32>(old.info.format), old.info.size.width,
                            old.info.size.height, old.info.block.width, old.info.block.height,
                            old.info.block.depth, old.stale_target.birth,
                            old.stale_target.use_epoch, old.stale_target.overwrite_epoch,
                            old.stale_target.evidence.size());
                        LOG_ERROR(
                            HW_GPU,
                            "Stale target verify: target type {} depth {} levels {} layers {} "
                            "samples {} stride {} spacing {} native {} flags {:#x} origin {} "
                            "forbidden {} defined {} references {}",
                            static_cast<u32>(old.info.type), old.info.size.depth,
                            old.info.resources.levels, old.info.resources.layers,
                            old.info.num_samples, old.info.layer_stride,
                            old.info.tile_width_spacing, old.info.native_tiling,
                            static_cast<u32>(old.flags), old.stale_target.render_target_origin,
                            old.stale_target.forbidden_history, old.stale_target.contents_defined,
                            !NoReferences(cache, id, old));
                        for (const auto& evidence : old.stale_target.evidence) {
                            LOG_ERROR(
                                HW_GPU,
                                "Stale target verify: writer CPU {:#x} bytes {} birth {} type {} "
                                "format {} {}x{}x{} levels {} layers {} samples {} block {}/{}/{} "
                                "stride {} spacing {} native {}",
                                evidence.address, evidence.bytes, evidence.birth,
                                static_cast<u32>(evidence.info.type),
                                static_cast<u32>(evidence.info.format), evidence.info.size.width,
                                evidence.info.size.height, evidence.info.size.depth,
                                evidence.info.resources.levels, evidence.info.resources.layers,
                                evidence.info.num_samples, evidence.info.block.width,
                                evidence.info.block.height, evidence.info.block.depth,
                                evidence.info.layer_stride, evidence.info.tile_width_spacing,
                                evidence.info.native_tiling);
                        }
                        old.stale_target.Forbid();
                        return;
                    }
                    ++cache.trace_stale_verified;
                    LOG_DEBUG(HW_GPU,
                              "Stale target verify: drop GPU {:#x} CPU {:#x} bytes {} birth {} "
                              "writers {}",
                              old.gpu_addr, old.cpu_addr, old.guest_size_bytes,
                              old.stale_target.birth, old.stale_target.evidence.size());
                }
                drops.push_back(id);
            });
        for (const ImageId id : drops) {
            auto& old = cache.slot_images[id];
            // Recheck mutable protection immediately before normal delayed retirement.
            if (!GpuOnlyRenderTarget(old) || !StaleMarkerCovers(old) ||
                !NoReferences(cache, id, old) || !old.stale_target.NoUseSinceOverwrite() ||
                !old.stale_target.EveryLiveByteOverwritten()) {
                continue;
            }
            ++cache.trace_stale_drops;
            cache.trace_stale_drop_bytes += old.guest_size_bytes;
            cache.MarkStale(old);
            if (True(old.flags & ImageFlagBits::Tracked)) {
                cache.UntrackImage(old, id);
            }
            cache.UnregisterImage(id);
            cache.DeleteImage(id);
        }
    }
};

} // namespace VideoCommon
