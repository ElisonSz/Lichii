// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "common/logging.h"
#include "video_core/texture_cache/image_base.h"

namespace VideoCommon::ImageJoin {

inline bool stats_enabled{};
inline bool verify{};

/// All counters are owned by the texture cache's back thread. Fixed creation-count windows
/// avoid clock reads and TickFrame changes. Frame counts include the two endpoint frames.
struct Counters {
    u64 first_frame{};
    u64 creations{};
    std::array<u64, 5> origins{}; // copy, graphics TIC, compute TIC, render target, clear target
    u64 astc{};
    u64 views_new_image{};
    u64 views_kept_image{};
    u64 resolved{};
    u64 left_alias{};
    u64 right_alias{};
    u64 bad_overlap{};
    u64 ignored{};
    u64 metadata_builds{};
    u64 metadata_reuses{};
    u64 sparse_queries_skipped{};
    u64 deleted{};
    u64 retired{};
    u64 retire_partners{};
    u64 retired_address_returns{};
    u64 retired_same_shape{};
    u64 retired_within_age{};
    u64 verified{};
    u64 mismatches{};
};
inline Counters counters;

/// Bounded evidence of age-rule deletions returning later. This never affects cache decisions.
/// A shape match is deliberately explicit; an address return alone isn't image reuse evidence.
class RetirementHistory {
public:
    struct Entry {
        ImageInfo info;
        DAddr cpu_address;
        u64 frame;
    };
    static constexpr size_t Capacity = 8192;

    void Retire(const ImageBase& image, u64 frame) {
        if (entries.size() >= Capacity) {
            entries.clear();
        }
        entries.insert_or_assign(image.gpu_addr, Entry{image.info, image.cpu_addr, frame});
    }

    [[nodiscard]] std::optional<Entry> Take(GPUVAddr address) {
        const auto found = entries.find(address);
        if (found == entries.end()) {
            return std::nullopt;
        }
        const Entry entry = found->second;
        entries.erase(found);
        return entry;
    }

    [[nodiscard]] static bool SameShape(const Entry& old, const ImageInfo& info, DAddr address) {
        const auto& previous = old.info;
        return old.cpu_address == address && previous.format == info.format &&
               previous.type == info.type && previous.size == info.size &&
               previous.resources == info.resources && previous.block == info.block &&
               previous.layer_stride == info.layer_stride &&
               previous.maybe_unaligned_layer_stride == info.maybe_unaligned_layer_stride &&
               previous.num_samples == info.num_samples &&
               previous.tile_width_spacing == info.tile_width_spacing &&
               previous.native_tiling == info.native_tiling &&
               previous.is_sparse == info.is_sparse && previous.astc_opaque == info.astc_opaque;
    }

private:
    std::unordered_map<GPUVAddr, Entry> entries;
};
inline RetirementHistory retirement_history;

inline void Created(const ImageBase& image, std::string_view origin, u64 frame, u64 retire_age) {
    if (!stats_enabled) {
        return;
    }
    if (counters.creations == 0) {
        counters.first_frame = frame;
    }
    ++counters.creations;
    const size_t index = origin == "graphics-tic"    ? 1
                         : origin == "compute-tic"   ? 2
                         : origin == "render-target" ? 3
                         : origin == "clear-target"  ? 4
                                                     : 0;
    ++counters.origins[index];
    counters.astc += VideoCore::Surface::IsPixelFormatASTC(image.info.format) ? 1U : 0U;
    if (const auto previous = retirement_history.Take(image.gpu_addr)) {
        ++counters.retired_address_returns;
        if (RetirementHistory::SameShape(*previous, image.info, image.cpu_addr)) {
            ++counters.retired_same_shape;
            counters.retired_within_age += frame - previous->frame <= retire_age ? 1U : 0U;
        }
    }
    if (counters.creations < 2048) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Images churn: frames={} created={} origins copy/gfx/compute/rt/clear={}/{}/{}/{}/{} "
             "astc={} view_requests new/kept_image={}/{} joins={} aliases left/right={}/{} bad={} "
             "ignored={} deleted={} retired={} "
             "partners={} retired_address_returns={} same_shape={} within_age={} metadata "
             "builds/reuses={}/{} sparse_queries_skipped={} verified={} mismatches={}",
             frame - counters.first_frame + 1, counters.creations, counters.origins[0],
             counters.origins[1], counters.origins[2], counters.origins[3], counters.origins[4],
             counters.astc, counters.views_new_image, counters.views_kept_image, counters.resolved,
             counters.left_alias, counters.right_alias, counters.bad_overlap, counters.ignored,
             counters.deleted, counters.retired, counters.retire_partners,
             counters.retired_address_returns, counters.retired_same_shape,
             counters.retired_within_age, counters.metadata_builds, counters.metadata_reuses,
             counters.sparse_queries_skipped, counters.verified, counters.mismatches);
    counters = {};
}

inline void Retired(const ImageBase& image, u64 frame) {
    if (stats_enabled) {
        ++counters.retired;
        retirement_history.Retire(image, frame);
    }
}

} // namespace VideoCommon::ImageJoin
