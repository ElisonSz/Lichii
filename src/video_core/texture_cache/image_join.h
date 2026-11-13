// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdlib>
#include <optional>

#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon::ImageJoin {

inline bool reuse_metadata{};
inline bool sparse_check{};

inline void ReadSwitches() {
    const auto enabled = [](const char* name) {
        const char* const value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    reuse_metadata = enabled("EDEN_NX2_IMAGES_JOIN");
    sparse_check = enabled("EDEN_NX2_IMAGES_SPARSE_CHECK");
}

enum class Relation { LeftAlias, RightAlias, BadOverlap };

/// One proposed image's metadata for the reverse subresource test. ResolveOverlap can change
/// its address and resources during the scan; discard the metadata on every successful resolve.
/// A forward subresource match needs none of it, so build only on the first reverse test.
class Metadata {
public:
    void Invalidate() {
        candidate.reset();
    }

    [[nodiscard]] const ImageBase& Get(const ImageInfo& info, GPUVAddr gpu_address,
                                       DAddr cpu_address) {
        if (!candidate) {
            candidate.emplace(info, gpu_address, cpu_address);
        }
        return *candidate;
    }

private:
    std::optional<ImageBase> candidate;
};

[[nodiscard]] inline Relation Classify(Metadata& metadata, const ImageInfo& info,
                                       GPUVAddr gpu_address, DAddr cpu_address,
                                       const ImageBase& overlap, bool broken_views,
                                       bool native_bgr) {
    constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
    if (IsSubresource(info, overlap, gpu_address, options, broken_views, native_bgr)) {
        return Relation::LeftAlias;
    }
    if (IsSubresource(overlap.info, metadata.Get(info, gpu_address, cpu_address), overlap.gpu_addr,
                      options, broken_views, native_bgr)) {
        return Relation::RightAlias;
    }
    return Relation::BadOverlap;
}

/// The mapping query is pure. Its result can affect Sparse only for a sparse image.
/// Keeping both orders here also lets a no-GPU test assert the query's call count.
template <class ContinuousQuery>
[[nodiscard]] bool IsSparse(bool requested, bool skip_unused_query, ContinuousQuery&& continuous) {
    return skip_unused_query ? requested && !continuous() : !continuous() && requested;
}

} // namespace VideoCommon::ImageJoin
