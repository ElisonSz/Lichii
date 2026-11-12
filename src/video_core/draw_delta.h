// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

enum class DrawDeltaTextureResult : u8 { Match, NotCaptured, Handles, Descriptors };

enum class DrawDeltaGuardResult : u8 { Match, State, Submitted, RenderPass };

struct DrawDeltaGuard {
    u64 serial{};
    u64 map_serial{};
    u64 tick{};
    u64 render_pass{};
};

[[nodiscard]] constexpr DrawDeltaGuardResult CheckDrawDeltaGuard(const DrawDeltaGuard& previous,
                                                                 const DrawDeltaGuard& current,
                                                                 bool pass_active,
                                                                 bool pipeline_bound) {
    if (previous.serial != current.serial || previous.map_serial != current.map_serial) {
        return DrawDeltaGuardResult::State;
    }
    if (previous.tick != current.tick) {
        return DrawDeltaGuardResult::Submitted;
    }
    if (previous.render_pass != current.render_pass || !pass_active || !pipeline_bound) {
        return DrawDeltaGuardResult::RenderPass;
    }
    return DrawDeltaGuardResult::Match;
}

/// Keep the descriptors that actually selected the views, not a second read of guest memory.
/// Readers return an optional descriptor: absent/unmapped entries cannot authorize reuse.
/// Epochs invalidate IDs even when a deleted/recreated image has identical descriptor bytes.
template <typename ImageDescriptor, typename SamplerDescriptor>
class DrawDeltaTextures {
public:
    using Handle = std::pair<u32, u32>;

    template <typename ReadImage, typename ReadSampler>
    void Capture(std::span<const Handle> handles, u64 images, u64 samplers, ReadImage&& read_image,
                 ReadSampler&& read_sampler) {
        valid = false;
        entries.clear();
        for (const Handle& handle : handles) {
            const auto image = read_image(handle.first);
            if (!image) {
                return;
            }
            const auto sampler = handle.second == ~0U
                                    ? SamplerDescriptor{}
                                    : read_sampler(handle.second);
            if (!sampler) {
                return;
            }
            entries.push_back({handle, *image, *sampler});
        }
        image_epoch = images;
        sampler_epoch = samplers;
        valid = true;
    }

    template <typename ReadImage, typename ReadSampler>
    [[nodiscard]] DrawDeltaTextureResult Match(std::span<const Handle> handles, u64 images,
                                               u64 samplers, ReadImage&& read_image,
                                               ReadSampler&& read_sampler) const {
        if (!valid) {
            return DrawDeltaTextureResult::NotCaptured;
        }
        if (images != image_epoch || samplers != sampler_epoch) {
            return DrawDeltaTextureResult::Descriptors;
        }
        if (handles.size() != entries.size()) {
            return DrawDeltaTextureResult::Handles;
        }
        for (size_t index = 0; index < handles.size(); ++index) {
            const Entry& entry = entries[index];
            if (handles[index] != entry.handle) {
                return DrawDeltaTextureResult::Handles;
            }
            const auto image = read_image(entry.handle.first);
            const auto sampler = entry.handle.second == ~0U
                                    ? SamplerDescriptor{}
                                    : read_sampler(entry.handle.second);
            if (!image || !sampler || *image != entry.image || *sampler != entry.sampler) {
                return DrawDeltaTextureResult::Descriptors;
            }
        }
        return DrawDeltaTextureResult::Match;
    }

private:
    struct Entry {
        Handle handle;
        ImageDescriptor image;
        SamplerDescriptor sampler;
    };
    std::vector<Entry> entries;
    u64 image_epoch{};
    u64 sampler_epoch{};
    bool valid{};
};

} // namespace VideoCommon
