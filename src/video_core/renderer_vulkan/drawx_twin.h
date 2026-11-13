// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/graphics_stage_info.h"
#include "video_core/shader_handle_program.h"

namespace Vulkan {

/// EDEN_NX2_DRAWX_TWIN2: whether two stage records resolve the same image state from the same
/// memory. Views, samplers, image descriptor entries and rescaling words come from nothing else
/// than the four image descriptor lists (types, formats, counts, constant buffer slots and
/// offsets, written flags; PushImageDescriptors), the handles read for them (ConfigureImpl,
/// EvaluateDeltaHandles) and HandleMustBeNull, which also reads the storage bindings. So they
/// must be equal, every texture element must be nulled alike, and a handle a program computes
/// needs the same programs, the same storage bindings they read through and the same handle
/// shader hash (its warnings).
[[nodiscard]] inline bool ResolvesSameImageState(const GraphicsStageInfo& lhs,
                                                 const GraphicsStageInfo& rhs) {
    if (!std::ranges::equal(lhs.texture_buffer_descriptors, rhs.texture_buffer_descriptors) ||
        !std::ranges::equal(lhs.image_buffer_descriptors, rhs.image_buffer_descriptors) ||
        !std::ranges::equal(lhs.texture_descriptors, rhs.texture_descriptors) ||
        !std::ranges::equal(lhs.image_descriptors, rhs.image_descriptors)) {
        return false;
    }
    for (const auto& desc : lhs.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            if (VideoCommon::HandleMustBeNull(lhs, desc, index) !=
                VideoCommon::HandleMustBeNull(rhs, desc, index)) {
                return false;
            }
        }
    }
    const auto programmed = [](const auto& descriptors) {
        return std::ranges::any_of(
            descriptors, [](const auto& descriptor) { return descriptor.handle_program != ~0U; });
    };
    if (!programmed(lhs.texture_buffer_descriptors) && !programmed(lhs.image_buffer_descriptors) &&
        !programmed(lhs.texture_descriptors) && !programmed(lhs.image_descriptors)) {
        return true;
    }
    return lhs.handle_shader_hash == rhs.handle_shader_hash &&
           std::ranges::equal(lhs.storage_buffers_descriptors, rhs.storage_buffers_descriptors) &&
           std::ranges::equal(
               lhs.handle_programs, rhs.handle_programs,
               [](const auto& left, const auto& right) { return std::ranges::equal(left, right); });
}

} // namespace Vulkan
