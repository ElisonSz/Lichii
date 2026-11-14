// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/renderer_vulkan/pipeline_destroy_timing.h"
#include "video_core/vulkan_common/vulkan_pipeline_handles.h"

namespace Vulkan {

inline void DestroyPipelineHandles(PipelineHandles& handles, PipelineDestroyTime& time) {
    auto begin = PipelineDestroyTime::Clock::now();
    // Helpers are VkPipelines too: include every vkDestroyPipeline in the same measurement.
    handles.distribution_pipeline.reset();
    handles.task_compute_pipeline.reset();
    handles.mesh_compute_pipeline.reset();
    handles.pipeline.reset();
    time.pipelines_ms += PipelineDestroyTime::Milliseconds(begin);
    begin = PipelineDestroyTime::Clock::now();
    handles.distribution_layout.reset();
    handles.distribution_set_layout.reset();
    handles.distribution_module.reset();
    handles.mesh_vertex_module.reset();
    handles.descriptor_update_template.reset();
    handles.pipeline_layout.reset();
    handles.descriptor_set_layout.reset();
    for (auto& module : handles.spv_modules) {
        module.reset();
    }
    time.other_ms += PipelineDestroyTime::Milliseconds(begin);
}

} // namespace Vulkan
