// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// Only owning handles: no guest memory, shader jobs, descriptor pools or frontend references.
struct PipelineHandles {
    std::array<vk::ShaderModule, 5> spv_modules;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;
    vk::ShaderModule mesh_vertex_module;
    vk::Pipeline mesh_compute_pipeline;
    vk::Pipeline task_compute_pipeline;
    vk::ShaderModule distribution_module;
    vk::DescriptorSetLayout distribution_set_layout;
    vk::PipelineLayout distribution_layout;
    vk::Pipeline distribution_pipeline;
};

} // namespace Vulkan
