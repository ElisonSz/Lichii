// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"

namespace VideoCommon::Nx2FastDraw {

u32 ClassifyBreak(Reason reason, u32 method) {
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using Regs = Maxwell3D::Regs;
    const auto within = [method](size_t first, size_t bytes) {
        return method >= first && method < first + bytes / sizeof(u32);
    };
    if (reason == Reason::DeltaOverflow) {
        return ClassBit(BreakClass::DeltaOverflow);
    }
    if (reason == Reason::InlineData) {
        return ClassBit(BreakClass::MethodUpload);
    }
    if (reason != Reason::Register && reason != Reason::Method) {
        return ClassBit(BreakClass::Other);
    }
    if (method == MAXWELL3D_REG_INDEX(clear_report_value) ||
        within(MAXWELL3D_REG_INDEX(render_enable), sizeof(Regs::RenderEnable)) ||
        method == MAXWELL3D_REG_INDEX(render_enable_override)) {
        return ClassBit(BreakClass::QueryCondition);
    }
    if (reason == Reason::Method) {
        switch (method) {
        case MAXWELL3D_REG_INDEX(report_semaphore.query):
            return ClassBit(BreakClass::MethodReport);
        case MAXWELL3D_REG_INDEX(wait_for_idle):
        case MAXWELL3D_REG_INDEX(fragment_barrier):
        case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        case MAXWELL3D_REG_INDEX(sync_info):
            return ClassBit(BreakClass::MethodBarrier);
        case MAXWELL3D_REG_INDEX(launch_dma):
        case MAXWELL3D_REG_INDEX(inline_data):
            return ClassBit(BreakClass::MethodUpload);
        default:
            return ClassBit(BreakClass::MethodControl);
        }
    }
    if (within(MAXWELL3D_REG_INDEX(vertex_streams),
               sizeof(Regs::VertexStream) * Regs::NumVertexArrays) ||
        within(MAXWELL3D_REG_INDEX(vertex_stream_instances), sizeof(Regs::VertexStreamInstances)) ||
        within(MAXWELL3D_REG_INDEX(vertex_attrib_format),
               sizeof(Regs::VertexAttribute) * Regs::NumVertexAttributes)) {
        return ClassBit(BreakClass::VertexLayout);
    }
    if (method == MAXWELL3D_REG_INDEX(vertex_id_base)) {
        return ClassBit(BreakClass::VertexId);
    }
    if (within(MAXWELL3D_REG_INDEX(pipelines), sizeof(Regs::Pipeline) * Regs::MaxShaderProgram) ||
        method == MAXWELL3D_REG_INDEX(program_region.address_high) ||
        method == MAXWELL3D_REG_INDEX(program_region.address_low)) {
        return ClassBit(BreakClass::Shader);
    }
    if (within(MAXWELL3D_REG_INDEX(viewport_transform),
               sizeof(Regs::ViewportTransform) * Regs::NumViewports) ||
        within(MAXWELL3D_REG_INDEX(viewports), sizeof(Regs::Viewport) * Regs::NumViewports) ||
        within(MAXWELL3D_REG_INDEX(scissor_test), sizeof(Regs::scissor_test)) ||
        within(MAXWELL3D_REG_INDEX(blend_color), sizeof(Regs::BlendColor)) ||
        within(MAXWELL3D_REG_INDEX(blend), sizeof(Regs::Blend)) ||
        within(MAXWELL3D_REG_INDEX(blend_per_target),
               sizeof(Regs::BlendPerTarget) * Regs::NumRenderTargets) ||
        within(MAXWELL3D_REG_INDEX(color_mask), sizeof(Regs::ColorMask) * Regs::NumRenderTargets) ||
        within(MAXWELL3D_REG_INDEX(stencil_enable), 0x20) ||
        method == MAXWELL3D_REG_INDEX(depth_test_enable) ||
        method == MAXWELL3D_REG_INDEX(depth_write_enabled) ||
        method == MAXWELL3D_REG_INDEX(depth_test_func) ||
        method == MAXWELL3D_REG_INDEX(blend_per_target_enabled) ||
        method == MAXWELL3D_REG_INDEX(color_mask_common) ||
        method == MAXWELL3D_REG_INDEX(depth_bounds_enable) ||
        method == MAXWELL3D_REG_INDEX(stencil_two_side_enable) ||
        within(MAXWELL3D_REG_INDEX(stencil_back_op), sizeof(Regs::StencilOp)) ||
        within(MAXWELL3D_REG_INDEX(depth_bounds), sizeof(Regs::depth_bounds)) ||
        within(MAXWELL3D_REG_INDEX(logic_op), sizeof(Regs::LogicOp)) ||
        within(MAXWELL3D_REG_INDEX(gl_cull_test_enabled), 3 * sizeof(u32)) ||
        within(MAXWELL3D_REG_INDEX(polygon_mode_front), 2 * sizeof(u32)) ||
        within(MAXWELL3D_REG_INDEX(polygon_offset_point_enable), 3 * sizeof(u32)) ||
        method == MAXWELL3D_REG_INDEX(rasterize_enable) ||
        method == MAXWELL3D_REG_INDEX(viewport_clip_control) ||
        method == MAXWELL3D_REG_INDEX(point_size) ||
        method == MAXWELL3D_REG_INDEX(line_width_smooth) ||
        method == MAXWELL3D_REG_INDEX(line_width_aliased)) {
        return ClassBit(BreakClass::FixedFunction);
    }
    return ClassBit(BreakClass::Other);
}

bool IsExperimentBreak(Reason reason, u32 method) {
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using Regs = Maxwell3D::Regs;
    if (reason == Reason::Method) {
        return method == MAXWELL3D_REG_INDEX(clear_report_value) ||
               method == MAXWELL3D_REG_INDEX(render_enable.mode);
    }
    if (reason != Reason::Register) {
        return false;
    }
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 attributes = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    constexpr u32 condition = MAXWELL3D_REG_INDEX(render_enable);
    return method == MAXWELL3D_REG_INDEX(vertex_id_base) ||
           (method >= streams && method < streams + Regs::NumVertexArrays * 4 &&
            (method - streams) % 4 == 0) ||
           (method >= attributes && method < attributes + Regs::NumVertexAttributes) ||
           method == MAXWELL3D_REG_INDEX(clear_report_value) ||
           (method >= condition && method < condition + sizeof(Regs::RenderEnable) / sizeof(u32));
}

void DeviceSpans::Update(const Tegra::MemoryManager& memory_, GPUVAddr address_, u64 size_) {
    const u64 serial = memory_.MapSerial();
    if (memory == &memory_ && address == address_ && size == size_ && map_serial == serial) {
        return;
    }
    memory = &memory_;
    address = address_;
    size = size_;
    map_serial = serial;
    spans.clear();
    for (const auto& [device, bytes] : memory_.GetSubmappedDeviceRange(address_, size_)) {
        spans.emplace_back(device, device + bytes);
    }
    std::ranges::sort(spans);
    size_t merged = 0;
    for (const auto& span : spans) {
        if (merged != 0 && span.first <= spans[merged - 1].second) {
            spans[merged - 1].second = (std::max)(spans[merged - 1].second, span.second);
        } else {
            spans[merged++] = span;
        }
    }
    spans.resize(merged);
}

bool DeviceSpans::Intersects(DAddr begin, DAddr end) const {
    const auto span = std::ranges::upper_bound(spans, begin, {}, &std::pair<DAddr, DAddr>::second);
    return span != spans.end() && span->first < end;
}

} // namespace VideoCommon::Nx2FastDraw
