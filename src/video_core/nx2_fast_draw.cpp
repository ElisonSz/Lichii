// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>

#include "common/logging.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_tested.h"

namespace VideoCommon::Nx2FastDraw {

void ReadDrawReuseSwitches() {
    const auto on = [](const char* name) {
        const char* const value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    g_draw_reuse = Nx2Tested::On("EDEN_NX2_DRAW_REUSE");
    g_draw_reuse_verify = g_draw_reuse && on("EDEN_NX2_DRAW_REUSE_VERIFY");
    if (g_draw_reuse) {
        g_draw_delta = true;
        g_draw_delta_verify = g_draw_delta_verify || g_draw_reuse_verify;
    }
    g_draw_vertex_input = Nx2Tested::On("EDEN_NX2_DRAW_VERTEX_INPUT");
    g_draw_handle_eval = Nx2Tested::On("EDEN_NX2_DRAW_HANDLE_EVAL");
    Shader::g_lean_handle_programs = g_draw_handle_eval;
}

void ReadDrawXSwitches() {
    const auto on = [](const char* name) {
        const char* const value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    g_drawx_neutral = Nx2Tested::On("EDEN_NX2_DRAWX_NEUTRAL");
    g_drawx_pipestate = Nx2Tested::On("EDEN_NX2_DRAWX_PIPESTATE");
    g_drawx_uniforms = Nx2Tested::On("EDEN_NX2_DRAWX_UNIFORMS");
    g_drawx_twin = Nx2Tested::On("EDEN_NX2_DRAWX_TWIN");
    g_drawx_verify = on("EDEN_NX2_DRAWX_VERIFY");
    g_drawx_keep_audit = g_drawx_verify || on("EDEN_NX2_DRAWX_KEEP_AUDIT");
    g_drawx_no_keeps = on("EDEN_NX2_DRAWX_NO_KEEPS");
    g_drawx_coalesce = on("EDEN_NX2_DRAWX_COALESCE");
    g_drawx_coalesce_verify = g_drawx_coalesce && on("EDEN_NX2_DRAWX_COALESCE_VERIFY");
    g_drawx_dynstate = Nx2Tested::On("EDEN_NX2_DRAWX_DYNSTATE");
    g_drawx_twin2 = Nx2Tested::On("EDEN_NX2_DRAWX_TWIN2");
    g_drawx_twin = g_drawx_twin || g_drawx_twin2;
    // The calls' neutrality leans on REUSE's checks of the writes the caches heard of (texture
    // headers, handle inputs), which the plain delta path does not make.
    g_drawx_calls = Nx2Tested::On("EDEN_NX2_DRAWX_CALLS") && g_draw_reuse;
    g_drawx_calls2 = on("EDEN_NX2_DRAWX_CALLS2") && g_draw_reuse;
    g_drawx_clears = on("EDEN_NX2_DRAWX_CLEARS");
    g_drawx_clip = on("EDEN_NX2_DRAWX_CLIP");
    Nx2CallSites::g_enabled = on("EDEN_NX2_DRAWX_CALL_SITES");
    g_drawx_registers = g_drawx_neutral || g_drawx_pipestate || g_drawx_twin || g_drawx_dynstate ||
                        g_drawx_clears || g_drawx_clip;
    for (u32 method = 0; method < g_drawx_register_class.size(); ++method) {
        DrawXRegister kind = ClassifyDrawXRegister(method);
        if ((kind == DrawXRegister::Neutral && !g_drawx_neutral) ||
            (kind == DrawXRegister::PipelineState && !g_drawx_pipestate) ||
            (kind == DrawXRegister::Shader && !g_drawx_twin) ||
            ((kind == DrawXRegister::DynamicState || kind == DrawXRegister::DynamicPipeline) &&
             !g_drawx_dynstate) ||
            (kind == DrawXRegister::Clear && !g_drawx_clears) ||
            (kind == DrawXRegister::Clip && !g_drawx_clip)) {
            kind = DrawXRegister::Breaks;
        }
        g_drawx_register_class[method] = kind;
    }
    if (g_drawx_registers || g_drawx_uniforms || g_drawx_twin || g_drawx_verify ||
        g_drawx_keep_audit || g_drawx_no_keeps || g_drawx_coalesce || g_drawx_calls ||
        g_drawx_calls2 || Nx2CallSites::g_enabled) {
        LOG_INFO(HW_GPU,
                 "Track D draw switches: neutral={} pipestate={} uniforms={} twin={} verify={} "
                 "keep_audit={} no_keeps={} coalesce={} coalesce_verify={} dynstate={} twin2={} "
                 "calls={} calls2={} clears={} clip={} call_sites={}",
                 g_drawx_neutral, g_drawx_pipestate, g_drawx_uniforms, g_drawx_twin, g_drawx_verify,
                 g_drawx_keep_audit, g_drawx_no_keeps, g_drawx_coalesce, g_drawx_coalesce_verify,
                 g_drawx_dynstate, g_drawx_twin2, g_drawx_calls, g_drawx_calls2, g_drawx_clears,
                 g_drawx_clip, Nx2CallSites::g_enabled);
    }
    if (Nx2Tested::On("EDEN_NX2_DRAWX_CALLS") && !g_draw_reuse) {
        LOG_WARNING(HW_GPU, "EDEN_NX2_DRAWX_CALLS needs EDEN_NX2_DRAW_REUSE=1: left off");
    }
    if (on("EDEN_NX2_DRAWX_CALLS2") && !g_draw_reuse) {
        LOG_WARNING(HW_GPU, "EDEN_NX2_DRAWX_CALLS2 needs EDEN_NX2_DRAW_REUSE=1: left off");
    }
}

void CheckCoalesced(const Delta& taken) {
    CoalesceVerify& verify = g_coalesce_verify;
    if (!taken.overflowed) {
        const auto check = [&](const std::vector<std::pair<u64, u64>>& ranges,
                               const std::array<u64, Delta::MaxWrites>& begins,
                               const std::array<u64, Delta::MaxWrites>& ends, u32 count,
                               const char* kind) {
            for (const auto& [begin, end] : ranges) {
                ++verify.checked;
                bool covered = false;
                for (u32 index = 0; index < count && !covered; ++index) {
                    covered = begins[index] <= begin && end <= ends[index];
                }
                if (!covered && verify.mismatches++ < 16) {
                    LOG_ERROR(HW_GPU,
                              "DrawX coalesce verify: {} write [{:#x}, {:#x}) is in no list range "
                              "({} ranges)",
                              kind, begin, end, count);
                }
            }
        };
        check(verify.gpu, taken.write_begin, taken.write_end, taken.num_writes, "GPU");
        check(verify.device, taken.device_begin, taken.device_end, taken.num_device_writes,
              "device");
    }
    verify.Clear();
}

DrawXRegister ClassifyDrawXRegister(u32 method) {
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using Regs = Maxwell3D::Regs;
    // Turing MME memory reads: SET_MME_MEM_ADDRESS_A/B, the data RAM address and the two read
    // triggers (MME_DMA_READ, MME_DMA_READ_FIFOED). Only the macro engines read them (the macro
    // DMA read, Maxwell3D::FrontDmaRead); the words it reads reach draws through the registers
    // the macro writes, which break as any other write.
    constexpr u32 mme_memory_first = 0x154;
    constexpr u32 mme_memory_end = 0x159;
    if (method >= mme_memory_first && method < mme_memory_end) {
        return DrawXRegister::Neutral;
    }
    // SET_CONSTANT_COLOR_RENDERING: nothing in the emulator reads it.
    constexpr u32 constant_color = MAXWELL3D_REG_INDEX(const_color_rendering);
    if (method >= constant_color &&
        method < constant_color + sizeof(Regs::ConstantColorRendering) / sizeof(u32)) {
        return DrawXRegister::Neutral;
    }
    // EDEN_NX2_DRAWX_DYNSTATE: the depth bias values are read only by UpdateDepthBias (dynamic
    // state by its DepthBias dirty flag, on both paths); sub_tiling_perf and opportunistic_early_z
    // are read by nothing.
    const auto within = [method](u32 first, size_t bytes) {
        return method >= first && method < first + bytes / sizeof(u32);
    };
    // SET_SHADOW_RAM_CONTROL as a register write (ProcessDirtyRegisters): read by no draw
    // preparation; the method that sets the mode is taken in Maxwell3D::ProcessMethodCall.
    if (method == MAXWELL3D_REG_INDEX(depth_bias) ||
        method == MAXWELL3D_REG_INDEX(slope_scale_depth_bias) ||
        method == MAXWELL3D_REG_INDEX(depth_bias_clamp) ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control) ||
        within(MAXWELL3D_REG_INDEX(sub_tiling_perf), sizeof(Regs::SubTilingPerf)) ||
        within(MAXWELL3D_REG_INDEX(opportunistic_early_z), sizeof(Regs::OpportunisticEarlyZ))) {
        return DrawXRegister::DynamicState;
    }
    // mandated_early_z is in the pipeline key (FixedPipelineState::early_z) and read nowhere else.
    if (method == MAXWELL3D_REG_INDEX(mandated_early_z)) {
        return DrawXRegister::DynamicPipeline;
    }
    // EDEN_NX2_DRAWX_CLEARS: CLEAR_SURFACE's register is read by clears only (RasterizerVulkan::
    // Clear, TextureCache::ClearCovers).
    if (method == MAXWELL3D_REG_INDEX(clear_surface)) {
        return DrawXRegister::Clear;
    }
    // EDEN_NX2_DRAWX_CLIP: SURFACE_CLIP sets the RenderTargets and Viewports dirty flags only (not
    // the ColorBuffer or ZetaBuffer ones that look render target images up): delta draws run
    // UpdateRenderTargets (the framebuffer's size), UpdateDynamicStates and the render area words
    // again (GraphicsPipeline::ConfigureDelta), as a full resolve does.
    if (within(MAXWELL3D_REG_INDEX(surface_clip), sizeof(Regs::SurfaceClip))) {
        return DrawXRegister::Clip;
    }
    // Fixed-function state feeds the pipeline key and dynamic state only: no part of a resolve
    // the delta path keeps reads it, and UpdateDynamicStates runs on both paths.
    const u32 kind = ClassifyBreak(Reason::Register, method);
    if (kind == ClassBit(BreakClass::FixedFunction)) {
        return DrawXRegister::PipelineState;
    }
    // Shader programs (and their region) select the pipeline: only its lookup reads them.
    if (kind == ClassBit(BreakClass::Shader)) {
        return DrawXRegister::Shader;
    }
    return DrawXRegister::Breaks;
}

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
    const auto submapped = memory_.GetSubmappedDeviceRange(address_, size_);
    // Segments come in GPU order, joined where device memory continues.
    linear = submapped.size() == 1 && submapped[0].second == size_;
    linear_base = linear ? submapped[0].first : 0;
    for (const auto& [device, bytes] : submapped) {
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
