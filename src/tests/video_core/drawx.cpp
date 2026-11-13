// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Track D (EDEN_NX2_DRAWX_NEUTRAL, _PIPESTATE, _UNIFORMS): which 3D register writes still end the
// fast draw delta chain, and which breaks end uniform keeps on full draws.

#include <cstdlib>

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_fast_draw.h"

namespace {

using Tegra::Engines::Maxwell3D;
namespace FastDraw = VideoCommon::Nx2FastDraw;
using FastDraw::DrawXRegister;

constexpr u32 NumRegs = Maxwell3D::Regs::NUM_REGS;

/// Sets the four switches as the environment would and rebuilds the register table.
void ReadSwitches(bool neutral, bool pipestate, bool uniforms) {
    const auto set = [](const char* name, bool on) {
#ifdef _WIN32
        _putenv_s(name, on ? "1" : "0");
#else
        setenv(name, on ? "1" : "0", 1);
#endif
    };
    set("EDEN_NX2_DRAWX_NEUTRAL", neutral);
    set("EDEN_NX2_DRAWX_PIPESTATE", pipestate);
    set("EDEN_NX2_DRAWX_UNIFORMS", uniforms);
    set("EDEN_NX2_DRAWX_VERIFY", false);
    FastDraw::ReadDrawXSwitches();
}

bool InRange(u32 method, u32 first, size_t bytes) {
    return method >= first && method < first + bytes / sizeof(u32);
}

} // Anonymous namespace

TEST_CASE("DrawX: neutral registers are the MME memory read and constant colour blocks",
          "[video_core][drawx]") {
    constexpr u32 constant_color = MAXWELL3D_REG_INDEX(const_color_rendering);
    for (u32 method = 0; method < NumRegs; ++method) {
        const bool neutral =
            (method >= 0x154 && method <= 0x158) ||
            InRange(method, constant_color, sizeof(Maxwell3D::Regs::ConstantColorRendering));
        CHECK((FastDraw::ClassifyDrawXRegister(method) == DrawXRegister::Neutral) == neutral);
    }
    // Neighbours stay as they were.
    CHECK(FastDraw::ClassifyDrawXRegister(0x153) != DrawXRegister::Neutral);
    CHECK(FastDraw::ClassifyDrawXRegister(0x159) != DrawXRegister::Neutral);
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(stencil_back_ref)) !=
          DrawXRegister::Neutral);
}

TEST_CASE("DrawX: pipeline state registers are exactly the fixed-function break class",
          "[video_core][drawx]") {
    const u32 fixed_function = FastDraw::ClassBit(FastDraw::BreakClass::FixedFunction);
    u32 pipeline_state = 0;
    for (u32 method = 0; method < NumRegs; ++method) {
        const DrawXRegister kind = FastDraw::ClassifyDrawXRegister(method);
        if (kind == DrawXRegister::Neutral) {
            continue;
        }
        const bool fixed = FastDraw::ClassifyBreak(FastDraw::Reason::Register, method) ==
                           fixed_function;
        CHECK((kind == DrawXRegister::PipelineState) == fixed);
        pipeline_state += fixed ? 1 : 0;
    }
    CHECK(pipeline_state > 0);
}

TEST_CASE("DrawX: what a delta draw keeps still ends the chain", "[video_core][drawx]") {
    const auto breaks = [](u32 method) {
        return FastDraw::ClassifyDrawXRegister(method) == DrawXRegister::Breaks;
    };
    // Shaders, texture and sampler tables, render targets, vertex layout, transform feedback.
    for (u32 method = MAXWELL3D_REG_INDEX(pipelines);
         method < MAXWELL3D_REG_INDEX(pipelines) +
                      sizeof(Maxwell3D::Regs::Pipeline) * Maxwell3D::Regs::MaxShaderProgram / 4;
         ++method) {
        CHECK(breaks(method));
    }
    for (u32 word = 0; word < sizeof(Maxwell3D::Regs::TexHeader) / 4; ++word) {
        CHECK(breaks(MAXWELL3D_REG_INDEX(tex_header) + word));
    }
    for (u32 word = 0; word < sizeof(Maxwell3D::Regs::TexSampler) / 4; ++word) {
        CHECK(breaks(MAXWELL3D_REG_INDEX(tex_sampler) + word));
    }
    CHECK(breaks(MAXWELL3D_REG_INDEX(sampler_binding)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(rt)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(rt_control)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(zeta)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(vertex_attrib_format)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(vertex_streams)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(transform_feedback_enabled)));
    CHECK(breaks(MAXWELL3D_REG_INDEX(surface_clip)));
    // Fixed-function examples.
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(scissor_test)) ==
          DrawXRegister::PipelineState);
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(viewport_transform)) ==
          DrawXRegister::PipelineState);
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(depth_test_enable)) ==
          DrawXRegister::PipelineState);
}

TEST_CASE("DrawX: the register table follows the switches", "[video_core][drawx]") {
    for (int mask = 0; mask < 4; ++mask) {
        const bool neutral = (mask & 1) != 0;
        const bool pipestate = (mask & 2) != 0;
        ReadSwitches(neutral, pipestate, false);
        CHECK(FastDraw::g_drawx_registers == (neutral || pipestate));
        for (u32 method = 0; method < NumRegs; ++method) {
            const DrawXRegister full = FastDraw::ClassifyDrawXRegister(method);
            DrawXRegister expected = full;
            if ((full == DrawXRegister::Neutral && !neutral) ||
                (full == DrawXRegister::PipelineState && !pipestate)) {
                expected = DrawXRegister::Breaks;
            }
            CHECK(FastDraw::DrawXRegisterClass(method) == expected);
        }
        // Out of range: as without the switches.
        CHECK(FastDraw::DrawXRegisterClass(NumRegs) == DrawXRegister::Breaks);
        CHECK(FastDraw::DrawXRegisterClass(0xffffffffU) == DrawXRegister::Breaks);
    }
    ReadSwitches(false, false, false);
}

TEST_CASE("DrawX: only non-register breaks end uniform keeps on full draws",
          "[video_core][drawx]") {
    FastDraw::SetEnabledForTest(true);
    ReadSwitches(false, false, true);
    const u64 keep = FastDraw::g_keep_serial.load();
    const u64 serial = FastDraw::Serial();
    FastDraw::Break(FastDraw::Reason::Register, MAXWELL3D_REG_INDEX(pipelines) + 3);
    CHECK(FastDraw::g_keep_serial.load() == keep);
    CHECK(FastDraw::Serial() == serial + 1);
    for (u32 reason = 0; reason < static_cast<u32>(FastDraw::Reason::Count); ++reason) {
        const auto kind = static_cast<FastDraw::Reason>(reason);
        const u64 before = FastDraw::g_keep_serial.load();
        FastDraw::Break(kind, 0);
        CHECK(FastDraw::g_keep_serial.load() == before + (kind == FastDraw::Reason::Register ? 0 : 1));
    }
    // Without the switch the keep serial does not move.
    ReadSwitches(false, false, false);
    const u64 off = FastDraw::g_keep_serial.load();
    FastDraw::Break(FastDraw::Reason::Fence);
    FastDraw::Break(FastDraw::Reason::Method, MAXWELL3D_REG_INDEX(wait_for_idle));
    CHECK(FastDraw::g_keep_serial.load() == off);
    // Fast draws off: no break counts at all.
    ReadSwitches(false, false, true);
    FastDraw::SetEnabledForTest(false);
    const u64 disabled = FastDraw::g_keep_serial.load();
    FastDraw::Break(FastDraw::Reason::Fence);
    CHECK(FastDraw::g_keep_serial.load() == disabled);
    FastDraw::SetEnabledForTest(true);
    ReadSwitches(false, false, false);
}
