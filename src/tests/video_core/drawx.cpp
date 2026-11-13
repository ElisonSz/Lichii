// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Track D (EDEN_NX2_DRAWX_NEUTRAL, _PIPESTATE, _UNIFORMS, _TWIN): which 3D register writes still end
// the fast draw delta chain, and which breaks end uniform keeps on full draws.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_call_sites.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/renderer_vulkan/drawx_twin.h"

namespace {

using Tegra::Engines::Maxwell3D;
namespace FastDraw = VideoCommon::Nx2FastDraw;
using FastDraw::DrawXRegister;

constexpr u32 NumRegs = Maxwell3D::Regs::NUM_REGS;

void SetSwitch(const char* name, bool on) {
#ifdef _WIN32
    _putenv_s(name, on ? "1" : "0");
#else
    setenv(name, on ? "1" : "0", 1);
#endif
}

/// Sets the switches as the environment would and rebuilds the register table.
void ReadSwitches(bool neutral, bool pipestate, bool uniforms, bool twin = false,
                  bool dynstate = false, bool coalesce = false, bool clears = false,
                  bool clip = false) {
    SetSwitch("EDEN_NX2_DRAWX_NEUTRAL", neutral);
    SetSwitch("EDEN_NX2_DRAWX_PIPESTATE", pipestate);
    SetSwitch("EDEN_NX2_DRAWX_UNIFORMS", uniforms);
    SetSwitch("EDEN_NX2_DRAWX_TWIN", twin);
    SetSwitch("EDEN_NX2_DRAWX_VERIFY", false);
    SetSwitch("EDEN_NX2_DRAWX_DYNSTATE", dynstate);
    SetSwitch("EDEN_NX2_DRAWX_COALESCE", coalesce);
    SetSwitch("EDEN_NX2_DRAWX_COALESCE_VERIFY", coalesce);
    SetSwitch("EDEN_NX2_DRAWX_TWIN2", false);
    SetSwitch("EDEN_NX2_DRAWX_CALLS", false);
    SetSwitch("EDEN_NX2_DRAWX_CALLS2", false);
    SetSwitch("EDEN_NX2_DRAWX_CLEARS", clears);
    SetSwitch("EDEN_NX2_DRAWX_CLIP", clip);
    SetSwitch("EDEN_NX2_DRAWX_CALL_SITES", false);
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

TEST_CASE("DrawX: pipeline state and shader registers are exactly their break classes",
          "[video_core][drawx]") {
    const u32 fixed_function = FastDraw::ClassBit(FastDraw::BreakClass::FixedFunction);
    const u32 shader = FastDraw::ClassBit(FastDraw::BreakClass::Shader);
    u32 pipeline_state = 0;
    u32 shaders = 0;
    for (u32 method = 0; method < NumRegs; ++method) {
        const DrawXRegister kind = FastDraw::ClassifyDrawXRegister(method);
        if (kind == DrawXRegister::Neutral) {
            continue;
        }
        const u32 breaks = FastDraw::ClassifyBreak(FastDraw::Reason::Register, method);
        CHECK((kind == DrawXRegister::PipelineState) == (breaks == fixed_function));
        CHECK((kind == DrawXRegister::Shader) == (breaks == shader));
        pipeline_state += breaks == fixed_function ? 1 : 0;
        shaders += breaks == shader ? 1 : 0;
    }
    CHECK(pipeline_state > 0);
    // Six programs of 16 words and the program region's two address words.
    CHECK(shaders == sizeof(Maxwell3D::Regs::Pipeline) * Maxwell3D::Regs::MaxShaderProgram / 4 + 2);
}

TEST_CASE("DrawX: what a delta draw keeps still ends the chain", "[video_core][drawx]") {
    const auto breaks = [](u32 method) {
        return FastDraw::ClassifyDrawXRegister(method) == DrawXRegister::Breaks;
    };
    // Shader programs only select the pipeline (looked up again under EDEN_NX2_DRAWX_TWIN).
    for (u32 method = MAXWELL3D_REG_INDEX(pipelines);
         method < MAXWELL3D_REG_INDEX(pipelines) +
                      sizeof(Maxwell3D::Regs::Pipeline) * Maxwell3D::Regs::MaxShaderProgram / 4;
         ++method) {
        CHECK(FastDraw::ClassifyDrawXRegister(method) == DrawXRegister::Shader);
    }
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(program_region.address_high)) ==
          DrawXRegister::Shader);
    // Texture and sampler tables, render targets, vertex layout, transform feedback.
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
    // The surface clip only sizes the framebuffer, which delta draws update as full ones do: its
    // own switch (EDEN_NX2_DRAWX_CLIP), breaking without it.
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(surface_clip)) ==
          DrawXRegister::Clip);
    // Fixed-function examples.
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(scissor_test)) ==
          DrawXRegister::PipelineState);
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(viewport_transform)) ==
          DrawXRegister::PipelineState);
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(depth_test_enable)) ==
          DrawXRegister::PipelineState);
}

TEST_CASE("DrawX: the register table follows the switches", "[video_core][drawx]") {
    for (int mask = 0; mask < 64; ++mask) {
        const bool neutral = (mask & 1) != 0;
        const bool pipestate = (mask & 2) != 0;
        const bool twin = (mask & 4) != 0;
        const bool dynstate = (mask & 8) != 0;
        const bool clears = (mask & 16) != 0;
        const bool clip = (mask & 32) != 0;
        ReadSwitches(neutral, pipestate, false, twin, dynstate, false, clears, clip);
        CHECK(FastDraw::g_drawx_registers ==
              (neutral || pipestate || twin || dynstate || clears || clip));
        for (u32 method = 0; method < NumRegs; ++method) {
            const DrawXRegister full = FastDraw::ClassifyDrawXRegister(method);
            DrawXRegister expected = full;
            if ((full == DrawXRegister::Neutral && !neutral) ||
                (full == DrawXRegister::PipelineState && !pipestate) ||
                (full == DrawXRegister::Shader && !twin) ||
                ((full == DrawXRegister::DynamicState || full == DrawXRegister::DynamicPipeline) &&
                 !dynstate) ||
                (full == DrawXRegister::Clear && !clears) ||
                (full == DrawXRegister::Clip && !clip)) {
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

TEST_CASE("DrawX: uniform keep chains end at draws that never bound their uniforms",
          "[video_core][drawx]") {
    FastDraw::KeepChain chain;
    // The first draw has no previous draw to keep by.
    u64 taken = chain.Take();
    CHECK(taken == 0);
    CHECK(!chain.Keeps(taken, 5, false));
    // Then each draw keeps while g_keep_serial stays and no write list overflowed; a draw that
    // could not keep still links the next one.
    taken = chain.Take();
    CHECK(chain.Keeps(taken, 5, false));
    taken = chain.Take();
    CHECK(!chain.Keeps(taken, 5, true));
    taken = chain.Take();
    CHECK(chain.Keeps(taken, 5, false));
    taken = chain.Take();
    CHECK(!chain.Keeps(taken, 6, false));
    CHECK(chain.ended_early == 0);
    // A draw that took g_delta and returned before its uniforms: the next one keeps nothing, even
    // though g_keep_serial is the same (its g_delta lacks what came before the returned draw).
    (void)chain.Take();
    taken = chain.Take();
    CHECK(taken == 0);
    CHECK(chain.ended_early == 1);
    CHECK(!chain.Keeps(taken, 6, false));
    taken = chain.Take();
    CHECK(chain.Keeps(taken, 6, false));
    // Fast draws turned off and on again.
    (void)chain.Take();
    chain.Reset();
    taken = chain.Take();
    CHECK(taken == 0);
    CHECK(chain.ended_early == 1);
    CHECK(!chain.Keeps(taken, 6, false));
}

TEST_CASE("DrawX: dynamic state registers are exactly the depth bias, sub tiling, shadow RAM "
          "control and early-Z ones",
          "[video_core][drawx]") {
    // SET_SHADOW_RAM_CONTROL also as a register write (ProcessDirtyRegisters breaks before the
    // method's own handler runs).
    const std::array<u32, 4> depth_bias{
        MAXWELL3D_REG_INDEX(depth_bias), MAXWELL3D_REG_INDEX(slope_scale_depth_bias),
        MAXWELL3D_REG_INDEX(depth_bias_clamp), MAXWELL3D_REG_INDEX(shadow_ram_control)};
    u32 dynamic = 0;
    for (u32 method = 0; method < NumRegs; ++method) {
        const DrawXRegister kind = FastDraw::ClassifyDrawXRegister(method);
        const bool expected = std::ranges::find(depth_bias, method) != depth_bias.end() ||
                              InRange(method, MAXWELL3D_REG_INDEX(sub_tiling_perf),
                                      sizeof(Maxwell3D::Regs::SubTilingPerf)) ||
                              InRange(method, MAXWELL3D_REG_INDEX(opportunistic_early_z),
                                      sizeof(Maxwell3D::Regs::OpportunisticEarlyZ));
        CHECK((kind == DrawXRegister::DynamicState) == expected);
        CHECK((kind == DrawXRegister::DynamicPipeline) ==
              (method == MAXWELL3D_REG_INDEX(mandated_early_z)));
        dynamic += kind == DrawXRegister::DynamicState ? 1 : 0;
        if (kind == DrawXRegister::DynamicState || kind == DrawXRegister::DynamicPipeline) {
            // None of them was in a class the other switches already take.
            CHECK(FastDraw::ClassifyBreak(FastDraw::Reason::Register, method) ==
                  FastDraw::ClassBit(FastDraw::BreakClass::Other));
        }
    }
    CHECK(dynamic == 4 + sizeof(Maxwell3D::Regs::SubTilingPerf) / 4 +
                         sizeof(Maxwell3D::Regs::OpportunisticEarlyZ) / 4);
    // Depth bias enables come from the polygon offset registers, in the pipeline key: unchanged.
    CHECK(FastDraw::ClassifyDrawXRegister(MAXWELL3D_REG_INDEX(polygon_offset_point_enable)) ==
          DrawXRegister::PipelineState);
}

TEST_CASE("DrawX: coalesced write lists cover every range and stay bounded",
          "[video_core][drawx]") {
    using FastDraw::Delta;
    u32 random = 7;
    const auto next = [&] {
        random = random * 1664525U + 1013904223U;
        return random;
    };
    u64 merges = 0;
    u64 refusals = 0;
    for (int round = 0; round < 2000; ++round) {
        std::array<u64, Delta::MaxWrites> begins{};
        std::array<u64, Delta::MaxWrites> ends{};
        u32 count = 0;
        std::vector<std::pair<u64, u64>> exact;
        // A ring of constant data with a few far away tables, as games write them.
        const u64 ring = 0x100000ULL + (next() % 64) * 0x10000ULL;
        for (int write = 0; write < 48; ++write) {
            const u32 kind = next() % 8;
            // Tables far away from the ring and from each other: some lists run out of room.
            const u64 base = kind == 0 ? 0x80000000ULL + (next() % 64) * 0x1000000ULL : ring;
            const u64 begin = base + (next() % 0x8000) / 4 * 4;
            const u64 end = begin + 4 + (next() % 64) / 4 * 4;
            const u32 before = count;
            const auto previous_begins = begins;
            const auto previous_ends = ends;
            bool merged = false;
            if (!FastDraw::CoalesceRange(begins, ends, count, begin, end, merged)) {
                ++refusals;
                // Refused only when every merge would add more than the limit of unwritten bytes.
                REQUIRE(count == Delta::MaxWrites);
                for (u32 index = 0; index < count; ++index) {
                    const u64 low = (std::min)(begins[index], begin);
                    const u64 high = (std::max)(ends[index], end);
                    CHECK(high - low - (ends[index] - begins[index]) - (end - begin) >
                          FastDraw::CoalesceGrowthLimit);
                }
                CHECK(begins == previous_begins);
                CHECK(ends == previous_ends);
                continue;
            }
            exact.emplace_back(begin, end);
            merges += merged ? 1 : 0;
            // A full list can still extend its last range; anything else it takes is a merge.
            CHECK((!merged || before == Delta::MaxWrites));
            CHECK((merged || count == before + 1 || ends[before - 1] == end));
            // Entries only grow.
            for (u32 index = 0; index < before; ++index) {
                CHECK(begins[index] <= previous_begins[index]);
                CHECK(ends[index] >= previous_ends[index]);
            }
            // Every range taken lies inside one entry.
            for (const auto& [first, last] : exact) {
                bool covered = false;
                for (u32 index = 0; index < count && !covered; ++index) {
                    covered = begins[index] <= first && last <= ends[index];
                }
                REQUIRE(covered);
            }
        }
    }
    CHECK(merges > 1000);
    CHECK(refusals > 0);
}

TEST_CASE("DrawX: with COALESCE a full write list no longer ends the chain",
          "[video_core][drawx]") {
    FastDraw::SetEnabledForTest(true);
    const auto write_many = [](u32 count) {
        for (u32 index = 0; index < count; ++index) {
            const u64 begin = 0x200000 + u64{index} * 0x40; // gaps of 0x30 bytes
            FastDraw::NoteConstantData(begin, begin + 0x10);
            FastDraw::NoteDeviceData(0x9000000 + u64{index} * 0x40,
                                     0x9000000 + u64{index} * 0x40 + 0x10);
        }
    };
    // Off: the 17th range overflows and breaks, as before.
    ReadSwitches(false, false, false);
    FastDraw::g_delta = {};
    u64 serial = FastDraw::Serial();
    write_many(40);
    CHECK(FastDraw::Serial() > serial);
    CHECK(FastDraw::g_delta.overflowed);
    CHECK(!FastDraw::g_delta.drawx_skipped);
    // On: no break, the lists cover every range, and the delta is marked for verify attribution.
    ReadSwitches(false, false, false, false, false, true);
    REQUIRE(FastDraw::g_drawx_coalesce);
    REQUIRE(FastDraw::g_drawx_coalesce_verify);
    FastDraw::g_delta = {};
    FastDraw::g_coalesce_verify.Clear();
    FastDraw::g_coalesce_verify.mismatches = 0;
    FastDraw::g_coalesce_verify.checked = 0;
    serial = FastDraw::Serial();
    const u64 merges = FastDraw::g_coalesced_writes;
    write_many(40);
    CHECK(FastDraw::Serial() == serial);
    CHECK(!FastDraw::g_delta.overflowed);
    CHECK(FastDraw::g_delta.drawx_skipped);
    CHECK(FastDraw::g_delta.num_writes == FastDraw::Delta::MaxWrites);
    CHECK(FastDraw::g_coalesced_writes == merges + 2 * (40 - FastDraw::Delta::MaxWrites));
    // Written data reaches what lies under a merged range, and nothing far away.
    const FastDraw::Delta delta = FastDraw::g_delta;
    FastDraw::DeviceSpans spans;
    CHECK(delta.Reaches(0x200000 + 39 * 0x40, 4, spans));
    CHECK(!delta.Reaches(0x400000, 0x100, spans));
    FastDraw::CheckCoalesced(delta);
    CHECK(FastDraw::g_coalesce_verify.checked == 80);
    CHECK(FastDraw::g_coalesce_verify.mismatches == 0);
    // A list that lost a range is a mismatch.
    write_many(2);
    FastDraw::Delta missing = FastDraw::g_delta;
    missing.num_writes = 1;
    FastDraw::CheckCoalesced(missing);
    CHECK(FastDraw::g_coalesce_verify.mismatches == 1);
    CHECK(FastDraw::g_coalesce_verify.gpu.empty());
    // Far apart ranges still overflow (and break) once no merge stays within the limit.
    FastDraw::g_delta = {};
    serial = FastDraw::Serial();
    for (u32 index = 0; index < FastDraw::Delta::MaxWrites + 1; ++index) {
        const u64 begin = u64{index} * 0x1000000;
        FastDraw::NoteConstantData(begin, begin + 0x10);
    }
    CHECK(FastDraw::Serial() > serial);
    CHECK(FastDraw::g_delta.overflowed);
    FastDraw::g_delta = {};
    FastDraw::g_coalesce_verify.Clear();
    ReadSwitches(false, false, false);
}

TEST_CASE("DrawX: deferred calls are neutral only with CALLS and REUSE", "[video_core][drawx]") {
    const auto read = [](bool calls, bool reuse) {
        SetSwitch("EDEN_NX2_DRAW_REUSE", reuse);
        SetSwitch("EDEN_NX2_DRAW_DELTA", false);
        FastDraw::g_draw_delta = false;
        FastDraw::ReadDrawReuseSwitches();
        ReadSwitches(false, false, false);
        SetSwitch("EDEN_NX2_DRAWX_CALLS", calls);
        FastDraw::ReadDrawXSwitches();
    };
    // On the back, inside the call: attribution and statistics only, and only with the switch.
    const auto note_call = [] {
        FastDraw::g_delta = {};
        u64 counter = 0;
        FastDraw::NoteNeutralCall(counter);
        return std::pair{FastDraw::g_delta.drawx_skipped, counter};
    };
    read(false, false);
    CHECK(FastDraw::CallFlags() == 0);
    CHECK(note_call() == std::pair{false, u64{0}});
    read(true, false);
    CHECK(!FastDraw::g_drawx_calls);
    CHECK(FastDraw::CallFlags() == 0);
    CHECK(note_call() == std::pair{false, u64{0}});
    read(false, true);
    CHECK(FastDraw::CallFlags() == 0);
    CHECK(note_call() == std::pair{false, u64{0}});
    read(true, true);
    CHECK(FastDraw::g_drawx_calls);
    CHECK(FastDraw::CallFlags() == FastDraw::SplitFastDrawNeutral);
    const u64 serial = FastDraw::Serial();
    CHECK(note_call() == std::pair{true, u64{1}});
    // Being neutral is the split's flag; noting the call ends nothing by itself.
    CHECK(FastDraw::Serial() == serial);
    read(false, false);
    FastDraw::g_draw_delta = false;
    FastDraw::g_delta = {};
}

TEST_CASE("DrawX: twins by image descriptors resolve the same image state", "[video_core][drawx]") {
    using Vulkan::GraphicsStageInfo;
    using Vulkan::ResolvesSameImageState;
    Shader::Info base{};
    Shader::TextureDescriptor texture{};
    texture.type = Shader::TextureType::Color2D;
    texture.cbuf_index = 1;
    texture.cbuf_offset = 0x20;
    texture.count = 2;
    texture.size_shift = 3;
    base.texture_descriptors.push_back(texture);
    base.constant_buffer_mask = 0x3;
    base.constant_buffer_used_sizes[0] = 0x100;
    // Another program of the stage: other uniform sizes and inputs, the same textures.
    Shader::Info other = base;
    other.constant_buffer_used_sizes[0] = 0x200;
    other.constant_buffer_mask = 0x7;
    const GraphicsStageInfo lhs{base};
    const GraphicsStageInfo rhs{other};
    CHECK(!lhs.Matches(other));
    CHECK(ResolvesSameImageState(lhs, rhs));
    CHECK(ResolvesSameImageState(rhs, lhs));

    // Another slot, type or count is another state.
    for (int variant = 0; variant < 3; ++variant) {
        Shader::Info changed = other;
        auto& desc = changed.texture_descriptors[0];
        if (variant == 0) {
            desc.cbuf_offset = 0x24;
        } else if (variant == 1) {
            desc.type = Shader::TextureType::ColorArray2D;
        } else {
            desc.count = 1;
        }
        CHECK(!ResolvesSameImageState(lhs, GraphicsStageInfo{changed}));
    }
    Shader::Info with_image = other;
    Shader::ImageDescriptor image{};
    image.type = Shader::TextureType::Color2D;
    image.cbuf_index = 1;
    image.cbuf_offset = 0x40;
    image.count = 1;
    with_image.image_descriptors.push_back(image);
    CHECK(!ResolvesSameImageState(lhs, GraphicsStageInfo{with_image}));

    // A storage buffer whose address word is a texture's handle word nulls that texture
    // (HandleMustBeNull): only on the side that has it.
    Shader::StorageBufferDescriptor storage{};
    storage.cbuf_index = 1;
    storage.cbuf_offset = 0x28; // element 1 of the texture (0x20 + (1 << 3))
    storage.count = 1;
    Shader::Info nulling = other;
    nulling.storage_buffers_descriptors.push_back(storage);
    CHECK(!ResolvesSameImageState(lhs, GraphicsStageInfo{nulling}));
    Shader::Info nulling_too = base;
    nulling_too.storage_buffers_descriptors.push_back(storage);
    CHECK(ResolvesSameImageState(GraphicsStageInfo{nulling_too}, GraphicsStageInfo{nulling}));
    // A storage buffer elsewhere changes nothing for constant buffer handles.
    Shader::Info elsewhere = other;
    storage.cbuf_offset = 0x80;
    elsewhere.storage_buffers_descriptors.push_back(storage);
    CHECK(ResolvesSameImageState(lhs, GraphicsStageInfo{elsewhere}));

    // Handles computed by programs: the programs, the storage bindings and the hash must match.
    Shader::Info programmed = base;
    programmed.texture_descriptors[0].handle_program = 0;
    programmed.handle_programs.push_back({1, 2, 3});
    programmed.storage_buffers_descriptors.push_back(storage);
    programmed.handle_shader_hash = 0x1234;
    Shader::Info programmed_too = programmed;
    programmed_too.constant_buffer_used_sizes[0] = 0x300;
    const GraphicsStageInfo program_lhs{programmed};
    CHECK(ResolvesSameImageState(program_lhs, GraphicsStageInfo{programmed_too}));
    Shader::Info other_program = programmed_too;
    other_program.handle_programs[0] = {1, 2, 4};
    CHECK(!ResolvesSameImageState(program_lhs, GraphicsStageInfo{other_program}));
    Shader::Info other_storage = programmed_too;
    other_storage.storage_buffers_descriptors[0].cbuf_offset = 0x90;
    CHECK(!ResolvesSameImageState(program_lhs, GraphicsStageInfo{other_storage}));
    Shader::Info other_hash = programmed_too;
    other_hash.handle_shader_hash = 0x4321;
    CHECK(!ResolvesSameImageState(program_lhs, GraphicsStageInfo{other_hash}));
}

TEST_CASE("DrawX: clear and clip registers are exactly CLEAR_SURFACE and SURFACE_CLIP",
          "[video_core][drawx]") {
    u32 clears = 0;
    u32 clips = 0;
    for (u32 method = 0; method < NumRegs; ++method) {
        const DrawXRegister kind = FastDraw::ClassifyDrawXRegister(method);
        CHECK((kind == DrawXRegister::Clear) == (method == MAXWELL3D_REG_INDEX(clear_surface)));
        CHECK((kind == DrawXRegister::Clip) == InRange(method, MAXWELL3D_REG_INDEX(surface_clip),
                                                       sizeof(Maxwell3D::Regs::SurfaceClip)));
        clears += kind == DrawXRegister::Clear ? 1 : 0;
        clips += kind == DrawXRegister::Clip ? 1 : 0;
        if (kind == DrawXRegister::Clear || kind == DrawXRegister::Clip) {
            // Not in a class another switch takes already.
            CHECK(FastDraw::ClassifyBreak(FastDraw::Reason::Register, method) ==
                  FastDraw::ClassBit(FastDraw::BreakClass::Other));
        }
    }
    CHECK(clears == 1);
    CHECK(clips == 2);
    // Off: they break as before; on: no break, attributed to the switches.
    ReadSwitches(false, false, false);
    CHECK(FastDraw::DrawXRegisterClass(MAXWELL3D_REG_INDEX(clear_surface)) ==
          DrawXRegister::Breaks);
    CHECK(FastDraw::DrawXRegisterClass(MAXWELL3D_REG_INDEX(surface_clip)) == DrawXRegister::Breaks);
    ReadSwitches(false, false, false, false, false, false, true, true);
    CHECK(FastDraw::g_drawx_clears);
    CHECK(FastDraw::g_drawx_clip);
    CHECK(FastDraw::g_drawx_registers);
    CHECK(FastDraw::DrawXRegisterClass(MAXWELL3D_REG_INDEX(clear_surface)) == DrawXRegister::Clear);
    CHECK(FastDraw::DrawXRegisterClass(MAXWELL3D_REG_INDEX(surface_clip) + 1) ==
          DrawXRegister::Clip);
    ReadSwitches(false, false, false);
}

TEST_CASE("DrawX: CALLS2 flags only with CALLS2 and REUSE", "[video_core][drawx]") {
    const auto read = [](bool calls2, bool reuse) {
        SetSwitch("EDEN_NX2_DRAW_REUSE", reuse);
        SetSwitch("EDEN_NX2_DRAW_DELTA", false);
        FastDraw::g_draw_delta = false;
        FastDraw::ReadDrawReuseSwitches();
        ReadSwitches(false, false, false);
        SetSwitch("EDEN_NX2_DRAWX_CALLS2", calls2);
        FastDraw::ReadDrawXSwitches();
    };
    read(false, false);
    CHECK(FastDraw::CallFlags2() == 0);
    read(true, false);
    CHECK(!FastDraw::g_drawx_calls2);
    CHECK(FastDraw::CallFlags2() == 0);
    read(false, true);
    CHECK(FastDraw::CallFlags2() == 0);
    CHECK(FastDraw::CallFlags() == 0);
    read(true, true);
    CHECK(FastDraw::g_drawx_calls2);
    CHECK(FastDraw::CallFlags2() == FastDraw::SplitFastDrawNeutral);
    // Independent of CALLS.
    CHECK(FastDraw::CallFlags() == 0);
    read(false, false);
    FastDraw::g_draw_delta = false;
}

TEST_CASE("DrawX: call site names keep the lambda's file and line or the type",
          "[video_core][drawx]") {
    namespace Sites = VideoCommon::Nx2CallSites;
    CHECK(Sites::ShortName("const char *VideoCommon::Nx2CallSites::NameOf() [Func = (lambda at "
                           "C:\\work\\lichii\\src\\video_core\\engines\\puller.cpp:871:13)]") ==
          "video_core\\engines\\puller.cpp:871:13");
    CHECK(Sites::ShortName("const char *VideoCommon::Nx2CallSites::NameOf() [Func = (lambda at "
                           "/home/x/src/lichii/src/video_core/gpu_thread.cpp:232:21)]") ==
          "video_core/gpu_thread.cpp:232:21");
    CHECK(
        Sites::ShortName("constexpr const char* VideoCommon::Nx2CallSites::NameOf() [with Func = "
                         "Tegra::Control::Scheduler::BindChannel(Tegra::GPU&, s32)::<lambda()>]") ==
        "Tegra::Control::Scheduler::BindChannel(Tegra::GPU&, s32)::<lambda()>");
    CHECK(Sites::ShortName("const char *__cdecl VideoCommon::Nx2CallSites::NameOf<struct "
                           "Tegra::Engines::RunAheadCheck>(void) noexcept") ==
          "struct Tegra::Engines::RunAheadCheck");
    CHECK(Sites::ShortName("plain") == "plain");
    [[maybe_unused]] const auto lambda = [] {};
    const std::string_view own = Sites::ShortName(Sites::NameOf<decltype(lambda)>());
#if defined(__clang__)
    CHECK(own.find("drawx.cpp:") != std::string_view::npos);
    CHECK(own.find("src") == std::string_view::npos);
#elif defined(__GNUC__)
    CHECK(own.find("lambda") != std::string_view::npos);
#endif
    CHECK(!own.empty());
}

TEST_CASE("DrawX: call sites count the calls and the full draws they broke",
          "[video_core][drawx]") {
    namespace Sites = VideoCommon::Nx2CallSites;
    Sites::ResetForTests();
    Sites::g_enabled = true;
    constexpr u64 macro_thunk = 0x1000;
    constexpr u64 bind_thunk = 0x2000;
    Sites::Name(macro_thunk, "[Func = (lambda at /src/video_core/engines/maxwell_3d.cpp:1865:9)]");
    Sites::Name(bind_thunk, "[Func = (lambda at /src/video_core/control/scheduler.cpp:27:13)]");
    CHECK(Sites::SiteCountForTests() == 2);

    // A back macro alone before a draw: its only cause and its last break.
    Sites::NoteBreak(macro_thunk);
    Sites::NoteBreak(macro_thunk);
    auto breaks = Sites::TakeDraw();
    CHECK(breaks.sites == 1);
    CHECK(breaks.last == 1);
    CHECK(!breaks.other);
    Sites::NoteFullDraw(breaks, true);
    // A bind, then a macro: both sites (several), the macro last.
    Sites::NoteBreak(bind_thunk);
    Sites::NoteBreak(macro_thunk);
    breaks = Sites::TakeDraw();
    CHECK(breaks.sites == 3);
    Sites::NoteFullDraw(breaks, true);
    // A macro, then a register break: the call is neither the last break nor the only cause.
    Sites::NoteBreak(macro_thunk);
    FastDraw::Break(FastDraw::Reason::Register, 0x123);
    breaks = Sites::TakeDraw();
    CHECK(breaks.other);
    Sites::NoteFullDraw(breaks, false);
    // A register break, then a bind: the bind is the last break, not the only cause.
    FastDraw::Break(FastDraw::Reason::Register, 0x123);
    FastDraw::Break(FastDraw::Reason::Call);
    Sites::NoteBreak(bind_thunk);
    breaks = Sites::TakeDraw();
    CHECK(breaks.other);
    Sites::NoteFullDraw(breaks, true);
    // A call of a site never named: counted apart.
    Sites::NoteBreak(0x3000);
    breaks = Sites::TakeDraw();
    CHECK(breaks.unnamed);
    CHECK(breaks.last == 0);
    Sites::NoteFullDraw(breaks, true);
    // Neutral calls are counted as such.
    Sites::NoteNeutral(bind_thunk);
    // Nothing since: an empty set.
    breaks = Sites::TakeDraw();
    CHECK(breaks.sites == 0);
    CHECK(!breaks.other);

    const auto macro = Sites::CountsForTests(0);
    const auto bind = Sites::CountsForTests(1);
    CHECK(Sites::ShortName(macro.name) == "video_core/engines/maxwell_3d.cpp:1865:9");
    CHECK(macro.breaks == 4);
    CHECK(macro.last_draws == 2);
    CHECK(macro.sole_draws == 1);
    CHECK(bind.breaks == 2);
    CHECK(bind.neutral == 1);
    CHECK(bind.last_draws == 1);
    CHECK(bind.sole_draws == 0);
    // The window's report logs and starts over.
    Sites::Report(1.0);
    CHECK(Sites::CountsForTests(0).breaks == 0);
    CHECK(Sites::CountsForTests(0).last_draws == 0);
    CHECK(Sites::SiteCountForTests() == 2);

    // Named once per function object type.
    [[maybe_unused]] const auto first = [] {};
    Sites::NameOnce<decltype(first), false>(0x4000);
    Sites::NameOnce<decltype(first), false>(0x4000);
    CHECK(Sites::SiteCountForTests() == 3);
    Sites::g_enabled = false;
    Sites::ResetForTests();
}
