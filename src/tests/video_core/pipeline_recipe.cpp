// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "common/zstd_compression.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "video_core/native_lowering.h"
#include "video_core/renderer_vulkan/pipeline_recipe.h"
#include "video_core/shader_stage_translation.h"

namespace {
using VideoCommon::ShaderPools;
constexpr u64 MakeCbufKey(u32 index, u32 offset) {
    return (static_cast<u64>(index) << 32) | offset;
}

constexpr u64 MakeIndirectCbufKey(u32 index, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (static_cast<u64>(indirect & 0x3fffff) << 40) |
           MakeCbufKey(index, offset);
}

/// A captured environment of tools/shader-sweep/fixtures. It answers as FileEnvironment answers
/// from the record it was taken from (recorded values only), without the rest of video_core.
class FixtureEnvironment final : public Shader::Environment {
public:
    explicit FixtureEnvironment(const nlohmann::json& env) {
        const auto& parameters = env.at("parameters");
        local_memory_size = parameters.at(0);
        texture_bound = parameters.at(1);
        start_address = parameters.at(2);
        read_lowest = parameters.at(3);
        read_highest = parameters.at(4);
        viewport_transform_state = parameters.at(5);
        stage = static_cast<Shader::Stage>(parameters.at(6).get<u32>());
        for (const auto& word : env.at("code")) {
            code.push_back(std::stoull(word.get<std::string>(), nullptr, 16));
        }
        for (const auto& entry : env.at("texture_types")) {
            texture_types.emplace(entry.at(0), static_cast<Shader::TextureType>(entry.at(1)));
        }
        for (const auto& entry : env.at("texture_formats")) {
            texture_formats.emplace(entry.at(0),
                                    static_cast<Shader::TexturePixelFormat>(entry.at(1)));
        }
        for (const auto& entry : env.at("cbuf_values")) {
            cbuf_values.emplace(entry.at(0), entry.at(1));
        }
        cbuf_values.erase(MakeCbufKey(~0U, 0)); // The record's read properties
        for (const auto& entry : env.at("storage_pointers")) {
            storage_pointers.emplace(entry.at(0), entry.at(1));
        }
        for (const auto& entry : env.at("replacements")) {
            replacements.emplace(entry.at(0), static_cast<Shader::ReplaceConstant>(entry.at(1)));
        }
        const std::string tail_hex = env.at("tail");
        std::vector<u8> tail;
        for (size_t i = 0; i + 1 < tail_hex.size(); i += 2) {
            tail.push_back(static_cast<u8>(std::stoul(tail_hex.substr(i, 2), nullptr, 16)));
        }
        size_t offset = 0;
        const auto read = [&](auto& value) {
            REQUIRE(offset + sizeof(value) <= tail.size());
            std::memcpy(&value, tail.data() + offset, sizeof(value));
            offset += sizeof(value);
        };
        if (stage == Shader::Stage::Compute) {
            read(workgroup_size);
            read(shared_memory_size);
        } else {
            read(sph);
            if (stage == Shader::Stage::Geometry) {
                read(gp_passthrough_mask);
            }
        }
        is_proprietary_driver = texture_bound == 2 || IsNativeProgram();
    }

    /// Where PipelineCache starts the control flow graph of this stage
    u32 CfgOffset() const {
        return start_address +
               (stage == Shader::Stage::Compute ? 0 : u32{sizeof(Shader::ProgramHeader)});
    }

    u64 ReadInstruction(u32 address) override {
        if (address < read_lowest || address > read_highest) {
            throw Shader::LogicError("Out of bounds address {}", address);
        }
        return code[(address - read_lowest) / sizeof(u64)];
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) {
            return *translated;
        }
        if (Shader::Ampere::HasNativeAddressMap(code)) {
            throw Shader::RuntimeError("Invalid cached native indirect branch address {:#x}",
                                       address);
        }
        return address;
    }
    bool IsNativeProgram() const override {
        return Shader::Ampere::HasNativeAddressMap(code);
    }
    bool HasNativeInstructions() const override {
        return IsNativeProgram();
    }
    bool IsCbufStorageBufferPointer(u32 index, u32 offset) override {
        return Lookup(storage_pointers, MakeCbufKey(index, offset)) != 0;
    }
    bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) override {
        return Lookup(storage_pointers, MakeIndirectCbufKey(index, offset, indirect)) != 0;
    }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        return Lookup(cbuf_values, MakeCbufKey(index, offset));
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return Lookup(texture_types, handle);
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return Lookup(texture_formats, handle);
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        using F = Shader::TexturePixelFormat;
        switch (ReadTexturePixelFormat(handle)) {
        case F::A8B8G8R8_SINT:
        case F::A8B8G8R8_UINT:
        case F::A2B10G10R10_UINT:
        case F::R8_SINT:
        case F::R8_UINT:
        case F::R16G16B16A16_SINT:
        case F::R16G16B16A16_UINT:
        case F::R32G32B32A32_UINT:
        case F::R32G32B32A32_SINT:
        case F::R32G32_SINT:
        case F::R16_UINT:
        case F::R16_SINT:
        case F::R16G16_UINT:
        case F::R16G16_SINT:
        case F::R8G8_SINT:
        case F::R8G8_UINT:
        case F::R32G32_UINT:
        case F::R32_UINT:
        case F::R32_SINT:
            return true;
        default:
            return false;
        }
    }
    u32 ReadViewportTransformState() override {
        return viewport_transform_state;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return local_memory_size;
    }
    u32 SharedMemorySize() const override {
        return shared_memory_size;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return workgroup_size;
    }
    bool HasHLEMacroState() const override {
        return !replacements.empty();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        const auto it = replacements.find(MakeCbufKey(bank, offset));
        return it == replacements.end() ? std::nullopt : std::optional{it->second};
    }
    void Dump(u64, u64) override {}

private:
    template <typename Map>
    static typename Map::mapped_type Lookup(const Map& map, typename Map::key_type key) {
        const auto it = map.find(key);
        if (it == map.end()) {
            throw Shader::LogicError("Unrecorded environment value {:#x}", u64{key});
        }
        return it->second;
    }

    std::vector<u64> code;
    std::unordered_map<u32, Shader::TextureType> texture_types;
    std::unordered_map<u32, Shader::TexturePixelFormat> texture_formats;
    std::unordered_map<u64, u32> cbuf_values;
    std::unordered_map<u64, u8> storage_pointers;
    std::unordered_map<u64, Shader::ReplaceConstant> replacements;
    u32 local_memory_size{};
    u32 texture_bound{};
    u32 read_lowest{};
    u32 read_highest{};
    u32 viewport_transform_state{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};
};

struct Fixture {
    std::string name;
    nlohmann::json environment;
};

std::vector<Fixture> LoadFixtures() {
    std::vector<Fixture> fixtures;
    for (const auto& entry : std::filesystem::directory_iterator{SHADER_SWEEP_FIXTURES}) {
        if (entry.path().extension() != ".json") {
            continue;
        }
        std::ifstream file{entry.path()};
        fixtures.push_back(
            {entry.path().stem().string(), nlohmann::json::parse(file).at("environment")});
    }
    std::ranges::sort(fixtures, {}, &Fixture::name);
    REQUIRE(fixtures.size() >= 4);
    return fixtures;
}

/// The offline host and profile of tools/shader-sweep
Shader::HostTranslateInfo Host() {
    Shader::HostTranslateInfo host{};
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_conditional_barrier = true;
    host.support_viewport_index_layer = true;
    host.support_geometry_shader_passthrough = true;
    host.ApplyDescriptorLimitPolicy();
    return host;
}

Shader::Profile Profile(const Shader::HostTranslateInfo& host) {
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010400;
    profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = host.min_ssbo_alignment;
    profile.support_int64 = true;
    profile.support_int16 = true;
    profile.support_int8 = true;
    profile.support_vote = true;
    profile.support_shuffle_relative = true;
    profile.support_geometry_streams = true;
    profile.support_geometry_shader_passthrough = true;
    profile.support_viewport_index_layer_non_geometry = true;
    profile.support_multi_viewport = true;
    profile.max_user_clip_distances = 8;
    profile.max_subgroup_size = 32;
    profile.support_typeless_image_loads = true;
    return profile;
}

// Native instruction programs from shader_cache_v2_programs.cpp. Use valid minimal stage
// headers here: that test fills unrelated header bytes to exercise program-record identity.
std::vector<Fixture> NativeFixtures() {
    using Shader::Ampere::Instruction;
    constexpr Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const std::vector<Instruction> plain{{{0x06007947, 0x10 | 3, 0x0b800000, 0}}, exit, exit};
    const std::vector<Instruction> pc{{{0x00007918, 0, 0, 0}}, {{0x0014734e, 0, 0, 0}}, exit};
    const Instruction control{{0xff057321, 0x000018ff, 0x00009000, 0x000e6200}};
    const Instruction evaluation{{0xff057321, 0x000018ff, 0x00001000, 0x000e6200}};
    std::vector<Fixture> fixtures;
    for (const auto stage :
         {Shader::Stage::VertexB, Shader::Stage::Fragment, Shader::Stage::Geometry,
          Shader::Stage::TessellationControl, Shader::Stage::TessellationEval}) {
        const bool geometry = stage == Shader::Stage::Geometry;
        const bool tess_control = stage == Shader::Stage::TessellationControl;
        const bool tess_eval = stage == Shader::Stage::TessellationEval;
        const auto native = geometry       ? pc
                            : tess_control ? std::vector{control, exit}
                            : tess_eval    ? std::vector{evaluation, exit}
                                           : plain;
        const VideoCommon::NativeLoweringInputs inputs{
            .geometry = geometry,
            .tessellation_stage = tess_control ? Shader::Ampere::TessellationStage::Control
                                  : tess_eval  ? Shader::Ampere::TessellationStage::Evaluation
                                               : Shader::Ampere::TessellationStage::None,
            .native_code_address = 0x2000};
        auto lowered = VideoCommon::LowerNativeProgram(native, inputs);
        REQUIRE(static_cast<bool>(lowered));
        Shader::ProgramHeader header{};
        header.common3.output_topology.Assign(Shader::OutputTopology::PointList);
        header.common4.max_output_vertices.Assign(1);
        const auto code = VideoCommon::MakeLoweredNativeCode(native, header, 128, {}, lowered);
        nlohmann::json environment;
        environment["parameters"] = {0,
                                     2,
                                     0,
                                     0,
                                     static_cast<u32>((code->code.size() - 1) * sizeof(u64)),
                                     1,
                                     static_cast<u32>(stage)};
        environment["code"] = nlohmann::json::array();
        for (const u64 word : code->code) {
            environment["code"].push_back(fmt::format("{:016x}", word));
        }
        for (const auto* field : {"texture_types", "texture_formats", "cbuf_values",
                                  "storage_pointers", "replacements"}) {
            environment[field] = nlohmann::json::array();
        }
        std::string tail;
        for (const u8 byte : std::span{reinterpret_cast<const u8*>(&header), sizeof(header)}) {
            tail += fmt::format("{:02x}", byte);
        }
        if (geometry) {
            tail += std::string(64, '0');
        }
        environment["tail"] = tail;
        fixtures.push_back({"v2 program stage " + std::to_string(static_cast<u32>(stage)),
                            std::move(environment)});
    }
    return fixtures;
}

Vulkan::GraphicsPipelineInput TranslateFixture(const Fixture& fixture) {
    FixtureEnvironment environment{fixture.environment};
    Vulkan::PipelineEnvironmentRecorder recorded{environment};
    ShaderPools pools;
    const auto host = Host();
    Shader::Maxwell::Flow::CFG cfg{recorded, pools.flow_block, environment.CfgOffset()};
    auto program = Shader::Maxwell::TranslateProgram(pools.inst, pools.block, recorded, cfg, host);
    Shader::Maxwell::ConvertLegacyToGeneric(program, {});
    Vulkan::GraphicsPipelineInput result;
    const size_t slot = program.stage == Shader::Stage::Fragment              ? 4
                        : program.stage == Shader::Stage::Geometry            ? 3
                        : program.stage == Shader::Stage::TessellationEval    ? 2
                        : program.stage == Shader::Stage::TessellationControl ? 1
                                                                              : 0;
    result.present[slot] = true;
    result.infos[slot] = program.info;
    Shader::Backend::Bindings bindings;
    result.spirv[slot] = Shader::Backend::SPIRV::EmitSPIRV(Profile(host), {}, program, bindings);
    result.environments.push_back(recorded.Take());
    return result;
}
} // namespace

TEST_CASE("Pipeline recipe compresses and restores captured shader creation inputs",
          "[video_core][pipeline_keep][spirv]") {
    size_t raw_total{}, compressed_total{}, allocated_total{}, count{};
    auto fixtures = LoadFixtures();
    auto native = NativeFixtures();
    fixtures.insert(fixtures.end(), native.begin(), native.end());
    for (const auto& fixture : fixtures) {
        INFO(fixture.name);
        const auto input = TranslateFixture(fixture);
        Vulkan::GraphicsPipelineRecipe recipe;
        recipe.Store(input);
        auto restored = recipe.Restore();
        REQUIRE(restored.has_value());
        CHECK(restored->SameShaders(input));
        CHECK(restored->SameShaders(TranslateFixture(fixture)));
        FixtureEnvironment current{fixture.environment};
        std::array<Shader::Environment*, 1> environments{&current};
        CHECK(restored->Matches(environments));
        CHECK(recipe.Bytes() > 0);
        CHECK(recipe.Bytes() < recipe.RawBytes());
        for (int cycle = 0; cycle < 8; ++cycle) {
            recipe.Store(*restored);
            restored = recipe.Restore();
            REQUIRE(restored.has_value());
            CHECK(restored->SameShaders(input));
        }
        raw_total += recipe.RawBytes();
        compressed_total += recipe.Bytes();
        allocated_total += recipe.AllocatedBytes();
        ++count;
        std::cout << "recipe " << fixture.name << ": raw=" << recipe.RawBytes()
                  << " compressed=" << recipe.Bytes() << " allocated=" << recipe.AllocatedBytes()
                  << '\n';
    }
    std::cout << "recipes=" << count << " raw=" << raw_total << " compressed=" << compressed_total
              << " allocated=" << allocated_total
              << " projection24000=" << allocated_total * 24000 / count << '\n';
}

TEST_CASE("Pipeline recipe preserves mesh auxiliary shaders and descriptor handle arrays",
          "[video_core][pipeline_keep]") {
    auto input = TranslateFixture(LoadFixtures().front());
    auto& info = input.infos[0];
    info.texture_descriptors.push_back({.type = Shader::TextureType::ColorArray2D,
                                        .has_secondary = true,
                                        .cbuf_index = 2,
                                        .cbuf_offset = 24,
                                        .secondary_cbuf_index = 3,
                                        .secondary_cbuf_offset = 32,
                                        .count = 7,
                                        .size_shift = 3,
                                        .handle_program = 0});
    info.image_descriptors.push_back({.type = Shader::TextureType::Color2D,
                                      .format = Shader::ImageFormat::R32_UINT,
                                      .is_written = true,
                                      .cbuf_index = 4,
                                      .count = 3,
                                      .is_byte_addressed = true,
                                      .handle_program = 1});
    info.texture_buffer_descriptors.push_back({.cbuf_index = 1, .count = 2});
    info.image_buffer_descriptors.push_back({.is_integer = true, .count = 4});
    info.handle_programs = {{1, 2, 3}, {7, 8, 9, 10}};
    info.storage_buffers_descriptors.push_back(
        {.cbuf_index = 4,
         .cbuf_offset = 16,
         .count = 1,
         .is_written = true,
         .indirect_offset = 32,
         .write_extent = {.bounded = true, .const_end = 128}});
    info.legacy_stores_mapping.emplace(Shader::IR::Attribute::PositionX,
                                       Shader::IR::Attribute::Generic0X);
    info.loads.Set(Shader::IR::Attribute::PositionX);
    info.stores.Set(Shader::IR::Attribute::Generic0X);
    info.nvn_buffer_used.set(15);
    input.mesh.emplace();
    input.mesh->has_task = true;
    input.mesh->vertex_spirv = {1, 2, 3};
    input.mesh->distribution_spirv = {4, 5, 6};
    input.mesh->mesh_stage = 1;
    input.mesh->block_words = 256;
    input.mesh->max_vertices = 32;
    input.mesh->max_primitives = 16;
    input.mesh->position_lines = {0, 1, 2, 3};
    input.mesh->payload_stride = 17;
    input.mesh->payload_binding = 7;
    input.mesh->offsets_binding = 8;
    input.mesh->args_binding = 9;
    Vulkan::GraphicsPipelineRecipe recipe;
    recipe.Store(input);
    const auto restored = recipe.Restore();
    REQUIRE(restored.has_value());
    CHECK(restored->SameShaders(input));
    auto changed = *restored;
    changed.mesh->distribution_spirv[1] ^= 1;
    CHECK_FALSE(changed.SameShaders(input));
    changed = *restored;
    changed.infos[0].handle_programs[0][1] ^= 1;
    CHECK_FALSE(changed.SameShaders(input));
    changed = *restored;
    changed.infos[0].storage_buffers_descriptors.back().indirect_offset += 4;
    CHECK_FALSE(changed.SameShaders(input));
    changed = *restored;
    changed.spirv[0].push_back(99);
    CHECK_FALSE(changed.SameShaders(input));
}

TEST_CASE("Pipeline recipe rejects changed environment reads", "[video_core][pipeline_keep]") {
    const auto fixtures = LoadFixtures();
    size_t changed_reads{};
    for (const auto& fixture : fixtures) {
        auto input = TranslateFixture(fixture);
        for (auto& call : input.environments[0].reads) {
            FixtureEnvironment current{fixture.environment};
            std::array<Shader::Environment*, 1> environments{&current};
            call[4] ^= 1;
            CHECK_FALSE(input.Matches(environments));
            call[4] ^= 1;
            ++changed_reads;
        }
        FixtureEnvironment current{fixture.environment};
        std::array<Shader::Environment*, 1> environments{&current};
        input.environments[0].mesh_registers[0] ^= 1;
        CHECK_FALSE(input.Matches(environments));
        input.environments[0].mesh_registers[0] ^= 1;
        input.environments[0].header.common0.shader_type.Assign(7);
        CHECK_FALSE(input.Matches(environments));
    }
    CHECK(changed_reads > 0);
    std::cout << "environment reads changed=" << changed_reads << '\n';
}

TEST_CASE("Pipeline recipe records negative answers and every environment operation",
          "[video_core][pipeline_keep]") {
    auto fixture = NativeFixtures().front();
    auto& json = fixture.environment;
    json["cbuf_values"].push_back({MakeCbufKey(0, 0), 42});
    json["storage_pointers"].push_back({MakeCbufKey(0, 0), 0});
    json["storage_pointers"].push_back({MakeIndirectCbufKey(0, 0, 8), 0});
    json["texture_types"].push_back({7, static_cast<u32>(Shader::TextureType::Color2D)});
    json["texture_formats"].push_back({7, static_cast<u32>(Shader::TexturePixelFormat::R32_UINT)});
    FixtureEnvironment source{json};
    Vulkan::PipelineEnvironmentRecorder recorder{source};
    CHECK(recorder.IsNativeProgram());
    CHECK(recorder.HasNativeInstructions());
    CHECK_FALSE(recorder.IsCbufStorageBufferPointer(0, 0));
    CHECK_FALSE(recorder.IsCbufIndirectStorageBufferPointer(0, 0, 8));
    CHECK(recorder.ReadCbufValue(0, 0) == 42);
    CHECK(recorder.ReadTextureType(7) == Shader::TextureType::Color2D);
    CHECK(recorder.ReadTexturePixelFormat(7) == Shader::TexturePixelFormat::R32_UINT);
    CHECK(recorder.IsTexturePixelFormatInteger(7));
    CHECK(recorder.ReadViewportTransformState() == source.ReadViewportTransformState());
    CHECK(recorder.TextureBoundBuffer() == source.TextureBoundBuffer());
    CHECK(recorder.LocalMemorySize() == source.LocalMemorySize());
    CHECK(recorder.SharedMemorySize() == source.SharedMemorySize());
    CHECK(recorder.WorkgroupSize() == source.WorkgroupSize());
    CHECK_FALSE(recorder.HasHLEMacroState());
    CHECK_FALSE(recorder.GetReplaceConstBuffer(0, 0));
    auto dependencies = recorder.Take();
    REQUIRE(dependencies.reads.size() == 17);
    CHECK(dependencies.Matches(source));
    for (auto& call : dependencies.reads) {
        call[4] ^= 1;
        CHECK_FALSE(dependencies.Matches(source));
        call[4] ^= 1;
    }
}

TEST_CASE("Pipeline recipe compression checks size and damaged frames",
          "[video_core][pipeline_keep]") {
    const std::vector<u8> raw{0, 1, 2, 3, 0, 1, 2, 3};
    auto compressed = Common::Compression::CompressDataZSTD(raw.data(), raw.size(), 1);
    REQUIRE(!compressed.empty());
    auto restored = Common::Compression::TryDecompressDataZSTD(compressed, raw.size());
    REQUIRE(restored.has_value());
    CHECK(*restored == raw);
    CHECK_FALSE(Common::Compression::TryDecompressDataZSTD(compressed, raw.size() - 1));
    compressed.pop_back();
    CHECK_FALSE(Common::Compression::TryDecompressDataZSTD(compressed, raw.size()));
    Vulkan::GraphicsPipelineRecipe empty;
    CHECK_FALSE(empty.Restore());
}
