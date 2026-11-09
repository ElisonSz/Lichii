// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "shader_recompiler/environment.h"

namespace Vulkan {

namespace PipelineKeep {
inline bool Switch(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    return value ? std::string_view{value} == "1" : fallback;
}
inline std::atomic_bool enabled{Switch("EDEN_NX2_PIPELINE_KEEP", true)};
inline std::atomic_bool check{Switch("EDEN_NX2_PIPELINE_KEEP_CHECK", false)};
inline std::atomic<u64> count{}, bytes{}, allocated{}, raw_bytes{};
inline std::atomic<u64> restored{}, restore_ns{}, restore_gpu_ns{}, built{}, build_ns{}, checks{},
    mismatches{}, rejected{};
inline void ReadSwitch() {
    enabled.store(Switch("EDEN_NX2_PIPELINE_KEEP", true), std::memory_order_relaxed);
    check.store(Switch("EDEN_NX2_PIPELINE_KEEP_CHECK", false), std::memory_order_relaxed);
}
} // namespace PipelineKeep

/// Mesh emulation without VK_EXT_mesh_shader: the mesh program becomes a compute shader that
/// writes a storage buffer, and this auxiliary vertex shader draws its output.
struct MeshEmulationSetup {
    std::vector<u32> vertex_spirv;
    u32 output_binding{};
    u32 block_words{};    ///< output words per mesh workgroup
    u32 max_primitives{}; ///< triangles per mesh workgroup
    // Task output feeds the mesh program at mesh_stage through the distribution pass.
    bool has_task{};
    u32 mesh_stage{};
    u32 payload_stride{};
    u32 payload_binding{};
    u32 offsets_binding{};
    u32 args_binding{};
    std::vector<u32> distribution_spirv;
    u32 max_vertices{};
    u32 vertex_lines{};
    std::array<u32, 4> position_lines{~0U, ~0U, ~0U, ~0U};

    bool operator==(const MeshEmulationSetup&) const = default;
};

/// A translation's actual environment calls, including negative replacement/pointer answers.
struct PipelineEnvironment {
    Shader::ProgramHeader header{};
    Shader::ProgramHeader primitive_header{};
    std::array<u32, 8> passthrough{};
    std::array<u32, 3> mesh_registers{};
    Shader::Stage stage{};
    u32 start{};
    bool proprietary{};
    // Each call is {operation, argument 0, argument 1, argument 2, answer}.
    std::vector<std::array<u64, 5>> reads;

    bool Matches(Shader::Environment& environment) const;
};

class PipelineEnvironmentRecorder final : public Shader::Environment {
public:
    explicit PipelineEnvironmentRecorder(Shader::Environment& source);
    PipelineEnvironment Take() {
        return std::move(record);
    }
    u64 ReadInstruction(u32 address) override;
    u32 TranslateProgramAddress(u32 address) const override;
    bool IsNativeProgram() const override;
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override;
    bool IsCbufIndirectStorageBufferPointer(u32 bank, u32 offset, u32 indirect) override;
    u32 ReadCbufValue(u32 bank, u32 offset) override;
    Shader::TextureType ReadTextureType(u32 handle) override;
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;
    bool IsTexturePixelFormatInteger(u32 handle) override;
    u32 ReadViewportTransformState() override;
    u32 TextureBoundBuffer() const override;
    u32 LocalMemorySize() const override;
    u32 SharedMemorySize() const override;
    std::array<u32, 3> WorkgroupSize() const override;
    bool HasHLEMacroState() const override;
    bool HasNativeInstructions() const override;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;
    std::vector<std::pair<u32, u32>> NullStoragePointerSlots() const override;
    std::vector<u64> MappedStoragePointerKeys() const override;
    void Dump(u64 pipeline_hash, u64 shader_hash) override;

private:
    u64 Save(u64 operation, u64 a, u64 b, u64 c, u64 answer) const;
    Shader::Environment& source;
    mutable PipelineEnvironment record;
};

/// Before-emission Info is what GraphicsPipeline copies for descriptor writes and layouts.
struct GraphicsPipelineInput {
    static constexpr size_t Stages = 5;
    std::array<u32, 8> build_settings{};
    std::array<bool, Stages> present{};
    std::array<Shader::Info, Stages> infos;
    std::array<std::vector<u32>, Stages> spirv;
    std::optional<MeshEmulationSetup> mesh;
    std::vector<PipelineEnvironment> environments;

    bool SameShaders(const GraphicsPipelineInput& other) const;
    bool Matches(std::span<Shader::Environment* const> current) const;
};

/// Session-local, deterministic field encoding; never stores pointers or container object bytes.
class GraphicsPipelineRecipe {
public:
    GraphicsPipelineRecipe() = default;
    ~GraphicsPipelineRecipe();
    GraphicsPipelineRecipe(const GraphicsPipelineRecipe&) = delete;
    GraphicsPipelineRecipe& operator=(const GraphicsPipelineRecipe&) = delete;
    void Store(const GraphicsPipelineInput& input);
    std::optional<GraphicsPipelineInput> Restore() const;
    size_t Bytes() const {
        return compressed.size();
    }
    size_t AllocatedBytes() const {
        return compressed.capacity() + sizeof(*this);
    }
    size_t RawBytes() const {
        return raw_bytes;
    }

private:
    size_t raw_bytes{};
    std::vector<u8> compressed;
};

} // namespace Vulkan
