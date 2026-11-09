// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "common/zstd_compression.h"
#include "video_core/renderer_vulkan/pipeline_recipe.h"

namespace Vulkan {
namespace {
enum Read : u64 {
    Native,
    Pointer,
    Indirect,
    Cbuf,
    Texture,
    Format,
    Integer,
    Viewport,
    Bound,
    Local,
    Shared,
    Workgroup,
    Hle,
    NativeInstructions,
    Replacement,
};

u64 Answer(Shader::Environment& env, const std::array<u64, 5>& call) {
    const u32 a = static_cast<u32>(call[1]);
    const u32 b = static_cast<u32>(call[2]);
    const u32 c = static_cast<u32>(call[3]);
    switch (call[0]) {
    case Native:
        return env.IsNativeProgram();
    case Pointer:
        return env.IsCbufStorageBufferPointer(a, b);
    case Indirect:
        return env.IsCbufIndirectStorageBufferPointer(a, b, c);
    case Cbuf:
        return env.ReadCbufValue(a, b);
    case Texture:
        return static_cast<u64>(env.ReadTextureType(a));
    case Format:
        return static_cast<u64>(env.ReadTexturePixelFormat(a));
    case Integer:
        return env.IsTexturePixelFormatInteger(a);
    case Viewport:
        return env.ReadViewportTransformState();
    case Bound:
        return env.TextureBoundBuffer();
    case Local:
        return env.LocalMemorySize();
    case Shared:
        return env.SharedMemorySize();
    case Workgroup:
        return env.WorkgroupSize()[a];
    case Hle:
        return env.HasHLEMacroState();
    case NativeInstructions:
        return env.HasNativeInstructions();
    case Replacement: {
        const auto result = env.GetReplaceConstBuffer(a, b);
        return result ? static_cast<u64>(*result) : ~u64{0};
    }
    default:
        throw std::runtime_error("Invalid pipeline environment call");
    }
}

// The format is session-local. Scalars use host byte order; containers and descriptors are
// encoded field by field, so padding and allocator state never affect comparisons or storage.
struct Writer {
    std::vector<u8> bytes;
    template <typename T>
    void Scalar(T& value) {
        const auto* data = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), data, data + sizeof(value));
    }
    template <typename T>
    void operator()(T& value);
};
struct Reader {
    std::span<const u8> bytes;
    template <typename T>
    void Scalar(T& value) {
        if (bytes.size() < sizeof(value)) {
            throw std::runtime_error("Truncated pipeline recipe");
        }
        std::memcpy(&value, bytes.data(), sizeof(value));
        bytes = bytes.subspan(sizeof(value));
    }
    template <typename T>
    void operator()(T& value);
};

template <typename A, typename T>
void Fields(A& archive, T& value) {
    if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
        archive.Scalar(value);
    } else if constexpr (requires {
                             value.resize(size_t{});
                             value.size();
                             value.begin();
                         }) {
        u64 count = value.size();
        archive(count);
        if constexpr (std::is_same_v<A, Reader>) {
            if (count > archive.bytes.size()) {
                throw std::runtime_error("Invalid recipe count");
            }
            value.resize(static_cast<size_t>(count));
        }
        for (auto& element : value) {
            archive(element);
        }
    } else if constexpr (requires {
                             value.begin();
                             value.end();
                         }) {
        for (auto& element : value) {
            archive(element);
        }
    } else {
        static_assert(sizeof(T) == 0, "Missing recipe field encoding");
    }
}
template <typename A, size_t N>
void Fields(A& archive, std::bitset<N>& value) {
    for (size_t bit = 0; bit < N; ++bit) {
        bool set = value[bit];
        archive(set);
        if constexpr (std::is_same_v<A, Reader>) {
            value[bit] = set;
        }
    }
}
template <typename A>
void Fields(A& archive, Shader::VaryingState& value) {
    archive(value.mask);
}
template <typename A>
void Fields(A& archive, Shader::Info& value) {
    // Keep this walk complete when Info gains members: layouts and per-draw descriptor writes
    // both consume it, and SameShaders compares the entire object during a verification run.
    archive(value.is_task);
    archive(value.is_mesh);
    archive(value.uses_workgroup_id);
    archive(value.uses_local_invocation_id);
    archive(value.uses_invocation_id);
    archive(value.uses_invocation_info);
    archive(value.uses_sample_id);
    archive(value.uses_is_helper_invocation);
    archive(value.uses_subgroup_invocation_id);
    archive(value.uses_subgroup_shuffles);
    archive(value.uses_quad_shuffles);
    archive(value.uses_patches);
    archive(value.interpolation);
    archive(value.loads);
    archive(value.stores);
    archive(value.passthrough);
    archive(value.legacy_stores_mapping);
    archive(value.loads_indexed_attributes);
    archive(value.stores_frag_color);
    archive(value.frag_color_components);
    archive(value.stores_sample_mask);
    archive(value.stores_frag_depth);
    archive(value.stores_tess_level_outer);
    archive(value.stores_tess_level_inner);
    archive(value.stores_indexed_attributes);
    archive(value.stores_global_memory);
    archive(value.uses_local_memory);
    archive(value.uses_fp16);
    archive(value.uses_fp64);
    archive(value.uses_fp16_denorms_flush);
    archive(value.uses_fp16_denorms_preserve);
    archive(value.uses_fp32_denorms_flush);
    archive(value.uses_fp32_denorms_preserve);
    archive(value.uses_int8);
    archive(value.uses_int16);
    archive(value.uses_int64);
    archive(value.uses_image_1d);
    archive(value.uses_sampled_1d);
    archive(value.uses_sparse_residency);
    archive(value.uses_demote_to_helper_invocation);
    archive(value.uses_subgroup_vote);
    archive(value.uses_subgroup_mask);
    archive(value.uses_fswzadd);
    archive(value.uses_derivatives);
    archive(value.uses_typeless_image_reads);
    archive(value.uses_typeless_image_writes);
    archive(value.uses_image_buffers);
    archive(value.uses_shared_increment);
    archive(value.uses_shared_decrement);
    archive(value.uses_global_increment);
    archive(value.uses_global_decrement);
    archive(value.uses_atomic_f32_add);
    archive(value.uses_atomic_f16x2_add);
    archive(value.uses_atomic_f16x2_min);
    archive(value.uses_atomic_f16x2_max);
    archive(value.uses_atomic_f32x2_add);
    archive(value.uses_atomic_f32x2_min);
    archive(value.uses_atomic_f32x2_max);
    archive(value.uses_atomic_s32_min);
    archive(value.uses_atomic_s32_max);
    archive(value.uses_int64_bit_atomics);
    archive(value.uses_global_memory);
    archive(value.requires_coherent_global_memory);
    archive(value.requires_coherent_image_memory);
    archive(value.uses_cooperative_matrix);
    archive(value.native_global_bounds);
    archive(value.uses_atomic_image_u32);
    archive(value.uses_shadow_lod);
    archive(value.uses_rescaling_uniform);
    archive(value.uses_cbuf_indirect);
    archive(value.uses_render_area);
    archive(value.used_constant_buffer_types);
    archive(value.used_storage_buffer_types);
    archive(value.used_indirect_cbuf_types);
    archive(value.constant_buffer_mask);
    archive(value.constant_buffer_used_sizes);
    archive(value.nvn_buffer_base);
    archive(value.nvn_buffer_used);
    archive(value.requires_layer_emulation);
    archive(value.emulated_layer);
    archive(value.used_clip_distances);
    archive(value.constant_buffer_descriptors);
    archive(value.storage_buffers_descriptors);
    archive(value.texture_buffer_descriptors);
    archive(value.image_buffer_descriptors);
    archive(value.texture_descriptors);
    archive(value.handle_programs);
    archive(value.image_descriptors);
}
template <typename A>
void Fields(A& archive, Shader::ConstantBufferDescriptor& value) {
    archive(value.index);
    archive(value.count);
}
template <typename A>
void Fields(A& archive, Shader::StorageBufferDescriptor& value) {
    archive(value.cbuf_index);
    archive(value.cbuf_offset);
    archive(value.count);
    archive(value.is_written);
    archive(value.indirect_offset);
    archive(value.write_extent);
}
template <typename A>
void Fields(A& archive, Shader::StorageWriteExtent& value) {
    archive(value.bounded);
    archive(value.guarded);
    archive(value.const_end);
    archive(value.local_stride);
    archive(value.group_stride);
    archive(value.guard_const);
    archive(value.guard_local_stride);
    archive(value.guard_group_stride);
}
template <typename A>
void Fields(A& archive, Shader::TextureBufferDescriptor& value) {
    archive(value.has_secondary);
    archive(value.cbuf_index);
    archive(value.cbuf_offset);
    archive(value.shift_left);
    archive(value.secondary_cbuf_index);
    archive(value.secondary_cbuf_offset);
    archive(value.secondary_shift_left);
    archive(value.count);
    archive(value.size_shift);
    archive(value.handle_program);
}
template <typename A>
void Fields(A& archive, Shader::ImageBufferDescriptor& value) {
    archive(value.format);
    archive(value.is_written);
    archive(value.is_read);
    archive(value.is_integer);
    archive(value.cbuf_index);
    archive(value.cbuf_offset);
    archive(value.count);
    archive(value.size_shift);
    archive(value.handle_program);
}
template <typename A>
void Fields(A& archive, Shader::TextureDescriptor& value) {
    archive(value.type);
    archive(value.is_depth);
    archive(value.is_multisample);
    archive(value.is_integer);
    archive(value.has_secondary);
    archive(value.cbuf_index);
    archive(value.cbuf_offset);
    archive(value.shift_left);
    archive(value.secondary_cbuf_index);
    archive(value.secondary_cbuf_offset);
    archive(value.secondary_shift_left);
    archive(value.count);
    archive(value.size_shift);
    archive(value.handle_program);
}
template <typename A>
void Fields(A& archive, Shader::ImageDescriptor& value) {
    archive(value.type);
    archive(value.format);
    archive(value.is_written);
    archive(value.is_read);
    archive(value.is_integer);
    archive(value.cbuf_index);
    archive(value.cbuf_offset);
    archive(value.count);
    archive(value.size_shift);
    archive(value.is_byte_addressed);
    archive(value.handle_program);
}
template <typename A>
void Fields(A& archive, std::map<Shader::IR::Attribute, Shader::IR::Attribute>& value) {
    u64 count = value.size();
    archive(count);
    if constexpr (std::is_same_v<A, Reader>) {
        if (count > archive.bytes.size()) {
            throw std::runtime_error("Invalid recipe map");
        }
        for (u64 index = 0; index < count; ++index) {
            Shader::IR::Attribute first{}, second{};
            archive(first);
            archive(second);
            value.emplace(first, second);
        }
    } else {
        for (auto [first, second] : value) {
            archive(first);
            archive(second);
        }
    }
}
template <typename A>
void Fields(A& archive, Shader::ProgramHeader& value) {
    static_assert(std::is_trivially_copyable_v<Shader::ProgramHeader>);
    archive.Scalar(value);
}
template <typename A>
void Fields(A& archive, PipelineEnvironment& value) {
    archive(value.header);
    archive(value.primitive_header);
    archive(value.passthrough);
    archive(value.mesh_registers);
    archive(value.stage);
    archive(value.start);
    archive(value.proprietary);
    archive(value.reads);
}
template <typename A>
void Fields(A& archive, MeshEmulationSetup& value) {
    archive(value.vertex_spirv);
    archive(value.output_binding);
    archive(value.block_words);
    archive(value.max_primitives);
    archive(value.has_task);
    archive(value.mesh_stage);
    archive(value.payload_stride);
    archive(value.payload_binding);
    archive(value.offsets_binding);
    archive(value.args_binding);
    archive(value.distribution_spirv);
    archive(value.max_vertices);
    archive(value.vertex_lines);
    archive(value.position_lines);
}
template <typename A>
void Fields(A& archive, GraphicsPipelineInput& value) {
    archive(value.build_settings);
    archive(value.present);
    archive(value.infos);
    archive(value.spirv);
    bool has_mesh = value.mesh.has_value();
    archive(has_mesh);
    if constexpr (std::is_same_v<A, Reader>) {
        if (has_mesh) {
            value.mesh.emplace();
        }
    }
    if (has_mesh) {
        archive(*value.mesh);
    }
    archive(value.environments);
}
template <typename T>
void Writer::operator()(T& value) {
    Fields(*this, value);
}
template <typename T>
void Reader::operator()(T& value) {
    Fields(*this, value);
}
} // namespace

PipelineEnvironmentRecorder::PipelineEnvironmentRecorder(Shader::Environment& original)
    : source{original} {
    sph = original.SPH();
    mesh_primitive_header = original.MeshPrimitiveHeader();
    gp_passthrough_mask = original.GpPassthroughMask();
    mesh_registers = original.MeshRegisters();
    stage = original.ShaderStage();
    start_address = original.StartAddress();
    is_proprietary_driver = original.IsProprietaryDriver();
    record.header = sph;
    record.primitive_header = mesh_primitive_header;
    record.passthrough = gp_passthrough_mask;
    record.mesh_registers = mesh_registers;
    record.stage = stage;
    record.start = start_address;
    record.proprietary = is_proprietary_driver;
}
u64 PipelineEnvironmentRecorder::Save(u64 operation, u64 a, u64 b, u64 c, u64 answer) const {
    const std::array<u64, 5> call{operation, a, b, c, answer};
    if (std::ranges::find(record.reads, call) == record.reads.end()) {
        record.reads.push_back(call);
    }
    return answer;
}
u64 PipelineEnvironmentRecorder::ReadInstruction(u32 address) {
    return source.ReadInstruction(address);
}
u32 PipelineEnvironmentRecorder::TranslateProgramAddress(u32 address) const {
    return source.TranslateProgramAddress(address);
}
bool PipelineEnvironmentRecorder::IsNativeProgram() const {
    return Save(Native, 0, 0, 0, source.IsNativeProgram()) != 0;
}
bool PipelineEnvironmentRecorder::IsCbufStorageBufferPointer(u32 bank, u32 offset) {
    return Save(Pointer, bank, offset, 0, source.IsCbufStorageBufferPointer(bank, offset)) != 0;
}
bool PipelineEnvironmentRecorder::IsCbufIndirectStorageBufferPointer(u32 bank, u32 offset,
                                                                     u32 indirect) {
    return Save(Indirect, bank, offset, indirect,
                source.IsCbufIndirectStorageBufferPointer(bank, offset, indirect)) != 0;
}
u32 PipelineEnvironmentRecorder::ReadCbufValue(u32 bank, u32 offset) {
    return static_cast<u32>(Save(Cbuf, bank, offset, 0, source.ReadCbufValue(bank, offset)));
}
Shader::TextureType PipelineEnvironmentRecorder::ReadTextureType(u32 handle) {
    return static_cast<Shader::TextureType>(
        Save(Texture, handle, 0, 0, static_cast<u64>(source.ReadTextureType(handle))));
}
Shader::TexturePixelFormat PipelineEnvironmentRecorder::ReadTexturePixelFormat(u32 handle) {
    return static_cast<Shader::TexturePixelFormat>(
        Save(Format, handle, 0, 0, static_cast<u64>(source.ReadTexturePixelFormat(handle))));
}
bool PipelineEnvironmentRecorder::IsTexturePixelFormatInteger(u32 handle) {
    return Save(Integer, handle, 0, 0, source.IsTexturePixelFormatInteger(handle)) != 0;
}
u32 PipelineEnvironmentRecorder::ReadViewportTransformState() {
    return static_cast<u32>(Save(Viewport, 0, 0, 0, source.ReadViewportTransformState()));
}
u32 PipelineEnvironmentRecorder::TextureBoundBuffer() const {
    return static_cast<u32>(Save(Bound, 0, 0, 0, source.TextureBoundBuffer()));
}
u32 PipelineEnvironmentRecorder::LocalMemorySize() const {
    return static_cast<u32>(Save(Local, 0, 0, 0, source.LocalMemorySize()));
}
u32 PipelineEnvironmentRecorder::SharedMemorySize() const {
    return static_cast<u32>(Save(Shared, 0, 0, 0, source.SharedMemorySize()));
}
std::array<u32, 3> PipelineEnvironmentRecorder::WorkgroupSize() const {
    const auto result = source.WorkgroupSize();
    for (size_t index = 0; index < result.size(); ++index) {
        Save(Workgroup, index, 0, 0, result[index]);
    }
    return result;
}
bool PipelineEnvironmentRecorder::HasHLEMacroState() const {
    return Save(Hle, 0, 0, 0, source.HasHLEMacroState()) != 0;
}
bool PipelineEnvironmentRecorder::HasNativeInstructions() const {
    return Save(NativeInstructions, 0, 0, 0, source.HasNativeInstructions()) != 0;
}
std::optional<Shader::ReplaceConstant> PipelineEnvironmentRecorder::GetReplaceConstBuffer(
    u32 bank, u32 offset) {
    const auto result = source.GetReplaceConstBuffer(bank, offset);
    Save(Replacement, bank, offset, 0, result ? static_cast<u64>(*result) : ~u64{0});
    return result;
}
std::vector<std::pair<u32, u32>> PipelineEnvironmentRecorder::NullStoragePointerSlots() const {
    return source.NullStoragePointerSlots();
}
std::vector<u64> PipelineEnvironmentRecorder::MappedStoragePointerKeys() const {
    return source.MappedStoragePointerKeys();
}
void PipelineEnvironmentRecorder::Dump(u64 pipeline_hash, u64 shader_hash) {
    source.Dump(pipeline_hash, shader_hash);
}

bool PipelineEnvironment::Matches(Shader::Environment& environment) const {
    if (std::memcmp(&header, &environment.SPH(), sizeof(header)) != 0 ||
        std::memcmp(&primitive_header, &environment.MeshPrimitiveHeader(),
                    sizeof(primitive_header)) != 0 ||
        (stage == Shader::Stage::Geometry && passthrough != environment.GpPassthroughMask()) ||
        mesh_registers != environment.MeshRegisters() || stage != environment.ShaderStage() ||
        start != environment.StartAddress() || proprietary != environment.IsProprietaryDriver()) {
        return false;
    }
    return std::ranges::all_of(
        reads, [&](const auto& call) { return Answer(environment, call) == call[4]; });
}
bool GraphicsPipelineInput::Matches(std::span<Shader::Environment* const> current) const {
    if (current.size() != environments.size()) {
        return false;
    }
    for (size_t index = 0; index < current.size(); ++index) {
        if (!environments[index].Matches(*current[index])) {
            return false;
        }
    }
    return true;
}
bool GraphicsPipelineInput::SameShaders(const GraphicsPipelineInput& other) const {
    return build_settings == other.build_settings && present == other.present &&
           infos == other.infos && spirv == other.spirv && mesh == other.mesh;
}
GraphicsPipelineRecipe::~GraphicsPipelineRecipe() {
    if (!compressed.empty()) {
        PipelineKeep::count.fetch_sub(1);
        PipelineKeep::bytes.fetch_sub(Bytes());
        PipelineKeep::allocated.fetch_sub(AllocatedBytes());
        PipelineKeep::raw_bytes.fetch_sub(raw_bytes);
    }
}
void GraphicsPipelineRecipe::Store(const GraphicsPipelineInput& input) {
    // Writer only reads; sharing the field walk with Reader avoids two divergent encodings.
    Writer writer;
    writer(const_cast<GraphicsPipelineInput&>(input));
    auto encoded =
        Common::Compression::CompressDataZSTD(writer.bytes.data(), writer.bytes.size(), 1);
    if (!compressed.empty()) {
        PipelineKeep::count.fetch_sub(1);
        PipelineKeep::bytes.fetch_sub(Bytes());
        PipelineKeep::allocated.fetch_sub(AllocatedBytes());
        PipelineKeep::raw_bytes.fetch_sub(raw_bytes);
    }
    raw_bytes = writer.bytes.size();
    compressed = std::move(encoded);
    compressed.shrink_to_fit();
    if (!compressed.empty()) {
        PipelineKeep::count.fetch_add(1);
        PipelineKeep::bytes.fetch_add(Bytes());
        PipelineKeep::allocated.fetch_add(AllocatedBytes());
        PipelineKeep::raw_bytes.fetch_add(raw_bytes);
    }
}
std::optional<GraphicsPipelineInput> GraphicsPipelineRecipe::Restore() const {
    auto bytes = Common::Compression::TryDecompressDataZSTD(compressed, raw_bytes);
    if (!bytes || bytes->size() != raw_bytes) {
        return std::nullopt;
    }
    try {
        Reader reader{*bytes};
        GraphicsPipelineInput result;
        reader(result);
        if (!reader.bytes.empty()) {
            return std::nullopt;
        }
        return result;
    } catch (const std::runtime_error&) {
        return std::nullopt;
    }
}
} // namespace Vulkan
