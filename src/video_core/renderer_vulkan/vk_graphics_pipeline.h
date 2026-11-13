// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/draw_delta.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/evicted_pointer_slots.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/graphics_stage_info.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/pipeline_destroy_timing.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_pipeline_handles.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineBinaries;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// Mesh emulation on a device without VK_EXT_mesh_shader: the mesh program (stage 0) is built as
/// a compute shader that writes its output to a storage buffer, and this vertex shader draws it
/// (see Shader::Backend::SPIRV::EmitMeshEmulationVertex).
struct MeshEmulationSetup {
    std::vector<u32> vertex_spirv;
    u32 output_binding{};
    u32 block_words{};    ///< output words per mesh workgroup
    u32 max_primitives{}; ///< triangles per mesh workgroup
    // With a task program (stage 0, the mesh program at mesh_stage): its payload buffer and the
    // distribution pass's offsets and arguments (Shader::RuntimeInfo), and that pass's shader.
    bool has_task{};
    u32 mesh_stage{};
    u32 payload_stride{};
    u32 payload_binding{};
    u32 offsets_binding{};
    u32 args_binding{};
    std::vector<u32> distribution_spirv;
    // Output layout, for the readback diagnostics.
    u32 max_vertices{};
    u32 vertex_lines{};
    std::array<u32, 4> position_lines{~0U, ~0U, ~0U, ~0U}; ///< line of PositionX..W
};

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, PipelineBinaries* binaries,
        VideoCore::ShaderNotify* shader_notify, const Device& device,
        DescriptorPool& descriptor_pool, GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, const PipelineBuildQueue& build_queue,
        PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos, GraphicsStageInfoPool& stage_pool,
        std::function<std::array<vk::ShaderModule, NUM_STAGES>()> emit_stages = {},
        std::optional<MeshEmulationSetup> mesh_emulation = std::nullopt);

    /// Call only after shader jobs and all GPU uses have completed. Release build captures while
    /// their scheduler, caches and guest memory are alive; only handles may outlive the renderer.
    PipelineHandles TakeHandles();
    PipelineOrigin destroy_origin{PipelineOrigin::Evicted};

    static std::chrono::nanoseconds TakeHoldTime() noexcept;

    struct CpuMemory {
        size_t count{};
        size_t object{};
        size_t transitions{};
        size_t descriptors{};
        size_t allocator{};
        size_t helper_spirv{};

        void Add(const CpuMemory& other) {
            count += other.count;
            object += other.object;
            transitions += other.transitions;
            descriptors += other.descriptors;
            allocator += other.allocator;
            helper_spirv += other.helper_spirv;
        }
    };
    /// Called only after builds, draws and command workers have released the pipeline.
    CpuMemory RetainedCpuMemory() const;

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    const std::array<u64, 6>& ShaderHashes() const noexcept {
        return key.unique_hashes;
    }

    struct SampledImage {
        u32 stage;
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The sampled textures of the last configured draw, kept while a sampled frame dump runs.
    std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    /// Color components of a render target the fragment shader writes, as VkColorComponentFlags
    VkColorComponentFlags ColorOutputMask(size_t render_target) const noexcept {
        return (color_output_mask >> (render_target * 4)) & 0xF;
    }

    /// Color components the fragment shader writes, 4 bits per render target
    u32 ColorOutputMask() const noexcept {
        return color_output_mask;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }

    /// Switch 2 task/mesh pipeline: drawn with vkCmdDrawMeshTasksEXT.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return stage_infos[0]->is_task || stage_infos[0]->is_mesh;
    }

    /// Mesh pipeline drawn through the compute emulation (MeshEmulationSetup).
    [[nodiscard]] bool IsEmulatedMesh() const noexcept {
        return mesh_emulation.has_value();
    }

    /// Output bytes the emulation needs for one mesh workgroup.
    [[nodiscard]] u64 MeshEmulationBlockBytes() const noexcept {
        return mesh_emulation ? u64{mesh_emulation->block_words} * 4 : 0;
    }

    /// Vertices of the emulation's triangle list for one mesh workgroup.
    [[nodiscard]] u32 MeshEmulationVerticesPerGroup() const noexcept {
        return mesh_emulation ? mesh_emulation->max_primitives * 3 : 0;
    }

    /// The emulation's setup (output layout), or null.
    [[nodiscard]] const MeshEmulationSetup* MeshEmulation() const noexcept {
        return mesh_emulation ? &*mesh_emulation : nullptr;
    }

    /// The emulation's output buffer and workgroup count for the next Configure.
    void SetMeshEmulationOutput(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                u32 groups) noexcept {
        mesh_output_buffer = buffer;
        mesh_output_offset = offset;
        mesh_output_size = size;
        mesh_groups = groups;
    }

    struct MeshEmulationBuffer {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        VkDeviceSize size{};
    };
    /// With a task program: the task payload, offset and argument buffers, the arena counter,
    /// and the words of the output arena (the draw's group count is the task's).
    void SetMeshEmulationTaskBuffers(const MeshEmulationBuffer& payload,
                                     const MeshEmulationBuffer& offsets,
                                     const MeshEmulationBuffer& args,
                                     const MeshEmulationBuffer& counter,
                                     u32 capacity_words) noexcept {
        mesh_task_buffers = {payload, offsets, args, counter};
        mesh_capacity = capacity_words;
    }

    [[nodiscard]] bool MeshEmulationHasTask() const noexcept {
        return mesh_emulation && mesh_emulation->has_task;
    }

    [[nodiscard]] u32 MeshEmulationPayloadStride() const noexcept {
        return mesh_emulation ? mesh_emulation->payload_stride : 0;
    }
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    bool Configure(bool is_indexed) {
        return configure_func(this, is_indexed);
    }

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it = std::ranges::find(transitions, current_key, &Transition::key);
        return it != transitions.end() ? it->pipeline : nullptr;
    }

    /// The frame the pipeline cache last handed this pipeline out (evicts long unused ones).
    void MarkUsed(u64 frame) noexcept {
        last_use_frame = frame;
        used = true;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use_frame;
    }

    /// Whether the cache has handed this pipeline out at all (frame 0 included), as opposed to a
    /// pipeline prebuilt from the disk cache that no draw has used yet.
    [[nodiscard]] bool WasUsed() const noexcept {
        return used;
    }

    /// The constant buffer slots each program binds as storage buffer pointers
    [[nodiscard]] const VideoCommon::EvictedPointerSlots::Slots& MappedPointers() const noexcept {
        return mapped_pointers;
    }

    void SetMappedPointers(VideoCommon::EvictedPointerSlots::Slots slots) noexcept {
        mapped_pointers = std::move(slots);
    }

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transitions.clear();
    }

    [[nodiscard]] bool WritesStorage() const noexcept {
        return writes_storage;
    }

    /// Writes storage images or image buffers (their images are marked modified when resolved).
    [[nodiscard]] bool WritesImages() const noexcept {
        return writes_images;
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return build->IsBuilt();
    }

    /// Builds it on this thread, for a pipeline made with PipelineBuildQueue::owner_builds
    void Build() {
        build->Build();
    }

    /// Waits up to `timeout` for the build; true when built.
    [[nodiscard]] bool WaitBuilt(std::chrono::microseconds timeout) const {
        return build->WaitFor(timeout);
    }

    [[nodiscard]] bool Failed() const noexcept {
        return build->Failed();
    }

    /// Fast draw (nx2_fast_draw.h): whether draws can keep the image views and samplers the last
    /// captured Configure resolved. Handles computed at draw time are resolved on every draw.
    [[nodiscard]] bool TakesDeltaDraws() const noexcept {
        return delta_blocker == DeltaBlocker::None && delta_views_valid;
    }

    /// Why TakesDeltaDraws is false (diagnostics).
    enum class DeltaBlocker : u8 {
        None,
        WritesImages,
        Mesh,
        ComputedHandles,
        NotConfigured,
        Count,
    };
    [[nodiscard]] DeltaBlocker DeltaDrawBlocker() const noexcept {
        return delta_blocker != DeltaBlocker::None ? delta_blocker
               : delta_views_valid                 ? DeltaBlocker::None
                                                   : DeltaBlocker::NotConfigured;
    }

    /// Constant buffer slots per stage the pipeline reads texture and image handles from.
    [[nodiscard]] const std::array<u32, NUM_STAGES>& HandleConstantBuffers() const noexcept {
        return handle_constant_buffers;
    }

    /// Resolves and binds every buffer as Configure does (storage buffer addresses are read from
    /// the constant buffers again) and records the draw's descriptors with the image views,
    /// samplers and texture buffers the last captured Configure resolved. False when the draw
    /// can't be recorded, as for Configure.
    bool ConfigureDelta(bool is_indexed);

    /// The next Configure keeps the views and samplers it resolves for delta draws. With
    /// `verify`, it first compares them with the ones kept before; DeltaMismatch reports it.
    void CaptureNextDelta(bool verify) noexcept {
        capture_delta = true;
        verify_delta = verify;
    }
    [[nodiscard]] bool DeltaMismatch() const noexcept {
        return delta_mismatch;
    }

    /// Reevaluate handles (including storage-backed programs), then compare TIC/TSC contents.
    [[nodiscard]] VideoCommon::DrawDeltaTextureResult CheckDrawDeltaTextures();

    /// EDEN_NX2_DRAW_REUSE: what the handles of the stages are computed from (constant buffer
    /// slots, storage bindings), and the stages the last captured Configure resolved.
    [[nodiscard]] const VideoCommon::DrawHandleInputs& DeltaHandleInputs() const noexcept {
        return delta_handle_inputs;
    }
    [[nodiscard]] const std::array<bool, NUM_STAGES>& ConfiguredStages() const noexcept {
        return configured_stages;
    }

    /// EDEN_NX2_DRAW_REUSE: evaluates the handles again (as Configure does) and compares them with
    /// the ones the last captured Configure resolved; texture and sampler headers are not read
    /// (the delta chain breaks on anything that writes them). BackedOff: the pipeline's recent
    /// checks kept failing, so this one was skipped (DeltaCheckBackoff).
    enum class DeltaHandleCheck : u8 { Held, Changed, BackedOff };
    [[nodiscard]] DeltaHandleCheck CheckDeltaHandles();

    /// Whether an image the kept views sample has an upload pending or a newer alias to copy in:
    /// ConfigureDelta prepares none, Configure does.
    [[nodiscard]] bool DeltaImagesChange() const noexcept;

    /// EDEN_NX2_DRAWX_TWIN: whether `other` resolves exactly the bindings this pipeline resolves:
    /// the same configure specialization (stages) and the same interned stage infos, so the same
    /// descriptors, handle programs and stage properties (GraphicsStageInfoPool shares an info
    /// only between programs whose retained properties and descriptors all match).
    [[nodiscard]] bool SameBindingsAs(const GraphicsPipeline& other) const noexcept {
        return configure_func == other.configure_func && stage_infos == other.stage_infos;
    }
    /// EDEN_NX2_DRAWX_TWIN: takes what the last captured resolve of `source` (SameBindingsAs)
    /// kept for delta draws: views, samplers, image descriptors, rescaling words, handles.
    void TakeDeltaState(const GraphicsPipeline& source);

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) { return pl->ConfigureImpl<Spec>(is_indexed); };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    template <typename Spec>
    bool ConfigureImpl(bool is_indexed);

    void KeepImageDescriptors(size_t stage, size_t first_entry);

    /// The handles of the configured stages from current memory, in Configure's order, into
    /// draw_delta_handles.
    void EvaluateDeltaHandles();

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    void MakePipeline(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    PipelineBinaries* const binaries; ///< Shader cache v2's driver layer, else null
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    bool (*configure_func)(GraphicsPipeline*, bool){};

    struct Transition {
        GraphicsPipelineCacheKey key;
        GraphicsPipeline* pipeline;
    };
    std::vector<Transition> transitions;
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    u64 last_use_frame{};
    bool used{};

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    GraphicsStageInfos stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    std::vector<SampledImage> sampled_images;
    u32 num_textures{};
    bool fragment_has_color0_output{};
    u32 color_output_mask{};
    bool writes_storage{};
    /// Every texture is sampled with a handle read from a constant buffer: ConfigureImpl can
    /// resolve them from texture_memo.
    bool texture_memo_eligible{};
    VideoCommon::SampledTextureMemo texture_memo;
    bool writes_images{};

    DeltaBlocker delta_blocker{};
    bool delta_views_valid{};
    VideoCommon::DrawDeltaTextures<Tegra::Texture::TICEntry, Tegra::Texture::TSCEntry>
        draw_delta_textures;
    std::vector<std::pair<u32, u32>> draw_delta_handles;
    /// EDEN_NX2_DRAW_REUSE: the handles the last captured Configure resolved, what they are
    /// computed from, and the back-off for pipelines whose handles keep changing.
    std::vector<std::pair<u32, u32>> delta_handles;
    bool delta_handles_valid{};
    VideoCommon::DrawHandleInputs delta_handle_inputs;
    VideoCommon::DeltaCheckBackoff delta_check_backoff;
    bool capture_delta{};
    bool verify_delta{};
    bool delta_mismatch{};
    std::array<bool, NUM_STAGES> configured_stages{};
    std::array<u32, NUM_STAGES> handle_constant_buffers{};
    std::vector<VideoCommon::ImageViewInOut> delta_views;
    std::vector<VideoCommon::SamplerId> delta_samplers;
    /// Delta draws: the image descriptor entries (first entry and count per
    /// stage) and rescaling words pushed for delta_views, which delta draws push again while no
    /// image changed its scale (TextureCache::ScaleEpoch).
    std::vector<DescriptorUpdateEntry> kept_image_entries;
    std::array<std::pair<u32, u32>, NUM_STAGES> kept_image_ranges{};
    std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> kept_rescaling{};
    u64 kept_scale_epoch{};
    bool kept_images_valid{};

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    const bool worker_only_build;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};

    // Mesh emulation (MeshEmulationSetup).
    std::optional<MeshEmulationSetup> mesh_emulation;
    vk::ShaderModule mesh_vertex_module;
    vk::Pipeline mesh_compute_pipeline;
    VkDescriptorSet mesh_descriptor_set{}; ///< shared by the dispatch and the draw it feeds
    VkBuffer mesh_output_buffer{};
    VkDeviceSize mesh_output_offset{};
    VkDeviceSize mesh_output_size{};
    u32 mesh_groups{};
    // With a task program: its compute pipeline, the distribution pass between it and the mesh
    // dispatch, and the per-draw buffers (payload, offsets, arguments, arena counter).
    vk::Pipeline task_compute_pipeline;
    vk::ShaderModule distribution_module;
    vk::DescriptorSetLayout distribution_set_layout;
    vk::PipelineLayout distribution_layout;
    vk::Pipeline distribution_pipeline;
    DescriptorAllocator distribution_allocator;
    std::array<MeshEmulationBuffer, 4> mesh_task_buffers{};
    u32 mesh_capacity{};

    std::optional<PipelineBuild> build;
};

} // namespace Vulkan
