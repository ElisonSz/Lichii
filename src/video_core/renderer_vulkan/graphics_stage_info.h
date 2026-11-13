// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "shader_recompiler/shader_info.h"

namespace Vulkan {

/// Only the shader properties used by graphics pipeline creation and draws. Compiler analysis
/// and the fixed-capacity descriptor arrays in Shader::Info do not belong to a live pipeline.
struct GraphicsStageProperties {
    explicit GraphicsStageProperties(const Shader::Info& info)
        : handle_shader_hash{info.handle_shader_hash}, loads{info.loads},
          constant_buffer_mask{info.constant_buffer_mask},
          constant_buffer_used_sizes{info.constant_buffer_used_sizes},
          stores_frag_color{info.stores_frag_color}, frag_color_components{info.frag_color_components},
          stores_global_memory{info.stores_global_memory}, is_task{info.is_task}, is_mesh{info.is_mesh},
          uses_render_area{info.uses_render_area}, uses_subgroup_vote{info.uses_subgroup_vote},
          uses_subgroup_invocation_id{info.uses_subgroup_invocation_id},
          uses_subgroup_shuffles{info.uses_subgroup_shuffles},
          uses_subgroup_mask{info.uses_subgroup_mask}, uses_quad_shuffles{info.uses_quad_shuffles},
          uses_fswzadd{info.uses_fswzadd},
          removed_conditional_barrier{info.removed_conditional_barrier} {}

    bool operator==(const GraphicsStageProperties&) const = default;

    u64 handle_shader_hash{};
    Shader::VaryingState loads;
    u32 constant_buffer_mask{};
    std::array<u32, Shader::Info::MAX_CBUFS> constant_buffer_used_sizes{};
    std::array<bool, 8> stores_frag_color{};
    std::array<u8, 8> frag_color_components{};
    bool stores_global_memory{};
    bool is_task{};
    bool is_mesh{};
    bool uses_render_area{};
    bool uses_subgroup_vote{};
    bool uses_subgroup_invocation_id{};
    bool uses_subgroup_shuffles{};
    bool uses_subgroup_mask{};
    bool uses_quad_shuffles{};
    bool uses_fswzadd{};
    bool removed_conditional_barrier{};
};

/// Descriptors and handle bytecode occupy one allocation per distinct shader record. Immutable
/// spans let every pipeline use it directly without copying fixed-capacity compiler containers.
struct GraphicsStageInfo : GraphicsStageProperties {
    explicit GraphicsStageInfo(const Shader::Info& info) : GraphicsStageProperties{info} {
        size_t bytes{};
        const auto reserve = [&]<typename T>(const T& values) {
            using Element = typename T::value_type;
            bytes = Aligned(bytes, alignof(Element)) + values.size() * sizeof(Element);
        };
        reserve(info.constant_buffer_descriptors);
        reserve(info.storage_buffers_descriptors);
        reserve(info.texture_buffer_descriptors);
        reserve(info.image_buffer_descriptors);
        reserve(info.texture_descriptors);
        reserve(info.image_descriptors);
        bytes = Aligned(bytes, alignof(std::span<const u32>)) +
                info.handle_programs.size() * sizeof(std::span<const u32>);
        for (const auto& program : info.handle_programs) {
            reserve(program);
        }
        storage_bytes = bytes;
        if (bytes == 0) {
            return;
        }
        storage = std::make_unique<std::byte[]>(bytes);
        size_t offset{};
        const auto copy = [&]<typename T>(const T& values) {
            using Element = typename T::value_type;
            offset = Aligned(offset, alignof(Element));
            auto* const begin = reinterpret_cast<Element*>(storage.get() + offset);
            std::uninitialized_copy(values.begin(), values.end(), begin);
            offset += values.size() * sizeof(Element);
            return std::span<const Element>{begin, values.size()};
        };
        constant_buffer_descriptors = copy(info.constant_buffer_descriptors);
        storage_buffers_descriptors = copy(info.storage_buffers_descriptors);
        texture_buffer_descriptors = copy(info.texture_buffer_descriptors);
        image_buffer_descriptors = copy(info.image_buffer_descriptors);
        texture_descriptors = copy(info.texture_descriptors);
        image_descriptors = copy(info.image_descriptors);
        offset = Aligned(offset, alignof(std::span<const u32>));
        auto* const programs = reinterpret_cast<std::span<const u32>*>(storage.get() + offset);
        offset += info.handle_programs.size() * sizeof(std::span<const u32>);
        for (size_t index = 0; index < info.handle_programs.size(); ++index) {
            std::construct_at(programs + index, copy(info.handle_programs[index]));
        }
        handle_programs = {programs, info.handle_programs.size()};
    }

    bool Matches(const Shader::Info& info) const {
        return static_cast<const GraphicsStageProperties&>(*this) ==
                   GraphicsStageProperties{info} &&
               std::ranges::equal(constant_buffer_descriptors, info.constant_buffer_descriptors) &&
               std::ranges::equal(storage_buffers_descriptors, info.storage_buffers_descriptors) &&
               std::ranges::equal(texture_buffer_descriptors, info.texture_buffer_descriptors) &&
               std::ranges::equal(image_buffer_descriptors, info.image_buffer_descriptors) &&
               std::ranges::equal(texture_descriptors, info.texture_descriptors) &&
               std::ranges::equal(image_descriptors, info.image_descriptors) &&
               std::ranges::equal(
                   handle_programs, info.handle_programs,
                   [](const auto& lhs, const auto& rhs) { return std::ranges::equal(lhs, rhs); });
    }

    size_t RetainedBytes() const {
        return sizeof(*this) + storage_bytes;
    }

    std::span<const Shader::ConstantBufferDescriptor> constant_buffer_descriptors;
    std::span<const Shader::StorageBufferDescriptor> storage_buffers_descriptors;
    std::span<const Shader::TextureBufferDescriptor> texture_buffer_descriptors;
    std::span<const Shader::ImageBufferDescriptor> image_buffer_descriptors;
    std::span<const Shader::TextureDescriptor> texture_descriptors;
    std::span<const Shader::ImageDescriptor> image_descriptors;
    std::span<const std::span<const u32>> handle_programs;

private:
    static size_t Aligned(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    // These descriptor types and spans have trivial destructors; their storage needs no walk.
    static_assert(std::is_trivially_copyable_v<Shader::ConstantBufferDescriptor>);
    static_assert(std::is_trivially_copyable_v<Shader::StorageBufferDescriptor>);
    static_assert(std::is_trivially_copyable_v<Shader::TextureBufferDescriptor>);
    static_assert(std::is_trivially_copyable_v<Shader::ImageBufferDescriptor>);
    static_assert(std::is_trivially_copyable_v<Shader::TextureDescriptor>);
    static_assert(std::is_trivially_copyable_v<Shader::ImageDescriptor>);
    static_assert(std::is_trivially_copyable_v<std::span<const u32>>);
    std::unique_ptr<std::byte[]> storage;
    size_t storage_bytes{};
};

using GraphicsStageInfos = std::array<std::shared_ptr<const GraphicsStageInfo>, 5>;

/// Weak ownership lets eviction reclaim unused shader records. The guest hash only selects a
/// bucket: translation can specialize a shader for the other stages and pipeline state, so every
/// retained property and descriptor must also match before it can be shared.
class GraphicsStageInfoPool {
public:
    std::shared_ptr<const GraphicsStageInfo> Intern(u64 hash, const Shader::Info* info) {
        if (!info) {
            static const auto empty = std::make_shared<const GraphicsStageInfo>(Shader::Info{});
            return empty;
        }
        std::scoped_lock lock{mutex};
        auto& bucket = shaders[hash];
        for (auto it = bucket.begin(); it != bucket.end();) {
            if (auto existing = it->lock()) {
                if (existing->Matches(*info)) {
                    return existing;
                }
                ++it;
            } else {
                it = bucket.erase(it);
            }
        }
        auto result = std::make_shared<const GraphicsStageInfo>(*info);
        bucket.emplace_back(result);
        return result;
    }

    struct Stats {
        size_t records{};
        size_t bytes{};
        size_t copies{};
        size_t unshared_bytes{};
        size_t largest{};
    };

    Stats LiveStats() {
        std::scoped_lock lock{mutex};
        Stats stats;
        for (const auto& [hash, bucket] : shaders) {
            for (const auto& weak : bucket) {
                if (const auto record = weak.lock()) {
                    const size_t bytes = record->RetainedBytes();
                    const size_t copies = static_cast<size_t>(record.use_count() - 1);
                    if (copies == 0) {
                        continue;
                    }
                    ++stats.records;
                    stats.bytes += bytes;
                    stats.copies += copies;
                    stats.unshared_bytes += bytes * copies;
                    stats.largest = (std::max)(stats.largest, bytes);
                }
            }
        }
        return stats;
    }

    void Prune() {
        std::scoped_lock lock{mutex};
        for (auto it = shaders.begin(); it != shaders.end();) {
            std::erase_if(it->second, [](const auto& weak) { return weak.expired(); });
            if (it->second.empty()) {
                it = shaders.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    std::mutex mutex;
    std::unordered_map<u64, std::vector<std::weak_ptr<const GraphicsStageInfo>>> shaders;
};

} // namespace Vulkan
