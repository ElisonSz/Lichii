// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

namespace VideoCommon {

/// EDEN_NX2_DRAW_REUSE: what a pipeline's texture and image handles are computed from, per stage.
/// RasterizerVulkan compares the constant buffer binds and data written since the previous draw
/// with it; while nothing reaches it, the handles the previous draw resolved are the current ones.
struct DrawHandleInputs {
    static constexpr size_t NumStages = 5;
    static constexpr u32 NumCbufs = 18; ///< Constant buffer slots per stage (Maxwell)
    static constexpr u32 NumStorage = 64;

    /// Constant buffer slots handles are read from: the descriptors' own slots, the slots handle
    /// programs read, and the slots holding the addresses of the storage buffers they read.
    std::array<u32, NumStages> cbufs{};
    /// Storage buffer bindings (indices of the stage's storage buffer descriptors) programs read.
    std::array<u64, NumStages> storage{};
    /// Any data written may change a handle: a program reads a storage buffer whose address comes
    /// from a pointer table, or the scan met an operation it doesn't know.
    std::array<bool, NumStages> any_write{};

    void AddCbuf(size_t stage, u32 index) {
        // Slots past the last one read nothing (the evaluators' cbuf_address gives 0).
        if (index < NumCbufs) {
            cbufs[stage] |= 1U << index;
        }
    }

    /// Adds what handle program `code` of `stage` reads (Shader::EvaluateHandleProgram).
    template <typename StorageDescriptors>
    void AddProgram(size_t stage, std::span<const u32> code,
                    const StorageDescriptors& storage_descriptors) {
        using Shader::HandleOp;
        const auto unknown = [&] {
            cbufs[stage] = (1U << NumCbufs) - 1;
            any_write[stage] = true;
        };
        for (size_t pc = 0; pc < code.size();) {
            const u32 op = code[pc++];
            const size_t left = code.size() - pc;
            switch (static_cast<HandleOp>(op)) {
            case HandleOp::Immediate:
            case HandleOp::Tee:
            case HandleOp::Load:
                pc += left < 1 ? left : 1;
                break;
            case HandleOp::Cbuf:
                if (left < 2) {
                    return unknown();
                }
                AddCbuf(stage, code[pc]);
                pc += 2;
                break;
            case HandleOp::CbufDynamic:
                if (left < 1) {
                    return unknown();
                }
                AddCbuf(stage, code[pc]);
                pc += 1;
                break;
            case HandleOp::Storage: {
                if (left < 1) {
                    return unknown();
                }
                const u32 binding = code[pc];
                pc += 1;
                if (binding >= storage_descriptors.size()) {
                    break; // StorageBufferBase gives 0: nothing is read
                }
                if (binding >= NumStorage) {
                    return unknown();
                }
                storage[stage] |= u64{1} << binding;
                const auto& descriptor = storage_descriptors[binding];
                AddCbuf(stage, descriptor.cbuf_index);
                if (descriptor.indirect_offset != ~0U) {
                    any_write[stage] = true; // the table entry is read from other memory
                }
                break;
            }
            default:
                if (op > static_cast<u32>(HandleOp::Index)) {
                    return unknown();
                }
                break; // no operands
            }
        }
    }
};

/// Whether the constant buffer binds (`binds`, per stage) or data written since the previous draw
/// may have changed a handle of the stages in `stages`. cbuf_reached(stage, slot) and
/// storage_reached(stage, binding) tell whether data written reaches that constant buffer's or
/// storage buffer's memory; they are only asked when `data_written`.
template <typename CbufReached, typename StorageReached>
[[nodiscard]] bool HandleInputsReached(const DrawHandleInputs& inputs,
                                       const std::array<bool, DrawHandleInputs::NumStages>& stages,
                                       const std::array<u32, DrawHandleInputs::NumStages>& binds,
                                       bool data_written, CbufReached&& cbuf_reached,
                                       StorageReached&& storage_reached) {
    for (size_t stage = 0; stage < DrawHandleInputs::NumStages; ++stage) {
        if (stages[stage] && (binds[stage] & inputs.cbufs[stage]) != 0) {
            return true;
        }
    }
    if (!data_written) {
        return false;
    }
    for (size_t stage = 0; stage < DrawHandleInputs::NumStages; ++stage) {
        if (!stages[stage]) {
            continue;
        }
        if (inputs.any_write[stage]) {
            return true;
        }
        for (u32 mask = inputs.cbufs[stage]; mask != 0; mask &= mask - 1) {
            if (cbuf_reached(stage, static_cast<u32>(std::countr_zero(mask)))) {
                return true;
            }
        }
        for (u64 mask = inputs.storage[stage]; mask != 0; mask &= mask - 1) {
            if (storage_reached(stage, static_cast<u32>(std::countr_zero(mask)))) {
                return true;
            }
        }
    }
    return false;
}

/// EDEN_NX2_DRAW_REUSE: a pipeline whose delta draws keep finding changed handles pays for every
/// evaluation twice (the check, then the full resolve). After Threshold failures in a row it
/// goes to the full resolve directly, trying the check again every Period-th time.
class DeltaCheckBackoff {
public:
    static constexpr u32 Threshold = 8;
    static constexpr u32 Period = 8;

    [[nodiscard]] bool ShouldTry() noexcept {
        if (failures < Threshold) {
            return true;
        }
        return ++skipped % Period == 0;
    }

    void Note(bool held) noexcept {
        if (held) {
            failures = 0;
            skipped = 0;
        } else if (failures < Threshold) {
            ++failures;
        }
    }

private:
    u32 failures{};
    u32 skipped{};
};

/// EDEN_NX2_DRAW_VERTEX_INPUT: the registers RasterizerVulkan::UpdateVertexInput builds a dynamic
/// vertex input state from: every attribute's format word, every binding's stride, divisor and
/// instancing, the bindings the buffer cache repacks, and how many of each the device takes. The
/// build reads nothing else, so equal keys build equal states.
struct VertexInputKey {
    static constexpr size_t MaxEntries = 32;

    std::array<u32, MaxEntries> attributes{};
    std::array<u32, MaxEntries> strides{};
    std::array<u32, MaxEntries> divisors{};
    u32 instanced{};
    u32 repacked{};
    u32 num_attributes{};
    u32 num_bindings{};

    [[nodiscard]] bool operator==(const VertexInputKey&) const = default;

    [[nodiscard]] u64 Hash() const noexcept {
        u64 hash = 0x9E3779B97F4A7C15ULL ^ ((u64{num_attributes} << 32) | num_bindings);
        const auto mix = [&hash](u64 value) {
            hash = (hash ^ value) * 0xFF51AFD7ED558CCDULL;
            hash ^= hash >> 32;
        };
        for (size_t index = 0; index < MaxEntries; index += 2) {
            mix(u64{attributes[index]} | (u64{attributes[index + 1]} << 32));
            mix(u64{strides[index]} | (u64{strides[index + 1]} << 32));
            mix(u64{divisors[index]} | (u64{divisors[index + 1]} << 32));
        }
        mix(u64{instanced} | (u64{repacked} << 32));
        return hash;
    }
};

/// Built vertex input states by their keys, direct-mapped. A state is immutable once built and
/// shared with the recorded commands that set it, so replacing an entry never frees a state a
/// pending command still reads.
template <typename State>
class VertexInputCache {
public:
    static constexpr size_t NumEntries = 256;

    [[nodiscard]] std::shared_ptr<const State> Find(const VertexInputKey& key, u64 hash) const {
        if (entries.empty()) {
            return nullptr;
        }
        const Entry& entry = entries[hash % NumEntries];
        if (entry.state == nullptr || entry.hash != hash || !(entry.key == key)) {
            return nullptr;
        }
        return entry.state;
    }

    void Insert(const VertexInputKey& key, u64 hash, std::shared_ptr<const State> state) {
        if (entries.empty()) {
            entries.resize(NumEntries);
        }
        Entry& entry = entries[hash % NumEntries];
        entry.hash = hash;
        entry.key = key;
        entry.state = std::move(state);
    }

private:
    struct Entry {
        u64 hash{};
        VertexInputKey key{};
        std::shared_ptr<const State> state;
    };
    std::vector<Entry> entries;
};

enum class DrawDeltaTextureResult : u8 { Match, NotCaptured, Handles, Descriptors };

enum class DrawDeltaGuardResult : u8 { Match, State, Submitted, RenderPass };

struct DrawDeltaGuard {
    u64 serial{};
    u64 map_serial{};
    u64 tick{};
    u64 render_pass{};
};

[[nodiscard]] constexpr DrawDeltaGuardResult CheckDrawDeltaGuard(const DrawDeltaGuard& previous,
                                                                 const DrawDeltaGuard& current,
                                                                 bool pass_active,
                                                                 bool pipeline_bound) {
    if (previous.serial != current.serial || previous.map_serial != current.map_serial) {
        return DrawDeltaGuardResult::State;
    }
    if (previous.tick != current.tick) {
        return DrawDeltaGuardResult::Submitted;
    }
    if (previous.render_pass != current.render_pass || !pass_active || !pipeline_bound) {
        return DrawDeltaGuardResult::RenderPass;
    }
    return DrawDeltaGuardResult::Match;
}

/// Keep the descriptors that actually selected the views, not a second read of guest memory.
/// Readers return an optional descriptor: absent/unmapped entries cannot authorize reuse.
/// Epochs invalidate IDs even when a deleted/recreated image has identical descriptor bytes.
template <typename ImageDescriptor, typename SamplerDescriptor>
class DrawDeltaTextures {
public:
    using Handle = std::pair<u32, u32>;

    template <typename ReadImage, typename ReadSampler>
    void Capture(std::span<const Handle> handles, u64 images, u64 samplers, ReadImage&& read_image,
                 ReadSampler&& read_sampler) {
        valid = false;
        entries.clear();
        for (const Handle& handle : handles) {
            const auto image = read_image(handle.first);
            if (!image) {
                return;
            }
            const auto sampler = handle.second == ~0U
                                    ? SamplerDescriptor{}
                                    : read_sampler(handle.second);
            if (!sampler) {
                return;
            }
            entries.push_back({handle, *image, *sampler});
        }
        image_epoch = images;
        sampler_epoch = samplers;
        valid = true;
    }

    template <typename ReadImage, typename ReadSampler>
    [[nodiscard]] DrawDeltaTextureResult Match(std::span<const Handle> handles, u64 images,
                                               u64 samplers, ReadImage&& read_image,
                                               ReadSampler&& read_sampler) const {
        if (!valid) {
            return DrawDeltaTextureResult::NotCaptured;
        }
        if (images != image_epoch || samplers != sampler_epoch) {
            return DrawDeltaTextureResult::Descriptors;
        }
        if (handles.size() != entries.size()) {
            return DrawDeltaTextureResult::Handles;
        }
        for (size_t index = 0; index < handles.size(); ++index) {
            const Entry& entry = entries[index];
            if (handles[index] != entry.handle) {
                return DrawDeltaTextureResult::Handles;
            }
            const auto image = read_image(entry.handle.first);
            const auto sampler = entry.handle.second == ~0U
                                    ? SamplerDescriptor{}
                                    : read_sampler(entry.handle.second);
            if (!image || !sampler || *image != entry.image || *sampler != entry.sampler) {
                return DrawDeltaTextureResult::Descriptors;
            }
        }
        return DrawDeltaTextureResult::Match;
    }

private:
    struct Entry {
        Handle handle;
        ImageDescriptor image;
        SamplerDescriptor sampler;
    };
    std::vector<Entry> entries;
    u64 image_epoch{};
    u64 sampler_epoch{};
    bool valid{};
};

} // namespace VideoCommon
