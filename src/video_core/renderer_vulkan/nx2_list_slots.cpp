// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nx2_list_slots.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

void Nx2ListSlots::Release(u32 slot) {
    if (slot >= SlotCount) {
        return;
    }
    if (holds[slot].fetch_sub(1, std::memory_order_acq_rel) == 0) {
        holds[slot].fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR(Render_Vulkan, "Nx2 list slot {} released without a hold", slot);
    }
}

void Nx2ListSlots::Snapshot(VkBuffer src, u64 src_offset, u32 count, u32 slot) {
    if (src == VK_NULL_HANDLE || slot >= SlotCount || count == 0 ||
        count > Tegra::Nx2ListIndirect::SlotWords) {
        return;
    }
    Prepare();
    const VkDeviceSize base = VkDeviceSize{slot} * SlotBytes;
    copies.clear();
    copies.push_back(VkBufferCopy{src_offset, base, VkDeviceSize{count} * 4});
    if (count == Tegra::Nx2ListIndirect::SlotWords) {
        // The grid as DispatchNativeCompute takes it: height and depth are 16 bits. The upper
        // halves of their words in the slot stay 0 from the fill (Prepare).
        copies.push_back(VkBufferCopy{src_offset, base + DispatchOffset, 4});
        copies.push_back(VkBufferCopy{src_offset + 4, base + DispatchOffset + 4, 2});
        copies.push_back(VkBufferCopy{src_offset + 8, base + DispatchOffset + 8, 2});
    }
    RecordCopies(src, *buffer, copies);
}

void Nx2ListSlots::NoteOwed(DAddr device_addr, u32 value, u32 guest_word) {
    // The upload wrote over older owed words at these bytes.
    for (auto it = owed.lower_bound(device_addr >= 3 ? device_addr - 3 : 0);
         it != owed.end() && it->first < device_addr + 4;) {
        Release(Tegra::Nx2ListIndirect::PlaceholderSlot(it->second));
        it = owed.erase(it);
    }
    if (guest_word != value || !Tegra::Nx2ListIndirect::IsPlaceholder(value)) {
        return; // It wrote nothing there
    }
    holds[Tegra::Nx2ListIndirect::PlaceholderSlot(value)].fetch_add(1, std::memory_order_acq_rel);
    owed.emplace(device_addr, value);
}

void Nx2ListSlots::NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z) {
    std::array<u32, 3> slots{NoEntry, NoEntry, NoEntry};
    size_t count = 0;
    for (const u32 word : {x, y, z}) {
        if (!Tegra::Nx2ListIndirect::IsPlaceholder(word)) {
            continue;
        }
        const u32 slot = Tegra::Nx2ListIndirect::PlaceholderSlot(word);
        if (std::ranges::find(slots, slot) == slots.end()) {
            slots[count++] = slot;
        }
    }
    if (count == 0 && qmd_slots.empty()) {
        return;
    }
    // The new holds first: an old hold of the same slot is dropped after them.
    for (size_t i = 0; i < count; ++i) {
        holds[slots[i]].fetch_add(1, std::memory_order_acq_rel);
    }
    const auto it = qmd_slots.find(device_addr);
    if (it == qmd_slots.end()) {
        if (count != 0) {
            qmd_slots.emplace(device_addr, slots);
        }
        return;
    }
    for (const u32 slot : it->second) {
        if (slot != NoEntry) {
            Release(slot);
        }
    }
    if (count == 0) {
        qmd_slots.erase(it);
    } else {
        it->second = slots;
    }
}

void Nx2ListSlots::Read(std::span<const u32> slots, std::span<u32> words) {
    std::ranges::fill(words, 0U);
    if (!buffer) {
        return;
    }
    if (!readback) {
        readback = memory_allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = VkDeviceSize{ReadbackSlots} * SlotWordBytes,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Download);
    }
    for (size_t first = 0; first < slots.size(); first += ReadbackSlots) {
        const size_t count = (std::min<size_t>)(ReadbackSlots, slots.size() - first);
        copies.clear();
        for (size_t i = 0; i < count; ++i) {
            copies.push_back(VkBufferCopy{VkDeviceSize{slots[first + i]} * SlotBytes,
                                          VkDeviceSize{i} * SlotWordBytes, SlotWordBytes});
        }
        RecordCopies(*buffer, *readback, copies);
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            // A fence makes device writes available, not visible to the host.
            static constexpr VkMemoryBarrier HOST_BARRIER{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                   HOST_BARRIER);
        });
        scheduler.Finish();
        readback.Invalidate();
        const size_t bytes =
            (std::min)(count * SlotWordBytes, words.size_bytes() - first * SlotWordBytes);
        std::memcpy(words.data() + first * Tegra::Nx2ListIndirect::SlotWords,
                    readback.Mapped().data(), bytes);
    }
}

void Nx2ListSlots::Prepare() {
    if (buffer) {
        return;
    }
    buffer = memory_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = VkDeviceSize{SlotCount} * SlotBytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::DeviceLocal);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([dst = *buffer](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.FillBuffer(dst, 0, VK_WHOLE_SIZE, 0);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0, WRITE_BARRIER);
    });
}

void Nx2ListSlots::RecordCopies(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> regions) {
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    static constexpr VkMemoryBarrier WRITE_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([src, dst, list = std::vector<VkBufferCopy>(regions.begin(), regions.end())](
                         vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        cmdbuf.CopyBuffer(src, dst, list);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0, WRITE_BARRIER);
    });
}

} // namespace Vulkan
