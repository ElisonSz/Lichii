// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/oot_whiteout_trace.h"
#include "video_core/renderer_vulkan/vk_oot_whiteout.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
void OotWhiteoutDetector::Capture(ImageView &view, u64 shader, u64 frame, bool tonemap) {
    using VideoCommon::OotWhiteout::Encoding;
    using VideoCore::Surface::PixelFormat;
    if (!VideoCommon::OotWhiteout::Enabled() || frame % 15 != 0)
        return;
    const Image *const image = view.BackingImage();
    if (!image || image->info.num_samples != 1 || image->info.type != VideoCommon::ImageType::e2D)
        return;
    if (!image->HandleMatchesRescaling()) {
        ++dropped;
        return;
    }
    Encoding encoding;
    switch (image->info.format) {
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::A8B8G8R8_SRGB:
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::B8G8R8A8_SRGB:
        if (!tonemap)
            return;
        encoding = Encoding::Rgba8;
        break;
    case PixelFormat::B10G11R11_FLOAT:
        encoding = Encoding::R11G11B10;
        break;
    case PixelFormat::R16G16B16A16_FLOAT:
        encoding = Encoding::Rgba16;
        break;
    default:
        return;
    }
    const u32 level = static_cast<u32>(view.range.base.level);
    const u32 layer = static_cast<u32>(view.range.base.layer);
    const auto &resolution = Settings::values.resolution_info;
    const u32 width = std::max(1U, (image->IsRescaled() ? resolution.ScaleUp(image->info.size.width)
                                                        : image->info.size.width) >>
                                       level);
    const u32 height =
        std::max(1U, (image->IsRescaled() ? resolution.ScaleUp(image->info.size.height)
                                          : image->info.size.height) >>
                         level);
    // A size/format candidate, not an assertion that this is a bloom resource.
    if (!tonemap && (width > 128 || height > 128))
        return;
    if (sampling_frame != frame) {
        sampling_frame = frame;
        captured.clear();
        hdr_samples = 0;
        tonemap_sample = false;
    }
    if (tonemap ? tonemap_sample : hdr_samples >= 4)
        return;
    const std::array<u64, 3> key{image->gpu_addr, level, layer};
    if (std::ranges::find(captured, key) != captured.end())
        return;
    auto slot = std::ranges::find_if(samples, [](const Sample &sample) { return !sample.pending; });
    if (slot == samples.end()) {
        ++dropped;
        return;
    }
    constexpr u32 Capacity = 128 * 128 * 8;
    if (!slot->buffer) {
        slot->buffer = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = Capacity,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::Download);
    }
    const u32 stride = encoding == Encoding::Rgba16 ? 8 : 4;
    const u32 columns = tonemap ? std::min(width, 32U) : width;
    const u32 rows = tonemap ? std::min(height, 18U) : height;
    std::vector<VkBufferImageCopy> copies;
    const VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
    if (tonemap) {
        copies.reserve(columns * rows);
        for (u32 y = 0; y < rows; ++y) {
            for (u32 x = 0; x < columns; ++x) {
                copies.push_back(VkBufferImageCopy{
                    .bufferOffset = static_cast<VkDeviceSize>(y * columns + x) * stride,
                    .imageSubresource = subresource,
                    .imageOffset = {static_cast<s32>((2 * x + 1) * width / (2 * columns)),
                                    static_cast<s32>((2 * y + 1) * height / (2 * rows)), 0},
                    .imageExtent = {1, 1, 1},
                });
            }
        }
    } else {
        copies.push_back(
            VkBufferImageCopy{.imageSubresource = subresource, .imageExtent = {width, height, 1}});
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = image->Handle(), destination = *slot->buffer,
                      copies = std::move(copies)](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        cmdbuf.CopyImageToBuffer(source, VK_IMAGE_LAYOUT_GENERAL, destination, copies);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                               after);
    });
    slot->tick = scheduler.CurrentTick();
    slot->frame = frame;
    slot->address = image->gpu_addr;
    slot->shader = shader;
    slot->width = width;
    slot->height = height;
    slot->level = level;
    slot->layer = layer;
    slot->pixels = columns * rows;
    slot->bytes = columns * rows * stride;
    slot->encoding = encoding;
    slot->tonemap = tonemap;
    slot->pending = true;
    captured.push_back(key);
    if (tonemap)
        tonemap_sample = true;
    else
        ++hdr_samples;
}

void OotWhiteoutDetector::Collect(u64 frame) {
    namespace Trace = VideoCommon::OotWhiteout;
    if (!Trace::Enabled())
        return;
    if (frame % 600 == 0) {
        LOG_INFO(
            Render_Vulkan,
            "OoT whiteout heartbeat: frame={} completed={} dropped={} pending={} grid-unknown={} "
            "argument-read={}",
            frame, completed, dropped,
            std::ranges::count_if(samples, [](const Sample &sample) { return sample.pending; }),
            Trace::counters[static_cast<size_t>(Trace::Event::GridUnknown)].load(
                std::memory_order_relaxed),
            Trace::counters[static_cast<size_t>(Trace::Event::ArgumentRead)].load(
                std::memory_order_relaxed));
    }
    std::vector<Sample *> ready;
    for (auto &sample : samples)
        if (sample.pending && scheduler.IsFree(sample.tick))
            ready.push_back(&sample);
    std::ranges::stable_sort(ready, {}, [](const Sample *sample) { return sample->frame; });
    for (Sample *const pointer : ready) {
        auto &sample = *pointer;
        sample.buffer.Invalidate();
        const auto mapped = sample.buffer.Mapped();
        const auto stats = Trace::Inspect(
            {reinterpret_cast<const u8 *>(mapped.data()), sample.bytes}, sample.encoding);
        const bool suspicious = stats.Suspicious(sample.tonemap);
        auto state = std::ranges::find_if(seen, [&](const Seen &entry) {
            return entry.used && entry.address == sample.address && entry.shader == sample.shader &&
                   entry.level == sample.level && entry.layer == sample.layer;
        });
        if (state == seen.end()) {
            state = std::ranges::find_if(seen, [](const Seen &entry) { return !entry.used; });
            if (state == seen.end())
                state = std::ranges::min_element(seen, {}, &Seen::last_log);
            *state = {.address = sample.address,
                      .shader = sample.shader,
                      .level = sample.level,
                      .layer = sample.layer,
                      .used = true};
        }
        const bool first = !state->observed;
        const bool changed = state->suspicious != suspicious;
        // Log the first sampled suspect frame immediately, then at most every 300 frames per
        // target. A healthy heartbeat and recovery distinguish no samples from a clean sample.
        if (first || changed ||
            (sample.frame >= state->last_log && sample.frame - state->last_log >= 300)) {
            LOG_INFO(Render_Vulkan,
                     "OoT whiteout sample: frame={} tick={} kind={} gpu={:x} shader={:016x} "
                     "phase=mid-frame size={}x{} mip={} layer={} pixels={} nonfinite={} white={} bright={} max={} "
                     "state={} completed={} dropped={}",
                     sample.frame, sample.tick, sample.tonemap ? "tonemap" : "hdr-candidate",
                     sample.address, sample.shader, sample.width, sample.height, sample.level,
                     sample.layer, stats.pixels, stats.nonfinite, stats.white, stats.bright,
                     stats.maximum, suspicious ? "suspect" : "clear", ++completed, dropped);
            state->last_log = sample.frame;
        } else {
            ++completed;
        }
        if (!first && changed && suspicious && (sample.tonemap || stats.nonfinite != 0)) {
            Trace::Dump(sample.frame);
        }
        state->observed = true;
        state->suspicious = suspicious;
        sample.pending = false;
    }
}
} // namespace Vulkan
