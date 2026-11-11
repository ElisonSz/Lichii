// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <vector>
#include "video_core/oot_whiteout_pixels.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class MemoryAllocator;
class ImageView;
class Scheduler;
// Bounded asynchronous diagnostic. Copies only; never submits or waits. Buffers live through
// their completion tick. Rasterizer shutdown finishes GPU work before destroying this object.
class OotWhiteoutDetector {
  public:
    OotWhiteoutDetector(MemoryAllocator &allocator_, Scheduler &scheduler_)
        : allocator{allocator_}, scheduler{scheduler_} {}
    void Capture(ImageView &view, u64 shader, u64 frame, bool tonemap);
    void Collect(u64 frame);

  private:
    struct Sample {
        vk::Buffer buffer;
        u64 tick{}, frame{}, address{}, shader{};
        u32 width{}, height{}, level{}, layer{}, pixels{}, bytes{};
        VideoCommon::OotWhiteout::Encoding encoding{};
        bool tonemap{}, pending{};
    };
    struct Seen {
        u64 address{}, shader{}, last_log{};
        u32 level{}, layer{};
        bool used{}, observed{}, suspicious{};
    };
    MemoryAllocator &allocator;
    Scheduler &scheduler;
    std::array<Sample, 16> samples{};
    std::array<Seen, 32> seen{};
    u64 sampling_frame{~0ULL}, dropped{}, completed{};
    std::vector<std::array<u64, 3>> captured;
    u32 hdr_samples{};
    bool tonemap_sample{};
};
} // namespace Vulkan
