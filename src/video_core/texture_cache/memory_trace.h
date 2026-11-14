// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <map>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

struct ImageTraceShape {
    u32 format;
    u32 type;
    u32 width;
    u32 height;
    u32 depth;
    s32 levels;
    s32 layers;
    u32 samples;
    u32 kind;
    bool scaled;
    u32 host_format{};
    u64 address_space{};
    auto operator<=>(const ImageTraceShape&) const = default;
};

struct ImageTraceRecord {
    ImageTraceShape shape;
    u64 gpu;
    u64 cpu;
    u64 guest_bytes;
    std::array<u64, 3> allocated; // Original/sparse, scaled, compute upload buffer.
    u64 age;
    bool written;
    bool alias;
    bool bad_overlap;
    bool decoding;
};

struct ImageTraceGroup {
    ImageTraceShape shape;
    std::vector<const ImageTraceRecord*> images;
    std::array<u64, 3> allocated{};
    u64 distinct_gpu{};
    u64 distinct_cpu{};
    u64 overlapping_cpu{}; // Intervals intersecting any earlier interval, not pair count.
    u64 guest_union{};
    u64 first_gpu{};
    u64 last_gpu{};
    u64 oldest{};
    u64 written{};
    u64 aliases{};
    u64 bad_overlaps{};
    u64 decoding{};
    u64 recent{};

    u64 Bytes() const {
        return allocated[0] + allocated[1] + allocated[2];
    }
};

// O(n log n), run only in a memory trace. CPU overlaps are evaluated within each shape;
// the cache's alias and bad-overlap flags also describe relationships to other shapes.
inline std::vector<ImageTraceGroup> GroupImageMemory(std::span<const ImageTraceRecord> records) {
    std::map<ImageTraceShape, std::vector<const ImageTraceRecord*>> shapes;
    for (const auto& record : records) {
        shapes[record.shape].push_back(&record);
    }
    std::vector<ImageTraceGroup> groups;
    for (auto& [shape, images] : shapes) {
        ImageTraceGroup group{.shape = shape, .images = std::move(images)};
        std::sort(group.images.begin(), group.images.end(),
                  [](const auto* lhs, const auto* rhs) { return lhs->gpu < rhs->gpu; });
        group.first_gpu = group.images.front()->gpu;
        group.last_gpu = group.images.back()->gpu;
        u64 previous_gpu{};
        bool first = true;
        for (const auto* record : group.images) {
            group.distinct_gpu += first || record->gpu != previous_gpu ? 1 : 0;
            previous_gpu = record->gpu;
            first = false;
            for (size_t i = 0; i < group.allocated.size(); ++i) {
                group.allocated[i] += record->allocated[i];
            }
            group.oldest = (std::max)(group.oldest, record->age);
            group.recent += record->age < 10 ? 1 : 0;
            group.written += record->written ? 1 : 0;
            group.aliases += record->alias ? 1 : 0;
            group.bad_overlaps += record->bad_overlap ? 1 : 0;
            group.decoding += record->decoding ? 1 : 0;
        }
        std::sort(group.images.begin(), group.images.end(),
                  [](const auto* lhs, const auto* rhs) { return lhs->cpu < rhs->cpu; });
        u64 previous_cpu{};
        u64 interval_end{};
        first = true;
        for (const auto* record : group.images) {
            group.distinct_cpu += first || record->cpu != previous_cpu ? 1 : 0;
            previous_cpu = record->cpu;
            const u64 end = record->cpu + record->guest_bytes;
            if (!first && record->cpu < interval_end && record->guest_bytes != 0) {
                ++group.overlapping_cpu;
            }
            const u64 start = first ? record->cpu : (std::max)(record->cpu, interval_end);
            group.guest_union += end > start ? end - start : 0;
            interval_end = first ? end : (std::max)(interval_end, end);
            first = false;
        }
        groups.push_back(std::move(group));
    }
    std::sort(groups.begin(), groups.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.Bytes() > rhs.Bytes(); });
    return groups;
}

} // namespace VideoCommon
