// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/lru_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/texture_cache/collection_budget.h"
#include "video_core/texture_cache/memory_accounting.h"
#include "video_core/texture_cache/memory_trace.h"
#include "video_core/texture_cache/types.h"

namespace {
using namespace VideoCommon;

struct LruTraits {
    using ObjectType = size_t;
    using TickType = u64;
};

struct CollectionResult {
    size_t examined{};
    size_t deleted{};
};

// Use the collector's real LRU and budget, with the same skip-before-delete callback order.
CollectionResult Collect(size_t protected_count, size_t eligible_count, bool scan_skips) {
    Common::LeastRecentlyUsedCache<LruTraits> lru;
    for (size_t i = 0; i < protected_count + eligible_count; ++i) {
        lru.Insert(i, 10);
    }
    CollectionBudget budget;
    budget.scan_past_skips = scan_skips;
    budget.Configure(40);
    CollectionResult result;
    lru.ForEachItemBelow(20, [&](size_t image) {
        if (!budget.Examine()) {
            return true;
        }
        ++result.examined;
        if (image < protected_count) {
            return false;
        }
        ++result.deleted;
        budget.Deleted();
        return false;
    });
    return result;
}

ImageTraceRecord Record(u64 gpu, u64 cpu, u64 size, u64 bytes) {
    return {.shape = {16, 1, 1280, 720, 1, 1, 1, 1, 0, false},
            .gpu = gpu,
            .cpu = cpu,
            .guest_bytes = size,
            .allocated = {bytes, 0, 0},
            .age = 20,
            .written = true,
            .alias = false,
            .bad_overlap = false,
            .decoding = false};
}
} // namespace

TEST_CASE("Null texture slots cannot become framebuffer attachments",
          "[video_core][texture_memory]") {
    const std::array<ImageViewId, 4> ids{ImageViewId{}, NULL_IMAGE_VIEW_ID, ImageViewId{1},
                                         ImageViewId{19}};
    size_t attachments{};
    for (const auto id : ids) {
        // GetFramebufferId uses this predicate for both color and depth attachment pointers.
        attachments += IsRenderTargetView(id) ? 1 : 0;
    }
    CHECK(attachments == 2);
    CHECK_FALSE(IsRenderTargetView(NULL_IMAGE_VIEW_ID));
    CHECK_FALSE(IsRenderTargetView(ImageViewId{}));
    CHECK(IsRenderTargetView(ImageViewId{1}));
}

TEST_CASE("Collection reaches an eligible image behind forty decoding images",
          "[video_core][texture_memory]") {
    const auto legacy = Collect(40, 1, false);
    CHECK(legacy.deleted == 0);
    const auto result = Collect(40, 1, true);
    CHECK(result.examined == 41);
    CHECK(result.deleted == 1);
}

TEST_CASE("Collection remains bounded by scans and successful deletions",
          "[video_core][texture_memory]") {
    const auto protected_result = Collect(400, 1, true);
    CHECK(protected_result.examined == 320);
    CHECK(protected_result.deleted == 0);
    const auto eligible_result = Collect(5, 400, true);
    CHECK(eligible_result.examined == 45);
    CHECK(eligible_result.deleted == 40);
}

TEST_CASE("Releasing a scaled image cannot wrap a lagging usage reading",
          "[video_core][texture_memory]") {
    CHECK(ReleaseImageMemory(1, 3, 4) == 0);
    CHECK(ReleaseImageMemory(0, 3, 4) == 0);
    CHECK(ReleaseImageMemory(10, 3, 4) == 3);
    CHECK(ReleaseImageMemory(10, 20, 0) == 0);
    CHECK(ReleaseImageMemory(0, 0, 0) == 0);
    CHECK(ReleaseImageMemory(10, (std::numeric_limits<u64>::max)(), 4) == 0);
}

TEST_CASE("Image trace distinguishes reused addresses from distinct allocations",
          "[video_core][texture_memory]") {
    std::array records{Record(1000, 100, 100, 400), Record(1000, 100, 100, 400),
                       Record(1200, 200, 100, 400)};
    records[0].allocated = {400, 800, 50};
    records[0].alias = true;
    records[1].bad_overlap = true;
    records[1].age = 2;
    const auto groups = GroupImageMemory(records);
    REQUIRE(groups.size() == 1);
    const auto& group = groups.front();
    CHECK(group.images.size() == 3);
    CHECK(group.distinct_gpu == 2);
    CHECK(group.distinct_cpu == 2);
    CHECK(group.overlapping_cpu == 1);
    CHECK(group.guest_union == 200);
    CHECK(group.Bytes() == 2050);
    CHECK(group.written == 3);
    CHECK(group.aliases == 1);
    CHECK(group.bad_overlaps == 1);
    CHECK(group.recent == 1);
    CHECK(group.oldest == 20);
}

TEST_CASE("Image trace union handles nested intervals and touching endpoints",
          "[video_core][texture_memory]") {
    const std::array records{Record(1000, 100, 300, 1), Record(1010, 150, 10, 1),
                             Record(1020, 200, 100, 1), Record(1030, 400, 50, 1)};
    const auto groups = GroupImageMemory(records);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].overlapping_cpu == 2);
    CHECK(groups[0].guest_union == 350);
}

TEST_CASE("Image trace separates guest and host shapes and address spaces",
          "[video_core][texture_memory]") {
    std::array<ImageTraceRecord, 9> records{};
    for (auto& record : records) {
        record = Record(1000, 100, 100, 1);
    }
    records[1].shape.format = 17;
    records[2].shape.kind = 1;
    records[3].shape.levels = 2;
    records[4].shape.layers = 2;
    records[5].shape.samples = 4;
    records[6].shape.scaled = true;
    records[6].allocated[1] = 50;
    records[7].shape.host_format = 97;
    records[8].shape.address_space = 1;
    const auto groups = GroupImageMemory(records);
    REQUIRE(groups.size() == 9);
    CHECK(groups.front().shape.scaled);
    CHECK(groups.front().Bytes() == 51);
    CHECK(GroupImageMemory({}).empty());
}

TEST_CASE("Image ages follow touches and LRU slot reuse", "[video_core][texture_memory]") {
    Common::LeastRecentlyUsedCache<LruTraits> lru;
    const size_t index = lru.Insert(1, 4);
    CHECK(lru.GetTick(index) == 4);
    lru.Touch(index, 8);
    CHECK(lru.GetTick(index) == 8);
    lru.Touch(index, 5);
    CHECK(lru.GetTick(index) == 8);
    lru.Free(index);
    const size_t reused = lru.Insert(2, 9);
    CHECK(lru.GetTick(reused) == 9);
}

TEST_CASE("Retired owner inspection does not change retirement", "[video_core][texture_memory]") {
    DelayedDestructionRing<u64, 2> ring;
    ring.Push(u64{7});
    ring.Tick();
    ring.Push(u64{11});
    u64 bytes{};
    ring.ForEach([&](u64 value) { bytes += value; });
    CHECK(bytes == 18);
    ring.Tick();
    bytes = 0;
    ring.ForEach([&](u64 value) { bytes += value; });
    CHECK(bytes == 11);
}
