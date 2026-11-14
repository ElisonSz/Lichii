// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/word_manager.h"

namespace {
using Runs = std::vector<std::pair<VAddr, size_t>>;
struct Tracker {
    std::vector<std::pair<Runs, s32>> calls;
    void UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta) {
        calls.emplace_back(Runs{ranges.begin(), ranges.end()}, delta);
    }
};
using Manager = VideoCommon::WordManager<Tracker, 16, 4ULL << 20>;

struct RestoreSwitch {
    bool word{VideoCommon::MemTrack::g_word};
    ~RestoreSwitch() {
        VideoCommon::MemTrack::g_word = word;
    }
};
} // namespace

TEST_CASE("MemTrack single-page marks preserve every state bit and protection update",
          "[video_core][memtrack]") {
    RestoreSwitch restore;
    std::mt19937_64 random{0x0400486024dec000};
    Tracker reference_tracker;
    Tracker optimized_tracker;
    constexpr VAddr Base = 0x5000000;
    Manager reference{Base, reference_tracker};
    Manager optimized{Base, optimized_tracker};
    for (std::size_t step = 0; step < 10000; ++step) {
        // Include independent GPU/CPU/cached/untracked/preflushable bits and clean-word states.
        for (std::size_t index = 0; index < reference.heap.size(); ++index) {
            reference.heap[index] = random();
        }
        optimized.heap = reference.heap;
        for (std::size_t index = 0; index < Manager::num_words; ++index) {
            reference.NoteCleanWord(index);
            optimized.NoteCleanWord(index);
        }
        const u64 offset = random() % (4ULL << 20);
        const u64 page_remaining =
            VideoCommon::BYTES_PER_PAGE - (offset & (VideoCommon::BYTES_PER_PAGE - 1));
        // Some calls cross pages, are empty, or cross the end: those must retain the old path.
        const u64 size = step % 3 == 0 ? random() % 12000 : 1 + random() % page_remaining;
        VideoCommon::MemTrack::g_word = false;
        reference.ChangeRegionState(VideoCommon::Type::CPU, true, Base + offset, size);
        VideoCommon::MemTrack::g_word = true;
        optimized.ChangeRegionState(VideoCommon::Type::CPU, true, Base + offset, size);
        REQUIRE(optimized.heap == reference.heap);
        REQUIRE(optimized.clean_words == reference.clean_words);
        REQUIRE(optimized_tracker.calls == reference_tracker.calls);
        reference_tracker.calls.clear();
        optimized_tracker.calls.clear();
    }
}

TEST_CASE("MemTrack CPU marks unprotect only on the first write and rearm after upload",
          "[video_core][memtrack]") {
    RestoreSwitch restore;
    VideoCommon::MemTrack::g_word = true;
    Tracker tracker;
    constexpr VAddr Base = 0x4000000;
    Manager manager{Base, tracker};
    constexpr u64 Page = VideoCommon::BYTES_PER_PAGE;
    manager.ChangeRegionState(VideoCommon::Type::CPU, false, Base + Page, Page);
    REQUIRE(tracker.calls.size() == 1);
    REQUIRE(tracker.calls.back().second == 1);
    manager.ChangeRegionState(VideoCommon::Type::CPU, true, Base + Page + 7, 4);
    REQUIRE(tracker.calls.size() == 2);
    REQUIRE(tracker.calls.back().first == Runs{{Base + Page, Page}});
    REQUIRE(tracker.calls.back().second == -1);
    manager.ChangeRegionState(VideoCommon::Type::CPU, true, Base + Page + 31, 8);
    REQUIRE(tracker.calls.size() == 2);
    manager.ForEachModifiedRange(VideoCommon::Type::CPU, true, Base + Page, static_cast<s64>(Page),
                                 [](VAddr, size_t) {});
    REQUIRE(tracker.calls.size() == 3);
    REQUIRE(tracker.calls.back().second == 1);
    manager.ChangeRegionState(VideoCommon::Type::CPU, true, Base + Page + 15, 1);
    REQUIRE(tracker.calls.size() == 4);
    REQUIRE(tracker.calls.back().second == -1);
}
