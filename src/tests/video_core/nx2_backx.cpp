// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/gpu_dirty_memory_manager.h"
#include "video_core/buffer_cache/usage_tracker.h"
#include "video_core/nx2_backx.h"

namespace {

namespace BackX = VideoCommon::BackX;

/// Sets one BackX switch until the end of the scope.
class SwitchScope {
public:
    SwitchScope(bool& flag_, bool value) : flag{flag_}, previous{flag_} {
        flag = value;
    }
    ~SwitchScope() {
        flag = previous;
    }
    SwitchScope(const SwitchScope&) = delete;
    SwitchScope& operator=(const SwitchScope&) = delete;

private:
    bool& flag;
    bool previous;
};

u64 Counted(BackX::Counter counter) {
    return BackX::g_counters[static_cast<std::size_t>(counter)].load(std::memory_order_relaxed);
}

/// The 64-byte blocks of [address, address + size).
void AddBlocks(std::set<u64>& blocks, u64 address, u64 size) {
    for (u64 block = address >> 6; block <= (address + size - 1) >> 6; ++block) {
        blocks.insert(block);
    }
}

} // Anonymous namespace

TEST_CASE("BackX tick: bounded usage resets answer as the base's full resets",
          "[video_core][nx2_backx]") {
    // The same operations on two trackers, one with EDEN_NX2_BACKX_TICK (and its verify): every
    // IsUsed answer and every word match, and verify finds no word outside the tracked extent.
    const SwitchScope verify{BackX::g_verify, true};
    const u64 mismatches = Counted(BackX::Counter::Mismatches);
    for (u64 seed = 1; seed <= 40; ++seed) {
        std::mt19937_64 rng{seed};
        const u64 buffer_size = 4096 * (1 + rng() % 300) + rng() % 4096;
        VideoCommon::UsageTracker base{buffer_size};
        VideoCommon::UsageTracker bounded{buffer_size};
        const auto range = [&] {
            const u64 offset = rng() % buffer_size;
            const u64 limit = buffer_size - offset;
            const u64 kind = rng() % 4;
            const u64 size = kind == 0   ? 1 + rng() % 64
                             : kind == 1 ? 1 + rng() % 8192
                             : kind == 2 ? 1 + rng() % (64 * 4096)
                                         : limit;
            return std::pair{offset, (std::min)(size, limit)};
        };
        for (int step = 0; step < 2000; ++step) {
            const u64 op = rng() % 16;
            if (op < 8) {
                const auto [offset, size] = range();
                {
                    const SwitchScope tick{BackX::g_tick, false};
                    base.Track(offset, size);
                }
                {
                    const SwitchScope tick{BackX::g_tick, true};
                    bounded.Track(offset, size);
                }
            } else if (op < 9) {
                {
                    const SwitchScope tick{BackX::g_tick, false};
                    base.Reset();
                }
                {
                    const SwitchScope tick{BackX::g_tick, true};
                    bounded.Reset();
                }
                REQUIRE(std::ranges::equal(base.Pages(), bounded.Pages()));
            } else {
                const auto [offset, size] = range();
                REQUIRE(base.IsUsed(offset, size) == bounded.IsUsed(offset, size));
            }
        }
        REQUIRE(std::ranges::equal(base.Pages(), bounded.Pages()));
    }
    REQUIRE(Counted(BackX::Counter::Mismatches) == mismatches);
}

TEST_CASE("BackX tick: a reset clears only the extent tracked since the last one",
          "[video_core][nx2_backx]") {
    const SwitchScope tick{BackX::g_tick, true};
    const SwitchScope stats{BackX::g_stats, true};
    // 4 MiB + 1: 1025 words.
    VideoCommon::UsageTracker tracker{4 * 1024 * 1024 + 1};
    tracker.Track(5 * 4096 + 100, 3 * 4096);
    const u64 cleared = Counted(BackX::Counter::TickWordsCleared);
    const u64 skipped = Counted(BackX::Counter::TickWordsSkipped);
    tracker.Reset();
    // Words 5..8 were set.
    REQUIRE(Counted(BackX::Counter::TickWordsCleared) - cleared == 4);
    REQUIRE(Counted(BackX::Counter::TickWordsSkipped) - skipped == tracker.Pages().size() - 4);
    REQUIRE(std::ranges::all_of(tracker.Pages(), [](u64 word) { return word == 0; }));
    REQUIRE_FALSE(tracker.IsUsed(5 * 4096 + 100, 1));
    // A reset with nothing tracked since clears nothing.
    tracker.Reset();
    REQUIRE(Counted(BackX::Counter::TickWordsCleared) - cleared == 4);
}

TEST_CASE("BackX report: the page filter holds every page of the set",
          "[video_core][nx2_backx]") {
    // A model of read_back_pages: at most 64 pages, the least recently added one evicted.
    for (u64 seed = 1; seed <= 20; ++seed) {
        std::mt19937_64 rng{seed};
        BackX::PageSetFilter filter;
        std::vector<u64> pages;
        for (int step = 0; step < 5000; ++step) {
            // Pages from a pool larger than the filter's 4096 buckets, so buckets collide.
            const u64 page = (rng() % 20000) << 12;
            if (std::ranges::find(pages, page) == pages.end()) {
                if (pages.size() >= 64) {
                    pages.erase(pages.begin() + static_cast<std::ptrdiff_t>(rng() % pages.size()));
                    filter.Rebuild(pages, [](u64 entry) { return entry; });
                }
                filter.Mark(page);
                pages.push_back(page);
            }
            for (const u64 member : pages) {
                REQUIRE(filter.MayHold(member));
                REQUIRE(filter.MayHold(member + 4095));
            }
        }
        // After a rebuild, a page whose bucket holds no member reads clear.
        filter.Rebuild(pages, [](u64 entry) { return entry; });
        std::set<u64> buckets;
        for (const u64 member : pages) {
            buckets.insert((member >> 12) % 4096);
        }
        for (u64 page = 0; page < 8192; ++page) {
            REQUIRE(filter.MayHold(page << 12) == buckets.contains(page % 4096));
        }
    }
}

TEST_CASE("BackX report: readers without the lock always see pages that stay in the set",
          "[video_core][nx2_backx]") {
    BackX::PageSetFilter filter;
    std::mutex mutex;
    std::vector<u64> pages;
    // Pinned pages enter first and never leave; the others come and go.
    std::vector<u64> pinned;
    for (u64 i = 0; i < 16; ++i) {
        pinned.push_back((i * 7919 + 13) << 12);
    }
    {
        std::scoped_lock lock{mutex};
        for (const u64 page : pinned) {
            filter.Mark(page);
            pages.push_back(page);
        }
    }
    std::atomic<bool> stop{false};
    std::atomic<u64> misses{0};
    std::atomic<u64> reads{0};
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 3; ++reader) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                for (const u64 page : pinned) {
                    if (!filter.MayHold(page)) {
                        misses.fetch_add(1, std::memory_order_relaxed);
                    }
                    reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    std::mt19937_64 rng{7};
    for (int step = 0; step < 200000; ++step) {
        std::scoped_lock lock{mutex};
        const u64 page = (rng() % 100000) << 12;
        if (std::ranges::find(pages, page) != pages.end()) {
            continue;
        }
        if (pages.size() >= 64) {
            // Never a pinned page (the first 16).
            const auto victim = 16 + static_cast<std::ptrdiff_t>(rng() % (pages.size() - 16));
            pages.erase(pages.begin() + victim);
            filter.Rebuild(pages, [](u64 entry) { return entry; });
        }
        filter.Mark(page);
        pages.push_back(page);
    }
    stop.store(true, std::memory_order_relaxed);
    for (std::thread& reader : readers) {
        reader.join();
    }
    REQUIRE(reads.load() > 0);
    REQUIRE(misses.load() == 0);
}

TEST_CASE("BackX gather: a manager shows a collected write until it is gathered",
          "[core][nx2_backx]") {
    Core::GPUDirtyMemoryManager manager;
    std::vector<std::pair<PAddr, size_t>> gathered;
    std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
        gathered.emplace_back(address, size);
    };
    REQUIRE_FALSE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE(gathered.empty());
    REQUIRE_FALSE(manager.MayHavePending());

    // In `current`.
    manager.Collect(0x10040, 0x10);
    REQUIRE(manager.MayHavePending());
    manager.Collect(0x10080, 0x8);
    REQUIRE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE_FALSE(gathered.empty());
    REQUIRE_FALSE(manager.MayHavePending());

    // In back_buffer and `current`.
    gathered.clear();
    manager.Collect(0x20000, 0x40);
    manager.Collect(0x30000, 0x40);
    manager.Collect(0x40000, 0x40);
    REQUIRE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE(gathered.size() == 3);
    REQUIRE_FALSE(manager.MayHavePending());
    gathered.clear();
    manager.Gather(callback);
    REQUIRE(gathered.empty());

    // A write already collected collects nothing new, and still shows until gathered.
    manager.Collect(0x50000, 0x4);
    manager.Collect(0x50000, 0x4);
    REQUIRE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE(gathered.size() == 1);
    REQUIRE_FALSE(manager.MayHavePending());
}

TEST_CASE("BackX gather: gathering only when a manager shows a write loses none",
          "[core][nx2_backx]") {
    // One collecting thread per manager, as each guest core has its own; the GPU thread gathers
    // only when MayHavePending says so, at the same time.
    for (u64 seed = 1; seed <= 8; ++seed) {
        Core::GPUDirtyMemoryManager manager;
        std::set<u64> collected;
        std::set<u64> gathered;
        std::atomic<bool> done{false};
        u64 checks = 0;
        u64 skipped = 0;
        std::thread collector([&] {
            std::mt19937_64 rng{seed};
            for (int i = 0; i < 300000; ++i) {
                // A few 2 KiB pages, so both the compare-exchange and the push path run.
                const u64 page = rng() % (seed % 2 == 0 ? 3 : 64);
                const u64 offset = rng() % 2040;
                const u64 size = 1 + rng() % 8;
                const u64 address = 0x100000 + page * 2048 + offset;
                manager.Collect(address, size);
                AddBlocks(collected, address, size);
            }
            done.store(true, std::memory_order_release);
        });
        std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
            AddBlocks(gathered, address, size);
        };
        while (!done.load(std::memory_order_acquire)) {
            ++checks;
            if (manager.MayHavePending()) {
                manager.Gather(callback);
            } else {
                ++skipped;
            }
        }
        collector.join();
        if (manager.MayHavePending()) {
            manager.Gather(callback);
        }
        INFO("checks " << checks << " skipped " << skipped);
        REQUIRE_FALSE(manager.MayHavePending());
        const size_t before = gathered.size();
        manager.Gather(callback);
        REQUIRE(gathered.size() == before);
        REQUIRE(std::ranges::includes(gathered, collected));
        REQUIRE(checks > 0);
    }
}
