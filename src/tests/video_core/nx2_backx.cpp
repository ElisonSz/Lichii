// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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

TEST_CASE("BackX gather: a write crossing a 2 KiB granule leaves the write before it pending",
          "[core][nx2_backx]") {
    // BuildTransform gives a write crossing the manager's 2 KiB granule an empty mask (an
    // out-of-range shift). Its push moves the transform before it to back_buffer and leaves an
    // empty `current`: a check of `current` alone missed that write (round 1's verify mismatches).
    struct Write {
        PAddr address;
        size_t size;
    };
    constexpr std::array<Write, 4> crossing{
        {{0x107fc, 8}, {0x10700, 0x200}, {0x10400, 0x800}, {0x107c0, 0x41}}};
    for (const Write& write : crossing) {
        Core::GPUDirtyMemoryManager manager;
        manager.Collect(0x20000, 4);
        manager.Collect(write.address, write.size);
        REQUIRE(manager.MayHavePending());
        std::set<u64> gathered;
        std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
            AddBlocks(gathered, address, size);
        };
        manager.Gather(callback);
        REQUIRE(gathered.contains(0x20000 >> 6));
        REQUIRE_FALSE(manager.MayHavePending());
    }
}

TEST_CASE("BackX gather: collect epochs count each Collect twice with verify on",
          "[core][nx2_backx]") {
    const SwitchScope epochs{BackX::g_gather_epochs, true};
    Core::GPUDirtyMemoryManager manager;
    REQUIRE(manager.CollectEpoch() == 0);
    manager.Collect(0x10000, 4);
    REQUIRE(manager.CollectEpoch() == 2);
    // Every return path: nothing new, the push path, a crossing write.
    manager.Collect(0x10000, 4);
    manager.Collect(0x30000, 4);
    manager.Collect(0x107fc, 8);
    REQUIRE(manager.CollectEpoch() == 8);
    {
        const SwitchScope off{BackX::g_gather_epochs, false};
        manager.Collect(0x40000, 4);
    }
    REQUIRE(manager.CollectEpoch() == 8);
}

TEST_CASE("BackX gather: gathering only when a manager shows a write loses none",
          "[core][nx2_backx]") {
    // One collecting thread per manager, as each guest core has its own; the GPU thread gathers
    // only when MayHavePending says so, at the same time. Some writes cross a 2 KiB granule. What
    // the base would gather of each write is what a second manager, gathered after every write,
    // returns for it (crossing writes lose bytes there too: BuildTransform).
    for (u64 seed = 1; seed <= 8; ++seed) {
        Core::GPUDirtyMemoryManager manager;
        Core::GPUDirtyMemoryManager shadow;
        std::set<u64> expected;
        std::set<u64> gathered;
        std::atomic<bool> done{false};
        u64 checks = 0;
        u64 skipped = 0;
        u64 crossing = 0;
        std::thread collector([&] {
            std::mt19937_64 rng{seed};
            std::function<void(PAddr, size_t)> shadow_callback = [&](PAddr address, size_t size) {
                AddBlocks(expected, address, size);
            };
            for (int i = 0; i < 300000; ++i) {
                // A few 2 KiB pages, so both the compare-exchange and the push path run.
                const u64 page = rng() % (seed % 2 == 0 ? 3 : 64);
                const u64 offset = rng() % 2048;
                const u64 size = rng() % 16 == 0 ? 1 + rng() % 0x400 : 1 + rng() % 16;
                const u64 address = 0x100000 + page * 2048 + offset;
                crossing += offset + size > 2048 ? 1 : 0;
                manager.Collect(address, size);
                shadow.Collect(address, size);
                shadow.Gather(shadow_callback);
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
        INFO("checks " << checks << " skipped " << skipped << " crossing writes " << crossing);
        REQUIRE_FALSE(manager.MayHavePending());
        const size_t before = gathered.size();
        manager.Gather(callback);
        REQUIRE(gathered.size() == before);
        REQUIRE(std::ranges::includes(gathered, expected));
        REQUIRE(crossing > 0);
        REQUIRE(checks > 0);
    }
}

TEST_CASE("BackX gather: the verify classification finds no write the check missed",
          "[core][nx2_backx]") {
    // GPU::Impl's verify: read the collect epochs, check, gather anyway; writes found after a skip
    // with an even, unchanged epoch were there at the check (a gap), else they raced with it. The
    // collector pauses after some crossing writes, as a guest core does other work.
    const SwitchScope epochs{BackX::g_gather_epochs, true};
    u64 gaps = 0;
    u64 races = 0;
    for (u64 seed = 1; seed <= 4; ++seed) {
        Core::GPUDirtyMemoryManager manager;
        std::atomic<bool> done{false};
        std::thread collector([&] {
            std::mt19937_64 rng{seed};
            for (int i = 0; i < 100000; ++i) {
                const u64 page = rng() % 8;
                const u64 offset = rng() % 2048;
                const u64 size = 1 + rng() % 64;
                manager.Collect(0x100000 + page * 2048 + offset, size);
                if (offset + size > 2048 && rng() % 4 == 0) {
                    const auto until =
                        std::chrono::steady_clock::now() + std::chrono::microseconds{5};
                    while (std::chrono::steady_clock::now() < until) {
                    }
                }
            }
            done.store(true, std::memory_order_release);
        });
        std::function<void(PAddr, size_t)> callback = [](PAddr, size_t) {};
        bool found = false;
        std::function<void(PAddr, size_t)> verify_callback = [&](PAddr, size_t) { found = true; };
        while (!done.load(std::memory_order_acquire)) {
            const u64 epoch = manager.CollectEpoch();
            if (manager.MayHavePending()) {
                manager.Gather(callback);
                continue;
            }
            found = false;
            manager.Gather(verify_callback);
            if (!found) {
                continue;
            }
            if ((epoch & 1) == 0 && manager.CollectEpoch() == epoch) {
                ++gaps;
            } else {
                ++races;
            }
        }
        collector.join();
    }
    INFO("races " << races);
    REQUIRE(gaps == 0);
}

TEST_CASE("BackX collect split: every byte of a write crossing 2 KiB granules is collected",
          "[core][nx2_backx]") {
    // Without the switch the base loses such writes (BuildTransform); with it each granule gets
    // its own transform. Each write is collected alone and gathered: exactly its 64-byte blocks.
    const SwitchScope split{BackX::g_collect_split, true};
    std::mt19937_64 rng{11};
    u64 crossing = 0;
    for (int i = 0; i < 20000; ++i) {
        Core::GPUDirtyMemoryManager manager;
        const PAddr address = 0x100000 + rng() % 0x4000;
        const size_t size = 1 + rng() % 0x1000;
        crossing += (address % 2048) + size > 2048 ? 1 : 0;
        manager.Collect(address, size);
        std::set<u64> gathered;
        std::function<void(PAddr, size_t)> callback = [&](PAddr a, size_t s) {
            AddBlocks(gathered, a, s);
        };
        manager.Gather(callback);
        std::set<u64> expected;
        AddBlocks(expected, address, size);
        REQUIRE(gathered == expected);
        REQUIRE_FALSE(manager.MayHavePending());
    }
    REQUIRE(crossing > 1000);
}

TEST_CASE("BackX collect split: a racing gatherer gets every byte of every write",
          "[core][nx2_backx]") {
    const SwitchScope split{BackX::g_collect_split, true};
    for (u64 seed = 1; seed <= 4; ++seed) {
        Core::GPUDirtyMemoryManager manager;
        std::set<u64> collected;
        std::set<u64> gathered;
        std::atomic<bool> done{false};
        std::thread collector([&] {
            std::mt19937_64 rng{seed};
            for (int i = 0; i < 100000; ++i) {
                const PAddr address = 0x100000 + rng() % 0x8000;
                const size_t size = rng() % 8 == 0 ? 1 + rng() % 0x1000 : 1 + rng() % 16;
                manager.Collect(address, size);
                AddBlocks(collected, address, size);
            }
            done.store(true, std::memory_order_release);
        });
        std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
            AddBlocks(gathered, address, size);
        };
        while (!done.load(std::memory_order_acquire)) {
            if (manager.MayHavePending()) {
                manager.Gather(callback);
            }
        }
        collector.join();
        if (manager.MayHavePending()) {
            manager.Gather(callback);
        }
        REQUIRE(gathered == collected);
    }
}
