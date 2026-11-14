// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
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

// Ported from jpkto/Lichii 7a5e5357cb (src/tests/video_core/nx2_backx.cpp).

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

/// OPS: an operation that logs its id when it is destroyed (moved-from ones log nothing).
class Tracer {
public:
    Tracer(int id_, std::vector<int>* log_) : id{id_}, log{log_} {}
    Tracer(Tracer&& other) noexcept : id{other.id}, log{std::exchange(other.log, nullptr)} {}
    Tracer& operator=(Tracer&&) = delete;
    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    ~Tracer() {
        if (log != nullptr) {
            log->push_back(id);
        }
    }

private:
    int id;
    std::vector<int>* log;
};

/// OPS: a released fence of the fence manager's batch, with its operation list.
struct TracerEntry {
    std::vector<Tracer> operations;
};

std::vector<TracerEntry> MakeTracerEntries(std::vector<int>& log, int entries, int per_entry) {
    std::vector<TracerEntry> result(static_cast<std::size_t>(entries));
    int id = 0;
    for (TracerEntry& entry : result) {
        entry.operations.reserve(static_cast<std::size_t>(per_entry));
        for (int i = 0; i < per_entry; ++i) {
            entry.operations.emplace_back(id++, &log);
        }
    }
    return result;
}

/// OPS: a std::mutex that runs a hook each time it is given back, as if another thread had
/// taken it in between.
class HookMutex {
public:
    void lock() {
        inner.lock();
        held = true;
        ++locks;
    }
    void unlock() {
        held = false;
        inner.unlock();
        if (hook) {
            hook();
        }
    }

    std::mutex inner;
    bool held = false;
    std::size_t locks = 0;
    std::function<void()> hook;
};

/// OPS: the query cache's registrations (newest location per address, under cache_mutex) and
/// UnregisterPending's step for one completed query, as a model.
struct QueryModel {
    void Register(u64 address, u64 location) {
        std::scoped_lock lock{mutex};
        newest[address] = location;
        address_of[location] = address;
    }

    /// With `mutex` held.
    void Unregister(u64 location) {
        const auto it = newest.find(address_of.at(location));
        if (it != newest.end() && it->second == location) {
            newest.erase(it);
        }
        freed.insert(location);
    }

    std::mutex mutex;
    std::map<u64, u64> newest;
    std::map<u64, u64> address_of;
    std::set<u64> freed;
};

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
    // Bounded resets are always on now (nx2): the base's full reset is a fresh tracker.
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
                base = VideoCommon::UsageTracker{buffer_size};
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
    // A model of read_back_pages (64 pages here; the cache keeps up to 512 by default), a
    // random member evicted.
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
    // BuildTransform gave a write crossing the manager's 2 KiB granule an empty mask (an
    // out-of-range shift). Its push moved the transform before it to back_buffer and left an
    // empty `current`: a check of `current` alone missed that write (round 1's verify mismatches).
    // Collect now takes such a write a granule at a time, so `current` keeps bits.
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

TEST_CASE("BackX gather: a write crossing a 2 KiB page is collected whole and shows",
          "[core][nx2_backx]") {
    // A transform covers 2 KiB in 64-byte bits. A write past its page's 32 bits must still show
    // (MayHavePending) and be gathered to its last byte.
    Core::GPUDirtyMemoryManager manager;
    std::vector<std::pair<PAddr, size_t>> gathered;
    std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
        gathered.emplace_back(address, size);
    };
    for (const auto& [offset, size] :
         {std::pair<u64, u64>{600, 1960}, {0, 4096}, {1000, 3000}, {2047, 2}, {64, 2048}}) {
        CAPTURE(offset, size);
        manager.Collect(0x7000, 0x40); // Something already collected on another page
        manager.Collect(0x10000 + offset, size);
        REQUIRE(manager.MayHavePending());
        gathered.clear();
        manager.Gather(callback);
        std::set<u64> blocks;
        for (const auto& [address, length] : gathered) {
            AddBlocks(blocks, address, length);
        }
        for (u64 block = (0x10000 + offset) >> 6; block <= (0x10000 + offset + size - 1) >> 6;
             ++block) {
            REQUIRE(blocks.contains(block));
        }
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

TEST_CASE("BackX gather: Collects counts the collects that add bits", "[core][nx2_backx]") {
    // The verify tells a race from a miss by this count, so it must move for every collect a
    // gather can find (a page split counting each page) and only when counting is on.
    Core::GPUDirtyMemoryManager manager;
    manager.Collect(0x10000, 0x40);
    REQUIRE(manager.Collects() == 0);
    Core::GPUDirtyMemoryManager::count_collects = true;
    manager.Collect(0x10000, 0x40);
    REQUIRE(manager.Collects() == 0);
    manager.Collect(0x10040, 0x40);
    REQUIRE(manager.Collects() == 1);
    manager.Collect(0x7000, 0x40);
    REQUIRE(manager.Collects() == 2);
    manager.Collect(0x107c0, 0x80);
    REQUIRE(manager.Collects() == 4);
    Core::GPUDirtyMemoryManager::count_collects = false;
}

TEST_CASE("BackX gather: gathering only when a manager shows a write loses none",
          "[core][nx2_backx]") {
    // One collecting thread per manager, as each guest core has its own; the GPU thread gathers
    // only when MayHavePending says so, at the same time. Some writes cross a 2 KiB granule. What
    // the base would gather of each write is what a second manager, gathered after every write,
    // returns for it; now that Collect takes a crossing write a granule at a time (nx2), that is
    // every byte of every write (`collected`).
    for (u64 seed = 1; seed <= 8; ++seed) {
        Core::GPUDirtyMemoryManager manager;
        Core::GPUDirtyMemoryManager shadow;
        std::set<u64> expected;
        std::set<u64> collected;
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
                AddBlocks(collected, address, size);
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
        REQUIRE(std::ranges::includes(gathered, collected));
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
    // The base lost such writes (BuildTransform); Collect now gives each granule its own transform
    // (the switch's split, always on since nx2's fix). Each write is collected alone and gathered:
    // exactly its 64-byte blocks.
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

TEST_CASE("BackX verify: mismatches are counted per check", "[video_core][nx2_backx]") {
    const u64 gather = BackX::Mismatches(BackX::Check::Gather);
    const u64 report = BackX::Mismatches(BackX::Check::Report);
    const u64 total = Counted(BackX::Counter::Mismatches);
    for (int i = 0; i < 100; ++i) {
        BackX::Mismatch(BackX::Check::Gather, "test", 0, 0);
    }
    BackX::Mismatch(BackX::Check::Report, "test", 0x1000, 4);
    REQUIRE(BackX::Mismatches(BackX::Check::Gather) - gather == 100);
    REQUIRE(BackX::Mismatches(BackX::Check::Report) - report == 1);
    REQUIRE(Counted(BackX::Counter::Mismatches) - total == 101);
}

TEST_CASE("BackX verify: nx2's mismatches count as Other, its skipped gathers' misses as Gather",
          "[video_core][nx2_backx]") {
    const SwitchScope stats{BackX::g_stats, true};
    const u64 other = BackX::Mismatches(BackX::Check::Other);
    const u64 gather = BackX::Mismatches(BackX::Check::Gather);
    const u64 total = Counted(BackX::Counter::Mismatches);
    const u64 races = Counted(BackX::Counter::GatherRaces);
    BackX::Mismatch("test", 0x2000, 8);
    REQUIRE(BackX::Mismatches(BackX::Check::Other) - other == 1);
    // A skip whose gather found nothing, or writes collected after its check, missed nothing.
    BackX::SkippedGather(false, false, {});
    BackX::SkippedGather(false, true, {});
    BackX::SkippedGather(true, true, {0x3000, 0x40});
    REQUIRE(BackX::Mismatches(BackX::Check::Gather) == gather);
    REQUIRE(Counted(BackX::Counter::GatherRaces) - races == 1);
    BackX::SkippedGather(true, false, {0x3000, 0x40});
    REQUIRE(BackX::Mismatches(BackX::Check::Gather) - gather == 1);
    REQUIRE(Counted(BackX::Counter::Mismatches) - total == 2);
    REQUIRE(Counted(BackX::Counter::GatherRaces) - races == 1);
}

TEST_CASE("BackX ops: a recycled list comes back empty with the storage it grew to",
          "[video_core][nx2_backx]") {
    BackX::ListRecycler<std::vector<int>, 4, 1024> recycler;
    std::vector<int> list;
    REQUIRE_FALSE(recycler.Take(list));
    REQUIRE(list.capacity() == 0);

    struct Entry {
        std::vector<int> operations;
    };
    std::vector<Entry> entries(3);
    std::vector<const int*> storage;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        entries[i].operations.assign(100 * (i + 1), 7);
        storage.push_back(entries[i].operations.data());
    }
    const auto list_of = [](Entry& entry) -> std::vector<int>& { return entry.operations; };
    recycler.Give(entries, list_of);
    REQUIRE(recycler.Kept() == 3);
    for (const Entry& entry : entries) {
        REQUIRE(entry.operations.empty());
    }
    // The last list given comes back first, empty, with its storage.
    for (std::size_t i = entries.size(); i-- > 0;) {
        std::vector<int> taken;
        REQUIRE(recycler.Take(taken));
        REQUIRE(taken.empty());
        REQUIRE(taken.capacity() >= 100 * (i + 1));
        REQUIRE(taken.data() == storage[i]);
    }
    REQUIRE(recycler.Kept() == 0);
    REQUIRE_FALSE(recycler.Take(list));
}

TEST_CASE("BackX ops: lists are emptied as destroying their entries would, and few are kept",
          "[video_core][nx2_backx]") {
    // The base: batch.clear() destroys the entries and so their operations.
    std::vector<int> base_log;
    {
        std::vector<TracerEntry> entries = MakeTracerEntries(base_log, 6, 3);
        entries.clear();
    }
    REQUIRE(base_log.size() == 18);

    std::vector<int> log;
    BackX::ListRecycler<std::vector<Tracer>, 4, 1024> recycler;
    std::vector<TracerEntry> entries = MakeTracerEntries(log, 6, 3);
    entries[1].operations.shrink_to_fit(); // still holds its three
    entries.emplace_back();                // no storage: not kept
    recycler.Give(entries,
                  [](TracerEntry& entry) -> std::vector<Tracer>& { return entry.operations; });
    REQUIRE(log == base_log);
    REQUIRE(recycler.Kept() == 4);
    // Entries past the fourth keep their (empty) lists for their own destruction.
    REQUIRE(entries[4].operations.capacity() != 0);
    REQUIRE(entries[5].operations.capacity() != 0);
    entries.clear();
    REQUIRE(log == base_log);

    // A list without storage is not kept, and one grown past MaxCapacity is freed as before.
    BackX::ListRecycler<std::vector<int>, 4, 64> small;
    struct Entry {
        std::vector<int> operations;
    };
    std::vector<Entry> lists(3);
    lists[1].operations.reserve(65);
    lists[2].operations.reserve(64);
    small.Give(lists, [](Entry& entry) -> std::vector<int>& { return entry.operations; });
    REQUIRE(small.Kept() == 1);
    std::vector<int> taken;
    REQUIRE(small.Take(taken));
    REQUIRE(taken.capacity() == 64);
}

TEST_CASE("BackX ops: a filling thread and an emptying thread run every operation once in order",
          "[video_core][nx2_backx]") {
    using OperationList = std::vector<std::function<void()>>;
    BackX::ListRecycler<OperationList, 16, 4096> recycler;
    std::mutex queue_guard;
    std::deque<OperationList> queue; // pending_operations
    std::atomic<bool> done{false};
    std::vector<u64> ran;
    u64 reused = 0;
    bool taken_not_empty = false; // Catch2 asserts on this thread only
    constexpr u64 Operations = 200000;
    std::thread filler([&] { // SignalFence
        std::mt19937_64 rng{42};
        OperationList list;
        list.reserve(64);
        for (u64 next = 0; next < Operations;) {
            const u64 count = 1 + rng() % 200;
            for (u64 i = 0; i < count && next < Operations; ++i, ++next) {
                list.emplace_back([&ran, next] { ran.push_back(next); });
            }
            {
                std::scoped_lock lock{queue_guard};
                queue.emplace_back(std::move(list));
            }
            list.clear();
            if (recycler.Take(list)) {
                ++reused;
                taken_not_empty |= !list.empty();
            } else {
                list.reserve(64);
            }
        }
        done.store(true, std::memory_order_release);
    });
    struct Released {
        OperationList operations;
    };
    std::vector<Released> batch; // ReleaseThreadFunc
    while (true) {
        const bool last = done.load(std::memory_order_acquire);
        {
            std::scoped_lock lock{queue_guard};
            while (!queue.empty() && batch.size() < 8) {
                batch.push_back({std::move(queue.front())});
                queue.pop_front();
            }
        }
        if (batch.empty()) {
            if (last) {
                break;
            }
            std::this_thread::yield();
            continue;
        }
        for (Released& released : batch) {
            for (auto& operation : released.operations) {
                operation();
            }
        }
        recycler.Give(batch,
                      [](Released& released) -> OperationList& { return released.operations; });
        batch.clear();
    }
    filler.join();
    REQUIRE_FALSE(taken_not_empty);
    REQUIRE(ran.size() == Operations);
    for (u64 i = 0; i < Operations; ++i) {
        REQUIRE(ran[i] == i);
    }
    REQUIRE(reused > 0);
    REQUIRE(recycler.Kept() <= 16);
}

TEST_CASE("BackX ops: chunked passes visit every element once, in order, with the lock held",
          "[video_core][nx2_backx]") {
    for (const std::size_t size : {std::size_t{0}, std::size_t{1}, std::size_t{31}, std::size_t{32},
                                   std::size_t{33}, std::size_t{100}, std::size_t{1000}}) {
        std::vector<std::size_t> items(size);
        for (std::size_t i = 0; i < size; ++i) {
            items[i] = i;
        }
        HookMutex mutex;
        std::vector<std::size_t> visited;
        std::vector<std::size_t> released_after; // elements done at each unlock
        mutex.hook = [&] { released_after.push_back(visited.size()); };
        const std::size_t holds =
            BackX::ForEachLockedInChunks<32>(items, mutex, [&](std::size_t item) {
                REQUIRE(mutex.held);
                visited.push_back(item);
            });
        REQUIRE(visited == items);
        REQUIRE(holds == (size + 31) / 32);
        REQUIRE(mutex.locks == holds);
        REQUIRE(released_after.size() == holds);
        for (std::size_t i = 0; i < released_after.size(); ++i) {
            REQUIRE(released_after[i] == (std::min)(size, (i + 1) * 32));
        }
    }
}

TEST_CASE("BackX ops: reports between unregister chunks leave each address as the base would",
          "[video_core][nx2_backx]") {
    for (u64 seed = 1; seed <= 20; ++seed) {
        std::mt19937_64 rng{seed};
        QueryModel model;
        // Registered locations (1..400) over 64 addresses, so many are rewritten; about half
        // completed and wait in pending_unregister.
        std::vector<u64> pending;
        for (u64 location = 1; location <= 400; ++location) {
            model.Register(rng() % 64, location);
            if (rng() % 2 == 0) {
                pending.push_back(location);
            }
        }
        const std::map<u64, u64> before = model.newest;
        const std::set<u64> pending_set(pending.begin(), pending.end());
        // Reports register new locations between chunks.
        HookMutex mutex;
        std::map<u64, u64> reported; // address -> last location a report registered
        u64 next_location = 1000;
        mutex.hook = [&] {
            for (int i = 0; i < 3; ++i) {
                const u64 address = rng() % 64;
                std::scoped_lock lock{mutex.inner};
                model.newest[address] = next_location;
                model.address_of[next_location] = address;
                reported[address] = next_location++;
            }
        };
        BackX::ForEachLockedInChunks<32>(pending, mutex,
                                         [&](u64 location) { model.Unregister(location); });
        REQUIRE(model.freed == pending_set);
        for (u64 address = 0; address < 64; ++address) {
            const auto now = model.newest.find(address);
            if (const auto it = reported.find(address); it != reported.end()) {
                // A report's location stays: no completed query erases it.
                REQUIRE(now != model.newest.end());
                REQUIRE(now->second == it->second);
                continue;
            }
            // Untouched by reports: the base's pass, whole.
            const auto old = before.find(address);
            if (old == before.end() || pending_set.contains(old->second)) {
                REQUIRE(now == model.newest.end());
            } else {
                REQUIRE(now != model.newest.end());
                REQUIRE(now->second == old->second);
            }
        }
        // No freed query's location stays registered.
        for (const auto& [address, location] : model.newest) {
            REQUIRE_FALSE(model.freed.contains(location));
        }
    }
}

TEST_CASE("BackX ops: a report thread racing chunked passes never sees a freed location",
          "[video_core][nx2_backx]") {
    QueryModel model;
    std::vector<u64> pending;
    std::mt19937_64 rng{7};
    for (u64 location = 1; location <= 20000; ++location) {
        model.Register(rng() % 256, location);
        if (rng() % 2 == 0) {
            pending.push_back(location);
        }
    }
    std::atomic<bool> done{false};
    std::atomic<u64> reports{0};
    bool freed_seen = false; // Catch2 asserts on this thread only
    std::thread reporter([&] {
        std::mt19937_64 report_rng{8};
        for (u64 location = 100000; !done.load(std::memory_order_acquire); ++location) {
            const u64 address = report_rng() % 256;
            std::scoped_lock lock{model.mutex};
            // CounterReport: the location it replaces must not be one already freed.
            if (const auto it = model.newest.find(address); it != model.newest.end()) {
                freed_seen |= model.freed.contains(it->second);
            }
            model.newest[address] = location;
            model.address_of[location] = address;
            reports.fetch_add(1, std::memory_order_release);
        }
    });
    while (reports.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }
    BackX::ForEachLockedInChunks<32>(pending, model.mutex,
                                     [&](u64 location) { model.Unregister(location); });
    done.store(true, std::memory_order_release);
    reporter.join();
    REQUIRE_FALSE(freed_seen);
    REQUIRE(model.freed.size() == pending.size());
    for (const auto& [address, location] : model.newest) {
        REQUIRE_FALSE(model.freed.contains(location));
    }
}
