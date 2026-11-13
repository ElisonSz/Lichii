// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

/// The GPU back thread's work outside draw preparation (EDEN_NX2_BACKX_*): locks and passes the
/// base takes where an exact check without the lock shows they would change nothing. Guest CPU
/// write traps hold the buffer cache's lock across page protection system calls, so every
/// acquisition the back makes can meet one and spin (3 % of the back's time on Castle Town).
/// nx2 ported the gather, inline write, usage reset and apply parts (7a5e5357cb and f5edeabdbb)
/// as unconditional behaviour and fixed crossing writes as COLLECT_SPLIT does;
/// EDEN_NX2_BACKX_VERIFY checks its skips too (the Mismatch without a Check, SkippedGather).
namespace VideoCommon::BackX {

/// EDEN_NX2_BACKX_INLINE=1: an inline-to-memory write takes the buffer cache's lock only when one
/// of its pages may hold GPU-written data or a command list (otherwise InlineMemory would find
/// nothing to upload), and invalidates the query cache only where its page filter marks a query.
inline bool g_inline{};
/// EDEN_NX2_BACKX_REPORT=1: a report takes the buffer cache's lock (NoteReportPending) only when
/// its page may be one the guest read GPU-written data from (read-back pages).
inline bool g_report{};
/// EDEN_NX2_BACKX_GATHER=1: InvalidateGPUCache gathers collected CPU writes (two locks per core's
/// manager) only when a manager may hold one. nx2 does that always now; with _VERIFY the switch
/// picks its own verify (CollectEpoch) over nx2's (Collects, SkippedGather).
inline bool g_gather{};
/// _GATHER with _VERIFY: GPUDirtyMemoryManager::Collect counts its calls (CollectEpoch), so a
/// verify gather that finds writes tells a Collect racing with the check from a check that missed
/// a write.
inline bool g_gather_epochs{};
/// EDEN_NX2_BACKX_COLLECT_SPLIT=1: a CPU write into GPU-written memory that crosses one of the
/// GPUDirtyMemoryManager's 2 KiB granules is collected granule by granule. The base built one
/// transform from the write's first granule with a mask shift out of range past it: such a write
/// lost its bytes (often all), so they never reached the GPU's copy, and a later download could
/// write the GPU's older bytes over them in guest memory. nx2 fixed the base the same way, so
/// Collect always splits now and the switch changes nothing; the stats line counts crossing
/// writes ("crossing writes").
inline bool g_collect_split{};
/// EDEN_NX2_BACKX_TICK=1: a buffer's usage tracker clears only the words it set since its last
/// reset (BufferCacheRuntime::TickFrame cleared every word of every used buffer each frame). nx2
/// made that unconditional, so the switch changes nothing now; the stats line still counts the
/// resets and their words.
inline bool g_tick{};
/// EDEN_NX2_BACKX_APPLY=1: collected CPU writes are applied with each cache's lock taken once for
/// all of them (OnCacheInvalidations took the buffer cache's twice, the texture cache's once, per
/// write). The caches are independent and their page counts commute.
inline bool g_apply{};
/// EDEN_NX2_BACKX_SUBMIT_DRAWS=n (1 to 4096): a fence flush Scheduler::DeferFlush deferred is
/// submitted after at most n more draws, dispatches or clears (RasterizerVulkan::FlushWork),
/// instead of with the next budget flush (128 draws), a fence 500 us after the last flush or the
/// back's idle hook. Only when or where a busy back holds fences back (track S's run-ahead): it
/// adds submissions. 0 (off) by default.
inline u32 g_submit_draws{};
/// EDEN_NX2_BACKX_OPS=1: the fence manager reuses operation lists the fencing thread emptied
/// instead of allocating one per fence, and the query cache's UnregisterPending gives
/// cache_mutex back every 32 entries (the back's reports waited for whole passes).
inline bool g_ops{};
/// EDEN_NX2_BACKX_FLUSH=1: the in-place upload attempt of the GPU thread's own writes
/// (EDEN_NX2_PROTECT_INLINE) asks the tracker first, so the attempts it declines skip the page
/// table lookup and the GPU-written range search.
inline bool g_flush{};
/// EDEN_NX2_BACKX_VERIFY=1: every skip above, and every one nx2 takes unconditionally, also runs
/// the base's own check, logs "BackX verify mismatch" where that check would have done something,
/// and then does what the base does. The GPU dirty memory managers count their collects
/// (Core::GPUDirtyMemoryManager::count_collects).
inline bool g_verify{};
/// EDEN_NX2_BACKX_STATS=1: one "BackX" line every 10 s with the counters below.
inline bool g_stats{};

enum class Counter : std::size_t {
    InlineFast,       ///< inline writes that skipped the buffer cache's lock
    InlineLocked,     ///< inline writes that took it
    QuerySkipped,     ///< inline writes whose query cache invalidation was skipped
    QueryRun,         ///< inline writes that invalidated the query cache
    ReportFast,       ///< reports that skipped the lock of NoteReportPending
    ReportLocked,     ///< reports that took it
    GatherSkipped,    ///< InvalidateGPUCache calls that found no manager with a collected write
    GatherRun,        ///< calls that gathered
    GatherEmpty,      ///< gathers that found nothing (a write collected and gathered meanwhile)
    GatherRaces,      ///< verify: skipped gathers' writes collected after the check (as the base)
    CrossingWrites,   ///< collected CPU writes crossing a 2 KiB granule (split)
    TickResets,       ///< usage tracker resets
    TickWordsCleared, ///< words cleared by them
    TickWordsSkipped, ///< words the base would have cleared too (already zero)
    ApplyPasses,      ///< OnCacheInvalidations calls with each lock taken once
    ApplyRanges,      ///< collected writes they applied
    EarlySubmits,     ///< deferred fence flushes submitted after EDEN_NX2_BACKX_SUBMIT_DRAWS
    OpsReused,        ///< operation lists reused instead of allocated
    OpsAllocated,     ///< operation lists the free list could not give
    UnregisterPasses, ///< UnregisterPending passes with entries
    UnregisterChunks, ///< their cache_mutex holds
    TrackerFirst,     ///< in-place attempts declined by the tracker before the range search
    Mismatches,       ///< verify: skips the base's check contradicted
    Count,
};

inline std::array<std::atomic<u64>, static_cast<std::size_t>(Counter::Count)> g_counters{};

/// Counts only with the stats or verify line on (else one predictable branch).
inline void Count(Counter counter, u64 amount = 1) {
    if (g_stats || g_verify) [[unlikely]] {
        g_counters[static_cast<std::size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
    }
}

/// Verify: the skips the base's own check can contradict.
enum class Check : std::size_t {
    InlineGpuData, ///< INLINE: an inline write into GPU-written buffer data
    InlineQuery,   ///< INLINE: an inline write over a cached query
    Report,        ///< REPORT: a report on a read-back page
    Gather,        ///< GATHER: collected CPU writes no manager showed (no Collect raced)
    Tick,          ///< TICK: a usage word set outside the tracked extent
    Other,         ///< a skip named by its message alone (nx2's Mismatch without a Check)
    Count,
};

/// Verify mismatches of each check since the start (each kind logged 16 times at most, Other 64
/// times as nx2 logs them, so one kind cannot hide the others).
inline std::array<std::atomic<u64>, static_cast<std::size_t>(Check::Count)> g_mismatches{};

/// Verify: a skip the base's own check contradicts.
inline void Mismatch(Check check, const char* what, u64 address, u64 size) {
    g_counters[static_cast<std::size_t>(Counter::Mismatches)].fetch_add(1,
                                                                        std::memory_order_relaxed);
    const u64 seen =
        g_mismatches[static_cast<std::size_t>(check)].fetch_add(1, std::memory_order_relaxed);
    if (seen < (check == Check::Other ? 64U : 16U)) {
        LOG_ERROR(HW_GPU, "BackX verify mismatch: {} at {:#x}+{:#x}", what, address, size);
    }
}

/// Verify: the same, for a caller that names the skip by `what` alone (nx2's form): Check::Other.
inline void Mismatch(const char* what, u64 address, u64 size) {
    Mismatch(Check::Other, what, address, size);
}

/// Verify of a gather that nx2's InvalidateGPUCache skipped: `found` when the gather found writes,
/// `collected` when a manager collected one after the skip's check (a race the next call covers
/// either way). Writes found with nothing collected since are a miss (Check::Gather). Logs the
/// counts every 10 s; the stats line counts the races too (GatherRaces, as GATHER's own verify).
inline void SkippedGather(bool found, bool collected, std::pair<u64, u64> first_write) {
    static std::mutex mutex;
    static u64 skips{};
    static u64 races{};
    static u64 misses{};
    static auto last_log = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++skips;
    if (found && collected) {
        ++races;
        Count(Counter::GatherRaces);
    } else if (found) {
        ++misses;
        Mismatch(Check::Gather, "collected CPU write while no manager showed one",
                 first_write.first, first_write.second);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log < std::chrono::seconds{10}) {
        return;
    }
    LOG_INFO(HW_GPU,
             "BackX gather checks in 10 s: {} skipped gathers, {} found writes collected after "
             "the check, {} found writes no manager showed (must stay 0)",
             skips, races, misses);
    skips = races = misses = 0;
    last_log = now;
}

/// EDEN_NX2_BACKX_REPORT: whether a device page may be in a small set of pages that changes under
/// a lock while others ask without it. One bit per page index modulo 4096. A page's bit is set
/// before the page enters the set; after pages leave, Rebuild stores every word anew from the
/// pages still in it, so a page that stays in the set keeps its bit in each value a reader can
/// load. Mark and Rebuild need the set's lock; MayHold does not.
class PageSetFilter {
public:
    static constexpr u64 PageBits = 12;
    static constexpr std::size_t NumWords = 64;

    void Mark(u64 address) noexcept {
        const u64 index = Index(address);
        words[index / 64].fetch_or(u64{1} << (index % 64), std::memory_order_release);
    }

    /// `pages`: the device addresses of the pages now in the set.
    template <typename Range, typename Projection>
    void Rebuild(const Range& pages, Projection&& projection) noexcept {
        std::array<u64, NumWords> rebuilt{};
        for (const auto& entry : pages) {
            const u64 index = Index(projection(entry));
            rebuilt[index / 64] |= u64{1} << (index % 64);
        }
        for (std::size_t i = 0; i < NumWords; ++i) {
            words[i].store(rebuilt[i], std::memory_order_release);
        }
    }

    [[nodiscard]] bool MayHold(u64 address) const noexcept {
        const u64 index = Index(address);
        return ((words[index / 64].load(std::memory_order_acquire) >> (index % 64)) & 1) != 0;
    }

private:
    [[nodiscard]] static constexpr u64 Index(u64 address) noexcept {
        return (address >> PageBits) % (NumWords * 64);
    }

    std::array<std::atomic<u64>, NumWords> words{};
};

/// EDEN_NX2_BACKX_OPS: lists one thread empties and hands back for another to fill again, so the
/// filling thread reuses their storage instead of allocating a list (and regrowing it) per use.
/// Keeps at most MaxLists lists of at most MaxCapacity elements' storage.
template <typename List, std::size_t MaxLists, std::size_t MaxCapacity>
class ListRecycler {
public:
    ListRecycler() {
        lists.reserve(MaxLists);
    }

    /// Swaps a kept list into `list`, which is empty; false, leaving `list` alone, when none is
    /// kept.
    [[nodiscard]] bool Take(List& list) {
        std::scoped_lock lock{guard};
        if (lists.empty()) {
            return false;
        }
        list.swap(lists.back());
        lists.pop_back();
        return true;
    }

    /// Empties the list `list_of(entry)` of each of `entries` in their order, as destroying the
    /// entries would, then keeps those lists' storage while fewer than MaxLists are kept.
    template <typename Entries, typename ListOf>
    void Give(Entries& entries, ListOf&& list_of) {
        for (auto& entry : entries) {
            list_of(entry).clear();
        }
        std::scoped_lock lock{guard};
        for (auto& entry : entries) {
            if (lists.size() >= MaxLists) {
                return;
            }
            List& list = list_of(entry);
            if (list.capacity() != 0 && list.capacity() <= MaxCapacity) {
                lists.push_back(std::move(list));
            }
        }
    }

    [[nodiscard]] std::size_t Kept() {
        std::scoped_lock lock{guard};
        return lists.size();
    }

private:
    std::mutex guard;
    std::vector<List> lists;
};

/// EDEN_NX2_BACKX_OPS: calls `func` on the elements of `items` in order with `mutex` held, taking
/// it anew for every Chunk elements, so that threads waiting for it get it in between. Returns how
/// often it took the mutex.
template <std::size_t Chunk, typename Items, typename Mutex, typename Func>
std::size_t ForEachLockedInChunks(const Items& items, Mutex& mutex, Func&& func) {
    static_assert(Chunk > 0);
    std::size_t holds = 0;
    for (std::size_t begin = 0; begin < items.size(); begin += Chunk) {
        const std::size_t end = (std::min)(items.size(), begin + Chunk);
        std::scoped_lock lock{mutex};
        for (std::size_t i = begin; i < end; ++i) {
            func(items[i]);
        }
        ++holds;
    }
    return holds;
}

[[nodiscard]] inline bool ReadFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

/// Called by VideoCore::CreateGPU, before the caches exist.
inline void ReadSwitches() {
    g_inline = ReadFlag("EDEN_NX2_BACKX_INLINE");
    g_report = ReadFlag("EDEN_NX2_BACKX_REPORT");
    g_gather = ReadFlag("EDEN_NX2_BACKX_GATHER");
    g_collect_split = ReadFlag("EDEN_NX2_BACKX_COLLECT_SPLIT");
    g_tick = ReadFlag("EDEN_NX2_BACKX_TICK");
    g_apply = ReadFlag("EDEN_NX2_BACKX_APPLY");
    g_ops = ReadFlag("EDEN_NX2_BACKX_OPS");
    g_flush = ReadFlag("EDEN_NX2_BACKX_FLUSH");
    g_submit_draws = 0;
    if (const char* value = std::getenv("EDEN_NX2_BACKX_SUBMIT_DRAWS")) {
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        g_submit_draws = parsed <= 4096 ? static_cast<u32>(parsed) : 4096U;
    }
    g_verify = ReadFlag("EDEN_NX2_BACKX_VERIFY");
    g_stats = ReadFlag("EDEN_NX2_BACKX_STATS");
    g_gather_epochs = g_gather && g_verify;
    if (g_inline || g_report || g_gather || g_collect_split || g_tick || g_apply ||
        g_submit_draws != 0 || g_ops || g_flush || g_verify || g_stats) {
        LOG_INFO(HW_GPU,
                 "BackX: inline {} report {} gather {} collect split {} tick {} apply {} submit "
                 "after {} draws ops {} flush {} verify {} stats {}",
                 g_inline, g_report, g_gather, g_collect_split, g_tick, g_apply, g_submit_draws,
                 g_ops, g_flush, g_verify, g_stats);
    }
}

/// nx2's name for ReadSwitches (its version read EDEN_NX2_BACKX_VERIFY alone).
inline void ReadSwitch() {
    ReadSwitches();
}

[[nodiscard]] inline u64 Mismatches(Check check) {
    return g_mismatches[static_cast<std::size_t>(check)].load(std::memory_order_relaxed);
}

[[nodiscard]] inline u64 Take(Counter counter) {
    return g_counters[static_cast<std::size_t>(counter)].exchange(0, std::memory_order_relaxed);
}

/// The stats line, at most once every 10 s (called once a frame by RasterizerVulkan::TickFrame).
inline void Report() {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point window_start = Clock::now();
    const Clock::time_point now = Clock::now();
    if (now - window_start < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    window_start = now;
    const u64 inline_fast = Take(Counter::InlineFast);
    const u64 inline_locked = Take(Counter::InlineLocked);
    const u64 query_skipped = Take(Counter::QuerySkipped);
    const u64 query_run = Take(Counter::QueryRun);
    const u64 report_fast = Take(Counter::ReportFast);
    const u64 report_locked = Take(Counter::ReportLocked);
    const u64 gather_skipped = Take(Counter::GatherSkipped);
    const u64 gather_run = Take(Counter::GatherRun);
    const u64 gather_empty = Take(Counter::GatherEmpty);
    const u64 gather_races = Take(Counter::GatherRaces);
    const u64 crossing_writes = Take(Counter::CrossingWrites);
    const u64 tick_resets = Take(Counter::TickResets);
    const u64 tick_cleared = Take(Counter::TickWordsCleared);
    const u64 tick_skipped = Take(Counter::TickWordsSkipped);
    const u64 apply_passes = Take(Counter::ApplyPasses);
    const u64 apply_ranges = Take(Counter::ApplyRanges);
    const u64 early_submits = Take(Counter::EarlySubmits);
    const u64 mismatches = Take(Counter::Mismatches);
    LOG_INFO(HW_GPU,
             "BackX {:.1f} s: inline writes {} without the buffer lock, {} with it; query "
             "invalidations {} skipped, {} run; reports {} without the buffer lock, {} with it; "
             "gathers {} skipped, {} run ({} found nothing, {} skipped with writes collected after "
             "the check); crossing writes {} (split); usage resets {} clearing {} words, {} words "
             "left alone; collected writes {} in {} passes; deferred fence flushes submitted "
             "early {}; verify mismatches {} (since start: inline {}, query {}, report {}, gather "
             "{}, tick {}, other {})",
             seconds, inline_fast, inline_locked, query_skipped, query_run, report_fast,
             report_locked, gather_skipped, gather_run, gather_empty, gather_races, crossing_writes,
             tick_resets, tick_cleared, tick_skipped, apply_ranges, apply_passes, early_submits,
             mismatches, Mismatches(Check::InlineGpuData), Mismatches(Check::InlineQuery),
             Mismatches(Check::Report), Mismatches(Check::Gather), Mismatches(Check::Tick),
             Mismatches(Check::Other));
    // OPS and FLUSH count only while switched on.
    const u64 ops_reused = Take(Counter::OpsReused);
    const u64 ops_allocated = Take(Counter::OpsAllocated);
    const u64 unregister_passes = Take(Counter::UnregisterPasses);
    const u64 unregister_chunks = Take(Counter::UnregisterChunks);
    const u64 tracker_first = Take(Counter::TrackerFirst);
    LOG_INFO(HW_GPU,
             "BackX3 {:.1f} s: operation lists {} reused, {} allocated; unregister passes {} in {} "
             "lock holds; in-place attempts declined by the tracker first {}",
             seconds, ops_reused, ops_allocated, unregister_passes, unregister_chunks,
             tracker_first);
}

} // namespace VideoCommon::BackX
