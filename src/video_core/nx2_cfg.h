// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>

#include "common/common_types.h"
#include "common/logging.h"

/// Track C (EDEN_NX2_CFG_*): work in draw preparation that repeats although its inputs did not
/// change. Every switch is off by default, read once by VideoCore::CreateGPU.
namespace VideoCommon::Cfg {

/// EDEN_NX2_CFG_TOUCH=1: a buffer or image whose last LeastRecentlyUsedCache::Touch was of the same
/// item at the same frame tick skips the Touch (it would change nothing; see TouchMemo).
inline bool g_touch{};
/// EDEN_NX2_CFG_LOOKUP=1: GraphicsPipeline::Next compares a dense digest per transition before it
/// compares whole keys (vk_graphics_pipeline.h TransitionDigest).
inline bool g_lookup{};
/// EDEN_NX2_CFG_SSBO=1: a graphics storage binding whose range the slot's last
/// VirtualRangeCache::Query found in one piece skips the Query while the range cache has not
/// changed since and no unmap is pending (BufferCache::ResolveMultiRangeStorage).
inline bool g_ssbo{};
/// TEST: EDEN_NX2_CFG_BANK_STATS=1 (diagnostic, costs a copy and a compare per re-upload): counts
/// the once-a-frame re-uploads of native programs' 64 KiB banks
/// (BufferCache::BindHostGraphicsUniformBuffer) and how many found the bank's bytes unchanged
/// since its previous re-upload, the share an exact skip could save.
inline bool g_bank_stats{};
/// EDEN_NX2_CFG_VERIFY=1: every skipped Touch checks that the item's tick already is the frame's,
/// every digest lookup also runs the base's search, and every skipped Query runs; a mismatch is
/// counted, the first 16 are logged, and the base's answer is used.
inline bool g_verify{};
/// Any of the switches: the stats line every 10 s.
inline bool g_any{};

enum class Counter : std::size_t {
    BufferTouchesSkipped, ///< TouchBuffer calls that skipped the LRU cache
    BufferTouches,        ///< TouchBuffer calls that touched it
    ImageTouchesSkipped,  ///< PrepareImage touches that skipped it
    ImageTouches,         ///< PrepareImage touches that touched it
    Searches,             ///< GraphicsPipeline::Next searches past the pipeline's own key
    SearchEntries,        ///< transitions those searches passed over
    SearchCompares,       ///< whole key compares they ran (digest matches)
    SearchHits,           ///< searches that found the key
    RangeQueries,         ///< storage range queries made (graphics, with the memo)
    RangeQueriesSkipped,  ///< storage range queries skipped (or, with verify, checked)
    BankUploads,          ///< bank re-uploads (EDEN_NX2_CFG_BANK_STATS)
    BankBytes,            ///< their bytes
    BankUnchanged,        ///< re-uploads of the bytes the bank's previous re-upload read
    Verified,             ///< verify: skipped touches and searches checked
    Mismatches,           ///< verify: checks that contradicted the skip
    Count,
};

/// On cache lines of their own (the size rounds up to whole lines): the back thread writes them
/// for every touch.
struct alignas(64) Counters {
    std::array<std::atomic<u64>, static_cast<std::size_t>(Counter::Count)> values{};
};
inline Counters g_counters{};

/// The GPU back thread counts: a plain load and store, not a locked add (another thread's rare
/// count may be lost; the counters are statistics).
inline void Count(Counter counter, u64 amount = 1) noexcept {
    auto& value = g_counters.values[static_cast<std::size_t>(counter)];
    value.store(value.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

inline std::atomic<u64> g_mismatches_total{};
/// Mismatches logged in the current stats window (Report resets it).
inline std::atomic<u64> g_mismatches_logged{};

/// Verify: a check contradicted a skip. The caller then does what the base does. The first 16 of
/// each 10 s window are logged.
inline void Mismatch(const char* what, u64 first, u64 second) {
    Count(Counter::Mismatches);
    g_mismatches_total.fetch_add(1, std::memory_order_relaxed);
    if (g_mismatches_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU, "Track C verify mismatch: {} ({:#x}, {:#x})", what, first, second);
    }
}

/// The LRU item and tick of an object's last LeastRecentlyUsedCache::Touch made through it.
/// Touch(id, tick) changes nothing once the item's tick is at least `tick`. An item's tick is set
/// only by Insert (to the tick it is given) and raised only by Touch; Free leaves it. Both caches
/// always pass their current frame tick. So after Touch(id, tick), and for the rest of that tick,
/// a Touch(id, tick) is a no-op, whether the item was freed or handed to another object since.
struct TouchMemo {
    std::size_t id = (std::numeric_limits<std::size_t>::max)();
    u64 tick = 0;
};

/// EDEN_NX2_CFG_TOUCH: true when Touch(id, tick) may be skipped; otherwise notes it, and the
/// caller touches. `item_tick()` reads the item's tick (verify only).
template <typename ItemTick>
[[nodiscard]] inline bool SkipTouch(TouchMemo& memo, std::size_t id, u64 tick, Counter skipped,
                                    Counter touched, ItemTick&& item_tick) {
    if (memo.id == id && memo.tick == tick) {
        if (g_verify) [[unlikely]] {
            Count(Counter::Verified);
            const u64 now = item_tick();
            if (now < tick) {
                Mismatch("LRU touch skipped although the item's tick was older", now, tick);
                return false;
            }
        }
        Count(skipped);
        return true;
    }
    memo.id = id;
    memo.tick = tick;
    Count(touched);
    return false;
}

/// EDEN_NX2_CFG_LOOKUP: one step of a key digest.
[[nodiscard]] constexpr u64 MixDigest(u64 digest, u64 value) noexcept {
    digest = (digest ^ value) * 0xBF58476D1CE4E5B9ULL;
    return digest ^ (digest >> 31);
}

/// EDEN_NX2_CFG_LOOKUP: a digest of what a pipeline key comparison looks at first, the six shader
/// hashes (`unique_hashes`) and the first two words of the fixed state (`state`). For key types
/// whose every comparison covers those words (GraphicsPipelineCacheKey: unique_hashes, then
/// state.Size() bytes, never fewer than 16), equal keys have equal digests.
template <typename Key>
[[nodiscard]] u64 KeyDigest(const Key& key) noexcept {
    std::array<u64, 2> state_words;
    static_assert(sizeof(key.state) >= sizeof(state_words));
    std::memcpy(state_words.data(), &key.state, sizeof(state_words));
    u64 digest = 0x9E3779B97F4A7C15ULL;
    for (const u64 hash : key.unique_hashes) {
        digest = MixDigest(digest, hash);
    }
    digest = MixDigest(digest, state_words[0]);
    return MixDigest(digest, state_words[1]);
}

/// EDEN_NX2_CFG_LOOKUP: the first index whose digest is `digest` and for which `equal(index)`
/// holds, else digests.size(). When equal keys have equal digests, that is the first index whose
/// key equals, the one a search over the keys in order finds. `compares` counts the `equal` calls.
template <typename Equal>
[[nodiscard]] std::size_t FindByDigest(std::span<const u64> digests, u64 digest, Equal&& equal,
                                       u64& compares) {
    for (std::size_t index = 0; index < digests.size(); ++index) {
        if (digests[index] != digest) {
            continue;
        }
        ++compares;
        if (equal(index)) {
            return index;
        }
    }
    return digests.size();
}

[[nodiscard]] inline bool ReadFlag(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

/// Called by VideoCore::CreateGPU, before the caches exist.
inline void ReadSwitches() {
    g_touch = ReadFlag("EDEN_NX2_CFG_TOUCH");
    g_lookup = ReadFlag("EDEN_NX2_CFG_LOOKUP");
    g_ssbo = ReadFlag("EDEN_NX2_CFG_SSBO");
    g_bank_stats = ReadFlag("EDEN_NX2_CFG_BANK_STATS");
    g_verify = ReadFlag("EDEN_NX2_CFG_VERIFY");
    g_any = g_touch || g_lookup || g_ssbo || g_bank_stats || g_verify;
    if (g_any) {
        LOG_INFO(HW_GPU,
                 "Track C draw preparation switches: touch={} lookup={} ssbo={} bank_stats={} "
                 "verify={}",
                 g_touch, g_lookup, g_ssbo, g_bank_stats, g_verify);
    }
}

[[nodiscard]] inline u64 Take(Counter counter) {
    return g_counters.values[static_cast<std::size_t>(counter)].exchange(0,
                                                                         std::memory_order_relaxed);
}

/// The stats line, at most once every 10 s (called once a frame by PipelineCache::TickFrame).
inline void Report() {
    if (!g_any) {
        return;
    }
    using Clock = std::chrono::steady_clock;
    static Clock::time_point window_start = Clock::now();
    const Clock::time_point now = Clock::now();
    if (now - window_start < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    window_start = now;
    g_mismatches_logged.store(0, std::memory_order_relaxed);
    const u64 buffer_skipped = Take(Counter::BufferTouchesSkipped);
    const u64 buffer_touched = Take(Counter::BufferTouches);
    const u64 image_skipped = Take(Counter::ImageTouchesSkipped);
    const u64 image_touched = Take(Counter::ImageTouches);
    const u64 searches = Take(Counter::Searches);
    const u64 entries = Take(Counter::SearchEntries);
    const u64 compares = Take(Counter::SearchCompares);
    const u64 hits = Take(Counter::SearchHits);
    const u64 range_queries = Take(Counter::RangeQueries);
    const u64 range_skipped = Take(Counter::RangeQueriesSkipped);
    const u64 bank_uploads = Take(Counter::BankUploads);
    const u64 bank_bytes = Take(Counter::BankBytes);
    const u64 bank_unchanged = Take(Counter::BankUnchanged);
    const u64 verified = Take(Counter::Verified);
    const u64 mismatches = Take(Counter::Mismatches);
    LOG_INFO(HW_GPU,
             "Track C {:.1f} s: buffer LRU touches {} skipped, {} made; image LRU touches {} "
             "skipped, {} made; pipeline transition searches {} over {} entries, {} whole key "
             "compares, {} found; storage range queries {} skipped, {} made; bank re-uploads {} "
             "({} KiB), {} unchanged; verify checked {}, mismatches {} ({} since start) | "
             "touch={} lookup={} ssbo={} bank_stats={} verify={}",
             seconds, buffer_skipped, buffer_touched, image_skipped, image_touched, searches,
             entries, compares, hits, range_skipped, range_queries, bank_uploads, bank_bytes >> 10,
             bank_unchanged, verified, mismatches,
             g_mismatches_total.load(std::memory_order_relaxed), g_touch, g_lookup, g_ssbo,
             g_bank_stats, g_verify);
}

} // namespace VideoCommon::Cfg
