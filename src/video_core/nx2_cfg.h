// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_tested.h"

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
/// EDEN_NX2_CFG_TIC=1: deleting an image view resolves again only the texture header entries
/// bound to a view deleted since they were read (and those bound to nothing), not every entry of
/// every channel's image tables; texture memos hold while none of their views was deleted
/// (DescriptorTable::ReadImageEntry, TextureCache::NoteDeletedViews).
inline bool g_tic{};
/// EDEN_NX2_CFG_VERTEX=1: a vertex stream that is disabled or empty is resolved without
/// translating its address (the binding is null whatever it gives), and a vertex stream or index
/// buffer whose address range and the page table are those of the slot's last translation reuses
/// it (BufferCache::UpdateVertexBuffer, UpdateIndexBuffer).
inline bool g_vertex{};
/// EDEN_NX2_CFG_CBUF=1: a graphics uniform buffer bound at the address and with the page table of
/// the slot's last bind reuses that bind's translation (BufferCache::BindGraphicsUniformBuffer).
inline bool g_cbuf{};
/// EDEN_NX2_CFG_VERIFY=1: every skipped Touch checks that the item's tick already is the frame's,
/// every digest lookup also runs the base's search, every skipped Query runs, every kept texture
/// header entry is resolved as the base would (without its side effects), every texture memo hit
/// is checked against the full resolve, and every reused translation is made again; a mismatch is
/// counted, the first 16 of each window are logged, and the base's answer is used.
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
    TicKept,              ///< texture header entries read again after a view deletion and kept
    TicResolved,          ///< such entries resolved again (view deleted, bound to nothing, map)
    TicFull,              ///< deletions that invalidated a table in full (the page table changed)
    MemoKept,             ///< texture memo hits after a deletion of none of their views
    MemoDropped,          ///< texture memos dropped because one of their views was deleted
    VertexSkipped,        ///< disabled or empty vertex streams resolved without a translation
    VertexReused,         ///< vertex streams that reused the slot's last translation
    VertexTranslated,     ///< vertex streams translated
    IndexReused,          ///< index buffers that reused the last translation
    IndexTranslated,      ///< index buffers translated
    CbufReused,           ///< uniform buffer binds that reused the slot's last translation
    CbufTranslated,       ///< uniform buffer binds translated
    Verified,             ///< verify: skipped work checked
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

/// EDEN_NX2_CFG_TIC: image view deletions, counted from 1 (Clock), and per view slot the count of
/// its last deletion. An image table entry stamped with the count at its last read
/// (DescriptorTable::stamps) is bound to a view deleted since exactly when DiedAfter(its view's
/// slot, its stamp): a deletion after the stamp gives the slot a larger count, and a view the slot
/// held before was deleted before the view the entry was bound to was made, so before the read.
class ViewDeaths {
public:
    explicit ViewDeaths(u32 start = 1) : deletions{start} {}

    [[nodiscard]] u32 Clock() const noexcept {
        return deletions;
    }

    /// One deletion of the views whose slots `slots` lists. True when the count started again
    /// from 1 first: every table must then be invalidated in full, so that no entry and no memo
    /// keeps a count from before (what the base does for every deletion anyway).
    template <typename Slots>
    bool Note(const Slots& slots) {
        bool restarted = false;
        if (deletions == (std::numeric_limits<u32>::max)()) [[unlikely]] {
            std::fill(deaths.begin(), deaths.end(), u32{0});
            deletions = 1;
            restarted = true;
        }
        ++deletions;
        for (const auto slot_value : slots) {
            const std::size_t slot = static_cast<std::size_t>(slot_value);
            if (slot >= deaths.size()) {
                deaths.resize((std::max)(slot + 1, deaths.size() * 2));
            }
            deaths[slot] = deletions;
        }
        return restarted;
    }

    [[nodiscard]] bool DiedAfter(u32 slot, u32 stamp) const noexcept {
        return slot < deaths.size() && deaths[slot] > stamp;
    }

private:
    u32 deletions;
    std::vector<u32> deaths;
};

/// EDEN_NX2_CFG_VERTEX, _CBUF: the last translation of a binding slot's GPU address. A
/// translation reads the address space's page table alone, which changes only in
/// MemoryManager::PageTableOp and BigPageTableOp; each bumps MapSerial once it is done. So the
/// translation of the same address in the same address space is the same while MapSerial is the
/// one read before it was made. (An operation still running when MapSerial is read is a race the
/// base has too: its translation may come before or after the change.)
struct TranslationMemo {
    const void* memory{}; ///< the address space (Tegra::MemoryManager) translated in
    u64 map_serial{};     ///< its MapSerial, read before the translation
    u64 gpu_begin{};      ///< the address translated
    u64 gpu_end{};        ///< what else the result depends on (a vertex stream's end), else 0
    u64 device_addr{};
    u32 size{}; ///< a vertex stream's size after the continuous range clamp, else 0
    bool mapped{};
    bool valid{};
};

/// A translation result: the device address, if mapped, and the size that goes with it.
using Translation = std::pair<std::optional<u64>, u32>;

/// EDEN_NX2_CFG_VERTEX, _CBUF: the slot's last translation when it holds (same address space,
/// MapSerial, address and end), else `translate()`, the base's computation, noted for next time.
/// `map_serial` is to be read before the call. Verify makes the translation again and uses it.
template <typename Translate>
[[nodiscard]] Translation Translated(TranslationMemo& memo, const void* memory, u64 map_serial,
                                     u64 gpu_begin, u64 gpu_end, Counter reused, Counter translated,
                                     Translate&& translate) {
    if (memo.valid && memo.memory == memory && memo.map_serial == map_serial &&
        memo.gpu_begin == gpu_begin && memo.gpu_end == gpu_end) {
        const Translation kept{memo.mapped ? std::optional<u64>{memo.device_addr} : std::nullopt,
                               memo.size};
        if (!g_verify) [[likely]] {
            Count(reused);
            return kept;
        }
        Count(Counter::Verified);
        const Translation expected = translate();
        if (expected == kept) {
            Count(reused);
            return kept;
        }
        Mismatch("a reused address translation differs from the page table's", gpu_begin,
                 expected.first.value_or(0));
        memo.device_addr = expected.first.value_or(0);
        memo.size = expected.second;
        memo.mapped = expected.first.has_value();
        return expected;
    }
    const Translation result = translate();
    memo = TranslationMemo{
        .memory = memory,
        .map_serial = map_serial,
        .gpu_begin = gpu_begin,
        .gpu_end = gpu_end,
        .device_addr = result.first.value_or(0),
        .size = result.second,
        .mapped = result.first.has_value(),
        .valid = true,
    };
    Count(translated);
    return result;
}

[[nodiscard]] inline bool ReadFlag(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

/// Called by VideoCore::CreateGPU, before the caches exist.
inline void ReadSwitches() {
    g_touch = Nx2Tested::On("EDEN_NX2_CFG_TOUCH");
    g_lookup = Nx2Tested::On("EDEN_NX2_CFG_LOOKUP");
    g_ssbo = Nx2Tested::On("EDEN_NX2_CFG_SSBO");
    g_bank_stats = ReadFlag("EDEN_NX2_CFG_BANK_STATS");
    g_tic = Nx2Tested::On("EDEN_NX2_CFG_TIC");
    g_vertex = Nx2Tested::On("EDEN_NX2_CFG_VERTEX");
    g_cbuf = ReadFlag("EDEN_NX2_CFG_CBUF");
    g_verify = ReadFlag("EDEN_NX2_CFG_VERIFY");
    g_any =
        g_touch || g_lookup || g_ssbo || g_bank_stats || g_tic || g_vertex || g_cbuf || g_verify;
    if (g_any) {
        LOG_INFO(HW_GPU,
                 "Track C draw preparation switches: touch={} lookup={} ssbo={} bank_stats={} "
                 "tic={} vertex={} cbuf={} verify={}",
                 g_touch, g_lookup, g_ssbo, g_bank_stats, g_tic, g_vertex, g_cbuf, g_verify);
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
    const u64 tic_kept = Take(Counter::TicKept);
    const u64 tic_resolved = Take(Counter::TicResolved);
    const u64 tic_full = Take(Counter::TicFull);
    const u64 memo_kept = Take(Counter::MemoKept);
    const u64 memo_dropped = Take(Counter::MemoDropped);
    const u64 vertex_skipped = Take(Counter::VertexSkipped);
    const u64 vertex_reused = Take(Counter::VertexReused);
    const u64 vertex_translated = Take(Counter::VertexTranslated);
    const u64 index_reused = Take(Counter::IndexReused);
    const u64 index_translated = Take(Counter::IndexTranslated);
    const u64 cbuf_reused = Take(Counter::CbufReused);
    const u64 cbuf_translated = Take(Counter::CbufTranslated);
    const u64 verified = Take(Counter::Verified);
    const u64 mismatches = Take(Counter::Mismatches);
    LOG_INFO(HW_GPU,
             "Track C {:.1f} s: buffer LRU touches {} skipped, {} made; image LRU touches {} "
             "skipped, {} made; pipeline transition searches {} over {} entries, {} whole key "
             "compares, {} found; storage range queries {} skipped, {} made; bank re-uploads {} "
             "({} KiB), {} unchanged; texture headers after deletions {} kept, {} resolved again, "
             "{} tables invalidated in full; texture memos {} kept over deletions, {} dropped; "
             "vertex streams {} without translation, {} reused, {} translated; index buffers {} "
             "reused, {} translated; uniform binds {} reused, {} translated; verify checked {}, "
             "mismatches {} ({} since start) | touch={} lookup={} ssbo={} bank_stats={} tic={} "
             "vertex={} cbuf={} verify={}",
             seconds, buffer_skipped, buffer_touched, image_skipped, image_touched, searches,
             entries, compares, hits, range_skipped, range_queries, bank_uploads, bank_bytes >> 10,
             bank_unchanged, tic_kept, tic_resolved, tic_full, memo_kept, memo_dropped,
             vertex_skipped, vertex_reused, vertex_translated, index_reused, index_translated,
             cbuf_reused, cbuf_translated, verified, mismatches,
             g_mismatches_total.load(std::memory_order_relaxed), g_touch, g_lookup, g_ssbo,
             g_bank_stats, g_tic, g_vertex, g_cbuf, g_verify);
}

} // namespace VideoCommon::Cfg
