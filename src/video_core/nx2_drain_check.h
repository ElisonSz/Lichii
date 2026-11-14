// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"

// EDEN_NX2_DRAIN_CHECK compares command entries that skipped a wait at their place on the back,
// and verifies the QMD view's releases, descriptor reads, late reads and scheduler tables.
// EDEN_NX2_ENTRY_CHECK also compares submitted and live command words with guest memory.
// Acquire decisions retain this branch's EDEN_NX2_ACQUIRE_VERIFY checks in Puller.
namespace VideoCommon::Nx2DrainCheck {

// A plain flag: the front checks it at every command list and acquire.
inline bool g_enabled = std::getenv("EDEN_NX2_DRAIN_CHECK") != nullptr;
inline bool g_entries = std::getenv("EDEN_NX2_ENTRY_CHECK") != nullptr;
inline bool g_any = g_enabled || g_entries;

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_enabled = std::getenv("EDEN_NX2_DRAIN_CHECK") != nullptr;
    g_entries = std::getenv("EDEN_NX2_ENTRY_CHECK") != nullptr;
    g_any = g_enabled || g_entries;
}

enum Path : u32 { SubmitCopy, Memory, GpuCopy };
constexpr size_t ComparedPaths = 2;

struct State {
    // The front's.
    std::atomic<u64> skipped{}; ///< Entries that did not wait for queued writers
    std::array<std::atomic<u64>, ComparedPaths> compared{};
    // The back's.
    std::atomic<u64> skipped_checked{};
    std::atomic<u64> skipped_written{}; ///< On pages GPU work had made GPU-written at their place
    std::atomic<u64> skipped_differ{};  ///< Of those, entries with words differing after a flush
    std::atomic<u64> skipped_words{};   ///< Their differing words
    std::array<std::atomic<u64>, ComparedPaths> differ{};
    std::array<std::atomic<u64>, ComparedPaths> differ_words{};
    // Each check logs its first differing words: the drain check's must not be crowded out by
    // the entry check's, which the CPU rewriting a command ring can make many.
    std::atomic<u32> skipped_logged{};
    std::atomic<u32> compared_logged{};

    std::chrono::steady_clock::time_point last_entry_report{};
    /// QMD methods the front valued and the releases compared for them, those that differed,
    /// and methods it could not value (their releases are not compared).
    std::atomic<u64> qmd_methods{};
    std::atomic<u64> qmd_releases{};
    std::atomic<u64> qmd_wrong{};
    std::atomic<u64> qmd_unvalued{};
    std::atomic<u32> qmd_logged{};
    /// Descriptors compared for the valued methods, and those the back read otherwise; QMD tables
    /// compared, and those that differed.
    std::atomic<u64> qmd_reads{};
    std::atomic<u64> qmd_reads_wrong{};
    std::atomic<u32> qmd_reads_logged{};
    std::atomic<u64> qmd_tables{};
    std::atomic<u64> qmd_tables_wrong{};
    /// Of the tables, those compared right after the front valued signals late; signals valued
    /// late whose descriptors were compared, those the back read otherwise, and those whose back
    /// reads were gone (more than LateReadsKept waited); QMD methods the front valued after the
    /// back could have run them (their entry published).
    std::atomic<u64> qmd_tables_late{};
    std::atomic<u64> qmd_late_signals{};
    std::atomic<u64> qmd_late_wrong{};
    std::atomic<u64> qmd_late_missing{};
    std::atomic<u64> qmd_published_early{};
    /// EDEN_NX2_RUNAHEAD_LANDINGS: releases the back made fence operations, compared once landed
    /// (LandingRelease), and of those, landed otherwise than noted.
    std::atomic<u64> qmd_landed{};
    std::atomic<u64> qmd_landed_wrong{};
    std::chrono::steady_clock::time_point last_report{}; ///< The front's
};

inline State& Get() {
    static State state;
    return state;
}

constexpr u32 MaxLoggedWords = 16;

/// The back found `now` in guest memory at `word_address` of the entry at `entry_address`, where
/// the front parsed `parsed`. `skipped`: the drain check's comparison, else the entry check's.
inline void LogWord(bool skipped, Path path, u64 entry_address, u64 word_address, u32 parsed,
                    u32 now) {
    State& state = Get();
    std::atomic<u32>& logged = skipped ? state.skipped_logged : state.compared_logged;
    if (logged.fetch_add(1, std::memory_order_relaxed) >= MaxLoggedWords) {
        return;
    }
    if (skipped) {
        LOG_WARNING(HW_GPU,
                    "Drain check: an entry that did not wait for queued writers differs from "
                    "GPU-written memory at {:#x} (entry {:#x}): parsed {:08x}, guest memory "
                    "{:08x} after a flush at its place",
                    word_address, entry_address, parsed, now);
        return;
    }
    LOG_WARNING(HW_GPU,
                "Entry check: an entry parsed from {} differs from guest memory at {:#x} (entry "
                "{:#x}): parsed {:08x}, guest memory {:08x} at its place",
                path == SubmitCopy ? "its submit-time copy" : "guest memory", word_address,
                entry_address, parsed, now);
}

/// front could not tell (`known` false), or that value at least (`at_least`).
struct LandingRelease;
struct QmdRelease {
    u64 device;
    u32 value;
    bool known;
    bool at_least;
    /// The back's record of a release it made a fence operation (EDEN_NX2_RUNAHEAD_LANDINGS): its
    /// value is compared once it landed (`value` is not the value then).
    std::shared_ptr<LandingRelease> landing;

    bool operator==(const QmdRelease&) const = default;
};

/// A QMD release the back made a fence operation (EDEN_NX2_RUNAHEAD_LANDINGS) lands on the fencing
/// thread, with the value the word holds then: known once it landed (QmdReleaseLanded), and the
/// front's note for it once the back compared the method's releases (CheckQmdReleases). Whichever
/// comes second compares them.
struct LandingRelease {
    std::mutex mutex;
    std::optional<u32> landed;
    std::optional<QmdRelease> noted;
};

/// A QMD the scheduling read (NativeQmdScheduler::Signal's read): its GPU address and words.
struct QmdRead {
    u64 address;
    std::array<u32, 64> words;
};

/// The releases the back ran for QMD launches since the last check (the back's thread only).
inline std::vector<QmdRelease>& BackQmdReleases() {
    static std::vector<QmdRelease> released;
    return released;
}

/// The QMDs the back's scheduling read since the last check (the back's thread only).
inline std::vector<QmdRead>& BackQmdReads() {
    static std::vector<QmdRead> reads;
    return reads;
}

/// Where one signal's reads end in BackQmdReads (no QMD lives at that address).
inline constexpr u64 SignalEnd = ~u64{0};

/// The descriptors the back read for the signals that waited on the front, by the position after
/// their entry and their place in it, until the front valued them late (the back's thread only).
inline std::map<std::pair<u64, u32>, std::vector<QmdRead>>& LateBackReads() {
    static std::map<std::pair<u64, u32>, std::vector<QmdRead>> reads;
    return reads;
}
inline constexpr size_t LateReadsKept = 16384;

/// A signal the front valued late (it waited), at the entry ending at `position`, `index`-th of
/// the entry's signals, and the descriptors it read for it.
struct LateQmdRead {
    u64 position;
    u32 index;
    std::vector<QmdRead> reads;
};

/// The split's back's scheduling ran one signal (KeplerCompute::ProcessNativeMethod).
inline void NoteQmdSignalEnd(bool on_back) {
    if (on_back) {
        BackQmdReads().push_back({SignalEnd, {}});
    }
}

/// The front valued a QMD method after its entry was published: the back could have run it.
inline void NotePublishedEarly() {
    State& state = Get();
    state.qmd_published_early.fetch_add(1, std::memory_order_relaxed);
    if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU, "QMD view check: the front valued a QMD method the back could run");
    }
}

/// The split's back runs a release of a QMD launch (KeplerCompute::ProcessNativeMethod).
inline void NoteQmdRelease(u64 device_address, u32 value, bool on_back) {
    if (on_back) {
        BackQmdReleases().push_back({device_address, value, true, false, nullptr});
    }
}

/// The split's back made a release of a QMD launch a fence operation (EDEN_NX2_RUNAHEAD_LANDINGS):
/// recorded in its place, compared once it landed. The fence operation reports the value
/// (QmdReleaseLanded).
[[nodiscard]] inline std::shared_ptr<LandingRelease> NoteQmdReleaseLanding(u64 device_address,
                                                                           bool on_back) {
    if (!on_back) {
        return nullptr;
    }
    auto landing = std::make_shared<LandingRelease>();
    BackQmdReleases().push_back({device_address, 0, true, false, landing});
    return landing;
}

/// With `landing.mutex` held, both known.
inline void CompareLanding(const LandingRelease& landing, u64 device) {
    State& state = Get();
    state.qmd_landed.fetch_add(1, std::memory_order_relaxed);
    const QmdRelease& note = *landing.noted;
    const u32 value = *landing.landed;
    if (!note.known || (note.at_least ? value >= note.value : value == note.value)) {
        return;
    }
    state.qmd_landed_wrong.fetch_add(1, std::memory_order_relaxed);
    if (state.qmd_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU,
                  "QMD view check: a release the back made a fence operation landed {:#x}={:#x}, "
                  "noted {}{:#x}",
                  device, value, note.at_least ? ">=" : "=", note.value);
    }
}

/// The fencing thread: the release landed with `value` in its word.
inline void QmdReleaseLanded(LandingRelease& landing, u64 device, u32 value) {
    std::scoped_lock lock{landing.mutex};
    landing.landed = value;
    if (landing.noted) {
        CompareLanding(landing, device);
    }
}

/// The back: the front's note for the release.
inline void ExpectQmdRelease(LandingRelease& landing, const QmdRelease& note) {
    std::scoped_lock lock{landing.mutex};
    landing.noted = QmdRelease{note.device, note.value, note.known, note.at_least, nullptr};
    if (landing.landed) {
        CompareLanding(landing, note.device);
    }
}

/// The split's back's scheduling read a QMD (KeplerCompute::ProcessNativeMethod).
inline void NoteQmdRead(u64 address, const std::array<u32, 64>& words, bool on_back) {
    if (on_back) {
        BackQmdReads().push_back({address, words});
    }
}

/// The back compared the front's QMD table with its own at the same place (`same`); `late`: right
/// after the front valued signals late.
inline void NoteQmdTables(bool same, size_t states, bool late = false) {
    State& state = Get();
    state.qmd_tables.fetch_add(1, std::memory_order_relaxed);
    if (late) {
        state.qmd_tables_late.fetch_add(1, std::memory_order_relaxed);
    }
    if (!same) {
        state.qmd_tables_wrong.fetch_add(1, std::memory_order_relaxed);
        if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(HW_GPU,
                      "QMD view check: the front's table ({} QMDs) differs from the back's{}",
                      states, late ? " after it valued signals late" : "");
        }
    }
}

/// The back, after the signals the front valued late: the descriptors it read for them against
/// the ones it read itself running them.
inline void CheckLateQmdReads(std::span<const LateQmdRead> late) {
    State& state = Get();
    auto& kept = LateBackReads();
    for (const LateQmdRead& signal : late) {
        const auto it = kept.find({signal.position, signal.index});
        if (it == kept.end()) {
            state.qmd_late_missing.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        state.qmd_late_signals.fetch_add(1, std::memory_order_relaxed);
        const auto& back_reads = it->second;
        bool same = back_reads.size() == signal.reads.size();
        for (size_t i = 0; same && i < back_reads.size(); ++i) {
            same = back_reads[i].address == signal.reads[i].address &&
                   back_reads[i].words == signal.reads[i].words;
        }
        if (!same) {
            state.qmd_late_wrong.fetch_add(1, std::memory_order_relaxed);
            if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                std::string text;
                for (size_t i = 0; i < (std::max)(back_reads.size(), signal.reads.size()); ++i) {
                    const QmdRead* back = i < back_reads.size() ? &back_reads[i] : nullptr;
                    const QmdRead* front = i < signal.reads.size() ? &signal.reads[i] : nullptr;
                    text += fmt::format(" | {:#x} by the back, {:#x} by the front",
                                        back ? back->address : 0, front ? front->address : 0);
                    for (size_t word = 0; back && front && word < 64; ++word) {
                        if (back->words[word] != front->words[word]) {
                            text += fmt::format(", word {} {:#x} for {:#x}", word,
                                                back->words[word], front->words[word]);
                            break;
                        }
                    }
                }
                LOG_ERROR(HW_GPU,
                          "QMD view check: the back read other descriptors for a signal the "
                          "front valued late (position {:#x}, signal {}){}",
                          signal.position, signal.index, text);
            }
        }
        kept.erase(it);
    }
}

/// The back, right after QMD methods the front sent: what it read and released for them against
/// what the front read and noted (`front_reads`, `predicted`, in order), or nothing to compare
/// where the front could not value them (`valued` false).
/// `wait_position`: the signals of the entry ending there waited on the front from the
/// `first_waiting`-th on (0: none): their reads are kept for CheckLateQmdReads.
inline void CheckQmdReleases(std::span<const QmdRelease> predicted,
                             std::span<const QmdRead> front_reads, bool valued,
                             u64 wait_position = 0, u32 first_waiting = 0) {
    State& state = Get();
    auto& released = BackQmdReleases();
    auto& all_reads = BackQmdReads();
    // By signal: each ends with a SignalEnd.
    std::vector<QmdRead> back_reads;
    std::vector<std::vector<QmdRead>> signals(1);
    size_t ends = 0;
    for (QmdRead& read : all_reads) {
        if (read.address == SignalEnd) {
            signals.emplace_back();
            ++ends;
            continue;
        }
        back_reads.push_back(read);
        signals.back().push_back(std::move(read));
    }
    signals.resize(ends);
    all_reads.clear();
    if (!valued) {
        state.qmd_unvalued.fetch_add(1, std::memory_order_relaxed);
        released.clear();
        auto& kept = LateBackReads();
        if (wait_position != 0) {
            // Kept from before the split last started over at position 0.
            kept.erase(kept.lower_bound({wait_position, 0}), kept.end());
        }
        for (u32 index = first_waiting; wait_position != 0 && index < signals.size(); ++index) {
            if (kept.size() >= LateReadsKept) {
                kept.erase(kept.begin()); // The oldest: the front lost or dropped it
            }
            kept.insert_or_assign({wait_position, index}, std::move(signals[index]));
        }
        return;
    }
    state.qmd_reads.fetch_add(back_reads.size(), std::memory_order_relaxed);
    bool same_reads = back_reads.size() == front_reads.size();
    for (size_t i = 0; same_reads && i < back_reads.size(); ++i) {
        same_reads = back_reads[i].address == front_reads[i].address &&
                     back_reads[i].words == front_reads[i].words;
    }
    if (!same_reads) {
        state.qmd_reads_wrong.fetch_add(1, std::memory_order_relaxed);
        if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::string text;
            for (size_t i = 0; i < (std::max)(back_reads.size(), front_reads.size()); ++i) {
                const QmdRead* back = i < back_reads.size() ? &back_reads[i] : nullptr;
                const QmdRead* front = i < front_reads.size() ? &front_reads[i] : nullptr;
                text += fmt::format(" | {:#x} by the back, {:#x} by the front",
                                    back ? back->address : 0, front ? front->address : 0);
                for (size_t word = 0; back && front && word < 64; ++word) {
                    if (back->words[word] != front->words[word]) {
                        text += fmt::format(", word {} {:#x} for {:#x}", word, back->words[word],
                                            front->words[word]);
                        break;
                    }
                }
            }
            LOG_ERROR(HW_GPU, "QMD view check: the back read other descriptors{}", text);
        }
    }
    state.qmd_methods.fetch_add(1, std::memory_order_relaxed);
    state.qmd_releases.fetch_add(released.size(), std::memory_order_relaxed);
    bool same = released.size() == predicted.size();
    for (size_t i = 0; same && i < released.size(); ++i) {
        const QmdRelease& note = predicted[i];
        // One made a fence operation is compared once it landed.
        same =
            released[i].device == note.device &&
            (released[i].landing || !note.known ||
             (note.at_least ? released[i].value >= note.value : released[i].value == note.value));
    }
    if (same) {
        for (size_t i = 0; i < released.size(); ++i) {
            if (released[i].landing) {
                ExpectQmdRelease(*released[i].landing, predicted[i]);
            }
        }
    }
    if (!same) {
        state.qmd_wrong.fetch_add(1, std::memory_order_relaxed);
        if (state.qmd_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::string text;
            for (const QmdRelease& release : released) {
                text += release.landing
                            ? fmt::format(" {:#x}=(as it lands)", release.device)
                            : fmt::format(" {:#x}={:#x}", release.device, release.value);
            }
            text += " | noted";
            for (const QmdRelease& note : predicted) {
                text += fmt::format(" {:#x}{}{:#x}", note.device,
                                    !note.known     ? "=?"
                                    : note.at_least ? ">="
                                                    : "=",
                                    note.value);
            }
            LOG_ERROR(HW_GPU, "QMD view check: the back released{}", text);
        }
    }
    released.clear();
}

/// The front, between commands.
inline void MaybeReportEntries() {
    if (!g_any) {
        return;
    }
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    if (state.last_entry_report == std::chrono::steady_clock::time_point{}) {
        state.last_entry_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_entry_report).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_entry_report = now;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    if (g_enabled) {
        const u64 skipped = take(state.skipped);
        const u64 checked = take(state.skipped_checked);
        const u64 written = take(state.skipped_written);
        const u64 differ = take(state.skipped_differ);
        const u64 words = take(state.skipped_words);
        LOG_INFO(HW_GPU,
                 "Drain checks in {:.1f} s: entries that did not wait for queued writers {} ({} "
                 "checked at their place, on pages GPU work had made GPU-written {}, differing "
                 "from guest memory after a flush {} in {} words)",
                 seconds, skipped, checked, written, differ, words);
    }
    if (g_entries) {
        const u64 copies = take(state.compared[SubmitCopy]);
        const u64 copies_differ = take(state.differ[SubmitCopy]);
        const u64 copies_words = take(state.differ_words[SubmitCopy]);
        const u64 reads = take(state.compared[Memory]);
        const u64 reads_differ = take(state.differ[Memory]);
        const u64 reads_words = take(state.differ_words[Memory]);
        LOG_INFO(HW_GPU,
                 "Entry checks in {:.1f} s: entries compared with guest memory at their place, run "
                 "from their submit-time copy {} ({} differ in {} words), from guest memory {} ({} "
                 "differ in {} words)",
                 seconds, copies, copies_differ, copies_words, reads, reads_differ, reads_words);
    }
}

inline void MaybeReport() {
    MaybeReportEntries();
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    if (state.last_report == std::chrono::steady_clock::time_point{}) {
        state.last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_report).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_report = now;
    const auto take = [](std::atomic<u64>& value) {
        return value.exchange(0, std::memory_order_relaxed);
    };
    const u64 qmd_methods = take(state.qmd_methods);
    const u64 qmd_releases = take(state.qmd_releases);
    const u64 qmd_wrong = take(state.qmd_wrong);
    const u64 qmd_unvalued = take(state.qmd_unvalued);
    const u64 qmd_reads = take(state.qmd_reads);
    const u64 qmd_reads_wrong = take(state.qmd_reads_wrong);
    const u64 qmd_tables = take(state.qmd_tables);
    const u64 qmd_tables_wrong = take(state.qmd_tables_wrong);
    const u64 qmd_tables_late = take(state.qmd_tables_late);
    const u64 qmd_late_signals = take(state.qmd_late_signals);
    const u64 qmd_late_wrong = take(state.qmd_late_wrong);
    const u64 qmd_late_missing = take(state.qmd_late_missing);
    const u64 qmd_published_early = take(state.qmd_published_early);
    const u64 qmd_landed = take(state.qmd_landed);
    const u64 qmd_landed_wrong = take(state.qmd_landed_wrong);
    LOG_INFO(HW_GPU,
             "QMD drain checks in {:.1f} s: methods valued {} with {} releases, released otherwise "
             "than noted {}, with {} descriptors read, read otherwise {}; valued after the back "
             "could run them {}; signals valued late {}, read otherwise {}, not compared {}; "
             "tables compared {} (after valuing late {}), differing {}; not valued {}; releases "
             "made fence operations compared as they landed {}, landed otherwise than noted {}",
             seconds, qmd_methods, qmd_releases, qmd_wrong, qmd_reads, qmd_reads_wrong,
             qmd_published_early, qmd_late_signals, qmd_late_wrong, qmd_late_missing, qmd_tables,
             qmd_tables_late, qmd_tables_wrong, qmd_unvalued, qmd_landed, qmd_landed_wrong);
}

} // namespace VideoCommon::Nx2DrainCheck

// QMD counters for EDEN_NX2_GPU_PROFILE and EDEN_NX2_ACQUIRE_VERIFY. Front thread only.
namespace VideoCommon::Nx2AcquireStats {

// A plain flag: the front checks it at every acquire and release it sends.
inline bool g_enabled = Nx2Profile::Enabled() || std::getenv("EDEN_NX2_ACQUIRE_VERIFY") != nullptr;

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_enabled = Nx2Profile::Enabled() || std::getenv("EDEN_NX2_ACQUIRE_VERIFY") != nullptr;
}

/// Who the front sent releases to a word for (State::noted bits).
enum class Releaser : u32 {
    Puller,        ///< Semaphore methods (WriteLong, SEM_EXECUTE release)
    Report3D,      ///< 3D report semaphores
    ReportCompute, ///< Compute report semaphores
    Copy,          ///< Copy engine semaphores
    Qmd,           ///< QMD launches the front valued (FrontQmdView)
    Count,
};

/// Why the front's QMD view could not value a signal (FrontQmdView::Signal), or one that waited.
enum class QmdLost : u32 {
    NotSynced,        ///< Its table is not the back's yet (the back has not caught up since)
    Unplaced,         ///< Work that may write guest memory anywhere was still queued
    GpuWritten,       ///< A QMD it read is on a page GPU work wrote
    Pending,          ///< A write into a QMD it read was still queued
    ReleasedInto,     ///< A release of the same signal writes into a QMD it read
    Mappings,         ///< Work that may change the GPU's mappings was still queued
    WrittenAfter,     ///< Waiting, a QMD it reads was written after it
    MappedAfter,      ///< Waiting, the mappings may have changed after it
    LateReleasedInto, ///< Waiting, a release of it writes a QMD read late
    TooMany,          ///< Too many signals waited
    Count,
};

/// Where the front's QMD view took the back's table over (FrontQmdView::TakeOver).
enum class QmdTakeOver : u32 {
    FrontStart,      ///< An engine's front started, with the back drained
    BetweenCommands, ///< The back had run everything sent (GPU thread loop)
    QmdMethod,       ///< The same, at a QMD method
    Count,
};

/// What the QMD view noted for a release.
enum class QmdNote : u32 {
    Exact,
    AtLeast, ///< A maximum with a word it can't tell
    Unknown, ///< A reduction of a word it can't tell
    Count,
};

struct State {
    u64 passed_by_qmd{}; ///< Of the passes by a queued release, at words QMD launches release
    u64 qmd_valued{};
    std::array<u64, static_cast<size_t>(QmdLost::Count)> qmd_lost{};
    /// Of qmd_lost[Unplaced] and qmd_lost[Mappings]: by what sent the last such write
    std::array<u64, static_cast<size_t>(UnplacedKind::Count)> qmd_lost_unplaced{};
    std::array<u64, static_cast<size_t>(UnplacedKind::Count)> qmd_lost_mappings{};
    std::array<u64, static_cast<size_t>(QmdNote::Count)> qmd_notes{};
    u64 qmd_inline_reads{}; ///< Descriptors read from an inline QMD still queued
    /// Signals that waited first, by why, and behind one; valued late, and how often the last
    /// waiting one was; waits that lost the table, by why, and the waiting signals they dropped;
    /// waiting signals takeovers dropped (the back had run them).
    std::array<u64, static_cast<size_t>(QmdLost::Count)> qmd_waits{};
    u64 qmd_behind{};
    u64 qmd_late{};
    u64 qmd_caught_up{};
    std::array<u64, static_cast<size_t>(QmdLost::Count)> qmd_late_lost{};
    u64 qmd_late_dropped{};
    u64 qmd_takeover_dropped{};
    /// Of qmd_waits[Pending]: by what wrote the QMD (FrontQmdView::NoteWaitWriter)
    std::map<std::string, u64> qmd_wait_writers;
    std::array<u64, static_cast<size_t>(QmdTakeOver::Count)> qmd_syncs{};
    std::chrono::steady_clock::duration qmd_sync_time{};
    std::unordered_map<u64, u8> noted; ///< Releaser bits by word
    std::chrono::steady_clock::time_point last_report{};
};

inline State& Get() {
    static State state;
    return state;
}

/// The front sent a release to the word at `device_address`.
inline void NoteRelease(u64 device_address, Releaser releaser) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    if (state.noted.size() < 65536) {
        state.noted[device_address & ~u64{3}] = static_cast<u8>(1U << static_cast<u32>(releaser));
    }
}

/// The QMD view valued a signal (`lost` empty) or not; `kind`: what sent the last write that
/// may write anywhere (Unplaced) or change the mappings (Mappings).
inline void NoteQmdSignal(std::optional<QmdLost> lost, UnplacedKind kind = UnplacedKind::Other) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    if (!lost) {
        ++state.qmd_valued;
        return;
    }
    ++state.qmd_lost[static_cast<size_t>(*lost)];
    if (*lost == QmdLost::Unplaced) {
        ++state.qmd_lost_unplaced[static_cast<size_t>(kind)];
    } else if (*lost == QmdLost::Mappings) {
        ++state.qmd_lost_mappings[static_cast<size_t>(kind)];
    }
}

/// A signal the QMD view could not value at once waits: the first by `why` (`kind`: what sent the
/// last write that may write anywhere or change the mappings), or behind one (`why` empty).
inline void NoteQmdWait(std::optional<QmdLost> why, UnplacedKind kind = UnplacedKind::Other) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    if (!why) {
        ++state.qmd_behind;
        return;
    }
    ++state.qmd_waits[static_cast<size_t>(*why)];
    if (*why == QmdLost::Unplaced) {
        ++state.qmd_lost_unplaced[static_cast<size_t>(kind)];
    } else if (*why == QmdLost::Mappings) {
        ++state.qmd_lost_mappings[static_cast<size_t>(kind)];
    }
}

/// A signal waits for a write into a QMD by `writer`, still queued.
inline void NoteQmdWaitWriter(const std::string& writer) {
    if (g_enabled) {
        ++Get().qmd_wait_writers[writer];
    }
}

/// The QMD view valued a waiting signal late; `caught_up`: none waits now.
inline void NoteQmdLate(bool caught_up) {
    if (g_enabled) {
        State& state = Get();
        ++state.qmd_late;
        state.qmd_caught_up += caught_up ? 1 : 0;
    }
}

/// The QMD view lost its table valuing a waiting signal late (`why`), dropping `dropped` waiting
/// signals.
inline void NoteQmdLateLost(QmdLost why, size_t dropped) {
    if (g_enabled) {
        State& state = Get();
        ++state.qmd_late_lost[static_cast<size_t>(why)];
        state.qmd_late_dropped += dropped;
    }
}

/// A takeover dropped `dropped` waiting signals: the back had run them.
inline void NoteQmdTakeOverDropped(size_t dropped) {
    if (g_enabled) {
        Get().qmd_takeover_dropped += dropped;
    }
}

/// The QMD view read a descriptor from an inline QMD still queued.
inline void NoteQmdInlineRead() {
    if (g_enabled) {
        ++Get().qmd_inline_reads;
    }
}

/// The QMD view noted a release so.
inline void NoteQmdNote(QmdNote note) {
    if (g_enabled) {
        ++Get().qmd_notes[static_cast<size_t>(note)];
    }
}

/// The QMD view took the back's table over, at `where`, in `time`.
inline void NoteQmdSync(QmdTakeOver where, std::chrono::steady_clock::duration time) {
    if (g_enabled) {
        State& state = Get();
        ++state.qmd_syncs[static_cast<size_t>(where)];
        state.qmd_sync_time += time;
    }
}

inline void NoteQueuedPass(u64 device_address) {
    if (!g_enabled)
        return;
    State& state = Get();
    const auto it = state.noted.find(device_address & ~u64{3});
    if (it != state.noted.end() && ((it->second >> static_cast<u32>(Releaser::Qmd)) & 1)) {
        ++state.passed_by_qmd;
    }
}

/// The front, between commands.
inline void MaybeReport() {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    if (state.last_report == std::chrono::steady_clock::time_point{}) {
        state.last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_report).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_report = now;
    const auto& lost = state.qmd_lost;
    const auto& notes = state.qmd_notes;
    const auto& syncs = state.qmd_syncs;
    const auto kinds = [](const auto& counts) {
        std::string text;
        for (size_t kind = 0; kind < counts.size(); ++kind) {
            if (counts[kind] != 0) {
                text += fmt::format("{}{} {}", text.empty() ? " (" : ", ",
                                    unplaced_kind_names[kind], counts[kind]);
            }
        }
        return text.empty() ? text : text + ")";
    };
    // The writers a signal waited for most.
    std::vector<std::pair<u64, std::string>> writers;
    for (const auto& [writer, count] : state.qmd_wait_writers) {
        writers.emplace_back(count, writer);
    }
    std::sort(writers.rbegin(), writers.rend());
    std::string by_writer;
    for (size_t i = 0; i < writers.size() && i < 6; ++i) {
        by_writer +=
            fmt::format("{}{} {}", i == 0 ? " (by " : ", ", writers[i].second, writers[i].first);
    }
    if (!by_writer.empty()) {
        by_writer += ")";
    }
    const auto& waits = state.qmd_waits;
    const auto& late_lost = state.qmd_late_lost;
    const auto index = [](QmdLost why) { return static_cast<size_t>(why); };
    u64 waited_first = 0;
    u64 lost_late = 0;
    for (size_t why = 0; why < waits.size(); ++why) {
        waited_first += waits[why];
        lost_late += late_lost[why];
    }
    LOG_INFO(
        HW_GPU,
        "QMD view in {:.1f} s (EDEN_NX2_GPU_PROFILE / EDEN_NX2_ACQUIRE_VERIFY): signals valued {}, "
        "valued late {} "
        "(caught up {} times) | waited first {} (a write into the QMD queued {}{}, work "
        "that may write anywhere queued {}{}, a change of the mappings queued {}{}), behind an "
        "earlier one {}; lost while waiting {} (a QMD written after it {}, the mappings "
        "changed after it {}, a release into a QMD read late {}, too many waiting {}; {} "
        "waiting dropped), dropped at takeovers {} | not valued: table not synced yet {}, "
        "QMD on a GPU-written page {}, a release of the signal into it {} | descriptors read "
        "from an inline QMD still queued {} | releases noted exact {}, at least {}, unknown "
        "{} | tables taken over {} in {:.3f} ms (at a front's start {}, between commands {}, "
        "at QMD methods {}) | acquires passed by QMD releases {}",
        seconds, state.qmd_valued, state.qmd_late, state.qmd_caught_up, waited_first,
        waits[index(QmdLost::Pending)], by_writer, waits[index(QmdLost::Unplaced)],
        kinds(state.qmd_lost_unplaced), waits[index(QmdLost::Mappings)],
        kinds(state.qmd_lost_mappings), state.qmd_behind, lost_late,
        late_lost[index(QmdLost::WrittenAfter)], late_lost[index(QmdLost::MappedAfter)],
        late_lost[index(QmdLost::LateReleasedInto)], late_lost[index(QmdLost::TooMany)],
        state.qmd_late_dropped, state.qmd_takeover_dropped, lost[index(QmdLost::NotSynced)],
        lost[index(QmdLost::GpuWritten)], lost[index(QmdLost::ReleasedInto)],
        state.qmd_inline_reads, notes[0], notes[1], notes[2], syncs[0] + syncs[1] + syncs[2],
        std::chrono::duration<double, std::milli>(state.qmd_sync_time).count(), syncs[0], syncs[1],
        syncs[2], state.passed_by_qmd);
    auto noted_words = std::move(state.noted);
    state = State{};
    state.noted = std::move(noted_words);
    state.last_report = now;
}

} // namespace VideoCommon::Nx2AcquireStats
