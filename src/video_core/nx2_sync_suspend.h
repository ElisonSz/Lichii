// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "video_core/nx2_tested.h"

// EDEN_NX2_SYNC_SUSPEND=1: a Switch 2 semaphore acquire that the split's semaphore view could
// decide only once the back has run the entry of a fence-flagged release to its word (the
// positioned wait of Puller::AcquireWithoutDrain) suspends its channel at once, as the baseline
// does after that wait whenever the release has not landed yet, instead of holding the whole
// front until the back reaches the release. The front meanwhile sends the other work queued
// for it, so the back has entries past the release when it gets there.
//
// Only where the baseline's own outcome after the wait can be that suspension: the back has run
// every write sent before the release that could still change the word as it runs (compute and
// copy work, writes the front could not place, writes landing at run;
// GpuSplit::QueuedSemaphoreWrite::settle), and neither a latch nor the word in memory, as the
// view reads it while the release is pending, meets the acquire. The channel then resumes on
// the same latch, view and poll as any suspended one; before the back runs the release, its
// 1 ms drain valve does not run (the baseline was still waiting), and a front with nothing
// else to do waits for that entry as before. With the switch off nothing here runs.
namespace VideoCommon::SyncSuspend {

// Loaded by CreateGPU before any GPU thread starts; never changed during a session.
inline bool enabled{};
/// EDEN_NX2_SYNC_SUSPEND=2, a measurement: compute and copy work queued before the release does
/// not hold the suspension back. Such work could release the word unseen (a QMD release the
/// front could not value) so that the baseline passes after its wait where this suspends: the
/// channel still resumes only once the view or a latch lets it, but after work sent later.
/// Counted as `deviations` where it shows.
inline bool relaxed{};

inline void ReadSwitch() {
    // Unset, =1 in the tested title (Nx2Tested); =2 stays a measurement.
    const char* const value = std::getenv("EDEN_NX2_SYNC_SUSPEND");
    const bool one =
        value != nullptr ? value[0] == '1' && value[1] == '\0' : Nx2Tested::TestedTitle();
    const bool two = value != nullptr && value[0] == '2' && value[1] == '\0';
    enabled = one || two;
    relaxed = two;
}

/// Commands the CPU queued for the GPU thread that its front has not taken yet (counted only
/// while enabled, from the GPU thread's start: incremented before the push, decremented after
/// the pop). Signed, so a miscount can only err towards "nothing queued".
inline std::atomic<s64> queued_commands{};
/// The front: suspended channels other than the one it is dispatching.
inline u64 other_pending{};

/// Whether the front has other work than the channel it dispatches.
[[nodiscard]] inline bool WorkQueued() {
    return queued_commands.load(std::memory_order_relaxed) > 0 || other_pending != 0;
}

/// What an acquire waiting for a fence-flagged release's entry does.
enum class Decision : u8 {
    Suspend, ///< Suspend now: the baseline does so after the wait unless the release lands first
    NoWork,  ///< Nothing else for the front to send: wait for the entry as the baseline does
    Settle,  ///< Work sent before the release may still change the word: wait for it first
    Met,     ///< A latch or the word in memory meets it: the baseline passes after the wait
};

/// `settled()`: the back ran the work before the release that could change the word as it runs.
/// `met()`: the waiter is latched, or the word as the view reads it with the release pending
/// meets the acquire. Each is asked only when the ones before allow a suspension.
template <typename Settled, typename Met>
[[nodiscard]] Decision Decide(bool work_queued, Settled&& settled, Met&& met) {
    if (!work_queued) {
        return Decision::NoWork;
    }
    if (!settled()) {
        return Decision::Settle;
    }
    if (met()) {
        return Decision::Met;
    }
    return Decision::Suspend;
}

/// How long the front waits at once for an early-suspended release's entry when it has nothing
/// else to do; then it polls and looks for new commands, and waits again.
inline constexpr std::chrono::milliseconds IdleWaitLimit{2};

using Clock = std::chrono::steady_clock;

[[nodiscard]] inline u64 Nanoseconds(Clock::duration duration) {
    const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return count > 0 ? static_cast<u64>(count) : 0;
}

/// The front's counts, logged every 10 s and cleared (Nx2AcquireDetails::ReportSyncSuspend).
struct Counters {
    u64 acquires{};   ///< Acquires the semaphore view decided
    u64 fenced{};     ///< Of those: first answer, a wait for a fence-flagged release's entry
    u64 early{};      ///< Of those: suspended early
    u64 no_work{};    ///< ...kept waiting: nothing else for the front to send
    u64 met{};        ///< ...kept waiting: a latch or memory meets it (the baseline passes)
    u64 changed{};    ///< ...kept waiting: the view's answer changed during the settle wait
    u64 settle_waits{};
    u64 settle_ns{};
    u64 waits{};      ///< Positioned waits left on the baseline path (every kind)
    u64 wait_ns{};
    u64 reached{};    ///< Early suspensions whose release entry the front saw the back run
    u64 reached_ns{}; ///< Their time from the suspension to then: front waiting avoided
    u64 reached_words{}; ///< Ring words the front sent meanwhile
    u64 unreached_resumes{}; ///< Early suspensions resumed before that (latch, other write)
    /// Early suspensions a poll passed on the word in memory while their release was still
    /// pending: where the baseline would have passed right after its wait (a CPU write, or work
    /// the settle wait did not cover: EDEN_NX2_SYNC_SUSPEND=2).
    u64 deviations{};
    u64 idle_waits{}; ///< The front, with nothing else to do, waits for such an entry
    u64 idle_wait_ns{};
    u64 idle_timeouts{};
};
inline Counters counters;

/// Back idle spans (from finding the ring empty to having work again), while enabled.
struct BackCounters {
    std::atomic<u64> idle_spans{};
    std::atomic<u64> idle_ns{};
};
inline BackCounters back;

inline void NoteWait(Clock::time_point begin) {
    ++counters.waits;
    counters.wait_ns += Nanoseconds(Clock::now() - begin);
}

/// An early suspension's release entry ran: suspended at `since` with the front at ring position
/// `sent`, now at `position` (a restarted split starts at 0 again: then no words count).
inline void NoteReached(Clock::time_point since, u64 sent, u64 position) {
    ++counters.reached;
    counters.reached_ns += Nanoseconds(Clock::now() - since);
    counters.reached_words += position > sent ? position - sent : 0;
}

} // namespace VideoCommon::SyncSuspend
