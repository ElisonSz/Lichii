// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>

#include "common/common_types.h"

// EDEN_NX2_RUNAHEAD=1: the split GPU thread's front stays ahead of its back where the baseline
// makes it wait for writes that cannot reach the word it acquires.
//
// A Switch 2 semaphore acquire whose word in memory (or whose last release sent) already meets it,
// but that the semaphore view cannot call settled (an equal acquire, or a word the mirror diverts),
// waits in the baseline until the back has run every compute and copy entry sent before it
// (GpuSplit::LastWriterPosition: every compute macro call, upload, report, QMD method and copy
// method), then asks the view again. With the switch the acquire waits only for the entries that
// can change the word other than through the releases the view sees at it
// (GpuSplit::RunAheadReach): QMD signals whose releases the front did not note when it sent them,
// placed writes over the word's 4 bytes (copies, uploads, QMD releases placed late, report bytes
// past the words noted), writes the front could not place, and, on a page GPU work ever wrote,
// every compute and copy entry as before. Nothing else sent before the acquire can change the
// word, so the view the baseline asks after its wait answers as it answers once those ran: the
// same decision, sooner. Two more pieces keep the front ahead: a QMD signaled while the inline
// upload that writes it is still queued is read from the upload's words the front sent
// (FrontQmdView), so the front keeps valuing QMD releases; and a greater-or-equal acquire on the
// syncpoint shim that the guest syncpoint meets already passes without draining the split
// (syncpoints only grow).
//
// The writes a dispatch's shaders store go to the GPU caches: only on a page GPU work wrote may
// they reach guest memory as acquires read it (Nx2MayBeGpuWritten), and there the reach stays
// the baseline's. EDEN_NX2_RUNAHEAD_VERIFY=1 checks every acquire passed sooner on the back, at
// its place in the ring, against the word as the baseline would read it after its wait.
namespace VideoCommon::RunAhead {

// Loaded by CreateGPU before any GPU thread starts; never changed during a session.
inline bool enabled{};
/// EDEN_NX2_RUNAHEAD_VERIFY=1 (with or without EDEN_NX2_RUNAHEAD=1, which it implies).
inline bool verify{};
/// EDEN_NX2_RUNAHEAD_STATS=1, or a verifier: counts and logs every 10 s.
inline bool stats{};
/// EDEN_NX2_RUNAHEAD_VALVE=1: a suspended acquire's 1 ms valve has the back submit what its idle
/// hook defers and send the fence work, without draining the split (Puller::PollWithoutDrain).
inline bool valve{};
/// EDEN_NX2_RUNAHEAD_VALVE_VERIFY=1 (implies the valve): drains as well (the baseline's valve), and
/// counts the passes that drain found then; the valve alone finds them at a later poll (latch or
/// view), unless only a drain makes the release land, which a hang would show (ReportLongWait).
inline bool valve_verify{};
/// EDEN_NX2_RUNAHEAD_W1=1: an acquire that a fence-flagged release already sent before it will meet
/// once it lands (its waiter latches then) passes now, not once the host GPU ran the work before
/// the release (Puller::AcquireWithoutDrain); the back runs what follows once the release landed
/// (GpuSplit::HoldBackUntilLanded), so only the front's decoding goes on sooner.
inline bool w1{};
/// EDEN_NX2_RUNAHEAD_W1_VERIFY=1 (implies W1): each such pass keeps a waiter of its own that the
/// release must latch.
inline bool w1_verify{};
/// EDEN_NX2_RUNAHEAD_LANDINGS=1: W1 passes (as EDEN_NX2_RUNAHEAD_W1) without holding the back:
/// from the pass until it closes a landing window, the back makes every guest-visible landing a
/// fence operation (after the release's fence), and the front notes the releases it sends as
/// landing once a fence ran (GpuSplit::OpenLandingWindow). Takes precedence over the W1 hold.
inline bool landings{};
/// EDEN_NX2_RUNAHEAD_LANDINGS_VERIFY=1 (implies LANDINGS): each window's acquire keeps a waiter
/// its release must latch before any landing converted in the window is applied, and the front
/// counts its reads of a word whose converted landing is still pending.
inline bool landings_verify{};
/// EDEN_NX2_RUNAHEAD_QMD=1: the QMD view places the writes to the QMDs it reads by exact range from
/// the run-ahead write log, not by hashed 256-byte granule (FrontQmdView::ReadDescriptor).
inline bool qmd{};

[[nodiscard]] inline bool SwitchIs(const char* name, char value) {
    const char* const set = std::getenv(name);
    return set != nullptr && set[0] == value && set[1] == '\0';
}

inline void ReadSwitch() {
    verify = SwitchIs("EDEN_NX2_RUNAHEAD_VERIFY", '1');
    qmd = SwitchIs("EDEN_NX2_RUNAHEAD_QMD", '1');
    // The QMD view's exact targets come from the write log the reach keeps.
    enabled = verify || qmd || SwitchIs("EDEN_NX2_RUNAHEAD", '1');
    valve_verify = SwitchIs("EDEN_NX2_RUNAHEAD_VALVE_VERIFY", '1');
    valve = valve_verify || SwitchIs("EDEN_NX2_RUNAHEAD_VALVE", '1');
    w1_verify = SwitchIs("EDEN_NX2_RUNAHEAD_W1_VERIFY", '1');
    w1 = w1_verify || SwitchIs("EDEN_NX2_RUNAHEAD_W1", '1');
    landings_verify = SwitchIs("EDEN_NX2_RUNAHEAD_LANDINGS_VERIFY", '1');
    landings = landings_verify || SwitchIs("EDEN_NX2_RUNAHEAD_LANDINGS", '1');
    stats = (enabled || valve || w1 || landings) &&
            (verify || valve_verify || w1_verify || landings_verify ||
             SwitchIs("EDEN_NX2_RUNAHEAD_STATS", '1'));
}

/// What the reach of an acquire is (GpuSplit::RunAheadReach): the last kind of entry sent before
/// it that may still change its word.
enum class Reach : u8 {
    None,       ///< Nothing but the releases the view sees at the word
    Unvalued,   ///< A QMD signal whose releases the front did not note when it sent it
    Placed,     ///< A placed write over the word: a copy, an upload, a QMD release placed late
    Unplaced,   ///< A write the front could not place
    NewPage,    ///< The word's page was not watched yet: writes sent before were not logged
    Lost,       ///< The log dropped writes the back may still have to run, or no page slot left
    GpuWritten, ///< GPU work wrote the word's page: every compute and copy entry, as before
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reach::Count)> reach_names{
    "none",     "unvalued QMD", "placed write",    "unplaced write",
    "new page", "log lost",     "GPU-written page"};

struct Answer {
    u64 position; ///< The acquire waits for the back to run the entries before this, at most
    Reach reason;
};

/// EDEN_NX2_RUNAHEAD_W1: whether a fence-flagged release decides an acquire now
/// (GpuSplit::W1Release), or why not.
enum class W1 : u8 {
    Pass,       ///< The release meets it and will latch its waiter as it lands
    NotW1,      ///< The view's answer does not wait for a fence-flagged release
    NotPlain,   ///< The release is a reduction, a maximum, or a value the front can't tell
    Unmet,      ///< Its payload does not meet the acquire
    Unlatched,  ///< The acquire's waiter has no slot
    Overwritten ///< A write over the word was sent after it (it may drop the release)
};

/// The front's counts, logged every 10 s and cleared (ReportCounters; EDEN_NX2_RUNAHEAD_STATS).
struct Counters {
    u64 acquires{}; ///< Acquires the view decided with a reach
    u64 earlier{};  ///< Of those: the reach is before the writers the baseline waits for
    u64 sooner{};   ///< Passed with the baseline's writers still queued (the baseline waited)
    u64 waits{};    ///< Positioned waits for a reach (the writers-before path)
    u64 wait_ns{};
    std::array<u64, static_cast<size_t>(Reach::Count)> wait_reasons{};
    u64 scanned{};  ///< Logged writes the reaches looked at
    u64 shim_met{}; ///< Syncpoint shim acquires met without a drain
    u64 shim_drained{};
    u64 upload_qmds{}; ///< QMD descriptors read from a queued inline upload
    u64 upload_qmds_partial{}; ///< ..._QMD: of those, the upload wrote part, memory the rest
    u64 unvalued{};    ///< QMD signals whose releases the front did not note when it sent them
    u64 lead_sum{};    ///< Ring words the back had yet to run at the acquires (the front's lead)
    u64 lead_max{};
    u64 valve_drains{}; ///< Polls' 1 ms valve drains (each runs the back dry, whatever the lead)
    u64 valve_ns{};
    u64 valve_requests{};     ///< EDEN_NX2_RUNAHEAD_VALVE: valves that only asked for the submit
    u64 valve_drain_passes{}; ///< ..._VERIFY: of those, passes only the drain found
    u64 valve_wait_max_ns{};  ///< ...VALVE: the longest an acquire had waited at one of them
    u64 w1_candidates{};      ///< Acquires a fence-flagged release sent before them decides
    u64 w1_passes{};          ///< EDEN_NX2_RUNAHEAD_W1: of those, passed at once
    u64 w1_poll_passes{};     ///< Suspended acquires a fenced release sent since let go on
    u64 w1_not_plain{};       ///< Not passed: the last release is a reduction, maximum, or unknown
    u64 w1_unmet{};           ///< Not passed: its payload does not meet the acquire
    u64 w1_unlatched{};       ///< Not passed: no waiter slot (the release could not latch it)
    u64 w1_confirmed{};       ///< ..._VERIFY: passes whose own waiter a write latched
    u64 w1_unconfirmed{};     ///< ..._VERIFY: passes no write latched within 250 ms
    u64 w1_unchecked{};       ///< ..._VERIFY: passes with no check slot free
    u64 w1_overwritten{};     ///< Not passed: a write over the word was sent after the release
    u64 w1_holds{};   ///< Passes whose release had not landed: the back waits for it (W1 hold)
    u64 windows_opened{};      ///< LANDINGS: landing windows opened at a W1 pass
    u64 windows_extended{};    ///< ... passes on a later release while one was open
    u64 windows_closed{};      ///< ... closes sent once the windows' releases landed
    u64 windows_acked{};       ///< ... closes the back acknowledged (it converts no more)
    u64 notes_fenced{};        ///< ... releases noted as landing once a fence ran, not at run
    u64 checks_unregistered{}; ///< ..._VERIFY: windows without a check waiter (no slot)
    u64 qmd_exact{};  ///< EDEN_NX2_RUNAHEAD_QMD: QMD reads with the exact placement
    u64 qmd_sooner{}; ///< Of those: read now where the granules held them back
    u64 uploads{};    ///< Compute inline uploads seen (exec)
    std::array<u64, 5> upload_rejects{}; ///< Not kept: layout, size, placeholder, short, other
};
inline Counters counters;

/// The back's verify results (any thread may read them, the back writes them).
struct VerifyCounters {
    std::atomic<u64> checked{};
    std::atomic<u64> mismatches{};
    std::atomic<u64> flushes{}; ///< Of mismatches: the buffer cache holds the word GPU-written
    /// Changed while a fence-flagged release to the word was pending at the decision: a landing
    /// the baseline's timing allows as well, not a mismatch of the reach
    std::atomic<u64> raced{};
};
inline VerifyCounters verified;

/// EDEN_NX2_RUNAHEAD_W1: the back's waits in a W1 hold (GpuSplit::WaitLanded), cleared by the log.
struct HoldCounters {
    std::atomic<u64> waits{}; ///< Holds the back reached before their release landed
    std::atomic<u64> wait_ns{};
};
inline HoldCounters holds;

/// EDEN_NX2_RUNAHEAD_LANDINGS: the back's and the fencing thread's counts, cleared by the log.
struct LandingCounters {
    std::atomic<u64> reports{};    ///< Payload reports that would land at once, made fenced
    std::atomic<u64> operations{}; ///< Fence operations that would run at once, attached
    std::atomic<u64> pending_max{};
    std::atomic<u64> extended{}; ///< Entry boundaries after a close with landings still pending
    std::atomic<u64> fences{};   ///< Fences made for pending landings (close, extension)
    std::atomic<u64> ordered{};  ///< ..._VERIFY: applied after their window's release landed
    std::atomic<u64> out_of_order{};
    std::atomic<u64> unchecked{};       ///< ..._VERIFY: their window's check slot was reused
    std::atomic<u64> early_reads{};     ///< ..._VERIFY: front reads of a word with one pending
    std::atomic<u64> overwrite_waits{}; ///< Writes over reports waiting for the window to land
    std::atomic<u64> overwrite_ns{};
};
inline LandingCounters landing_counts;

/// EDEN_NX2_RUNAHEAD_LANDINGS_VERIFY: a window's check, a waiter its release must latch (the
/// waiter's owner is the slot). The front registers and retires it; anyone reads it.
struct LandingCheck {
    std::atomic<u32> generation{};
    std::atomic<bool> latched{}; ///< Seen latched by the front (set before it removed the waiter)
    bool registered{};           ///< Front only: the waiter is registered
};
inline std::array<LandingCheck, 16> landing_checks;
/// A check as an entry carries it: slot + 1 in the low byte (0: none), generation above.
[[nodiscard]] inline u64 PackLandingCheck(size_t slot, u32 generation) {
    return static_cast<u64>(slot + 1) | (u64{generation} << 8);
}

/// EDEN_NX2_RUNAHEAD_LANDINGS: set around a fence operation that is a guest-visible landing
/// (FenceManager::SignalFence converts only those): syncpoint increments, QMD reductions.
/// `device`: the word it writes, for the verify (0 if not one word).
struct LandingScope {
    explicit LandingScope(u64 device_ = 0) : previous{device}, was_active{active} {
        active = true;
        device = device_;
    }
    ~LandingScope() {
        active = was_active;
        device = previous;
    }
    LandingScope(const LandingScope&) = delete;
    LandingScope& operator=(const LandingScope&) = delete;
    static inline thread_local bool active{};
    static inline thread_local u64 device{};

private:
    u64 previous;
    bool was_active;
};

} // namespace VideoCommon::RunAhead
