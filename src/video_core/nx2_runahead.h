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
/// EDEN_NX2_RUNAHEAD_STATS=1, or the verifier: counts and logs every 10 s.
inline bool stats{};

[[nodiscard]] inline bool SwitchIs(const char* name, char value) {
    const char* const set = std::getenv(name);
    return set != nullptr && set[0] == value && set[1] == '\0';
}

inline void ReadSwitch() {
    verify = SwitchIs("EDEN_NX2_RUNAHEAD_VERIFY", '1');
    enabled = verify || SwitchIs("EDEN_NX2_RUNAHEAD", '1');
    stats = enabled && (verify || SwitchIs("EDEN_NX2_RUNAHEAD_STATS", '1'));
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
    u64 unvalued{};    ///< QMD signals whose releases the front did not note when it sent them
};
inline Counters counters;

/// The back's verify results (any thread may read them, the back writes them).
struct VerifyCounters {
    std::atomic<u64> checked{};
    std::atomic<u64> mismatches{};
    std::atomic<u64> flushes{}; ///< Of mismatches: the buffer cache holds the word GPU-written
};
inline VerifyCounters verified;

} // namespace VideoCommon::RunAhead
