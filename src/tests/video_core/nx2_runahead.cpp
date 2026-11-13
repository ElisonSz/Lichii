// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_runahead.h"

namespace VideoCommon {
struct RunAheadTestAccess {
    /// An entry of `words` words was just sent (as Commit does, without the ring).
    static void Send(GpuSplit& split, u64 words) {
        split.last_entry_start = split.write_position;
        split.write_position += words;
    }
    /// The back ran everything before `position` and stored it.
    static void Consume(GpuSplit& split, u64 position) {
        split.consumed.store(position, std::memory_order_seq_cst);
    }
    /// As MaybeStart with the switch on.
    static void Start(GpuSplit& split) {
        split.run_ahead_log =
            std::make_unique<GpuSplit::RunAheadWrite[]>(GpuSplit::RunAheadLogSize);
    }
    /// The entry just sent writes [device, device + size) (NoteWrite's placement of one piece).
    static void Write(GpuSplit& split, u64 device, u64 size) {
        split.NoteDeviceWriteAt(device, size, split.write_position);
    }
    /// A write of an entry sent earlier, placed now (a QMD release valued late).
    static void WriteAt(GpuSplit& split, u64 device, u64 size, u64 position) {
        split.NoteDeviceWriteAt(device, size, position);
    }
    static void CoarseWrite(GpuSplit& split, u64 device, u64 size) {
        split.NoteCoarseDeviceWriteAt(device, size, split.write_position);
    }
    static u64 Logged(const GpuSplit& split) {
        return split.run_ahead_count;
    }
    static u64 LogSize() {
        return GpuSplit::RunAheadLogSize;
    }
    static u64 MaxPages() {
        return GpuSplit::MaxWatchedPages;
    }
    /// The fenced writes of the entries before `position` landed.
    static void Land(GpuSplit& split, u64 position) {
        split.fenced_landed.store(split.EntryTag(position), std::memory_order_release);
    }
    /// The hashed granule sets' answer for device memory (MemoryWritesTarget's, one piece).
    static u64 GranuleTarget(const GpuSplit& split, u64 device, u64 size) {
        return (std::max)(split.last_unplaced_write, split.DeviceWritesTarget(device, size));
    }
    static size_t GranuleSet(u64 device) {
        return GpuSplit::WriteSlot(device >> GpuSplit::WriteGranuleBits);
    }
    /// As MaybeStart's ring, without a back thread (entries are run by hand, RunEntry).
    static void StartRing(GpuSplit& split) {
        split.ring = std::make_unique<u64[]>(GpuSplit::RingWords);
        split.active = true;
    }
    static void StopRing(GpuSplit& split) {
        split.active = false;
        split.ring.reset();
    }
    static u64 WritePosition(const GpuSplit& split) {
        return split.write_position;
    }
    static u64 LastEntry(const GpuSplit& split) {
        return split.last_entry_start;
    }
    /// Runs the deferred call sent at `position` as the back does (Execute's Kind::Call).
    static bool RunEntry(GpuSplit& split, u64 position) {
        split.back_entry_start = position;
        const u64* const entry = &split.ring[position & GpuSplit::RingMask];
        if ((entry[0] & 0xf) != static_cast<u64>(GpuSplit::Kind::Call) ||
            (entry[0] & GpuSplit::FastDrawNeutral) == 0) {
            return false;
        }
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return true;
    }
    /// The back registered a report of the entry at `position` that lands once its fence ran.
    static void PendFenced(GpuSplit& split, u64 position) {
        std::scoped_lock lock{split.fenced_entries_mutex};
        split.fenced_entries.push_back({split.EntryTag(position)});
    }
    /// Its fence ran (a fence thread).
    static void CompleteFenced(GpuSplit& split, u64 position) {
        split.CompleteFencedEntry(split.EntryTag(position));
    }
    static u64 Tag(const GpuSplit& split, u64 position) {
        return split.EntryTag(position);
    }
    /// The thread acts as the back (GpuSplit::ConvertsLandings looks at it).
    static void ActAsBack(bool back) {
        GpuSplit::on_back_thread = back;
    }
    /// The back finished entries up to `position` (BackLoop's seal).
    static void Seal(GpuSplit& split, u64 position) {
        split.SealFencedEntries(position);
    }
    /// BackLoop's check at an entry boundary.
    static void Boundary(GpuSplit& split, u64 position) {
        if (split.landing_closing) {
            split.MaybeEndLandings(position);
        }
    }
    static bool Converting(const GpuSplit& split) {
        return split.landing_converting;
    }
    static void ClearLandings(GpuSplit& split) {
        split.landing_window_open = false;
        split.landing_close_pending = false;
        split.landing_converting = false;
        split.landing_closing = false;
    }
    static bool Landed(const GpuSplit& split, u64 position) {
        return split.fenced_landed.load(std::memory_order_acquire) > split.EntryTag(position);
    }
    /// The front's window state alone (what NoteSemaphoreWrite looks at), without its entries.
    static void SetWindowOpen(GpuSplit& split, bool open) {
        split.landing_window_open = open;
    }
    static void SetClosePending(GpuSplit& split, bool pending) {
        split.landing_close_pending = pending;
    }
    /// The check the back gives the landings it converts (as the window's open or extension sets
    /// it).
    static void SetLandingCheck(GpuSplit& split, u64 check) {
        split.landing_check = check;
    }
};
} // namespace VideoCommon

namespace {

using Split = VideoCommon::GpuSplit;
using Access = VideoCommon::RunAheadTestAccess;
using Reach = VideoCommon::RunAhead::Reach;
using ViewKind = Split::SemaphoreView::Kind;
namespace RunAhead = VideoCommon::RunAhead;

struct SwitchScope {
    explicit SwitchScope(bool value) : previous{RunAhead::enabled}, stats{RunAhead::stats} {
        RunAhead::enabled = value;
        RunAhead::stats = false;
    }
    ~SwitchScope() {
        RunAhead::enabled = previous;
        RunAhead::stats = stats;
    }
    SwitchScope(const SwitchScope&) = delete;
    SwitchScope& operator=(const SwitchScope&) = delete;
    bool previous;
    bool stats;
};

/// A started split with the switch on, its word's page watched since position 0 and run past.
struct Fixture {
    SwitchScope on{true};
    std::unique_ptr<Split> split = std::make_unique<Split>();
    static constexpr u64 Word = 0x2b010;

    Fixture() {
        Access::Start(*split);
        (void)split->RunAheadReach(Word, 0); // Watches the page from position 0
        Access::Send(*split, 2);
        Access::Consume(*split, 2);
    }
    RunAhead::Answer Reach(u64 writers_before) {
        return split->RunAheadReach(Word, writers_before);
    }
};

TEST_CASE("Run-ahead switch values", "[video_core][runahead]") {
    const bool enabled = RunAhead::enabled;
    const bool verify = RunAhead::verify;
    const bool stats = RunAhead::stats;
    const auto set = [](const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        if (value[0] == '\0') {
            unsetenv(name);
        } else {
            setenv(name, value, 1);
        }
#endif
    };
    set("EDEN_NX2_RUNAHEAD", "1");
    set("EDEN_NX2_RUNAHEAD_VERIFY", "");
    set("EDEN_NX2_RUNAHEAD_STATS", "");
    RunAhead::ReadSwitch();
    CHECK((RunAhead::enabled && !RunAhead::verify && !RunAhead::stats));
    set("EDEN_NX2_RUNAHEAD_STATS", "1");
    RunAhead::ReadSwitch();
    CHECK((RunAhead::enabled && !RunAhead::verify && RunAhead::stats));
    set("EDEN_NX2_RUNAHEAD", "");
    RunAhead::ReadSwitch();
    CHECK((!RunAhead::enabled && !RunAhead::stats)); // Stats alone count nothing
    set("EDEN_NX2_RUNAHEAD_VERIFY", "1");
    RunAhead::ReadSwitch();
    CHECK((RunAhead::enabled && RunAhead::verify && RunAhead::stats)); // Verify implies both
    set("EDEN_NX2_RUNAHEAD_VERIFY", "");
    set("EDEN_NX2_RUNAHEAD_STATS", "");
    set("EDEN_NX2_RUNAHEAD", "2");
    RunAhead::ReadSwitch();
    CHECK(!RunAhead::enabled);
    set("EDEN_NX2_RUNAHEAD", "");
    RunAhead::ReadSwitch();
    CHECK(!RunAhead::enabled);
    // EDEN_NX2_RUNAHEAD_W1_MAX: off by default, on only with "1", implies nothing else.
    const bool w1_max = RunAhead::w1_max;
    CHECK_FALSE(RunAhead::w1_max);
    set("EDEN_NX2_RUNAHEAD_W1_MAX", "1");
    RunAhead::ReadSwitch();
    CHECK((RunAhead::w1_max && !RunAhead::enabled && !RunAhead::landings && !RunAhead::w1 &&
           !RunAhead::stats));
    set("EDEN_NX2_RUNAHEAD_W1_MAX", "2");
    RunAhead::ReadSwitch();
    CHECK_FALSE(RunAhead::w1_max);
    set("EDEN_NX2_RUNAHEAD_W1_MAX", "");
    RunAhead::ReadSwitch();
    CHECK_FALSE(RunAhead::w1_max);
    RunAhead::w1_max = w1_max;
    // EDEN_NX2_RUNAHEAD_WINDOW_BASE: likewise.
    const bool window_base = RunAhead::window_base;
    CHECK_FALSE(RunAhead::window_base);
    set("EDEN_NX2_RUNAHEAD_WINDOW_BASE", "1");
    RunAhead::ReadSwitch();
    CHECK((RunAhead::window_base && RunAhead::enabled && !RunAhead::landings && !RunAhead::stats));
    set("EDEN_NX2_RUNAHEAD_WINDOW_BASE", "0");
    RunAhead::ReadSwitch();
    CHECK_FALSE(RunAhead::window_base);
    set("EDEN_NX2_RUNAHEAD_WINDOW_BASE", "");
    RunAhead::ReadSwitch();
    CHECK((!RunAhead::window_base && !RunAhead::enabled));
    RunAhead::window_base = window_base;
    RunAhead::enabled = enabled;
    RunAhead::verify = verify;
    RunAhead::stats = stats;
}

TEST_CASE("Run-ahead off logs nothing", "[video_core][runahead]") {
    SwitchScope off{false};
    auto split = std::make_unique<Split>();
    Access::Start(*split); // Even with a log, nothing writes it while the switch is off
    Access::Send(*split, 4);
    Access::Write(*split, Fixture::Word, 16);
    split->NoteSemaphoreWrite(Fixture::Word, 5, true, Split::Landing::AtRun);
    CHECK(Access::Logged(*split) == 0);
    // No log at all (the switch was off at the start): the reach is the baseline's.
    auto bare = std::make_unique<Split>();
    Access::Send(*bare, 8);
    const auto answer = bare->RunAheadReach(Fixture::Word, 6);
    CHECK(answer.position == 6);
    CHECK(answer.reason == Reach::Lost);
}

TEST_CASE("Run-ahead reach of a word nothing writes", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 10); // Compute and copy work that writes elsewhere
    const auto answer = f.Reach(12);
    CHECK(answer.position == 0);
    CHECK(answer.reason == Reach::None);
}

TEST_CASE("Run-ahead reach of a newly watched page", "[video_core][runahead]") {
    SwitchScope on{true};
    auto split = std::make_unique<Split>();
    Access::Start(*split);
    Access::Send(*split, 10);
    Access::Write(*split, 0x5000, 4); // Not logged: the page is not watched yet
    CHECK(Access::Logged(*split) == 0);
    Access::Send(*split, 10);
    // The first acquire on the page: every write sent before may have hit it.
    auto answer = split->RunAheadReach(0x5000, 15);
    CHECK(answer.position == 15);
    CHECK(answer.reason == Reach::NewPage);
    Access::Send(*split, 2);
    Access::Write(*split, 0x5000, 4); // Logged now, with the entry that ends at 22
    CHECK(Access::Logged(*split) == 1);
    Access::Send(*split, 2);
    Access::Consume(*split, 20); // Past the watch start, not past the write at 22
    answer = split->RunAheadReach(0x5000, 24);
    CHECK(answer.position == 22);
    CHECK(answer.reason == Reach::Placed);
    Access::Consume(*split, 22);
    answer = split->RunAheadReach(0x5000, 24);
    CHECK(answer.reason == Reach::None);
}

TEST_CASE("Run-ahead reach of placed writes over the word", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word - 0x40, 0x100); // A copy whose range covers the word
    const u64 copy = f.split->Position();
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word - 0x10, 16); // The 16-byte slot before: not the word
    Access::Write(*f.split, Fixture::Word + 4, 4);     // The next word
    Access::Send(*f.split, 4);
    auto answer = f.Reach(f.split->Position());
    CHECK(answer.position == copy);
    CHECK(answer.reason == Reach::Placed);
    Access::Consume(*f.split, copy);
    answer = f.Reach(f.split->Position());
    CHECK(answer.position == 0); // Granules would still count the neighbours
    // A coarse write (copies over 16 KiB) is logged piece by piece, as placed.
    Access::Send(*f.split, 4);
    Access::CoarseWrite(*f.split, Fixture::Word & ~u64{0xfff}, 0x1000);
    answer = f.Reach(f.split->Position());
    CHECK(answer.position == f.split->Position());
    CHECK(answer.reason == Reach::Placed);
}

TEST_CASE("Run-ahead leaves releases noted at the word to the view", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    // A 16-byte report at the word: payload and high word noted, the timestamp after them not.
    Access::Write(*f.split, Fixture::Word, 16);
    f.split->NoteSemaphoreWrite(Fixture::Word, 7, true, Split::Landing::AtRun);
    f.split->NoteSemaphoreWrite(Fixture::Word + 4, 0, true, Split::Landing::AtRun);
    const u64 report = f.split->Position();
    Access::Send(*f.split, 4);
    CHECK(f.Reach(f.split->Position()).reason == Reach::None);
    CHECK(f.split->RunAheadReach(Fixture::Word + 4, f.split->Position()).reason == Reach::None);
    const auto timestamp = f.split->RunAheadReach(Fixture::Word + 8, f.split->Position());
    CHECK(timestamp.position == report);
    CHECK(timestamp.reason == Reach::Placed);
    // A release noted at another position does not explain this write.
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word, 4);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 9, true, Split::Landing::AtRun);
    CHECK(f.Reach(f.split->Position()).reason == Reach::Placed);
}

TEST_CASE("Run-ahead reach of QMD signals the front did not value", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    f.split->NoteUnvaluedSignal(true); // Waits to be valued
    const u64 waiting = f.split->Position();
    Access::Send(*f.split, 4);
    auto answer = f.Reach(f.split->Position());
    CHECK(answer.position == waiting);
    CHECK(answer.reason == Reach::Unvalued);
    // Valued late: its releases are placed (elsewhere here) and it no longer counts.
    Access::WriteAt(*f.split, 0x9000, 16, waiting);
    f.split->NoteWaitingSignalsDone(false);
    CHECK(f.Reach(f.split->Position()).reason == Reach::None);
    // Valued late with a release into the word: placed at the signal's position.
    Access::Send(*f.split, 4);
    f.split->NoteUnvaluedSignal(true);
    const u64 second = f.split->Position();
    Access::Send(*f.split, 4);
    Access::WriteAt(*f.split, Fixture::Word, 4, second);
    f.split->NoteWaitingSignalsDone(false);
    answer = f.Reach(f.split->Position());
    CHECK(answer.position == second);
    CHECK(answer.reason == Reach::Placed);
    Access::Consume(*f.split, second);
    // Dropped unvalued: unknown until the back ran it.
    Access::Send(*f.split, 4);
    f.split->NoteUnvaluedSignal(true);
    const u64 dropped = f.split->Position();
    f.split->NoteWaitingSignalsDone(true);
    Access::Send(*f.split, 4);
    answer = f.Reach(f.split->Position());
    CHECK(answer.position == dropped);
    CHECK(answer.reason == Reach::Unvalued);
    // Not valued at all (the view lost its table).
    Access::Send(*f.split, 4);
    f.split->NoteUnvaluedSignal(false);
    const u64 lost = f.split->Position();
    Access::Send(*f.split, 4);
    answer = f.Reach(f.split->Position());
    CHECK(answer.position == lost);
    Access::Consume(*f.split, lost);
    CHECK(f.Reach(f.split->Position()).reason == Reach::None);
}

TEST_CASE("Run-ahead reach of writes the front could not place", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    f.split->NoteUnplacedWrite();
    const u64 unplaced = f.split->Position();
    Access::Send(*f.split, 4);
    const auto answer = f.Reach(f.split->Position());
    CHECK(answer.position == unplaced);
    CHECK(answer.reason == Reach::Unplaced);
}

TEST_CASE("Run-ahead keeps the baseline's writers on GPU-written pages", "[video_core][runahead]") {
    Fixture f;
    Tegra::Nx2MarkGpuWritten(Fixture::Word & ~u64{0xfff}, 0x1000);
    Access::Send(*f.split, 4);
    const auto answer = f.Reach(5);
    Tegra::Nx2ClearGpuWrittenPages();
    CHECK(answer.position == 5);
    CHECK(answer.reason == Reach::GpuWritten);
}

TEST_CASE("Run-ahead reach never passes the baseline's writers", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    const u64 writers = f.split->Position();
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word, 4); // After the writers: the baseline does not wait
    Access::Send(*f.split, 4);
    const auto answer = f.Reach(writers);
    CHECK(answer.position == writers);
    CHECK(answer.reason == Reach::Placed);
}

TEST_CASE("Run-ahead log overflow and page limits", "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    const u64 first = f.split->Position();
    for (u64 i = 0; i <= Access::LogSize(); ++i) {
        Access::Write(*f.split, 0x2b800, 4); // Same page, another word
        Access::Send(*f.split, 1);
    }
    // The first write fell out of the log unconsumed: anything up to it may have hit the word.
    auto answer = f.Reach(f.split->Position());
    CHECK(answer.position >= first);
    CHECK(answer.reason == Reach::Lost);
    Access::Consume(*f.split, f.split->Position());
    CHECK(f.Reach(f.split->Position()).reason == Reach::None);
    // Pages past the limit keep the baseline's writers.
    for (u64 page = 0; page < Access::MaxPages(); ++page) {
        (void)f.split->RunAheadReach((page + 0x100) << 12, 0);
    }
    answer = f.split->RunAheadReach(u64{0x7777} << 12, 3);
    CHECK(answer.position == 3);
    CHECK(answer.reason == Reach::Lost);
}

TEST_CASE("Run-ahead scan stops at writes logged before the back's position",
          "[video_core][runahead]") {
    Fixture f;
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word, 4);
    Access::Send(*f.split, 4);
    const u64 run = f.split->Position();
    // Placed late at an old position, logged after the back ran it: nothing to wait for.
    Access::WriteAt(*f.split, Fixture::Word, 4, 3);
    Access::Consume(*f.split, run);
    Access::Send(*f.split, 4);
    CHECK(f.Reach(f.split->Position()).reason == Reach::None);
}

/// The view's answer for an acquire on a word only the model writes, the front's notes aside.
struct ModelWord {
    u32 memory{};
    bool Matches(u32 value, u32 target, bool equal) const {
        return equal ? value == target : value >= target;
    }
};

TEST_CASE("Run-ahead passes where the baseline passes after its wait, sooner",
          "[video_core][runahead]") {
    // An equal acquire (a word that does not only advance): the baseline waits for every
    // compute and copy entry sent before it, then asks the view again.
    Fixture f;
    ModelWord word{.memory = 5};
    const auto view = [&](u64 writers) {
        return f.split->ViewSemaphore(
            Fixture::Word, false, writers, [](u32 value) { return value == 5; },
            [&](bool) -> std::optional<u32> { return word.memory; });
    };
    Access::Send(*f.split, 8); // Dispatches and copies writing elsewhere
    f.split->NoteWriter();
    const u64 writers = f.split->LastWriterPosition();
    Access::Send(*f.split, 2); // The acquire's own entries
    const auto base = view(writers);
    CHECK(base.kind == ViewKind::Unknown);
    CHECK(base.position == writers);
    const auto reach = f.Reach(writers);
    CHECK(view(reach.position).kind == ViewKind::Met);
    // The baseline after its wait: the same answer.
    Access::Consume(*f.split, writers);
    CHECK(view(writers).kind == ViewKind::Met);
}

TEST_CASE("Run-ahead model: reach decisions match the baseline after its wait",
          "[video_core][runahead]") {
    // Random entries before an acquire on the word: writes elsewhere, placed writes over the
    // word, releases noted at it, QMD signals valued or not (that may or may not write it), writes
    // the front could not place. The model knows what each entry does to the word; the split
    // knows what the front noted. Whenever the view with the reach passes the acquire, the view
    // the baseline asks once the back ran its writers (all compute and copy entries) passes too,
    // on the same word value, which no entry between the reach and the acquire changed.
    std::mt19937_64 random{0x5eed};
    u64 checked = 0;
    u64 sooner = 0;
    for (u32 round = 0; round < 400; ++round) {
        Fixture f;
        struct Entry {
            u64 position;
            std::optional<u32> writes; ///< What the back writes to the word running it
            bool writer;               ///< A compute or copy entry (LastWriterPosition)
        };
        std::vector<Entry> entries;
        u32 memory = 3;
        const u32 count = 1 + static_cast<u32>(random() % 12);
        for (u32 i = 0; i < count; ++i) {
            Access::Send(*f.split, 1 + random() % 4);
            Entry entry{f.split->Position(), std::nullopt, false};
            const u32 value = static_cast<u32>(random() % 8);
            switch (random() % 7) {
            case 0: // Compute or copy work writing elsewhere
                entry.writer = true;
                Access::Write(*f.split, 0x9000 + (random() % 64) * 4, 4);
                break;
            case 1: // A copy over the word
                entry.writer = true;
                entry.writes = value;
                Access::Write(*f.split, Fixture::Word - 8, 32);
                break;
            case 2: // A release noted at the word (a report, a QMD release valued at once)
                entry.writer = random() % 2 == 0;
                entry.writes = value;
                Access::Write(*f.split, Fixture::Word, 16);
                f.split->NoteSemaphoreWrite(Fixture::Word, value, true, Split::Landing::AtRun);
                break;
            case 3: // A QMD signal the front did not value: it may write the word
                entry.writer = true;
                if (random() % 2 == 0) {
                    entry.writes = value;
                }
                f.split->NoteUnvaluedSignal(false);
                break;
            case 4: // A write the front could not place
                if (random() % 2 == 0) {
                    entry.writes = value;
                }
                f.split->NoteUnplacedWrite();
                break;
            case 5: // A neighbouring word in the same granule
                entry.writer = random() % 2 == 0;
                Access::Write(*f.split, Fixture::Word + 0x10, 16);
                break;
            default: // 3D work
                break;
            }
            if (entry.writer) {
                f.split->NoteWriter();
            }
            entries.push_back(entry);
        }
        const u64 writers = f.split->LastWriterPosition();
        Access::Send(*f.split, 2); // The acquire
        const u32 target = static_cast<u32>(random() % 8);
        const bool equal = random() % 2 == 0;
        // The back has run a random prefix: the word holds what those entries wrote.
        const u64 ran = random() % (f.split->Position() + 1);
        Access::Consume(*f.split, ran);
        for (const Entry& entry : entries) {
            if (entry.position <= ran && entry.writes) {
                memory = *entry.writes;
            }
        }
        const auto matches = [&](u32 value) { return equal ? value == target : value >= target; };
        const auto memory_now = [&](bool) -> std::optional<u32> { return memory; };
        const auto reach = f.Reach(writers);
        const auto fast =
            f.split->ViewSemaphore(Fixture::Word, false, reach.position, matches, memory_now);
        if (fast.kind != ViewKind::Met && fast.kind != ViewKind::MetByQueued) {
            continue;
        }
        // The baseline: the back runs its writers (and maybe more, but never past the acquire),
        // then the view is asked again with the word as those entries left it.
        u32 after = memory;
        for (const Entry& entry : entries) {
            if (entry.position > ran && entry.position <= writers && entry.writes) {
                after = *entry.writes;
            }
        }
        Access::Consume(*f.split, (std::max)(ran, writers));
        const auto base = f.split->ViewSemaphore(Fixture::Word, false, writers, matches,
                                                 [&](bool) -> std::optional<u32> { return after; });
        ++checked;
        sooner += ran < writers ? 1 : 0;
        INFO("round " << round);
        CHECK((base.kind == ViewKind::Met || base.kind == ViewKind::MetByQueued));
        CHECK(reach.position <= writers);
    }
    CHECK(checked > 50);
    CHECK(sooner > 10);
}

// Round 2: W1 passes, exact QMD placement.

using W1 = RunAhead::W1;

TEST_CASE("Run-ahead W1 decisions", "[video_core][runahead]") {
    Fixture f;
    const auto equals = [](u32 target) { return [target](u32 value) { return value == target; }; };
    const auto unknown_at = [](u64 position) {
        return Split::SemaphoreView{ViewKind::Unknown, position, true};
    };
    std::optional<Split::QueuedSemaphoreWrite> release;
    // A fence-flagged report with a known payload: the acquire's view waits for its entry.
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 7, true, Split::Landing::Fenced);
    const u64 report = f.split->Position();
    Access::Send(*f.split, 2);
    CHECK(f.split->W1Release(Fixture::Word, unknown_at(report), true, 0, equals(7), &release) ==
          W1::Pass);
    REQUIRE(release.has_value());
    CHECK(release->payload == 7);
    CHECK(release->position == report);
    // Not met by its payload, no waiter slot, another position, a wait for writers.
    CHECK(f.split->W1Release(Fixture::Word, unknown_at(report), true, 0, equals(8), nullptr) ==
          W1::Unmet);
    CHECK(f.split->W1Release(Fixture::Word, unknown_at(report), false, 0, equals(7), nullptr) ==
          W1::Unlatched);
    CHECK(f.split->W1Release(Fixture::Word, unknown_at(report + 1), true, 0, equals(7), nullptr) ==
          W1::NotW1);
    CHECK(f.split->W1Release(Fixture::Word, Split::SemaphoreView{ViewKind::Unknown, report}, true,
                             0, equals(7), nullptr) == W1::NotW1);
    CHECK(f.split->W1Release(Fixture::Word, Split::SemaphoreView{ViewKind::Met}, true, 0, equals(7),
                             nullptr) == W1::NotW1);
    // Ran but not landed: memory does not meet it yet (NotMet); the release still decides.
    Access::Consume(*f.split, f.split->Position());
    CHECK(f.split->W1Release(Fixture::Word, Split::SemaphoreView{ViewKind::NotMet}, true, 0,
                             equals(7), nullptr) == W1::Pass);
    // Landed: memory decides, as always.
    Access::Land(*f.split, f.split->Position());
    CHECK(f.split->W1Release(Fixture::Word, Split::SemaphoreView{ViewKind::NotMet}, true, 0,
                             equals(7), nullptr) == W1::NotW1);
    // A reduction or maximum, or a release behind a pending fenced one: the front can't tell.
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 0, false, Split::Landing::Fenced);
    const u64 reduction = f.split->Position();
    CHECK(f.split->W1Release(Fixture::Word, unknown_at(reduction), true, 0, equals(0), nullptr) ==
          W1::NotPlain);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 4, 3, true, Split::Landing::Fenced);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 4, 5, true, Split::Landing::AtRun);
    CHECK(f.split->W1Release(Fixture::Word + 4, Split::SemaphoreView{ViewKind::NotMet}, true, 0,
                             equals(5), nullptr) == W1::NotPlain);
    // A write over the word sent after a release (an upload, a copy) can drop it at the back: the
    // baseline decides from what that write left. A write next to it, or one before it, does not.
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 0x20, 6, true, Split::Landing::Fenced);
    const u64 later = f.split->Position();
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word + 0x24, 4);
    CHECK(f.split->W1Release(Fixture::Word + 0x20, unknown_at(later), true, 0, equals(6),
                             nullptr) == W1::Pass);
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word + 0x20, 4);
    CHECK(f.split->W1Release(Fixture::Word + 0x20, unknown_at(later), true, 0, equals(6),
                             nullptr) == W1::Overwritten);
    // A release landing at run is not W1 (the view waits for its entry and memory decides).
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 8, 2, true, Split::Landing::AtRun);
    CHECK(f.split->W1Release(Fixture::Word + 8, unknown_at(f.split->Position()), true, 0, equals(2),
                             nullptr) == W1::NotW1);
}

TEST_CASE("Run-ahead W1 passes where the release latches the baseline's waiter",
          "[video_core][runahead]") {
    // The baseline's acquire waits suspended with a registered waiter; the release, landing,
    // latches it even when later writes (CPU, other releases) change the word again. Whenever
    // W1Release lets the acquire go on, that landing latches it: the baseline goes on as well.
    std::mt19937_64 random{0x1a7c};
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    u64 passes = 0;
    for (u32 round = 0; round < 300; ++round) {
        Fixture f;
        const u64 device = Fixture::Word;
        const u32 payload = static_cast<u32>(random() % 6);
        const u32 target = static_cast<u32>(random() % 6);
        const bool gequal = random() % 2 == 0;
        const u32 operation = gequal ? Tegra::Nx2SemaphoreWaiters::AcquireGequal
                                     : Tegra::Nx2SemaphoreWaiters::AcquireEqual;
        Access::Send(*f.split, 4);
        f.split->NoteSemaphoreWrite(device, payload, true, Split::Landing::Fenced);
        const u64 report = f.split->Position();
        Access::Send(*f.split, 2);
        int owner = 0;
        REQUIRE(waiters.Add(&owner, device, target, operation));
        const auto matches = [&](u32 value) { return gequal ? value >= target : value == target; };
        const W1 decision =
            f.split->W1Release(device, Split::SemaphoreView{ViewKind::Unknown, report, true}, true,
                               0, matches, nullptr);
        // Writes before the landing that don't meet it, then the landing, then later writes.
        for (u32 i = random() % 3; i > 0; --i) {
            const u32 value = static_cast<u32>(random() % 6);
            if (!matches(value)) {
                waiters.OnWrite(device, value);
            }
        }
        CHECK(!waiters.Latched(&owner));
        waiters.OnWrite(device, payload); // The release lands
        waiters.OnWrite(device, static_cast<u32>(random() % 6));
        if (decision == W1::Pass) {
            ++passes;
            CHECK(waiters.Latched(&owner));
        } else {
            CHECK(decision == W1::Unmet);
            CHECK(!matches(payload));
        }
        waiters.Remove(&owner);
    }
    CHECK(passes > 50);
}

TEST_CASE("Run-ahead exact placement of the writes a QMD read waits for",
          "[video_core][runahead]") {
    Fixture f;
    constexpr u64 Qmd = 0x7000'0000; // A QMD's device address (256 bytes)
    // First read of the page: writes sent before it were not logged.
    Access::Send(*f.split, 4);
    auto exact = f.split->LoggedWritesTarget(Qmd, 256);
    REQUIRE(exact.has_value());
    CHECK(*exact == f.split->Position());
    Access::Consume(*f.split, f.split->Position());
    // A write into the QMD, one next to it in the same page, and writes that evict the QMD's
    // granule from its hashed 4-way set.
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Qmd + 0x40, 0x20);
    const u64 into = f.split->Position();
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Qmd + 0x100, 0x100); // The next QMD
    u64 evictions = 0;
    for (u64 granule = 0x10000; evictions < 8; ++granule) {
        const u64 device = granule << 8;
        if (Access::GranuleSet(device) != Access::GranuleSet(Qmd)) {
            continue;
        }
        Access::Send(*f.split, 2);
        Access::Write(*f.split, device, 4); // Another page: not logged
        ++evictions;
    }
    Access::Send(*f.split, 2);
    exact = f.split->LoggedWritesTarget(Qmd, 256);
    REQUIRE(exact.has_value());
    CHECK(*exact == into);
    // The granules answer with the latest position evicted from the set: a write elsewhere.
    CHECK(Access::GranuleTarget(*f.split, Qmd, 256) > into);
    // Once the back ran the write into it, nothing queued writes the QMD.
    Access::Consume(*f.split, into);
    exact = f.split->LoggedWritesTarget(Qmd, 256);
    REQUIRE(exact.has_value());
    CHECK(*exact <= into);
    CHECK(f.split->Ran(*exact));
    CHECK(!f.split->Ran(Access::GranuleTarget(*f.split, Qmd, 256)));
    // A late placement (a QMD release valued late) at an earlier, unrun position counts.
    Access::WriteAt(*f.split, Qmd + 0x80, 4, into + 2);
    exact = f.split->LoggedWritesTarget(Qmd, 256);
    REQUIRE(exact.has_value());
    CHECK(*exact == into + 2);
    // A write sent after a signal that the back already ran still orders after the signal: a
    // view valuing the signal late must not read the QMD the write changed.
    const u64 signal = f.split->Position();
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Qmd + 0x68, 4); // Word 26
    const u64 rewrite = f.split->Position();
    Access::Send(*f.split, 2);
    Access::Consume(*f.split, f.split->Position());
    exact = f.split->LoggedWritesTarget(Qmd, 256);
    REQUIRE(exact.has_value());
    CHECK(*exact == rewrite);
    CHECK(*exact > signal);
}

} // namespace

TEST_CASE("Run-ahead W1 hold: the back runs what follows once the release landed",
          "[video_core][runahead]") {
    Fixture f;
    Access::StartRing(*f.split);
    RunAhead::holds.waits.store(0);
    // The fence-flagged release Q, in an entry the back ran: its report waits for its fence.
    Access::Send(*f.split, 4);
    const u64 q = Access::LastEntry(*f.split);
    f.split->NoteSemaphoreWrite(Fixture::Word, 7, true, Split::Landing::Fenced);
    Access::PendFenced(*f.split, q);
    // A release that lands at run, sent after it to the word: the word holds either until Q
    // landed, so a hold on the word still waits for Q.
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 8, true, Split::Landing::AtRun);
    const auto entry = f.split->UnlandedFencedEntry(Fixture::Word);
    REQUIRE(entry == Access::Tag(*f.split, q)); // Q's, not the later release's
    const u64 hold = Access::WritePosition(*f.split);
    f.split->HoldBackUntilLanded(*entry);
    REQUIRE(Access::WritePosition(*f.split) > hold);
    std::atomic<bool> done{false};
    std::atomic<bool> ran{false};
    std::thread back{[&] {
        ran.store(Access::RunEntry(*f.split, hold));
        done.store(true);
    }};
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    CHECK_FALSE(done.load()); // Sealed past Q, but Q's fence has not run
    CHECK_FALSE(Access::Landed(*f.split, q));
    Access::CompleteFenced(*f.split, q);
    back.join();
    CHECK(done.load());
    CHECK(ran.load()); // A deferred call that leaves draws as they are
    CHECK(Access::Landed(*f.split, q));
    CHECK(RunAhead::holds.waits.load() == 1);
    // Landed: nothing to hold for, nor for a word whose last release lands at run.
    CHECK_FALSE(f.split->UnlandedFencedEntry(Fixture::Word).has_value());
    f.split->NoteSemaphoreWrite(Fixture::Word + 0x10, 1, true, Split::Landing::AtRun);
    CHECK_FALSE(f.split->UnlandedFencedEntry(Fixture::Word + 0x10).has_value());
    // A hold whose release lands before the back reaches it does not wait.
    Access::Send(*f.split, 4);
    const u64 q2 = Access::LastEntry(*f.split);
    f.split->NoteSemaphoreWrite(Fixture::Word, 9, true, Split::Landing::Fenced);
    Access::PendFenced(*f.split, q2);
    const auto entry2 = f.split->UnlandedFencedEntry(Fixture::Word);
    REQUIRE(entry2 == Access::Tag(*f.split, q2));
    const u64 hold2 = Access::WritePosition(*f.split);
    f.split->HoldBackUntilLanded(*entry2);
    REQUIRE(Access::WritePosition(*f.split) > hold2);
    Access::CompleteFenced(*f.split, q2); // Not sealed past it yet: not landed
    CHECK_FALSE(Access::Landed(*f.split, q2));
    CHECK(Access::RunEntry(*f.split, hold2)); // Seals: lands at once
    CHECK(Access::Landed(*f.split, q2));
    CHECK(RunAhead::holds.waits.load() == 1);
    Access::StopRing(*f.split);
}

namespace {
struct LandingsScope {
    LandingsScope() : was{RunAhead::landings} {
        RunAhead::landings = true;
    }
    ~LandingsScope() {
        RunAhead::landings = was;
        Access::ActAsBack(false);
    }
    LandingsScope(const LandingsScope&) = delete;
    LandingsScope& operator=(const LandingsScope&) = delete;
    bool was;
};

bool NotedFenced(const Split& split, u64 word) {
    const auto queued = split.QueuedSemaphoreWriteAt(word);
    return queued && queued->fenced;
}
} // namespace

TEST_CASE("Run-ahead landings: a window converts until its close is acknowledged",
          "[video_core][runahead]") {
    Fixture f;
    LandingsScope on;
    Access::StartRing(*f.split);
    constexpr u64 Other = Fixture::Word + 0x100;
    // The fence-flagged release Q, run by the back: its report waits for its fence.
    Access::Send(*f.split, 4);
    const u64 q = Access::LastEntry(*f.split);
    f.split->NoteSemaphoreWrite(Fixture::Word, 7, true, Split::Landing::Fenced);
    Access::PendFenced(*f.split, q);
    // An acquire went on before Q landed: the window opens.
    const u64 open = Access::WritePosition(*f.split);
    f.split->OpenLandingWindow(Access::Tag(*f.split, q), 0, nullptr);
    REQUIRE(Access::WritePosition(*f.split) > open);
    // A release sent in the window is noted as landing once a fence ran.
    Access::Send(*f.split, 4);
    const u64 r = Access::LastEntry(*f.split);
    f.split->NoteSemaphoreWrite(Other, 1, true, Split::Landing::AtRun);
    CHECK(NotedFenced(*f.split, Other));
    // The back runs the window's entry and then the release: it converts its landing.
    Access::ActAsBack(true);
    CHECK_FALSE(f.split->ConvertsLandings());
    REQUIRE(Access::RunEntry(*f.split, open));
    CHECK(f.split->ConvertsLandings());
    auto landing = f.split->ConvertedLanding(Other, true);
    Access::PendFenced(*f.split, r); // Its report holds the entry's fenced writes
    Access::Seal(*f.split, Access::WritePosition(*f.split));
    Access::ActAsBack(false);
    CHECK_FALSE(f.split->ConvertsLandings()); // Only the back converts
    // Q has not landed: no close yet.
    f.split->CheckLandingWindow(nullptr);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Other + 0x10, 1, true, Split::Landing::AtRun);
    CHECK(NotedFenced(*f.split, Other + 0x10));
    // Q lands: the front closes the window; until the back acknowledges it, notes stay fenced.
    Access::CompleteFenced(*f.split, q);
    REQUIRE(Access::Landed(*f.split, q));
    const u64 close = Access::WritePosition(*f.split);
    f.split->CheckLandingWindow(nullptr);
    REQUIRE(Access::WritePosition(*f.split) > close);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Other + 0x20, 1, true, Split::Landing::AtRun);
    CHECK(NotedFenced(*f.split, Other + 0x20));
    // The back runs the close with a converted landing still pending: it keeps converting.
    Access::ActAsBack(true);
    REQUIRE(Access::RunEntry(*f.split, close));
    Access::Boundary(*f.split, Access::WritePosition(*f.split));
    CHECK(f.split->ConvertsLandings());
    f.split->CheckLandingWindow(nullptr);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Other + 0x30, 1, true, Split::Landing::AtRun);
    CHECK(NotedFenced(*f.split, Other + 0x30));                // Not acknowledged yet
    CHECK(f.split->QueuedSemaphoreWriteAt(Other).has_value()); // Not landed: not in memory
    // The converted landing ran: the back stops converting and acknowledges.
    landing.reset();
    Access::CompleteFenced(*f.split, r);
    Access::Boundary(*f.split, Access::WritePosition(*f.split));
    CHECK_FALSE(f.split->ConvertsLandings());
    Access::ActAsBack(false);
    f.split->CheckLandingWindow(nullptr);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Other + 0x40, 1, true, Split::Landing::AtRun);
    CHECK_FALSE(NotedFenced(*f.split, Other + 0x40));
    Access::StopRing(*f.split);
}

TEST_CASE("Run-ahead landings model: the back converts only what the front noted fenced, and "
          "lands nothing at run before a converted landing",
          "[video_core][runahead]") {
    std::mt19937 random{0x1a4d};
    u64 converted = 0;
    u64 at_run = 0;
    u64 extended = 0;
    for (int round = 0; round < 200; ++round) {
        Fixture f;
        LandingsScope on;
        Access::StartRing(*f.split);
        enum class Kind { Release, Fenced, Protocol };
        struct Sent {
            u64 position;
            Kind kind;
            u64 word;
            bool noted_fenced;
        };
        std::vector<Sent> sent;
        u64 next_run = 0; // Index into `sent` the back runs next
        // Fence order (FIFO): Q reports and converted landings complete in the order the back made
        // them; a landing completes as its token drops.
        struct Op {
            u64 position;
            std::shared_ptr<void> token; // Converted landings only
        };
        std::vector<Op> fence_ops;
        size_t completed_ops = 0;
        std::vector<u64> unlanded_q;
        u64 word_counter = 0;
        const auto front_step = [&] {
            const int choice = static_cast<int>(random() % 10);
            if (choice < 5) { // A release that lands at run
                Access::Send(*f.split, 2);
                const u64 word = Fixture::Word + 0x1000 + (word_counter++ % 64) * 0x10;
                f.split->NoteSemaphoreWrite(word, 1, true, Split::Landing::AtRun);
                sent.push_back({Access::LastEntry(*f.split), Kind::Release, word,
                                NotedFenced(*f.split, word)});
            } else if (choice < 7) { // A fence-flagged release
                Access::Send(*f.split, 2);
                f.split->NoteSemaphoreWrite(Fixture::Word, 3, true, Split::Landing::Fenced);
                sent.push_back({Access::LastEntry(*f.split), Kind::Fenced, Fixture::Word, true});
                unlanded_q.push_back(Access::LastEntry(*f.split));
            } else if (choice < 9) { // An acquire on a release not landed goes on: a window
                if (const auto entry = f.split->UnlandedFencedEntry(Fixture::Word)) {
                    const u64 before = Access::WritePosition(*f.split);
                    f.split->OpenLandingWindow(*entry, 0, nullptr);
                    if (Access::WritePosition(*f.split) != before) {
                        sent.push_back({before, Kind::Protocol, 0, false});
                    }
                }
            } else { // Between methods: maybe close, take an acknowledgement
                const u64 before = Access::WritePosition(*f.split);
                f.split->CheckLandingWindow(nullptr);
                if (Access::WritePosition(*f.split) != before) {
                    sent.push_back({before, Kind::Protocol, 0, false});
                }
            }
        };
        const auto back_step = [&] {
            if (next_run == sent.size()) {
                return;
            }
            const Sent& entry = sent[next_run++];
            Access::ActAsBack(true);
            if (entry.kind == Kind::Protocol) {
                REQUIRE(Access::RunEntry(*f.split, entry.position));
            } else if (entry.kind == Kind::Fenced) {
                Access::PendFenced(*f.split, entry.position);
                fence_ops.push_back({entry.position, {}});
            } else if (f.split->ConvertsLandings()) {
                // Converted: the front must have noted it fenced.
                CHECK(entry.noted_fenced);
                ++converted;
                Access::PendFenced(*f.split, entry.position);
                fence_ops.push_back({entry.position, f.split->ConvertedLanding(entry.word, true)});
            } else {
                // Lands at run: nothing converted may still be pending.
                ++at_run;
                for (size_t i = completed_ops; i < fence_ops.size(); ++i) {
                    CHECK_FALSE(fence_ops[i].token);
                }
            }
            const u64 end =
                next_run < sent.size() ? sent[next_run].position : Access::WritePosition(*f.split);
            Access::Seal(*f.split, end);
            const bool closing_with_pending = Access::Converting(*f.split);
            Access::Boundary(*f.split, end);
            extended += closing_with_pending && Access::Converting(*f.split) ? 1 : 0;
            Access::ActAsBack(false);
        };
        const auto fence_step = [&] {
            if (completed_ops == fence_ops.size()) {
                return;
            }
            Op& op = fence_ops[completed_ops++];
            op.token.reset();
            Access::CompleteFenced(*f.split, op.position);
            std::erase(unlanded_q, op.position);
        };
        for (int step = 0; step < 300; ++step) {
            const int who = static_cast<int>(random() % 3);
            if (who == 0) {
                front_step();
            } else if (who == 1) {
                back_step();
            } else {
                fence_step();
            }
        }
        Access::StopRing(*f.split);
    }
    // The model reaches every state: conversions, landings at run, and closes that wait.
    CHECK(converted > 100);
    CHECK(at_run > 100);
    CHECK(extended > 10);
}

TEST_CASE("Run-ahead landings verify: a converted landing checks its window's release",
          "[video_core][runahead]") {
    Fixture f;
    LandingsScope on;
    const bool verify_was = RunAhead::landings_verify;
    RunAhead::landings_verify = true;
    Access::StartRing(*f.split);
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    auto& counts = RunAhead::landing_counts;
    counts.ordered.store(0);
    counts.out_of_order.store(0);
    counts.unchecked.store(0);
    // The window's check: a waiter for the acquire (equal 7) its release must latch.
    auto& check = RunAhead::landing_checks[0];
    const u32 generation = check.generation.load() + 1;
    check.generation.store(generation);
    check.latched.store(false);
    REQUIRE(waiters.Add(&check, Fixture::Word, 7, Tegra::Nx2SemaphoreWaiters::AcquireEqual));
    f.split->OpenLandingWindow(1, RunAhead::PackLandingCheck(0, generation), nullptr);
    Access::ActAsBack(true);
    REQUIRE(Access::RunEntry(*f.split, 2));
    auto early = f.split->ConvertedLanding(Fixture::Word + 0x40, true);
    CHECK(f.split->LandingPendingAt(Fixture::Word + 0x40));
    Split::CheckConvertedLanding(early); // Applied before the release landed
    CHECK(counts.out_of_order.load() == 1);
    waiters.OnWrite(Fixture::Word, 7); // The release lands
    auto later = f.split->ConvertedLanding(Fixture::Word + 0x40, true);
    Split::CheckConvertedLanding(later);
    CHECK(counts.ordered.load() == 1);
    // The slot went to a later window: not checked.
    check.generation.store(generation + 1);
    Split::CheckConvertedLanding(later);
    CHECK(counts.unchecked.load() == 1);
    early.reset();
    later.reset();
    CHECK_FALSE(f.split->LandingPendingAt(Fixture::Word + 0x40));
    waiters.Remove(&check);
    Access::ActAsBack(false);
    RunAhead::landings_verify = verify_was;
    Access::StopRing(*f.split);
}

TEST_CASE("Run-ahead landings: a write over reports waits for the window and its landings",
          "[video_core][runahead]") {
    Fixture f;
    LandingsScope on;
    Access::StartRing(*f.split);
    RunAhead::landing_counts.overwrite_waits.store(0);
    Access::Send(*f.split, 4);
    const u64 q = Access::LastEntry(*f.split);
    f.split->NoteSemaphoreWrite(Fixture::Word, 7, true, Split::Landing::Fenced);
    Access::PendFenced(*f.split, q);
    const u64 open = Access::WritePosition(*f.split);
    f.split->OpenLandingWindow(Access::Tag(*f.split, q), 0, nullptr);
    Access::ActAsBack(true);
    REQUIRE(Access::RunEntry(*f.split, open));
    auto landing = f.split->ConvertedLanding(Fixture::Word + 0x40, true);
    Access::ActAsBack(false);
    std::atomic<bool> done{false};
    std::thread back{[&] {
        Access::ActAsBack(true);
        f.split->WaitBeforeOverwrite();
        Access::ActAsBack(false);
        done.store(true);
    }};
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    CHECK_FALSE(done.load()); // The window's release has not landed
    Access::CompleteFenced(*f.split, q);
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    CHECK_FALSE(done.load()); // A converted landing is still pending
    landing.reset();
    back.join();
    CHECK(done.load());
    CHECK(RunAhead::landing_counts.overwrite_waits.load() == 1);
    Access::StopRing(*f.split);
}

TEST_CASE("Run-ahead landings: the words of converted landings are known while they are pending",
          "[video_core][runahead]") {
    Fixture f;
    LandingsScope on;
    const bool verify_was = RunAhead::landings_verify;
    RunAhead::landings_verify = false; // The view holds reads by them without the verify too
    CHECK_FALSE(f.split->LandingPendingAt(Fixture::Word));
    auto first = f.split->ConvertedLanding(Fixture::Word + 2, true); // Its word
    auto second = f.split->ConvertedLanding(Fixture::Word, false);
    auto wordless = f.split->ConvertedLanding(0, false); // A syncpoint increment
    CHECK(f.split->LandingPendingAt(Fixture::Word));
    CHECK_FALSE(f.split->LandingPendingAt(Fixture::Word + 4));
    first.reset();
    CHECK(f.split->LandingPendingAt(Fixture::Word)); // The second still is
    second.reset();
    CHECK_FALSE(f.split->LandingPendingAt(Fixture::Word));
    wordless.reset();
    CHECK_FALSE(f.split->LandingPendingAt(0));
    RunAhead::landings_verify = verify_was;
}

// Round 4: W1 passes on maxima (EDEN_NX2_RUNAHEAD_W1_MAX), what the releases the plain rule does
// not pass are, and writes over the words of pending landings.

namespace {
struct MaxScope {
    explicit MaxScope(bool value) : was{RunAhead::w1_max} {
        RunAhead::w1_max = value;
    }
    ~MaxScope() {
        RunAhead::w1_max = was;
    }
    MaxScope(const MaxScope&) = delete;
    MaxScope& operator=(const MaxScope&) = delete;
    bool was;
};

Split::SemaphoreView UnknownAt(u64 position) {
    return Split::SemaphoreView{ViewKind::Unknown, position, true};
}
} // namespace

TEST_CASE("Run-ahead W1 on maxima (EDEN_NX2_RUNAHEAD_W1_MAX)", "[video_core][runahead]") {
    Fixture f;
    const auto reaches_to = [](u32 target) {
        return [target](u32 value) { return value >= target; };
    };
    const auto equals = [](u32 target) { return [target](u32 value) { return value == target; }; };
    // A maximum sent while a landing window is open (a QMD completion): the front notes it as
    // landing once a fence ran, at least its payload (no base: the word's last write is fenced).
    Access::SetWindowOpen(*f.split, true);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 9, true, true, Split::SemaphoreWriter::QmdReduction);
    const u64 maximum = f.split->Position();
    Access::SetWindowOpen(*f.split, false);
    Access::Send(*f.split, 2);
    const auto queued = f.split->QueuedSemaphoreWriteAt(Fixture::Word);
    REQUIRE(queued.has_value());
    CHECK((queued->fenced && queued->windowed && queued->at_least && queued->known &&
           !queued->raced));
    std::optional<Split::QueuedSemaphoreWrite> release;
    {
        // Off: not plain, as before (the release is still handed out, for the stats).
        MaxScope off{false};
        CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), true, 0, reaches_to(9),
                                 &release, true) == W1::NotPlain);
        REQUIRE(release.has_value());
        CHECK(release->at_least);
    }
    MaxScope on{true};
    release.reset();
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), true, 0, reaches_to(9), &release,
                             true) == W1::Pass);
    REQUIRE(release.has_value());
    CHECK(release->payload == 9);
    CHECK(release->position == maximum);
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), true, 0, reaches_to(4), nullptr,
                             true) == W1::Pass);
    // Ran but not landed, memory short of it (the view's NotMet on an advancing word).
    CHECK(f.split->W1Release(Fixture::Word, Split::SemaphoreView{ViewKind::NotMet}, true, 0,
                             reaches_to(9), nullptr, true) == W1::Pass);
    // Short of the acquire; an equal (or mask) acquire; no waiter slot.
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), true, 0, reaches_to(10), nullptr,
                             true) == W1::NotPlain);
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), true, 0, equals(9), nullptr,
                             false) == W1::NotPlain);
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), false, 0, reaches_to(9), nullptr,
                             true) == W1::Unlatched);
    // Another position: not W1 at all.
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum + 1), true, 0, reaches_to(9), nullptr,
                             true) == W1::NotW1);
    // A write over the word sent after it.
    Access::Send(*f.split, 4);
    Access::Write(*f.split, Fixture::Word, 4);
    CHECK(f.split->W1Release(Fixture::Word, UnknownAt(maximum), true, 0, reaches_to(9), nullptr,
                             true) == W1::Overwritten);
    // A value the front can't tell (a reduction without a base) stays: only maxima pass.
    Access::SetWindowOpen(*f.split, true);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 8, 0, false, false,
                                Split::SemaphoreWriter::ReductionCompute);
    Access::SetWindowOpen(*f.split, false);
    CHECK(f.split->W1Release(Fixture::Word + 8, UnknownAt(f.split->Position()), true, 0,
                             reaches_to(0), nullptr, true) == W1::NotPlain);
}

TEST_CASE("Run-ahead W1_MAX passes where the maximum's landing latches the baseline's waiter",
          "[video_core][runahead]") {
    // The baseline's acquire waits with a registered waiter until the maximum lands (in a window
    // the landing is a fence operation): it writes max(word, payload) and latches the waiter with
    // what acquires see then, through the mirror's shadow too. Whenever W1Release lets the
    // acquire go on, that landing latches it, whatever the word held: the baseline goes on too.
    std::mt19937_64 random{0x4d41};
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    auto& shadow = Tegra::Nx2MirrorShadow::Get();
    shadow.Clear();
    MaxScope on{true};
    u64 passes = 0;
    u64 mirrored_passes = 0;
    for (u32 round = 0; round < 600; ++round) {
        Fixture f;
        const u64 device = Fixture::Word;
        const u32 payload = static_cast<u32>(random() % 8);
        const u32 target = static_cast<u32>(random() % 8);
        constexpr std::array operations{Tegra::Nx2SemaphoreWaiters::AcquireEqual,
                                        Tegra::Nx2SemaphoreWaiters::AcquireGequal,
                                        Tegra::Nx2SemaphoreWaiters::AcquireMask};
        const u32 operation = operations[random() % operations.size()];
        const bool reaches = operation == Tegra::Nx2SemaphoreWaiters::AcquireGequal;
        const auto matches = [&](u32 value) {
            return operation == Tegra::Nx2SemaphoreWaiters::AcquireEqual    ? value == target
                   : operation == Tegra::Nx2SemaphoreWaiters::AcquireGequal ? value >= target
                                                                            : (value & target) != 0;
        };
        Access::SetWindowOpen(*f.split, true);
        Access::Send(*f.split, 4);
        f.split->NoteSemaphoreWrite(device, payload, true, true,
                                    Split::SemaphoreWriter::QmdReduction);
        Access::SetWindowOpen(*f.split, false);
        const u64 position = f.split->Position();
        Access::Send(*f.split, 2);
        int owner = 0;
        REQUIRE(waiters.Add(&owner, device, target, operation));
        const W1 decision =
            f.split->W1Release(device, UnknownAt(position), true, 0, matches, nullptr, reaches);
        // The word as the maximum lands: anything (earlier writes, the CPU); the mirror's shadow
        // may hold a value of its own for acquires.
        const u32 guest = static_cast<u32>(random() % 8);
        const auto combine = [payload](u32 current) { return (std::max)(current, payload); };
        u32 seen = combine(guest);
        const bool mirrored = random() % 2 == 0;
        if (mirrored) {
            shadow.Set(device, static_cast<u32>(random() % 8), guest);
            seen = shadow.NoteCombinedRelease(device, 0, guest, combine(guest), combine);
        }
        waiters.OnWrite(device, seen);                           // The maximum lands
        waiters.OnWrite(device, static_cast<u32>(random() % 8)); // Later writes change nothing
        if (decision == W1::Pass) {
            ++passes;
            mirrored_passes += mirrored ? 1 : 0;
            CHECK(reaches);
            CHECK(payload >= target);
            CHECK(waiters.Latched(&owner));
        } else {
            CHECK(decision == W1::NotPlain);
            CHECK((!reaches || payload < target));
        }
        waiters.Remove(&owner);
        shadow.Clear();
    }
    CHECK(passes > 50);
    CHECK(mirrored_passes > 20);
}

TEST_CASE("Run-ahead W1 kinds of the releases the plain rule does not pass",
          "[video_core][runahead]") {
    using Kind = RunAhead::W1Kind;
    using Writer = Split::SemaphoreWriter;
    const auto write = [](bool known, bool at_least, Writer writer, bool raced = false) {
        Split::QueuedSemaphoreWrite queued{5, known, at_least, true, writer, 10};
        queued.raced = raced;
        return queued;
    };
    CHECK(Split::W1KindOf(write(true, true, Writer::QmdReduction), W1::Pass, true, true) ==
          Kind::MaxMet);
    CHECK(Split::W1KindOf(write(true, true, Writer::QmdReduction), W1::NotPlain, true, false) ==
          Kind::MaxShort);
    CHECK(Split::W1KindOf(write(true, true, Writer::QmdReduction), W1::NotPlain, false, true) ==
          Kind::MaxOther);
    CHECK(Split::W1KindOf(write(false, false, Writer::QmdReduction), W1::NotPlain, true, false) ==
          Kind::QmdReduction);
    CHECK(Split::W1KindOf(write(false, false, Writer::ReductionCompute), W1::NotPlain, true,
                          false) == Kind::ComputeReduction);
    CHECK(Split::W1KindOf(write(false, false, Writer::Report3D), W1::NotPlain, false, false) ==
          Kind::Counter);
    CHECK(Split::W1KindOf(write(false, false, Writer::Report3D, true), W1::NotPlain, true, false) ==
          Kind::Raced);
    CHECK(Split::W1KindOf(write(false, false, Writer::Puller), W1::NotPlain, true, false) ==
          Kind::Unknown);
    CHECK(Split::W1KindOf(write(true, false, Writer::Puller), W1::Unmet, true, false) ==
          Kind::Unmet);
    CHECK(Split::W1KindOf(write(true, false, Writer::Puller), W1::Unlatched, true, true) ==
          Kind::Other);
    CHECK(Split::W1KindOf(write(true, false, Writer::Copy), W1::Overwritten, false, true) ==
          Kind::Other);
    CHECK(RunAhead::W1OperationIndex(Tegra::Nx2SemaphoreWaiters::AcquireEqual) == 0);
    CHECK(RunAhead::W1OperationIndex(Tegra::Nx2SemaphoreWaiters::AcquireGequal) == 1);
    CHECK(RunAhead::W1OperationIndex(Tegra::Nx2SemaphoreWaiters::AcquireMask) == 2);
    // The flags as NoteSemaphoreWrite sets them.
    Fixture f;
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 3, true, Split::Landing::Fenced); // Fence-flagged
    const auto own = f.split->QueuedSemaphoreWriteAt(Fixture::Word);
    REQUIRE(own.has_value());
    CHECK((own->fenced && own->known && !own->windowed && !own->raced));
    // Behind it, a release landing at run holds either value until it landed: raced.
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word, 4, true, Split::Landing::AtRun);
    const auto raced = f.split->QueuedSemaphoreWriteAt(Fixture::Word);
    REQUIRE(raced.has_value());
    CHECK((raced->fenced && !raced->known && raced->raced && !raced->windowed));
    // In a window (or until the back acknowledged its close), one landing at run is fenced.
    Access::SetWindowOpen(*f.split, true);
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 8, 1, true, Split::Landing::AtRun);
    const auto windowed = f.split->QueuedSemaphoreWriteAt(Fixture::Word + 8);
    REQUIRE(windowed.has_value());
    CHECK((windowed->fenced && windowed->known && windowed->windowed && !windowed->raced));
    // A fence-flagged one in a window is fenced on its own.
    Access::Send(*f.split, 4);
    f.split->NoteSemaphoreWrite(Fixture::Word + 0x10, 1, true, Split::Landing::Fenced);
    const auto flagged = f.split->QueuedSemaphoreWriteAt(Fixture::Word + 0x10);
    REQUIRE(flagged.has_value());
    CHECK((flagged->fenced && !flagged->windowed));
    Access::SetWindowOpen(*f.split, false);
}

TEST_CASE("Run-ahead landings: a write over the words of pending landings is found",
          "[video_core][runahead]") {
    Fixture f;
    LandingsScope on;
    constexpr u64 Word = Fixture::Word + 0x40;
    CHECK_FALSE(f.split->LandingPendingIn(Fixture::Word, 0x100)); // None pending
    auto landing = f.split->ConvertedLanding(Word, false);        // A QMD maximum's word
    auto wordless = f.split->ConvertedLanding(0, false);          // A syncpoint increment
    CHECK(f.split->LandingPendingIn(Fixture::Word, 0x100));
    CHECK(f.split->LandingPendingIn(Word, 4));
    CHECK(f.split->LandingPendingIn(Word + 2, 1));          // One byte of the word
    CHECK(f.split->LandingPendingIn(Word - 4, 5));          // Into its first byte
    CHECK_FALSE(f.split->LandingPendingIn(Word - 4, 4));    // Right before it
    CHECK_FALSE(f.split->LandingPendingIn(Word + 4, 0x10)); // Right after it
    CHECK_FALSE(f.split->LandingPendingIn(Word, 0));        // Nothing written
    // Other pages, one or many (more than the page slots: the exact look under the lock).
    CHECK_FALSE(f.split->LandingPendingIn(Word + 0x1000, 0x2000));
    CHECK(f.split->LandingPendingIn(0, 0x80000));
    CHECK_FALSE(f.split->LandingPendingIn(Word + 0x1000, 0x80000));
    landing.reset();
    CHECK_FALSE(f.split->LandingPendingIn(Fixture::Word, 0x100));
    CHECK_FALSE(f.split->LandingPendingIn(0, 0x80000));
    wordless.reset();
}

// Round 5: the landings verify's own bookkeeping, QMD reductions valued from a window's order
// (EDEN_NX2_RUNAHEAD_WINDOW_BASE), and releases made fence operations compared as they land.

namespace {
struct WindowBaseScope {
    WindowBaseScope() : was{RunAhead::window_base} {
        RunAhead::window_base = true;
    }
    ~WindowBaseScope() {
        RunAhead::window_base = was;
    }
    WindowBaseScope(const WindowBaseScope&) = delete;
    WindowBaseScope& operator=(const WindowBaseScope&) = delete;
    bool was;
};

void ClearLandingChecks() {
    for (auto& check : RunAhead::landing_checks) {
        Tegra::Nx2SemaphoreWaiters::Get().Remove(&check);
        check.registered = false;
    }
}
} // namespace

TEST_CASE("Run-ahead landings verify: the front's bookkeeping never reads as out of order",
          "[video_core][runahead]") {
    // Every window's release latches its check before a landing of the window is applied. The
    // front meanwhile moves latches to the flag (SweepLandingChecks) and gives slots to later
    // windows (StartLandingCheck). A landing checked during either counts as ordered or
    // unchecked, never out of order (t46/t48: the reader read the flag before the sweep set it,
    // then found the waiter the sweep had removed).
    Fixture f;
    LandingsScope on;
    const bool verify_was = RunAhead::landings_verify;
    RunAhead::landings_verify = true;
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    auto& counts = RunAhead::landing_counts;
    counts.ordered.store(0);
    counts.out_of_order.store(0);
    counts.unchecked.store(0);
    constexpr u64 Word = Fixture::Word + 0x80;
    const Split::QueuedSemaphoreWrite release{7, true, false, true, Split::SemaphoreWriter::Puller,
                                              4};
    int own = 0; // The acquire's own waiter (not registered: only the check's latches)
    std::mutex mutex;
    // Landings of the last windows: the fencing thread applies the newest and older ones, whose
    // slots later windows take.
    std::array<std::shared_ptr<void>, 40> recent;
    size_t sent_windows = 0;
    std::atomic<bool> done{false};
    std::thread fencing{[&] {
        std::mt19937 pick{0x77};
        while (!done.load(std::memory_order_acquire)) {
            std::shared_ptr<void> token;
            {
                std::scoped_lock lock{mutex};
                if (sent_windows != 0) {
                    const size_t back = pick() % 4 == 0 ? pick() % recent.size() : 0;
                    token = recent[(sent_windows - 1 + recent.size() - back) % recent.size()];
                }
            }
            for (int i = 0; token && i < 16; ++i) {
                Split::CheckConvertedLanding(token);
            }
        }
    }};
    std::mt19937 random{0x0dd5};
    std::atomic<u32> spin_sink{};
    for (int round = 0; round < 20000; ++round) {
        const u64 check = Split::StartLandingCheck(
            Word, 7, Tegra::Nx2SemaphoreWaiters::AcquireEqual, 0, nullptr, release, 0, &own, false);
        REQUIRE(check != 0);
        Access::SetLandingCheck(*f.split, check);
        auto token = f.split->ConvertedLanding(Word + 4, false);
        waiters.OnWrite(Word, 7); // The window's release lands: its check latches
        {
            std::scoped_lock lock{mutex};
            recent[sent_windows++ % recent.size()] = std::move(token);
        }
        for (u32 spin = static_cast<u32>(random() % 256); spin > 0; --spin) {
            spin_sink.fetch_add(1, std::memory_order_relaxed);
        }
        // In every other stretch no acquire sweeps: the slot's next window removes the latched
        // waiter itself.
        if ((round / 64) % 2 == 0 && random() % 4 != 0) {
            Split::SweepLandingChecks();
        }
    }
    done.store(true, std::memory_order_release);
    fencing.join();
    recent = {};
    CHECK(counts.out_of_order.load() == 0);
    CHECK(counts.ordered.load() > 1000);
    ClearLandingChecks();
    Access::SetLandingCheck(*f.split, 0);
    RunAhead::landings_verify = verify_was;
}

TEST_CASE("Run-ahead landings verify: a release that landed before the check counts as landed",
          "[video_core][runahead]") {
    Fixture f;
    LandingsScope on;
    const bool verify_was = RunAhead::landings_verify;
    RunAhead::landings_verify = true;
    auto& counts = RunAhead::landing_counts;
    counts.ordered.store(0);
    counts.out_of_order.store(0);
    constexpr u64 Word = Fixture::Word + 0x90;
    const Split::QueuedSemaphoreWrite release{3, true, false, true, Split::SemaphoreWriter::Puller,
                                              4};
    int own = 0;
    // The word met the acquire when the pass read it (the release landed before any waiter was
    // added): one read decides it for the W1 check and this one alike.
    const u64 met = Split::StartLandingCheck(Word, 3, Tegra::Nx2SemaphoreWaiters::AcquireEqual, 0,
                                             nullptr, release, 0, &own, true);
    Access::SetLandingCheck(*f.split, met);
    Split::CheckConvertedLanding(f.split->ConvertedLanding(Word + 4, false));
    CHECK(counts.ordered.load() == 1);
    // Not met, nothing latched: out of order.
    const u64 unmet = Split::StartLandingCheck(Word, 3, Tegra::Nx2SemaphoreWaiters::AcquireEqual, 0,
                                               nullptr, release, 0, &own, false);
    Access::SetLandingCheck(*f.split, unmet);
    Split::CheckConvertedLanding(f.split->ConvertedLanding(Word + 4, false));
    CHECK(counts.out_of_order.load() == 1);
    ClearLandingChecks();
    Access::SetLandingCheck(*f.split, 0);
    RunAhead::landings_verify = verify_was;
}

TEST_CASE("Run-ahead window base: whether the release before a reduction lands right before it",
          "[video_core][runahead]") {
    using Refusal = RunAhead::BaseRefusal;
    constexpr u64 Word = Fixture::Word + 0x40;
    const auto target = [](const Split& split) { return Access::GranuleTarget(split, Word, 4); };
    // A release noted at the word (its bytes placed first, as every sender places them).
    const auto release = [](Split& split, u32 payload, Split::Landing landing) {
        Access::Send(split, 4);
        Access::Write(split, Word, 4);
        split.NoteSemaphoreWrite(Word, payload, true, landing);
    };
    SECTION("nothing pending at the word: memory decides") {
        Fixture f;
        Access::SetWindowOpen(*f.split, true);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
    }
    SECTION("a fence-flagged release, a window open") {
        Fixture f;
        release(*f.split, 0x10, Split::Landing::Fenced);
        Access::SetWindowOpen(*f.split, true);
        CHECK(f.split->WindowBase(Word, target(*f.split), false) == 0x10u);
        // Shaders wrote the page.
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), true).has_value());
        // Closing: the reduction may land as the back runs it, before the release.
        Access::SetWindowOpen(*f.split, false);
        RunAhead::counters.base_refusals = {};
        RunAhead::stats = true;
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
        CHECK(RunAhead::counters.base_refusals[static_cast<size_t>(Refusal::Closing)] == 1);
        RunAhead::stats = false;
    }
    SECTION("a release landing at run, sent before the window opened") {
        Fixture f;
        release(*f.split, 0x11, Split::Landing::AtRun);
        Access::SetWindowOpen(*f.split, true);
        CHECK(f.split->WindowBase(Word, target(*f.split), false) == 0x11u);
    }
    SECTION("a release a window made fenced") {
        Fixture f;
        Access::SetWindowOpen(*f.split, true);
        release(*f.split, 0x12, Split::Landing::AtRun);
        CHECK(f.split->WindowBase(Word, target(*f.split), false) == 0x12u);
    }
    SECTION("a value the front can't tell, or only at least") {
        Fixture f;
        Access::Send(*f.split, 4);
        Access::Write(*f.split, Word, 4);
        f.split->NoteSemaphoreWrite(Word, 0, false, Split::Landing::Fenced); // A counter
        Access::SetWindowOpen(*f.split, true);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
        Access::Send(*f.split, 4);
        Access::Write(*f.split, Word, 4);
        f.split->NoteSemaphoreWrite(Word, 9, true, true, Split::SemaphoreWriter::QmdReduction);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
    }
    SECTION("written over after it, next to it") {
        Fixture f;
        release(*f.split, 0x13, Split::Landing::Fenced);
        Access::SetWindowOpen(*f.split, true);
        // The word beside it, in the same granule: the exact log tells them apart.
        Access::Send(*f.split, 4);
        Access::Write(*f.split, Word + 4, 4);
        CHECK(f.split->WindowBase(Word, target(*f.split), false) == 0x13u);
        // Over it.
        Access::Send(*f.split, 4);
        Access::Write(*f.split, Word, 4);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
    }
    SECTION("a QMD signal the front did not value, sent after it") {
        Fixture f;
        release(*f.split, 0x14, Split::Landing::Fenced);
        Access::SetWindowOpen(*f.split, true);
        Access::Send(*f.split, 4);
        f.split->NoteUnvaluedSignal(false);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
    }
    SECTION("only a close pending, a fenced release before it pending: it may land before that") {
        Fixture f;
        release(*f.split, 0x15, Split::Landing::Fenced);
        Access::SetClosePending(*f.split, true);
        release(*f.split, 0x16, Split::Landing::AtRun); // Fenced by the close, maybe at run
        const auto noted = f.split->QueuedSemaphoreWriteAt(Word);
        REQUIRE(noted.has_value());
        CHECK((noted->windowed && noted->known && !noted->ordered));
        Access::SetClosePending(*f.split, false);
        Access::SetWindowOpen(*f.split, true);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
        // The same once the earlier one landed: either way it lands last.
        Fixture g;
        release(*g.split, 0x15, Split::Landing::Fenced);
        Access::Land(*g.split, g.split->Position());
        Access::SetClosePending(*g.split, true);
        release(*g.split, 0x16, Split::Landing::AtRun);
        const auto ordered = g.split->QueuedSemaphoreWriteAt(Word);
        REQUIRE(ordered.has_value());
        CHECK((ordered->windowed && ordered->ordered));
        Access::SetClosePending(*g.split, false);
        Access::SetWindowOpen(*g.split, true);
        CHECK(g.split->WindowBase(Word, target(*g.split), false) == 0x16u);
    }
    SECTION("a release landing at run behind a fenced one: either value") {
        Fixture f;
        release(*f.split, 0x17, Split::Landing::Fenced);
        release(*f.split, 0x18, Split::Landing::AtRun);
        const auto raced = f.split->QueuedSemaphoreWriteAt(Word);
        REQUIRE(raced.has_value());
        CHECK((raced->raced && !raced->known && !raced->ordered));
        Access::SetWindowOpen(*f.split, true);
        CHECK_FALSE(f.split->WindowBase(Word, target(*f.split), false).has_value());
    }
    SECTION("a window open: what it notes is ordered") {
        Fixture f;
        release(*f.split, 0x19, Split::Landing::Fenced);
        Access::SetWindowOpen(*f.split, true);
        release(*f.split, 0x1a, Split::Landing::AtRun); // Converted: after the one before
        const auto noted = f.split->QueuedSemaphoreWriteAt(Word);
        REQUIRE(noted.has_value());
        CHECK((noted->windowed && noted->ordered));
        CHECK(f.split->WindowBase(Word, target(*f.split), false) == 0x1au);
    }
}

TEST_CASE("Run-ahead window base model: a QMD reduction lands with the base the front noted",
          "[video_core][runahead]") {
    // The front sends releases landing at run, fence-flagged ones, uploads over the word, QMD
    // signals it can't value, and QMD reductions (an add) it values as FrontQmdView::WordAt does:
    // from a release landing at run, from the window's order (WindowBase), or from memory. Landing
    // windows open and close as acquires go on before fenced releases landed. The back converts
    // what lands in a window, the fencing thread lands fence operations in order. Whenever the
    // front valued a reduction, the word holds that base as the reduction lands.
    std::mt19937 random{0x5b45};
    u64 window_bases = 0;
    u64 other_bases = 0;
    u64 landed_checked = 0;
    u64 converted_reductions = 0;
    WindowBaseScope base_on;
    constexpr u64 W = Fixture::Word + 0x40;
    for (int round = 0; round < 400; ++round) {
        Fixture f;
        LandingsScope on;
        Access::StartRing(*f.split);
        enum class Kind { AtRun, Fenced, Reduction, Upload, Unvalued, Protocol };
        struct Sent {
            u64 position;
            Kind kind;
            u32 value; ///< The payload, the add, or what an upload or unvalued signal writes
            std::optional<u32> base; ///< A reduction's, as the front noted it
            bool open;               ///< A reduction noted with a window open
        };
        std::vector<Sent> sent;
        size_t next_run = 0;
        struct Op {
            u64 position;
            Kind kind;
            u32 value;
            std::optional<u32> base;
            std::shared_ptr<void> token;
        };
        std::deque<Op> fence_ops;
        u32 memory = 0;
        bool synced = true; // The QMD view values signals (else until a takeover)
        const auto land = [&](Kind kind, u32 value, const std::optional<u32>& base) {
            if (kind == Kind::Reduction) {
                if (base) {
                    CHECK(memory == *base);
                    ++landed_checked;
                }
                memory += value;
            } else {
                memory = value;
            }
        };
        const auto fence_step = [&] {
            if (fence_ops.empty()) {
                return;
            }
            Op op = std::move(fence_ops.front());
            fence_ops.pop_front();
            land(op.kind, op.value, op.base);
            op.token.reset();
            Access::CompleteFenced(*f.split, op.position);
        };
        const auto front_step = [&] {
            if (!synced && f.split->Ran(f.split->Position())) {
                synced = true; // Taken over once the back ran everything sent
            }
            const int choice = static_cast<int>(random() % 16);
            const u32 value = static_cast<u32>(random() % 1000);
            if (choice < 3) {
                Access::Send(*f.split, 2);
                Access::Write(*f.split, W, 4);
                f.split->NoteSemaphoreWrite(W, value, true, Split::Landing::AtRun);
                sent.push_back({Access::LastEntry(*f.split), Kind::AtRun, value, {}, false});
            } else if (choice < 6) {
                Access::Send(*f.split, 2);
                Access::Write(*f.split, W, 4);
                f.split->NoteSemaphoreWrite(W, value, true, Split::Landing::Fenced);
                sent.push_back({Access::LastEntry(*f.split), Kind::Fenced, value, {}, false});
            } else if (choice < 10) {
                if (!synced) {
                    return;
                }
                Access::Send(*f.split, 2);
                const u64 target = Access::GranuleTarget(*f.split, W, 4);
                std::optional<u32> base;
                bool from_window = false;
                if (const auto queued = f.split->QueuedSemaphoreWriteAt(W)) {
                    if (queued->known && !queued->at_least && !queued->fenced &&
                        queued->position >= target) {
                        base = queued->payload;
                    } else {
                        base = f.split->WindowBase(W, target, false);
                        from_window = base.has_value();
                    }
                } else if (f.split->MemoryHoldsBase(W, target, false)) {
                    base = memory;
                }
                const u32 add = 1 + value % 7;
                Access::Write(*f.split, W, 4);
                f.split->NoteSemaphoreWrite(W, base ? *base + add : 0, base.has_value(),
                                            Split::Landing::AtRun, false,
                                            Split::SemaphoreWriter::QmdReduction, from_window);
                sent.push_back({Access::LastEntry(*f.split), Kind::Reduction, add, base,
                                f.split->LandingWindowOpen()});
                window_bases += from_window ? 1 : 0;
                other_bases += base && !from_window ? 1 : 0;
            } else if (choice < 11) {
                Access::Send(*f.split, 2);
                Access::Write(*f.split, W, 4);
                sent.push_back({Access::LastEntry(*f.split), Kind::Upload, value, {}, false});
            } else if (choice < 12) {
                Access::Send(*f.split, 2);
                f.split->NoteUnvaluedSignal(false);
                synced = false;
                sent.push_back({Access::LastEntry(*f.split), Kind::Unvalued, value, {}, false});
            } else if (choice < 14) {
                if (const auto entry = f.split->UnlandedFencedEntry(W)) {
                    const u64 before = Access::WritePosition(*f.split);
                    f.split->OpenLandingWindow(*entry, 0, nullptr);
                    if (Access::WritePosition(*f.split) != before) {
                        sent.push_back({before, Kind::Protocol, 0, {}, false});
                    }
                }
            } else {
                const u64 before = Access::WritePosition(*f.split);
                f.split->CheckLandingWindow(nullptr);
                if (Access::WritePosition(*f.split) != before) {
                    sent.push_back({before, Kind::Protocol, 0, {}, false});
                }
            }
        };
        const auto back_step = [&] {
            if (next_run == sent.size()) {
                return;
            }
            const Sent entry = sent[next_run++];
            Access::ActAsBack(true);
            if (entry.kind == Kind::Protocol) {
                REQUIRE(Access::RunEntry(*f.split, entry.position));
            } else if (entry.kind == Kind::Fenced) {
                Access::PendFenced(*f.split, entry.position);
                fence_ops.push_back({entry.position, Kind::Fenced, entry.value, {}, {}});
            } else if (entry.kind == Kind::Upload) {
                if (f.split->LandingPendingIn(W, 4)) {
                    // Round 4: the upload waits for the landings pending there
                    // (WaitBeforeOverwrite).
                    while (!fence_ops.empty()) {
                        fence_step();
                    }
                }
                land(Kind::Upload, entry.value, {});
            } else {
                if (entry.kind == Kind::Reduction && entry.open) {
                    CHECK(f.split->ConvertsLandings()); // Sent in a window: a fence operation
                }
                if (f.split->ConvertsLandings()) {
                    converted_reductions += entry.kind == Kind::Reduction ? 1 : 0;
                    Access::PendFenced(*f.split, entry.position);
                    fence_ops.push_back({entry.position, entry.kind, entry.value, entry.base,
                                         f.split->ConvertedLanding(W, false)});
                } else {
                    land(entry.kind, entry.value, entry.base);
                }
            }
            const u64 end =
                next_run < sent.size() ? sent[next_run].position : Access::WritePosition(*f.split);
            Access::Seal(*f.split, end);
            Access::Consume(*f.split, end);
            Access::Boundary(*f.split, end);
            Access::ActAsBack(false);
        };
        for (int step = 0; step < 400; ++step) {
            const int who = static_cast<int>(random() % 3);
            if (who == 0) {
                front_step();
            } else if (who == 1) {
                back_step();
            } else {
                fence_step();
            }
        }
        while (next_run < sent.size() || !fence_ops.empty()) {
            back_step();
            fence_step();
        }
        Access::StopRing(*f.split);
    }
    CHECK(window_bases > 100);
    CHECK(other_bases > 100);
    CHECK(converted_reductions > 100);
    CHECK(landed_checked > 300);
}

TEST_CASE("Run-ahead landings: a QMD release made a fence operation is compared as it lands",
          "[video_core][runahead]") {
    namespace Check = VideoCommon::Nx2DrainCheck;
    auto& state = Check::Get();
    const auto landed = [&state] { return state.qmd_landed.load(); };
    const auto wrong = [&state] { return state.qmd_landed_wrong.load(); };
    const u64 landed_before = landed();
    const u64 wrong_before = wrong();
    // Landed first, then the note: compared then.
    Check::LandingRelease first;
    Check::QmdReleaseLanded(first, 0x10, 5);
    CHECK(landed() == landed_before);
    Check::ExpectQmdRelease(first, {0x10, 5, true, false, nullptr});
    CHECK(landed() == landed_before + 1);
    CHECK(wrong() == wrong_before);
    // The note first; it landed otherwise.
    Check::LandingRelease second;
    Check::ExpectQmdRelease(second, {0x10, 6, true, false, nullptr});
    Check::QmdReleaseLanded(second, 0x10, 7);
    CHECK(wrong() == wrong_before + 1);
    // At least, and a value the front could not tell.
    Check::LandingRelease at_least;
    Check::ExpectQmdRelease(at_least, {0x10, 6, true, true, nullptr});
    Check::QmdReleaseLanded(at_least, 0x10, 7);
    Check::LandingRelease unknown;
    Check::ExpectQmdRelease(unknown, {0x10, 0, false, false, nullptr});
    Check::QmdReleaseLanded(unknown, 0x10, 7);
    CHECK(wrong() == wrong_before + 1);
    CHECK(landed() == landed_before + 4);
    // The back's check after the method hands the notes over in order; the others it compares.
    Check::BackQmdReleases().clear();
    Check::BackQmdReads().clear();
    auto fenced = Check::NoteQmdReleaseLanding(0x20, true);
    REQUIRE(fenced);
    Check::NoteQmdRelease(0x24, 3, true);
    CHECK_FALSE(Check::NoteQmdReleaseLanding(0x28, false)); // Off the back: not recorded
    const u64 methods_wrong = state.qmd_wrong.load();
    const std::vector<Check::QmdRelease> predicted{{0x20, 9, true, false, nullptr},
                                                   {0x24, 3, true, false, nullptr}};
    Check::CheckQmdReleases(predicted, {}, true);
    CHECK(state.qmd_wrong.load() == methods_wrong);
    Check::QmdReleaseLanded(*fenced, 0x20, 9);
    CHECK(landed() == landed_before + 5);
    CHECK(wrong() == wrong_before + 1);
}

TEST_CASE("Run-ahead landings: a read after a drain waits for the word's landings, bounded",
          "[video_core][runahead]") {
    // ReadSemaphoreWord: the baseline's read after a drain finds every landing sent before. A
    // converted one to the word still pending after the bound decides nothing (not met).
    Fixture f;
    LandingsScope on;
    constexpr u64 Word = Fixture::Word + 0xa0;
    CHECK(f.split->WaitForLandingsAt(Word, std::chrono::milliseconds{1}));
    auto landing = f.split->ConvertedLanding(Word, false);
    CHECK_FALSE(f.split->WaitForLandingsAt(Word, std::chrono::milliseconds{2}));
    CHECK(f.split->WaitForLandingsAt(Word + 4, std::chrono::milliseconds{1})); // Another word
    std::thread fence{[&landing] {
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
        landing.reset(); // It lands
    }};
    CHECK(f.split->WaitForLandingsAt(Word, std::chrono::seconds{10}));
    fence.join();
}
