// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
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

} // namespace
