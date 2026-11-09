// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "video_core/native_qmd.h"
#include "video_core/nx2.h"

namespace VideoCommon {

inline bool g_qmd_view = [] {
    const char* value = std::getenv("EDEN_NX2_QMD_VIEW");
    return value == nullptr || value[0] != '0';
}();

inline void ReadQmdViewSwitch() {
    const char* value = std::getenv("EDEN_NX2_QMD_VIEW");
    g_qmd_view = value == nullptr || value[0] != '0';
}

/// The QMD scheduling of every channel (Tegra::NativeQmdScheduler), run on the split GPU thread's
/// front as its back will run it, on a table of its own: what the launches release is known when
/// they are sent, for the front's view of the semaphore words (GpuSplit::NoteSemaphoreWrite).
///
/// The table is the back's from the start: an engine's front takes it over with the back drained
/// when it takes the engine's macros over (TakeOver). It stays the back's as long as the front
/// reads the descriptors the back will read. The front reads one when it sends a signal, the back
/// when it runs it, both from guest memory without the GPU caches. The front takes a descriptor:
/// - from the signal's own inline QMD, which the back writes right before;
/// - from memory, once the back ran every write sent there (GpuSplit::MemoryWritesTarget: the
///   writes placed in its granule and the unplaced ones), on pages GPU work never wrote
///   (Nx2MayBeGpuWritten);
/// - from the inline QMD a LOAD_INLINE_QMD_DATA still queued writes there, when that is the last
///   write sent there (NoteInlineWrite).
/// No change of the GPU's mappings may be queued either (GpuSplit::LastMappingChange). Memory holds
/// what the back finds there only while the back has not run the signal's entry yet: the front
/// sends and values it under a GpuSplit::PublishHold.
///
/// Where the back has yet to run a write the front can't read past (into a QMD the signal reads,
/// or anywhere, or to the mappings), the signal waits, with what it changed in the table taken
/// back (NativeQmdScheduler::BeginAttempt), and every signal sent after it waits behind it: the
/// table stays the back's as of the signals valued. The front values the waiting ones in order
/// once the back ran what the first waits for (CatchUp). Late, the back may have run the signal
/// and what followed it, so memory holds what the back read only where nothing sent after the
/// signal writes the QMD, and no release of the waiting signals writes a QMD read late. Their
/// releases are placed at their signals' positions, not noted. Where the front can't read what
/// the back read at all (a QMD on a GPU-written page, written after the signal, released into,
/// or the mappings changed after it), it loses the table: it stops valuing, and takes the back's
/// table over the next time the back has run everything sent, between commands or at a QMD
/// method (MaybeTakeOver).
///
/// `Memory`, the channel's GPU address space as the back reads it, has:
/// - bool IsMapped(GPUVAddr, u64), void Read(GPUVAddr, void*, u64), u32 ReadWord(GPUVAddr) (as
///   MemoryManager::Read<u32>), bool InSyncpointShim(GPUVAddr, u64);
/// - std::optional<u64> Device(GPUVAddr), bool GpuWritten(u64 device, u64 size);
/// - u64 WritesTarget(const GpuSplit&, GPUVAddr, u64), void NoteWrite(GpuSplit&, GPUVAddr, u64),
///   void NoteWriteAt(GpuSplit&, GPUVAddr, u64 size, u64 position);
/// - const void* Id(): the address space's, the same for every Memory of it.
class FrontQmdView {
public:
    enum class ReadFailure {
        NotSynced,
        Unplaced,
        GpuWritten,
        Pending,
        ReleasedInto,
        MappedAfter,
        Mappings,
        TooMany,
        WrittenAfter,
        LateReleasedInto
    };
    enum class TakeOverSite { FrontStart, BetweenCommands, QmdMethod };
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;

    /// The view the compute engines of every channel feed: they share the back's table.
    static FrontQmdView& Get() {
        static FrontQmdView view;
        return view;
    }

    /// The back ran everything sent and runs nothing (the split drained, or it waits for work):
    /// takes over the table `back` shares with the other engines. `where`: TEST.
    void TakeOver(const Tegra::NativeQmdScheduler& back,
                  TakeOverSite where = TakeOverSite::QmdMethod) {
        scheduler.CopyStates(back);
        synced = true;
        ClearWaiting();
    }

    /// The front, between commands and before QMD methods: takes the back's table over if the
    /// view lost it or signals wait, and the back has run everything sent.
    void MaybeTakeOver(const GpuSplit& split, const Tegra::NativeQmdScheduler& back,
                       TakeOverSite where = TakeOverSite::QmdMethod) {
        if (!Current() && split.Ran(split.Position())) {
            TakeOver(back, where);
        }
    }
    void MaybeTakeOver(const GpuSplit& split, TakeOverSite where = TakeOverSite::QmdMethod) {
        MaybeTakeOver(split, back_table, where);
    }

    /// The engines schedule QMDs the front does not see (the split stopped).
    void Lose() {
        synced = false;
        inline_writes_count = 0;
        ClearWaiting();
    }

    [[nodiscard]] bool Synced() const {
        return synced;
    }
    /// Whether the table is the back's as of everything sent: synced, and no signal waits.
    [[nodiscard]] bool Current() const {
        return synced && waiting.empty();
    }

    /// LOAD_INLINE_QMD_DATA was just sent: running it, the back writes `qmd` to the QMD at
    /// `device`. Until then the front reads it from here (ReadDescriptor).
    void NoteInlineWrite(const GpuSplit& split, u64 device, const Descriptor& qmd) {
        InlineWrite& written = inline_writes[inline_writes_count % inline_writes.size()];
        ++inline_writes_count;
        written.device = device;
        written.position = split.Position();
        written.qmd = qmd;
        written.overwritten = false;
    }

    /// The scheduling of PCAS action `argument` on the QMD at `address` that the back runs at the
    /// entry just sent (KeplerCompute::ProcessNativeMethod); `inline_qmd`: the QMD it writes there
    /// right before. Notes what the launches it runs release, or has the signal wait. The entry
    /// must still be unpublished (GpuSplit::PublishHold): the back must not have run it.
    template <typename Memory>
    void Signal(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 argument,
                const Descriptor* inline_qmd) {
        if (!g_qmd_view) {
            Lose();
            split.NoteUnplacedWrite();
            return;
        }
        split.NoteCacheWrite(); // The dispatches' shader stores
        using Lost = ReadFailure;
        const u64 position = split.Position();
        if (synced) {
            CatchUp(split, memory);
        }
        if (!synced) {
            Unvalued(split, Lost::NotSynced);
            return;
        }
        if (!waiting.empty()) {
            // Behind a signal still waiting.
            Wait(split, position, memory.Id(), address, argument, inline_qmd, std::nullopt);
            return;
        }
        const Outcome outcome =
            Attempt(split, memory, position, address, argument, inline_qmd, false);
        if (outcome.valued) {
            return;
        }
        if (outcome.retry_after) {
            retry_after = *outcome.retry_after;
            Wait(split, position, memory.Id(), address, argument, inline_qmd, outcome.lost);
            return;
        }
        synced = false;
        Unvalued(split, outcome.lost);
    }

private:
    struct Release {
        GPUVAddr address;
        u32 payload;
        u32 flags;
    };

    /// An inline QMD a LOAD_INLINE_QMD_DATA sent writes (NoteInlineWrite).
    struct InlineWrite {
        u64 device{};
        u64 position{};
        Descriptor qmd{};
        bool overwritten{}; ///< A release of the same entry writes into it after
    };

    /// A signal sent at `position` that waits to be valued (`inline_qmd` if `has_inline`).
    struct WaitingSignal {
        u64 position;
        const void* memory; ///< Memory::Id of the address space it reads
        GPUVAddr address;
        u32 argument;
        bool has_inline;
        Descriptor inline_qmd;
    };

    /// How an attempt to value a signal ended: valued, or taken back, to try again once the back
    /// ran `retry_after`, or for good (the view loses its table).
    struct Outcome {
        bool valued;
        std::optional<u64> retry_after;
        ReadFailure lost;
        u64 qmd{}; ///< TEST: the QMD it could not read
    };

    /// Signals that wait, at most: a back that far behind leaves the view to its next takeover.
    static constexpr size_t MaxWaiting = 4096;

    /// A signal the front can't value, its table lost: no release of it is noted, and the view
    /// takes the back's table over once the back ran it.
    void Unvalued(const GpuSplit& split, ReadFailure lost) {}

    /// The signal sent at `position` waits, first (`why`: what it waits for) or behind the others.
    /// None of its releases is noted: they are placed once it is valued.
    void Wait(const GpuSplit& split, u64 position, const void* memory_id, GPUVAddr address,
              u32 argument, const Descriptor* inline_qmd, std::optional<ReadFailure> why) {
        if (waiting.size() >= MaxWaiting) {
            synced = false;
            ClearWaiting();
            return;
        }
        WaitingSignal& signal = waiting.emplace_back();
        signal.position = position;
        signal.memory = memory_id;
        signal.address = address;
        signal.argument = argument;
        signal.has_inline = inline_qmd != nullptr;
        if (inline_qmd != nullptr) {
            signal.inline_qmd = *inline_qmd;
        }
    }

    void ClearWaiting() {
        waiting.clear();
        late_reads.clear();
        retry_after = 0;
    }

    /// Values the waiting signals that read `memory` in order, from the first, as long as the back
    /// ran what the first waits for.
    template <typename Memory>
    void CatchUp(GpuSplit& split, const Memory& memory) {
        while (!waiting.empty()) {
            const WaitingSignal& first = waiting.front();
            if (first.memory != memory.Id() || !split.Ran(retry_after)) {
                return;
            }
            const Outcome outcome =
                Attempt(split, memory, first.position, first.address, first.argument,
                        first.has_inline ? &first.inline_qmd : nullptr, true);
            if (outcome.retry_after) {
                retry_after = *outcome.retry_after;
                return;
            }
            if (!outcome.valued) {
                synced = false;
                ClearWaiting();
                return;
            }
            waiting.pop_front();
            retry_after = 0;
        }
        // Every signal sent before the reads made late is valued: none of their releases waits.
        late_reads.clear();
    }

    /// Runs the signal sent at `position` on the table as the back runs it, and notes what its
    /// launches release, or places it if `late`. Takes it back where the front can't read a QMD
    /// as the back reads it.
    template <typename Memory>
    Outcome Attempt(GpuSplit& split, const Memory& memory, u64 position, GPUVAddr address,
                    u32 argument, const Descriptor* inline_qmd, bool late) {
        using Lost = ReadFailure;
        // The back finds the mappings every change sent before the signal made, and none after it.
        const u64 mapping_change = split.LastMappingChange();
        if (mapping_change > position) {
            return {false, std::nullopt, Lost::MappedAfter};
        }
        if (!split.Ran(mapping_change)) {
            return {false, mapping_change, Lost::Mappings};
        }
        releases.clear();
        attempt_reads.clear();
        std::optional<Outcome> failure;
        const auto read = [&](u64 qmd_address, Descriptor& descriptor) {
            if (!memory.IsMapped(qmd_address, sizeof(descriptor))) {
                return false;
            }
            if (inline_qmd != nullptr && qmd_address == address) {
                descriptor = *inline_qmd;
            } else if (const auto unread =
                           ReadDescriptor(split, memory, qmd_address, position, descriptor)) {
                failure = unread;
                return false; // Stops the scheduling, which is taken back
            }
            // A release this signal ran before writes into the QMD.
            for (const Release& earlier : releases) {
                if (earlier.address + 16 > qmd_address &&
                    earlier.address < qmd_address + sizeof(descriptor)) {
                    failure = Outcome{false, std::nullopt, Lost::ReleasedInto};
                    return false;
                }
            }
            attempt_reads.push_back(qmd_address);
            return true;
        };
        const auto release = [&](u64 destination, u32 payload, u32 flags) {
            const bool four_words = (flags >> 30) == 0;
            if (!memory.IsMapped(destination, four_words ? 16 : sizeof(u32))) {
                return false;
            }
            releases.push_back({destination, payload, flags});
            return true;
        };
        scheduler.BeginAttempt();
        scheduler.Signal(address, argument, read, release, [](const Descriptor&) { return true; });
        if (!failure && late && ReleasesIntoLateReads()) {
            failure = Outcome{false, std::nullopt, Lost::LateReleasedInto};
        }
        if (failure) {
            scheduler.RollbackAttempt();
            return *failure;
        }
        scheduler.CommitAttempt();
        if (late) {
            for (const u64 read_address : attempt_reads) {
                late_reads.insert(read_address >> 8);
            }
            for (const Release& done : releases) {
                memory.NoteWriteAt(split, done.address, (done.flags >> 30) == 0 ? 16 : 4, position);
            }
        } else {
            for (const Release& noted : releases) {
                NoteRelease(split, memory, noted.address, noted.payload, noted.flags);
            }
        }
        return {true, std::nullopt, Lost::NotSynced};
    }

    /// Late: whether a release of the signal writes a QMD it read, or one a signal valued late
    /// before read. The back may have run the release before the front read that QMD.
    [[nodiscard]] bool ReleasesIntoLateReads() const {
        for (const Release& done : releases) {
            const u64 first = done.address >> 8;
            const u64 last = (done.address + 15) >> 8;
            for (const u64 read_address : attempt_reads) {
                if ((read_address >> 8) == first || (read_address >> 8) == last) {
                    return true;
                }
            }
            if (late_reads.contains(first) || late_reads.contains(last)) {
                return true;
            }
        }
        return false;
    }

    /// Reads into `descriptor` the QMD at `qmd_address` as the back read it, or will, running the
    /// signal sent at `position`, or tells why the front can't.
    template <typename Memory>
    [[nodiscard]] std::optional<Outcome> ReadDescriptor(const GpuSplit& split, const Memory& memory,
                                                        GPUVAddr qmd_address, u64 position,
                                                        Descriptor& descriptor) const {
        using Lost = ReadFailure;
        const auto device = memory.Device(qmd_address);
        if (!device || memory.GpuWritten(*device, sizeof(descriptor))) {
            return Outcome{false, std::nullopt, Lost::GpuWritten};
        }
        // The last write sent there, placed or not (the inline write covers the whole QMD).
        const u64 target = memory.WritesTarget(split, qmd_address, sizeof(descriptor));
        const Lost lost = target == split.LastUnplacedWrite() ? Lost::Unplaced : Lost::Pending;
        if (target > position) {
            // Sent after the signal (it waited): memory may hold it, the back read what was before.
            return Outcome{false, std::nullopt, Lost::WrittenAfter, qmd_address};
        }
        if (split.Ran(target)) {
            // Read only once Ran saw the back past the write: the back stores its position after
            // the writes before it. A read before that look may be older than the write. The back
            // can run a write published shortly before the signal (to an idle back, after a few
            // words) while the front reads, and store its position before the front looks.
            memory.Read(qmd_address, descriptor.data(), sizeof(descriptor));
            return std::nullopt;
        }
        if (const InlineWrite* written = QueuedInlineWrite(split, *device);
            written != nullptr && written->position == target) {
            descriptor = written->qmd;
            return std::nullopt;
        }
        return Outcome{false, target, lost, qmd_address};
    }

    /// The last inline QMD sent to the QMD at `device` that the back has not written yet.
    [[nodiscard]] const InlineWrite* QueuedInlineWrite(const GpuSplit& split, u64 device) const {
        const size_t count = (std::min)(inline_writes_count, inline_writes.size());
        for (size_t i = 1; i <= count; ++i) {
            const InlineWrite& written =
                inline_writes[(inline_writes_count - i) % inline_writes.size()];
            if (split.Ran(written.position)) {
                break; // The back wrote it, and the older ones
            }
            if (written.device == device) {
                return written.overwritten ? nullptr : &written;
            }
        }
        return nullptr;
    }

    /// A release the back runs with the entry just sent, as ProcessNativeMethod's: what the word
    /// holds once the back ran it.
    template <typename Memory>
    void NoteRelease(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 payload,
                     u32 flags) {
        const bool four_words = (flags >> 30) == 0;
        const bool reduction = (flags & (1U << 26)) != 0;
        const bool maximum = !reduction && !four_words && (flags & (1U << 19)) != 0;
        const u64 size = four_words ? 16 : sizeof(u32);
        const auto device = memory.Device(address);
        // The word a reduction or a maximum combines the payload with, before this write.
        std::optional<u32> base;
        if (device && (reduction || maximum)) {
            base = WordAt(split, memory, address, *device);
        }
        memory.NoteWrite(split, address, size);
        if (!device) {
            return;
        }
        // An inline QMD this entry wrote before is no longer what memory holds there.
        for (size_t i = 1; i <= (std::min)(inline_writes_count, inline_writes.size()); ++i) {
            InlineWrite& written = inline_writes[(inline_writes_count - i) % inline_writes.size()];
            if (written.position != split.Position()) {
                break;
            }
            if (written.device < *device + size && *device < written.device + sizeof(Descriptor)) {
                written.overwritten = true;
            }
        }
        if (reduction) {
            const bool is_signed = ((flags >> 24) & 1) != 0;
            const u32 value =
                base ? Tegra::ReduceSemaphore((flags >> 20) & 7, is_signed, *base, payload) : 0;
            split.NoteSemaphoreWrite(*device, value, base.has_value());
        } else if (four_words) {
            // The payload as 64 bits, a timestamp after it.
            split.NoteSemaphoreWrite(*device, payload, true, GpuSplit::Landing::Fenced);
            split.NoteSemaphoreWrite(*device + 4, 0, true, GpuSplit::Landing::Fenced);
        } else if (maximum) {
            // Whatever the word held, it holds the payload at least.
            const u32 value = base ? (std::max)(*base, payload) : payload;
            split.NoteSemaphoreWrite(*device, value, true, !base.has_value());
        } else {
            split.NoteSemaphoreWrite(*device, payload, true, GpuSplit::Landing::Fenced);
        }
    }

    /// The word the back finds at `address` (`device`) when it runs the entry just sent, where
    /// the front knows it: the last release sent there, if nothing sent since may have written
    /// over it, else memory, once the back ran every write sent there before the entry. The entry
    /// itself is held unpublished (PublishHold): an idle back that ran it before this read would
    /// leave its own increment in memory, and the front would count it twice.
    template <typename Memory>
    [[nodiscard]] std::optional<u32> WordAt(const GpuSplit& split, const Memory& memory,
                                            GPUVAddr address, u64 device) const {
        // The back reads the syncpoint shim from the syncpoints.
        if (memory.InSyncpointShim(address, sizeof(u32)) ||
            Tegra::Nx2MirrorShadow::Get().Diverts(device)) {
            return std::nullopt;
        }
        // The last write sent to the word's granule, placed or not.
        const u64 target = memory.WritesTarget(split, address, sizeof(u32));
        if (const auto queued = split.QueuedSemaphoreWriteAt(device)) {
            if (queued->known && !queued->at_least && !queued->fenced &&
                queued->position >= target) {
                return queued->payload;
            }
            return std::nullopt;
        }
        if (memory.GpuWritten(device, sizeof(u32)) || !split.Ran(target)) {
            return std::nullopt;
        }
        return memory.ReadWord(address);
    }

    Tegra::NativeQmdScheduler scheduler{Tegra::NativeQmdScheduler::OwnTable{}};
    /// The table the engines of every channel share, which the back's scheduling uses.
    Tegra::NativeQmdScheduler back_table;
    std::vector<Release> releases;
    std::vector<u64> attempt_reads; ///< The QMDs the attempt read
    bool synced{};
    /// Signals that wait to be valued, oldest first; the first may be once the back ran
    /// retry_after. late_reads: the granules of the QMDs read late since the waiting began.
    std::deque<WaitingSignal> waiting;
    u64 retry_after{};
    std::unordered_set<u64> late_reads;
    /// The last inline QMDs sent, oldest first from inline_writes_count on: a back a millisecond
    /// behind has some ten to write, a few times that in long frames.
    std::array<InlineWrite, 256> inline_writes{};
    size_t inline_writes_count{};
};

} // namespace VideoCommon
