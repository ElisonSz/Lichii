// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "video_core/native_qmd.h"
#include "video_core/nx2.h"

namespace VideoCommon {

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
/// No change of the GPU's mappings may be queued either (GpuSplit::LastMappingChange). Where the
/// front can't tell what the back reads, its table may no longer be the back's: it stops valuing,
/// the launches count as writing guest memory anywhere, and it takes the back's table over again
/// the next time the back has run everything sent, between commands or at a QMD method
/// (MaybeTakeOver).
///
/// `Memory`, the channel's GPU address space as the back reads it, has:
/// - bool IsMapped(GPUVAddr, u64), void Read(GPUVAddr, void*, u64), u32 ReadWord(GPUVAddr) (as
///   MemoryManager::Read<u32>), bool InSyncpointShim(GPUVAddr, u64);
/// - std::optional<u64> Device(GPUVAddr), bool GpuWritten(u64 device, u64 size);
/// - u64 WritesTarget(const GpuSplit&, GPUVAddr, u64), void NoteWrite(GpuSplit&, GPUVAddr, u64).
class FrontQmdView {
public:
    enum class ReadFailure { NotSynced, Unplaced, GpuWritten, Pending, ReleasedInto };
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;

    /// The view the compute engines of every channel feed: they share the back's table.
    static FrontQmdView& Get() {
        static FrontQmdView view;
        return view;
    }

    /// The back ran everything sent and runs nothing (the split drained, or it waits for work):
    /// takes over the table `back` shares with the other engines.
    void TakeOver(const Tegra::NativeQmdScheduler& back) {
        scheduler.CopyStates(back);
        synced = true;
    }

    /// The front, between commands and before QMD methods: takes the back's table over if the
    /// view lost it and the back has run everything sent.
    void MaybeTakeOver(const GpuSplit& split, const Tegra::NativeQmdScheduler& back) {
        if (!synced && split.Ran(split.Position())) {
            TakeOver(back);
        }
    }
    void MaybeTakeOver(const GpuSplit& split) {
        MaybeTakeOver(split, back_table);
    }

    /// The engines schedule QMDs the front does not see (the split stopped).
    void Lose() {
        synced = false;
        inline_writes_count = 0;
    }

    [[nodiscard]] bool Synced() const {
        return synced;
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
    /// right before. Notes what the launches it runs release.
    template <typename Memory>
    void Signal(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 argument,
                const Descriptor* inline_qmd) {
        split.NoteCacheWrite(); // The dispatches' shader stores
        using Lost = ReadFailure;
        if (!synced) {
            Unvalued(split, Lost::NotSynced);
            return;
        }
        // A change of the mappings still queued may unmap what the back reads or releases to.
        if (!split.Ran(split.LastMappingChange())) {
            synced = false;
            Unvalued(split, Lost::Unplaced);
            return;
        }
        releases.clear();
        std::optional<Lost> lost;
        const auto read = [&](u64 qmd_address, Descriptor& descriptor) {
            if (!memory.IsMapped(qmd_address, sizeof(descriptor))) {
                return false;
            }
            if (inline_qmd != nullptr && qmd_address == address) {
                descriptor = *inline_qmd;
            } else if (const auto unknown = ReadDescriptor(split, memory, qmd_address, descriptor);
                       unknown && !lost) {
                lost = unknown;
            }
            // A release this signal ran before writes into the QMD.
            for (const Release& earlier : releases) {
                if (!lost && earlier.address + 16 > qmd_address &&
                    earlier.address < qmd_address + sizeof(descriptor)) {
                    lost = Lost::ReleasedInto;
                }
            }
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
        scheduler.Signal(address, argument, read, release, [](const Descriptor&) { return true; });
        if (lost) {
            synced = false;
            Unvalued(split, *lost);
            return;
        }
        for (const Release& noted : releases) {
            NoteRelease(split, memory, noted.address, noted.payload, noted.flags);
        }
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

    /// A signal the front can't value: its releases may write guest memory anywhere (they change
    /// no mappings).
    void Unvalued(GpuSplit& split, ReadFailure lost) {
        split.NoteUnplacedWrite();
    }

    /// Reads into `descriptor` the QMD at `qmd_address` as the back will read it running the entry
    /// just sent, or tells why the front can't.
    template <typename Memory>
    [[nodiscard]] std::optional<ReadFailure> ReadDescriptor(const GpuSplit& split,
                                                            const Memory& memory,
                                                            GPUVAddr qmd_address,
                                                            Descriptor& descriptor) const {
        using Lost = ReadFailure;
        memory.Read(qmd_address, descriptor.data(), sizeof(descriptor));
        const auto device = memory.Device(qmd_address);
        if (!device || memory.GpuWritten(*device, sizeof(descriptor))) {
            return Lost::GpuWritten;
        }
        // The last write sent there, placed or not (the inline write covers the whole QMD).
        const u64 target = memory.WritesTarget(split, qmd_address, sizeof(descriptor));
        if (split.Ran(target)) {
            return std::nullopt;
        }
        if (const InlineWrite* written = QueuedInlineWrite(split, *device);
            written != nullptr && written->position == target) {
            descriptor = written->qmd;
            return std::nullopt;
        }
        return target == split.LastUnplacedWrite() ? Lost::Unplaced : Lost::Pending;
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
    /// over it, else memory, once the back ran every write sent there.
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
    bool synced{};
    /// The last inline QMDs sent, oldest first from inline_writes_count on: a back a millisecond
    /// behind has some ten to write, a few times that in long frames.
    std::array<InlineWrite, 256> inline_writes{};
    size_t inline_writes_count{};
};

} // namespace VideoCommon
