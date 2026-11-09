// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
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
/// The front reads the descriptors when it sends a signal, the back when it runs it, both from
/// guest memory without the GPU caches. They read the same where no map still queued can change
/// and nothing queued writes there but through the caches (GpuSplit::MemoryWritesTarget) onto
/// pages GPU work never wrote (Nx2MayBeGpuWritten). Where the front can't tell, its table may
/// differ from the back's until it takes the back's over again (Sync), and the launches count as
/// writing guest memory anywhere.
///
/// `Memory`, the channel's GPU address space as the back reads it, has:
/// - bool IsMapped(GPUVAddr, u64), void Read(GPUVAddr, void*, u64), u32 ReadWord(GPUVAddr) (as
///   MemoryManager::Read<u32>), bool InSyncpointShim(GPUVAddr, u64);
/// - std::optional<u64> Device(GPUVAddr), bool GpuWritten(u64 device, u64 size);
/// - u64 WritesTarget(const GpuSplit&, GPUVAddr, u64), void NoteWrite(GpuSplit&, GPUVAddr, u64).
class FrontQmdView {
public:
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;

    /// Before a QMD method is sent: takes over the table `back` shares with the other engines once
    /// the back ran everything sent.
    void Sync(const GpuSplit& split, const Tegra::NativeQmdScheduler& back) {
        if (!synced && split.Ran(split.Position())) [[unlikely]] {
            scheduler.CopyStates(back);
            synced = true;
        }
    }

    /// The engines schedule QMDs the front does not see (the split stopped, a front starts).
    void Lose() {
        synced = false;
    }

    [[nodiscard]] bool Synced() const {
        return synced;
    }

    /// The scheduling of PCAS action `argument` on the QMD at `address` that the back runs at the
    /// entry just sent (KeplerCompute::ProcessNativeMethod); `inline_qmd`: the QMD it writes there
    /// right before. Notes what the launches it runs release.
    template <typename Memory>
    void Signal(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 argument,
                const Descriptor* inline_qmd) {
        split.NoteCacheWrite(); // The dispatches' shader stores
        if (!synced) {
            split.NoteAnyWrite();
            return;
        }
        releases.clear();
        // A map still queued may change what the back reads, mapped or not.
        bool exact = split.Ran(split.LastUnplacedWrite());
        const auto read = [&](u64 qmd_address, Descriptor& descriptor) {
            if (!memory.IsMapped(qmd_address, sizeof(descriptor))) {
                return false;
            }
            if (inline_qmd != nullptr && qmd_address == address) {
                descriptor = *inline_qmd;
            } else {
                memory.Read(qmd_address, descriptor.data(), sizeof(descriptor));
                const auto device = memory.Device(qmd_address);
                exact = exact && device.has_value() &&
                        !memory.GpuWritten(*device, sizeof(descriptor)) &&
                        split.Ran(memory.WritesTarget(split, qmd_address, sizeof(descriptor)));
            }
            // A release this signal ran before writes into the QMD.
            for (const Release& earlier : releases) {
                exact = exact && (earlier.address + 16 <= qmd_address ||
                                  earlier.address >= qmd_address + sizeof(descriptor));
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
        if (!exact) {
            synced = false;
            split.NoteAnyWrite();
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

    /// A release the back runs with the entry just sent, as ProcessNativeMethod's: what the word
    /// holds once the back ran it.
    template <typename Memory>
    void NoteRelease(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 payload,
                     u32 flags) {
        const bool four_words = (flags >> 30) == 0;
        const bool reduction = (flags & (1U << 26)) != 0;
        const bool maximum = !reduction && !four_words && (flags & (1U << 19)) != 0;
        const auto device = memory.Device(address);
        // The word a reduction or a maximum combines the payload with, before this write.
        std::optional<u32> base;
        if (device && (reduction || maximum)) {
            base = WordAt(split, memory, address, *device);
        }
        memory.NoteWrite(split, address, four_words ? 16 : sizeof(u32));
        if (!device) {
            return;
        }
        if (reduction) {
            const bool is_signed = ((flags >> 24) & 1) != 0;
            split.NoteSemaphoreWrite(
                *device,
                base ? Tegra::ReduceSemaphore((flags >> 20) & 7, is_signed, *base, payload) : 0,
                base.has_value());
        } else if (four_words) {
            // The payload as 64 bits, a timestamp after it.
            split.NoteSemaphoreWrite(*device, payload, true, GpuSplit::Landing::Fenced);
            split.NoteSemaphoreWrite(*device + 4, 0, true, GpuSplit::Landing::Fenced);
        } else if (maximum) {
            // Whatever the word held, it holds the payload at least.
            split.NoteSemaphoreWrite(*device, base ? (std::max)(*base, payload) : payload, true,
                                     !base.has_value());
        } else {
            split.NoteSemaphoreWrite(*device, payload, true, GpuSplit::Landing::Fenced);
        }
    }

    /// The word the back finds at `address` (`device`) when it runs the entry just sent, where
    /// the front knows it: the last release sent there while the back may not have run it, else
    /// memory, once nothing queued may write there.
    template <typename Memory>
    [[nodiscard]] std::optional<u32> WordAt(const GpuSplit& split, const Memory& memory,
                                            GPUVAddr address, u64 device) const {
        // The back reads the syncpoint shim from the syncpoints.
        if (memory.InSyncpointShim(address, sizeof(u32)) ||
            Tegra::Nx2MirrorShadow::Get().Diverts(device)) {
            return std::nullopt;
        }
        if (const auto queued = split.QueuedSemaphoreWriteAt(device)) {
            if (queued->known && !queued->at_least && !queued->fenced &&
                queued->position >= split.LastUnplacedWrite()) {
                return queued->payload;
            }
            return std::nullopt;
        }
        if (memory.GpuWritten(device, sizeof(u32)) ||
            !split.Ran(memory.WritesTarget(split, address, sizeof(u32)))) {
            return std::nullopt;
        }
        return memory.ReadWord(address);
    }

    Tegra::NativeQmdScheduler scheduler{Tegra::NativeQmdScheduler::OwnTable{}};
    std::vector<Release> releases;
    bool synced{};
};

} // namespace VideoCommon
