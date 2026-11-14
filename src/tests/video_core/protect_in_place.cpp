// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/alignment.h"
#include "common/common_types.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/nx2_protect.h"

namespace {

namespace Nx2Protect = VideoCommon::Nx2Protect;
using VideoCommon::InPlaceCheck;

constexpr u64 PAGE = 4096;
constexpr u64 HIGH_PAGE = u64{1} << 22;

/// The device memory manager's cached-page counts: a page is protected while its count is above
/// zero, and each 0->1 or 1->0 change is a protection change (a VirtualProtect in the emulator).
class Device {
public:
    void UpdatePagesCachedCount(DAddr addr, size_t size, s32 delta) {
        Apply(addr, size, delta);
    }

    void UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta) {
        for (const auto& [addr, size] : ranges) {
            Apply(addr, size, delta);
        }
    }

    [[nodiscard]] bool Protected(u64 page) const {
        const auto it = counts.find(page);
        return it != counts.end() && it->second > 0;
    }

    std::vector<std::tuple<DAddr, size_t, s32>> calls;
    u64 protects = 0;
    u64 unprotects = 0;

private:
    void Apply(DAddr addr, size_t size, s32 delta) {
        calls.emplace_back(addr, size, delta);
        for (u64 page = addr / PAGE; page < (addr + size + PAGE - 1) / PAGE; ++page) {
            int& count = counts[page];
            count += delta;
            REQUIRE(count >= 0);
            if (delta > 0 && count == 1) {
                ++protects;
            }
            if (delta < 0 && count == 0) {
                ++unprotects;
            }
        }
    }

    std::map<u64, int> counts;
};

using Tracker = VideoCommon::MemoryTrackerBase<Device>;

struct RestoreSwitches {
    bool stats{Nx2Protect::g_stats};
    bool in_place{Nx2Protect::g_inline};
    ~RestoreSwitches() {
        Nx2Protect::g_stats = stats;
        Nx2Protect::g_inline = in_place;
    }
};

/// Guest memory, one host buffer over all of it and the buffer cache's real memory tracker, driven
/// the way the emulator drives them:
/// - a guest CPU write traps on a protected page (RasterizerVulkan::OnCPUWrite ->
///   BufferCache::WriteMemory: CPU-modified mark), then stores; on an unprotected page it stores,
/// - the GPU thread writes guest memory itself (inline-to-memory) and leaves the caches'
///   invalidation for FlushCaching; pages holding GPU-written data take the bytes in place at
///   once instead (BufferCache::InlineMemory),
/// - FlushCaching (RasterizerVulkan::InnerInvalidation) marks those ranges CPU-modified
///   (WriteMemory), or with EDEN_NX2_PROTECT_INLINE uploads tracked pages in place
///   (BufferCache::WriteGpuMemoryInPlace, mirrored below),
/// - a draw or dispatch flushes, then synchronizes what it reads (SynchronizeBuffer: upload scan,
///   whole pages from guest memory except GPU-written bytes) and must find in the buffer every
///   byte written before it,
/// - a shader write is synchronized first, then GPU-written (MarkWrittenBuffer), and downloaded
///   into guest memory later.
/// CPU writes into pages holding GPU-written bytes (collected and applied later) are not
/// modelled: the change under test never handles those pages.
class Model {
public:
    /// Crosses a 4 MiB manager boundary.
    static constexpr DAddr Base = 16 * HIGH_PAGE - 8 * PAGE;
    static constexpr u64 Pages = 24;
    static constexpr u64 Size = Pages * PAGE;

    explicit Model(bool in_place_) : in_place{in_place_}, tracker{device} {
        guest.assign(Size, 0);
        buffer.assign(Size, 0);
        gpu_value.assign(Size, 0);
        gpu_owned.assign(Size, false);
    }

    [[nodiscard]] bool AnyGpuOwned(u64 offset, u64 size) const {
        const u64 begin = Common::AlignDown(offset, PAGE);
        const u64 end = (std::min)(Common::AlignUp(offset + size, PAGE), Size);
        return std::any_of(gpu_owned.begin() + static_cast<std::ptrdiff_t>(begin),
                           gpu_owned.begin() + static_cast<std::ptrdiff_t>(end),
                           [](bool owned) { return owned; });
    }

    void CpuWrite(u64 offset, u64 size, u8 value) {
        bool trapped = false;
        for (u64 page = (Base + offset) / PAGE; page < (Base + offset + size + PAGE - 1) / PAGE;
             ++page) {
            trapped |= device.Protected(page);
        }
        if (trapped) {
            ++cpu_traps;
            tracker.MarkRegionAsCpuModified(Base + offset, size);
        }
        std::fill_n(guest.begin() + static_cast<std::ptrdiff_t>(offset), size, value);
    }

    void GpuSelfWrite(u64 offset, u64 size, u8 value) {
        std::fill_n(guest.begin() + static_cast<std::ptrdiff_t>(offset), size, value);
        if (AnyGpuOwned(offset, size)) {
            for (u64 i = offset; i < offset + size; ++i) {
                buffer[i] = value;
                gpu_owned[i] = false;
            }
            return;
        }
        pending.emplace_back(offset, size);
    }

    void Flush() {
        Batch batch{};
        for (const auto& [offset, size] : pending) {
            if (!(in_place && InPlace(offset, size, batch))) {
                tracker.MarkRegionAsCpuModified(Base + offset, size);
            }
        }
        pending.clear();
    }

    void Use(u64 offset, u64 size, bool dirty_scan) {
        Flush();
        Sync(offset, size, dirty_scan);
        Check(offset, size);
        ++uses;
    }

    /// `flush`: a draw or dispatch flushes first; a DMA copy (BufferCache::DMACopy) does not, so
    /// writes still waiting for FlushCaching may find GPU-written bytes on their pages.
    void GpuWriteBuffer(u64 offset, u64 size, u8 value, bool flush) {
        if (flush) {
            Flush();
        }
        Sync(offset, size, true);
        for (u64 i = offset; i < offset + size; ++i) {
            buffer[i] = value;
            gpu_value[i] = value;
            gpu_owned[i] = true;
        }
        tracker.MarkRegionAsGpuModified(Base + offset, size);
    }

    void Download(u64 offset, u64 size) {
        for (u64 i = offset; i < offset + size; ++i) {
            if (gpu_owned[i]) {
                guest[i] = buffer[i];
                gpu_owned[i] = false;
            }
        }
        tracker.ForEachDownloadRangeAndClear(Base + offset, size, [](u64, u64) {});
    }

    /// The buffer holds every byte the guest wrote, and the GPU's bytes where it wrote last.
    void Check(u64 offset, u64 size) const {
        u64 stale = 0;
        for (u64 i = offset; i < offset + size; ++i) {
            if (buffer[i] != (gpu_owned[i] ? gpu_value[i] : guest[i])) {
                ++stale;
            }
        }
        REQUIRE(stale == 0);
    }

    Device device;
    std::vector<u8> guest;
    std::vector<u8> buffer;
    u64 cpu_traps = 0;
    u64 in_place_uploads = 0;
    u64 uses = 0;
    bool in_place;
    Tracker tracker;

private:
    struct Batch {
        DAddr begin{};
        DAddr end{};
    };

    void Sync(u64 offset, u64 size, bool dirty_scan) {
        const auto upload = [&](u64 address, u64 range_size) {
            for (u64 i = address - Base; i < address - Base + range_size; ++i) {
                if (!gpu_owned[i]) {
                    buffer[i] = guest[i];
                }
            }
        };
        if (dirty_scan) {
            tracker.ForEachDirtyUploadRange(Base + offset, size, upload);
        } else {
            tracker.ForEachUploadRange(Base + offset, size, upload);
        }
    }

    /// BufferCache::WriteGpuMemoryInPlace without its notes (one buffer holds every page here).
    bool InPlace(u64 offset, u64 size, Batch& batch) {
        const DAddr page_begin = Common::AlignDown(Base + offset, PAGE);
        const DAddr page_end = Common::AlignUp(Base + offset + size, PAGE);
        if (batch.begin <= page_begin && page_end <= batch.end) {
            return true;
        }
        if (page_end - page_begin > 16 * PAGE || AnyGpuOwned(offset, size) ||
            tracker.CheckInPlace(page_begin, page_end - page_begin) != InPlaceCheck::Clean) {
            return false;
        }
        for (u64 i = page_begin - Base; i < page_end - Base; ++i) {
            buffer[i] = guest[i];
        }
        tracker.MarkInPlace(page_begin, page_end - page_begin);
        batch = Batch{.begin = page_begin, .end = page_end};
        ++in_place_uploads;
        return true;
    }

    std::vector<u8> gpu_value;
    std::vector<bool> gpu_owned;
    std::vector<std::pair<u64, u64>> pending;
};

/// One random step: a CPU write, a GPU self-write, a use, a shader write or a download.
void RandomStep(Model& model, std::mt19937_64& random) {
    const u64 offset = random() % Model::Size;
    const u64 size = 1 + random() % (std::min<u64>)(Model::Size - offset, 3 * PAGE);
    const u8 value = static_cast<u8>(random());
    switch (random() % 10) {
    case 0:
    case 1:
    case 2:
        if (!model.AnyGpuOwned(offset, size)) {
            model.CpuWrite(offset, size, value);
        }
        break;
    case 3:
    case 4:
        model.GpuSelfWrite(offset, (std::min<u64>)(size, 512), value);
        break;
    case 5:
    case 6:
    case 7:
        model.Use(offset, size, random() % 2 == 0);
        break;
    case 8:
        if (const u64 kind = random() % 4; kind < 2) {
            model.GpuWriteBuffer(offset, (std::min<u64>)(size, 256), value, kind == 0);
        }
        break;
    default:
        model.Download(offset, size);
        break;
    }
}

} // namespace

TEST_CASE("Nx2Protect in place: page states", "[video_core][nx2_protect]") {
    RestoreSwitches restore;
    Nx2Protect::g_stats = false;
    Nx2Protect::g_inline = true;
    Device device;
    Tracker tracker{device};
    constexpr DAddr base = 16 * HIGH_PAGE - 2 * PAGE;
    // No manager yet, and a new manager starts with every page CPU-modified.
    REQUIRE(tracker.CheckInPlace(base, PAGE) == InPlaceCheck::Untracked);
    tracker.MarkRegionAsCpuModified(base, 4 * PAGE);
    REQUIRE(tracker.CheckInPlace(base, 4 * PAGE) == InPlaceCheck::Untracked);
    REQUIRE(tracker.CheckInPlace(base, 0) == InPlaceCheck::Untracked);

    // An upload scan tracks the pages (across the manager boundary): clean.
    tracker.ForEachUploadRange(base, 4 * PAGE, [](u64, u64) {});
    REQUIRE(device.protects == 4);
    REQUIRE(tracker.CheckInPlace(base, 4 * PAGE) == InPlaceCheck::Clean);
    REQUIRE(tracker.CheckInPlace(base + 3 * PAGE, 1) == InPlaceCheck::Clean);

    // An in-place upload holds until the next scan over the page; nothing else changes.
    tracker.MarkInPlace(base + PAGE, PAGE);
    REQUIRE(tracker.CheckInPlace(base + PAGE, 16) == InPlaceCheck::Uploaded);
    REQUIRE(tracker.CheckInPlace(base, PAGE) == InPlaceCheck::Clean);
    REQUIRE(tracker.CheckInPlace(base + 2 * PAGE, 2 * PAGE) == InPlaceCheck::Clean);
    REQUIRE(!tracker.IsRegionCpuModified(base, 4 * PAGE));
    REQUIRE(device.unprotects == 0);
    tracker.ForEachDirtyUploadRange(base + PAGE + 100, 8, [](u64, u64) { FAIL("nothing to upload"); });
    REQUIRE(tracker.CheckInPlace(base + PAGE, PAGE) == InPlaceCheck::Clean);

    // GPU-written and preflushable pages are not clean; a CPU mark untracks.
    tracker.MarkRegionAsGpuModified(base + 2 * PAGE, 8);
    REQUIRE(tracker.CheckInPlace(base + 2 * PAGE, PAGE) == InPlaceCheck::GpuState);
    tracker.UnmarkRegionAsGpuModified(base + 2 * PAGE, 8);
    tracker.MarkRegionAsPreflushable(base + 3 * PAGE, 8);
    REQUIRE(tracker.CheckInPlace(base + 3 * PAGE, PAGE) == InPlaceCheck::GpuState);
    tracker.UnmarkRegionAsPreflushable(base + 3 * PAGE, 8);
    tracker.MarkRegionAsCpuModified(base + 8, 4);
    REQUIRE(tracker.CheckInPlace(base, PAGE) == InPlaceCheck::Untracked);
    REQUIRE(device.unprotects == 1);

    // Without the switch a scan leaves the in-place bits alone (no other code reads them).
    Nx2Protect::g_inline = false;
    tracker.MarkInPlace(base + 2 * PAGE, PAGE);
    tracker.ForEachUploadRange(base, 4 * PAGE, [](u64, u64) {});
    REQUIRE(tracker.CheckInPlace(base + 2 * PAGE, PAGE) == InPlaceCheck::Uploaded);
}

TEST_CASE("Nx2Protect in place: every write before a use reaches the buffer",
          "[video_core][nx2_protect]") {
    RestoreSwitches restore;
    Nx2Protect::g_stats = false;
    for (u64 seed = 0; seed < 200; ++seed) {
        std::mt19937_64 random_off{seed};
        std::mt19937_64 random_on{seed};
        Nx2Protect::g_inline = false;
        Model off{false};
        for (int step = 0; step < 400; ++step) {
            RandomStep(off, random_off);
        }
        Nx2Protect::g_inline = true;
        Model on{true};
        for (int step = 0; step < 400; ++step) {
            RandomStep(on, random_on);
        }
        // The same guest memory either way, and every use found its bytes (Model::Check).
        REQUIRE(off.guest == on.guest);
        REQUIRE(off.uses == on.uses);
        REQUIRE(off.in_place_uploads == 0);
        // Whatever is left over is uploaded at the next use.
        off.Use(0, Model::Size, false);
        Nx2Protect::g_inline = true;
        on.Use(0, Model::Size, false);
        REQUIRE(off.buffer == on.buffer);
    }
}

TEST_CASE("Nx2Protect in place: a dispatch's own constant data stays protected",
          "[video_core][nx2_protect]") {
    RestoreSwitches restore;
    Nx2Protect::g_stats = false;
    const auto run = [](bool in_place) {
        Nx2Protect::g_inline = in_place;
        Model model{in_place};
        // Each dispatch writes its parameters with inline-to-memory and reads them as a constant
        // buffer, next to a table the guest CPU wrote once.
        model.CpuWrite(PAGE, 2 * PAGE, 7);
        for (u8 dispatch = 0; dispatch < 100; ++dispatch) {
            model.GpuSelfWrite(3 * PAGE + 64 * (dispatch % 8), 256, dispatch);
            model.Use(PAGE, 3 * PAGE, dispatch % 2 == 0);
        }
        return std::make_tuple(model.device.protects, model.device.unprotects,
                               model.in_place_uploads);
    };
    const auto [base_protects, base_unprotects, base_in_place] = run(false);
    const auto [protects, unprotects, in_place] = run(true);
    // As before: each dispatch unprotects the page and its upload scan protects it again.
    REQUIRE(base_protects >= 100);
    REQUIRE(base_unprotects >= 99);
    REQUIRE(base_in_place == 0);
    // In place: the first upload protects the pages, and they stay protected.
    REQUIRE(protects == 3);
    REQUIRE(unprotects == 0);
    REQUIRE(in_place >= 99);
}

TEST_CASE("Nx2Protect in place: written again before a use falls back",
          "[video_core][nx2_protect]") {
    RestoreSwitches restore;
    Nx2Protect::g_stats = false;
    Nx2Protect::g_inline = true;
    Model model{true};
    model.Use(0, 2 * PAGE, false);
    REQUIRE(model.device.protects == 2);
    model.GpuSelfWrite(16, 16, 1);
    model.Flush();
    REQUIRE(model.in_place_uploads == 1);
    // A second flush before any use: CPU-modified as before (one unprotect), and the next use
    // uploads the page again and protects it.
    model.GpuSelfWrite(32, 16, 2);
    model.Flush();
    REQUIRE(model.in_place_uploads == 1);
    REQUIRE(model.device.unprotects == 1);
    model.Use(0, 2 * PAGE, true);
    REQUIRE(model.device.protects == 3);
    // Two ranges on one page in one flush: one upload.
    model.GpuSelfWrite(64, 8, 3);
    model.GpuSelfWrite(512, 8, 4);
    model.Use(0, PAGE, false);
    REQUIRE(model.in_place_uploads == 2);
    REQUIRE(model.device.unprotects == 1);
}

TEST_CASE("Nx2Protect: stats and the in-place switch off change no tracking",
          "[video_core][nx2_protect]") {
    RestoreSwitches restore;
    Nx2Protect::g_inline = false;
    for (u64 seed = 0; seed < 50; ++seed) {
        std::mt19937_64 random_plain{seed};
        std::mt19937_64 random_stats{seed};
        Nx2Protect::g_stats = false;
        Model plain{false};
        for (int step = 0; step < 400; ++step) {
            RandomStep(plain, random_plain);
        }
        Nx2Protect::g_stats = true;
        Model stats{false};
        for (int step = 0; step < 400; ++step) {
            RandomStep(stats, random_stats);
        }
        // The same count changes in the same order, the same buffer and guest bytes.
        REQUIRE(plain.device.calls == stats.device.calls);
        REQUIRE(plain.buffer == stats.buffer);
        REQUIRE(plain.guest == stats.guest);
        for (DAddr page = Model::Base; page < Model::Base + Model::Size; page += PAGE) {
            REQUIRE(stats.tracker.CheckInPlace(page, PAGE) != InPlaceCheck::Uploaded);
        }
    }
    // The stats saw rearms and logged a window (the first note opens it without ReadSwitch).
    const std::string line = Nx2Protect::LastLogLine();
    REQUIRE(line.starts_with("Nx2Protect "));
    REQUIRE(line.find("rearmed pages") != std::string::npos);
}

TEST_CASE("Nx2Protect churn table", "[video_core][nx2_protect]") {
    Nx2Protect::ChurnTable table;
    constexpr u64 ms = 1'000'000;
    u64 now = 1000 * ms;
    // A page rearmed every frame (20 ms) builds a streak; the fourth rearm in a row is steady.
    auto result = table.Note(7, now, 0);
    REQUIRE(result.interval == Nx2Protect::Interval::First);
    REQUIRE(result.first_in_window);
    REQUIRE(result.streak == 0);
    for (u32 frame = 1; frame <= 4; ++frame) {
        now += 20 * ms;
        result = table.Note(7, now, 0);
        REQUIRE(result.interval == Nx2Protect::Interval::Under25ms);
        REQUIRE(result.streak == frame);
        REQUIRE(!result.first_in_window);
        REQUIRE(result.steady == (frame == 3));
    }
    // A pause longer than two frames ends the streak; a new window counts the page again.
    now += 60 * ms;
    result = table.Note(7, now, 1);
    REQUIRE(result.interval == Nx2Protect::Interval::Under100ms);
    REQUIRE(result.streak == 0);
    REQUIRE(result.first_in_window);
    // Twice in one frame.
    now += 2 * ms;
    result = table.Note(7, now, 1);
    REQUIRE(result.interval == Nx2Protect::Interval::Under4ms);
    REQUIRE(result.streak == 1);

    for (u32 i = 0; i < 5; ++i) {
        now += 10 * ms;
        table.Note(9, now, 1);
    }
    std::array<Nx2Protect::ChurnTable::Top, 4> top{};
    REQUIRE(table.Busiest(1, false, top) == 2);
    REQUIRE(top[0].page == 9);
    REQUIRE(top[0].rearms == 5);
    REQUIRE(top[1].page == 7);
    REQUIRE(top[1].rearms == 2);
    REQUIRE(table.Busiest(1, true, top) == 1);
    REQUIRE(top[0].page == 9);
    REQUIRE(table.Busiest(0, false, top) == 0);

    // Pages not rearmed for 20 s are forgotten.
    REQUIRE(table.Size() == 2);
    table.Prune(now + Nx2Protect::ChurnTable::ForgetNs + 1);
    REQUIRE(table.Size() == 0);
}
