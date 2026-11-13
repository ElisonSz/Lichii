// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>

#include "common/assert.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_sync_trace.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_acquire_stats.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_runahead.h"
#include "video_core/nx2_sync_suspend.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

namespace {

namespace Details = VideoCommon::Nx2AcquireDetails;

constexpr u32 Gf100BindClassMask = 0xffff;
constexpr u32 Gf100BindValidMask =  0x1f0000 | Gf100BindClassMask;

bool WatchSemaphore(GPUVAddr address) {
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    return watch != 0 && (address & ~GPUVAddr{0xf}) == watch;
}

// Register the acquire word and its mirror before the first decision, including a queued
// answer that never reads RAM. Host pages let unrelated stores skip device translation.
void TrackSemaphoreView(DmaPusher& dma_pusher, GPUVAddr address, DAddr device) {
    if (!VideoCommon::SemRecorder::Enabled()) {
        return;
    }
    auto& memory = dma_pusher.memory_manager;
    VideoCommon::SemRecorder::Track(device, memory.GetPointer(address));
    const GPUVAddr mirror = Nx2SemaphoreMirror(address);
    const GPUVAddr partner = mirror != 0                        ? mirror
                             : Nx2InSemaphoreFirstPage(address) ? address + 0x1000
                                                                : 0;
    if (partner != 0) {
        if (const auto paired_device = memory.GpuToCpuAddress(partner)) {
            VideoCommon::SemRecorder::Track(*paired_device, memory.GetPointer(partner));
        }
    }
}

// Reads a semaphore word as a GPU acquire sees it in memory: a mirrored release the guest memory
// does not hold counts (Nx2MirrorShadow). Without a flush: the buffer cache may hold the word
// newer where the GPU wrote its page (ReadSemaphoreWord, AskView).
// EDEN_NX2_RUNAHEAD_LANDINGS_VERIFY: the front reads a word as settled while a landing converted in
// a landing window is still to reach it (it noted the write fenced: the view should not read it).
void NoteLandingRead(DAddr device, GPUVAddr address, const char* what) {
    if (!VideoCommon::GpuSplit::Get().LandingPendingAt(device)) {
        return;
    }
    VideoCommon::RunAhead::landing_counts.early_reads.fetch_add(1, std::memory_order_relaxed);
    if (static std::atomic<u32> logged{}; logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(HW_GPU,
                    "Run-ahead landings verify: {} of semaphore {:#x} (device {:#x}) while a "
                    "converted landing to it is pending",
                    what, address, device);
    }
}

u32 ReadSemaphoreMemory(DmaPusher& dma_pusher, GPUVAddr address, std::optional<DAddr> device,
                        bool* shadow_used = nullptr) {
    if (device) {
        TrackSemaphoreView(dma_pusher, address, *device);
    }
    const u32 guest = dma_pusher.memory_manager.Read<u32>(address);
    if (!device) {
        return guest;
    }
    const auto mirrored = Nx2MirrorShadow::Get().Read(*device, guest);
    if (shadow_used != nullptr) {
        *shadow_used = mirrored.has_value();
    }
    if (VideoCommon::SemRecorder::Enabled()) {
        VideoCommon::SemRecorder::Add({.kind = "memory",
                                       .address = *device,
                                       .gpu = address,
                                       .value = mirrored.value_or(guest),
                                       .before = guest,
                                       .flags = u64{mirrored.has_value()}});
    }
    if (WatchSemaphore(address)) {
        LOG_INFO(HW_GPU, "Semaphore watch: memory {:#x} guest {:#x} shadow {} seen {:#x}", address,
                 guest, mirrored.has_value(), mirrored.value_or(guest));
    }
    return mirrored.value_or(guest);
}

// Reads a semaphore word as a GPU acquire sees it, flushing a GPU-written copy out of the buffer
// cache first.
u32 ReadSemaphoreWord(DmaPusher& dma_pusher, GPUVAddr address) {
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (device && dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                         VideoCommon::CacheType::BufferCache)) {
        const bool watched = WatchSemaphore(address);
        const u32 before = watched ? dma_pusher.memory_manager.Read<u32>(address) : 0;
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = dma_pusher.rasterizer, device = *device] {
                VideoCommon::SemRecorder::Record("drain-flush-begin", device);
                rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
                VideoCommon::SemRecorder::Record("drain-flush-end", device);
            },
            VideoCommon::DrainSite::SemaphoreFlush);
        if (watched) {
            LOG_INFO(HW_GPU, "Semaphore watch: drain flush {:#x} guest {:#x} -> {:#x}", address,
                     before, dma_pusher.memory_manager.Read<u32>(address));
        }
    }
    if (device && VideoCommon::RunAhead::landings_verify) [[unlikely]] {
        NoteLandingRead(*device, address, "a read after a drain");
    }
    const u32 value = ReadSemaphoreMemory(dma_pusher, address, device);
    if (WatchSemaphore(address)) {
        LOG_INFO(HW_GPU, "Semaphore watch: drained {:#x} guest {:#x} seen {:#x}", address,
                 dma_pusher.memory_manager.Read<u32>(address), value);
    }
    return value;
}

// With the GPU thread split (gpu_split.h) the rasterizer belongs to the back thread: work that
// only feeds it runs there in order; reads of what it wrote first wait for it (DrainBack).
template <typename Func>
void OnBack(const Func& func,
            VideoCommon::GpuSplit::Writes writes = VideoCommon::GpuSplit::Writes::Anywhere,
            u64 flags = 0) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(func, flags, writes);
    } else {
        func();
    }
}

// EDEN_NX2_DRAWX_CALLS (releases deferred with Nx2FastDraw::CallFlags()): a release writes at
// most 16 bytes at its address (payload and timestamp), now or when its fence is released, which
// ends the fast draw chain itself. The bytes count as constant buffer data written, as the 3D
// report method's do under EDEN_NX2_DRAW_REUSE. Nothing when the switch is off.
void NoteReleaseForFastDraw(VideoCore::RasterizerInterface* rasterizer, GPUVAddr address) {
    constexpr u64 ReleaseBytes = 16;
    rasterizer->NoteFastDrawWrite(address, ReleaseBytes);
}

constexpr auto FenceWrites = VideoCommon::GpuSplit::Writes::Caches;

void DrainBack(VideoCommon::DrainSite site = VideoCommon::DrainSite::Acquire) {
    VideoCommon::GpuSplit::Get().Drain(site);
}

// The fence manager records into the scheduler and downloads the CPU's writes over GPU data:
// the back's state, which a drained back still uses from its idle hook (it submits).
void ReleaseFences(DmaPusher& dma_pusher) {
    VideoCommon::GpuSplit::Get().Run(
        [rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
        VideoCommon::DrainSite::FenceRelease);
}

// A release with `properties` was just sent to the back: tell the front what it writes, and when
// it lands (AskView).
void NoteRelease(DmaPusher& dma_pusher, GPUVAddr address, u32 payload,
                 VideoCommon::QueryPropertiesFlags properties) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        return;
    }
    split.NoteWrite(dma_pusher.memory_manager, address,
                    True(properties & VideoCommon::QueryPropertiesFlags::HasTimeout) ? 16 : 4);
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return;
    }
    using VideoCommon::QueryPropertiesFlags;
    // A fence-flagged report lands once the fence after it ran (QueryCacheBase::CounterReport).
    const auto landing = True(properties & QueryPropertiesFlags::IsAFence)
                             ? VideoCommon::GpuSplit::Landing::Fenced
                             : VideoCommon::GpuSplit::Landing::AtRun;
    VideoCommon::SemRecorder::Add({.kind = "front-puller",
                                   .address = *device,
                                   .gpu = address,
                                   .value = payload,
                                   .position = split.Position(),
                                   .flags = static_cast<u64>(properties)});
    split.NoteSemaphoreWrite(*device, payload, true, landing, false,
                             VideoCommon::GpuSplit::SemaphoreWriter::Puller);
    if (True(properties & QueryPropertiesFlags::HasTimeout)) {
        // The payload as 64 bits (a timestamp follows). A mirrored one into a word with direct
        // releases goes to the mirror's shadow instead, and leaves the high word as it is.
        const bool mirrored = True(properties & QueryPropertiesFlags::Nx2Mirror);
        split.NoteSemaphoreWrite(*device + 4, 0, !mirrored, landing, false,
                                 VideoCommon::GpuSplit::SemaphoreWriter::Puller);
    }
}

// Whether the back writes a payload report without a fence flag to memory when it runs it
// (CounterReport): below High GPU accuracy, with fences that run their operations at once
// (FenceManager::SignalFence). Fence-flagged reports land once their fence ran
// (GpuSplit::Landing::Fenced).
bool ReleasesLandAtOnce() {
    const bool delay_fence =
        Settings::IsGPUFenceBehaviorDefault()
            ? Settings::IsGPULevelHigh()
            : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
    return !delay_fence && !Settings::IsGPULevelHigh();
}

// Whether the front decides Switch 2 acquires from its view of the semaphore words (AskView)
// instead of draining the split: the back writes each release to memory as it runs it, or once
// its fence ran (GpuSplit::Landing), and the front sees every release it sends, its macros'
// included.
bool ViewApplies() {
    const auto& split = VideoCommon::GpuSplit::Get();
    return split.Active() && split.FrontMacros() && ReleasesLandAtOnce();
}

using ViewAnswer = VideoCommon::GpuSplit::SemaphoreView;

// The split's view of an acquire (GpuSplit::ViewSemaphore), and words the buffer cache may hold
// newer than memory.
ViewAnswer AskView(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, DAddr device) {
    using Kind = ViewAnswer::Kind;
    TrackSemaphoreView(dma_pusher, acquire.address, device);
    const auto memory = [&](bool fenced) -> std::optional<u32> {
        if (!fenced && VideoCommon::RunAhead::landings_verify) [[unlikely]] {
            NoteLandingRead(device, acquire.address, "a settled read");
        }
        bool shadow_used = false;
        const u32 value = ReadSemaphoreMemory(dma_pusher, acquire.address, device, &shadow_used);
        // A positioned wait consumes reports, not their fence callbacks. An older direct
        // callback can still remove this shadow before the latest fenced release lands.
        if (fenced && shadow_used) {
            return std::nullopt;
        }
        return value;
    };
    // The words Switch 2 syncs on only advance (NVN2 even releases QMD completions as maxima,
    // KeplerCompute::ProcessNativeMethod). A word the mirror diverts (Nx2MirrorShadow) can
    // instead fall behind when a direct release replaces the mirrored counter.
    const bool advances = acquire.WaitsToReach() && !Nx2MirrorShadow::Get().Diverts(device);
    if (Nx2MayBeGpuWritten(device, sizeof(u32)) &&
        dma_pusher.rasterizer->MustFlushRegion(device, sizeof(u32),
                                               VideoCommon::CacheType::BufferCache)) {
        // Guest memory is not authoritative while the cache owns these bytes. A direct
        // semaphore release may have advanced RAM without updating that copy; the download
        // used by the drain path can therefore lower the word, even for a >= acquire.
        VideoCommon::SemRecorder::Record("decision-flush", device, acquire.value);
        return {Kind::Flush};
    }
    const auto answer = VideoCommon::GpuSplit::Get().ViewSemaphore(
        device, advances, acquire.writers_before,
        [&acquire](u32 value) { return acquire.Matches(value); }, memory);
    VideoCommon::SemRecorder::Add({.kind = "decision",
                                   .address = device,
                                   .gpu = acquire.address,
                                   .value = acquire.value,
                                   .before = acquire.writers_before,
                                   .position = answer.position,
                                   .flags = static_cast<u64>(answer.kind) |
                                            (u64{acquire.operation} << 8) | (u64{advances} << 16)});
    return answer;
}

// Has the back flush the word out of the buffer cache after the work sent before: memory holds
// it once the back ran the flush (PendingAcquire::flush_position).
template <bool Track>
void RequestFlush(DmaPusher& dma_pusher, Puller::PendingAcquire& acquire, DAddr device) {
    auto& split = VideoCommon::GpuSplit::Get();
    split.Defer(
        [rasterizer = dma_pusher.rasterizer, memory_manager = &dma_pusher.memory_manager, device,
         address = acquire.address, target = acquire.value,
         watched = WatchSemaphore(acquire.address)] {
            const u32 before = watched ? memory_manager->Read<u32>(address) : 0;
            VideoCommon::SemRecorder::Record("view-flush-begin", device);
            rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
            VideoCommon::SemRecorder::Record("view-flush-end", device);
            if (watched) {
                LOG_INFO(HW_GPU,
                         "Semaphore watch: view flush {:#x} target {:#x} guest {:#x} -> {:#x}",
                         address, target, before, memory_manager->Read<u32>(address));
            }
        },
        0, VideoCommon::GpuSplit::Writes::Caches);
    acquire.flush_position = split.Position();
    if constexpr (Track) {
        acquire.stats_flushed = true;
    }
    VideoCommon::Count(VideoCommon::g_acquire_stats.flushes);
}

// The fence work an acquire that still waits after the drain sent (ReleaseFences), sent once the
// back ran everything before the acquire: the release that work runs then counts as it did (the
// waiter is registered), and acquires the queued work releases send none.
void MaybeReleaseFences(DmaPusher& dma_pusher, Puller::PendingAcquire& acquire) {
    if (acquire.fences_released || !VideoCommon::GpuSplit::Get().Ran(acquire.sent_at)) {
        return;
    }
    acquire.fences_released = true;
    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
}

// Logs once an acquire that waits for more than 2 s. `current`: the word as read after a drain,
// or nothing to read it from memory.
void ReportLongWait(DmaPusher& dma_pusher, Puller::PendingAcquire& acquire,
                    std::optional<u32> current) {
    if (acquire.reported ||
        std::chrono::steady_clock::now() - acquire.since <= std::chrono::seconds{2}) {
        return;
    }
    acquire.reported = true;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    const GPUVAddr words = acquire.address & ~GPUVAddr{0xf};
    LOG_WARNING(
        HW_GPU,
        "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s; device "
        "{:#x}, buffer cache holds GPU data: {}, words {:08x} {:08x} {:08x} {:08x}",
        dma_pusher.channel_state.bind_id, acquire.address, acquire.operation, acquire.value,
        current.value_or(ReadSemaphoreMemory(dma_pusher, acquire.address, device)),
        device.value_or(0),
        device && dma_pusher.rasterizer->MustFlushRegion(*device, 16, VideoCommon::CacheType::All),
        dma_pusher.memory_manager.Read<u32>(words), dma_pusher.memory_manager.Read<u32>(words + 4),
        dma_pusher.memory_manager.Read<u32>(words + 8),
        dma_pusher.memory_manager.Read<u32>(words + 12));
}

// EDEN_NX2_ACQUIRE_VERIFY: what the drain path decides instead (the waiter is registered, so a
// release the back ran latches it).
bool MetAfterDrain(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, const void* waiter,
                   VideoCommon::DrainSite site) {
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    VideoCommon::SemRecorder::Add({.kind = "verify-drain-begin",
                                   .address = device.value_or(0),
                                   .gpu = acquire.address,
                                   .value = acquire.value});
    DrainBack(site);
    const u32 current = ReadSemaphoreWord(dma_pusher, acquire.address);
    const bool met = acquire.Matches(current) || Nx2SemaphoreWaiters::Get().Latched(waiter);
    VideoCommon::SemRecorder::Add({.kind = "verify-drain-end",
                                   .address = device.value_or(0),
                                   .gpu = acquire.address,
                                   .value = acquire.value,
                                   .after = current,
                                   .flags = u64{met}});
    return met;
}

// Whether a fence-flagged release to the word at `device` may still land.
bool FencedWritePending(DAddr device) {
    const auto queued = VideoCommon::GpuSplit::Get().QueuedSemaphoreWriteAt(device);
    return queued && queued->fenced;
}

// EDEN_NX2_RUNAHEAD_VERIFY: an acquire the view passed with its reach run but the baseline's
// writers still queued, checked on the back at its place in the ring: everything sent before it
// has run, as after the baseline's wait (which nothing between the reach and the acquire could
// change). The word as acquires read it must still meet it, and the buffer cache must not hold
// it GPU-written (the baseline's view would flush it first).
struct RunAheadCheck {
    VideoCore::RasterizerInterface* rasterizer;
    MemoryManager* memory;
    GPUVAddr address;
    DAddr device;
    u64 sent_at;
    u64 reach;
    u64 writers;
    u32 value;
    u32 operation;
    VideoCommon::RunAhead::Reach reason;
    /// A fence-flagged release to the word was pending at the decision: it may land at any time,
    /// for the baseline too (its view reads memory then as well), so a change counts apart.
    bool fenced;

    void operator()() const {
        auto& verified = VideoCommon::RunAhead::verified;
        verified.checked.fetch_add(1, std::memory_order_relaxed);
        const bool flush =
            Nx2MayBeGpuWritten(device, sizeof(u32)) &&
            rasterizer->MustFlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
        const u32 guest = memory->Read<u32>(address);
        const auto shadow = Nx2MirrorShadow::Get().Read(device, guest);
        const u32 seen = shadow.value_or(guest);
        const bool met = operation == Nx2SemaphoreWaiters::AcquireEqual    ? seen == value
                         : operation == Nx2SemaphoreWaiters::AcquireGequal ? seen >= value
                                                                           : (seen & value) != 0;
        if (met && !flush) {
            return;
        }
        if (fenced && !flush) {
            verified.raced.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        verified.mismatches.fetch_add(1, std::memory_order_relaxed);
        if (flush) {
            verified.flushes.fetch_add(1, std::memory_order_relaxed);
        }
        if (static std::atomic<u32> logged{}; logged.fetch_add(1) < 32) {
            LOG_WARNING(HW_GPU,
                        "Run-ahead verify: semaphore {:#x} (device {:#x}, op {}, {:#x}) passed "
                        "before the writers at {:#x} ran (reach {:#x}, {}), acquire at {:#x}; "
                        "now {:#x} (guest {:#x}, shadow {}), GPU-written in the buffer cache {}",
                        address, device, operation, value, writers, reach,
                        VideoCommon::RunAhead::reach_names[static_cast<size_t>(reason)], sent_at,
                        seen, guest, shadow.has_value(), flush);
        }
    }
};

// EDEN_NX2_RUNAHEAD_W1_VERIFY: an acquire passed before its fence-flagged release landed. It has a
// waiter of its own (the slot's address is the owner), which the release must latch as it lands,
// as it would have latched the baseline's suspended acquire. Front only.
struct W1Check {
    GPUVAddr address{};
    u64 device{};
    u64 position{}; ///< Of the release
    u32 value{};
    u32 operation{};
    u32 payload{};
    std::chrono::steady_clock::time_point since{};
    bool live{};
};
std::array<W1Check, 16> g_w1_checks{};

/// How long a release may take to land before its pass counts as unconfirmed (frames with
/// texture uploads or pipeline builds hold the host queue for tens of milliseconds).
constexpr auto W1CheckLimit = std::chrono::milliseconds{250};

void SweepW1Checks(bool all) {
    auto& waiters = Nx2SemaphoreWaiters::Get();
    auto& counts = VideoCommon::RunAhead::counters;
    const auto now = std::chrono::steady_clock::now();
    for (W1Check& check : g_w1_checks) {
        if (!check.live) {
            continue;
        }
        if (waiters.Latched(&check)) {
            ++counts.w1_confirmed;
        } else if (all || now - check.since > W1CheckLimit) {
            ++counts.w1_unconfirmed;
            if (static std::atomic<u32> logged{}; logged.fetch_add(1) < 32) {
                LOG_WARNING(HW_GPU,
                            "Run-ahead W1 verify: semaphore {:#x} (device {:#x}, op {}, {:#x}) "
                            "passed before the release of {:#x} at {:#x} landed; no write met it "
                            "within {} ms",
                            check.address, check.device, check.operation, check.value,
                            check.payload, check.position, W1CheckLimit.count());
            }
        } else {
            continue;
        }
        waiters.Remove(&check);
        check.live = false;
    }
}

void StartW1Check(const Puller::PendingAcquire& acquire, u64 device, u64 position, u32 payload,
                  const void* waiter) {
    auto& waiters = Nx2SemaphoreWaiters::Get();
    auto& counts = VideoCommon::RunAhead::counters;
    const auto free = std::ranges::find(g_w1_checks, false, &W1Check::live);
    if (free == g_w1_checks.end() ||
        !waiters.Add(&*free, device, acquire.value, acquire.operation)) {
        ++counts.w1_unchecked;
        return;
    }
    // A landing before the check's waiter was added latched the acquire's own (still added).
    if (waiters.Latched(waiter)) {
        waiters.Remove(&*free);
        ++counts.w1_confirmed;
        return;
    }
    *free = {acquire.address,
             device,
             position,
             acquire.value,
             acquire.operation,
             payload,
             std::chrono::steady_clock::now(),
             true};
}

// EDEN_NX2_RUNAHEAD_W1: the fence-flagged release that decides an acquire the view can't pass now
// (it waits for the release's entry, or the word does not meet it yet), if the acquire may go on
// at once: the last release sent to the word, not landed yet, writes a payload the front knows
// (no reduction or maximum) that meets the acquire, and the acquire's waiter has a slot, so the
// release latches it as it lands even if a later write changes the word. The baseline goes on
// then at the latest; until then nothing else decides (see AcquireWithoutDrain).
std::optional<VideoCommon::GpuSplit::QueuedSemaphoreWrite> W1Release(
    const VideoCommon::GpuSplit& split, const Puller::PendingAcquire& acquire, u64 device,
    const ViewAnswer& answer, bool count) {
    using VideoCommon::RunAhead::W1;
    std::optional<VideoCommon::GpuSplit::QueuedSemaphoreWrite> release;
    const W1 decision = split.W1Release(
        device, answer, acquire.registered, acquire.writers_before,
        [&acquire](u32 value) { return acquire.Matches(value); }, &release);
    if (count && VideoCommon::RunAhead::stats && decision != W1::NotW1) [[unlikely]] {
        auto& counts = VideoCommon::RunAhead::counters;
        ++counts.w1_candidates;
        counts.w1_not_plain += decision == W1::NotPlain ? 1 : 0;
        counts.w1_unmet += decision == W1::Unmet ? 1 : 0;
        // Without a waiter slot the baseline sees the release only if a poll finds it.
        counts.w1_unlatched += decision == W1::Unlatched ? 1 : 0;
        counts.w1_overwritten += decision == W1::Overwritten ? 1 : 0;
    }
    return decision == W1::Pass ? release : std::nullopt;
}

// EDEN_NX2_RUNAHEAD_LANDINGS_VERIFY: a waiter the window's release must latch before any landing
// converted in the window is applied (GpuSplit::CheckConvertedLanding). Front only.
size_t g_landing_slot{};

// Retires the checks whose waiters latched, so waiter slots stay free for acquires.
void SweepLandingChecks() {
    auto& waiters = Nx2SemaphoreWaiters::Get();
    for (auto& check : VideoCommon::RunAhead::landing_checks) {
        if (check.registered && waiters.Latched(&check)) {
            check.latched.store(true, std::memory_order_release);
            waiters.Remove(&check);
            check.registered = false;
        }
    }
}

u64 StartLandingCheck(const Puller::PendingAcquire& acquire, u64 device, const void* waiter) {
    auto& waiters = Nx2SemaphoreWaiters::Get();
    auto& checks = VideoCommon::RunAhead::landing_checks;
    const size_t slot = g_landing_slot++ % checks.size();
    auto& check = checks[slot];
    if (check.registered) {
        // The slot's earlier window: landings still pending there count as unchecked.
        waiters.Remove(&check);
        check.registered = false;
    }
    // A new generation before the latch is cleared: a reader that sees it cleared sees the
    // generation move as well (CheckConvertedLanding).
    const u32 generation = check.generation.load(std::memory_order_relaxed) + 1;
    check.generation.store(generation, std::memory_order_release);
    check.latched.store(false, std::memory_order_release);
    if (!waiters.Add(&check, device, acquire.value, acquire.operation)) {
        VideoCommon::RunAhead::counters.checks_unregistered += 1;
        return 0;
    }
    check.registered = true;
    // A release that landed before the check's waiter was added latched the acquire's own.
    if (waiters.Latched(waiter)) {
        check.latched.store(true, std::memory_order_release);
    }
    return VideoCommon::RunAhead::PackLandingCheck(slot, generation);
}

// EDEN_NX2_RUNAHEAD_W1: the acquire on `device` goes on before its fence-flagged release landed.
// The fence work the baseline's poll sends once the back ran what was sent before the acquire
// (MaybeReleaseFences: a fence for the reports that wait for the next one, those of the release's
// own entry included, and the deferred submit) goes out now, then the back holds until the release
// landed (GpuSplit::HoldBackUntilLanded).
//
// EDEN_NX2_RUNAHEAD_LANDINGS instead: no fence work and no hold. The back runs on, and makes every
// guest-visible landing of what follows a fence operation, run after the release's fence, until
// the window closes (GpuSplit::OpenLandingWindow). `waiter`: the acquire's own (still added).
void HoldForW1(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, u64 device,
               const void* waiter) {
    auto& split = VideoCommon::GpuSplit::Get();
    const auto entry = split.UnlandedFencedEntry(device);
    if (!entry) {
        return;
    }
    if (VideoCommon::RunAhead::landings) {
        const u64 check =
            VideoCommon::RunAhead::landings_verify ? StartLandingCheck(acquire, device, waiter) : 0;
        split.OpenLandingWindow(*entry, check, dma_pusher.rasterizer);
        return;
    }
    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
    split.HoldBackUntilLanded(*entry);
}

// EDEN_NX2_RUNAHEAD: a greater-or-equal acquire on the syncpoint shim (no device memory there:
// the view does not apply, and the baseline drains the split, then reads the syncpoint) that the
// guest syncpoint meets already. Syncpoints only grow, so the value read after the drain meets it
// too: the baseline's decision without the drain.
bool ShimAcquireMet(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire) {
    auto& memory = dma_pusher.memory_manager;
    if (acquire.operation != Nx2SemaphoreWaiters::AcquireGequal || (acquire.address & 3) != 0 ||
        !VideoCommon::GpuSplit::Get().Active() ||
        !memory.OverlapsSyncpointShim(acquire.address, sizeof(u32)) ||
        memory.GpuToCpuAddress(acquire.address)) {
        return false;
    }
    const bool met = acquire.Matches(memory.Read<u32>(acquire.address));
    if (VideoCommon::RunAhead::stats) {
        ++(met ? VideoCommon::RunAhead::counters.shim_met
               : VideoCommon::RunAhead::counters.shim_drained);
    }
    return met;
}

void DumpSemaphoreHistory(DmaPusher& dma_pusher, GPUVAddr address, DAddr device) {
    using namespace VideoCommon;
    if (!SemRecorder::TakeDump()) {
        return;
    }
    // Region membership is GPU-VA based; events and their pairs use device addresses so aliases
    // of this word share one history. Resolve both pages without assuming contiguous backing.
    const GPUVAddr mirror = Nx2SemaphoreMirror(address);
    const GPUVAddr paired = mirror != 0                        ? mirror
                            : Nx2InSemaphoreFirstPage(address) ? address + 0x1000
                                                               : 0;
    const u64 pair =
        paired != 0 ? dma_pusher.memory_manager.GpuToCpuAddress(paired).value_or(0) : 0;
    const auto history = SemRecorder::Snapshot(device, pair);
    LOG_WARNING(HW_GPU,
                "Sem history begin: gpu={:#x} word={:#x} pair={:#x} events={} end={} dropped={} "
                "busy={} untracked={}",
                address, device, pair, history.events.size(), history.end, history.dropped,
                history.busy, SemRecorder::Get().untracked.load(std::memory_order_relaxed));
    const auto log_event = [](const SemRecorder::Event& event) {
        LOG_WARNING(HW_GPU,
                    "Sem history: seq={} ns={} tid={} kind={} word={:#x} size={:#x} gpu={:#x} "
                    "pair={:#x} value={:#x} before={:#x} after={:#x} pos={:#x} entry={:#x} "
                    "consumed={:#x} fence={:#x} flags={:#x}",
                    event.sequence, event.timestamp, event.thread, event.kind, event.address,
                    event.size, event.gpu, event.pair, event.value, event.before, event.after,
                    event.position, event.entry, event.consumed, event.fence, event.flags);
    };
    for (const auto& event : history.events) {
        log_event(event);
    }
    log_event(GpuSplit::Get().SemaphoreRecordState(device, "state"));
    if (pair != 0) {
        log_event(GpuSplit::Get().SemaphoreRecordState(pair, "state"));
    }
    LOG_WARNING(HW_GPU, "Sem history end: word={:#x}", device);
}

// EDEN_NX2_ACQUIRE_VERIFY: whether a pass of the view stands against the drain path. One the
// drained word contradicts waits instead. A fenced release that landed during the drain can
// change the word the view read, as it can on the drain path itself: counted apart.
bool VerifyPass(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, const void* waiter,
                DAddr device, bool after_wait, VideoCommon::DrainSite site) {
    if (!VideoCommon::g_acquire_verify) {
        return true;
    }
    auto& stats = VideoCommon::g_acquire_stats;
    const bool fenced = FencedWritePending(device);
    VideoCommon::SemRecorder::Add({.kind = "verify-pass",
                                   .address = device,
                                   .gpu = acquire.address,
                                   .value = acquire.value,
                                   .flags = u64{after_wait} | (u64{fenced} << 1)});
    if (WatchSemaphore(acquire.address)) {
        const u32 guest = dma_pusher.memory_manager.Read<u32>(acquire.address);
        const auto queued = VideoCommon::GpuSplit::Get().QueuedSemaphoreWriteAt(device);
        LOG_INFO(HW_GPU,
                 "Semaphore watch: view pass {:#x} target {:#x} guest {:#x} "
                 "diverts {} dirty {} queued {} known {} fenced {} after_wait {}",
                 acquire.address, acquire.value, guest, Nx2MirrorShadow::Get().Diverts(device),
                 dma_pusher.rasterizer->MustFlushRegion(device, sizeof(u32),
                                                        VideoCommon::CacheType::BufferCache),
                 queued.has_value(), queued && queued->known, fenced, after_wait);
    }
    if (MetAfterDrain(dma_pusher, acquire, waiter, site)) {
        return true;
    }
    const bool fence_landed = fenced && !FencedWritePending(device);
    DumpSemaphoreHistory(dma_pusher, acquire.address, device);
    if (fence_landed) {
        VideoCommon::Count(stats.raced);
        return false;
    }
    VideoCommon::Count(stats.mismatches);
    if (after_wait) {
        VideoCommon::Count(stats.mismatches_waited);
    }
    if (static std::atomic<u32> logged{}; logged.fetch_add(1) < 16) {
        LOG_WARNING(HW_GPU,
                    "Acquire verify: channel {} passed semaphore {:#x} (op {}, {:#x}) without a "
                    "drain, the word is {:#x} after it",
                    dma_pusher.channel_state.bind_id, acquire.address, acquire.operation,
                    acquire.value, ReadSemaphoreMemory(dma_pusher, acquire.address, device));
    }
    return false;
}

// EDEN_NX2_ACQUIRE_VERIFY: whether a wait of the view stands against the drain path. One the
// drained word or a latch ends passes instead.
bool VerifyWait(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, const void* waiter,
                std::atomic<u64>& late, VideoCommon::DrainSite site) {
    if (!VideoCommon::g_acquire_verify || !MetAfterDrain(dma_pusher, acquire, waiter, site)) {
        return true;
    }
    VideoCommon::Count(late);
    return false;
}

} // Anonymous namespace

bool Puller::PendingAcquire::Matches(u32 current) const {
    switch (static_cast<GpuSemaphoreOperation>(operation)) {
    case GpuSemaphoreOperation::AcquireEqual:
        return current == value;
    case GpuSemaphoreOperation::AcquireGequal:
        return current >= value;
    case GpuSemaphoreOperation::AcquireMask:
        return (current & value) != 0;
    default:
        return true;
    }
}

bool Puller::PendingAcquire::WaitsToReach() const {
    return static_cast<GpuSemaphoreOperation>(operation) == GpuSemaphoreOperation::AcquireGequal;
}

bool Puller::CanResumeAcquire(DmaPusher& dma_pusher) {
    return Details::g_enabled ? CanResumeAcquireImpl<true>(dma_pusher)
                              : CanResumeAcquireImpl<false>(dma_pusher);
}

template <bool Track>
bool Puller::CanResumeAcquireImpl(DmaPusher& dma_pusher) {
    if (!pending_acquire || pending_acquire->met) {
        return true;
    }
    auto& stats = VideoCommon::g_acquire_stats;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
    if (device && ViewApplies()) {
        pending_acquire->met = PollWithoutDrain<Track>(dma_pusher, *device);
        return pending_acquire->met;
    }
    VideoCommon::Count(stats.drained);
    VideoCommon::Count(device ? stats.drain_settings : stats.drain_word);
    // The drain below runs the release's entry too.
    pending_acquire->early_position.reset();
    // The release may still be queued for the split GPU thread's back half (and the caches
    // ReadSemaphoreWord asks are its state).
    DrainBack();
    const u32 current = ReadSemaphoreWord(dma_pusher, pending_acquire->address);
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        if (static std::atomic<u32> rescued{};
            !pending_acquire->Matches(current) && rescued.fetch_add(1) < 64) {
            LOG_INFO(HW_GPU, "Channel {} met semaphore {:#x} value {:#x} at its release (now {:#x})",
                     dma_pusher.channel_state.bind_id, pending_acquire->address,
                     pending_acquire->value, current);
        }
        if constexpr (Track) {
            Details::NoteResume(this, *pending_acquire, Details::Resume::Drain);
        }
        return true;
    }
    if (pending_acquire->Matches(current)) {
        if constexpr (Track) {
            Details::NoteResume(this, *pending_acquire, Details::Resume::Drain);
        }
        return true;
    }
    // With delayed fence operations (Balanced or Accurate fences, accurate GPU), a release
    // queued after the acquire released the fences waits in the caches for the next fence. With
    // every channel waiting none comes: release them again now and then while waiting. (The
    // semaphore view, PollWithoutDrain, applies only while releases without a fence flag land at
    // once: never with delayed fence operations. Fence-flagged ones signal their own fence.)
    const bool delayed = Settings::IsGPUFenceBehaviorDefault()
                             ? Settings::IsGPULevelHigh()
                             : Settings::IsGPUFenceBehaviorBalanced() ||
                                   Settings::IsGPUFenceBehaviorAccurate();
    if (const auto now = std::chrono::steady_clock::now();
        delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
        pending_acquire->last_fences = now;
        ReleaseFences(dma_pusher);
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            pending_acquire->Matches(ReadSemaphoreWord(dma_pusher, pending_acquire->address))) {
            if constexpr (Track) {
                Details::NoteResume(this, *pending_acquire, Details::Resume::Drain);
            }
            return true;
        }
    }
    ReportLongWait(dma_pusher, *pending_acquire, current);
    return false;
}

template <bool Track>
bool Puller::PollWithoutDrain(DmaPusher& dma_pusher, u64 device) {
    using Kind = ViewAnswer::Kind;
    using VideoCommon::DrainSite;
    PendingAcquire& acquire = *pending_acquire;
    auto& split = VideoCommon::GpuSplit::Get();
    auto& stats = VideoCommon::g_acquire_stats;
    if (VideoCommon::RunAhead::landings) [[unlikely]] {
        split.CheckLandingWindow(dma_pusher.rasterizer); // Between methods, as at an acquire
    }
    // A release the back ran latches the waiter, even if a later one changed the word again.
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        VideoCommon::Count(stats.latched);
        if constexpr (Track) {
            Details::NoteResume(this, acquire, Details::Resume::Latch);
        }
        return true;
    }
    if (acquire.split_start != split.Starts()) {
        // Sent before the split restarted: everything sent before it ran (Stop joins the back).
        acquire.split_start = split.Starts();
        acquire.writers_before = 0;
        acquire.sent_at = 0;
        acquire.flush_position.reset();
        acquire.early_position.reset();
    }
    if (acquire.flush_position && split.Ran(*acquire.flush_position)) {
        acquire.flush_position.reset();
    }
    bool verified = false;
    if (!acquire.flush_position) {
        const ViewAnswer answer = AskView(dma_pusher, acquire, device);
        switch (answer.kind) {
        case Kind::Met:
        case Kind::MetByQueued:
            verified = VideoCommon::g_acquire_verify;
            if (!VerifyPass(dma_pusher, acquire, this, device, false, DrainSite::AcquirePoll)) {
                break;
            }
            if (answer.kind == Kind::MetByQueued) {
                // What the front reads after the acquire waits for the work sent before the
                // release.
                split.NoteSync();
                VideoCommon::Count(stats.by_queued);
                VideoCommon::Nx2AcquireStats::NoteQueuedPass(device);
            }
            VideoCommon::Count(stats.passed);
            if constexpr (Track) {
                Details::NoteResume(this, acquire,
                                    answer.kind == Kind::MetByQueued ? Details::Resume::Queued
                                                                     : Details::Resume::Memory);
            }
            if (acquire.early_position && FencedWritePending(device)) [[unlikely]] {
                ++VideoCommon::SyncSuspend::counters.deviations;
            }
            return true;
        case Kind::Flush:
            RequestFlush<Track>(dma_pusher, acquire, device);
            break;
        case Kind::NotMet:
        case Kind::Unknown:
            if ((VideoCommon::RunAhead::w1 || VideoCommon::RunAhead::landings) &&
                !VideoCommon::g_acquire_verify) [[unlikely]] {
                // EDEN_NX2_RUNAHEAD_W1: a fenced release sent since (another channel's) meets it.
                if (const auto queued = W1Release(split, acquire, device, answer, false)) {
                    if (VideoCommon::RunAhead::w1_verify) {
                        StartW1Check(acquire, device, queued->position, queued->payload, this);
                    }
                    HoldForW1(dma_pusher, acquire, device, this);
                    split.NoteSync();
                    VideoCommon::Count(stats.passed);
                    VideoCommon::Count(stats.by_queued);
                    VideoCommon::Nx2AcquireStats::NoteQueuedPass(device);
                    if constexpr (Track) {
                        Details::NoteResume(this, acquire, Details::Resume::Queued);
                    }
                    VideoCommon::RunAhead::counters.w1_poll_passes +=
                        VideoCommon::RunAhead::stats ? 1 : 0;
                    return true;
                }
            }
            // Poll again later, without waiting for the back: until it runs a release that meets
            // the acquire (which latches the waiter) or the work before `position`, nothing the
            // front sees meets it.
            break;
        }
    }
    if (acquire.early_position) [[unlikely]] {
        // Suspended before the back ran the release's entry (EDEN_NX2_SYNC_SUSPEND), while the
        // baseline still waited for it: until the back ran it the release cannot have landed,
        // so its 1 ms valve starts from then, as from the baseline's suspension.
        acquire.last_drain = std::chrono::steady_clock::now();
        if (split.Ran(*acquire.early_position)) {
            VideoCommon::SyncSuspend::NoteReached(acquire.early_since, acquire.early_sent,
                                                  split.Position());
            acquire.early_position.reset();
        }
    }
    // Work only a drain finishes (the back's idle hook submits what the host GPU writes) still
    // lands: a waiting acquire drains at most once per 1 ms (PantACRO4life's valve, b837e742c4,
    // against a Mario Kart World hang).
    if (const auto now = std::chrono::steady_clock::now();
        now - acquire.last_drain >= std::chrono::milliseconds{1}) {
        acquire.last_drain = now;
        if (VideoCommon::RunAhead::valve) [[unlikely]] {
            // EDEN_NX2_RUNAHEAD_VALVE: what the drain is for is the back's idle-hook submit (a
            // release that lands only once that work is on the host GPU: the Mario Kart World
            // hang) and the fence work; its read found the word met almost never (run I: 6 of
            // 2,101), and it emptied the front's lead each time. The back submits at its next
            // entry boundary instead, the fence work goes out once it ran everything before the
            // acquire, and the view and the latch go on deciding at every poll.
            split.RequestIdleHook();
            MaybeReleaseFences(dma_pusher, acquire);
            if (VideoCommon::RunAhead::stats) {
                auto& counts = VideoCommon::RunAhead::counters;
                ++counts.valve_requests;
                // A release that only a drain lands would show here first (and at 2 s below).
                counts.valve_wait_max_ns = (std::max)(
                    counts.valve_wait_max_ns,
                    static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(now - acquire.since)
                            .count()));
            }
            if (!VideoCommon::RunAhead::valve_verify) {
                ReportLongWait(dma_pusher, acquire, std::nullopt);
                return false;
            }
            // _VERIFY: the baseline's valve follows; a pass only its drain finds is counted.
        }
        VideoCommon::Count(stats.drained);
        VideoCommon::Count(stats.valve_drains);
        if constexpr (Track) {
            ++Details::Get().valves[Details::Operation(acquire.operation)];
        }
        const auto valve_begin = VideoCommon::RunAhead::stats
                                     ? VideoCommon::SyncSuspend::Clock::now()
                                     : VideoCommon::SyncSuspend::Clock::time_point{};
        DrainBack(DrainSite::AcquirePoll);
        if (VideoCommon::RunAhead::stats) [[unlikely]] {
            ++VideoCommon::RunAhead::counters.valve_drains;
            VideoCommon::RunAhead::counters.valve_ns += VideoCommon::SyncSuspend::Nanoseconds(
                VideoCommon::SyncSuspend::Clock::now() - valve_begin);
        }
        const u32 current = ReadSemaphoreWord(dma_pusher, acquire.address);
        if (Nx2SemaphoreWaiters::Get().Latched(this) || acquire.Matches(current)) {
            if (VideoCommon::RunAhead::valve) [[unlikely]] {
                ++VideoCommon::RunAhead::counters.valve_drain_passes;
                if (static std::atomic<u32> logged{}; logged.fetch_add(1) < 16) {
                    LOG_WARNING(HW_GPU,
                                "Run-ahead valve verify: semaphore {:#x} (op {}, {:#x}) met only "
                                "after the drain: {:#x}",
                                acquire.address, acquire.operation, acquire.value, current);
                }
            }
            VideoCommon::Count(stats.valve_passes);
            if constexpr (Track) {
                ++Details::Get().valve_passes[Details::Operation(acquire.operation)];
                Details::NoteResume(this, acquire, Details::Resume::Valve);
            }
            return true;
        }
        MaybeReleaseFences(dma_pusher, acquire);
        ReportLongWait(dma_pusher, acquire, current);
        return false;
    }
    if (!verified &&
        !VerifyWait(dma_pusher, acquire, this, stats.late_polls, DrainSite::AcquirePoll)) {
        if constexpr (Track) {
            Details::NoteResume(this, acquire, Details::Resume::Verify);
        }
        return true;
    }
    VideoCommon::Count(stats.polls);
    MaybeReleaseFences(dma_pusher, acquire);
    ReportLongWait(dma_pusher, acquire, std::nullopt);
    return false;
}

bool Puller::ResumeAcquire(DmaPusher& dma_pusher) {
    if (!CanResumeAcquire(dma_pusher)) {
        return false;
    }
    // Only a resume from a suspension counts: DmaPusher::Step calls this before every entry, and
    // counting those made the count odd ("suspended") at random, so submit-time snapshots were
    // thrown away for about half of all entries (CommandList::snapshot_suspensions).
    if (pending_acquire) {
        if (pending_acquire->early_position) [[unlikely]] {
            // Met before a poll saw the release's entry run (EDEN_NX2_SYNC_SUSPEND).
            const auto& split = VideoCommon::GpuSplit::Get();
            if (split.Ran(*pending_acquire->early_position)) {
                VideoCommon::SyncSuspend::NoteReached(pending_acquire->early_since,
                                                      pending_acquire->early_sent,
                                                      split.Position());
            } else {
                ++VideoCommon::SyncSuspend::counters.unreached_resumes;
            }
        }
        pending_acquire.reset();
        dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
        Nx2SemaphoreWaiters::Get().Remove(this);
    }
    return true;
}

void Puller::ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_DEBUG(HW_GPU, "Binding subchannel {} to engine {:#x}", method_call.subchannel, method_call.argument);
    u32 engine = method_call.argument;
    if ((engine & ~Gf100BindClassMask) != 0 && (engine & ~Gf100BindValidMask) == 0) {
        engine &= Gf100BindClassMask;
    }
    // Switch 2 binds Ampere classes; run them on the Maxwell engines (same method layout for
    // most state). Class 0 unbinds the subchannel.
    switch (engine & 0xFFFF) {
    case 0xC797: // AMPERE_B
        engine = static_cast<u32>(EngineID::MAXWELL_B);
        break;
    case 0xC7C0: // AMPERE_COMPUTE_B
        engine = static_cast<u32>(EngineID::KEPLER_COMPUTE_B);
        break;
    case 0xC7B5: // AMPERE_DMA_COPY_B
        engine = static_cast<u32>(EngineID::MAXWELL_DMA_COPY_A);
        break;
    case 0x0:
        if (g_nx2_ampere) {
            LOG_DEBUG(HW_GPU, "Switch 2 unbinds subchannel {}", method_call.subchannel);
            return;
        }
        break;
    default:
        break;
    }

    const auto engine_id = static_cast<EngineID>(engine);
    bound_engines[method_call.subchannel] = engine_id;
    switch (engine_id) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->fermi_2d, method_call.subchannel, EngineTypes::Fermi2D);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_3d, method_call.subchannel, EngineTypes::Maxwell3D);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_compute, method_call.subchannel, EngineTypes::KeplerCompute);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_dma, method_call.subchannel, EngineTypes::MaxwellDMA);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_memory, method_call.subchannel, EngineTypes::KeplerMemory);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->nv01_timer, method_call.subchannel, EngineTypes::Nv01Timer);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine {:04X}", engine_id);
        break;
    }
}

void Puller::ProcessFenceActionMethod(DmaPusher& dma_pusher) {
    switch (regs.fence_action.op) {
    case Puller::FenceOperation::Acquire:
        // UNIMPLEMENTED_MSG("Channel Scheduling pending.");
        // WaitFence(regs.fence_action.syncpoint_id, regs.fence_value);
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case Puller::FenceOperation::Increment:
        OnBack([rasterizer = dma_pusher.rasterizer,
                id = static_cast<u32>(
                    regs.fence_action.syncpoint_id.Value())] { rasterizer->SignalSyncPoint(id); },
               FenceWrites);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented operation {}", regs.fence_action.op.Value());
        break;
    }
}

void Puller::ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher) {
    if (Details::g_enabled) {
        ProcessSemaphoreTriggerMethodImpl<true>(dma_pusher);
    } else {
        ProcessSemaphoreTriggerMethodImpl<false>(dma_pusher);
    }
}

template <bool Track>
void Puller::ProcessSemaphoreTriggerMethodImpl(DmaPusher& dma_pusher) {
    const auto semaphoreOperationMask = 0xF;
    const auto op = GpuSemaphoreOperation(regs.semaphore_trigger & semaphoreOperationMask);
    // EDEN_NX2_SEM_WATCH=<gpu address>: log every release/acquire on the 16 bytes there.
    if (WatchSemaphore(regs.semaphore_address.SemaphoreAddress())) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} as {} device {:#x} op {} {:#x} value {:#x} (now {:#x})",
                 dma_pusher.channel_state.bind_id, dma_pusher.memory_manager.GetID(),
                 dma_pusher.memory_manager.GpuToCpuAddress(regs.semaphore_address.SemaphoreAddress()).value_or(0),
                 static_cast<u32>(op),
                 regs.semaphore_address.SemaphoreAddress(), regs.semaphore_sequence,
                 dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress()));
    }
    if (op == GpuSemaphoreOperation::WriteLong) {
        const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
        const u32 payload = regs.semaphore_sequence;
        // RELEASE_SIZE (bit 24) set means a 4-byte release without the timestamp; writing 16
        // bytes there clobbers the words next to it (Switch 2 packs semaphores tightly).
        const auto properties = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0
                                    ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout;
        const GPUVAddr mirror = g_nx2_ampere ? Nx2SemaphoreMirror(sequence_address) : 0;
        if (mirror != 0) {
            LOG_DEBUG(HW_GPU, "Switch 2 semaphore {:#x}={:#x} mirrored to {:#x}", sequence_address, payload, mirror);
        }
        OnBack(
            [rasterizer = dma_pusher.rasterizer, sequence_address, mirror, properties, payload] {
                NoteReleaseForFastDraw(rasterizer, sequence_address);
                rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload, properties,
                                  payload, 0);
                if (mirror != 0) {
                    NoteReleaseForFastDraw(rasterizer, mirror);
                    rasterizer->Query(mirror, VideoCommon::QueryType::Payload,
                                      properties | VideoCommon::QueryPropertiesFlags::Nx2Mirror,
                                      payload, 0);
                }
            },
            VideoCommon::GpuSplit::Writes::Noted, VideoCommon::Nx2FastDraw::CallFlags());
        NoteRelease(dma_pusher, sequence_address, payload, properties);
        if (mirror != 0) {
            // The mirrored word is what GPU acquires read, in guest memory or the shadow.
            NoteRelease(dma_pusher, mirror, payload,
                        properties | VideoCommon::QueryPropertiesFlags::Nx2Mirror);
        }
    } else if (g_nx2_ampere) {
        // Switch 2 waits on semaphores the CPU (or another channel) releases, e.g. once it has
        // written the commands this channel runs next: suspend the channel until the value is
        // met instead of reading ahead (approach from changes.patch).
        PendingAcquire acquire{
            .address = regs.semaphore_address.SemaphoreAddress(),
            .value = regs.semaphore_sequence,
            .operation = static_cast<u32>(op),
            .since = std::chrono::steady_clock::now(),
            .last_fences = std::chrono::steady_clock::now(),
            .last_drain = std::chrono::steady_clock::now(),
            .reported = false,
        };
        if (op != GpuSemaphoreOperation::AcquireEqual && op != GpuSemaphoreOperation::AcquireGequal &&
            op != GpuSemaphoreOperation::AcquireMask) {
            LOG_ERROR(HW_GPU, "Invalid semaphore operation {}", static_cast<u32>(op));
            return;
        }
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireChecks, 1);
        Details::AcquireScope<Track> capture{acquire};
        SCOPE_EXIT {
            capture.Finish(acquire, dma_pusher.memory_manager, pending_acquire.has_value());
        };
        if (AcquireWithoutDrain<Track>(dma_pusher, acquire, capture.Data())) {
            return;
        }
        if (VideoCommon::RunAhead::enabled && ShimAcquireMet(dma_pusher, acquire)) [[unlikely]] {
            if constexpr (Track) {
                capture.Data()->decision = Details::Decision::Memory;
            }
            return;
        }
        auto& stats = VideoCommon::g_acquire_stats;
        VideoCommon::Count(stats.drained);
        VideoCommon::Count(dma_pusher.memory_manager.GpuToCpuAddress(acquire.address)
                               ? stats.drain_settings
                               : stats.drain_word);
        DrainBack();
        if (acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
            return;
        }
        // Registered before the fences run, so a release among them that meets the value counts
        // even if a later one overwrites it.
        auto& waiters = Nx2SemaphoreWaiters::Get();
        if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
            device && !waiters.Add(this, *device, acquire.value, acquire.operation)) {
            if (static std::atomic<u32> full{}; full.fetch_add(1) < 8) {
                LOG_ERROR(HW_GPU, "Channel {} waits on semaphore {:#x} unlatched: {} channels wait",
                          dma_pusher.channel_state.bind_id, acquire.address,
                          Nx2SemaphoreWaiters::MaxWaiters);
            }
        }
        ReleaseFences(dma_pusher);
        if (!waiters.Latched(this) && !acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
            if (VideoCommon::SyncTrace::enabled) {
                VideoCommon::SyncTrace::Suspend(this);
            }
            pending_acquire = acquire;
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
        } else {
            waiters.Remove(this);
        }
    } else {
        DrainBack();
        do {
            const u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
            regs.acquire_source = true;
            regs.acquire_value = regs.semaphore_sequence;
            if (op == GpuSemaphoreOperation::AcquireEqual) {
                regs.acquire_active = true;
                regs.acquire_mode = false;
                if (word != regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireGequal) {
                regs.acquire_active = true;
                regs.acquire_mode = true;
                if (word < regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (word && regs.semaphore_sequence == 0) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else {
                LOG_ERROR(HW_GPU, "Invalid semaphore operation");
            }
        } while (false);
    }
}

template <bool Track, typename Observation>
bool Puller::AcquireWithoutDrain(DmaPusher& dma_pusher, PendingAcquire& acquire,
                                 Observation* observed) {
    using Kind = ViewAnswer::Kind;
    using VideoCommon::DrainSite;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device || !ViewApplies()) {
        return false;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    auto& stats = VideoCommon::g_acquire_stats;
    acquire.split_start = split.Starts();
    acquire.writers_before = split.LastWriterPosition();
    acquire.sent_at = split.Position();
    // EDEN_NX2_RUNAHEAD: only the entries that can change the word count as its writers.
    namespace RunAhead = VideoCommon::RunAhead;
    const u64 base_writers = acquire.writers_before;
    RunAhead::Answer reach{base_writers, RunAhead::Reach::None};
    if (RunAhead::enabled) [[unlikely]] {
        reach = split.RunAheadReach(*device, base_writers);
        acquire.writers_before = reach.position;
    }
    // Registered before the view reads the word: a release the back runs from now on latches the
    // waiter even if a later one overwrites it before a poll looks (no drain orders them).
    auto& waiters = Nx2SemaphoreWaiters::Get();
    acquire.registered = waiters.Add(this, *device, acquire.value, acquire.operation);
    if (!acquire.registered) {
        if (static std::atomic<u32> full{}; full.fetch_add(1) < 8) {
            LOG_ERROR(HW_GPU, "Channel {} waits on semaphore {:#x} unlatched: {} channels wait",
                      dma_pusher.channel_state.bind_id, acquire.address,
                      Nx2SemaphoreWaiters::MaxWaiters);
        }
    }
    if (RunAhead::w1_verify) [[unlikely]] {
        SweepW1Checks(false);
    }
    if (RunAhead::landings) [[unlikely]] {
        // Between methods: no release is half sent (the notes follow the window,
        // OpenLandingWindow).
        split.CheckLandingWindow(dma_pusher.rasterizer);
        if (RunAhead::landings_verify) {
            SweepLandingChecks();
        }
    }
    ViewAnswer answer = AskView(dma_pusher, acquire, *device);
    // EDEN_NX2_RUNAHEAD_STATS / _VERIFY: whether the baseline would wait for its writers here (the
    // view's answer with them), where the reach lets this acquire go on sooner.
    bool base_waits = false;
    if (RunAhead::stats) [[unlikely]] {
        ++RunAhead::counters.acquires;
        const u64 lead = split.Backlog();
        RunAhead::counters.lead_sum += lead;
        RunAhead::counters.lead_max = (std::max)(RunAhead::counters.lead_max, lead);
        if (reach.position < base_writers) {
            ++RunAhead::counters.earlier;
            PendingAcquire base = acquire;
            base.writers_before = base_writers;
            const ViewAnswer base_answer = AskView(dma_pusher, base, *device);
            base_waits = base_answer.kind == Kind::Unknown && !base_answer.write_dependency &&
                         base_answer.position > reach.position;
        }
    }
    if ((RunAhead::w1 || RunAhead::landings) && !VideoCommon::g_acquire_verify &&
        ((answer.kind == Kind::Unknown && answer.write_dependency) || answer.kind == Kind::NotMet))
        [[unlikely]] {
        // EDEN_NX2_RUNAHEAD_W1: the last release sent to the word lands once a fence ran (W1),
        // and its payload meets the acquire. The baseline waits for that landing, then goes on
        // (the release latches the waiter as it lands, even if a later write changes the word):
        // its decision is to go on, and only the time differs. The release is in the ring before
        // the acquire, and the back runs nothing sent after it until the release landed
        // (HoldBackUntilLanded): releases that land as the back runs them, and its memory writes,
        // keep the baseline's order. The front's reads after the acquire have waits of their own
        // (GPU-written lists after NoteSync, macro DMA reads drain through the hold, QMD reads),
        // and words the CPU writes once the release landed cannot be ordered against the GPU's
        // reads after an acquire on it.
        if (const auto queued = W1Release(split, acquire, *device, answer, true)) {
            if (RunAhead::w1_verify) {
                StartW1Check(acquire, *device, queued->position, queued->payload, this);
            }
            HoldForW1(dma_pusher, acquire, *device, this);
            waiters.Remove(this);
            // What the front reads after the acquire waits for the work sent before the release,
            // as after a pass on a queued release (GpuSplit::NoteSync).
            split.NoteSync();
            VideoCommon::Count(stats.passed);
            VideoCommon::Count(stats.by_queued);
            VideoCommon::Nx2AcquireStats::NoteQueuedPass(*device);
            if constexpr (Track) {
                observed->decision = Details::Decision::Queued;
            }
            RunAhead::counters.w1_passes += RunAhead::stats ? 1 : 0;
            return true;
        }
    }
    bool waited = false;
    if (VideoCommon::SyncSuspend::enabled) [[unlikely]] {
        ++VideoCommon::SyncSuspend::counters.acquires;
        if (answer.kind == Kind::Unknown && answer.write_dependency &&
            SuspendEarly<Track>(dma_pusher, acquire, *device, answer, waited, observed)) {
            return true;
        }
    }
    // The value is in memory once the back ran that work, before what was sent after it. A
    // release the front can't value, then the writers before the acquire: two waits at most.
    for (u32 waits = 0; answer.kind == Kind::Unknown && waits < 2; ++waits) {
        VideoCommon::Count(stats.positioned);
        bool actually_waited;
        const auto wait_begin = VideoCommon::SyncSuspend::enabled || RunAhead::stats
                                    ? VideoCommon::SyncSuspend::Clock::now()
                                    : VideoCommon::SyncSuspend::Clock::time_point{};
        if constexpr (Track) {
            auto wait = observed->Begin(split, *device, answer.position, acquire.operation,
                                        answer.write_dependency);
            const auto begin = Details::Clock::now();
            actually_waited = split.WaitForPosition(answer.position, DrainSite::Acquire);
            observed->End(wait, begin, actually_waited);
        } else {
            actually_waited = split.WaitForPosition(answer.position, DrainSite::Acquire);
        }
        if (VideoCommon::SyncSuspend::enabled) [[unlikely]] {
            VideoCommon::SyncSuspend::NoteWait(wait_begin);
        }
        if (RunAhead::stats && !answer.write_dependency) [[unlikely]] {
            // A wait for the writers (the reach): counted by what the reach found.
            ++RunAhead::counters.waits;
            ++RunAhead::counters.wait_reasons[static_cast<size_t>(reach.reason)];
            RunAhead::counters.wait_ns += VideoCommon::SyncSuspend::Nanoseconds(
                VideoCommon::SyncSuspend::Clock::now() - wait_begin);
        }
        if (actually_waited) {
            VideoCommon::Count(stats.positioned_waited);
        }
        waited = true;
        answer = AskView(dma_pusher, acquire, *device);
    }
    if constexpr (Track) {
        observed->decision = Details::Decision::Suspended;
    }
    bool pass = false;
    switch (answer.kind) {
    case Kind::Met:
    case Kind::MetByQueued:
        pass = VerifyPass(dma_pusher, acquire, this, *device, waited, DrainSite::Acquire);
        if (pass && base_waits && !split.Ran(base_writers)) [[unlikely]] {
            // The baseline would still wait for its writers here.
            ++RunAhead::counters.sooner;
            if (RunAhead::verify) {
                split.Defer(RunAheadCheck{dma_pusher.rasterizer, &dma_pusher.memory_manager,
                                          acquire.address, *device, acquire.sent_at, reach.position,
                                          base_writers, acquire.value, acquire.operation,
                                          reach.reason, FencedWritePending(*device)},
                            VideoCommon::GpuSplit::FastDrawNeutral,
                            VideoCommon::GpuSplit::Writes::Noted);
            }
        }
        if (pass && answer.kind == Kind::MetByQueued) {
            // What the front reads after the acquire waits for the work sent before the release
            // (GpuSplit::NoteSync).
            split.NoteSync();
            VideoCommon::Count(stats.by_queued);
            VideoCommon::Nx2AcquireStats::NoteQueuedPass(*device);
        }
        if (pass) {
            if constexpr (Track) {
                observed->decision = waited                             ? Details::Decision::Waited
                                     : answer.kind == Kind::MetByQueued ? Details::Decision::Queued
                                                                        : Details::Decision::Memory;
            }
            VideoCommon::Count(stats.passed);
            if (waited) {
                VideoCommon::Count(stats.after_wait);
            }
        }
        break;
    case Kind::Flush:
        RequestFlush<Track>(dma_pusher, acquire, *device);
        break;
    case Kind::NotMet:
    case Kind::Unknown:
        break;
    }
    if (!pass && waiters.Latched(this)) {
        if constexpr (Track) {
            observed->decision = Details::Decision::Latched;
        }
        VideoCommon::Count(stats.latched);
        pass = true;
    }
    if (!pass && !VerifyWait(dma_pusher, acquire, this, stats.late, DrainSite::Acquire)) {
        if constexpr (Track) {
            observed->decision = Details::Decision::Drained;
        }
        pass = true;
    }
    if (pass) {
        waiters.Remove(this);
        return true;
    }
    Suspend(dma_pusher, acquire);
    return true;
}

void Puller::Suspend(DmaPusher& dma_pusher, PendingAcquire& acquire) {
    // Suspended: a release the back runs latches the waiter, and the polls see the others in
    // memory or in the view (PollWithoutDrain).
    VideoCommon::Count(VideoCommon::g_acquire_stats.suspended);
    if (VideoCommon::SyncTrace::enabled) {
        VideoCommon::SyncTrace::Suspend(this);
    }
    acquire.last_drain = std::chrono::steady_clock::now();
    MaybeReleaseFences(dma_pusher, acquire);
    pending_acquire = acquire;
    dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
}

template <bool Track, typename Answer, typename Observation>
bool Puller::SuspendEarly(DmaPusher& dma_pusher, PendingAcquire& acquire, u64 device,
                          Answer& answer, bool& waited, Observation* observed) {
    namespace Suspension = VideoCommon::SyncSuspend;
    using Kind = ViewAnswer::Kind;
    using Decision = Suspension::Decision;
    auto& split = VideoCommon::GpuSplit::Get();
    auto& counts = Suspension::counters;
    // The fence-flagged release the view waits for, as long as that is what it waits for.
    const auto release = [&]() -> std::optional<VideoCommon::GpuSplit::QueuedSemaphoreWrite> {
        if (answer.kind != Kind::Unknown || !answer.write_dependency) {
            return std::nullopt;
        }
        const auto queued = split.QueuedSemaphoreWriteAt(device);
        if (!queued || !queued->fenced || queued->position != answer.position) {
            return std::nullopt;
        }
        return queued;
    };
    // The verifier drains at each decision: it compares the baseline path only.
    auto queued = VideoCommon::g_acquire_verify ? std::nullopt : release();
    if (!queued) {
        return false;
    }
    ++counts.fenced;
    // After the wait, the baseline passes on a latch, or on the word in memory as the view reads
    // it while the release is pending (a mirror shadow an older fence callback may still remove
    // doesn't count: AskView). Anything else suspends unless the release lands during the wait.
    const auto met = [&] {
        if (Nx2SemaphoreWaiters::Get().Latched(this)) {
            return true;
        }
        bool shadow_used = false;
        const u32 value = ReadSemaphoreMemory(dma_pusher, acquire.address, device, &shadow_used);
        return !shadow_used && acquire.Matches(value);
    };
    const auto decide = [&] {
        return Suspension::Decide(
            Suspension::WorkQueued(), [&] { return split.Ran(queued->settle); }, met);
    };
    Decision decision = decide();
    if (decision == Decision::Settle) {
        // Work sent before the release may still change the word as the back runs it: the
        // baseline's wait passes it first, so wait for it here too, then ask again.
        VideoCommon::Count(VideoCommon::g_acquire_stats.positioned);
        const auto begin = Suspension::Clock::now();
        const bool actually_waited =
            split.WaitForPosition(queued->settle, VideoCommon::DrainSite::Acquire);
        ++counts.settle_waits;
        counts.settle_ns += Suspension::Nanoseconds(Suspension::Clock::now() - begin);
        if (actually_waited) {
            VideoCommon::Count(VideoCommon::g_acquire_stats.positioned_waited);
        }
        waited = waited || actually_waited;
        answer = AskView(dma_pusher, acquire, device);
        queued = release();
        decision = queued ? decide() : Decision::Settle;
        if (decision == Decision::Settle) {
            ++counts.changed;
            return false;
        }
    }
    switch (decision) {
    case Decision::NoWork:
        ++counts.no_work;
        return false;
    case Decision::Met:
        ++counts.met;
        return false;
    case Decision::Settle:
        ++counts.changed;
        return false;
    case Decision::Suspend:
        break;
    }
    acquire.early_position = answer.position;
    acquire.early_since = Suspension::Clock::now();
    acquire.early_sent = split.Position();
    ++counts.early;
    if constexpr (Track) {
        observed->decision = Details::Decision::Suspended;
    }
    Suspend(dma_pusher, acquire);
    return true;
}

void Puller::ProcessSemaphoreRelease(DmaPusher& dma_pusher) {
    const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
    const u32 payload = regs.semaphore_release;
    OnBack(
        [rasterizer = dma_pusher.rasterizer, sequence_address, payload] {
            NoteReleaseForFastDraw(rasterizer, sequence_address);
            rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload,
                              VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
        },
        VideoCommon::GpuSplit::Writes::Noted, VideoCommon::Nx2FastDraw::CallFlags());
    NoteRelease(dma_pusher, sequence_address, payload, VideoCommon::QueryPropertiesFlags::IsAFence);
}

void Puller::ProcessSemaphoreAcquire(DmaPusher& dma_pusher) {
    DrainBack();
    // The CPU may rewrite what the next draws read once it sees the release this waits for.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
    const auto value = regs.semaphore_acquire;
    while (word != value) {
        regs.acquire_active = true;
        regs.acquire_value = value;
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
        // TODO(kemathe73) figure out how to do the acquire_timeout
        regs.acquire_mode = false;
        regs.acquire_source = false;
    }
}

// Ampere SEM_EXECUTE (method 0x1B): operation in bits 2:0 on the semaphore at
// SEM_ADDR_HI:SEM_ADDR_LO (methods 0x18/0x17) with payload SEM_PAYLOAD_LO (0x19).
void Puller::ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument) {
    const GPUVAddr address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                             (regs.reg_array[0x17] & ~3U);
    const u32 payload = regs.reg_array[0x19];
    const u32 operation = argument & 7;
    if (operation == 1) { // RELEASE
        OnBack(
            [rasterizer = dma_pusher.rasterizer, address, payload] {
                NoteReleaseForFastDraw(rasterizer, address);
                rasterizer->Query(address, VideoCommon::QueryType::Payload,
                                  VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
            },
            VideoCommon::GpuSplit::Writes::Noted, VideoCommon::Nx2FastDraw::CallFlags());
        NoteRelease(dma_pusher, address, payload, VideoCommon::QueryPropertiesFlags::IsAFence);
        return;
    }
    // Acquires: never block the GPU thread on them; they are checked for the warning below only.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    const auto ready = [operation, payload](u32 value) {
        return operation == 0   ? value == payload
               : operation == 2 ? value >= payload
               : operation == 3 ? static_cast<s32>(value - payload) >= 0
               : operation == 4 ? (value & payload) != 0
               : operation == 5 ? (value & payload) == 0
                                : true;
    };
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    std::optional<u32> word;
    if (device && ViewApplies()) {
        // The last release sent to the word tells while the back hasn't run it, else memory as a
        // GPU acquire reads it (through the mirror's shadow). What the front reads after the
        // acquire waits for the work sent before it, as after a drain.
        auto& split = VideoCommon::GpuSplit::Get();
        split.NoteSync();
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
        const auto queued = split.QueuedSemaphoreWriteAt(*device);
        if (!queued || queued->fenced) {
            word = ReadSemaphoreMemory(dma_pusher, address, device);
        } else if (queued->known && !queued->at_least) {
            word = queued->payload;
        }
    } else {
        DrainBack();
        ReleaseFences(dma_pusher);
        word = ReadSemaphoreMemory(dma_pusher, address, device);
    }
    if (word && !ready(*word)) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(HW_GPU, "SEM_EXECUTE op {} at {:#x}: {:#x} vs {:#x} not met, continuing",
                        operation, address, *word, payload);
        }
    }
}

/// Calls a GPU puller method.
void Puller::CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    static const bool nx2_trace = std::getenv("EDEN_NX2_PULLER_TRACE") != nullptr;
    if (nx2_trace) {
        LOG_INFO(HW_GPU, "nx2 puller sub={} method={:#x} arg={:#x}", method_call.subchannel,
                 method_call.method, method_call.argument);
    }
    regs.reg_array[method_call.method] = method_call.argument;
    const auto method = static_cast<BufferMethods>(method_call.method);

    switch (method) {
    case BufferMethods::BindObject: {
        ProcessBindMethod(dma_pusher, method_call);
        break;
    }
    case BufferMethods::Nop:
    case BufferMethods::SemaphoreAddressHigh:
    case BufferMethods::SemaphoreAddressLow:
    case BufferMethods::SemaphoreSequencePayload:
    case BufferMethods::SyncpointPayload:
    case BufferMethods::WrcacheFlush:
        break;
    case BufferMethods::RefCnt:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->SignalReference(); },
               FenceWrites);
        break;
    case BufferMethods::SyncpointOperation:
        ProcessFenceActionMethod(dma_pusher);
        break;
    case BufferMethods::WaitForIdle:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->WaitForIdle(); },
               VideoCommon::GpuSplit::Writes::Caches);
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case BufferMethods::SemaphoreOperation: {
        ProcessSemaphoreTriggerMethod(dma_pusher);
        break;
    }
    case BufferMethods::NonStallInterrupt: {
        LOG_ERROR(HW_GPU, "Special puller engine method NonStallInterrupt not implemented");
        break;
    }
    case BufferMethods::MemOpA: {
        LOG_TRACE(HW_GPU, "Memory Operation A");
        break;
    }
    case BufferMethods::MemOpB: {
        // Implement this better.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); },
               VideoCommon::GpuSplit::Writes::Caches);
        break;
    }
    case BufferMethods::MemOpC:
    case BufferMethods::MemOpD: {
        LOG_TRACE(HW_GPU, "Memory Operation C,D");
        break;
    }
    case BufferMethods::SemaphoreAcquire: {
        // On Ampere host classes 0x17..0x1A are SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI.
        if (!g_nx2_ampere) {
            ProcessSemaphoreAcquire(dma_pusher);
        }
        break;
    }
    case BufferMethods::SemaphoreRelease: {
        if (g_nx2_ampere) {
            ProcessSemaphoreExecute(dma_pusher, method_call.argument);
        } else {
            ProcessSemaphoreRelease(dma_pusher);
        }
        break;
    }
    case BufferMethods::Yield: {
        // TODO(Kmather73): Research and implement this method.
        LOG_ERROR(HW_GPU, "Special puller engine method Yield not implemented");
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Special puller engine method {:X} not implemented", method);
        if (static bool dumped = false; g_nx2_ampere && !dumped) {
            dumped = true;
            dma_pusher.DumpCurrentEntry("unknown host method");
        }
        break;
    }
}

/// Calls a GPU engine method.
static bool Nx2MethodTrace() {
    static const bool trace = std::getenv("EDEN_NX2_METHOD_TRACE") != nullptr;
    return trace;
}

void Puller::CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    const EngineID engine = bound_engines[method_call.subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 method sub={} engine={:04X} method={:#x} arg={:#x}",
                 method_call.subchannel, static_cast<u32>(engine), method_call.method,
                 method_call.argument);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU engine multivalue method.
void Puller::CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    const EngineID engine = bound_engines[subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 multimethod sub={} engine={:04X} method={:#x} amount={} first={:#x}",
                 subchannel, static_cast<u32>(engine), method, amount, base_start[0]);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU method.
void Puller::CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method_call.method, method_call.subchannel);
    ASSERT(method_call.subchannel < bound_engines.size());

    if (ExecuteMethodOnEngine(dma_pusher, method_call.method)) {
        CallEngineMethod(dma_pusher, method_call);
    } else {
        CallPullerMethod(dma_pusher, method_call);
    }
}

/// Calls a GPU multivalue method.
void Puller::CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method, subchannel);
    ASSERT(subchannel < bound_engines.size());
    if (ExecuteMethodOnEngine(dma_pusher, method)) {
        CallEngineMultiMethod(dma_pusher, method, subchannel, base_start, amount, methods_pending);
    } else {
        for (u32 i = 0; i < amount; i++) {
            CallPullerMethod(dma_pusher, MethodCall{
                method,
                base_start[i],
                subchannel,
                methods_pending - i,
            });
        }
    }
}

/// Determines where the method should be executed.
[[nodiscard]] bool Puller::ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method) {
    const auto buffer_method = BufferMethods(method);
    return buffer_method >= BufferMethods::NonPullerMethods;
}

} // namespace Tegra::Engines
