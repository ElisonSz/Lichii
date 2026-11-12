// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "common/common_types.h"

/// Page protection of rasterizer-cached guest memory (Memory::RasterizerMarkRegionCached): who
/// changes it and how often, and fewer changes.
namespace VideoCommon::Nx2Protect {

/// EDEN_NX2_PROTECT_STATS=1: counts protection changes by source, guest CPU traps on cached pages
/// and the pages the buffer cache's upload scans track again, in one "Nx2Protect" line every 10 s.
inline bool g_stats{};
/// EDEN_NX2_PROTECT_INLINE=1: guest memory the GPU thread wrote itself (FlushCaching) is uploaded
/// into the buffer pages that hold it when those are tracked, instead of untracking them
/// (BufferCache::WriteGpuMemoryInPlace): no unprotect now and no protect at the next upload scan.
inline bool g_inline{};

/// Called by VideoCore::CreateGPU, after nx2_switches.txt has been loaded.
void ReadSwitch();

/// What changed the cached-page counts that changed a protection (stats only). The innermost
/// SourceScope of the calling thread.
enum class Source : u8 {
    Other,      ///< none below: unmaps, buffer deletion and garbage collection, texture uploads
    CpuWrite,   ///< a guest CPU write trap (RasterizerVulkan::OnCPUWrite)
    GpuWrite,   ///< the GPU thread's own writes, applied at FlushCaching (InnerInvalidation)
    Collected,  ///< CPU writes into GPU-written pages, applied later (OnCacheInvalidation)
    Invalidate, ///< RasterizerVulkan::InvalidateRegion (device memory writes, services)
    Inline,     ///< inline-to-memory applied at once (AccelerateInlineToMemory)
    Rearm,      ///< an upload scan tracking pages again (WordManager::ForEachModifiedRange)
    Count,
};

/// Which tracker's count changed: the buffer cache's (batched) or a single range (textures,
/// shaders: DeviceMemoryManager::UpdatePagesCachedCount).
enum class Tracker : u8 {
    Buffer,
    Single,
    Count,
};

inline thread_local Source t_source = Source::Other;
inline thread_local Tracker t_tracker = Tracker::Buffer;
/// EDEN_NX2_PROTECT_INLINE: the running FlushCaching is a compute dispatch's, which ends the render
/// pass anyway, so an in-place upload may end it too (DispatchFlushScope).
inline thread_local bool t_dispatch_flush = false;

/// Around RasterizerVulkan::DispatchCompute's FlushCaching (one branch without the switch).
class DispatchFlushScope {
public:
    DispatchFlushScope() noexcept {
        if (g_inline) [[unlikely]] {
            t_dispatch_flush = true;
            active = true;
        }
    }
    ~DispatchFlushScope() {
        if (active) [[unlikely]] {
            t_dispatch_flush = false;
        }
    }
    DispatchFlushScope(const DispatchFlushScope&) = delete;
    DispatchFlushScope& operator=(const DispatchFlushScope&) = delete;

private:
    bool active{};
};

/// Sets the calling thread's source until the end of the scope (stats only, else one branch).
class SourceScope {
public:
    explicit SourceScope(Source source) noexcept {
        if (g_stats) [[unlikely]] {
            previous = t_source;
            t_source = source;
            active = true;
        }
    }
    ~SourceScope() {
        if (active) [[unlikely]] {
            t_source = previous;
        }
    }
    SourceScope(const SourceScope&) = delete;
    SourceScope& operator=(const SourceScope&) = delete;

private:
    Source previous{};
    bool active{};
};

class TrackerScope {
public:
    explicit TrackerScope(Tracker tracker) noexcept {
        if (g_stats) [[unlikely]] {
            previous = t_tracker;
            t_tracker = tracker;
            active = true;
        }
    }
    ~TrackerScope() {
        if (active) [[unlikely]] {
            t_tracker = previous;
        }
    }
    TrackerScope(const TrackerScope&) = delete;
    TrackerScope& operator=(const TrackerScope&) = delete;

private:
    Tracker previous{};
    bool active{};
};

/// Steady clock in nanoseconds.
[[nodiscard]] u64 NowNs() noexcept;

/// A HostMemory::Protect call of RasterizerMarkRegionCached: `cached` takes access away.
/// `cpu_core`: on a guest CPU core thread. `now`: NowNs() after the call.
void NoteProtect(bool cached, u64 size, u64 ns, bool cpu_core, u64 now);

/// A guest CPU write of a rasterizer-cached page (Memory::HandleRasterizerWrite).
void NoteCpuWrite(u64 vaddr);
/// The write was not in the core's last area: it reached the caches (GPU::OnCPUWrite), which
/// collected it for later when `collected`.
void NoteCpuWriteAsked(bool collected);
/// A guest CPU read of a rasterizer-cached page (Memory::HandleRasterizerDownload).
void NoteCpuRead(u64 vaddr);
/// The read was not in the core's last area: it asked the caches (GPU::OnCPURead).
void NoteCpuReadAsked();

/// An upload scan tracks these device runs again (each holds whole pages).
void NoteRearm(std::span<const std::pair<u64, std::size_t>> runs);
/// An upload scan over [begin, end) (device addresses), before its clean words are skipped.
void NoteScan(DAddr begin, DAddr end);

/// The last Nx2Protect line logged (tests).
[[nodiscard]] std::string LastLogLine();

/// What BufferCache::WriteGpuMemoryInPlace did with one range of the GPU's own writes.
enum class InPlace : u8 {
    Uploaded,       ///< its pages were uploaded and stay tracked
    Repeated,       ///< its pages were uploaded for an earlier range of the same flush
    Large,          ///< more pages than one upload takes: untracked as before
    NoBuffer,       ///< no single buffer holds its pages: untracked as before
    GpuData,        ///< GPU-written or preflushable data on its pages: as before
    Untracked,      ///< a page is not tracked (CPU-modified or never synchronized): as before
    AgainBeforeUse, ///< uploaded in place before and no upload scan since: untracked as before
    RenderPass,     ///< the upload would end the render pass: untracked as before
    Count,
};
void NoteInPlace(InPlace result, u64 bytes);

/// Time since a page's previous rearm (ChurnTable buckets).
enum class Interval : u8 {
    First, ///< not rearmed in the last 20 s
    Under4ms,
    Under12ms,
    Under25ms,
    Under50ms,
    Under100ms,
    Later,
    Count,
};

/// Rearm history per device page: how long since its previous rearm, and how many rearms in a
/// row came at most StreakNs apart (about two frames at 50 fps): pages the guest CPU writes
/// every frame build long streaks.
class ChurnTable {
public:
    static constexpr u64 StreakNs = 40'000'000;
    static constexpr u64 ForgetNs = 20'000'000'000;

    struct Result {
        Interval interval{};
        u32 streak{};          ///< rearms in a row before this one, each within StreakNs
        bool first_in_window{}; ///< the page's first rearm in this window
        bool steady{};          ///< streak reached 3 for the first time in this window
    };

    Result Note(u64 page, u64 now, u32 window);

    /// Drops pages not rearmed for ForgetNs.
    void Prune(u64 now);

    struct Top {
        u64 page{};
        u32 rearms{};
        u32 streak{};
    };
    /// Fills `out` with the pages rearmed most often in `window`, most first (only those with a
    /// streak of 3 or more when `steady_only`), and returns how many it filled.
    std::size_t Busiest(u32 window, bool steady_only, std::span<Top> out) const;

    [[nodiscard]] std::size_t Size() const noexcept;

    ChurnTable();
    ~ChurnTable();
    ChurnTable(const ChurnTable&) = delete;
    ChurnTable& operator=(const ChurnTable&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCommon::Nx2Protect
