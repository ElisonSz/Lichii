// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <vector>
#include <fmt/ranges.h>
#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::OotWhiteout {
// Diagnostic only. No guest reads, GPU waits, or allocations in the event path. Start/Stop are
// owned by the rasterizer; worker events carry the frame current when they actually execute.
enum class Event : size_t {
    Draw,
    Dispatch,
    Indirect,
    Slot,
    Build,
    DriverBuilt,
    Skip,
    GraphicsWait,
    ComputeWait,
    Failed,
    CollectDownload,
    CollectDrop,
    StaleSkip,
    Refresh,
    Evict,
    Flush,
    UploadBytes,
    Binding,
    GridUnknown,
    ArgumentRead,
    Copy,
    Count
};
inline constexpr std::array Names{
    "draw",         "dispatch",      "indirect",     "slot",   "build",        "driver-built",
    "skip",         "graphics-wait", "compute-wait", "failed", "gc-download",  "gc-drop",
    "stale-skip",   "refresh",       "evict",        "flush",  "upload-bytes", "binding",
    "grid-unknown", "argument-read", "copy"};
static_assert(Names.size() == static_cast<size_t>(Event::Count));
inline std::atomic_bool enabled{};
inline std::array<std::atomic<u64>, static_cast<size_t>(Event::Count)> counters{};
struct Detail {
    u64 frame{}, serial{};
    Event event{};
    u64 a{}, b{}, c{}, d{};
};
struct Frame {
    u64 frame{};
    std::array<u64, static_cast<size_t>(Event::Count)> totals{};
};
inline std::mutex mutex;
// Separate rings keep frequent uploads/flushes from erasing preceding pipeline/GC events.
inline constexpr size_t DetailsPerKind = 64;
inline std::array<std::array<Detail, DetailsPerKind>, Names.size()> details{};
inline std::array<u64, Names.size()> detail_counts{};
inline std::array<Frame, 600> frames{};
inline u64 serial{}, frame_serial{}, current_frame{};

inline bool Enabled() { return enabled.load(std::memory_order_relaxed); }
inline void Start(u64 title) {
    const char *value = std::getenv("EDEN_NX2_WHITEOUT");
    const bool active = title == 0x0400486024DEC000ULL && value && std::string_view{value} == "1";
    enabled.store(false, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    serial = frame_serial = current_frame = 0;
    details = {};
    detail_counts = {};
    frames = {};
    for (auto &counter : counters)
        counter.store(0, std::memory_order_relaxed);
    enabled.store(active, std::memory_order_relaxed);
    if (active)
        LOG_INFO(Render_Vulkan, "OoT whiteout armed: interval=15 frames grid=32x18 pending=16 "
                                "history=600 frames/64 events per kind");
}
inline void Count(Event event, u64 amount = 1) {
    if (Enabled())
        counters[static_cast<size_t>(event)].fetch_add(amount, std::memory_order_relaxed);
}
// a..d are event-specific unsigned fields; their layout is documented in whiteout.md.
inline void Add(Event event, u64 a = 0, u64 b = 0, u64 c = 0, u64 d = 0, u64 amount = 1) {
    if (!Enabled())
        return;
    Count(event, amount);
    std::scoped_lock lock{mutex};
    const size_t index = static_cast<size_t>(event);
    const u64 next = serial++;
    details[index][detail_counts[index]++ % DetailsPerKind] = {
        current_frame, next, event, a, b, c, d};
}
inline void Tick(u64 frame) {
    if (!Enabled())
        return;
    std::scoped_lock lock{mutex};
    auto &entry = frames[frame_serial++ % frames.size()];
    entry.frame = current_frame;
    for (size_t i = 0; i < counters.size(); ++i)
        entry.totals[i] = counters[i].load(std::memory_order_relaxed);
    current_frame = frame;
}
inline void Dump(u64 sample_frame) {
    // Snapshot under the lock; slow logging must not hold up shader-build workers.
    std::vector<Detail> event_snapshot;
    std::vector<Frame> frame_snapshot;
    u64 event_total{}, frame_total{}, latest{};
    {
        std::scoped_lock lock{mutex};
        event_total = serial;
        frame_total = frame_serial;
        latest = current_frame;
        for (size_t kind = 0; kind < details.size(); ++kind) {
            const u64 count = detail_counts[kind];
            const u64 first = count > DetailsPerKind ? count - DetailsPerKind : 0;
            for (u64 i = first; i < count; ++i)
                event_snapshot.push_back(details[kind][i % DetailsPerKind]);
        }
        const u64 first_frame = frame_serial > frames.size() ? frame_serial - frames.size() : 0;
        for (u64 i = first_frame; i < frame_serial; i += 15)
            frame_snapshot.push_back(frames[i % frames.size()]);
    }
    std::ranges::sort(event_snapshot, {}, &Detail::serial);
    LOG_WARNING(
        Render_Vulkan,
        "OoT whiteout history: sample_frame={} current_frame={} events={} overwritten={} frames={}",
        sample_frame, latest, event_total, event_total - event_snapshot.size(), frame_total);
    // Cumulative totals retain even events whose individual detail has been overwritten.
    for (const auto &entry : frame_snapshot) {
        LOG_WARNING(Render_Vulkan, "OoT whiteout totals: frame={} values={}", entry.frame,
                    fmt::join(entry.totals, ","));
    }
    for (const auto &entry : event_snapshot) {
        LOG_WARNING(Render_Vulkan,
                    "OoT whiteout event: frame={} seq={} kind={} a={:x} b={:x} c={:x} d={:x}",
                    entry.frame, entry.serial, Names[static_cast<size_t>(entry.event)], entry.a,
                    entry.b, entry.c, entry.d);
    }
}
} // namespace VideoCommon::OotWhiteout
