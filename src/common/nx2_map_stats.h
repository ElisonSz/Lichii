// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace Common::Nx2MapStats {

/// EDEN_NX2_MAP_STATS=1 (diagnostic): what the guest's MapPhysicalMemory and UnmapPhysicalMemory
/// calls cost the host, and how much of a new view lands on the same backing memory as the view
/// the guest unmapped there before (which a host view kept across the unmap would not remap).
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_MAP_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

enum class Counter : size_t {
    SvcMapCalls,
    SvcMapBytes,
    SvcMapNs,
    SvcUnmapCalls,
    SvcUnmapBytes,
    SvcUnmapNs,
    HostMaps,
    HostMapBytes,
    HostMapSplits,
    HostMapNs,
    HostUnmaps,
    HostUnmapPlaceholders,
    HostUnmapBytes,
    HostUnmapNs,
    SameBackingBytes,
    // Where the host's time goes: waits for the placeholder lock, and each kind of system call.
    MapLockNs,
    UnmapLockNs,
    ProtectLockNs,
    Lookups,
    LookupLockNs,
    Splits,
    SplitNs,
    MapViews,
    MapViewNs,
    UnmapViews,
    UnmapViewNs,
    Coalesces,
    CoalesceNs,
    Protects,
    VirtualProtects,
    VirtualProtectNs,
    Count,
};

inline std::array<std::atomic<u64>, static_cast<size_t>(Counter::Count)> g_counters{};

inline void Add(Counter counter, u64 value) {
    g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

[[nodiscard]] inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Counts one call and its time when the statistics are on.
class Timer {
public:
    Timer(Counter calls_, Counter ns_)
        : calls{calls_}, ns{ns_}, on{Enabled()}, start{on ? NowNs() : 0} {}
    ~Timer() {
        if (on) [[unlikely]] {
            Add(calls, 1);
            Add(ns, NowNs() - start);
        }
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    Counter calls;
    Counter ns;
    bool on;
    u64 start;
};

/// The process's page faults so far (host_memory.cpp; 0 where unknown).
[[nodiscard]] u64 PageFaultCount();

/// After each counted SVC: logs and clears the counters once at least 10 s have passed.
inline void MaybeReport() {
    static std::atomic<u64> window_start{};
    const u64 now = NowNs();
    u64 start = window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
        return;
    }
    if (now - start < 10'000'000'000ULL ||
        !window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
        return;
    }
    std::array<u64, static_cast<size_t>(Counter::Count)> value{};
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
    }
    const auto get = [&value](Counter counter) { return value[static_cast<size_t>(counter)]; };
    const auto mib = [&get](Counter counter) {
        return static_cast<double>(get(counter)) / (1024.0 * 1024.0);
    };
    const auto ms = [&get](Counter counter) { return static_cast<double>(get(counter)) / 1e6; };
    LOG_INFO(Kernel,
             "Guest physical memory {:.1f} s: MapPhysicalMemory {} calls, {:.1f} MiB, {:.1f} ms "
             "(host {} views, {:.1f} MiB, {} splits, {:.1f} ms); UnmapPhysicalMemory {} calls, "
             "{:.1f} MiB, {:.1f} ms (host {} unmaps, {} placeholders, {:.1f} MiB, {:.1f} ms); "
             "new views on the backing their range had before: {:.1f} MiB",
             static_cast<double>(now - start) / 1e9, get(Counter::SvcMapCalls),
             mib(Counter::SvcMapBytes), ms(Counter::SvcMapNs), get(Counter::HostMaps),
             mib(Counter::HostMapBytes), get(Counter::HostMapSplits), ms(Counter::HostMapNs),
             get(Counter::SvcUnmapCalls), mib(Counter::SvcUnmapBytes), ms(Counter::SvcUnmapNs),
             get(Counter::HostUnmaps), get(Counter::HostUnmapPlaceholders),
             mib(Counter::HostUnmapBytes), ms(Counter::HostUnmapNs),
             mib(Counter::SameBackingBytes));
    static std::atomic<u64> last_faults{};
    const u64 faults = PageFaultCount();
    const u64 new_faults = faults - last_faults.exchange(faults, std::memory_order_relaxed);
    LOG_INFO(Kernel,
             "Host memory {:.1f} s: lock waits map {:.1f} ms, unmap {:.1f} ms, protect {:.1f} ms, "
             "{} lookups {:.1f} ms; splits {} {:.1f} ms, views mapped {} {:.1f} ms, unmapped {} "
             "{:.1f} ms, coalesces {} {:.1f} ms; {} protects, {} VirtualProtect {:.1f} ms; {} page "
             "faults",
             static_cast<double>(now - start) / 1e9, ms(Counter::MapLockNs),
             ms(Counter::UnmapLockNs), ms(Counter::ProtectLockNs), get(Counter::Lookups),
             ms(Counter::LookupLockNs), get(Counter::Splits), ms(Counter::SplitNs),
             get(Counter::MapViews), ms(Counter::MapViewNs), get(Counter::UnmapViews),
             ms(Counter::UnmapViewNs), get(Counter::Coalesces), ms(Counter::CoalesceNs),
             get(Counter::Protects), get(Counter::VirtualProtects), ms(Counter::VirtualProtectNs),
             new_faults);
}

} // namespace Common::Nx2MapStats
