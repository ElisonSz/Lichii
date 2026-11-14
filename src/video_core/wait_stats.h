// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::WaitStats {

enum class Reason : u8 {
    Other,
    Finish,
    Fence,
    Query,
    HostRead,
    GpuWrittenList,
    Present,
    Swapchain,
    DescriptorChunk,
    MultiRangeRetire,
    Download,
    AstcDecode,
    Trace,
    ResourceReuse,
    ChunkLimit,
    Count,
};

inline constexpr size_t ReasonCount = static_cast<size_t>(Reason::Count);
inline constexpr std::array<std::string_view, ReasonCount> reason_names{
    "other",    "finish",      "fence",     "query",      "host_read",
    "gpu_list", "present",     "swapchain", "descriptor", "multi_range",
    "download", "astc_decode", "trace",     "resource",   "chunk_limit",
};

enum class Thread : u8 {
    Other,
    Front,
    Back,
    Worker,
    Present,
    Fencing,
    Count,
};

inline constexpr size_t ThreadCount = static_cast<size_t>(Thread::Count);
inline constexpr size_t ThreadSlots = 64;
inline constexpr std::array<std::string_view, ThreadCount> thread_names{
    "other", "front", "back", "worker", "present", "fencing",
};

struct WaitCounter {
    std::atomic<u64> count{};
    std::atomic<u64> ns{};
    std::atomic<u64> over_1ms{};
    std::atomic<u64> longest_ns{};
};

inline bool enabled{};
inline std::chrono::steady_clock::time_point window_start{};
inline std::mutex report_mutex;
inline std::array<WaitCounter, ReasonCount * ThreadSlots> waits{};
inline std::array<std::atomic<u64>, ReasonCount> submissions{};
inline std::array<std::atomic<u64>, ReasonCount> deferred_submissions{};
inline std::atomic<u64> total_submissions{};
inline std::atomic<u64> total_deferred{};
inline thread_local Reason current_reason{Reason::Other};
inline thread_local Thread current_thread{Thread::Other};
inline std::atomic<size_t> next_thread_slot{1};
inline thread_local size_t thread_slot{};
inline std::array<std::atomic<size_t>, ThreadSlots> thread_roles{};
inline std::array<std::atomic<size_t>, ThreadSlots> thread_ids{};

inline size_t ThreadSlot() {
    if (thread_slot == 0) {
        const size_t assigned = next_thread_slot.fetch_add(1, std::memory_order_relaxed);
        thread_slot = assigned < ThreadSlots ? assigned : 0;
        thread_ids[thread_slot].store(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                      std::memory_order_relaxed);
    }
    thread_roles[thread_slot].store(static_cast<size_t>(current_thread), std::memory_order_relaxed);
    return thread_slot;
}

inline void ReadSwitch() {
    const char* const value = std::getenv("EDEN_NX2_WAIT_STATS");
    enabled = value != nullptr && value[0] == '1';
    window_start = std::chrono::steady_clock::now();
}

[[nodiscard]] inline bool Enabled() {
    return enabled;
}

[[nodiscard]] inline Reason CurrentReason() {
    return current_reason;
}

[[nodiscard]] inline u64 CurrentReasonBit() {
    return u64{1} << static_cast<u8>(current_reason);
}

class ReasonScope {
public:
    explicit ReasonScope(Reason reason) : active{Enabled()} {
        if (active) {
            previous = current_reason;
            current_reason = previous == Reason::Other ? reason : previous;
        }
    }
    ~ReasonScope() {
        if (active) {
            current_reason = previous;
        }
    }
    ReasonScope(const ReasonScope&) = delete;
    ReasonScope& operator=(const ReasonScope&) = delete;

private:
    Reason previous{};
    bool active;
};

class ThreadScope {
public:
    explicit ThreadScope(Thread thread) : active{Enabled()} {
        if (active) {
            previous = current_thread;
            current_thread = thread;
        }
    }
    ~ThreadScope() {
        if (active) {
            current_thread = previous;
        }
    }
    ThreadScope(const ThreadScope&) = delete;
    ThreadScope& operator=(const ThreadScope&) = delete;

private:
    Thread previous{};
    bool active;
};

// Explicit site tags preserve a more specific event supplied by their caller.
template <typename Func>
decltype(auto) WithReason(Reason reason, Func&& func) {
    if (Enabled()) {
        const ReasonScope scope{reason};
        return std::forward<Func>(func)();
    }
    return std::forward<Func>(func)();
}

inline void Max(std::atomic<u64>& target, u64 value) {
    u64 old = target.load(std::memory_order_relaxed);
    while (old < value && !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
    }
}

inline void Report() {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{report_mutex};
    const auto now = std::chrono::steady_clock::now();
    if (now - window_start < std::chrono::seconds{10}) {
        return;
    }
    std::string text;
    for (size_t reason = 0; reason < ReasonCount; ++reason) {
        const u64 submit_count = submissions[reason].exchange(0, std::memory_order_relaxed);
        const u64 deferred_count =
            deferred_submissions[reason].exchange(0, std::memory_order_relaxed);
        u64 total_waits{};
        u64 total_ns{};
        u64 total_over{};
        u64 longest{};
        for (size_t thread = 0; thread < ThreadSlots; ++thread) {
            auto& counter = waits[reason * ThreadSlots + thread];
            const u64 count = counter.count.exchange(0, std::memory_order_relaxed);
            const u64 ns = counter.ns.exchange(0, std::memory_order_relaxed);
            const u64 over = counter.over_1ms.exchange(0, std::memory_order_relaxed);
            const u64 max_ns = counter.longest_ns.exchange(0, std::memory_order_relaxed);
            total_waits += count;
            total_ns += ns;
            total_over += over;
            longest = (std::max)(longest, max_ns);
            if (count != 0) {
                text += fmt::format(
                    " {}.{}#{:x}={}/{:.3f}ms/>1ms:{}/max:{:.3f}ms", reason_names[reason],
                    thread_names[thread_roles[thread].load(std::memory_order_relaxed)],
                    thread_ids[thread].load(std::memory_order_relaxed), count,
                    static_cast<double>(ns) / 1e6, over, static_cast<double>(max_ns) / 1e6);
            }
        }
        if (submit_count != 0 || deferred_count != 0 || total_waits != 0) {
            text += fmt::format(" {}=wait:{}/{:.3f}ms/>1ms:{}/max:{:.3f}ms submit:{} deferred:{}",
                                reason_names[reason], total_waits,
                                static_cast<double>(total_ns) / 1e6, total_over,
                                static_cast<double>(longest) / 1e6, submit_count, deferred_count);
        }
    }
    LOG_INFO(Render_Vulkan, "Wait stats: {:.3f}s submissions={} deferred={}{}",
             std::chrono::duration<double>(now - window_start).count(),
             total_submissions.exchange(0, std::memory_order_relaxed),
             total_deferred.exchange(0, std::memory_order_relaxed), text.empty() ? " none" : text);
    window_start = now;
}

inline void RecordSubmission(u64 reason_mask, bool deferred) {
    if (!Enabled()) {
        return;
    }
    total_submissions.fetch_add(1, std::memory_order_relaxed);
    if (deferred) {
        total_deferred.fetch_add(1, std::memory_order_relaxed);
    }
    for (size_t reason = 0; reason < ReasonCount; ++reason) {
        if ((reason_mask & (u64{1} << reason)) != 0) {
            submissions[reason].fetch_add(1, std::memory_order_relaxed);
            if (deferred) {
                deferred_submissions[reason].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

class WaitScope {
public:
    WaitScope() : active{Enabled()} {
        if (active) {
            reason = CurrentReason();
            slot = ThreadSlot();
            start = std::chrono::steady_clock::now();
        }
    }
    ~WaitScope() {
        if (!active) {
            return;
        }
        const u64 elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - start)
                                                 .count());
        auto& counter = waits[static_cast<size_t>(reason) * ThreadSlots + slot];
        counter.count.fetch_add(1, std::memory_order_relaxed);
        counter.ns.fetch_add(elapsed, std::memory_order_relaxed);
        if (elapsed > 1'000'000) {
            counter.over_1ms.fetch_add(1, std::memory_order_relaxed);
        }
        Max(counter.longest_ns, elapsed);
    }
    WaitScope(const WaitScope&) = delete;
    WaitScope& operator=(const WaitScope&) = delete;

private:
    Reason reason{};
    size_t slot{};
    bool active;
    std::chrono::steady_clock::time_point start{};
};

} // namespace VideoCommon::WaitStats
