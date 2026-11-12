// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

// TEST: CPU observations, not GPU timestamps. Never changes submission or wait policy.
namespace VideoCommon::SyncTrace {
inline bool enabled{}; // Set once in CreateGPU, before GPU threads start.
using Time = s64;
inline Time Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
constexpr size_t Capacity = 65536;
enum Point : size_t { R, Q, S, W0, W1, L, F, P, B, SubmissionReady, Points };
inline constexpr std::array<std::pair<Point, Point>, 11> Intervals{{
    {R, Q},
    {Q, S},
    {S, W1},
    {W0, W1},
    {W1, L},
    {L, F},
    {F, P},
    {P, B},
    {R, B},
    {W0, SubmissionReady},
    {SubmissionReady, W1},
}};
inline constexpr std::array<const char*, 11> Names{
    "R-Q", "Q-S", "S-W1", "W0-W1", "W1-L", "L-F", "F-P", "P-B", "R-B", "W0-ready", "ready-W1"};
struct Report {
    u64 id{};
    std::array<Time, Points> at{};
    std::array<bool, Intervals.size()> sampled{};
    bool stubbed{};
    bool issued{};
    bool discarded{};
    bool matched{};
    bool suspended{};
};
using Token = std::shared_ptr<Report>;
struct IdleTotals {
    u64 pending_count{}, clear_count{}, slept{}, hooks{}, cancelled{};
    Time pending_ns{}, clear_ns{};
    // before submission, completion observation, publication, after publication, unknown
    std::array<Time, 5> phases{};
};
struct Window {
    u64 reports{}, stubbed{}, deferred{}, discarded{}, landed{}, no_waiter{}, no_suspended{};
    u64 dropped{}, expired{}, unobserved{}, no_entry{};
    u64 early_submit{}, early_observe{}, deferred_at_queue{}, batched{}, coalesced{};
    std::array<std::vector<Time>, Intervals.size()> samples;
    IdleTotals idle;
};
struct IdleSpan {
    Time start{};
    bool slept{}, hook{};
    Token report;
    Time end{};
};
struct State {
    std::mutex mutex;
    u64 next_id{};
    Time since{};
    Window window;
    std::map<u64, Token> reports;
    std::map<u64, Token> pending;
    std::map<std::pair<const void*, u64>, std::vector<Token>> ticks;
    std::map<const void*, std::vector<Token>> ready;
    std::map<const void*, Token> waiters;
    std::map<const void*, bool> suspended;
    std::map<u64, std::vector<Token>> entries;
    IdleSpan idle;
    std::vector<IdleSpan> closed_idle;
};
inline State state;
inline std::atomic<u64> next_entry{std::numeric_limits<u64>::max()};
inline thread_local u64 current_report{};
inline thread_local const void* front_owner{};
inline thread_local bool front_ready{};
inline thread_local bool idle_active{};

inline void Sample(Report& report) { // state.mutex held; tolerate milestones arriving out of order.
    for (size_t i = 0; i < Intervals.size(); ++i) {
        const auto [first, last] = Intervals[i];
        if (!report.sampled[i] && report.at[first] != 0 && report.at[last] != 0) {
            state.window.early_submit +=
                first == Q && last == S && report.at[last] < report.at[first] ? 1U : 0U;
            state.window.early_observe +=
                first == L && last == F && report.at[last] < report.at[first] ? 1U : 0U;
            auto& values = state.window.samples[i];
            if (values.size() < Capacity) {
                values.push_back(report.at[last] - report.at[first]);
            } else {
                ++state.window.dropped;
            }
            report.sampled[i] = true;
        }
    }
}
inline void Mark(u64 id, Point point, Time time = 0) {
    if (id == 0) {
        return;
    }
    const Time stamp = time != 0 ? time : Now();
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.reports.find(id); it != state.reports.end()) {
        auto& report = *it->second;
        if (report.at[point] == 0) {
            report.at[point] = stamp;
            if (point == Q && !report.stubbed && !report.issued) {
                ++state.window.deferred;
            }
            Sample(report);
        }
    }
}
inline u64 Begin() {
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    if (state.since == 0) {
        state.since = stamp;
    }
    if (state.reports.size() >= Capacity) {
        ++state.window.dropped;
        return 0;
    }
    auto report = std::make_shared<Report>();
    report->id = ++state.next_id;
    report->at[R] = stamp; // CounterReport entry: closest report-only hook, no draw clocks.
    state.reports.emplace(report->id, report);
    state.pending.emplace(report->id, report);
    ++state.window.reports;
    return report->id;
}
inline void Attach(u64 id, const void* scheduler, u64 tick, bool stubbed, bool deferred) {
    if (id == 0) {
        return;
    }
    std::scoped_lock lock{state.mutex};
    const auto it = state.reports.find(id);
    if (it == state.reports.end()) {
        return;
    }
    it->second->stubbed = stubbed;
    state.window.stubbed += stubbed ? 1U : 0U;
    state.window.deferred_at_queue += deferred ? 1U : 0U;
    if (!stubbed) {
        state.ticks[{scheduler, tick}].push_back(it->second);
    }
}
inline void Issued(const void* scheduler, u64 tick) {
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.ticks.find({scheduler, tick}); it != state.ticks.end()) {
        for (const auto& report : it->second) {
            report->issued = true;
        }
    }
}
inline void Submitted(const void* scheduler, u64 tick) {
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    const auto it = state.ticks.find({scheduler, tick});
    if (it == state.ticks.end()) {
        return;
    }
    for (const auto& report : it->second) {
        report->at[S] = stamp;
        Sample(*report);
    }
    state.ticks.erase(it);
}
inline void BatchObserved(u64 id) {
    if (id == 0) {
        return;
    }
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.reports.find(id); it != state.reports.end()) {
        it->second->at[W0] = stamp;
        it->second->at[W1] = stamp;
        ++state.window.batched;
        Sample(*it->second);
    }
}
inline void Matched(u64 id, const void* owner) { // before publishing the first latch
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.reports.find(id); it != state.reports.end()) {
        it->second->matched = true;
        if (const auto waiting = state.suspended.find(owner); waiting != state.suspended.end()) {
            it->second->suspended |= waiting->second;
        }
        state.waiters[owner] = it->second;
    }
}
inline void Suspend(const void* owner) {
    std::scoped_lock lock{state.mutex};
    state.suspended[owner] = true;
    if (const auto it = state.waiters.find(owner); it != state.waiters.end()) {
        it->second->suspended = true;
    }
}
inline void RemoveWaiter(const void* owner) {
    std::scoped_lock lock{state.mutex};
    state.suspended.erase(owner);
    state.waiters.erase(owner);
}
inline void Observe(u64 id, const void* owner) {
    if (id == 0) {
        return;
    }
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    const auto it = state.reports.find(id);
    if (it == state.reports.end() || it->second->at[F] != 0) {
        return;
    }
    const auto& report = it->second;
    report->at[F] = stamp;
    state.ready[owner].push_back(report);
    if (front_owner == owner) {
        front_ready = true;
    }
    Sample(*report);
}
inline void Finish(u64 id, bool discarded) {
    if (id == 0) {
        return;
    }
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.reports.find(id); it != state.reports.end()) {
        auto& report = *it->second;
        report.discarded = discarded;
        if (discarded) {
            ++state.window.discarded;
        } else {
            report.at[L] = stamp;
            ++state.window.landed;
            state.window.no_waiter += report.matched ? 0U : 1U;
            Sample(report);
        }
        state.pending.erase(id);
    }
}
// Wrap channel dispatch, including its deferred BindChannel entry on ResumePending.
class FrontScope {
public:
    explicit FrontScope(const void* owner) {
        if (!enabled) {
            return;
        }
        previous_owner = front_owner;
        previous_ready = front_ready;
        front_owner = owner;
        std::scoped_lock lock{state.mutex};
        const auto it = state.ready.find(owner);
        front_ready = it != state.ready.end();
    }
    ~FrontScope() {
        if (!enabled) {
            return;
        }
        front_owner = previous_owner;
        front_ready = previous_ready;
    }

private:
    const void* previous_owner{};
    bool previous_ready{};
};
inline void Commit(u64 start) {
    if (!front_ready) {
        return;
    }
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.ready.find(front_owner); it != state.ready.end()) {
        auto& tokens = state.entries[start];
        tokens.insert(tokens.end(), it->second.begin(), it->second.end());
        state.ready.erase(it);
        next_entry.store(state.entries.begin()->first, std::memory_order_release);
    }
    front_ready = false;
}
inline void Publish(u64 first, u64 end) {
    if (next_entry.load(std::memory_order_acquire) == std::numeric_limits<u64>::max()) {
        return;
    }
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    for (auto it = state.entries.lower_bound(first); it != state.entries.end() && it->first < end;
         ++it) {
        for (const auto& report : it->second) {
            if (report->at[P] == 0) {
                report->at[P] = stamp;
                Sample(*report);
            }
        }
    }
}
inline void BackEntry(u64 start) {
    if (start < next_entry.load(std::memory_order_acquire)) {
        return;
    }
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.entries.find(start); it != state.entries.end()) {
        for (const auto& report : it->second) {
            report->at[B] = stamp;
            Sample(*report);
        }
        state.entries.erase(it);
        next_entry.store(state.entries.empty() ? std::numeric_limits<u64>::max()
                                               : state.entries.begin()->first,
                         std::memory_order_release);
    }
}
// Execute can coalesce consecutive constant-buffer entries. If it consumed the target
// inside that batch, use the end-of-Execute observation and count the approximation.
inline void BackConsumed(u64 end) {
    if (next_entry.load(std::memory_order_acquire) >= end) {
        return;
    }
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    for (auto it = state.entries.begin(); it != state.entries.end() && it->first < end;) {
        for (const auto& report : it->second) {
            report->at[B] = stamp;
            ++state.window.coalesced;
            Sample(*report);
        }
        it = state.entries.erase(it);
    }
    next_entry.store(state.entries.empty() ? std::numeric_limits<u64>::max()
                                           : state.entries.begin()->first,
                     std::memory_order_release);
}
inline void IdleBegin() {
    if (idle_active) {
        return;
    }
    idle_active = true;
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    if (state.idle.start == 0) {
        state.idle = {stamp, false, false,
                      state.pending.empty() ? Token{} : state.pending.begin()->second};
    }
}
inline void IdleFlag(bool slept) {
    std::scoped_lock lock{state.mutex};
    (slept ? state.idle.slept : state.idle.hook) = true;
}
// Called with state.mutex held. Completed spans can precede their report's milestones.
inline void AttributeIdle(const IdleSpan& span) {
    auto& phases = state.window.idle.phases;
    const auto& report = *span.report;
    const Time duration = span.end - span.start;
    const Time submitted = report.at[S];
    const Time completed = report.at[W1];
    const Time landed = report.at[L];
    if (report.stubbed || report.discarded || submitted == 0 || completed == 0 || landed == 0 ||
        completed < submitted || landed < completed) {
        phases[4] += duration;
        return;
    }
    Time cursor = span.start;
    const std::array<Time, 3> boundaries{submitted, completed, landed};
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const Time end = std::clamp(boundaries[i], cursor, span.end);
        phases[i] += end - cursor;
        cursor = end;
    }
    phases[3] += span.end - cursor;
}
inline void IdleEnd(bool cancelled = false) {
    if (!idle_active) {
        return;
    }
    idle_active = false;
    const Time stamp = Now();
    std::scoped_lock lock{state.mutex};
    IdleSpan span = std::exchange(state.idle, {});
    if (span.start == 0) {
        return;
    }
    span.end = stamp;
    auto& totals = state.window.idle;
    totals.slept += span.slept ? 1U : 0U;
    totals.hooks += span.hook ? 1U : 0U;
    totals.cancelled += cancelled ? 1U : 0U;
    if (!span.report) {
        ++totals.clear_count;
        totals.clear_ns += stamp - span.start;
    } else if (state.closed_idle.size() < Capacity) {
        state.closed_idle.push_back(std::move(span));
    } else {
        ++totals.pending_count;
        totals.pending_ns += stamp - span.start;
        totals.phases[4] += stamp - span.start;
        ++state.window.dropped;
    }
}
inline void StopSplit() {
    std::scoped_lock lock{state.mutex};
    state.entries.clear();
    state.ready.clear();
    next_entry.store(std::numeric_limits<u64>::max(), std::memory_order_release);
}
inline std::string Distribution(std::vector<Time>& values) {
    if (values.empty()) {
        return "n=0";
    }
    std::sort(values.begin(), values.end());
    double total = 0;
    for (const Time value : values) {
        total += static_cast<double>(value);
    }
    const size_t count = values.size();
    const double median = count % 2 != 0 ? static_cast<double>(values[count / 2])
                                         : (static_cast<double>(values[count / 2 - 1]) +
                                            static_cast<double>(values[count / 2])) /
                                               2.0;
    return fmt::format("n={} mean={:.3f} med={:.3f} p90={:.3f} max={:.3f}", count,
                       total / static_cast<double>(count) / 1000.0, median / 1000.0,
                       static_cast<double>(values[(count * 9 + 9) / 10 - 1]) / 1000.0,
                       static_cast<double>(values.back()) / 1000.0);
}
inline void MaybeReport() {
    const Time stamp = Now();
    Window snapshot;
    size_t outstanding = 0;
    size_t idle_unresolved = 0;
    double seconds = 0;
    {
        std::scoped_lock lock{state.mutex};
        if (state.since == 0 || stamp - state.since < 10'000'000'000LL) {
            return;
        }
        seconds = static_cast<double>(stamp - state.since) / 1e9;
        state.since = stamp;
        // Attribute only when the representative report has resolved (or expired).
        for (auto it = state.closed_idle.begin(); it != state.closed_idle.end();) {
            const auto& report = *it->report;
            if (report.at[L] != 0 || report.discarded || stamp - report.at[R] > 30'000'000'000LL) {
                ++state.window.idle.pending_count;
                state.window.idle.pending_ns += it->end - it->start;
                AttributeIdle(*it);
                it = state.closed_idle.erase(it);
            } else {
                ++it;
            }
        }
        // Keep unfinished reports across windows, but bound stale state after shutdown/unmaps.
        for (auto it = state.reports.begin(); it != state.reports.end();) {
            const auto& report = *it->second;
            const bool terminal = report.discarded || (report.at[Q] != 0 && report.at[L] != 0 &&
                                                       (!report.matched || report.at[B] != 0));
            const bool expired = stamp - report.at[R] > 30'000'000'000LL;
            if (terminal || expired) {
                state.window.expired += !terminal && expired ? 1U : 0U;
                state.window.unobserved += report.matched && report.at[F] == 0 ? 1U : 0U;
                state.window.no_entry += report.at[F] != 0 && report.at[B] == 0 ? 1U : 0U;
                state.window.no_suspended += report.at[L] != 0 && !report.suspended ? 1U : 0U;
                state.pending.erase(it->first);
                it = state.reports.erase(it);
            } else {
                ++it;
            }
        }
        // Remove auxiliary references to expired records. Tokens held by active hooks stay valid.
        const auto stale = [](const Token& token) { return !state.reports.contains(token->id); };
        for (auto it = state.ticks.begin(); it != state.ticks.end();) {
            std::erase_if(it->second, stale);
            it = it->second.empty() ? state.ticks.erase(it) : std::next(it);
        }
        for (auto it = state.entries.begin(); it != state.entries.end();) {
            std::erase_if(it->second, stale);
            it = it->second.empty() ? state.entries.erase(it) : std::next(it);
        }
        for (auto it = state.ready.begin(); it != state.ready.end();) {
            std::erase_if(it->second, stale);
            it = it->second.empty() ? state.ready.erase(it) : std::next(it);
        }
        std::erase_if(state.waiters, [&](const auto& item) { return stale(item.second); });
        next_entry.store(state.entries.empty() ? std::numeric_limits<u64>::max()
                                               : state.entries.begin()->first,
                         std::memory_order_release);
        outstanding = state.reports.size();
        idle_unresolved = state.closed_idle.size();
        snapshot = std::exchange(state.window, {});
    }
    LOG_INFO(HW_GPU,
             "Sync trace {:.3f}s: reports={} landed={} stubbed={} deferred={} discarded={} "
             "no_waiter={} no_suspended={} outstanding={} dropped={} expired={} unobserved={} "
             "no_entry={} early_submit={} early_observe={} deferred_at_queue={} idle_unresolved={} "
             "batched={} coalesced={}",
             seconds, snapshot.reports, snapshot.landed, snapshot.stubbed, snapshot.deferred,
             snapshot.discarded, snapshot.no_waiter, snapshot.no_suspended, outstanding,
             snapshot.dropped, snapshot.expired, snapshot.unobserved, snapshot.no_entry,
             snapshot.early_submit, snapshot.early_observe, snapshot.deferred_at_queue,
             idle_unresolved, snapshot.batched, snapshot.coalesced);
    std::string intervals;
    for (size_t i = 0; i < Names.size(); ++i) {
        intervals += fmt::format(" {}[{}]", Names[i], Distribution(snapshot.samples[i]));
    }
    LOG_INFO(HW_GPU, "Sync trace us:{}", intervals);
    const auto& idle = snapshot.idle;
    LOG_INFO(
        HW_GPU,
        "Sync trace idle: pending={} us/{}, clear={} us/{}, slept={} hook={} cancelled={}; "
        "pending phases us: before_S={} S-W1={} W1-L={} after_L={} unknown={}",
        static_cast<double>(idle.pending_ns) / 1000.0, idle.pending_count,
        static_cast<double>(idle.clear_ns) / 1000.0, idle.clear_count, idle.slept, idle.hooks,
        idle.cancelled, static_cast<double>(idle.phases[0]) / 1000.0,
        static_cast<double>(idle.phases[1]) / 1000.0, static_cast<double>(idle.phases[2]) / 1000.0,
        static_cast<double>(idle.phases[3]) / 1000.0, static_cast<double>(idle.phases[4]) / 1000.0);
}
inline void ReadSwitch() {
    const char* value = std::getenv("EDEN_NX2_SYNC_TRACE");
    enabled = value != nullptr && value[0] == '1' && value[1] == '\0';
    // No resetting IDs: callbacks retained across a renderer teardown cannot hit a reused ID.
}
} // namespace VideoCommon::SyncTrace
