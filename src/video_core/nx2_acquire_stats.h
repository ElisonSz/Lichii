// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_wake.h"
#include "video_core/nx2_sync_suspend.h"

// All accounting belongs to the front. The latch publishes its source in its existing atomic;
// neither the back nor a fence callback touches these counters or takes an accounting lock.
namespace VideoCommon::Nx2AcquireDetails {

inline bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_ACQUIRE_STATS");
    return value != nullptr && value[0] == '1';
}();

inline void ReadSwitch() {
    const char* value = std::getenv("EDEN_NX2_ACQUIRE_STATS");
    g_enabled = value != nullptr && value[0] == '1';
}

using Clock = std::chrono::steady_clock;
using Writer = GpuSplit::SemaphoreWriter;
using Source = Tegra::Nx2SemaphoreWaiters::Source;

enum class Decision : u8 { Memory, Queued, Waited, Latched, Suspended, Drained, Count };
enum class Resume : u8 { Latch, Memory, Queued, Drain, Valve, Verify, Count };
enum class Outcome : u8 { Passed, Suspended, Drained, Count };
inline constexpr size_t WriterCount = static_cast<size_t>(Writer::Count);
inline constexpr std::array<const char*, WriterCount> writer_names{
    "other",       "puller",        "3D report",     "compute report", "compute reduction",
    "QMD release", "QMD reduction", "copy semaphore"};
inline constexpr std::array<const char*, 3> operation_names{"equal", "greater or equal", "mask"};

inline size_t Operation(u32 operation) {
    return operation == Tegra::Nx2SemaphoreWaiters::AcquireEqual    ? 0
           : operation == Tegra::Nx2SemaphoreWaiters::AcquireGequal ? 1
                                                                    : 2;
}

inline u64 Nanoseconds(Clock::duration duration) {
    return static_cast<u64>(
        (std::max)(std::chrono::nanoseconds::rep{0},
                   std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()));
}

struct Timing {
    u64 count{};
    u64 ns{};
    u64 max_ns{};
    u64 over_1ms{};
    void Add(u64 duration) {
        ++count;
        ns += duration;
        max_ns = (std::max)(max_ns, duration);
        over_1ms += duration > 1'000'000 ? 1 : 0;
    }
};

// Exact, lower bound, unknown. A lower bound is useful knowledge, but isn't an exact payload.
inline size_t Knowledge(const GpuSplit::QueuedSemaphoreWrite& write) {
    return !write.known ? 2 : write.at_least ? 1 : 0;
}

struct WaitGroup {
    Timing time;
    u64 actually_waited{};
    u64 backlog_sum{};
    u64 backlog_max{};
    u64 target_gap_sum{};
    u64 target_gap_max{};
    std::array<u64, 3> outcomes{};
};

// writer * 18 + knowledge * 6 + fenced * 3 + operation. The writer is Other for a
// writers-before dependency; its kind and payload cannot be inferred from a release elsewhere.
inline constexpr size_t GroupCount = WriterCount * 18;
struct Word {
    Timing time;
    std::array<u64, 3> operations{};
    std::array<u64, WriterCount * 3> writers{};
    std::array<u64, 3> outcomes{};
    u64 writers_before{};
};
struct State {
    std::array<std::array<u64, static_cast<size_t>(Decision::Count)>, 3> acquires{};
    std::array<u64, 3> valves{};
    std::array<u64, 3> valve_passes{};
    std::array<u64, static_cast<size_t>(Resume::Count)> resumes{};
    std::array<u64, static_cast<size_t>(Source::Count)> latches{};
    Timing suspended_time;
    u64 after_flush{};
    u64 suspended{};
    u64 suspended_gpu_written{};
    u64 suspended_flush{};
    std::array<u64, WriterCount> suspended_writers{};
    u64 suspended_no_writer{};
    std::array<WaitGroup, GroupCount> waits{};
    WaitGroup total;
    WaitGroup writers_before;
    std::unordered_map<u64, Word> words;
    std::unordered_map<u64, u16> noted; ///< Historical writer kinds, retained across windows
    u64 noted_overflow{};
    Word overflow;
    Clock::time_point last_report{};
};
inline State& Get() {
    static State state;
    return state;
}

struct Wait {
    size_t group{};
    u64 device{};
    u64 ns{};
    u64 backlog{};
    u64 target_gap{};
    bool waited{};
    bool writers_before{};
};

struct Observation {
    Decision decision{Decision::Drained};
    std::array<Wait, 2> waits{};
    size_t wait_count{};

    // Capture before WaitForPosition: afterwards a release may be retired from the view.
    Wait Begin(const GpuSplit& split, u64 device, u64 target, u32 operation,
               bool write_dependency) const {
        const auto queued = split.SemaphoreWriteForStats(device);
        const bool specific = write_dependency && queued && queued->position == target;
        const size_t writer = specific ? static_cast<size_t>(queued->writer) : 0;
        const size_t knowledge = specific ? Knowledge(*queued) : 2;
        const size_t fenced = specific && queued->fenced ? 1 : 0;
        const u64 backlog = split.Backlog();
        const u64 consumed_position = split.Position() - backlog;
        return {writer * 18 + knowledge * 6 + fenced * 3 + Operation(operation),
                device,
                0,
                backlog,
                target > consumed_position ? target - consumed_position : 0,
                false,
                !write_dependency};
    }
    void End(Wait wait, Clock::time_point begin, bool actually_waited) {
        wait.ns = Nanoseconds(Clock::now() - begin);
        wait.waited = actually_waited;
        waits[wait_count++] = wait;
    }
};

inline void AddWait(const Wait& wait, Outcome outcome) {
    State& state = Get();
    const auto add_group = [&](WaitGroup& group) {
        group.time.Add(wait.ns);
        group.actually_waited += wait.waited ? 1 : 0;
        group.backlog_sum += wait.backlog;
        group.backlog_max = (std::max)(group.backlog_max, wait.backlog);
        group.target_gap_sum += wait.target_gap;
        group.target_gap_max = (std::max)(group.target_gap_max, wait.target_gap);
        ++group.outcomes[static_cast<size_t>(outcome)];
    };
    add_group(state.total);
    add_group(state.waits[wait.group]);
    if (wait.writers_before) {
        add_group(state.writers_before);
    }
    auto found = state.words.find(wait.device);
    if (found == state.words.end() && state.words.size() < 65536) {
        found = state.words.try_emplace(wait.device).first;
    }
    Word& word = found == state.words.end() ? state.overflow : found->second;
    word.time.Add(wait.ns);
    ++word.operations[wait.group % 3];
    ++word.writers[(wait.group / 18) * 3 + (wait.group / 6) % 3];
    ++word.outcomes[static_cast<size_t>(outcome)];
    word.writers_before += wait.writers_before ? 1 : 0;
}

// The disabled specialization compiles away. Puller chooses its specialization once per new
// acquire (or poll), so disabling statistics adds one predictable branch to that path.
template <bool Enabled>
class AcquireScope {
public:
    template <typename Acquire>
    explicit AcquireScope(Acquire&) {}
    Observation* Data() {
        return nullptr;
    }
    template <typename Acquire, typename Memory>
    void Finish(const Acquire&, const Memory&, bool) {}
};

template <>
class AcquireScope<true> {
public:
    template <typename Acquire>
    explicit AcquireScope(Acquire& acquire)
        : operation{Operation(acquire.operation)}, address{acquire.address} {}
    Observation* Data() {
        return &observation;
    }
    // Finish explicitly after the normal path has made its final decision.
    template <typename Acquire, typename Memory>
    void Finish(const Acquire& acquire, const Memory& memory, bool suspended) {
        State& state = Get();
        ++state.acquires[operation][static_cast<size_t>(observation.decision)];
        const Outcome outcome = observation.decision == Decision::Drained ? Outcome::Drained
                                : suspended                               ? Outcome::Suspended
                                                                          : Outcome::Passed;
        for (size_t i = 0; i < observation.wait_count; ++i) {
            AddWait(observation.waits[i], outcome);
        }
        if (!suspended) {
            return;
        }
        ++state.suspended;
        state.suspended_flush += acquire.stats_flushed ? 1 : 0;
        if (const auto device = memory.GpuToCpuAddress(address)) {
            state.suspended_gpu_written += Tegra::Nx2MayBeGpuWritten(*device, sizeof(u32)) ? 1 : 0;
            const auto noted = state.noted.find(*device & ~u64{3});
            if (noted == state.noted.end()) {
                ++state.suspended_no_writer;
            } else {
                for (size_t writer = 0; writer < WriterCount; ++writer) {
                    state.suspended_writers[writer] += (noted->second >> writer) & 1U;
                }
            }
        } else {
            ++state.suspended_no_writer;
        }
    }

private:
    size_t operation;
    GPUVAddr address;
    Observation observation;
};

inline void NoteWriter(u64 device, Writer writer) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    auto found = state.noted.find(device);
    if (found == state.noted.end()) {
        if (state.noted.size() == 65536) {
            ++state.noted_overflow;
            return;
        }
        found = state.noted.try_emplace(device, u16{0}).first;
    }
    found->second |= static_cast<u16>(1U << static_cast<u32>(writer));
}

// Called only by the collecting specialization, once when a pending acquire becomes runnable.
template <typename Acquire>
inline void NoteResume(const void* owner, Acquire& acquire, Resume how) {
    if (acquire.stats_resumed) {
        return;
    }
    acquire.stats_resumed = true;
    State& state = Get();
    ++state.resumes[static_cast<size_t>(how)];
    if (how == Resume::Latch) {
        ++state.latches[static_cast<size_t>(Tegra::Nx2SemaphoreWaiters::Get().LatchSource(owner))];
    }
    state.after_flush += acquire.stats_flushed ? 1 : 0;
    state.suspended_time.Add(Nanoseconds(Clock::now() - acquire.since));
}

inline std::string TimingText(const Timing& time) {
    return fmt::format("{} waits, {:.3f} ms total, {:.3f} ms max, {} over 1 ms", time.count,
                       static_cast<double>(time.ns) / 1e6, static_cast<double>(time.max_ns) / 1e6,
                       time.over_1ms);
}

inline void ReportFastWake() {
    if (!FastWake::enabled) {
        return;
    }
    static auto last_report = Clock::now();
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    auto& counts = FastWake::counters;
    LOG_INFO(HW_GPU,
             "Fast wake {:.3f}s: latch_signals={} command_signals={} pending_waits={} "
             "pending_notified={} "
             "pending_timeouts={} position_signals={} position_waits={} position_timeouts={} "
             "idle_publishes={} back_signals={} back_waits={} back_timeouts={}",
             seconds, take(counts.latch_signals), take(counts.command_signals),
             take(counts.pending_waits), take(counts.pending_notified),
             take(counts.pending_timeouts), take(counts.position_signals),
             take(counts.position_waits), take(counts.position_timeouts),
             take(counts.idle_publishes), take(counts.back_signals), take(counts.back_waits),
             take(counts.back_timeouts));
}

// EDEN_NX2_SYNC_SUSPEND: what the early suspensions did and the waiting left, every 10 s.
inline void ReportSyncSuspend() {
    namespace Suspension = SyncSuspend;
    if (!Suspension::enabled) {
        return;
    }
    static u32 calls{};
    if (++calls % 64 != 0) {
        return;
    }
    static auto last_report = Clock::now();
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    const auto ms = [](u64 ns) { return static_cast<double>(ns) / 1e6; };
    auto& counts = Suspension::counters;
    const u64 idle_spans = Suspension::back.idle_spans.exchange(0, std::memory_order_relaxed);
    const u64 idle_ns = Suspension::back.idle_ns.exchange(0, std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "Sync suspend{} {:.1f} s: {} acquires, {} wait for a fence-flagged release: {} "
             "suspended early ({:.2f} ms until the back ran the release, {} ring words sent "
             "meanwhile, {} met before, {} deviations), kept waiting {} (nothing else queued "
             "{}, met {}, changed {}); settle waits {} {:.2f} ms; other positioned waits {} "
             "{:.2f} ms; idle front waits {} {:.2f} ms ({} timed out); back idle {} spans "
             "{:.2f} ms",
             Suspension::relaxed ? " (relaxed)" : "", seconds, counts.acquires, counts.fenced,
             counts.early, ms(counts.reached_ns), counts.reached_words, counts.unreached_resumes,
             counts.deviations, counts.no_work + counts.met + counts.changed, counts.no_work,
             counts.met, counts.changed, counts.settle_waits, ms(counts.settle_ns), counts.waits,
             ms(counts.wait_ns), counts.idle_waits, ms(counts.idle_wait_ns),
             counts.idle_timeouts, idle_spans, ms(idle_ns));
    counts = {};
}

// EDEN_NX2_RUNAHEAD_STATS (or _VERIFY): what the reaches decided, every 10 s.
inline void ReportRunAhead() {
    if (!RunAhead::stats) {
        return;
    }
    static u32 calls{};
    if (++calls % 64 != 0) {
        return;
    }
    static auto last_report = Clock::now();
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    auto& counts = RunAhead::counters;
    std::string reasons;
    for (size_t i = 0; i < counts.wait_reasons.size(); ++i) {
        reasons += fmt::format("{}{} {}", i == 0 ? "" : ", ", RunAhead::reach_names[i],
                               counts.wait_reasons[i]);
    }
    auto& verified = RunAhead::verified;
    // Round 2: the valve, W1 passes, the QMD view's exact placement and the uploads it may read.
    LOG_INFO(
        HW_GPU,
        "Run-ahead front {:.1f} s: valves without a drain {} (only the drain found {}, longest "
        "wait at one {:.1f} ms); W1 "
        "candidates {}, passed {} (not plain {}, payload unmet {}, no waiter slot {}, written "
        "over after it {}), "
        "suspended ones passed at a poll {}, holds {} (the back waited {}, {:.2f} ms), verify "
        "confirmed {}, unconfirmed {}, unchecked {}; QMD reads placed exactly {} (read sooner "
        "{}); compute uploads {} (not one linear line {}, too long {}, with list placeholders "
        "{}), QMDs last written by part of an upload {} (read with the uploads over it {})",
        seconds, counts.valve_requests, counts.valve_drain_passes,
        static_cast<double>(counts.valve_wait_max_ns) / 1e6, counts.w1_candidates, counts.w1_passes,
        counts.w1_not_plain, counts.w1_unmet, counts.w1_unlatched, counts.w1_overwritten,
        counts.w1_poll_passes, counts.w1_holds,
        RunAhead::holds.waits.exchange(0, std::memory_order_relaxed),
        static_cast<double>(RunAhead::holds.wait_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
        counts.w1_confirmed, counts.w1_unconfirmed, counts.w1_unchecked, counts.qmd_exact,
        counts.qmd_sooner, counts.uploads, counts.upload_rejects[0], counts.upload_rejects[1],
        counts.upload_rejects[2], counts.upload_rejects[3], counts.upload_qmds_partial);
    // Round 4: the W1 candidates the rule for plain payloads does not pass, by what their release
    // is and by the acquire's operation; the passes on maxima (EDEN_NX2_RUNAHEAD_W1_MAX).
    std::string kinds;
    for (size_t i = 0; i < counts.w1_kinds.size(); ++i) {
        const auto& operations = counts.w1_kinds[i];
        kinds += fmt::format("{}{} {}/{}/{}", i == 0 ? "" : ", ", RunAhead::w1_kind_names[i],
                             operations[0], operations[1], operations[2]);
    }
    std::string refusals;
    for (size_t i = 0; i < counts.base_refusals.size(); ++i) {
        refusals += fmt::format("{}{} {}", i == 0 ? "" : ", ", RunAhead::base_refusal_names[i],
                                counts.base_refusals[i]);
    }
    LOG_INFO(HW_GPU,
             "Run-ahead W1 releases {:.1f} s (acquires equal/greater or equal/mask): {}; made "
             "fenced by a landing window {}; maxima passed {} (verify confirmed {}, unconfirmed "
             "{}); QMD reductions and maxima valued from the window's order {} (no base: {}), "
             "passed on them {} (verify confirmed {}, unconfirmed {}); idle front waits with "
             "1/2/3+ channels suspended {}/{}/{}",
             seconds, kinds, counts.w1_windowed, counts.w1_max_passes, counts.w1_max_confirmed,
             counts.w1_max_unconfirmed, counts.window_bases, refusals, counts.w1_base_passes,
             counts.w1_base_confirmed, counts.w1_base_unconfirmed, counts.idle_channels[0],
             counts.idle_channels[1], counts.idle_channels[2]);
    if (RunAhead::landings) {
        auto& landing = RunAhead::landing_counts;
        const auto take = [](std::atomic<u64>& value) {
            return value.exchange(0, std::memory_order_relaxed);
        };
        LOG_INFO(HW_GPU,
                 "Run-ahead landings {:.1f} s: windows opened {}, extended {}, closed {} "
                 "(acknowledged {}), releases noted fenced {}; landings converted: reports {}, "
                 "fence operations {}, pending max {}; entries after a close with some pending "
                 "{}, fences made for them {}; writes over reports or landings that waited {} "
                 "({:.2f} ms); settled reads of a word with one pending: words that only grow {}, "
                 "held until it landed {}; reads after a drain that waited for one {} (gave up "
                 "{}); verify ordered {}, out of order {}, unchecked {}, windows without a check "
                 "{}, fence-flagged reports dropped before they landed {}",
                 seconds, counts.windows_opened, counts.windows_extended, counts.windows_closed,
                 counts.windows_acked, counts.notes_fenced, take(landing.reports),
                 take(landing.operations), take(landing.pending_max), take(landing.extended),
                 take(landing.fences), take(landing.overwrite_waits),
                 static_cast<double>(take(landing.overwrite_ns)) / 1e6,
                 take(landing.advancing_reads), take(landing.held_reads), take(landing.drain_waits),
                 take(landing.early_reads), take(landing.ordered), take(landing.out_of_order),
                 take(landing.unchecked), counts.checks_unregistered,
                 take(landing.dropped_reports));
    }
    const double lead_avg = counts.acquires == 0 ? 0.0
                                                 : static_cast<double>(counts.lead_sum) /
                                                       static_cast<double>(counts.acquires);
    LOG_INFO(HW_GPU,
             "Run-ahead {:.1f} s: {} acquires with a reach, {} before the baseline's writers, {} "
             "passed while the baseline would still wait; writer waits {} {:.2f} ms ({}); {} "
             "logged writes scanned; front lead at acquires {:.0f} words avg, {} max; 1 ms valve "
             "drains {} {:.2f} ms; shim acquires met without a drain {}, drained {}; QMDs read "
             "from queued uploads {}, QMD signals unvalued {}; verify {} checked, {} mismatches "
             "({} GPU-written), {} changed by a fenced release landing",
             seconds, counts.acquires, counts.earlier, counts.sooner, counts.waits,
             static_cast<double>(counts.wait_ns) / 1e6, reasons, counts.scanned, lead_avg,
             counts.lead_max, counts.valve_drains, static_cast<double>(counts.valve_ns) / 1e6,
             counts.shim_met, counts.shim_drained, counts.upload_qmds, counts.unvalued,
             verified.checked.exchange(0, std::memory_order_relaxed),
             verified.mismatches.exchange(0, std::memory_order_relaxed),
             verified.flushes.exchange(0, std::memory_order_relaxed),
             verified.raced.exchange(0, std::memory_order_relaxed));
    counts = {};
}

inline void Report() {
    ReportFastWake();
    ReportSyncSuspend();
    ReportRunAhead();
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    const auto now = Clock::now();
    if (state.last_report == Clock::time_point{}) {
        state.last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_report).count();
    if (seconds < 10.0) {
        return;
    }
    for (size_t op = 0; op < 3; ++op) {
        const auto& counts = state.acquires[op];
        LOG_INFO(HW_GPU,
                 "Acquire stats in {:.1f} s: {}: memory {}, queued {}, after positioned wait {}, "
                 "latched {}, suspended {}, drained {}; 1 ms valve {} (passed {})",
                 seconds, operation_names[op], counts[0], counts[1], counts[2], counts[3],
                 counts[4], counts[5], state.valves[op], state.valve_passes[op]);
    }
    const auto& resumes = state.resumes;
    const auto& latches = state.latches;
    LOG_INFO(HW_GPU,
             "Acquire stats resumes: latch {} (payload {}, query {}, QMD reduction {}, QMD max {}, "
             "other {}), memory {}, queued {}, drained {}, valve {}, verify {}; after flush {}; {}",
             resumes[0], latches[1], latches[2], latches[3], latches[4], latches[0], resumes[1],
             resumes[2], resumes[3], resumes[4], resumes[5], state.after_flush,
             TimingText(state.suspended_time));
    std::string suspended_writers;
    for (size_t i = 0; i < WriterCount; ++i) {
        suspended_writers +=
            fmt::format("{}{} {}", i == 0 ? "" : ", ", writer_names[i], state.suspended_writers[i]);
    }
    LOG_INFO(
        HW_GPU,
        "Acquire stats suspensions: {} total, GPU-written {}, flush requested {}; releases sent "
        "by: {}; none recorded {}; writer history overflow {}",
        state.suspended, state.suspended_gpu_written, state.suspended_flush, suspended_writers,
        state.suspended_no_writer, state.noted_overflow);
    constexpr std::array<const char*, 3> knowledge_names{"exact", "lower bound", "unknown"};
    const auto log_group = [](const std::string& label, const WaitGroup& group) {
        if (group.time.count == 0) {
            return;
        }
        LOG_INFO(
            HW_GPU,
            "Acquire positioned: {}: {}, actually waited {}; backlog words avg {:.1f} max "
            "{}, target gap words avg {:.1f} max {}; after: passed {}, suspended {}, drained {}",
            label, TimingText(group.time), group.actually_waited,
            static_cast<double>(group.backlog_sum) / static_cast<double>(group.time.count),
            group.backlog_max,
            static_cast<double>(group.target_gap_sum) / static_cast<double>(group.time.count),
            group.target_gap_max, group.outcomes[0], group.outcomes[1], group.outcomes[2]);
    };
    for (size_t i = 0; i < GroupCount; ++i) {
        log_group(fmt::format("{}, payload {}, fenced {}, {}", writer_names[i / 18],
                              knowledge_names[(i / 6) % 3], (i / 3) % 2, operation_names[i % 3]),
                  state.waits[i]);
    }
    log_group("all", state.total);
    log_group("writers before (subset of other)", state.writers_before);
    std::vector<std::pair<u64, const Word*>> words;
    words.reserve(state.words.size());
    for (const auto& [device, word] : state.words) {
        words.emplace_back(device, &word);
    }
    std::sort(words.begin(), words.end(), [](const auto& left, const auto& right) {
        return left.second->time.ns != right.second->time.ns
                   ? left.second->time.ns > right.second->time.ns
                   : left.first < right.first;
    });
    const auto log_word = [&](const std::string& label, const Word& word) {
        std::string writers;
        for (size_t i = 0; i < word.writers.size(); ++i) {
            if (word.writers[i] != 0) {
                writers +=
                    fmt::format("{}{} payload {} {}", writers.empty() ? "" : ", ",
                                writer_names[i / 3], knowledge_names[i % 3], word.writers[i]);
            }
        }
        LOG_INFO(HW_GPU,
                 "Acquire positioned word {}: {}; operations equal {}, greater or equal {}, mask "
                 "{}; writers {}; writers-before {}; after: passed {}, suspended {}, drained {}",
                 label, TimingText(word.time), word.operations[0], word.operations[1],
                 word.operations[2], writers, word.writers_before, word.outcomes[0],
                 word.outcomes[1], word.outcomes[2]);
    };
    for (size_t i = 0; i < (std::min)(size_t{8}, words.size()); ++i) {
        log_word(fmt::format("{:#x}", words[i].first), *words[i].second);
    }
    if (state.overflow.time.count != 0) {
        log_word("overflow (65536 word limit)", state.overflow);
    }
    auto noted = std::move(state.noted);
    state = State{};
    state.noted = std::move(noted);
    state.last_report = now;
}

} // namespace VideoCommon::Nx2AcquireDetails
