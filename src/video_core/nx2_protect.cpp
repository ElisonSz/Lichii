// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/nx2_protect.h"

namespace VideoCommon::Nx2Protect {

namespace {

constexpr u64 PageBits = 12;
constexpr u64 PageMask = (u64{1} << PageBits) - 1;
constexpr u64 LogPeriodNs = 10'000'000'000;
/// Pages of one rearmed run followed one by one; the rest of a long run is only counted.
constexpr std::size_t PagesPerRun = 64;
/// Churn pages whose upload scans are counted in the next window.
constexpr std::size_t WatchSize = 64;

constexpr std::size_t NumSources = static_cast<std::size_t>(Source::Count);
constexpr std::size_t NumTrackers = static_cast<std::size_t>(Tracker::Count);

struct Counter {
    std::atomic<u64> calls{};
    std::atomic<u64> bytes{};
    std::atomic<u64> ns{};
};

/// [source][tracker][protect (1) or unprotect (0)][on a CPU core (1) or not (0)]
std::array<Counter, NumSources * NumTrackers * 2 * 2> g_protect;

constexpr std::size_t ProtectIndex(std::size_t source, std::size_t tracker, bool cached,
                                   bool cpu_core) {
    return ((source * NumTrackers + tracker) * 2 + (cached ? 1 : 0)) * 2 + (cpu_core ? 1 : 0);
}

/// Distinct pages, approximately: a hashed bitmap of 65536 bits.
struct PageSet {
    std::array<std::atomic<u64>, 1024> words{};
    std::atomic<u64> distinct{};

    void Add(u64 page) noexcept {
        const u64 hash = (page * 0x9E3779B97F4A7C15ULL) >> 48;
        const u64 bit = u64{1} << (hash & 63);
        std::atomic<u64>& word = words[static_cast<std::size_t>(hash >> 6)];
        if ((word.load(std::memory_order_relaxed) & bit) != 0) {
            return;
        }
        if ((word.fetch_or(bit, std::memory_order_relaxed) & bit) == 0) {
            distinct.fetch_add(1, std::memory_order_relaxed);
        }
    }

    u64 Take() noexcept {
        for (std::atomic<u64>& word : words) {
            word.store(0, std::memory_order_relaxed);
        }
        return distinct.exchange(0, std::memory_order_relaxed);
    }
};

struct TrapTotals {
    std::atomic<u64> traps{};
    std::atomic<u64> asked{};
    std::atomic<u64> collected{};
    PageSet pages;
};
TrapTotals g_writes;
TrapTotals g_reads;

/// Trap counts of this thread not yet added to the totals: the cores trap far more often than
/// pages change protection.
struct LocalTraps {
    u64 write_traps{};
    u64 write_asked{};
    u64 write_collected{};
    u64 read_traps{};
    u64 read_asked{};
    u32 pending{};
};
thread_local LocalTraps t_traps;

void FlushTraps() noexcept {
    LocalTraps& local = t_traps;
    g_writes.traps.fetch_add(local.write_traps, std::memory_order_relaxed);
    g_writes.asked.fetch_add(local.write_asked, std::memory_order_relaxed);
    g_writes.collected.fetch_add(local.write_collected, std::memory_order_relaxed);
    g_reads.traps.fetch_add(local.read_traps, std::memory_order_relaxed);
    g_reads.asked.fetch_add(local.read_asked, std::memory_order_relaxed);
    local = LocalTraps{};
}

void CountTrap() noexcept {
    if (++t_traps.pending >= 64) {
        FlushTraps();
    }
}

std::array<std::atomic<u64>, static_cast<std::size_t>(InPlace::Count)> g_in_place_count;
std::atomic<u64> g_in_place_bytes{};

/// The churn pages of a window and their upload scans; two, so that a scan reading one while
/// the log rebuilds the other never mixes them.
struct WatchSet {
    std::array<std::atomic<u64>, WatchSize> pages{};
    std::array<std::atomic<u32>, WatchSize> scans{};
    std::array<std::atomic<u32>, WatchSize> rearms{};
    std::atomic<u32> count{};
};
std::array<WatchSet, 2> g_watch;
std::atomic<u32> g_watch_active{};

/// Index of the first watched page at or past `page`.
std::size_t WatchLowerBound(const WatchSet& set, u32 count, u64 page) noexcept {
    std::size_t low = 0;
    std::size_t high = count;
    while (low < high) {
        const std::size_t middle = (low + high) / 2;
        if (set.pages[middle].load(std::memory_order_relaxed) < page) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

struct ChurnWindow {
    u64 runs{};
    u64 pages{};      ///< pages followed one by one
    u64 bulk_pages{}; ///< pages of long runs past PagesPerRun
    u64 distinct{};
    u64 steady{}; ///< distinct pages whose streak reached 3
    std::array<u64, static_cast<std::size_t>(Interval::Count)> intervals{};
    std::array<u64, 4> streaks{}; ///< 0, 1, 2, 3 or more
};

struct ChurnState {
    std::mutex mutex;
    ChurnTable table;
    ChurnWindow window;
    u32 window_id{};
};
ChurnState& Churn() {
    static ChurnState state;
    return state;
}

std::mutex g_log_mutex;
std::atomic<u64> g_next_log{};
u64 g_window_start{};
std::string g_last_line; ///< `g_log_mutex`

std::string Ms(u64 ns) {
    return fmt::format("{:.2f}", static_cast<double>(ns) / 1e6);
}

std::string MiB(u64 bytes) {
    return fmt::format("{:.2f}", static_cast<double>(bytes) / static_cast<double>(1u << 20));
}

constexpr std::array<const char*, NumSources> SourceNames{
    "other", "cpu-write", "gpu-write", "collected", "invalidate", "inline", "rearm",
};

void Log(u64 now) {
    const u64 span_ns = now > g_window_start ? now - g_window_start : 1;
    g_window_start = now;
    const double seconds = static_cast<double>(span_ns) / 1e9;

    // Protection changes
    u64 protect_calls = 0, protect_bytes = 0, protect_ns = 0;
    u64 unprotect_calls = 0, unprotect_bytes = 0, unprotect_ns = 0, unprotect_core_ns = 0,
        protect_core_ns = 0;
    std::string sources;
    for (std::size_t source = 0; source < NumSources; ++source) {
        for (std::size_t tracker = 0; tracker < NumTrackers; ++tracker) {
            std::string part;
            for (const bool cached : {true, false}) {
                u64 calls = 0, bytes = 0, ns = 0, core_calls = 0;
                for (const bool cpu_core : {false, true}) {
                    Counter& counter = g_protect[ProtectIndex(source, tracker, cached, cpu_core)];
                    const u64 c = counter.calls.exchange(0, std::memory_order_relaxed);
                    const u64 b = counter.bytes.exchange(0, std::memory_order_relaxed);
                    const u64 n = counter.ns.exchange(0, std::memory_order_relaxed);
                    calls += c;
                    bytes += b;
                    ns += n;
                    if (cpu_core) {
                        core_calls += c;
                        (cached ? protect_core_ns : unprotect_core_ns) += n;
                    }
                }
                if (calls == 0) {
                    continue;
                }
                (cached ? protect_calls : unprotect_calls) += calls;
                (cached ? protect_bytes : unprotect_bytes) += bytes;
                (cached ? protect_ns : unprotect_ns) += ns;
                part += fmt::format(" {} {}/{}ms{}", cached ? "P" : "U", calls, Ms(ns),
                                    core_calls != 0 ? fmt::format(" ({} on cores)", core_calls)
                                                    : std::string{});
            }
            if (!part.empty()) {
                sources += fmt::format("; {}{}{}", SourceNames[source],
                                       tracker == static_cast<std::size_t>(Tracker::Single)
                                           ? "(texture/shader)"
                                           : "",
                                       part);
            }
        }
    }

    // Guest CPU traps on cached pages
    FlushTraps();
    const u64 write_traps = g_writes.traps.exchange(0, std::memory_order_relaxed);
    const u64 write_asked = g_writes.asked.exchange(0, std::memory_order_relaxed);
    const u64 write_collected = g_writes.collected.exchange(0, std::memory_order_relaxed);
    const u64 write_pages = g_writes.pages.Take();
    const u64 read_traps = g_reads.traps.exchange(0, std::memory_order_relaxed);
    const u64 read_asked = g_reads.asked.exchange(0, std::memory_order_relaxed);
    const u64 read_pages = g_reads.pages.Take();

    // Rearm churn, the watched pages' scans, and the next window's watched pages
    ChurnWindow churn;
    std::string busiest;
    u64 watched = 0, watch_scans = 0, watch_rearms = 0;
    {
        ChurnState& state = Churn();
        std::scoped_lock lock{state.mutex};
        churn = state.window;
        state.window = {};

        const u32 active = g_watch_active.load(std::memory_order_relaxed);
        WatchSet& current = g_watch[active];
        watched = current.count.load(std::memory_order_relaxed);
        for (std::size_t i = 0; i < watched; ++i) {
            watch_scans += current.scans[i].load(std::memory_order_relaxed);
            watch_rearms += current.rearms[i].load(std::memory_order_relaxed);
        }

        std::array<ChurnTable::Top, 4> top{};
        const std::size_t found = state.table.Busiest(state.window_id, false, top);
        for (std::size_t i = 0; i < found; ++i) {
            busiest += fmt::format(" {:#x} x{} s{}", top[i].page << PageBits, top[i].rearms,
                                   top[i].streak);
        }

        std::array<ChurnTable::Top, WatchSize> steady{};
        const std::size_t steady_count = state.table.Busiest(state.window_id, true, steady);
        std::sort(steady.begin(), steady.begin() + static_cast<std::ptrdiff_t>(steady_count),
                  [](const ChurnTable::Top& a, const ChurnTable::Top& b) { return a.page < b.page; });
        WatchSet& next = g_watch[active ^ 1];
        next.count.store(0, std::memory_order_relaxed);
        for (std::size_t i = 0; i < WatchSize; ++i) {
            next.pages[i].store(i < steady_count ? steady[i].page : ~u64{0},
                                std::memory_order_relaxed);
            next.scans[i].store(0, std::memory_order_relaxed);
            next.rearms[i].store(0, std::memory_order_relaxed);
        }
        next.count.store(static_cast<u32>(steady_count), std::memory_order_release);
        g_watch_active.store(active ^ 1, std::memory_order_release);

        state.table.Prune(now);
        ++state.window_id;
    }

    // In-place uploads of the GPU's own writes
    std::array<u64, static_cast<std::size_t>(InPlace::Count)> in_place{};
    for (std::size_t i = 0; i < in_place.size(); ++i) {
        in_place[i] = g_in_place_count[i].exchange(0, std::memory_order_relaxed);
    }
    const u64 in_place_bytes = g_in_place_bytes.exchange(0, std::memory_order_relaxed);
    std::string in_place_text;
    if (g_inline) {
        const auto count = [&](InPlace result) { return in_place[static_cast<std::size_t>(result)]; };
        in_place_text = fmt::format(
            " | in place: {} uploads {} MiB, {} repeated; as before: untracked {}, again before "
            "use {}, gpu data {}, no buffer {}, render pass {}, large {}",
            count(InPlace::Uploaded), MiB(in_place_bytes), count(InPlace::Repeated),
            count(InPlace::Untracked), count(InPlace::AgainBeforeUse), count(InPlace::GpuData),
            count(InPlace::NoBuffer), count(InPlace::RenderPass), count(InPlace::Large));
    }

    const auto interval = [&](Interval bucket) {
        return churn.intervals[static_cast<std::size_t>(bucket)];
    };
    g_last_line = fmt::format(
        "Nx2Protect {:.1f} s: protect {} calls {} MiB {} ms ({} ms on cores), unprotect {} calls {} "
        "MiB {} ms ({} ms on cores){} | CPU traps: writes {} on ~{} pages ({} reached the caches, "
        "{} collected), reads {} on ~{} pages ({} asked the caches) | rearmed pages {} in {} runs "
        "(+{} in long runs), {} distinct, {} distinct rearmed 4+ times in a row; again after <4ms "
        "{} <12ms {} <25ms {} <50ms {} <100ms {} later {} first {}; streak 0:{} 1:{} 2:{} 3+:{} | "
        "busiest{} | {} watched churn pages: {} upload scans over them, {} rearms{}",
        seconds, protect_calls, MiB(protect_bytes), Ms(protect_ns), Ms(protect_core_ns),
        unprotect_calls, MiB(unprotect_bytes), Ms(unprotect_ns), Ms(unprotect_core_ns), sources,
        write_traps, write_pages, write_asked, write_collected, read_traps, read_pages, read_asked,
        churn.pages, churn.runs, churn.bulk_pages, churn.distinct, churn.steady,
        interval(Interval::Under4ms), interval(Interval::Under12ms), interval(Interval::Under25ms),
        interval(Interval::Under50ms), interval(Interval::Under100ms), interval(Interval::Later),
        interval(Interval::First), churn.streaks[0], churn.streaks[1], churn.streaks[2],
        churn.streaks[3], busiest.empty() ? std::string{" none"} : busiest, watched, watch_scans,
        watch_rearms, in_place_text);
    LOG_INFO(HW_GPU, "{}", g_last_line);
}

void MaybeLog(u64 now) {
    if (now < g_next_log.load(std::memory_order_relaxed)) {
        return;
    }
    std::unique_lock lock{g_log_mutex, std::try_to_lock};
    if (!lock.owns_lock() || now < g_next_log.load(std::memory_order_relaxed)) {
        return;
    }
    g_next_log.store(now + LogPeriodNs, std::memory_order_relaxed);
    Log(now);
}

} // Anonymous namespace

void ReadSwitch() {
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    g_stats = enabled("EDEN_NX2_PROTECT_STATS");
    g_inline = enabled("EDEN_NX2_PROTECT_INLINE");
    if (g_stats) {
        std::scoped_lock lock{g_log_mutex};
        g_window_start = NowNs();
        g_next_log.store(g_window_start + LogPeriodNs, std::memory_order_relaxed);
    }
    if (g_stats || g_inline) {
        LOG_INFO(HW_GPU, "Nx2Protect: stats {}, in-place uploads of the GPU's own writes {}",
                 g_stats ? "on" : "off", g_inline ? "on" : "off");
    }
}

u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void NoteProtect(bool cached, u64 size, u64 ns, bool cpu_core, u64 now) {
    Counter& counter = g_protect[ProtectIndex(static_cast<std::size_t>(t_source),
                                              static_cast<std::size_t>(t_tracker), cached,
                                              cpu_core)];
    counter.calls.fetch_add(1, std::memory_order_relaxed);
    counter.bytes.fetch_add(size, std::memory_order_relaxed);
    counter.ns.fetch_add(ns, std::memory_order_relaxed);
    MaybeLog(now);
}

void NoteCpuWrite(u64 vaddr) {
    ++t_traps.write_traps;
    g_writes.pages.Add(vaddr >> PageBits);
    CountTrap();
}

void NoteCpuWriteAsked(bool collected) {
    ++t_traps.write_asked;
    if (collected) {
        ++t_traps.write_collected;
    }
}

void NoteCpuRead(u64 vaddr) {
    ++t_traps.read_traps;
    g_reads.pages.Add(vaddr >> PageBits);
    CountTrap();
}

void NoteCpuReadAsked() {
    ++t_traps.read_asked;
}

void NoteRearm(std::span<const std::pair<u64, std::size_t>> runs) {
    if (runs.empty()) {
        return;
    }
    const u64 now = NowNs();
    {
        ChurnState& state = Churn();
        std::scoped_lock lock{state.mutex};
        const u32 active = g_watch_active.load(std::memory_order_relaxed);
        WatchSet& watch = g_watch[active];
        const u32 watched = watch.count.load(std::memory_order_relaxed);
        for (const auto& [address, size] : runs) {
            ++state.window.runs;
            const u64 first = address >> PageBits;
            const u64 pages = (size + PageMask) >> PageBits;
            const u64 followed = (std::min<u64>)(pages, PagesPerRun);
            state.window.bulk_pages += pages - followed;
            for (u64 page = first; page < first + followed; ++page) {
                const ChurnTable::Result result = state.table.Note(page, now, state.window_id);
                ++state.window.pages;
                ++state.window.intervals[static_cast<std::size_t>(result.interval)];
                ++state.window.streaks[(std::min<u32>)(result.streak, 3)];
                if (result.first_in_window) {
                    ++state.window.distinct;
                }
                if (result.steady) {
                    ++state.window.steady;
                }
                if (watched != 0) {
                    const std::size_t i = WatchLowerBound(watch, watched, page);
                    if (i < watched && watch.pages[i].load(std::memory_order_relaxed) == page) {
                        watch.rearms[i].fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }
    MaybeLog(now);
}

void NoteScan(DAddr begin, DAddr end) {
    WatchSet& watch = g_watch[g_watch_active.load(std::memory_order_acquire)];
    const u32 watched = watch.count.load(std::memory_order_acquire);
    if (watched == 0 || end <= begin) {
        return;
    }
    const u64 last = (end + PageMask) >> PageBits;
    for (std::size_t i = WatchLowerBound(watch, watched, begin >> PageBits);
         i < watched && watch.pages[i].load(std::memory_order_relaxed) < last; ++i) {
        // The scans of a watched page, whether it was dirty or not: kept unprotected, the page
        // would be uploaded at each one.
        watch.scans[i].fetch_add(1, std::memory_order_relaxed);
    }
}

std::string LastLogLine() {
    std::scoped_lock lock{g_log_mutex};
    return g_last_line;
}

void NoteInPlace(InPlace result, u64 bytes) {
    g_in_place_count[static_cast<std::size_t>(result)].fetch_add(1, std::memory_order_relaxed);
    if (bytes != 0) {
        g_in_place_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

struct ChurnTable::Impl {
    struct Page {
        u64 last_ns{};
        u32 window{};
        u32 rearms{}; ///< in `window`
        u32 streak{};
        bool steady_noted{}; ///< `steady` reported in `window`
    };
    std::unordered_map<u64, Page> pages;
};

ChurnTable::ChurnTable() : impl{std::make_unique<Impl>()} {}

ChurnTable::~ChurnTable() = default;

ChurnTable::Result ChurnTable::Note(u64 page, u64 now, u32 window) {
    const auto [it, inserted] = impl->pages.try_emplace(page);
    Impl::Page& entry = it->second;
    Result result{};
    if (inserted) {
        result.interval = Interval::First;
        entry.streak = 0;
    } else {
        const u64 since = now > entry.last_ns ? now - entry.last_ns : 0;
        if (since < 4'000'000) {
            result.interval = Interval::Under4ms;
        } else if (since < 12'000'000) {
            result.interval = Interval::Under12ms;
        } else if (since < 25'000'000) {
            result.interval = Interval::Under25ms;
        } else if (since < 50'000'000) {
            result.interval = Interval::Under50ms;
        } else if (since < 100'000'000) {
            result.interval = Interval::Under100ms;
        } else {
            result.interval = Interval::Later;
        }
        entry.streak = since <= StreakNs ? entry.streak + 1 : 0;
    }
    if (inserted || entry.window != window) {
        entry.window = window;
        entry.rearms = 0;
        entry.steady_noted = false;
        result.first_in_window = true;
    }
    ++entry.rearms;
    entry.last_ns = now;
    result.streak = entry.streak;
    if (entry.streak >= 3 && !entry.steady_noted) {
        entry.steady_noted = true;
        result.steady = true;
    }
    return result;
}

void ChurnTable::Prune(u64 now) {
    std::erase_if(impl->pages, [now](const auto& item) {
        return now > item.second.last_ns && now - item.second.last_ns > ForgetNs;
    });
}

std::size_t ChurnTable::Busiest(u32 window, bool steady_only, std::span<Top> out) const {
    if (out.empty()) {
        return 0;
    }
    std::vector<Top> candidates;
    for (const auto& [page, entry] : impl->pages) {
        if (entry.window != window || (steady_only && entry.streak < 3)) {
            continue;
        }
        candidates.push_back(Top{.page = page, .rearms = entry.rearms, .streak = entry.streak});
    }
    const std::size_t count = (std::min)(candidates.size(), out.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(count),
                      candidates.end(), [](const Top& a, const Top& b) {
                          return a.rearms != b.rearms ? a.rearms > b.rearms : a.page < b.page;
                      });
    std::copy_n(candidates.begin(), count, out.begin());
    return count;
}

std::size_t ChurnTable::Size() const noexcept {
    return impl->pages.size();
}

} // namespace VideoCommon::Nx2Protect
