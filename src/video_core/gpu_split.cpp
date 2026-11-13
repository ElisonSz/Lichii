// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <thread>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __GLIBC__
#include <execinfo.h>
#endif
#include <cstring>
#include <string>
#include <string_view>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_sync_trace.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/nx2_acquire_stats.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"

#include "video_core/wait_stats.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};

/// TEST: an engine's name, by the class it was bound as (MethodEndingAt).
[[nodiscard]] std::string_view EngineName(std::optional<Tegra::Engines::EngineTypes> type) {
    using Tegra::Engines::EngineTypes;
    if (!type) {
        return "an unbound engine";
    }
    switch (*type) {
    case EngineTypes::Nv01Timer:
        return "Nv01Timer";
    case EngineTypes::KeplerCompute:
        return "KeplerCompute";
    case EngineTypes::Maxwell3D:
        return "Maxwell3D";
    case EngineTypes::Fermi2D:
        return "Fermi2D";
    case EngineTypes::MaxwellDMA:
        return "MaxwellDMA";
    case EngineTypes::KeplerMemory:
        return "KeplerMemory";
    }
    return "an engine";
}

bool BackTickEnabled() {
    const char* const value = std::getenv("EDEN_NX2_BACK_TICK");
    return value == nullptr || value[0] != '0';
}

std::atomic<bool> g_back_tick_enabled{BackTickEnabled()};

BackIdleTrace::FrontState WaitState(DrainSite site) {
    return site == DrainSite::ListWriters || site == DrainSite::ListSync ||
                   site == DrainSite::ListFlush
               ? BackIdleTrace::FrontState::Writers
               : BackIdleTrace::FrontState::Drain;
}

// A wait of the front for its back, counted by site while g_acquire_stats counts.
class CountedWait {
public:
    explicit CountedWait(DrainSite site_) : site{static_cast<size_t>(site_)} {
        if (g_acquire_counting) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~CountedWait() {
        if (!start) {
            return;
        }
        Count(g_acquire_stats.site_drains[site]);
        Count(g_acquire_stats.site_drain_ns[site],
              static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - *start)
                                   .count()));
    }

private:
    size_t site;
    std::optional<std::chrono::steady_clock::time_point> start;
};

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (!instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        stack += fmt::format(" {}+{:#x}", base_name ? base_name + 1 : name,
                             reinterpret_cast<uintptr_t>(frames[i]) -
                                 reinterpret_cast<uintptr_t>(module));
    }
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on thread {} while the back runs; stack:{}",
              where, thread, stack);
#else
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (on_back_thread || !instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
#ifdef __GLIBC__
    void* frames[40];
    const int count = backtrace(frames, 40);
    if (char** const symbols = backtrace_symbols(frames, count)) {
        for (int i = 1; i < count; ++i) {
            stack += fmt::format("\n  {}", symbols[i]);
        }
        std::free(symbols);
    }
#endif
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on a thread other than the back; stack:{}", where,
              stack);
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::ReadTickSwitch() {
    g_back_tick_enabled.store(BackTickEnabled(), std::memory_order_relaxed);
}

bool GpuSplit::TickEnabled() {
    return g_back_tick_enabled.load(std::memory_order_relaxed);
}

void GpuSplit::RequestTick() {
    if (!TickEnabled()) {
        return;
    }
    u64 state = tick_state.load(std::memory_order_relaxed);
    const u64 generation = state & ~(TickAccepting | TickRequested);
    do {
        if ((state & TickAccepting) == 0 ||
            (state & ~(TickAccepting | TickRequested)) != generation) {
            return;
        }
    } while (!tick_state.compare_exchange_weak(
        state, state | TickRequested, std::memory_order_seq_cst, std::memory_order_relaxed));
    // Paired with the back's check after setting back_waiting: one side sees the other.
    if (back_waiting.load(std::memory_order_seq_cst)) {
        WakeBack();
    }
}

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    const char* const verify = std::getenv("EDEN_NX2_ACQUIRE_VERIFY");
    g_acquire_verify = verify != nullptr && verify[0] == '1';
    g_acquire_counting = g_acquire_verify || Nx2Profile::Enabled() ||
                         std::getenv("EDEN_NX2_ACQUIRE_STATS") != nullptr ||
                         std::getenv("EDEN_NX2_GUEST_THREAD_STATS") != nullptr;
    profile_publishes = Nx2Profile::Enabled();
    // The methods sent are logged for the profile's and the verify's reports (MethodEndingAt),
    // not for the statistics switches alone.
    log_methods = g_acquire_verify || Nx2Profile::Enabled();
    ring = std::make_unique<u64[]>(RingWords);
    if (RunAhead::enabled) {
        run_ahead_log = std::make_unique<RunAheadWrite[]>(RunAheadLogSize);
    }
    ++starts; // Before the back starts: it tags the entries of fenced writes with it (EntryTag)
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}",
             front_macros ? ", macros on the first" : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    WakeBack();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    last_sync_position = 0;
    synced_writer_position = 0;
    last_entry_start = 0;
    back_entry_start = 0;
    {
        std::scoped_lock lock{fenced_entries_mutex};
        fenced_entries.clear();
        fenced_sealed = 0;
        fenced_landed.store(0, std::memory_order_release);
    }
    semaphore_writes.clear();
    semaphore_writes_limit = 1024;
    last_any_write = 0;
    last_unplaced_write = 0;
    last_mapping_change = 0;
    last_unplaced_kind = UnplacedKind::Other;
    last_mapping_kind = UnplacedKind::Other;
    first_unpublished_ticks = 0;
    back_idle_since.store(0, std::memory_order_relaxed);
    last_noted_write = 0;
    write_sets.fill({});
    coarse_sets.fill({});
    run_ahead_log.reset();
    run_ahead_count = 0;
    run_ahead_lost = 0;
    last_unknown_writer = 0;
    run_ahead_waiting = 0;
    watched_pages.fill({});
    watched_count = 0;
    landing_window_open = false;
    landing_close_pending = false;
    landing_window_tag = 0;
    landing_close_end = 0;
    landing_converting = false;
    landing_closing = false;
    landing_uncommitted = false;
    landing_check = 0;
    landing_back_tag = 0;
    landing_rasterizer = nullptr;
    landing_ack.store(0, std::memory_order_relaxed);
    published_local = 0;
    publish_held = false;
    logged_method_count = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    state_engine = nullptr;
    state_count = 0;
    state_breaks = false;
    state_anywhere = false;
    state_flags.fill(0);
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_idle_at.store(~u64{0}, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
}

GpuSplit::FencedEntry::~FencedEntry() {
    split.CompleteFencedEntry(tag);
}

std::shared_ptr<void> GpuSplit::TrackFencedWrite() {
    if (!on_back_thread) {
        return {};
    }
    if (!back_fenced_entry) {
        const u64 tag = EntryTag(back_entry_start);
        std::scoped_lock lock{fenced_entries_mutex};
        fenced_entries.push_back({tag});
        back_fenced_entry = std::make_shared<FencedEntry>(*this, tag);
    }
    return back_fenced_entry;
}

void GpuSplit::PublishFencedEntries() {
    while (!fenced_entries.empty() && fenced_entries.front().complete) {
        fenced_entries.pop_front();
    }
    // FenceManager normally completes reports in FIFO order. Do not rely on it: a completed
    // later entry must never retire an earlier unfinished report, even across fence managers.
    // sealed also covers entries with no deferred reports (including a query dropped on entry).
    const u64 landed = fenced_entries.empty() ? fenced_sealed
                                              : std::min(fenced_sealed, fenced_entries.front().tag);
    fenced_landed.store(landed, std::memory_order_release);
    if (landing_waits) [[unlikely]] {
        landed_cv.notify_all();
    }
}

std::optional<u64> GpuSplit::UnlandedFencedEntry(u64 device_address) const {
    const auto it = semaphore_writes.find(device_address & ~u64{3});
    if (!active || it == semaphore_writes.end() || !it->second.fenced ||
        fenced_landed.load(std::memory_order_acquire) > it->second.entry) {
        return std::nullopt;
    }
    return it->second.entry;
}

void GpuSplit::HoldBackUntilLanded(u64 entry) {
    // It changes nothing a draw resolves to (the idle hook's submit runs between entries too).
    Defer([this, entry] { WaitLanded(entry); }, FastDrawNeutral, Writes::Noted);
    if (RunAhead::stats) {
        ++RunAhead::counters.w1_holds;
    }
}

void GpuSplit::OpenLandingWindow(u64 entry, u64 check, VideoCore::RasterizerInterface* rasterizer) {
    if (!active) {
        return;
    }
    auto& counts = RunAhead::counters;
    if (!landing_window_open) {
        landing_window_open = true;
        landing_close_pending = false; // Notes stay fenced: the window is open again
        landing_window_tag = entry;
        Defer([this, entry, check, rasterizer] { BackOpenLanding(entry, check, rasterizer); },
              FastDrawNeutral, Writes::Noted);
        counts.windows_opened += RunAhead::stats ? 1 : 0;
        return;
    }
    if (entry > landing_window_tag) {
        // Fences run in order: once this later release landed, the window's earlier ones have.
        landing_window_tag = entry;
        Defer(
            [this, entry, check] {
                landing_check = check;
                landing_back_tag = (std::max)(landing_back_tag, entry);
            },
            FastDrawNeutral, Writes::Noted);
        counts.windows_extended += RunAhead::stats ? 1 : 0;
    }
}

void GpuSplit::CheckLandingWindow(VideoCore::RasterizerInterface* rasterizer) {
    auto& counts = RunAhead::counters;
    if (landing_close_pending && landing_ack.load(std::memory_order_acquire) >= landing_close_end) {
        landing_close_pending = false;
        counts.windows_acked += RunAhead::stats ? 1 : 0;
    }
    if (!landing_window_open ||
        fenced_landed.load(std::memory_order_acquire) <= landing_window_tag) {
        return;
    }
    landing_window_open = false;
    landing_close_pending = true;
    Defer([this, rasterizer] { BackCloseLanding(rasterizer); }, FastDrawNeutral, Writes::Noted);
    landing_close_end = write_position;
    counts.windows_closed += RunAhead::stats ? 1 : 0;
}

void GpuSplit::BackOpenLanding(u64 entry, u64 check, VideoCore::RasterizerInterface* rasterizer) {
    landing_back_tag = (std::max)(landing_back_tag, entry);
    landing_converting = true;
    landing_closing = false;
    landing_check = check;
    landing_rasterizer = rasterizer;
    // The submit the baseline's back made once it ran dry at the acquire: the release's fence
    // goes to the host GPU now.
    if (idle_hook != nullptr) {
        idle_hook(idle_argument);
    }
}

void GpuSplit::BackCloseLanding(VideoCore::RasterizerInterface* rasterizer) {
    landing_closing = true;
    landing_rasterizer = rasterizer;
    // Converted landings not attached to a fence yet go with the next one: make it now.
    MakeLandingFence();
}

void GpuSplit::MakeLandingFence() {
    if (!landing_uncommitted || landing_rasterizer == nullptr) {
        return;
    }
    landing_uncommitted = false;
    landing_rasterizer->SignalReference();
    landing_fence_at = std::chrono::steady_clock::now();
    RunAhead::landing_counts.fences.fetch_add(1, std::memory_order_relaxed);
}

void GpuSplit::MaybeEndLandings(u64 position) {
    if (landing_pending.load(std::memory_order_acquire) == 0) {
        // Nothing converted is pending: what lands from here on can land at run again.
        landing_converting = false;
        landing_closing = false;
        landing_check = 0;
        landing_ack.store(position, std::memory_order_release);
        return;
    }
    RunAhead::landing_counts.extended.fetch_add(1, std::memory_order_relaxed);
    if (landing_uncommitted &&
        std::chrono::steady_clock::now() - landing_fence_at > std::chrono::milliseconds{1}) {
        MakeLandingFence();
    }
}

struct GpuSplit::LandingToken {
    LandingToken(GpuSplit& split_, u64 device_, u64 check_)
        : split{split_}, device{device_}, check{check_} {}
    ~LandingToken() {
        if (device != 0) {
            std::scoped_lock lock{split.landing_words_mutex};
            if (const auto it = split.landing_words.find(device);
                it != split.landing_words.end() && --it->second == 0) {
                split.landing_words.erase(it);
            }
            split.landing_pages[LandingPageSlot(device)].fetch_sub(1, std::memory_order_release);
        }
        if (split.landing_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // The last one: a write waiting before it overwrites may go on (WaitBeforeOverwrite).
            std::scoped_lock lock{split.fenced_entries_mutex};
            if (split.landing_waits) {
                split.landed_cv.notify_all();
            }
        }
    }
    LandingToken(const LandingToken&) = delete;
    LandingToken& operator=(const LandingToken&) = delete;
    GpuSplit& split;
    u64 device; ///< ..._VERIFY: the word it writes (0: not tracked)
    u64 check;  ///< ..._VERIFY: its window's check (RunAhead::PackLandingCheck), 0: none
};

std::shared_ptr<void> GpuSplit::ConvertedLanding(u64 device, bool report) {
    auto& counts = RunAhead::landing_counts;
    const u64 pending = landing_pending.fetch_add(1, std::memory_order_acq_rel) + 1;
    (report ? counts.reports : counts.operations).fetch_add(1, std::memory_order_relaxed);
    for (u64 max = counts.pending_max.load(std::memory_order_relaxed);
         pending > max &&
         !counts.pending_max.compare_exchange_weak(max, pending, std::memory_order_relaxed);) {
    }
    // Its word: the view holds settled reads of it until it landed (puller's SettledReadDecides).
    const u64 word = device & ~u64{3};
    if (word != 0) {
        std::scoped_lock lock{landing_words_mutex};
        ++landing_words[word];
        landing_pages[LandingPageSlot(word)].fetch_add(1, std::memory_order_release);
    }
    return std::make_shared<LandingToken>(*this, word, landing_check);
}

namespace {
/// EDEN_NX2_RUNAHEAD_LANDINGS_VERIFY: the check slot the next window takes (front only).
size_t g_landing_slot{};
} // Anonymous namespace

u64 GpuSplit::StartLandingCheck(u64 device, u32 value, u32 operation, u64 address,
                                const void* memory, const QueuedSemaphoreWrite& release,
                                u64 release_entry, const void* waiter, bool met) {
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    auto& checks = RunAhead::landing_checks;
    const size_t slot = g_landing_slot++ % checks.size();
    auto& check = checks[slot];
    // The new generation first: a reader that no longer finds the earlier window's waiter, or
    // finds the latch cleared or this window's fields, finds the generation moved as well
    // (CheckConvertedLanding): the earlier window's landings still pending count as unchecked.
    const u32 generation = check.generation.load(std::memory_order_relaxed) + 1;
    check.generation.store(generation, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    if (check.registered) {
        waiters.Remove(&check);
        check.registered = false;
    }
    check.latched.store(false, std::memory_order_relaxed);
    check.device.store(device, std::memory_order_relaxed);
    check.value.store(value, std::memory_order_relaxed);
    check.operation.store(operation, std::memory_order_relaxed);
    check.address.store(address, std::memory_order_relaxed);
    check.memory.store(memory, std::memory_order_relaxed);
    check.release_position.store(release.position, std::memory_order_relaxed);
    check.release_entry.store(release_entry, std::memory_order_relaxed);
    check.release_payload.store(release.payload, std::memory_order_relaxed);
    check.release_writer.store(static_cast<u8>(release.writer), std::memory_order_relaxed);
    check.release_at_least.store(release.at_least, std::memory_order_relaxed);
    check.release_windowed.store(release.windowed, std::memory_order_relaxed);
    if (!waiters.Add(&check, device, value, operation)) {
        RunAhead::counters.checks_unregistered += 1;
        return 0;
    }
    check.registered = true;
    // A release that landed before the check's waiter was added latched the acquire's own, or,
    // landed before the acquire's waiter was added too (its entry not complete yet), was in memory
    // when the caller read the word (`met`).
    if (met || waiters.Latched(waiter)) {
        check.latched.store(true, std::memory_order_relaxed);
    }
    return RunAhead::PackLandingCheck(slot, generation);
}

void GpuSplit::SweepLandingChecks() {
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    for (auto& check : RunAhead::landing_checks) {
        if (check.registered && waiters.Latched(&check)) {
            // The flag before the waiter goes: a reader that no longer finds the waiter finds the
            // flag set (CheckConvertedLanding looks at the waiter first).
            check.latched.store(true, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            waiters.Remove(&check);
            check.registered = false;
        }
    }
}

void GpuSplit::CheckConvertedLanding(const std::shared_ptr<void>& token) {
    auto& counts = RunAhead::landing_counts;
    const auto* const landing = static_cast<const LandingToken*>(token.get());
    if (landing == nullptr || landing->check == 0) {
        counts.unchecked.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const size_t slot = static_cast<size_t>(landing->check & 0xff) - 1;
    const u32 generation = static_cast<u32>(landing->check >> 8);
    auto& check = RunAhead::landing_checks[slot % RunAhead::landing_checks.size()];
    if (check.generation.load(std::memory_order_acquire) != generation) {
        counts.unchecked.fetch_add(1, std::memory_order_relaxed); // The slot went to a later window
        return;
    }
    // The waiter, then the flag, then the generation again (RunAhead::LandingCheck): a sweep moves
    // a latch from the waiter to the flag, a later window moves the generation first.
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    const bool in_table = waiters.Latched(&check);
    std::atomic_thread_fence(std::memory_order_acquire);
    const bool flagged = check.latched.load(std::memory_order_relaxed);
    const u64 device = check.device.load(std::memory_order_relaxed);
    const u32 value = check.value.load(std::memory_order_relaxed);
    const u32 operation = check.operation.load(std::memory_order_relaxed);
    const u64 address = check.address.load(std::memory_order_relaxed);
    const auto* const memory =
        static_cast<const Tegra::MemoryManager*>(check.memory.load(std::memory_order_relaxed));
    const u64 release_position = check.release_position.load(std::memory_order_relaxed);
    const u64 release_entry = check.release_entry.load(std::memory_order_relaxed);
    const u32 release_payload = check.release_payload.load(std::memory_order_relaxed);
    const u32 release_writer = check.release_writer.load(std::memory_order_relaxed);
    const bool release_at_least = check.release_at_least.load(std::memory_order_relaxed);
    const bool release_windowed = check.release_windowed.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (check.generation.load(std::memory_order_relaxed) != generation) {
        counts.unchecked.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (in_table || flagged) {
        counts.ordered.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    counts.out_of_order.fetch_add(1, std::memory_order_relaxed);
    if (static std::atomic<u32> logged{}; logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        // What decides it in the next run: whether the release's entry landed by now (as the
        // front counts it), the acquire's word now (the release dropped, or written over), and
        // whether the check is latched now (a reader that ran into the front's bookkeeping).
        const bool latched_now =
            waiters.Latched(&check) || check.latched.load(std::memory_order_acquire);
        const u32 generation_now = check.generation.load(std::memory_order_acquire);
        const bool entry_landed =
            Get().fenced_landed.load(std::memory_order_acquire) > release_entry;
        const u32 guest = memory != nullptr ? memory->Read<u32>(address) : 0;
        const auto shadow = Tegra::Nx2MirrorShadow::Get().Peek(device);
        LOG_ERROR(HW_GPU,
                  "Run-ahead landings verify: a converted landing (word {:#x}) was applied before "
                  "its window's release landed (the window's acquire: device {:#x}, op {}, {:#x}; "
                  "its release: {}{:#x} at {:#x}, writer {}, {}; check slot {}, generation {}). "
                  "Now: the release's entry {} as the front counts, the acquire's word holds "
                  "{:#x}{}, the check is {}latched (generation {})",
                  landing->device, device, operation, value, release_at_least ? "at least " : "",
                  release_payload, release_position, release_writer,
                  release_windowed ? "fenced by a window" : "fence-flagged", slot, generation,
                  entry_landed ? "landed" : "not landed", guest,
                  shadow ? fmt::format(" (mirror shadow {:#x} set over {:#x})", shadow->first,
                                       shadow->second)
                         : std::string{},
                  latched_now ? "" : "not ", generation_now);
    }
}

void GpuSplit::WaitBeforeOverwrite() {
    const auto start = std::chrono::steady_clock::now();
    const auto done = [this] {
        return landing_pending.load(std::memory_order_acquire) == 0 &&
               fenced_landed.load(std::memory_order_acquire) > landing_back_tag;
    };
    if (done()) {
        return;
    }
    // As WaitLanded: what they wait for goes to a fence and to the host GPU first.
    SealFencedEntries(back_entry_start);
    MakeLandingFence();
    if (idle_hook != nullptr) {
        idle_hook(idle_argument);
    }
    const std::stop_token stop = back_thread.get_stop_token();
    bool warned = false;
    std::unique_lock lock{fenced_entries_mutex};
    landing_waits = true;
    while (!done() && !stop.stop_requested()) {
        // In the middle of an engine method: no tick is served here, unlike between entries.
        if (landed_cv.wait_for(lock, std::chrono::milliseconds{1}) != std::cv_status::timeout) {
            continue;
        }
        if (!warned && std::chrono::steady_clock::now() - start > std::chrono::seconds{2}) {
            warned = true;
            LOG_WARNING(HW_GPU,
                        "Run-ahead landings: a write waits 2 s for {} converted landings and the "
                        "window's releases (entry {:#x}, landed {:#x})",
                        landing_pending.load(std::memory_order_relaxed), landing_back_tag,
                        fenced_landed.load(std::memory_order_relaxed));
        }
    }
    landing_waits = false;
    lock.unlock();
    auto& counts = RunAhead::landing_counts;
    counts.overwrite_waits.fetch_add(1, std::memory_order_relaxed);
    counts.overwrite_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count()),
        std::memory_order_relaxed);
}

bool GpuSplit::LandingPendingAt(u64 device) const {
    if (landing_pending.load(std::memory_order_acquire) == 0) {
        return false;
    }
    std::scoped_lock lock{landing_words_mutex};
    return landing_words.contains(device & ~u64{3});
}

bool GpuSplit::WaitForLandingsAt(u64 device, std::chrono::milliseconds limit) const {
    const auto start = std::chrono::steady_clock::now();
    while (LandingPendingAt(device)) {
        if (std::chrono::steady_clock::now() - start > limit) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

bool GpuSplit::LandingPendingIn(u64 device, u64 size) const {
    if (size == 0 || landing_pending.load(std::memory_order_acquire) == 0) {
        return false;
    }
    const u64 begin = device & ~u64{3};
    const u64 end = device + size;
    // Most writes in a window go to pages no pending landing is on: no lock for those.
    if (((end - 1) >> 12) - (begin >> 12) < LandingPageSlots) {
        bool any = false;
        for (u64 page = begin >> 12; page <= (end - 1) >> 12 && !any; ++page) {
            any = landing_pages[LandingPageSlot(page << 12)].load(std::memory_order_acquire) != 0;
        }
        if (!any) {
            return false;
        }
    }
    std::scoped_lock lock{landing_words_mutex};
    // A few dozen words at most (the pending landings).
    return std::ranges::any_of(landing_words, [begin, end](const auto& word) {
        return word.first >= begin && word.first < end;
    });
}

void GpuSplit::WaitLanded(u64 entry) {
    if (fenced_landed.load(std::memory_order_acquire) > entry) {
        return;
    }
    // Every entry before this one ran, so their reports are registered (as after a batch).
    SealFencedEntries(back_entry_start);
    if (fenced_landed.load(std::memory_order_acquire) > entry) {
        return;
    }
    // The release's fence may wait for the deferred submit only this thread makes
    // (FenceManager::WaitPendingFences); nothing is recorded while it waits.
    if (idle_hook != nullptr) {
        idle_hook(idle_argument);
    }
    const auto start = std::chrono::steady_clock::now();
    const std::stop_token stop = back_thread.get_stop_token();
    bool warned = false;
    std::unique_lock lock{fenced_entries_mutex};
    landing_waits = true;
    while (fenced_landed.load(std::memory_order_relaxed) <= entry && !stop.stop_requested()) {
        if (landed_cv.wait_for(lock, std::chrono::milliseconds{1}) != std::cv_status::timeout) {
            continue;
        }
        // Serve a guest read meanwhile, as between entries.
        lock.unlock();
        MaybeTick();
        lock.lock();
        if (!warned && std::chrono::steady_clock::now() - start > std::chrono::seconds{2}) {
            warned = true;
            LOG_WARNING(HW_GPU,
                        "Run-ahead W1: the back waits 2 s for a fence-flagged release to land "
                        "(entry {:#x}, landed {:#x})",
                        entry, fenced_landed.load(std::memory_order_relaxed));
        }
    }
    landing_waits = false;
    lock.unlock();
    RunAhead::holds.waits.fetch_add(1, std::memory_order_relaxed);
    RunAhead::holds.wait_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count()),
        std::memory_order_relaxed);
}

void GpuSplit::CompleteFencedEntry(u64 tag) {
    std::scoped_lock lock{fenced_entries_mutex};
    const auto it =
        std::lower_bound(fenced_entries.begin(), fenced_entries.end(), tag,
                         [](const auto& pending, u64 value) { return pending.tag < value; });
    // Stop clears the queue; a callback retained by an earlier start cannot land a new record.
    if (it == fenced_entries.end() || it->tag != tag) {
        return;
    }
    it->complete = true;
    SemRecorder::Add({.kind = "fence-entry-complete", .entry = tag});
    PublishFencedEntries();
}

void GpuSplit::SealFencedEntries(u64 position) {
    std::scoped_lock lock{fenced_entries_mutex};
    fenced_sealed = EntryTag(position);
    PublishFencedEntries();
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const CountedWait counted{DrainSite::RingFull};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            Nx2Profile::AddDrain(DrainSite::RingFull, start);
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::NoteFirstUnpublished() {
    first_unpublished_ticks = Nx2Profile::Ticks();
}

void GpuSplit::NoteEarlyPublish() {
    Nx2Profile::Add(Nx2Profile::Counter::EarlyPublishes, 1);
}

void GpuSplit::NoteLatePublish() {
    const u64 idle_since = back_idle_since.load(std::memory_order_relaxed);
    if (idle_since == 0 || first_unpublished_ticks == 0) {
        return;
    }
    // The back could have run the words since it ran out of work or since the first was sent.
    const u64 since = std::max(idle_since, first_unpublished_ticks);
    const u64 now = Nx2Profile::Ticks();
    if (now > since) {
        Nx2Profile::Add(Nx2Profile::Counter::LatePublishTicks, now - since);
        Nx2Profile::Add(Nx2Profile::Counter::LatePublishes, 1);
    }
}

void GpuSplit::Publish() {
    FlushState();
    if (published_local == write_position) {
        return;
    }
    if (profile_publishes) {
        NoteLatePublish(); // TEST
    }
    if (SyncTrace::enabled) {
        SyncTrace::Publish(published_local, write_position);
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        WakeBack();
    }
}

void GpuSplit::Drain(DrainSite site) {
    FlushState();
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position && BackQuiet()) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const BackIdleTrace::FrontWait idle_trace{WaitState(site)};
    const CountedWait counted{site};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    // Everything consumed is not enough: the back submits deferred work (Scheduler::DeferFlush)
    // from its idle hook afterwards, and the caller may use the scheduler once this returns.
    bool hook_asked = false;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            if (BackQuiet()) {
                publish_every = PublishAfterWait;
                Nx2Profile::AddDrain(site, start);
                return;
            }
            if (!hook_asked) {
                RequestIdleHook();
                hook_asked = true;
            }
            // Backoff sleeps only while entries are unconsumed: spin until the hook is done.
            if (spins < 16384) {
                Pause();
            } else {
                std::this_thread::yield();
            }
            continue;
        }
        Backoff(spins, write_position);
    }
}

bool GpuSplit::WaitForWriters() {
    return WaitFor(std::min(last_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitForSync() {
    return WaitFor(std::min(last_sync_position, write_position), DrainSite::ListSync);
}

void GpuSplit::NoteSemaphoreWrite(u64 device_address, u32 payload, bool known, Landing landing,
                                  bool at_least, SemaphoreWriter writer, bool window_base) {
    bool windowed = false;
    if (landing == Landing::AtRun && (landing_window_open || landing_close_pending)) [[unlikely]] {
        // EDEN_NX2_RUNAHEAD_LANDINGS: the back may make it a fence operation (OpenLandingWindow).
        landing = Landing::Fenced;
        windowed = true;
        RunAhead::counters.notes_fenced += RunAhead::stats ? 1 : 0;
    }
    if (semaphore_writes.size() >= semaphore_writes_limit) {
        // The words of releases in memory hold their values (the back ran them, and fenced ones
        // landed).
        boost::unordered::erase_if(semaphore_writes,
                                   [this](const auto& entry) { return InMemory(entry.second); });
        semaphore_writes_limit = std::max<size_t>(1024, semaphore_writes.size() * 2);
    }
    // The release is in the last entry sent (a copy engine launch in a long multi-method run is
    // in an earlier piece: its fence then counts it in memory later, never too early).
    SemaphoreWrite write{
        .position = write_position,
        .entry = EntryTag(last_entry_start),
        .payload = payload,
        .known = known,
        .at_least = at_least,
        .fenced = landing == Landing::Fenced,
        .writer = writer,
        .settle = 0,
        .windowed = windowed,
        .raced = false,
        .ordered = false,
        .window_base = window_base,
    };
    const u64 word = device_address & ~u64{3};
    const auto previous = semaphore_writes.find(word);
    // A fenced write sent before that has not landed (one landing at run that such a write made
    // fenced as well).
    const bool earlier_pending =
        previous != semaphore_writes.end() && previous->second.fenced &&
        fenced_landed.load(std::memory_order_acquire) <= previous->second.entry;
    if (!write.fenced && earlier_pending) {
        // A fenced write sent before may land after this one: until it did, the word holds
        // either.
        write.entry = previous->second.entry;
        write.known = false;
        write.fenced = true;
        write.raced = true;
    }
    // EDEN_NX2_RUNAHEAD_WINDOW_BASE: it lands after every write sent to the word before it, for
    // certain, when it lands as the back runs it (nothing fenced before it pending: else it is
    // raced), when it is fence-flagged (fences run in order), when a window is open (the back makes
    // it a fence operation), or, with only a close pending (the back may land it at run), when no
    // fenced write sent before is pending.
    write.ordered =
        !write.raced && (!write.fenced || !windowed || landing_window_open || !earlier_pending);
    if (SyncSuspend::enabled) [[unlikely]] {
        write.settle = SettleBefore(word, write, landing == Landing::AtRun);
    }
    if (RunAhead::enabled) [[unlikely]] {
        NoteRunAheadWord(word);
    }
    semaphore_writes.insert_or_assign(word, write);
    if (SemRecorder::Relevant(word)) {
        SemRecorder::Add({.kind = "note",
                          .address = word,
                          .value = payload,
                          .after = write.payload,
                          .position = write.position,
                          .entry = write.entry,
                          .consumed = consumed.load(std::memory_order_acquire),
                          .fence = fenced_landed.load(std::memory_order_acquire),
                          .flags = u64{known} | (u64{at_least} << 1) |
                                   (u64{landing == Landing::Fenced} << 2) |
                                   (u64{write.known} << 3) | (u64{write.fenced} << 4)});
    }
    Nx2AcquireDetails::NoteWriter(word, writer);
    using Releaser = Nx2AcquireStats::Releaser;
    const auto releaser =
        writer == SemaphoreWriter::Report3D ? Releaser::Report3D
        : writer == SemaphoreWriter::ReportCompute || writer == SemaphoreWriter::ReductionCompute
            ? Releaser::ReportCompute
        : writer == SemaphoreWriter::Copy ? Releaser::Copy
        : writer == SemaphoreWriter::QmdRelease || writer == SemaphoreWriter::QmdReduction
            ? Releaser::Qmd
            : Releaser::Puller;
    Nx2AcquireStats::NoteRelease(word, releaser);
}

SemRecorder::Event GpuSplit::SemaphoreRecordState(u64 device_address, const char* kind) const {
    const auto it = semaphore_writes.find(device_address & ~u64{3});
    const bool found = it != semaphore_writes.end();
    const SemaphoreWrite write = found ? it->second : SemaphoreWrite{};
    // The lookup's actual return is logged separately: progress can advance between these
    // diagnostic loads, so state alone must not be mistaken for the decision's snapshot.
    return {.kind = kind,
            .address = device_address,
            .value = write.payload,
            .position = write.position,
            .entry = write.entry,
            .consumed = consumed.load(std::memory_order_acquire),
            .fence = fenced_landed.load(std::memory_order_acquire),
            .flags = u64{found} | (u64{write.known} << 1) | (u64{write.at_least} << 2) |
                     (u64{write.fenced} << 3) | (u64{found && InMemory(write)} << 4)};
}

std::optional<GpuSplit::QueuedSemaphoreWrite> GpuSplit::QueuedSemaphoreWriteAt(
    u64 device_address) const {
    const auto it = semaphore_writes.find(device_address & ~u64{3});
    const bool queued = it != semaphore_writes.end() && !InMemory(it->second);
    if (SemRecorder::Relevant(device_address)) {
        auto event = SemaphoreRecordState(device_address, "lookup");
        event.flags |= u64{queued} << 5;
        SemRecorder::Add(event);
    }
    if (!queued) {
        return std::nullopt;
    }
    return QueuedSemaphoreWrite{it->second.payload, it->second.known,      it->second.at_least,
                                it->second.fenced,  it->second.writer,     it->second.position,
                                it->second.settle,  it->second.windowed,   it->second.raced,
                                it->second.ordered, it->second.window_base};
}

std::optional<u32> GpuSplit::WindowBase(u64 device, u64 target, bool gpu_written) const {
    using Refusal = RunAhead::BaseRefusal;
    const auto refuse = [](Refusal why) -> std::optional<u32> {
        RunAhead::counters.base_refusals[static_cast<size_t>(why)] += RunAhead::stats ? 1 : 0;
        return std::nullopt;
    };
    const auto queued = QueuedSemaphoreWriteAt(device);
    if (!queued) {
        return std::nullopt; // In memory as far as the front knows: memory decides
    }
    if (!landing_window_open) {
        return refuse(Refusal::Closing);
    }
    if (!queued->known || queued->at_least) {
        return refuse(Refusal::PreviousUnknown);
    }
    if (!queued->ordered) {
        return refuse(Refusal::PreviousRacing);
    }
    // The last write sent over the word: that release's own (placed when it was noted), unless
    // something was written there since. Both answers bound it from above.
    u64 written = target;
    if (const auto logged = LoggedWritesTarget(device & ~u64{3}, sizeof(u32))) {
        written = (std::min)(written, (std::max)({*logged, last_unplaced_write, run_ahead_lost}));
    }
    if (written > queued->position) {
        return refuse(Refusal::WrittenAfter);
    }
    // Their releases are noted nowhere: they may write the word, and land between the two.
    if ((std::max)(last_unknown_writer, run_ahead_waiting) > queued->position) {
        return refuse(Refusal::UnvaluedAfter);
    }
    if (gpu_written) {
        return refuse(Refusal::GpuWritten); // Shader stores may reach it through the caches
    }
    return queued->payload;
}

bool GpuSplit::MemoryHoldsBase(u64 device, u64 target, bool gpu_written) const {
    if (gpu_written || !Ran(target)) {
        return false;
    }
    if (LandingPendingAt(device)) [[unlikely]] {
        // It lands there before the reduction: memory does not hold what that combines with yet.
        using Refusal = RunAhead::BaseRefusal;
        RunAhead::counters.base_refusals[static_cast<size_t>(Refusal::LandingPending)] +=
            RunAhead::stats ? 1 : 0;
        return false;
    }
    return true;
}

u64 GpuSplit::SettleBefore(u64 word, const SemaphoreWrite& write, bool lands_at_run) const {
    // Compute and copy work sent before may release the word without the front's knowing (QMD
    // releases), and a write the front could not place may hit it (the measurement mode leaves
    // out the former).
    u64 settle = SyncSuspend::relaxed ? last_unplaced_write
                                      : (std::max)(last_writer_position, last_unplaced_write);
    if (lands_at_run) {
        // The word holds it once the back ran it, also when it only counts as fenced because an
        // earlier fenced write may still land after it.
        settle = (std::max)(settle, write.position);
    }
    if (const auto it = semaphore_writes.find(word);
        it != semaphore_writes.end() && !InMemory(it->second)) {
        // The write it replaces: one landing at run changes the word once the back runs it, a
        // fenced one once its fence ran (whenever that is), after what it followed.
        const SemaphoreWrite& previous = it->second;
        settle = (std::max)(settle, previous.settle);
        if (!previous.fenced) {
            settle = (std::max)(settle, previous.position);
        }
    }
    return (std::min)(settle, write_position);
}

std::optional<GpuSplit::QueuedSemaphoreWrite> GpuSplit::SemaphoreWriteForStats(
    u64 device_address) const {
    const auto it = semaphore_writes.find(device_address & ~u64{3});
    if (it == semaphore_writes.end()) {
        return std::nullopt;
    }
    const auto& write = it->second;
    return QueuedSemaphoreWrite{write.payload, write.known,    write.at_least,   write.fenced,
                                write.writer,  write.position, write.settle,     write.windowed,
                                write.raced,   write.ordered,  write.window_base};
}

namespace {

/// Calls `func(device_address, size)` for each mapped piece of [address, address + size) within
/// one 4 KiB GPU page; nothing is read or written where no device memory is mapped.
template <typename Func>
void ForEachDeviceRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                        Func&& func) {
    constexpr GPUVAddr page_size = 0x1000;
    const GPUVAddr end = address + size;
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = std::min((begin & ~(page_size - 1)) + page_size, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            func(*device, page_end - begin);
        }
        begin = page_end;
    }
}

} // Anonymous namespace

void GpuSplit::NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    NoteWriteAt(memory, address, size, write_position);
}

void GpuSplit::NoteWriteAt(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                           u64 position) {
    if (size == 0) {
        return;
    }
    if (size > MaxCoarseNotedBytes || address + size < address) {
        NoteUnplacedWrite();
        TagUnplaced(UnplacedKind::Oversized); // TEST
        return;
    }
    if (size > MaxNotedBytes) {
        ForEachDeviceRange(memory, address, size, [this, position](DAddr device, u64 bytes) {
            NoteCoarseDeviceWriteAt(device, bytes, position);
        });
        return;
    }
    ForEachDeviceRange(memory, address, size, [this, position](DAddr device, u64 bytes) {
        NoteDeviceWriteAt(device, bytes, position);
    });
}

void GpuSplit::NoteSetWrite(WriteSet& set, u64 granule, u64 position) {
    const u32 tag = static_cast<u32>(granule);
    size_t oldest = 0;
    for (size_t way = 0; way < WriteWays; ++way) {
        if (set.granules[way] == tag) {
            set.positions[way] = std::max(set.positions[way], position);
            return;
        }
        if (set.positions[way] < set.positions[oldest]) {
            oldest = way;
        }
    }
    // The granule's earlier writes, if any, are among the evicted ones.
    const u64 granule_before = set.evicted;
    set.evicted = std::max(set.evicted, set.positions[oldest]);
    set.granules[oldest] = tag;
    set.positions[oldest] = std::max(position, granule_before);
}

u64 GpuSplit::SetTarget(const WriteSet& set, u64 granule) {
    const u32 tag = static_cast<u32>(granule);
    for (size_t way = 0; way < WriteWays; ++way) {
        if (set.granules[way] == tag) {
            return set.positions[way];
        }
    }
    return set.evicted;
}

void GpuSplit::NoteDeviceWriteAt(DAddr address, u64 size, u64 position) {
    const u64 last = (address + size - 1) >> WriteGranuleBits;
    for (u64 granule = address >> WriteGranuleBits; granule <= last; ++granule) {
        NoteSetWrite(write_sets[WriteSlot(granule)], granule, position);
    }
    last_noted_write = std::max(last_noted_write, position);
    if (RunAhead::enabled) [[unlikely]] {
        LogRunAheadWrite(address, size, position);
    }
}

void GpuSplit::NoteCoarseDeviceWriteAt(DAddr address, u64 size, u64 position) {
    const u64 last = (address + size - 1) >> CoarseGranuleBits;
    for (u64 granule = address >> CoarseGranuleBits; granule <= last; ++granule) {
        NoteSetWrite(coarse_sets[CoarseSlot(granule)], granule, position);
    }
    last_noted_write = std::max(last_noted_write, position);
    if (RunAhead::enabled) [[unlikely]] {
        LogRunAheadWrite(address, size, position);
    }
}

const GpuSplit::WatchedPage* GpuSplit::FindWatchedPage(u64 page) const {
    // Linear probing; pages are never removed during a start (Stop clears them all).
    for (size_t i = 0, slot = WatchedSlot(page); i < WatchedPageSlots;
         ++i, slot = (slot + 1) % WatchedPageSlots) {
        const WatchedPage& watched = watched_pages[slot];
        if (watched.key == page + 1) {
            return &watched;
        }
        if (watched.key == 0) {
            return nullptr;
        }
    }
    return nullptr;
}

const GpuSplit::WatchedPage* GpuSplit::WatchPage(u64 page) const {
    if (const WatchedPage* const watched = FindWatchedPage(page)) {
        return watched;
    }
    if (watched_count >= MaxWatchedPages) {
        return nullptr;
    }
    for (size_t i = 0, slot = WatchedSlot(page); i < WatchedPageSlots;
         ++i, slot = (slot + 1) % WatchedPageSlots) {
        WatchedPage& watched = watched_pages[slot];
        if (watched.key == 0) {
            // Writes to the page sent before now were not logged: until the back ran them all,
            // any may have written the word.
            watched = {page + 1, write_position};
            ++watched_count;
            return &watched;
        }
    }
    return nullptr;
}

void GpuSplit::LogRunAheadWrite(DAddr address, u64 size, u64 position) {
    if (size == 0 || !run_ahead_log) {
        return;
    }
    // Pieces of a write never cross a 4 KiB page (ForEachDeviceRange); check both ends anyway.
    const u64 first_page = address >> 12;
    const u64 last_page = (address + size - 1) >> 12;
    if (FindWatchedPage(first_page) == nullptr &&
        (last_page == first_page || FindWatchedPage(last_page) == nullptr)) {
        return;
    }
    if (size > ~u32{0}) {
        run_ahead_lost = (std::max)(run_ahead_lost, position);
        return;
    }
    RunAheadWrite& entry = run_ahead_log[run_ahead_count % RunAheadLogSize];
    if (run_ahead_count >= RunAheadLogSize && entry.appended > consumed_cache) {
        // The back may not have run the write this slot held: it is no longer looked at.
        run_ahead_lost = (std::max)(run_ahead_lost, entry.position);
    }
    entry = {address, position, write_position, static_cast<u32>(size), 0};
    ++run_ahead_count;
}

void GpuSplit::NoteRunAheadWord(u64 word) {
    if (!run_ahead_log) {
        return;
    }
    // The release's own write was placed (logged) just before, at the same position.
    const u64 newest = (std::min<u64>)(run_ahead_count, 4);
    for (u64 i = 1; i <= newest; ++i) {
        RunAheadWrite& entry = run_ahead_log[(run_ahead_count - i) % RunAheadLogSize];
        if (entry.appended != write_position) {
            return;
        }
        const u64 offset = word - entry.begin;
        if (word >= entry.begin && offset + sizeof(u32) <= entry.size && offset % 4 == 0 &&
            offset / 4 < 32) {
            entry.noted |= u32{1} << (offset / 4);
            return;
        }
    }
}

RunAhead::Answer GpuSplit::RunAheadReach(u64 word, u64 writers_before) {
    using RunAhead::Reach;
    const u64 base = (std::min)(writers_before, write_position);
    if (Tegra::Nx2MayBeGpuWritten(word, sizeof(u32))) {
        // Dispatches' shader stores may reach guest memory here (through the caches): each
        // compute and copy entry may change the word, as the baseline counts.
        return {base, Reach::GpuWritten};
    }
    const WatchedPage* const page = run_ahead_log ? WatchPage(word >> 12) : nullptr;
    if (page == nullptr) {
        return {base, Reach::Lost};
    }
    RunAhead::Answer answer{0, Reach::None};
    const auto consider = [&answer](u64 position, Reach reason) {
        if (position > answer.position) {
            answer = {position, reason};
        }
    };
    consider(page->since, Reach::NewPage);
    consider((std::max)(last_unknown_writer, run_ahead_waiting), Reach::Unvalued);
    consider(last_unplaced_write, Reach::Unplaced);
    consider(run_ahead_lost, Reach::Lost);
    // Newest first, down to the writes logged before the back's position: everything older was
    // logged at a position the back has run, and none takes effect later than it was logged.
    const u64 done = consumed.load(std::memory_order_acquire);
    const u64 logged = (std::min<u64>)(run_ahead_count, RunAheadLogSize);
    u64 scanned = 0;
    for (u64 i = 1; i <= logged; ++i) {
        const RunAheadWrite& entry = run_ahead_log[(run_ahead_count - i) % RunAheadLogSize];
        if (entry.appended <= done) {
            break;
        }
        ++scanned;
        if (entry.position <= answer.position || entry.position <= done ||
            entry.begin >= word + sizeof(u32) || word >= entry.begin + entry.size) {
            continue;
        }
        const u64 offset = word - entry.begin;
        const bool noted = word >= entry.begin && offset % 4 == 0 && offset / 4 < 32 &&
                           ((entry.noted >> (offset / 4)) & 1) != 0;
        if (!noted) {
            consider(entry.position, Reach::Placed);
        }
    }
    if (RunAhead::stats) {
        RunAhead::counters.scanned += scanned;
    }
    if (answer.position <= done) {
        answer = {0, Reach::None};
    }
    if (answer.position >= base) {
        // Not before what the baseline waits for: its wait, counted by the cause found.
        answer.position = base;
    }
    return answer;
}

u64 GpuSplit::WritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) const {
    if (size > MaxNotedBytes || address + size < address) {
        return std::max(last_any_write, last_noted_write);
    }
    u64 target = last_any_write;
    ForEachDeviceRange(memory, address, size, [this, &target](DAddr device, u64 bytes) {
        target = std::max(target, DeviceWritesTarget(device, bytes));
    });
    return target;
}

u64 GpuSplit::MemoryWritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                 u64 size) const {
    u64 target = last_unplaced_write;
    if (size > MaxNotedBytes || address + size < address) {
        target = std::max(target, last_noted_write);
    } else {
        ForEachDeviceRange(memory, address, size, [this, &target](DAddr device, u64 bytes) {
            target = std::max(target, DeviceWritesTarget(device, bytes));
        });
    }
    return std::min(target, write_position);
}

u64 GpuSplit::ExactWritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                u64 size) const {
    const u64 granules = MemoryWritesTarget(memory, address, size);
    if (!run_ahead_log || size == 0 || size > MaxNotedBytes || address + size < address) {
        return granules;
    }
    // The writes the front could not place, and those the log dropped, may be anywhere.
    u64 target = (std::max)(last_unplaced_write, run_ahead_lost);
    bool known = true;
    ForEachDeviceRange(memory, address, size, [this, &target, &known](DAddr device, u64 bytes) {
        if (const auto piece = LoggedWritesTarget(device, bytes)) {
            target = (std::max)(target, *piece);
        } else {
            known = false;
        }
    });
    // Both bound the last write sent there from above: the smaller one is as safe.
    return known ? (std::min)(target, granules) : granules;
}

std::optional<u64> GpuSplit::LoggedWritesTarget(DAddr address, u64 size) const {
    if (!run_ahead_log || size == 0) {
        return std::nullopt;
    }
    u64 target = 0;
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        const WatchedPage* const watched = WatchPage(page);
        if (watched == nullptr) {
            return std::nullopt;
        }
        // Writes to it sent before it was watched were not logged.
        target = (std::max)(target, watched->since);
    }
    // Newest first: the last write sent over the range, whether the back ran it or not. The QMD
    // view orders it against the signal it values (a write sent after a signal valued late counts
    // even once it ran: memory holds it, the back read what was before). No write logged at or
    // before the target can place one later (a position never passes its `appended`).
    constexpr u64 MaxScan = 1024;
    const u64 logged = (std::min<u64>)(run_ahead_count, RunAheadLogSize);
    u64 scanned = 0;
    for (u64 i = 1; i <= logged; ++i) {
        const RunAheadWrite& entry = run_ahead_log[(run_ahead_count - i) % RunAheadLogSize];
        if (entry.appended <= target) {
            break;
        }
        if (++scanned > MaxScan) {
            return std::nullopt; // The granules decide
        }
        if (entry.position > target && entry.begin < address + size &&
            address < entry.begin + entry.size) {
            target = entry.position;
        }
    }
    if (RunAhead::stats) {
        RunAhead::counters.scanned += scanned;
    }
    return target;
}

u64 GpuSplit::DeviceWritesTarget(DAddr address, u64 size) const {
    u64 target = 0;
    const u64 last = (address + size - 1) >> WriteGranuleBits;
    for (u64 granule = address >> WriteGranuleBits; granule <= last; ++granule) {
        target = std::max(target, SetTarget(write_sets[WriteSlot(granule)], granule));
    }
    const u64 last_coarse = (address + size - 1) >> CoarseGranuleBits;
    for (u64 granule = address >> CoarseGranuleBits; granule <= last_coarse; ++granule) {
        target = std::max(target, SetTarget(coarse_sets[CoarseSlot(granule)], granule));
    }
    return target;
}

bool GpuSplit::WaitForWrites(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    FlushState(); // A block that may issue a deferred draw notes its write anywhere.
    return WaitFor(std::min(WritesTarget(memory, address, size), write_position),
                   DrainSite::MacroDmaWrites);
}

bool GpuSplit::WaitForSyncedWriters(DrainSite site) {
    return WaitFor(std::min(synced_writer_position, write_position), site);
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    FlushState();
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    const BackIdleTrace::FrontWait idle_trace{WaitState(site)};
    const CountedWait counted{site};
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            // The back may run out of what was sent before long.
            publish_every = PublishAfterWait;
            Nx2Profile::AddDrain(site, start);
            return true;
        }
        Backoff(spins, target);
    }
}

bool GpuSplit::WaitForPositionFor(u64 position, std::chrono::nanoseconds limit) {
    const u64 target = (std::min)(position, write_position);
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return true;
    }
    const BackIdleTrace::FrontWait idle_trace{BackIdleTrace::FrontState::Drain};
    // As WaitFor: the back stores its position as soon as it is past the target.
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const auto deadline = std::chrono::steady_clock::now() + limit;
    bool reached = false;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            reached = true;
            break;
        }
        if (spins < 16384) {
            Pause();
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline || !back_running.load(std::memory_order_acquire)) {
            break;
        }
        // The back stores `consumed` before it reads front_event_waiting (NotifyFront), the front
        // sets that before it reads `consumed`: one sees the other, and the event stays set.
        front_event_waiting.store(true, std::memory_order_seq_cst);
        if (consumed.load(std::memory_order_seq_cst) < target) {
            front_event.SleepFor(deadline - now);
        }
        front_event_waiting.store(false, std::memory_order_seq_cst);
    }
    wait_target.store(~u64{0}, std::memory_order_relaxed);
    if (reached) {
        // The back may run out of what was sent before long.
        publish_every = PublishAfterWait;
    }
    return reached;
}

void GpuSplit::Backoff(u32 spins, u64 target) {
    if (spins < 16384) {
        Pause();
        return;
    }
    if (!back_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        return;
    }
    if (FastWake::enabled) {
        front_waiting.store(true, std::memory_order_seq_cst);
        const u64 seen = position_signal.Snapshot();
        if (consumed.load(std::memory_order_seq_cst) < target) {
            FastWake::counters.position_waits.fetch_add(1, std::memory_order_relaxed);
            if (position_signal.Wait(seen, std::chrono::milliseconds{1}) ==
                FastWake::Signal::Result::Timeout) {
                FastWake::counters.position_timeouts.fetch_add(1, std::memory_order_relaxed);
            }
        }
        front_waiting.store(false, std::memory_order_seq_cst);
        return;
    }
    // A long wait sleeps until the back stores a new position (port of #161's front_wake): the
    // back checks front_waiting after each store, the front checks the position after setting it.
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) < target) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::FlushStateSlow() {
    // Cleared first: the entry's own SelectEngine, Reserve and Commit flush again.
    const u32 count = std::exchange(state_count, 0);
    SelectEngine(state_engine);
    const size_t size = 1 + StateFlagWords + count;
    u64* const entry = Reserve(size);
    entry[0] = static_cast<u64>(Kind::State) | (state_breaks ? HasBreak : 0) |
               (u64{state_break & 0xffff} << 8) | (u64{count} << 32);
    std::memcpy(entry + 1, state_flags.data(), sizeof(state_flags));
    std::memcpy(entry + 1 + StateFlagWords, state_writes.data(), size_t{count} * sizeof(u64));
    Commit(size);
    if (state_anywhere) {
        // A deferred draw the block may issue writes guest memory through the GPU caches only,
        // as FrontNoteWrites notes a draw: macro DMA reads wait for it (WritesTarget), reads that
        // bypass the caches and the mappings don't.
        NoteCacheWrite();
    }
    state_breaks = false;
    state_anywhere = false;
    state_flags.fill(0);
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment,
                           Writes writes) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
    if (log_methods) {
        LogMethod(engine, method); // TEST
    }
    NoteWrites(writes);
}

void GpuSplit::LogMethod(const Tegra::Engines::EngineInterface* engine, u32 method) {
    if (!logged_methods) {
        logged_methods = std::make_unique<SentMethod[]>(LoggedMethods);
    }
    logged_methods[logged_method_count++ & (LoggedMethods - 1)] = {write_position,
                                                                   engine->bound_type, method};
}

std::string GpuSplit::MethodEndingAt(u64 end) const {
    if (!logged_methods || logged_method_count == 0) {
        return {};
    }
    // Ends grow with the count: a binary search over the ones kept.
    u64 low = logged_method_count > LoggedMethods ? logged_method_count - LoggedMethods : 0;
    u64 high = logged_method_count;
    while (low < high) {
        const u64 middle = low + (high - low) / 2;
        if (logged_methods[middle & (LoggedMethods - 1)].end < end) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low == logged_method_count) {
        return {};
    }
    const SentMethod& sent = logged_methods[low & (LoggedMethods - 1)];
    if (sent.end != end) {
        return {};
    }
    // Not typeid: without RTTI (MSVC's /GR-) it reads type data the engines don't have.
    return fmt::format("{} {:#x}", EngineName(sent.engine), sent.method);
}

void GpuSplit::DeferFunction(std::function<void()>&& func, Writes writes) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer(
        [heap] {
            (*heap)();
            delete heap;
        },
        0, writes);
}

size_t GpuSplit::Execute(const u64* entry, size_t words) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if (engine->fast_maxwell3d) {
            // Nx2Perf: as below without the profile zones (off with fast_maxwell3d) and the
            // virtual call.
            using Tegra::Engines::Maxwell3D;
            auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
            constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
            if (method - cb_data < 16 && (head & HasSegment) == 0) {
                // Single constant buffer words in a row (macros write them one by one) go up in
                // one write. They are executable: each entry would only consume the sink first.
                std::array<u32, 64> methods;
                std::array<u32, 64> arguments;
                methods[0] = method;
                arguments[0] = argument;
                u32 count = 1;
                while (count < methods.size() && count < words) {
                    const u64 next = entry[count];
                    const u32 next_method = static_cast<u32>((next >> 8) & 0xffff);
                    if (static_cast<Kind>(next & 0xf) != Kind::Method || (next & HasSegment) != 0 ||
                        next_method - cb_data >= 16) {
                        break;
                    }
                    methods[count] = next_method;
                    arguments[count] = static_cast<u32>(next >> 32);
                    ++count;
                }
                maxwell3d->ConsumeSink(*system);
                if (count == 1) {
                    maxwell3d->CallMethod(*system, method, argument, true);
                } else {
                    maxwell3d->CallConstBufferWords(*system, methods.data(), arguments.data(),
                                                    count);
                }
                return count;
            }
            if ((head & Direct) != 0) {
                maxwell3d->ConsumeSink(*system);
                maxwell3d->CallMethod(*system, method, argument, true);
            } else if (!maxwell3d->execution_mask[method]) {
                maxwell3d->method_sink.emplace_back(method, argument);
            } else {
                maxwell3d->ConsumeSink(*system);
                if ((head & HasSegment) != 0) {
                    maxwell3d->current_dma_segment = entry[1];
                }
                maxwell3d->CallMethod(*system, method, argument, (head & IsLast) != 0);
            }
            return (head & HasSegment) != 0 ? 2 : 1;
        }
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        if ((head & Direct) != 0) {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::State: {
        const u32 count = static_cast<u32>(head >> 32);
        const u32 last_break = (head & HasBreak) != 0 ? static_cast<u32>((head >> 8) & 0xffff)
                                                      : Tegra::Engines::Maxwell3D::NoBreak;
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->WriteStateRegisters(*system,
                                  std::span<const u64, StateFlagWords>{entry + 1, StateFlagWords},
                                  std::span<const u64>{entry + 1 + StateFlagWords, count},
                                  last_break);
        return 1 + StateFlagWords + count;
    }
    case Kind::Call: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
            if (Nx2CallSites::g_enabled) [[unlikely]] {
                Nx2CallSites::NoteBreak(entry[1]); // EDEN_NX2_DRAWX_CALL_SITES
            }
        } else if (Nx2CallSites::g_enabled) [[unlikely]] {
            Nx2CallSites::NoteNeutral(entry[1]);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return 2 + payload;
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    const StutterTrace::ThreadScope stutter_thread{StutterTrace::Thread::Back};
    Common::SetCurrentThreadName("GPUBack");
    const VideoCommon::WaitStats::ThreadScope wait_thread{VideoCommon::WaitStats::Thread::Back};
    on_back_thread = true;
    tick_state.store(tick_state.load(std::memory_order_relaxed) + TickGeneration + TickAccepting,
                     std::memory_order_release);
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (Settings::values.pin_gpu_threads.GetValue()) {
        Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    }
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    // EDEN_NX2_SYNC_SUSPEND: when the current idle span began (none while the back has work).
    std::optional<std::chrono::steady_clock::time_point> idle_since;
    // EDEN_NX2_BACK_IDLE_TRACE: the same in the trace's clock (0 while the back has work).
    const bool idle_trace = BackIdleTrace::Enabled();
    u64 trace_idle_since = 0;
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread", true);
        MaybeTick();
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            if (SyncTrace::enabled) {
                SyncTrace::IdleBegin();
            }
            if (SyncSuspend::enabled && !idle_since) [[unlikely]] {
                idle_since = std::chrono::steady_clock::now();
            }
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            if (idle_trace && trace_idle_since == 0) {
                trace_idle_since = BackIdleTrace::Now();
            }
            // The front publishes what it holds after a few more words from now on.
            back_idle_at.store(read_position, std::memory_order_relaxed);
            if (profile_publishes) {
                back_idle_since.store(Nx2Profile::Ticks(), std::memory_order_relaxed);
            }
            const auto spin_start = g_acquire_counting ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
            // A bounded sleep can time out with no new work. Do not repeat the spin budget
            // after the idle hook has already made the back quiet.
            const u32 spin_limit =
                FastWake::enabled && back_quiet.load(std::memory_order_relaxed) ? 0U : 8192U;
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < spin_limit && available == read_position &&
                                !hook_requested.load(std::memory_order_relaxed) &&
                                (tick_state.load(std::memory_order_relaxed) & TickRequested) == 0;
                 ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (g_acquire_counting) {
                Count(g_acquire_stats.back_waits);
                Count(g_acquire_stats.back_spin_ns,
                      static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - spin_start)
                                           .count()));
            }
            if (available == read_position) {
                hook_requested.store(false, std::memory_order_seq_cst);
                // The front may already see everything consumed: Drain() waits for back_quiet
                // too, so the deferred submit can't overlap whatever the front does next.
                if (!back_quiet.load(std::memory_order_relaxed)) {
                    if (idle_hook != nullptr) {
                        if (SyncTrace::enabled) {
                            SyncTrace::IdleFlag(false);
                        }
                        idle_hook(idle_argument);
                    }
                    back_quiet.store(true, std::memory_order_release);
                }
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                const u64 fast_seen = FastWake::enabled ? back_signal.Snapshot() : 0;
                available = published.load(std::memory_order_seq_cst);
                // Idle-hook and tick requests made since the hooks ran must prevent sleeping.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst) &&
                    (tick_state.load(std::memory_order_seq_cst) & TickRequested) == 0) {
                    Count(g_acquire_stats.back_sleeps);
                    if (SyncTrace::enabled) {
                        SyncTrace::IdleFlag(true);
                    }
                    if (FastWake::enabled) {
                        FastWake::counters.back_waits.fetch_add(1, std::memory_order_relaxed);
                        if (back_signal.Wait(fast_seen, std::chrono::milliseconds{1}) ==
                            FastWake::Signal::Result::Timeout) {
                            FastWake::counters.back_timeouts.fetch_add(1,
                                                                       std::memory_order_relaxed);
                        }
                    } else {
                        wake.wait(seen, std::memory_order_seq_cst);
                    }
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        if (SyncTrace::enabled) {
            SyncTrace::IdleEnd();
        }
        if (idle_since) [[unlikely]] {
            SyncSuspend::back.idle_spans.fetch_add(1, std::memory_order_relaxed);
            SyncSuspend::back.idle_ns.fetch_add(
                SyncSuspend::Nanoseconds(std::chrono::steady_clock::now() - *idle_since),
                std::memory_order_relaxed);
            idle_since.reset();
        }
        if (trace_idle_since != 0) {
            BackIdleTrace::NoteBackIdle(trace_idle_since, BackIdleTrace::Now());
            trace_idle_since = 0;
        }
        if (profile_publishes) {
            back_idle_since.store(0, std::memory_order_relaxed);
        }
        u64 last_store = read_position;
        // Before any entry runs, so a front that sees these entries consumed also sees this.
        back_quiet.store(false, std::memory_order_seq_cst);
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            back_entry_start = read_position;
            if (SyncTrace::enabled) {
                SyncTrace::BackEntry(read_position);
            }
            const size_t size =
                Execute(&ring[index], std::min<u64>(available - read_position, RingWords - index));
            read_position += size != 0 ? size : RingWords - index;
            if (SyncTrace::enabled) {
                SyncTrace::BackConsumed(read_position);
            }
            // Keep registration open through Execute: an entry can send several reports, and
            // the first fence can finish before the back has even sent the last one.
            back_fenced_entry.reset();
            if (landing_closing) [[unlikely]] {
                MaybeEndLandings(read_position);
            }
            if (hook_requested.load(std::memory_order_relaxed) &&
                idle_hook != nullptr) [[unlikely]] {
                hook_requested.store(false, std::memory_order_seq_cst);
                idle_hook(idle_argument);
            }
            // Serve a guest read before the next entry, even with more work queued on the front.
            MaybeTick();
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                SealFencedEntries(read_position);
                consumed.store(read_position, std::memory_order_seq_cst);
                last_store = read_position;
                NotifyFront();
            }
        }
        SealFencedEntries(read_position);
        consumed.store(read_position, std::memory_order_seq_cst);
        NotifyFront();
        if (landing_uncommitted) [[unlikely]] {
            // EDEN_NX2_RUNAHEAD_LANDINGS: about to run dry; converted landings not attached to a
            // fence yet would wait for the next one, which nothing may make while the back sleeps.
            MakeLandingFence();
        }
    }
    if (SyncTrace::enabled) {
        SyncTrace::IdleEnd(true);
        SyncTrace::StopSplit();
    }
    // Stop accepting ticks before leaving. A reader still has its normal queued GPU tick:
    // it runs on the front after a settings stop, or #1058 releases it at final shutdown.
    tick_state.fetch_and(~(TickAccepting | TickRequested), std::memory_order_seq_cst);
    back_running.store(false, std::memory_order_seq_cst);
    if (FastWake::enabled) {
        position_signal.Notify();
    }
    if (SyncSuspend::enabled) {
        front_event.Set();
    }
    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

void GpuSplit::ReportAcquire() {
    static u32 calls{};
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    if (++calls % 256 != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds{10};
    auto& stats = g_acquire_stats;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 passed = take(stats.passed);
    const u64 by_queued = take(stats.by_queued);
    const u64 after_wait = take(stats.after_wait);
    const u64 latched = take(stats.latched);
    const u64 suspended = take(stats.suspended);
    const u64 positioned = take(stats.positioned);
    const u64 positioned_waited = take(stats.positioned_waited);
    const u64 flushes = take(stats.flushes);
    const u64 drained = take(stats.drained);
    const u64 drain_operation = take(stats.drain_operation);
    const u64 drain_settings = take(stats.drain_settings);
    const u64 drain_word = take(stats.drain_word);
    const u64 drain_queued = take(stats.drain_queued);
    const u64 queued_unknown = take(stats.queued_unknown);
    const u64 queued_unknown_met = take(stats.queued_unknown_met);
    const u64 trigger_not_met = take(stats.trigger_not_met);
    const u64 trigger_not_met_twice = take(stats.trigger_not_met_twice);
    const u64 valve_drains = take(stats.valve_drains);
    const u64 valve_passes = take(stats.valve_passes);
    const u64 polls = take(stats.polls);
    const u64 acquire_polls = take(stats.acquire_polls);
    const u64 list_polls = take(stats.list_polls);
    const u64 front_polls = take(stats.front_polls);
    const u64 front_sleeps = take(stats.front_sleeps);
    const u64 front_woken = take(stats.front_woken);
    const u64 front_stored = take(stats.front_stored);
    const u64 front_found_polling = take(stats.front_found_polling);
    const u64 front_found_late = take(stats.front_found_late);
    const u64 early_publishes = take(stats.early_publishes);
    const u64 settles = take(stats.settles);
    const u64 back_waits = take(stats.back_waits);
    const u64 back_sleeps = take(stats.back_sleeps);
    const double spin_ms = static_cast<double>(take(stats.back_spin_ns)) / 1e6;
    LOG_INFO(HW_GPU,
             "Acquires: {} passed without a drain ({} by a queued release, {} after a positioned "
             "wait), {} met by a latch, {} suspended, {} positioned waits ({} waited), {} word "
             "flushes, {} drained ({} by the 1 ms valve, of those {} met; {} equal or mask, {} by "
             "the settings, {} unmapped or GPU-written word, {} queued release unknown or short), "
             "{} polls waited without a drain in 10 s; polls: {} of acquires, {} of GPU-written "
             "list copies; back: {} waits for work, {} slept, {:.1f} ms spinning; {} early "
             "publishes, {} idle hooks asked by polls",
             passed, by_queued, after_wait, latched, suspended, positioned, positioned_waited,
             flushes, drained, valve_drains, valve_passes, drain_operation, drain_settings,
             drain_word, drain_queued, polls, acquire_polls, list_polls, back_waits, back_sleeps,
             spin_ms, early_publishes, settles);
    LOG_INFO(HW_GPU,
             "GPU thread waits for suspended channels in 10 s: {} in hardware, {} in the OS, {} "
             "ended by a wake, {} by a store to a semaphore word; channels that moved without "
             "either: {} found while polling, {} found late in the OS wait",
             front_polls, front_sleeps, front_woken, front_stored, front_found_polling,
             front_found_late);
    std::string sites;
    for (size_t site = 0; site < drain_site_names.size(); ++site) {
        const u64 count = take(stats.site_drains[site]);
        const u64 ns = take(stats.site_drain_ns[site]);
        if (count != 0) {
            sites += fmt::format("{}{} {} ({:.1f} ms)", sites.empty() ? "" : ", ",
                                 drain_site_names[site], count, static_cast<double>(ns) / 1e6);
        }
    }
    LOG_INFO(HW_GPU,
             "Front waits for the back in 10 s by site: {}; acquire methods not met without a "
             "drain: {} ({} still not met after it, which drain again to release the fences); "
             "drains for a queued release of unknown value (a reduction or a counter) to a word "
             "memory already meets: {} of {}",
             sites.empty() ? "none" : sites, trigger_not_met, trigger_not_met_twice,
             queued_unknown_met, queued_unknown);
    const u64 mismatches = take(stats.mismatches);
    const u64 mismatches_waited = take(stats.mismatches_waited);
    const u64 raced = take(stats.raced);
    const u64 unlanded = take(stats.unlanded);
    const u64 late = take(stats.late);
    const u64 late_polls = take(stats.late_polls);
    if (!g_acquire_verify) {
        return;
    }
    // A pass of the view the drained word contradicts is an error of the view; the rest is
    // expected: a fenced release may land during the drain, and waits end at a latch or a poll.
    if (mismatches != 0 || unlanded != 0) {
        LOG_WARNING(HW_GPU,
                    "Acquire verify: {} passed but not met after a drain ({} after a positioned "
                    "wait), {} changed by a fenced release landing during the drain, {} met by a "
                    "queued release not in memory after it; {} suspended and {} polls kept "
                    "waiting though a drain meets them, in 10 s",
                    mismatches, mismatches_waited, raced, unlanded, late, late_polls);
    } else {
        LOG_INFO(HW_GPU,
                 "Acquire verify: {} passed but not met after a drain ({} after a positioned "
                 "wait), {} changed by a fenced release landing during the drain, {} met by a "
                 "queued release not in memory after it; {} suspended and {} polls kept waiting "
                 "though a drain meets them, in 10 s",
                 mismatches, mismatches_waited, raced, unlanded, late, late_polls);
    }
}

} // namespace VideoCommon
