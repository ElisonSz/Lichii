// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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
#include "video_core/nx2_acquire_stats.h"
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
    (void)where;
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
    g_acquire_counting = g_acquire_verify || Nx2Profile::Enabled();
    profile_publishes = Nx2Profile::Enabled();
    log_methods = g_acquire_counting;
    ring = std::make_unique<u64[]>(RingWords);
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
    published_local = 0;
    publish_held = false;
    logged_method_count = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
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
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position && BackQuiet()) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
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
                                  bool at_least, SemaphoreWriter writer) {
    if (semaphore_writes.size() >= semaphore_writes_limit) {
        // The words of releases in memory hold their values.
        std::erase_if(semaphore_writes,
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
    };
    const u64 word = device_address & ~u64{3};
    if (const auto it = semaphore_writes.find(word);
        !write.fenced && it != semaphore_writes.end() && it->second.fenced &&
        fenced_landed.load(std::memory_order_acquire) <= it->second.entry) {
        // A fenced write sent before may land after this one: until it did, the word holds
        // either.
        write.entry = it->second.entry;
        write.known = false;
        write.fenced = true;
    }
    if (SyncSuspend::enabled) [[unlikely]] {
        write.settle = SettleBefore(word, write, landing == Landing::AtRun);
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
    return QueuedSemaphoreWrite{it->second.payload, it->second.known,  it->second.at_least,
                                it->second.fenced,  it->second.writer, it->second.position,
                                it->second.settle};
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
    return QueuedSemaphoreWrite{write.payload, write.known,  write.at_least, write.fenced,
                                write.writer,  write.position, write.settle};
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
}

void GpuSplit::NoteCoarseDeviceWriteAt(DAddr address, u64 size, u64 position) {
    const u64 last = (address + size - 1) >> CoarseGranuleBits;
    for (u64 granule = address >> CoarseGranuleBits; granule <= last; ++granule) {
        NoteSetWrite(coarse_sets[CoarseSlot(granule)], granule, position);
    }
    last_noted_write = std::max(last_noted_write, position);
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
    return WaitFor(std::min(WritesTarget(memory, address, size), write_position),
                   DrainSite::MacroDmaWrites);
}

bool GpuSplit::WaitForSyncedWriters(DrainSite site) {
    return WaitFor(std::min(synced_writer_position, write_position), site);
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
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
    case Kind::Call: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
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
    const u64 drain_settings = take(stats.drain_settings);
    const u64 drain_word = take(stats.drain_word);
    const u64 valve_drains = take(stats.valve_drains);
    const u64 valve_passes = take(stats.valve_passes);
    const u64 polls = take(stats.polls);
    const u64 acquire_polls = take(stats.acquire_polls);
    const u64 list_polls = take(stats.list_polls);
    const u64 early_publishes = take(stats.early_publishes);
    const u64 settles = take(stats.settles);
    const u64 back_waits = take(stats.back_waits);
    const u64 back_sleeps = take(stats.back_sleeps);
    const double spin_ms = static_cast<double>(take(stats.back_spin_ns)) / 1e6;
    LOG_INFO(HW_GPU,
             "Acquires: {} passed without a drain ({} by a queued release, {} after a positioned "
             "wait), {} met by a latch, {} suspended, {} positioned waits ({} waited), {} word "
             "flushes, {} drained ({} by the 1 ms valve, of those {} met; {} by the settings, {} "
             "unmapped word), {} polls waited without a drain in 10 s; polls: {} of acquires, {} "
             "of GPU-written list copies; back: {} waits for work, {} slept, {:.1f} ms spinning; "
             "{} early publishes, {} idle hooks asked by polls",
             passed, by_queued, after_wait, latched, suspended, positioned, positioned_waited,
             flushes, drained, valve_drains, valve_passes, drain_settings, drain_word, polls,
             acquire_polls, list_polls, back_waits, back_sleeps, spin_ms, early_publishes, settles);
    const u64 mismatches = take(stats.mismatches);
    const u64 mismatches_waited = take(stats.mismatches_waited);
    const u64 raced = take(stats.raced);
    const u64 late = take(stats.late);
    const u64 late_polls = take(stats.late_polls);
    if (!g_acquire_verify) {
        return;
    }
    // A pass of the view the drained word contradicts is an error of the view; the rest is
    // expected: a fenced release may land during the drain, and waits end at a latch or a poll.
    if (mismatches != 0) {
        LOG_WARNING(HW_GPU,
                    "Acquire verify: {} passed but not met after a drain ({} after a positioned "
                    "wait), {} changed by a fenced release landing during the drain; {} "
                    "suspended and {} polls kept waiting though a drain meets them, in 10 s",
                    mismatches, mismatches_waited, raced, late, late_polls);
    } else {
        LOG_INFO(HW_GPU,
                 "Acquire verify: {} passed but not met after a drain ({} after a positioned "
                 "wait), {} changed by a fenced release landing during the drain; {} suspended "
                 "and {} polls kept waiting though a drain meets them, in 10 s",
                 mismatches, mismatches_waited, raced, late, late_polls);
    }
}

} // namespace VideoCommon
