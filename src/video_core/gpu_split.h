// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA) the caches may still write
    MacroDmaWrites, ///< Those reads: the queued work that can write the words (WaitForWrites)
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    RingFull,       ///< The ring is full: the front waits for the back to make room
    BackRegister,   ///< A register a macro the back ran may have written (Maxwell3D)
    BackCode,       ///< The macro code changes under the macros the back has yet to run
    ListCopies,     ///< Lists on GPU-written pages: the dispatches and copies before the last sync
    AcquirePoll,    ///< A suspended acquire's poll drains once per 1 ms (Puller::PollWithoutDrain)
    Count,
};

/// How semaphore acquires and the polls of suspended ones were decided, counted with
/// EDEN_NX2_GPU_PROFILE or EDEN_NX2_ACQUIRE_VERIFY and logged every 10 s (GpuSplit::ReportAcquire).
struct AcquireStats {
    std::atomic<u64> passed{};     ///< Acquires and polls the semaphore view passed (no drain)
    std::atomic<u64> by_queued{};  ///< Of passed: met by a release the back had not run yet
    std::atomic<u64> after_wait{}; ///< Of passed: acquires, after a positioned wait
    std::atomic<u64> latched{};    ///< Acquires and polls met by a release the back ran (latch)
    std::atomic<u64> suspended{};  ///< Acquires the view suspended
    std::atomic<u64> positioned{}; ///< Positioned waits of acquires for queued work
    std::atomic<u64> positioned_waited{}; ///< Of those: the back had not run it yet
    std::atomic<u64> flushes{}; ///< Words the back flushes out of the buffer cache for the view
    std::atomic<u64> drained{}; ///< Acquires and polls that drained the split
    std::atomic<u64> drain_settings{}; ///< Of drained: the view doesn't apply (settings, macros)
    std::atomic<u64> drain_word{};     ///< ...the word is unmapped
    std::atomic<u64> valve_drains{};   ///< ...polls of a suspended acquire, once per 1 ms
    std::atomic<u64> valve_passes{};   ///< Of valve_drains: the drained word or a latch met it
    std::atomic<u64> polls{};     ///< Polls that keep waiting without a drain
    std::atomic<u64> acquire_polls{}; ///< Polls of suspended channels waiting on an acquire
    std::atomic<u64> list_polls{};    ///< ...and on a command list copy the GPU writes
    std::atomic<u64> early_publishes{}; ///< Publishes to a back that waits for work (Commit)
    std::atomic<u64> settles{};   ///< Idle hooks asked of a back that ran everything (SettleIdleBack)
    std::atomic<u64> back_waits{};      ///< The back ran out of work
    std::atomic<u64> back_sleeps{};     ///< Of those: it went to sleep
    std::atomic<u64> back_spin_ns{};    ///< How long it spun for more work before that
    std::atomic<u64> mismatches{}; ///< VERIFY: passed, yet the word after a drain doesn't meet it
    std::atomic<u64> mismatches_waited{}; ///< Of mismatches: passed after a positioned wait
    /// VERIFY: passed, the word after a drain doesn't meet it, and a fenced release to it landed
    /// during the drain: the drain path itself reads either value then.
    std::atomic<u64> raced{};
    std::atomic<u64> late{};       ///< VERIFY: an acquire suspended, yet a drain meets it
    std::atomic<u64> late_polls{}; ///< VERIFY: a poll kept waiting, yet a drain meets it
};
inline AcquireStats g_acquire_stats;
/// Whether g_acquire_stats counts (EDEN_NX2_GPU_PROFILE or EDEN_NX2_ACQUIRE_VERIFY), as of the
/// split's last start.
inline bool g_acquire_counting{};
/// EDEN_NX2_ACQUIRE_VERIFY=1: acquires and polls the semaphore view decides drain anyway and
/// compare; where the drained word disagrees, the drain's answer stands.
inline bool g_acquire_verify{};

inline void Count(std::atomic<u64>& counter, u64 amount = 1) {
    if (g_acquire_counting) {
        counter.fetch_add(amount, std::memory_order_relaxed);
    }
}

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",           "fetch drain",   "acquire",    "semaphore flush",   "fence release",
    "list writers",    "list sync",     "list flush", "macro dma read",    "macro dma writes",
    "macro dma flush", "gp sync fence", "map",        "front start",       "front verify",
    "ring full",       "back register", "back code",  "list copy writers", "acquire poll",
};

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// How often the split started: state the front copied during an earlier start is stale.
    [[nodiscard]] u32 Starts() const {
        return starts;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Back thread: the entry it runs, as NoteFencedWrite takes it; elsewhere nothing.
    [[nodiscard]] std::optional<u64> BackEntry() const {
        if (!on_back_thread) {
            return std::nullopt;
        }
        return EntryTag(back_entry_start);
    }
    /// Any thread: the operation of a report the back sent in `entry` (BackEntry) ran with its
    /// fence, so the report is in memory, and so are those of the entries before it: fences run
    /// their operations in order (QueryCacheBase::CounterReport, FenceManager::SignalFence).
    void NoteFencedWrite(u64 entry) {
        u64 landed = fenced_landed.load(std::memory_order_relaxed);
        while (landed <= entry &&
               !fenced_landed.compare_exchange_weak(landed, entry + 1, std::memory_order_release,
                                                    std::memory_order_relaxed)) {
        }
    }
    /// Marks the calling thread as the GPU thread (the front, or the only one without the split).
    static void MarkGpuThread() {
        on_gpu_thread = true;
    }
    /// Whether the calling thread owns the Vulkan scheduler: the back while it runs, else the
    /// GPU thread. Other threads (CPU cores, services) must not record or submit work.
    [[nodiscard]] static bool OwnsScheduler() {
        return Get().BackRunning() ? on_back_thread : on_gpu_thread;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// What the sender of an entry tells about the guest memory the back writes running it, for
    /// the front's reads of guest memory (WaitForWrites).
    enum class Writes : u8 {
        Anywhere, ///< Unknown: a read anywhere waits for the entry
        Noted,    ///< Nothing, or what the sender notes right after sending it (NoteWrite)
    };

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment, Writes writes = Writes::Anywhere) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
        if (writes == Writes::Anywhere) {
            NoteAnyWrite();
        }
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment,
                     Writes writes = Writes::Anywhere);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Writes writes = Writes::Anywhere) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
        if (writes == Writes::Anywhere) {
            NoteAnyWrite();
        }
    }

    /// Defer flag: the call doesn't end fast draw reuse (nx2_fast_draw.h).
    static constexpr u64 FastDrawNeutral = 1 << 7;

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// `flags`: FastDrawNeutral when `func` can't change what a draw resolves to.
    template <typename Func>
    void Defer(const Func& func, u64 flags = 0, Writes writes = Writes::Anywhere) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | flags | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload);
        if (writes == Writes::Anywhere) {
            NoteAnyWrite();
        }
    }
    /// Defer, calling `func` with a copy of `words` (a std::span<const u32>).
    template <typename Func>
    void DeferWords(const Func& func, std::span<const u32> words,
                    Writes writes = Writes::Anywhere) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t function_words = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        const size_t payload = function_words + 1 + (words.size() + 1) / 2;
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallWordsThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        entry[2 + function_words] = words.size();
        if (!words.empty()) {
            entry[1 + payload] = 0; // Padding after an odd count
            std::memcpy(entry + 3 + function_words, words.data(), words.size_bytes());
        }
        Commit(2 + payload);
        if (writes == Writes::Anywhere) {
            NoteAnyWrite();
        }
    }
    void DeferFunction(std::function<void()>&& func);
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// The front has nothing to send while it polls suspended channels. A back that ran everything
    /// runs its idle hook (the deferred submit) and sleeps now instead of spinning for more, as it
    /// did when every poll drained (Drain asks for the hook), but without the wait.
    void SettleIdleBack() {
        if (hook_requested.load(std::memory_order_relaxed) || BackQuiet() ||
            consumed.load(std::memory_order_acquire) != write_position) {
            return;
        }
        Count(g_acquire_stats.settles);
        RequestIdleHook();
    }
    /// Logs g_acquire_stats every 10 s (front, while they count).
    void ReportAcquire();
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);
    /// The ring position after the entries sent so far, for Ran and WaitForPosition.
    [[nodiscard]] u64 Position() const {
        return write_position;
    }
    /// Waits until the back executed the entries sent before `position`, not what came after.
    /// Returns whether any of them was still queued.
    bool WaitForPosition(u64 position, DrainSite site) {
        return WaitFor((std::min)(position, write_position), site);
    }
    /// Whether the back executed everything sent before `position`. It may tell so a little
    /// late: the back stores its position every few thousand words and when it runs out of work.
    [[nodiscard]] bool Ran(u64 position) const {
        return consumed.load(std::memory_order_acquire) >= position;
    }

    /// Compute or copy work was just sent: what can write GPU-written command lists, and
    /// semaphore words without the front's knowing (QMD releases).
    void NoteWriter() {
        last_writer_position = write_position;
    }
    /// The position after the last compute or copy work sent (NoteWriter).
    [[nodiscard]] u64 LastWriterPosition() const {
        return (std::min)(last_writer_position, write_position);
    }
    /// Whether compute or copy work sent so far may still be queued.
    [[nodiscard]] bool WritersPending() const {
        return !Ran(LastWriterPosition());
    }
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires note it when a release still queued meets them): the guest's GPU
    /// work before it is complete before the commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
        synced_writer_position = last_writer_position;
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();
    /// When a release the back runs reaches memory.
    enum class Landing : u8 {
        AtRun,  ///< As the back runs it: payload reports without a fence flag, reductions, maxima
        Fenced, ///< Once the fence signaled after it ran (QueryCacheBase::CounterReport):
                ///< fence-flagged reports, counter reports
    };
    /// A semaphore release was just sent: once it landed (`landing`), the word at
    /// `device_address` holds `payload`, or a value the front can't tell (`known` false: a
    /// reduction, a counter report).
    void NoteSemaphoreWrite(u64 device_address, u32 payload, bool known,
                            Landing landing = Landing::AtRun);
    struct QueuedSemaphoreWrite {
        u32 payload;
        bool known;
        /// It lands once a fence ran, at no position the front can wait for; or an earlier write
        /// that does may still land after it.
        bool fenced;
        u64 position; ///< Once the back ran this (Ran), the word holds the write, unless `fenced`
    };
    /// The last write sent to the semaphore word at `device_address`, if it may not be in memory
    /// yet.
    [[nodiscard]] std::optional<QueuedSemaphoreWrite> QueuedSemaphoreWriteAt(
        u64 device_address) const;

    /// What the front can tell of a semaphore acquire without waiting for the back (the
    /// semaphore view, Puller::AskView).
    struct SemaphoreView {
        enum class Kind : u8 {
            Met,         ///< By the word in memory
            MetByQueued, ///< By the last release sent to the word, which the back has not run yet
            NotMet,      ///< Nothing sent to the word meets it yet
            Unknown,     ///< Memory tells once the back ran the work before `position`
            Flush,       ///< Shaders wrote the word: the buffer cache may hold it newer (Puller)
        };
        Kind kind;
        u64 position{};
    };
    /// The view of an acquire on the word at `device_address`, as far as the writes sent to it
    /// tell. `matches(value)`: whether the word holding `value` meets the acquire. `advances`: a
    /// greater-or-equal acquire on a word that only advances. `writers_before`: the
    /// LastWriterPosition when the acquire was sent. `memory()`: the word in memory, as GPU
    /// acquires read it.
    template <typename Matches, typename Memory>
    [[nodiscard]] SemaphoreView ViewSemaphore(u64 device_address, bool advances, u64 writers_before,
                                              Matches&& matches, Memory&& memory) const {
        using ViewKind = SemaphoreView::Kind;
        // Memory that meets a greater-or-equal acquire now still does once the back ran the
        // queued work (the word only advances). Another acquire holds only for the value the
        // word has when the channel reaches it: compute and copy work sent before it may still
        // release the word without the front's knowing (QMD releases), so a value that meets it
        // stands once the back ran that work.
        const bool settled = advances || Ran(writers_before);
        // The last release the front sent to the word decides while it may not be in memory.
        const auto queued = QueuedSemaphoreWriteAt(device_address);
        if (queued && queued->fenced) {
            // It lands once a fence ran, at no position the front can wait for: memory decides,
            // as after a drain, and the landing latches the waiter or a poll finds the word.
            // Writes sent to the word before it are in memory once the back ran it. A
            // greater-or-equal acquire memory meets now meets it then too.
            if (!Ran(queued->position) && !(queued->known && advances)) {
                return {ViewKind::Unknown, queued->position};
            }
        } else if (queued) {
            if (!queued->known) {
                return {ViewKind::Unknown, queued->position};
            }
            if (!matches(queued->payload)) {
                return {ViewKind::NotMet};
            }
            if (settled || queued->position >= writers_before) {
                return {ViewKind::MetByQueued};
            }
            return {ViewKind::Unknown, writers_before};
        }
        if (!matches(memory())) {
            return {ViewKind::NotMet};
        }
        return settled ? SemaphoreView{ViewKind::Met}
                       : SemaphoreView{ViewKind::Unknown, writers_before};
    }
    /// Waits until the back executed the compute and copy work sent before the last NoteSync:
    /// the writers a command list fetched now is ordered after. Returns whether any of it was
    /// still queued. `site`: whose wait it is, for EDEN_NX2_GPU_PROFILE.
    bool WaitForSyncedWriters(DrainSite site = DrainSite::ListWriters);

    /// The entry just sent (Writes::Noted) writes [address, address + size) of `memory`.
    void NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);
    /// The entry just sent may write guest memory anywhere.
    void NoteAnyWrite() {
        last_any_write = write_position;
    }
    /// Waits until the back executed every entry sent so far that can write guest memory in
    /// [address, address + size) of `memory`, not the others. Returns whether one was queued.
    /// Writes are placed by the device memory the GPU addresses map to when noted and read: GPU
    /// maps drain the split first (GPU::MapInOrder), the unmaps it is sent count as anywhere.
    bool WaitForWrites(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

private:
    friend struct GpuSplitTestAccess;

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;
    /// Words a back waiting for work gets without a publish_every (Commit).
    static constexpr size_t PublishAfterWait = 8;

    /// NoteWrite places writes by 256-byte granule of device memory, up to MaxNotedBytes long
    /// (longer ones count as anywhere), in write_positions slots hashed by granule: granules that
    /// share a slot share its position, so a read may wait for a write elsewhere, never miss one.
    static constexpr u32 WriteGranuleBits = 8;
    static constexpr u64 MaxNotedBytes = 0x4000;
    static constexpr u32 WriteSlotBits = 10;

    static size_t WriteSlot(u64 granule) {
        return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> (64 - WriteSlotBits));
    }
    void NoteDeviceWrite(DAddr address, u64 size);
    /// The position WaitForWrites waits for.
    [[nodiscard]] u64 WritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                   u64 size) const;
    [[nodiscard]] u64 DeviceWritesTarget(DAddr address, u64 size) const;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }
    template <typename Func>
    static void CallWordsThunk(const u64* payload) {
        constexpr size_t function_words = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        (*reinterpret_cast<const Func*>(payload))(
            std::span<const u32>{reinterpret_cast<const u32*>(payload + function_words + 1),
                                 static_cast<size_t>(payload[function_words])});
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words) {
        last_entry_start = write_position;
        write_position += words;
        const u64 unpublished = write_position - published_local;
        if (unpublished >= publish_every) {
            Publish();
            return;
        }
        // The back waits for work at the last position published (back_idle_at): it gets these
        // words after a few, not after publish_every. A drain used to restart it like that.
        if (unpublished >= PublishAfterWait &&
            back_idle_at.load(std::memory_order_relaxed) == published_local) {
            Count(g_acquire_stats.early_publishes);
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);

    Core::System* system{};
    bool active{};
    u32 starts{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_sync_position{};
    u64 synced_writer_position{};
    u64 last_entry_start{}; ///< Where the last entry sent starts
    struct SemaphoreWrite {
        u64 position; ///< write_position once it was sent
        /// `fenced`: the entry (EntryTag) whose fenced writes must have landed (NoteFencedWrite)
        u64 entry;
        u32 payload;
        bool known;
        bool fenced;
    };
    /// The front's NoteSemaphoreWrite, by word address.
    std::unordered_map<u64, SemaphoreWrite> semaphore_writes;
    size_t semaphore_writes_limit = 1024; ///< Forget what the back ran past this many
    /// After the last entry that may write guest memory anywhere (NoteAnyWrite), the last one
    /// NoteWrite placed, and the last placed write in each slot's granules.
    u64 last_any_write{};
    u64 last_noted_write{};
    std::array<u64, size_t{1} << WriteSlotBits> write_positions{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    u64 back_entry_start{}; ///< Where the entry the back runs starts (BackEntry)

    /// An entry's position as fenced writes name it: with this start of the split in the high
    /// bits, so a fence of an earlier start that runs late lands nothing of this one (positions
    /// stay far below 2^48).
    [[nodiscard]] u64 EntryTag(u64 position) const {
        return (u64{starts} << 48) | position;
    }
    /// Whether the word holds `write` (as far as no other write changed it since): the back ran
    /// it, and a fenced one landed.
    [[nodiscard]] bool InMemory(const SemaphoreWrite& write) const {
        return write.position <= consumed.load(std::memory_order_acquire) &&
               (!write.fenced || fenced_landed.load(std::memory_order_acquire) > write.entry);
    }

    /// Whether the back is done with the Vulkan scheduler until it gets more work: set after its
    /// idle hook ran, cleared before it executes entries.
    [[nodiscard]] bool BackQuiet() const {
        return back_quiet.load(std::memory_order_acquire) ||
               !back_running.load(std::memory_order_acquire);
    }

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    static inline thread_local bool on_gpu_thread{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};
    std::atomic<bool> back_quiet{true};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// One past the last entry (EntryTag) whose fenced writes landed (NoteFencedWrite): the
    /// fencing thread writes it, the front reads it.
    alignas(128) std::atomic<u64> fenced_landed{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    /// The back's position when it last ran out of work, stored before it waits: while
    /// it equals published_local, the back waits for the words the front holds (Commit). Port of
    /// the publish part of 62db242b34 (jpkto, PR #805).
    alignas(128) std::atomic<u64> back_idle_at{~u64{0}};
    std::jthread back_thread;
};

} // namespace VideoCommon
