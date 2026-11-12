// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "common/common_types.h"
#include "video_core/nx2_fast_wake.h"
#include "video_core/nx2_sem_recorder.h"
#include "video_core/nx2_sync_trace.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
enum class EngineTypes : u32;
}

namespace VideoCommon {

/// TEST (EDEN_NX2_GPU_PROFILE / EDEN_NX2_ACQUIRE_VERIFY): what sent the last write the front could
/// not place (GpuSplit::TagUnplaced), for why the QMD view could not value a signal.
enum class UnplacedKind : u8 {
    Other,      ///< A deferred call nobody tagged
    RunCommand, ///< A function the CPU queued in order (unmaps, events)
    Engine,     ///< An engine method the front sends as it is (blits)
    Oversized,  ///< A placed write longer than the coarse slots take
    QmdLaunch,  ///< A QMD launch the view could not value
    InlineQmd,  ///< An inline QMD behind a change of the mappings
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(UnplacedKind::Count)>
    unplaced_kind_names{"other",        "CPU functions", "engines", "oversized writes",
                        "QMD launches", "inline QMDs"};

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
    /// Back thread: the entry it runs; elsewhere nothing.
    [[nodiscard]] std::optional<u64> BackEntry() const {
        if (!on_back_thread) {
            return std::nullopt;
        }
        return EntryTag(back_entry_start);
    }
    /// Back thread: retain this in each deferred report operation until it finishes or is
    /// discarded. The entry stays pending until all its reports complete and Execute returns,
    /// so an early fence cannot land a report the same entry has yet to send.
    [[nodiscard]] std::shared_ptr<void> TrackFencedWrite();
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
        WakeBack();
    }

    /// Set before the GPU threads start, cleared after they have joined.
    void SetTickHook(void (*hook)(void*), void* argument) {
        tick_hook = hook;
        tick_argument = argument;
    }
    /// Any thread: ask the back to tick after its current entry, or wake it if idle. Callers
    /// also queue a normal GPU tick, which serves requests when the split stops or is disabled.
    void RequestTick();
    /// Reload EDEN_NX2_BACK_TICK after the frontend loads nx2_switches.txt.
    static void ReadTickSwitch();
    [[nodiscard]] static bool TickEnabled();

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// What the sender of an entry tells about the guest memory the back writes running it, for
    /// the front's reads of guest memory (WaitForWrites, MemoryWritesTarget).
    enum class Writes : u8 {
        /// Unknown: a read anywhere waits for the entry, and it may change the GPU's mappings
        /// (unmaps; GPU maps drain the split)
        Anywhere,
        /// Guest memory anywhere, the mappings not: engine work whose writes nobody placed
        Unplaced,
        /// The GPU caches anywhere (shader stores, copies into buffers), guest memory only through
        /// them, where GPU work wrote (Nx2MayBeGpuWritten): a read of the caches waits for it, a
        /// read of other guest memory not (MemoryWritesTarget)
        Caches,
        Noted, ///< Nothing, or what the sender notes right after sending it (NoteWrite)
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
        if (log_methods) {
            LogMethod(engine, method); // TEST
        }
        NoteWrites(writes);
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
        if (log_methods) {
            LogMethod(engine, method); // TEST
        }
        NoteWrites(writes);
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
        NoteWrites(writes);
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
        NoteWrites(writes);
    }
    void DeferFunction(std::function<void()>&& func, Writes writes = Writes::Anywhere);
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
    class PublishHold {
    public:
        PublishHold(GpuSplit& split_, bool hold) : split{split_}, was_held{split_.publish_held} {
            if (hold) {
                split.publish_held = true;
            }
        }
        ~PublishHold() {
            if (split.publish_held != was_held) {
                split.publish_held = was_held;
                split.PublishIfDue();
            }
        }
        PublishHold(const PublishHold&) = delete;
        PublishHold& operator=(const PublishHold&) = delete;

    private:
        GpuSplit& split;
        bool was_held;
    };

    void PublishSoon() {
        publish_every = PublishAfterWait;
    }
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
    [[nodiscard]] std::string MethodEndingAt(u64 end) const;
    [[nodiscard]] u64 Backlog() const {
        return write_position - consumed.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 Unpublished() const {
        return write_position - published_local;
    }
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);
    /// The ring position after the entries sent so far, for Ran and WaitForPosition.
    [[nodiscard]] u64 Position() const {
        return write_position;
    }
    /// Waits until the back executed the entries sent before `position`, not what came after.
    /// Returns whether any of them was still queued.
    bool WaitForPosition(u64 position, DrainSite site) {
        if (SemRecorder::Enabled()) {
            SemRecorder::Add({.kind = "wait-begin",
                              .position = position,
                              .consumed = consumed.load(std::memory_order_acquire)});
        }
        const bool waited = WaitFor((std::min)(position, write_position), site);
        if (SemRecorder::Enabled()) {
            SemRecorder::Add({.kind = "wait-end",
                              .position = position,
                              .consumed = consumed.load(std::memory_order_acquire)});
        }
        return waited;
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
    /// Diagnostic identity of a noted write; it never participates in the semaphore view.
    enum class SemaphoreWriter : u8 {
        Other,
        Puller,
        Report3D,
        ReportCompute,
        ReductionCompute,
        QmdRelease,
        QmdReduction,
        Copy,
        Count,
    };
    /// A semaphore release was just sent: once it landed (`landing`), the word at
    /// `device_address` holds `payload`, or a value the front can't tell (`known` false: a
    /// reduction, a counter report).
    void NoteSemaphoreWrite(u64 device_address, u32 payload, bool known,
                            Landing landing = Landing::AtRun, bool at_least = false,
                            SemaphoreWriter writer = SemaphoreWriter::Other);
    void NoteSemaphoreWrite(u64 device_address, u32 payload, bool known, bool at_least,
                            SemaphoreWriter writer = SemaphoreWriter::Other) {
        NoteSemaphoreWrite(device_address, payload, known, Landing::AtRun, at_least, writer);
    }
    struct QueuedSemaphoreWrite {
        u32 payload;
        bool known;
        bool at_least;
        /// It lands once a fence ran, at no position the front can wait for; or an earlier write
        /// that does may still land after it.
        bool fenced;
        SemaphoreWriter writer;
        u64 position; ///< Once the back ran this (Ran), the word holds the write, unless `fenced`
    };
    /// The last write sent to the semaphore word at `device_address`, if it may not be in memory
    /// yet.
    [[nodiscard]] std::optional<QueuedSemaphoreWrite> QueuedSemaphoreWriteAt(
        u64 device_address) const;

    /// Diagnostic snapshot, including a write that reached memory since the view was asked.
    /// Only statistics calls this: retiring a write must not erase the cause of a wait.
    [[nodiscard]] std::optional<QueuedSemaphoreWrite> SemaphoreWriteForStats(
        u64 device_address) const;
    [[nodiscard]] SemRecorder::Event SemaphoreRecordState(u64 device_address,
                                                          const char* kind) const;

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
        constexpr SemaphoreView(Kind kind_, u64 position_ = 0, bool write_dependency_ = false)
            : kind{kind_}, write_dependency{write_dependency_}, position{position_} {}
        Kind kind;
        bool write_dependency; ///< Unknown: position names the noted write to this word
        u64 position;
    };
    /// The view of an acquire on the word at `device_address`, as far as the writes sent to it
    /// tell. `matches(value)`: whether the word holding `value` meets the acquire. `advances`: a
    /// greater-or-equal acquire on a word that only advances. `writers_before`: the
    /// LastWriterPosition when the acquire was sent. `memory(fenced)`: the word in memory, as GPU
    /// acquires read it, or none if that source is unreliable while a fenced write is pending.
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
            // Earlier fenced writes can still land after the back ran this entry. The memory
            // reader must reject a source those callbacks can invalidate (a mirror shadow).
            if (!Ran(queued->position) && !(queued->known && advances)) {
                return {ViewKind::Unknown, queued->position, true};
            }
        } else if (queued) {
            if (!queued->known) {
                return {ViewKind::Unknown, queued->position, true};
            }
            if (queued->at_least) {
                return advances && matches(queued->payload)
                           ? SemaphoreView{ViewKind::MetByQueued}
                           : SemaphoreView{ViewKind::Unknown, queued->position, true};
            }
            if (!matches(queued->payload)) {
                return {ViewKind::NotMet};
            }
            if (settled || queued->position >= writers_before) {
                return {ViewKind::MetByQueued};
            }
            return {ViewKind::Unknown, writers_before};
        }
        const std::optional<u32> current = memory(queued && queued->fenced);
        if (!current || !matches(*current)) {
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
    void NoteWriteAt(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size, u64 position);
    /// The entry just sent may write guest memory anywhere.
    void NoteAnyWrite() {
        NoteUnplacedWrite();
        last_mapping_change = write_position;
    }
    void NoteUnplacedWrite() {
        last_any_write = write_position;
        last_unplaced_write = write_position;
    }
    /// The entry just sent may write the GPU caches anywhere (Writes::Caches).
    void NoteCacheWrite() {
        last_any_write = write_position;
    }
    void NoteWrites(Writes writes) {
        if (writes == Writes::Anywhere) {
            NoteAnyWrite();
        } else if (writes == Writes::Unplaced) {
            NoteUnplacedWrite();
        } else if (writes == Writes::Caches) {
            NoteCacheWrite();
        }
    }
    /// The position after the last entry sent that may write guest memory where the front does
    /// not know (NoteAnyWrite), outside what the caches hold.
    [[nodiscard]] u64 LastUnplacedWrite() const {
        return (std::min)(last_unplaced_write, write_position);
    }
    [[nodiscard]] u64 LastMappingChange() const {
        return (std::min)(last_mapping_change, write_position);
    }
    void TagUnplaced(UnplacedKind kind) {
        if (last_unplaced_write == write_position) {
            last_unplaced_kind = kind;
        }
        if (last_mapping_change == write_position) {
            last_mapping_kind = kind;
        }
    }
    [[nodiscard]] UnplacedKind LastUnplacedKind() const {
        return last_unplaced_kind;
    }
    [[nodiscard]] UnplacedKind LastMappingKind() const {
        return last_mapping_kind;
    }
    /// Once the back ran everything before the position returned, no entry sent so far writes
    /// guest memory in [address, address + size) of `memory` but through the GPU caches
    /// (Writes::Caches): what the back reads there without them, as the QMD scheduler does, is in
    /// memory then. Placed as WaitForWrites places writes.
    [[nodiscard]] u64 MemoryWritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                         u64 size) const;
    /// Waits until the back executed every entry sent so far that can write guest memory in
    /// [address, address + size) of `memory`, not the others. Returns whether one was queued.
    /// Writes are placed by the device memory the GPU addresses map to when noted and read: GPU
    /// maps drain the split first (GPU::MapInOrder), the unmaps it is sent count as anywhere.
    bool WaitForWrites(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

private:
    friend struct GpuSplitTestAccess;
    friend struct FastWakeTestAccess;

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

    /// NoteWrite places writes by 256-byte granule of device memory, up to MaxNotedBytes long,
    /// in sets of WriteWays slots hashed by granule. Writes up to MaxCoarseNotedBytes (copies,
    /// uploads) go by 64 KiB granule into coarse sets, longer ones count as unplaced. A slot holds
    /// a granule and the position of its last write; a write to a granule its set lacks takes the
    /// slot written longest ago, and the set keeps the latest position of the writes it so
    /// dropped (`evicted`). A read waits for its granule's own last write while the set holds
    /// it, else for the evicted ones: it may wait for a write elsewhere, never miss one, and the
    /// writes to other granules hashed there (the constant buffer words of queued draws) hold up
    /// a read of a QMD only once they pushed it out of its set.
    static constexpr u32 WriteGranuleBits = 8;
    static constexpr u64 MaxNotedBytes = 0x4000;
    static constexpr u32 WriteSlotBits = 12; ///< Sets
    static constexpr u32 CoarseGranuleBits = 16;
    static constexpr u64 MaxCoarseNotedBytes = u64{16} << 20;
    static constexpr u32 CoarseSlotBits = 10;
    static constexpr size_t WriteWays = 4;

    /// One cache line.
    struct WriteSet {
        std::array<u64, WriteWays> positions;
        /// The granules' low 32 bits: granules that agree in them share a slot.
        std::array<u32, WriteWays> granules;
        u64 evicted;
        u64 unused;
    };
    static_assert(sizeof(WriteSet) == 64);
    /// The set a granule is hashed to.
    static size_t WriteSlot(u64 granule) {
        return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> (64 - WriteSlotBits));
    }
    static size_t CoarseSlot(u64 granule) {
        return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> (64 - CoarseSlotBits));
    }
    /// A write to `granule` at `position` (the newest, or an earlier one placed late).
    static void NoteSetWrite(WriteSet& set, u64 granule, u64 position);
    /// The latest position of the writes to `granule` in its set, at most.
    [[nodiscard]] static u64 SetTarget(const WriteSet& set, u64 granule);
    void NoteDeviceWrite(DAddr address, u64 size) {
        NoteDeviceWriteAt(address, size, write_position);
    }
    void NoteDeviceWriteAt(DAddr address, u64 size, u64 position);
    void NoteCoarseDeviceWrite(DAddr address, u64 size) {
        NoteCoarseDeviceWriteAt(address, size, write_position);
    }
    void NoteCoarseDeviceWriteAt(DAddr address, u64 size, u64 position);
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
        if (profile_publishes && write_position == published_local) {
            NoteFirstUnpublished();
        }
        last_entry_start = write_position;
        if (SyncTrace::enabled) {
            SyncTrace::Commit(write_position);
        }
        write_position += words;
        PublishIfDue();
    }
    void PublishIfDue() {
        if (publish_held) {
            return;
        }
        const u64 unpublished = write_position - published_local;
        if (FastWake::enabled && unpublished != 0 &&
            back_idle_at.load(std::memory_order_relaxed) == published_local) {
            FastWake::counters.idle_publishes.fetch_add(1, std::memory_order_relaxed);
            Publish();
            return;
        }
        if (unpublished >= publish_every) {
            Publish();
            return;
        }
        // The back waits for work at the last position published (back_idle_at): it gets these
        // words after a few, not after publish_every. A drain used to restart it like that.
        if (unpublished >= PublishAfterWait &&
            back_idle_at.load(std::memory_order_relaxed) == published_local) {
            Count(g_acquire_stats.early_publishes);
            if (profile_publishes) {
                NoteEarlyPublish();
            }
            Publish();
        }
    }
    void NoteFirstUnpublished();
    void NoteEarlyPublish();
    void NoteLatePublish();
    void WaitForSpace(size_t words);
    /// TEST (EDEN_NX2_GPU_PROFILE / EDEN_NX2_ACQUIRE_VERIFY): the engine method entries sent, by
    /// the position after them.
    struct SentMethod {
        u64 end;
        std::optional<Tegra::Engines::EngineTypes> engine; ///< The class it was bound as
        u32 method;
    };
    static constexpr size_t LoggedMethods = size_t{1} << 16;
    /// EDEN_NX2_GPU_PROFILE / EDEN_NX2_ACQUIRE_VERIFY as the split started (MaybeStart).
    bool log_methods{};
    void LogMethod(const Tegra::Engines::EngineInterface* engine, u32 method);
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
            if (FastWake::enabled) {
                if (position_signal.Notify()) {
                    FastWake::counters.position_signals.fetch_add(1, std::memory_order_relaxed);
                }
                return;
            }
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void WakeBack() {
        if (FastWake::enabled) {
            if (back_signal.Notify()) {
                FastWake::counters.back_signals.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    void BackLoop(std::stop_token stop_token);
    /// The back, between entries. Ticking may record deferred work even when the ring is empty.
    void MaybeTick() {
        if ((tick_state.load(std::memory_order_relaxed) & TickRequested) == 0) {
            return;
        }
        const u64 state = tick_state.fetch_and(~TickRequested, std::memory_order_acquire);
        if ((state & TickRequested) != 0 && tick_hook != nullptr) {
            back_quiet.store(false, std::memory_order_seq_cst);
            tick_hook(tick_argument);
        }
    }
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its
    /// size.
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
        /// `fenced`: the entry (EntryTag) whose deferred reports must all have completed
        u64 entry;
        u32 payload;
        bool known;
        bool at_least;
        bool fenced;
        SemaphoreWriter writer;
    };
    /// The front's NoteSemaphoreWrite, by word address.
    std::unordered_map<u64, SemaphoreWrite> semaphore_writes;
    size_t semaphore_writes_limit = 1024; ///< Forget what the back ran past this many
    /// After the last entry that may write guest memory anywhere (NoteAnyWrite), the last one
    /// NoteWrite placed, and the last placed write in each slot's granules.
    u64 last_any_write{};
    u64 last_unplaced_write{};
    u64 last_mapping_change{};
    UnplacedKind last_unplaced_kind{};
    UnplacedKind last_mapping_kind{};
    u64 last_noted_write{};
    alignas(64) std::array<WriteSet, size_t{1} << WriteSlotBits> write_sets{};
    alignas(64) std::array<WriteSet, size_t{1} << CoarseSlotBits> coarse_sets{};
    bool profile_publishes{};
    u64 first_unpublished_ticks{};
    std::unique_ptr<SentMethod[]> logged_methods;
    u64 logged_method_count{};
    bool publish_held{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    u64 back_entry_start{}; ///< Where the entry the back runs starts (BackEntry)
    struct FencedEntry {
        FencedEntry(GpuSplit& owner, u64 entry) : split{owner}, tag{entry} {}
        GpuSplit& split;
        u64 tag;
        ~FencedEntry();
    };
    std::shared_ptr<FencedEntry> back_fenced_entry;
    struct PendingFencedEntry {
        u64 tag;
        bool complete{};
    };
    /// Registration is on the back; completion can run on any fence or cancellation thread.
    /// The front only reads fenced_landed, never this lock or queue.
    std::mutex fenced_entries_mutex;
    std::deque<PendingFencedEntry> fenced_entries;
    u64 fenced_sealed{}; ///< EntryTag of the position after the last published consumed entry
    void CompleteFencedEntry(u64 tag);
    void SealFencedEntries(u64 position);
    void PublishFencedEntries(); ///< With fenced_entries_mutex held

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
    void (*tick_hook)(void*){};
    void* tick_argument{};
    // One state arbitrates requests with back exit. Each start advances the generation, so a
    // caller delayed across Stop/Start cannot leave a request for the next back thread.
    static constexpr u64 TickAccepting = 1;
    static constexpr u64 TickRequested = 2;
    static constexpr u64 TickGeneration = 4;
    std::atomic<u64> tick_state{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};
    std::atomic<bool> back_quiet{true};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// Entries before this EntryTag have finished every deferred report or discarded them.
    /// Published under fenced_entries_mutex, never beyond the back's sealed position.
    alignas(128) std::atomic<u64> fenced_landed{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    FastWake::Signal back_signal;
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    FastWake::Signal position_signal;
    /// The back's position when it last ran out of work, stored before it waits: while
    /// it equals published_local, the back waits for the words the front holds (Commit). Port of
    /// the publish part of 62db242b34 (jpkto, PR #805).
    alignas(128) std::atomic<u64> back_idle_at{~u64{0}};
    alignas(128) std::atomic<u64> back_idle_since{};
    std::jthread back_thread;
};

} // namespace VideoCommon
