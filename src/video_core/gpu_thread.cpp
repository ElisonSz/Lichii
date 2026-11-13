// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#ifdef __linux__
#include <sys/prctl.h>
#elif defined(_WIN32)
#include "common/windows/timer_resolution.h"
#endif

#include "common/assert.h"
#include "common/dkb_cpu_trace.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/front_qmd.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_thread.h"
#include "video_core/host1x/host1x.h"
#include "video_core/nx2_acquire_stats.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_fast_wake.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_sync_suspend.h"
#include "video_core/nx2_sync_trace.h"
#include "video_core/renderer_base.h"

#include "video_core/wait_stats.h"

namespace VideoCommon::GPUThread {

namespace {

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    // A command an earlier session's front left in its queue at shutdown is gone with it.
    VideoCommon::SyncSuspend::queued_commands.store(0, std::memory_order_relaxed);
    thread = std::jthread([&](std::stop_token stop_token) {
        const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuCommand};
        const VideoCommon::StutterTrace::ThreadScope stutter_thread{
            VideoCommon::StutterTrace::Thread::Front};
        Common::SetCurrentThreadName("GPU");
        const VideoCommon::WaitStats::ThreadScope wait_thread{
            VideoCommon::WaitStats::Thread::Front};
        VideoCommon::GpuSplit::MarkGpuThread();
#ifdef __linux__
        // The pending-channel polls below sleep 50 us; the default timer slack adds up to 50 more.
        prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        if (Settings::values.pin_gpu_threads.GetValue()) {
            Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        }
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the polling below).
        auto last_progress = std::chrono::steady_clock::now();
        namespace Profile = VideoCommon::Nx2Profile;
        auto& split = VideoCommon::GpuSplit::Get();
        while (!stop_token.stop_requested()) {
            Profile::Report();
            split.MaybeStart(system);
            if (VideoCommon::g_acquire_counting) {
                split.ReportAcquire();
            }
            if (VideoCommon::SyncTrace::enabled) {
                VideoCommon::SyncTrace::MaybeReport();
            }
            VideoCommon::Nx2DrainCheck::MaybeReport();
            VideoCommon::Nx2AcquireStats::MaybeReport();
            VideoCommon::Nx2AcquireDetails::Report();
            if (scheduler.HasPendingChannels()) {
                // Take this before polling: a latch between the poll and sleep must not be lost.
                const u64 pending_generation =
                    VideoCommon::FastWake::enabled ? VideoCommon::FastWake::pending.Snapshot() : 0;
                // Suspended channels wait on semaphores the CPU releases: poll them meanwhile.
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                if (resumed) {
                    last_progress = std::chrono::steady_clock::now();
                }
                if (split.Active()) {
                    VideoCommon::FrontQmdView::Get().MaybeTakeOver(
                        split, VideoCommon::Nx2AcquireStats::QmdTakeOver::BetweenCommands);
                }
                if (!state.queue.TryPop(next)) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    if (VideoCommon::SyncSuspend::enabled && split.Active()) [[unlikely]] {
                        // A channel suspended before the back ran the release it waits for
                        // (EDEN_NX2_SYNC_SUSPEND): with nothing else to send, wait for that entry
                        // as the baseline's acquire did, then poll from there. Bounded: after
                        // IdleWaitLimit, new commands and the channels are looked at again.
                        if (const auto target = scheduler.EarlyWaitPosition();
                            target && !split.Ran(*target)) {
                            namespace Suspension = VideoCommon::SyncSuspend;
                            const auto begin = Suspension::Clock::now();
                            const bool reached =
                                split.WaitForPositionFor(*target, Suspension::IdleWaitLimit);
                            auto& counts = Suspension::counters;
                            ++counts.idle_waits;
                            counts.idle_wait_ns +=
                                Suspension::Nanoseconds(Suspension::Clock::now() - begin);
                            counts.idle_timeouts += reached ? 0 : 1;
                            last_progress = std::chrono::steady_clock::now();
                            continue;
                        }
                    }
                    // The CPU usually releases the semaphore soon after: poll for half a
                    // millisecond, then in short sleeps. Ocarina of Time waits here ~330 times a
                    // second and its present waits for the work behind them: with 1 ms sleeps the
                    // present waited ~17 ms a frame in Kokiri village, with 50 us ~1 ms. A channel
                    // still waiting after 2 s (menus, loading) goes back to 1 ms sleeps.
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    // Words sent since the last publish (a poll's flush or fence work, a resumed
                    // channel's) would wait out the sleep below while the back runs out of work:
                    // hand them over first. Publish does nothing when nothing new was sent.
                    if (split.Active()) {
                        split.Publish();
                    }
                    split.SettleIdleBack();
                    const auto waited = std::chrono::steady_clock::now() - last_progress;
                    if (waited < std::chrono::microseconds{500}) {
                        std::this_thread::yield();
                    } else if (VideoCommon::FastWake::enabled) {
                        auto& wake_counts = VideoCommon::FastWake::counters;
                        wake_counts.pending_waits.fetch_add(1, std::memory_order_relaxed);
                        // Keep Windows fallback polling near the old timer tick; real latches
                        // and queued commands interrupt it rather than waiting for the timer.
#ifdef _WIN32
                        constexpr auto poll_timeout = std::chrono::microseconds{500};
#else
                        constexpr auto poll_timeout = std::chrono::microseconds{50};
#endif
                        const auto timeout = waited < std::chrono::seconds{2}
                                                 ? poll_timeout
                                                 : std::chrono::microseconds{1000};
                        const auto result =
                            VideoCommon::FastWake::pending.Wait(pending_generation, timeout);
                        auto& counter = result == VideoCommon::FastWake::Signal::Result::Notified
                                            ? wake_counts.pending_notified
                                            : wake_counts.pending_timeouts;
                        counter.fetch_add(1, std::memory_order_relaxed);
                    } else if (waited < std::chrono::seconds{2}) {
#ifdef _WIN32
                        // sleep_for(50us) rounds up to Sleep(1) (>= 1 ms) on Windows;
                        // SleepForOneTick uses NtDelayExecution(-1) for one high-res timer tick.
                        Common::Windows::SleepForOneTick();
#else
                        std::this_thread::sleep_for(std::chrono::microseconds{50});
#endif
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    }
                    continue;
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (VideoCommon::SyncSuspend::enabled) [[unlikely]] {
                VideoCommon::SyncSuspend::queued_commands.fetch_sub(1, std::memory_order_relaxed);
            }
            if (split.Active()) {
                split.PublishSoon();
                VideoCommon::FrontQmdView::Get().MaybeTakeOver(
                    split, VideoCommon::Nx2AcquireStats::QmdTakeOver::BetweenCommands);
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    Profile::Scope scope{Profile::Zone::Push};
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    // TickWork ends fast draw reuse itself when it runs a request.
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); },
                                VideoCommon::GpuSplit::FastDrawNeutral,
                                VideoCommon::GpuSplit::Writes::Caches);
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        // GPU memory maps unmap caches (UnmapMemory downloads): the back's state.
                        split.DeferFunction(std::move(run->func));
                        split.Drain(VideoCommon::DrainSite::Map);
                    } else {
                        split.DeferFunction(std::move(run->func));
                        split.TagUnplaced(VideoCommon::UnplacedKind::RunCommand);
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = flush->addr,
                                 size = flush->size] { rasterizer_ptr->FlushRegion(addr, size); },
                                0, VideoCommon::GpuSplit::Writes::Caches);
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    // EDEN_NX2_DRAWX_CALLS2: the caches note the write (BufferCache::
                    // WriteMemory) or break (texture and shader invalidations), as for the CPU
                    // writes other threads apply directly.
                    split.Defer(
                        [rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                            rasterizer_ptr->OnCacheInvalidation(addr, size);
                        },
                        VideoCommon::Nx2FastDraw::CallFlags2(),
                        VideoCommon::GpuSplit::Writes::Noted);
                } else {
                    ASSERT(false);
                }
                split.Defer([state_ptr = &state, fence = next.fence,
                             block = next.block] { SignalCommand(*state_ptr, fence, block); },
                            VideoCommon::GpuSplit::FastDrawNeutral,
                            VideoCommon::GpuSplit::Writes::Noted);
                split.Publish();
                last_progress = std::chrono::steady_clock::now();
                continue;
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Push};
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
            last_progress = std::chrono::steady_clock::now();
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads) {
    PushCommand(RunCommand{std::move(func), before_later_reads}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    if (VideoCommon::SyncSuspend::enabled) [[unlikely]] {
        // Before the push: the front takes it off only after it is counted.
        VideoCommon::SyncSuspend::queued_commands.fetch_add(1, std::memory_order_relaxed);
    }
    state.queue.EmplaceWait(std::move(command_data), fence, block);
    if (VideoCommon::FastWake::enabled && VideoCommon::FastWake::pending.Notify()) {
        VideoCommon::FastWake::counters.command_signals.fetch_add(1, std::memory_order_relaxed);
    }

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        if (VideoCommon::FastWake::enabled) {
            VideoCommon::FastWake::pending.Notify();
        }
        thread.join();
    }
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
