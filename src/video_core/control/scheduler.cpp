// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <memory>
#include <vector>

#include "common/assert.h"
#include "core/core.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_sync_suspend.h"
#include "video_core/nx2_sync_trace.h"

namespace Tegra::Control {

void Scheduler::BindChannel(GPU& gpu, s32 bind_id) {
    // With the GPU thread split the rasterizer's channel switches with the methods around it, on
    // the back thread (deferred outside the lock: the back takes it to bind).
    // EDEN_NX2_DRAWX_CALLS2: binding the channel bound already does nothing on the back
    // (GPU::BindChannel returns at once), so it doesn't end the fast draw chain there.
    const bool rebind = bind_id == last_bind_id;
    last_bind_id = bind_id;
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(
            [this, gpu_ptr = &gpu, bind_id] {
                std::scoped_lock lk{scheduling_guard};
                gpu_ptr->BindChannel(bind_id);
            },
            rebind ? VideoCommon::Nx2FastDraw::CallFlags2() : 0,
            VideoCommon::GpuSplit::Writes::Noted);
        return;
    }
    std::scoped_lock lk{scheduling_guard};
    gpu.BindChannel(bind_id);
}

void Scheduler::Push(GPU& gpu, s32 channel, CommandList&& entries) {
    std::shared_ptr<ChannelState> channel_state;
    {
        std::unique_lock lk(scheduling_guard);
        auto it = channels.find(channel);
        ASSERT(it != channels.end());
        channel_state = it->second;
    }
    const VideoCommon::SyncTrace::FrontScope sync_scope{
        VideoCommon::SyncTrace::enabled ? &channel_state->payload->dma_pusher.puller : nullptr};
    BindChannel(gpu, channel_state->bind_id);
    // Process commands outside the lock to reduce contention.
    // Multiple channels can prepare their commands in parallel.
    channel_state->payload->dma_pusher.Push(std::move(entries));
    if (VideoCommon::SyncSuspend::enabled) [[unlikely]] {
        NoteOtherPending(*channel_state);
    }
    channel_state->payload->dma_pusher.DispatchCalls();
    if (channel_state->payload->dma_pusher.HasPendingAcquire() &&
        std::find(pending_channels.begin(), pending_channels.end(), channel_state) ==
            pending_channels.end()) {
        pending_channels.push_back(channel_state);
        any_pending.store(true, std::memory_order_release);
    }
    if (!pending_channels.empty()) {
        ResumePending(gpu);
    }
}

bool Scheduler::ResumePending(GPU& gpu) {
    bool any_progress = false;
    // Running one channel can release the semaphore another waits for: retry until none moves.
    for (bool progress = true; progress;) {
        progress = false;
        for (const auto& state : pending_channels) {
            auto& dma_pusher = state->payload->dma_pusher;
            // Powered off (shutting down), DispatchCalls returns at once and the acquire stays
            // pending: resuming it again counted as progress forever and the GPU thread never
            // reached its stop request (closing the game hung).
            if (!dma_pusher.system.IsPoweredOn()) {
                return any_progress;
            }
            if (!dma_pusher.HasPendingAcquire() || !dma_pusher.CanResumeAcquire()) {
                continue;
            }
            if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                split.PublishSoon();
            }
            VideoCommon::BackIdleTrace::SetFrontState(
                VideoCommon::BackIdleTrace::FrontState::Parse);
            const VideoCommon::SyncTrace::FrontScope sync_scope{
                VideoCommon::SyncTrace::enabled ? &dma_pusher.puller : nullptr};
            BindChannel(gpu, state->bind_id);
            if (VideoCommon::SyncSuspend::enabled) [[unlikely]] {
                NoteOtherPending(*state);
            }
            dma_pusher.DispatchCalls();
            progress = true;
            any_progress = true;
        }
    }
    std::erase_if(pending_channels, [](const auto& state) {
        return !state->payload->dma_pusher.HasPendingAcquire();
    });
    any_pending.store(!pending_channels.empty(), std::memory_order_release);
    return any_progress;
}

void Scheduler::NoteOtherPending(const ChannelState& channel) const {
    u64 others = 0;
    for (const auto& state : pending_channels) {
        const bool other = state.get() != &channel;
        others += (other && state->payload->dma_pusher.HasPendingAcquire()) ? 1 : 0;
    }
    VideoCommon::SyncSuspend::other_pending = others;
}

std::optional<u64> Scheduler::EarlyWaitPosition() const {
    std::optional<u64> earliest;
    for (const auto& state : pending_channels) {
        if (const auto position = state->payload->dma_pusher.puller.EarlyPosition()) {
            earliest = earliest ? (std::min)(*earliest, *position) : *position;
        }
    }
    return earliest;
}

bool Scheduler::WaitsForListCopy() const {
    return std::ranges::any_of(pending_channels, [](const auto& state) {
        return state->payload->dma_pusher.WaitsForListCopy();
    });
}

void Scheduler::DeclareChannel(std::shared_ptr<ChannelState> new_channel) {
    s32 channel = new_channel->bind_id;
    std::unique_lock lk(scheduling_guard);
    channels.emplace(channel, new_channel);
}

} // namespace Tegra::Control
