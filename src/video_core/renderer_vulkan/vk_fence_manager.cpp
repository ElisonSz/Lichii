// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_sync_trace.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

InnerFence::~InnerFence() = default;

void InnerFence::Queue() {
    if (is_stubbed) {
        if (VideoCommon::SyncTrace::enabled) {
            VideoCommon::SyncTrace::Attach(sync_trace_id, &scheduler, 0, true, false);
        }
        return;
    }
    const VideoCommon::WaitStats::ReasonScope reason_scope{VideoCommon::WaitStats::Reason::Fence};
    if (VideoCommon::WaitStats::Enabled()) {
        wait_reason = VideoCommon::WaitStats::CurrentReason();
    }
    // Get the current tick so we can wait for it
    wait_tick = scheduler.CurrentTick();
    const bool deferred = scheduler.DeferFlush();
    if (VideoCommon::SyncTrace::enabled) {
        VideoCommon::SyncTrace::Attach(sync_trace_id, &scheduler, wait_tick, false, deferred);
    }
    if (!deferred) {
        scheduler.Flush();
    }
}

bool InnerFence::IsSignaled() const {
    if (is_stubbed) {
        return true;
    }
    return scheduler.IsFree(wait_tick);
}

void InnerFence::Wait() {
    const VideoCommon::WaitStats::ReasonScope reason_scope{wait_reason};
    if (is_stubbed) {
        return;
    }
    // A deferred fence flush (Scheduler::DeferFlush) may not be submitted yet. Only the
    // split's back may submit; other threads (the fencing thread) ask it to and wait. If the back
    // has stopped, nothing else uses the scheduler and Wait below submits it.
    if (!scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread() && scheduler.WaitSubmitted(wait_tick)) {
        if (VideoCommon::SyncTrace::enabled) {
            VideoCommon::SyncTrace::Mark(sync_trace_id, VideoCommon::SyncTrace::SubmissionReady);
        }
        scheduler.GetMasterSemaphore().Wait(wait_tick);
        return;
    }
    if (VideoCommon::SyncTrace::enabled) {
        VideoCommon::SyncTrace::Mark(sync_trace_id, VideoCommon::SyncTrace::SubmissionReady);
    }
    scheduler.Wait(wait_tick);
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, const Device& device_, Scheduler& scheduler_)
    : GenericFenceManager{rasterizer_, gpu_, texture_cache_, buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    fence->Queue();
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

void FenceManager::SubmitDeferredWork() {
    if (VideoCommon::GpuSplit::OnBackThread()) {
        scheduler.FlushDeferred();
    }
}

} // namespace Vulkan
