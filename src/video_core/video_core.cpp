// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>

#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/front_qmd.h"
#include "video_core/gpu_split.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memtrack_translation.h"
#include "video_core/nx2_acquire_stats.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_backx.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_download_trace.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_fast_wake.h"
#include "video_core/nx2_list_indirect.h"
#include "video_core/nx2_list_indirect_stats.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_protect.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/nx2_sync_suspend.h"
#include "video_core/nx2_sync_trace.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_null/renderer_null.h"
#include "video_core/stream_stats.h"
#include "video_core/wait_stats.h"
#ifdef HAS_OPENGL
#include "video_core/renderer_opengl/renderer_opengl.h"
#endif
#include "video_core/renderer_vulkan/async_teardown.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/texture_cache/image_join.h"
#include "video_core/texture_cache/stale_targets.h"
#include "video_core/video_core.h"

namespace {

std::unique_ptr<VideoCore::RendererBase> CreateRenderer(Core::System& system, Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu, std::unique_ptr<Core::Frontend::GraphicsContext> context) {
    [[maybe_unused]] auto& device_memory = system.Host1x().MemoryManager();
    switch (Settings::values.renderer_backend.GetValue()) {
#ifdef HAS_OPENGL
    case Settings::RendererBackend::OpenGL_GLSL:
    case Settings::RendererBackend::OpenGL_GLASM:
    case Settings::RendererBackend::OpenGL_SPIRV:
        return std::make_unique<OpenGL::RendererOpenGL>(emu_window, device_memory, gpu, std::move(context));
#endif
    case Settings::RendererBackend::Vulkan:
        return std::make_unique<Vulkan::RendererVulkan>(emu_window, device_memory, gpu, std::move(context));
    case Settings::RendererBackend::Null:
        return std::make_unique<Null::RendererNull>(emu_window, gpu, std::move(context));
    default:
        return nullptr;
    }
}

} // Anonymous namespace

namespace VideoCore {

void WaitForAsyncTeardown() {
    Vulkan::AsyncTeardown::Get().Wait();
}

/// @brief Creates an emulated GPU instance using the given system context.
void CreateGPU(std::optional<Tegra::GPU>& gpu, Core::Frontend::EmuWindow& emu_window,
               Core::System& system) {
    Settings::UpdateRescalingInfo();
    VideoCommon::FastWake::ReadSwitch();
    VideoCommon::SyncSuspend::ReadSwitch();
    VideoCommon::SyncTrace::ReadSwitch();
    VideoCommon::StreamStats::ReadSwitch();
    VideoCommon::WaitStats::ReadSwitch();
    VideoCommon::Nx2Profile::ReadSwitch();
    VideoCommon::ReadQmdViewSwitch();
    VideoCommon::Nx2AcquireStats::ReadSwitch();
    VideoCommon::Nx2AcquireDetails::ReadSwitch();
    VideoCommon::StutterTrace::StartSession();
    Tegra::Nx2CmdPageDiag::ReadSwitch();
    VideoCommon::Nx2DownloadTrace::ReadSwitch();
    VideoCore::AsyncReadback::ReadSwitch();
    VideoCommon::Nx2Perf::ReadSwitch();
    VideoCommon::MemTrack::ReadSwitch();
    if (VideoCommon::MemTrack::g_word) {
        LOG_INFO(HW_GPU, "Nx2MemTrack single-page CPU marks enabled");
    }
    if (VideoCommon::MemTrack::g_translate) {
        LOG_INFO(HW_GPU, "Nx2MemTrack translation cache enabled: entries={} verify={} stats={}",
                 VideoCommon::MemTrack::TranslationCache::NumEntries,
                 VideoCommon::MemTrack::g_verify, VideoCommon::MemTrack::g_stats);
    }
    Tegra::Nx2ListIndirect::ReadSwitch();
    VideoCommon::Nx2DrainCheck::ReadSwitch();
    Tegra::Nx2ListIndirect::Stats::ReadSwitch();
    VideoCommon::GpuSplit::ReadTickSwitch();
    VideoCommon::ReadUniformStreamReuseSwitch();
    VideoCommon::Nx2FastDraw::ReadDrawCostStatsSwitch();
    VideoCommon::Nx2FastDraw::ReadDrawDeltaSwitch();
    VideoCommon::Nx2FastDraw::ReadDrawReuseSwitches();
    VideoCommon::StaleTargets::ReadSwitch();
    VideoCommon::ImageJoin::ReadSwitches();
    VideoCommon::Nx2Protect::ReadSwitch();
    VideoCommon::BackX::ReadSwitches();

    const auto nvdec_value = Settings::values.nvdec_emulation.GetValue();
    const bool use_nvdec = nvdec_value != Settings::NvdecEmulation::Off;
    const bool use_async = Settings::values.use_asynchronous_gpu_emulation.GetValue();
    gpu.emplace(system, use_async, use_nvdec);
    auto context = emu_window.CreateSharedContext();
    auto scope = context->Acquire();
    try {
        auto renderer = CreateRenderer(system, emu_window, *gpu, std::move(context));
        gpu->BindRenderer(std::move(renderer));
    } catch (const std::runtime_error& exception) {
        scope.Cancel();
        LOG_ERROR(HW_GPU, "Failed to initialize GPU: {}", exception.what());
        gpu.reset();
    }
}

} // namespace VideoCore
