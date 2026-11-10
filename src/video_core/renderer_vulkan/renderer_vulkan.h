// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>
#include <variant>

#include "common/dynamic_library.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"
#include "video_core/teardown_timing.h"
#ifdef HAS_LSFG
#include "video_core/renderer_vulkan/present/frame_gen.h"
#endif
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_turbo_mode.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core::Memory {
class Memory;
}

namespace Tegra {
class GPU;
}

namespace Vulkan {

Device CreateDevice(const vk::Instance& instance, const vk::InstanceDispatch& dld,
                    VkSurfaceKHR surface);

class RendererVulkan final : public VideoCore::RendererBase {
public:
    explicit RendererVulkan(Core::Frontend::EmuWindow& emu_window,
                            Tegra::MaxwellDeviceMemoryManager& device_memory_, Tegra::GPU& gpu_,
                            std::unique_ptr<Core::Frontend::GraphicsContext> context_);
    ~RendererVulkan() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return &rasterizer;
    }

    [[nodiscard]] std::string GetDeviceVendor() const override {
        return device.GetDriverName();
    }

    [[nodiscard]] VideoCore::FrameGenStatus GetFrameGenStatus() const override {
#ifdef HAS_LSFG
        return frame_gen.CurrentStatus();
#else
        return {.state = VideoCore::FrameGenState::UnsupportedBuild};
#endif
    }

    // Enhanced platform-specific initialization
    void InitializePlatformSpecific();

private:
    Common::DkbFrameTrace::RateLimit trace_cpu_rate;
    std::chrono::steady_clock::time_point trace_last_frame{};
    u64 trace_frame{};

    void InterpolateFrames(Frame* prev_frame, Frame* curr_frame);
    Frame* previous_frame = nullptr;  // Store the previous frame for interpolation
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer command_buffer);
    void Report() const;

    vk::Buffer RenderToBuffer(std::span<const Tegra::FramebufferConfig> framebuffers,
                              const Layout::FramebufferLayout& layout, VkFormat format,
                              VkDeviceSize buffer_size);
    void RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers);
    void RenderAppletCaptureLayer(std::span<const Tegra::FramebufferConfig> framebuffers);

    std::vector<Tegra::FramebufferConfig> applet_capture_layers;
    std::vector<Tegra::FramebufferConfig> screenshot_layer_scratch;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Tegra::GPU& gpu;

    VideoCommon::TeardownClock teardown_clock;
    VideoCommon::TeardownMarker teardown_library{teardown_clock, "driver owner handoff or join"};
    struct DriverOwner;
    struct DriverOwnerDeleter {
        void operator()(DriverOwner* owner) const;
    };
    static DriverOwner* CreateDriverOwner(Core::Frontend::EmuWindow& window,
                                          Core::Frontend::GraphicsContext* graphics_context);
    std::unique_ptr<DriverOwner, DriverOwnerDeleter> driver_owner;
    std::shared_ptr<Common::DynamicLibrary>& library;
    vk::InstanceDispatch& dld;

    // Keep original handles for compatibility with existing code
    vk::Instance& instance;
    vk::DebugUtilsMessenger& debug_messenger;
    vk::SurfaceKHR& surface;
    Device& device;
    VideoCommon::TeardownMarker teardown_memory{teardown_clock, "renderer state"};
    MemoryAllocator& memory_allocator;
    StateTracker state_tracker;
    VideoCommon::TeardownMarker teardown_scheduler{teardown_clock, "command scheduler"};
    Scheduler scheduler;
    VideoCommon::TeardownMarker teardown_swapchain{teardown_clock, "swapchain"};
    Swapchain swapchain;
    VideoCommon::TeardownMarker teardown_present{teardown_clock, "present manager"};
    PresentManager present_manager;
    VideoCommon::TeardownMarker teardown_blits{teardown_clock, "screen blitters"};
    BlitScreen blit_swapchain;
    BlitScreen blit_capture;
    BlitScreen blit_applet;
    VideoCommon::TeardownMarker teardown_rasterizer{teardown_clock, "rasterizer"};
    RasterizerVulkan rasterizer;
    VideoCommon::TeardownMarker teardown_tail{teardown_clock, "applet frame, turbo and frame gen"};
#ifdef HAS_LSFG
    FrameGen frame_gen;
#endif
    std::optional<TurboMode> turbo_mode;

    Frame applet_frame;
};

} // namespace Vulkan
