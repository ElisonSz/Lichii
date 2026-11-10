// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <mutex>
#include "common/logging.h"

namespace Vulkan {

// TEST: sums measure time on all workers, not elapsed wall time.
enum class PipelineOrigin : size_t { Evicted, GraphicsCache, UnclaimedPrebuild, Compute, Count };

struct PipelineDestroyTime {
    using Clock = std::chrono::steady_clock;
    double pipelines_ms{};
    double other_ms{};
    double cpu_ms{};

    double Total() const {
        return pipelines_ms + other_ms + cpu_ms;
    }

    static double Milliseconds(Clock::time_point begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }
};

struct PipelineDestroyStats {
    struct Group {
        size_t count{};
        PipelineDestroyTime sum;
        double slowest_ms{};
    };
    mutable std::mutex mutex;
    std::array<Group, static_cast<size_t>(PipelineOrigin::Count)> groups{};

    void Add(PipelineOrigin origin, const PipelineDestroyTime& time) {
        std::scoped_lock lock{mutex};
        auto& group = groups[static_cast<size_t>(origin)];
        ++group.count;
        group.sum.pipelines_ms += time.pipelines_ms;
        group.sum.other_ms += time.other_ms;
        group.sum.cpu_ms += time.cpu_ms;
        group.slowest_ms = std::max(group.slowest_ms, time.Total());
    }

    void Report() const {
        std::scoped_lock lock{mutex};
        constexpr std::array names{"evicted", "graphics cache", "unclaimed prebuild", "compute"};
        Group total;
        for (size_t index = 0; index < groups.size(); ++index) {
            const auto& group = groups[index];
            LOG_INFO(Render_Vulkan,
                     "Pipeline destroy parts: {} count {}, vkDestroyPipeline sum {:.3f} ms, "
                     "other Vulkan sum {:.3f} ms, CPU frees sum {:.3f} ms, slowest {:.3f} ms",
                     names[index], group.count, group.sum.pipelines_ms, group.sum.other_ms,
                     group.sum.cpu_ms, group.slowest_ms);
            total.count += group.count;
            total.sum.pipelines_ms += group.sum.pipelines_ms;
            total.sum.other_ms += group.sum.other_ms;
            total.sum.cpu_ms += group.sum.cpu_ms;
            total.slowest_ms = std::max(total.slowest_ms, group.slowest_ms);
        }
        LOG_INFO(Render_Vulkan,
                 "Pipeline destroy parts: all count {}, vkDestroyPipeline sum {:.3f} ms, "
                 "other Vulkan sum {:.3f} ms, CPU frees sum {:.3f} ms, slowest {:.3f} ms",
                 total.count, total.sum.pipelines_ms, total.sum.other_ms, total.sum.cpu_ms,
                 total.slowest_ms);
    }
};

} // namespace Vulkan
