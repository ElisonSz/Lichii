// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Nx2PresentStats {

/// EDEN_NX2_PRESENT_STATS=1 (diagnostic): the cadence of the host's swapchain presents, which a
/// variable refresh rate display shows as it is. The bench's frame times are the game's present
/// calls, paced by the guest's vblank emulation; these are when frames reach the display.
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_PRESENT_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// After each swapchain present, from one thread at a time (the present thread, or the GPU
/// thread without async presentation). Every 10 s: the intervals' mean and percentiles, how many
/// exceed a 60 Hz frame by 10 % and by 50 %, and a 1 ms histogram.
inline void NotePresent() {
    if (!Enabled()) [[likely]] {
        return;
    }
    using Clock = std::chrono::steady_clock;
    static Clock::time_point last_present{};
    static Clock::time_point window_start{};
    static std::vector<double> intervals;
    const Clock::time_point now = Clock::now();
    if (last_present == Clock::time_point{}) {
        window_start = now;
    } else {
        intervals.push_back(std::chrono::duration<double, std::milli>(now - last_present).count());
    }
    last_present = now;
    if (intervals.empty() || now - window_start < std::chrono::seconds{10}) {
        return;
    }
    std::vector<double> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&sorted](double fraction) {
        const auto index = static_cast<size_t>(fraction * static_cast<double>(sorted.size()));
        return sorted[(std::min)(index, sorted.size() - 1)];
    };
    double total_ms = 0.0;
    size_t over_10_percent = 0;
    size_t over_50_percent = 0;
    std::array<u32, 41> histogram{};
    for (const double interval_ms : intervals) {
        total_ms += interval_ms;
        over_10_percent += interval_ms > 18.33 ? 1 : 0;
        over_50_percent += interval_ms > 25.0 ? 1 : 0;
        ++histogram[(std::min)(static_cast<size_t>(interval_ms), histogram.size() - 1)];
    }
    std::string bins;
    for (size_t bin = 0; bin < histogram.size(); ++bin) {
        if (histogram[bin] != 0) {
            bins += fmt::format("{}{}:{}", bins.empty() ? "" : ",", bin, histogram[bin]);
        }
    }
    LOG_INFO(Render_Vulkan,
             "Host presents {:.1f} s: {} intervals, mean {:.2f} ms, p50 {:.2f}, p90 {:.2f}, p99 "
             "{:.2f}, max {:.2f}; over 18.3 ms {}, over 25 ms {}; hist {}",
             std::chrono::duration<double>(now - window_start).count(), intervals.size(),
             total_ms / static_cast<double>(intervals.size()), percentile(0.5), percentile(0.9),
             percentile(0.99), sorted.back(), over_10_percent, over_50_percent, bins);
    intervals.clear();
    window_start = now;
}

} // namespace VideoCommon::Nx2PresentStats
