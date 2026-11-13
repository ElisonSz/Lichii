// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <string>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace VideoCommon::GpuTrace {

struct ReplayOptions {
    std::filesystem::path trace;
    /// Gets frame-NNNN.png per present and frames.json.
    std::filesystem::path out;
    /// Stop after this many presents, 0 for all.
    u32 max_frames = 0;
};

/// Replays a GPU trace without a window: restores the snapshot, feeds the recorded records
/// through the GPU in order and captures every present. `system` must not be initialized; the
/// replay initializes it from the trace and shuts it down. See docs/nx2/gpu-trace.md.
[[nodiscard]] bool Replay(Core::System& system, const ReplayOptions& options, std::string& error);

} // namespace VideoCommon::GpuTrace
