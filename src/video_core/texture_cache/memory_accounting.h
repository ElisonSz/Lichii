// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"

namespace VideoCommon {

// Driver readings can lag allocations and omit allocator slack. Release both backing images
// with one saturating subtraction, even when the scaled allocation alone exceeds the reading.
constexpr u64 ReleaseImageMemory(u64 usage, u64 original, u64 scaled) {
    usage -= (std::min)(usage, scaled);
    return usage - (std::min)(usage, original);
}

} // namespace VideoCommon
