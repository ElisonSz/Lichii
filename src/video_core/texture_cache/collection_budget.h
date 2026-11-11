// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace VideoCommon {

// Bound both CPU work and evictions. Protected candidates consume scan work, but must not
// prevent the collector from reaching eligible images behind them in the same LRU window.
struct CollectionBudget {
    size_t deletions{};
    size_t scans{};
    bool scan_past_skips{};

    void Configure(size_t count) {
        deletions = count;
        scans = count * 8;
    }

    bool Examine() {
        if (deletions == 0 || scans == 0) {
            return false;
        }
        --scans;
        if (!scan_past_skips) {
            --deletions;
        }
        return true;
    }

    void Deleted() {
        if (scan_past_skips) {
            --deletions;
        }
    }
};

} // namespace VideoCommon
