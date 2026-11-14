// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>

#include "common/logging.h"

namespace VideoCommon {

// TEST: markers precede members so they report after those members' automatic destruction.
// Activate only in the owner's destructor body; partially constructed objects report nothing.
class TeardownClock {
public:
    void Start() {
        begin = std::chrono::steady_clock::now();
        active = true;
    }

    void Step(const char* name) {
        if (!active) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        LOG_INFO(Render_Vulkan, "Renderer teardown: {} took {:.1f} ms", name,
                 std::chrono::duration<double, std::milli>(now - begin).count());
        begin = now;
    }

private:
    std::chrono::steady_clock::time_point begin{};
    bool active{};
};

struct TeardownMarker {
    ~TeardownMarker() {
        clock.Step(name);
    }
    TeardownClock& clock;
    const char* name;
};

} // namespace VideoCommon
