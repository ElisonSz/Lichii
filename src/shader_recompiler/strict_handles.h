// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdlib>
#include <mutex>
#include <string_view>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"

namespace Shader {
/// Process-wide startup snapshot. Only the exact value 1 enables the experimental rules.
inline const bool StrictHandles = [] {
    const char* const setting = std::getenv("EDEN_NX2_STRICT_HANDLES");
    return setting != nullptr && std::string_view{setting} == "1";
}();
/// All strict fallback diagnostics share one line per shader across translation and execution.
inline void WarnStrictHandles(u64 shader_hash, const char* reason) {
    if (!StrictHandles)
        return;
    static std::mutex warning_mutex;
    static std::unordered_set<u64> warned;
    {
        std::scoped_lock lock{warning_mutex};
        if (!warned.insert(shader_hash).second)
            return;
    }
    LOG_WARNING(Shader, "Strict handles: shader {:016x}: {}", shader_hash, reason);
}
} // namespace Shader
