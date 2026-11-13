// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstddef>
#include <string>
#include <fmt/format.h>

#include "dynarmic/backend/x64/perf_map.h"
#include "common/common_types.h"
#if defined(__linux__) && !defined(__ANDROID__)
#    include <cstdio>
#    include <cstdlib>
#    include <mutex>
#    include <sys/types.h>
#    include <unistd.h>

namespace Dynarmic::Backend::X64 {

namespace {
std::mutex mutex;
std::FILE* file = nullptr;

void OpenFile() {
    const char* perf_dir = std::getenv("PERF_BUILDID_DIR");
    if (!perf_dir) {
        file = nullptr;
        return;
    }

    const pid_t pid = getpid();
    const std::string filename = fmt::format("{:s}/perf-{:d}.map", perf_dir, pid);

    file = std::fopen(filename.c_str(), "w");
    if (!file) {
        return;
    }

    std::setvbuf(file, nullptr, _IONBF, 0);
}
}  // anonymous namespace

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (start == end) {
        // Nothing to register
        return;
    }

    std::lock_guard guard{mutex};

    if (!file) {
        OpenFile();
        if (!file) {
            return;
        }
    }

    const std::string line = fmt::format("{:016x} {:016x} {:s}\n", reinterpret_cast<u64>(start), reinterpret_cast<u64>(end) - reinterpret_cast<u64>(start), friendly_name);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
}
}  // namespace detail

void PerfMapClear() {
    std::lock_guard guard{mutex};

    if (!file) {
        return;
    }

    std::fclose(file);
    file = nullptr;
    OpenFile();
}

}  // namespace Dynarmic::Backend::X64

#endif

#if defined(_WIN32)
#    include <cstdio>
#    include <cstdlib>
#    include <mutex>

#    include <process.h>

namespace Dynarmic::Backend::X64 {

// EDEN_NX2_JIT_MAP=<folder> (diagnostics): every compiled block as "start size name" in
// <folder>/jit-map-<pid>.txt, like Linux's perf map, so a sampler's JIT addresses can be mapped
// to the guest code they run. A cache clear writes a "# clear <n>" line instead of starting over:
// samples taken before it then point at the blocks above it.
namespace {
std::mutex mutex;
std::FILE* file = nullptr;
bool opened = false;
unsigned clears = 0;

void OpenFile() {
    opened = true;
    const char* const folder = std::getenv("EDEN_NX2_JIT_MAP");
    if (folder == nullptr || folder[0] == '\0') {
        return;
    }
    const std::string filename = fmt::format("{:s}/jit-map-{:d}.txt", folder, _getpid());
    file = std::fopen(filename.c_str(), "w");
}
}  // anonymous namespace

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (start == end) {
        return;
    }
    std::lock_guard guard{mutex};
    if (!opened) {
        OpenFile();
    }
    if (!file) {
        return;
    }
    const std::string line = fmt::format("{:016x} {:016x} {:s}\n", reinterpret_cast<u64>(start),
                                         reinterpret_cast<u64>(end) - reinterpret_cast<u64>(start),
                                         friendly_name);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
    std::fflush(file);
}
}  // namespace detail

void PerfMapClear() {
    std::lock_guard guard{mutex};
    if (!file) {
        return;
    }
    const std::string line = fmt::format("# clear {:d}\n", ++clears);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
    std::fflush(file);
}

}  // namespace Dynarmic::Backend::X64

#endif
