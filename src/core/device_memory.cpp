// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>

#include "common/logging.h"
#include "common/nx2_map_stats.h"
#include "common/thread.h"
#endif

#include "core/device_memory.h"
#include "hle/kernel/board/nintendo/nx/k_system_control.h"

namespace Core {

#ifdef HAS_NCE
constexpr size_t VirtualReserveSize = 1ULL << 38;
#else
constexpr size_t VirtualReserveSize = 1ULL << 39;
#endif

DeviceMemory::DeviceMemory()
    : buffer{Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize(),
             VirtualReserveSize} {
#ifdef _WIN32
    // EDEN_NX2_ZERO_TRIM=0 (comparisons): no trim passes, so pages the guest cleared stay in the
    // working set and a later touch of them takes no page fault.
    if (const char* const value = std::getenv("EDEN_NX2_ZERO_TRIM");
        value != nullptr && value[0] == '0') {
        LOG_INFO(HW_Memory, "EDEN_NX2_ZERO_TRIM=0: zero-page trimming is off");
        return;
    }
    zero_trim_thread = std::jthread{[this](std::stop_token stop) { TrimZeroPages(stop); }};
#endif
}

DeviceMemory::~DeviceMemory() = default;

#ifdef _WIN32
// Windows can't punch holes in the guest memory section, so pages holding only zeros (games clear
// their heaps while booting: Ocarina leaves ~7 GB) stay resident until they leave the working set.
// A pass comes every 5 s while they keep turning up, backing off to once every five minutes: in
// game a pass finds a few MiB but walks the whole working set (~1.3 s on this thread).
void DeviceMemory::TrimZeroPages(std::stop_token stop) {
    using namespace std::chrono_literals;
    Common::SetCurrentThreadName("ZeroTrim");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    constexpr size_t Busy = size_t{64} << 20;
    std::mutex mutex;
    std::condition_variable_any wake;
    std::chrono::seconds interval = 5s;
    while (true) {
        {
            std::unique_lock lock{mutex};
            wake.wait_for(lock, stop, interval, [] { return false; });
        }
        if (stop.stop_requested()) {
            return;
        }
        // A guest streaming memory in or out goes first (Ocarina of Time's route start maps and
        // unmaps ~800 MiB in ~12 s): a pass holds the kernel's working-set lock, which each of its
        // maps and unmaps needs, for most of a second. Look again after the same interval.
        constexpr u64 QuietNs = 5'000'000'000ULL;
        if (Common::HostMemory::NanosecondsSinceMapChange() < QuietNs) {
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        const auto [resident, trimmed] = buffer.TrimZeroPages();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (Common::Nx2MapStats::Enabled()) {
            LOG_INFO(HW_Memory, "Zero trim pass: {} MiB of {} MiB resident in {} ms",
                     trimmed >> 20, resident >> 20, ms);
        }
        if (Common::HostMemory::NanosecondsSinceMapChange() < QuietNs) {
            continue; // the pass gave way: no backing off
        }
        if (trimmed < Busy) {
            interval = (std::min)(interval * 2, std::chrono::seconds{300});
            continue;
        }
        interval = 5s;
        LOG_INFO(HW_Memory,
                 "Trimmed {} MiB of all-zero guest pages out of {} MiB resident in {} ms; the kernel "
                 "zero-filled {} MiB so far",
                 trimmed >> 20, resident >> 20, ms, Common::HostMemory::ZeroFilledBytes() >> 20);
    }
}
#endif

} // namespace Core
