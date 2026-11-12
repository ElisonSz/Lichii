// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>

#include "common/common_types.h"

namespace VideoCommon::MemTrack {

inline bool g_translate{};
inline bool g_word{};
inline bool g_verify{};
inline bool g_stats{};

// Called before the GPU threads start, after nx2_switches.txt has been loaded.
inline void ReadSwitch() {
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    g_translate = enabled("EDEN_NX2_MEMTRACK_TRANSLATE");
    g_word = enabled("EDEN_NX2_MEMTRACK_WORD");
    g_verify = enabled("EDEN_NX2_MEMTRACK_VERIFY");
    g_stats = enabled("EDEN_NX2_MEMTRACK_STATS");
}

// Device addresses only: host backing, protection and cached-page counts are never cached.
// One instance belongs to one thread. Map/unmap must have the same external synchronization as
// the page-table walk; a serial is an invalidation token, not a lock for a concurrent remap.
class TranslationCache {
public:
    static constexpr std::size_t NumEntries = 64;

    template <typename Resolve>
    std::optional<DAddr> Translate(std::size_t owner_id, u64 map_serial, GPUVAddr address,
                                   u64 page_bits, Resolve&& resolve, bool& hit) {
        if (owner != owner_id || serial != map_serial) {
            valid = 0;
            owner = owner_id;
            serial = map_serial;
        }
        const u64 page = address >> page_bits;
        const u64 offset = address & ((u64{1} << page_bits) - 1);
        const std::size_t slot = static_cast<std::size_t>((page ^ (page >> 6)) & (NumEntries - 1));
        const u64 bit = u64{1} << slot;
        auto& entry = entries[slot];
        hit = (valid & bit) != 0 && entry.page == page;
        if (hit) {
            return entry.device + offset;
        }
        // Resolve the actual address, retaining the original bounds checks and error logging.
        const auto device = resolve(address);
        if (device) {
            entry = {page, *device - offset};
            valid |= bit;
        }
        return device;
    }

    void Clear() {
        valid = 0;
    }

private:
    struct Entry {
        u64 page{};
        DAddr device{};
    };
    std::array<Entry, NumEntries> entries{};
    std::size_t owner{};
    u64 serial{};
    u64 valid{};
};

} // namespace VideoCommon::MemTrack
