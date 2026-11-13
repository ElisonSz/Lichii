// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename T, typename Id = u32>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    void Invalidate() noexcept {
        // Cleared on the next read: the texture cache invalidates every channel's tables for
        // each image it deletes, dozens of times a frame while its collector runs.
        invalidated = true;
        ++changes;
        ++layout_changes;
    }

    /// EDEN_NX2_CFG_TIC: what Invalidate becomes for a deleted image view. The base resolves every
    /// entry again at its next read; here ReadImageEntry resolves again those that can resolve
    /// differently. A full invalidation only when the page table changed since the read bits were
    /// last cleared in full (`map_serial`: the channel's MapSerial now). Returns true for a full
    /// one. Either way the state other readers see follows the base: `changes` counts it, and
    /// `deleted_since_read` stands for `invalidated` until the next read.
    bool InvalidateViews(u64 map_serial) noexcept {
        if (!invalidated && clear_map_serial != UNKNOWN_MAP_SERIAL &&
            clear_map_serial != map_serial) {
            Invalidate();
            return true;
        }
        deleted_since_read = true;
        ++changes;
        return false;
    }

    std::span<Id> LiveBoundIds() noexcept {
        return std::span{bound_ids}.first(
            (std::min)(bound_ids.size(), size_t{current_limit} + 1));
    }

    template <class MemoryManager>
    [[nodiscard]] std::pair<T, bool> Read(const MemoryManager& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        std::pair<T, bool> result;
        // This runs for every texture and sampler of every draw. An entry within one page is
        // contiguous in host memory, so a single page lookup replaces the generic block read.
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    /// EDEN_NX2_CFG_TIC: how ReadImageEntry treated an entry read again with the same bytes.
    enum class EntryRead : u8 {
        Plain,    ///< as Read: not read since the last clear, other bytes, or no view deleted since
        Kept,     ///< a view was deleted since its last read, and the entry keeps its view
        Resolved, ///< a view was deleted since its last read, and the entry is to be resolved again
    };

    /// EDEN_NX2_CFG_TIC: Read for an image table, `deletions` being the texture cache's count of
    /// view deletions so far and `died_after(id, stamp)` whether view `id` was deleted after the
    /// count was `stamp`. The base resolves an entry again at its first read after any view
    /// deletion (Invalidate). Of those entries, with their bytes unchanged, this resolves again
    /// only those whose view was deleted since their last read, those bound to nothing (a failed
    /// creation is tried again, as in the base) and, once the page table changed since the read
    /// bits were last cleared in full, all; the others count as unchanged (EntryRead::Kept).
    /// Resolving one of those again would bind its view again: the same bytes give the same
    /// texture header, the page table (IsValidEntry) is the same, and the header's image_views
    /// entry, which bound the view, is removed only with the view.
    template <class MemoryManager, class DiedAfter>
    [[nodiscard]] std::pair<T, bool> ReadImageEntry(const MemoryManager& gpu_memory, u32 index,
                                                    u32 deletions, DiedAfter&& died_after,
                                                    EntryRead& how) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        deleted_since_read = false;
        if (clear_map_serial == UNKNOWN_MAP_SERIAL) {
            clear_map_serial = gpu_memory.MapSerial();
        }
        if (stamps.size() != descriptors.size()) {
            // Values only count for entries read since the last clear.
            stamps.resize(descriptors.size());
        }
        how = EntryRead::Plain;
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        std::pair<T, bool> result;
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
            if (!result.second && stamps[index] != deletions) {
                const Id id = bound_ids[index];
                const bool again = id == Id{0} || died_after(id, stamps[index]) ||
                                   gpu_memory.MapSerial() != clear_map_serial;
                how = again ? EntryRead::Resolved : EntryRead::Kept;
                result.second = again;
            }
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            result.second = true;
        }
        stamps[index] = deletions;
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    /// EDEN_NX2_CFG_TIC: whether entry `index` holds what Read left there and no view was deleted
    /// since (the base: read since `invalidated` was last cleared, and not invalidated since). A
    /// deletion counts before the tables hear of it, so it leaves no stamp equal to the count.
    [[nodiscard]] bool HoldsReadEntry(u32 index, u32 deletions) const noexcept {
        return !invalidated && (read_descriptors[index / 64] & (1ULL << (index % 64))) != 0 &&
               index < stamps.size() && stamps[index] == deletions;
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        ++changes;
        ++layout_changes;
        deleted_since_read = false;
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        ClearReadDescriptors();
        invalidated = false;
        //
        descriptors.resize(num_descriptors);
        bound_ids.resize(num_descriptors, Id{0});
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    std::vector<Id> bound_ids;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    bool invalidated{}; ///< read_descriptors is to be cleared before the next read
    /// Bumped by every Invalidate and Refresh: the ids and descriptors read before are only good
    /// while it stays the same.
    u64 changes{};

    /// EDEN_NX2_CFG_TIC (image tables only): no MapSerial read since the last full clear.
    static constexpr u64 UNKNOWN_MAP_SERIAL = ~u64{0};
    /// Per entry, the texture cache's view deletion count at its last ReadImageEntry.
    std::vector<u32> stamps;
    /// The channel's MapSerial at the first ReadImageEntry after the read bits were last cleared
    /// in full: every entry read since was resolved with that page table or a later one.
    u64 clear_map_serial = UNKNOWN_MAP_SERIAL;
    /// Bumped by Refresh and by full invalidations (Invalidate), not by InvalidateViews.
    u64 layout_changes{};
    /// A view was deleted since the last read (where the base has `invalidated`).
    bool deleted_since_read{};

private:
    void ClearReadDescriptors() noexcept {
        clear_map_serial = UNKNOWN_MAP_SERIAL;
        // Reserved capacity is not addressable. A later limit increase clears its new range.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        std::fill_n(read_descriptors.begin(), words, u64{0});
        const size_t ids = (std::min)(bound_ids.size(), size_t{current_limit} + 1);
        std::fill_n(bound_ids.begin(), ids, Id{0});
    }
};

} // namespace VideoCommon
