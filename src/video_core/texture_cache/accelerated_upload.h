// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/stream_stats.h"

namespace VideoCommon {

// Upload/Stream mappings are write-only CPU destinations. When a content key is needed, keep
// the same guest read in ordinary memory and copy it to staging, never read staging back.
template <typename Read>
void ReadAcceleratedUpload(std::span<u8> staging, std::vector<u8>& cached, bool keep_cached,
                           Read&& read) {
    if (!keep_cached) {
        read(staging);
        return;
    }
    // Includes gathering the scratch bytes and copying them to the write-only staging map.
    // The guest read also has its own GuestReadNs scope at the call site.
    const StreamStats::Scope scope{StreamStats::Counter::KeyReadNs};
    cached.resize(staging.size());
    read(std::span{cached});
    std::copy(cached.begin(), cached.end(), staging.begin());
}

} // namespace VideoCommon
