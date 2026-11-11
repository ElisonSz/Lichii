// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace Vulkan {

// Byte equality is mandatory: the hash only selects candidates. Payloads own immutable snapshots.
// A caller may evict only snapshots whose last GPU use completed, including pending captures.
template <typename Payload>
class AstcContentCache {
public:
    struct Entry {
        uint64_t hash;
        std::vector<uint64_t> layout;
        std::vector<uint8_t> compressed;
        uint64_t charge;
        Payload payload;
    };
    explicit AstcContentCache(uint64_t capacity_) : capacity{capacity_} {}

    Entry* Find(uint64_t hash, std::span<const uint64_t> layout,
                std::span<const uint8_t> compressed) {
        const auto found = std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
            return entry.hash == hash && std::ranges::equal(entry.layout, layout) &&
                   std::ranges::equal(entry.compressed, compressed);
        });
        if (found == entries.end()) {
            return nullptr;
        }
        entries.splice(entries.end(), entries, found);
        return &entries.back();
    }

    template <typename IsFree>
    bool MakeRoom(uint64_t charge, IsFree&& is_free) {
        if (charge > capacity) {
            return false;
        }
        for (auto entry = entries.begin();
             resident > capacity - charge && entry != entries.end();) {
            if (is_free(entry->payload)) {
                resident -= entry->charge;
                entry = entries.erase(entry);
            } else {
                ++entry;
            }
        }
        return resident <= capacity - charge;
    }

    void Insert(uint64_t hash, std::vector<uint64_t> layout, std::vector<uint8_t> compressed,
                uint64_t charge, Payload payload) {
        entries.push_back(
            {hash, std::move(layout), std::move(compressed), charge, std::move(payload)});
        resident += charge;
    }
    uint64_t Resident() const {
        return resident;
    }
    std::size_t Size() const {
        return entries.size();
    }

private:
    uint64_t capacity;
    uint64_t resident{};
    std::list<Entry> entries;
};

// A bounded sample of the first unique session contents. Existing observations remain useful
// after saturation; new unique contents are dropped explicitly rather than allocating forever.
template <typename Key, typename Value, std::size_t Capacity>
class AstcSeenContents {
public:
    const Value* Find(const Key& key) const {
        const auto found = entries.find(key);
        return found == entries.end() ? nullptr : &found->second;
    }
    bool Insert(const Key& key, const Value& value) {
        if (entries.size() == Capacity) {
            return false;
        }
        entries.try_emplace(key, value);
        return true;
    }
    std::size_t Size() const {
        return entries.size();
    }

private:
    std::map<Key, Value> entries;
};

// Both owned, mapped readbacks must exist before any commands reference either one. A skipped
// comparison leaves the caller's independent decode intact and never waits. Non-OOM exceptions
// from the allocator propagate normally. Cap is 32 MiB per readback (64 MiB total).
template <typename Allocate, typename Compare>
bool WithAstcVerificationBuffers(uint64_t bytes, Allocate&& allocate, Compare&& compare) {
    if (bytes == 0 || bytes > (uint64_t{32} << 20)) {
        return false;
    }
    auto reference = allocate();
    if (!reference || reference.Mapped().size() < bytes) {
        return false;
    }
    auto cached = allocate();
    if (!cached || cached.Mapped().size() < bytes) {
        return false;
    }
    compare(reference, cached);
    return true;
}

// Byte offset used by astc_decoder.comp, in a 2D block-linear mip/layer. Keeping padding out of
// the content key lets otherwise identical streams match when guest allocation padding changes.
inline uint64_t AstcBlockOffset(uint32_t x, uint32_t y, uint32_t block_size, uint32_t x_shift,
                                uint32_t block_height, uint32_t block_height_mask,
                                bool native_tiling) {
    const uint32_t byte_x = x * 16;
    uint32_t gob = ((byte_x & 32U) << 3U) | ((y & 6U) << 5U) | ((byte_x & 16U) << 1U) |
                   ((y & 1U) << 4U) | (byte_x & 15U);
    if (native_tiling) {
        const uint32_t difference = ((gob >> 5U) ^ (gob >> 6U)) & 1U;
        gob ^= (difference << 5U) | (difference << 6U);
    }
    const uint32_t block_y = y >> 3U;
    return uint64_t{block_y >> block_height} * block_size +
           (uint64_t{block_y & block_height_mask} << 9U) + (uint64_t{byte_x >> 6U} << x_shift) +
           gob;
}

struct AstcBlockLayout {
    uint64_t offset;
    uint32_t layers;
    uint32_t layer_stride;
    uint32_t width;
    uint32_t height;
    uint32_t block_size;
    uint32_t x_shift;
    uint32_t block_height;
    uint32_t block_height_mask;
    bool native_tiling;
};

// Append only the blocks the decoder reads, in level/layer/row order. Reject incomplete inputs.
inline bool AppendAstcBlocks(std::vector<uint8_t>& output, std::span<const uint8_t> input,
                             const AstcBlockLayout& block) {
    constexpr std::size_t limit = std::size_t{32} << 20;
    if (block.width == 0 || block.height == 0 || block.layers == 0 || block.block_height > 31 ||
        block.x_shift > 31 || block.offset > input.size()) {
        return false;
    }
    const uint64_t plane = uint64_t{block.width} * block.height;
    if (plane > limit / 16 || block.layers > limit / 16 / plane) {
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(plane * block.layers * 16);
    if (output.size() > limit - count) {
        return false;
    }
    const std::size_t begin = output.size();
    output.resize(begin + count);
    std::size_t destination = begin;
    for (uint32_t layer = 0; layer < block.layers; ++layer) {
        for (uint32_t y = 0; y < block.height; ++y) {
            for (uint32_t x = 0; x < block.width; ++x) {
                const uint64_t offset =
                    block.offset + uint64_t{layer} * block.layer_stride +
                    AstcBlockOffset(x, y, block.block_size, block.x_shift, block.block_height,
                                    block.block_height_mask, block.native_tiling);
                if (offset > input.size() || input.size() - offset < 16) {
                    output.resize(begin);
                    return false;
                }
                std::copy_n(input.data() + offset, 16, output.data() + destination);
                destination += 16;
            }
        }
    }
    return true;
}

} // namespace Vulkan
