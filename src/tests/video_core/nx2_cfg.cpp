// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/lru_cache.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/nx2_cfg.h"

namespace {

namespace Cfg = VideoCommon::Cfg;

/// Sets a Track C switch until the end of the scope.
class SwitchScope {
public:
    SwitchScope(bool& flag_, bool value) : flag{flag_}, previous{flag_} {
        flag = value;
    }
    ~SwitchScope() {
        flag = previous;
    }
    SwitchScope(const SwitchScope&) = delete;
    SwitchScope& operator=(const SwitchScope&) = delete;

private:
    bool& flag;
    bool previous;
};

struct LruTraits {
    using ObjectType = u32;
    using TickType = u64;
};
using Lru = Common::LeastRecentlyUsedCache<LruTraits>;

/// An object of a cache: its LRU item while registered, and its memo.
struct Object {
    bool registered{};
    size_t lru_id = (std::numeric_limits<size_t>::max)();
    Cfg::TouchMemo memo;
};

/// The objects of the items, least recently used first.
std::vector<u32> Contents(Lru& lru) {
    std::vector<u32> contents;
    lru.ForEachItemBelow(static_cast<u64>((std::numeric_limits<s64>::max)()),
                         [&](u32 object) { contents.push_back(object); });
    return contents;
}

} // Anonymous namespace

TEST_CASE("Cfg: skipped LRU touches leave the cache as every touch does", "[video_core][nx2_cfg]") {
    SwitchScope touch{Cfg::g_touch, true};
    SwitchScope verify{Cfg::g_verify, true};
    std::mt19937_64 rng{0xC0FFEEULL};
    u64 skipped_total = 0;
    for (int run = 0; run < 200; ++run) {
        Lru base;
        Lru skipping;
        constexpr u32 NumObjects = 24;
        std::array<Object, NumObjects> base_objects{};
        std::array<Object, NumObjects> objects{};
        u64 tick = 0;
        const u64 mismatches_before = Cfg::g_mismatches_total.load();
        for (int step = 0; step < 2000; ++step) {
            const u32 object = static_cast<u32>(rng() % NumObjects);
            switch (rng() % 16) {
            case 0: // A frame ends.
                ++tick;
                break;
            case 1: // Register (Insert reuses freed items first, as both caches do).
                if (!objects[object].registered) {
                    base_objects[object].lru_id = base.Insert(object, tick);
                    objects[object].lru_id = skipping.Insert(object, tick);
                    REQUIRE(base_objects[object].lru_id == objects[object].lru_id);
                    base_objects[object].registered = objects[object].registered = true;
                }
                break;
            case 2: // Unregister: the object keeps its stale id and memo.
                if (objects[object].registered) {
                    base.Free(base_objects[object].lru_id);
                    skipping.Free(objects[object].lru_id);
                    base_objects[object].registered = objects[object].registered = false;
                }
                break;
            case 3: { // Garbage collection of items older than a cutoff.
                const u64 cutoff = tick >= 2 ? tick - 2 : 0;
                std::vector<u32> collected;
                base.ForEachItemBelow(cutoff, [&](u32 id) { collected.push_back(id); });
                std::vector<u32> collected_skipping;
                skipping.ForEachItemBelow(cutoff,
                                          [&](u32 id) { collected_skipping.push_back(id); });
                REQUIRE(collected == collected_skipping);
                for (const u32 id : collected) {
                    base.Free(base_objects[id].lru_id);
                    skipping.Free(objects[id].lru_id);
                    base_objects[id].registered = objects[id].registered = false;
                }
                break;
            }
            default: // A draw binds or prepares it.
                if (objects[object].registered) {
                    base.Touch(base_objects[object].lru_id, tick);
                    const size_t id = objects[object].lru_id;
                    if (Cfg::SkipTouch(
                            objects[object].memo, id, tick, Cfg::Counter::BufferTouchesSkipped,
                            Cfg::Counter::BufferTouches, [&] { return skipping.GetTick(id); })) {
                        ++skipped_total;
                    } else {
                        skipping.Touch(id, tick);
                    }
                }
                break;
            }
            REQUIRE(Contents(base) == Contents(skipping));
            for (u32 index = 0; index < NumObjects; ++index) {
                if (objects[index].registered) {
                    REQUIRE(base.GetTick(base_objects[index].lru_id) ==
                            skipping.GetTick(objects[index].lru_id));
                }
            }
        }
        REQUIRE(Cfg::g_mismatches_total.load() == mismatches_before);
    }
    // Both outcomes occur often: a touch at a new tick or of a new item, and a repeated one.
    REQUIRE(skipped_total > 5000U);
}

TEST_CASE("Cfg: a memo that claims a touch the item lacks fails verify", "[video_core][nx2_cfg]") {
    SwitchScope touch{Cfg::g_touch, true};
    SwitchScope verify{Cfg::g_verify, true};
    Lru lru;
    const size_t id = lru.Insert(1, 3);
    Cfg::TouchMemo memo{.id = id, .tick = 5}; // claims a touch at 5; the item is at 3
    const u64 before = Cfg::g_mismatches_total.load();
    REQUIRE_FALSE(Cfg::SkipTouch(memo, id, 5, Cfg::Counter::ImageTouchesSkipped,
                                 Cfg::Counter::ImageTouches, [&] { return lru.GetTick(id); }));
    REQUIRE(Cfg::g_mismatches_total.load() == before + 1U);
    // Another item or tick is never skipped.
    REQUIRE_FALSE(Cfg::SkipTouch(memo, id + 1, 5, Cfg::Counter::ImageTouchesSkipped,
                                 Cfg::Counter::ImageTouches, [] { return u64{9}; }));
    REQUIRE_FALSE(Cfg::SkipTouch(memo, id + 1, 6, Cfg::Counter::ImageTouchesSkipped,
                                 Cfg::Counter::ImageTouches, [] { return u64{9}; }));
    REQUIRE(Cfg::SkipTouch(memo, id + 1, 6, Cfg::Counter::ImageTouchesSkipped,
                           Cfg::Counter::ImageTouches, [] { return u64{9}; }));
}

namespace {

/// A key compared as GraphicsPipelineCacheKey is: the shader hashes, then Size() bytes of the
/// state, where Size() depends on flags in the state's first word and is never below 16.
struct MockState {
    std::array<u64, 8> words;

    [[nodiscard]] size_t Size() const noexcept {
        switch (words[0] & 3) {
        case 0:
            return 16;
        case 1:
            return 24;
        case 2:
            return 40;
        default:
            return sizeof(words);
        }
    }
};

struct MockKey {
    std::array<u64, 6> unique_hashes;
    MockState state;

    bool operator==(const MockKey& rhs) const noexcept {
        if (unique_hashes != rhs.unique_hashes) {
            return false;
        }
        return std::memcmp(&state, &rhs.state, state.Size()) == 0;
    }
};

MockKey RandomKey(std::mt19937_64& rng) {
    // Few distinct values per field, so lists hold many keys sharing hashes and prefixes, and
    // keys that differ only past 16 bytes (equal or not, depending on Size()).
    MockKey key{};
    const u64 shader_set = rng() % 3;
    for (size_t index = 0; index < key.unique_hashes.size(); ++index) {
        key.unique_hashes[index] = (shader_set + index) * 0x1000;
    }
    if (rng() % 8 == 0) {
        key.unique_hashes[rng() % 6] ^= 1;
    }
    for (size_t index = 0; index < key.state.words.size(); ++index) {
        key.state.words[index] = rng() % (index < 2 ? 4 : 2);
    }
    return key;
}

} // Anonymous namespace

TEST_CASE("Cfg: equal keys have equal digests", "[video_core][nx2_cfg]") {
    std::mt19937_64 rng{7};
    u64 equal_pairs = 0;
    for (int iteration = 0; iteration < 200000; ++iteration) {
        const MockKey lhs = RandomKey(rng);
        MockKey rhs = lhs;
        // Bytes the comparison does not look at may differ.
        for (size_t word = (lhs.state.Size() + 7) / 8; word < rhs.state.words.size(); ++word) {
            rhs.state.words[word] = rng();
        }
        REQUIRE(lhs == rhs);
        REQUIRE(Cfg::KeyDigest(lhs) == Cfg::KeyDigest(rhs));
        const MockKey other = RandomKey(rng);
        if (lhs == other) {
            ++equal_pairs;
            REQUIRE(Cfg::KeyDigest(lhs) == Cfg::KeyDigest(other));
        }
    }
    REQUIRE(equal_pairs > 0U);
}

TEST_CASE("Cfg: the digest search finds what a linear search finds", "[video_core][nx2_cfg]") {
    std::mt19937_64 rng{11};
    u64 found = 0;
    u64 missing = 0;
    u64 compares_total = 0;
    u64 scanned_total = 0;
    for (int run = 0; run < 2000; ++run) {
        std::vector<MockKey> keys;
        std::vector<u64> digests;
        const size_t count = rng() % 64;
        for (size_t index = 0; index < count; ++index) {
            keys.push_back(RandomKey(rng));
            digests.push_back(Cfg::KeyDigest(keys.back()));
        }
        for (int query = 0; query < 64; ++query) {
            const MockKey key =
                !keys.empty() && rng() % 2 == 0 ? keys[rng() % keys.size()] : RandomKey(rng);
            u64 compares = 0;
            const size_t index = Cfg::FindByDigest(
                std::span<const u64>(digests), Cfg::KeyDigest(key),
                [&](size_t candidate) { return keys[candidate] == key; }, compares);
            const size_t expected =
                static_cast<size_t>(std::ranges::find(keys, key) - keys.begin());
            REQUIRE(index == expected);
            REQUIRE(compares <= (index < keys.size() ? index + 1 : keys.size()));
            if (index < keys.size()) {
                ++found;
            } else {
                ++missing;
            }
            compares_total += compares;
            scanned_total += index < keys.size() ? index + 1 : keys.size();
        }
    }
    REQUIRE(found > 1000U);
    REQUIRE(missing > 1000U);
    // The digests spare most whole key compares.
    REQUIRE(compares_total * 2 < scanned_total);
}

namespace {

/// GPU pages mapped to device pages (or not), as VirtualRangeCache::Query reads them.
struct MockMemory {
    static constexpr u64 Base = 0x10000000;
    static constexpr u64 PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;
    std::vector<std::optional<u64>> pages;

    [[nodiscard]] size_t GetID() const noexcept {
        return 1;
    }

    [[nodiscard]] std::optional<u64> DevicePage(u64 gpu_addr) const {
        if (gpu_addr < Base || ((gpu_addr - Base) >> PageBits) >= pages.size()) {
            return std::nullopt;
        }
        return pages[(gpu_addr - Base) >> PageBits];
    }

    [[nodiscard]] std::optional<u64> GpuToCpuAddress(u64 gpu_addr) const {
        const std::optional<u64> page = DevicePage(gpu_addr);
        if (!page) {
            return std::nullopt;
        }
        return (*page << PageBits) | (gpu_addr & (PageSize - 1));
    }

    /// The mapped parts of the range, split where the device pages stop following each other.
    [[nodiscard]] boost::container::small_vector<std::pair<u64, std::size_t>, 32> GetSubmappedRange(
        u64 gpu_addr, std::size_t size) const {
        boost::container::small_vector<std::pair<u64, std::size_t>, 32> result;
        u64 address = gpu_addr;
        const u64 end = gpu_addr + size;
        std::optional<u64> previous;
        while (address < end) {
            const u64 next = (std::min)(end, (address & ~(PageSize - 1)) + PageSize);
            const std::optional<u64> page = DevicePage(address);
            if (page) {
                if (previous && *previous + 1 == *page && !result.empty() &&
                    result.back().first + result.back().second == address) {
                    result.back().second += next - address;
                } else {
                    result.emplace_back(address, next - address);
                }
            }
            previous = page;
            address = next;
        }
        return result;
    }
};

struct RangeMemo {
    u64 gpu_addr{};
    u32 size{};
    u64 generation{};
    bool valid{};
};

} // Anonymous namespace

TEST_CASE("Cfg: a range query known in one piece repeats without changing the cache",
          "[video_core][nx2_cfg]") {
    std::mt19937_64 rng{23};
    u64 known_total = 0;
    u64 queries_total = 0;
    for (int run = 0; run < 200; ++run) {
        MockMemory memory;
        memory.pages.resize(64);
        for (size_t page = 0; page < memory.pages.size(); ++page) {
            if (rng() % 8 != 0) {
                memory.pages[page] = rng() % 4 == 0 ? rng() % 1024 + 1 : page + 100;
            }
        }
        VideoCommon::VirtualRangeCache cache;
        std::array<RangeMemo, 6> memos{};
        for (int step = 0; step < 3000; ++step) {
            const u64 kind = rng() % 64;
            if (kind == 0) {
                // A page mapped or remapped without a notification, as Map does.
                memory.pages[rng() % memory.pages.size()] = rng() % 2 == 0 ? rng() % 1024 + 1 : 7;
                continue;
            }
            if (kind == 1) {
                // An unmap: the buffer cache tells the range cache (deferred to the next Query).
                const u64 page = rng() % memory.pages.size();
                const u64 count = rng() % 4 + 1;
                for (u64 index = page; index < (std::min)(page + count, u64{memory.pages.size()});
                     ++index) {
                    memory.pages[index].reset();
                }
                cache.Unmap(memory.GetID(), MockMemory::Base + page * MockMemory::PageSize,
                            count * MockMemory::PageSize);
                continue;
            }
            const size_t slot = static_cast<size_t>(rng() % memos.size());
            RangeMemo& memo = memos[slot];
            // Draws mostly bind a slot to the range it had; sometimes to another one.
            const bool again = memo.valid && rng() % 4 != 0;
            const u64 gpu_addr = again ? memo.gpu_addr
                                       : MockMemory::Base + (rng() % 8) * 3 * MockMemory::PageSize +
                                             (rng() % 2) * 0x40;
            const u32 size = again ? memo.size : static_cast<u32>((rng() % 5 + 1) * 0x900);
            const bool known = memo.valid && memo.gpu_addr == gpu_addr && memo.size == size &&
                               memo.generation == cache.Generation() && !cache.HasDeferred();
            const u64 generation_before = cache.Generation();
            const VideoCommon::VirtualSegments* const found = cache.Query(memory, gpu_addr, size);
            const bool single = found == nullptr || found->size() < 2;
            ++queries_total;
            if (known) {
                ++known_total;
                REQUIRE(single);
                REQUIRE(cache.Generation() == generation_before);
            }
            if (single) {
                memo = RangeMemo{.gpu_addr = gpu_addr,
                                 .size = size,
                                 .generation = cache.Generation(),
                                 .valid = true};
            } else {
                memo.valid = false;
            }
        }
    }
    // Both outcomes occur often.
    REQUIRE(known_total > 50000U);
    REQUIRE(known_total * 4 < queries_total * 3);
}
