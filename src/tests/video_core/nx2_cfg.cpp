// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/nx2_cfg.h"
#include "video_core/texture_cache/descriptor_table.h"

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

namespace {

/// A texture header for the table tests: the address IsValidEntry looks at, and the rest.
struct MockTic {
    u64 address{};
    u64 format{};
    bool operator==(const MockTic&) const = default;
};

struct MockTicHash {
    size_t operator()(const MockTic& tic) const noexcept {
        return static_cast<size_t>(tic.address * 0x9E3779B97F4A7C15ULL ^ tic.format);
    }
};

using TicTable = VideoCommon::DescriptorTable<MockTic, Common::SlotId>;
constexpr Common::SlotId NULL_VIEW{0};
constexpr u64 TABLE_BASE = 0x40000;
constexpr u64 NUM_ADDRESSES = 12;
constexpr u64 NULL_FORMAT = 5; ///< CreateImageView gives no view for it (and no failure)

/// What the texture cache's VisitImageView reads and changes: guest memory with the table, the
/// page table, the channel's image_views map and the image view slots.
struct TicWorld {
    std::vector<MockTic> words;
    std::array<bool, NUM_ADDRESSES> mapped{};
    u64 map_serial = 1;
    std::vector<bool> alive{true}; ///< slot 0: the null view
    std::vector<u32> free_slots;   ///< reused last freed first, as Common::SlotVector does
    std::unordered_map<MockTic, Common::SlotId, MockTicHash> image_views;
    bool out_of_memory = false;

    [[nodiscard]] const u8* GetPointer(u64 gpu_addr) const {
        const u64 offset = gpu_addr - TABLE_BASE;
        if (gpu_addr < TABLE_BASE || offset / sizeof(MockTic) >= words.size()) {
            return nullptr;
        }
        return reinterpret_cast<const u8*>(words.data()) + offset;
    }
    void ReadBlockUnsafe(u64 gpu_addr, void* out, size_t size) const {
        std::memcpy(out, GetPointer(gpu_addr), size);
    }
    [[nodiscard]] u64 MapSerial() const {
        return map_serial;
    }
    [[nodiscard]] bool Valid(const MockTic& tic) const {
        return tic.address != 0 && mapped[tic.address % NUM_ADDRESSES];
    }
    Common::SlotId Create() {
        if (!free_slots.empty()) {
            const u32 slot = free_slots.back();
            free_slots.pop_back();
            alive[slot] = true;
            return Common::SlotId{slot};
        }
        alive.push_back(true);
        return Common::SlotId{static_cast<u32>(alive.size() - 1)};
    }
    /// Deletes a view: RemoveImageViewReferences, then the slot is freed.
    void Delete(Common::SlotId id) {
        std::erase_if(image_views, [id](const auto& entry) { return entry.second == id; });
        alive[id.Value()] = false;
        free_slots.push_back(id.Value());
    }
    [[nodiscard]] std::vector<u32> LiveViews() const {
        std::vector<u32> live;
        for (u32 slot = 1; slot < alive.size(); ++slot) {
            if (alive[slot]) {
                live.push_back(slot);
            }
        }
        return live;
    }
};

/// One channel's image table and its texture cache, with or without EDEN_NX2_CFG_TIC.
struct TicModel {
    bool per_entry{};
    TicWorld world;
    TicTable table;
    Cfg::ViewDeaths deaths;
    u64 kept = 0;
    u64 resolved_again = 0;
    u64 full = 0;

    /// The deletion of `slots` (DeleteImage or InvalidateScale): the views die, the tables hear.
    void DeleteViews(const std::vector<u32>& slots) {
        for (const u32 slot : slots) {
            world.Delete(Common::SlotId{slot});
        }
        if (!per_entry) {
            table.Invalidate();
            return;
        }
        if (deaths.Note(slots)) {
            table.Invalidate();
            return;
        }
        full += table.InvalidateViews(world.MapSerial()) ? 1 : 0;
    }

    /// VisitImageView. `prepare_deletes` stands for PrepareImageView deleting the view it
    /// prepares (scaling its image), before the entry is bound to it.
    Common::SlotId Visit(u32 index, bool prepare_deletes) {
        using EntryRead = TicTable::EntryRead;
        EntryRead how = EntryRead::Plain;
        const std::pair<MockTic, bool> read =
            per_entry ? table.ReadImageEntry(
                            world, index, deaths.Clock(),
                            [this](Common::SlotId id, u32 stamp) {
                                return deaths.DiedAfter(id.Value(), stamp);
                            },
                            how)
                      : table.Read(world, index);
        kept += how == EntryRead::Kept ? 1 : 0;
        resolved_again += how == EntryRead::Resolved ? 1 : 0;
        if (!read.second) {
            // Prepared on this path too (PrepareImageView of the bound view).
            const Common::SlotId bound = table.bound_ids[index];
            if (prepare_deletes && bound != NULL_VIEW) {
                DeleteViews({bound.Value()});
            }
            return bound;
        }
        const MockTic& tic = read.first;
        if (!world.Valid(tic)) {
            table.bound_ids[index] = NULL_VIEW;
            return NULL_VIEW;
        }
        const auto [it, inserted] = world.image_views.try_emplace(tic);
        if (inserted) {
            if (world.out_of_memory) {
                world.image_views.erase(it);
                table.bound_ids[index] = NULL_VIEW;
                return NULL_VIEW;
            }
            it->second = tic.format == NULL_FORMAT ? NULL_VIEW : world.Create();
        }
        const Common::SlotId view = it->second;
        if (prepare_deletes && view != NULL_VIEW) {
            DeleteViews({view.Value()});
        }
        table.bound_ids[index] = view;
        return view;
    }

    /// What CaptureDrawDeltaTextures takes as read: the base's cleared read bits and invalidated.
    [[nodiscard]] bool Holds(u32 index) const {
        if (per_entry) {
            return table.HoldsReadEntry(index, deaths.Clock());
        }
        return !table.invalidated &&
               (table.read_descriptors[index / 64] & (u64{1} << (index % 64))) != 0;
    }
};

MockTic RandomTic(std::mt19937_64& rng) {
    return MockTic{.address = rng() % (NUM_ADDRESSES + 2), .format = rng() % 6};
}

} // Anonymous namespace

TEST_CASE("Cfg: per-entry texture header invalidation binds what full invalidation binds",
          "[video_core][nx2_cfg]") {
    std::mt19937_64 rng{31};
    u64 reads = 0;
    u64 kept = 0;
    u64 resolved_again = 0;
    u64 full = 0;
    for (int run = 0; run < 300; ++run) {
        TicModel base;
        TicModel tic;
        tic.per_entry = true;
        if (run % 50 == 7) {
            // Close to the count's wrap.
            tic.deaths = Cfg::ViewDeaths{(std::numeric_limits<u32>::max)() - 40};
        }
        u32 limit = static_cast<u32>(rng() % 90 + 10);
        std::vector<MockTic> words(256);
        for (MockTic& word : words) {
            word = RandomTic(rng);
        }
        std::array<bool, NUM_ADDRESSES> mapped{};
        for (bool& address : mapped) {
            address = rng() % 4 != 0;
        }
        for (TicModel* model : {&base, &tic}) {
            model->world.words = words;
            model->world.mapped = mapped;
            (void)model->table.Synchronize(TABLE_BASE, limit);
        }
        for (int step = 0; step < 3000; ++step) {
            const u64 kind = rng() % 100;
            if (kind < 6) {
                // The game writes a header.
                const size_t index = static_cast<size_t>(rng() % (limit + 1));
                const MockTic word = RandomTic(rng);
                base.world.words[index] = word;
                tic.world.words[index] = word;
            } else if (kind < 16) {
                // Images deleted: one to three live views (the same in both worlds).
                const std::vector<u32> live = base.world.LiveViews();
                REQUIRE(live == tic.world.LiveViews());
                if (live.empty()) {
                    continue;
                }
                std::vector<u32> slots;
                const u64 count = rng() % 3 + 1;
                for (u64 i = 0; i < count; ++i) {
                    const u32 slot = live[static_cast<size_t>(rng() % live.size())];
                    if (std::ranges::find(slots, slot) == slots.end()) {
                        slots.push_back(slot);
                    }
                }
                base.DeleteViews(slots);
                tic.DeleteViews(slots);
            } else if (kind < 18) {
                // A map or unmap.
                const size_t address = static_cast<size_t>(rng() % NUM_ADDRESSES);
                for (TicModel* model : {&base, &tic}) {
                    model->world.mapped[address] = !model->world.mapped[address];
                    ++model->world.map_serial;
                }
            } else if (kind < 20) {
                const bool oom = rng() % 2 == 0;
                base.world.out_of_memory = oom;
                tic.world.out_of_memory = oom;
            } else if (kind == 20 && rng() % 8 == 0) {
                // The game moves or resizes the table.
                limit = static_cast<u32>(rng() % 90 + 10);
                (void)base.table.Synchronize(TABLE_BASE, limit);
                (void)tic.table.Synchronize(TABLE_BASE, limit);
            } else if (kind == 21) {
                // Out of memory at the end of a frame: TickFrame invalidates in full.
                base.table.Invalidate();
                tic.table.Invalidate();
            } else {
                const u32 index = static_cast<u32>(rng() % (limit + 1));
                const bool prepare_deletes = rng() % 64 == 0;
                const Common::SlotId expected = base.Visit(index, prepare_deletes);
                const Common::SlotId got = tic.Visit(index, prepare_deletes);
                ++reads;
                REQUIRE(got == expected);
            }
            REQUIRE(base.table.changes == tic.table.changes);
            for (u32 index = 0; index <= limit; ++index) {
                REQUIRE(base.Holds(index) == tic.Holds(index));
            }
        }
        REQUIRE(base.world.image_views == tic.world.image_views);
        kept += tic.kept;
        resolved_again += tic.resolved_again;
        full += tic.full;
    }
    // Every path ran many times.
    CAPTURE(kept, resolved_again, full, reads);
    REQUIRE(kept > 5000U);
    REQUIRE(resolved_again > 5000U);
    REQUIRE(full > 1000U);
    REQUIRE(reads > 500000U);
}

TEST_CASE("Cfg: view deaths order deletions and reads, also across the count's wrap",
          "[video_core][nx2_cfg]") {
    Cfg::ViewDeaths deaths{(std::numeric_limits<u32>::max)() - 2};
    const u32 read_before = deaths.Clock();
    REQUIRE_FALSE(deaths.Note(std::vector<u32>{3}));
    REQUIRE(deaths.DiedAfter(3, read_before));
    REQUIRE_FALSE(deaths.DiedAfter(3, deaths.Clock()));
    REQUIRE_FALSE(deaths.DiedAfter(4, read_before));
    REQUIRE_FALSE(deaths.DiedAfter(1000, read_before));
    REQUIRE_FALSE(deaths.Note(std::vector<u32>{4}));
    REQUIRE(deaths.Clock() == (std::numeric_limits<u32>::max)());
    // The next deletion starts the count again and says so: the caller invalidates in full.
    REQUIRE(deaths.Note(std::vector<u32>{5}));
    REQUIRE(deaths.Clock() == 2U);
    REQUIRE(deaths.DiedAfter(5, 1));
    REQUIRE_FALSE(deaths.DiedAfter(3, 1));
    REQUIRE_FALSE(deaths.DiedAfter(4, 1));
}
