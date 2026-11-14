// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/slot_vector.h"

namespace {
struct ConstructOnly {
    explicit ConstructOnly(u32 value_) : value{std::make_unique<u32>(value_)} {}
    ConstructOnly(ConstructOnly&&) noexcept = default;
    ConstructOnly& operator=(ConstructOnly&&) = delete;
    std::unique_ptr<u32> value;
};
static_assert(std::is_nothrow_move_constructible_v<ConstructOnly>);
static_assert(!std::is_move_assignable_v<ConstructOnly>);
} // namespace

TEST_CASE("SlotVector relocates nonassignable owners without losing ownership", "[common]") {
    Common::SlotVector<ConstructOnly> slots;
    std::vector<Common::SlotId> ids;
    for (u32 value = 0; value < 513; ++value) {
        ids.push_back(slots.insert(value));
    }
    for (u32 value = 0; value < 513; ++value) {
        REQUIRE(*slots[ids[value]].value == value);
    }
    slots.erase(ids[17]);
    const auto replacement = slots.insert(u32{900});
    REQUIRE(replacement == ids[17]);
    REQUIRE(*slots[replacement].value == 900);
    REQUIRE(slots.size() == 513);
}

TEST_CASE("SlotVector begins at the first live entry after empty bitmap words", "[common]") {
    Common::SlotVector<u32> slots;
    std::vector<Common::SlotId> ids;
    for (u32 i = 0; i < 513; ++i) {
        ids.push_back(slots.insert(i));
    }
    for (u32 i = 0; i < 512; ++i) {
        slots.erase(ids[i]);
    }
    auto it = slots.begin();
    REQUIRE(it != slots.end());
    REQUIRE((*it).first == ids[512]);
    REQUIRE(*(*it).second == 512);
    REQUIRE(++it == slots.end());
    slots.erase(ids[512]);
    REQUIRE(slots.begin() == slots.end());
}

TEST_CASE("SlotVector sparse iteration matches a live-entry model", "[common]") {
    Common::SlotVector<u32> slots;
    std::map<u32, u32> live;
    std::mt19937 random{9813};
    for (u32 batch = 0; batch < 200; ++batch) {
        for (u32 i = 0; i < 100; ++i) {
            if (!live.empty() && (random() % 3) == 0) {
                auto selected = live.begin();
                std::advance(selected, random() % live.size());
                slots.erase(Common::SlotId{selected->first});
                live.erase(selected);
            } else {
                const u32 value = random();
                const auto id = slots.insert(value);
                live.emplace(id.Value(), value);
            }
        }
        auto expected = live.begin();
        for (const auto [id, value] : slots) {
            REQUIRE(expected != live.end());
            REQUIRE(id.Value() == expected->first);
            REQUIRE(*value == expected->second);
            ++expected;
        }
        REQUIRE(expected == live.end());
    }
}
