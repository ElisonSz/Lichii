// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/memtrack_translation.h"

namespace {
constexpr u64 PageBits = 12;
constexpr u64 PageSize = u64{1} << PageBits;

// Independent two-level page-table model: a mapped big page wins, otherwise the small table is
// consulted, just as GpuToCpuAddress does (including fallback under a reserved big entry).
struct PageTable {
    enum class State { Free, Reserved, Mapped };
    struct Entry {
        State state{};
        DAddr device{};
    };
    static constexpr std::size_t SmallPages = 256;
    static constexpr std::size_t PagesPerBig = 16;
    std::array<Entry, SmallPages> small{};
    std::array<Entry, SmallPages / PagesPerBig> big{};
    u64 serial{};

    std::optional<DAddr> Walk(GPUVAddr address) const {
        if (address >= SmallPages * PageSize) {
            return std::nullopt;
        }
        const auto& big_entry = big[address / (PagesPerBig * PageSize)];
        if (big_entry.state == State::Mapped) {
            return big_entry.device + address % (PagesPerBig * PageSize);
        }
        const auto& small_entry = small[address / PageSize];
        if (small_entry.state != State::Mapped) {
            return std::nullopt;
        }
        return small_entry.device + address % PageSize;
    }
};
} // namespace

TEST_CASE("MemTrack translations reuse pages but not byte offsets", "[video_core][memtrack]") {
    VideoCommon::MemTrack::TranslationCache cache;
    bool hit{};
    u64 walks{};
    const auto resolve = [&](GPUVAddr address) -> std::optional<DAddr> {
        ++walks;
        return address + 0x800000;
    };
    REQUIRE(cache.Translate(0, 0, 0x1234, PageBits, resolve, hit) == 0x801234);
    REQUIRE_FALSE(hit);
    REQUIRE(cache.Translate(0, 0, 0x1fff, PageBits, resolve, hit) == 0x801fff);
    REQUIRE(hit);
    REQUIRE(cache.Translate(0, 0, 0x2000, PageBits, resolve, hit) == 0x802000);
    REQUIRE_FALSE(hit);
    REQUIRE(walks == 2);
}

TEST_CASE("MemTrack translations expire on remap, unmap and address-space change",
          "[video_core][memtrack]") {
    VideoCommon::MemTrack::TranslationCache cache;
    PageTable table;
    bool hit{};
    const auto resolve = [&](GPUVAddr address) { return table.Walk(address); };
    table.small[0] = {PageTable::State::Mapped, 0}; // Device address zero is a valid mapping.
    REQUIRE(cache.Translate(0, table.serial, 31, PageBits, resolve, hit) == 31);
    REQUIRE_FALSE(hit);
    table.small[0].device = 0x8000; // Same entry type, different mapping.
    table.serial += 2;
    REQUIRE(cache.Translate(0, table.serial, 31, PageBits, resolve, hit) == 0x801f);
    REQUIRE_FALSE(hit);
    table.small[0].state = PageTable::State::Free;
    table.serial += 2;
    REQUIRE_FALSE(cache.Translate(0, table.serial, 31, PageBits, resolve, hit));
    REQUIRE_FALSE(hit);
    table.small[0].state = PageTable::State::Mapped;
    // Negative results aren't cached, so even a first backing needs a fresh walk.
    REQUIRE(cache.Translate(0, table.serial, 31, PageBits, resolve, hit) == 0x801f);
    REQUIRE_FALSE(hit);
    table.small[0].device = 0x4000;
    REQUIRE(cache.Translate(1, table.serial, 31, PageBits, resolve, hit) == 0x401f);
    REQUIRE_FALSE(hit);
    cache.Clear();
    REQUIRE(cache.Translate(1, table.serial, 31, PageBits, resolve, hit) == 0x401f);
    REQUIRE_FALSE(hit);
}

TEST_CASE("MemTrack translation collisions never alias another page", "[video_core][memtrack]") {
    VideoCommon::MemTrack::TranslationCache cache;
    bool hit{};
    const auto resolve = [](GPUVAddr address) -> std::optional<DAddr> {
        return address + 0x900000;
    };
    // page 0 and page 65 hash to the same slot.
    REQUIRE(cache.Translate(0, 0, 17, PageBits, resolve, hit) == 0x900011);
    REQUIRE(cache.Translate(0, 0, 65 * PageSize + 19, PageBits, resolve, hit) ==
            0x900000 + 65 * PageSize + 19);
    REQUIRE_FALSE(hit);
    REQUIRE(cache.Translate(0, 0, 17, PageBits, resolve, hit) == 0x900011);
    REQUIRE_FALSE(hit);
}

TEST_CASE("MemTrack cached translations match a changing big and small page table",
          "[video_core][memtrack]") {
    VideoCommon::MemTrack::TranslationCache cache;
    PageTable table;
    std::mt19937_64 random{0x45abcd};
    bool hit{};
    const auto resolve = [&](GPUVAddr address) { return table.Walk(address); };
    for (std::size_t iteration = 0; iteration < 1000; ++iteration) {
        const auto small_index = static_cast<std::size_t>(random() % table.small.size());
        const auto big_index = static_cast<std::size_t>(random() % table.big.size());
        table.small[small_index] = {static_cast<PageTable::State>(random() % 3),
                                    (random() % 0x10000) * PageSize};
        table.big[big_index] = {static_cast<PageTable::State>(random() % 3),
                                (random() % 0x10000) * PageSize};
        table.serial += 2;
        for (std::size_t query = 0; query < 100; ++query) {
            const GPUVAddr address = random() % ((PageTable::SmallPages + 1) * PageSize);
            REQUIRE(cache.Translate(0, table.serial, address, PageBits, resolve, hit) ==
                    table.Walk(address));
            const GPUVAddr last_byte = (address & ~(PageSize - 1)) + PageSize - 1;
            REQUIRE(cache.Translate(0, table.serial, last_byte, PageBits, resolve, hit) ==
                    table.Walk(last_byte));
        }
    }
}

TEST_CASE("MemTrack translation caches are independent between threads", "[video_core][memtrack]") {
    const auto translate = [](DAddr device) {
        thread_local VideoCommon::MemTrack::TranslationCache cache;
        bool hit{};
        return cache.Translate(
            0, 0, 0x123, PageBits,
            [device](GPUVAddr address) -> std::optional<DAddr> { return device + address; }, hit);
    };
    std::optional<DAddr> first;
    std::optional<DAddr> second;
    std::thread first_thread{[&] { first = translate(0x1000); }};
    std::thread second_thread{[&] { second = translate(0x2000); }};
    first_thread.join();
    second_thread.join();
    REQUIRE(first == 0x1123);
    REQUIRE(second == 0x2123);
}
