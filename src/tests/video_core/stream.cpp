// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/cityhash.h"
#include "video_core/renderer_vulkan/astc_content_cache.h"
#include "video_core/stream_stats.h"
#include "video_core/texture_cache/accelerated_upload.h"
#include "video_core/textures/decoders.h"

namespace {
struct Snapshot {
    u64 tick;
    std::vector<u8> pixels;
};

void CacheChecks() {
    Vulkan::AstcContentCache<Snapshot> cache{100};
    const std::vector<u64> layout{1, 37, 48, 96, 2, 0, 7};
    const std::vector<u8> blocks{1, 2, 3, 4};
    u64 completed{};
    const auto is_free = [&](const Snapshot& snapshot) { return snapshot.tick <= completed; };
    REQUIRE(cache.Find(42, layout, blocks) == nullptr);
    REQUIRE(cache.MakeRoom(60, is_free));
    cache.Insert(42, layout, blocks, 60, Snapshot{2, {9, 8, 7}});
    auto* hit = cache.Find(42, layout, blocks);
    REQUIRE(hit != nullptr);
    REQUIRE(hit->payload.pixels == std::vector<u8>({9, 8, 7}));
    auto changed = blocks;
    changed.back() ^= 1;
    REQUIRE(!cache.Find(42, layout, changed)); // Deliberate hash collision/guest rewrite.
    for (size_t field = 0; field < layout.size(); ++field) {
        auto different = layout;
        ++different[field]; // Format, dimensions, layers, skipped mips, retained levels.
        REQUIRE(!cache.Find(42, different, blocks));
    }
    REQUIRE(!cache.MakeRoom(101, is_free));
    REQUIRE(!cache.MakeRoom(41, is_free)); // Capture still in flight: no wait and no eviction.
    REQUIRE(cache.Size() == size_t{1});
    REQUIRE(cache.Resident() == u64{60});
    completed = 2;
    REQUIRE(cache.MakeRoom(41, is_free));
    REQUIRE(cache.Size() == size_t{0});
    REQUIRE(cache.Resident() == u64{0});

    cache.Insert(1, layout, blocks, 40, Snapshot{2, {1}});
    cache.Insert(2, layout, blocks, 40, Snapshot{2, {2}});
    REQUIRE(cache.Find(1, layout, blocks)); // Touch oldest -> most recently used.
    REQUIRE(cache.MakeRoom(40, is_free));
    REQUIRE(!cache.Find(2, layout, blocks));
    REQUIRE(cache.Find(1, layout, blocks));
    cache.Find(1, layout, blocks)->payload.tick = 3; // Recorded hit pins its snapshot too.
    REQUIRE(!cache.MakeRoom(100, is_free));
    completed = 3;
    REQUIRE(cache.MakeRoom(100, is_free)); // Pressure purge preserves pending GPU ownership.
    REQUIRE(cache.Resident() == u64{0});

    // Two identical first-use requests: old behavior decodes twice, cache decodes only once.
    unsigned decodes{};
    for (unsigned request = 0; request < 2; ++request) {
        if (!cache.Find(7, layout, blocks)) {
            ++decodes;
            REQUIRE(cache.MakeRoom(40, is_free));
            cache.Insert(7, layout, blocks, 40, Snapshot{3, {6, 5, 4}});
        }
    }
    REQUIRE(decodes == 1U);
    unsigned previous_decodes{};
    for (unsigned request = 0; request < 2; ++request) {
        ++previous_decodes; // HEAD has no content lookup before AssembleInto's dispatches.
    }
    REQUIRE(previous_decodes == 2U);
    REQUIRE(decodes < previous_decodes);
}

u32 Deposit(u32 value, u32 mask) {
    u32 result{};
    for (u32 bit = 1; mask != 0; bit <<= 1) {
        if ((value & bit) != 0) {
            result |= mask & (~mask + 1);
        }
        mask &= mask - 1;
    }
    return result;
}

void BlockChecks() {
    for (bool native : {false, true}) {
        for (u32 height_log = 0; height_log <= 5; ++height_log) {
            for (u32 width : {1U, 3U, 4U, 5U, 13U, 33U, 65U}) {
                for (u32 height : {1U, 7U, 8U, 9U, 17U, 65U}) {
                    const u32 gobs = (width + 3) / 4;
                    const u32 block_bytes = gobs * (512U << height_log);
                    const u32 stride =
                        ((height + (8U << height_log) - 1) >> (3 + height_log)) * block_bytes;
                    const Vulkan::AstcBlockLayout block{32,
                                                        2,
                                                        stride,
                                                        width,
                                                        height,
                                                        block_bytes,
                                                        9 + height_log,
                                                        height_log,
                                                        (1U << height_log) - 1,
                                                        native};
                    std::vector<u8> input(32 + size_t{stride} * 2, 0xBA);
                    std::vector<u8> linear(size_t{width} * height * 16);
                    for (size_t index = 0; index < linear.size(); ++index) {
                        linear[index] = static_cast<u8>((index * 13 + index / 17) & 255);
                    }
                    std::vector<u8> expected;
                    for (u32 layer = 0; layer < 2; ++layer) {
                        auto destination =
                            std::span{input}.subspan(32 + size_t{layer} * stride, stride);
                        Tegra::Texture::SwizzleTexture(destination, linear, 16, width, height, 1,
                                                       height_log, 0, 1, native);
                        expected.insert(expected.end(), linear.begin(), linear.end());
                        for (u32 y = 0; y < height; ++y) {
                            for (u32 x = 0; x < width; ++x) {
                                const u64 base = Tegra::Texture::GetGOBOffset(width, height, x, y,
                                                                              height_log, 16);
                                u32 within = Deposit(x * 16, Tegra::Texture::SWIZZLE_X_BITS) |
                                             Deposit(y, Tegra::Texture::SWIZZLE_Y_BITS);
                                if (native) {
                                    within = (within & ~96U) | ((within & 32U) << 1) |
                                             ((within & 64U) >> 1);
                                }
                                REQUIRE(Vulkan::AstcBlockOffset(x, y, block_bytes, 9 + height_log,
                                                                height_log, (1U << height_log) - 1,
                                                                native) == base + within);
                            }
                        }
                    }
                    std::vector<u8> extracted;
                    REQUIRE(Vulkan::AppendAstcBlocks(extracted, input, block));
                    REQUIRE(extracted ==
                            expected); // Actual CPU swizzler is the independent oracle.
                    input[0] ^= 1;     // Allocation padding must not change canonical blocks.
                    std::vector<u8> padded;
                    REQUIRE(Vulkan::AppendAstcBlocks(padded, input, block));
                    REQUIRE(padded == expected);
                    auto truncated = std::span<const u8>{input}.first(32);
                    const auto previous = extracted;
                    REQUIRE_FALSE(Vulkan::AppendAstcBlocks(extracted, truncated, block));
                    REQUIRE(extracted == previous);
                }
            }
        }
    }
    std::vector<u8> output{1, 2};
    const std::array<u8, 16> input{};
    REQUIRE(!Vulkan::AppendAstcBlocks(output, input,
                                      {0, 1, 16, 0xFFFFFFFFU, 0xFFFFFFFFU, 512, 9, 0, 0, false}));
    REQUIRE(output == std::vector<u8>({1, 2}));
}

u64 Count(VideoCommon::StreamStats::Counter counter) {
    return VideoCommon::StreamStats::counters[static_cast<size_t>(counter)].load();
}
void StatsChecks(bool stats_enabled) {
    using namespace VideoCommon::StreamStats;
    REQUIRE(Enabled() == stats_enabled);
    Reset();
    FrameTracker tracker;
    tracker.Graphics(1);
    tracker.Decode(1);
    tracker.Decode(1);
    tracker.FinishFrame();
    tracker.Decode(2);
    tracker.Graphics(3); // Same frame, separate submissions: still a mixed frame.
    tracker.FinishFrame();
    tracker.Graphics(4);
    tracker.FinishFrame();
    tracker.Decode(5);
    tracker.Graphics(5);
    tracker.Graphics(5);
    tracker.FinishFrame();
    tracker.FinishFrame();
    REQUIRE(Count(Counter::DecodeFrames) == (stats_enabled ? u64{3} : u64{0}));
    REQUIRE(Count(Counter::MixedFrames) == (stats_enabled ? u64{3} : u64{0}));
    REQUIRE(Count(Counter::MixedSubmissions) == (stats_enabled ? u64{2} : u64{0}));
    Reset();
    tracker = {};
    tracker.Graphics(6);
    tracker.FinishFrame();
    tracker.Decode(6); // A command buffer may span frames; previous-frame graphics is not current.
    tracker.FinishFrame();
    REQUIRE(Count(Counter::MixedFrames) == u64{0});
    REQUIRE(Count(Counter::MixedSubmissions) == (stats_enabled ? u64{1} : u64{0}));
    {
        const Scope decode_barrier{Counter::DecodeBarrierNs};
        const Scope transfer_barrier{Counter::TransferBarrierNs};
        const Scope descriptor{Counter::DescriptorNs};
    }
    REQUIRE(Count(Counter::DecodeBarriers) == (stats_enabled ? u64{1} : u64{0}));
    REQUIRE(Count(Counter::TransferBarriers) == (stats_enabled ? u64{1} : u64{0}));
    REQUIRE(Count(Counter::DescriptorUpdates) == (stats_enabled ? u64{1} : u64{0}));
    Upload(true, false, 100);
    Upload(false, true, 200);
    Upload(false, false, 300);
    REQUIRE(Count(Counter::BytesAstc) == (stats_enabled ? u64{100} : u64{0}));
    REQUIRE(Count(Counter::BytesBcn) == (stats_enabled ? u64{200} : u64{0}));
    REQUIRE(Count(Counter::BytesOther) == (stats_enabled ? u64{300} : u64{0}));
    std::thread worker([] {
        for (unsigned i = 0; i < 10000; ++i)
            Add(Counter::Dispatches);
    });
    for (unsigned i = 0; i < 10000; ++i)
        Add(Counter::Dispatches);
    worker.join();
    REQUIRE(Count(Counter::Dispatches) == (stats_enabled ? u64{20000} : u64{0}));
    counters[static_cast<size_t>(Counter::CacheBytes)].store(123);
    const auto previous_window = window_start;
    Report();
    REQUIRE(window_start == previous_window); // Ten-second gate.
    REQUIRE(Count(Counter::Dispatches) == (stats_enabled ? u64{20000} : u64{0}));
    window_start -= std::chrono::seconds{11};
    Report();
    if (stats_enabled) {
        REQUIRE(window_start > previous_window);
        REQUIRE(Count(Counter::Dispatches) == u64{0});
        REQUIRE(Count(Counter::CacheBytes) == u64{123});
    } else {
        REQUIRE(window_start == previous_window - std::chrono::seconds{11});
    }
}
} // namespace

TEST_CASE("ASTC content equality, layout and in-flight ownership", "[video_core][stream]") {
    CacheChecks();
}

TEST_CASE("ASTC canonical blocks match the independent CPU swizzler", "[video_core][stream]") {
    BlockChecks();
}

TEST_CASE("Stream counters honor startup flags and reporting windows", "[video_core][stream]") {
    using namespace VideoCommon::StreamStats;
    const bool saved_enabled = enabled;
    const auto saved_start = window_start;
    std::array<u64, static_cast<size_t>(Counter::Count)> saved_counters{};
    for (size_t index = 0; index < counters.size(); ++index) {
        saved_counters[index] = counters[index].load();
        counters[index].store(0);
    }
    struct Restore {
        bool flag;
        std::chrono::steady_clock::time_point start;
        decltype(saved_counters)& values;
        ~Restore() {
            enabled = flag;
            window_start = start;
            for (size_t index = 0; index < counters.size(); ++index) {
                counters[index].store(values[index]);
            }
        }
    } restore{saved_enabled, saved_start, saved_counters};
    enabled = false;
    const auto before_reset = window_start;
    counters[static_cast<size_t>(Counter::Lookups)].store(7);
    Reset();
    REQUIRE(window_start == before_reset);
    REQUIRE(Count(Counter::Lookups) == u64{7});
    counters[static_cast<size_t>(Counter::Lookups)].store(0);
    StatsChecks(false);
    enabled = true;
    StatsChecks(true);
}

TEST_CASE("ASTC repeat history stops retaining new content at capacity", "[video_core][stream]") {
    Vulkan::AstcSeenContents<u64, u64, 2> history;
    REQUIRE(history.Insert(u64{1}, u64{10}));
    REQUIRE(history.Insert(u64{2}, u64{20}));
    REQUIRE_FALSE(history.Insert(u64{3}, u64{30}));
    REQUIRE(history.Size() == size_t{2});
    REQUIRE(history.Find(3) == nullptr);
    REQUIRE(history.Find(1) != nullptr);
    REQUIRE(*history.Find(1) == u64{10});
    REQUIRE(*history.Find(2) == u64{20});
}

TEST_CASE("ASTC verification records only after two bounded mapped allocations",
          "[video_core][stream]") {
    struct Download {
        bool allocated;
        std::array<u8, 16> bytes{};
        explicit operator bool() const {
            return allocated;
        }
        std::span<u8> Mapped() {
            return allocated ? std::span{bytes} : std::span<u8>{};
        }
    };
    for (unsigned failed : {1U, 2U, 0U}) {
        unsigned allocations{};
        unsigned recordings{};
        const auto allocate = [&] { return Download{++allocations != failed}; };
        const auto compare = [&](auto&, auto&) {
            REQUIRE(allocations == 2U);
            ++recordings;
        };
        REQUIRE(Vulkan::WithAstcVerificationBuffers(16, allocate, compare) == (failed == 0U));
        REQUIRE(recordings == (failed == 0U ? 1U : 0U));
        REQUIRE(allocations == (failed == 1U ? 1U : 2U));
    }
    unsigned allocations{};
    const auto allocate = [&] {
        ++allocations;
        return Download{true};
    };
    const auto compare = [](auto&, auto&) { FAIL("Oversized readbacks must not record commands"); };
    REQUIRE_FALSE(Vulkan::WithAstcVerificationBuffers(0, allocate, compare));
    REQUIRE_FALSE(Vulkan::WithAstcVerificationBuffers((u64{32} << 20) + 1, allocate, compare));
    REQUIRE(allocations == 0U);
    REQUIRE_FALSE(Vulkan::WithAstcVerificationBuffers(17, allocate, compare));
    REQUIRE(allocations == 1U); // Unmapped/short mappings are also an explicit skip.
    const auto lost_device = []() -> Download { throw std::runtime_error("device lost"); };
    REQUIRE_THROWS_AS(Vulkan::WithAstcVerificationBuffers(16, lost_device, compare),
                      std::runtime_error);
}

TEST_CASE("Stream switches are read explicitly at session startup", "[video_core][stream]") {
    using namespace VideoCommon::StreamStats;
    struct Restore {
        bool stats = enabled;
        bool cache = cache_enabled;
        bool verify = verify_enabled;
        ~Restore() {
            enabled = stats;
            cache_enabled = cache;
            verify_enabled = verify;
        }
    } restore;
    const auto expected = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    ReadSwitch();
    REQUIRE(Enabled() == expected("EDEN_NX2_STREAM_STATS"));
    REQUIRE(cache_enabled == expected("EDEN_NX2_ASTC_CACHE"));
    REQUIRE(verify_enabled == expected("EDEN_NX2_ASTC_CACHE_VERIFY"));
}

TEST_CASE("ASTC keys retain guest bytes across skipped mip tails", "[video_core][stream]") {
    for (bool native : {false, true}) {
        std::vector<Vulkan::AstcBlockLayout> levels;
        u32 stride = 0;
        for (u32 level = 0; level < 5; ++level) {
            const u32 width = (std::max)(33U >> level, 1U);
            const u32 height = (std::max)(17U >> level, 1U);
            const u32 height_log = level < 2 ? 1U : 0U;
            const u32 block_bytes = ((width + 3) / 4) * (512U << height_log);
            levels.push_back({stride, 2, 0, width, height, block_bytes, 9 + height_log,
                              height_log, (1U << height_log) - 1, native});
            stride += ((height + (8U << height_log) - 1) >> (3 + height_log)) * block_bytes;
        }
        std::vector<u8> guest(size_t{stride} * 2);
        for (size_t index = 0; index < guest.size(); ++index) {
            guest[index] = static_cast<u8>((index * 13 + index / 17) & 255);
        }
        for (u32 skip : {0U, 2U}) {
            const size_t offset = levels[skip].offset;
            const u32 tail = stride - static_cast<u32>(offset);
            std::vector<u8> staging(size_t{tail} * 2);
            std::vector<u8> cached;
            unsigned reads = 0;
            const auto read = [&](std::span<u8> destination) {
                ++reads;
                for (size_t layer = 0; layer < 2; ++layer) {
                    std::copy_n(guest.data() + layer * stride + offset, tail,
                                destination.data() + layer * tail);
                }
            };
            VideoCommon::ReadAcceleratedUpload(staging, cached, true, read);
            REQUIRE(reads == 1U);
            REQUIRE(cached == staging);
            std::vector<u8> old_key;
            std::vector<u8> new_key;
            for (size_t level = skip; level < levels.size(); ++level) {
                auto old_layout = levels[level];
                old_layout.layer_stride = stride;
                auto new_layout = old_layout;
                new_layout.offset -= offset;
                new_layout.layer_stride = tail;
                REQUIRE(Vulkan::AppendAstcBlocks(old_key, guest, old_layout));
                REQUIRE(Vulkan::AppendAstcBlocks(new_key, cached, new_layout));
            }
            REQUIRE(old_key == new_key);
            const auto hash64 = [](const std::vector<u8>& bytes) {
                return Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            };
            const auto hash128 = [](const std::vector<u8>& bytes) {
                return Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            };
            REQUIRE(hash64(old_key) == hash64(new_key));
            REQUIRE(hash128(old_key) == hash128(new_key));
            std::fill(staging.begin(), staging.end(), 0);
            REQUIRE(cached != staging);
            std::vector<u8> disabled;
            reads = 0;
            VideoCommon::ReadAcceleratedUpload(staging, disabled, false,
                [&](std::span<u8> destination) {
                    REQUIRE(destination.data() == staging.data());
                    read(destination);
                });
            REQUIRE(reads == 1U);
            REQUIRE(disabled.empty());
            REQUIRE(staging == cached);
        }
    }
}
