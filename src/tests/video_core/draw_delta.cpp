// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "video_core/draw_delta.h"

namespace {
using Snapshot = VideoCommon::DrawDeltaTextures<u64, u64>;
using Result = VideoCommon::DrawDeltaTextureResult;

struct Textures {
    std::array<u64, 4> images{11, 12, 13, 14};
    std::array<u64, 4> samplers{21, 22, 23, 24};
    std::array<Snapshot::Handle, 3> handles{{{0, 1}, {2, 3}, {2, 3}}};
    u64 image_epoch{1};
    u64 sampler_epoch{1};
    Snapshot snapshot;

    void Capture() {
        snapshot.Capture(
            handles, image_epoch, sampler_epoch, [&](u32 index) { return Read(images, index); },
            [&](u32 index) { return Read(samplers, index); });
    }
    Result Match() const {
        return snapshot.Match(
            handles, image_epoch, sampler_epoch, [&](u32 index) { return Read(images, index); },
            [&](u32 index) { return Read(samplers, index); });
    }
    static std::optional<u64> Read(const std::array<u64, 4>& entries, u32 index) {
        return index < entries.size() ? std::optional<u64>{entries[index]} : std::nullopt;
    }
};
} // namespace

TEST_CASE("Draw delta compares handles and descriptor contents independently", "[draw_delta]") {
    Textures textures;
    CHECK(textures.Match() == Result::NotCaptured);
    textures.Capture();
    REQUIRE(textures.Match() == Result::Match);

    SECTION("changed handle from a rebound constant buffer or storage program") {
        textures.handles[1].first = 1;
        CHECK(textures.Match() == Result::Handles);
    }
    SECTION("changed sampler selection") {
        textures.handles[1].second = 2;
        CHECK(textures.Match() == Result::Handles);
    }
    SECTION("same handles with rewritten TIC") {
        textures.images[2] = 99;
        CHECK(textures.Match() == Result::Descriptors);
    }
    SECTION("same handles with rewritten TSC") {
        textures.samplers[3] = 99;
        CHECK(textures.Match() == Result::Descriptors);
    }
    SECTION("unrelated descriptor rewrite keeps the selected descriptors") {
        textures.images[1] = 99;
        textures.samplers[0] = 99;
        CHECK(textures.Match() == Result::Match);
    }
    SECTION("array elements and duplicates retain order") {
        std::swap(textures.handles[0], textures.handles[1]);
        CHECK(textures.Match() == Result::Handles);
    }
}

TEST_CASE("Draw delta supports non-sampled image and buffer bindings", "[draw_delta]") {
    using Descriptor = VideoCommon::DrawDeltaTextures<u64, u64>;
    Descriptor snapshot;
    const std::array<Descriptor::Handle, 2> handles{{{0, ~0U}, {1, ~0U}}};
    const std::array<u64, 2> images{11, 12};
    const auto read_image = [&](u32 index) -> std::optional<u64> {
        return index < images.size() ? std::optional<u64>{images[index]} : std::nullopt;
    };
    const auto read_sampler = [](u32) -> std::optional<u64> {
        FAIL("non-sampled bindings must not read a sampler");
        return std::nullopt;
    };
    snapshot.Capture(handles, 1, 1, read_image, read_sampler);
    CHECK(snapshot.Match(handles, 1, 1, read_image, read_sampler) == Result::Match);
}

TEST_CASE("Draw delta rejects invalidated IDs even when descriptor bytes are identical",
          "[draw_delta]") {
    Textures textures;
    textures.Capture();
    SECTION("image deletion or TIC table refresh") {
        ++textures.image_epoch;
    }
    SECTION("sampler invalidation or TSC table refresh") {
        ++textures.sampler_epoch;
    }
    CHECK(textures.Match() == Result::Descriptors);
    textures.Capture();
    CHECK(textures.Match() == Result::Match);
}

TEST_CASE("Draw delta cannot capture missing descriptors or keep a previous successful capture",
          "[draw_delta]") {
    Textures textures;
    textures.Capture();
    textures.handles[2].first = 4;
    textures.Capture();
    CHECK(textures.Match() == Result::NotCaptured);
    textures.handles[2].first = 2;
    CHECK(textures.Match() == Result::NotCaptured);
    textures.Capture();
    CHECK(textures.Match() == Result::Match);
}

TEST_CASE("Draw delta handles empty pipelines and descriptor count changes", "[draw_delta]") {
    Snapshot snapshot;
    const auto read = [](u32) -> std::optional<u64> { return std::nullopt; };
    snapshot.Capture({}, 1, 1, read, read);
    CHECK(snapshot.Match({}, 1, 1, read, read) == Result::Match);
    const std::array<Snapshot::Handle, 1> handles{{{0, 0}}};
    CHECK(snapshot.Match(handles, 1, 1, read, read) == Result::Handles);
    CHECK(snapshot.Match({}, 2, 1, read, read) == Result::Descriptors);
}

TEST_CASE("Draw delta compares against the resolved bytes rather than rereading at capture",
          "[draw_delta]") {
    Textures textures;
    textures.Capture();
    // Models a descriptor changed after VisitImageView read it: retaining its ID must compare
    // against the bytes Visit read, not against a fresh capture of guest memory.
    textures.images[0] ^= 1;
    CHECK(textures.Match() == Result::Descriptors);
    textures.images[0] ^= 1;
    CHECK(textures.Match() == Result::Match);
    textures.handles[1].second = 4;
    textures.Capture();
    CHECK(textures.Match() == Result::NotCaptured);
}

TEST_CASE("Draw delta texture decisions agree with a fresh selected-descriptor resolve",
          "[draw_delta]") {
    Textures textures;
    textures.Capture();
    const auto original_handles = textures.handles;
    const auto original_images = textures.images;
    const auto original_samplers = textures.samplers;
    u32 random = 7;
    for (u32 iteration = 0; iteration < 20000; ++iteration) {
        random = random * 1664525U + 1013904223U;
        textures.handles = original_handles;
        textures.images = original_images;
        textures.samplers = original_samplers;
        const u32 index = (random >> 8) % 4;
        switch (random % 4) {
        case 0:
            textures.images[index] ^= 1;
            break;
        case 1:
            textures.samplers[index] ^= 1;
            break;
        case 2:
            textures.handles[index % 3].first = (random >> 16) % 4;
            break;
        case 3:
            break;
        }
        bool equal = textures.handles == original_handles;
        for (size_t element = 0; element < original_handles.size(); ++element) {
            const auto [image, sampler] = original_handles[element];
            equal &= textures.images[image] == original_images[image] &&
                     textures.samplers[sampler] == original_samplers[sampler];
        }
        REQUIRE((textures.Match() == Result::Match) == equal);
    }
}

TEST_CASE("Draw delta rechecks state after guest handle reads", "[draw_delta]") {
    using VideoCommon::CheckDrawDeltaGuard;
    using VideoCommon::DrawDeltaGuard;
    using GuardResult = VideoCommon::DrawDeltaGuardResult;
    const DrawDeltaGuard previous{1, 2, 3, 4};
    DrawDeltaGuard current = previous;
    CHECK(CheckDrawDeltaGuard(previous, current, true, true) == GuardResult::Match);
    SECTION("CPU invalidation or another engine") {
        ++current.serial;
        CHECK(CheckDrawDeltaGuard(previous, current, true, true) == GuardResult::State);
    }
    SECTION("mapping changes") {
        ++current.map_serial;
        CHECK(CheckDrawDeltaGuard(previous, current, true, true) == GuardResult::State);
    }
    SECTION("upload submits a command buffer") {
        ++current.tick;
        CHECK(CheckDrawDeltaGuard(previous, current, true, true) == GuardResult::Submitted);
    }
    SECTION("query handler ends the pass") {
        CHECK(CheckDrawDeltaGuard(previous, current, false, true) == GuardResult::RenderPass);
    }
    SECTION("pass replaced or graphics pipeline displaced by intervening work") {
        ++current.render_pass;
        CHECK(CheckDrawDeltaGuard(previous, current, true, true) == GuardResult::RenderPass);
        CHECK(CheckDrawDeltaGuard(previous, previous, true, false) == GuardResult::RenderPass);
    }
}
