// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "video_core/ocarina_camera.h"

namespace {
VideoCommon::OcarinaCameraBuffer Camera(f32 width = 2560.0f, f32 height = 1440.0f) {
    VideoCommon::OcarinaCameraBuffer values{};
    for (size_t index = 0; index < 16; ++index) {
        values[0x30 / sizeof(f32) + index] = 100.0f + static_cast<f32>(index);
        values[0x180 / sizeof(f32) + index] = 200.0f + static_cast<f32>(index);
    }
    values[0xa8 / sizeof(f32)] = -1.0f;
    values[0xac / sizeof(f32)] = 0.0f;
    values[0x120 / sizeof(f32)] = width;
    values[0x124 / sizeof(f32)] = height;
    return values;
}
using VideoCommon::DecodeOcarinaCamera;
using VideoCommon::OcarinaCameraIssue;
} // namespace

TEST_CASE("Ocarina camera needs only the bytes through its previous matrix",
          "[video_core][ocarina]") {
    const auto values = Camera();
    REQUIRE(sizeof(values) == 0x1c0);
    const auto camera = DecodeOcarinaCamera(values, 2560, 1440, false);
    REQUIRE(camera.issue == OcarinaCameraIssue::None);
    CHECK(camera.width == 2560);
    CHECK(camera.height == 1440);
    for (size_t index = 0; index < 16; ++index) {
        CHECK(camera.current[index] == 100.0f + static_cast<f32>(index));
        CHECK(camera.previous[index] == 200.0f + static_cast<f32>(index));
    }
}

TEST_CASE("Ocarina camera refuses a buffer missing any part of its previous matrix",
          "[video_core][ocarina]") {
    const auto values = Camera();
    for (const size_t count : {size_t{0}, size_t{0x180 / sizeof(f32)}, values.size() - 1}) {
        CHECK(DecodeOcarinaCamera(std::span{values}.first(count), 2560, 1440, false).issue ==
              OcarinaCameraIssue::ShortBuffer);
    }
}

TEST_CASE("Ocarina camera keeps scene extent and perspective checks", "[video_core][ocarina]") {
    CHECK(DecodeOcarinaCamera(Camera(1280, 720), 2560, 1440, false).issue ==
          OcarinaCameraIssue::Extent);
    CHECK(DecodeOcarinaCamera(Camera(2561, 1440), 2560, 1440, true).issue ==
          OcarinaCameraIssue::Extent);
    CHECK(DecodeOcarinaCamera(Camera(2560.5f, 1440), 2561, 1440, true).issue ==
          OcarinaCameraIssue::Extent);
    CHECK(DecodeOcarinaCamera(Camera(320, 180), 320, 180, false).issue ==
          OcarinaCameraIssue::Extent);
    for (const f32 invalid : {std::numeric_limits<f32>::quiet_NaN(),
                              std::numeric_limits<f32>::infinity(), -1.0f}) {
        CHECK(DecodeOcarinaCamera(Camera(invalid, 1440), 2560, 1440, false).issue ==
              OcarinaCameraIssue::Extent);
    }
    auto orthographic = Camera();
    orthographic[0xa8 / sizeof(f32)] = 0.0f;
    orthographic[0xac / sizeof(f32)] = 1.0f;
    CHECK(DecodeOcarinaCamera(orthographic, 2560, 1440, false).issue ==
          OcarinaCameraIssue::Projection);
}

TEST_CASE("Ocarina camera accepts DLSS padding only within the existing tolerance",
          "[video_core][ocarina]") {
    CHECK(DecodeOcarinaCamera(Camera(2276, 1280), 2288, 1280, true).issue ==
          OcarinaCameraIssue::None);
    CHECK(DecodeOcarinaCamera(Camera(2276, 1280), 2288, 1280, false).issue ==
          OcarinaCameraIssue::Extent);
    CHECK(DecodeOcarinaCamera(Camera(2276, 1280), 2292, 1280, true).issue ==
          OcarinaCameraIssue::Extent);
}
