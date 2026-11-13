// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "common/common_types.h"
#include "common/settings.h"
#include "video_core/nx2_tested.h"

namespace VideoCommon::Nx2Tested {

namespace {
/// The Legend of Zelda: Ocarina of Time (Switch 2).
constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
} // Anonymous namespace

bool TestedTitle() {
    return Settings::GetCurrentProgramID() == OcarinaOfTime;
}

bool On(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr ? value[0] == '1' : TestedTitle();
}

} // namespace VideoCommon::Nx2Tested
