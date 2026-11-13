// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace VideoCommon::Nx2Tested {

/// Whether the running title is the one the EDEN_NX2_* features read through On were measured and
/// verified in: Ocarina of Time (Switch 2), its Castle Town benchmark with their verifiers clean.
[[nodiscard]] bool TestedTitle();

/// Whether such a feature switch is on. Unset, it is on in the tested title and off in every other
/// one; set, "1" turns it on and anything else off, in any title.
[[nodiscard]] bool On(const char* name);

} // namespace VideoCommon::Nx2Tested
