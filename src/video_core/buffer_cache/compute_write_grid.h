// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <optional>
#include "common/common_types.h"

namespace VideoCommon {
// A list slot's QMD grid holds tagged placeholders, not group counts. As with an ordinary
// indirect dispatch, only the GPU knows the real grid; grid-dependent write bounds are unknown.
template <class Engine> std::optional<std::array<u32, 3>> ComputeWriteGrid(const Engine &engine) {
    if (engine.GetIndirectComputeAddress() || engine.GetIndirectListSlot())
        return std::nullopt;
    const auto &qmd = engine.launch_description;
    return std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                              qmd.grid_dim_z.Value()};
}
} // namespace VideoCommon
