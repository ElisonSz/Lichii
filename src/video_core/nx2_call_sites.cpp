// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>

#include "common/logging.h"
#include "video_core/nx2_call_sites.h"

namespace VideoCommon::Nx2CallSites {

namespace {

constexpr u32 MaxSites = 64;

struct Site {
    std::atomic<u64> thunk{};
    const char* name{};
    u64 breaks{};     ///< Calls that ended the chain
    u64 neutral{};    ///< Calls deferred as FastDrawNeutral
    u64 last_draws{}; ///< Full draws whose last break was one of this site's calls
    u64 sole_draws{}; ///< Full draws only this site's calls broke
};

std::array<Site, MaxSites> sites;
std::atomic<u32> site_count{};

// The back's, since the previous draw took them.
u64 draw_sites{};
u32 draw_last{};
bool draw_unnamed{};
u32 cached_index{};

// The back's, per window.
u64 unnamed_calls{};
u64 call_last_draws{};
u64 call_only_draws{};
u64 one_site_draws{};
u64 several_site_draws{};

[[nodiscard]] u32 Find(u64 thunk) noexcept {
    const u32 count = site_count.load(std::memory_order_acquire);
    if (cached_index < count &&
        sites[cached_index].thunk.load(std::memory_order_relaxed) == thunk) {
        return cached_index;
    }
    for (u32 index = 0; index < count; ++index) {
        if (sites[index].thunk.load(std::memory_order_acquire) == thunk) {
            cached_index = index;
            return index;
        }
    }
    return MaxSites;
}

} // Anonymous namespace

void Name(u64 thunk, const char* name) noexcept {
    const u32 count = site_count.load(std::memory_order_relaxed);
    if (count >= MaxSites) {
        return;
    }
    sites[count].name = name;
    sites[count].thunk.store(thunk, std::memory_order_release);
    site_count.store(count + 1, std::memory_order_release);
}

void NoteBreak(u64 thunk) noexcept {
    const u32 index = Find(thunk);
    if (index == MaxSites) {
        ++unnamed_calls;
        draw_unnamed = true;
        draw_last = 0;
        return;
    }
    ++sites[index].breaks;
    draw_sites |= u64{1} << index;
    draw_last = index + 1;
}

void NoteNeutral(u64 thunk) noexcept {
    if (const u32 index = Find(thunk); index != MaxSites) {
        ++sites[index].neutral;
    }
}

DrawBreaks TakeDraw() noexcept {
    const DrawBreaks breaks{
        .sites = draw_sites,
        .last = draw_last,
        .unnamed = draw_unnamed,
        .other = g_other_break.exchange(false, std::memory_order_relaxed),
    };
    draw_sites = 0;
    draw_last = 0;
    draw_unnamed = false;
    return breaks;
}

void NoteFullDraw(const DrawBreaks& breaks, bool call_last) noexcept {
    if (call_last) {
        ++call_last_draws;
        if (breaks.last != 0) {
            ++sites[breaks.last - 1].last_draws;
        }
    }
    if (breaks.other || (breaks.sites == 0 && !breaks.unnamed)) {
        return;
    }
    ++call_only_draws;
    if (!breaks.unnamed && std::has_single_bit(breaks.sites)) {
        ++one_site_draws;
        ++sites[static_cast<size_t>(std::countr_zero(breaks.sites))].sole_draws;
    } else {
        ++several_site_draws;
    }
}

void Report(double seconds) {
    if (seconds <= 0.0) {
        return;
    }
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    const u32 count = site_count.load(std::memory_order_acquire);
    std::array<u32, MaxSites> order{};
    std::iota(order.begin(), order.begin() + count, 0U);
    std::sort(order.begin(), order.begin() + count, [](u32 lhs, u32 rhs) {
        const Site& left = sites[lhs];
        const Site& right = sites[rhs];
        if (left.last_draws != right.last_draws) {
            return left.last_draws > right.last_draws;
        }
        if (left.breaks != right.breaks) {
            return left.breaks > right.breaks;
        }
        return left.neutral > right.neutral;
    });
    LOG_INFO(
        HW_GPU,
        "DrawX call sites: {:.1f} s | full draws/s whose last break was a deferred call={:.0f} "
        "| broken only by deferred calls={:.0f} (of one site {:.0f}, of several {:.0f}) | "
        "calls of unnamed sites/s={:.0f} | {} sites named",
        seconds, rate(call_last_draws), rate(call_only_draws), rate(one_site_draws),
        rate(several_site_draws), rate(unnamed_calls), count);
    constexpr u32 Listed = 16;
    for (u32 rank = 0; rank < (std::min)(count, Listed); ++rank) {
        const Site& site = sites[order[rank]];
        if (site.breaks == 0 && site.neutral == 0) {
            break;
        }
        LOG_INFO(HW_GPU,
                 "DrawX call site: calls/s ending the chain={:.0f} neutral={:.0f} | full draws/s "
                 "it last broke={:.0f}, it alone broke={:.0f} | {}",
                 rate(site.breaks), rate(site.neutral), rate(site.last_draws),
                 rate(site.sole_draws), ShortName(site.name != nullptr ? site.name : "?"));
    }
    for (u32 index = 0; index < count; ++index) {
        Site& site = sites[index];
        site.breaks = 0;
        site.neutral = 0;
        site.last_draws = 0;
        site.sole_draws = 0;
    }
    unnamed_calls = 0;
    call_last_draws = 0;
    call_only_draws = 0;
    one_site_draws = 0;
    several_site_draws = 0;
}

std::string_view ShortName(std::string_view signature) noexcept {
    std::string_view name = signature;
    if (const size_t func = name.find("Func = "); func != std::string_view::npos) {
        // Clang and GCC: "... [Func = <type>]".
        name.remove_prefix(func + 7);
        if (const size_t end = name.rfind(']'); end != std::string_view::npos) {
            name = name.substr(0, end);
        }
    } else if (const size_t open = name.find("NameOf<"); open != std::string_view::npos) {
        // MSVC: "... NameOf<<type>>(void) noexcept".
        name.remove_prefix(open + 7);
        if (const size_t close = name.rfind(">(void)"); close != std::string_view::npos) {
            name = name.substr(0, close);
        }
    }
    // A clang lambda, "(lambda at <path>:<line>:<column>)": the path from the source folder on.
    if (const size_t at = name.find("lambda at "); at != std::string_view::npos) {
        std::string_view path = name.substr(at + 10);
        if (!path.empty() && path.back() == ')') {
            path.remove_suffix(1);
        }
        const size_t slash = path.rfind("src/");
        const size_t backslash = path.rfind("src\\");
        size_t from = std::string_view::npos;
        if (slash != std::string_view::npos && backslash != std::string_view::npos) {
            from = (std::max)(slash, backslash);
        } else if (slash != std::string_view::npos) {
            from = slash;
        } else {
            from = backslash;
        }
        if (from != std::string_view::npos) {
            path.remove_prefix(from + 4);
        }
        return path;
    }
    return name;
}

SiteCounts CountsForTests(u32 index) noexcept {
    if (index >= site_count.load(std::memory_order_acquire)) {
        return {};
    }
    const Site& site = sites[index];
    return {.name = site.name != nullptr ? std::string_view{site.name} : std::string_view{},
            .breaks = site.breaks,
            .neutral = site.neutral,
            .last_draws = site.last_draws,
            .sole_draws = site.sole_draws};
}

u32 SiteCountForTests() noexcept {
    return site_count.load(std::memory_order_acquire);
}

void ResetForTests() noexcept {
    for (Site& site : sites) {
        site.thunk.store(0, std::memory_order_relaxed);
        site.name = nullptr;
        site.breaks = 0;
        site.neutral = 0;
        site.last_draws = 0;
        site.sole_draws = 0;
    }
    site_count.store(0, std::memory_order_release);
    draw_sites = 0;
    draw_last = 0;
    draw_unnamed = false;
    cached_index = 0;
    g_other_break.store(false, std::memory_order_relaxed);
    unnamed_calls = 0;
    call_last_draws = 0;
    call_only_draws = 0;
    one_site_draws = 0;
    several_site_draws = 0;
}

} // namespace VideoCommon::Nx2CallSites
