// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <string_view>

#include "common/common_types.h"

/// EDEN_NX2_DRAWX_CALL_SITES=1 (statistics; read by Nx2FastDraw::ReadDrawXSwitches): which deferred
/// calls (GpuSplit Kind::Call entries without FastDrawNeutral) end the fast draw delta chain. A
/// site is the function object type the front deferred, named from the compiler's signature of
/// NameOf<Func> (a lambda by its file and line with clang). Per site and second: the calls that
/// broke the chain, the full draws whose last break was one of them, and the full draws that only
/// this site's calls broke since the previous draw (what making that site neutral would turn into
/// delta candidates). Front thread: Name. Back thread: everything else but g_other_break.
namespace VideoCommon::Nx2CallSites {

inline bool g_enabled{};

/// The compiler's signature of this function, which names `Func` (ShortName keeps that part).
template <typename Func>
[[nodiscard]] constexpr const char* NameOf() noexcept {
#if defined(__clang__) || defined(__GNUC__)
    return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    return __FUNCSIG__;
#else
    return "unnamed call site";
#endif
}

/// Front thread: `thunk` (the entry's function) runs calls deferred at the site `name`.
void Name(u64 thunk, const char* name) noexcept;

/// Front thread, in GpuSplit's deferring functions: names `thunk` the first time `Func` is
/// deferred.
template <typename Func, bool Words>
void NameOnce(u64 thunk) noexcept {
    static std::atomic<bool> named{};
    if (!named.load(std::memory_order_relaxed) &&
        !named.exchange(true, std::memory_order_relaxed)) {
        Name(thunk, NameOf<Func>());
    }
}

/// Any thread (Nx2FastDraw::Break): a break that is not a deferred call happened.
inline std::atomic<bool> g_other_break{};

/// Back thread: a call of `thunk` ran without FastDrawNeutral (after its Break).
void NoteBreak(u64 thunk) noexcept;
/// Back thread: a call of `thunk` deferred as FastDrawNeutral ran.
void NoteNeutral(u64 thunk) noexcept;

/// What broke the chain since the previous draw took it.
struct DrawBreaks {
    u64 sites{};    ///< One bit per site index
    u32 last{};     ///< 1 + the index of the site that broke last, or 0
    bool unnamed{}; ///< A call whose site was not named (more sites than the table holds)
    bool other{};   ///< A break that is not a deferred call
};
/// Back thread, at each draw's TakeReusableDraw: takes and forgets what broke since the previous.
[[nodiscard]] DrawBreaks TakeDraw() noexcept;
/// Back thread: the draw that took `breaks` was prepared in full because the chain broke;
/// `call_last`: its last break was a deferred call.
void NoteFullDraw(const DrawBreaks& breaks, bool call_last) noexcept;

/// Back thread: logs the window's sites (most full draws first) and starts a new window.
void Report(double seconds);

/// The part of a NameOf signature that names the site: a lambda's file (from its "src" folder),
/// line and column, or the type.
[[nodiscard]] std::string_view ShortName(std::string_view signature) noexcept;

/// Tests: per site index, the counts so far.
struct SiteCounts {
    std::string_view name;
    u64 breaks{};
    u64 neutral{};
    u64 last_draws{};
    u64 sole_draws{};
};
[[nodiscard]] SiteCounts CountsForTests(u32 index) noexcept;
[[nodiscard]] u32 SiteCountForTests() noexcept;
/// Tests: forgets every site and count.
void ResetForTests() noexcept;

} // namespace VideoCommon::Nx2CallSites
