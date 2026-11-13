// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <string>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/common_types.h"
#include "common/logging.h"

#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

/// Macro calls on the split GPU thread's front (EDEN_NX2_MACRO_*): Ocarina of Time makes ~50,000
/// a frame, ~7 register writes each, and the front ran each write through a call chain (JIT,
/// thunk, emit lambda, FrontWrite) of its own.
namespace VideoCommon::Nx2Macro {

/// EDEN_NX2_MACRO_BATCH=1: the front's macro calls buffer their writes in the generated code and
/// hand them over in batches (TuringMacroJit::ExecuteBatched): before a register read, a DMA read,
/// when the buffer is full and when the call ends. A plain 3D register write then takes
/// FrontWrite's own steps in a loop (Maxwell3D::FrontWriteBatch); every other write, and every
/// compute write, goes where it went before, in the same order.
inline bool g_batch{};
/// EDEN_NX2_MACRO_FASTCALL=1: a 3D macro known to run on the front (FrontRunOnBack decided so for
/// its start and nothing changed since) skips FrontRunOnBack's checks: one lookup by start.
inline bool g_fastcall{};
/// EDEN_NX2_MACRO_VERIFY=n (n >= 1): one batched 3D call in n (1: every call) is run again on
/// copies with the unbatched program (TuringMacroJit::Execute), its register reads answered as
/// the batched call saw them, and must emit the same writes in the same order with the same
/// result and data RAM; in that call every plain write also goes through FrontWrite, which must
/// do what the batch would have done. Calls that made a DMA read are not run again.
inline u32 g_verify_every{};
/// EDEN_NX2_MACRO_STATS=1: one "Macro batch" line every 10 s with the counters below.
inline bool g_stats{};

/// Counted on the front GPU thread only (its macro calls), with the stats or verify line on.
enum class Counter : std::size_t {
    Calls,           ///< 3D macro calls batched
    ComputeCalls,    ///< compute macro calls batched
    FastCalls,       ///< 3D calls that skipped FrontRunOnBack (EDEN_NX2_MACRO_FASTCALL)
    Flushes,         ///< batches handed over
    Buffered,        ///< writes in them
    PlainSent,       ///< plain 3D writes the batch loop sent
    PlainSame,       ///< plain 3D writes that changed nothing (not sent, as FrontWrite does)
    OtherWrites,     ///< 3D writes in batches that went to FrontWrite
    VerifiedCalls,   ///< calls run again unbatched
    VerifySkipped,   ///< calls not run again (DMA reads)
    CallMismatches,  ///< of the calls run again: other writes, reads, result or data RAM
    WritesChecked,   ///< plain writes checked against FrontWrite
    WriteMismatches, ///< of them: FrontWrite sent, kept or shadowed another value
    Count,
};

inline std::array<u64, static_cast<std::size_t>(Counter::Count)> g_counts{};

inline void Count(Counter counter, u64 amount = 1) {
    if (g_stats || g_verify_every != 0) [[unlikely]] {
        g_counts[static_cast<std::size_t>(counter)] += amount;
    }
}

[[nodiscard]] inline bool ReadFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

/// Called by VideoCore::CreateGPU, before the GPU threads start.
inline void ReadSwitches() {
    g_batch = ReadFlag("EDEN_NX2_MACRO_BATCH");
    g_fastcall = ReadFlag("EDEN_NX2_MACRO_FASTCALL");
    g_verify_every = 0;
    if (const char* value = std::getenv("EDEN_NX2_MACRO_VERIFY")) {
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        g_verify_every = parsed > 1000000 ? 1000000U : static_cast<u32>(parsed);
    }
    g_stats = ReadFlag("EDEN_NX2_MACRO_STATS");
    if (g_batch || g_fastcall || g_verify_every != 0 || g_stats) {
        LOG_INFO(HW_GPU, "Macro switches: batch {} fastcall {} verify one call in {} stats {}",
                 g_batch, g_fastcall, g_verify_every, g_stats);
    }
}

/// Whether to run this batched 3D call again (EDEN_NX2_MACRO_VERIFY).
[[nodiscard]] inline bool SampleVerify() {
    if (g_verify_every == 0) {
        return false;
    }
    static u32 countdown{};
    if (countdown == 0) {
        countdown = g_verify_every;
    }
    return --countdown == 0;
}

/// The stats line, at most once every 10 s (the front calls it every 4096 batched calls).
inline void Report() {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point window_start = Clock::now();
    const Clock::time_point now = Clock::now();
    if (now - window_start < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    window_start = now;
    const auto take = [](Counter counter) {
        return std::exchange(g_counts[static_cast<std::size_t>(counter)], u64{0});
    };
    const u64 calls = take(Counter::Calls);
    const u64 compute_calls = take(Counter::ComputeCalls);
    const u64 fast_calls = take(Counter::FastCalls);
    const u64 flushes = take(Counter::Flushes);
    const u64 buffered = take(Counter::Buffered);
    const u64 plain_sent = take(Counter::PlainSent);
    const u64 plain_same = take(Counter::PlainSame);
    const u64 other_writes = take(Counter::OtherWrites);
    const u64 verified = take(Counter::VerifiedCalls);
    const u64 verify_skipped = take(Counter::VerifySkipped);
    const u64 call_mismatches = take(Counter::CallMismatches);
    const u64 writes_checked = take(Counter::WritesChecked);
    const u64 write_mismatches = take(Counter::WriteMismatches);
    LOG_INFO(HW_GPU,
             "Macro batch {:.1f} s: calls 3D {} compute {} ({} skipped FrontRunOnBack); batches "
             "{} with {} writes; plain 3D writes {} sent, {} same value (not sent), {} other 3D "
             "writes to FrontWrite; verify: {} calls run again ({} not: DMA reads), {} "
             "mismatches; {} plain writes checked, {} mismatches",
             seconds, calls, compute_calls, fast_calls, flushes, buffered, plain_sent, plain_same,
             other_writes, verified, verify_skipped, call_mismatches, writes_checked,
             write_mismatches);
}

#ifdef ARCHITECTURE_x86_64
/// EDEN_NX2_MACRO_VERIFY: what one batched call did, and the copies to run it again from.
class CallCheck {
public:
    using Word = Tegra::TuringMacroJit::Word;

    CallCheck(u32 start_, std::span<const Word> parameters_, std::span<const Word> data_ram_)
        : start{start_}, parameters(parameters_.begin(), parameters_.end()),
          data_ram(data_ram_.begin(), data_ram_.end()) {}

    void NoteRead(Word address, Word value) {
        reads.emplace_back(address, value);
    }
    void NoteWrite(Word method, Word value) {
        writes.emplace_back(method, value);
        dma = dma || method == 0x157 || method == 0x158;
    }
    void NoteWrites(std::span<const Word> pairs) {
        for (std::size_t i = 0; i + 1 < pairs.size(); i += 2) {
            NoteWrite(pairs[i], pairs[i + 1]);
        }
    }

    /// Runs the call again with the unbatched program and compares; `data_ram_after` is the data
    /// RAM the batched call left, `macro` its method (for the log).
    void Compare(Tegra::TuringMacroJit& jit, std::span<const Word> code, u64 generation,
                 const Tegra::TuringMacroJit::Result& batched, std::span<const Word> data_ram_after,
                 u32 macro) {
        if (dma) {
            Count(Counter::VerifySkipped);
            return;
        }
        Count(Counter::VerifiedCalls);
        jit.Prepare(code, generation);
        std::vector<std::pair<Word, Word>> replayed;
        std::size_t next_read = 0;
        bool read_differs = false;
        std::vector<Word> fifo = parameters;
        const auto result = jit.Execute(
            start, fifo, std::span<Word>{data_ram},
            [&](Word address) -> Word {
                if (next_read >= reads.size() || reads[next_read].first != address) {
                    read_differs = true;
                    return 0;
                }
                return reads[next_read++].second;
            },
            [&](Word method, Word value) {
                replayed.emplace_back(method, value);
                return method < 0xe00;
            });
        const bool same_ram = std::ranges::equal(data_ram, data_ram_after);
        if (!read_differs && next_read == reads.size() && replayed == writes &&
            result.success == batched.success && result.consumed == batched.consumed &&
            result.error == batched.error && same_ram) {
            return;
        }
        Count(Counter::CallMismatches);
        static u32 logged{};
        if (logged >= 32) {
            return;
        }
        ++logged;
        std::size_t first = 0;
        while (first < replayed.size() && first < writes.size() &&
               replayed[first] == writes[first]) {
            ++first;
        }
        const auto pair_text = [](const std::vector<std::pair<Word, Word>>& list,
                                  std::size_t index) {
            return index < list.size()
                       ? fmt::format("{:#x}={:#x}", list[index].first, list[index].second)
                       : std::string{"none"};
        };
        LOG_ERROR(HW_GPU,
                  "Macro batch verify mismatch: macro {:#x} start {:#x} parameters {:08x}: {} "
                  "writes batched, {} unbatched, first difference at {} ({} / {}); reads {} of "
                  "{}{}; result {} {} '{}' / {} {} '{}'; data RAM {}",
                  macro, start,
                  fmt::join(std::span<const Word>{parameters}.first(
                                std::min<std::size_t>(parameters.size(), 8)),
                            ","),
                  writes.size(), replayed.size(), first, pair_text(writes, first),
                  pair_text(replayed, first), next_read, reads.size(),
                  read_differs ? " (other address)" : "", batched.success, batched.consumed,
                  batched.error, result.success, result.consumed, result.error,
                  same_ram ? "same" : "differs");
    }

private:
    u32 start;
    std::vector<Word> parameters;
    std::vector<Word> data_ram;
    std::vector<std::pair<Word, Word>> reads;
    std::vector<std::pair<Word, Word>> writes;
    bool dma{};
};
#endif

} // namespace VideoCommon::Nx2Macro
