// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// TuringMacroJit::ExecuteBatched (EDEN_NX2_MACRO_BATCH) against Execute: the same writes, reads,
// FIFO, data RAM and result for every program and parameter list.

#include <array>
#include <initializer_list>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"

#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"

namespace {

/// A Turing MME instruction from its fields (TuringMacroInterpreter::Decode).
struct Mme {
    bool end{};
    u32 pred_mode{};
    u32 pred_reg{};
    std::array<u32, 2> op{}, dst{24, 24}, src_a{24, 24}, src_b{24, 24}, imm{};
    std::array<u32, 2> out_method{}, out_emit{};
};

std::vector<u32> Code(std::span<const Mme> instructions) {
    std::vector<u32> code;
    for (const Mme& f : instructions) {
        std::array<u32, 3> bits{};
        const auto field = [&bits](u32 offset, u32 count, u32 value) {
            for (u32 bit = 0; bit < count; ++bit) {
                bits[(offset + bit) / 32] |= ((value >> bit) & 1) << ((offset + bit) % 32);
            }
        };
        field(0, 1, f.end ? 1 : 0);
        field(1, 4, f.pred_mode);
        field(5, 5, f.pred_reg);
        for (u32 lane = 0; lane < 2; ++lane) {
            field(10 + lane * 36, 5, f.op[lane]);
            field(15 + lane * 36, 5, f.dst[lane]);
            field(20 + lane * 36, 5, f.src_a[lane]);
            field(25 + lane * 36, 5, f.src_b[lane]);
            field(30 + lane * 36, 16, f.imm[lane]);
            field(82 + lane * 7, 3, f.out_method[lane]);
            field(85 + lane * 7, 4, f.out_emit[lane]);
        }
        code.insert(code.end(), {bits[2], bits[1], bits[0]});
    }
    return code;
}

using DataRam = std::array<u32, 0xc00>;

/// What a call did, as its caller saw it.
struct Run {
    std::vector<std::pair<u32, u32>> writes;
    std::vector<u32> reads;
    std::vector<u32> fifo;
    DataRam ram{};
    bool success{};
    u32 pc{};
    std::size_t consumed{};
    std::string error;
    /// Batched: pairs flushed that Execute would have emitted through `emit` at once.
    u32 bad_flushes{};
    u32 flushes{};

    bool operator==(const Run& other) const {
        return writes == other.writes && reads == other.reads && fifo == other.fifo &&
               ram == other.ram && success == other.success && pc == other.pc &&
               consumed == other.consumed && error == other.error;
    }
};

/// A caller whose effects show the order of everything: a register read returns a value of the
/// writes it saw, MME_DMA_READ_FIFOED (0x158) appends words of them to the FIFO, MME_DMA_READ
/// (0x157) stores one into the data RAM, and methods from 0xe00 on are refused.
Run Call(Tegra::TuringMacroJit& jit, bool batched, u32 start, std::vector<u32> parameters,
         const DataRam& ram) {
    Run run;
    run.ram = ram;
    const auto apply = [&](u32 method, u32 value) {
        if (method >= 0xe00) {
            return false;
        }
        run.writes.emplace_back(method, value);
        const u32 seen = static_cast<u32>(run.writes.size());
        if (method == 0x158) {
            for (u32 i = 0; i < (value & 7); ++i) {
                parameters.push_back(seen * 16 + i);
            }
        } else if (method == 0x157) {
            run.ram[(value * 7 + seen) % run.ram.size()] = seen * 3 + 1;
        }
        return true;
    };
    const auto read = [&](u32 address) {
        run.reads.push_back(address);
        return address * 31 + static_cast<u32>(run.writes.size());
    };
    Tegra::TuringMacroJit::Result result;
    if (batched) {
        result = jit.ExecuteBatched(start, parameters, run.ram, read, apply,
                                    [&](std::span<const u32> pairs) {
                                        ++run.flushes;
                                        for (std::size_t i = 0; i + 1 < pairs.size(); i += 2) {
                                            const u32 method = pairs[i];
                                            if (method >= 0xe00 || method == 0x157 ||
                                                method == 0x158 || !apply(method, pairs[i + 1])) {
                                                ++run.bad_flushes;
                                            }
                                        }
                                    });
    } else {
        result = jit.Execute(start, parameters, run.ram, read, apply);
    }
    run.success = result.success;
    run.pc = result.pc;
    run.consumed = result.consumed;
    run.error = std::string{result.error};
    run.fifo = parameters;
    return run;
}

/// Sends the next parameter to `method` (an immediate with its increment in bits 12-15).
Mme Fifo(u32 method, bool end = false) {
    return Mme{.end = end, .imm = {method, 0}, .out_method = {5, 0}, .out_emit = {3, 0}};
}

/// 32-bit random words (std::mt19937's result type is wider on some hosts).
struct Rng {
    std::mt19937 engine;
    u32 operator()() {
        return static_cast<u32>(engine());
    }
};

u32 Pick(Rng& rng, std::initializer_list<u32> values) {
    return *(values.begin() + rng() % values.size());
}

/// A method address for an immediate: plain, DMA reads, refused, with an increment.
u32 RandomMethod(Rng& rng) {
    const u32 method = rng() % 4 != 0 ? 0x100 + rng() % 0x40
                                      : Pick(rng, {0x157, 0x158, 0xdfe, 0xdff, 0xe00, 0xe02});
    return method | ((rng() % 3) << 12);
}

Mme RandomInstruction(Rng& rng) {
    Mme f;
    f.end = rng() % 6 == 0;
    if (rng() % 3 == 0) {
        f.pred_mode = rng() % 16;
        f.pred_reg = rng() % 5 == 0 ? 24 + rng() % 6 : rng() % 8;
    }
    for (u32 lane = 0; lane < 2; ++lane) {
        const u32 kind = rng() % 100;
        u32 op;
        if (kind < 40) {
            op = Pick(rng, {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15});
        } else if (kind < 50) {
            op = 16;
        } else if (kind < 60) {
            op = 17 + rng() % 5;
        } else if (kind < 68) {
            op = 22; // MME_READ
        } else if (kind < 71) {
            op = 23; // LOOP
        } else if (kind < 74) {
            op = 24; // JAL
        } else if (kind < 84) {
            op = 25 + rng() % 5; // branches
        } else if (kind < 97) {
            op = 30 + rng() % 2; // DREAD / DWRITE
        } else {
            op = 7; // EXTENDED
        }
        f.op[lane] = op;
        f.dst[lane] = rng() % 4 == 0 ? 24 : rng() % 8;
        const auto source = [&rng] {
            const u32 kind_of_source = rng() % 10;
            if (kind_of_source < 5) {
                return rng() % 8;
            }
            if (kind_of_source < 8) {
                return 24 + rng() % 4; // zero, immediates
            }
            return 28 + rng() % 2; // FIFO loads
        };
        f.src_a[lane] = source();
        f.src_b[lane] = source();
        switch (op) {
        case 16:
            f.imm[lane] = (rng() % 33 << 10) | (rng() % 32 << 5) | (rng() % 32);
            break;
        case 23:
            f.imm[lane] = rng() % 6;
            break;
        case 24:
            f.imm[lane] = 1 + rng() % 4;
            break;
        case 25:
        case 26:
        case 27:
        case 28:
        case 29: {
            const u32 offset = rng() % 16 == 0 ? (0x1000 & 0x1fff) : (rng() % 9 - 4) & 0x1fff;
            f.imm[lane] = offset | (rng() % 2 << 14) | (rng() % 2 << 15);
            break;
        }
        case 7:
            f.imm[lane] = rng() % 2 == 0 ? 0x1000 : RandomMethod(rng);
            break;
        default:
            f.imm[lane] = rng() % 3 == 0 ? rng() % 0x10000 : RandomMethod(rng);
            break;
        }
        if (op == 7 && rng() % 2 == 0) {
            f.src_a[lane] = 25; // its immediate: the load barrier (0x1000) or not
        }
        if ((op == 30 || op == 31) && rng() % 4 != 0) {
            f.src_a[lane] = 25; // a data RAM address in range, mostly
            f.imm[lane] = rng() % 0xc00;
        }
        if (rng() % 3 == 0) {
            f.out_method[lane] = Pick(rng, {1, 2, 3, 4, 5, 6, 8, 9, 10});
        }
        if (rng() % 2 == 0) {
            // Now and then an invalid selector (7, 11), which fails the call.
            f.out_emit[lane] =
                rng() % 20 == 0 ? Pick(rng, {7, 11}) : Pick(rng, {1, 2, 3, 4, 5, 6, 8, 9, 10});
        }
    }
    return f;
}

void Compare(Tegra::TuringMacroJit& jit, u32 start, const std::vector<u32>& parameters,
             const DataRam& ram) {
    const Run expected = Call(jit, false, start, parameters, ram);
    const Run batched = Call(jit, true, start, parameters, ram);
    INFO("start " << start << ", " << parameters.size() << " parameters, " << expected.writes.size()
                  << " writes, error '" << expected.error << "' / '" << batched.error << "'");
    CHECK(batched.bad_flushes == 0);
    CHECK(batched == expected);
}

} // Anonymous namespace

TEST_CASE("Batched native macro calls write what unbatched ones do", "[video_core][native_macro]") {
    Rng rng{std::mt19937{0x5eed}};
    DataRam ram{};
    for (u32 i = 0; i < ram.size(); ++i) {
        ram[i] = i * 0x9e3779b9U;
    }
    for (u32 program = 0; program < 1500; ++program) {
        std::vector<Mme> instructions(2 + rng() % 24);
        for (Mme& instruction : instructions) {
            instruction = RandomInstruction(rng);
        }
        const auto code = Code(instructions);
        Tegra::TuringMacroJit jit;
        jit.Prepare(code, program);
        jit.PrepareBatched(code, program);
        for (u32 call = 0; call < 6; ++call) {
            std::vector<u32> parameters(rng() % 24);
            for (u32& word : parameters) {
                word = rng() % 4 == 0 ? rng() : RandomMethod(rng);
            }
            const u32 start = call < 3 ? 0 : rng() % static_cast<u32>(instructions.size() + 1);
            Compare(jit, start, parameters, ram);
        }
    }
}

TEST_CASE("Batched native macro calls: full buffers, reads and DMA reads between writes",
          "[video_core][native_macro]") {
    // A loop of FIFO writes longer than the buffer, then a register read, a DMA read into the
    // FIFO and writes of what it read.
    std::vector<Mme> instructions;
    // r1 = parameter 0 (the count), then LOOP r1 over one instruction sending parameters.
    instructions.push_back(Mme{.op = {0, 0}, .dst = {1, 24}, .src_a = {28, 24}, .src_b = {24, 24}});
    instructions.push_back(Mme{.op = {23, 0}, .src_a = {1, 24}, .imm = {2, 0}});
    instructions.push_back(Fifo(0x1100)); // method 0x100, increment 1
    // r2 = MME_READ(0x200)
    instructions.push_back(Mme{
        .op = {22, 0}, .dst = {2, 24}, .src_a = {25, 24}, .src_b = {24, 24}, .imm = {0x200, 0}});
    // emit r2 to 0x300, then a DMA read of 3 words into the FIFO, then two of them out.
    instructions.push_back(Mme{.op = {0, 0},
                               .dst = {24, 24},
                               .src_a = {2, 24},
                               .src_b = {24, 24},
                               .imm = {0x300, 0},
                               .out_method = {5, 0},
                               .out_emit = {1, 0}});
    instructions.push_back(Mme{.op = {0, 0},
                               .dst = {24, 24},
                               .src_a = {24, 24},
                               .src_b = {24, 24},
                               .imm = {0x158, 3},
                               .out_method = {5, 0},
                               .out_emit = {6, 0}});
    instructions.push_back(Fifo(0x1310));
    instructions.push_back(Fifo(0x1310, true));
    instructions.push_back(Fifo(0x1310));
    const auto code = Code(instructions);
    Tegra::TuringMacroJit jit;
    jit.Prepare(code, 1);
    jit.PrepareBatched(code, 1);
    const DataRam ram{};
    for (const u32 count : {0U, 1U, 255U, 256U, 257U, 600U}) {
        std::vector<u32> parameters{count};
        for (u32 i = 0; i < count; ++i) {
            parameters.push_back(i * 3);
        }
        const Run expected = Call(jit, false, 0, parameters, ram);
        const Run batched = Call(jit, true, 0, parameters, ram);
        INFO("count " << count);
        REQUIRE(expected.success);
        CHECK(expected.writes.size() == count + 5);
        CHECK(batched.bad_flushes == 0);
        CHECK(batched == expected);
        // The read saw every write before it: its value counts them.
        CHECK(expected.writes[count] == std::pair<u32, u32>{0x300, 0x200 * 31 + count});
        // Flushes: before the read, the DMA read, and every full buffer.
        CHECK(batched.flushes >= count / Tegra::TuringMacroJit::BatchCapacity);
    }
}
#endif
