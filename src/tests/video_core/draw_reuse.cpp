// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Lane B (EDEN_NX2_DRAW_REUSE, EDEN_NX2_DRAW_VERTEX_INPUT, EDEN_NX2_DRAW_HANDLE_EVAL): the GPU-free
// rules in video_core/draw_delta.h and Shader::EvaluateHandleProgram.

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/shader_info.h"
#include "video_core/draw_delta.h"

namespace {
using Shader::HandleOp;
using VideoCommon::DrawHandleInputs;

constexpr u32 Op(HandleOp op) {
    return static_cast<u32>(op);
}

struct StorageDescriptor {
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 indirect_offset{~0U};
};

/// Guest memory by 32-bit word; words never written read as zero.
struct Memory {
    std::map<u64, u32> words;

    u32 Read32(u64 address) const {
        const auto it = words.find(address);
        return it == words.end() ? 0U : it->second;
    }
    u64 Read64(u64 address) const {
        return u64{Read32(address)} | (u64{Read32(address + 4)} << 32);
    }
    void Write32(u64 address, u32 value) {
        words[address] = value;
    }
    void Write64(u64 address, u64 value) {
        Write32(address, static_cast<u32>(value));
        Write32(address + 4, static_cast<u32>(value >> 32));
    }
};

/// splitmix64: consecutive draws are independent (a power-of-two LCG's low bits are not, and
/// left whole regions of the property test unvisited).
struct Random {
    u64 state;
    u32 Next() {
        u64 z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return static_cast<u32>((z ^ (z >> 31)) >> 8);
    }
};

constexpr u64 StorageAlignment = 0x100;

/// One stage's bindings as the rasterizer sees them: constant buffers by slot, storage buffers
/// read through descriptors (address and size words in a constant buffer).
struct Stage {
    std::array<u64, DrawHandleInputs::NumCbufs> cbuf_address{};
    std::array<u32, DrawHandleInputs::NumCbufs> cbuf_size{};
    std::vector<StorageDescriptor> storage;

    /// What EvaluateTextureHandle reads (shader_handle_program.h): constant buffer words at the
    /// bound address, storage words at the descriptor's address aligned down.
    std::optional<u32> Evaluate(const Memory& memory, std::span<const u32> program) const {
        return Shader::EvaluateHandleProgram(
            program,
            [&](u32 index, u32 offset) -> std::optional<u32> {
                if (index >= cbuf_address.size() || cbuf_address[index] == 0) {
                    return 0U;
                }
                return memory.Read32(cbuf_address[index] + offset);
            },
            [&](u32 binding, u32 offset) -> std::optional<u32> {
                const u64 base = StorageBase(memory, binding);
                if (base == 0) {
                    return 0U;
                }
                return memory.Read32(base + offset);
            });
    }

    u64 StorageBase(const Memory& memory, u32 binding) const {
        if (binding >= storage.size() || cbuf_address[storage[binding].cbuf_index] == 0) {
            return 0;
        }
        const StorageDescriptor& descriptor = storage[binding];
        u64 base = memory.Read64(cbuf_address[descriptor.cbuf_index] + descriptor.cbuf_offset);
        if (base != 0 && descriptor.indirect_offset != ~0U) {
            base = memory.Read64(base + descriptor.indirect_offset);
        }
        return base & ~(StorageAlignment - 1);
    }

    /// The range the buffer cache binds for the descriptor (StorageBufferBinding): the address
    /// aligned down to the size word's end.
    std::pair<u64, u64> StorageRange(const Memory& memory, u32 binding) const {
        const StorageDescriptor& descriptor = storage[binding];
        const u64 word = cbuf_address[descriptor.cbuf_index] + descriptor.cbuf_offset;
        u64 address = memory.Read64(word);
        if (address != 0 && descriptor.indirect_offset != ~0U) {
            address = memory.Read64(address + descriptor.indirect_offset);
        }
        const u64 size = memory.Read32(word + 8);
        return {address & ~(StorageAlignment - 1), address + size};
    }
};

// Programs over the stage below (operands follow the operation, see Shader::HandleOp).
// A: the handle word at c2[0x10].
const std::vector<u32> program_cbuf{Op(HandleOp::Cbuf), 2, 0x10};
// B: a table of handles in storage buffer 0, indexed by the material word c3[0x20].
const std::vector<u32> program_table{
    Op(HandleOp::Cbuf),    3, 0x20, Op(HandleOp::Immediate), 4, Op(HandleOp::Mul),
    Op(HandleOp::Storage), 0};
// C: a word of c1 at an offset read from c3[0x24], or'd with the low bits of storage 1's word 8.
const std::vector<u32> program_dynamic{
    Op(HandleOp::Cbuf),        3, 0x24, Op(HandleOp::CbufDynamic), 1,
    Op(HandleOp::Immediate),   8, Op(HandleOp::Storage),           1,
    Op(HandleOp::Immediate),   0xff, Op(HandleOp::And),            Op(HandleOp::Or)};
} // namespace

TEST_CASE("Draw reuse finds the constant buffers and storage buffers handle programs read",
          "[draw_reuse]") {
    const std::vector<StorageDescriptor> storage{{0, 0x40}, {0, 0x50}, {4, 0x0, 0x18}};
    DrawHandleInputs inputs;
    inputs.AddProgram(0, program_cbuf, storage);
    CHECK(inputs.cbufs[0] == (1U << 2));
    CHECK(inputs.storage[0] == 0);
    CHECK_FALSE(inputs.any_write[0]);

    inputs.AddProgram(0, program_table, storage);
    // c3 for the index, c0 for the storage buffer's address.
    CHECK(inputs.cbufs[0] == ((1U << 2) | (1U << 3) | (1U << 0)));
    CHECK(inputs.storage[0] == 1);

    inputs.AddProgram(4, program_dynamic, storage);
    CHECK(inputs.cbufs[4] == ((1U << 3) | (1U << 1) | (1U << 0)));
    CHECK(inputs.storage[4] == 2);
    CHECK_FALSE(inputs.any_write[4]);

    SECTION("a pointer table makes any write reach the handles") {
        const std::vector<u32> indirect{Op(HandleOp::Immediate), 0, Op(HandleOp::Storage), 2};
        inputs.AddProgram(1, indirect, storage);
        CHECK(inputs.any_write[1]);
        CHECK(inputs.cbufs[1] == (1U << 4));
    }
    SECTION("unknown operations and cut operands take every slot") {
        inputs.AddProgram(1, std::vector<u32>{Op(HandleOp::Index) + 1}, storage);
        CHECK(inputs.any_write[1]);
        CHECK(inputs.cbufs[1] == (1U << DrawHandleInputs::NumCbufs) - 1);
        inputs.AddProgram(2, std::vector<u32>{Op(HandleOp::Cbuf), 2}, storage);
        CHECK(inputs.any_write[2]);
    }
    SECTION("reads that read nothing add nothing") {
        // A slot past the last and a binding past the descriptors evaluate to zero.
        const std::vector<u32> nothing{Op(HandleOp::Cbuf),      40, 0, Op(HandleOp::Immediate),
                                       0,                       Op(HandleOp::Storage), 9,
                                       Op(HandleOp::Add)};
        inputs.AddProgram(3, nothing, storage);
        CHECK(inputs.cbufs[3] == 0);
        CHECK(inputs.storage[3] == 0);
        CHECK_FALSE(inputs.any_write[3]);
    }
}

TEST_CASE("Draw reuse asks about memory only when data was written", "[draw_reuse]") {
    DrawHandleInputs inputs;
    inputs.cbufs[0] = 1U << 3;
    inputs.storage[4] = 1U << 1;
    std::array<bool, 5> stages{true, false, false, false, true};
    std::array<u32, 5> binds{};
    u32 cbuf_questions = 0;
    u32 storage_questions = 0;
    bool cbuf_written = false;
    bool storage_written = false;
    const auto reached = [&](bool data_written) {
        return VideoCommon::HandleInputsReached(
            inputs, stages, binds, data_written,
            [&](size_t stage, u32 slot) {
                ++cbuf_questions;
                CHECK(stage == 0);
                CHECK(slot == 3);
                return cbuf_written;
            },
            [&](size_t stage, u32 binding) {
                ++storage_questions;
                CHECK(stage == 4);
                CHECK(binding == 1);
                return storage_written;
            });
    };
    CHECK_FALSE(reached(false));
    CHECK(cbuf_questions == 0);
    CHECK_FALSE(reached(true));
    CHECK(cbuf_questions == 1);
    CHECK(storage_questions == 1);
    cbuf_written = true;
    CHECK(reached(true));
    cbuf_written = false;
    storage_written = true;
    CHECK(reached(true));
    storage_written = false;

    SECTION("a bind of a slot the handles read") {
        binds[0] = 1U << 3;
        CHECK(reached(false));
    }
    SECTION("binds of other slots and stages") {
        binds[0] = 1U << 2;
        binds[1] = 1U << 3; // stage 1 isn't configured
        CHECK_FALSE(reached(false));
    }
    SECTION("stages the pipeline doesn't configure are not asked") {
        stages[4] = false;
        storage_written = true;
        CHECK_FALSE(reached(true));
    }
    SECTION("any write reaches a stage reading through a pointer table") {
        inputs.any_write[4] = true;
        CHECK_FALSE(reached(false));
        CHECK(reached(true));
    }
}

TEST_CASE("Draw reuse: handles are unchanged whenever nothing reached their inputs",
          "[draw_reuse]") {
    // A stage reading handles three ways, then random binds and writes. Whenever the rule says
    // nothing reached the inputs, evaluating again must give the same handles.
    Stage stage;
    stage.storage = {{0, 0x40}, {0, 0x50}};
    const std::array<std::span<const u32>, 3> programs{program_cbuf, program_table,
                                                        program_dynamic};
    DrawHandleInputs inputs;
    for (const auto& program : programs) {
        inputs.AddProgram(0, program, stage.storage);
    }
    const std::array<bool, 5> stages{true, false, false, false, false};

    Memory memory;
    Random random{12345};
    const auto next = [&random] { return random.Next(); };
    // Constant buffers at 0x10000 + slot * 0x1000; storage buffers at 0x40000 and 0x48000.
    for (u32 slot = 0; slot < 5; ++slot) {
        stage.cbuf_address[slot] = 0x10000 + u64{slot} * 0x1000;
        stage.cbuf_size[slot] = 0x100;
    }
    const auto fill = [&] {
        // Writes to c0 can move the storage buffers: put them back now and then.
        memory.Write64(stage.cbuf_address[0] + 0x40, 0x40000);
        memory.Write32(stage.cbuf_address[0] + 0x48, 0x400);
        memory.Write64(stage.cbuf_address[0] + 0x50, 0x48020); // not aligned: read aligned down
        memory.Write32(stage.cbuf_address[0] + 0x58, 0x100);
        memory.Write32(stage.cbuf_address[2] + 0x10, next());
        memory.Write32(stage.cbuf_address[3] + 0x20, next() % 64);
        memory.Write32(stage.cbuf_address[3] + 0x24, (next() % 32) * 4);
        for (u32 word = 0; word < 0x100; word += 4) {
            memory.Write32(0x40000 + word, next());
            memory.Write32(stage.cbuf_address[1] + word, next());
        }
        memory.Write32(0x48000 + 8, next());
    };
    fill();
    const auto evaluate = [&] {
        std::array<std::optional<u32>, 3> handles;
        for (size_t index = 0; index < programs.size(); ++index) {
            handles[index] = stage.Evaluate(memory, programs[index]);
        }
        return handles;
    };

    u32 untouched = 0;
    u32 reached_count = 0;
    for (u32 iteration = 0; iteration < 20000; ++iteration) {
        const auto before = evaluate();
        // What the buffer cache holds from the previous draw's binds.
        const std::array<std::pair<u64, u64>, 2> storage_ranges{stage.StorageRange(memory, 0),
                                                                stage.StorageRange(memory, 1)};
        std::array<u32, 5> binds{};
        std::vector<std::pair<u64, u64>> writes; // [begin, end)
        const u32 events = next() % 3;
        for (u32 event = 0; event < events; ++event) {
            const u32 kind = next() % 8;
            if (kind == 0) {
                // Rebind a slot elsewhere, written first (a game double-buffering constants);
                // two slots may end up at the same address.
                const u32 slot = next() % 5;
                const u64 to = 0x30000 + u64{next() % 4} * 0x1000;
                for (u32 word = 0; word < 0x100; word += 4) {
                    memory.Write32(to + word, memory.Read32(stage.cbuf_address[slot] + word));
                }
                memory.Write32(to + (next() % 0x40) * 4, next());
                writes.emplace_back(to, to + 0x100);
                if (stage.cbuf_address[slot] != to) {
                    binds[0] |= 1U << slot;
                    stage.cbuf_address[slot] = to;
                }
                continue;
            }
            // Constant buffer data: a few words somewhere in the constant buffers, the storage
            // buffers or memory nothing reads.
            static constexpr std::array<u64, 6> regions{0x10000, 0x12000, 0x13000,
                                                        0x40000, 0x48000, 0x70000};
            const u64 begin = regions[next() % regions.size()] + (next() % 0x100) / 4 * 4;
            const u64 end = begin + 4 * (1 + next() % 4);
            for (u64 address = begin; address < end; address += 4) {
                memory.Write32(address, next());
            }
            writes.emplace_back(begin, end);
        }
        const auto overlaps = [&](u64 begin, u64 end) {
            for (const auto& [write_begin, write_end] : writes) {
                if (write_begin < end && begin < write_end) {
                    return true;
                }
            }
            return false;
        };
        // The rule sees the bindings the previous draw used (as the rasterizer does: the buffer
        // cache still holds them, and a write to the words giving them reaches a constant buffer).
        const bool reached = VideoCommon::HandleInputsReached(
            inputs, stages, binds, !writes.empty(),
            [&](size_t, u32 slot) {
                return overlaps(stage.cbuf_address[slot],
                                stage.cbuf_address[slot] + stage.cbuf_size[slot]);
            },
            [&](size_t, u32 binding) {
                return overlaps(storage_ranges[binding].first, storage_ranges[binding].second);
            });
        const auto after = evaluate();
        if (!reached) {
            ++untouched;
            REQUIRE(after == before);
        } else {
            ++reached_count;
        }
        if (next() % 16 == 0) {
            fill();
        }
    }
    // Both outcomes occur, so the property is exercised.
    CHECK(untouched > 1000);
    CHECK(reached_count > 1000);
}

TEST_CASE("Draw reuse backs off from checks that keep failing", "[draw_reuse]") {
    using Backoff = VideoCommon::DeltaCheckBackoff;
    Backoff backoff;
    for (u32 attempt = 0; attempt < Backoff::Threshold; ++attempt) {
        REQUIRE(backoff.ShouldTry());
        backoff.Note(false);
    }
    // One try in every Period while it keeps failing.
    u32 tries = 0;
    for (u32 attempt = 0; attempt < Backoff::Period * 10; ++attempt) {
        if (backoff.ShouldTry()) {
            ++tries;
            backoff.Note(false);
        }
    }
    CHECK(tries == 10);
    // A success restores every check.
    while (!backoff.ShouldTry()) {
    }
    backoff.Note(true);
    for (u32 attempt = 0; attempt < Backoff::Threshold; ++attempt) {
        CHECK(backoff.ShouldTry());
    }
}

TEST_CASE("Vertex input states are found by every register they are built from",
          "[draw_reuse]") {
    struct State {
        u32 built_from;
    };
    VideoCommon::VertexInputCache<State> cache;
    VideoCommon::VertexInputKey key{.num_attributes = 32, .num_bindings = 32};
    key.attributes[0] = 0x3e400000;
    key.strides[0] = 16;
    CHECK(cache.Find(key, key.Hash()) == nullptr);
    cache.Insert(key, key.Hash(), std::make_shared<const State>(State{1}));
    const auto found = cache.Find(key, key.Hash());
    REQUIRE(found != nullptr);
    CHECK(found->built_from == 1);

    const auto differs = [&](auto&& change) {
        VideoCommon::VertexInputKey other = key;
        change(other);
        return !(other == key) && cache.Find(other, other.Hash()) == nullptr;
    };
    CHECK(differs([](auto& k) { k.attributes[31] = 1; }));
    CHECK(differs([](auto& k) { k.strides[5] = 4; }));
    CHECK(differs([](auto& k) { k.divisors[0] = 2; }));
    CHECK(differs([](auto& k) { k.instanced = 1U << 7; }));
    CHECK(differs([](auto& k) { k.repacked = 1U; }));
    CHECK(differs([](auto& k) { k.num_attributes = 16; }));
    CHECK(differs([](auto& k) { k.num_bindings = 16; }));

    SECTION("a replaced entry stays alive for the commands that recorded it") {
        // Find colliding keys: same slot, different registers.
        VideoCommon::VertexInputKey other = key;
        for (u32 value = 1;; ++value) {
            other.attributes[1] = value;
            if (other.Hash() % VideoCommon::VertexInputCache<State>::NumEntries ==
                key.Hash() % VideoCommon::VertexInputCache<State>::NumEntries) {
                break;
            }
        }
        cache.Insert(other, other.Hash(), std::make_shared<const State>(State{2}));
        CHECK(cache.Find(key, key.Hash()) == nullptr);
        CHECK(found->built_from == 1);
        CHECK(cache.Find(other, other.Hash())->built_from == 2);
    }
}

namespace {
/// Fills the stack the evaluator's arrays will occupy with non-zero bytes, so a read of an element
/// it never wrote would differ between the zeroing and the lean evaluation.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void DirtyStack(u8 value) {
    volatile u8 junk[16384];
    for (size_t index = 0; index < sizeof(junk); ++index) {
        junk[index] = static_cast<u8>(value + index);
    }
}
} // namespace

TEST_CASE("Handle programs give the same results with and without zeroing their stack",
          "[draw_reuse]") {
    // Random programs, valid and malformed, over random constant buffer and storage words.
    Random random{99};
    const auto next = [&random] { return random.Next(); };
    const auto read_cbuf = [&](u32 index, u32 offset) -> std::optional<u32> {
        if (index == 7) {
            return std::nullopt; // a failed read
        }
        return index * 0x10000 + offset;
    };
    const auto read_storage = [&](u32 binding, u32 offset) -> std::optional<u32> {
        return (binding << 24) ^ (offset * 0x9E3779B9U);
    };
    // Binary operations, by code.
    std::vector<u32> binary;
    for (u32 code = Op(HandleOp::Add); code <= Op(HandleOp::LXor); ++code) {
        if (code != Op(HandleOp::Neg) && code != Op(HandleOp::Not) && code != Op(HandleOp::LNot)) {
            binary.push_back(code);
        }
    }
    u32 valid_results = 0;
    for (u32 iteration = 0; iteration < 20000; ++iteration) {
        // Mostly well-formed programs (tracking the stack depth), some broken on purpose.
        std::vector<u32> program;
        u32 depth = 0;
        u32 teed = 0; // slots 0..teed-1 are written
        const u32 length = 1 + next() % 24;
        for (u32 op = 0; op < length; ++op) {
            const u32 choice = next() % 100;
            if (choice < 3) {
                program.push_back(next() % (Op(HandleOp::Index) + 2)); // anything
                continue;
            }
            if (depth == 0 || choice < 35) {
                switch (next() % 4) {
                case 0:
                    program.insert(program.end(), {Op(HandleOp::Immediate), next()});
                    break;
                case 1:
                    program.insert(program.end(), {Op(HandleOp::Cbuf), next() % 8, next() % 0x100});
                    break;
                case 2:
                    program.push_back(Op(HandleOp::Index));
                    break;
                default:
                    // Sometimes a slot never written, which fails the program.
                    program.insert(program.end(), {Op(HandleOp::Load), teed == 0 || next() % 8 == 0
                                                                           ? next() % 256
                                                                           : next() % teed});
                    break;
                }
                ++depth;
                continue;
            }
            if (choice < 55) {
                static constexpr std::array<HandleOp, 5> unary{HandleOp::Neg, HandleOp::Not,
                                                               HandleOp::LNot,
                                                               HandleOp::CbufDynamic,
                                                               HandleOp::Storage};
                const HandleOp code = unary[next() % unary.size()];
                program.push_back(Op(code));
                if (code == HandleOp::CbufDynamic || code == HandleOp::Storage) {
                    program.push_back(next() % 8);
                }
                continue;
            }
            if (choice < 65) {
                const u32 slot = next() % 8;
                program.insert(program.end(), {Op(HandleOp::Tee), slot});
                teed = std::max(teed, slot + 1);
                continue;
            }
            if (depth >= 3 && choice < 72) {
                program.push_back(Op(HandleOp::Select));
                depth -= 2;
                continue;
            }
            if (depth >= 2) {
                program.push_back(binary[next() % binary.size()]);
                --depth;
            }
        }
        for (; depth > 1; --depth) {
            program.push_back(binary[next() % binary.size()]);
        }
        const u32 element = next() % 4;
        Shader::g_lean_handle_programs = false;
        DirtyStack(static_cast<u8>(next()));
        const auto zeroed = Shader::EvaluateHandleProgram(program, read_cbuf, read_storage, element);
        Shader::g_lean_handle_programs = true;
        DirtyStack(static_cast<u8>(next()));
        const auto lean = Shader::EvaluateHandleProgram(program, read_cbuf, read_storage, element);
        Shader::g_lean_handle_programs = false;
        REQUIRE(zeroed == lean);
        valid_results += zeroed.has_value() ? 1 : 0;
    }
    CHECK(valid_results > 500);
}
