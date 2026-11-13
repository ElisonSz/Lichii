// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/uniform_stream_reuse.h"

namespace {
void Check(bool condition, const char* message) {
    INFO(message);
    REQUIRE(condition);
}

struct Memory {
    std::array<u8, 8192> bytes{};
    u32 gathered{}; ///< Reads through ReadBlockUnsafe
    u32 in_place{}; ///< Ranges GetSpan handed out
    void ReadBlockUnsafe(u64 address, void* destination, u32 size) {
        Check(address + size <= bytes.size(), "read outside guest memory");
        std::memcpy(destination, bytes.data() + address, size);
        ++gathered;
    }
    /// As the device memory manager: a range in one piece of host memory, here one that doesn't
    /// cross a 4 KiB page (as if the pages after it were mapped elsewhere).
    u8* GetSpan(u64 address, size_t size) {
        if (address / 4096 != (address + size - 1) / 4096) {
            return nullptr;
        }
        ++in_place;
        return bytes.data() + address;
    }
};

struct Runtime {
    struct StreamUniformRef {
        size_t allocation;
    };
    std::vector<std::vector<u8>> allocations;
    StreamUniformRef last{}, bound{};
    u64 tick{1};
    u64 CurrentTick() const {
        return tick;
    }
    StreamUniformRef LastStreamUniformRef() const {
        return last;
    }
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        allocations.emplace_back(size);
        last = bound = StreamUniformRef{allocations.size() - 1};
        return allocations.back();
    }
    void RebindStreamUniform(StreamUniformRef ref, u32 size) {
        Check(ref.allocation < allocations.size(), "reused a recycled allocation");
        Check(allocations[ref.allocation].size() == size, "reused wrong allocation size");
        bound = ref;
    }
    void NextTick() {
        ++tick;
        allocations.clear();
    }
};

void CheckBound(const Runtime& runtime, const Memory& memory, u64 address, u32 size) {
    const auto& bytes = runtime.allocations.at(runtime.bound.allocation);
    Check(bytes.size() == size &&
              std::memcmp(bytes.data(), memory.bytes.data() + address, size) == 0,
          "bound uniform data differs from guest memory");
}
} // namespace

TEST_CASE("Uniform stream reuse preserves data through mutation and session resets",
          "[video_core]") {
    using VideoCommon::BindSmallUniformReusing;
    Memory memory;
    Runtime runtime;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    auto bind = [&](size_t stage, u32 slot, u64 address, u32 size) {
        Check(BindSmallUniformReusing(runtime, memory, stage, slot, slot, address, size),
              "valid small uniform rejected");
        CheckBound(runtime, memory, address, size);
    };
    bind(0, 0, 0, 64);
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 1, "identical bytes were copied twice");
    memory.bytes[7] = 23;
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 2, "changed bytes reused stale data");
    bind(0, 1, 0, 64);
    bind(1, 0, 0, 64);
    Check(runtime.allocations.size() == 4, "stage or slot shared the wrong allocation");
    runtime.NextTick();
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 1, "a new command buffer reused an old allocation");
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 2, "session reset retained an old allocation");
    bind(4, 17, 0, 4096);
    const auto before = runtime.allocations.size();
    for (auto [stage, slot, size] :
         {std::array<u32, 3>{5, 0, 64}, {0, 18, 64}, {0, 0, 0}, {0, 0, 4097}}) {
        Check(!BindSmallUniformReusing(runtime, memory, stage, slot, slot, 0, size),
              "invalid uniform range did not use the fallback");
    }
    Check(runtime.allocations.size() == before, "fallback allocated stream memory");

    u32 random = 17;
    for (u32 index = 0; index < 10000; ++index) {
        random = random * 1664525U + 1013904223U;
        if (index % 37 == 0)
            runtime.NextTick();
        if (index % 503 == 0)
            VideoCommon::ResetUniformStreamReuse<Runtime>();
        const u32 size = 1 + random % 4096;
        const u64 address = (random >> 12) % (memory.bytes.size() - 4096);
        if (index % 3 == 0)
            memory.bytes[address + random % size] ^= u8{0x5a};
        const size_t stage = (random >> 20) % 5;
        const u32 slot = (random >> 24) % 18;
        bind(stage, slot, address, size);
        bind(stage, slot, address, size);
    }
    Check(memory.gathered > 1000 && memory.in_place > 1000,
          "the reads did not take both the gathered and the in-place path");
}

TEST_CASE("Uniform keep audit sorts keeps as RebindKeptUniform treats them", "[video_core]") {
    using VideoCommon::AuditKeptUniform;
    using VideoCommon::BindSmallUniformReusing;
    using VideoCommon::KeptUniform;
    using VideoCommon::RebindKeptUniform;
    Memory memory;
    Runtime runtime;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    const auto audit = [&](size_t stage, u32 slot, u64 address, u32 size) {
        return AuditKeptUniform(runtime, memory, stage, slot, address, size);
    };
    const auto stream = [&](size_t stage, u32 slot, u64 address, u32 size) {
        Check(BindSmallUniformReusing(runtime, memory, stage, slot, slot, address, size),
              "valid small uniform rejected");
    };

    // Nothing streamed yet, invalid slots and sizes: refused, as the keep refuses them.
    Check(audit(0, 3, 0x100, 0x40).result == KeptUniform::RefusedSize, "unstreamed slot kept");
    Check(!RebindKeptUniform(runtime, 0, 3, 0x100, 0x40), "unstreamed slot rebound");
    for (auto [stage, slot, size] :
         {std::array<u32, 3>{5, 0, 64}, {0, 18, 64}, {0, 3, 0}, {0, 3, 4097}}) {
        Check(audit(stage, slot, 0x100, size).result == KeptUniform::RefusedSize,
              "invalid range not refused");
    }

    memory.bytes[0x100] = 1;
    stream(0, 3, 0x100, 0x40);
    Check(audit(0, 3, 0x100, 0x40).result == KeptUniform::Held, "unchanged range not held");
    Check(RebindKeptUniform(runtime, 0, 3, 0x100, 0x40), "held range not rebound");

    // Another pipeline's size, another address: refused whatever memory holds (the verify before
    // the audit counted these as changed ranges).
    Check(audit(0, 3, 0x100, 0x30).result == KeptUniform::RefusedSize, "other size kept");
    Check(!RebindKeptUniform(runtime, 0, 3, 0x100, 0x30), "other size rebound");
    Check(audit(0, 3, 0x200, 0x40).result == KeptUniform::RefusedAddress, "other address kept");
    Check(!RebindKeptUniform(runtime, 0, 3, 0x200, 0x40), "other address rebound");

    // A write nothing reported: the keep would bind stale bytes.
    memory.bytes[0x117] = 0xab;
    const auto stale = audit(0, 3, 0x100, 0x40);
    Check(stale.result == KeptUniform::Stale, "changed range not stale");
    Check(stale.offset == 0x17 && stale.was == 0 && stale.now == 0xab000000U,
          "stale byte or words wrong");
    Check(RebindKeptUniform(runtime, 0, 3, 0x100, 0x40), "the keep rule changed");

    // Another command buffer: refused, and still told apart when the bytes changed too.
    runtime.NextTick();
    Check(audit(0, 3, 0x100, 0x40).result == KeptUniform::RefusedTickStale,
          "changed range of another command buffer not told apart");
    memory.bytes[0x117] = 0;
    Check(audit(0, 3, 0x100, 0x40).result == KeptUniform::RefusedTick,
          "unchanged range of another command buffer not refused");
    Check(!RebindKeptUniform(runtime, 0, 3, 0x100, 0x40),
          "range of another command buffer rebound");

    // A last word shorter than four bytes.
    stream(1, 2, 0x300, 6);
    memory.bytes[0x305] = 0x7f;
    const auto tail = audit(1, 2, 0x300, 6);
    Check(tail.result == KeptUniform::Stale && tail.offset == 5 && tail.was == 0 &&
              tail.now == 0x7f00U,
          "short last word compared wrong");

    // Random binds, writes and command buffers: bound keeps are exactly the held and stale ones,
    // and held means the slot's bytes are guest memory's.
    u32 random = 91;
    for (u32 index = 0; index < 20000; ++index) {
        random = random * 1664525U + 1013904223U;
        if (index % 41 == 0) {
            runtime.NextTick();
        }
        const u32 size = 1 + (random >> 4) % 512;
        const u64 address = ((random >> 13) % 64) * 64;
        const size_t stage = (random >> 21) % 5;
        const u32 slot = (random >> 24) % 18;
        if (index % 3 == 0) {
            memory.bytes[address + (random >> 8) % size] ^= u8{0x24};
        }
        if (index % 2 == 0) {
            stream(stage, slot, address, size);
        }
        const auto check = audit(stage, slot, address, size);
        const bool bound = RebindKeptUniform(runtime, stage, slot, address, size);
        Check(bound == (check.result == KeptUniform::Held || check.result == KeptUniform::Stale),
              "audit and keep disagree on whether the range is bound");
        if (check.result == KeptUniform::Held) {
            CheckBound(runtime, memory, address, size);
        }
    }
}
