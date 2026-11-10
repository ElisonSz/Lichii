// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::SemRecorder {

// CPU slow stores and scalar GPU memory accesses only need a load on the disabled path.
inline const bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_ACQUIRE_VERIFY");
    return value != nullptr && value[0] == '1';
}();

inline bool Enabled() {
    return g_enabled;
}

// All addresses are device addresses, except gpu (an optional GPU VA). Range events retain
// their extent so a download can be found even when it began before the semaphore word.
struct Event {
    const char* kind{};
    u64 address{};
    u64 size{4};
    u64 gpu{};
    u64 pair{};
    u64 value{};
    u64 before{};
    u64 after{};
    u64 position{};
    u64 entry{};
    u64 consumed{};
    u64 fence{};
    u64 flags{};
    u64 sequence{};
    u64 timestamp{};
    u64 thread{};
};

inline constexpr size_t Capacity = 65536;
struct Slot {
    std::atomic_flag busy{};
    Event event{};
};
struct State {
    std::array<Slot, Capacity> slots;
    std::atomic<u64> next{1};
    std::atomic<u64> threads{1};
    std::atomic<u64> dropped{};
    std::atomic<u32> dumps{};
    std::array<std::atomic<u64>, 4096> words{};
    std::atomic<u64> untracked{};
};

inline State& Get() {
    // No ring allocation or initialization in ordinary runs.
    static auto state = std::make_unique<State>();
    return *state;
}

inline void Track(u64 address) {
    if (!Enabled() || address == 0) {
        return;
    }
    auto& state = Get();
    const u64 word = address & ~u64{3};
    for (size_t probe = 0; probe < state.words.size(); ++probe) {
        auto& cell = state.words[((word >> 2) + probe) % state.words.size()];
        u64 empty = 0;
        if (cell.load(std::memory_order_relaxed) == word ||
            cell.compare_exchange_strong(empty, word, std::memory_order_relaxed) || empty == word) {
            return;
        }
    }
    state.untracked.fetch_add(1, std::memory_order_relaxed);
}

inline void Add(Event event) {
    if (!Enabled()) {
        return;
    }
    auto& state = Get();
    static thread_local const u64 thread_id = state.threads.fetch_add(1, std::memory_order_relaxed);
    event.thread = thread_id;
    event.timestamp = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count());
    event.sequence = state.next.fetch_add(1, std::memory_order_relaxed);
    auto& slot = state.slots[event.sequence % Capacity];
    // Never wait behind a preempted producer (including one a full wrap behind). Readers use
    // the same guard: copying non-atomic event fields without it would be a C++ data race.
    if (slot.busy.test_and_set(std::memory_order_acquire)) {
        state.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (slot.event.sequence < event.sequence) {
        slot.event = event;
    } else {
        state.dropped.fetch_add(1, std::memory_order_relaxed);
    }
    slot.busy.clear(std::memory_order_release);
}

inline void Record(const char* kind, u64 address, u64 value = 0, u64 before = 0, u64 after = 0,
                   u64 flags = 0) {
    Add({.kind = kind,
         .address = address,
         .value = value,
         .before = before,
         .after = after,
         .flags = flags});
}

// Input bytes only: never sample guest memory here, since a diagnostic read could race a writer
// or trigger a download. Emit the complete supplied words and the exact extent of partial ones.
inline void Bytes(const char* kind, u64 address, const void* bytes, u64 size) {
    if (!Enabled() || size == 0) {
        return;
    }
    const auto* data = static_cast<const u8*>(bytes);
    for (u64 offset = 0; offset < size;) {
        const u64 count = (std::min)(u64{4} - ((address + offset) & 3), size - offset);
        u32 value{};
        if (data != nullptr) {
            std::memcpy(&value, data + offset, static_cast<size_t>(count));
        }
        Add({.kind = kind, .address = address + offset, .size = count, .value = value});
        offset += count;
    }
}

// Bulk copies and CPU slow-path stores can be enormous. Retain their range unconditionally,
// and supplied values for words already identified by a semaphore read, note or report.
inline void TrackedBytes(const char* kind, u64 address, const void* bytes, u64 size,
                         bool record_range = true) {
    if (!Enabled() || size == 0) {
        return;
    }
    if (record_range) {
        Add({.kind = kind, .address = address, .size = size});
    }
    const auto* data = static_cast<const u8*>(bytes);
    if (size <= 16) {
        const u64 first = address & ~u64{3};
        for (u64 word = first; word - first < size + (address - first); word += 4) {
            auto& state = Get();
            bool tracked = false;
            for (size_t probe = 0; probe < state.words.size(); ++probe) {
                const u64 candidate = state.words[((word >> 2) + probe) % state.words.size()].load(
                    std::memory_order_relaxed);
                if (candidate == word) {
                    tracked = true;
                    break;
                }
                if (candidate == 0) {
                    break;
                }
            }
            if (tracked) {
                const u64 start = (std::max)(word, address);
                const u64 count = (std::min)(word + 4, address + size) - start;
                Bytes(kind, start, data != nullptr ? data + (start - address) : nullptr, count);
            }
        }
        return;
    }
    for (const auto& cell : Get().words) {
        const u64 word = cell.load(std::memory_order_relaxed);
        if (word != 0 && word < address + size && (word >= address || address - word < 4)) {
            const u64 start = (std::max)(word, address);
            const u64 count = (std::min)(word + 4, address + size) - start;
            Bytes(kind, start, data != nullptr ? data + (start - address) : nullptr, count);
        }
    }
}

struct History {
    std::vector<Event> events;
    u64 end{};
    u64 dropped{};
    u64 busy{};
};

inline History Snapshot(u64 word, u64 pair = 0, size_t limit = 300) {
    History history;
    if (!Enabled()) {
        return history;
    }
    auto& state = Get();
    history.end = state.next.load(std::memory_order_acquire);
    history.dropped = state.dropped.load(std::memory_order_relaxed);
    const u64 begin = history.end > Capacity ? history.end - Capacity : 1;
    for (auto& slot : state.slots) {
        if (slot.busy.test_and_set(std::memory_order_acquire)) {
            ++history.busy;
            continue;
        }
        const Event event = slot.event;
        slot.busy.clear(std::memory_order_release);
        const auto covers = [&event](u64 address) {
            return address != 0 && event.address < address + 4 &&
                   (event.address >= address || address - event.address < event.size);
        };
        // address==0 denotes a split-wide progress marker (positioned waits and fences).
        if (event.sequence >= begin && event.sequence < history.end &&
            (event.address == 0 || covers(word) || covers(pair) || event.pair == word ||
             (pair != 0 && event.pair == pair))) {
            history.events.push_back(event);
        }
    }
    std::ranges::sort(history.events, {}, &Event::sequence);
    if (history.events.size() > limit) {
        history.events.erase(history.events.begin(),
                             history.events.end() - static_cast<std::ptrdiff_t>(limit));
    }
    return history;
}

inline bool TakeDump() {
    return Enabled() && Get().dumps.fetch_add(1, std::memory_order_relaxed) < 5;
}

// Called when the next nvdrv session starts, before its GPU/fence users run. Monotonically
// increasing thread IDs remain unambiguous even when the host reuses a thread.
inline void Reset() {
    if (!Enabled()) {
        return;
    }
    auto& state = Get();
    for (auto& slot : state.slots) {
        while (slot.busy.test_and_set(std::memory_order_acquire)) {
        }
        slot.event = {};
        slot.busy.clear(std::memory_order_release);
    }
    for (auto& word : state.words) {
        word.store(0, std::memory_order_relaxed);
    }
    state.untracked.store(0, std::memory_order_relaxed);
    state.next.store(1, std::memory_order_relaxed);
    state.dropped.store(0, std::memory_order_relaxed);
    state.dumps.store(0, std::memory_order_relaxed);
}

} // namespace VideoCommon::SemRecorder
