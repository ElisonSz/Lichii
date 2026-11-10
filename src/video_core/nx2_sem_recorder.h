// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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
// Pages are registered only by the view. Published links are immutable until session reset;
// producers read atomic word bits without locking or probing a full word table.
struct Page {
    u64 number{};
    std::array<std::atomic<u64>, 16> bits{};
    Page* bucket_next{};
    Page* all_next{};
};
struct State {
    std::array<Slot, Capacity> slots;
    std::atomic<u64> next{1};
    std::atomic<u64> threads{1};
    std::atomic<u64> dropped{};
    std::atomic<u32> dumps{};
    std::atomic<u64> untracked{};
    std::array<std::atomic<Page*>, 4096> buckets{};
    std::atomic<Page*> pages{};
    std::array<std::atomic<u64>, 1024> host_pages{};
    std::mutex registration_mutex;
    std::vector<std::unique_ptr<Page>> owned_pages;
};

// Irrelevant producers must not allocate the ring or initialize its static storage.
inline std::atomic<State*> g_state{};

inline State& Get() {
    static auto state = [] {
        auto created = std::make_unique<State>();
        g_state.store(created.get(), std::memory_order_release);
        return created;
    }();
    return *state;
}

inline size_t Bucket(u64 page) {
    return static_cast<size_t>((page ^ (page >> 12) ^ (page >> 24)) & 4095);
}

inline Page* FindPage(const State& state, u64 number) {
    auto* page = state.buckets[Bucket(number)].load(std::memory_order_acquire);
    while (page != nullptr && page->number != number) {
        page = page->bucket_next;
    }
    return page;
}

// This filter uses host addresses, so it can reject CPU stores and GPU scalar accesses before
// any device translation. Hash collisions only add work; they cannot omit a tracked page.
inline bool MayHostRange(const void* pointer, u64 size) {
    if (!Enabled() || size == 0) {
        return false;
    }
    const auto* state = g_state.load(std::memory_order_acquire);
    if (state == nullptr || state->pages.load(std::memory_order_acquire) == nullptr) {
        return false;
    }
    const u64 first = static_cast<u64>(reinterpret_cast<uintptr_t>(pointer)) >> 12;
    const u64 count =
        ((static_cast<u64>(reinterpret_cast<uintptr_t>(pointer)) & 4095) + size - 1) / 4096 + 1;
    if (count >= 65536) {
        return true;
    }
    // Inspect bitmap cells rather than walking every page of a large CPU block. The modulo
    // interval can wrap once; collisions cause a conservative slow path, never a missed store.
    for (u64 offset = 0; offset < count;) {
        const u64 bit = (first + offset) & 65535;
        const u64 amount = (std::min)(count - offset, u64{64} - (bit & 63));
        const u64 mask = (~u64{0} >> (64 - amount)) << (bit & 63);
        if ((state->host_pages[static_cast<size_t>(bit >> 6)].load(std::memory_order_relaxed) &
             mask) != 0) {
            return true;
        }
        offset += amount;
    }
    return false;
}

template <typename Func>
inline void ForWords(const State& state, u64 address, u64 size, Func&& visit) {
    if (size == 0) {
        return;
    }
    const auto inspect = [&](const Page& page) {
        const u64 base = page.number << 12;
        if (base >= address ? base - address >= size : address - base >= 4096) {
            return;
        }
        const u64 first = address > base ? (address - base) / 4 : 0;
        const u64 last = (std::min)(u64{1023}, (address + size - 1 - base) / 4);
        for (u64 cell = first / 64; cell <= last / 64; ++cell) {
            u64 bits = page.bits[static_cast<size_t>(cell)].load(std::memory_order_relaxed);
            if (cell == first / 64) {
                bits &= ~u64{0} << (first & 63);
            }
            if (cell == last / 64) {
                bits &= ~u64{0} >> (63 - (last & 63));
            }
            while (bits != 0) {
                const auto index = static_cast<u64>(std::countr_zero(bits));
                visit(base + (cell * 64 + index) * 4);
                bits &= bits - 1;
            }
        }
    };
    const u64 first_page = address >> 12;
    const u64 last_page = (address + size - 1) >> 12;
    if (last_page - first_page <= 16) {
        for (u64 number = first_page; number <= last_page; ++number) {
            if (const auto* page = FindPage(state, number)) {
                inspect(*page);
            }
        }
    } else {
        for (auto* page = state.pages.load(std::memory_order_acquire); page != nullptr;
             page = page->all_next) {
            inspect(*page);
        }
    }
}

inline bool Relevant(u64 address, u64 size = 4) {
    if (!Enabled()) {
        return false;
    }
    const auto* state = g_state.load(std::memory_order_acquire);
    if (state == nullptr) {
        return false;
    }
    if (size != 0 && size <= 4 && (address & 3) + size <= 4) {
        const auto* page = FindPage(*state, address >> 12);
        if (page == nullptr) {
            return false;
        }
        const u64 index = (address & 4095) >> 2;
        return (page->bits[static_cast<size_t>(index >> 6)].load(std::memory_order_relaxed) &
                (u64{1} << (index & 63))) != 0;
    }
    bool found = false;
    ForWords(*state, address, size, [&](u64) { found = true; });
    return found;
}

inline void TrackHostPage(State& state, const void* host) {
    if (host == nullptr) {
        return;
    }
    const u64 bit = (static_cast<u64>(reinterpret_cast<uintptr_t>(host)) >> 12) & 65535;
    auto& cell = state.host_pages[static_cast<size_t>(bit >> 6)];
    const u64 mask = u64{1} << (bit & 63);
    if ((cell.load(std::memory_order_relaxed) & mask) == 0) {
        cell.fetch_or(mask, std::memory_order_relaxed);
    }
}

// Device words survive backing-page changes. Register the replacement host page before its
// mapping is published, so stores between the remap and the next view read remain visible.
inline void RemapHostPage(u64 address, const void* host) {
    if (Relevant(address, 4096)) {
        TrackHostPage(Get(), host);
    }
}

inline void Track(u64 address, const void* host = nullptr) {
    if (!Enabled() || address == 0) {
        return;
    }
    auto& state = Get();
    TrackHostPage(state, host);
    const u64 number = address >> 12;
    auto* page = FindPage(state, number);
    if (page == nullptr) {
        std::scoped_lock lock{state.registration_mutex};
        page = FindPage(state, number);
        if (page == nullptr) {
            auto created = std::make_unique<Page>();
            page = created.get();
            page->number = number;
            page->bucket_next = state.buckets[Bucket(number)].load(std::memory_order_relaxed);
            page->all_next = state.pages.load(std::memory_order_relaxed);
            state.owned_pages.push_back(std::move(created));
            state.buckets[Bucket(number)].store(page, std::memory_order_release);
            state.pages.store(page, std::memory_order_release);
        }
    }
    const u64 index = (address & 4095) >> 2;
    auto& cell = page->bits[static_cast<size_t>(index >> 6)];
    const u64 mask = u64{1} << (index & 63);
    if ((cell.load(std::memory_order_relaxed) & mask) == 0) {
        cell.fetch_or(mask, std::memory_order_relaxed);
    }
}

// Call only after an exact word/range test. Bulk byte extraction has already done that test.
inline void AddRelevant(Event event) {
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

inline void Add(Event event) {
    if (!Enabled()) {
        return;
    }
    const auto* active = g_state.load(std::memory_order_acquire);
    if (active == nullptr ||
        (event.address == 0 ? active->pages.load(std::memory_order_acquire) == nullptr
                            : !Relevant(event.address, event.size) &&
                                  (event.pair == 0 || !Relevant(event.pair)))) {
        return;
    }
    AddRelevant(event);
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
    if (!Relevant(address, size)) {
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

// Only intersecting registered words are inspected. A large copy visits registered pages,
// not every cell of a fixed-capacity table or every page of the copy.
inline void TrackedBytes(const char* kind, u64 address, const void* bytes, u64 size,
                         bool record_range = true) {
    if (!Enabled()) {
        return;
    }
    if (size != 0 && size <= 4 && (address & 3) + size <= 4) {
        if (Relevant(address, size)) {
            u32 value{};
            if (bytes != nullptr) {
                std::memcpy(&value, bytes, static_cast<size_t>(size));
            }
            if (record_range) {
                AddRelevant({.kind = kind, .address = address, .size = size});
            }
            AddRelevant({.kind = kind, .address = address, .size = size, .value = value});
        }
        return;
    }
    const auto* state = g_state.load(std::memory_order_acquire);
    if (state == nullptr) {
        return;
    }
    bool noted = false;
    const auto* data = static_cast<const u8*>(bytes);
    ForWords(*state, address, size, [&](u64 word) {
        if (record_range && !noted) {
            AddRelevant({.kind = kind, .address = address, .size = size});
            noted = true;
        }
        const u64 start = (std::max)(word, address);
        const u64 count = (std::min)(word + 4, address + size) - start;
        u32 value{};
        if (data != nullptr) {
            std::memcpy(&value, data + (start - address), static_cast<size_t>(count));
        }
        AddRelevant({.kind = kind, .address = start, .size = count, .value = value});
    });
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
    // Keep the pages themselves: a producer still running from the last session may be walking
    // them (a CPU store passing the host filter), so freeing one here could be used after free.
    // Clearing their words untracks them.
    for (auto* page = state.pages.load(std::memory_order_acquire); page != nullptr;
         page = page->all_next) {
        for (auto& bits : page->bits) {
            bits.store(0, std::memory_order_relaxed);
        }
    }
    for (auto& bits : state.host_pages) {
        bits.store(0, std::memory_order_relaxed);
    }
    state.untracked.store(0, std::memory_order_relaxed);
    state.next.store(1, std::memory_order_relaxed);
    state.dropped.store(0, std::memory_order_relaxed);
    state.dumps.store(0, std::memory_order_relaxed);
}

} // namespace VideoCommon::SemRecorder
