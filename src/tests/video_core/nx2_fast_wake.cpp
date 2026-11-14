// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_wake.h"

namespace VideoCommon {
struct FastWakeTestAccess {
    static void Prepare(GpuSplit& split) {
        split.write_position = 100;
        split.published_local = 100;
        split.published.store(100);
        split.publish_every = GpuSplit::PublishEvery;
        split.back_idle_at.store(100);
    }
    static void Commit(GpuSplit& split, size_t words) {
        split.Commit(words);
    }
    static u64 Published(const GpuSplit& split) {
        return split.published.load();
    }
    static void ArmFront(GpuSplit& split) {
        split.front_waiting.store(true);
    }
    static u64 PositionGeneration(const GpuSplit& split) {
        return split.position_signal.Snapshot();
    }
    static void Consume(GpuSplit& split, u64 position) {
        split.consumed.store(position);
        split.NotifyFront();
    }
    static void ArmBack(GpuSplit& split) {
        split.back_waiting.store(true);
    }
    static u64 BackGeneration(const GpuSplit& split) {
        return split.back_signal.Snapshot();
    }
    static void WakeBack(GpuSplit& split) {
        split.WakeBack();
    }
};
} // namespace VideoCommon

namespace VideoCommon::FastWake {
struct SignalTestAccess {
    static bool Waiting(const Signal& signal) {
        return signal.waiting.load(std::memory_order_seq_cst);
    }
};
} // namespace VideoCommon::FastWake

namespace {

namespace Wake = VideoCommon::FastWake;
using namespace std::chrono_literals;

struct SwitchScope {
    explicit SwitchScope(bool value = true) : previous{Wake::enabled} {
        Wake::enabled = value;
    }
    ~SwitchScope() {
        Wake::enabled = previous;
    }
    bool previous;
};

} // namespace

TEST_CASE("Fast wake disabled leaves notification state untouched", "[video_core][fast_wake]") {
    SwitchScope switch_scope{false};
    Wake::Signal signal;
    const u64 before = signal.Snapshot();
    REQUIRE_FALSE(signal.Notify());
    REQUIRE(signal.Snapshot() == before);
    REQUIRE(signal.Wait(before, 1ms) == Wake::Signal::Result::Disabled);
    Tegra::Nx2SemaphoreWaiters waiters;
    const int owner{};
    const u64 pending_before = Wake::pending.Snapshot();
    REQUIRE(waiters.Add(&owner, 0x1000, 7, Tegra::Nx2SemaphoreWaiters::AcquireEqual));
    waiters.OnWrite(0x1000, 7);
    REQUIRE(waiters.Latched(&owner));
    REQUIRE(Wake::pending.Snapshot() == pending_before);
}

TEST_CASE("Fast wake remembers a notify before wait is armed", "[video_core][fast_wake]") {
    SwitchScope switch_scope;
    Wake::Signal signal;
    const u64 before = signal.Snapshot();
    REQUIRE_FALSE(signal.Notify());
    REQUIRE(signal.Wait(before, 1s) == Wake::Signal::Result::Notified);
    // A notification is a hint to recheck, not a permanently signalled event.
    REQUIRE(signal.Wait(signal.Snapshot(), 0ns) == Wake::Signal::Result::Timeout);
}

TEST_CASE("Fast wake notifies an OS sleeper and publishes preceding stores",
          "[video_core][fast_wake]") {
    SwitchScope switch_scope;
    Wake::Signal signal;
    const u64 before = signal.Snapshot();
    int payload = 0;
    std::atomic<bool> armed_signal{};
    std::jthread producer([&] {
        payload = 42;
        const auto deadline = std::chrono::steady_clock::now() + 1s;
        while (!Wake::SignalTestAccess::Waiting(signal) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        // Allow the waiter to enter the OS too; correctness must not depend on the delay.
        std::this_thread::sleep_for(1ms);
        armed_signal.store(signal.Notify(), std::memory_order_release);
    });
    REQUIRE(signal.Wait(before, 1s) == Wake::Signal::Result::Notified);
    REQUIRE(payload == 42);
    producer.join();
    REQUIRE(armed_signal.load(std::memory_order_acquire));
}

TEST_CASE("Fast wake timeout races cannot erase the next generation", "[video_core][fast_wake]") {
    SwitchScope switch_scope;
    Wake::Signal signal;
    std::atomic<u32> request{};
    std::atomic<u32> complete{};
    std::jthread producer([&](std::stop_token stop_token) {
        for (u32 round = 1; round <= 256; ++round) {
            while (request.load(std::memory_order_acquire) < round) {
                if (stop_token.stop_requested()) {
                    return;
                }
                std::this_thread::yield();
            }
            signal.Notify();
            complete.store(round, std::memory_order_release);
        }
    });
    for (u32 round = 1; round <= 256; ++round) {
        const u64 before = signal.Snapshot();
        request.store(round, std::memory_order_release);
        const auto result = signal.Wait(before, 1us);
        REQUIRE(
            (result == Wake::Signal::Result::Notified || result == Wake::Signal::Result::Timeout));
        const auto deadline = std::chrono::steady_clock::now() + 1s;
        while (complete.load(std::memory_order_acquire) < round &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        REQUIRE(complete.load(std::memory_order_acquire) == round);
        // Whether the first wait timed out or was notified, the generation is never reset.
        REQUIRE(signal.Wait(before, 0ns) == Wake::Signal::Result::Notified);
    }
}

TEST_CASE("First matching semaphore latch wakes once without changing latch rules",
          "[video_core][fast_wake]") {
    SwitchScope switch_scope;
    Tegra::Nx2SemaphoreWaiters waiters;
    const int equal{}, mask{};
    REQUIRE(waiters.Add(&equal, 0x1000, 7, Tegra::Nx2SemaphoreWaiters::AcquireEqual));
    REQUIRE(waiters.Add(&mask, 0x2000, 4, Tegra::Nx2SemaphoreWaiters::AcquireMask));
    const u64 before = Wake::pending.Snapshot();
    waiters.OnWrite(0x1000, 6);
    REQUIRE(Wake::pending.Snapshot() == before);
    waiters.OnWrite(0x1000, 7);
    REQUIRE(Wake::pending.Snapshot() == before + 1);
    REQUIRE(Wake::pending.Wait(before, 1s) == Wake::Signal::Result::Notified);
    waiters.OnWrite(0x1000, 7);
    waiters.OnWrite(0x1000, 0);
    REQUIRE(Wake::pending.Snapshot() == before + 1);
    REQUIRE(waiters.Latched(&equal));
    waiters.OnWrite(0x2000, 6);
    REQUIRE(Wake::pending.Snapshot() == before + 2);
    REQUIRE(waiters.Latched(&mask));
    waiters.Remove(&equal);
    REQUIRE(waiters.Add(&equal, 0x1000, 7, Tegra::Nx2SemaphoreWaiters::AcquireEqual));
    REQUIRE_FALSE(waiters.Latched(&equal));
    waiters.OnWrite(0x1000, 7);
    REQUIRE(Wake::pending.Snapshot() == before + 3);
}

TEST_CASE("Idle publication preserves holds and the disabled eight word threshold",
          "[video_core][fast_wake]") {
    using Access = VideoCommon::FastWakeTestAccess;
    for (const bool fast : {false, true}) {
        SwitchScope switch_scope{fast};
        VideoCommon::GpuSplit split;
        Access::Prepare(split);
        {
            VideoCommon::GpuSplit::PublishHold hold{split, true};
            Access::Commit(split, 1);
            REQUIRE(Access::Published(split) == 100);
        }
        REQUIRE(Access::Published(split) == (fast ? 101 : 100));
        if (!fast) {
            Access::Commit(split, 7);
            REQUIRE(Access::Published(split) == 108);
        }
    }
}

TEST_CASE("Production split notifications preserve disabled atomics and opt in generations",
          "[video_core][fast_wake]") {
    using Access = VideoCommon::FastWakeTestAccess;
    for (const bool fast : {false, true}) {
        SwitchScope switch_scope{fast};
        VideoCommon::GpuSplit split;
        Access::ArmFront(split);
        Access::Consume(split, 123);
        REQUIRE(split.Ran(123));
        REQUIRE(Access::PositionGeneration(split) == (fast ? 1 : 0));
        REQUIRE(Access::BackGeneration(split) == 0);
        Access::ArmBack(split);
        Access::WakeBack(split);
        REQUIRE(Access::BackGeneration(split) == (fast ? 1 : 0));
        REQUIRE(Access::PositionGeneration(split) == (fast ? 1 : 0));
    }
}

TEST_CASE("Latch wake cannot satisfy a positioned dependency", "[video_core][fast_wake]") {
    SwitchScope switch_scope;
    Wake::Signal position_signal;
    const u64 position_before = position_signal.Snapshot();
    Tegra::Nx2SemaphoreWaiters waiters;
    const int owner{};
    REQUIRE(waiters.Add(&owner, 0x1000, 7, Tegra::Nx2SemaphoreWaiters::AcquireEqual));
    const u64 pending_before = Wake::pending.Snapshot();
    waiters.OnWrite(0x1000, 7);
    REQUIRE(Wake::pending.Wait(pending_before, 0ns) == Wake::Signal::Result::Notified);
    REQUIRE(position_signal.Wait(position_before, 1us) == Wake::Signal::Result::Timeout);
}
