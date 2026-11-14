// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/thread.h"

namespace VideoCommon::FastWake {

// Loaded by CreateGPU before starting any GPU threads; never changed during a session.
inline bool enabled{};

inline void ReadSwitch() {
    const char* const value = std::getenv("EDEN_NX2_FAST_WAKE");
    enabled = value != nullptr && value[0] == '1' && value[1] == '\0';
}

// One waiting thread, any number of notifiers. Snapshot before checking the work predicate.
// The generation covers notifications before arming; the event covers those after arming.
class Signal {
public:
    enum class Result { Disabled, Notified, Timeout };

    [[nodiscard]] u64 Snapshot() const {
        return generation.load(std::memory_order_seq_cst);
    }

    // Returns whether an armed waiter was signalled, rather than just advancing the generation.
    bool Notify() {
        if (!enabled) {
            return false;
        }
        generation.fetch_add(1, std::memory_order_seq_cst);
        if (!waiting.load(std::memory_order_seq_cst)) {
            return false;
        }
        event.Set();
        return true;
    }

    Result Wait(u64 observed, std::chrono::nanoseconds timeout) {
        if (!enabled) {
            return Result::Disabled;
        }
        event.Reset();
        waiting.store(true, std::memory_order_seq_cst);
        if (Snapshot() == observed) {
            // SleepFor uses an interruptible high-resolution timer on Windows, a timed CV
            // elsewhere. It deliberately avoids Event::WaitFor's hardware polling path.
            event.SleepFor(timeout);
        }
        waiting.store(false, std::memory_order_seq_cst);
        return Snapshot() != observed ? Result::Notified : Result::Timeout;
    }

private:
    friend struct SignalTestAccess;
    std::atomic<u64> generation{};
    std::atomic<bool> waiting{};
    Common::Event event;
};

inline Signal pending;

// Atomic because position/latch notifications originate on the back or fencing threads.
struct Counters {
    std::atomic<u64> latch_signals{};
    std::atomic<u64> command_signals{};
    std::atomic<u64> pending_waits{};
    std::atomic<u64> pending_notified{};
    std::atomic<u64> pending_timeouts{};
    std::atomic<u64> position_signals{};
    std::atomic<u64> position_waits{};
    std::atomic<u64> position_timeouts{};
    std::atomic<u64> idle_publishes{};
    std::atomic<u64> back_signals{};
    std::atomic<u64> back_waits{};
    std::atomic<u64> back_timeouts{};
};
inline Counters counters;

} // namespace VideoCommon::FastWake
