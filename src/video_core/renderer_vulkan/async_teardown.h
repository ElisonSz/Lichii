// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include "common/assert.h"
#include "common/logging.h"

namespace Vulkan {

inline bool AsyncTeardownEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_TEARDOWN");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// One previous device may be retiring. Join before creating the next instance: driver cache
/// files and optional integrations have process-wide state, even though Vulkan devices do not.
/// The core also joins at destruction, before its frontend and logging infrastructure disappear.
class AsyncTeardown {
public:
    ~AsyncTeardown() {
        Wait();
    }

    void Wait() {
        std::scoped_lock lock{mutex};
        if (worker.joinable()) {
            worker.join();
            const char* check = std::getenv("EDEN_NX2_PIPELINE_TEARDOWN_CHECK");
            if (check != nullptr && check[0] == '1') {
                ASSERT(!active.load(std::memory_order_acquire));
                LOG_INFO(Render_Vulkan, "Async teardown check: joined, active owners 0");
            }
        }
    }

    template <typename Owner>
    void Retire(std::unique_ptr<Owner> owner) {
        std::scoped_lock lock{mutex};
        if (worker.joinable()) {
            worker.join();
        }
        // Keep ownership here until thread creation succeeds; resource exhaustion must fall
        // back to synchronous destruction rather than leak a live Vulkan device.
        active.store(true, std::memory_order_release);
        try {
            worker = std::thread{[this, raw = owner.get()] {
                delete raw;
                active.store(false, std::memory_order_release);
            }};
            owner.release();
        } catch (const std::system_error& error) {
            LOG_WARNING(Render_Vulkan, "Async Vulkan teardown: using synchronous cleanup: {}",
                        error.what());
            owner.reset();
            active.store(false, std::memory_order_release);
        }
    }

    static AsyncTeardown& Get() {
        // Qt's global System pointer is constructed before main, but its pointee is created
        // later. A function-static registry would die before that System's destructor calls
        // Wait(). Keep this small registry address valid through static destruction; every
        // device and thread is still released by explicit frontend/core joins.
        static AsyncTeardown* const teardown = new AsyncTeardown;
        return *teardown;
    }

private:
    std::mutex mutex;
    std::thread worker;
    std::atomic_bool active{};
};

} // namespace Vulkan
