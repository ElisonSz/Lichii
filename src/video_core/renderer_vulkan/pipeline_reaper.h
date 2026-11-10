// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/thread.h"

namespace Vulkan {

/// Owns pipelines that neither a draw nor the GPU can reference anymore. Destruction never runs
/// under the queue lock. Keep the device, scheduler and descriptor pools alive until Drain returns.
template <typename Pipeline>
class PipelineReaper {
public:
    PipelineReaper() {
        AddWorkers(1);
    }

    ~PipelineReaper() {
        Drain();
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        ready.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }

    PipelineReaper(const PipelineReaper&) = delete;
    PipelineReaper& operator=(const PipelineReaper&) = delete;

    void Retire(std::unique_ptr<Pipeline> pipeline) {
        if (!pipeline) {
            return;
        }
        {
            std::scoped_lock lock{mutex};
            pending.push_back(std::move(pipeline));
            ++submitted;
        }
        ready.notify_one();
    }

    /// Transfer only completed GPU work, up to the frame's retirement budget. Keeping this check
    /// at the ownership boundary also covers command buffers still queued on the Vulkan worker.
    template <typename Retired, typename IsFree>
    void RetireCompleted(Retired& retired, size_t limit, const IsFree& is_free) {
        while (limit != 0 && !retired.empty() && is_free(retired.front().first)) {
            Retire(std::move(retired.front().second));
            retired.pop_front();
            --limit;
        }
    }

    /// Only the owning thread adds workers. Production uses one: increasing this did not improve
    /// driver destruction throughput. Tests can exercise concurrent retirement with more workers.
    void AddWorkers(size_t count) {
        while (workers.size() < count) {
            workers.emplace_back([this] { Run(); });
        }
    }

    /// Withdraw queued ownership for device-lifetime teardown. Only an already active destructor
    /// still needs Drain(); withdrawn objects are no longer included in this reaper's counts.
    std::deque<std::unique_ptr<Pipeline>> TakePending() {
        std::scoped_lock lock{mutex};
        std::deque<std::unique_ptr<Pipeline>> result;
        result.swap(pending);
        submitted -= result.size();
        if (active == 0) {
            drained.notify_all();
        }
        return result;
    }

    void Drain() {
        std::unique_lock lock{mutex};
        drained.wait(lock, [this] { return pending.empty() && active == 0; });
    }

    std::pair<size_t, size_t> Counts() const {
        std::scoped_lock lock{mutex};
        return {submitted, destroyed};
    }

private:
    void Run() {
        Common::SetCurrentThreadName("VkPipelineReaper");
        std::unique_lock lock{mutex};
        while (true) {
            ready.wait(lock, [this] { return stopping || !pending.empty(); });
            if (pending.empty()) {
                return;
            }
            auto pipeline = std::move(pending.front());
            pending.pop_front();
            ++active;
            lock.unlock();
            pipeline.reset();
            lock.lock();
            --active;
            ++destroyed;
            if (pending.empty() && active == 0) {
                drained.notify_all();
            }
        }
    }

    mutable std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable drained;
    std::deque<std::unique_ptr<Pipeline>> pending;
    size_t active{};
    size_t submitted{};
    size_t destroyed{};
    bool stopping{};
    std::vector<std::thread> workers;
};

} // namespace Vulkan
