// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include "common/container/unordered_map.h"

#include "video_core/dma_pusher.h"

namespace Tegra {

class GPU;

namespace Control {

struct ChannelState;

class Scheduler {
public:
    void Push(GPU& gpu, s32 channel, CommandList&& entries);

    /// Runs channels whose pending semaphore acquire is now met; true if any channel moved.
    bool ResumePending(GPU& gpu);
    [[nodiscard]] bool HasPendingChannels() const {
        return !pending_channels.empty();
    }
    /// Whether a suspended channel waits for the host GPU's copy of a command list.
    [[nodiscard]] bool WaitsForListCopy() const;
    /// HasPendingChannels for threads other than the GPU thread.
    [[nodiscard]] bool HasPendingChannelsAnyThread() const {
        return any_pending.load(std::memory_order_acquire);
    }
    /// EDEN_NX2_SYNC_SUSPEND: the earliest ring position a pending channel's acquire, suspended
    /// before the back ran its release, waits for (Puller::EarlyPosition).
    [[nodiscard]] std::optional<u64> EarlyWaitPosition() const;
    /// The channels suspended on an acquire (GPU thread only).
    [[nodiscard]] size_t PendingChannelCount() const {
        return pending_channels.size();
    }

    void DeclareChannel(std::shared_ptr<ChannelState> new_channel);

private:
    void BindChannel(GPU& gpu, s32 bind_id);
    /// EDEN_NX2_SYNC_SUSPEND: before `channel` dispatches, how many other channels are pending.
    void NoteOtherPending(const ChannelState& channel) const;

    ::Common::unordered_map<s32, std::shared_ptr<ChannelState>> channels;
    std::mutex scheduling_guard;
    std::vector<std::shared_ptr<ChannelState>> pending_channels; ///< Suspended on an acquire
    std::atomic<bool> any_pending{};
    /// The channel the last BindChannel bound: the GPU's once the binds sent before ran (the
    /// front's thread only).
    s32 last_bind_id = -1;
};

} // namespace Control

} // namespace Tegra
