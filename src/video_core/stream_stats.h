// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::StreamStats {

// Read once before renderer/worker creation; unchanged for the GPU session.
inline bool enabled{};
inline bool cache_enabled{};
inline bool verify_enabled{};
inline void ReadSwitch() {
    const auto read = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    enabled = read("EDEN_NX2_STREAM_STATS");
    cache_enabled = read("EDEN_NX2_ASTC_CACHE");
    verify_enabled = read("EDEN_NX2_ASTC_CACHE_VERIFY");
}
inline bool Enabled() {
    return enabled;
}

enum class Counter : std::size_t {
    Lookups,
    Reused,
    TicChanges,
    ViewCreates,
    DecodeBarriers,
    TransferBarriers,
    DescriptorUpdates,
    Created,
    Destroyed,
    CreatedAstc,
    DestroyedAstc,
    UploadAstc,
    BytesAstc,
    UploadBcn,
    BytesBcn,
    UploadOther,
    BytesOther,
    Decodes,
    Dispatches,
    Repeats,
    RepeatsOtherAddress,
    LayoutChanges,
    HashSkipped,
    HistoryDropped,
    CpuWrites,
    CpuUnmaps,
    GpuUnmaps,
    Collected,
    DecodeFrames,
    MixedFrames,
    MixedSubmissions,
    Submissions,
    RenderPassBreaks,
    Finishes,
    GpuWaits,
    CacheHits,
    CacheMisses,
    CacheEvictions,
    CacheRejected,
    CacheBytes,
    CreateNs,
    StagingNs,
    GuestReadNs,
    HashNs,
    ConvertNs,
    DecodeBackNs,
    DecodeWorkerNs,
    DecodeBarrierNs,
    TransferWorkerNs,
    TransferBarrierNs,
    DescriptorNs,
    SubmitNs,
    FinishNs,
    QueueSubmitNs,
    GpuWaitNs,
    GpuSamples,
    GpuSkipped,
    GpuDecodeNs,
    GpuBarrierNs,
    VerifyHits,
    VerifyBadBytes,
    VerifySkipped,
    Count,
};

inline std::array<std::atomic<u64>, static_cast<std::size_t>(Counter::Count)> counters{};
inline std::chrono::steady_clock::time_point window_start{};

inline void Add(Counter counter, u64 value = 1) {
    if (Enabled()) {
        counters[static_cast<std::size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
    }
}

inline void Upload(bool astc, bool bcn, u64 bytes) {
    Add(astc ? Counter::UploadAstc : bcn ? Counter::UploadBcn : Counter::UploadOther);
    Add(astc ? Counter::BytesAstc : bcn ? Counter::BytesBcn : Counter::BytesOther, bytes);
}

class Scope {
public:
    explicit Scope(Counter counter_) : counter{counter_} {
        if (Enabled()) {
            start = std::chrono::steady_clock::now();
            if (counter == Counter::DecodeBarrierNs) {
                Add(Counter::DecodeBarriers);
            } else if (counter == Counter::TransferBarrierNs) {
                Add(Counter::TransferBarriers);
            } else if (counter == Counter::DescriptorNs) {
                Add(Counter::DescriptorUpdates);
            }
        }
    }
    ~Scope() {
        if (Enabled()) {
            Add(counter, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - start)
                                              .count()));
        }
    }

private:
    Counter counter;
    std::chrono::steady_clock::time_point start{};
};

// Called on the recording thread, once per texture-cache frame. Worker counters are atomic.
inline void Reset() {
    if (!Enabled()) {
        return;
    }
    for (auto& counter : counters) {
        counter.store(0, std::memory_order_relaxed);
    }
    window_start = std::chrono::steady_clock::now();
}

inline void Report() {
    if (!Enabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - window_start < std::chrono::seconds{10}) {
        return;
    }
    std::array<u64, static_cast<std::size_t>(Counter::Count)> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        // CacheBytes is a gauge; everything else is a window delta.
        values[index] = index == static_cast<std::size_t>(Counter::CacheBytes)
                            ? counters[index].load(std::memory_order_relaxed)
                            : counters[index].exchange(0, std::memory_order_relaxed);
    }
    const auto count = [&](Counter counter) { return values[static_cast<std::size_t>(counter)]; };
    const auto ms = [&](Counter counter) { return static_cast<double>(count(counter)) / 1e6; };
    LOG_INFO(
        Render_Vulkan,
        "Stream stats: {:.3f}s lookups={} reused={} tic_changes={} view_creates={} images={}/{} "
        "astc_images={}/{} "
        "uploads_astc={}/{}B bcn={}/{}B other={}/{}B "
        "decodes={} dispatches={} repeats={} other_address={} layout_changes={} hash_skipped={} "
        "history_dropped={} "
        "invalidations_write={} cpu_unmap={} gpu_unmap={} collected={} "
        "decode_frames={} mixed_frames={} mixed_submissions={} submissions={} pass_breaks={} "
        "finishes={} gpu_waits={} "
        "cache_hit={} miss={} evict={} reject={} resident={}B decode_barriers={} "
        "transfer_barriers={} descriptor_updates={} "
        "cpu_ms create={:.3f} staging={:.3f} guest_read={:.3f} hash={:.3f} "
        "convert={:.3f} decode_back={:.3f} decode_worker={:.3f} decode_barrier={:.3f} "
        "transfer_worker={:.3f} transfer_barrier={:.3f} descriptor={:.3f} submit={:.3f} "
        "finish={:.3f} queue_submit={:.3f} gpu_wait={:.3f} "
        "gpu_samples={} skipped={} gpu_ms decode={:.3f} barriers={:.3f} verify_hits={} "
        "verify_bad_bytes={} verify_skipped={}",
        std::chrono::duration<double>(now - window_start).count(), count(Counter::Lookups),
        count(Counter::Reused), count(Counter::TicChanges), count(Counter::ViewCreates),
        count(Counter::Created), count(Counter::Destroyed), count(Counter::CreatedAstc),
        count(Counter::DestroyedAstc), count(Counter::UploadAstc), count(Counter::BytesAstc),
        count(Counter::UploadBcn), count(Counter::BytesBcn), count(Counter::UploadOther),
        count(Counter::BytesOther), count(Counter::Decodes), count(Counter::Dispatches),
        count(Counter::Repeats), count(Counter::RepeatsOtherAddress), count(Counter::LayoutChanges),
        count(Counter::HashSkipped), count(Counter::HistoryDropped), count(Counter::CpuWrites),
        count(Counter::CpuUnmaps), count(Counter::GpuUnmaps), count(Counter::Collected),
        count(Counter::DecodeFrames), count(Counter::MixedFrames), count(Counter::MixedSubmissions),
        count(Counter::Submissions), count(Counter::RenderPassBreaks), count(Counter::Finishes),
        count(Counter::GpuWaits), count(Counter::CacheHits), count(Counter::CacheMisses),
        count(Counter::CacheEvictions), count(Counter::CacheRejected), count(Counter::CacheBytes),
        count(Counter::DecodeBarriers), count(Counter::TransferBarriers),
        count(Counter::DescriptorUpdates), ms(Counter::CreateNs), ms(Counter::StagingNs),
        ms(Counter::GuestReadNs), ms(Counter::HashNs), ms(Counter::ConvertNs),
        ms(Counter::DecodeBackNs), ms(Counter::DecodeWorkerNs), ms(Counter::DecodeBarrierNs),
        ms(Counter::TransferWorkerNs), ms(Counter::TransferBarrierNs), ms(Counter::DescriptorNs),
        ms(Counter::SubmitNs), ms(Counter::FinishNs), ms(Counter::QueueSubmitNs),
        ms(Counter::GpuWaitNs), count(Counter::GpuSamples), count(Counter::GpuSkipped),
        ms(Counter::GpuDecodeNs), ms(Counter::GpuBarrierNs), count(Counter::VerifyHits),
        count(Counter::VerifyBadBytes), count(Counter::VerifySkipped));
    window_start = now;
}

// Recording-thread state: a render pass identifies a graphics command buffer, including clears.
// FinishFrame is deliberately independent of presentation/CPU tracing switches.
class FrameTracker {
public:
    void Graphics(u64 tick) {
        graphics_frame = true;
        graphics_tick = tick;
        if (decode_tick == tick) {
            if (mixed_tick != tick) {
                mixed_tick = tick;
                Add(Counter::MixedSubmissions);
            }
        }
    }
    void Decode(u64 tick) {
        decode_frame = true;
        decode_tick = tick;
        if (graphics_tick == tick && mixed_tick != tick) {
            mixed_tick = tick;
            Add(Counter::MixedSubmissions);
        }
    }
    void FinishFrame() {
        if (decode_frame) {
            Add(Counter::DecodeFrames);
        }
        if (decode_frame && graphics_frame) {
            Add(Counter::MixedFrames);
        }
        decode_frame = graphics_frame = false;
    }

private:
    u64 graphics_tick{};
    u64 decode_tick{};
    u64 mixed_tick{};
    bool decode_frame{};
    bool graphics_frame{};
};

inline FrameTracker frames;

} // namespace VideoCommon::StreamStats
