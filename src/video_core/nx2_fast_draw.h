// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

/// A draw that follows a draw of the same pipeline with only light state changes in between takes
/// the delta path. It keeps the previous draw's pipeline, image views, samplers and texture buffer
/// bindings, and resolves and binds every buffer (index, vertex, uniform, storage, texture) again
/// as a full resolve does, so guest memory written without a notification is streamed as in a
/// full resolve.
/// Light changes are draw parameters, index buffer and vertex stream registers, constant buffer
/// binds and inline constant buffer data (collected in g_delta: the rasterizer resolves in full
/// when they reach the pipeline's texture handles at their GPU addresses or through another
/// mapping of their device memory). Anything else that could change what a resolve produces bumps
/// a serial: a 3D register that changes, any 3D method with side effects (uploads, reports,
/// clears, barriers, ...), constant buffer data reaching the texture or sampler headers (the same
/// way), any other engine, deferred calls on the split's back (except the ones marked
/// GpuSplit::FastDrawNeutral), GPU::TickWork requests, texture cache writes, unmaps and
/// deletions, shader removals, fence releases, HLE macros and frames. The rasterizer takes the
/// delta path only while the serial, the GPU memory map, the command buffer, the render pass and
/// the bound pipeline are the ones the previous draw left.
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the pipeline, views,
/// samplers or kept buffer bindings differ.
/// Experimental query reuse still executes every query handler. Experimental vertex-layout
/// reuse requires dynamic vertex input, rechecks the full pipeline key and emits vertex input
/// again; static input or a different shader input type falls back to full preparation.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    DeltaOverflow,
    Map,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "fence release", "indirect reset", "frame",           "texture header data",
    "delta overflow", "memory map",
};

inline std::atomic<bool> g_enabled{true};

/// Initialized at GPU creation, before command processing starts. Hot switches are plain flags.
inline bool g_draw_delta{};
inline bool g_draw_delta_verify{};
inline void ReadDrawDeltaSwitch() {
    const char* const enabled = std::getenv("EDEN_NX2_DRAW_DELTA");
    const char* const verify = std::getenv("EDEN_NX2_DRAW_DELTA_VERIFY");
    g_draw_delta = enabled != nullptr && enabled[0] == '1';
    g_draw_delta_verify = g_draw_delta && verify != nullptr && verify[0] == '1';
}

/// EDEN_NX2_DRAW_REUSE=1: the delta draws EDEN_NX2_DRAW_DELTA takes (it turns that switch on),
/// checked without reading guest memory unless data written since the previous draw reaches the
/// pipeline's handle inputs (RasterizerVulkan::DeltaHandleInputsReached), and then by the
/// handles alone; report targets count as data written, which keeps uniform keeps valid across
/// reports. EDEN_NX2_DRAW_REUSE_VERIFY=1 prepares those draws in full and compares (diagnostic).
inline bool g_draw_reuse{};
inline bool g_draw_reuse_verify{};
/// EDEN_NX2_DRAW_VERTEX_INPUT=1: dynamic vertex input states are kept by the registers they are
/// built from (VideoCommon::VertexInputCache) and recorded by reference.
inline bool g_draw_vertex_input{};
/// EDEN_NX2_DRAW_HANDLE_EVAL=1: handle programs run without zeroing their stack and slots
/// (Shader::g_lean_handle_programs).
inline bool g_draw_handle_eval{};
/// Reads the four switches above; called by VideoCore::CreateGPU after ReadDrawDeltaSwitch.
void ReadDrawReuseSwitches();

/// Track D (EDEN_NX2_DRAWX_*): the delta chain ends only on what it keeps.
/// EDEN_NX2_DRAWX_NEUTRAL=1: writes to 3D registers that no draw preparation reads (the MME memory
/// read registers, the constant colour rendering block) no longer end it.
inline bool g_drawx_neutral{};
/// EDEN_NX2_DRAWX_PIPESTATE=1: a fixed-function register write marks the delta (pipeline_state)
/// instead of ending it; the next draw looks its pipeline up again and keeps the delta only when
/// the same pipeline object comes back (as the vertex layout case does).
inline bool g_drawx_pipestate{};
/// EDEN_NX2_DRAWX_UNIFORMS=1: uniform keeps (BufferCache::UniformRangeKept) also on full draws
/// when nothing but register writes ended the delta chain since the previous draw (g_keep_serial).
inline bool g_drawx_uniforms{};
/// EDEN_NX2_DRAWX_TWIN=1: when the pipeline looked up again for a vertex layout or fixed-function
/// change is another object that resolves exactly the same bindings (the same configure
/// specialization and interned stage infos: descriptors, handle programs, properties), the
/// delta draw goes on with it, taking what the previous pipeline's last resolve kept.
inline bool g_drawx_twin{};
/// EDEN_NX2_DRAWX_VERIFY=1: delta candidates are prepared in full and compared (as
/// EDEN_NX2_FAST_DRAW_VERIFY), kept uniforms are compared with guest memory; counted per window.
inline bool g_drawx_verify{};
/// g_drawx_neutral || g_drawx_pipestate: the one flag the register write path tests.
inline bool g_drawx_registers{};
/// Reads the switches above; called by VideoCore::CreateGPU after ReadDrawReuseSwitches.
void ReadDrawXSwitches();

/// What a 3D register write does to the delta chain under the switches above.
enum class DrawXRegister : u8 {
    Breaks,        ///< As without the switches
    Neutral,       ///< Read by no draw preparation: no break (EDEN_NX2_DRAWX_NEUTRAL)
    PipelineState, ///< Fixed-function state: Delta::pipeline_state (EDEN_NX2_DRAWX_PIPESTATE)
};
/// One entry per 3D register (Maxwell3D::Regs::NUM_REGS), filled by ReadDrawXSwitches.
inline std::array<DrawXRegister, 0xe00> g_drawx_register_class{};
[[nodiscard]] inline DrawXRegister DrawXRegisterClass(u32 method) {
    return method < g_drawx_register_class.size() ? g_drawx_register_class[method]
                                                  : DrawXRegister::Breaks;
}
/// The class each register has when both switches are on (tests and ReadDrawXSwitches).
[[nodiscard]] DrawXRegister ClassifyDrawXRegister(u32 method);

/// Every break other than a 3D register write (methods, macros, other engines, fences, tick work,
/// invalidations, maps, overflowing write lists, ...): what can write guest memory unseen by the
/// write lists. Counted only with EDEN_NX2_DRAWX_UNIFORMS (any thread).
inline std::atomic<u64> g_keep_serial{1};

/// Any thread: whether fast draws run (always, unless a test turned them off). The rasterizer
/// latches it per draw (on the thread that draws) and starts with a full resolve when it turns on.
[[nodiscard]] inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Tests only.
inline void SetEnabledForTest(bool on) {
    g_enabled.store(on, std::memory_order_relaxed);
}

/// EDEN_NX2_FAST_DRAW_VERIFY=1, read once.
[[nodiscard]] inline bool VerifyEnabled() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// Experiments are read once when the first 3D engine is created, after switch-file loading.
/// 0 = off, 1 = on, 2 = alternate off/on in successive ten-second draw report windows.
struct Experiments {
    u8 vertex_id{};
    u8 vertex_layout{};
    u8 queries{};

    [[nodiscard]] bool Alternating() const {
        return vertex_id == 2 || vertex_layout == 2 || queries == 2;
    }
};

[[nodiscard]] inline u8 ReadExperiment(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && (value[0] == '1' || value[0] == '2')
               ? static_cast<u8>(value[0] - '0')
               : u8{0};
}

[[nodiscard]] inline const Experiments& ExperimentSettings() {
    static const Experiments settings{ReadExperiment("EDEN_NX2_REUSE_VERTEX_ID"),
                                      ReadExperiment("EDEN_NX2_REUSE_VERTEX_LAYOUT"),
                                      ReadExperiment("EDEN_NX2_REUSE_QUERIES")};
    return settings;
}

inline std::atomic<bool> g_alternate_on{};

[[nodiscard]] inline bool ExperimentActive(u8 mode) {
    return mode == 1 || (mode == 2 && g_alternate_on.load(std::memory_order_relaxed));
}

/// A plain flag: checked for every draw and every break, where a function's static is checked
/// for its initialization on every call (MSVC reads a thread-local for it).
inline std::atomic<bool> g_draw_cost_stats{};

/// Called by VideoCore::CreateGPU, after the frontend has loaded nx2_switches.txt.
inline void ReadDrawCostStatsSwitch() {
    const char* const value = std::getenv("EDEN_NX2_DRAW_COST_STATS");
    g_draw_cost_stats.store((value != nullptr && value[0] == '1') ||
                                ExperimentSettings().Alternating(),
                            std::memory_order_relaxed);
}

[[nodiscard]] inline bool DrawCostStatsEnabled() {
    return g_draw_cost_stats.load(std::memory_order_relaxed);
}

/// A draw's union of breaks, rather than just its last break. Only collected with cost stats.
/// Method classes describe side effects; register classes describe the state being changed.
enum class BreakClass : u8 {
    VertexLayout,
    QueryCondition,
    Shader,
    FixedFunction,
    MethodReport,
    MethodBarrier,
    MethodUpload,
    MethodControl,
    DeltaOverflow,
    VertexId,
    Other,
    Count,
};
inline constexpr size_t NumBreakClasses = static_cast<size_t>(BreakClass::Count);
inline constexpr std::array<const char*, NumBreakClasses> class_names{
    "vertex_layout",  "query_condition", "shader",        "fixed_function",
    "method_report",  "method_barrier",  "method_upload", "method_control",
    "delta_overflow", "vertex_id",       "other"};

[[nodiscard]] constexpr u32 ClassBit(BreakClass kind) {
    return 1U << static_cast<u32>(kind);
}
[[nodiscard]] u32 ClassifyBreak(Reason reason, u32 method);
inline std::atomic<u32> g_break_classes{};
/// All method break events, including those hidden by a later break (not draw counts).
inline std::array<std::atomic<u64>, 0xe00> g_method_breaks{};
inline std::atomic<u64> g_gpu_overflows{};
inline std::atomic<u64> g_device_overflows{};

inline std::atomic<u64> g_overflow_draws{};
inline std::array<std::atomic<u64>, 2> g_shader_writes{}; // identical, changed
inline std::atomic<u64> g_bookkeeping_samples{};
inline std::atomic<u64> g_bookkeeping_ns{};

[[nodiscard]] inline u64 CostNow() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Recording-thread attribution, including nested WFI work inside cache invalidation. Counts
/// both operations and handlers containing each effect; no worker-thread globals are sampled.
enum class Handler : size_t { Report, TextureInvalidate, WFI, Fragment, FragmentResolve, Count };
enum class Effect : size_t {
    QueryEnd,
    PassEnd,
    Submit,
    HostWait,
    GpuBarrier,
    QueryCopyWait,
    WorkerWait,
    GuestWait,
    Count
};
inline constexpr size_t NumHandlers = static_cast<size_t>(Handler::Count);
inline constexpr size_t NumEffects = static_cast<size_t>(Effect::Count);
inline constexpr std::array<const char*, NumHandlers> handler_names{
    "report", "texture_invalidate", "wfi", "fragment", "fragment_resolve"};
inline constexpr std::array<const char*, NumEffects> effect_names{
    "query_end",   "pass_end",        "submit",      "host_wait",
    "gpu_barrier", "query_copy_wait", "worker_wait", "guest_wait"};
struct HandlerStats {
    std::atomic<u64> calls{};
    std::atomic<u64> ns{};
    std::array<std::atomic<u64>, NumEffects> effects{};
    std::array<std::atomic<u64>, NumEffects> with_effect{};
};
inline std::array<HandlerStats, NumHandlers> g_handlers;
inline std::array<std::atomic<u64>, NumEffects> g_effect_operations{};
struct HandlerScope;
inline thread_local HandlerScope* g_handler_scope{};
struct HandlerScope {
    HandlerScope* previous{};
    u64 start{};
    Handler kind;
    u32 effects{};
    explicit HandlerScope(Handler handler) : kind{handler} {
        if (DrawCostStatsEnabled()) {
            previous = g_handler_scope;
            g_handler_scope = this;
            start = CostNow();
        }
    }
    ~HandlerScope() {
        if (start == 0) {
            return;
        }
        const u64 elapsed = CostNow() - start;
        g_handler_scope = previous;
        auto& stats = g_handlers[static_cast<size_t>(kind)];
        stats.calls.fetch_add(1, std::memory_order_relaxed);
        stats.ns.fetch_add(elapsed, std::memory_order_relaxed);
        for (size_t index = 0; index < NumEffects; ++index) {
            if ((effects & (1U << index)) != 0) {
                stats.with_effect[index].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    HandlerScope(const HandlerScope&) = delete;
    HandlerScope& operator=(const HandlerScope&) = delete;
};
inline void NoteEffect(Effect effect) {
    if (!DrawCostStatsEnabled()) {
        return;
    }
    const size_t index = static_cast<size_t>(effect);
    g_effect_operations[index].fetch_add(1, std::memory_order_relaxed);
    for (auto* scope = g_handler_scope; scope != nullptr; scope = scope->previous) {
        scope->effects |= 1U << index;
        g_handlers[static_cast<size_t>(scope->kind)].effects[index].fetch_add(
            1, std::memory_order_relaxed);
    }
}

/// Start a clean report window; diagnostic serials deliberately remain monotonic.
inline void ResetCostEvents() {
    for (auto& count : g_method_breaks) {
        count.exchange(0, std::memory_order_relaxed);
    }
    g_gpu_overflows.exchange(0, std::memory_order_relaxed);
    g_device_overflows.exchange(0, std::memory_order_relaxed);
    g_overflow_draws.exchange(0, std::memory_order_relaxed);
    for (auto& count : g_shader_writes) {
        count.exchange(0, std::memory_order_relaxed);
    }
    g_bookkeeping_samples.exchange(0, std::memory_order_relaxed);
    g_bookkeeping_ns.exchange(0, std::memory_order_relaxed);
    for (auto& operation : g_effect_operations) {
        operation.exchange(0, std::memory_order_relaxed);
    }
    for (auto& stats : g_handlers) {
        stats.calls.exchange(0, std::memory_order_relaxed);
        stats.ns.exchange(0, std::memory_order_relaxed);
        for (size_t index = 0; index < NumEffects; ++index) {
            stats.effects[index].exchange(0, std::memory_order_relaxed);
            stats.with_effect[index].exchange(0, std::memory_order_relaxed);
        }
    }
}

inline std::atomic<u64> g_serial{1};
/// Diagnostic serial ignoring only the three experiment-eligible register/method breaks. It
/// never authorizes reuse: off-window candidates still execute full preparation.
inline std::atomic<u64> g_match_serial{1};
/// The last break's reason and, for Register and Method, its 3D method (diagnostics).
inline std::atomic<u32> g_last_break{};

[[nodiscard]] bool IsExperimentBreak(Reason reason, u32 method);

inline void Break(Reason reason, u32 method = 0) {
    if (Enabled()) {
        g_serial.fetch_add(1, std::memory_order_relaxed);
        if (g_drawx_uniforms && reason != Reason::Register) {
            g_keep_serial.fetch_add(1, std::memory_order_relaxed);
        }
        g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
        if (DrawCostStatsEnabled()) {
            // Sample only the optional bookkeeping, including its extra serial. The clock pair
            // and sample-result atomics are outside this interval. Sampling avoids two clock
            // reads on every break; report sample count so extrapolation is explicit.
            static thread_local u32 sample_sequence{};
            const bool sample = (++sample_sequence & 1023U) == 0;
            const u64 start = sample ? CostNow() : 0;
            if (!IsExperimentBreak(reason, method)) {
                g_match_serial.fetch_add(1, std::memory_order_relaxed);
            }
            g_break_classes.fetch_or(ClassifyBreak(reason, method), std::memory_order_relaxed);
            if (reason == Reason::Method && method < g_method_breaks.size()) {
                g_method_breaks[method].fetch_add(1, std::memory_order_relaxed);
            }
            if (sample) {
                g_bookkeeping_ns.fetch_add(CostNow() - start, std::memory_order_relaxed);
                g_bookkeeping_samples.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

inline u64 Serial() {
    return g_serial.load(std::memory_order_relaxed);
}

/// The device memory a GPU range maps to. A write through another GPU address that maps the
/// same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless that range of memory was translated last
    /// and memory's mappings haven't changed since.
    void Update(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const;

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    std::vector<std::pair<DAddr, DAddr>> spans; ///< Sorted, disjoint [begin, end) pairs
};

/// Light changes since the last draw that can reach texture handles. Written and read on the
/// thread that runs the 3D engine and the rasterizer (the split's back).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    /// Constant buffer data written, by GPU address (unmapped addresses included)...
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    /// ...and by the device address it landed on.
    std::array<DAddr, MaxWrites> device_begin{};
    std::array<DAddr, MaxWrites> device_end{};
    u32 num_device_writes{};
    bool vertex_layout{}; ///< Requires a fresh pipeline lookup, then dynamic vertex input.
    /// EDEN_NX2_DRAWX_PIPESTATE: a fixed-function register changed; requires a fresh pipeline
    /// lookup that selects the same pipeline object.
    bool pipeline_state{};
    /// EDEN_NX2_DRAWX_NEUTRAL / _PIPESTATE: a write that ends the chain without the switches did
    /// not (statistics and verify attribution only).
    bool drawx_skipped{};
    bool queries{};       ///< Query side effects already ran, including conditional rendering.
    bool vertex_id{};
    u8 match_cases{};  ///< Three-bit experiment class, observed in both on and off windows.
    bool overflowed{}; ///< At least one lost interval since the preceding draw attempt.

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    /// Whether data written reaches [address, address + size), whose device memory is spans: at
    /// one of its GPU addresses, or through any GPU address that maps its device memory.
    [[nodiscard]] bool Reaches(GPUVAddr address, u64 size, const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (spans.Intersects(device_begin[write], device_end[write])) {
                return true;
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// Appends [begin, end) to a delta's range list, or extends its last range when that ends at
/// begin. False when the list is full.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return true;
    }
    if (count == Delta::MaxWrites) {
        return false;
    }
    begins[count] = begin;
    ends[count] = end;
    ++count;
    return true;
}

inline void NoteBind(size_t stage, u32 slot) {
    if (Enabled()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// Inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (!Enabled()) {
        return;
    }
    Delta& delta = g_delta;
    if (!AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end)) {
        delta.overflowed = true;
        Break(Reason::DeltaOverflow);
        if (DrawCostStatsEnabled()) {
            g_gpu_overflows.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

/// The device addresses [begin, end) that inline constant buffer data landed on.
inline void NoteDeviceData(DAddr begin, DAddr end) {
    if (!Enabled()) {
        return;
    }
    Delta& delta = g_delta;
    if (!AppendRange(delta.device_begin, delta.device_end, delta.num_device_writes, begin, end)) {
        delta.overflowed = true;
        Break(Reason::DeltaOverflow);
        if (DrawCostStatsEnabled()) {
            g_device_overflows.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

} // namespace VideoCommon::Nx2FastDraw
