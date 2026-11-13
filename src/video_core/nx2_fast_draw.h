// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nx2_call_sites.h"

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
    /// A write list lost a range (only with EDEN_NX2_DRAWX_COALESCE: by default a full list widens
    /// its closest range, AppendRange).
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

/// Whether pipelines take their vertex input as dynamic state (VK_EXT_vertex_input_dynamic_state):
/// then the vertex attribute formats are draw parameters, as only each attribute's type class is
/// in the pipeline key. Set by the pipeline cache before the GPU threads run, read on the back.
inline std::atomic<bool> g_dynamic_vertex_input{};

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
/// built from (VideoCommon::VertexInputCache) and recorded by reference. nx2 does that always
/// (RasterizerVulkan::UpdateVertexInput); the switch still logs the cache's hits and misses.
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
/// nx2 keeps uniforms on every draw until a synchronisation point (SyncSerial), so full draws keep
/// them without the switch; with it the DrawX window counts the full draws that keep them and
/// those outside a keep chain (which only nx2's rule keeps).
inline bool g_drawx_uniforms{};
/// EDEN_NX2_DRAWX_TWIN=1: when the pipeline looked up again for a vertex layout, fixed-function or
/// shader program change is another object that keeps the same image state (the same configure
/// specialization, and per stage the same interned stage info or none with images on either
/// side), the delta draw goes on with it, taking what the previous pipeline's last resolve kept.
/// Shader program writes then mark the delta (pipeline_state) instead of ending it.
inline bool g_drawx_twin{};
/// EDEN_NX2_DRAWX_VERIFY=1: delta candidates are prepared in full and compared (as
/// EDEN_NX2_FAST_DRAW_VERIFY), kept uniforms are compared with guest memory; counted per window.
inline bool g_drawx_verify{};
/// EDEN_NX2_DRAWX_KEEP_AUDIT=1 (also on with EDEN_NX2_DRAWX_VERIFY=1): every uniform buffer a keep
/// would bind (on delta, twin and full draws) is bound the original way instead and classified as
/// RebindKeptUniform would treat it: refused (size, address, command buffer) or bound, and then
/// whether guest memory still holds what the kept allocation holds (BufferCache's audit lines).
/// Unlike EDEN_NX2_DRAWX_VERIFY it leaves the delta draws on.
inline bool g_drawx_keep_audit{};
/// EDEN_NX2_DRAWX_NO_KEEPS=1: no uniform keeps at all, also none on delta draws (where they are on
/// by default): every small uniform buffer is compared with the bytes its slot streamed last.
inline bool g_drawx_no_keeps{};
/// EDEN_NX2_DRAWX_COALESCE=1: a full constant data write list (Delta::MaxWrites GPU and device
/// ranges) takes a further range by widening the entry it grows least (CoalesceRange), by at most
/// CoalesceGrowthLimit bytes nobody wrote; a range no merge takes within that ends the chain
/// (Reason::DeltaOverflow). Without the switch a full list widens its closest range however far
/// (AppendRange) and the draw keeps no uniforms (Delta::no_uniform_keeps). Every reader of the
/// lists treats a range as maybe written, so a covering range only makes them more careful, and
/// bounded merges keep uniform keeps. EDEN_NX2_DRAWX_COALESCE_VERIFY=1 also keeps every range
/// exactly and checks the lists cover them when a draw takes them.
inline bool g_drawx_coalesce{};
inline bool g_drawx_coalesce_verify{};
/// EDEN_NX2_DRAWX_DYNSTATE=1: depth bias values (read only by UpdateDepthBias, dynamic state on
/// both paths) and sub_tiling_perf (read by nothing) no longer end the chain, nor does
/// SET_SHADOW_RAM_CONTROL, as a method or as a register write (it only selects how later register
/// writes are taken); the early-Z registers (in the pipeline key) mark the delta for a pipeline
/// lookup as fixed-function state.
inline bool g_drawx_dynstate{};
/// EDEN_NX2_DRAWX_TWIN2=1 (implies EDEN_NX2_DRAWX_TWIN): a twin may also differ in a stage whose
/// image descriptors are the same (GraphicsPipeline::KeepsImageStateOf).
inline bool g_drawx_twin2{};
/// EDEN_NX2_DRAWX_CALLS=1, only with EDEN_NX2_DRAW_REUSE: deferred calls that change nothing a
/// delta draw keeps without ending the chain themselves go to the back as neutral (CallFlags).
inline bool g_drawx_calls{};
/// EDEN_NX2_DRAWX_CALLS2=1, only with EDEN_NX2_DRAW_REUSE: as CALLS for more call sites
/// (CallFlags2): back-run macros (their writes are classified as any other, and query results a
/// DMA read writes out are recorded as written: RasterizerVulkan::FlushQueryResults), channel
/// binds that keep the bound channel, fence signals and waits (SignalSyncPoint, SignalReference,
/// ReleaseFences), semaphore word flushes, cache invalidations and list slot bookkeeping.
inline bool g_drawx_calls2{};
/// EDEN_NX2_DRAWX_CLEARS=1: CLEAR_SURFACE (the method and its register) no longer ends the chain.
/// A clear records into the render pass the next draw uses (which then goes on), or ends it, or
/// binds a helper pipeline (then the next draw is full anyway); images it retires break as
/// deletions do.
inline bool g_drawx_clears{};
/// EDEN_NX2_DRAWX_CLIP=1: SURFACE_CLIP no longer ends the chain. It sizes the framebuffer
/// (UpdateRenderTargets, which delta draws run too, without looking images up for it), the
/// viewports (dynamic state) and the render area words (computed again by delta draws).
inline bool g_drawx_clip{};
/// g_drawx_neutral || g_drawx_pipestate || g_drawx_twin || g_drawx_dynstate || g_drawx_clears ||
/// g_drawx_clip: the one flag the register write path tests.
inline bool g_drawx_registers{};
/// Reads the switches above; called by VideoCore::CreateGPU after ReadDrawReuseSwitches.
void ReadDrawXSwitches();

/// The kind of draw a uniform keep is on (BufferCache::BeginGraphicsUniforms, keep audit).
enum class KeepSource : u8 {
    Delta, ///< A delta draw with the previous draw's pipeline
    Twin,  ///< A delta draw with a twin of it (EDEN_NX2_DRAWX_TWIN)
    Full,  ///< A full draw
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(KeepSource::Count)>
    keep_source_names{"delta", "twin", "full"};

/// EDEN_NX2_DRAWX_UNIFORMS: links full draws into keep chains (the test line's full-draw keep rule,
/// counted in the DrawX window since nx2 keeps uniforms on every draw). g_keep_serial must not
/// have changed since the previous draw began binding its uniforms, and that draw must be the one
/// that took g_delta last: TakeReusableDraw empties g_delta for each draw, so after a draw that
/// returned before binding its uniforms (no pipeline yet, a skipped mesh draw) the next draw's
/// g_delta lacks the binds and constant data written before it. Take() when the draw takes
/// g_delta, Keeps() when it begins binding its uniforms.
class KeepChain {
public:
    /// The serial the previous draw left for this one: 0 (none) unless it reached Keeps().
    [[nodiscard]] u64 Take() noexcept {
        ended_early += open ? 1 : 0;
        open = true;
        return std::exchange(serial, 0);
    }

    /// Whether a full draw may keep uniforms: `taken` (Take's value) is g_keep_serial `now` and
    /// no write list overflowed. Leaves `now` for the next draw.
    [[nodiscard]] bool Keeps(u64 taken, u64 now, bool overflowed) noexcept {
        open = false;
        serial = now;
        return taken != 0 && taken == now && !overflowed;
    }

    void Reset() noexcept {
        serial = 0;
        open = false;
    }

    /// Draws that took g_delta and returned before binding their uniforms (statistics).
    u64 ended_early{};

private:
    u64 serial{}; ///< g_keep_serial when the previous draw reached Keeps (it starts at 1)
    bool open{};  ///< A draw took g_delta and has not reached Keeps yet
};

/// The GpuSplit entry flag of a call that doesn't end fast draw reuse (GpuSplit::FastDrawNeutral,
/// which nx2_fast_draw.cpp checks this against).
inline constexpr u64 SplitFastDrawNeutral = u64{1} << 7;

/// EDEN_NX2_DRAWX_CALLS: the GpuSplit::Defer flags of a deferred call that changes nothing a delta
/// draw keeps, or ends the chain itself when it does (a submit changes the command buffer, fence
/// releases break, written guest memory is recorded as constant data). Off: 0, as before.
[[nodiscard]] inline u64 CallFlags() noexcept {
    return g_drawx_calls ? SplitFastDrawNeutral : 0;
}
/// EDEN_NX2_DRAWX_CALLS2: as CallFlags, for the sites g_drawx_calls2 lists. Off: 0, as before.
[[nodiscard]] inline u64 CallFlags2() noexcept {
    return g_drawx_calls2 ? SplitFastDrawNeutral : 0;
}

/// What a 3D register write does to the delta chain under the switches above.
enum class DrawXRegister : u8 {
    Breaks,          ///< As without the switches
    Neutral,         ///< Read by no draw preparation: no break (EDEN_NX2_DRAWX_NEUTRAL)
    PipelineState,   ///< Fixed-function state: Delta::pipeline_state (EDEN_NX2_DRAWX_PIPESTATE)
    Shader,          ///< Shader programs: Delta::pipeline_state (EDEN_NX2_DRAWX_TWIN)
    DynamicState,    ///< Dynamic state only, or read by nothing: no break (EDEN_NX2_DRAWX_DYNSTATE)
    DynamicPipeline, ///< In the pipeline key: Delta::pipeline_state (EDEN_NX2_DRAWX_DYNSTATE)
    Clear,           ///< CLEAR_SURFACE: no break (EDEN_NX2_DRAWX_CLEARS)
    Clip,            ///< SURFACE_CLIP: no break (EDEN_NX2_DRAWX_CLIP)
};
/// One entry per 3D register (Maxwell3D::Regs::NUM_REGS), filled by ReadDrawXSwitches.
inline std::array<DrawXRegister, 0xe00> g_drawx_register_class{};
[[nodiscard]] inline DrawXRegister DrawXRegisterClass(u32 method) {
    return method < g_drawx_register_class.size() ? g_drawx_register_class[method]
                                                  : DrawXRegister::Breaks;
}
/// The class each register has when every switch is on (tests and ReadDrawXSwitches).
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

/// On its own cache line, away from what every thread reads (g_enabled).
alignas(64) inline std::atomic<u64> g_serial{1};
/// Diagnostic serial ignoring only the three experiment-eligible register/method breaks. It
/// never authorizes reuse: off-window candidates still execute full preparation.
inline std::atomic<u64> g_match_serial{1};
/// The last break's reason and, for Register and Method, its 3D method (diagnostics). Stored by
/// every break, also those that skip the add below, so a register block's one break names what
/// its writes one by one would.
inline std::atomic<u32> g_last_break{};

/// The thread that draws reads the serial before each draw (Serial). Once one of its breaks
/// bumped the serial since that read, more change nothing it compares, so they skip the locked
/// add (every changed 3D register breaks). Threads that never read it (the front's puller, guest
/// CPU write traps, unmaps) always bump. From lucastaonline's 865afaa4f3 (#1161).
inline thread_local bool t_reads_serial{};
inline thread_local bool t_serial_bumped{};

/// Bumped where the guest CPU may have synchronised with the GPU: fences, semaphores, syncpoints,
/// reports, frames, other engines and requests from other threads. After one the guest may have
/// written memory the caches were not told about (pages that aren't write-protected), so uniform
/// keeps read their buffers again.
inline std::atomic<u64> g_sync_serial{1};

inline void NoteSync() {
    g_sync_serial.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline u64 SyncSerial() {
    return g_sync_serial.load(std::memory_order_acquire);
}

[[nodiscard]] bool IsExperimentBreak(Reason reason, u32 method);

inline void Break(Reason reason, u32 method = 0) {
    // 3D methods that let the CPU synchronise (reports, syncpoints) note it where they reach the
    // rasterizer. Invalidations end image and shader lifetimes, not guest memory writes, and an
    // overflowing write list ends its draw's uniform keeps itself (Delta::no_uniform_keeps).
    if (reason != Reason::Register && reason != Reason::Method && reason != Reason::ConstantData &&
        reason != Reason::TextureHeaders && reason != Reason::DeltaOverflow &&
        reason != Reason::Invalidation) {
        NoteSync();
    }
    if (!Enabled()) {
        return;
    }
    g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    if (!(t_reads_serial && t_serial_bumped)) {
        g_serial.fetch_add(1, std::memory_order_relaxed);
        t_serial_bumped = t_reads_serial;
    }
    // Every break, also those that skip the add above.
    if (Nx2CallSites::g_enabled && reason != Reason::Call) [[unlikely]] {
        Nx2CallSites::g_other_break.store(true, std::memory_order_relaxed);
    }
    if (g_drawx_uniforms && reason != Reason::Register) {
        g_keep_serial.fetch_add(1, std::memory_order_relaxed);
    }
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

/// The thread that draws: what it compares fast draws with (and from then on its next break
/// bumps the serial again).
inline u64 Serial() {
    t_reads_serial = true;
    t_serial_bumped = false;
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

    /// The GPU address the range starts at.
    [[nodiscard]] GPUVAddr Address() const noexcept {
        return address;
    }

    /// Whether the range maps to one run of device memory in GPU order, starting at LinearBase.
    [[nodiscard]] bool IsLinear() const noexcept {
        return linear;
    }
    [[nodiscard]] DAddr LinearBase() const noexcept {
        return linear_base;
    }

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    bool linear{};
    DAddr linear_base{};
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
    /// Set when the ranges were widened (AppendRange), lost a range (EDEN_NX2_DRAWX_COALESCE) or
    /// reached a buffer of texture handles between its handles: the draw keeps no uniforms, as
    /// when such writes ended draw reuse.
    bool no_uniform_keeps{};
    bool vertex_layout{}; ///< Requires a fresh pipeline lookup, then dynamic vertex input.
    /// EDEN_NX2_DRAWX_PIPESTATE: a fixed-function register changed; requires a fresh pipeline
    /// lookup that selects the same pipeline object (or a twin, EDEN_NX2_DRAWX_TWIN).
    bool pipeline_state{};
    /// EDEN_NX2_DRAWX_TWIN: of those, a shader program register (statistics only).
    bool shader_state{};
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

    /// Reaches for only the bytes [base + first, base + second) of each of words, all inside the
    /// range spans was updated with. Device writes are compared word by word when that range maps
    /// linearly, else with the whole range.
    [[nodiscard]] bool ReachesWords(GPUVAddr base, std::span<const std::pair<u32, u32>> words,
                                    const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            for (const auto& [first, second] : words) {
                if (write_begin[write] < base + second && base + first < write_end[write]) {
                    return true;
                }
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (!spans.Intersects(device_begin[write], device_end[write])) {
                continue;
            }
            if (!spans.IsLinear()) {
                return true;
            }
            for (const auto& [first, second] : words) {
                const DAddr begin = spans.LinearBase() + (base + first - spans.Address());
                const DAddr end = begin + (second - first);
                if (device_begin[write] < end && begin < device_end[write]) {
                    return true;
                }
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// Moves g_delta into `out` and clears it, copying only the ranges in use (Delta is ~0.5 KiB and
/// taken for every draw). Every other member is copied and cleared too.
inline void TakeDelta(Delta& out) {
    Delta& delta = g_delta;
    out.binds = delta.binds;
    out.num_writes = delta.num_writes;
    std::copy_n(delta.write_begin.begin(), delta.num_writes, out.write_begin.begin());
    std::copy_n(delta.write_end.begin(), delta.num_writes, out.write_end.begin());
    out.num_device_writes = delta.num_device_writes;
    std::copy_n(delta.device_begin.begin(), delta.num_device_writes, out.device_begin.begin());
    std::copy_n(delta.device_end.begin(), delta.num_device_writes, out.device_end.begin());
    out.no_uniform_keeps = delta.no_uniform_keeps;
    out.vertex_layout = delta.vertex_layout;
    out.pipeline_state = delta.pipeline_state;
    out.shader_state = delta.shader_state;
    out.drawx_skipped = delta.drawx_skipped;
    out.queries = delta.queries;
    out.vertex_id = delta.vertex_id;
    out.match_cases = delta.match_cases;
    out.overflowed = delta.overflowed;
    delta.binds = {};
    delta.num_writes = 0;
    delta.num_device_writes = 0;
    delta.no_uniform_keeps = false;
    delta.vertex_layout = false;
    delta.pipeline_state = false;
    delta.shader_state = false;
    delta.drawx_skipped = false;
    delta.queries = false;
    delta.vertex_id = false;
    delta.match_cases = 0;
    delta.overflowed = false;
}

/// Adds [begin, end) to a delta's range list: extends a range it touches or overlaps, else
/// appends it. A full list widens the range that grows least instead (true then), which only
/// makes Reaches more conservative.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (begin <= ends[i] && begins[i] <= end) {
            begins[i] = (std::min)(begins[i], begin);
            ends[i] = (std::max)(ends[i], end);
            return false;
        }
    }
    if (count < Delta::MaxWrites) {
        begins[count] = begin;
        ends[count] = end;
        ++count;
        return false;
    }
    u32 best = 0;
    u64 best_growth = ~u64{0};
    for (u32 i = 0; i < count; ++i) {
        const u64 growth = ((std::max)(ends[i], end) - (std::min)(begins[i], begin)) -
                           (ends[i] - begins[i]);
        if (growth < best_growth) {
            best = i;
            best_growth = growth;
        }
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    return true;
}

/// EDEN_NX2_DRAWX_COALESCE's first step: appends [begin, end) to a delta's range list, or extends
/// its last range when that ends at begin. False when the list is full.
inline bool AppendOrExtendLast(std::array<u64, Delta::MaxWrites>& begins,
                               std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin,
                               u64 end) {
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

/// EDEN_NX2_DRAWX_COALESCE: the most bytes nobody wrote that one merge may add to a range. A list
/// whose every entry is further away than this from the new range overflows as before.
inline constexpr u64 CoalesceGrowthLimit = 0x10000;

/// AppendOrExtendLast; when the list is full, [begin, end) widens the entry it grows by the
/// fewest bytes nobody wrote (none for a range inside or touching an entry), and `merged` is set.
/// False when every merge would add more than CoalesceGrowthLimit such bytes.
inline bool CoalesceRange(std::array<u64, Delta::MaxWrites>& begins,
                          std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end,
                          bool& merged) {
    if (AppendOrExtendLast(begins, ends, count, begin, end)) {
        return true;
    }
    u32 best = Delta::MaxWrites;
    u64 best_growth = CoalesceGrowthLimit + 1;
    for (u32 index = 0; index < count; ++index) {
        const u64 low = (std::min)(begins[index], begin);
        const u64 high = (std::max)(ends[index], end);
        const u64 written = (ends[index] - begins[index]) + (end - begin);
        const u64 growth = high - low > written ? high - low - written : 0;
        if (growth < best_growth) {
            best_growth = growth;
            best = index;
            if (growth == 0) {
                break;
            }
        }
    }
    if (best == Delta::MaxWrites) {
        return false;
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    merged = true;
    return true;
}

/// EDEN_NX2_DRAWX_COALESCE statistics (the back's thread only): ranges merged into a full list,
/// and ranges a full list still could not take.
inline u64 g_coalesced_writes{};
inline u64 g_coalesce_overflows{};

/// EDEN_NX2_DRAWX_COALESCE_VERIFY: every range noted since the delta was taken, exactly
/// (CheckCoalesced compares them with the lists when a draw takes the delta).
struct CoalesceVerify {
    std::vector<std::pair<u64, u64>> gpu;
    std::vector<std::pair<u64, u64>> device;
    u64 checked{};    ///< Ranges compared
    u64 mismatches{}; ///< Ranges no list entry covers
    void Clear() {
        gpu.clear();
        device.clear();
    }
};
inline CoalesceVerify g_coalesce_verify;
/// Checks that `taken`'s lists cover every range noted for it (unless it overflowed, when its
/// readers already treat everything as written), counts and logs ranges they miss, and forgets
/// the ranges.
void CheckCoalesced(const Delta& taken);

/// Appends to one of the delta's range lists as the switches say. False when the range is lost
/// (only with EDEN_NX2_DRAWX_COALESCE: AppendRange widens a full list however far).
inline bool NoteRange(Delta& delta, std::array<u64, Delta::MaxWrites>& begins,
                      std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (!g_drawx_coalesce) {
        delta.no_uniform_keeps |= AppendRange(begins, ends, count, begin, end);
        return true;
    }
    bool merged = false;
    if (!CoalesceRange(begins, ends, count, begin, end, merged)) {
        ++g_coalesce_overflows;
        return false;
    }
    if (merged) {
        // A write that ended the chain without the switch (verify attribution, statistics).
        ++g_coalesced_writes;
        delta.drawx_skipped = true;
    }
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
    if (g_drawx_coalesce_verify) [[unlikely]] {
        g_coalesce_verify.gpu.emplace_back(begin, end);
    }
    if (!NoteRange(delta, delta.write_begin, delta.write_end, delta.num_writes, begin, end)) {
        // The lists miss the range: the chain ends, and the next draw keeps no uniforms.
        delta.overflowed = true;
        delta.no_uniform_keeps = true;
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
    if (g_drawx_coalesce_verify) [[unlikely]] {
        g_coalesce_verify.device.emplace_back(begin, end);
    }
    if (!NoteRange(delta, delta.device_begin, delta.device_end, delta.num_device_writes, begin,
                   end)) {
        // As NoteConstantData's.
        delta.overflowed = true;
        delta.no_uniform_keeps = true;
        Break(Reason::DeltaOverflow);
        if (DrawCostStatsEnabled()) {
            g_device_overflows.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

/// EDEN_NX2_DRAWX_CALLS statistics (the back's thread only): command list ends and semaphore
/// release targets that went on without ending the chain.
inline u64 g_neutral_list_ends{};
inline u64 g_neutral_releases{};

/// EDEN_NX2_DRAWX_CALLS, on the back inside a call deferred with CallFlags(): it would have ended
/// the chain without the switch (verify attribution, statistics). No effect when the switch is
/// off.
inline void NoteNeutralCall(u64& counter) {
    if (g_drawx_calls) {
        g_delta.drawx_skipped = true;
        ++counter;
    }
}

} // namespace VideoCommon::Nx2FastDraw
