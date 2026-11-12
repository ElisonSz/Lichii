// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_sync_suspend.h"

namespace VideoCommon {
struct SyncSuspendTestAccess {
    /// An entry of `words` words was just sent (as Commit does, without the ring).
    static void Send(GpuSplit& split, u64 words) {
        split.last_entry_start = split.write_position;
        split.write_position += words;
    }
    /// The back ran everything before `position` and stored it.
    static void Consume(GpuSplit& split, u64 position) {
        split.consumed.store(position, std::memory_order_seq_cst);
        split.NotifyFront();
    }
    /// The fenced writes of the entries before `position` landed.
    static void Land(GpuSplit& split, u64 position) {
        split.fenced_landed.store(split.EntryTag(position), std::memory_order_release);
    }
    static void SetRunning(GpuSplit& split, bool running) {
        split.back_running.store(running, std::memory_order_seq_cst);
    }
    static void ArmEvent(GpuSplit& split) {
        split.front_event_waiting.store(true, std::memory_order_seq_cst);
    }
    static bool EventArmed(const GpuSplit& split) {
        return split.front_event_waiting.load(std::memory_order_seq_cst);
    }
    static bool EventSet(const GpuSplit& split) {
        return split.front_event.IsSet();
    }
    static u64 Published(const GpuSplit& split) {
        return split.published.load(std::memory_order_seq_cst);
    }
    static u64 WaitTarget(const GpuSplit& split) {
        return split.wait_target.load(std::memory_order_seq_cst);
    }
};
} // namespace VideoCommon

namespace {

using Split = VideoCommon::GpuSplit;
using Access = VideoCommon::SyncSuspendTestAccess;
using Landing = Split::Landing;
using Waiters = Tegra::Nx2SemaphoreWaiters;
namespace Suspension = VideoCommon::SyncSuspend;

struct SwitchScope {
    explicit SwitchScope(bool value, bool relaxed = false)
        : previous{Suspension::enabled}, previous_relaxed{Suspension::relaxed} {
        Suspension::enabled = value;
        Suspension::relaxed = relaxed;
    }
    ~SwitchScope() {
        Suspension::enabled = previous;
        Suspension::relaxed = previous_relaxed;
    }
    SwitchScope(const SwitchScope&) = delete;
    SwitchScope& operator=(const SwitchScope&) = delete;
    bool previous;
    bool previous_relaxed;
};

void SetSwitch(const char* value) {
#ifdef _WIN32
    _putenv_s("EDEN_NX2_SYNC_SUSPEND", value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv("EDEN_NX2_SYNC_SUSPEND", value, 1);
    } else {
        unsetenv("EDEN_NX2_SYNC_SUSPEND");
    }
#endif
}

constexpr u64 WordX = 0x2b000;
constexpr u64 WordY = 0x2b010;

} // namespace

TEST_CASE("Sync suspend settle positions cover the work that can still change the word",
          "[video_core][sync_suspend]") {
    SwitchScope scope{true};
    auto split = std::make_unique<Split>();
    Access::Send(*split, 10);
    split->NoteWriter(); // Compute or copy work ending at 10
    Access::Send(*split, 6);
    split->NoteSemaphoreWrite(WordX, 1, true, Landing::Fenced);
    auto queued = split->QueuedSemaphoreWriteAt(WordX);
    REQUIRE(queued);
    REQUIRE(queued->fenced);
    REQUIRE(queued->position == 16);
    REQUIRE(queued->settle == 10);

    SECTION("a queued write landing at run before it") {
        Access::Send(*split, 4);
        split->NoteSemaphoreWrite(WordY, 5, true, Landing::AtRun); // at 20
        Access::Send(*split, 5);
        split->NoteSemaphoreWrite(WordY, 6, true, Landing::Fenced); // at 25
        queued = split->QueuedSemaphoreWriteAt(WordY);
        REQUIRE(queued);
        REQUIRE(queued->settle == 20);
    }
    SECTION("a write before it that the back already ran") {
        Access::Send(*split, 4);
        split->NoteSemaphoreWrite(WordY, 5, true, Landing::AtRun); // at 20
        Access::Consume(*split, 20);
        Access::Send(*split, 5);
        split->NoteSemaphoreWrite(WordY, 6, true, Landing::Fenced); // at 25
        queued = split->QueuedSemaphoreWriteAt(WordY);
        REQUIRE(queued);
        REQUIRE(queued->settle == 10);
    }
    SECTION("a pending fenced write before it counts with what it followed, not its own entry") {
        Access::Send(*split, 4);
        split->NoteWriter(); // at 20
        Access::Send(*split, 1);
        split->NoteSemaphoreWrite(WordY, 5, true, Landing::Fenced); // at 21, settle 20
        Access::Send(*split, 4);
        split->NoteSemaphoreWrite(WordY, 6, true, Landing::Fenced); // at 25
        queued = split->QueuedSemaphoreWriteAt(WordY);
        REQUIRE(queued);
        REQUIRE(queued->fenced);
        REQUIRE(queued->settle == 20);
    }
    SECTION("a write landing at run behind a pending fenced one waits for itself") {
        Access::Send(*split, 4);
        split->NoteSemaphoreWrite(WordX, 2, true, Landing::AtRun); // at 20
        queued = split->QueuedSemaphoreWriteAt(WordX);
        REQUIRE(queued);
        REQUIRE(queued->fenced); // May still be overtaken by the fenced one at 16
        REQUIRE_FALSE(queued->known);
        REQUIRE(queued->settle == 20);
    }
    SECTION("a write the front could not place") {
        Access::Send(*split, 14);
        split->NoteUnplacedWrite(); // at 30
        Access::Send(*split, 1);
        split->NoteSemaphoreWrite(WordY, 1, true, Landing::Fenced); // at 31
        queued = split->QueuedSemaphoreWriteAt(WordY);
        REQUIRE(queued);
        REQUIRE(queued->settle == 30);
    }
    SECTION("landed and ran: no longer queued") {
        Access::Consume(*split, 16);
        Access::Land(*split, 16);
        REQUIRE_FALSE(split->QueuedSemaphoreWriteAt(WordX));
    }
}

TEST_CASE("Sync suspend measurement mode leaves compute and copy work out of the settle position",
          "[video_core][sync_suspend]") {
    SwitchScope scope{true, true};
    auto split = std::make_unique<Split>();
    Access::Send(*split, 10);
    split->NoteWriter(); // at 10
    Access::Send(*split, 2);
    split->NoteUnplacedWrite(); // at 12
    Access::Send(*split, 4);
    split->NoteSemaphoreWrite(WordX, 1, true, Landing::AtRun); // at 16
    Access::Send(*split, 4);
    split->NoteWriter(); // at 20
    Access::Send(*split, 1);
    split->NoteSemaphoreWrite(WordX, 2, true, Landing::Fenced); // at 21
    const auto queued = split->QueuedSemaphoreWriteAt(WordX);
    REQUIRE(queued);
    // The unplaced write and the queued write landing at run still count, the writers not.
    REQUIRE(queued->settle == 16);
}

TEST_CASE("Sync suspend switch values", "[video_core][sync_suspend]") {
    const bool was_enabled = Suspension::enabled;
    const bool was_relaxed = Suspension::relaxed;
    const auto read_switch = [](const char* value) {
        SetSwitch(value);
        Suspension::ReadSwitch();
        return std::pair{Suspension::enabled, Suspension::relaxed};
    };
    REQUIRE(read_switch(nullptr) == std::pair{false, false});
    REQUIRE(read_switch("0") == std::pair{false, false});
    REQUIRE(read_switch("1") == std::pair{true, false});
    REQUIRE(read_switch("2") == std::pair{true, true});
    REQUIRE(read_switch("12") == std::pair{false, false});
    REQUIRE(read_switch("yes") == std::pair{false, false});
    SetSwitch(nullptr);
    Suspension::enabled = was_enabled;
    Suspension::relaxed = was_relaxed;
}

TEST_CASE("Sync suspend off leaves the semaphore notes as they were",
          "[video_core][sync_suspend]") {
    SwitchScope scope{false};
    auto split = std::make_unique<Split>();
    Access::Send(*split, 10);
    split->NoteWriter();
    Access::Send(*split, 4);
    split->NoteSemaphoreWrite(WordY, 5, true, Landing::AtRun);
    Access::Send(*split, 2);
    split->NoteSemaphoreWrite(WordY, 6, true, Landing::Fenced);
    const auto queued = split->QueuedSemaphoreWriteAt(WordY);
    REQUIRE(queued);
    REQUIRE(queued->payload == 6);
    REQUIRE(queued->known);
    REQUIRE(queued->fenced);
    REQUIRE(queued->position == 16);
    REQUIRE(queued->settle == 0);
    // The back's notification leaves the bounded wait's event alone.
    Access::ArmEvent(*split);
    Access::Consume(*split, 16);
    REQUIRE_FALSE(Access::EventSet(*split));
}

TEST_CASE("Sync suspend decisions ask only what they need", "[video_core][sync_suspend]") {
    using Decision = Suspension::Decision;
    u32 settled_asked = 0;
    u32 met_asked = 0;
    const auto decide = [&](bool work, bool settled, bool met) {
        return Suspension::Decide(
            work,
            [&] {
                ++settled_asked;
                return settled;
            },
            [&] {
                ++met_asked;
                return met;
            });
    };
    REQUIRE(decide(false, true, false) == Decision::NoWork);
    REQUIRE(settled_asked == 0);
    REQUIRE(met_asked == 0);
    REQUIRE(decide(true, false, false) == Decision::Settle);
    REQUIRE(settled_asked == 1);
    REQUIRE(met_asked == 0);
    REQUIRE(decide(true, true, true) == Decision::Met);
    REQUIRE(decide(true, true, false) == Decision::Suspend);
    REQUIRE(settled_asked == 3);
    REQUIRE(met_asked == 2);

    Suspension::queued_commands.store(0);
    Suspension::other_pending = 0;
    REQUIRE_FALSE(Suspension::WorkQueued());
    Suspension::queued_commands.store(-2);
    REQUIRE_FALSE(Suspension::WorkQueued());
    Suspension::queued_commands.store(1);
    REQUIRE(Suspension::WorkQueued());
    Suspension::queued_commands.store(0);
    Suspension::other_pending = 1;
    REQUIRE(Suspension::WorkQueued());
    Suspension::other_pending = 0;
}

TEST_CASE("Sync suspend bounded position wait", "[video_core][sync_suspend]") {
    using namespace std::chrono_literals;
    SwitchScope scope{true};
    auto split = std::make_unique<Split>();
    Access::Send(*split, 100);
    Access::SetRunning(*split, true);

    SECTION("a position already run returns at once and publishes nothing") {
        Access::Consume(*split, 60);
        REQUIRE(split->WaitForPositionFor(50, 1s));
        REQUIRE(Access::Published(*split) == 0);
    }
    SECTION("the back's notification ends the wait well before the limit") {
        const auto begin = std::chrono::steady_clock::now();
        std::thread back([&] {
            std::this_thread::sleep_for(20ms);
            // Past the spin phase by now: the front sleeps on the event.
            Access::Consume(*split, 40);
            std::this_thread::sleep_for(5ms);
            Access::Consume(*split, 80);
        });
        REQUIRE(split->WaitForPositionFor(80, 5s));
        back.join();
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        REQUIRE(elapsed < 2s);
        REQUIRE(Access::Published(*split) == 100); // Published before waiting
        REQUIRE(Access::WaitTarget(*split) == ~u64{0});
        REQUIRE_FALSE(Access::EventArmed(*split));
    }
    SECTION("the target clamps to what was sent") {
        std::thread back([&] {
            std::this_thread::sleep_for(10ms);
            Access::Consume(*split, 100);
        });
        REQUIRE(split->WaitForPositionFor(1000, 5s));
        back.join();
    }
    SECTION("no progress: the wait ends at its limit") {
        const auto begin = std::chrono::steady_clock::now();
        REQUIRE_FALSE(split->WaitForPositionFor(80, 30ms));
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        REQUIRE(elapsed >= 30ms);
        REQUIRE(elapsed < 2s);
        REQUIRE(Access::WaitTarget(*split) == ~u64{0});
    }
    SECTION("a stopped back ends the wait") {
        Access::SetRunning(*split, false);
        REQUIRE_FALSE(split->WaitForPositionFor(80, 5s));
    }
    SECTION("many short waits never lose a notification") {
        u64 position = 0;
        for (u32 round = 0; round < 64; ++round) {
            Access::Send(*split, 1);
            const u64 target = split->Position();
            std::thread back([&split, target] { Access::Consume(*split, target); });
            REQUIRE(split->WaitForPositionFor(target, 5s));
            back.join();
            position = target;
        }
        REQUIRE(split->Ran(position));
    }
}

namespace {

// A model of two or more channels on the split, for the order early suspensions give the ring.
// The split's positions, semaphore notes, settle positions and view are the production code
// (GpuSplit), the waiter latches too (Nx2SemaphoreWaiters), the decision is
// SyncSuspend::Decide; the puller and scheduler flow around them follows
// Puller::AcquireWithoutDrain / PollWithoutDrain and Scheduler::Push / ResumePending.
enum class Op : u8 { Method, Writer, Release, Acquire };

struct Command {
    Op op{};
    std::string label;
    u64 word{};
    u32 value{};
    Landing landing{Landing::AtRun};
};

Command Method(std::string label) {
    return {Op::Method, std::move(label)};
}
Command Writer(std::string label) {
    return {Op::Writer, std::move(label)};
}
Command Release(std::string label, u64 word, u32 value, Landing landing) {
    return {Op::Release, std::move(label), word, value, landing};
}
Command Acquire(u64 word, u32 value) {
    return {Op::Acquire, "acquire", word, value};
}

struct List {
    char channel;
    std::vector<Command> commands;
};

/// When a fence-flagged release lands: as the back runs it, or only once the front has nothing
/// else to send (a host GPU slower than the front).
enum class Fences : u8 { Fast, Slow };

struct Model {
    struct Channel {
        char id{};
        std::deque<Command> commands;
        bool suspended{};
        Command acquire;
        u64 writers_before{};
        std::optional<u64> early_position;
    };
    struct Entry {
        char channel{};
        std::string label;
        std::optional<Command> release;
        bool landed{};
    };

    Model(bool early_, Fences fences_, std::vector<List> lists, std::map<u64, u32> memory_ = {})
        : early{early_}, fences{fences_}, memory{std::move(memory_)} {
        for (auto& list : lists) {
            queue.push_back(std::move(list));
        }
    }

    bool early;
    Fences fences;
    std::unique_ptr<Split> split = std::make_unique<Split>();
    Waiters waiters;
    std::map<u64, u32> memory;
    std::deque<List> queue;
    std::vector<std::unique_ptr<Channel>> channels; ///< In the order they appeared
    std::vector<Entry> ring; ///< Entry i ends at ring position i + 1
    u64 ran{};
    u32 early_suspensions{};
    u32 suspensions{};
    std::vector<std::string> violations;

    [[nodiscard]] std::vector<std::string> Order() const {
        std::vector<std::string> order;
        for (const Entry& entry : ring) {
            order.push_back(std::string{entry.channel} + ":" + entry.label);
        }
        return order;
    }

    Channel& ChannelFor(char id) {
        for (const auto& channel : channels) {
            if (channel->id == id) {
                return *channel;
            }
        }
        auto& added = channels.emplace_back(std::make_unique<Channel>());
        added->id = id;
        return *added;
    }

    static bool Matches(const Command& acquire, u32 current) {
        return current >= acquire.value;
    }

    void Send(Channel& channel, const Command& command) {
        ring.push_back({channel.id, command.label,
                        command.op == Op::Release ? std::optional<Command>{command} : std::nullopt,
                        false});
        Access::Send(*split, 1);
        if (command.op == Op::Writer) {
            split->NoteWriter();
        } else if (command.op == Op::Release) {
            split->NoteSemaphoreWrite(command.word, command.value, true, command.landing);
        }
    }

    void Land(size_t index) {
        Entry& entry = ring[index];
        entry.landed = true;
        memory[entry.release->word] = entry.release->value;
        waiters.OnWrite(entry.release->word, entry.release->value);
        // As GpuSplit::PublishFencedEntries: every entry before the first one the back ran
        // whose fenced release has not landed.
        u64 landed = ran;
        for (size_t i = 0; i < ran; ++i) {
            if (ring[i].release && ring[i].release->landing == Landing::Fenced && !ring[i].landed) {
                landed = i;
                break;
            }
        }
        Access::Land(*split, landed);
    }

    /// The back runs the ring up to `position`.
    void RunTo(u64 position) {
        position = (std::min)(position, u64{ring.size()});
        while (ran < position) {
            const size_t index = static_cast<size_t>(ran);
            Entry& entry = ring[index];
            if (const auto& release = entry.release) {
                if (release->landing == Landing::AtRun) {
                    entry.landed = true;
                    memory[release->word] = release->value;
                    waiters.OnWrite(release->word, release->value);
                }
            }
            ++ran;
            Access::Consume(*split, ran);
            if (entry.release && entry.release->landing == Landing::Fenced &&
                fences == Fences::Fast) {
                Land(index);
            }
        }
    }

    /// The host GPU catches up: every fenced release the back ran lands.
    void LandAll() {
        for (size_t i = 0; i < ran; ++i) {
            if (ring[i].release && ring[i].release->landing == Landing::Fenced && !ring[i].landed) {
                Land(i);
            }
        }
    }

    Split::SemaphoreView View(const Channel& channel) const {
        const Command& acquire = channel.acquire;
        return split->ViewSemaphore(
            acquire.word, false, channel.writers_before,
            [&acquire](u32 value) { return Matches(acquire, value); },
            [this, &acquire](bool) -> std::optional<u32> {
                const auto it = memory.find(acquire.word);
                return it != memory.end() ? it->second : 0U;
            });
    }

    [[nodiscard]] bool WorkQueued(const Channel& self) const {
        if (!queue.empty()) {
            return true;
        }
        return std::any_of(channels.begin(), channels.end(), [&self](const auto& channel) {
            return channel.get() != &self && channel->suspended;
        });
    }

    /// The front's decision on an acquire: true when the channel goes on.
    bool AcquireNow(Channel& channel, const Command& acquire) {
        using Kind = Split::SemaphoreView::Kind;
        channel.acquire = acquire;
        channel.writers_before = split->LastWriterPosition();
        REQUIRE(waiters.Add(&channel, acquire.word, acquire.value, Waiters::AcquireGequal));
        auto answer = View(channel);
        if (early && answer.kind == Kind::Unknown && answer.write_dependency) {
            const auto release = [&]() -> std::optional<Split::QueuedSemaphoreWrite> {
                if (answer.kind != Kind::Unknown || !answer.write_dependency) {
                    return std::nullopt;
                }
                const auto queued = split->QueuedSemaphoreWriteAt(acquire.word);
                if (!queued || !queued->fenced || queued->position != answer.position) {
                    return std::nullopt;
                }
                return queued;
            };
            auto queued = release();
            const auto met = [&] {
                return waiters.Latched(&channel) || Matches(acquire, memory[acquire.word]);
            };
            const auto decide = [&] {
                return Suspension::Decide(
                    WorkQueued(channel), [&] { return split->Ran(queued->settle); }, met);
            };
            auto decision = queued ? decide() : Suspension::Decision::NoWork;
            if (decision == Suspension::Decision::Settle) {
                RunTo(queued->settle);
                answer = View(channel);
                queued = release();
                decision = queued ? decide() : Suspension::Decision::Settle;
            }
            if (decision == Suspension::Decision::Suspend) {
                channel.early_position = answer.position;
                channel.suspended = true;
                ++early_suspensions;
                ++suspensions;
                return false;
            }
        }
        for (u32 waits = 0; answer.kind == Kind::Unknown && waits < 2; ++waits) {
            RunTo(answer.position);
            answer = View(channel);
        }
        if (answer.kind == Kind::Met || answer.kind == Kind::MetByQueued ||
            waiters.Latched(&channel)) {
            Resume(channel);
            return true;
        }
        channel.suspended = true;
        ++suspensions;
        return false;
    }

    /// The scheduler's poll of a suspended channel (PollWithoutDrain).
    bool CanResume(Channel& channel) {
        using Kind = Split::SemaphoreView::Kind;
        if (waiters.Latched(&channel)) {
            return true;
        }
        const auto answer = View(channel);
        if (answer.kind == Kind::Met || answer.kind == Kind::MetByQueued) {
            return true;
        }
        if (channel.early_position && split->Ran(*channel.early_position)) {
            channel.early_position.reset();
        }
        return false;
    }

    void Resume(Channel& channel) {
        // Never earlier than the baseline: a latch, or the word meets it in memory or once the
        // back ran a release the view saw queued before what this channel sends next.
        const auto answer = View(channel);
        const bool met = waiters.Latched(&channel) ||
                         Matches(channel.acquire, memory[channel.acquire.word]) ||
                         answer.kind == Split::SemaphoreView::Kind::MetByQueued;
        if (!met) {
            violations.push_back(std::string{"resumed unmet: "} + channel.id);
        }
        waiters.Remove(&channel);
        channel.suspended = false;
        channel.early_position.reset();
    }

    void Dispatch(Channel& channel) {
        while (!channel.suspended && !channel.commands.empty()) {
            const Command command = channel.commands.front();
            channel.commands.pop_front();
            if (command.op == Op::Acquire) {
                AcquireNow(channel, command);
            } else {
                Send(channel, command);
            }
        }
    }

    void ResumePending() {
        for (bool progress = true; progress;) {
            progress = false;
            for (const auto& channel : channels) {
                if (channel->suspended && CanResume(*channel)) {
                    Resume(*channel);
                    Dispatch(*channel);
                    progress = true;
                }
            }
        }
    }

    [[nodiscard]] bool AnySuspended() const {
        return std::any_of(channels.begin(), channels.end(),
                           [](const auto& channel) { return channel->suspended; });
    }

    std::optional<u64> EarlyWaitPosition() const {
        std::optional<u64> earliest;
        for (const auto& channel : channels) {
            if (channel->suspended && channel->early_position) {
                earliest = earliest ? (std::min)(*earliest, *channel->early_position)
                                    : *channel->early_position;
            }
        }
        return earliest;
    }

    /// The GPU thread's loop until everything ran.
    void Run() {
        for (u32 steps = 0; steps < 1000; ++steps) {
            ResumePending();
            if (!queue.empty()) {
                List list = std::move(queue.front());
                queue.pop_front();
                Channel& channel = ChannelFor(list.channel);
                for (auto& command : list.commands) {
                    channel.commands.push_back(std::move(command));
                }
                Dispatch(channel);
                ResumePending();
                continue;
            }
            if (!AnySuspended()) {
                RunTo(ring.size());
                LandAll();
                return;
            }
            if (early) {
                if (const auto target = EarlyWaitPosition(); target && !split->Ran(*target)) {
                    RunTo(*target); // The idle front's bounded wait
                    continue;
                }
            }
            // Nothing to send: time passes, the back runs everything and the GPU catches up.
            RunTo(ring.size());
            LandAll();
        }
        violations.push_back("did not finish");
    }
};

/// A 3D channel A ends with a fence-flagged release of X = 1 (after compute work), B waits for
/// it, C has independent work queued behind B.
std::vector<List> FencedScenario() {
    return {
        {'A',
         {Method("a1"), Writer("dispatch"), Method("a2"), Release("R", WordX, 1, Landing::Fenced),
          Method("a3")}},
        {'B', {Acquire(WordX, 1), Method("b1"), Method("b2")}},
        {'C', {Method("c1"), Method("c2")}},
    };
}

std::vector<std::string> RunModel(bool early, Fences fences, std::vector<List> lists,
                                  std::map<u64, u32> memory = {}, u32* early_count = nullptr) {
    Model model{early, fences, std::move(lists), std::move(memory)};
    model.Run();
    for (const auto& violation : model.violations) {
        UNSCOPED_INFO(violation);
    }
    REQUIRE(model.violations.empty());
    if (early_count != nullptr) {
        *early_count = model.early_suspensions;
    }
    return model.Order();
}

std::vector<std::string> Labels(std::initializer_list<const char*> labels) {
    return {labels.begin(), labels.end()};
}

} // namespace

TEST_CASE("Sync suspend gives the order the baseline gives when the release lands late",
          "[video_core][sync_suspend]") {
    SwitchScope scope{true};
    const auto slow = RunModel(false, Fences::Slow, FencedScenario());
    const auto fast = RunModel(false, Fences::Fast, FencedScenario());
    // The baseline: B passes after its wait when the fence is quick, else C goes first.
    REQUIRE(slow == Labels({"A:a1", "A:dispatch", "A:a2", "A:R", "A:a3", "C:c1", "C:c2", "B:b1",
                            "B:b2"}));
    REQUIRE(fast == Labels({"A:a1", "A:dispatch", "A:a2", "A:R", "A:a3", "B:b1", "B:b2", "C:c1",
                            "C:c2"}));
    u32 early_slow = 0;
    u32 early_fast = 0;
    const auto suspended_slow =
        RunModel(true, Fences::Slow, FencedScenario(), {}, &early_slow);
    const auto suspended_fast =
        RunModel(true, Fences::Fast, FencedScenario(), {}, &early_fast);
    REQUIRE(early_slow == 1);
    REQUIRE(early_fast == 1);
    // Whatever the fence does, the early suspension gives the baseline's late-fence order.
    REQUIRE(suspended_slow == slow);
    REQUIRE(suspended_fast == slow);
}

TEST_CASE("Sync suspend leaves the baseline order where the baseline passes or has no choice",
          "[video_core][sync_suspend]") {
    SwitchScope scope{true};
    SECTION("the word already meets the acquire") {
        const std::map<u64, u32> memory{{WordX, 1}};
        for (const Fences fences : {Fences::Slow, Fences::Fast}) {
            u32 early = 0;
            const auto off = RunModel(false, fences, FencedScenario(), memory);
            const auto on = RunModel(true, fences, FencedScenario(), memory, &early);
            REQUIRE(early == 0);
            REQUIRE(on == off);
        }
    }
    SECTION("nothing else is queued") {
        std::vector<List> lists = FencedScenario();
        lists.pop_back(); // No C
        for (const Fences fences : {Fences::Slow, Fences::Fast}) {
            u32 early = 0;
            const auto off = RunModel(false, fences, lists);
            const auto on = RunModel(true, fences, lists, {}, &early);
            REQUIRE(early == 0);
            REQUIRE(on == off);
        }
    }
    SECTION("a queued write landing at run before the release meets it") {
        // A first releases X = 1 as the back runs it, then the fenced X = 2: the baseline's wait
        // runs the first, the latch passes B. The settle wait does the same.
        std::vector<List> lists{
            {'A',
             {Method("a1"), Release("P", WordX, 1, Landing::AtRun), Method("a2"),
              Release("R", WordX, 2, Landing::Fenced), Method("a3")}},
            {'B', {Acquire(WordX, 1), Method("b1")}},
            {'C', {Method("c1")}},
        };
        for (const Fences fences : {Fences::Slow, Fences::Fast}) {
            u32 early = 0;
            const auto off = RunModel(false, fences, lists);
            const auto on = RunModel(true, fences, lists, {}, &early);
            REQUIRE(early == 0);
            REQUIRE(on == off);
            REQUIRE(on == Labels({"A:a1", "A:P", "A:a2", "A:R", "A:a3", "B:b1", "C:c1"}));
        }
    }
}

TEST_CASE("Sync suspend keeps each channel's order with two waiting channels",
          "[video_core][sync_suspend]") {
    SwitchScope scope{true};
    const std::vector<List> lists{
        {'A',
         {Method("a1"), Release("R1", WordX, 1, Landing::Fenced), Method("a2"),
          Release("R2", WordY, 1, Landing::Fenced), Method("a3")}},
        {'B', {Acquire(WordX, 1), Method("b1"), Release("S", 0x2c000, 1, Landing::AtRun)}},
        {'D', {Acquire(WordY, 1), Method("d1")}},
        {'C', {Method("c1"), Method("c2")}},
        {'B', {Method("b2")}},
    };
    const auto slow = RunModel(false, Fences::Slow, lists);
    u32 early_slow = 0;
    u32 early_fast = 0;
    const auto on_slow = RunModel(true, Fences::Slow, lists, {}, &early_slow);
    const auto on_fast = RunModel(true, Fences::Fast, lists, {}, &early_fast);
    REQUIRE(early_slow == 2);
    REQUIRE(early_fast == 2);
    REQUIRE(on_slow == slow);
    REQUIRE(on_fast == slow);
    // Per channel the commands keep their order, and each continuation follows its release.
    const auto index = [&on_fast](const char* label) {
        return std::find(on_fast.begin(), on_fast.end(), label) - on_fast.begin();
    };
    REQUIRE(index("B:b1") > index("A:R1"));
    REQUIRE(index("B:S") > index("B:b1"));
    REQUIRE(index("B:b2") > index("B:S"));
    REQUIRE(index("D:d1") > index("A:R2"));
    REQUIRE(index("C:c2") > index("C:c1"));
}
