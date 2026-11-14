// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include "common/common_funcs.h"
#include "common/fs/path_util.h"
#include "common/hex_util.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/nso.h"

namespace {

constexpr u64 OcarinaTitleId = 0x0400486024DEC000;
constexpr u64 OtherTitleId = 0x0100000000010000;
constexpr std::string_view MaxResolution = "Max Resolution (built-in)";
constexpr std::string_view DynamicFps = "Dynamic FPS by Dark (built-in)";
constexpr std::string_view FpsUnlock = "FPS Unlock (built-in)";
constexpr std::string_view OcarinaBuildIdHex = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348";
constexpr FileSys::PatchManager::BuildID OcarinaBuildId{
    0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0,
    0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48,
};
constexpr size_t PatchOffset = sizeof(Loader::NSOHeader) + 0x805BD0;
constexpr u32 TbzInstruction = 0x360041E8;
constexpr u32 BranchInstruction = 0x1400020F;
constexpr u32 NopInstruction = 0xD503201F;
constexpr size_t DynamicCallOffset = sizeof(Loader::NSOHeader) + 0x6E6CC4;
constexpr size_t DynamicRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46200;
constexpr size_t DynamicStateOffset = sizeof(Loader::NSOHeader) + 0x4E9FE00;
constexpr size_t DynamicRoutineSize = 120;
constexpr size_t UnlockRoutineSize = 128;
constexpr size_t DynamicStateSize = 0x98;
constexpr size_t WaterHookOffset = sizeof(Loader::NSOHeader) + 0x475F8;
constexpr size_t WaterMulOffset = sizeof(Loader::NSOHeader) + 0x4760C;
constexpr size_t WaterRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46300;
constexpr size_t WaterRoutineSize = 148;
constexpr u32 WaterOriginalInstruction = 0x1E380009;
constexpr u32 WaterMulOriginalInstruction = 0x1B097D08;
constexpr u32 WaterCallInstruction = 0x94EFFB42;
constexpr std::string_view WaterRoutine =
    "0101221E0008211E0820201E0D040054C99200B0294138912A010291EB0309AA6C0140F9"
    "9F0115EB600100546B4100917F010AEB61FFFF544C0140B92B512C8B750100F97F0900B9"
    "8C0500118C0900124C0100B9610940BD0028211E01102C1E0228211E4800381E1F050071"
    "4A000054280080520101221E0038211E01103C1E0048211E600900BDC0035FD6E8031F2A"
    "C0035FD6";
constexpr size_t UnlockStateSize = 0xA8;
constexpr size_t SoundEntryOffset = sizeof(Loader::NSOHeader) + 0x415300;
constexpr size_t SoundStepOffset = sizeof(Loader::NSOHeader) + 0x415334;
constexpr size_t SoundGateOffset = sizeof(Loader::NSOHeader) + 0x3C46800;
constexpr size_t SoundFeedOffset = sizeof(Loader::NSOHeader) + 0x3C46280;
constexpr u32 SoundEntryOriginalInstruction = 0xD503237F; // pacibsp
constexpr u32 SoundEntryBranchInstruction = 0x14E0C540;   // b main+0x3C46800
constexpr u32 SoundStepOriginalInstruction = 0xBD401100;  // ldr s0, [x8, #0x10]
constexpr u32 SoundStepCallInstruction = 0x94E0C3D3;      // bl main+0x3C46280
constexpr std::string_view SoundGate =
    "D09200B010823A9102030035C9810090294944F9290140F9301140BD11102E1E0022311EAA010054"
    "120240BD522A301E4022311ECB000054523A311E120200BD29008052090600B90A000014120200BD"
    "1F0600B9C0035FD61F2003D529008052090600B903000014090640B9690000347F2303D5A43A1F17"
    "C0035FD6";
constexpr std::string_view SoundFeed = "001140BD10102E1E0068301EC0035FD6";
constexpr u32 DynamicCallInstruction = 0x94D57D4F;
constexpr u32 DynamicOriginalInstruction = 0x1E201828;

constexpr std::array<std::pair<Common::FS::EdenPath, const char*>, 4> TestDirectories{{
    {Common::FS::EdenPath::NANDDir, "nand"},
    {Common::FS::EdenPath::SDMCDir, "sdmc"},
    {Common::FS::EdenPath::LoadDir, "load"},
    {Common::FS::EdenPath::DumpDir, "dump"},
}};

void WriteIps(const std::filesystem::path& path, u32 word, size_t offset = PatchOffset) {
    const bool ips32 = offset > 0xFFFFFF;
    std::vector<u8> ips =
        ips32 ? std::vector<u8>{'I', 'P', 'S', '3', '2'} : std::vector<u8>{'P', 'A', 'T', 'C', 'H'};
    if (ips32) {
        ips.push_back(static_cast<u8>(offset >> 24));
    }
    ips.push_back(static_cast<u8>(offset >> 16));
    ips.push_back(static_cast<u8>(offset >> 8));
    ips.push_back(static_cast<u8>(offset));
    ips.push_back(0);
    ips.push_back(sizeof(word));
    for (size_t i = 0; i < sizeof(word); ++i) {
        ips.push_back(static_cast<u8>(word >> (i * 8)));
    }
    if (ips32) {
        ips.push_back('E');
    }
    ips.insert(ips.end(), {'E', 'O', 'F'});
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<const char*>(ips.data()), static_cast<std::streamsize>(ips.size()));
}

class PatchManagerTestContext {
public:
    explicit PatchManagerTestContext(std::optional<u32> user_mod_word = std::nullopt,
                                     size_t user_mod_offset = PatchOffset) {
        for (size_t i = 0; i < TestDirectories.size(); ++i) {
            const auto& [path, name] = TestDirectories[i];
            previous[i] = Common::FS::GetEdenPath(path);
            std::filesystem::create_directories(root / name);
            Common::FS::SetEdenPath(path, root / name);
        }
        const auto title_dir = root / "load" / fmt::format("{:016X}", OcarinaTitleId);
        std::filesystem::create_directories(title_dir);
        if (user_mod_word) {
            const auto exefs = title_dir / "UserMod" / "exefs";
            std::filesystem::create_directories(exefs);
            WriteIps(exefs / fmt::format("{}.ips", OcarinaBuildIdHex), *user_mod_word,
                     user_mod_offset);
        }
        system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
        system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
        system.GetFileSystemController().CreateFactories(*system.GetFilesystem());
    }

    ~PatchManagerTestContext() {
        for (size_t i = 0; i < TestDirectories.size(); ++i) {
            Common::FS::SetEdenPath(TestDirectories[i].first, previous[i]);
        }
        Settings::values.disabled_addons.erase(OcarinaTitleId);
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    PatchManagerTestContext(const PatchManagerTestContext&) = delete;
    PatchManagerTestContext& operator=(const PatchManagerTestContext&) = delete;

    FileSys::PatchManager Manager(u64 title_id = OcarinaTitleId) {
        return FileSys::PatchManager{title_id, system.GetFileSystemController(),
                                     system.GetContentProvider()};
    }

private:
    std::array<std::filesystem::path, TestDirectories.size()> previous;
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("builtin-patch-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Core::System system;
};

std::vector<u8> MakeMain(const FileSys::PatchManager::BuildID& build_id, u32 word) {
    std::vector<u8> nso(PatchOffset + 0x100);
    Loader::NSOHeader header{};
    header.magic = Common::MakeMagic('N', 'S', 'O', '0');
    header.build_id = build_id;
    std::memcpy(nso.data(), &header, sizeof(header));
    std::memcpy(nso.data() + PatchOffset, &word, sizeof(word));
    return nso;
}

u32 WordAt(const std::vector<u8>& nso, size_t offset = PatchOffset) {
    REQUIRE(nso.size() >= offset + sizeof(u32));
    u32 word;
    std::memcpy(&word, nso.data() + offset, sizeof(word));
    return word;
}

u32 PatchMain(PatchManagerTestContext& context, const FileSys::PatchManager::BuildID& build_id,
              u32 word, const std::string& name = "main", u64 title_id = OcarinaTitleId) {
    return WordAt(context.Manager(title_id).PatchNSO(MakeMain(build_id, word), name));
}

} // Anonymous namespace

TEST_CASE("Built-in Max Resolution patch is gated on build id and original word",
          "[patch_manager]") {
    PatchManagerTestContext context;

    SECTION("Matching build id and original word applies") {
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == BranchInstruction);
    }

    SECTION("Wrong original word skips") {
        REQUIRE(PatchMain(context, OcarinaBuildId, NopInstruction) == NopInstruction);
    }

    SECTION("Other build id skips") {
        auto other = OcarinaBuildId;
        other[0] ^= 1;
        REQUIRE(PatchMain(context, other, TbzInstruction) == TbzInstruction);
    }

    SECTION("Other module skips") {
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction, "sdk") == TbzInstruction);
    }

    SECTION("Other title skips") {
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction, "main", OtherTitleId) ==
                TbzInstruction);
    }

    SECTION("Disabled in Add-Ons skips") {
        Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == TbzInstruction);
    }
}

TEST_CASE("Built-in Max Resolution patch defers to a user mod at the same offset",
          "[patch_manager]") {
    SECTION("A different word from the mod is kept") {
        PatchManagerTestContext context{NopInstruction};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == NopInstruction);
    }

    SECTION("The same word from the mod is applied once") {
        PatchManagerTestContext context{BranchInstruction};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == BranchInstruction);
    }

    SECTION("Disabling the built-in patch keeps the mod") {
        PatchManagerTestContext context{BranchInstruction};
        Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == BranchInstruction);
    }
}

TEST_CASE("Built-in Max Resolution patch is listed and loaded for Ocarina's main",
          "[patch_manager]") {
    PatchManagerTestContext context;
    const auto manager = context.Manager();

    REQUIRE(manager.HasNSOPatch(OcarinaBuildId, "main"));
    REQUIRE_FALSE(manager.HasNSOPatch(OcarinaBuildId, "sdk"));
    REQUIRE_FALSE(context.Manager(OtherTitleId).HasNSOPatch(OcarinaBuildId, "main"));

    const auto builtin_enabled = [&] {
        const auto patches = manager.GetPatches();
        const auto it = std::ranges::find(patches, MaxResolution, &FileSys::Patch::name);
        REQUIRE(it != patches.end());
        REQUIRE(it->type == FileSys::PatchType::Mod);
        return it->enabled;
    };
    REQUIRE(builtin_enabled());

    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    REQUIRE_FALSE(builtin_enabled());
}

namespace {

std::vector<u8> MakeDynamicMain() {
    auto nso = MakeMain(OcarinaBuildId, TbzInstruction);
    nso.resize(DynamicStateOffset + UnlockStateSize);
    Loader::NSOHeader header;
    std::memcpy(&header, nso.data(), sizeof(header));
    header.segments[0].size = 63201744;
    header.segments[1].location = 63205376;
    header.segments[1].size = 12603520;
    header.segments[2].location = 75812864;
    header.segments[2].size = 5217152;
    header.segments[2].bss_size = 1414272;
    std::memcpy(nso.data(), &header, sizeof(header));
    std::memcpy(nso.data() + DynamicCallOffset, &DynamicOriginalInstruction,
                sizeof(DynamicOriginalInstruction));
    std::memcpy(nso.data() + WaterHookOffset, &WaterOriginalInstruction,
                sizeof(WaterOriginalInstruction));
    std::memcpy(nso.data() + WaterMulOffset, &WaterMulOriginalInstruction,
                sizeof(WaterMulOriginalInstruction));
    std::memcpy(nso.data() + SoundEntryOffset, &SoundEntryOriginalInstruction,
                sizeof(SoundEntryOriginalInstruction));
    std::memcpy(nso.data() + SoundStepOffset, &SoundStepOriginalInstruction,
                sizeof(SoundStepOriginalInstruction));
    return nso;
}

} // Anonymous namespace

TEST_CASE("Built-in FPS Unlock defaults on and replaces Dynamic FPS", "[patch_manager]") {
    PatchManagerTestContext context;
    const auto manager = context.Manager();
    const auto listed = [&](std::string_view name) {
        const auto patches = manager.GetPatches();
        const auto it = std::ranges::find(patches, name, &FileSys::Patch::name);
        REQUIRE(it != patches.end());
        REQUIRE(it->type == FileSys::PatchType::Mod);
        return *it;
    };
    const auto routine = [](const std::vector<u8>& nso, size_t size) {
        return Common::HexToString(std::span{nso}.subspan(DynamicRoutineOffset, size));
    };
    constexpr std::string_view RoutineHead =
        "2818201ECA9200B04A0138912BE03BD54C0140F94B0100F9CC0200B46B010CCB0DE03BD5"
        "4E0180D2AE09CE9A7F010EEBCB818B9A4C0540F94C0000B5EC030BAA6E010CCB8C0D8E8B"
        "4C0500F98201239EA301239E4218231E4208211E";
    const auto water_patched = [](const std::vector<u8>& nso) {
        return WordAt(nso, WaterHookOffset) == WaterCallInstruction &&
               WordAt(nso, WaterMulOffset) == NopInstruction &&
               Common::HexToString(std::span{nso}.subspan(WaterRoutineOffset, WaterRoutineSize)) ==
                   WaterRoutine;
    };
    const auto water_untouched = [](const std::vector<u8>& nso) {
        return WordAt(nso, WaterHookOffset) == WaterOriginalInstruction &&
               WordAt(nso, WaterMulOffset) == WaterMulOriginalInstruction &&
               WordAt(nso, WaterRoutineOffset) == 0;
    };
    const auto hex_at = [](const std::vector<u8>& nso, size_t offset, size_t size) {
        return Common::HexToString(std::span{nso}.subspan(offset, size));
    };
    const auto sound_gate_patched = [&](const std::vector<u8>& nso) {
        return WordAt(nso, SoundEntryOffset) == SoundEntryBranchInstruction &&
               WordAt(nso, SoundStepOffset) == SoundStepCallInstruction &&
               hex_at(nso, SoundGateOffset, SoundGate.size() / 2) == SoundGate &&
               hex_at(nso, SoundFeedOffset, SoundFeed.size() / 2) == SoundFeed;
    };
    const auto sound_gate_untouched = [](const std::vector<u8>& nso) {
        return WordAt(nso, SoundEntryOffset) == SoundEntryOriginalInstruction &&
               WordAt(nso, SoundStepOffset) == SoundStepOriginalInstruction &&
               WordAt(nso, SoundGateOffset) == 0 && WordAt(nso, SoundFeedOffset) == 0;
    };
    const auto nso = MakeDynamicMain();

    // Default: FPS Unlock applies, Dynamic FPS is listed as replaced by it when it applies.
    REQUIRE(listed(FpsUnlock).enabled);
    REQUIRE(listed(DynamicFps).enabled);
    REQUIRE(listed(DynamicFps).version ==
            fmt::format("IPS, replaced by {} when it applies", FpsUnlock));
    auto patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(routine(patched, UnlockRoutineSize) ==
            std::string{RoutineHead} + "E300001C4268231EC300001C4278231E4840201EC0035FD6"
                                       "1F2003D50AD7A33C0000C040");
    REQUIRE(WordAt(patched) == BranchInstruction);
    REQUIRE(water_patched(patched));
    REQUIRE(sound_gate_patched(patched));

    // Unchecking only Dynamic FPS keeps FPS Unlock.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{DynamicFps}};
    REQUIRE_FALSE(listed(DynamicFps).enabled);
    REQUIRE(routine(manager.PatchNSO(nso, "main"), UnlockRoutineSize) ==
            routine(patched, UnlockRoutineSize));

    // Without FPS Unlock, Dark's routine applies unchanged.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{FpsUnlock}};
    REQUIRE_FALSE(listed(FpsUnlock).enabled);
    REQUIRE(listed(DynamicFps).version == "IPS");
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(routine(patched, DynamicRoutineSize) ==
            std::string{RoutineHead} + "C300001C0309231E4020231E4D0000544840201EC0035FD6"
                                       "0AD7833F");
    REQUIRE(WordAt(patched, DynamicRoutineOffset + DynamicRoutineSize) == 0);
    REQUIRE(water_patched(patched));
    // Dark's step stays at or above 1.0, so the sound manager keeps running every frame.
    REQUIRE(sound_gate_untouched(patched));

    // Both unchecked: hook and code cave stay untouched.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{FpsUnlock},
                                                        std::string{DynamicFps}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicOriginalInstruction);
    REQUIRE(WordAt(patched, DynamicRoutineOffset) == 0);
    REQUIRE(water_untouched(patched));
    REQUIRE(sound_gate_untouched(patched));

    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(WordAt(patched) == TbzInstruction);

    const auto unrelated = context.Manager(OtherTitleId).GetPatches();
    REQUIRE(std::ranges::find(unrelated, DynamicFps, &FileSys::Patch::name) == unrelated.end());
    REQUIRE(std::ranges::find(unrelated, FpsUnlock, &FileSys::Patch::name) == unrelated.end());
}

TEST_CASE("Built-in FPS Unlock skips the sound gate's sites only as a whole patch",
          "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    auto nso = MakeDynamicMain();
    SECTION("Wrong original job entry") {
        nso[SoundEntryOffset] ^= 1;
    }
    SECTION("Wrong original step load") {
        nso[SoundStepOffset] ^= 1;
    }
    SECTION("Nonzero gate cave") {
        nso[SoundGateOffset + 4] = 1;
    }
    SECTION("Nonzero sound state") {
        nso[DynamicStateOffset + 0xA4] = 1;
    }
    // FPS Unlock doesn't apply, so Dark's routine does, without the sound gate.
    const auto patched = context.Manager().PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(WordAt(patched, DynamicRoutineOffset + DynamicRoutineSize) == 0);
    REQUIRE(WordAt(patched, SoundStepOffset) != SoundStepCallInstruction);
    REQUIRE(WordAt(patched, SoundEntryOffset) != SoundEntryBranchInstruction);
}

TEST_CASE("Built-in Dynamic FPS rejects mismatched or truncated modules transactionally",
          "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    auto nso = MakeDynamicMain();
    std::string module = "main";
    u64 title = OcarinaTitleId;
    SECTION("Wrong build") {
        nso[offsetof(Loader::NSOHeader, build_id)] ^= 1;
    }
    SECTION("Wrong module") {
        module = "sdk";
    }
    SECTION("Wrong title") {
        title = OtherTitleId;
    }
    SECTION("Incomplete header") {
        nso.resize(sizeof(Loader::NSOHeader) - 1);
    }
    SECTION("Incomplete call") {
        nso.resize(DynamicCallOffset + 3);
    }
    SECTION("Incomplete routine") {
        nso.resize(DynamicRoutineOffset + DynamicRoutineSize - 1);
    }
    SECTION("Incomplete timer state") {
        nso.resize(DynamicStateOffset + DynamicStateSize - 1);
    }
    SECTION("Timer state outside BSS") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[2].bss_size = 15;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    SECTION("Wrong original hook") {
        nso[DynamicCallOffset] ^= 1;
    }
    SECTION("Nonzero code cave") {
        nso[DynamicRoutineOffset + DynamicRoutineSize - 1] = 1;
    }
    SECTION("Nonzero timer state") {
        nso[DynamicStateOffset + DynamicStateSize - 1] = 1;
    }
    SECTION("Wrong original water hook") {
        nso[WaterHookOffset] ^= 1;
    }
    SECTION("Wrong original water multiply") {
        nso[WaterMulOffset] ^= 1;
    }
    SECTION("Nonzero water code cave") {
        nso[WaterRoutineOffset + WaterRoutineSize - 1] = 1;
    }
    SECTION("Code cave outside executable padding") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[0].size = 0x3C46000;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    SECTION("Code cave overlaps read-only segment") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[1].location = 0x3C46200;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    REQUIRE(context.Manager(title).PatchNSO(nso, module) == nso);
}

TEST_CASE("Built-in Dynamic FPS leaves both records alone when a user mod changes either",
          "[patch_manager]") {
    size_t offset = DynamicCallOffset;
    u32 mod_word = 0x11223344;
    SECTION("User call takes precedence") {}
    SECTION("Identical call still takes precedence for the whole patch") {
        mod_word = DynamicCallInstruction;
    }
    SECTION("User routine takes precedence") {
        offset = DynamicRoutineOffset;
    }
    SECTION("Last routine byte takes precedence") {
        offset = DynamicRoutineOffset + DynamicRoutineSize - sizeof(u32);
    }
    SECTION("User timer state takes precedence") {
        offset = DynamicStateOffset;
    }
    PatchManagerTestContext context{mod_word, offset};
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    const auto nso = MakeDynamicMain();
    auto expected = nso;
    std::memcpy(expected.data() + offset, &mod_word, sizeof(mod_word));
    REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
}

TEST_CASE("Built-in Texture Pool x2 doubles the allocator's initial pool size",
          "[patch_manager]") {
    constexpr std::string_view TexturePool = "Texture Pool x2 (built-in)";
    constexpr size_t CallOffset = sizeof(Loader::NSOHeader) + 0xC2A36C;
    constexpr size_t RoutineOffset = sizeof(Loader::NSOHeader) + 0x3C462C0;
    constexpr u32 StpInstruction = 0xA90127E8;
    constexpr u32 CallInstruction = 0x94C06FD5;
    PatchManagerTestContext context;
    const auto manager = context.Manager();
    auto nso = MakeDynamicMain();
    std::memcpy(nso.data() + CallOffset, &StpInstruction, sizeof(StpInstruction));

    auto patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, CallOffset) == CallInstruction);
    REQUIRE(Common::HexToString(std::span{patched}.subspan(RoutineOffset, 12)) ==
            "08F97FD3E82701A9C0035FD6");
    REQUIRE(WordAt(patched, RoutineOffset + 12) == 0);

    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{TexturePool}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, CallOffset) == StpInstruction);
    REQUIRE(WordAt(patched, RoutineOffset) == 0);

    Settings::values.disabled_addons.erase(OcarinaTitleId);
    std::memcpy(nso.data() + CallOffset, &NopInstruction, sizeof(NopInstruction));
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, CallOffset) == NopInstruction);
    REQUIRE(WordAt(patched, RoutineOffset) == 0);
}

TEST_CASE("Built-in DLSS render scale follows the NVIDIA DLSS quality", "[patch_manager]") {
#ifdef EGLANTINE_STREAMLINE
    constexpr bool DlssBuild = true;
#else
    constexpr bool DlssBuild = false;
#endif
    constexpr size_t ScaleOffset = sizeof(Loader::NSOHeader) + 0x806420;
    constexpr u32 ScaleOne = 0x52A7F009;
    struct SettingsGuard {
        ~SettingsGuard() {
            Settings::values.nvidia_dlss.SetValue(false);
            Settings::values.nvidia_dlss_quality.SetValue(Settings::NvidiaDlssQuality::Automatic);
            Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
        }
    } guard;
    PatchManagerTestContext context;
    auto nso = MakeMain(OcarinaBuildId, TbzInstruction);
    nso.resize(ScaleOffset + sizeof(u32));
    std::memcpy(nso.data() + ScaleOffset, &ScaleOne, sizeof(ScaleOne));
    const auto scale = [&] { return WordAt(context.Manager().PatchNSO(nso, "main"), ScaleOffset); };

    Settings::values.nvidia_dlss.SetValue(true);
    Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Docked);
    using Quality = Settings::NvidiaDlssQuality;
    for (const auto& [quality, word] : std::array<std::pair<Quality, u32>, 6>{{
             {Quality::Automatic, ScaleOne},
             {Quality::Native, ScaleOne},
             {Quality::Quality, ScaleOne},
             {Quality::Balanced, 0x52A7EBE9},
             {Quality::Performance, 0x52A7E809},
             {Quality::UltraPerformance, 0x52A7E009},
         }}) {
        Settings::values.nvidia_dlss_quality.SetValue(quality);
        CHECK(scale() == (DlssBuild ? word : ScaleOne));
        CHECK(Settings::AppliedOcarinaRenderScale() ==
              (DlssBuild && word != ScaleOne ? std::optional{quality} : std::nullopt));
    }

    Settings::values.nvidia_dlss_quality.SetValue(Quality::Performance);
    Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
    CHECK(scale() == ScaleOne);
    Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Docked);
    Settings::values.nvidia_dlss.SetValue(false);
    CHECK(scale() == ScaleOne);
    Settings::values.nvidia_dlss.SetValue(true);
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    CHECK(scale() == ScaleOne);
    CHECK_FALSE(Settings::AppliedOcarinaRenderScale());

    for (const auto& patch : context.Manager().GetPatches()) {
        CHECK(patch.name.find("render scale") == std::string::npos);
    }
}
