// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "video_core/shader_handle_program.h"
#include "video_core/textures/texture.h"

namespace {
namespace IR = Shader::IR;
using IR::Opcode;
using IR::Value;

class HandleEnvironment final : public Shader::Environment {
public:
    bool integer_texture{};
    Shader::TextureType texture_type{Shader::TextureType::Color2D};
    Shader::TexturePixelFormat pixel_format{Shader::TexturePixelFormat::A8B8G8R8_UNORM};
    u32 format_reads{};
    u32 pixel_format_reads{};
    u32 expected_handle{0xabc};
    std::map<std::pair<u32, u32>, u32> cbuf_values{{{0, 0}, 0xabc}};

    u64 ReadInstruction(u32) override { throw std::logic_error("Unexpected instruction read"); }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        const auto it = cbuf_values.find({index, offset});
        REQUIRE(it != cbuf_values.end());
        return it->second;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return texture_type;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        ++pixel_format_reads;
        return pixel_format;
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        REQUIRE(handle == expected_handle);
        ++format_reads;
        return integer_texture;
    }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

struct HandleProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block block{pool};
    IR::Block left{pool};
    IR::Block right{pool};
    IR::Block merge{pool};
    IR::Block* insertion_block{&block};
    IR::Program program;
    HandleEnvironment env;
    bool handle_table_arrays{true};

    HandleProgram() {
        program.stage = Shader::Stage::Fragment;
        program.blocks = {&block};
        program.post_order_blocks = {&block};
    }

    Value Emit(Opcode opcode, std::initializer_list<Value> args = {}) {
        insertion_block->AppendNewInst(opcode, args);
        return Value{&insertion_block->back()};
    }

    Value Storage(u32 offset) {
        return Emit(Opcode::LoadStorage32, {Value{0U}, Value{offset}});
    }

    Value Add(Value lhs, Value rhs) {
        return Emit(Opcode::IAdd32, {lhs, rhs});
    }

    Value Phi(Value first, Value second) {
        block.AddBranch(&left);
        block.AddBranch(&right);
        left.AddBranch(&merge);
        right.AddBranch(&merge);
        program.blocks = {&block, &left, &right, &merge};
        program.post_order_blocks = {&merge, &right, &left, &block};
        insertion_block = &merge;
        const Value value{Emit(Opcode::Phi)};
        value.Inst()->SetFlags(IR::Type::U32);
        value.Inst()->AddPhiOperand(&left, first);
        value.Inst()->AddPhiOperand(&right, second);
        return value;
    }

    IR::Inst& Sample(Value handle, Shader::TextureType type = Shader::TextureType::Color2D) {
        IR::IREmitter ir{*insertion_block};
        IR::TextureInstInfo flags{};
        flags.type.Assign(type);
        return *ir.ImageSampleExplicitLod(handle,
                                         ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f)),
                                         ir.Imm32(0.0f), {}, flags).Inst();
    }

    void Run() {
        Shader::HostTranslateInfo host;
        host.support_handle_table_arrays = handle_table_arrays;
        host.ApplyDescriptorLimitPolicy();
        Shader::Optimization::TexturePass(env, program, host);
    }

    const Shader::TextureDescriptor& Descriptor(const IR::Inst& sample) const {
        REQUIRE(sample.GetOpcode() == Opcode::ImageSampleExplicitLod);
        const u32 index{sample.Flags<IR::TextureInstInfo>().descriptor_index};
        REQUIRE(index < program.info.texture_descriptors.size());
        return program.info.texture_descriptors[index];
    }

    u32 Evaluate(const IR::Inst& sample, u32 first = 0x1234, u32 second = 0x50) const {
        const u32 index{Descriptor(sample).handle_program};
        REQUIRE(index < program.info.handle_programs.size());
        const auto result{Shader::EvaluateHandleProgram(
            program.info.handle_programs[index],
            [](u32, u32) -> u32 { throw std::logic_error("Unexpected evaluator cbuf read"); },
            [=](u32 binding, u32 offset) {
                REQUIRE(binding == 0);
                REQUIRE((offset == 0 || offset == 4));
                return offset == 0 ? first : second;
            })};
        REQUIRE(result.has_value());
        return *result;
    }
};
}

TEST_CASE("Sampled descriptors distinguish static and computed handles",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    auto& first{shader.Sample(shader.Storage(0))};
    auto& second{shader.Sample(shader.Storage(4))};
    auto& repeated{shader.Sample(shader.Storage(0))};
    auto& fixed{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}}))};
    shader.Run();

    CHECK(shader.program.info.handle_programs.size() == 2);
    CHECK(shader.program.info.texture_descriptors.size() == 3);
    CHECK(shader.Descriptor(first).handle_program != shader.Descriptor(second).handle_program);
    CHECK(shader.Descriptor(first).handle_program == shader.Descriptor(repeated).handle_program);
    CHECK(shader.Descriptor(fixed).handle_program == ~0U);
    CHECK(first.Flags<IR::TextureInstInfo>().descriptor_index ==
          repeated.Flags<IR::TextureInstInfo>().descriptor_index);
    CHECK(first.Flags<IR::TextureInstInfo>().descriptor_index !=
          fixed.Flags<IR::TextureInstInfo>().descriptor_index);
    CHECK(shader.Evaluate(first) == 0x1234);
    CHECK(shader.Evaluate(second) == 0x50);
    CHECK(shader.Evaluate(repeated, 0x5678) == 0x5678);
}

TEST_CASE("Computed handles do not share the static texture format cache",
          "[video_core][texture_handles]") {
    for (const bool dynamic_first : {false, true}) {
        CAPTURE(dynamic_first);
        HandleProgram shader;
        shader.env.integer_texture = true;
        const Value dynamic{shader.Storage(0)};
        const Value fixed{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
        auto& first{shader.Sample(dynamic_first ? dynamic : fixed)};
        auto& second{shader.Sample(dynamic_first ? fixed : dynamic)};
        auto& third{shader.Sample(fixed)};
        shader.Run();

        const auto& static_sample{dynamic_first ? second : first};
        const auto& dynamic_sample{dynamic_first ? first : second};
        CHECK(shader.program.info.texture_descriptors.size() == 2);
        CHECK(static_sample.Flags<IR::TextureInstInfo>().is_integer == 1);
        CHECK(third.Flags<IR::TextureInstInfo>().is_integer == 1);
        CHECK(dynamic_sample.Flags<IR::TextureInstInfo>().is_integer == 0);
        CHECK(shader.Descriptor(static_sample).is_integer);
        CHECK_FALSE(shader.Descriptor(dynamic_sample).is_integer);
        CHECK(shader.env.format_reads == 1);
    }
}

TEST_CASE("A sampled handle keeps a storage-indexed cbuf texture with its sampler",
          "[video_core][texture_handles]") {
    for (const bool sampler_first : {false, true}) {
        HandleProgram shader;
        shader.env.cbuf_values = {{{6, 0}, 0x123}, {{2, 0x570}, 0x24000000}};
        shader.env.expected_handle = 0x24000123;
        const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
        const Value texture{shader.Emit(Opcode::GetCbufU32, {Value{6U}, offset})};
        const Value sampler{shader.Emit(Opcode::GetCbufU32, {Value{2U}, Value{0x570U}})};
        auto& sample{shader.Sample(shader.Emit(Opcode::BitwiseOr32,
                                              {sampler_first ? sampler : texture,
                                               sampler_first ? texture : sampler}))};
        shader.Run();
        const auto& descriptor = shader.Descriptor(sample);
        CHECK(descriptor.handle_program == ~0U);
        CHECK(descriptor.count > 1);
        CHECK(descriptor.size_shift == 4);
        CHECK(descriptor.cbuf_index == 6);
        CHECK(descriptor.cbuf_offset == 0);
        CHECK(descriptor.has_secondary);
        CHECK(descriptor.secondary_cbuf_index == 2);
        CHECK(descriptor.secondary_cbuf_offset == 0x570);
        CHECK(shader.program.info.handle_programs.empty());
        REQUIRE_FALSE(sample.Arg(0).IsEmpty());
        const auto* selection = sample.Arg(0).InstRecursive();
        REQUIRE(selection->GetOpcode() == Opcode::UMin32);
        const auto* index = selection->Arg(0).InstRecursive();
        REQUIRE(index->GetOpcode() == Opcode::ShiftRightLogical32);
        CHECK(index->Arg(0).Resolve() == offset.Resolve());
        CHECK(index->Arg(1).U32() == 4);
    }
}

TEST_CASE("Combining static cbuf handles keeps the static descriptor path",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    const Value texture{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
    const Value sampler{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
    auto& sample{shader.Sample(shader.Emit(Opcode::BitwiseOr32, {texture, sampler}))};
    shader.Run();
    CHECK(shader.Descriptor(sample).handle_program == ~0U);
    CHECK(shader.Descriptor(sample).has_secondary);
    CHECK(shader.program.info.handle_programs.empty());
}

TEST_CASE("Texture arrays reserve the final static and buffer resources",
          "[video_core][texture_handles]") {
    for (const u32 arrays : {1U, 2U}) {
        HandleProgram shader;
        const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
        auto& sample{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, offset}))};
        shader.Run();
        auto& info = shader.program.info;
        const auto descriptor = info.texture_descriptors.front();
        // Other textures: descriptors equal to one another would be one after a cut.
        for (u32 i = 1; i < arrays; ++i) {
            auto other = descriptor;
            other.cbuf_offset = 0x100 * i;
            info.texture_descriptors.push_back(other);
        }
        for (u32 i = 0; i < 3; ++i) {
            auto fixed = descriptor;
            fixed.count = 1;
            fixed.is_descriptor_array = false;
            fixed.cbuf_offset = 0x400 + 8 * i;
            info.texture_descriptors.push_back(fixed);
        }
        for (u32 i = 0; i < 5; ++i) {
            info.constant_buffer_descriptors.push_back({.index = i, .count = 1});
        }
        for (u32 i = 0; i < 2; ++i) {
            info.storage_buffers_descriptors.push_back({.count = 1});
        }
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        host.max_per_stage_resources = 16;
        host.max_per_stage_descriptor_sampled_images = 16;
        host.max_descriptor_set_sampled_images = 16;
        Shader::Optimization::LimitSampledTextureArrays(shader.program, host);
        for (u32 i = 0; i < arrays; ++i) {
            CHECK(info.texture_descriptors[i].count == 6 / arrays);
        }
        CHECK(Shader::NumDescriptors(info.texture_descriptors) == 9);
        CHECK(sample.Arg(0).InstRecursive()->Arg(1).U32() == 6 / arrays - 1);
        if (arrays == 2) {
            info.texture_descriptors[0].count = 2;
            info.texture_descriptors[1].count = 12;
            host.max_per_stage_resources = 24;
            host.max_per_stage_descriptor_sampled_images = 24;
            host.max_descriptor_set_sampled_images = 24;
            Shader::Optimization::LimitSampledTextureArrays(shader.program, host);
            CHECK(info.texture_descriptors[0].count == 2);
            CHECK(info.texture_descriptors[1].count == 12);
        }
        host.max_per_stage_resources = 10;
        REQUIRE_THROWS(Shader::Optimization::LimitSampledTextureArrays(shader.program, host));
    }
}

TEST_CASE("Texture arrays fit the sampler limits of their stage and pipeline",
          "[video_core][texture_handles]") {
    std::array<HandleProgram, 2> stages;
    std::array<IR::Inst*, 2> samples{};
    for (size_t i = 0; i < stages.size(); ++i) {
        HandleProgram& shader{stages[i]};
        const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
        samples[i] = &shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, offset}));
        shader.Run();
        REQUIRE(shader.program.info.texture_descriptors.front().count > 16);
    }
    auto& first = stages[0].program.info;
    auto& second = stages[1].program.info;
    // Texel buffers take sampled images, not samplers.
    for (u32 i = 0; i < 4; ++i) {
        first.texture_buffer_descriptors.push_back({.count = 1});
    }
    Shader::HostTranslateInfo host;
    host.ApplyDescriptorLimitPolicy();
    host.max_per_stage_descriptor_sampled_images = 12;
    host.max_per_stage_descriptor_samplers = 8;
    Shader::Optimization::LimitSampledTextureArrays(stages[0].program, host);
    CHECK(first.texture_descriptors.front().count == 8);
    CHECK(samples[0]->Arg(0).InstRecursive()->Arg(1).U32() == 7);

    // The descriptor set limits count the textures of both stages.
    for (const u32 offset : {0x100U, 0x108U}) {
        auto fixed = second.texture_descriptors.front();
        fixed.count = 1;
        fixed.is_descriptor_array = false;
        fixed.cbuf_offset = offset;
        second.texture_descriptors.push_back(fixed);
    }
    host.max_descriptor_set_samplers = 20;
    const std::array programs{&stages[0].program, &stages[1].program};
    Shader::Optimization::LimitPipelineSampledTextureArrays(programs, host);
    CHECK(first.texture_descriptors.front().count == 8);
    CHECK(samples[0]->Arg(0).InstRecursive()->Arg(1).U32() == 7);
    CHECK(second.texture_descriptors.front().count == 9);
    CHECK(samples[1]->Arg(0).InstRecursive()->Arg(1).U32() == 8);

    host.max_descriptor_set_samplers = 3;
    REQUIRE_THROWS(Shader::Optimization::LimitPipelineSampledTextureArrays(programs, host));
}

TEST_CASE("Texture arrays a cap cut to one element still count as arrays",
          "[video_core][texture_handles]") {
    // With one sampler a stage, the pass cuts the array to one element before it has the other
    // textures of the stage. The limits of the stage and of the pipeline still count it.
    Shader::HostTranslateInfo host;
    host.ApplyDescriptorLimitPolicy();
    host.max_per_stage_descriptor_samplers = 1;
    std::array<HandleProgram, 2> stages;
    HandleProgram& shader{stages[0]};
    const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
    auto& sample{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, offset}))};
    Shader::Optimization::TexturePass(shader.env, shader.program, host);
    auto& info{shader.program.info};
    REQUIRE(info.texture_descriptors.size() == 1);
    const auto array{info.texture_descriptors.front()};
    CHECK(array.count == 1);
    CHECK(array.is_descriptor_array);
    CHECK(sample.Arg(0).IsEmpty());
    CHECK_NOTHROW(Shader::Optimization::LimitSampledTextureArrays(shader.program, host));

    // With a texture of another stage, the pipeline exceeds a set of one sampler.
    auto single{array};
    single.is_descriptor_array = false;
    single.cbuf_offset = 0x100;
    stages[1].program.info.texture_descriptors.push_back(single);
    host.max_descriptor_set_samplers = 1;
    const std::array programs{&stages[0].program, &stages[1].program};
    REQUIRE_THROWS(Shader::Optimization::LimitPipelineSampledTextureArrays(programs, host));

    // With one of its own, the stage exceeds its limit.
    info.texture_descriptors.push_back(single);
    REQUIRE_THROWS(Shader::Optimization::LimitSampledTextureArrays(shader.program, host));
    // A stage reading its textures on their own is left as the guest has it.
    info.texture_descriptors.front().is_descriptor_array = false;
    CHECK_NOTHROW(Shader::Optimization::LimitSampledTextureArrays(shader.program, host));
}

TEST_CASE("Texture descriptors a cap makes equal share one binding",
          "[video_core][texture_handles]") {
    // An array of the textures at c0[0] and the texture at c0[0] read on its own: cut to one
    // element, the array is that texture.
    HandleProgram shader;
    const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
    auto& indexed{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, offset}))};
    auto& single{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}}))};
    shader.Run();
    auto& info{shader.program.info};
    REQUIRE(info.texture_descriptors.size() == 2);
    REQUIRE(info.texture_descriptors[0].size_shift != info.texture_descriptors[1].size_shift);
    // The rescaling pass's test of the second descriptor.
    auto& scaled{*shader.Emit(Opcode::IsTextureScaled, {Value{1U}}).Inst()};

    Shader::HostTranslateInfo host;
    host.ApplyDescriptorLimitPolicy();
    host.max_per_stage_descriptor_samplers = 1;
    Shader::Optimization::LimitSampledTextureArrays(shader.program, host);
    REQUIRE(info.texture_descriptors.size() == 1);
    CHECK(info.texture_descriptors[0].count == 1);
    CHECK(info.texture_descriptors[0].is_descriptor_array);
    const auto index = [](const IR::Inst& inst) -> u32 {
        return inst.Flags<IR::TextureInstInfo>().descriptor_index;
    };
    CHECK(index(indexed) == 0);
    CHECK(index(single) == 0);
    CHECK(indexed.Arg(0).IsEmpty());
    CHECK(scaled.Arg(0).U32() == 0);
}

TEST_CASE("Failed Phi handle subtrees discard their cached values",
          "[video_core][texture_handles]") {
    for (const bool nested : {false, true}) {
        for (const bool retained_slot : {false, true}) {
            CAPTURE(nested, retained_slot);
            HandleProgram shader;
            const Value shared{shader.Storage(0)};
            const Value retained{shader.Storage(4)};
            const Value unsupported{shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}})};
            const Value subtree{nested ? shader.Add(shared, retained) : shared};
            const Value failed{shader.Add(subtree, unsupported)};
            Value fallback{shader.Add(subtree, Value{7U})};
            if (retained_slot) {
                fallback = shader.Add(fallback, retained);
            }
            Value handle{shader.Phi(failed, fallback)};
            if (retained_slot) {
                handle = shader.Add(retained, handle);
            }
            auto& sample{shader.Sample(handle)};
            shader.Run();

            if (Shader::StrictHandles) {
                CHECK(shader.program.info.handle_programs.empty());
                CHECK(shader.Descriptor(sample).bind_null);
                continue;
            }
            REQUIRE(shader.program.info.handle_programs.size() == 1);
            CHECK(shader.Evaluate(sample) ==
                  0x1234U + 7U + (nested ? 0x50U : 0U) + (retained_slot ? 2U * 0x50U : 0U));
            CHECK(shader.Evaluate(sample, 0x5678, 0x90) ==
                  0x5678U + 7U + (nested ? 0x90U : 0U) + (retained_slot ? 2U * 0x90U : 0U));
        }
    }
}

namespace {
struct HandleTestMemory {
    std::map<GPUVAddr, u32> words;
    mutable u32 reads{};
    template <typename T>
    T Read(GPUVAddr address) const {
        ++reads;
        const auto word = [&](GPUVAddr at) -> u64 {
            const auto it = words.find(at);
            return it != words.end() ? it->second : 0U;
        };
        if constexpr (sizeof(T) == 8) {
            return static_cast<T>(word(address) | (word(address + 4) << 32));
        } else {
            return static_cast<T>(word(address));
        }
    }
};
} // namespace

TEST_CASE("Handle tables are read for every element at once", "[video_core][texture_handles]") {
    constexpr GPUVAddr cbuf = 0x1000;
    HandleTestMemory memory;
    memory.words[cbuf + 0x10] = 0x2000; // the storage buffer address
    memory.words[cbuf + 0x14] = 0;
    memory.words[cbuf + 0x20] = 0x60; // the table entry size
    for (u32 element = 0; element < 4; ++element) {
        memory.words[0x2000 + 0x10 + element * 0x60] = 0x100 + element;
    }
    Shader::Info info;
    info.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    using Op = Shader::HandleOp;
    // The word at Index * cbuf0[0x20] + 0x10 of the storage buffer.
    info.handle_programs.push_back({static_cast<u32>(Op::Index), static_cast<u32>(Op::Cbuf), 0U,
                                    0x20U, static_cast<u32>(Op::Mul),
                                    static_cast<u32>(Op::Immediate), 0x10U,
                                    static_cast<u32>(Op::Add), static_cast<u32>(Op::Storage), 0U});
    u32 cbuf_reads{};
    const auto cbuf_address = [&cbuf_reads](u32 index) -> GPUVAddr {
        ++cbuf_reads;
        return index == 0 ? cbuf : 0;
    };
    std::vector<u32> handles;
    VideoCommon::EvaluateTextureHandles(info, 0, memory, cbuf_address, 0, 4,
                                        [&handles](u32 element, std::optional<u32> handle) {
                                            CHECK(element == handles.size());
                                            REQUIRE(handle.has_value());
                                            handles.push_back(*handle);
                                        });
    CHECK(handles == std::vector<u32>{0x100, 0x101, 0x102, 0x103});
    // The entry size and the storage buffer address, once each for all the elements.
    CHECK(cbuf_reads == 2);
    for (u32 element = 0; element < 4; ++element) {
        CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 0, element) ==
              0x100 + element);
    }
}

TEST_CASE("Handle tables read storage buffers past binding 31 from their own base",
          "[video_core][texture_handles]") {
    // Native programs bind up to Info::MAX_SSBOS storage buffers (Ocarina of Time binds 37).
    // Each buffer's base is read once for all the elements, tracked by a mask of the bindings
    // read so far. With 32 bits, binding 33 took binding 1's bit and read no base.
    constexpr GPUVAddr cbuf = 0x1000;
    HandleTestMemory memory;
    Shader::Info info;
    for (u32 binding = 0; binding < 34; ++binding) {
        info.storage_buffers_descriptors.push_back(
            {.cbuf_index = 0, .cbuf_offset = 0x10 * binding, .count = 1, .is_written = false});
        memory.words[cbuf + 0x10 * binding] = 0x10000 + 0x100 * binding; // the buffer address
        memory.words[cbuf + 0x10 * binding + 4] = 0;
        memory.words[0x10000 + 0x100 * binding] = binding; // the word each buffer holds
    }
    using Op = Shader::HandleOp;
    // Binding 1's first word plus binding 33's.
    info.handle_programs.push_back({static_cast<u32>(Op::Immediate), 0U,
                                    static_cast<u32>(Op::Storage), 1U,
                                    static_cast<u32>(Op::Immediate), 0U,
                                    static_cast<u32>(Op::Storage), 33U, static_cast<u32>(Op::Add)});
    const auto cbuf_address = [](u32 index) -> GPUVAddr { return index == 0 ? cbuf : 0; };
    std::vector<u32> handles;
    VideoCommon::EvaluateTextureHandles(info, 0, memory, cbuf_address, 0, 2,
                                        [&handles](u32, std::optional<u32> handle) {
                                            REQUIRE(handle.has_value());
                                            handles.push_back(*handle);
                                        });
    CHECK(handles == std::vector<u32>{34, 34});
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 0) == 34U);
}

namespace {
// A light loop's shadow map handle: each invocation reads a light index from its own list, at an
// address only it knows, then the handle of that light from a table of 96-byte entries.
struct LightTable {
    Value light;
    Value handle;

    explicit LightTable(HandleProgram& shader) {
        const Value list_offset{shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}})};
        light = shader.Emit(Opcode::LoadStorage32, {Value{0U}, list_offset});
        const Value entry{shader.Emit(Opcode::IMul32, {light, Value{96U}})};
        const Value handle_offset{shader.Add(entry, Value{16U})};
        handle = shader.Emit(Opcode::LoadStorage32, {Value{1U}, handle_offset});
    }
};

u32 EvaluateTableElement(const HandleProgram& shader, const Shader::TextureDescriptor& descriptor,
                         u32 element) {
    REQUIRE(descriptor.handle_program < shader.program.info.handle_programs.size());
    const auto result{Shader::EvaluateHandleProgram(
        shader.program.info.handle_programs[descriptor.handle_program],
        [](u32, u32) -> u32 { throw std::logic_error("Unexpected evaluator cbuf read"); },
        [](u32 binding, u32 offset) -> u32 {
            REQUIRE(binding == 1);
            return 0x1000 + offset;
        },
        element)};
    REQUIRE(result.has_value());
    return *result;
}
} // namespace

TEST_CASE("A handle read from a table at a per-invocation index is a descriptor array",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    const LightTable table{shader};
    auto& first{shader.Sample(table.handle)};
    auto& second{shader.Sample(table.handle)};
    shader.Run();

    REQUIRE(shader.program.info.texture_descriptors.size() == 1);
    const auto& descriptor{shader.Descriptor(first)};
    CHECK(descriptor.count == 64);
    CHECK(descriptor.size_shift == 0);
    // Element k holds the handle of table entry k.
    for (const u32 element : {0U, 1U, 63U}) {
        CHECK(EvaluateTableElement(shader, descriptor, element) == 0x1000 + element * 96 + 16);
    }
    // Each instruction indexes the array with the light index it read, clamped to the array.
    for (const IR::Inst* const sample : {&first, &second}) {
        REQUIRE_FALSE(sample->Arg(0).IsEmpty());
        const IR::Inst* const selection{sample->Arg(0).InstRecursive()};
        REQUIRE(selection->GetOpcode() == Opcode::UMin32);
        CHECK(selection->Arg(0).Resolve() == table.light.Resolve());
        CHECK(selection->Arg(1).U32() == 63);
    }
}

TEST_CASE("A table index not defined where the texture is read takes the first element",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.insertion_block = &shader.left;
    const LightTable table{shader};
    shader.insertion_block = &shader.right;
    const Value other{shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R1}})};
    auto& sample{shader.Sample(shader.Phi(table.handle, other))};
    shader.Run();

    const auto& descriptor{shader.Descriptor(sample)};
    CHECK(descriptor.count == 1);
    if (Shader::StrictHandles) {
        CHECK(descriptor.bind_null); // This Phi also contains a genuinely different handle.
    } else {
        CHECK(EvaluateTableElement(shader, descriptor, 0) == 0x1000 + 16);
    }
    CHECK(sample.Arg(0).IsEmpty());
}

TEST_CASE("An unavailable array index retains element zero without rejecting the texture",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.block.AddBranch(&shader.left);
    shader.block.AddBranch(&shader.right);
    shader.program.blocks = {&shader.block, &shader.left, &shader.right};
    shader.program.post_order_blocks = {&shader.right, &shader.left, &shader.block};
    shader.insertion_block = &shader.left;
    const LightTable table{shader};
    shader.insertion_block = &shader.right;
    auto& sample = shader.Sample(table.handle);
    shader.Run();
    const auto& descriptor = shader.Descriptor(sample);
    CHECK_FALSE(descriptor.bind_null);
    CHECK(descriptor.count == 1);
    CHECK(EvaluateTableElement(shader, descriptor, 0) == 0x1010U);
    CHECK(sample.Arg(0).IsEmpty());
}

TEST_CASE("Hosts without handle table arrays bind null for a per-invocation handle",
          "[video_core][texture_handles]") {
    for (const bool supported : {false, true}) {
        CAPTURE(supported);
        HandleProgram shader;
        shader.handle_table_arrays = supported;
        shader.env.cbuf_values = {{{0, 0x520}, 0xabc}};
        // The table's address is a constant buffer word, not the sampled texture handle.
        const Value list_offset{shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}})};
        const Value light{shader.Emit(Opcode::LoadStorage32, {Value{0U}, list_offset})};
        const Value entry{shader.Emit(Opcode::IMul32, {light, Value{96U}})};
        const Value table{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0x520U}})};
        const Value handle{
            shader.Emit(Opcode::LoadStorage32, {Value{1U}, shader.Add(entry, table)})};
        auto& sample{shader.Sample(handle)};
        shader.Run();

        const auto& descriptor{shader.Descriptor(sample)};
        CHECK(descriptor.count == (supported ? 64U : 1U));
        CHECK((descriptor.handle_program != ~0U) == supported);
        CHECK(descriptor.bind_null == !supported);
        CHECK(VideoCommon::HandleMustBeNull(shader.program.info, descriptor, 0) == !supported);
        if (supported) {
            const auto result{Shader::EvaluateHandleProgram(
                shader.program.info.handle_programs[descriptor.handle_program],
                [](u32 cbuf_index, u32 offset) {
                    REQUIRE(cbuf_index == 0);
                    REQUIRE(offset == 0x520);
                    return 0xabcU;
                },
                [](u32 binding, u32 offset) {
                    REQUIRE(binding == 1);
                    return offset;
                },
                0)};
            CHECK(result == 0xabcU);
        }
    }
}

TEST_CASE("Handle tracking skips the floating-point math a reused register leads to",
          "[video_core][texture_handles]") {
    // The handle register holds a storage word on one path and a float result on the other: no
    // constant buffer word behind the float math is the handle.
    HandleProgram shader;
    shader.insertion_block = &shader.left;
    const Value handle{shader.Storage(0)};
    shader.insertion_block = &shader.right;
    const Value word{shader.Emit(Opcode::GetCbufU32, {Value{3U}, Value{0x10U}})};
    const Value sum{
        shader.Emit(Opcode::FPAdd32, {shader.Emit(Opcode::BitCastF32U32, {word}), Value{1.0f}})};
    const Value reused{shader.Emit(Opcode::BitCastU32F32, {sum})};
    auto& sample{shader.Sample(shader.Phi(handle, reused))};
    shader.Run();

    if (Shader::StrictHandles) {
        CHECK(shader.Descriptor(sample).bind_null);
    } else {
        CHECK(shader.Descriptor(sample).handle_program != ~0U);
        CHECK(shader.Evaluate(sample) == 0x1234);
    }
}

TEST_CASE("Dual vertex programs join the descriptors and storage buffers of both halves",
          "[video_core][texture_handles]") {
    // Vertex A samples a 2D and a 3D texture with the handle at offset 4 of its storage buffer 0,
    // and stores that word. Vertex B samples a texture with the handle at offset 0 of its buffer
    // 0, and Vertex A's 2D texture and an array texture with the handle Vertex A reads, at offset
    // 4 of its buffer 1, which is Vertex A's buffer 0; it stores that word in its buffer 0. Vertex
    // B's buffers swap bindings when they join, and its textures go past Vertex A's or onto them.
    using Shader::TextureType;
    HandleProgram vertex_a;
    HandleProgram vertex_b;
    vertex_a.program.stage = Shader::Stage::VertexA;
    vertex_b.program.stage = Shader::Stage::VertexB;
    // Outputs a sample's result, which keeps the sample through dead code elimination.
    size_t outputs{};
    const auto keep = [&outputs](HandleProgram& shader, IR::Inst& sample) {
        IR::IREmitter ir{*shader.insertion_block};
        ir.SetAttribute(IR::Attribute::Generic0X + outputs++ * 4,
                        IR::F32{ir.CompositeExtract(Value{&sample}, 0)}, ir.Imm32(0));
    };
    const Value word_a{vertex_a.Storage(4)};
    auto& sample_a{vertex_a.Sample(word_a)};
    auto& volume_a{vertex_a.Sample(word_a, TextureType::Color3D)};
    keep(vertex_a, sample_a);
    keep(vertex_a, volume_a);
    auto& store_a{*vertex_a.Emit(Opcode::WriteStorage32, {Value{0U}, Value{8U}, word_a}).Inst()};
    auto& sample_b{vertex_b.Sample(vertex_b.Storage(0))};
    const Value word_b{vertex_b.Emit(Opcode::LoadStorage32, {Value{1U}, Value{4U}})};
    auto& shared_b{vertex_b.Sample(word_b)};
    auto& array_b{vertex_b.Sample(word_b, TextureType::ColorArray2D)};
    keep(vertex_b, sample_b);
    keep(vertex_b, shared_b);
    keep(vertex_b, array_b);
    auto& store_b{*vertex_b.Emit(Opcode::WriteStorage32, {Value{0U}, Value{8U}, word_b}).Inst()};
    vertex_a.Run();
    vertex_b.Run();
    vertex_a.program.info.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    vertex_b.program.info.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x20, .count = 1, .is_written = false});
    vertex_b.program.info.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    REQUIRE(vertex_a.program.info.handle_programs.size() == 1);
    REQUIRE(vertex_b.program.info.handle_programs.size() == 2);
    REQUIRE(vertex_a.program.info.texture_descriptors.size() == 2);
    REQUIRE(vertex_b.program.info.texture_descriptors.size() == 3);
    const auto program_a{vertex_a.program.info.handle_programs.front()};
    for (HandleProgram* const half : {&vertex_a, &vertex_b}) {
        half->program.syntax_list = {
            {{.block = &half->block}, IR::AbstractSyntaxNode::Type::Block},
            {{}, IR::AbstractSyntaxNode::Type::Return},
        };
    }
    const Shader::HostTranslateInfo host{};
    const IR::Program merged{Shader::Maxwell::MergeDualVertexPrograms(
        vertex_a.program, vertex_b.program, vertex_b.env, host)};

    const auto& info{merged.info};
    REQUIRE(info.storage_buffers_descriptors.size() == 2);
    CHECK(info.storage_buffers_descriptors[0].cbuf_offset == 0x10);
    CHECK(info.storage_buffers_descriptors[1].cbuf_offset == 0x20);
    // Vertex A's program keeps its index, and Vertex B's reading the same word is that one.
    REQUIRE(info.handle_programs.size() == 2);
    CHECK(info.handle_programs.front() == program_a);
    REQUIRE(info.texture_descriptors.size() == 4);
    // The storage buffer whose address constant buffer 0 holds at 0x10 is at 0x2000, the one at
    // 0x20 at 0x3000; a program reading the wrong one gets 0xd0 or 0xd4.
    constexpr GPUVAddr cbuf = 0x1000;
    HandleTestMemory memory;
    memory.words[cbuf + 0x10] = 0x2000;
    memory.words[cbuf + 0x20] = 0x3000;
    memory.words[0x2004] = 0xa0;
    memory.words[0x3000] = 0xb0;
    memory.words[0x2000] = 0xd0;
    memory.words[0x3004] = 0xd4;
    const auto cbuf_address = [](u32 index) -> GPUVAddr { return index == 0 ? cbuf : 0; };
    // Each instruction reads the texture and the storage buffer it read before the merge.
    const auto descriptor_index = [](const IR::Inst& sample) -> u32 {
        return sample.Flags<IR::TextureInstInfo>().descriptor_index;
    };
    const auto texture = [&](const IR::Inst& sample) {
        REQUIRE(descriptor_index(sample) < info.texture_descriptors.size());
        const auto& descriptor{info.texture_descriptors[descriptor_index(sample)]};
        REQUIRE(descriptor.handle_program < info.handle_programs.size());
        const auto handle = VideoCommon::EvaluateTextureHandle(info, descriptor.handle_program,
                                                               memory, cbuf_address, 0);
        REQUIRE(handle.has_value());
        return std::pair{descriptor.type, *handle};
    };
    CHECK(texture(sample_a) == std::pair{TextureType::Color2D, 0xa0U});
    CHECK(texture(volume_a) == std::pair{TextureType::Color3D, 0xa0U});
    CHECK(texture(sample_b) == std::pair{TextureType::Color2D, 0xb0U});
    CHECK(texture(shared_b) == std::pair{TextureType::Color2D, 0xa0U});
    CHECK(texture(array_b) == std::pair{TextureType::ColorArray2D, 0xa0U});
    CHECK(descriptor_index(shared_b) == descriptor_index(sample_a));
    const auto buffer = [&](const IR::Inst& inst) {
        REQUIRE(inst.Arg(0).IsImmediate());
        REQUIRE(inst.Arg(0).U32() < info.storage_buffers_descriptors.size());
        return info.storage_buffers_descriptors[inst.Arg(0).U32()].cbuf_offset;
    };
    CHECK(buffer(*word_a.Inst()) == 0x10);
    CHECK(buffer(store_a) == 0x10);
    CHECK(buffer(*word_b.Inst()) == 0x10);
    CHECK(buffer(store_b) == 0x20);
}

TEST_CASE("Joined descriptor indices follow the list each instruction reads",
          "[video_core][texture_handles]") {
    // Descriptor i of each kind moves to element i of its kind's list, storage buffer binding i
    // to element i of the bindings.
    HandleProgram shader;
    const Value none{};
    const auto texture = [&](Opcode opcode, std::initializer_list<Value> args,
                             Shader::TextureType type, u32 index) -> IR::Inst& {
        IR::Inst& inst{*shader.Emit(opcode, args).Inst()};
        IR::TextureInstInfo flags{};
        flags.type.Assign(type);
        flags.descriptor_index.Assign(index);
        inst.SetFlags(flags);
        return inst;
    };
    using Shader::TextureType;
    auto& texel_fetch{
        texture(Opcode::ImageFetch, {none, none, none, none, none}, TextureType::Buffer, 0)};
    auto& sample{
        texture(Opcode::ImageSampleImplicitLod, {none, none, none, none}, TextureType::Color2D, 1)};
    auto& texel_read{texture(Opcode::ImageRead, {none, none}, TextureType::Buffer, 0)};
    auto& image_write{texture(Opcode::ImageWrite, {none, none, none}, TextureType::Color2D, 1)};
    auto& image_atomic{
        texture(Opcode::ImageAtomicIAdd32, {none, none, none}, TextureType::Color2D, 0)};
    auto& texture_scaled{*shader.Emit(Opcode::IsTextureScaled, {Value{1U}}).Inst()};
    auto& image_scaled{*shader.Emit(Opcode::IsImageScaled, {Value{1U}}).Inst()};
    auto& load{*shader.Emit(Opcode::LoadStorage32, {Value{1U}, none}).Inst()};
    auto& store{*shader.Emit(Opcode::WriteStorage64, {Value{0U}, none, none}).Inst()};
    auto& atomic{*shader.Emit(Opcode::StorageAtomicIAdd32, {Value{2U}, none, none}).Inst()};
    auto& cbuf{*shader.Emit(Opcode::GetCbufU32, {Value{1U}, Value{0U}}).Inst()};

    const Shader::Optimization::JoinedDescriptors joined{
        .texture_buffers = {5},
        .image_buffers = {6},
        .textures = {7, 8},
        .images = {9, 10},
    };
    Shader::Optimization::RemapTextureDescriptors(shader.program, joined);
    const std::array<u32, 3> bindings{3, 4, 5};
    Shader::Optimization::RemapStorageBindings(shader.program, bindings);

    const auto descriptor_index = [](const IR::Inst& inst) -> u32 {
        return inst.Flags<IR::TextureInstInfo>().descriptor_index;
    };
    CHECK(descriptor_index(texel_fetch) == 5);
    CHECK(descriptor_index(sample) == 8);
    CHECK(descriptor_index(texel_read) == 6);
    CHECK(descriptor_index(image_write) == 10);
    CHECK(descriptor_index(image_atomic) == 9);
    CHECK(texture_scaled.Arg(0).U32() == 8);
    CHECK(image_scaled.Arg(0).U32() == 10);
    CHECK(load.Arg(0).U32() == 4);
    CHECK(store.Arg(0).U32() == 3);
    CHECK(atomic.Arg(0).U32() == 5);
    // A constant buffer index is no storage buffer binding.
    CHECK(cbuf.Arg(0).U32() == 1);
}

TEST_CASE("Texture arrays that become distinct descriptors share the cap",
          "[video_core][texture_handles]") {
    // One table of textures sampled with two samplers: two descriptors, so two arrays, which the
    // third sample (with the first sampler) doesn't add to.
    HandleProgram shader;
    shader.env.cbuf_values = {{{6, 0}, 0x123}, {{2, 0x570}, 0x24000000}, {{2, 0x580}, 0x24000000}};
    shader.env.expected_handle = 0x24000123;
    const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
    const Value texture{shader.Emit(Opcode::GetCbufU32, {Value{6U}, offset})};
    std::vector<IR::Inst*> samples;
    for (const u32 sampler_offset : {0x570U, 0x580U, 0x570U}) {
        const Value sampler{shader.Emit(Opcode::GetCbufU32, {Value{2U}, Value{sampler_offset}})};
        samples.push_back(&shader.Sample(shader.Emit(Opcode::BitwiseOr32, {texture, sampler})));
    }
    Shader::HostTranslateInfo host;
    host.ApplyDescriptorLimitPolicy();
    host.max_per_stage_descriptor_sampled_images = 16;
    host.max_descriptor_set_sampled_images = 16;
    Shader::Optimization::TexturePass(shader.env, shader.program, host);

    const auto& descriptors{shader.program.info.texture_descriptors};
    REQUIRE(descriptors.size() == 2);
    CHECK(descriptors[0].secondary_cbuf_offset != descriptors[1].secondary_cbuf_offset);
    for (const auto& descriptor : descriptors) {
        CHECK(descriptor.count == 8);
    }
    CHECK(samples[0]->Flags<IR::TextureInstInfo>().descriptor_index ==
          samples[2]->Flags<IR::TextureInstInfo>().descriptor_index);
}

TEST_CASE("A size query of a table shares the texture array of its sample",
          "[video_core][texture_handles]") {
    // The size query carries no type until the pass gives it the sample's: one descriptor, so one
    // array, which takes the whole budget. A 3D sample of the table adds a second array.
    for (const bool volume : {false, true}) {
        CAPTURE(volume);
        HandleProgram shader;
        const LightTable table{shader};
        auto& sample{shader.Sample(table.handle)};
        IR::IREmitter ir{*shader.insertion_block};
        auto& query{*ir.ImageQueryDimension(table.handle, ir.Imm32(0), ir.Imm1(true)).Inst()};
        if (volume) {
            shader.Sample(table.handle, Shader::TextureType::Color3D);
        }
        Shader::HostTranslateInfo host;
        host.support_handle_table_arrays = true;
        host.ApplyDescriptorLimitPolicy();
        host.max_per_stage_descriptor_sampled_images = 16;
        host.max_descriptor_set_sampled_images = 16;
        Shader::Optimization::TexturePass(shader.env, shader.program, host);

        const auto& descriptors{shader.program.info.texture_descriptors};
        REQUIRE(descriptors.size() == (volume ? 2U : 1U));
        for (const auto& descriptor : descriptors) {
            CHECK(descriptor.count == (volume ? 8U : 16U));
        }
        if (!volume) {
            CHECK(query.Flags<IR::TextureInstInfo>().descriptor_index ==
                  sample.Flags<IR::TextureInstInfo>().descriptor_index);
        }
    }
}

TEST_CASE("Storage handle words are read from the host-aligned buffer base",
          "[video_core][texture_handles]") {
    // The program's storage offsets are relative to the buffer address aligned down to the host
    // storage alignment (StorageOffset), where the buffer cache binds the buffer.
    constexpr GPUVAddr cbuf = 0x1000;
    HandleTestMemory memory;
    memory.words[cbuf + 0x10] = 0x2008; // the storage buffer address, 8 bytes past 16
    memory.words[cbuf + 0x14] = 0;
    memory.words[0x2008] = 0x11; // the handle the shader reads
    memory.words[0x2010] = 0x22;

    Shader::Info info;
    info.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    // Offset 8 from the base aligned to 16, as StorageOffset gives it for a read at 0x2008.
    info.handle_programs.push_back({static_cast<u32>(Shader::HandleOp::Immediate), 8U,
                                    static_cast<u32>(Shader::HandleOp::Storage), 0U});
    const auto cbuf_address = [](u32 index) -> GPUVAddr { return index == 0 ? cbuf : 0; };
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 16) == 0x11U);
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 4) == 0x22U);
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 0) == 0x22U);

    SECTION("Indirect buffers align the address read from the table, not the table") {
        // The constant buffer points at a table that is not aligned. The buffer address is read
        // from the table entry at its exact address, then aligned down.
        memory.words[cbuf + 0x10] = 0x3004;
        memory.words[0x3004 + 0x8] = 0x2008;
        memory.words[0x3004 + 0xc] = 0;
        info.storage_buffers_descriptors[0].indirect_offset = 0x8;
        CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 16) == 0x11U);
        CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 4) == 0x22U);
    }
}

TEST_CASE("Handle table support distinguishes cached environment reads",
          "[video_core][texture_handles]") {
    Shader::HostTranslateInfo host;
    const u32 without_tables{host.EnvironmentReadProperties()};
    CHECK(((without_tables >> 10) & 1U) == static_cast<u32>(Shader::StrictHandles));
    host.support_handle_table_arrays = true;
    CHECK(host.EnvironmentReadProperties() == (without_tables | (1U << 9)));
}

TEST_CASE("Task and mesh texture arrays share the compute stage budget",
          "[video_core][texture_handles]") {
    // Each guest stage fits by itself; their arrays, texel buffers and auxiliary output buffers
    // together must fit one host compute stage. A fragment stage's textures are outside it.
    for (const bool sampler_limited : {false, true}) {
        CAPTURE(sampler_limited);
        std::array<HandleProgram, 2> stages;
        for (HandleProgram& shader : stages) {
            const LightTable table{shader};
            shader.Sample(table.handle);
            shader.Run();
            shader.program.info.constant_buffer_descriptors.push_back({.index = 0, .count = 1});
            shader.program.info.storage_buffers_descriptors.push_back({.count = 1});
        }
        stages[0].program.info.texture_buffer_descriptors.push_back({.count = 1});
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        host.max_per_stage_resources = 21;
        host.max_per_stage_descriptor_sampled_images = 32;
        host.max_per_stage_descriptor_samplers = sampler_limited ? 8 : 32;
        const std::array programs{&stages[0].program, &stages[1].program};
        // Four auxiliary buffers, four cbuf/ssbo resources and a texel buffer leave 12 textures.
        Shader::Optimization::LimitSharedStageSampledTextureArrays(programs, host, 4);
        for (const auto& shader : stages) {
            CHECK(shader.program.info.texture_descriptors.front().count ==
                  (sampler_limited ? 4U : 6U));
        }
        host.max_per_stage_resources = 10; // only one texture slot for two arrays
        REQUIRE_THROWS(
            Shader::Optimization::LimitSharedStageSampledTextureArrays(programs, host, 4));
    }
}

TEST_CASE("Joined storage descriptors retain the bounds of both halves' writes",
          "[video_core][texture_handles]") {
    Shader::Info base;
    Shader::Info source;
    base.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    source.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = true});
    source.storage_buffers_descriptors.front().write_extent = {.bounded = true, .const_end = 32};
    Shader::Optimization::JoinStorageInfo(base, source);
    REQUIRE(base.storage_buffers_descriptors.size() == 1);
    CHECK(base.storage_buffers_descriptors.front().is_written);
    CHECK(base.storage_buffers_descriptors.front().write_extent.bounded);
    CHECK(base.storage_buffers_descriptors.front().write_extent.const_end == 32);
    source.storage_buffers_descriptors.front().write_extent.const_end = 64;
    Shader::Optimization::JoinStorageInfo(base, source);
    CHECK(base.storage_buffers_descriptors.front().write_extent.const_end == 64);
    source.storage_buffers_descriptors.front().write_extent = {};
    Shader::Optimization::JoinStorageInfo(base, source);
    CHECK_FALSE(base.storage_buffers_descriptors.front().write_extent.bounded);
}

TEST_CASE("An unresolved sampled texture handle binds null", "[video_core][texture_handles]") {
    HandleProgram shader;
    const Value cbuf_word{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0x540U}})};
    const Value float_word{shader.Emit(Opcode::BitCastF32U32, {cbuf_word})};
    const Value is_nan{shader.Emit(Opcode::FPIsNan32, {float_word})};
    const Value handle{shader.Emit(Opcode::LoadStorage32, {Value{0U}, is_nan})};
    auto& sample{shader.Sample(handle)};
    shader.Run();

    const auto& descriptor{shader.Descriptor(sample)};
    CHECK(descriptor.bind_null);
    CHECK(VideoCommon::HandleMustBeNull(shader.program.info, descriptor, 0));
}

TEST_CASE("A texture handle word that holds a storage buffer address is not a handle",
          "[video_core][texture_handles]") {
    Shader::Info info;
    info.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0x520, .count = 1, .is_written = false});
    Shader::TextureDescriptor desc{};
    desc.cbuf_index = 0;
    desc.cbuf_offset = 0x520;
    desc.count = 1;
    desc.size_shift = 3;
    CHECK(VideoCommon::HandleMustBeNull(info, desc, 0));
    desc.cbuf_offset = 0x524; // the address's upper word
    CHECK(VideoCommon::HandleMustBeNull(info, desc, 0));
    desc.cbuf_offset = 0x528;
    CHECK_FALSE(VideoCommon::HandleMustBeNull(info, desc, 0));
    desc.cbuf_offset = 0x518;
    CHECK(VideoCommon::HandleMustBeNull(info, desc, 1)); // the second element
    desc.cbuf_offset = 0x520;
    desc.cbuf_index = 2;
    CHECK_FALSE(VideoCommon::HandleMustBeNull(info, desc, 0));
    desc.cbuf_index = 0;
    desc.handle_program = 0; // computed for each draw: a real handle
    CHECK_FALSE(VideoCommon::HandleMustBeNull(info, desc, 0));
}

TEST_CASE("Nested handle shifts preserve default binding and strict evaluation",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    const Value word{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
    const Value first{shader.Emit(Opcode::ShiftLeftLogical32, {word, Value{2U}})};
    const Value second{shader.Emit(Opcode::ShiftLeftLogical32, {first, Value{3U}})};
    auto& sample{shader.Sample(second)};
    shader.env.expected_handle = 0x5ffcU;
    shader.Run();
    const auto& descriptor{shader.Descriptor(sample)};
    if (!Shader::StrictHandles) {
        CHECK(descriptor.handle_program == ~0U);
        CHECK(descriptor.shift_left == 3);
        return;
    }
    REQUIRE(descriptor.handle_program != ~0U);
    CHECK(Shader::EvaluateHandleProgram(
              shader.program.info.handle_programs[descriptor.handle_program],
              [](u32, u32) { return 0xabcU; }, [](u32, u32) { return 0U; }) == 0x15780U);
}

TEST_CASE("Malformed handle programs cannot synthesize a texture handle",
          "[video_core][texture_handles]") {
    using Op = Shader::HandleOp;
    const auto word = [](Op op) { return static_cast<u32>(op); };
    std::vector<std::vector<u32>> malformed{
        {},
        {word(Op::Immediate)},
        {word(Op::Cbuf), 0U},
        {word(Op::CbufDynamic), 0U},
        {word(Op::Storage), 0U},
        {word(Op::Immediate), 7U, word(Op::Add)},
        {word(Op::Neg)},
        {word(Op::Select)},
        {word(Op::Tee), 0U},
        {word(Op::Load), 0U},
        {word(Op::Load), 256U},
        {word(Op::Immediate), 7U, word(Op::Tee), 256U},
        {word(Op::Immediate), 7U, word(Op::Tee)},
        {word(Op::Immediate), 7U, ~0U},
    };
    std::vector<u32> overflow;
    for (u32 index = 0; index < 65; ++index) {
        overflow.insert(overflow.end(), {word(Op::Immediate), index});
    }
    for (u32 index = 0; index < 64; ++index) {
        overflow.push_back(word(Op::Add));
    }
    malformed.push_back(std::move(overflow));
    for (const auto& code : malformed) {
        CHECK_FALSE(Shader::EvaluateHandleProgram(
            code, [](u32, u32) -> u32 { throw std::logic_error("Invalid cbuf read"); },
            [](u32, u32) -> u32 { throw std::logic_error("Invalid storage read"); }));
    }
}

TEST_CASE("Cbuf handle masks evaluate the entire expression", "[video_core][texture_handles]") {
    HandleProgram shader;
    const Value word{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
    auto& sample{shader.Sample(shader.Emit(Opcode::BitwiseAnd32, {word, Value{0xffU}}))};
    shader.Run();
    const auto& descriptor{shader.Descriptor(sample)};
    if (!Shader::StrictHandles) {
        CHECK(descriptor.handle_program == ~0U);
        CHECK(descriptor.cbuf_offset == 0);
        return;
    }
    REQUIRE(descriptor.handle_program != ~0U);
    const auto result{Shader::EvaluateHandleProgram(
        shader.program.info.handle_programs[descriptor.handle_program],
        [](u32, u32) { return 0xabcU; }, [](u32, u32) { return 0U; })};
    CHECK(result == 0xbcU);
}

TEST_CASE("Failed handle reads bind null while a known zero remains texture zero",
          "[video_core][texture_handles]") {
    using Op = Shader::HandleOp;
    const auto word = [](Op op) { return static_cast<u32>(op); };
    constexpr GPUVAddr cbuf = 0x1000;
    HandleTestMemory memory;
    Shader::Info info;
    info.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1});
    memory.words[cbuf + 0x10] = 0x2000;
    // Zero is a real, successfully read handle. Missing bindings and malformed programs fail.
    info.handle_programs = {{word(Op::Immediate), 0U},
                            {word(Op::Cbuf), 0U, 0U},
                            {word(Op::Cbuf), 1U, 0U},
                            {word(Op::Immediate), 0U, word(Op::Storage), 0U},
                            {word(Op::Immediate), 0U, word(Op::Storage), 1U},
                            {word(Op::Immediate), 0U, word(Op::Storage), 64U},
                            {word(Op::Immediate)}};
    const auto cbuf_address = [](u32 index) -> GPUVAddr { return index == 0 ? cbuf : 0; };
    for (u32 program = 0; program <= info.handle_programs.size(); ++program) {
        CAPTURE(program);
        const bool known =
            program == 0 || program == 1 || program == 3 || (!Shader::StrictHandles && program < 7);
        const auto single =
            VideoCommon::EvaluateTextureHandle(info, program, memory, cbuf_address, 0);
        CHECK(single.has_value() == known);
        if (known)
            CHECK(*single == 0U);
        for (const bool linked : {false, true}) {
            CHECK(VideoCommon::TexturePairOrNull(single, linked) ==
                  (known ? std::pair{0U, 0U} : VideoCommon::NULL_TEXTURE_HANDLE));
        }
        u32 outputs{};
        VideoCommon::EvaluateTextureHandles(info, program, memory, cbuf_address, 0, 3,
                                            [&](u32 element, std::optional<u32> handle) {
                                                CHECK(element == outputs++);
                                                CHECK(handle == single);
                                            });
        CHECK(outputs == 3);
    }
    memory.words[cbuf + 0x10] = 0; // A missing storage pointer is also a failed read.
    CHECK(VideoCommon::EvaluateTextureHandle(info, 3, memory, cbuf_address, 0).has_value() ==
          !Shader::StrictHandles);
    VideoCommon::EvaluateTextureHandles(info, 3, memory, cbuf_address, 0, 3,
                                        [](u32, std::optional<u32> handle) {
                                            CHECK(handle.has_value() == !Shader::StrictHandles);
                                        });
}

TEST_CASE("Handle read failure follows the selected value and survives Tee and Load",
          "[video_core][texture_handles]") {
    using Op = Shader::HandleOp;
    const auto word = [](Op op) { return static_cast<u32>(op); };
    HandleTestMemory memory;
    Shader::Info info;
    for (const bool storage : {false, true}) {
        for (const Op operation : {Op::Select, Op::LAnd, Op::LOr}) {
            for (const bool selected : {false, true}) {
                const u32 lhs = operation == Op::LOr ? !selected : selected;
                std::vector<u32> code{word(Op::Immediate), lhs};
                if (operation == Op::Select) {
                    code.insert(code.end(), {word(Op::Immediate), 7U});
                }
                if (storage)
                    code.insert(code.end(), {word(Op::Immediate), 0U, word(Op::Storage), 0U});
                else
                    code.insert(code.end(), {word(Op::Cbuf), 0U, 0U});
                code.insert(code.end(), {word(Op::Tee), 0U});
                // Load uses the same failed entry, then discards the original via Select.
                code.insert(code.end(),
                            {word(Op::Immediate), 1U, word(Op::Load), 0U, word(Op::Immediate), 9U,
                             word(Op::Select), word(Op::Add), word(operation)});
                info.handle_programs = {code};
                const auto address = [](u32) -> GPUVAddr { return 0; };
                const bool used = operation == Op::Select ? !selected : selected;
                const auto result = VideoCommon::EvaluateTextureHandle(info, 0, memory, address, 0);
                CHECK(result.has_value() == (!Shader::StrictHandles || !used));
                VideoCommon::EvaluateTextureHandles(
                    info, 0, memory, address, 0, 3,
                    [&](u32, std::optional<u32> handle) { CHECK(handle == result); });
                if (result && operation == Op::Select)
                    CHECK(*result == (selected ? 7U : 0U));
            }
        }
    }
    CHECK(memory.reads == 0);
}

TEST_CASE("Phi reloads compare the cbuf computation rather than instruction identity",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.env.cbuf_values = {{{1, 0x10}, 0xabc}};
    shader.insertion_block = &shader.left;
    const Value first = shader.Emit(Opcode::GetCbufU32, {Value{1U}, Value{0x10U}});
    shader.insertion_block = &shader.right;
    const Value second = shader.Emit(Opcode::GetCbufU32, {Value{1U}, Value{0x10U}});
    auto& sample = shader.Sample(shader.Phi(first, second));
    shader.Run();
    CHECK_FALSE(shader.Descriptor(sample).bind_null);
    if (Shader::StrictHandles) {
        const auto& desc = shader.Descriptor(sample);
        REQUIRE(desc.handle_program < shader.program.info.handle_programs.size());
        CHECK(Shader::EvaluateHandleProgram(
                  shader.program.info.handle_programs[desc.handle_program],
                  [](u32 index, u32 offset) {
                      CHECK(index == 1);
                      CHECK(offset == 0x10);
                      return 0xabcU;
                  },
                  [](u32, u32) { return 0U; }) == 0xabcU);
    }
}

TEST_CASE("Strict mode retains a uniquely resolved storage image and texel buffer",
          "[video_core][texture_handles]") {
    for (const bool buffer : {false, true}) {
        HandleProgram shader;
        const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
        const Value lane = shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}});
        const Value handle = shader.Add(word, lane);
        IR::TextureInstInfo flags{};
        flags.type.Assign(buffer ? Shader::TextureType::Buffer : Shader::TextureType::Color2D);
        auto& image = *shader.Emit(Opcode::BindlessImageWrite, {handle, {}, {}}).Inst();
        image.SetFlags(flags);
        REQUIRE_NOTHROW(shader.Run());
        if (buffer) {
            REQUIRE(shader.program.info.image_buffer_descriptors.size() == 1);
            CHECK(shader.program.info.image_buffer_descriptors.front().cbuf_offset == 0);
        } else {
            REQUIRE(shader.program.info.image_descriptors.size() == 1);
            CHECK(shader.program.info.image_descriptors.front().cbuf_offset == 0);
        }
    }
}

TEST_CASE("Strict computed handles retain the old integer and volume metadata",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.env.integer_texture = true;
    shader.env.texture_type = Shader::TextureType::Color3D;
    const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
    const Value handle = shader.Add(word, Value{1U});
    IR::IREmitter ir{shader.block};
    IR::TextureInstInfo flags{};
    auto& query = *ir.ImageQueryDimension(handle, ir.Imm32(0U), ir.Imm1(false)).Inst();
    shader.Run();
    const auto& desc = shader.program.info.texture_descriptors.front();
    CHECK(desc.type == Shader::TextureType::Color3D);
    CHECK(desc.is_integer);
    CHECK(query.Flags<IR::TextureInstInfo>().is_integer == 1);
    CHECK((desc.handle_program != ~0U) == Shader::StrictHandles);
}

TEST_CASE("Default handle mode preserves the safe cbuf search and missing-read zero",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
    auto& sample = shader.Sample(shader.Add(word, Value{3U}));
    shader.Run();
    const auto& desc = shader.Descriptor(sample);
    CHECK((desc.handle_program != ~0U) == Shader::StrictHandles);
    if (!Shader::StrictHandles) {
        CHECK(desc.cbuf_index == 0);
        CHECK(desc.cbuf_offset == 0);
        CHECK_FALSE(desc.bind_null);
    }
}

TEST_CASE("Strict computed texel handles retain the previous signed normalized format",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.env.texture_type = Shader::TextureType::Buffer;
    shader.env.pixel_format = Shader::TexturePixelFormat::R16_SNORM;
    const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
    const Value handle = shader.Add(word, Value{1U});
    IR::TextureInstInfo flags{};
    flags.type.Assign(Shader::TextureType::Buffer);
    auto& fetch = *shader.Emit(Opcode::BindlessImageFetch, {handle, Value{0U}, {}, {}, {}}).Inst();
    fetch.SetFlags(flags);
    shader.Run();
    REQUIRE(shader.program.info.texture_buffer_descriptors.size() == 1);
    CHECK(shader.env.pixel_format_reads == 1);
    CHECK((shader.program.info.texture_buffer_descriptors.front().handle_program != ~0U) ==
          Shader::StrictHandles);
    CHECK(std::ranges::any_of(shader.block.Instructions(), [](const IR::Inst& inst) {
        return inst.GetOpcode() == Opcode::FPMul32;
    }));
}

TEST_CASE("Strict writable fallback rejects competing handles but retains one repeated source",
          "[video_core][texture_handles]") {
    for (const bool same : {false, true}) {
        HandleProgram shader;
        shader.env.cbuf_values = {{{0, 0}, 0xabc}, {{0, 4}, 0xabc}};
        const Value first = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
        const Value second = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{same ? 0U : 4U}});
        const Value lane = shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}});
        const Value handle = shader.Add(shader.Add(first, second), lane);
        IR::TextureInstInfo flags{};
        flags.type.Assign(Shader::TextureType::Color2D);
        auto& image = *shader.Emit(Opcode::BindlessImageWrite, {handle, {}, {}}).Inst();
        image.SetFlags(flags);
        if (Shader::StrictHandles && !same) {
            REQUIRE_THROWS_AS(shader.Run(), Shader::NotImplementedException);
            CHECK(shader.program.info.image_descriptors.empty());
        } else {
            REQUIRE_NOTHROW(shader.Run());
            REQUIRE(shader.program.info.image_descriptors.size() == 1);
        }
    }
}

TEST_CASE("Failed handle addresses never read guest memory", "[video_core][texture_handles]") {
    using Op = Shader::HandleOp;
    const auto word = [](Op op) { return static_cast<u32>(op); };
    u32 cbuf_reads{};
    u32 storage_reads{};
    const std::vector<u32> code{word(Op::Cbuf),    0U, 0U, word(Op::Immediate), 16U, word(Op::Add),
                                word(Op::Storage), 0U};
    CHECK_FALSE(Shader::EvaluateHandleProgram(
        code,
        [&](u32, u32) -> std::optional<u32> {
            ++cbuf_reads;
            return std::nullopt;
        },
        [&](u32, u32) {
            ++storage_reads;
            return 7U;
        }));
    CHECK(cbuf_reads == 1);
    CHECK(storage_reads == 0);
}

TEST_CASE("Writable legacy handles cannot come from predicates or addresses",
          "[video_core][texture_handles]") {
    for (const bool buffer : {false, true}) {
        for (const bool predicate : {false, true}) {
            HandleProgram shader;
            const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
            const Value lane = shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}});
            Value handle;
            if (predicate) {
                const Value condition = shader.Emit(Opcode::IEqual, {word, Value{0U}});
                handle = shader.Emit(Opcode::SelectU32, {condition, lane, Value{0U}});
            } else {
                const Value offset = shader.Emit(Opcode::IMul32, {word, lane});
                handle = shader.Emit(Opcode::GetCbufU32, {Value{0U}, offset});
            }
            IR::TextureInstInfo flags{};
            flags.type.Assign(buffer ? Shader::TextureType::Buffer : Shader::TextureType::Color2D);
            auto& image = *shader.Emit(Opcode::BindlessImageWrite, {handle, {}, {}}).Inst();
            image.SetFlags(flags);
            if (Shader::StrictHandles) {
                REQUIRE_THROWS_AS(shader.Run(), Shader::NotImplementedException);
                CHECK(shader.program.info.image_descriptors.empty());
                CHECK(shader.program.info.image_buffer_descriptors.empty());
            } else {
                REQUIRE_NOTHROW(shader.Run());
                CHECK(shader.program.info.image_descriptors.size() +
                          shader.program.info.image_buffer_descriptors.size() ==
                      1);
            }
        }
    }
}

TEST_CASE("Phi handle equality checks complete computations", "[video_core][texture_handles]") {
    // Both trees are independently allocated. Only identical operations, flags and inputs match.
    for (u32 difference = 0; difference < 4; ++difference) {
        CAPTURE(difference);
        HandleProgram shader;
        shader.env.cbuf_values = {{{1, 0x10}, 0xabc}, {{1, 0x14}, 0xabc}};
        shader.insertion_block = &shader.left;
        const Value left_word = shader.Emit(Opcode::GetCbufU32, {Value{1U}, Value{0x10U}});
        const Value first = shader.Add(left_word, Value{3U});
        shader.insertion_block = &shader.right;
        const Value right_word =
            shader.Emit(Opcode::GetCbufU32, {Value{1U}, Value{difference == 1 ? 0x14U : 0x10U}});
        const Value second =
            shader.Emit(difference == 2 ? Opcode::ISub32 : Opcode::IAdd32, {right_word, Value{3U}});
        if (difference == 3)
            second.Inst()->SetFlags(1U);
        auto& sample = shader.Sample(shader.Phi(first, second));
        shader.Run();
        CHECK(shader.Descriptor(sample).bind_null == (Shader::StrictHandles && difference != 0));
        if (Shader::StrictHandles && difference == 0) {
            CHECK(Shader::EvaluateHandleProgram(
                      shader.program.info.handle_programs.front(), [](u32, u32) { return 0xabcU; },
                      [](u32, u32) { return 0U; }) == 0xabfU);
        }
    }
}

TEST_CASE("Review19 masks and single shifts retain legacy descriptors by default",
          "[video_core][texture_handles]") {
    for (u32 shape = 0; shape < 6; ++shape) {
        CAPTURE(shape);
        HandleProgram shader;
        shader.env.cbuf_values = {{{0, 0}, 5U}, {{0, 4}, 0x00300000U}};
        shader.env.integer_texture = true;
        shader.env.texture_type = Shader::TextureType::Color3D;
        shader.env.pixel_format = Shader::TexturePixelFormat::R32_UINT;
        const Value first = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
        const Value second = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{4U}});
        const Value low = shader.Emit(Opcode::BitwiseAnd32, {first, Value{0xfffffU}});
        const Value high = shader.Emit(Opcode::BitwiseAnd32, {second, Value{0xfff00000U}});
        Value handle = low;
        if (shape == 0)
            handle = shader.Emit(Opcode::BitwiseOr32, {low, second});
        if (shape == 1)
            handle = shader.Emit(Opcode::BitwiseOr32, {first, high});
        if (shape == 3)
            handle = shader.Emit(Opcode::BitwiseOr32, {low, high});
        if (shape == 4)
            handle = shader.Emit(Opcode::ShiftLeftLogical32, {first, Value{4U}});
        if (shape == 5) {
            const Value shifted = shader.Emit(Opcode::ShiftLeftLogical32, {second, Value{20U}});
            handle = shader.Emit(Opcode::BitwiseOr32, {first, shifted});
        }
        shader.env.expected_handle = shape == 0 ? 0x00300005U : shape == 4 ? 85U : 5U;
        IR::IREmitter ir{shader.block};
        auto& query = *ir.ImageQueryDimension(handle, ir.Imm32(0U), ir.Imm1(false)).Inst();
        shader.Run();
        REQUIRE(shader.program.info.texture_descriptors.size() == 1);
        const auto& desc = shader.program.info.texture_descriptors.front();
        CHECK(desc.type == Shader::TextureType::Color3D);
        CHECK(desc.is_integer);
        CHECK(query.Flags<IR::TextureInstInfo>().is_integer == 1);
        CHECK_FALSE(desc.bind_null);
        if (!Shader::StrictHandles || shape == 5) {
            CHECK(desc.handle_program == ~0U);
            CHECK(desc.has_secondary == (shape == 0 || shape == 1 || shape == 3 || shape == 5));
            CHECK(desc.cbuf_index == 0);
            CHECK(desc.cbuf_offset == 0);
            CHECK(desc.shift_left == (shape == 4 ? 4U : 0U));
            if (desc.has_secondary) {
                CHECK(desc.secondary_cbuf_offset == 4);
                CHECK(desc.secondary_shift_left == (shape == 0 ? 0U : 20U));
            }
            // Same fields read by the renderers: preserve texture and sampler, not just metadata.
            const u32 raw = desc.has_secondary ? (5U << desc.shift_left) |
                                                     (0x00300000U << desc.secondary_shift_left)
                                               : 5U;
            const u32 previous = shape == 0 ? 0x00300005U : 5U;
            CHECK(Tegra::Texture::TexturePair(raw, false) ==
                  Tegra::Texture::TexturePair(previous, false));
            // Memo and delta blockers inspect these same static/computed and secondary flags.
            CHECK((desc.handle_program == ~0U && !desc.has_secondary && !desc.bind_null) ==
                  (shape == 2 || shape == 4));
        } else {
            REQUIRE(desc.handle_program < shader.program.info.handle_programs.size());
            CHECK(Shader::EvaluateHandleProgram(
                      shader.program.info.handle_programs[desc.handle_program],
                      [](u32, u32 offset) { return offset == 0 ? 5U : 0x00300000U; },
                      [](u32, u32) { return 0U; }) == (shape == 2   ? 5U
                                                       : shape == 4 ? 80U
                                                       : shape == 5 ? 5U
                                                                    : 0x00300005U));
        }
    }
}

TEST_CASE("Review19 single shifts preserve signed normalized texel metadata",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.env.texture_type = Shader::TextureType::Buffer;
    shader.env.pixel_format = Shader::TexturePixelFormat::R16_SNORM;
    const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}});
    const Value handle = shader.Emit(Opcode::ShiftLeftLogical32, {word, Value{4U}});
    IR::TextureInstInfo flags{};
    flags.type.Assign(Shader::TextureType::Buffer);
    auto& fetch = *shader.Emit(Opcode::BindlessImageFetch, {handle, Value{0U}, {}, {}, {}}).Inst();
    fetch.SetFlags(flags);
    shader.Run();
    REQUIRE(shader.program.info.texture_buffer_descriptors.size() == 1);
    CHECK(shader.env.pixel_format_reads == 1);
    CHECK((shader.program.info.texture_buffer_descriptors.front().handle_program != ~0U) ==
          Shader::StrictHandles);
    CHECK(std::ranges::any_of(shader.block.Instructions(), [](const IR::Inst& inst) {
        return inst.GetOpcode() == Opcode::FPMul32;
    }));
}

TEST_CASE("Review19 equal storage reloads survive both Phi arms", "[video_core][texture_handles]") {
    for (const bool writes : {false, true}) {
        for (const bool different_offset : {false, true}) {
            HandleProgram shader;
            shader.insertion_block = &shader.left;
            const Value first = shader.Storage(0);
            shader.insertion_block = &shader.right;
            const Value offset = shader.Add(Value{0U}, Value{different_offset ? 4U : 0U});
            const Value second = shader.Emit(Opcode::LoadStorage32, {Value{0U}, offset});
            auto& sample = shader.Sample(shader.Phi(first, second));
            if (writes)
                shader.Emit(Opcode::WriteStorage32, {Value{0U}, Value{0U}, Value{7U}});
            shader.Run();
            // Equal arithmetic addresses must have equal structure as well as value.
            CHECK(shader.Descriptor(sample).bind_null == Shader::StrictHandles);
        }
        for (const bool immediate : {false, true}) {
            HandleProgram shader;
            shader.insertion_block = &shader.left;
            const Value first_offset = shader.Add(Value{4U}, Value{8U});
            const Value first = shader.Emit(Opcode::LoadStorage32,
                                            {Value{0U}, immediate ? Value{12U} : first_offset});
            shader.insertion_block = &shader.right;
            const Value second_offset = shader.Add(Value{4U}, Value{8U});
            const Value second = shader.Emit(Opcode::LoadStorage32,
                                             {Value{0U}, immediate ? Value{12U} : second_offset});
            auto& sample = shader.Sample(shader.Phi(first, second));
            if (writes)
                shader.Emit(Opcode::WriteStorage32, {Value{0U}, Value{0U}, Value{7U}});
            shader.Run();
            CHECK(shader.Descriptor(sample).bind_null == (Shader::StrictHandles && writes));
            if (!shader.Descriptor(sample).bind_null) {
                CHECK(Shader::EvaluateHandleProgram(
                          shader.program.info.handle_programs.front(), [](u32, u32) { return 0U; },
                          [](u32 binding, u32 offset) {
                              CHECK(binding == 0);
                              CHECK(offset == 12);
                              return 0xabcU;
                          }) == 0xabcU);
            }
        }
    }
}

TEST_CASE("Review19 writable Select keeps its legacy descriptor without skipping the draw",
          "[video_core][texture_handles]") {
    for (const bool byte_address : {false, true}) {
        HandleProgram shader;
        shader.env.cbuf_values = {{{0, 0x10}, 7U}, {{0, 0x14}, 9U}};
        shader.env.expected_handle = 9U;
        shader.env.pixel_format = Shader::TexturePixelFormat::R32_UINT;
        const Value lane = shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}});
        const Value condition = shader.Emit(Opcode::INotEqual, {lane, Value{0U}});
        const Value first = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0x10U}});
        const Value second = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0x14U}});
        const Value handle = shader.Emit(Opcode::SelectU32, {condition, first, second});
        IR::TextureInstInfo flags{};
        flags.type.Assign(Shader::TextureType::Color2D);
        flags.image_format.Assign(Shader::ImageFormat::R32_UINT);
        flags.image_byte_address.Assign(byte_address ? 1U : 0U);
        IR::IREmitter ir{shader.block};
        ir.ImageWrite(handle, ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U)),
                      ir.CompositeConstruct(ir.Imm32(1U), ir.Imm32(1U), ir.Imm32(1U), ir.Imm32(1U)),
                      flags);
        REQUIRE_NOTHROW(shader.Run());
        REQUIRE(shader.program.info.image_descriptors.size() == 1);
        const auto& desc = shader.program.info.image_descriptors.front();
        CHECK(desc.cbuf_offset == 0x14);
        CHECK(desc.handle_program == ~0U);
    }
}

TEST_CASE("Review19 global reloads remain equal after storage lowering",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    shader.env.cbuf_values = {{{0, 0x310}, 0x2000U}, {{0, 0x314}, 0U}};
    const auto reload = [&]() {
        IR::IREmitter ir{*shader.insertion_block};
        const auto low = ir.GetCbuf(ir.Imm32(0U), ir.Imm32(0x310U));
        const auto high = ir.GetCbuf(ir.Imm32(0U), ir.Imm32(0x314U));
        return Value{ir.LoadGlobal32(ir.PackUint2x32(ir.CompositeConstruct(low, high)))};
    };
    shader.insertion_block = &shader.left;
    const Value first = reload();
    shader.insertion_block = &shader.right;
    const Value second = reload();
    auto& sample = shader.Sample(shader.Phi(first, second));
    Shader::HostTranslateInfo host;
    host.min_ssbo_alignment = 16;
    host.support_int64 = true;
    Shader::Optimization::GlobalMemoryToStorageBufferPass(shader.env, shader.program, host);
    shader.Run();
    CHECK_FALSE(shader.Descriptor(sample).bind_null);
    REQUIRE(shader.program.info.storage_buffers_descriptors.size() == 1);
    CHECK(Shader::EvaluateHandleProgram(
              shader.program.info.handle_programs.front(), [](u32, u32) { return 0U; },
              [](u32 binding, u32 offset) {
                  CHECK(binding == 0);
                  CHECK(offset == 0);
                  return 0xabcU;
              }) == 0xabcU);
}

TEST_CASE("Review19 masked writable handles preserve descriptor mode",
          "[video_core][texture_handles]") {
    for (const bool byte_address : {false, true}) {
        HandleProgram shader;
        shader.env.cbuf_values = {{{0, 0x10}, 7U}};
        shader.env.expected_handle = 7U;
        shader.env.pixel_format = Shader::TexturePixelFormat::R32_UINT;
        const Value word = shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0x10U}});
        const Value handle = shader.Emit(Opcode::BitwiseAnd32, {word, Value{0xfffffU}});
        IR::TextureInstInfo flags{};
        flags.type.Assign(Shader::TextureType::Color2D);
        flags.image_format.Assign(Shader::ImageFormat::R32_UINT);
        flags.image_byte_address.Assign(byte_address ? 1U : 0U);
        IR::IREmitter ir{shader.block};
        ir.ImageWrite(handle, ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U)),
                      ir.CompositeConstruct(ir.Imm32(1U), ir.Imm32(1U), ir.Imm32(1U), ir.Imm32(1U)),
                      flags);
        REQUIRE_NOTHROW(shader.Run());
        REQUIRE(shader.program.info.image_descriptors.size() == 1);
        const auto& desc = shader.program.info.image_descriptors.front();
        CHECK(desc.cbuf_index == 0);
        CHECK(desc.cbuf_offset == 0x10);
        CHECK((desc.handle_program != ~0U) == (Shader::StrictHandles && !byte_address));
    }
}
