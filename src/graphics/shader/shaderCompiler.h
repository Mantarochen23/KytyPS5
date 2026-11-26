#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_

#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/shader.h"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
} // namespace HW

struct ShaderParams {
	std::span<const uint32_t> code;
	std::array<uint32_t, 40>  user_data {}; // 32 user SGPRs plus the merged-stage s0:s7 prefix.
	uint32_t                  user_data_count = 0;
	uint64_t                  hash = 0;
	std::span<const uint32_t> back_code;

	[[nodiscard]] uint64_t Base() const {
		return reinterpret_cast<uint64_t>(code.data());
	}
};

void BuildStageStaticKey(const ShaderVertexInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderPixelInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderComputeInputInfo& input_info, std::vector<uint32_t>& key);

// The preparers below fill an input info the caller value-initializes.
ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& input_info);
// Each stage's tess holds the VGT fields; the caller adds the layout reflected from the programs.
std::array<ShaderParams, 3>
PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            std::array<ShaderVertexInputInfo, 3>& input_info);
ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               input_info);
ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info);
struct ComputeShaderPrefetch {
	HW::CsStageRegisters cs_regs;
	bool                 dispatch_thread_dimensions = false;
};

// Decodes one compute SH register write into `cs_regs`; false if `offset` is not a CS register.
bool ApplyComputeShRegister(HW::CsStageRegisters& cs_regs, uint32_t offset, uint32_t value);

// Records a newly registered compute shader for background pre-translation.
void ShaderQueueComputePrefetch(const HW::CsStageRegisters& cs_regs,
                                bool dispatch_thread_dimensions);
[[nodiscard]] std::vector<ComputeShaderPrefetch> ShaderTakeComputePrefetches();
void ShaderSetComputePrefetchNotifier(Common::UniqueFunction<void>&& notifier);

// Non-fatal lookup for speculative compute prefetch; snapshots mapped code without touching guest
// memory. Returns false if the shader is not mapped.
bool TryPrepareComputeProgram(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& info,
                              ShaderParams& out,
                              std::shared_ptr<const std::vector<uint32_t>>& code_owner);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_ */
