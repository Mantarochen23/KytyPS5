#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SUBGROUPSIZE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SUBGROUPSIZE_H_

#include <cstdint>

namespace Libs::Graphics {

struct SubgroupSizeCaps {
	uint32_t default_size     = 0;
	uint32_t min_size         = 0;
	uint32_t max_size         = 0;
	uint32_t supported_stages = 0; // VkShaderStageFlags bitmask.
};

struct SubgroupSizeDecision {
	uint32_t expected_size    = 0;
	uint32_t required_size    = 0;
	bool     apply            = false;
	bool     mismatch_warning = false;
};

// Resolves the host subgroup size contract for one compiled shader stage.
// When the driver supports requiredSubgroupSize for `stage_bit` and the stage's
// expected host subgroup size is within [min_size, max_size], `apply` is true
// and the caller attaches VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.
// Otherwise, if the stage executes subgroup operations and its expected host
// subgroup size differs from the device default subgroup size, `mismatch_warning`
// is set so the caller emits a permanent diagnostic warning.
[[nodiscard]] constexpr SubgroupSizeDecision ResolveStageSubgroupSize(
    uint32_t stage_bit, uint32_t guest_wave_size, bool emulated_wave64_on_32,
    bool uses_subgroup_ops, const SubgroupSizeCaps& caps) noexcept {
	SubgroupSizeDecision decision {};
	if (guest_wave_size != 32u && guest_wave_size != 64u) {
		return decision;
	}
	const uint32_t expected = emulated_wave64_on_32 ? 32u : guest_wave_size;
	decision.expected_size  = expected;

	const bool stage_supported = (caps.supported_stages & stage_bit) != 0u;
	const bool size_supported =
	    caps.min_size != 0u && expected >= caps.min_size && expected <= caps.max_size;
	if (stage_supported && size_supported) {
		decision.required_size = expected;
		decision.apply         = true;
		return decision;
	}

	if (uses_subgroup_ops && caps.default_size != 0u && expected != caps.default_size) {
		decision.mismatch_warning = true;
	}
	return decision;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SUBGROUPSIZE_H_
