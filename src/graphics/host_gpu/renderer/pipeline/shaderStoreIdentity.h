#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREIDENTITY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREIDENTITY_H_

#include <cstdint>
#include <string>

namespace Libs::Graphics {

// The SPIR-V profile fields that identify a shader store, in the order the store header has always
// written them. A debug option that changes the emitted SPIR-V (--loop-guard) is deliberately NOT
// part of it: adding it to the header made every boot with a different option value reject the
// whole store and rewrite it (0 programs loaded). Such an option selects its own store file
// instead (ShaderStoreSuffix), so each variant keeps and reloads its own entries.
template <class Profile, class Visit> void VisitShaderStoreProfile(const Profile& profile, Visit&& visit) {
	visit(profile.maintenance8);
	visit(profile.native_float_min_max);
	visit(profile.device_clock_period_ps);
	visit(profile.robust_buffer_access);
	visit(profile.r8_srgb_views);
	visit(profile.r8g8_srgb_views);
	visit(profile.relaxed_float);
	visit(profile.ray_query);
}

// File-name part that separates stores whose SPIR-V differs by a debug option. Empty = default.
[[nodiscard]] inline std::string ShaderStoreSuffix(uint32_t loop_guard_limit) {
	return loop_guard_limit == 0 ? std::string {} : ".guard" + std::to_string(loop_guard_limit);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREIDENTITY_H_
