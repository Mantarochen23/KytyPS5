#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORECAPABILITY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORECAPABILITY_H_

#include <cstdint>

namespace Libs::Graphics {

// A translator capability generation. Failed translations are stored in the persistent
// shader store as negative entries so they are not retried on every boot. Bump this when
// the translator gains a capability that makes earlier rejections translatable: a failure
// recorded under an older generation no longer applies and is retried once. It is a field
// of the negative-cache key, not of the store format, so warm positive records and driver
// pipelines keep being reused.
inline constexpr uint64_t kTranslatorCapabilityVersion = 5;

[[nodiscard]] inline constexpr bool FailureCacheApplies(uint64_t stored_version) {
  return stored_version == kTranslatorCapabilityVersion;
}

// Per-record validity for ports that only widen what the translator accepts (common/ports.h,
// widens_translation). A stored failure remembers which such ports were on when it failed
// (`stored_ports`) and still applies only if no port is on now that was not then: a newly enabled
// port may translate it. A failure that predates the field has stored_ports = 0.
[[nodiscard]] inline constexpr bool FailureCacheApplies(uint64_t stored_version, uint32_t stored_ports,
                                                        uint32_t current_ports) {
  return FailureCacheApplies(stored_version) && (current_ports & ~stored_ports) == 0u;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORECAPABILITY_H_
