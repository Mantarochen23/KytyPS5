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
inline constexpr uint64_t kTranslatorCapabilityVersion = 3;

[[nodiscard]] inline constexpr bool FailureCacheApplies(uint64_t stored_version) {
  return stored_version == kTranslatorCapabilityVersion;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORECAPABILITY_H_
