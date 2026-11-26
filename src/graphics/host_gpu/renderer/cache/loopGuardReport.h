#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_LOOPGUARDREPORT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_LOOPGUARDREPORT_H_

// Layout and formatting of the loop-guard records in the tail of the fault buffer. A shader
// compiled with --loop-guard adds one record when an invocation used up its loop-exit budget:
// words [count, reserved] followed by LoopGuardSlotCount slots of {hash_lo, hash_hi, loop, 0}.
// `loop` is the IR block id of the loop header whose exit evaluation first ran over budget.

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {

inline constexpr uint32_t LoopGuardSlotCount  = 16;
inline constexpr uint32_t LoopGuardSlotDwords = 4;
inline constexpr uint32_t LoopGuardHeaderDwords = 2;
inline constexpr uint32_t LoopGuardAreaDwords =
    LoopGuardHeaderDwords + LoopGuardSlotCount * LoopGuardSlotDwords;
// Loop id of a shader the recompiler lowered to a block dispatcher: the guard then counts block
// transitions, not loop exits.
inline constexpr uint32_t LoopGuardDispatcherLoop = 0xfffffffeu;

[[nodiscard]] inline std::vector<std::string>
FormatLoopGuardTrips(std::span<const uint32_t> words, std::set<std::pair<uint64_t, uint32_t>>& seen) {
	std::vector<std::string> lines;
	if (words.size() < LoopGuardHeaderDwords || words[0] == 0u) return lines;
	const uint32_t count    = words[0];
	const uint32_t slots    = static_cast<uint32_t>((words.size() - LoopGuardHeaderDwords) /
                                                 LoopGuardSlotDwords);
	const uint32_t recorded = std::min({count, LoopGuardSlotCount, slots});
	for (uint32_t i = 0; i < recorded; i++) {
		const auto*    slot = words.data() + LoopGuardHeaderDwords + i * LoopGuardSlotDwords;
		const uint64_t hash = uint64_t {slot[0]} | (uint64_t {slot[1]} << 32u);
		const uint32_t loop = slot[2];
		if (!seen.emplace(hash, loop).second) continue;
		char text[192];
		if (loop == LoopGuardDispatcherLoop) {
			std::snprintf(text, sizeof(text),
			              "[loop-guard] shader=0x%016" PRIx64
			              " loop=dispatcher tripped_invocations=%u (first report of this shader)",
			              hash, count);
		} else {
			std::snprintf(text, sizeof(text),
			              "[loop-guard] shader=0x%016" PRIx64
			              " loop=B%u tripped_invocations=%u (first report of this loop)",
			              hash, loop, count);
		}
		lines.emplace_back(text);
	}
	return lines;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_LOOPGUARDREPORT_H_
