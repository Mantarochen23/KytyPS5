#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHLIMITS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHLIMITS_H_

// Size checks and diagnostics for compute dispatches whose group counts the guest chose.
//
// The guest command processor takes the three counts as full 32-bit values (COMPUTE_DIM_X/Y/Z),
// there is no flag or field in the upper bits: a count of 0x01000062 is 16,777,314 groups. Counts
// that look like that are computed by the program (here: ceil(count / group size) of a user data
// dword that held a float), and a dispatch of that size keeps the host GPU busy until the driver's
// ring timeout. A dispatch above the group limit is reported and skipped; one above the report
// threshold is reported (rate-limited per shader) and still runs.
//
//   --disable-port dispatch-guard     keeps the report, never skips
//   KYTY_DISPATCH_GUARD_MAX_GROUPS=n  the largest x * y * z let through (default 4194304)

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

constexpr uint32_t DispatchDefaultMaxGroups = 1u << 22u;
// A report is made from this many invocations (groups x threads) on.
constexpr uint64_t DispatchReportInvocations = 1ull << 26u;
constexpr uint32_t DispatchReportsPerShader  = 4;

// x * y * z, saturating (three full 32-bit counts do not fit 64 bits).
[[nodiscard]] constexpr uint64_t DispatchGroups(uint32_t x, uint32_t y, uint32_t z) {
	const uint64_t xy = uint64_t {x} * y;
	if (z != 0 && xy > UINT64_MAX / z) return UINT64_MAX;
	return xy * z;
}

// Total invocations; saturates instead of wrapping.
[[nodiscard]] constexpr uint64_t DispatchInvocations(uint32_t x, uint32_t y, uint32_t z,
                                                     uint64_t threads_per_group) {
	const uint64_t groups = DispatchGroups(x, y, z);
	if (threads_per_group != 0 && groups > UINT64_MAX / threads_per_group) return UINT64_MAX;
	return groups * threads_per_group;
}

// The limit text of KYTY_DISPATCH_GUARD_MAX_GROUPS; anything out of range keeps the default.
[[nodiscard]] inline uint32_t ParseDispatchGroupLimit(const char* text) {
	if (text == nullptr || text[0] == '\0') return DispatchDefaultMaxGroups;
	char*      end    = nullptr;
	const auto parsed = std::strtoull(text, &end, 0);
	if (end == text || *end != '\0' || parsed < 1024 || parsed > UINT32_MAX) {
		return DispatchDefaultMaxGroups;
	}
	return static_cast<uint32_t>(parsed);
}

[[nodiscard]] inline uint32_t DispatchGroupLimit() {
	static const uint32_t limit = ParseDispatchGroupLimit(std::getenv("KYTY_DISPATCH_GUARD_MAX_GROUPS"));
	return limit;
}

enum class DispatchVerdict : uint8_t {
	Run,    // normal
	Report, // runs, but is large enough to be named in the log
	Skip,   // above the group limit: cannot be valid
};

[[nodiscard]] constexpr DispatchVerdict JudgeDispatch(uint32_t x, uint32_t y, uint32_t z,
                                                      uint64_t threads_per_group,
                                                      uint32_t max_groups, bool may_skip) {
	if (may_skip && DispatchGroups(x, y, z) > max_groups) return DispatchVerdict::Skip;
	if (DispatchInvocations(x, y, z, threads_per_group) >= DispatchReportInvocations) {
		return DispatchVerdict::Report;
	}
	return DispatchVerdict::Run;
}

// True when counts the CPU has read cannot hurt the device, so that an indirect dispatch reading
// them needs no GPU-side check (dispatchGuard.h): each count is within the device's per-dimension
// limit and the total is within the group limit.
[[nodiscard]] constexpr bool CountsSafeWithoutGuard(uint32_t x, uint32_t y, uint32_t z,
                                                    uint32_t max_x, uint32_t max_y,
                                                    uint32_t max_z, uint32_t max_groups) {
	return x <= max_x && y <= max_y && z <= max_z &&
	       DispatchGroups(x, y, z) <= max_groups;
}

// What the GPU-side check of one set of counts left behind: the counts' guest address, the write
// stamp (BufferCache::WriteStamp) they had, the command buffer the check was recorded in, and the
// position of its result in the guard's ring (`sequence` is the check's running number).
struct ValidatedArgs {
	uint64_t guest_address = 0;
	uint64_t stamp         = 0;
	uint64_t slot          = 0;
	uint64_t sequence      = 0;
	uint64_t command       = 0;
};

// True when `entry` is still the check of the counts at `guest_address`: nothing wrote them since
// (equal non-zero stamp), it was recorded in the same command buffer, so that the queue orders its
// result before this use, and its ring slot was not reused yet (`next` is the running number of
// the next check, `slot_count` the ring size; a quarter of the ring is kept as margin).
[[nodiscard]] constexpr bool ValidationStillHolds(const ValidatedArgs& entry,
                                                  uint64_t guest_address, uint64_t stamp,
                                                  uint64_t command, uint64_t next,
                                                  uint64_t slot_count) {
	return stamp != 0 && entry.stamp == stamp && entry.guest_address == guest_address &&
	       entry.command == command && next >= entry.sequence &&
	       next - entry.sequence < slot_count - slot_count / 4;
}

// Which dword of `words` the x count was computed from: the first i with ceil(words[i] / group)
// == x (x > 1, so that the common 1 does not match everything). -1 when none does.
[[nodiscard]] inline int FindCountDword(const uint32_t* words, uint32_t num, uint32_t x,
                                        uint32_t threads_x) {
	if (x <= 1 || threads_x == 0) return -1;
	for (uint32_t i = 0; i < num; i++) {
		const uint64_t groups = (uint64_t {words[i]} + threads_x - 1u) / threads_x;
		if (groups == x) return static_cast<int>(i);
	}
	return -1;
}

[[nodiscard]] inline float DwordAsFloat(uint32_t value) {
	float result = 0.0f;
	std::memcpy(&result, &value, sizeof(result));
	return result;
}

// Counts the reports of each shader so that one shader that dispatches every frame is named a few
// times, not every frame.
class DispatchReportLimiter {
public:
	explicit DispatchReportLimiter(uint32_t per_shader = DispatchReportsPerShader)
	    : m_per_shader(per_shader) {}

	// True for the first `per_shader` calls of each key.
	[[nodiscard]] bool Allow(uint64_t key) {
		const std::lock_guard lock(m_mutex);
		return m_counts[key]++ < m_per_shader;
	}

private:
	uint32_t                                 m_per_shader;
	std::mutex                               m_mutex;
	std::unordered_map<uint64_t, uint32_t>   m_counts;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHLIMITS_H_ */
