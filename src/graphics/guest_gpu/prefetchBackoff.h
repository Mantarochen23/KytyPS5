#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_PREFETCHBACKOFF_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_PREFETCHBACKOFF_H_

#include <cstdint>

namespace Libs::Graphics {

// Decides which submissions get the scan that looks for compute shaders to translate ahead of
// their dispatch. A scan whose every shader was already known found nothing to do; after a few
// such scans in a row the next ones are skipped, doubling the number skipped up to
// 2^max_exponent - 1 between scans. A scan that did find something (a new or unmapped shader)
// brings the scanning back to every submission.
//
// max_exponent 0 scans every submission.
class PrefetchBackoff final {
public:
	static constexpr uint32_t QuietScansBeforeSkipping = 4;

	// True when this submission is to be scanned.
	[[nodiscard]] bool ShouldScan(uint32_t max_exponent) {
		if (max_exponent == 0u) {
			return true;
		}
		if (m_skip_left > 0u) {
			--m_skip_left;
			return false;
		}
		return true;
	}

	// Reports what a scan found: `quiet` when every shader it met was known already.
	void Scanned(bool quiet, uint32_t max_exponent) {
		if (!quiet || max_exponent == 0u) {
			m_quiet_scans = 0;
			m_skip_left   = 0;
			return;
		}
		if (m_quiet_scans < 31u) {
			++m_quiet_scans;
		}
		if (m_quiet_scans >= QuietScansBeforeSkipping) {
			const uint32_t step = m_quiet_scans - QuietScansBeforeSkipping + 1u;
			const uint32_t exponent = step < max_exponent ? step : max_exponent;
			m_skip_left             = (1u << exponent) - 1u;
		}
	}

	[[nodiscard]] uint32_t QuietScans() const { return m_quiet_scans; }
	[[nodiscard]] uint32_t SkipLeft() const { return m_skip_left; }

private:
	uint32_t m_quiet_scans = 0;
	uint32_t m_skip_left   = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_PREFETCHBACKOFF_H_
