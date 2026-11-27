#include "graphics/guest_gpu/prefetchBackoff.h"

#include <cstdio>
#include <cstdlib>

using Libs::Graphics::PrefetchBackoff;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PrefetchBackoffTests: failed: %s\n", message);
		std::abort();
	}
}

// Runs `count` submissions, all quiet when scanned; returns how many were scanned.
uint32_t Run(PrefetchBackoff& policy, uint32_t count, uint32_t exponent) {
	uint32_t scanned = 0;
	for (uint32_t i = 0; i < count; ++i) {
		if (policy.ShouldScan(exponent)) {
			++scanned;
			policy.Scanned(true, exponent);
		}
	}
	return scanned;
}

} // namespace

int main() {
	{
		PrefetchBackoff off;
		Check(Run(off, 1000, 0) == 1000, "exponent 0 scans every submission");
	}
	{
		PrefetchBackoff policy;
		Check(Run(policy, 4, 4) == 4, "the first quiet scans are not skipped");
		Check(policy.QuietScans() == 4 && policy.SkipLeft() == 1, "skipping starts after four");
		// Steady state: one scan in 2^4 submissions.
		const auto scanned = Run(policy, 10000, 4);
		Check(scanned > 10000 / 16 - 8 && scanned < 10000 / 16 + 8, "a scan in sixteen at most");
		// A scan that found something returns to every submission.
		for (uint32_t i = 0; i < 20 && !policy.ShouldScan(4); ++i) {
		}
		policy.Scanned(false, 4);
		Check(policy.SkipLeft() == 0 && policy.QuietScans() == 0, "news resets");
		Check(Run(policy, 4, 4) == 4, "scanning every submission again");
	}
	{
		PrefetchBackoff policy;
		// The skip grows by doubling and never exceeds the cap.
		uint32_t longest = 0;
		for (uint32_t i = 0; i < 200; ++i) {
			if (policy.ShouldScan(2)) {
				policy.Scanned(true, 2);
			}
			if (policy.SkipLeft() > longest) {
				longest = policy.SkipLeft();
			}
		}
		Check(longest == 3, "an exponent of 2 skips at most 3");
	}
	std::printf("PrefetchBackoffTests: ok\n");
	return 0;
}
