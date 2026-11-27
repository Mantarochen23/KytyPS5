#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPTAINT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPTAINT_H_

// Guest memory a skipped dispatch was going to write (--async-compute-wait-ms). A skipped dispatch
// leaves its output buffers with stale or zero contents; a later dispatch that reads such a
// buffer may loop on that data for ever (a traversal that never meets its terminator). The
// ranges the skipped dispatch would have written are remembered as tainted, a later dispatch that
// reads one is skipped as well (its own outputs become tainted: the taint follows the data), and
// a dispatch that really runs and writes a range clears it. A range also expires after a
// bounded time so a producer that never runs again cannot silence its consumers for ever.
//
// Only the buffers the shader names through descriptors are tracked: memory a shader reaches by
// a pointer it computes itself is not (the loop guard is the net for that).

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class SkipTaint {
public:
	struct Use {
		uint64_t address = 0;
		uint64_t size    = 0;
		bool     read    = false;
		bool     written = false;
	};

	static constexpr uint64_t DefaultMaxAgeMs = 5000;

	explicit SkipTaint(uint64_t max_age_ms = DefaultMaxAgeMs): m_max_age_ms(max_age_ms) {}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }
	[[nodiscard]] size_t Size() const { return m_ranges.size(); }

	// The written ranges of `uses` become tainted (the dispatch was skipped).
	void Taint(const std::vector<Use>& uses, uint64_t now_ms) {
		for (const auto& use: uses) {
			if (use.written && use.size != 0) {
				m_ranges.push_back({use.address, use.address + use.size, now_ms});
			}
		}
	}

	// True when a range `uses` reads is tainted and not yet expired.
	[[nodiscard]] bool ReadsTainted(const std::vector<Use>& uses, uint64_t now_ms) {
		Expire(now_ms);
		for (const auto& use: uses) {
			if (!use.read || use.size == 0) continue;
			for (const auto& range: m_ranges) {
				if (use.address < range.end && range.begin < use.address + use.size) return true;
			}
		}
		return false;
	}

	// The dispatch ran: what it wrote is fresh.
	void ClearWritten(const std::vector<Use>& uses) {
		for (const auto& use: uses) {
			if (use.written && use.size != 0) Subtract(use.address, use.address + use.size);
		}
	}

private:
	struct Range {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t at_ms = 0;
	};

	void Expire(uint64_t now_ms) {
		std::erase_if(m_ranges, [&](const Range& r) {
			return now_ms >= r.at_ms && now_ms - r.at_ms > m_max_age_ms;
		});
	}

	void Subtract(uint64_t begin, uint64_t end) {
		std::vector<Range> kept;
		kept.reserve(m_ranges.size() + 1);
		for (const auto& r: m_ranges) {
			if (end <= r.begin || r.end <= begin) {
				kept.push_back(r);
				continue;
			}
			if (r.begin < begin) kept.push_back({r.begin, begin, r.at_ms});
			if (end < r.end) kept.push_back({end, r.end, r.at_ms});
		}
		m_ranges = std::move(kept);
	}

	uint64_t           m_max_age_ms;
	std::vector<Range> m_ranges;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPTAINT_H_
