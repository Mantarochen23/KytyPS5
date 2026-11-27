#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FRAMEHOLES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FRAMEHOLES_H_

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <span>
#include <string>
#include <vector>

// Per-frame completeness accounting: counts every dropped draw and compute dispatch by reason,
// records affected render-target or written-buffer addresses, and emits one summary line per
// incomplete frame on flip. Header-only with no Vulkan dependency so it can be unit-tested.
namespace Libs::Graphics::FrameHoles {

enum class DrawSkipReason : uint8_t {
	ShaderBuilding,
	PipelineBuilding,
	FailedShader,
	PipelineFailed,
	UnsupportedStage,
	NoTargets,
	NullShader,
	RectNoExports,
	Count,
};

enum class DispatchSkipReason : uint8_t {
	Wait,
	Tainted,
	Refused,
	FailedShader,
	PipelineFailed,
	NullShader,
	Count,
};

[[nodiscard]] constexpr const char* Name(DrawSkipReason reason) noexcept {
	switch (reason) {
		case DrawSkipReason::ShaderBuilding: return "shader-building";
		case DrawSkipReason::PipelineBuilding: return "pipeline-building";
		case DrawSkipReason::FailedShader: return "failed-shader";
		case DrawSkipReason::PipelineFailed: return "pipeline-failed";
		case DrawSkipReason::UnsupportedStage: return "unsupported-stage";
		case DrawSkipReason::NoTargets: return "no-targets";
		case DrawSkipReason::NullShader: return "null-shader";
		case DrawSkipReason::RectNoExports: return "rect-no-exports";
		case DrawSkipReason::Count: break;
	}
	return "unknown";
}

[[nodiscard]] constexpr const char* Name(DispatchSkipReason reason) noexcept {
	switch (reason) {
		case DispatchSkipReason::Wait: return "wait";
		case DispatchSkipReason::Tainted: return "tainted";
		case DispatchSkipReason::Refused: return "refused";
		case DispatchSkipReason::FailedShader: return "failed-shader";
		case DispatchSkipReason::PipelineFailed: return "pipeline-failed";
		case DispatchSkipReason::NullShader: return "null-shader";
		case DispatchSkipReason::Count: break;
	}
	return "unknown";
}

class Tracker {
public:
	static constexpr size_t MaxTrackedTargets = 16;

	struct Snapshot {
		std::array<uint64_t, static_cast<size_t>(DrawSkipReason::Count)>     draws {};
		std::array<uint64_t, static_cast<size_t>(DispatchSkipReason::Count)> dispatches {};
		std::vector<uint64_t>                                                targets;
		uint64_t                                                             extra_targets = 0;
		uint64_t                                                             draws_total   = 0;
		uint64_t                                                             dispatches_total = 0;

		[[nodiscard]] bool Empty() const noexcept {
			return draws_total == 0 && dispatches_total == 0;
		}
	};

	void RecordDraw(DrawSkipReason reason, std::span<const uint64_t> targets = {}) {
		std::lock_guard lock(m_mutex);
		m_draws[static_cast<size_t>(reason)]++;
		m_draws_total++;
		AddTargetsLocked(targets);
	}

	void RecordDispatch(DispatchSkipReason reason, std::span<const uint64_t> buffers = {}) {
		std::lock_guard lock(m_mutex);
		m_dispatches[static_cast<size_t>(reason)]++;
		m_dispatches_total++;
		AddTargetsLocked(buffers);
	}

	[[nodiscard]] Snapshot TakeSnapshot() {
		std::lock_guard lock(m_mutex);
		Snapshot        out;
		out.draws            = m_draws;
		out.dispatches       = m_dispatches;
		out.targets          = std::move(m_targets);
		out.extra_targets    = m_extra_targets;
		out.draws_total      = m_draws_total;
		out.dispatches_total = m_dispatches_total;
		m_draws              = {};
		m_dispatches         = {};
		m_targets.clear();
		m_extra_targets    = 0;
		m_draws_total      = 0;
		m_dispatches_total = 0;
		return out;
	}

	[[nodiscard]] static std::string FormatLine(uint64_t frame, const Snapshot& snap) {
		if (snap.Empty()) {
			return {};
		}
		std::string line;
		char        buf[128];
		std::snprintf(buf, sizeof(buf), "[frame-holes] frame=%" PRIu64 " draws skipped=%" PRIu64 " (",
		              frame, snap.draws_total);
		line += buf;
		AppendReasons<DrawSkipReason>(line, snap.draws);
		std::snprintf(buf, sizeof(buf), ") dispatches skipped=%" PRIu64 " (",
		              snap.dispatches_total);
		line += buf;
		AppendReasons<DispatchSkipReason>(line, snap.dispatches);
		const uint64_t total_targets = snap.targets.size() + snap.extra_targets;
		std::snprintf(buf, sizeof(buf), ") targets affected=%" PRIu64, total_targets);
		line += buf;
		if (!snap.targets.empty()) {
			line += " (";
			const size_t shown = std::min<size_t>(snap.targets.size(), 8);
			for (size_t i = 0; i < shown; ++i) {
				if (i != 0) line += ", ";
				std::snprintf(buf, sizeof(buf), "0x%012" PRIx64, snap.targets[i]);
				line += buf;
			}
			if (total_targets > shown) {
				std::snprintf(buf, sizeof(buf), ", +%" PRIu64 " more", total_targets - shown);
				line += buf;
			}
			line += ")";
		}
		return line;
	}

	[[nodiscard]] std::string TakeFrameLine(uint64_t frame) {
		const auto snap = TakeSnapshot();
		return FormatLine(frame, snap);
	}

private:
	template <typename Enum, size_t N>
	static void AppendReasons(std::string& out, const std::array<uint64_t, N>& counts) {
		bool first = true;
		char buf[96];
		for (size_t i = 0; i < N; ++i) {
			if (counts[i] == 0) continue;
			if (!first) out += ", ";
			first = false;
			std::snprintf(buf, sizeof(buf), "%s=%" PRIu64, Name(static_cast<Enum>(i)), counts[i]);
			out += buf;
		}
		if (first) {
			out += "none";
		}
	}

	void AddTargetsLocked(std::span<const uint64_t> addrs) {
		for (const uint64_t addr: addrs) {
			if (addr == 0) continue;
			if (std::find(m_targets.begin(), m_targets.end(), addr) != m_targets.end()) {
				continue;
			}
			if (m_targets.size() < MaxTrackedTargets) {
				m_targets.push_back(addr);
			} else {
				m_extra_targets++;
			}
		}
	}

	std::mutex                                                           m_mutex;
	std::array<uint64_t, static_cast<size_t>(DrawSkipReason::Count)>     m_draws {};
	std::array<uint64_t, static_cast<size_t>(DispatchSkipReason::Count)> m_dispatches {};
	std::vector<uint64_t>                                                m_targets;
	uint64_t                                                             m_extra_targets    = 0;
	uint64_t                                                             m_draws_total      = 0;
	uint64_t                                                             m_dispatches_total = 0;
};

inline Tracker& Global() noexcept {
	static Tracker tracker;
	return tracker;
}

inline void NoteDrawSkip(DrawSkipReason reason, std::span<const uint64_t> targets = {}) {
	Global().RecordDraw(reason, targets);
}

inline void NoteDispatchSkip(DispatchSkipReason reason, std::span<const uint64_t> buffers = {}) {
	Global().RecordDispatch(reason, buffers);
}

inline void FlushFrame(uint64_t frame) {
	const auto line = Global().TakeFrameLine(frame);
	if (!line.empty()) {
		std::printf("%s\n", line.c_str());
	}
}

} // namespace Libs::Graphics::FrameHoles

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FRAMEHOLES_H_
