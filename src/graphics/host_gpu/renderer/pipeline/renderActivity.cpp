#include "graphics/host_gpu/renderer/pipeline/renderActivity.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"

#include <fmt/format.h>

namespace Libs::Graphics {

std::string RenderStallReporter::Describe(uint64_t stalled_ms, uint64_t frame, uint64_t now_ms,
                                          const RenderActivity::Snapshot& snapshot,
                                          uint32_t                        jobs_pending) {
	const auto in_phase_ms = now_ms >= snapshot.since_ms ? now_ms - snapshot.since_ms : 0;
	return fmt::format(
	    "[stall] no frame for {:.1f} s (frame {}): render thread in '{}'{}{} for {:.1f} s; shader "
	    "hash=0x{:016x} stage={}; last PM4 opcode 0x{:02x}, {} packets; shader worker jobs "
	    "pending {}",
	    static_cast<double>(stalled_ms) / 1000.0, frame, snapshot.phase,
	    snapshot.detail[0] != '\0' ? " state=" : "", snapshot.detail,
	    static_cast<double>(in_phase_ms) / 1000.0, snapshot.hash, snapshot.stage,
	    snapshot.opcode, snapshot.packets, jobs_pending);
}

RenderStallReporter::RenderStallReporter(uint32_t seconds): m_seconds(seconds) {
	if (m_seconds != 0) {
		m_thread = std::thread([this] { Run(); });
	}
}

RenderStallReporter::~RenderStallReporter() {
	{
		std::scoped_lock lock(m_mutex);
		m_stop = true;
	}
	m_wake.notify_all();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

void RenderStallReporter::Run() {
	auto     frame         = CurrentFrame();
	uint64_t progress_ms   = RenderActivity::NowMs();
	uint64_t last_print_ms = 0;
	const uint64_t period_ms = static_cast<uint64_t>(m_seconds) * 1000u;
	std::unique_lock lock(m_mutex);
	while (!m_stop) {
		m_wake.wait_for(lock, std::chrono::seconds(1), [this] { return m_stop; });
		if (m_stop) {
			break;
		}
		const auto now     = RenderActivity::NowMs();
		const auto current = CurrentFrame();
		if (current != frame) {
			frame         = current;
			progress_ms   = now;
			last_print_ms = 0;
			continue;
		}
		if (now - progress_ms < period_ms || (last_print_ms != 0 && now - last_print_ms < period_ms)) {
			continue;
		}
		last_print_ms = now;
		const auto text = Describe(now - progress_ms, current, now, RenderActivity::Read(),
		                           Detail::g_shader_worker_jobs.load(std::memory_order_relaxed));
		Log::WriteToConsoleAndLog(text + "\n");
	}
}

} // namespace Libs::Graphics
