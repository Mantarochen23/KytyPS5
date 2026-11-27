#include "graphics/host_gpu/renderer/pipeline/renderActivity.h"

#include "common/clockHold.h"
#include "common/guestPark.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "graphics/guest_gpu/command_processor/waitDiagnostics.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"

#include <fmt/format.h>

namespace Libs::Graphics {

std::string RenderStallReporter::Describe(uint64_t stalled_ms, uint64_t frame, uint64_t now_ms,
                                          const RenderActivity::Snapshot& snapshot,
                                          uint32_t jobs_pending, const StallExtras& extras) {
	const auto in_phase_ms = now_ms >= snapshot.since_ms ? now_ms - snapshot.since_ms : 0;
	auto       text        = fmt::format(
        "[stall] no frame for {:.1f} s (frame {}): render thread in '{}'{}{} for {:.1f} s; shader "
        "hash=0x{:016x} stage={}; last PM4 opcode 0x{:02x}, {} packets; shader worker jobs "
        "pending {}",
        static_cast<double>(stalled_ms) / 1000.0, frame, snapshot.phase,
        snapshot.detail[0] != '\0' ? " state=" : "", snapshot.detail,
        static_cast<double>(in_phase_ms) / 1000.0, snapshot.hash, snapshot.stage,
        snapshot.opcode, snapshot.packets, jobs_pending);
	if (extras.clock_hold_enabled) {
		text += extras.clock_held
		            ? fmt::format("; guest clock held for {:.1f} s (total held {:.1f} s in {} holds)",
		                          static_cast<double>(extras.clock_current_ms) / 1000.0,
		                          static_cast<double>(extras.clock_total_ms) / 1000.0,
		                          extras.clock_holds)
		            : fmt::format("; guest clock running (total held {:.1f} s in {} holds)",
		                          static_cast<double>(extras.clock_total_ms) / 1000.0,
		                          extras.clock_holds);
	}
	text += fmt::format("; {} guest threads parked", extras.parked_threads);
	return text;
}

RenderStallReporter::RenderStallReporter(uint32_t seconds, uint32_t clock_hold_ms,
                                         bool report_parked)
    : m_seconds(seconds), m_clock_hold_ms(clock_hold_ms), m_report_parked(report_parked) {
	if (m_seconds != 0 || m_clock_hold_ms != 0 || m_report_parked) {
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
	uint64_t last_park_ms  = 0;
	const uint64_t period_ms = static_cast<uint64_t>(m_seconds) * 1000u;
	// The clock hold reacts within a tenth of a second; the line needs a second at most.
	const auto tick = m_clock_hold_ms != 0 ? std::chrono::milliseconds(100)
	                                       : std::chrono::milliseconds(1000);
	auto& hold = Common::GuestClockHold();
	std::unique_lock lock(m_mutex);
	while (!m_stop) {
		m_wake.wait_for(lock, tick, [this] { return m_stop; });
		if (m_stop) {
			break;
		}
		const auto now     = RenderActivity::NowMs();
		const auto current = CurrentFrame();
		const bool progressed = current != frame;
		if (progressed) {
			frame         = current;
			progress_ms   = now;
			last_print_ms = 0;
		}
		const auto snapshot = RenderActivity::Read();
		if (m_clock_hold_ms != 0) {
			const bool want = Common::ShouldHoldClock(now - progress_ms, m_clock_hold_ms,
			                                          RenderActivity::BusyWithShaders(snapshot));
			const auto raw_ns   = Common::SteadyNs();
			uint64_t   ended_ns = 0;
			const auto change   = hold.Update(raw_ns, want, &ended_ns);
			if (change == Common::ClockHold::Change::Started) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "[clock-hold] started: no frame for {:.1f} s, render thread in '{}'; the guest "
				    "clock stands still ({} holds, {:.1f} s held before)\n",
				    static_cast<double>(now - progress_ms) / 1000.0, snapshot.phase,
				    hold.Read(raw_ns).holds,
				    static_cast<double>(hold.Read(raw_ns).completed_ns) / 1e9));
			} else if (change == Common::ClockHold::Change::Ended) {
				const auto state = hold.Read(raw_ns);
				Log::WriteToConsoleAndLog(fmt::format(
				    "[clock-hold] ended after {:.1f} s ({}); total held {:.1f} s in {} holds; the "
				    "guest clock continues from where it stopped; guest time reads that returned "
				    "an unchanged value (zero deltas) so far: {}\n",
				    static_cast<double>(ended_ns) / 1e9,
				    progressed ? "a frame was recorded"
				        : "the render thread is no longer busy with shaders",
				    static_cast<double>(state.completed_ns) / 1e9, state.holds,
				    hold.RepeatedReads()));
			}
		}
		if (m_report_parked && Common::GuestPark::Global().Count() != 0 &&
		    (last_park_ms == 0 || now - last_park_ms >= 10000)) {
			last_park_ms = now;
			Log::WriteToConsoleAndLog(fmt::format("[park] {} guest threads parked\n",
			                                      Common::GuestPark::Global().Count()));
		}
		if (m_seconds == 0 || progressed) {
			continue;
		}
		if (now - progress_ms < period_ms || (last_print_ms != 0 && now - last_print_ms < period_ms)) {
			continue;
		}
		last_print_ms = now;
		StallExtras extras;
		extras.clock_hold_enabled = m_clock_hold_ms != 0;
		const auto state          = hold.Read(Common::SteadyNs());
		extras.clock_held         = state.holding;
		extras.clock_current_ms   = state.current_ns / 1000000u;
		extras.clock_total_ms     = state.TotalNs() / 1000000u;
		extras.clock_holds        = state.holds;
		extras.parked_threads     = Common::GuestPark::Global().Count();
		const auto text = Describe(now - progress_ms, current, now, snapshot,
		                           Detail::g_shader_worker_jobs.load(std::memory_order_relaxed), extras);
		Log::WriteToConsoleAndLog(text + "\n");
		// Where each command queue is and what a waiting one waits for.
		for (const auto& line: WaitDiag::Global().Report(now)) {
			Log::WriteToConsoleAndLog(line + "\n");
		}
	}
}

} // namespace Libs::Graphics
