#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RENDERACTIVITY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RENDERACTIVITY_H_

#include "common/clockHold.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace Libs::Graphics {

// What the render (command processor) thread is doing, for the stall reporter. The thread notes a
// phase at the places it can spend a long time; another thread reads it when no frame was
// recorded for a while. The fields are separate relaxed atomics: a report may mix two phases, which
// is fine for a diagnostic line. Phase and detail are string literals.
namespace RenderActivity {

// Milliseconds on the steady clock's epoch. Noted at every change of phase, which is thousands of
// times a second: on Linux the coarse clock (a few ms resolution, no system call, a fraction of the
// cost) is enough for a stall report that counts seconds.
[[nodiscard]] inline uint64_t NowMs() noexcept {
#if defined(__linux__) && defined(CLOCK_MONOTONIC_COARSE)
	timespec now {};
	if (::clock_gettime(CLOCK_MONOTONIC_COARSE, &now) == 0) {
		return static_cast<uint64_t>(now.tv_sec) * 1000u + static_cast<uint64_t>(now.tv_nsec) / 1000000u;
	}
#endif
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

namespace Detail {
inline std::atomic<const char*> g_phase {"start"};
inline std::atomic<const char*> g_detail {""};
inline std::atomic<uint64_t>    g_hash {0};
inline std::atomic<const char*> g_stage {"-"};
// When the current phase, hash and detail began; kept while a retry notes the same ones again.
inline std::atomic<uint64_t>    g_since_ms {0};
// The PM4 opcode being processed and the number of packets processed so far.
inline std::atomic<uint32_t>    g_opcode {0};
inline std::atomic<uint64_t>    g_packets {0};
} // namespace Detail

// Notes what the thread is doing now. Calling it again with the same phase, hash and detail keeps
// the start time, so a dispatch retried every 50 ms shows how long it has waited in all.
inline void Note(const char* phase, uint64_t hash = 0, const char* stage = "-",
                 const char* detail = "") noexcept {
	using namespace Detail;
	if (g_phase.load(std::memory_order_relaxed) != phase ||
	    g_hash.load(std::memory_order_relaxed) != hash ||
	    g_detail.load(std::memory_order_relaxed) != detail) {
		g_since_ms.store(NowMs(), std::memory_order_relaxed);
	}
	g_phase.store(phase, std::memory_order_relaxed);
	g_detail.store(detail, std::memory_order_relaxed);
	g_hash.store(hash, std::memory_order_relaxed);
	g_stage.store(stage, std::memory_order_relaxed);
}

// Changes only the detail (for example the state of the job being waited for).
inline void NoteDetail(const char* detail) noexcept {
	using namespace Detail;
	if (g_detail.load(std::memory_order_relaxed) != detail) {
		g_since_ms.store(NowMs(), std::memory_order_relaxed);
		g_detail.store(detail, std::memory_order_relaxed);
	}
}

inline void NotePacket(uint32_t opcode) noexcept {
	Detail::g_opcode.store(opcode, std::memory_order_relaxed);
	Detail::g_packets.fetch_add(1, std::memory_order_relaxed);
}

// Notes a phase for a scope and returns to the previous one after it.
class Scope {
public:
	Scope(const char* phase, uint64_t hash = 0, const char* stage = "-",
	      const char* detail = "") noexcept
	    : m_phase(Detail::g_phase.load(std::memory_order_relaxed)),
	      m_detail(Detail::g_detail.load(std::memory_order_relaxed)),
	      m_hash(Detail::g_hash.load(std::memory_order_relaxed)),
	      m_stage(Detail::g_stage.load(std::memory_order_relaxed)) {
		Note(phase, hash, stage, detail);
	}
	~Scope() { Note(m_phase, m_hash, m_stage, m_detail); }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	const char* m_phase;
	const char* m_detail;
	uint64_t    m_hash;
	const char* m_stage;
};

struct Snapshot {
	const char* phase;
	const char* detail;
	uint64_t    hash;
	const char* stage;
	uint64_t    since_ms;
	uint32_t    opcode;
	uint64_t    packets;
};

[[nodiscard]] inline Snapshot Read() noexcept {
	using namespace Detail;
	return {g_phase.load(std::memory_order_relaxed),  g_detail.load(std::memory_order_relaxed),
	        g_hash.load(std::memory_order_relaxed),   g_stage.load(std::memory_order_relaxed),
	        g_since_ms.load(std::memory_order_relaxed), g_opcode.load(std::memory_order_relaxed),
	        g_packets.load(std::memory_order_relaxed)};
}

// Whether the render thread is busy with shader work or has every queue suspended for it: the
// states in which a long gap between frames is the emulator's doing, not the program's.
[[nodiscard]] inline bool BusyWithShaders(const Snapshot& snapshot) noexcept {
	const std::string_view phase(snapshot.phase);
	if (phase == "cp-blocked") {
		return std::string_view(snapshot.detail).starts_with("every queue waits");
	}
	for (const std::string_view busy: {"dispatch-shader", "dispatch-pipeline", "wait-pipeline",
	                                   "sync-translate", "draw-shaders", "pump-async", "wait-async",
	                                   "expire-translations", "cp-prefetch-scan",
	                                   "cp-prefetch-drain"}) {
		if (phase == busy) {
			return true;
		}
	}
	return false;
}

} // namespace RenderActivity

// What the stall line says about the guest clock and parked threads.
struct StallExtras {
	bool     clock_hold_enabled = false;
	bool     clock_held         = false;
	uint64_t clock_current_ms   = 0; // length of the running hold
	uint64_t clock_total_ms     = 0; // all holds, including the running one
	uint32_t clock_holds        = 0;
	uint32_t parked_threads     = 0;
};

// Starts a thread that prints one line when no frame was recorded for `seconds`, and again every
// `seconds` while it stays so. Idle cost: one wake-up per second. 0 starts nothing.
class RenderStallReporter {
public:
	// `seconds`: stall line period (0 = no line). `clock_hold_ms`: --guest-clock-hold-on-stall (0 =
	// off). `report_parked`: --guest-fault-park is on, so parked threads are listed.
	explicit RenderStallReporter(uint32_t seconds, uint32_t clock_hold_ms = 0,
	                             bool report_parked = false);
	~RenderStallReporter();
	RenderStallReporter(const RenderStallReporter&)            = delete;
	RenderStallReporter& operator=(const RenderStallReporter&) = delete;

	// The line for a stall of `stalled_ms`, from the activity read now. Public for tests.
	static std::string Describe(uint64_t stalled_ms, uint64_t frame, uint64_t now_ms,
	                            const RenderActivity::Snapshot& snapshot, uint32_t jobs_pending,
	                            const StallExtras& extras = {});

private:
	void Run();

	uint32_t                m_seconds;
	uint32_t                m_clock_hold_ms;
	bool                    m_report_parked;
	std::mutex              m_mutex;
	std::condition_variable m_wake;
	bool                    m_stop = false;
	std::thread             m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RENDERACTIVITY_H_
