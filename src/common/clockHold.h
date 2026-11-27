#ifndef KYTY_COMMON_CLOCK_HOLD_H_
#define KYTY_COMMON_CLOCK_HOLD_H_

// A guest clock that can stand still for a while and never jumps forward afterwards.
//
// The guest reads time as `raw - HeldNs(raw_ns)`: while a hold is active the subtracted amount grows
// as fast as the raw clock, so the guest value stays where it was; when the hold ends the amount
// stays, so the guest clock continues from the frozen value. The held time is never given back.
// One thread (the watcher) calls Update(); any thread may read. Everything is lock free so a guest
// time call costs a few loads, and with no hold ever started the value is returned untouched.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace Common {

class ClockHold {
public:
	enum class Change { None, Started, Ended };
	enum class Domain : uint32_t { Tsc = 0, Monotonic = 1, Realtime = 2, Count = 3 };

	struct State {
		bool     holding;
		uint64_t completed_ns; // total of finished holds
		uint64_t current_ns;   // length of the running hold at the asked time
		uint32_t holds;        // started so far
		[[nodiscard]] uint64_t TotalNs() const { return completed_ns + current_ns; }
	};

	// Single writer. `raw_ns` is the raw monotonic time now. Returns what changed; on Ended,
	// `*ended_ns` (if given) receives the length of the hold that just finished.
	Change Update(uint64_t raw_ns, bool want_hold, uint64_t* ended_ns = nullptr) {
		const auto start = m_start.load(std::memory_order_relaxed);
		if (want_hold && start == kNone) {
			Write(m_completed.load(std::memory_order_relaxed), raw_ns,
			      m_holds.load(std::memory_order_relaxed) + 1u);
			m_ever.store(true, std::memory_order_relaxed);
			return Change::Started;
		}
		if (!want_hold && start != kNone) {
			const uint64_t length = raw_ns > start ? raw_ns - start : 0;
			Write(m_completed.load(std::memory_order_relaxed) + length, kNone,
			      m_holds.load(std::memory_order_relaxed));
			if (ended_ns != nullptr) {
				*ended_ns = length;
			}
			return Change::Ended;
		}
		return Change::None;
	}

	[[nodiscard]] State Read(uint64_t raw_ns) const {
		uint64_t completed = 0;
		uint64_t start     = kNone;
		uint32_t holds     = 0;
		for (;;) {
			const auto before = m_seq.load(std::memory_order_acquire);
			if ((before & 1u) != 0) {
				continue;
			}
			completed = m_completed.load(std::memory_order_relaxed);
			start     = m_start.load(std::memory_order_relaxed);
			holds     = m_holds.load(std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_acquire);
			if (m_seq.load(std::memory_order_relaxed) == before) {
				break;
			}
		}
		State state {};
		state.holding      = start != kNone;
		state.completed_ns = completed;
		state.current_ns   = (start != kNone && raw_ns > start) ? raw_ns - start : 0;
		state.holds        = holds;
		return state;
	}

	// Nanoseconds the guest clock lags behind the raw clock at raw time `raw_ns`.
	[[nodiscard]] uint64_t HeldNs(uint64_t raw_ns) const {
		return m_ever.load(std::memory_order_relaxed) ? Read(raw_ns).TotalNs() : 0;
	}

	[[nodiscard]] bool Ever() const { return m_ever.load(std::memory_order_relaxed); }

	// The guest value of a raw reading in `domain`; `held` is HeldNs converted to the domain's
	// unit. Never lower than a value already returned for that domain.
	uint64_t Apply(Domain domain, uint64_t raw_value, uint64_t held) {
		if (!m_ever.load(std::memory_order_relaxed)) {
			return raw_value;
		}
		const uint64_t value = raw_value > held ? raw_value - held : 0;
		auto&          last  = m_last[static_cast<size_t>(domain)];
		auto           seen  = last.load(std::memory_order_relaxed);
		while (seen < value &&
		       !last.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
		}
		if (seen >= value) {
			// The guest gets a value it already had: a delta of 0 between two of its reads.
			m_repeats.fetch_add(1, std::memory_order_relaxed);
		}
		return seen > value ? seen : value;
	}

	// Readings (all domains) that returned a value already handed out in their domain. Each is a
	// zero time delta for the guest; a guest dividing by its frame time gets an infinity from it.
	[[nodiscard]] uint64_t RepeatedReads() const { return m_repeats.load(std::memory_order_relaxed); }

private:
	static constexpr uint64_t kNone = UINT64_MAX;

	void Write(uint64_t completed, uint64_t start, uint32_t holds) {
		const auto s = m_seq.load(std::memory_order_relaxed);
		m_seq.store(s + 1u, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		m_completed.store(completed, std::memory_order_relaxed);
		m_start.store(start, std::memory_order_relaxed);
		m_holds.store(holds, std::memory_order_relaxed);
		m_seq.store(s + 2u, std::memory_order_release);
	}

	std::atomic<uint32_t> m_seq {0};
	std::atomic<uint64_t> m_completed {0};
	std::atomic<uint64_t> m_start {kNone};
	std::atomic<uint32_t> m_holds {0};
	std::atomic<bool>     m_ever {false};
	std::atomic<uint64_t> m_last[static_cast<size_t>(Domain::Count)] {};
	std::atomic<uint64_t> m_repeats {0};
};

// The process wide guest clock hold.
inline ClockHold& GuestClockHold() {
	static ClockHold hold;
	return hold;
}

// Held nanoseconds expressed in counter ticks of `frequency` Hz.
[[nodiscard]] inline uint64_t HeldNsToTicks(uint64_t held_ns, uint64_t frequency) {
	return static_cast<uint64_t>((static_cast<unsigned __int128>(held_ns) * frequency) /
	                             1000000000u);
}

// Whether the clock should be held: no frame for longer than `threshold_ms` while the render side
// is busy with shader work or has queues suspended for it. 0 threshold = never.
[[nodiscard]] inline bool ShouldHoldClock(uint64_t stalled_ms, uint64_t threshold_ms,
                                          bool render_busy) {
	return threshold_ms != 0 && stalled_ms > threshold_ms && render_busy;
}

} // namespace Common

#endif /* KYTY_COMMON_CLOCK_HOLD_H_ */
