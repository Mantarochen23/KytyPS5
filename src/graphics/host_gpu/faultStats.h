#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTSTATS_H_

// Counters for the write-watch (page fault) path and the tuning of the fault-storm rule.
// Header only, so that every target that links the tracker can use it.
//
// The `[fault]` line is printed every 5 s by a thread of its own (never from the fault handler, which
// runs in a signal). An interval with no faults and no protection changes prints nothing.

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <utility>

namespace Libs::Graphics::FaultStats {

// Who took (or waited for) a tracking lock.
enum class LockSite : uint8_t { Other, Fault, Upload, Download, Mark, Unmark, Texture, Count };

inline const char* LockSiteName(LockSite site) {
	switch (site) {
		case LockSite::Fault: return "fault";
		case LockSite::Upload: return "upload";
		case LockSite::Download: return "download";
		case LockSite::Mark: return "mark";
		case LockSite::Unmark: return "unmark";
		case LockSite::Texture: return "texture";
		default: return "other";
	}
}

constexpr size_t kLockSites = static_cast<size_t>(LockSite::Count);

struct Counters {
	std::atomic<uint64_t> write_faults {0};
	std::atomic<uint64_t> read_faults {0};
	std::atomic<uint64_t> handler_ns {0};
	std::atomic<uint64_t> invalidate_ns {0};
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> protect_pages {0};   // pages armed (write- or read-watched)
	std::atomic<uint64_t> unprotect_pages {0}; // pages released
	std::atomic<uint64_t> protect_ns {0};
	std::atomic<uint64_t> fast_skips {0};      // faults on pages already CPU-dirty: no lock taken
	std::atomic<uint64_t> storm_marked {0};    // pages that became always-dirty
	std::atomic<uint64_t> storm_kept {0};      // page re-arms the rule skipped
	std::atomic<uint64_t> storm_expired {0};   // always-dirty pages that were re-armed
	std::array<std::atomic<uint64_t>, kLockSites> lock_wait_ns {};
	std::array<std::atomic<uint64_t>, kLockSites> lock_waits {};
};

inline Counters g_counters;

// The rule: a page whose write watch is re-armed and faults again `faults` times within `frames`
// frames stays unwatched, and counts as modified, for `hold` frames. faults == 0 = off.
struct StormTuning {
	std::atomic<uint32_t> faults {0};
	std::atomic<uint32_t> frames {8};
	std::atomic<uint32_t> hold {60};
	std::atomic<uint32_t> frame {0}; // the clock of the rule; set by the owner of the frame count
};

inline StormTuning g_storm;

inline void ConfigureStorm(uint32_t faults, uint32_t frames, uint32_t hold) {
	g_storm.frames.store(frames == 0 ? 1 : (frames > 30000 ? 30000 : frames));
	g_storm.hold.store(hold > 30000 ? 30000 : hold);
	g_storm.faults.store(faults > 250 ? 250 : faults);
}

inline void SetFrame(uint32_t frame) {
	g_storm.frame.store(frame, std::memory_order_relaxed);
}

inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(
	        std::chrono::steady_clock::now().time_since_epoch())
	        .count());
}

inline LockSite& CurrentSite() {
	static thread_local LockSite site = LockSite::Other;
	return site;
}

// Names the caller for the lock waits taken while it is alive.
class SiteScope final {
public:
	explicit SiteScope(LockSite site): m_previous(CurrentSite()) { CurrentSite() = site; }
	~SiteScope() { CurrentSite() = m_previous; }
	SiteScope(const SiteScope&)            = delete;
	SiteScope& operator=(const SiteScope&) = delete;

private:
	LockSite m_previous;
};

inline void NoteLockWait(uint64_t ns) {
	const auto site = static_cast<size_t>(CurrentSite());
	g_counters.lock_wait_ns[site].fetch_add(ns, std::memory_order_relaxed);
	g_counters.lock_waits[site].fetch_add(1, std::memory_order_relaxed);
}

// Faulting 64 KiB blocks of the current interval, counted in a small open-addressed table.
class HotBlocks final {
public:
	static constexpr size_t kSlots  = 4096;
	static constexpr size_t kProbes = 8;
	static constexpr size_t kTop    = 8;

	struct Entry {
		uint64_t block = 0; // address >> 16
		uint64_t count = 0;
	};

	void Note(uint64_t address) noexcept {
		const uint64_t key  = (address >> 16) + 1;
		auto           slot = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 52) % kSlots;
		for (size_t probe = 0; probe < kProbes; probe++, slot = (slot + 1) % kSlots) {
			uint64_t current = m_keys[slot].load(std::memory_order_relaxed);
			if (current == 0 &&
			    m_keys[slot].compare_exchange_strong(current, key, std::memory_order_relaxed)) {
				current = key;
			}
			if (current == key) {
				m_counts[slot].fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
		m_dropped.fetch_add(1, std::memory_order_relaxed);
	}

	// The most faulting blocks, descending; clears the table.
	size_t TakeTop(std::array<Entry, kTop>& top, uint64_t& dropped) noexcept {
		size_t used = 0;
		for (size_t slot = 0; slot < kSlots; slot++) {
			const auto key = m_keys[slot].exchange(0, std::memory_order_relaxed);
			const auto cnt = m_counts[slot].exchange(0, std::memory_order_relaxed);
			if (key == 0 || cnt == 0) {
				continue;
			}
			Entry entry {key - 1, cnt};
			size_t at = used < kTop ? used : kTop;
			if (used < kTop) {
				used++;
			} else if (top[kTop - 1].count >= cnt) {
				continue;
			} else {
				at = kTop - 1;
			}
			top[at] = entry;
			for (; at > 0 && top[at - 1].count < top[at].count; at--) {
				std::swap(top[at - 1], top[at]);
			}
		}
		dropped = m_dropped.exchange(0, std::memory_order_relaxed);
		return used;
	}

private:
	std::array<std::atomic<uint64_t>, kSlots> m_keys {};
	std::array<std::atomic<uint64_t>, kSlots> m_counts {};
	std::atomic<uint64_t>                     m_dropped {0};
};

inline HotBlocks g_hot_blocks;

inline std::atomic<uint64_t> g_last_report_ns {0};

// Prints the `[fault]` line for the time since the last call. `now` is a NowNs() value.
inline void Report(uint64_t now) {
	auto last = g_last_report_ns.exchange(now, std::memory_order_relaxed);
	if (last == 0 || now <= last) {
		return;
	}
	if (g_counters.write_faults.load(std::memory_order_relaxed) +
	        g_counters.read_faults.load(std::memory_order_relaxed) +
	        g_counters.protect_calls.load(std::memory_order_relaxed) ==
	    0) {
		return;
	}
	const double secs = static_cast<double>(now - last) / 1e9;
	auto&        c    = g_counters;
	const auto   take = [](std::atomic<uint64_t>& value) {
        return value.exchange(0, std::memory_order_relaxed);
	};
	const auto per_s = [secs](uint64_t value) { return static_cast<double>(value) / secs; };
	const auto ms_s  = [secs](uint64_t ns) { return static_cast<double>(ns) / 1e6 / secs; };

	const auto wr = take(c.write_faults), rd = take(c.read_faults);
	const auto handler = take(c.handler_ns), inval = take(c.invalidate_ns);
	const auto calls = take(c.protect_calls), armed = take(c.protect_pages),
	           released = take(c.unprotect_pages), prot_ns = take(c.protect_ns);
	const auto fast = take(c.fast_skips), marked = take(c.storm_marked),
	           kept = take(c.storm_kept), expired = take(c.storm_expired);
	std::printf("[fault] %.1fs write=%.0f/s read=%.0f/s armed=%.0f pg/s released=%.0f pg/s "
	            "protect-calls=%.0f/s protect=%.1f ms/s handler=%.1f ms/s invalidate=%.1f ms/s "
	            "fast-skip=%.0f/s storm(on=%u) marked=%" PRIu64 " kept=%.0f pg/s expired=%" PRIu64
	            "\n",
	            secs, per_s(wr), per_s(rd), per_s(armed), per_s(released), per_s(calls),
	            ms_s(prot_ns), ms_s(handler), ms_s(inval), per_s(fast),
	            g_storm.faults.load(std::memory_order_relaxed), marked, per_s(kept), expired);
	std::printf("[fault] lock-wait ms/s (waits/s):");
	for (size_t site = 0; site < kLockSites; site++) {
		const auto ns    = take(c.lock_wait_ns[site]);
		const auto waits = take(c.lock_waits[site]);
		std::printf(" %s=%.1f(%.0f)", LockSiteName(static_cast<LockSite>(site)), ms_s(ns),
		            per_s(waits));
	}
	std::printf("\n");
	std::array<HotBlocks::Entry, HotBlocks::kTop> top {};
	uint64_t                                      dropped = 0;
	const auto                                    used    = g_hot_blocks.TakeTop(top, dropped);
	std::printf("[fault] top 64K blocks:");
	for (size_t i = 0; i < used; i++) {
		std::printf(" 0x%" PRIx64 "=%" PRIu64, top[i].block << 16, top[i].count);
	}
	if (dropped != 0) {
		std::printf(" (untracked %" PRIu64 ")", dropped);
	}
	std::printf("\n");
	std::fflush(stdout);
}

// Starts the reporter once.
inline void StartReporter() {
	static std::atomic<bool> started {false};
	if (started.exchange(true)) {
		return;
	}
	g_last_report_ns.store(NowNs(), std::memory_order_relaxed);
	std::thread([] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(5));
			Report(NowNs());
		}
	}).detach();
}

} // namespace Libs::Graphics::FaultStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTSTATS_H_
