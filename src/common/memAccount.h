#ifndef KYTY_COMMON_MEM_ACCOUNT_H_
#define KYTY_COMMON_MEM_ACCOUNT_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Memory accounting of the process: owners that can count what they hold publish it in a Gauge, and
// one thread prints a `[mem]` line every 10 s and at exit: process RSS, swap, the owners' sum and
// what is left over (`unaccounted`, which is native heap that nobody counts, freed memory the
// allocator still holds, guest memory, driver memory). With a limit set (--mem-limit-gb) the same
// thread drops what can be dropped when RSS + swap passes it and stops the run if that does not help.
namespace Common::MemAccount {

// What one owner holds. Written by the owner at any time from any thread, read by the reporter.
class Gauge {
public:
	void Set(uint64_t bytes, uint64_t items = 0) {
		m_bytes.store(bytes, std::memory_order_relaxed);
		m_items.store(items, std::memory_order_relaxed);
	}
	// Adds or, when negative, removes (saturating at 0).
	void Add(int64_t bytes, int64_t items = 0) {
		Apply(m_bytes, bytes);
		Apply(m_items, items);
	}
	[[nodiscard]] uint64_t Bytes() const { return m_bytes.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Items() const { return m_items.load(std::memory_order_relaxed); }

private:
	static void Apply(std::atomic<uint64_t>& value, int64_t delta) {
		auto current = value.load(std::memory_order_relaxed);
		for (;;) {
			const auto next = delta >= 0 ? current + static_cast<uint64_t>(delta)
			                  : current > static_cast<uint64_t>(-delta)
			                      ? current - static_cast<uint64_t>(-delta)
			                      : uint64_t {0};
			if (value.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
				return;
			}
		}
	}
	std::atomic<uint64_t> m_bytes {0};
	std::atomic<uint64_t> m_items {0};
};

// The gauge of the named owner; created on first use and valid for the rest of the process.
// Owners print in the order they were first named.
Gauge& Owner(std::string_view name);

struct Entry {
	std::string name;
	uint64_t    bytes = 0;
	uint64_t    items = 0;
};
[[nodiscard]] std::vector<Entry> Snapshot();

// A figure only its owner can read (the Vulkan allocator): text such as `device=812MiB`, called on
// the reporting thread. Not part of the owners' sum (it is mostly not process memory).
using ExternalReader = std::function<std::string()>;
void SetExternal(std::string_view name, ExternalReader reader);
void ClearExternal(std::string_view name);

struct ProcessMemory {
	uint64_t rss      = 0;
	uint64_t swap     = 0;
	uint64_t peak_rss = 0; // 0 when the platform does not say
};
// The text of /proc/<pid>/status (kB fields) -> bytes.
[[nodiscard]] ProcessMemory ParseProcStatus(std::string_view text);
[[nodiscard]] ProcessMemory ReadProcessMemory();

struct HeapStats {
	bool     known    = false;
	uint64_t in_use   = 0; // malloc'd and not freed, mmapped chunks included
	uint64_t free_held = 0; // freed but still held by the allocator (arenas, fragmentation)
};
[[nodiscard]] HeapStats ReadHeapStats();

struct Sample {
	ProcessMemory                                    process;
	HeapStats                                        heap;
	std::vector<Entry>                               owners;
	std::vector<std::pair<std::string, std::string>> external;
};
[[nodiscard]] uint64_t Counted(const std::vector<Entry>& owners);
// RSS minus the owners' sum; negative when the owners' estimates are larger than the process.
[[nodiscard]] int64_t Unaccounted(uint64_t rss, uint64_t counted);
// `[mem] rss=.. swap=.. peak-rss=.. counted=.. unaccounted=.. | owner=..(n) ... | heap-in-use=.. ...`
[[nodiscard]] std::string FormatLine(const Sample& sample, std::string_view tag);

// The decision of --mem-limit-gb, from a sample every second. Over the limit: the first time, drop
// what can be dropped; still over `grace_ms` later, stop. Back under: starts over.
class LimitGuard {
public:
	enum class Action : uint8_t { None, DropCaches, Stop };
	explicit LimitGuard(uint64_t limit_bytes, uint64_t grace_ms = 10000)
	    : m_limit(limit_bytes), m_grace_ms(grace_ms) {}
	[[nodiscard]] Action Step(uint64_t now_ms, uint64_t rss_plus_swap) {
		if (m_limit == 0 || rss_plus_swap <= m_limit) {
			m_over = false;
			return Action::None;
		}
		if (!m_over) {
			m_over       = true;
			m_over_since = now_ms;
			return Action::DropCaches;
		}
		return now_ms - m_over_since >= m_grace_ms ? Action::Stop : Action::None;
	}

private:
	uint64_t m_limit;
	uint64_t m_grace_ms;
	bool     m_over       = false;
	uint64_t m_over_since = 0;
};

// Set by the guard: the owners of droppable caches free them on their own thread when they see it.
[[nodiscard]] bool TakeDropRequest();

// The line now (tag "now"), for a place that wants a figure at a precise moment.
void PrintNow(std::string_view tag);
void Start();
void Stop(); // prints the line `exit`
void EmergencyPrint();

struct Lifecycle {
	static constexpr const char* name               = "MemAccount";
	static constexpr auto        initialize         = Start;
	static constexpr auto        shutdown           = Stop;
	static constexpr auto        emergency_shutdown = EmergencyPrint;
};

} // namespace Common::MemAccount

#endif /* KYTY_COMMON_MEM_ACCOUNT_H_ */
