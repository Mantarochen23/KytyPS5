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
	// RSS split (Linux): anonymous (heap, driver), file-backed (mapped files, binaries) and shared
	// memory (the guest memory backing is a memfd, so its touched pages count here).
	uint64_t rss_anon  = 0;
	uint64_t rss_file  = 0;
	uint64_t rss_shmem = 0;
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

// One malloc arena from malloc_info: bytes freed but kept, and bytes the arena took from the system.
struct Arena {
	uint32_t nr           = 0;
	uint64_t free_bytes   = 0;
	uint64_t system_bytes = 0;
};
// The XML of malloc_info(3) -> its arenas (the trailing process-wide totals are not arenas).
[[nodiscard]] std::vector<Arena> ParseMallocInfo(std::string_view xml);
[[nodiscard]] std::vector<Arena> ReadArenas();

// zram swap: what the swapped pages really cost. `orig` is the data it holds, `compressed` the RAM
// it uses for it (mem_used_total of /sys/block/zram*/mm_stat, summed).
struct ZramStats {
	bool     known      = false;
	uint64_t orig       = 0;
	uint64_t compressed = 0;
};
// One mm_stat file: orig_data_size compr_data_size mem_used_total ...
[[nodiscard]] ZramStats ParseZramMmStat(std::string_view text);
[[nodiscard]] ZramStats ReadZram();
// RSS plus the swap at its real RAM cost: on a zram host swapped pages stay in RAM compressed.
[[nodiscard]] uint64_t Footprint(const ProcessMemory& process, const ZramStats& zram);

// The guest memory backing (a memfd on Linux) reports its committed size here once.
void NoteGuestBacking(uint64_t bytes);

struct Sample {
	ProcessMemory                                    process;
	HeapStats                                        heap;
	std::vector<Entry>                               owners;
	std::vector<std::pair<std::string, std::string>> external;
	ZramStats                                        zram;
	std::vector<Arena>                               arenas;
	uint64_t                                         guest_committed = 0;
};

// Every number that can grow, by name, from one sample (owners, heap, RSS parts, swap).
[[nodiscard]] std::vector<std::pair<std::string, uint64_t>> Holders(const Sample& sample);
// `heap-in-use +310MiB, rss-shmem +120MiB`: the holders that grew most between two samples.
[[nodiscard]] std::string FormatGrowth(const std::vector<std::pair<std::string, uint64_t>>& before,
                                       const std::vector<std::pair<std::string, uint64_t>>& after,
                                       size_t top = 4);

// Heap tuning of --heap-tune (arena cap, trim thresholds, mmap threshold); called once at start.
void ApplyHeapTune();
// Returns freed heap memory to the system at the end of a phase (store load, build burst) and prints
// one `[mem] trim` line with what came back. Does nothing unless --heap-tune is on.
void TrimHeap(std::string_view phase);
// The one `[exit]` line: what the process held when it ended and what is torn down. `reason` names
// the path (normal, fatal, signal, limit).
void PrintExit(std::string_view reason);
// Names the path the process is leaving by, for the `[exit]` line printed by the emergency path
// (fatal assert, quick exit, limit stop, SIGTERM). `reason` must be a string literal.
void NoteExitReason(const char* reason);
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
