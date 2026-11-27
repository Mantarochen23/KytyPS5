#include "common/memAccount.h"

#include "common/emulatorConfig.h"
#include "common/hostStats.h"
#include "common/subsystems.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace Common::MemAccount {

namespace {

constexpr uint64_t ReportIntervalMs = 10'000;

std::mutex                                 g_mutex;
std::vector<std::pair<std::string, std::unique_ptr<Gauge>>> g_owners;
std::vector<std::pair<std::string, ExternalReader>>         g_external;
std::atomic<bool>                          g_drop_requested {false};

std::mutex              g_thread_mutex;
std::condition_variable g_thread_wake;
bool                    g_stop = false;
std::thread             g_thread;
std::atomic<uint64_t>   g_peak_seen {0};
std::chrono::steady_clock::time_point g_start = std::chrono::steady_clock::now();

std::string Mib(uint64_t bytes) {
	return fmt::format("{:.0f}MiB", static_cast<double>(bytes) / 1048576.0);
}

Sample TakeSample() {
	Sample sample;
	sample.process = ReadProcessMemory();
	sample.heap    = ReadHeapStats();
	sample.owners  = Snapshot();
	std::vector<std::pair<std::string, ExternalReader>> readers;
	{
		std::scoped_lock lock(g_mutex);
		readers = g_external;
	}
	// A reader is called without the lock: it may take its owner's locks.
	for (const auto& [name, reader]: readers) {
		sample.external.emplace_back(name, reader());
	}
	return sample;
}

uint64_t ReadKb(std::string_view text, std::string_view key) {
	const auto at = text.find(key);
	if (at == std::string_view::npos) {
		return 0;
	}
	auto rest = text.substr(at + key.size());
	while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) {
		rest.remove_prefix(1);
	}
	uint64_t value = 0;
	std::from_chars(rest.data(), rest.data() + rest.size(), value);
	return value * 1024u;
}

} // namespace

Gauge& Owner(std::string_view name) {
	std::scoped_lock lock(g_mutex);
	for (auto& [key, gauge]: g_owners) {
		if (key == name) {
			return *gauge;
		}
	}
	return *g_owners.emplace_back(std::string(name), std::make_unique<Gauge>()).second;
}

std::vector<Entry> Snapshot() {
	std::vector<Entry> entries;
	std::scoped_lock   lock(g_mutex);
	entries.reserve(g_owners.size());
	for (const auto& [name, gauge]: g_owners) {
		entries.push_back({name, gauge->Bytes(), gauge->Items()});
	}
	return entries;
}

void SetExternal(std::string_view name, ExternalReader reader) {
	std::scoped_lock lock(g_mutex);
	for (auto& [key, existing]: g_external) {
		if (key == name) {
			existing = std::move(reader);
			return;
		}
	}
	g_external.emplace_back(std::string(name), std::move(reader));
}

void ClearExternal(std::string_view name) {
	std::scoped_lock lock(g_mutex);
	std::erase_if(g_external, [&](const auto& entry) { return entry.first == name; });
}

ProcessMemory ParseProcStatus(std::string_view text) {
	ProcessMemory memory;
	memory.rss      = ReadKb(text, "VmRSS:");
	memory.swap     = ReadKb(text, "VmSwap:");
	memory.peak_rss = ReadKb(text, "VmHWM:");
	return memory;
}

ProcessMemory ReadProcessMemory() {
	ProcessMemory memory;
#if defined(__linux__)
	std::ifstream file("/proc/self/status");
	if (file) {
		std::ostringstream text;
		text << file.rdbuf();
		memory = ParseProcStatus(text.str());
	}
#else
	static HostStatsSampler sampler;
	const auto              stats = sampler.Sample();
	memory.rss                    = stats.process_resident.value_or(0);
	memory.swap                   = stats.process_compressed.value_or(0);
#endif
	auto seen = g_peak_seen.load(std::memory_order_relaxed);
	while (memory.rss > seen && !g_peak_seen.compare_exchange_weak(seen, memory.rss)) {
	}
	memory.peak_rss = std::max(memory.peak_rss, g_peak_seen.load(std::memory_order_relaxed));
	return memory;
}

HeapStats ReadHeapStats() {
	HeapStats stats;
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 33)
	const auto info = mallinfo2();
	stats.known     = true;
	stats.in_use    = static_cast<uint64_t>(info.uordblks) + static_cast<uint64_t>(info.hblkhd);
	stats.free_held = static_cast<uint64_t>(info.fordblks);
#endif
#endif
	return stats;
}

uint64_t Counted(const std::vector<Entry>& owners) {
	uint64_t sum = 0;
	for (const auto& owner: owners) {
		sum += owner.bytes;
	}
	return sum;
}

int64_t Unaccounted(uint64_t rss, uint64_t counted) {
	return rss >= counted ? static_cast<int64_t>(rss - counted)
	                      : -static_cast<int64_t>(counted - rss);
}

std::string FormatLine(const Sample& sample, std::string_view tag) {
	const auto counted = Counted(sample.owners);
	const auto gap     = Unaccounted(sample.process.rss, counted);
	std::string line   = fmt::format(
        "[mem] {} t={:.0f}s rss={} swap={} total={} peak-rss={} counted={} unaccounted={}{}", tag,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count(),
        Mib(sample.process.rss), Mib(sample.process.swap),
        Mib(sample.process.rss + sample.process.swap), Mib(sample.process.peak_rss), Mib(counted),
        gap < 0 ? "-" : "", Mib(static_cast<uint64_t>(gap < 0 ? -gap : gap)));
	line += " |";
	for (const auto& owner: sample.owners) {
		line += fmt::format(" {}={}", owner.name, Mib(owner.bytes));
		if (owner.items != 0) {
			line += fmt::format("({})", owner.items);
		}
	}
	if (sample.heap.known) {
		line += fmt::format(" | heap-in-use={} heap-free-held={}", Mib(sample.heap.in_use),
		                    Mib(sample.heap.free_held));
	}
	for (const auto& [name, text]: sample.external) {
		line += fmt::format(" | {}: {}", name, text);
	}
	return line;
}

bool TakeDropRequest() {
	return g_drop_requested.exchange(false, std::memory_order_relaxed);
}

void PrintNow(std::string_view tag) {
	std::printf("%s\n", FormatLine(TakeSample(), tag).c_str());
	std::fflush(stdout);
}

namespace {

void Run() {
	const uint64_t limit = static_cast<uint64_t>(Config::MemLimitGb()) << 30;
	LimitGuard     guard(limit);
	const auto     begin = std::chrono::steady_clock::now();
	uint64_t       last_report_ms = 0;
	std::unique_lock lock(g_thread_mutex);
	while (!g_stop) {
		g_thread_wake.wait_for(lock, std::chrono::seconds(1), [] { return g_stop; });
		if (g_stop) {
			break;
		}
		const auto now_ms = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::milliseconds>(
		        std::chrono::steady_clock::now() - begin)
		        .count());
		lock.unlock();
		if (now_ms - last_report_ms >= ReportIntervalMs) {
			last_report_ms = now_ms;
			PrintNow("");
		}
		if (limit != 0) {
			const auto process = ReadProcessMemory();
			switch (guard.Step(now_ms, process.rss + process.swap)) {
				case LimitGuard::Action::None: break;
				case LimitGuard::Action::DropCaches:
					PrintNow("limit");
					std::printf("[mem] --mem-limit-gb %u passed (rss+swap %s): dropping caches; "
					            "the run is stopped if it is still over in 10 s\n",
					            Config::MemLimitGb(), Mib(process.rss + process.swap).c_str());
					g_drop_requested.store(true, std::memory_order_relaxed);
#if defined(__GLIBC__)
					malloc_trim(0);
#endif
					break;
				case LimitGuard::Action::Stop:
					PrintNow("limit-stop");
					std::printf("[mem] STOPPING: rss+swap %s is still over --mem-limit-gb %u 10 s "
					            "after the caches were dropped. Stopped by the emulator, not by the "
					            "kernel; read the table above.\n",
					            Mib(process.rss + process.swap).c_str(), Config::MemLimitGb());
					std::fflush(nullptr);
					Subsystems::EmergencyShutdownActive();
					std::fflush(nullptr);
					std::_Exit(75);
			}
		}
		lock.lock();
	}
}

} // namespace

void Start() {
	std::scoped_lock lock(g_thread_mutex);
	if (g_thread.joinable()) {
		return;
	}
	g_stop   = false;
	g_thread = std::thread(Run);
}

void Stop() {
	{
		std::scoped_lock lock(g_thread_mutex);
		g_stop = true;
	}
	g_thread_wake.notify_all();
	if (g_thread.joinable()) {
		g_thread.join();
	}
	PrintNow("exit");
}

void EmergencyPrint() {
	// Not from the reporting thread's join: it may be the thread that called this.
	PrintNow("emergency");
}

} // namespace Common::MemAccount
