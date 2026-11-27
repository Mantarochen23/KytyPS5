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
#if defined(__linux__)
#include <csignal>
#include <filesystem>
#include <unistd.h>
#endif

namespace Common::MemAccount {

namespace {

constexpr uint64_t ReportIntervalMs = 10'000;

// Never destroyed: owners count down from destructors that run during static destruction.
std::mutex& Mutex() {
	static auto* mutex = new std::mutex;
	return *mutex;
}
std::vector<std::pair<std::string, std::unique_ptr<Gauge>>>& Owners() {
	static auto* owners = new std::vector<std::pair<std::string, std::unique_ptr<Gauge>>>;
	return *owners;
}
// Held while the external readers run, so that a reader's owner can take it away (ClearExternal)
// before it is destroyed.
std::mutex& ReadMutex() {
	static auto* mutex = new std::mutex;
	return *mutex;
}
std::vector<std::pair<std::string, ExternalReader>>& Externals() {
	static auto* externals = new std::vector<std::pair<std::string, ExternalReader>>;
	return *externals;
}
std::atomic<bool>                          g_drop_requested {false};

std::mutex              g_thread_mutex;
std::condition_variable g_thread_wake;
bool                    g_stop = false;
std::thread             g_thread;
std::atomic<uint64_t>   g_peak_seen {0};
std::atomic<uint64_t>   g_guest_committed {0};
std::atomic<uint64_t>   g_trimmed_total {0};
std::atomic<bool>       g_exit_printed {false};
std::atomic<const char*> g_exit_reason {nullptr};
std::atomic<bool>       g_term_requested {false};
std::chrono::steady_clock::time_point g_start = std::chrono::steady_clock::now();

std::string Mib(uint64_t bytes) {
	return fmt::format("{:.0f}MiB", static_cast<double>(bytes) / 1048576.0);
}

Sample TakeSample(bool with_arenas = true) {
	Sample sample;
	sample.process = ReadProcessMemory();
	sample.heap    = ReadHeapStats();
	sample.owners  = Snapshot();
	sample.zram    = ReadZram();
	if (with_arenas) {
		sample.arenas = ReadArenas();
	}
	sample.guest_committed = g_guest_committed.load(std::memory_order_relaxed);
	std::scoped_lock read_lock(ReadMutex());
	std::vector<std::pair<std::string, ExternalReader>> readers;
	{
		std::scoped_lock lock(Mutex());
		readers = Externals();
	}
	// A reader is called without the table's lock: it may take its owner's locks.
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
	std::scoped_lock lock(Mutex());
	for (auto& [key, gauge]: Owners()) {
		if (key == name) {
			return *gauge;
		}
	}
	return *Owners().emplace_back(std::string(name), std::make_unique<Gauge>()).second;
}

std::vector<Entry> Snapshot() {
	std::vector<Entry> entries;
	std::scoped_lock   lock(Mutex());
	entries.reserve(Owners().size());
	for (const auto& [name, gauge]: Owners()) {
		entries.push_back({name, gauge->Bytes(), gauge->Items()});
	}
	return entries;
}

void SetExternal(std::string_view name, ExternalReader reader) {
	std::scoped_lock lock(Mutex());
	for (auto& [key, existing]: Externals()) {
		if (key == name) {
			existing = std::move(reader);
			return;
		}
	}
	Externals().emplace_back(std::string(name), std::move(reader));
}

void ClearExternal(std::string_view name) {
	std::scoped_lock read_lock(ReadMutex());
	std::scoped_lock lock(Mutex());
	std::erase_if(Externals(), [&](const auto& entry) { return entry.first == name; });
}

ProcessMemory ParseProcStatus(std::string_view text) {
	ProcessMemory memory;
	memory.rss      = ReadKb(text, "VmRSS:");
	memory.swap     = ReadKb(text, "VmSwap:");
	memory.peak_rss = ReadKb(text, "VmHWM:");
	memory.rss_anon  = ReadKb(text, "RssAnon:");
	memory.rss_file  = ReadKb(text, "RssFile:");
	memory.rss_shmem = ReadKb(text, "RssShmem:");
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

namespace {

// The value of attribute `name` ("size=\"123\"") in the tag that starts at `tag`.
uint64_t AttrValue(std::string_view tag, std::string_view name) {
	const std::string key = std::string(name) + "=\"";
	const auto        at  = tag.find(key);
	if (at == std::string_view::npos) {
		return 0;
	}
	auto rest = tag.substr(at + key.size());
	uint64_t value = 0;
	std::from_chars(rest.data(), rest.data() + rest.size(), value);
	return value;
}

} // namespace

std::vector<Arena> ParseMallocInfo(std::string_view xml) {
	std::vector<Arena> arenas;
	size_t             at = 0;
	while ((at = xml.find("<heap nr=\"", at)) != std::string_view::npos) {
		const auto end = xml.find("</heap>", at);
		if (end == std::string_view::npos) {
			break;
		}
		const auto body = xml.substr(at, end - at);
		Arena      arena;
		arena.nr = static_cast<uint32_t>(AttrValue(body, "nr"));
		size_t pos = 0;
		while ((pos = body.find("<total type=\"", pos)) != std::string_view::npos) {
			const auto close = body.find("/>", pos);
			const auto tag   = body.substr(pos, close == std::string_view::npos ? body.size() - pos
			                                                                    : close - pos);
			if (tag.find("\"fast\"") != std::string_view::npos ||
			    tag.find("\"rest\"") != std::string_view::npos) {
				arena.free_bytes += AttrValue(tag, "size");
			}
			pos += 13;
		}
		if (const auto sys = body.find("<system type=\"current\""); sys != std::string_view::npos) {
			arena.system_bytes = AttrValue(body.substr(sys), "size");
		}
		arenas.push_back(arena);
		at = end;
	}
	return arenas;
}

std::vector<Arena> ReadArenas() {
#if defined(__GLIBC__)
	char*  buffer = nullptr;
	size_t size   = 0;
	FILE*  stream = open_memstream(&buffer, &size);
	if (stream == nullptr) {
		return {};
	}
	malloc_info(0, stream);
	std::fclose(stream);
	std::vector<Arena> arenas;
	if (buffer != nullptr) {
		arenas = ParseMallocInfo(std::string_view(buffer, size));
		std::free(buffer);
	}
	return arenas;
#else
	return {};
#endif
}

ZramStats ParseZramMmStat(std::string_view text) {
	ZramStats stats;
	uint64_t  fields[3] = {0, 0, 0};
	size_t    count     = 0;
	const char* p       = text.data();
	const char* end     = p + text.size();
	while (count < 3 && p < end) {
		while (p < end && (*p == ' ' || *p == '\t' || *p == '\n')) {
			p++;
		}
		const auto result = std::from_chars(p, end, fields[count]);
		if (result.ec != std::errc()) {
			return stats;
		}
		p = result.ptr;
		count++;
	}
	if (count < 3) {
		return stats;
	}
	stats.known      = true;
	stats.orig       = fields[0];
	stats.compressed = fields[2];
	return stats;
}

ZramStats ReadZram() {
	ZramStats sum;
#if defined(__linux__)
	std::error_code error;
	for (const auto& entry: std::filesystem::directory_iterator("/sys/block", error)) {
		if (entry.path().filename().string().rfind("zram", 0) != 0) {
			continue;
		}
		std::ifstream file(entry.path() / "mm_stat");
		if (!file) {
			continue;
		}
		std::ostringstream text;
		text << file.rdbuf();
		const auto one = ParseZramMmStat(text.str());
		if (one.known) {
			sum.known = true;
			sum.orig += one.orig;
			sum.compressed += one.compressed;
		}
	}
#endif
	return sum;
}

uint64_t Footprint(const ProcessMemory& process, const ZramStats& zram) {
	if (!zram.known || zram.orig == 0 || zram.compressed >= zram.orig) {
		return process.rss + process.swap;
	}
	const auto share = static_cast<double>(zram.compressed) / static_cast<double>(zram.orig);
	return process.rss + static_cast<uint64_t>(static_cast<double>(process.swap) * share);
}

void NoteGuestBacking(uint64_t bytes) {
	g_guest_committed.fetch_add(bytes, std::memory_order_relaxed);
}

std::vector<std::pair<std::string, uint64_t>> Holders(const Sample& sample) {
	std::vector<std::pair<std::string, uint64_t>> holders;
	holders.emplace_back("rss-anon", sample.process.rss_anon);
	holders.emplace_back("rss-file", sample.process.rss_file);
	holders.emplace_back("rss-shmem(guest)", sample.process.rss_shmem);
	holders.emplace_back("swap", sample.process.swap);
	if (sample.heap.known) {
		holders.emplace_back("heap-in-use", sample.heap.in_use);
		holders.emplace_back("heap-free-held", sample.heap.free_held);
	}
	for (const auto& owner: sample.owners) {
		holders.emplace_back(owner.name, owner.bytes);
	}
	return holders;
}

std::string FormatGrowth(const std::vector<std::pair<std::string, uint64_t>>& before,
                         const std::vector<std::pair<std::string, uint64_t>>& after, size_t top) {
	std::vector<std::pair<std::string, int64_t>> grew;
	for (const auto& [name, now]: after) {
		uint64_t was = 0;
		for (const auto& [old_name, old_value]: before) {
			if (old_name == name) {
				was = old_value;
				break;
			}
		}
		if (now > was) {
			grew.emplace_back(name, static_cast<int64_t>(now - was));
		}
	}
	std::sort(grew.begin(), grew.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
	std::string text;
	for (size_t i = 0; i < grew.size() && i < top; i++) {
		text += fmt::format("{}{} +{}", i == 0 ? "" : ", ", grew[i].first,
		                    Mib(static_cast<uint64_t>(grew[i].second)));
	}
	return text.empty() ? "nothing" : text;
}

void ApplyHeapTune() {
#if defined(__GLIBC__)
	const auto arenas = Config::HeapTuneArenas();
	if (arenas == 0) {
		return;
	}
	// Arenas: one per core by default x8; every worker and guest thread that allocates gets its own and
	// keeps what it freed. Trim: give the top of an arena back once 64 MiB is free there. mmap: serve
	// blocks over 1 MiB from mmap so freeing them returns the pages at once (this also turns the
	// dynamic mmap threshold off, which would otherwise grow to 32 MiB and keep large buffers).
	mallopt(M_ARENA_MAX, static_cast<int>(arenas));
	mallopt(M_TRIM_THRESHOLD, 64 << 20);
	mallopt(M_MMAP_THRESHOLD, 1 << 20);
	std::printf("[mem] heap-tune: arenas<=%u trim-threshold=64MiB mmap-threshold=1MiB\n", arenas);
#endif
}

void TrimHeap(std::string_view phase) {
#if defined(__GLIBC__)
	if (Config::HeapTuneArenas() == 0) {
		return;
	}
	const auto before_heap = ReadHeapStats();
	const auto before_rss  = ReadProcessMemory().rss;
	malloc_trim(0);
	const auto after_heap = ReadHeapStats();
	const auto after_rss  = ReadProcessMemory().rss;
	const auto returned   = before_rss > after_rss ? before_rss - after_rss : 0;
	g_trimmed_total.fetch_add(returned, std::memory_order_relaxed);
	std::printf("[mem] trim %.*s: heap-free-held %s -> %s, rss %s -> %s, returned=%s\n",
	            static_cast<int>(phase.size()), phase.data(), Mib(before_heap.free_held).c_str(),
	            Mib(after_heap.free_held).c_str(), Mib(before_rss).c_str(), Mib(after_rss).c_str(),
	            Mib(returned).c_str());
	std::fflush(stdout);
#else
	(void)phase;
#endif
}

void NoteExitReason(const char* reason) {
	g_exit_reason.store(reason, std::memory_order_relaxed);
}

void PrintExit(std::string_view reason) {
	if (g_exit_printed.exchange(true)) {
		return;
	}
	const auto sample = TakeSample(false); // no allocator walk: this may be a fatal path
	std::string vk;
	for (const auto& [name, text]: sample.external) {
		vk += fmt::format(" {}: {}", name, text);
	}
	std::printf("[exit] reason=%.*s t=%.0fs rss=%s footprint=%s peak-rss=%s guest-backing=%s "
	            "(memfd, unnamed: no shm segment or file; the kernel frees it with the process) "
	            "heap-trimmed-total=%s%s | GPU memory is freed by the driver when the process ends\n",
	            static_cast<int>(reason.size()), reason.data(),
	            std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count(),
	            Mib(sample.process.rss).c_str(), Mib(Footprint(sample.process, sample.zram)).c_str(),
	            Mib(sample.process.peak_rss).c_str(), Mib(sample.guest_committed).c_str(),
	            Mib(g_trimmed_total.load(std::memory_order_relaxed)).c_str(), vk.c_str());
	std::fflush(stdout);
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
	line += fmt::format(" | footprint={}", Mib(Footprint(sample.process, sample.zram)));
	if (sample.zram.known) {
		line += fmt::format(" zram={}->{}", Mib(sample.zram.orig), Mib(sample.zram.compressed));
	}
	line += fmt::format(" rss-anon={} rss-file={} rss-shmem={}", Mib(sample.process.rss_anon),
	                    Mib(sample.process.rss_file), Mib(sample.process.rss_shmem));
	if (sample.guest_committed != 0) {
		line += fmt::format(" | guest: committed={} touched={}", Mib(sample.guest_committed),
		                    Mib(sample.process.rss_shmem));
	}
	if (!sample.arenas.empty()) {
		auto   by_held = sample.arenas;
		uint64_t held = 0;
		for (const auto& arena: by_held) {
			held += arena.free_bytes;
		}
		std::sort(by_held.begin(), by_held.end(),
		          [](const Arena& a, const Arena& b) { return a.free_bytes > b.free_bytes; });
		line += fmt::format(" | arenas={} held={} top:", by_held.size(), Mib(held));
		for (size_t i = 0; i < by_held.size() && i < 5; i++) {
			line += fmt::format(" #{}={}/{}", by_held[i].nr, Mib(by_held[i].free_bytes),
			                    Mib(by_held[i].system_bytes));
		}
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

#if defined(__linux__)
void OnTerm(int) {
	g_term_requested.store(true, std::memory_order_relaxed);
	alarm(8); // if the orderly stop below hangs, the default SIGALRM action ends the process
}
#endif

void Run() {
	const uint64_t limit = static_cast<uint64_t>(Config::MemLimitGb()) << 30;
	LimitGuard     guard(limit);
	const auto     begin = std::chrono::steady_clock::now();
	uint64_t       last_report_ms = 0;
	std::vector<std::pair<std::string, uint64_t>> previous_holders;
	uint64_t       burst_items_before = 0;
	uint64_t       last_burst_trim_ms = 0;
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
#if defined(__linux__)
		if (g_term_requested.load(std::memory_order_relaxed)) {
			// Time limit or kill by the watcher: print what we held, run the emergency path, then
			// die by the signal's default action so the exit status says so.
			NoteExitReason("sigterm");
			Subsystems::EmergencyShutdownActive();
			std::fflush(nullptr);
			std::signal(SIGTERM, SIG_DFL);
			std::raise(SIGTERM);
			std::_Exit(143);
		}
#endif
		if (now_ms - last_report_ms >= ReportIntervalMs) {
			last_report_ms = now_ms;
			PrintNow("");
			previous_holders = Holders(TakeSample());
		}
		if (Config::HeapTuneArenas() != 0) {
			// The end of a build burst: queued and running jobs went to zero after there were some.
			const auto items = Owner("jobs-queued").Items() + Owner("pipeline-jobs").Items();
			if (burst_items_before >= 8 && items == 0 && now_ms - last_burst_trim_ms >= 10'000) {
				last_burst_trim_ms = now_ms;
				TrimHeap("build-burst");
			}
			burst_items_before = items;
		}
		if (limit != 0) {
			// Every second, so only the two cheap reads: a full sample walks every allocator arena
			// under its lock (mallinfo2, malloc_info), which on a 10 GB heap takes a quarter of a
			// second and stalls every allocating thread for that long.
			const auto process   = ReadProcessMemory();
			const auto footprint = Footprint(process, ReadZram());
			const auto action    = guard.Step(now_ms, footprint);
			if (action == LimitGuard::Action::None) {
				lock.lock();
				continue;
			}
			const auto sample = TakeSample();
			switch (action) {
				case LimitGuard::Action::None: break;
				case LimitGuard::Action::DropCaches: {
					const auto grew = FormatGrowth(previous_holders, Holders(sample));
					PrintNow("limit");
					std::printf("[mem] --mem-limit-gb %u passed (footprint %s = rss %s + swap %s at its "
					            "real cost): dropping caches; the run is stopped if it is still over "
					            "in 10 s. Grew since the last report: %s\n",
					            Config::MemLimitGb(), Mib(footprint).c_str(), Mib(process.rss).c_str(),
					            Mib(process.swap).c_str(), grew.c_str());
					g_drop_requested.store(true, std::memory_order_relaxed);
#if defined(__GLIBC__)
					malloc_trim(0);
#endif
					break;
				}
				case LimitGuard::Action::Stop: {
					const auto grew = FormatGrowth(previous_holders, Holders(sample));
					PrintNow("limit-stop");
					std::printf("[mem] STOPPING: footprint %s is still over --mem-limit-gb %u 10 s after "
					            "the caches were dropped. Grew since the last report: %s. Stopped by "
					            "the emulator, not by the kernel; read the table above.\n",
					            Mib(footprint).c_str(), Config::MemLimitGb(), grew.c_str());
					NoteExitReason("limit-stop");
					std::fflush(nullptr);
					Subsystems::EmergencyShutdownActive();
					std::fflush(nullptr);
					std::_Exit(75);
				}
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
	ApplyHeapTune();
#if defined(__linux__)
	struct sigaction action {};
	action.sa_handler = OnTerm;
	sigemptyset(&action.sa_mask);
	sigaction(SIGTERM, &action, nullptr);
#endif
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
	PrintExit("normal");
}

void EmergencyPrint() {
	// Not from the reporting thread's join: it may be the thread that called this.
	PrintNow("emergency");
	const char* reason = g_exit_reason.load(std::memory_order_relaxed);
	PrintExit(reason != nullptr ? reason : "emergency");
}

} // namespace Common::MemAccount
