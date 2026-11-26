#include "common/shaderRegLedger.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Common::ShaderRegLedger {

namespace {

constexpr size_t StageCount = static_cast<size_t>(Stage::Count);
constexpr std::array<const char*, StageCount> StageNames = {
    "cs", "ps", "gs", "hs", "fs", "?"
};

// Milliseconds since process load; both paths use the same clock.
const auto Start = std::chrono::steady_clock::now();

struct StageRecord {
	int64_t  first_registration_ms = -1;
	uint64_t registrations          = 0;
	int64_t  first_use_ms           = -1;
	uint64_t uses                   = 0;
};

struct HashRecord {
	std::array<StageRecord, StageCount> stages;
	// Distinct ProgramKey values seen for this shader hash.
	std::unordered_set<uint64_t> variants;
};

std::mutex                                g_mutex;
std::unordered_map<uint64_t, HashRecord>  g_records;
std::atomic<uint64_t>                     g_create_calls {0};
std::atomic_flag                          g_printed; // Clear by default in C++20.

constexpr size_t DumpQueueCapacity = 512;

struct DumpEntry {
	uint64_t                                     hash = 0;
	std::string                                  stage;
	DumpRegistrationInfo                        info;
	std::shared_ptr<const std::vector<uint32_t>> code;
};

std::filesystem::path             g_dump_dir;
std::mutex                        g_dump_mutex;
std::condition_variable           g_dump_ready;
std::deque<DumpEntry>             g_dump_queue;
std::unordered_set<std::string>   g_dump_seen;
std::thread                       g_dump_thread;
bool                              g_dump_stop = false;
std::atomic<uint64_t>             g_dump_written {0};
std::atomic<uint64_t>             g_dump_dropped {0};
std::atomic<uint64_t>             g_dump_high_water {0};

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now() - Start)
	    .count();
}

template <typename... Args>
void AppendLine(std::string& text, fmt::format_string<Args...> format, Args&&... args) {
	text += "[shader-reg] ";
	fmt::format_to(std::back_inserter(text), format, std::forward<Args>(args)...);
	text += '\n';
}

std::string Counts(const std::array<uint64_t, StageCount>& values) {
	std::string text;
	for (size_t i = 0; i < StageCount; i++) {
		if (i != 0) text += ' ';
		fmt::format_to(std::back_inserter(text), "{}={}", StageNames[i], values[i]);
	}
	return text;
}

std::string MissingInputs(std::string_view stage) {
	if (stage == "cs") return R"(["user_data","compute_stage_input"])";
	if (stage == "ps") return R"(["user_data","pixel_stage_input"])";
	if (stage == "gs-front" || stage == "gs-back" || stage == "hs-front" ||
	    stage == "hs-back") {
		return R"(["user_data","graphics_stage_input","paired_stage_code","vertex_fetch_state"])";
	}
	return R"(["user_data","graphics_stage_input","vertex_fetch_state"])";
}

void WriteDumpEntry(const DumpEntry& entry, std::ofstream& index) {
	const auto name = fmt::format("{}_{:016x}.bin", entry.stage, entry.hash);
	const auto path = g_dump_dir / name;
	std::ofstream binary(path, std::ios::binary | std::ios::trunc);
	if (!binary || entry.code == nullptr) {
		g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	binary.write(reinterpret_cast<const char*>(entry.code->data()),
	             static_cast<std::streamsize>(entry.code->size() * sizeof(uint32_t)));
	binary.flush();
	if (!binary) {
		g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	binary.close();

	index << fmt::format(
	    "{{\"hash\":\"0x{:016x}\",\"stage\":\"{}\",\"file\":\"{}\","
	    "\"registered_size_bytes\":{},\"translated_size_bytes\":{},\"words\":{},"
	    "\"user_data_count\":null,\"user_data_layout\":{{\"direct_resources\":{},"
	    "\"sharp_resources\":{},\"eud_size_dwords\":{},\"srt_size_dwords\":{}}},"
	    "\"stage_input_info\":{{\"scratch_size_dwords\":{},\"input_semantics\":{}}},"
	    "\"missing_inputs\":{}}}\n",
	    entry.hash, entry.stage, name, entry.info.registered_size_bytes,
	    entry.info.translated_size_bytes, entry.code->size(), entry.info.direct_resources,
	    entry.info.sharp_resources, entry.info.eud_size_dwords, entry.info.srt_size_dwords,
	    entry.info.scratch_size_dwords, entry.info.input_semantics, MissingInputs(entry.stage));
	index.flush();
	if (index) g_dump_written.fetch_add(1, std::memory_order_relaxed);
	else g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
}

void DumpWorker() {
	std::ofstream index(g_dump_dir / "index.jsonl", std::ios::app);
	if (!index) {
		g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	for (;;) {
		DumpEntry entry;
		{
			std::unique_lock lock(g_dump_mutex);
			g_dump_ready.wait(lock, [] { return g_dump_stop || !g_dump_queue.empty(); });
			if (g_dump_queue.empty() && g_dump_stop) return;
			entry = std::move(g_dump_queue.front());
			g_dump_queue.pop_front();
		}
		WriteDumpEntry(entry, index);
	}
}

} // namespace

void Initialize() {
	g_dump_dir = Config::GetRegisteredShaderDumpFolder();
	if (g_dump_dir.empty()) return;
	std::error_code error;
	std::filesystem::create_directories(g_dump_dir, error);
	if (error) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Could not create registered shader dump directory {}: {}\n", g_dump_dir.string(),
		    error.message()));
		g_dump_dir.clear();
		return;
	}
	g_dump_stop = false;
	g_dump_thread = std::thread(DumpWorker);
}

void Shutdown() {
	if (g_dump_thread.joinable()) {
		{
			std::lock_guard lock(g_dump_mutex);
			g_dump_stop = true;
		}
		g_dump_ready.notify_one();
		g_dump_thread.join();
	}
	PrintSummary();
}

void NoteCreateCall() {
	g_create_calls.fetch_add(1, std::memory_order_relaxed);
}

void NoteRegistration(uint64_t hash, Stage stage) {
	const auto index = static_cast<size_t>(stage);
	if (index >= StageCount) return;
	const auto now = NowMs();
	std::lock_guard lock(g_mutex);
	auto&   record = g_records.try_emplace(hash).first->second;
	auto&   entry  = record.stages[index];
	if (entry.first_registration_ms < 0) entry.first_registration_ms = now;
	++entry.registrations;
}

void DumpRegistration(uint64_t hash, const DumpRegistrationInfo& info,
                      std::shared_ptr<const std::vector<uint32_t>> code) {
	if (g_dump_dir.empty() || code == nullptr) return;
	std::unique_lock lock(g_dump_mutex, std::try_to_lock);
	if (!lock.owns_lock() || g_dump_queue.size() >= DumpQueueCapacity) {
		g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto key = fmt::format("{}:{:016x}", info.binary_stage, hash);
	if (!g_dump_seen.insert(key).second) return;
	DumpEntry entry {.hash = hash, .stage = std::string(info.binary_stage), .info = info,
	                 .code = std::move(code)};
	// string_view in the queued metadata referred to the caller's static stage string. Keep no view
	// in the asynchronous record even though the current callers use static storage.
	entry.info.binary_stage = {};
	g_dump_queue.push_back(std::move(entry));
	const auto size = static_cast<uint64_t>(g_dump_queue.size());
	auto high = g_dump_high_water.load(std::memory_order_relaxed);
	while (size > high && !g_dump_high_water.compare_exchange_weak(
	                          high, size, std::memory_order_relaxed)) {}
	lock.unlock();
	g_dump_ready.notify_one();
}

void NoteUse(uint64_t hash, Stage stage, uint64_t variant_key) {
	const auto index = static_cast<size_t>(stage);
	if (index >= StageCount) return;
	const auto now = NowMs();
	std::lock_guard lock(g_mutex);
	auto&   record = g_records.try_emplace(hash).first->second;
	auto&   entry  = record.stages[index];
	if (entry.first_use_ms < 0) entry.first_use_ms = now;
	++entry.uses;
	record.variants.insert(variant_key);
}

void PrintSummary() {
	if (g_printed.test_and_set()) return;

	std::lock_guard lock(g_mutex);
	std::array<uint64_t, StageCount>       registered {};
	std::array<uint64_t, StageCount>       used {};
	std::array<uint64_t, StageCount>       registered_unused {};
	std::array<uint64_t, StageCount>       used_unregistered {};
	std::array<std::vector<int64_t>, StageCount> leads;

	for (const auto& item: g_records) {
		const auto& record = item.second;
		for (size_t i = 0; i < StageCount; i++) {
			const auto& entry = record.stages[i];
			const bool  reg   = entry.first_registration_ms >= 0;
			const bool  use   = entry.first_use_ms >= 0;
			if (reg) ++registered[i];
			if (use) ++used[i];
			if (reg && !use) ++registered_unused[i];
			if (use && !reg) ++used_unregistered[i];
			if (reg && use) leads[i].push_back(entry.first_use_ms - entry.first_registration_ms);
		}
	}

	std::string text;
	text += "[shader-reg] === shader registration ledger ===\n";
	AppendLine(text, "create-shader calls: {}", g_create_calls.load(std::memory_order_relaxed));
	AppendLine(text, "registered:                {}", Counts(registered));
	AppendLine(text, "used:                      {}", Counts(used));
	AppendLine(text, "registered but never used: {}", Counts(registered_unused));
	AppendLine(text, "used but never registered: {}", Counts(used_unregistered));
	for (size_t i = 0; i < StageCount; i++) {
		auto& values = leads[i];
		if (values.empty()) {
			AppendLine(text, "lead time ms (use - registration) {}: n=0", StageNames[i]);
			continue;
		}
		std::sort(values.begin(), values.end());
		const auto n       = values.size();
		const auto p90     = values[std::min(n - 1, (n * 9) / 10)];
		const auto median  = values[n / 2];
		AppendLine(text, "lead time ms (use - registration) {}: n={} min={} median={} p90={} max={}",
		           StageNames[i], n, values.front(), median, p90, values.back());
	}

	std::array<uint64_t, 4> histogram {}; // 1, 2, 3-4, 5+
	size_t                  hashes_with_variants = 0;
	for (const auto& item: g_records) {
		const auto count = item.second.variants.size();
		if (count == 0) continue;
		++hashes_with_variants;
		if (count == 1) histogram[0]++;
		else if (count == 2) histogram[1]++;
		else if (count <= 4) histogram[2]++;
		else histogram[3]++;
	}
	AppendLine(text, "variants per hash: 1={} 2={} 3-4={} 5+={} (hashes with uses={})",
	           histogram[0], histogram[1], histogram[2], histogram[3], hashes_with_variants);
	if (!g_dump_dir.empty()) {
		AppendLine(text, "registration dump: written={} dropped={} queue_high_water={}/{}",
		           g_dump_written.load(std::memory_order_relaxed),
		           g_dump_dropped.load(std::memory_order_relaxed),
		           g_dump_high_water.load(std::memory_order_relaxed), DumpQueueCapacity);
	}
	text += "[shader-reg] === end shader registration ledger ===\n";

	Log::WriteToConsoleAndLog(text);
}

} // namespace Common::ShaderRegLedger
