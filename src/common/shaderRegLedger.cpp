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
#include <nlohmann/json.hpp>
#include <optional>
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

// First-use records: a second bounded queue handled by the same writer thread.
constexpr size_t   UseQueueCapacity = 128;
constexpr uint64_t UseQueueBytes    = 256ull << 20;
std::deque<UseRecord>             g_use_queue;
uint64_t                          g_use_queue_bytes = 0;
std::unordered_set<std::string>   g_use_seen;
std::unordered_map<std::string, uint64_t> g_use_skipped;
std::atomic<uint64_t>             g_use_written {0};
std::atomic<uint64_t>             g_use_dropped {0};
std::atomic<uint64_t>             g_use_high_water {0};
std::atomic<uint64_t>             g_use_order {0};

uint64_t UseBytes(const UseRecord& record) {
	return record.memory.size() + record.input_info.size() +
	       (record.user_data.size() + record.back_code.size()) * sizeof(uint32_t);
}

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

std::vector<std::string> MissingInputs(const DumpEntry& entry) {
	if (entry.stage == "cs") {
		std::vector<std::string> missing {"user_data_values"};
		if (!entry.info.compute_input_known) missing.push_back("compute_stage_input");
		return missing;
	}
	if (entry.stage == "ps") return {"user_data_values", "pixel_stage_input"};
	if (entry.stage == "gs-front" || entry.stage == "gs-back" || entry.stage == "hs-front" ||
	    entry.stage == "hs-back") {
		return {"user_data_values", "graphics_stage_input", "paired_stage_code",
		        "vertex_fetch_state"};
	}
	return {"user_data_values", "graphics_stage_input", "vertex_fetch_state"};
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

	nlohmann::ordered_json record {
	    {"hash", fmt::format("0x{:016x}", entry.hash)},
	    {"stage", entry.stage},
	    {"file", name},
	    {"registered_size_bytes", entry.info.registered_size_bytes},
	    {"translated_size_bytes", entry.info.translated_size_bytes},
	    {"words", entry.code->size()},
	    {"user_data_count", entry.info.compute_input_known
	                            ? nlohmann::ordered_json(entry.info.user_data_count)
	                            : nlohmann::ordered_json(nullptr)},
	    {"user_data_layout",
	     {{"direct_resources", entry.info.direct_resources},
	      {"sharp_resources", entry.info.sharp_resources},
	      {"eud_size_dwords", entry.info.eud_size_dwords},
	      {"srt_size_dwords", entry.info.srt_size_dwords}}},
	    {"stage_input_info",
	     {{"known", entry.info.compute_input_known},
	      {"scratch_size_dwords", entry.info.scratch_size_dwords},
	      {"input_semantics", entry.info.input_semantics},
	      {"threads_num", std::array {entry.info.threads_num[0], entry.info.threads_num[1],
	                                   entry.info.threads_num[2]}},
	      {"lds_size_dwords", entry.info.lds_size_dwords},
	      {"wave_size", entry.info.wave_size},
	      {"float_mode", entry.info.float_mode},
	      {"group_id", std::array {entry.info.group_id[0], entry.info.group_id[1],
	                                entry.info.group_id[2]}},
	      {"thread_ids_num", entry.info.thread_ids_num},
	      {"tg_size_en", entry.info.tg_size_en},
	      {"workgroup_register", entry.info.workgroup_register},
	      {"dispatch_thread_dimensions", entry.info.dispatch_thread_dimensions}}},
	    {"missing_inputs", MissingInputs(entry)}};
	index << record.dump() << '\n';
	index.flush();
	if (index) g_dump_written.fetch_add(1, std::memory_order_relaxed);
	else g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
}

void SerializeUse(const UseRecord& record, uint64_t order, std::string& line,
                  std::vector<uint8_t>& blob) {
	blob.clear();
	const auto put = [&](const void* data, uint64_t bytes) {
		const auto begin = static_cast<uint64_t>(blob.size());
		const auto* first = static_cast<const uint8_t*>(data);
		if (bytes != 0) blob.insert(blob.end(), first, first + bytes);
		return nlohmann::ordered_json {{"offset", begin}, {"bytes", bytes}};
	};
	auto input  = put(record.input_info.data(), record.input_info.size());
	auto memory = put(record.memory.data(), record.memory.size());
	auto back   = put(record.back_code.data(), record.back_code.size() * sizeof(uint32_t));
	nlohmann::ordered_json ranges = nlohmann::ordered_json::array();
	for (const auto& range: record.ranges) ranges.push_back({range.address, range.bytes});
	nlohmann::ordered_json json {
	    {"kind", "use"},
	    {"order", order},
	    {"hash", fmt::format("0x{:016x}", record.hash)},
	    {"stage", record.stage},
	    {"file", fmt::format("use_{}_{:016x}.bin", record.stage, record.hash)},
	    {"options_wave_size", record.options_wave_size},
	    {"user_data_base", record.user_data_base},
	    {"user_data", record.user_data},
	    {"push_window", {record.push_begin, record.push_end}},
	    {"workgroup_counts", {record.workgroup_counts[0], record.workgroup_counts[1],
	                          record.workgroup_counts[2]}},
	    {"indirect_args_address", record.indirect_args_address},
	    {"shader_base", record.shader_base},
	    {"uses_image_heap", record.uses_image_heap},
	    {"memory_truncated", record.memory_truncated},
	    {"input_info", input},
	    {"memory", memory},
	    {"back_code", back},
	    {"ranges", ranges}};
	line = json.dump();
}

void WriteUseRecord(const UseRecord& record, uint64_t order, std::ofstream& index) {
	std::string          line;
	std::vector<uint8_t> blob;
	SerializeUse(record, order, line, blob);
	const auto name = fmt::format("use_{}_{:016x}.bin", record.stage, record.hash);
	std::ofstream file(g_dump_dir / name, std::ios::binary | std::ios::trunc);
	if (file) {
		file.write(reinterpret_cast<const char*>(blob.data()),
		           static_cast<std::streamsize>(blob.size()));
		file.flush();
	}
	if (!file) {
		g_use_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	file.close();
	index << line << '\n';
	index.flush();
	if (index) g_use_written.fetch_add(1, std::memory_order_relaxed);
	else g_use_dropped.fetch_add(1, std::memory_order_relaxed);
}

void DumpWorker() {
	std::ofstream index(g_dump_dir / "index.jsonl", std::ios::app);
	std::ofstream use_index(g_dump_dir / "use.jsonl", std::ios::app);
	if (!index) {
		g_dump_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	for (;;) {
		std::optional<DumpEntry> entry;
		std::optional<UseRecord> use;
		{
			std::unique_lock lock(g_dump_mutex);
			g_dump_ready.wait(lock, [] {
				return g_dump_stop || !g_dump_queue.empty() || !g_use_queue.empty();
			});
			if (g_dump_queue.empty() && g_use_queue.empty() && g_dump_stop) return;
			if (!g_dump_queue.empty()) {
				entry = std::move(g_dump_queue.front());
				g_dump_queue.pop_front();
			} else {
				use = std::move(g_use_queue.front());
				g_use_queue.pop_front();
				g_use_queue_bytes -= std::min(g_use_queue_bytes, UseBytes(*use));
			}
		}
		if (entry) WriteDumpEntry(*entry, index);
		else if (use && use_index) WriteUseRecord(*use, g_use_order.fetch_add(1), use_index);
		else g_use_dropped.fetch_add(1, std::memory_order_relaxed);
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

void SerializeUseRecord(const UseRecord& record, uint64_t order, std::string& line,
                        std::vector<uint8_t>& blob) {
	SerializeUse(record, order, line, blob);
}

bool UseDumpEnabled() {
	return !g_dump_dir.empty();
}

bool WantUse(uint64_t hash, std::string_view stage) {
	if (g_dump_dir.empty()) return false;
	std::lock_guard lock(g_dump_mutex);
	return g_use_seen.insert(fmt::format("{}:{:016x}", stage, hash)).second;
}

void NoteUseSkipped(std::string_view reason) {
	std::lock_guard lock(g_dump_mutex);
	++g_use_skipped[std::string(reason)];
}

void DumpUse(UseRecord&& record) {
	if (g_dump_dir.empty()) return;
	std::unique_lock lock(g_dump_mutex, std::try_to_lock);
	const auto bytes = UseBytes(record);
	if (!lock.owns_lock() || g_use_queue.size() >= UseQueueCapacity ||
	    g_use_queue_bytes + bytes > UseQueueBytes) {
		g_use_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	g_use_queue_bytes += bytes;
	g_use_queue.push_back(std::move(record));
	const auto size = static_cast<uint64_t>(g_use_queue.size());
	auto high = g_use_high_water.load(std::memory_order_relaxed);
	while (size > high && !g_use_high_water.compare_exchange_weak(
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
	if (!g_dump_dir.empty()) {
		std::string skipped;
		{
			std::lock_guard lock(g_dump_mutex);
			for (const auto& [reason, count]: g_use_skipped) {
				skipped += fmt::format(" {}={}", reason, count);
			}
		}
		AppendLine(text, "use dump: written={} dropped={} queue_high_water={}/{} skipped:{}",
		           g_use_written.load(std::memory_order_relaxed),
		           g_use_dropped.load(std::memory_order_relaxed),
		           g_use_high_water.load(std::memory_order_relaxed), UseQueueCapacity,
		           skipped.empty() ? " none" : skipped);
	}
	text += "[shader-reg] === end shader registration ledger ===\n";

	Log::WriteToConsoleAndLog(text);
}

} // namespace Common::ShaderRegLedger
