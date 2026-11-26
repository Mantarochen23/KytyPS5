#include "common/shaderRegLedger.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <fmt/format.h>
#include <iterator>
#include <mutex>
#include <string>
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

} // namespace

void Initialize() {
	// The clock starts at load, so no setup is needed here.
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
	text += "[shader-reg] === end shader registration ledger ===\n";

	Log::WriteToConsoleAndLog(text);
}

} // namespace Common::ShaderRegLedger
