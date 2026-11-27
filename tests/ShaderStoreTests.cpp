#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreDiff.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreFile.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace Libs::Graphics;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ShaderStoreTests: failed: %s\n", message);
		std::abort();
	}
}

ShaderComputeInputInfo MakeCompute() {
	ShaderComputeInputInfo info;
	info.wave_size           = 32;
	info.float_mode          = 0xc0;
	info.host_subgroup_size  = 32;
	info.thread_ids_num      = 1;
	info.workgroup_register  = 2;
	info.lds_size_dwords     = 128;
	info.threads_num[0]      = 8;
	info.threads_num[1]      = 8;
	info.threads_num[2]      = 1;
	info.group_id[0]         = true;
	info.tg_size_en          = true;
	return info;
}

// The same shader in two boots: everything the guest places differently from run to run changes,
// the key must not.
void TestKeyIgnoresGuestAddresses() {
	auto boot_a = MakeCompute();
	auto boot_b = MakeCompute();
	boot_a.indirect_args_address  = 0x7000'1234'0000ull;
	boot_b.indirect_args_address  = 0x7000'9876'5000ull;
	boot_a.workgroup_counts[0]    = 64;
	boot_b.workgroup_counts[0]    = 1920;
	boot_a.dispatch_threads_num[0] = 512;
	boot_b.dispatch_threads_num[0] = 7;
	boot_a.async_compute          = false;
	boot_b.async_compute          = true;
	std::vector<uint32_t> key_a;
	std::vector<uint32_t> key_b;
	BuildStageStaticKey(boot_a, key_a);
	BuildStageStaticKey(boot_b, key_b);
	Check(!key_a.empty() && key_a == key_b,
	      "addresses, dispatch sizes and the queue do not enter the program key");

	// What the key does contain: the shader's own shape.
	auto other = MakeCompute();
	other.threads_num[0] = 16;
	std::vector<uint32_t> key_c;
	BuildStageStaticKey(other, key_c);
	Check(key_c != key_a, "the workgroup size does");
	other = MakeCompute();
	other.host_subgroup_size = 64;
	BuildStageStaticKey(other, key_c);
	Check(key_c != key_a, "and the host subgroup size");
	other = MakeCompute();
	other.lds_storage = !other.lds_storage;
	BuildStageStaticKey(other, key_c);
	Check(key_c != key_a, "and where the shared memory lives (a store built for another device limit)");
}

std::vector<uint8_t> Record(uint8_t seed, size_t size) {
	std::vector<uint8_t> bytes(size);
	for (size_t i = 0; i < size; i++) {
		bytes[i] = static_cast<uint8_t>(seed + i * 7u);
	}
	return bytes;
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// All records of `bytes` after `header_size`, as the loader reads them.
std::vector<std::vector<uint8_t>> ReadRecords(const std::vector<uint8_t>& bytes, size_t header_size,
                                              size_t* rest = nullptr) {
	std::vector<std::vector<uint8_t>> out;
	std::span<const uint8_t>          records(bytes);
	records = records.subspan(header_size);
	for (;;) {
		std::vector<uint8_t> raw;
		size_t               frame = 0;
		if (ShaderStoreFile::Decode(records, false, raw, frame) !=
		    ShaderStoreFile::Status::Loaded) {
			break;
		}
		records = records.subspan(frame);
		out.push_back(std::move(raw));
	}
	if (rest != nullptr) {
		*rest = records.size();
	}
	return out;
}

bool WaitForFlush(const ShaderStoreWriter& store, uint64_t records) {
	for (int i = 0; i < 500; i++) {
		const auto counts = store.GetCounts();
		if (counts.pending == 0 && counts.flushed >= records) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	return false;
}

// A run that ends by an abort never calls Finish: whatever it had appended must already be in the
// file, and the next run must be able to append after it.
void TestDurableAppendAndReload() {
	const auto dir  = std::filesystem::temp_directory_path() /
	                  ("kyty-store-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	const auto path = dir / "store.shaders";
	std::filesystem::remove_all(dir);
	const std::vector<uint8_t> header = {'H', 'D', 'R', 1, 2, 3};

	std::vector<std::vector<uint8_t>> sent;
	{
		ShaderStoreWriter store;
		store.Open(path, header, 0);
		Check(static_cast<bool>(store), "the store opens");
		for (uint8_t i = 0; i < 6; i++) {
			sent.push_back(Record(i, 100u + i * 20000u));
			store.Append(std::vector<uint8_t>(sent.back()));
		}
		Check(WaitForFlush(store, sent.size()), "every finished record is on the disk within seconds");
		const auto counts = store.GetCounts();
		Check(counts.written == sent.size() && counts.failed == 0 && counts.flushed == counts.written,
		      "the counts say written, none failed, all flushed");

		// What an observer sees while the run is still going, i.e. after an abort.
		const auto bytes = ReadFile(path);
		Check(std::equal(header.begin(), header.end(), bytes.begin()), "the header is first");
		size_t     rest    = 0;
		const auto records = ReadRecords(bytes, header.size(), &rest);
		Check(records == sent && rest == 0, "all records reload, in order, with nothing left over");

		// The abort lands in the middle of the next record.
		store.Append(Record(9, 5000));
		Check(WaitForFlush(store, sent.size() + 1), "one more record");
	}
	// An aborted write leaves a partial frame behind the complete ones.
	auto bytes = ReadFile(path);
	const auto complete = ReadRecords(bytes, header.size());
	Check(complete.size() == 7, "seven records are complete");
	bytes.insert(bytes.end(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
	{
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	}
	size_t rest = 0;
	bytes       = ReadFile(path);
	const auto reread = ReadRecords(bytes, header.size(), &rest);
	Check(reread.size() == 7 && rest == 10, "the partial tail is not a record and is left alone");

	// The next boot keeps the complete records and appends behind them.
	{
		ShaderStoreWriter store;
		store.Open(path, header, bytes.size() - rest);
		store.Append(Record(42, 777));
		Check(WaitForFlush(store, 1), "the new boot's record is flushed too");
	}
	size_t     final_rest = 0;
	const auto final_records = ReadRecords(ReadFile(path), header.size(), &final_rest);
	Check(final_records.size() == 8 && final_records.back() == Record(42, 777) && final_rest == 0,
	      "the reloaded store holds the old records and the new one, without the torn tail");
	std::filesystem::remove_all(dir);
}

ShaderRecompiler::IR::ResourceSpecialization MakeSpecialization(uint32_t stride, uint32_t mips) {
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	specialization.buffers.resize(3);
	specialization.buffers[1].packed_stride = stride;
	specialization.images.resize(1);
	specialization.images[0].mip_count = mips;
	return specialization;
}

void TestMissDescriptions() {
	const auto have = MakeSpecialization(16, 4);
	const auto want = MakeSpecialization(32, 4);
	const auto same = StoreDiff::DiffSpecialization(have, have);
	Check(same.count == 0, "equal specializations do not differ");
	const auto diff = StoreDiff::DiffSpecialization(have, want);
	Check(diff.count == 1 && diff.first == "buffers[1].packed_stride 16 vs 32",
	      "the differing field is named with both values");

	const auto near = MakeSpecialization(16, 2);
	const std::vector<StoreDiff::ModuleInputs> modules = {
	    {.id = 0xA, .push_start = 4, .want_push_start = 4, .specialization = &have},
	    {.id = 0xB, .push_start = 4, .want_push_start = 8, .specialization = &near},
	};
	const auto text = StoreDiff::DescribeNearestModule(modules, want);
	Check(text.find("0x000000000000000a") != std::string::npos &&
	          text.find("buffers[1].packed_stride 16 vs 32") != std::string::npos,
	      "the nearest module is the one with the fewest differences");
	const auto push = StoreDiff::DescribeNearestModule(
	    {{.id = 0xC, .push_start = 4, .want_push_start = 8, .specialization = &want}}, want);
	Check(push.find("push window start 4 vs 8") != std::string::npos,
	      "a push window that moved is named");
	Check(StoreDiff::DescribeNearestModule({}, want) == "the program has no module",
	      "a program without modules says so");

	const auto key = StoreDiff::DescribeKeyDifference(8, {1, 2, 3}, 8, {1, 9, 3});
	Check(key == "static_state[1] 0x2 vs 0x9 (+0 more)", "one static word differs");
	Check(StoreDiff::DescribeKeyDifference(8, {1}, 9, {1}) == "user_data_count 8 vs 9 (+0 more)",
	      "or the user data count");
}

void TestLookupCounters() {
	StoreDiff::LookupStats stats;
	Check(stats.NoteDistinct(5) && !stats.NoteDistinct(5) && stats.NoteDistinct(6),
	      "a miss is counted once however often the dispatch retries");
	stats.lookups = 10;
	stats.hit     = 6;
	stats.miss_key = 3;
	stats.miss_absent = 1;
	Check(stats.Line() == "[store] lookups=10 hit=6 miss-key=3 miss-absent=1 distinct-miss=2",
	      "the counter line has the documented fields");
	for (uint32_t i = 0; i < StoreDiff::LookupStats::MaxDescribed; i++) {
		Check(stats.WantsDescription(), "the first twenty misses are described");
		stats.described++;
	}
	Check(!stats.WantsDescription(), "and no more");
}

} // namespace

int main() {
	TestKeyIgnoresGuestAddresses();
	TestDurableAppendAndReload();
	TestMissDescriptions();
	TestLookupCounters();
	std::printf("ShaderStoreTests: ok\n");
	return 0;
}
