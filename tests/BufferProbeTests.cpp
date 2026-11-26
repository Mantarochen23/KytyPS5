#include "graphics/host_gpu/renderer/cache/loopGuardReport.h"
#include "graphics/host_gpu/renderer/pipeline/bufferProbe.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreIdentity.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "BufferProbeTests: failed: %s\n", message);
		std::abort();
	}
}

void TestCompare() {
	std::vector<uint8_t> guest(3 * 4096 + 100, 0);
	auto                 host = guest;
	{
		const auto c = BufferProbe::Compare(guest.data(), host.data(), guest.size());
		Check(!c.Differs() && c.differing_pages == 0 && c.total_pages == 4 &&
		          c.first_diff == UINT64_MAX,
		      "identical buffers must not differ");
	}
	// One byte in page 1 and two in the short last page.
	host[4096 + 7]     = 0xab;
	host[3 * 4096 + 5] = 1;
	host[3 * 4096 + 99] = 2;
	for (uint32_t i = 0; i < 8; i++) guest[i * 4] = static_cast<uint8_t>(i + 1);
	for (uint32_t i = 0; i < 8; i++) host[i * 4] = static_cast<uint8_t>(i + 1);
	const auto c = BufferProbe::Compare(guest.data(), host.data(), guest.size());
	Check(c.Differs() && c.first_diff == 4096 + 7, "first differing offset");
	Check(c.differing_pages == 2 && c.total_pages == 4, "differing page count");
	Check(c.differing_bytes == 3, "differing byte count");
	Check(c.guest_head[0] == 1u && c.host_head[7] == 8u && c.guest_head[7] == 8u,
	      "head words of both sides");
	Check(c.host_at_diff[0] == 0xab000000u && c.guest_at_diff[0] == 0u,
	      "difference words start at the dword of the first difference");
	// A range whose tail is shorter than eight words reads zero past the end.
	const uint8_t a[6] {1, 0, 0, 0, 9, 9};
	const uint8_t b[6] {1, 0, 0, 0, 9, 8};
	const auto    d = BufferProbe::Compare(a, b, 6);
	Check(d.first_diff == 5 && d.host_head[2] == 0u && d.guest_head[1] == 0x0909u,
	      "short buffer tail");
}

void TestClassify() {
	using BufferProbe::CacheState;
	Check(BufferProbe::ClassifyCacheState(false, false) == CacheState::Synced, "synced");
	Check(BufferProbe::ClassifyCacheState(true, false) == CacheState::CpuDirty, "cpu dirty");
	Check(BufferProbe::ClassifyCacheState(false, true) == CacheState::GpuDirty, "gpu dirty");
	Check(BufferProbe::ClassifyCacheState(true, true) == CacheState::Both, "both");
	BufferProbe::Comparison same;
	BufferProbe::Comparison different;
	different.first_diff = 4;
	Check(!BufferProbe::IsUnexpectedDifference(same, CacheState::Synced), "no difference");
	Check(BufferProbe::IsUnexpectedDifference(different, CacheState::Synced),
	      "difference on a synced range is unexpected");
	Check(BufferProbe::IsUnexpectedDifference(different, CacheState::CpuDirty),
	      "difference after the upload on a cpu-dirty range is unexpected");
	Check(!BufferProbe::IsUnexpectedDifference(different, CacheState::GpuDirty),
	      "the GPU owns newer data on a gpu-dirty range");
}

void TestLineAndLimiter() {
	BufferProbe::Comparison c;
	c.compared_bytes = 8192;
	c.total_pages    = 2;
	c.first_diff     = 0x1004;
	c.differing_pages = 1;
	c.differing_bytes = 4;
	const auto line = BufferProbe::FormatLine(0x1234abcd5678ef01ull, 3, 1, 0x200008c000ull, 8192,
	                                          true, BufferProbe::CacheState::Synced, true, c);
	Check(line.find("shader=0x") != std::string::npos && line.find("1234abcd5678ef01") != std::string::npos &&
	          line.find("first_diff=0x1004") != std::string::npos &&
	          line.find("pages_differ=1/2") != std::string::npos &&
	          line.find("cache=synced") != std::string::npos &&
	          line.find("UNEXPECTED") != std::string::npos,
	      "probe line fields");
	BufferProbe::Comparison none;
	const auto ok = BufferProbe::FormatLine(1, 0, 0, 0x1000, 16, false,
	                                        BufferProbe::CacheState::GpuDirty, false, none);
	Check(ok.find("first_diff=none") != std::string::npos &&
	          ok.find("guest=unreadable") != std::string::npos &&
	          ok.find("UNEXPECTED") == std::string::npos,
	      "matching line");

	BufferProbe::Limiter limiter;
	uint32_t             printed = 0;
	for (uint32_t i = 0; i < 1000; i++) {
		const bool mismatch = i % 2 == 0;
		if (limiter.OnDispatch(mismatch, mismatch)) printed++;
		Check(limiter.SummaryDue() == ((i + 1) % 500 == 0), "summary every 500 dispatches");
	}
	// 40 mismatching dispatches plus the clean ones among the first 5 probed dispatches.
	Check(printed == BufferProbe::MaxMismatchPrints + 2 && limiter.Printed() == 40 &&
	          limiter.Mismatches() == 500 && limiter.Dispatches() == 1000,
	      "the first 40 mismatching and the first 5 probed dispatches print");
	Check(limiter.Summary().find("probed_dispatches=1000 mismatching=500 unexpected=500") !=
	          std::string::npos,
	      "summary text");
}

void TestStoreIdentity() {
	using Profile = ShaderRecompiler::Spirv::Profile;
	const auto collect = [](const Profile& p) {
		std::vector<uint64_t> values;
		VisitShaderStoreProfile(p, [&](auto v) { values.push_back(static_cast<uint64_t>(v)); });
		return values;
	};
	Profile plain {.maintenance8 = true, .device_clock_period_ps = 40, .ray_query = true};
	Profile guarded = plain;
	guarded.loop_guard_limit = 200000;
	const auto a = collect(plain);
	Check(a.size() == 8, "the store identity keeps the eight historic profile fields");
	Check(a == collect(guarded),
	      "a loop-guard limit must not change the store header (it made every boot reject the store)");
	Check(ShaderStoreSuffix(0).empty(), "default store file name is unchanged");
	Check(ShaderStoreSuffix(200000) == ".guard200000" && ShaderStoreSuffix(200000) != ShaderStoreSuffix(5),
	      "each loop-guard limit selects its own store file");
}

void TestLoopGuardReport() {
	std::vector<uint32_t> words(LoopGuardAreaDwords, 0u);
	std::set<std::pair<uint64_t, uint32_t>> seen;
	Check(FormatLoopGuardTrips(words, seen).empty(), "no trips, no lines");
	words[0] = 70; // More invocations than slots.
	const auto set_slot = [&](uint32_t slot, uint64_t hash, uint32_t loop) {
		auto* w = words.data() + LoopGuardHeaderDwords + slot * LoopGuardSlotDwords;
		w[0]    = static_cast<uint32_t>(hash);
		w[1]    = static_cast<uint32_t>(hash >> 32u);
		w[2]    = loop;
	};
	set_slot(0, 0x1234abcd5678ef01ull, 12);
	set_slot(1, 0x1234abcd5678ef01ull, 12); // Same loop again.
	set_slot(2, 0x1234abcd5678ef01ull, LoopGuardDispatcherLoop);
	for (uint32_t i = 3; i < LoopGuardSlotCount; i++) set_slot(i, 0x1111ull, 7);
	const auto lines = FormatLoopGuardTrips(words, seen);
	Check(lines.size() == 3u, "duplicates and slots past the area are dropped");
	Check(lines[0].starts_with("[loop-guard] shader=") &&
	          lines[0].find("1234abcd5678ef01 loop=B12 tripped_invocations=70 "
	                        "(first report of this loop)") != std::string::npos,
	      "loop line");
	Check(lines[1].find("loop=dispatcher") != std::string::npos, "dispatcher line");
	Check(lines[2].find("0000000000001111 loop=B7") != std::string::npos, "other shader");
	Check(FormatLoopGuardTrips(words, seen).empty(), "reported loops are not repeated");
	// The count can exceed the slots, never the words that exist.
	words[0] = 3;
	std::set<std::pair<uint64_t, uint32_t>> fresh;
	Check(FormatLoopGuardTrips(words, fresh).size() == 2u, "recorded slots follow the counter");
}

} // namespace

int main() {
	TestCompare();
	TestClassify();
	TestLineAndLimiter();
	TestStoreIdentity();
	TestLoopGuardReport();
	std::puts("BufferProbeTests: ok");
	return 0;
}
