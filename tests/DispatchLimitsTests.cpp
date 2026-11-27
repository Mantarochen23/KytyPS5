#include "graphics/host_gpu/renderer/dispatchLimits.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DispatchLimitsTests: failed: %s\n", message);
		std::abort();
	}
}

void TestCounts() {
	// The count is a full 32-bit value: nothing in the upper byte is a flag.
	Check(DispatchGroups(0x01000062u, 1, 1) == 16777314ull, "the upper byte is part of the count");
	Check(DispatchInvocations(0x01000062u, 1, 1, 64) == 16777314ull * 64u, "invocations");
	Check(DispatchInvocations(UINT32_MAX, UINT32_MAX, UINT32_MAX, 1024) == UINT64_MAX,
	      "invocations saturate");
	Check(DispatchGroups(UINT32_MAX, UINT32_MAX, UINT32_MAX) > UINT32_MAX, "groups do not wrap");
}

void TestJudge() {
	const uint32_t limit = DispatchDefaultMaxGroups;
	// A full-screen 4K pass of 8x8 groups.
	Check(JudgeDispatch(480, 270, 1, 64, limit, true) == DispatchVerdict::Run, "a frame pass runs");
	Check(JudgeDispatch(1, 1, 1, 1024, limit, true) == DispatchVerdict::Run, "one group runs");
	// Large but within the limit: named, still runs.
	Check(JudgeDispatch(1u << 20u, 1, 1, 64, limit, true) == DispatchVerdict::Report,
	      "67M invocations are reported");
	Check(JudgeDispatch(limit, 1, 1, 64, limit, true) == DispatchVerdict::Report,
	      "exactly the limit still runs");
	Check(JudgeDispatch(limit + 1, 1, 1, 64, limit, true) == DispatchVerdict::Skip,
	      "above the limit is skipped");
	Check(JudgeDispatch(0x01000062u, 1, 1, 64, limit, true) == DispatchVerdict::Skip,
	      "16.7 million groups are skipped");
	Check(JudgeDispatch(0x02fffaecu, 1, 1, 64, limit, true) == DispatchVerdict::Skip,
	      "50 million groups are skipped");
	// The groups multiply across axes.
	Check(JudgeDispatch(4096, 4096, 1, 1, limit, true) == DispatchVerdict::Skip,
	      "x * y above the limit is skipped");
	// With the guard port off nothing is skipped, but it is still named.
	Check(JudgeDispatch(0x01000062u, 1, 1, 64, limit, false) == DispatchVerdict::Report,
	      "disabled guard only reports");
	Check(JudgeDispatch(1000, 1, 1, 64, limit, false) == DispatchVerdict::Run, "small stays quiet");
}

void TestLimitText() {
	Check(ParseDispatchGroupLimit(nullptr) == DispatchDefaultMaxGroups, "unset");
	Check(ParseDispatchGroupLimit("") == DispatchDefaultMaxGroups, "empty");
	Check(ParseDispatchGroupLimit("65536") == 65536u, "decimal");
	Check(ParseDispatchGroupLimit("0x10000") == 65536u, "hex");
	Check(ParseDispatchGroupLimit("12") == DispatchDefaultMaxGroups, "below the floor");
	Check(ParseDispatchGroupLimit("99999999999") == DispatchDefaultMaxGroups, "above 32 bits");
	Check(ParseDispatchGroupLimit("12abc") == DispatchDefaultMaxGroups, "trailing text");
}

void TestCountDword() {
	// Four user dwords of a dispatch: a float count, ceil(count / 64) is the x count.
	const uint32_t words[4] = {0x012e000cu, 0u, 0x40014526u, 1u};
	Check(FindCountDword(words, 4, 0x01000515u, 64) == 2, "finds the dword the count came from");
	Check(FindCountDword(words, 4, 0x01000516u, 64) == -1, "no dword matches");
	Check(FindCountDword(words, 4, 1, 64) == -1, "a count of one matches nothing");
	Check(FindCountDword(words, 4, 5, 0) == -1, "no zero group size");
	const float two = DwordAsFloat(0x40014526u);
	Check(two > 2.0f && two < 2.01f, "the dword is a float near 2");
	Check(DwordAsFloat(0xbffebae2u) < -1.99f, "the negated twin");
}

void TestLimiter() {
	DispatchReportLimiter limiter(2);
	Check(limiter.Allow(7) && limiter.Allow(7) && !limiter.Allow(7), "two per shader");
	Check(limiter.Allow(8), "another shader has its own count");
	Check(!limiter.Allow(7), "stays limited");
}

} // namespace

int main() {
	TestCounts();
	TestJudge();
	TestLimitText();
	TestCountDword();
	TestLimiter();
	std::printf("DispatchLimitsTests: ok\n");
	return 0;
}
