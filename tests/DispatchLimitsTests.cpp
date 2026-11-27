#include "common/fpEnv.h"
#include "graphics/host_gpu/renderer/dispatchLimits.h"
#include "graphics/host_gpu/renderer/dispatchOrigin.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

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

void TestCountsSafeWithoutGuard() {
	const uint32_t limit = DispatchDefaultMaxGroups;
	Check(CountsSafeWithoutGuard(480, 270, 1, 65535, 65535, 65535, limit), "a frame pass needs no guard");
	// Each count is also bounded by the device's own per-dimension limit.
	Check(!CountsSafeWithoutGuard(1, 70000, 1, 65535, 65535, 65535, limit), "y above the device limit");
	Check(CountsSafeWithoutGuard(70000, 1, 1, 0xffffffffu, 65535, 65535, limit), "x within a 32-bit limit");
	Check(!CountsSafeWithoutGuard(0x01000062u, 1, 1, 0xffffffffu, 65535, 65535, limit),
	      "garbage counts within the device limit still exceed the group limit");
	Check(!CountsSafeWithoutGuard(4096, 4096, 1, 0xffffffffu, 65535, 65535, limit), "x * y above the limit");
}

void TestValidationStillHolds() {
	constexpr uint64_t Slots = 65536;
	const ValidatedArgs entry {0x1000, 7, 3, 10, 0xabc};
	Check(ValidationStillHolds(entry, 0x1000, 7, 0xabc, 11, Slots), "same stamp, buffer and address");
	Check(!ValidationStillHolds(entry, 0x1000, 8, 0xabc, 11, Slots), "a write changed the stamp");
	Check(!ValidationStillHolds(entry, 0x1000, 0, 0xabc, 11, Slots), "stamp 0 is never trusted");
	Check(!ValidationStillHolds({0x1000, 0, 3, 10, 0xabc}, 0x1000, 0, 0xabc, 11, Slots),
	      "an entry made without a stamp is never reused");
	Check(!ValidationStillHolds(entry, 0x1004, 7, 0xabc, 11, Slots), "another address");
	Check(!ValidationStillHolds(entry, 0x1000, 7, 0xdef, 11, Slots), "another command buffer");
	Check(ValidationStillHolds(entry, 0x1000, 7, 0xabc, 10 + Slots / 2, Slots), "half a ring later");
	Check(!ValidationStillHolds(entry, 0x1000, 7, 0xabc, 10 + Slots, Slots), "the slot was reused");
	Check(!ValidationStillHolds(entry, 0x1000, 7, 0xabc, 5, Slots), "a ring position from the past");
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
	Check(two > 2.0f && two < 2.03f, "the dword is a float near 2");
	Check(DwordAsFloat(0xbffebae2u) < -1.98f, "the negated twin");
}

void TestLimiter() {
	DispatchReportLimiter limiter(2);
	Check(limiter.Allow(7) && limiter.Allow(7) && !limiter.Allow(7), "two per shader");
	Check(limiter.Allow(8), "another shader has its own count");
	Check(!limiter.Allow(7), "stays limited");
}

void TestThreadCountsOfFloatPatterns() {
	// A thread-dimension dispatch of 64-wide groups: x groups = ceil(threads / 64), y and z as is.
	// The refused lines of the evidence are float bit patterns read as thread counts.
	auto groups_x = [](uint32_t threads) { return threads / 64u + (threads % 64u != 0u); };
	Check(groups_x(0xff800000u) == 0x3fe0000u, "-inf threads are 0x3fe0000 groups");
	Check(groups_x(0x7f800000u) == 0x1fe0000u, "+inf threads are 0x1fe0000 groups");
	Check(groups_x(0xbffebae2u) == 0x2fffaecu, "-1.99 threads are 0x2fffaec groups");
}

void TestDwordWindow() {
	const uint32_t words[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0xc0031502u,
	                            0x7f800000u};
	const auto text = FormatDwordWindow("[d] w ", words, 18, -16);
	Check(text == "[d] w -16: 00000001 00000002 00000003 00000004 00000005 00000006 00000007 "
	              "00000008 00000009 0000000a 0000000b 0000000c 0000000d 0000000e 0000000f "
	              "00000010\n[d] w +0: c0031502 7f800000\n",
	      "16 words per line, indexed from the packet header");
	Check(FormatDwordWindow("x", words, 0, 0).empty(), "nothing for no words");
}

void TestSubmitterCalls() {
	DispatchSubmitter submitter;
	Check(FormatSubmitterCalls(submitter) == "none", "no calls");
	submitter.calls_num = 2;
	submitter.calls[0]  = {"libA", "First", 0x10};
	submitter.calls[1]  = {"libB", "Second", 0x2a};
	Check(FormatSubmitterCalls(submitter) == "libA::First@0x10, libB::Second@0x2a", "oldest first");
	submitter.calls_num = 99;
	Check(!FormatSubmitterCalls(submitter).empty(), "a bad count is clamped");
}

void TestDispatchOriginScope() {
	Check(t_dispatch_origin == nullptr, "no origin outside a dispatch");
	DispatchOrigin outer {};
	{
		const ScopedDispatchOrigin a(&outer);
		Check(t_dispatch_origin == &outer, "set");
		DispatchOrigin inner {};
		{
			const ScopedDispatchOrigin b(&inner);
			Check(t_dispatch_origin == &inner, "nested");
		}
		Check(t_dispatch_origin == &outer, "restored");
	}
	Check(t_dispatch_origin == nullptr, "cleared");
}

void TestFpControl() {
	using Common::FpEnv;
	using Common::FpCheckSite;
	Check(Common::SameFpControl({0x9fc0u | 0x3fu, 0x037f}, Common::GuestFpEnv),
	      "sticky exception flags are not control");
	Check(!Common::SameFpControl({0x1f80u, 0x037f}, Common::GuestFpEnv),
	      "the Linux default (no FTZ, no DAZ) differs");
	Check(!Common::SameFpControl({0x9fc0u | 0x6000u, 0x037f}, Common::GuestFpEnv),
	      "a rounding mode differs");
	Check(!Common::SameFpControl(Common::GuestFpEnv, {0x9fc0u, 0x027f}), "x87 precision differs");
	Check(Common::SameFpControl(Common::GuestFpEnv, {0x9fc0u, 0x037fu | 0x1040u}),
	      "x87 reserved and infinity bits are ignored");
	// First deviation per site and thread.
	std::thread([] {
		Check(!Common::FpControlFirstDeviation(FpCheckSite::Submit, Common::GuestFpEnv), "as expected");
		Check(Common::FpControlFirstDeviation(FpCheckSite::Submit, {0x1f80u, 0x037f}), "first");
		Check(!Common::FpControlFirstDeviation(FpCheckSite::Submit, {0x1f80u, 0x037f}), "once");
		Check(Common::FpControlFirstDeviation(FpCheckSite::GuestFault, {0x1f80u, 0x037f}),
		      "per site");
		Check(Common::FpControlFirstDeviation(FpCheckSite::LibraryChanged, {0x9fc0u, 0x037f},
		                                      {0x1f80u, 0x037f}),
		      "against another expected state");
	}).join();
	std::thread([] {
		Check(Common::FpControlFirstDeviation(FpCheckSite::Submit, {0x1f80u, 0x037f}),
		      "per thread");
	}).join();
	// The state really is what the guest runs with: flush-to-zero turns a denormal product to 0.
	std::thread([] {
		const auto host = Common::ReadFpEnv();
		Common::WriteFpEnv(Common::GuestFpEnv);
		const auto env = Common::ReadFpEnv();
		volatile float tiny  = 1e-30f;
		volatile float scale = 1e-10f;
		volatile float product = tiny * scale;
		Common::WriteFpEnv(host);
		Check(Common::SameFpControl(env, Common::GuestFpEnv), "written state reads back");
		Check(product == 0.0f, "flush-to-zero under the guest state");
	}).join();
}

} // namespace

int main() {
	TestCounts();
	TestJudge();
	TestCountsSafeWithoutGuard();
	TestValidationStillHolds();
	TestLimitText();
	TestCountDword();
	TestLimiter();
	TestThreadCountsOfFloatPatterns();
	TestDwordWindow();
	TestSubmitterCalls();
	TestDispatchOriginScope();
	TestFpControl();
	std::printf("DispatchLimitsTests: ok\n");
	return 0;
}
