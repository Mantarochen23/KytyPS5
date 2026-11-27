#include "graphics/host_gpu/renderer/pipeline/buildTiming.h"
#include "graphics/host_gpu/renderer/pipeline/flowPlan.h"
#include "graphics/host_gpu/renderer/pipeline/frameHoles.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Libs::Graphics;
using FlowPlan::WaitAction;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FlowPlanTests: failed: %s\n", message);
		std::abort();
	}
}

void TestWaitForEver() {
	FlowPlan::DispatchWait wait;
	for (uint64_t now = 100; now < 100000; now += 50) {
		Check(wait.NotReady(0, now).action == WaitAction::Wait, "limit 0 never skips");
	}
}

void TestBoundedWait() {
	FlowPlan::DispatchWait wait;
	auto                   step = wait.NotReady(1000, 5000);
	Check(step.action == WaitAction::Wait && step.waited_ms == 0, "the first call starts the clock");
	step = wait.NotReady(1000, 5400);
	Check(step.action == WaitAction::Wait && step.waited_ms == 400, "waits below the limit");
	step = wait.NotReady(1000, 5999);
	Check(step.action == WaitAction::Wait, "one ms short of the limit still waits");
	step = wait.NotReady(1000, 6000);
	Check(step.action == WaitAction::Skip && step.waited_ms == 1000, "skips at the limit");
	// The skipped dispatch is over: the next one starts a new clock.
	step = wait.NotReady(1000, 6050);
	Check(step.action == WaitAction::Wait && step.waited_ms == 0, "a new dispatch waits afresh");
}

void TestDoneResets() {
	FlowPlan::DispatchWait wait;
	(void)wait.NotReady(100, 0);
	(void)wait.NotReady(100, 90);
	wait.Done(); // the shader came in time
	const auto step = wait.NotReady(100, 150);
	Check(step.action == WaitAction::Wait && step.waited_ms == 0, "a finished dispatch resets");
}

void TestClockGoingBack() {
	FlowPlan::DispatchWait wait;
	(void)wait.NotReady(10, 500);
	Check(wait.NotReady(10, 400).action == WaitAction::Wait, "a clock that went back never skips");
}

void TestSkipLedger() {
	FlowPlan::SkipLedger ledger;
	Check(ledger.Note(0xA, 1000) == 1, "first use of a hash is number 1");
	Check(ledger.Note(0xB, 1000) == 1, "another hash starts at 1");
	Check(ledger.Note(0xA, 1200) == 2, "second use of the first hash");
	Check(ledger.Total() == 3 && ledger.Hashes() == 2, "totals");
	Check(ledger.WaitedMs() == 3200, "waited time adds up");
	Check(ledger.TakePeriod() == 3, "the period counter holds the skips since the last report");
	Check(ledger.TakePeriod() == 0, "and starts again");
	ledger.Note(0xC, 1);
	Check(ledger.TakePeriod() == 1 && ledger.Total() == 4, "total keeps counting");
	Check(ledger.Uses(0xA) == 2 && ledger.Uses(0xD) == 0, "uses per hash");
}

void TestUseCounter() {
	FlowPlan::UseCounter served;
	Check(served.Note(1) == 1, "first use is announced");
	Check(served.Note(1) == 2 && served.Note(1) == 3, "later uses only count");
	Check(served.Note(2) == 1, "another hash");
	Check(served.Hashes() == 2 && served.Uses() == 4, "totals");
}

void TestRequestCounter() {
	FlowPlan::RequestCounter requests;
	Check(requests.Begin(7), "first call is a request");
	requests.Retry(7); // it ended without a module
	int counted = 0;
	for (int i = 0; i < 100; i++) {
		if (requests.Begin(7)) {
			counted++;
		}
		requests.Retry(7); // still waiting
	}
	Check(counted == 0, "100 retries are not new requests");
	Check(!requests.Begin(7), "the last retry is still a retry");
	Check(requests.Begin(7), "after it was served, the next use is a new request");
	Check(requests.Open() == 0, "nothing stays open");
}

void TestUserDataTracker() {
	constexpr uint32_t first = 0x240;
	FlowPlan::UserDataTracker data;
	Check(data.Covers(0), "no user data needs nothing");
	Check(!data.Covers(2), "unseen data is not known");
	Check(data.Note(first, 11, first), "a user data register");
	Check(!data.Covers(2), "one of two");
	Check(data.Note(first + 1 + 0x10000000u, 22, first), "type bits are ignored");
	Check(data.Covers(2) && !data.Covers(3), "both seen");
	Check(!data.Note(first - 1, 5, first), "a register before the block");
	Check(!data.Note(first + 16, 5, first), "a register after the block");
	Check(data.Value(0) == 11 && data.Value(1) == 22, "values");
	Check(!data.Covers(17), "more than the block is never known");
	data.Reset();
	Check(!data.Covers(1), "reset forgets");
	// The scan of a command stream: later writes win.
	data.Note(first, 1, first);
	data.Note(first, 2, first);
	Check(data.Covers(1) && data.Value(0) == 2, "the last write counts");
}

void TestBuildLine() {
	BuildTiming::Build build {.stage  = "cs",
	                          .hash   = 0x1234,
	                          .words  = 437000,
	                          .steps  = {.translate_us   = 30000,
	                                     .plan_us        = 4000,
	                                     .emit_us        = 110000,
	                                     .module_us      = 3000000,
	                                     .pipeline_us    = 500000,
	                                     .queue_wait_us  = 2000,
	                                     .total_us       = 3650000},
	                          .where  = "worker",
	                          .reason = "dispatch"};
	const auto line = BuildTiming::Line(build);
	// The hash is printed with sixteen digits.
	Check(line == "[build] cs hash=0x" + std::string(12, '0') + "1234 words=437000 "
	                  "translate=30.0ms plan=4.0ms emit=110.0ms module=3000.0ms pipeline=500.0ms "
	                  "queue-wait=2.0ms total=3650.0ms where=worker reason=dispatch",
	      "build line");
	build.number = 3;
	Check(BuildTiming::Line(build).ends_with(" repeat=#3"), "a repeat is named");
	build.number = 1;
	Check(BuildTiming::Line(build).find("repeat") == std::string::npos, "the first build is not a repeat");

	BuildTiming::Totals totals;
	Check(totals.Empty(), "empty");
	totals.Note(build);
	BuildTiming::Build pipeline {.stage = "pipeline-cs", .hash = 1, .words = 0,
	                             .steps = {.pipeline_us = 700000, .total_us = 700000},
	                             .where = "render", .reason = "dispatch"};
	totals.Note(pipeline);
	Check(totals.modules == 1 && totals.pipelines == 1, "a build without words is a pipeline");
	Check(totals.module_us == 3000000 && totals.pipeline_us == 1200000, "sums");
	Check(totals.render_builds == 1 && totals.render_us == 700000, "render thread builds");
	Check(!totals.Empty(), "not empty");
	const auto text = totals.Text();
	Check(text.find("1 modules") != std::string::npos &&
	          text.find("1 pipelines") != std::string::npos &&
	          text.find("1 on the render thread") != std::string::npos,
	      "totals text");
}

void TestFrameHolesSummary() {
	FrameHoles::Tracker tracker;
	Check(tracker.TakeFrameLine(1).empty(), "complete frame emits no frame-holes line");
	const uint64_t rt_addrs[]  = {0x100010000ull, 0x100020000ull};
	const uint64_t buf_addrs[] = {0x100020000ull, 0x100030000ull};
	tracker.RecordDraw(FrameHoles::DrawSkipReason::ShaderBuilding, rt_addrs);
	tracker.RecordDraw(FrameHoles::DrawSkipReason::FailedShader, {rt_addrs, 1});
	tracker.RecordDispatch(FrameHoles::DispatchSkipReason::Wait, buf_addrs);
	tracker.RecordDispatch(FrameHoles::DispatchSkipReason::Tainted, {buf_addrs + 1, 1});
	tracker.RecordDispatch(FrameHoles::DispatchSkipReason::Refused);
	const auto line = tracker.TakeFrameLine(42);
	Check(line.find("[frame-holes] frame=42 draws skipped=2 (shader-building=1, failed-shader=1)") !=
	          std::string::npos,
	      "draw skip reasons counted");
	Check(line.find("dispatches skipped=3 (wait=1, tainted=1, refused=1)") != std::string::npos,
	      "dispatch skip reasons counted");
	Check(line.find("targets affected=3 (0x000100010000, 0x000100020000, 0x000100030000)") !=
	          std::string::npos,
	      "distinct targets deduplicated and formatted");
	Check(tracker.TakeFrameLine(43).empty(), "counters reset after frame flush");
}

} // namespace

int main() {
	TestWaitForEver();
	TestBoundedWait();
	TestDoneResets();
	TestClockGoingBack();
	TestSkipLedger();
	TestUseCounter();
	TestRequestCounter();
	TestUserDataTracker();
	TestBuildLine();
	TestFrameHolesSummary();
	std::printf("FlowPlanTests: ok\n");
	return 0;
}
