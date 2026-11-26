#include "common/hangDump.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

using namespace Common::HangDump;
using std::chrono::milliseconds;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "HangDumpTests: failed: %s\n", message);
		std::abort();
	}
}

void TestEnvDetection() {
	Check(!DriverHangModeActive(nullptr), "null");
	Check(!DriverHangModeActive(""), "empty");
	Check(DriverHangModeActive("hang"), "plain");
	Check(DriverHangModeActive("nocache,hang"), "list tail");
	Check(DriverHangModeActive("hang,umr"), "list head");
	Check(DriverHangModeActive("a, hang ,b"), "spaces");
	Check(!DriverHangModeActive("hanging"), "prefix only");
	Check(!DriverHangModeActive("nohang"), "suffix only");
	Check(!DriverHangModeActive("shaders,info"), "unrelated");
}

// Scripted probe: returns the next state each call, the last one forever.
struct Script {
	std::vector<DumpState> states;
	size_t                 next = 0;
	milliseconds           slept {0};
	DumpState Probe() {
		const auto& s = states[next < states.size() ? next : states.size() - 1];
		++next;
		return s;
	}
};

DumpState State(bool found, uint32_t files, uint64_t bytes, bool last) {
	DumpState s;
	s.found          = found;
	s.path           = found ? "d" : "";
	s.file_count     = files;
	s.total_bytes    = bytes;
	s.has_last_chunk = last;
	return s;
}

WaitResult Run(Script& sc, WaitPolicy p) {
	return WaitForDriverDump(
	    p, [&] { return sc.Probe(); }, [&](milliseconds d) { sc.slept += d; });
}

void TestWait() {
	WaitPolicy p;
	p.timeout            = std::chrono::seconds(10);
	p.poll               = milliseconds(250);
	p.settle             = milliseconds(1000);
	p.settle_incomplete  = std::chrono::seconds(4);

	{ // disabled
		Script sc{{State(false, 0, 0, false)}};
		WaitPolicy off = p;
		off.timeout    = milliseconds(0);
		const auto r   = Run(sc, off);
		Check(r.outcome == WaitOutcome::Disabled && sc.slept.count() == 0, "disabled does not sleep");
	}
	{ // never starts: bounded by the timeout
		Script sc{{State(false, 0, 0, false)}};
		const auto r = Run(sc, p);
		Check(r.outcome == WaitOutcome::NotStarted, "not started");
		Check(sc.slept == std::chrono::seconds(10), "not started waits exactly the timeout");
	}
	{ // grows, then completes and settles
		Script sc{{State(false, 0, 0, false), State(true, 3, 100, false), State(true, 8, 900, false),
		           State(true, 12, 5000, true)}};
		const auto r = Run(sc, p);
		Check(r.outcome == WaitOutcome::Complete, "complete");
		Check(r.state.file_count == 12 && r.state.has_last_chunk, "complete state");
		// 3 growth polls + 4 settle polls of 250 ms
		Check(r.waited == milliseconds(250 * 7), "complete needs one settle after the last change");
	}
	{ // still growing at the timeout
		std::vector<DumpState> v;
		for (uint64_t i = 0; i < 100; ++i) v.push_back(State(true, 3, 100 + i, false));
		Script sc{v};
		const auto r = Run(sc, p);
		Check(r.outcome == WaitOutcome::TimedOut, "timed out");
		Check(sc.slept == std::chrono::seconds(10), "timed out waits exactly the timeout");
	}
	{ // stops changing without the last chunk
		Script sc{{State(true, 5, 10, false)}};
		const auto r = Run(sc, p);
		Check(r.outcome == WaitOutcome::StableIncomplete, "stable incomplete");
		Check(r.waited == std::chrono::seconds(4), "stable incomplete waits its settle time");
	}
	{ // timeout shorter than a poll step is respected
		Script sc{{State(false, 0, 0, false)}};
		WaitPolicy tiny = p;
		tiny.timeout    = milliseconds(100);
		const auto r    = Run(sc, tiny);
		Check(sc.slept == milliseconds(100) && r.outcome == WaitOutcome::NotStarted, "tiny timeout");
	}
}

void TestScan() {
	namespace fs   = std::filesystem;
	const auto root = fs::temp_directory_path() / "kyty_hang_dump_tests";
	std::error_code ec;
	fs::remove_all(root, ec);
	fs::create_directories(root / "radv_dumps_4242_2000-01-01_00.00.00");
	fs::create_directories(root / "radv_dumps_99_x"); // another pid
	std::ofstream(root / "radv_dumps_4242_2000-01-01_00.00.00" / "trace.log") << "abc";

	auto s = ScanDumpFolder(root.string().c_str(), 4242);
	Check(s.found && s.file_count == 1 && s.total_bytes == 3 && !s.has_last_chunk, "partial dump");
	std::ofstream(root / "radv_dumps_4242_2000-01-01_00.00.00" / "dmesg.log") << "xy";
	s = ScanDumpFolder(root.string().c_str(), 4242);
	Check(s.file_count == 2 && s.total_bytes == 5 && s.has_last_chunk, "complete dump");
	Check(!ScanDumpFolder(root.string().c_str(), 7).found, "other pid not matched");
	Check(!ScanDumpFolder((root / "missing").string().c_str(), 4242).found, "missing home");
	fs::remove_all(root, ec);
}

} // namespace

int main() {
	TestEnvDetection();
	TestWait();
	TestScan();
	std::puts("HangDumpTests: ok");
	return 0;
}
