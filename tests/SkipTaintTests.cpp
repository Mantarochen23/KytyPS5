#include "graphics/host_gpu/renderer/pipeline/skipTaint.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;
using Use = SkipTaint::Use;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SkipTaintTests: failed: %s\n", message);
		std::abort();
	}
}

void TestFollowsData() {
	SkipTaint taint;
	Check(taint.Empty() && !taint.ReadsTainted({{0x1000, 64, true, false}}, 0), "empty reads nothing");
	// The skipped producer writes B and reads A; A is not tainted.
	taint.Taint({{0x1000, 256, true, false}, {0x2000, 256, false, true}}, 100);
	Check(taint.Size() == 1, "only written ranges are tainted");
	Check(!taint.ReadsTainted({{0x1000, 256, true, false}}, 110), "the producer's input is clean");
	Check(taint.ReadsTainted({{0x2080, 4, true, false}}, 110), "a read inside the output is tainted");
	Check(taint.ReadsTainted({{0x1f00, 0x200, true, false}}, 110), "a read overlapping the edge is tainted");
	Check(!taint.ReadsTainted({{0x2100, 16, true, false}}, 110), "a read past the end is clean");
	Check(!taint.ReadsTainted({{0x2000, 256, false, true}}, 110), "a pure write does not read");
}

void TestClearAndSplit() {
	SkipTaint taint;
	taint.Taint({{0x2000, 0x300, false, true}}, 0);
	taint.ClearWritten({{0x2100, 0x100, false, true}});
	Check(taint.Size() == 2, "a write inside splits the range");
	Check(taint.ReadsTainted({{0x2000, 4, true, false}}, 1), "the head stays tainted");
	Check(!taint.ReadsTainted({{0x2100, 0x100, true, false}}, 1), "the written middle is clean");
	Check(taint.ReadsTainted({{0x22fc, 4, true, false}}, 1), "the tail stays tainted");
	taint.ClearWritten({{0x1000, 0x2000, false, true}});
	Check(taint.Empty(), "a covering write clears everything");
	taint.Taint({{0x3000, 16, false, true}}, 0);
	taint.ClearWritten({{0x3000, 16, true, false}});
	Check(taint.Size() == 1, "a read-only use clears nothing");
}

void TestTransitiveAndExpiry() {
	SkipTaint taint(1000);
	taint.Taint({{0x1000, 64, false, true}}, 0);
	const std::vector<Use> consumer {{0x1000, 64, true, false}, {0x5000, 64, false, true}};
	Check(taint.ReadsTainted(consumer, 10), "the consumer reads tainted data");
	taint.Taint(consumer, 10); // it is skipped: its output is tainted too
	Check(taint.ReadsTainted({{0x5000, 4, true, false}}, 20), "taint follows the data");
	Check(taint.ReadsTainted({{0x5000, 4, true, false}}, 1005), "the second range is younger");
	Check(!taint.ReadsTainted({{0x1000, 4, true, false}}, 1005), "the first range expired");
	Check(!taint.ReadsTainted({{0x5000, 4, true, false}}, 1011), "the second expired too");
	Check(taint.Empty(), "expired ranges are dropped");
}

} // namespace

int main() {
	TestFollowsData();
	TestClearAndSplit();
	TestTransitiveAndExpiry();
	std::printf("SkipTaintTests: ok\n");
	return 0;
}
