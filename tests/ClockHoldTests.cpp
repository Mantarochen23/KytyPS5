#include "common/clockHold.h"
#include "common/guestPark.h"
#include "graphics/host_gpu/renderer/pipeline/renderActivity.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using Common::ClockHold;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ClockHoldTests: failed: %s\n", message);
		std::abort();
	}
}

constexpr uint64_t MS = 1000000;

// What a guest reads at raw time t.
uint64_t Guest(ClockHold& hold, uint64_t raw) {
	return hold.Apply(ClockHold::Domain::Monotonic, raw, hold.HeldNs(raw));
}

void TestNothingHeldIsTheRawClock() {
	ClockHold hold;
	Check(Guest(hold, 5 * MS) == 5 * MS, "no hold: raw value");
	Check(hold.HeldNs(1000 * MS) == 0, "no hold: nothing held");
	Check(hold.Update(10 * MS, false) == ClockHold::Change::None, "no change when not wanted");
}

void TestHoldFreezesAndResumesWithoutJumping() {
	ClockHold hold;
	Check(hold.Update(100 * MS, true) == ClockHold::Change::Started, "hold starts");
	Check(Guest(hold, 100 * MS) == 100 * MS, "the clock reads the same at the start");
	Check(Guest(hold, 150 * MS) == 100 * MS, "frozen 50 ms into the hold");
	Check(Guest(hold, 900 * MS) == 100 * MS, "frozen 800 ms into the hold");
	Check(hold.Update(900 * MS, true) == ClockHold::Change::None, "still held: no change");
	uint64_t ended = 0;
	Check(hold.Update(1000 * MS, false, &ended) == ClockHold::Change::Ended, "hold ends");
	Check(ended == 900 * MS, "the hold lasted 900 ms");
	Check(Guest(hold, 1000 * MS) == 100 * MS, "no jump at the end");
	Check(Guest(hold, 1010 * MS) == 110 * MS, "continues from where it stopped");
	Check(hold.HeldNs(2000 * MS) == 900 * MS, "the held time stays subtracted for good");
	// A second hold adds up.
	Check(hold.Update(2000 * MS, true) == ClockHold::Change::Started, "second hold");
	Check(hold.Update(2500 * MS, false) == ClockHold::Change::Ended, "second hold ends");
	const auto state = hold.Read(3000 * MS);
	Check(!state.holding && state.holds == 2 && state.completed_ns == 1400 * MS, "totals add up");
	Check(Guest(hold, 3000 * MS) == 3000 * MS - 1400 * MS, "guest = raw - total held");
}

void TestNeverGoesBackwardsOrForwardsOfRaw() {
	ClockHold hold;
	uint64_t  previous = 0;
	uint64_t  raw      = 0;
	for (int step = 0; step < 5000; step++) {
		raw += 3 * MS;
		const bool want = (step / 97) % 2 == 1;
		hold.Update(raw, want);
		const auto guest = Guest(hold, raw);
		Check(guest >= previous, "the guest clock never goes backwards");
		Check(guest <= raw, "the guest clock never runs ahead of the raw clock");
		previous = guest;
	}
}

void TestLateReadingIsClamped() {
	ClockHold hold;
	hold.Update(100 * MS, true);
	hold.Update(300 * MS, false);
	// A thread that took its raw reading at 200 ms (during the hold) and applies it after the hold
	// ended would see the whole hold subtracted; the clamp keeps the value already handed out.
	const auto first = Guest(hold, 310 * MS);
	const auto late  = Guest(hold, 200 * MS);
	Check(late >= first || late == first, "a stale reading cannot move the clock back");
}

void TestDomainsAreClampedSeparately() {
	ClockHold hold;
	hold.Update(0, true);
	Check(hold.Apply(ClockHold::Domain::Tsc, 100, 0) == 100, "tsc domain");
	Check(hold.Apply(ClockHold::Domain::Realtime, 5, 0) == 5, "realtime domain keeps its own last value");
}

void TestRepeatedReadsAreCounted() {
	ClockHold hold;
	Check(Guest(hold, 5 * MS) == 5 * MS && hold.RepeatedReads() == 0, "never held: not counted");
	hold.Update(100 * MS, true);
	(void)Guest(hold, 100 * MS);
	const auto before = hold.RepeatedReads();
	(void)Guest(hold, 150 * MS);
	(void)Guest(hold, 200 * MS);
	Check(hold.RepeatedReads() == before + 2, "two frozen reads are two zero deltas");
	hold.Update(300 * MS, false);
	(void)Guest(hold, 310 * MS);
	const auto after = hold.RepeatedReads();
	(void)Guest(hold, 320 * MS);
	Check(hold.RepeatedReads() == after, "a moving clock adds none");
}

void TestTicks() {
	Check(Common::HeldNsToTicks(1000000000u, 1600000000u) == 1600000000u, "1 s = 1.6 G ticks");
	Check(Common::HeldNsToTicks(500 * MS, 1000000000u) == 500 * MS, "1 GHz is one tick per ns");
	Check(Common::HeldNsToTicks(UINT64_MAX / 2, 1600000000u) > 0, "no overflow");
}

void TestShouldHold() {
	Check(!Common::ShouldHoldClock(60000, 0, true), "threshold 0 = off");
	Check(!Common::ShouldHoldClock(900, 1000, true), "not long enough");
	Check(!Common::ShouldHoldClock(5000, 1000, false), "render not busy with shaders");
	Check(Common::ShouldHoldClock(5000, 1000, true), "stalled and busy");
}

void TestBusyClassification() {
	using Libs::Graphics::RenderActivity::BusyWithShaders;
	using Libs::Graphics::RenderActivity::Snapshot;
	Snapshot s {"dispatch-shader", "", 0, "cs", 0, 0, 0};
	Check(BusyWithShaders(s), "a dispatch waiting for its shader");
	s = {"wait-async", "waiting for a worker job", 0, "-", 0, 0, 0};
	Check(BusyWithShaders(s), "waiting for a worker");
	s = {"cp-blocked", "every queue waits for a shader or pipeline", 0, "-", 0, 0, 0};
	Check(BusyWithShaders(s), "all queues suspended for shaders");
	s = {"cp-blocked", "a queue waits on guest memory or a flip", 0, "-", 0, 0, 0};
	Check(!BusyWithShaders(s), "a queue waiting on guest memory is not shader work");
	s = {"cp-no-work", "nothing submitted", 0, "-", 0, 0, 0};
	Check(!BusyWithShaders(s), "idle");
	s = {"pm4", "packets", 0, "-", 0, 0, 0};
	Check(!BusyWithShaders(s), "plain packet processing");
}

void TestParkBookkeeping() {
	Common::GuestPark::Registry registry;
	Check(registry.Count() == 0, "starts empty");
	auto a = registry.Add(0x1000, 0, 7, "worker");
	Check(a.count == 1 && a.first_at_pc && a.at_pc == 1, "first fault at a pc");
	auto b = registry.Add(0x1000, 0, 8, "other");
	Check(b.count == 2 && !b.first_at_pc && b.at_pc == 2, "a second thread at the same pc parks too");
	auto c = registry.Add(0x2000, 0, 9, nullptr);
	Check(c.count == 3 && c.first_at_pc && c.at_pc == 1, "a new pc is a first again");
	Check(registry.Count() == 3, "three parked");
}

void TestParkBookkeepingIsThreadSafe() {
	Common::GuestPark::Registry registry;
	std::vector<std::thread>    threads;
	std::atomic<uint32_t>       firsts {0};
	for (int t = 0; t < 8; t++) {
		threads.emplace_back([&registry, &firsts, t] {
			for (int i = 0; i < 20; i++) {
				if (registry.Add(0x4000, 0, t, "t").first_at_pc) {
					firsts++;
				}
			}
		});
	}
	for (auto& thread: threads) {
		thread.join();
	}
	Check(registry.Count() == 160, "every fault counted");
	Check(firsts.load() == 1, "exactly one first fault at the pc");
}

void TestConcurrentReadersSeeAConsistentClock() {
	ClockHold         hold;
	std::atomic<bool> stop {false};
	std::atomic<bool> bad {false};
	std::vector<std::thread> readers;
	std::atomic<uint64_t> raw {0};
	for (int t = 0; t < 3; t++) {
		readers.emplace_back([&] {
			uint64_t previous = 0;
			while (!stop.load()) {
				const auto now   = raw.load();
				const auto guest = hold.Apply(ClockHold::Domain::Monotonic, now, hold.HeldNs(now));
				if (guest < previous || guest > raw.load()) {
					bad = true;
				}
				previous = guest;
			}
		});
	}
	for (uint64_t step = 1; step < 20000; step++) {
		raw.store(step * MS);
		hold.Update(step * MS, (step / 50) % 2 == 1);
	}
	stop = true;
	for (auto& thread: readers) {
		thread.join();
	}
	Check(!bad.load(), "concurrent readers never see the clock move backwards or ahead");
}

} // namespace

int main() {
	TestNothingHeldIsTheRawClock();
	TestHoldFreezesAndResumesWithoutJumping();
	TestNeverGoesBackwardsOrForwardsOfRaw();
	TestLateReadingIsClamped();
	TestDomainsAreClampedSeparately();
	TestRepeatedReadsAreCounted();
	TestTicks();
	TestShouldHold();
	TestBusyClassification();
	TestParkBookkeeping();
	TestParkBookkeepingIsThreadSafe();
	TestConcurrentReadersSeeAConsistentClock();
	std::printf("ClockHoldTests: ok\n");
	return 0;
}
