#include "graphics/host_gpu/renderer/pipeline/shaderWorkers.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <thread>

using namespace Libs::Graphics;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ShaderWorkersTests: failed: %s\n", message);
		std::abort();
	}
}

// The budget applies at least one job, stops once the clock passes it, and 0 means no limit.
void TestBudget() {
	uint64_t                           clock = 0;
	const auto                         now   = [&] { return clock; };
	std::deque<std::function<void()>> pending;
	int                                ran = 0;
	for (int i = 0; i < 10; i++) {
		pending.push_back([&] {
			ran++;
			clock += 400; // each job costs 400 us
		});
	}
	Check(ApplyWithinBudget(pending, 1000, now) == 3 && ran == 3 && pending.size() == 7,
	      "a 1 ms budget should apply three 400 us jobs");
	Check(ApplyWithinBudget(pending, 1, now) == 1, "one job is always applied");
	Check(ApplyWithinBudget(pending, 0, now) == 6 && pending.empty() && ran == 10,
	      "budget 0 applies everything");
	Check(ApplyWithinBudget(pending, 1000, now) == 0, "an empty backlog applies nothing");
}

// Every worker takes queued jobs, including ones nobody asked for lately.
void TestFeed() {
	constexpr uint32_t threads = 6;
	ShaderWorkers      workers(threads);
	std::atomic<int>   running {0};
	std::atomic<int>   peak {0};
	std::atomic<int>   done {0};
	std::atomic<bool>  release {false};
	constexpr int      jobs = 24;
	std::vector<ShaderWorkers::JobPtr> posted;
	for (int i = 0; i < jobs; i++) {
		posted.push_back(workers.Post([&] {
			const int now = ++running;
			int       old = peak.load();
			while (now > old && !peak.compare_exchange_weak(old, now)) {
			}
			while (!release.load()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			--running;
			done++;
		}));
		// Jobs that have gone stale before any worker looks at them.
		posted.back()->asked.store(AskClockMs() - 10 * ShaderWorkers::StaleAfterMs);
	}
	for (int wait = 0; wait < 2000 && workers.Busy() < threads; wait++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	Check(workers.Busy() == threads, "all workers should be busy while jobs are queued");
	Check(workers.Queued() == jobs - threads, "the rest should still be queued");
	release = true;
	for (int wait = 0; wait < 5000 && done.load() < jobs; wait++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	Check(done.load() == jobs, "all jobs should finish");
	Check(workers.FinishedJobs() >= static_cast<uint64_t>(jobs) - threads,
	      "finished jobs are counted");
	Check(peak.load() == static_cast<int>(threads), "the peak parallelism is the worker count");
}

} // namespace

int main() {
	TestBudget();
	TestFeed();
	std::puts("ShaderWorkersTests: ok");
	return 0;
}
