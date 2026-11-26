#include "graphics/host_gpu/renderer/pipeline/shaderWorkers.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>
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

// The default worker count follows the CPU: hardware threads minus four, at least two, and an
// explicit count wins.
void TestWorkerCount() {
	Check(ShaderWorkerCount(16) == 12, "16 threads -> 12 workers");
	Check(ShaderWorkerCount(8) == 4, "8 threads -> 4 workers");
	Check(ShaderWorkerCount(6) == 2 && ShaderWorkerCount(4) == 2 && ShaderWorkerCount(1) == 2 &&
	          ShaderWorkerCount(0) == 2,
	      "small CPUs get the minimum of two");
	Check(ShaderWorkerCount(16, 3) == 3 && ShaderWorkerCount(2, 9) == 9, "an explicit count wins");
}

// Jobs a draw waits for run first, oldest first; the others keep the order of the pools (urgent
// before normal, the oldest first on the single worker).
void TestWaitedFirst() {
	ShaderWorkers      workers(1);
	std::mutex         mutex;
	std::vector<int>   order;
	std::atomic<bool>  release {false};
	std::atomic<bool>  started {false};
	const auto         record = [&](int id) {
        return [&, id] {
            std::scoped_lock lock(mutex);
            order.push_back(id);
        };
	};
	// Holds the only worker while the others are queued.
	workers.Post([&] {
		started = true;
		while (!release.load()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	});
	for (int wait = 0; wait < 5000 && !started.load(); wait++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	Check(started.load(), "the blocker should be running");
	using Priority = ShaderWorkers::Priority;
	workers.Post(record(1));                                  // prefetched
	const auto two   = workers.Post(record(2));               // waited for later
	workers.Post(record(3), Priority::Urgent);                // urgent, not waited for
	workers.Post(record(4), Priority::Normal, /*waited=*/true);
	workers.Post(record(5));                                  // prefetched
	const auto six = workers.Post(record(6), Priority::Urgent);
	workers.MarkWaited(six);                                  // waited for, urgent
	workers.MarkWaited(two);
	release = true;
	for (int wait = 0; wait < 5000 && workers.FinishedJobs() < 7; wait++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::scoped_lock lock(mutex);
	// Waited: 2, 4, 6 by age. Then urgent 3, then the normal ones 1 and 5 by age.
	const std::vector<int> expected {2, 4, 6, 3, 1, 5};
	Check(order == expected, "waited jobs first and oldest first, then urgent, then normal");
}

} // namespace

int main() {
	TestBudget();
	TestFeed();
	TestWorkerCount();
	TestWaitedFirst();
	std::puts("ShaderWorkersTests: ok");
	return 0;
}
