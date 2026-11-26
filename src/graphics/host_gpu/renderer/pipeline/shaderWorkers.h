#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERWORKERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERWORKERS_H_

#include "common/common.h"
#include "common/uniqueFunction.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Milliseconds on a steady clock: when a draw last asked for a job.
[[nodiscard]] inline uint64_t AskClockMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Threads that run queued jobs at below-normal priority, so that the game's own threads go first.
// Urgent jobs go before normal ones. Within each kind a worker takes the job a draw asked for last,
// the scene on screen, instead of the oldest: first-come-first-served let a pipeline drawn every
// frame starve behind the new ones each frame brought. Worker 0 takes the oldest, so that a job
// that was asked for once is not left behind for good. Jobs no draw has asked for in StaleAfterMs
// are taken by the first BackgroundWorkers workers only, so that work nobody sees any more does
// not take the cores the game uses. (The scheduling follows usa-ratt/KytyPS5-GT7's pipeline
// compiler.)
class ShaderWorkers {
public:
	enum class Priority {
		// Translating a shader nothing has started on yet.
		Normal,
		// Finishing a shader that is already translated (module, pipeline). It makes a draw
		// drawable, so it does not wait behind the translations of other shaders.
		Urgent,
	};

	// A queued job; a draw that wants it again calls Touch.
	struct Job {
		Common::UniqueFunction<void> run;
		Priority                     priority = Priority::Normal;
		uint64_t                     sequence = 0;
		std::atomic<uint64_t>        asked {0};

		void Touch() noexcept { asked.store(AskClockMs(), std::memory_order_relaxed); }
	};
	using JobPtr = std::shared_ptr<Job>;

	// Jobs no draw asked for in StaleAfterMs are taken after the fresh ones; every worker takes
	// them, so that a long queue never leaves workers idle (the workers run at lowered priority,
	// which already keeps the game's own threads first).
	static constexpr uint32_t BackgroundWorkers = 2;
	static constexpr uint64_t StaleAfterMs      = 2000;

	explicit ShaderWorkers(uint32_t threads);
	~ShaderWorkers() { Stop(); }
	KYTY_CLASS_NO_COPY(ShaderWorkers);

	// Null once stopped.
	JobPtr Post(Common::UniqueFunction<void>&& job, Priority priority = Priority::Normal);
	// Moves a queued normal-priority job to the urgent queue and refreshes its ask timestamp.
	void Promote(const JobPtr& job);
	// The jobs that have not started.
	[[nodiscard]] size_t Queued();
	[[nodiscard]] bool   Stopped();
	// Workers running a job right now, the jobs finished so far, and the worker count.
	[[nodiscard]] uint32_t Busy() const { return m_busy.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t FinishedJobs() const { return m_finished.load(std::memory_order_relaxed); }
	[[nodiscard]] uint32_t ThreadCount() const { return m_thread_count; }
	// Drops the jobs that have not started and waits for the ones that have.
	void Stop();

private:
	// Under m_mutex: whether worker `index` may take a job now, and takes it.
	[[nodiscard]] bool   CanPick(uint32_t index) const;
	[[nodiscard]] JobPtr Pick(uint32_t index);

	std::mutex                  m_mutex;
	std::condition_variable_any m_ready;
	std::vector<JobPtr>         m_jobs;
	std::vector<JobPtr>         m_urgent_jobs;
	uint64_t                    m_next_sequence = 0;
	bool                        m_stopped       = false;
	std::atomic<uint32_t>       m_busy {0};
	std::atomic<uint64_t>       m_finished {0};
	uint32_t                    m_thread_count = 0;
	// Declared last, so the threads exist only once everything they use does.
	std::vector<std::jthread> m_threads;
};


// Applies the front of `pending` (calling each element) until `budget_us` microseconds have passed
// on `now_us`, but always at least one so a backlog drains. Budget 0 applies everything. Returns
// how many were applied; the rest stay at the front of `pending`.
template <typename Queue, typename Now>
size_t ApplyWithinBudget(Queue& pending, uint64_t budget_us, Now&& now_us) {
	const auto begin   = now_us();
	size_t     applied = 0;
	while (!pending.empty()) {
		auto apply = std::move(pending.front());
		pending.pop_front();
		apply();
		applied++;
		if (budget_us != 0 && now_us() - begin >= budget_us) {
			break;
		}
	}
	return applied;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERWORKERS_H_
