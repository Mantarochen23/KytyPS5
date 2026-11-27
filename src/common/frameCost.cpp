#include "common/frameCost.h"

#include "common/logging/log.h"

#include <array>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <vector>

namespace Common::FrameCost {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t CounterCount = static_cast<size_t>(Counter::Count);

struct ThreadCounters {
	std::array<std::atomic<uint64_t>, CounterCount> values {};
};

std::mutex                   g_threads_mutex;
std::vector<ThreadCounters*> g_threads;
std::mutex                   g_report_mutex;
std::array<uint64_t, CounterCount> g_previous {};
Clock::time_point                  g_window_start {};

ThreadCounters& Local() {
	thread_local ThreadCounters* counters = [] {
		auto* created = new ThreadCounters;
		std::scoped_lock lock {g_threads_mutex};
		g_threads.push_back(created);
		return created;
	}();
	return *counters;
}

std::array<uint64_t, CounterCount> Snapshot() {
	std::array<uint64_t, CounterCount> result {};
	std::scoped_lock lock {g_threads_mutex};
	for (const auto* counters: g_threads) {
		for (size_t i = 0; i < result.size(); ++i) {
			result[i] += counters->values[i].load(std::memory_order_relaxed);
		}
	}
	return result;
}

double Mean(const std::array<uint64_t, CounterCount>& delta, Counter counter,
            uint64_t frames) {
	return frames == 0 ? 0.0 : static_cast<double>(delta[static_cast<size_t>(counter)]) /
	                                static_cast<double>(frames);
}

double Milliseconds(const std::array<uint64_t, CounterCount>& delta, Counter counter,
	                uint64_t frames) {
	return Mean(delta, counter, frames) / 1'000'000.0;
}

} // namespace

void Add(Counter counter, uint64_t value) noexcept {
	Local().values[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

ScopedTimer::ScopedTimer(Counter duration) noexcept: m_duration(duration), m_start(Clock::now()) {}

ScopedTimer::~ScopedTimer() {
	Stop();
}

void ScopedTimer::Stop() noexcept {
	if (m_start == Clock::time_point {}) return;
	const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_start);
	m_start            = {};
	Add(m_duration, static_cast<uint64_t>(elapsed.count()));
}

void FramePresented() noexcept {
	Add(Counter::Frames);
	const auto now = Clock::now();
	std::scoped_lock report_lock {g_report_mutex};
	if (g_window_start == Clock::time_point {}) {
		g_window_start = now;
		g_previous     = Snapshot();
		return;
	}
	const auto elapsed = now - g_window_start;
	if (elapsed < std::chrono::seconds(5)) return;

	const auto totals = Snapshot();
	std::array<uint64_t, CounterCount> delta {};
	for (size_t i = 0; i < delta.size(); ++i) delta[i] = totals[i] - g_previous[i];
	const auto frames = delta[static_cast<size_t>(Counter::Frames)];
	if (frames != 0) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[frame] seconds={:.2f} frames={} draws={:.2f} dispatches={:.2f} "
		    "rebind={:.2f} entries={:.2f} resolve={:.2f} find_image={:.2f} "
		    "find_hits={:.2f} materialize={:.2f} find_module={:.2f} "
		    "ms/frame rebind={:.3f} resolve={:.3f} find_image={:.3f} "
		    "materialize={:.3f} find_module={:.3f} wait_gpu={:.3f} wait_guest={:.3f} "
		    "wait_lock={:.3f} idle={:.3f}\n",
		    std::chrono::duration<double>(elapsed).count(), frames,
		    Mean(delta, Counter::Draws, frames), Mean(delta, Counter::Dispatches, frames),
		    Mean(delta, Counter::ImageHeapRebindCalls, frames),
		    Mean(delta, Counter::ImageHeapEntriesVisited, frames),
		    Mean(delta, Counter::ResolveTextureCalls, frames),
		    Mean(delta, Counter::FindImageCalls, frames), Mean(delta, Counter::FindImageHits, frames),
		    Mean(delta, Counter::MaterializeResourcesCalls, frames),
		    Mean(delta, Counter::FindModuleCalls, frames),
		    Milliseconds(delta, Counter::ImageHeapRebindNs, frames),
		    Milliseconds(delta, Counter::ResolveTextureNs, frames),
		    Milliseconds(delta, Counter::FindImageNs, frames),
		    Milliseconds(delta, Counter::MaterializeResourcesNs, frames),
		    Milliseconds(delta, Counter::FindModuleNs, frames),
		    Milliseconds(delta, Counter::GpuFenceWaitNs, frames),
		    Milliseconds(delta, Counter::GuestMemoryWaitNs, frames),
		    Milliseconds(delta, Counter::LockWaitNs, frames),
		    Milliseconds(delta, Counter::IdleNs, frames)));
	}
	g_previous     = totals;
	g_window_start = now;
}

} // namespace Common::FrameCost
