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
thread_local ScopedTimer*          g_active_timer = nullptr;

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

double NanosecondsPerCall(const std::array<uint64_t, CounterCount>& delta, Counter duration,
	                      Counter calls) {
	const auto count = delta[static_cast<size_t>(calls)];
	return count == 0 ? 0.0 : static_cast<double>(delta[static_cast<size_t>(duration)]) /
	                            static_cast<double>(count);
}

} // namespace

void Add(Counter counter, uint64_t value) noexcept {
	Local().values[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

ScopedTimer::ScopedTimer(Counter duration) noexcept
    : m_duration(duration), m_start(Clock::now()), m_parent(g_active_timer) {
	g_active_timer = this;
}

ScopedTimer::~ScopedTimer() {
	Stop();
}

void ScopedTimer::Stop() noexcept {
	if (m_start == Clock::time_point {}) return;
	const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_start);
	m_start            = {};
	const auto elapsed_ns = static_cast<uint64_t>(elapsed.count());
	if (g_active_timer == this) g_active_timer = m_parent;
	if (m_parent != nullptr) m_parent->m_child_ns += elapsed_ns;
	Add(m_duration, elapsed_ns - std::min(elapsed_ns, m_child_ns));
}

namespace {
std::atomic<bool> g_store_loaded {false};
double            g_bad_seconds = 0.0; // under g_report_mutex
bool              g_bad_warned  = false;
} // namespace

void NoteStoreLoaded() noexcept {
	g_store_loaded.store(true, std::memory_order_relaxed);
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
		const auto total_ms =
		    std::chrono::duration<double, std::milli>(elapsed).count() / static_cast<double>(frames);
		const auto rebind_ms = Milliseconds(delta, Counter::ImageHeapRebindNs, frames);
		const auto resolve_ms = Milliseconds(delta, Counter::ResolveTextureNs, frames);
		const auto find_image_ms = Milliseconds(delta, Counter::FindImageNs, frames);
		const auto materialize_ms = Milliseconds(delta, Counter::MaterializeResourcesNs, frames);
		const auto find_module_ms = Milliseconds(delta, Counter::FindModuleNs, frames);
		const auto wait_gpu_ms = Milliseconds(delta, Counter::GpuFenceWaitNs, frames);
		const auto wait_guest_ms = Milliseconds(delta, Counter::GuestMemoryWaitNs, frames);
		const auto wait_lock_ms = Milliseconds(delta, Counter::LockWaitNs, frames);
		const auto idle_ms = Milliseconds(delta, Counter::IdleNs, frames);
		const auto named_ms = rebind_ms + resolve_ms + find_image_ms + materialize_ms +
		                      find_module_ms + wait_gpu_ms + wait_guest_ms + wait_lock_ms + idle_ms;
		Log::WriteToConsoleAndLog(fmt::format(
		    "[frame] seconds={:.2f} frames={} draws={:.2f} dispatches={:.2f} "
		    "rebind={:.2f} rebind_fast={:.2f} entries={:.2f} rebind_skipped={:.2f} "
		    "rebind_resolved={:.2f} resolve={:.2f} find_image={:.2f} find_hits={:.2f} "
		    "materialize={:.2f} materialize_hits={:.2f} materialize_misses={:.2f} "
		    "find_module={:.2f} module_hits={:.2f} module_misses={:.2f} "
		    "ms/frame rebind={:.3f} resolve={:.3f} find_image={:.3f} "
		    "materialize={:.3f} find_module={:.3f} wait_gpu={:.3f} wait_guest={:.3f} "
		    "wait_lock={:.3f} idle={:.3f} other={:.3f} total={:.3f} "
		    "ns/call materialize_hit={:.0f} materialize_miss={:.0f} module_hit={:.0f} "
		    "module_miss={:.0f} "
		    "memo_miss_empty={:.2f} memo_miss_heap_plan={:.2f} memo_miss_shader_base={:.2f} "
		    "memo_miss_indirect_args={:.2f} memo_miss_legacy={:.2f} memo_miss_user_data={:.2f} "
		    "memo_miss_workgroups={:.2f} memo_miss_unreadable={:.2f} memo_miss_range_changed={:.2f} "
		    "memo_miss_heap_changed={:.2f} memo_stored={:.2f} memo_not_stored={:.2f} "
		    "memo_evicted={:.2f} memo_heap_plan_calls={:.2f} memo_ranges_stamped={:.2f} "
		    "memo_ranges_compared={:.2f} memo_bytes_compared={:.0f} memo_ranges_tracked={:.2f} "
		    "memo_ranges_untracked={:.2f} memo_walk_changed={:.2f}\n",
		    std::chrono::duration<double>(elapsed).count(), frames,
		    Mean(delta, Counter::Draws, frames), Mean(delta, Counter::Dispatches, frames),
		    Mean(delta, Counter::ImageHeapRebindCalls, frames),
		    Mean(delta, Counter::ImageHeapRebindFastCalls, frames),
		    Mean(delta, Counter::ImageHeapEntriesVisited, frames),
		    Mean(delta, Counter::ImageHeapEntriesSkipped, frames),
		    Mean(delta, Counter::ImageHeapEntriesResolved, frames),
		    Mean(delta, Counter::ResolveTextureCalls, frames),
		    Mean(delta, Counter::FindImageCalls, frames), Mean(delta, Counter::FindImageHits, frames),
		    Mean(delta, Counter::MaterializeResourcesCalls, frames),
		    Mean(delta, Counter::MaterializationMemoHits, frames),
		    Mean(delta, Counter::MaterializationMemoMisses, frames),
		    Mean(delta, Counter::FindModuleCalls, frames),
		    Mean(delta, Counter::FindModuleHits, frames),
		    Mean(delta, Counter::FindModuleMisses, frames), rebind_ms, resolve_ms, find_image_ms,
		    materialize_ms, find_module_ms, wait_gpu_ms, wait_guest_ms, wait_lock_ms, idle_ms,
		    total_ms - named_ms, total_ms,
		    NanosecondsPerCall(delta, Counter::MaterializationMemoHitNs,
		                       Counter::MaterializationMemoHits),
		    NanosecondsPerCall(delta, Counter::MaterializationMemoMissNs,
		                       Counter::MaterializationMemoMisses),
		    NanosecondsPerCall(delta, Counter::FindModuleHitNs, Counter::FindModuleHits),
		    NanosecondsPerCall(delta, Counter::FindModuleMissNs, Counter::FindModuleMisses),
		    Mean(delta, Counter::MemoMissEmpty, frames),
		    Mean(delta, Counter::MemoMissHeapPlan, frames),
		    Mean(delta, Counter::MemoMissShaderBase, frames),
		    Mean(delta, Counter::MemoMissIndirectArgs, frames),
		    Mean(delta, Counter::MemoMissLegacyHash, frames),
		    Mean(delta, Counter::MemoMissUserData, frames),
		    Mean(delta, Counter::MemoMissWorkgroupCounts, frames),
		    Mean(delta, Counter::MemoMissUnreadable, frames),
		    Mean(delta, Counter::MemoMissRangeChanged, frames),
		    Mean(delta, Counter::MemoMissHeapChanged, frames),
		    Mean(delta, Counter::MemoStored, frames), Mean(delta, Counter::MemoNotStored, frames),
		    Mean(delta, Counter::MemoEvicted, frames),
		    Mean(delta, Counter::MemoHeapPlanCalls, frames),
		    Mean(delta, Counter::MemoRangesStamped, frames),
		    Mean(delta, Counter::MemoRangesCompared, frames),
		    Mean(delta, Counter::MemoBytesCompared, frames),
		    Mean(delta, Counter::MemoRangesTracked, frames),
		    Mean(delta, Counter::MemoRangesUntracked, frames),
		    Mean(delta, Counter::MemoWalkChanged, frames)));
		Log::WriteToConsoleAndLog(fmt::format(
		    "[frame] cuts breadcrumb_ops={:.2f} gpu_invalidate={:.2f} gpu_invalidate_skipped={:.2f} "
		    "gpu_invalidate_gen_bumps={:.2f} guard_gpu={:.2f} guard_cpu={:.2f} guard_skipped={:.2f} "
		    "prefetch_scans={:.2f} prefetch_skipped={:.2f} pump={:.2f} pump_skipped={:.2f}\n",
		    Mean(delta, Counter::BreadcrumbOps, frames),
		    Mean(delta, Counter::GpuInvalidateCalls, frames),
		    Mean(delta, Counter::GpuInvalidateSkipped, frames),
		    Mean(delta, Counter::GpuInvalidateGenBumps, frames),
		    Mean(delta, Counter::GuardGpuRuns, frames), Mean(delta, Counter::GuardCpuChecks, frames),
		    Mean(delta, Counter::GuardSkipped, frames), Mean(delta, Counter::PrefetchScans, frames),
		    Mean(delta, Counter::PrefetchSkipped, frames),
		    Mean(delta, Counter::PumpAsyncCalls, frames),
		    Mean(delta, Counter::PumpAsyncSkipped, frames)));
	}
	if (g_store_loaded.load(std::memory_order_relaxed) && !g_bad_warned) {
		const auto hits   = delta[static_cast<size_t>(Counter::FindModuleHits)];
		const auto misses = delta[static_cast<size_t>(Counter::FindModuleMisses)];
		if (misses > hits) {
			g_bad_seconds += std::chrono::duration<double>(elapsed).count();
		} else {
			g_bad_seconds = 0.0;
		}
		if (g_bad_seconds >= 10.0) {
			g_bad_warned = true;
			Log::WriteToConsoleAndLog(fmt::format(
			    "[frame] WARNING module-misses: module lookups missed more often than they hit for "
			    "{:.0f} s after the store load (last window hits={} misses={}); modules that exist "
			    "are not found, each use requests a new translation\n",
			    g_bad_seconds, hits, misses));
		}
	}
	g_previous     = totals;
	g_window_start = now;
}

} // namespace Common::FrameCost
