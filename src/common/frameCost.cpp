#include "common/frameCost.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <vector>

namespace Common::FrameCost {
namespace {

using Clock = std::chrono::steady_clock;

struct ThreadCounters {
	std::array<std::atomic<uint64_t>, CounterCount> values {};
};

std::mutex                         g_threads_mutex;
std::vector<ThreadCounters*>       g_threads;
std::mutex                         g_report_mutex;
std::array<uint64_t, CounterCount> g_previous {};
Clock::time_point                  g_window_start {};
thread_local ScopedTimer*          g_active_timer   = nullptr;
thread_local bool                  g_render_thread  = false;
thread_local WaitReason            g_wait_reason    = WaitReason::Other;
thread_local SubmitReason          g_submit_reason  = SubmitReason::Other;

ThreadCounters& Local() {
	thread_local ThreadCounters* counters = [] {
		auto* created = new ThreadCounters;
		std::scoped_lock lock {g_threads_mutex};
		g_threads.push_back(created);
		return created;
	}();
	return *counters;
}

double Mean(const std::array<uint64_t, CounterCount>& delta, Counter counter, uint64_t frames) {
	return frames == 0 ? 0.0
	                   : static_cast<double>(delta[static_cast<size_t>(counter)]) /
	                         static_cast<double>(frames);
}

double Milliseconds(const std::array<uint64_t, CounterCount>& delta, Counter counter,
                    uint64_t frames) {
	return Mean(delta, counter, frames) / 1'000'000.0;
}

double NanosecondsPerCall(const std::array<uint64_t, CounterCount>& delta, Counter duration,
                          Counter calls) {
	const auto count = delta[static_cast<size_t>(calls)];
	return count == 0 ? 0.0
	                  : static_cast<double>(delta[static_cast<size_t>(duration)]) /
	                        static_cast<double>(count);
}

} // namespace

void Add(Counter counter, uint64_t value) noexcept {
	Local().values[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void MarkRenderThread(bool is_render) noexcept {
	g_render_thread = is_render;
}

bool IsRenderThread() noexcept {
	return g_render_thread;
}

ScopedWaitReason::ScopedWaitReason(WaitReason reason) noexcept: m_previous(g_wait_reason) {
	g_wait_reason = reason;
}

ScopedWaitReason::~ScopedWaitReason() {
	g_wait_reason = m_previous;
}

WaitReason CurrentWaitReason() noexcept {
	return g_wait_reason;
}

void RecordRenderWaitReason(uint64_t elapsed_ns) noexcept {
	Counter calls = Counter::WaitOtherCalls;
	Counter ns    = Counter::WaitOtherNs;
	switch (g_wait_reason) {
		case WaitReason::Readback:
			calls = Counter::WaitReadbackCalls;
			ns    = Counter::WaitReadbackNs;
			break;
		case WaitReason::StreamWrap:
			calls = Counter::WaitStreamCalls;
			ns    = Counter::WaitStreamNs;
			break;
		case WaitReason::Unmap:
			calls = Counter::WaitUnmapCalls;
			ns    = Counter::WaitUnmapNs;
			break;
		case WaitReason::Gds:
			calls = Counter::WaitGdsCalls;
			ns    = Counter::WaitGdsNs;
			break;
		case WaitReason::FaultBuf:
			calls = Counter::WaitFaultCalls;
			ns    = Counter::WaitFaultNs;
			break;
		case WaitReason::Gc:
			calls = Counter::WaitGcCalls;
			ns    = Counter::WaitGcNs;
			break;
		case WaitReason::Rt:
			calls = Counter::WaitRtCalls;
			ns    = Counter::WaitRtNs;
			break;
		case WaitReason::Swapchain:
			calls = Counter::WaitSwapchainCalls;
			ns    = Counter::WaitSwapchainNs;
			break;
		case WaitReason::PriorityThread:
		case WaitReason::Other: break;
	}
	Add(calls);
	Add(ns, elapsed_ns);
}

ScopedSubmitReason::ScopedSubmitReason(SubmitReason reason) noexcept: m_previous(g_submit_reason) {
	g_submit_reason = reason;
}

ScopedSubmitReason::~ScopedSubmitReason() {
	g_submit_reason = m_previous;
}

SubmitReason CurrentSubmitReason() noexcept {
	return g_submit_reason;
}

void RecordSubmit() noexcept {
	Add(Counter::SubmitsTotal);
	switch (g_submit_reason) {
		case SubmitReason::Slice: Add(Counter::SubmitSlice); break;
		case SubmitReason::Flip: Add(Counter::SubmitFlip); break;
		case SubmitReason::Suspend: Add(Counter::SubmitSuspend); break;
		case SubmitReason::Sync: Add(Counter::SubmitSync); break;
		case SubmitReason::WriteBack: Add(Counter::SubmitWriteBack); break;
		case SubmitReason::FlushDraws: Add(Counter::SubmitFlushDraws); break;
		case SubmitReason::Present: Add(Counter::SubmitPresent); break;
		case SubmitReason::Other: Add(Counter::SubmitOther); break;
	}
}

ScopedTimer::ScopedTimer(Counter duration) noexcept
    : m_duration(duration), m_start(Clock::now()), m_parent(g_active_timer) {
	g_active_timer = this;
}

ScopedTimer::~ScopedTimer() {
	(void)Stop();
}

uint64_t ScopedTimer::Stop() noexcept {
	if (m_start == Clock::time_point {}) return 0;
	const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_start);
	m_start            = {};
	const auto elapsed_ns = static_cast<uint64_t>(elapsed.count());
	if (g_active_timer == this) g_active_timer = m_parent;
	if (m_parent != nullptr) m_parent->m_child_ns += elapsed_ns;
	const auto exclusive_ns = elapsed_ns - std::min(elapsed_ns, m_child_ns);
	Add(m_duration, exclusive_ns);
	return exclusive_ns;
}

std::array<uint64_t, CounterCount> SnapshotCounters() {
	std::array<uint64_t, CounterCount> result {};
	std::scoped_lock lock {g_threads_mutex};
	for (const auto* counters: g_threads) {
		for (size_t i = 0; i < result.size(); ++i) {
			result[i] += counters->values[i].load(std::memory_order_relaxed);
		}
	}
	return result;
}

ReportLines FormatWindow(const std::array<uint64_t, CounterCount>& delta, double elapsed_seconds) {
	const auto frames = delta[static_cast<size_t>(Counter::Frames)];
	if (frames == 0) {
		return {};
	}
	const auto total_ms       = (elapsed_seconds * 1000.0) / static_cast<double>(frames);
	const auto rebind_ms      = Milliseconds(delta, Counter::ImageHeapRebindNs, frames);
	const auto resolve_ms     = Milliseconds(delta, Counter::ResolveTextureNs, frames);
	const auto find_image_ms  = Milliseconds(delta, Counter::FindImageNs, frames);
	const auto materialize_ms = Milliseconds(delta, Counter::MaterializeResourcesNs, frames);
	const auto find_module_ms = Milliseconds(delta, Counter::FindModuleNs, frames);
	const auto wait_gpu_ms    = Milliseconds(delta, Counter::GpuFenceWaitNs, frames) +
	                            Milliseconds(delta, Counter::GpuPriorityOpWaitNs, frames);
	const auto wait_gpu_bg_ms = Milliseconds(delta, Counter::GpuFenceWaitBgNs, frames);
	const auto wait_guest_ms  = Milliseconds(delta, Counter::GuestMemoryWaitNs, frames);
	const auto wait_lock_ms   = Milliseconds(delta, Counter::LockWaitNs, frames);
	const auto idle_ms        = Milliseconds(delta, Counter::IdleNs, frames);
	const auto named_ms       = rebind_ms + resolve_ms + find_image_ms + materialize_ms +
	                            find_module_ms + wait_gpu_ms + wait_guest_ms + wait_lock_ms + idle_ms;

	ReportLines lines;
	lines.summary = fmt::format(
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
	    elapsed_seconds, frames, Mean(delta, Counter::Draws, frames),
	    Mean(delta, Counter::Dispatches, frames),
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
	    Mean(delta, Counter::FindModuleCalls, frames), Mean(delta, Counter::FindModuleHits, frames),
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
	    Mean(delta, Counter::MemoStored, frames),
	    Mean(delta, Counter::MemoNotStored, frames),
	    Mean(delta, Counter::MemoEvicted, frames),
	    Mean(delta, Counter::MemoHeapPlanCalls, frames),
	    Mean(delta, Counter::MemoRangesStamped, frames),
	    Mean(delta, Counter::MemoRangesCompared, frames),
	    Mean(delta, Counter::MemoBytesCompared, frames),
	    Mean(delta, Counter::MemoRangesTracked, frames),
	    Mean(delta, Counter::MemoRangesUntracked, frames),
	    Mean(delta, Counter::MemoWalkChanged, frames));

	lines.arch = fmt::format(
	    "[frame] arch wait_gpu_bg={:.3f} "
	    "wait_ms(rb={:.3f} stream={:.3f} unmap={:.3f} gds={:.3f} fault={:.3f} gc={:.3f} "
	    "rt={:.3f} swap={:.3f} prio={:.3f} other={:.3f}) "
	    "wait_n(rb={:.2f} stream={:.2f} unmap={:.2f} gds={:.2f} fault={:.2f} gc={:.2f} "
	    "rt={:.2f} swap={:.2f} prio={:.2f} other={:.2f}) "
	    "submits={:.2f}(slice={:.2f} flip={:.2f} susp={:.2f} sync={:.2f} wb={:.2f} "
	    "fdraw={:.2f} pres={:.2f} other={:.2f}) "
	    "calls_draw(dir={:.2f} ind_gpu={:.2f}/{:.2f} ind_cpu={:.2f}/{:.2f} ind_zero={:.2f} "
	    "coal={:.2f} quad_ex={:.2f} skip={:.2f}) "
	    "calls_disp(dir={:.2f} ind={:.2f} ind_tdim={:.2f} guard={:.2f} oa_ex={:.2f} "
	    "skip={:.2f} bvh={:.2f}) "
	    "barriers_all={:.2f} "
	    "draw_ns(n={} prog={:.0f} rt={:.0f} pipe={:.0f} bind={:.0f} commit={:.0f} rec={:.0f}) "
	    "disp_ns(n={} prog={:.0f} pipe={:.0f} bind={:.0f} guard={:.0f} commit={:.0f} rec={:.0f})\n",
	    wait_gpu_bg_ms, Milliseconds(delta, Counter::WaitReadbackNs, frames),
	    Milliseconds(delta, Counter::WaitStreamNs, frames),
	    Milliseconds(delta, Counter::WaitUnmapNs, frames),
	    Milliseconds(delta, Counter::WaitGdsNs, frames),
	    Milliseconds(delta, Counter::WaitFaultNs, frames),
	    Milliseconds(delta, Counter::WaitGcNs, frames),
	    Milliseconds(delta, Counter::WaitRtNs, frames),
	    Milliseconds(delta, Counter::WaitSwapchainNs, frames),
	    Milliseconds(delta, Counter::GpuPriorityOpWaitNs, frames),
	    Milliseconds(delta, Counter::WaitOtherNs, frames),
	    Mean(delta, Counter::WaitReadbackCalls, frames),
	    Mean(delta, Counter::WaitStreamCalls, frames), Mean(delta, Counter::WaitUnmapCalls, frames),
	    Mean(delta, Counter::WaitGdsCalls, frames), Mean(delta, Counter::WaitFaultCalls, frames),
	    Mean(delta, Counter::WaitGcCalls, frames), Mean(delta, Counter::WaitRtCalls, frames),
	    Mean(delta, Counter::WaitSwapchainCalls, frames),
	    Mean(delta, Counter::WaitPriorityOpCalls, frames),
	    Mean(delta, Counter::WaitOtherCalls, frames), Mean(delta, Counter::SubmitsTotal, frames),
	    Mean(delta, Counter::SubmitSlice, frames), Mean(delta, Counter::SubmitFlip, frames),
	    Mean(delta, Counter::SubmitSuspend, frames), Mean(delta, Counter::SubmitSync, frames),
	    Mean(delta, Counter::SubmitWriteBack, frames),
	    Mean(delta, Counter::SubmitFlushDraws, frames),
	    Mean(delta, Counter::SubmitPresent, frames), Mean(delta, Counter::SubmitOther, frames),
	    Mean(delta, Counter::DrawsDirect, frames),
	    Mean(delta, Counter::DrawsIndirectGpuPackets, frames),
	    Mean(delta, Counter::DrawsIndirectGpuSubdraws, frames),
	    Mean(delta, Counter::DrawsIndirectCpuPackets, frames),
	    Mean(delta, Counter::DrawsIndirectCpuSubdraws, frames),
	    Mean(delta, Counter::DrawsIndirectZero, frames),
	    Mean(delta, Counter::DrawsCoalesced, frames), Mean(delta, Counter::DrawsQuadExtra, frames),
	    Mean(delta, Counter::DrawsSkipped, frames), Mean(delta, Counter::DispatchesDirect, frames),
	    Mean(delta, Counter::DispatchesIndirect, frames),
	    Mean(delta, Counter::DispatchesIndirectThreadDim, frames),
	    Mean(delta, Counter::DispatchesGuard, frames),
	    Mean(delta, Counter::DispatchesOrderedAppendExtra, frames),
	    Mean(delta, Counter::DispatchesSkipped, frames),
	    Mean(delta, Counter::BvhDispatches, frames),
	    Mean(delta, Counter::BarriersAllCommands, frames),
	    delta[static_cast<size_t>(Counter::DrawStepSamples)],
	    NanosecondsPerCall(delta, Counter::DrawStepProgramsNs, Counter::DrawStepSamples),
	    NanosecondsPerCall(delta, Counter::DrawStepTargetsNs, Counter::DrawStepSamples),
	    NanosecondsPerCall(delta, Counter::DrawStepPipelineNs, Counter::DrawStepSamples),
	    NanosecondsPerCall(delta, Counter::DrawStepBindNs, Counter::DrawStepSamples),
	    NanosecondsPerCall(delta, Counter::DrawStepCommitNs, Counter::DrawStepSamples),
	    NanosecondsPerCall(delta, Counter::DrawStepRecordNs, Counter::DrawStepSamples),
	    delta[static_cast<size_t>(Counter::DispatchStepSamples)],
	    NanosecondsPerCall(delta, Counter::DispatchStepProgramNs, Counter::DispatchStepSamples),
	    NanosecondsPerCall(delta, Counter::DispatchStepPipelineNs, Counter::DispatchStepSamples),
	    NanosecondsPerCall(delta, Counter::DispatchStepBindNs, Counter::DispatchStepSamples),
	    NanosecondsPerCall(delta, Counter::DispatchStepGuardNs, Counter::DispatchStepSamples),
	    NanosecondsPerCall(delta, Counter::DispatchStepCommitNs, Counter::DispatchStepSamples),
	    NanosecondsPerCall(delta, Counter::DispatchStepRecordNs, Counter::DispatchStepSamples));
	return lines;
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
		g_previous     = SnapshotCounters();
		return;
	}
	const auto elapsed = now - g_window_start;
	if (elapsed < std::chrono::seconds(5)) return;

	const auto totals = SnapshotCounters();
	std::array<uint64_t, CounterCount> delta {};
	for (size_t i = 0; i < delta.size(); ++i) delta[i] = totals[i] - g_previous[i];
	const auto elapsed_seconds = std::chrono::duration<double>(elapsed).count();
	if (delta[static_cast<size_t>(Counter::Frames)] != 0) {
		const auto lines = FormatWindow(delta, elapsed_seconds);
		Log::WriteToConsoleAndLog(lines.summary);
		Log::WriteToConsoleAndLog(lines.arch);
	}
	if (g_store_loaded.load(std::memory_order_relaxed) && !g_bad_warned) {
		const auto hits   = delta[static_cast<size_t>(Counter::FindModuleHits)];
		const auto misses = delta[static_cast<size_t>(Counter::FindModuleMisses)];
		if (misses > hits) {
			g_bad_seconds += elapsed_seconds;
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
