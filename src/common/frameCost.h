#ifndef KYTY_COMMON_FRAME_COST_H_
#define KYTY_COMMON_FRAME_COST_H_

#include <chrono>
#include <cstdint>

namespace Common::FrameCost {

// Cheap, always-on counters for the five-second frame-cost proof line. Updates go to one
// lock-free counter block per calling thread, so hot renderer paths do not contend.
enum class Counter : uint8_t {
	Frames,
	Draws,
	Dispatches,
	ImageHeapRebindCalls,
	ImageHeapRebindFastCalls,
	ImageHeapEntriesVisited,
	ImageHeapEntriesSkipped,
	ImageHeapEntriesResolved,
	ResolveTextureCalls,
	FindImageCalls,
	FindImageHits,
	MaterializeResourcesCalls,
	MaterializationMemoHits,
	MaterializationMemoMisses,
	FindModuleCalls,
	FindModuleHits,
	FindModuleMisses,
	ImageHeapRebindNs,
	ResolveTextureNs,
	FindImageNs,
	MaterializeResourcesNs,
	FindModuleNs,
	MaterializationMemoHitNs,
	MaterializationMemoMissNs,
	FindModuleHitNs,
	FindModuleMissNs,
	GpuFenceWaitNs,
	GuestMemoryWaitNs,
	LockWaitNs,
	IdleNs,
	// Why the materialization memo missed (one per miss), and what its entries cost.
	MemoMissEmpty,
	MemoMissHeapPlan,
	MemoMissShaderBase,
	MemoMissIndirectArgs,
	MemoMissLegacyHash,
	MemoMissUserData,
	MemoMissWorkgroupCounts,
	MemoMissUnreadable,
	MemoMissRangeChanged,
	MemoMissHeapChanged,
	MemoStored,
	MemoNotStored,
	MemoEvicted,
	MemoHeapPlanCalls,
	MemoRangesStamped,
	MemoRangesCompared,
	MemoBytesCompared,
	MemoRangesTracked,
	MemoRangesUntracked,
	MemoWalkChanged,
	// Per-call cuts: debug breadcrumbs recorded, GPU-side texture invalidations, indirect dispatch
	// argument guards, the shader prefetch scan and the asynchronous-work pump.
	BreadcrumbOps,
	GpuInvalidateCalls,
	GpuInvalidateSkipped,
	GpuInvalidateGenBumps,
	GuardGpuRuns,
	GuardCpuChecks,
	GuardSkipped,
	PrefetchScans,
	PrefetchSkipped,
	PumpAsyncCalls,
	PumpAsyncSkipped,
	Count
};

void Add(Counter counter, uint64_t value = 1) noexcept;

class ScopedTimer final {
public:
	explicit ScopedTimer(Counter duration) noexcept;
	~ScopedTimer();
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

	void Stop() noexcept;

private:
	Counter                               m_duration;
	std::chrono::steady_clock::time_point m_start;
	ScopedTimer*                          m_parent   = nullptr;
	uint64_t                              m_child_ns = 0;
	// Per-call timers read the clock for one call in --frame-timer-sample and count that call's
	// time that many times over; a timer inside an unmeasured one is not measured either, one inside
	// a measured one is, with the same weight. Waits are always measured, with weight 1.
	uint32_t                              m_weight   = 1;
	bool                                  m_measured = true;
	bool                                  m_done     = false;
};

// Called when a store load that applied modules has finished: from then on, module lookups that
// miss more often than they hit for 10 s print one `[frame] WARNING module-misses ...` line.
void NoteStoreLoaded() noexcept;

// Counts a presented frame and emits one `[frame]` line after each five-second window.
void FramePresented() noexcept;

} // namespace Common::FrameCost

#endif // KYTY_COMMON_FRAME_COST_H_
