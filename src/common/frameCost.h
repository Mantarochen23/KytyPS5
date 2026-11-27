#ifndef KYTY_COMMON_FRAME_COST_H_
#define KYTY_COMMON_FRAME_COST_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

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
	// Render-thread vs. off-thread GPU wait split and reason breakdown.
	GpuFenceWaitBgNs,
	GpuPriorityOpWaitNs,
	WaitReadbackCalls,
	WaitReadbackNs,
	WaitStreamCalls,
	WaitStreamNs,
	WaitUnmapCalls,
	WaitUnmapNs,
	WaitGdsCalls,
	WaitGdsNs,
	WaitFaultCalls,
	WaitFaultNs,
	WaitGcCalls,
	WaitGcNs,
	WaitRtCalls,
	WaitRtNs,
	WaitSwapchainCalls,
	WaitSwapchainNs,
	WaitPriorityOpCalls,
	WaitOtherCalls,
	WaitOtherNs,
	// Queue submits by reason.
	SubmitsTotal,
	SubmitSlice,
	SubmitFlip,
	SubmitSuspend,
	SubmitSync,
	SubmitWriteBack,
	SubmitFlushDraws,
	SubmitPresent,
	SubmitOther,
	// True guest draw/dispatch counts, CPU expansions, and GPU guards/barriers.
	DrawsDirect,
	DrawsIndirectGpuPackets,
	DrawsIndirectGpuSubdraws,
	DrawsIndirectCpuPackets,
	DrawsIndirectCpuSubdraws,
	DrawsIndirectZero,
	DrawsCoalesced,
	DrawsQuadExtra,
	DrawsSkipped,
	DispatchesDirect,
	DispatchesIndirect,
	DispatchesIndirectThreadDim,
	DispatchesGuard,
	DispatchesOrderedAppendExtra,
	DispatchesSkipped,
	BvhDispatches,
	BarriersAllCommands,
	// 1-in-1,024 sampled per-step timings for executed draws and dispatches.
	DrawStepSamples,
	DrawStepProgramsNs,
	DrawStepTargetsNs,
	DrawStepPipelineNs,
	DrawStepBindNs,
	DrawStepCommitNs,
	DrawStepRecordNs,
	DispatchStepSamples,
	DispatchStepProgramNs,
	DispatchStepPipelineNs,
	DispatchStepBindNs,
	DispatchStepGuardNs,
	DispatchStepCommitNs,
	DispatchStepRecordNs,
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
	// Why an image-heap rebind missed its fast return (one per failed condition, so a miss can count
	// several), and which call sites bumped the two generations it compares. HeapSelfBump counts
	// slow rebinds that ended with the heap-change generation already past the one they stored.
	HeapMissPending,
	HeapMissDescriptorGen,
	HeapMissCacheGen,
	HeapMissHeapGen,
	HeapSelfBump,
	CacheSelfBump,
	HeapRepointCalls,
	HeapRepointSlots,
	HeapLayoutQuiet,
	HeapBumpCpuDirty,
	HeapBumpMaybeDirty,
	HeapBumpBufferModified,
	HeapBumpLayoutLeave,
	HeapBumpUntrackHead,
	HeapBumpUntrackTail,
	HeapBumpUnregister,
	HeapBumpUnmap,
	CacheBumpRegister,
	CacheBumpUnregister,
	CacheBumpStencil,
	CacheBumpGpuInvalidate,
	CacheBumpCpuInvalidate,
	CacheBumpClearMeta,
	CacheBumpTouchMeta,
	CacheBumpOther,
	// Runtime-indexed buffer table activity: candidates are paid only on rebuild.
	TableCandidates,
	TableDispatches,
	TableRebuilds,
	TableEnumerations,
	TableMarks,
	TableMarkRanges,
	// Rebuilt tables of typed or image-synced buffers with a candidate range inside a registered image.
	TableImageAliased,
	// Why a table was rebuilt (one per rebuild) and how a mark was resolved.
	TableRebuildKey,
	TableRebuildGeneration,
	TableRebuildGenerationOwn,
	TableRebuildEvicted,
	TableRebuildDisabled,
	TableMarksSkipped,
	Count
};

enum class WaitReason : uint8_t {
	Other,
	Readback,
	StreamWrap,
	Unmap,
	Gds,
	FaultBuf,
	Gc,
	Rt,
	Swapchain,
	PriorityThread,
};

enum class SubmitReason : uint8_t {
	Other,
	Slice,
	Flip,
	Suspend,
	Sync,
	WriteBack,
	FlushDraws,
	Present,
};

inline constexpr size_t CounterCount = static_cast<size_t>(Counter::Count);

void Add(Counter counter, uint64_t value = 1) noexcept;

// Marks the calling thread as the GPU render thread so GPU fence waits on background threads
// (priority completion runner, presenter) are reported separately from render-thread wait_gpu.
void               MarkRenderThread(bool is_render) noexcept;
[[nodiscard]] bool IsRenderThread() noexcept;

class ScopedWaitReason final {
public:
	explicit ScopedWaitReason(WaitReason reason) noexcept;
	~ScopedWaitReason();
	ScopedWaitReason(const ScopedWaitReason&)            = delete;
	ScopedWaitReason& operator=(const ScopedWaitReason&) = delete;

private:
	WaitReason m_previous;
};

[[nodiscard]] WaitReason CurrentWaitReason() noexcept;
void                     RecordRenderWaitReason(uint64_t elapsed_ns) noexcept;

class ScopedSubmitReason final {
public:
	explicit ScopedSubmitReason(SubmitReason reason) noexcept;
	~ScopedSubmitReason();
	ScopedSubmitReason(const ScopedSubmitReason&)            = delete;
	ScopedSubmitReason& operator=(const ScopedSubmitReason&) = delete;

private:
	SubmitReason m_previous;
};

[[nodiscard]] SubmitReason CurrentSubmitReason() noexcept;
void                       RecordSubmit() noexcept;

class ScopedTimer final {
public:
	explicit ScopedTimer(Counter duration) noexcept;
	~ScopedTimer();
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

	uint64_t Stop() noexcept;

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

// Lightweight 1-in-N step sampler. When inactive, AddTo() is a single branch with zero clock
// calls. Complete() resets the countdown only when a sampled call actually executes, so skipped
// calls never skew per-step means.
class StepClock final {
public:
	static constexpr uint32_t DefaultPeriod = 1024;
	static constexpr uint32_t InitialDelay  = 512;

	explicit StepClock(uint32_t& countdown) noexcept: m_countdown(&countdown) {
		if (countdown == 0) {
			m_active = true;
			m_last   = std::chrono::steady_clock::now();
		} else {
			--countdown;
		}
	}

	[[nodiscard]] bool Active() const noexcept { return m_active; }

	void AddTo(uint64_t& bucket_ns) noexcept {
		if (!m_active) return;
		const auto now = std::chrono::steady_clock::now();
		bucket_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_last).count());
		m_last = now;
	}

	void Complete(uint32_t period = DefaultPeriod) noexcept {
		if (m_active && m_countdown != nullptr) {
			*m_countdown = period > 0 ? period - 1u : 0u;
			m_active     = false;
		}
	}

private:
	uint32_t*                             m_countdown = nullptr;
	std::chrono::steady_clock::time_point m_last {};
	bool                                  m_active = false;
};

struct ReportLines {
	std::string summary;
	std::string arch;
	std::string cuts;
	std::string heap;
};

[[nodiscard]] std::array<uint64_t, CounterCount> SnapshotCounters();
[[nodiscard]] ReportLines FormatWindow(const std::array<uint64_t, CounterCount>& delta,
                                       double                                    elapsed_seconds);

// Called when a store load that applied modules has finished: from then on, module lookups that
// miss more often than they hit for 10 s print one `[frame] WARNING module-misses ...` line.
void NoteStoreLoaded() noexcept;

// Counts a presented frame and emits `[frame]` lines after each five-second window.
void FramePresented() noexcept;

} // namespace Common::FrameCost

#endif // KYTY_COMMON_FRAME_COST_H_
