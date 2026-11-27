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
	ImageHeapEntriesVisited,
	ResolveTextureCalls,
	FindImageCalls,
	FindImageHits,
	MaterializeResourcesCalls,
	FindModuleCalls,
	ImageHeapRebindNs,
	ResolveTextureNs,
	FindImageNs,
	MaterializeResourcesNs,
	FindModuleNs,
	GpuFenceWaitNs,
	GuestMemoryWaitNs,
	LockWaitNs,
	IdleNs,
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
};

// Counts a presented frame and emits one `[frame]` line after each five-second window.
void FramePresented() noexcept;

} // namespace Common::FrameCost

#endif // KYTY_COMMON_FRAME_COST_H_
