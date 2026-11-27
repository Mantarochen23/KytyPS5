#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GUESTREADWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GUESTREADWATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

// Write watches over the guest pages the shader materialization reads (descriptor tables, user
// data blocks), so a memoized materialization can tell without reading them that nothing wrote
// there. A page is watched while some entry depends on it. The first write to it faults, bumps the
// page's generation and releases the watch; the entry whose stamp holds the old generation is then
// compared by its bytes. Content that changes without a fault (host writes through the backing,
// downloads of GPU-written bytes) bumps the generation through Touched.
// Pages that were written more than `max_writes` times are not tracked: every range over them is
// compared by its bytes.
class GuestReadWatch final: public ReadStamps {
public:
	struct Environment {
		void* user = nullptr;
		// Whether the range may be tracked: mapped, and not backed by cached images.
		bool (*trackable)(void* user, uint64_t address, uint64_t size) = nullptr;
		// Whether the GPU holds bytes of the range that were not downloaded yet.
		bool (*gpu_holds)(void* user, uint64_t address, uint64_t size) = nullptr;
	};

	explicit GuestReadWatch(PageManager& page_manager): m_page_manager(page_manager) {}
	~GuestReadWatch();
	KYTY_CLASS_NO_COPY(GuestReadWatch);

	void Configure(uint32_t max_writes, Environment environment);

	bool Track(uint64_t address, uint64_t size, std::vector<PageStamp>& stamps) override;
	bool Unchanged(std::span<const PageStamp> stamps) override;
	bool GpuHolds(uint64_t address, uint64_t size) override;
	[[nodiscard]] std::shared_ptr<ShaderRecompiler::IR::MemoryChangeStamp> Track(
	    uint64_t address, uint64_t size) override;

	// A write is about to land over the range (CPU fault, host I/O into it, unmap). Releases the
	// watch of every watched page of it, on any thread.
	void Written(uint64_t address, uint64_t size);
	// The bytes of the range changed without a fault. Stamps over its pages stop being current.
	void Touched(uint64_t address, uint64_t size);

	[[nodiscard]] uint64_t WatchedPages() const noexcept {
		return m_watched.load(std::memory_order_relaxed);
	}

private:
	class Stamp;
	struct Page {
		std::atomic<uint64_t> generation {1};
		std::atomic<bool>     watched {false};
		uint32_t              writes = 0; // Guarded by m_mutex.
	};

	PageManager&                                      m_page_manager;
	Environment                                       m_environment;
	uint32_t                                          m_max_writes = 0;
	std::mutex                                        m_mutex;
	std::unordered_map<uint64_t, std::unique_ptr<Page>> m_pages; // By page index.
	std::atomic<uint64_t>                             m_watched {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GUESTREADWATCH_H_
