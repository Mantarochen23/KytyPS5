#include "graphics/host_gpu/renderer/cache/guestReadWatch.h"

#include "graphics/host_gpu/regionDefinitions.h"

#include <vector>

namespace Libs::Graphics {

namespace {
constexpr uint64_t PageSize = TRACKER_PAGE_SIZE;
}

GuestReadWatch::~GuestReadWatch() {
	std::scoped_lock lock {m_mutex};
	for (auto& [index, page]: m_pages) {
		if (page->watched.exchange(false)) {
			m_page_manager.UpdatePageWatchers<false>(index * PageSize, PageSize);
		}
	}
}

void GuestReadWatch::Configure(uint32_t max_writes, Environment environment) {
	std::scoped_lock lock {m_mutex};
	m_max_writes  = max_writes;
	m_environment = environment;
}

bool GuestReadWatch::Track(uint64_t address, uint64_t size, std::vector<PageStamp>& stamps) {
	if (size == 0u) return true;
	if (address + size < address || m_environment.trackable == nullptr ||
	    !m_environment.trackable(m_environment.user, address, size)) {
		return false;
	}
	const uint64_t first = address / PageSize;
	const uint64_t last  = (address + size - 1u) / PageSize;
	std::scoped_lock lock {m_mutex};
	std::vector<Page*> pages;
	pages.reserve(static_cast<size_t>(last - first + 1u));
	for (uint64_t index = first; index <= last; ++index) {
		auto& slot = m_pages[index];
		if (slot == nullptr) slot = std::make_unique<Page>();
		if (slot->writes > m_max_writes) return false;
		pages.push_back(slot.get());
	}
	// Arm runs of unwatched pages with one call each.
	for (size_t at = 0; at < pages.size();) {
		if (pages[at]->watched.load(std::memory_order_relaxed)) {
			++at;
			continue;
		}
		size_t end = at;
		while (end < pages.size() && !pages[end]->watched.load(std::memory_order_relaxed)) ++end;
		m_page_manager.UpdatePageWatchers<true>((first + at) * PageSize, (end - at) * PageSize);
		for (size_t i = at; i < end; ++i) pages[i]->watched.store(true, std::memory_order_release);
		m_watched.fetch_add(end - at, std::memory_order_relaxed);
		at = end;
	}
	for (auto* page: pages) {
		stamps.push_back({page, page->generation.load(std::memory_order_acquire)});
	}
	return true;
}

bool GuestReadWatch::Unchanged(std::span<const PageStamp> stamps) {
	for (const auto& stamp: stamps) {
		const auto* page = static_cast<const Page*>(stamp.page);
		if (page->generation.load(std::memory_order_acquire) != stamp.generation ||
		    !page->watched.load(std::memory_order_acquire)) {
			return false;
		}
	}
	return true;
}

bool GuestReadWatch::GpuHolds(uint64_t address, uint64_t size) {
	return m_environment.gpu_holds == nullptr || m_environment.gpu_holds(m_environment.user, address, size);
}

void GuestReadWatch::Written(uint64_t address, uint64_t size) {
	if (size == 0u || m_watched.load(std::memory_order_relaxed) == 0u) return;
	const uint64_t first = address / PageSize;
	const uint64_t last  = (address + size - 1u) / PageSize;
	std::scoped_lock lock {m_mutex};
	uint64_t run_begin = 0;
	uint64_t run_end   = 0; // Exclusive page index; the run is empty when equal to run_begin.
	const auto release = [&] {
		if (run_end != run_begin) {
			m_page_manager.UpdatePageWatchers<false>(run_begin * PageSize, (run_end - run_begin) * PageSize);
		}
		run_begin = run_end = 0;
	};
	for (uint64_t index = first; index <= last; ++index) {
		const auto found = m_pages.find(index);
		if (found == m_pages.end() || !found->second->watched.load(std::memory_order_relaxed)) continue;
		auto& page = *found->second;
		// The generation moves before the page is released, so a write that follows sees it.
		page.generation.fetch_add(1u, std::memory_order_acq_rel);
		++page.writes;
		page.watched.store(false, std::memory_order_release);
		m_watched.fetch_sub(1u, std::memory_order_relaxed);
		if (run_end != run_begin && run_end == index) {
			++run_end;
		} else {
			release();
			run_begin = index;
			run_end   = index + 1u;
		}
	}
	release();
}

void GuestReadWatch::Touched(uint64_t address, uint64_t size) {
	if (size == 0u || m_watched.load(std::memory_order_relaxed) == 0u) return;
	const uint64_t first = address / PageSize;
	const uint64_t last  = (address + size - 1u) / PageSize;
	std::scoped_lock lock {m_mutex};
	for (uint64_t index = first; index <= last; ++index) {
		const auto found = m_pages.find(index);
		if (found != m_pages.end()) found->second->generation.fetch_add(1u, std::memory_order_acq_rel);
	}
}

} // namespace Libs::Graphics
