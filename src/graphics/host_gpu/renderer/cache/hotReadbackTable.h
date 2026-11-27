#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOTREADBACKTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOTREADBACKTABLE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// How many scheduler ticks a buffer stays hot after its last synchronous readback.
inline constexpr uint64_t HotReadbackTicks = 512;
// Fixed cap keeps linear scans trivial and bounds the extra ranges a batch can add.
inline constexpr size_t MaxHotReadbackEntries = 64;
// Total extra bytes one batch may add to the download ring, far below the 64 MiB ring.
inline constexpr uint64_t MaxHotReadbackBytes = 4ull * 1024 * 1024;

// Remembers which buffers took a synchronous readback recently (faults, indirect arguments,
// colour-metadata views, CPU write faults). Entries hold only an address range and a tick, never
// a buffer reference, so a buffer deleted or merged since cannot be reached through a stale entry.
// Used on the GPU thread only.
class HotReadbackTable {
public:
	// Inserts a range, or refreshes the tick of the entry for the same buffer start. When full,
	// evicts the entry with the oldest tick.
	void Note(uint64_t address, uint64_t size, uint64_t tick) {
		for (auto& entry: m_entries) {
			if (entry.address == address) {
				entry.size = size;
				entry.tick = tick;
				return;
			}
		}
		if (m_entries.size() >= MaxHotReadbackEntries) {
			const auto oldest = std::min_element(
			    m_entries.begin(), m_entries.end(),
			    [](const Entry& left, const Entry& right) { return left.tick < right.tick; });
			*oldest = {address, size, tick};
			return;
		}
		m_entries.push_back({address, size, tick});
	}

	// Drops entries older than HotReadbackTicks, then calls visit(address, size) for the rest.
	template <class Visit>
	void ForEachHot(uint64_t now_tick, Visit&& visit) {
		std::erase_if(m_entries, [now_tick](const Entry& entry) {
			return now_tick > entry.tick && now_tick - entry.tick > HotReadbackTicks;
		});
		for (const auto& entry: m_entries) {
			visit(entry.address, entry.size);
		}
	}

	[[nodiscard]] bool TickOf(uint64_t address, uint64_t& tick) const {
		for (const auto& entry: m_entries) {
			if (entry.address == address) {
				tick = entry.tick;
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] size_t Size() const { return m_entries.size(); }

private:
	struct Entry {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t tick    = 0;
	};
	std::vector<Entry> m_entries;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOTREADBACKTABLE_H_ */
