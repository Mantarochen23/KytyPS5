#ifndef EMULATOR_SRC_KERNEL_PHYSICALCOVERAGE_H_
#define EMULATOR_SRC_KERNEL_PHYSICALCOVERAGE_H_

#include <cstdint>
#include <iterator>
#include <map>

namespace Libs::LibKernel::Memory {

// How many mappings cover each byte of physical memory. Physical ranges may be mapped more than
// once (aliases), so freeing an automatic block's range must leave out the parts another mapping
// still holds. Asking that of the mapping table means visiting every mapping (it is keyed by
// virtual address); titles that stream partially resident textures keep thousands of small
// mappings and unmap many per frame, which made that scan the hottest code of a run.
//
// Kept as boundaries: each key starts a run of addresses with the same count, up to the next key.
// Addresses below the first key, and runs with count 0, are uncovered. Adjacent runs never have
// the same count, so the map stays as small as the number of distinct covered pieces.
class PhysicalCoverage {
public:
	void Add(uint64_t start, uint64_t size) { Change(start, size, +1); }
	// The range must have been added before (by one or more Add calls covering it).
	void Remove(uint64_t start, uint64_t size) { Change(start, size, -1); }

	// Calls fn(first, last) for each maximal piece of [start, start + size) with a count above 0.
	template <class Fn>
	void ForEachCovered(uint64_t start, uint64_t size, Fn&& fn) const {
		if (size == 0) return;
		const uint64_t end   = start + size;
		auto           it    = m_bounds.upper_bound(start);
		uint64_t       pos   = start;
		int64_t        count = it == m_bounds.begin() ? 0 : std::prev(it)->second;
		bool           open  = false; // a covered piece started at `piece` and is not reported yet
		uint64_t       piece = 0;
		for (;;) {
			const uint64_t next = it == m_bounds.end() || it->first >= end ? end : it->first;
			if (pos < next) {
				if (count > 0 && !open) {
					open  = true;
					piece = pos;
				} else if (count <= 0 && open) {
					fn(piece, pos);
					open = false;
				}
			}
			if (next >= end) {
				if (open) fn(piece, end);
				return;
			}
			pos   = next;
			count = it->second;
			++it;
		}
	}

	[[nodiscard]] int64_t CountAt(uint64_t addr) const {
		auto it = m_bounds.upper_bound(addr);
		return it == m_bounds.begin() ? 0 : std::prev(it)->second;
	}
	[[nodiscard]] size_t BoundaryCount() const { return m_bounds.size(); }

private:
	// Makes `addr` a key holding the count of the run that contained it.
	std::map<uint64_t, int64_t>::iterator Split(uint64_t addr) {
		auto it = m_bounds.lower_bound(addr);
		if (it != m_bounds.end() && it->first == addr) return it;
		const int64_t count = it == m_bounds.begin() ? 0 : std::prev(it)->second;
		return m_bounds.emplace_hint(it, addr, count);
	}

	// Removes keys that start a run with the same count as the run before them (or a 0 run at
	// the start), around [first, last].
	void Merge(std::map<uint64_t, int64_t>::iterator first, uint64_t last) {
		if (first != m_bounds.begin()) --first;
		while (first != m_bounds.end() && first->first <= last) {
			const int64_t before =
			    first == m_bounds.begin() ? 0 : std::prev(first)->second;
			if (first->second == before) {
				first = m_bounds.erase(first);
			} else {
				++first;
			}
		}
	}

	void Change(uint64_t start, uint64_t size, int64_t delta) {
		if (size == 0 || UINT64_MAX - start < size) return;
		const uint64_t end   = start + size;
		auto           last  = Split(end);
		auto           first = Split(start);
		for (auto it = first; it != last; ++it) it->second += delta;
		Merge(first, end);
	}

	std::map<uint64_t, int64_t> m_bounds;
};

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_SRC_KERNEL_PHYSICALCOVERAGE_H_ */
