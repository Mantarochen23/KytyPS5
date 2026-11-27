#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <xxhash.h>

#include <algorithm>
#include <utility>

namespace Libs::Graphics {

MaterializationMemoSettings& GlobalMaterializationMemoSettings() {
	static MaterializationMemoSettings settings;
	return settings;
}

bool RecordingImageHeap::Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
                                 ShaderRecompiler::IR::ImageHeapView&          view) {
	if (m_inner == nullptr || !m_inner->Acquire(request, view)) {
		m_declined = true;
		return false;
	}
	m_acquired.push_back({.request    = request,
	                      .slots      = view.slots,
	                      .slot_count = view.slot_count,
	                      .kinds      = {view.kinds.begin(), view.kinds.end()}});
	return true;
}

void MemoizedMaterialization::Configure(uint32_t entries, ReadStamps* stamps) {
	m_capacity = std::max(entries, 1u);
	m_stamps   = stamps;
	while (m_entries.size() > m_capacity) m_entries.pop_back();
}

void MemoizedMaterialization::Clear() {
	m_entries.clear();
	current = 0;
}

uint64_t MemoizedMaterialization::ApproximateBytes() const {
	uint64_t bytes = 0;
	for (const auto& entry: m_entries) {
		bytes += sizeof(Entry);
		bytes += (entry->user_data.capacity() + entry->workgroup_counts.capacity()) * sizeof(uint32_t);
		bytes += entry->reads.capacity() * sizeof(Read);
		bytes += entry->pages.capacity() * sizeof(PageStamp);
		for (const auto& heap: entry->heap) {
			bytes += sizeof(heap) + heap.kinds.capacity() * sizeof(ShaderRecompiler::IR::DescriptorValue);
		}
		const auto& r = entry->resources;
		bytes += (r.buffers.capacity() + r.images.capacity() + r.samplers.capacity()) *
		         sizeof(ShaderRecompiler::IR::DescriptorValue);
		bytes += (r.flattened_srt.capacity() + r.user_data.capacity()) * sizeof(uint32_t);
		bytes += r.specialization_reads.capacity() * sizeof(std::pair<uint64_t, uint64_t>);
		const auto& s = entry->specialization;
		bytes += s.buffers.capacity() * sizeof(s.buffers[0]) + s.images.capacity() * sizeof(s.images[0]) +
		         s.samplers.capacity() * sizeof(s.samplers[0]);
	}
	return bytes;
}

bool MemoizedMaterialization::KeyMatches(const Entry&                                  entry,
                                         const ShaderRecompiler::IR::SrtRuntime& runtime,
                                         MemoOutcome&                            why) const {
	if (entry.shader_base != runtime.shader_base) {
		why = MemoOutcome::ShaderBase;
	} else if (entry.indirect_args_address != runtime.indirect_args_address) {
		why = MemoOutcome::IndirectArgs;
	} else if (entry.legacy_hash != runtime.legacy_hash) {
		why = MemoOutcome::LegacyHash;
	} else if (!std::ranges::equal(entry.user_data, runtime.user_data)) {
		why = MemoOutcome::UserData;
	} else if (!std::ranges::equal(entry.workgroup_counts, runtime.workgroup_counts)) {
		why = MemoOutcome::WorkgroupCounts;
	} else {
		return true;
	}
	return false;
}

bool MemoizedMaterialization::HashRange(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                        uint64_t address, uint64_t size, uint64_t& hash) {
	if (size == 0u) {
		hash = 0u;
		return true;
	}
	if (runtime.read_specialization_memory == nullptr || (size & 3u) != 0u ||
	    size / sizeof(uint32_t) > m_scratch.max_size()) {
		return false;
	}
	m_scratch.resize(static_cast<size_t>(size / sizeof(uint32_t)));
	if (!runtime.read_specialization_memory(runtime.userdata, address, m_scratch)) return false;
	hash = XXH3_64bits(m_scratch.data(), static_cast<size_t>(size));
	return true;
}

bool MemoizedMaterialization::ValidateReads(Entry&                                  entry,
                                            const ShaderRecompiler::IR::SrtRuntime& runtime,
                                            MemoOutcome& why, MemoLookupStats& stats) {
	for (auto& read: entry.reads) {
		if (read.tracked && m_stamps != nullptr) {
			const std::span<const PageStamp> pages(entry.pages.data() + read.first_page,
			                                       read.page_count);
			if (m_stamps->Unchanged(pages) && !m_stamps->GpuHolds(read.address, read.size)) {
				++stats.ranges_stamped;
				continue;
			}
		}
		// A page was written, or the GPU holds newer bytes: the bytes decide. Watch before reading,
		// so a write during the read invalidates the stamps this range gets next.
		std::vector<PageStamp> fresh;
		bool                   retracked = false;
		if (read.tracked && m_stamps != nullptr) {
			retracked = m_stamps->Track(read.address, read.size, fresh) &&
			            fresh.size() == read.page_count;
		}
		uint64_t hash = 0;
		if (!HashRange(runtime, read.address, read.size, hash)) {
			why = MemoOutcome::Unreadable;
			return false;
		}
		++stats.ranges_compared;
		stats.bytes_compared += read.size;
		if (hash != read.hash) {
			why = MemoOutcome::RangeChanged;
			return false;
		}
		if (retracked) {
			std::ranges::copy(fresh, entry.pages.begin() + read.first_page);
		} else {
			read.tracked = false;
		}
	}
	return true;
}

bool MemoizedMaterialization::ReplayHeap(const Entry&                            entry,
                                         const ShaderRecompiler::IR::SrtRuntime& runtime) {
	for (const auto& acquired: entry.heap) {
		ShaderRecompiler::IR::ImageHeapView view;
		if (runtime.image_heap == nullptr || !runtime.image_heap->Acquire(acquired.request, view) ||
		    view.slots != acquired.slots || view.slot_count != acquired.slot_count ||
		    !std::ranges::equal(view.kinds, acquired.kinds)) {
			return false;
		}
	}
	return true;
}

MemoizedMaterialization::Entry* MemoizedMaterialization::Lookup(
    const ShaderRecompiler::IR::SrtRuntime& runtime, bool uses_image_heap, MemoOutcome& outcome,
    MemoLookupStats& stats) {
	if (m_entries.empty()) {
		outcome = MemoOutcome::Empty;
		return nullptr;
	}
	if (uses_image_heap && m_capacity < 2u) {
		outcome = MemoOutcome::HeapPlan;
		return nullptr;
	}
	const Entry* newest = m_entries.front().get();
	for (const auto& entry: m_entries) {
		if (entry->used > newest->used) newest = entry.get();
	}
	MemoOutcome key_reason = MemoOutcome::Empty;
	MemoOutcome content    = MemoOutcome::Hit;
	for (const auto& candidate: m_entries) {
		auto&       entry = *candidate;
		MemoOutcome why   = MemoOutcome::Hit;
		if (!KeyMatches(entry, runtime, why)) {
			if (&entry == newest) key_reason = why;
			continue;
		}
		++stats.entries_tried;
		if (!ValidateReads(entry, runtime, why, stats)) {
			content = why;
			continue;
		}
		if (!ReplayHeap(entry, runtime)) {
			content = MemoOutcome::HeapChanged;
			continue;
		}
		entry.used = ++m_tick;
		outcome    = MemoOutcome::Hit;
		return &entry;
	}
	outcome = content != MemoOutcome::Hit ? content : key_reason;
	return nullptr;
}

MemoizedMaterialization::Entry* MemoizedMaterialization::Store(
    const ShaderRecompiler::IR::SrtRuntime&             runtime,
    const ShaderRecompiler::IR::ResourceSnapshot&       resources,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint64_t specialization_hash,
    RecordingImageHeap& heap, bool uses_image_heap, uint32_t& evicted, MemoLookupStats& stats) {
	if ((uses_image_heap && m_capacity < 2u) || heap.Declined()) return nullptr;
	// The heap tables are covered by the heap's own change tracking (replayed on a hit).
	auto ranges = resources.specialization_reads;
	for (const auto& acquired: heap.Take()) {
		const auto table = std::ranges::find(
		    ranges, std::pair<uint64_t, uint64_t> {acquired.request.base, acquired.request.size});
		if (table != ranges.end()) ranges.erase(table);
	}
	std::ranges::sort(ranges);
	ranges.erase(std::ranges::unique(ranges).begin(), ranges.end());

	auto fresh                   = std::make_unique<Entry>();
	fresh->user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	fresh->workgroup_counts.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
	fresh->shader_base           = runtime.shader_base;
	fresh->indirect_args_address = runtime.indirect_args_address;
	fresh->legacy_hash           = runtime.legacy_hash;
	fresh->reads.reserve(ranges.size());
	for (const auto& [address, size]: ranges) {
		Read read {.address = address, .size = size};
		// Watch before reading, so a write after the read changes the stamps.
		if (m_stamps != nullptr && size != 0u) {
			read.first_page = static_cast<uint32_t>(fresh->pages.size());
			read.tracked    = m_stamps->Track(address, size, fresh->pages);
			read.page_count = read.tracked ? static_cast<uint32_t>(fresh->pages.size()) - read.first_page
			                               : 0u;
		}
		if (!HashRange(runtime, address, size, read.hash)) return nullptr;
		++stats.ranges_compared;
		stats.bytes_compared += size;
		fresh->reads.push_back(read);
	}
	fresh->heap                = std::move(heap.Take());
	fresh->resources           = resources;
	fresh->specialization      = specialization;
	fresh->specialization_hash = specialization_hash;
	fresh->serial              = ++m_serials;
	fresh->used                = ++m_tick;
	if (m_entries.size() >= m_capacity) {
		const auto oldest = std::ranges::min_element(
		    m_entries, [](const auto& a, const auto& b) { return a->used < b->used; });
		m_entries.erase(oldest);
		++evicted;
	}
	m_entries.push_back(std::move(fresh));
	return m_entries.back().get();
}

} // namespace Libs::Graphics
