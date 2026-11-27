#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <xxhash.h>

#include <algorithm>
#include <cstring>
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

ShaderRecompiler::IR::SrtRuntime WalkReadHashes::Wrap(const ShaderRecompiler::IR::SrtRuntime& runtime) {
	m_inner = runtime;
	auto wrapped = runtime;
	if (runtime.read_specialization_memory != nullptr) {
		wrapped.userdata                   = this;
		wrapped.read_specialization_memory = &WalkReadHashes::Read;
	}
	return wrapped;
}

bool WalkReadHashes::Read(void* user, uint64_t address, std::span<uint32_t> values) {
	auto& self = *static_cast<WalkReadHashes*>(user);
	if (!self.m_inner.read_specialization_memory(self.m_inner.userdata, address, values)) return false;
	const uint64_t hash = values.empty() ? 0u : XXH3_64bits(values.data(), values.size_bytes());
	const auto [at, fresh] = self.m_seen.try_emplace(std::pair<uint64_t, uint64_t> {address, values.size_bytes()},
		                                                 Seen {hash, false});
	if (!fresh && at->second.hash != hash) at->second.unstable = true;
	return true;
}

WalkReadHashes::State WalkReadHashes::Find(uint64_t address, uint64_t size, uint64_t& hash) const {
	const auto found = m_seen.find(std::pair<uint64_t, uint64_t> {address, size});
	if (found == m_seen.end()) return State::NotRead;
	hash = found->second.hash;
	return found->second.unstable ? State::Unstable : State::Stable;
}

void MemoizedMaterialization::Configure(uint32_t entries, ReadStamps* stamps) {
	m_capacity = std::max(entries, 1u);
	m_stamps   = stamps;
	while (m_entries.size() > m_capacity) m_entries.pop_back();
}

void MemoizedMaterialization::Clear() {
	m_entries.clear();
	current = 0;
	++m_generation;
}

std::shared_ptr<MemoizedMaterialization::Entry> MemoizedMaterialization::ShareCurrent() const {
	if (current == 0) return {};
	for (const auto& entry: m_entries) {
		if (entry->serial == current) return entry;
	}
	return {};
}

uint64_t MemoizedMaterialization::ApproximateBytes(const Entry& entry_ref) {
	const auto* entry = &entry_ref;
	uint64_t bytes = sizeof(Entry);
	bytes += (entry->user_data.capacity() + entry->workgroup_counts.capacity()) * sizeof(uint32_t);
	bytes += entry->reads.capacity() * sizeof(Read) + entry->bytes.capacity();
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
	return bytes;
}

uint64_t MemoizedMaterialization::ApproximateBytes() const {
	uint64_t bytes = 0;
	for (const auto& entry: m_entries) bytes += ApproximateBytes(*entry);
	return bytes;
}

bool MemoizedMaterialization::KeyMatches(const Entry&                                  entry,
                                         const ShaderRecompiler::IR::SrtRuntime& runtime,
                                         MemoOutcome& why, size_t user_data_words) const {
	const auto user_data_equal = [&] {
		if (user_data_words == AllUserData) return std::ranges::equal(entry.user_data, runtime.user_data);
		return entry.user_data.size() == runtime.user_data.size() &&
		       user_data_words <= entry.user_data.size() &&
		       std::equal(entry.user_data.begin(), entry.user_data.begin() + user_data_words,
		                  runtime.user_data.begin());
	};
	if (entry.shader_base != runtime.shader_base) {
		why = MemoOutcome::ShaderBase;
	} else if (entry.indirect_args_address != runtime.indirect_args_address) {
		why = MemoOutcome::IndirectArgs;
	} else if (entry.legacy_hash != runtime.legacy_hash) {
		why = MemoOutcome::LegacyHash;
	} else if (!user_data_equal()) {
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
	constexpr uint64_t Chunk = 32u << 10u;
	if (size <= Chunk) {
		m_scratch.resize(static_cast<size_t>(size / sizeof(uint32_t)));
		if (!runtime.read_specialization_memory(runtime.userdata, address, m_scratch)) return false;
		hash = XXH3_64bits(m_scratch.data(), static_cast<size_t>(size));
		return true;
	}
	// A large table is read and hashed in pieces that stay in the cache; the streamed hash equals
	// the one-shot hash of the same bytes.
	m_scratch.resize(static_cast<size_t>(Chunk / sizeof(uint32_t)));
	XXH3_state_t* state = XXH3_createState();
	if (state == nullptr) return false;
	XXH3_64bits_reset(state);
	for (uint64_t done = 0; done < size; done += Chunk) {
		const auto piece = std::min(Chunk, size - done);
		const std::span<uint32_t> words(m_scratch.data(), static_cast<size_t>(piece / sizeof(uint32_t)));
		if (!runtime.read_specialization_memory(runtime.userdata, address + done, words)) {
			XXH3_freeState(state);
			return false;
		}
		XXH3_64bits_update(state, words.data(), static_cast<size_t>(piece));
	}
	hash = XXH3_64bits_digest(state);
	XXH3_freeState(state);
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
		bool same = false;
		if (read.has_bytes) {
			// Small: the bytes themselves decide, no hash.
			m_scratch.resize(static_cast<size_t>(read.size / sizeof(uint32_t)));
			if (!runtime.read_specialization_memory(runtime.userdata, read.address, m_scratch)) {
				why = MemoOutcome::Unreadable;
				return false;
			}
			same = std::memcmp(m_scratch.data(), entry.bytes.data() + read.byte_offset,
			                   static_cast<size_t>(read.size)) == 0;
			++stats.ranges_small;
		} else {
			uint64_t hash = 0;
			if (!HashRange(runtime, read.address, read.size, hash)) {
				why = MemoOutcome::Unreadable;
				return false;
			}
			same = hash == read.hash;
			++stats.ranges_large;
			stats.bytes_large += read.size;
		}
		++stats.ranges_compared;
		stats.bytes_compared += read.size;
		if (!same) {
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

bool MemoizedMaterialization::Revalidate(Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
                                         size_t user_data_words, MemoOutcome& outcome,
                                         MemoLookupStats& stats) {
	outcome = MemoOutcome::Hit;
	if (!KeyMatches(entry, runtime, outcome, user_data_words)) return false;
	++stats.entries_tried;
	if (!ValidateReads(entry, runtime, outcome, stats)) return false;
	if (!ReplayHeap(entry, runtime)) {
		outcome = MemoOutcome::HeapChanged;
		return false;
	}
	entry.used = ++m_tick;
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
    RecordingImageHeap& heap, bool uses_image_heap, uint32_t& evicted, MemoLookupStats& stats,
    const WalkReadHashes* walk) {
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

	auto fresh                   = std::make_shared<Entry>();
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
		if (size != 0u && size <= SmallRangeBytes) {
			// HashRange left the bytes in the scratch buffer.
			read.has_bytes   = true;
			read.byte_offset = static_cast<uint32_t>(fresh->bytes.size());
			const auto* data = reinterpret_cast<const uint8_t*>(m_scratch.data());
			fresh->bytes.insert(fresh->bytes.end(), data, data + size);
		}
		if (walk != nullptr) {
			// The range is armed now. If it holds other bytes than the walk read, a write landed
			// in between and the snapshot describes the older bytes: keep no entry for it.
			uint64_t seen = 0;
			const auto state = walk->Find(address, size, seen);
			if (state == WalkReadHashes::State::Unstable ||
			    (state == WalkReadHashes::State::Stable && seen != read.hash)) {
				stats.walk_changed = true;
				return nullptr;
			}
		}
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
