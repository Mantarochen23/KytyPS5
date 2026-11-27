#include "graphics/host_gpu/renderer/pipeline/callSiteTable.h"

#include <xxhash.h>

#include <algorithm>

namespace Libs::Graphics {

CallSiteTable::CallSiteTable(uint32_t sets, uint32_t ways):
    m_sets(std::max(sets, 1u)), m_ways(std::max(ways, 1u)) {
	m_records.resize(static_cast<size_t>(m_sets) * m_ways);
}

uint32_t CallSiteTable::SetOf(const CallSiteKey& key) const {
	const uint64_t words[3] = {key.packet, key.shader_hash, key.stage};
	return static_cast<uint32_t>(XXH3_64bits(words, sizeof(words)) % m_sets);
}

CallSiteTable::Record* CallSiteTable::Find(const CallSiteKey& key) {
	auto* set = &m_records[static_cast<size_t>(SetOf(key)) * m_ways];
	for (uint32_t way = 0; way < m_ways; ++way) {
		if (set[way].used && set[way].key == key) return &set[way];
	}
	return nullptr;
}

CallSiteOutcome CallSiteTable::Lookup(const CallSiteKey& key, const void* owner, uint64_t generation,
                                      MemoizedMaterialization&                memo,
                                      const ShaderRecompiler::IR::SrtRuntime& runtime,
                                      size_t user_data_words, Record*& found, CallSiteStats& stats) {
	found        = nullptr;
	auto* record = Find(key);
	if (record == nullptr) return CallSiteOutcome::NoRecord;
	if (record->owner != owner || record->generation != generation || record->entry == nullptr) {
		return CallSiteOutcome::Stale;
	}
	MemoOutcome     outcome = MemoOutcome::Hit;
	MemoLookupStats memo_stats;
	// Whether the old layout's hash was also derived is a diagnostic of the entry, not an input.
	auto probe        = runtime;
	probe.legacy_hash = record->entry->legacy_hash;
	const bool same   = memo.Revalidate(*record->entry, probe, user_data_words, outcome, memo_stats);
	stats.ranges_small += memo_stats.ranges_small;
	stats.ranges_large += memo_stats.ranges_large;
	stats.ranges_stamped += memo_stats.ranges_stamped;
	stats.bytes_compared += memo_stats.bytes_compared;
	stats.bytes_large += memo_stats.bytes_large;
	if (!same) {
		switch (outcome) {
			case MemoOutcome::HeapChanged: return CallSiteOutcome::Heap;
			case MemoOutcome::RangeChanged:
			case MemoOutcome::Unreadable: return CallSiteOutcome::Ranges;
			default: return CallSiteOutcome::Registers;
		}
	}
	// The result does not depend on the words the walk does not read; keep the entry's copy of the
	// vector (and the one inside its snapshot, which the draw reads offsets from) current.
	auto& entry = *record->entry;
	entry.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	entry.resources.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	record->touched = ++m_tick;
	found           = record;
	return CallSiteOutcome::Hit;
}

CallSiteTable::Record& CallSiteTable::Store(const CallSiteKey& key, const void* owner,
                                            uint64_t                                        generation,
                                            std::shared_ptr<MemoizedMaterialization::Entry> entry,
                                            void* module, bool& evicted) {
	evicted      = false;
	auto* record = Find(key);
	if (record == nullptr) {
		auto* set = &m_records[static_cast<size_t>(SetOf(key)) * m_ways];
		record    = &set[0];
		for (uint32_t way = 0; way < m_ways; ++way) {
			if (!set[way].used) {
				record = &set[way];
				break;
			}
			if (set[way].touched < record->touched) record = &set[way];
		}
		if (record->used) {
			evicted = true;
			m_bytes -= std::min(m_bytes, record->bytes);
		} else {
			++m_live;
		}
	} else {
		m_bytes -= std::min(m_bytes, record->bytes);
	}
	record->used       = true;
	record->key        = key;
	record->owner      = owner;
	record->generation = generation;
	record->entry      = std::move(entry);
	record->module     = module;
	record->touched    = ++m_tick;
	record->bytes =
	    record->entry != nullptr ? MemoizedMaterialization::ApproximateBytes(*record->entry) : 0u;
	m_bytes += record->bytes;
	return *record;
}

void CallSiteTable::Clear() {
	for (auto& record: m_records) record = {};
	m_live  = 0;
	m_bytes = 0;
}

} // namespace Libs::Graphics
