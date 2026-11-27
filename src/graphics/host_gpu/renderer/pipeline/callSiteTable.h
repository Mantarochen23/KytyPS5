#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CALLSITETABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CALLSITETABLE_H_

#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

// The command packet the command processor is executing on this thread (the address of its header),
// or 0 outside packet execution. A hint for CallSiteTable only: whatever it holds, a record is
// reused solely after its registers and bytes are compared.
inline thread_local uint64_t g_current_call_site = 0;

// Where a lookup was made: the packet that asked, the shader and the stage it asked for.
struct CallSiteKey {
	uint64_t packet      = 0;
	uint64_t shader_hash = 0;
	uint8_t  stage       = 0;
	[[nodiscard]] bool operator==(const CallSiteKey&) const = default;
};

// Why a call-site lookup did not reuse a record.
enum class CallSiteOutcome : uint8_t {
	Hit,
	NoRecord,    // nothing recorded for this call site (first call, or evicted)
	Stale,       // the owner replaced its plan since the record was made
	Registers,   // the walk's user-data words, a base or the dispatch inputs differ
	Ranges,      // a recorded range holds other bytes (or cannot be read)
	Heap,        // the resident heap hands out something else
	Count
};

struct CallSiteStats {
	uint32_t ranges_small   = 0;
	uint32_t ranges_large   = 0;
	uint32_t ranges_stamped = 0;
	uint64_t bytes_compared = 0;
	uint64_t bytes_large    = 0;
};

// A fixed-size, set-associative table of per-call-site records. A record holds the materialization
// a call site produced (the memo entry: the walk's registers, the guest ranges it read and their
// bytes, the specialization) and the module chosen for it. The module is selected by the
// specialization alone; the record reuses it when the registers the walk reads and the recorded
// bytes are what they were. No write stamp is required for correctness: without stamps every range
// is compared (small ones by memcmp, large ones by hash), with them an unchanged range is not read.
// Render thread only.
class CallSiteTable {
public:
	struct Record {
		bool                                            used = false;
		CallSiteKey                                     key;
		const void*                                     owner      = nullptr;
		uint64_t                                        generation = 0;
		std::shared_ptr<MemoizedMaterialization::Entry> entry;
		void*                                           module     = nullptr;
		uint64_t                                        touched    = 0;
		uint64_t                                        bytes      = 0;
	};

	CallSiteTable(uint32_t sets, uint32_t ways);

	// Finds the record of `key` and checks it against the inputs. `owner` is the program,
	// `generation` its memo generation; `user_data_words` how many user-data words the walk reads.
	// On Hit, `found` is the checked record and its entry is brought up to the current user data.
	[[nodiscard]] CallSiteOutcome Lookup(const CallSiteKey& key, const void* owner, uint64_t generation,
	                                     MemoizedMaterialization&                memo,
	                                     const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                     size_t user_data_words, Record*& found, CallSiteStats& stats);
	// Records the result of a lookup. `evicted` is set when a record of another call site is lost.
	Record& Store(const CallSiteKey& key, const void* owner, uint64_t generation,
	              std::shared_ptr<MemoizedMaterialization::Entry> entry, void* module,
	              bool& evicted);
	void Clear();

	[[nodiscard]] uint64_t Capacity() const { return m_records.size(); }
	[[nodiscard]] uint64_t Live() const { return m_live; }
	[[nodiscard]] uint64_t Bytes() const { return m_bytes; }

private:
	Record* Find(const CallSiteKey& key);
	[[nodiscard]] uint32_t SetOf(const CallSiteKey& key) const;

	std::vector<Record> m_records;
	uint32_t            m_sets  = 1;
	uint32_t            m_ways  = 1;
	uint64_t            m_tick  = 0;
	uint64_t            m_live  = 0;
	uint64_t            m_bytes = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CALLSITETABLE_H_
