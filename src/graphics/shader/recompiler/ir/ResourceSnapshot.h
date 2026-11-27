#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

// Immutable after one materialization finishes. The scan cache, materialization memo and active
// snapshot share this payload, so replaying a table does not allocate or copy thousands of values.
struct RuntimeBufferTableCandidates {
	std::array<uint64_t, 2>      key {};
	bool                         has_live_range = false;
	bool                         canonicalized = false;
	std::vector<DescriptorValue> entries;
	std::vector<uint32_t>        class_index;
};

// Opaque watch over guest memory used while materializing resources. A valid stamp proves the
// watched bytes are unchanged without reading or hashing them again.
class MemoryChangeStamp {
public:
	virtual ~MemoryChangeStamp() = default;
	[[nodiscard]] virtual bool Unchanged() const = 0;
};

class MemoryChangeSource {
public:
	virtual ~MemoryChangeSource() = default;
	[[nodiscard]] virtual std::shared_ptr<MemoryChangeStamp> Track(uint64_t address,
	                                                              uint64_t size) = 0;
};

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

struct ResourceSnapshot {
	// Runtime-indexed table for an indirect buffer root whose distinct descriptors exceeded
	// MaxIndirectBufferCandidates. Each entry maps a runtime buffer ordinal (1..entries.size()) to
	// its live guest V# and its 1-based specialization class child ordinal.
	struct IndirectBufferTable {
		uint32_t                     root_resource = 0;
		// Word index in flattened_srt holding [table_bda_lo, table_bda_hi, runtime_entry_count].
		uint32_t                     header_word = 0;
		// Address-free identity of the normalized candidates and their specialization classes.
		// The renderer uses both halves, plus the module/root identity, as its persistent BDA-table key.
			std::shared_ptr<RuntimeBufferTableCandidates> candidates =
			    std::make_shared<RuntimeBufferTableCandidates>();
	};

	std::vector<DescriptorValue>     buffers;
	std::vector<DescriptorValue>     images;
	std::vector<DescriptorValue>     samplers;
	std::vector<uint32_t>            flattened_srt;
	std::vector<uint32_t>            user_data;
	std::vector<IndirectBufferTable> indirect_buffer_tables;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	// With SrtRuntime::legacy_hash under the canonical-resources port: the hash of the
	// specialization the layout without the port would have given these inputs; else 0.
	uint64_t                    legacy_hash = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
