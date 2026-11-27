#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
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
		std::vector<DescriptorValue> entries;
		std::vector<uint32_t>        class_index;
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
