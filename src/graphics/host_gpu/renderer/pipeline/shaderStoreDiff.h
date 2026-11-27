#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREDIFF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREDIFF_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <unordered_set>
#include <vector>

// Counting the lookups of the shader cache, and saying which part of a key differs from the
// nearest one that exists, so that a module built again after it was stored can be explained.
namespace Libs::Graphics::StoreDiff {

// What the cache's lookups ended in. A lookup is one call that looks for a module for a stage.
struct LookupStats {
	uint64_t lookups     = 0;
	// A module was found.
	uint64_t hit         = 0;
	// The program is known (built earlier or loaded from the store) but none of its modules fit the
	// resources and push window of the call.
	uint64_t miss_key    = 0;
	// The cache has no program for the stage's key.
	uint64_t miss_absent = 0;
	// Requests that found no module at first and were served after a retry (a suspended dispatch
	// whose shader was built meanwhile).
	uint64_t served_late = 0;
	// Misses that were counted once per distinct key and resources, not once per retry.
	uint64_t distinct_misses = 0;
	// How many misses were described so far.
	uint32_t described = 0;
	std::unordered_set<uint64_t> seen;

	static constexpr uint32_t MaxDescribed = 20;

	// True the first time the miss identified by `identity` is seen.
	bool NoteDistinct(uint64_t identity) {
		if (!seen.insert(identity).second) {
			return false;
		}
		distinct_misses++;
		return true;
	}
	[[nodiscard]] bool WantsDescription() const { return described < MaxDescribed; }
	[[nodiscard]] std::string Line() const {
		auto line = fmt::format("[store] lookups={} hit={} miss-key={} miss-absent={} distinct-miss={}",
		                        lookups, hit, miss_key, miss_absent, distinct_misses);
		if (served_late != 0) {
			line += fmt::format(" served-late={}", served_late);
		}
		return line;
	}
};

// How two resource specializations differ: how many fields, and the first of them.
struct SpecDiff {
	size_t      count = 0;
	std::string first;
	void        Note(std::string text) {
        if (count++ == 0) {
            first = std::move(text);
        }
	}
};

[[nodiscard]] inline SpecDiff
DiffSpecialization(const ShaderRecompiler::IR::ResourceSpecialization& have,
                   const ShaderRecompiler::IR::ResourceSpecialization& want) {
	SpecDiff diff;
	const auto lists = [&](const char* name, size_t a, size_t b) {
		if (a != b) {
			diff.Note(fmt::format("{} count {} vs {}", name, a, b));
		}
	};
	lists("buffers", have.buffers.size(), want.buffers.size());
	lists("images", have.images.size(), want.images.size());
	lists("samplers", have.samplers.size(), want.samplers.size());
#define KYTY_SPEC_FIELD(list, index, field)                                                      \
	if (have.list[index].field != want.list[index].field) {                                      \
		diff.Note(fmt::format(#list "[{}]." #field " {} vs {}", index,                           \
		                      static_cast<uint64_t>(have.list[index].field),                     \
		                      static_cast<uint64_t>(want.list[index].field)));                   \
	}
	for (size_t i = 0; i < std::min(have.buffers.size(), want.buffers.size()); i++) {
		KYTY_SPEC_FIELD(buffers, i, packed_stride)
		KYTY_SPEC_FIELD(buffers, i, descriptor_format)
		KYTY_SPEC_FIELD(buffers, i, descriptor_swizzle)
		KYTY_SPEC_FIELD(buffers, i, zero_stride_oob)
		KYTY_SPEC_FIELD(buffers, i, scalar_empty)
		KYTY_SPEC_FIELD(buffers, i, indirect_root)
		KYTY_SPEC_FIELD(buffers, i, indirect_mapping_offset)
		KYTY_SPEC_FIELD(buffers, i, indirect_search_iterations)
	}
	for (size_t i = 0; i < std::min(have.images.size(), want.images.size()); i++) {
		KYTY_SPEC_FIELD(images, i, numeric_class)
		KYTY_SPEC_FIELD(images, i, dimension)
		KYTY_SPEC_FIELD(images, i, mip_count)
		KYTY_SPEC_FIELD(images, i, conversion_format)
		KYTY_SPEC_FIELD(images, i, shader_swizzle)
		KYTY_SPEC_FIELD(images, i, srgb_decode)
		KYTY_SPEC_FIELD(images, i, indirect_root)
		KYTY_SPEC_FIELD(images, i, indirect_mapping_offset)
		KYTY_SPEC_FIELD(images, i, indirect_search_iterations)
		KYTY_SPEC_FIELD(images, i, cube)
		KYTY_SPEC_FIELD(images, i, skip_degamma)
		KYTY_SPEC_FIELD(images, i, fmask)
		KYTY_SPEC_FIELD(images, i, heap)
	}
	for (size_t i = 0; i < std::min(have.samplers.size(), want.samplers.size()); i++) {
		KYTY_SPEC_FIELD(samplers, i, indirect_root)
		KYTY_SPEC_FIELD(samplers, i, indirect_mapping_offset)
		KYTY_SPEC_FIELD(samplers, i, indirect_search_iterations)
	}
#undef KYTY_SPEC_FIELD
	return diff;
}

// What a stored module was built for, to compare with what a call wants.
struct ModuleInputs {
	uint64_t                                            id = 0;
	// The start of the push window the module was built for, and the start the call's window has
	// for the module's layout.
	uint32_t                                            push_start = 0;
	uint32_t                                            want_push_start = 0;
	const ShaderRecompiler::IR::ResourceSpecialization* specialization = nullptr;
};

// The part of the wanted inputs that differs from the nearest of `modules` (fewest differences).
[[nodiscard]] inline std::string
DescribeNearestModule(const std::vector<ModuleInputs>& modules,
                      const ShaderRecompiler::IR::ResourceSpecialization& want) {
	if (modules.empty()) {
		return "the program has no module";
	}
	size_t      best_cost = SIZE_MAX;
	std::string best;
	for (const auto& module: modules) {
		const auto diff = DiffSpecialization(*module.specialization, want);
		const bool push = module.push_start != module.want_push_start;
		const auto cost = diff.count + (push ? 1u : 0u);
		if (cost >= best_cost) {
			continue;
		}
		best_cost = cost;
		best      = fmt::format("nearest module 0x{:016x} of {}: ", module.id, modules.size());
		if (push) {
			best += fmt::format("push window start {} vs {}", module.push_start, module.want_push_start);
			if (diff.count != 0) {
				best += "; ";
			}
		}
		if (diff.count != 0) {
			best += fmt::format("{} (+{} more)", diff.first, diff.count - 1);
		}
		if (cost == 0) {
			best += "no difference in specialization or push window (function code differs)";
		}
	}
	return best;
}

// Which part of a program key differs: the user data count or one word of the static state.
[[nodiscard]] inline std::string DescribeKeyDifference(uint32_t have_user_data,
                                                       const std::vector<uint32_t>& have_state,
                                                       uint32_t want_user_data,
                                                       const std::vector<uint32_t>& want_state) {
	std::string text;
	size_t      count = 0;
	const auto  note  = [&](std::string part) {
        if (count++ == 0) {
            text = std::move(part);
        }
	};
	if (have_user_data != want_user_data) {
		note(fmt::format("user_data_count {} vs {}", have_user_data, want_user_data));
	}
	if (have_state.size() != want_state.size()) {
		note(fmt::format("static_state size {} vs {}", have_state.size(), want_state.size()));
	}
	for (size_t i = 0; i < std::min(have_state.size(), want_state.size()); i++) {
		if (have_state[i] != want_state[i]) {
			note(fmt::format("static_state[{}] 0x{:x} vs 0x{:x}", i, have_state[i], want_state[i]));
		}
	}
	return count == 0 ? std::string("identical") : fmt::format("{} (+{} more)", text, count - 1);
}

} // namespace Libs::Graphics::StoreDiff

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREDIFF_H_
