#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  IR::PushData::Window push_data, const Profile& profile,
                                  bool emit_debug_info, EmitStats* stats) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {};
	EmitterState state(program, input_info, profile);
	state.push_data = push_data;
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	// Debug loop guard (off by default). Mesh shaders are skipped: their entry point is built
	// separately and would have to list the counter variables.
	if (program.stage == ShaderType::Compute) {
		state.loop_guard_limit = profile.loop_guard_compute_limit;
	} else if (program.stage != ShaderType::Mesh) {
		state.loop_guard_limit = profile.loop_guard_limit;
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	auto spirv = state.builder.Build();
	if (stats != nullptr) {
		*stats             = {};
		stats->total_words = spirv.size();
		std::vector<EmitContributor> entries;
		entries.reserve(state.emit_contributors.size() + 1u);
		uint64_t attributed = 0u;
		for (const auto& [key, entry]: state.emit_contributors) {
			attributed += entry.words;
			entries.push_back(entry);
		}
		if (spirv.size() > attributed) {
			entries.push_back({.kind       = "other",
			                   .resource   = 0u,
			                   .candidates = 0u,
			                   .accesses   = 1u,
			                   .words      = spirv.size() - attributed});
		}
		std::ranges::sort(entries, [](const EmitContributor& a, const EmitContributor& b) {
			return a.words > b.words;
		});
		const auto count = std::min<size_t>(entries.size(), stats->top_contributors.size());
		stats->contributor_count = static_cast<uint32_t>(count);
		for (size_t i = 0; i < count; ++i) {
			stats->top_contributors[i] = entries[i];
		}
	}
	return spirv;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
