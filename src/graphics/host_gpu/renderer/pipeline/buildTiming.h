#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BUILDTIMING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BUILDTIMING_H_

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <string_view>

// Where the time of building a shader module or a pipeline goes, one line per build. Nothing here
// needs Vulkan, so the line and the sums can be tested.
namespace Libs::Graphics::BuildTiming {

[[nodiscard]] inline uint64_t ElapsedUs(std::chrono::steady_clock::time_point begin) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now() - begin)
	                                 .count());
}

// The steps of one build, in microseconds. A step that did not run is 0.
struct Steps {
	// Guest code to IR, and the resource plan from the IR (the translation job).
	uint64_t translate_us = 0;
	uint64_t plan_us      = 0;
	// IR to SPIR-V words.
	uint64_t emit_us = 0;
	// The driver's module creation from the words.
	uint64_t module_us = 0;
	// The driver's pipeline creation.
	uint64_t pipeline_us = 0;
	// From the post of the job until a worker started it.
	uint64_t queue_wait_us = 0;
	// From the post of the job until it was done (queue wait included), or the time the render
	// thread spent when it built the step itself.
	uint64_t total_us = 0;
};

struct Build {
	// "cs", "vs", "ps", ... for a module, "pipeline-cs" or "pipeline-gfx" for a pipeline.
	const char* stage = "";
	uint64_t    hash  = 0;
	uint64_t    words = 0;
	Steps       steps;
	// "worker" or "render".
	const char* where = "worker";
	// "dispatch", "draw", "prefetch" or "warmup".
	const char* reason = "";
	// How often this hash was built in this run, 1 for the first; 0 = not counted.
	uint32_t    number = 0;
};

[[nodiscard]] inline double Ms(uint64_t us) { return static_cast<double>(us) / 1000.0; }

[[nodiscard]] inline std::string Line(const Build& build) {
	const auto& s = build.steps;
	auto line = fmt::format("[build] {} hash=0x{:016x} words={} translate={:.1f}ms plan={:.1f}ms "
	                   "emit={:.1f}ms module={:.1f}ms pipeline={:.1f}ms queue-wait={:.1f}ms "
	                   "total={:.1f}ms where={} reason={}",
	                   build.stage, build.hash, build.words, Ms(s.translate_us), Ms(s.plan_us),
	                   Ms(s.emit_us), Ms(s.module_us), Ms(s.pipeline_us), Ms(s.queue_wait_us),
	                   Ms(s.total_us), build.where, build.reason);
	if (build.number > 1) {
		// A repeat: the same hash built again in this run.
		line += fmt::format(" repeat=#{}", build.number);
	}
	return line;
}

// The builds of one status period (two seconds), for the `Async:` line.
struct Totals {
	uint64_t modules   = 0;
	uint64_t pipelines = 0;
	uint64_t translate_us = 0;
	uint64_t plan_us      = 0;
	uint64_t emit_us      = 0;
	uint64_t module_us    = 0;
	uint64_t pipeline_us  = 0;
	uint64_t queue_wait_us = 0;
	// Builds the render thread made itself, and the time they took.
	uint64_t render_builds = 0;
	uint64_t render_us     = 0;

	void Note(const Build& build) {
		const auto& s = build.steps;
		(build.words != 0 ? modules : pipelines)++;
		translate_us += s.translate_us;
		plan_us += s.plan_us;
		emit_us += s.emit_us;
		module_us += s.module_us;
		pipeline_us += s.pipeline_us;
		queue_wait_us += s.queue_wait_us;
		if (std::string_view(build.where) == "render") {
			render_builds++;
			render_us += s.total_us;
		}
	}
	[[nodiscard]] bool Empty() const { return modules + pipelines == 0; }
	[[nodiscard]] std::string Text() const {
		return fmt::format("built {} modules (translate {:.0f} ms, plan {:.0f} ms, emit {:.0f} ms, "
		                   "module {:.0f} ms), {} pipelines ({:.0f} ms), queue-wait {:.0f} ms, "
		                   "{} on the render thread ({:.0f} ms)",
		                   modules, Ms(translate_us), Ms(plan_us), Ms(emit_us), Ms(module_us),
		                   pipelines, Ms(pipeline_us), Ms(queue_wait_us), render_builds,
		                   Ms(render_us));
	}
};

} // namespace Libs::Graphics::BuildTiming

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BUILDTIMING_H_
