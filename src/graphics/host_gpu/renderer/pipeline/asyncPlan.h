#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPLAN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPLAN_H_

#include <algorithm>
#include <cstdint>
#include <vector>

// The decisions of --async-shader-compile that need no Vulkan, so that they can be tested: what a
// request for a stage does, whether a translation may also build its module, and which waiting
// translations expire. pipelineAsync.cpp applies them.
namespace Libs::Graphics::AsyncPlan {

// What a draw or dispatch that found no module for its resources does for a stage.
enum class Action : uint8_t {
	// A module job for these resources is running; mark it waited-for.
	JoinModuleJob,
	// A translation is waiting for these resources; post the module job from it.
	PostModuleJob,
	// The translation job is running; mark it waited-for.
	JoinTranslation,
	// Nothing exists for the stage: post a translation job.
	PostTranslation,
	// Prefetch only and the program is already known; do nothing.
	Nothing,
};

struct RequestState {
	// The cache has a program for the stage, from an earlier translation or from the shader store.
	bool has_program = false;
	// Only a prefetch asks (no draw or dispatch waits for it).
	bool prefetch_only = false;
	// A finished translation waits in the translation table.
	bool translation_ready = false;
	// A module job for exactly these resources and push window is in flight.
	bool module_job_in_flight = false;
	// The translation job for the stage is in flight.
	bool translation_in_flight = false;
};

[[nodiscard]] constexpr Action Decide(const RequestState& state) {
	if (state.has_program) {
		if (state.prefetch_only) {
			return Action::Nothing;
		}
		if (state.translation_ready) {
			return state.module_job_in_flight ? Action::JoinModuleJob : Action::PostModuleJob;
		}
	}
	return state.translation_in_flight ? Action::JoinTranslation : Action::PostTranslation;
}

[[nodiscard]] constexpr const char* Name(Action action) {
	switch (action) {
		case Action::JoinModuleJob: return "module job running";
		case Action::PostModuleJob: return "module job posted";
		case Action::JoinTranslation: return "translation running";
		case Action::PostTranslation: return "translation posted";
		case Action::Nothing: return "known program, prefetch";
	}
	return "";
}

// Whether a translation job may also build the stage's module before any draw has named its
// resources. Only the first request for a stage may: when the cache already has the program, either
// it came from the shader store or an earlier speculative module did not serve the dispatch that
// came back, and building the same module again would repeat for ever.
[[nodiscard]] constexpr bool MaySpeculate(bool enabled, bool compute, bool has_program) {
	return enabled && compute && !has_program;
}

// How long a finished translation waits for a draw to ask for it.
inline constexpr uint64_t TranslationTtlMs = 60000;

// Whether a translation made at `since` and last asked for at `asked` has expired at `now`.
[[nodiscard]] constexpr bool Expired(uint64_t now, uint64_t since, uint64_t asked,
                                     uint64_t ttl_ms = TranslationTtlMs) {
	return now - std::max(since, asked) > ttl_ms;
}

// Moves the expired entries of `table` (key -> {since, asked, ...}) into `dropped`, so that the
// caller frees them where it chooses. A request for a dropped stage translates it again.
template <typename Table, typename Dropped>
size_t TakeExpired(Table& table, uint64_t now, Dropped& dropped, uint64_t ttl_ms = TranslationTtlMs) {
	size_t count = 0;
	for (auto it = table.begin(); it != table.end();) {
		if (Expired(now, it->second.since, it->second.asked, ttl_ms)) {
			dropped.push_back(std::move(it->second));
			it = table.erase(it);
			count++;
		} else {
			++it;
		}
	}
	return count;
}

} // namespace Libs::Graphics::AsyncPlan

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPLAN_H_
