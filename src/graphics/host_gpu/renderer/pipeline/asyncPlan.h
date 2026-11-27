#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPLAN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPLAN_H_

#include <algorithm>
#include <cstdint>
#include <vector>

// The decisions of --async-shader-compile that need no Vulkan, so that they can be tested: what a
// request for a stage does, whether a translation may also build its module, and which waiting
// translations are dropped for the memory budget. pipelineAsync.cpp applies them.
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
		// The module job took the translation, so a running one is joined whether or not a
		// translation is still waiting: asking for a new translation then would redo the work.
		if (state.module_job_in_flight) {
			return Action::JoinModuleJob;
		}
		if (state.translation_ready) {
			return Action::PostModuleJob;
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
// came back, and building the same module again would repeat for ever. `mode` is
// --async-speculative-modules: 0 never; 1 only when the request carries the user data the dispatch
// will have (a dispatch itself, or the scan of the command stream); 2 also for the registration
// prefetch, which has none, so that its module is built from all-zero user data (the old
// behaviour, which yields a module no dispatch matches).
[[nodiscard]] constexpr bool MaySpeculate(uint32_t mode, bool compute, bool has_program,
                                          bool user_data_known) {
	return mode != 0 && compute && !has_program && (user_data_known || mode >= 2);
}

// Whether a prefetch translates a compute shader now. The scan of a submitted command buffer sees a
// dispatch that is about to run, so it always does. A registration only says the shader exists: it
// carries no user data and guesses default dispatch flags (async_compute=false,
// dispatch_thread_dimensions=false). Translating a registration solely because another ProgramKey
// of the same hash exists in the cache produces a translation under a mismatched key that no
// dispatch ever claims. So a registration translates only when user data are known (lookahead);
// otherwise it stays registered until a real dispatch or command-stream scan names the true key.
[[nodiscard]] constexpr bool TranslateAhead(bool from_registration, bool user_data_known,
                                            bool /*cache_has_hash*/) {
	return !from_registration || user_data_known;
}

// How long a finished translation may wait unclaimed in the translation table before it is retired
// loudly and its IR freed. Waiting entries are passive (no worker claims them without a matching
// draw or dispatch), so an unbounded wait keeps orphaned translations alive for the entire run.
// 0 = no time limit.
inline constexpr uint64_t DefaultWaitingTtlMs = 30'000;

// The memory the finished translations that no draw has asked for yet may hold, as the sum of their
// estimated IR sizes. Only when the translations still hold more than this are the least recently
// requested ones dropped, and a request for a dropped stage translates it again. 0 = no limit.
inline constexpr uint64_t DefaultIrBudgetBytes = uint64_t {2048} << 20;

// When a translation was last wanted: the later of its finish and the last request for it.
template <typename Entry> [[nodiscard]] constexpr uint64_t LastRequested(const Entry& entry) {
	return std::max(entry.since, entry.asked);
}

// What dropping for the budget took.
struct Dropped {
	size_t   count = 0;
	uint64_t bytes = 0;
	// How long ago the most recently requested of the dropped entries was requested.
	uint64_t newest_age_ms = 0;
	// What the table holds afterwards.
	size_t   kept       = 0;
	uint64_t kept_bytes = 0;
};

// Moves entries of `table` (key -> {since, asked, bytes, ...}) into `dropped`, least recently
// requested first, until the table holds at most `budget_bytes`; the caller frees them where it
// chooses. Nothing is dropped while the table is within the budget or the budget is 0.
template <typename Table, typename Out>
Dropped TakeOverBudget(Table& table, uint64_t now, uint64_t budget_bytes, Out& dropped) {
	Dropped result;
	uint64_t total = 0;
	for (const auto& entry: table) {
		total += entry.second.bytes;
	}
	if (budget_bytes != 0 && total > budget_bytes) {
		std::vector<typename Table::iterator> order;
		order.reserve(table.size());
		for (auto it = table.begin(); it != table.end(); ++it) {
			order.push_back(it);
		}
		std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
			return LastRequested(a->second) < LastRequested(b->second);
		});
		for (const auto& it: order) {
			if (total <= budget_bytes) {
				break;
			}
			total -= it->second.bytes;
			result.bytes += it->second.bytes;
			result.count++;
			result.newest_age_ms = now - LastRequested(it->second);
			dropped.push_back(std::move(it->second));
			table.erase(it);
		}
	}
	result.kept       = table.size();
	result.kept_bytes = total;
	return result;
}

// Moves entries of `table` whose last request is at least `ttl_ms` old into `dropped` so unclaimed
// waiting translations cannot linger indefinitely. Nothing is retired when `ttl_ms` is 0.
template <typename Table, typename Out>
Dropped TakeExpired(Table& table, uint64_t now, uint64_t ttl_ms, Out& dropped) {
	Dropped  result;
	uint64_t kept_bytes = 0;
	for (auto it = table.begin(); it != table.end();) {
		const auto last = LastRequested(it->second);
		const auto age  = now >= last ? now - last : uint64_t {0};
		if (ttl_ms != 0 && age >= ttl_ms) {
			result.bytes += it->second.bytes;
			result.count++;
			result.newest_age_ms =
			    result.count == 1 ? age : std::min(result.newest_age_ms, age);
			dropped.push_back(std::move(it->second));
			it = table.erase(it);
		} else {
			kept_bytes += it->second.bytes;
			++it;
		}
	}
	result.kept       = table.size();
	result.kept_bytes = kept_bytes;
	return result;
}

// A draw into a target the game was not seen to redraw waits for its shaders (see the decision in
// pipelineCache.cpp). "Redrawn" counts frames, so while no frame is recorded the count stands
// still and no target first drawn in that time can ever qualify: every such draw then builds on
// the render thread, one after another, and the frame it belongs to cannot finish. A title that
// loads a scene behind one long frame (a few hundred new shaders) does not get out of it, and a
// title with a hang watchdog aborts. The tracker tells how long the current frame has lasted, so
// that such draws may be skipped like redrawn ones once it has lasted `stall_ms`
// (--defer-new-targets-on-stall-ms; 0 = never).
struct FrameStallTracker {
	enum class Change : uint8_t { None, Started, Ended };

	uint64_t frame    = ~uint64_t {0};
	uint64_t since_ms = 0;
	bool     stalled  = false;
	// Draws let through while stalled: in the current (or last) stall, and in all.
	uint64_t deferred       = 0;
	uint64_t deferred_total = 0;

	// Called for a draw that would otherwise wait for its shaders. True when it may be skipped.
	// `change` reports the edges, for the log line.
	bool Update(uint64_t frame_now, uint64_t now_ms, uint32_t stall_ms, Change* change) {
		Change edge = Change::None;
		if (frame_now != frame) {
			if (stalled) edge = Change::Ended;
			frame    = frame_now;
			since_ms = now_ms;
			stalled  = false;
		}
		const bool want = stall_ms != 0 && now_ms >= since_ms && now_ms - since_ms >= stall_ms;
		if (want && !stalled) {
			stalled  = true;
			deferred = 0;
			edge     = Change::Started;
		}
		if (stalled) {
			deferred++;
			deferred_total++;
		}
		if (change != nullptr) *change = edge;
		return stalled;
	}
};

} // namespace Libs::Graphics::AsyncPlan

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPLAN_H_
