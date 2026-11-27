#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEASYNC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEASYNC_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderWorkers.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// The state --async-shader-compile adds to PipelineCache. Only pipelineCache.cpp and
// pipelineAsync.cpp include this.
namespace Libs::Graphics {

struct PipelineCache::StageInputs {
	ProgramKey key;
	std::variant<ShaderVertexInputInfo, ShaderPixelInputInfo, ShaderComputeInputInfo> info;
	std::vector<uint32_t>                                                             code;
	std::vector<uint32_t>                                     back_code;
	std::array<uint32_t, 40>                                  user_data {};
	ShaderRecompiler::Spirv::Profile                          profile;
	// The shader's guest address, for resolving its resources on a worker.
	uint64_t                                                  shader_base = 0;
	// Points into the members above, so a StageInputs is never copied or moved.
	ShaderRecompiler::CompileOptions                          options;

	StageInputs()                              = default;
	StageInputs(const StageInputs&)            = delete;
	StageInputs& operator=(const StageInputs&) = delete;
};

struct PipelineCache::Async {
	explicit Async(uint32_t threads): workers(threads) {}
	~Async();

	// A translation waiting for a draw to name the resources it is compiled for.
	struct Translation {
		std::shared_ptr<StageInputs>                       inputs;
		std::unique_ptr<ShaderRecompiler::TranslateResult> translated;
		// When the translation finished, and when a draw or dispatch last asked for it. Over
		// the IR budget the one with the older of the two goes first.
		uint64_t since = 0;
		uint64_t asked = 0;
		// Estimated memory of the translated IR.
		uint64_t bytes = 0;
		// What the translation job spent, to show in the build line of the module made from it.
		uint64_t translate_us = 0;
		uint64_t plan_us      = 0;
		// What asked for this translation ("prefetch", "dispatch", or "draw").
		const char* origin    = "unknown";
	};

	// A job a worker has or will run for a program, and what it does.
	struct Flight {
		ShaderWorkers::JobPtr job;
		bool                  translating = false;
	};

	// A stage compiled on a worker; frees the host module unless the cache took it.
	struct PendingStage {
		vk::Device    device;
		CompiledStage stage;
		bool          taken = false;

		PendingStage(vk::Device device, CompiledStage&& stage)
		    : device(device), stage(std::move(stage)) {}
		~PendingStage() {
			if (!taken && stage.module.module) {
				device.destroyShaderModule(stage.module.module, nullptr);
			}
		}
		PendingStage(const PendingStage&)            = delete;
		PendingStage& operator=(const PendingStage&) = delete;
	};

	// A finished job: `flight` is the id it was in `in_flight` under, 0 for pipelines.
	struct FinishedJob {
		uint64_t                     flight = 0;
		Common::UniqueFunction<void> apply;
		void operator()() { apply(); }
	};
	// What a worker finished, to be applied by the render thread.
	void Finish(Common::UniqueFunction<void>&& apply, uint64_t flight = 0);
	// Render thread only: a job's result is applied; it is no longer in flight or waited for.
	void Settle(uint64_t flight) {
		in_flight.erase(flight);
		wanted.erase(flight);
	}
	// Render thread only: a draw or dispatch is blocked on this job; it runs and is applied before
	// the jobs nobody waits for.
	void Want(uint64_t flight, const ShaderWorkers::JobPtr& job) {
		wanted.insert(flight);
		workers.MarkWaited(job);
	}
	// Render thread only: moves the finished jobs a draw waits for to the front of `backlog`.
	void PrioritizeBacklog();
	// Render thread only: moves what the workers finished into `backlog`.
	void TakeFinished();

	ShaderWorkers                             workers;
	std::mutex                                done_mutex;
	std::condition_variable                   done_cv;
	std::vector<FinishedJob> done;
	// Render thread only: finished jobs taken from `done` that PumpAsync's time budget has not
	// applied yet.
	std::deque<FinishedJob> backlog;
	// Render thread only, for the status line: time spent in PumpAsync and jobs applied since the
	// last report, and the worker job count at the last report.
	uint64_t pump_us          = 0;
	uint64_t pump_worst_us    = 0;
	uint32_t pump_applied     = 0;
	uint64_t reported_finished = 0;
	std::mutex                                notify_mutex;
	Common::UniqueFunction<void>              on_finish;
	// Render thread only: the translations and compilations being made, by id.
	std::unordered_map<uint64_t, Flight>                         in_flight;
	std::unordered_map<ProgramKey, Translation, ProgramKeyHash> translations;
	// Render thread only: compute shaders with GPU-selected calls (s_swappc_b64) that must
	// link callees from guest memory on the render thread.
	std::unordered_set<ProgramKey, ProgramKeyHash>               sync_only_keys;
	// Render thread only: the flights a draw or dispatch is blocked on.
	std::unordered_set<uint64_t> wanted;
	// Render thread only: pipelines being created on workers.
	uint32_t pipelines_building = 0;

	// Render thread only: what happened to the jobs, since the start. A job goes requested ->
	// translated -> (module) -> (pipeline) -> ready, or failed.
	struct Counters {
		uint64_t requested         = 0; // translation jobs posted
		uint64_t translated        = 0; // translations applied
		uint64_t translate_failed  = 0;
		uint64_t module_jobs       = 0; // module compile jobs posted
		uint64_t modules_ready     = 0;
		uint64_t module_failed     = 0;
		uint64_t speculative       = 0; // modules built with the translation, none posted
		uint64_t speculative_lost  = 0; // translation consumed, module unusable
		uint64_t pipeline_jobs     = 0;
		uint64_t pipelines_ready   = 0;
		uint64_t pipeline_failed   = 0;
		uint64_t expired           = 0; // translations dropped over the IR budget
		uint64_t retired           = 0; // waiting translations retired loudly (TTL/redundant/failed)
	};
	Counters counters;
	// Render thread only: retires a single waiting translation for `key` with a named reason and
	// frees its IR off the render thread.
	bool RetireWaiting(const ProgramKey& key, const char* reason);
	uint64_t last_store_report_ms = 0;
	uint64_t last_account_ms      = 0;
	// Owned by the PipelineCache: the builds and the skipped dispatches of the status period.
	BuildTiming::Totals*   build_totals = nullptr;
	FlowPlan::SkipLedger*  skipped_dispatches = nullptr;

	// Render thread only: when the game last drew into each render target, and in which of the
	// frames before that.
	struct TargetHistory {
		uint64_t frame = 0;
		// Bit n: drawn n frames before `frame`.
		uint8_t seen = 0;
	};
	std::unordered_map<uint64_t, TargetHistory> targets;
	// Records a draw into the target at `address` in the current frame. True when the game redraws
	// it every frame: it was drawn in at least three of the last six. Only then may a draw into it
	// be skipped while its shaders are built, because the next frame draws it again.
	bool NoteTargetDraw(uint64_t address);

	// Render thread only: what the render thread built itself since the last report, because a
	// draw or dispatch could not be skipped. These are the freezes async compile leaves.
	// Why the draw or dispatch could not be skipped.
	enum class StallReason : uint8_t { SmallTarget, NotRedrawn, Compute, WritesMemory, Count };
	struct StallCost {
		uint32_t shaders   = 0;
		uint32_t pipelines = 0;
		uint64_t us        = 0;
	};
	struct Stalls {
		std::array<StallCost, static_cast<size_t>(StallReason::Count)> by_reason {};
		uint64_t                                                       worst_us = 0;
	};
	Stalls      stalls;
	// Set by the code that decides whether a draw may be skipped, for the builds that follow.
	StallReason reason = StallReason::NotRedrawn;
	void        NoteStall(bool pipeline, std::chrono::steady_clock::time_point begin);

	// Render thread only: every two seconds while anything is pending, one line on the console
	// with what the workers are doing and how many draws were skipped meanwhile.
	void     ReportStatus();
	// The line `[store] written=.. pending=.. failed=..`.
	void     ReportStore(const ShaderStoreWriter::Counts& counts);
	uint64_t last_report_ms   = 0;
	uint64_t reported_skipped = 0;
	bool     was_busy         = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEASYNC_H_
