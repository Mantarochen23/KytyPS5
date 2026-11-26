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
		// When the translation finished, for the status line.
		uint64_t since = 0;
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

	// What a worker finished, to be applied by the render thread.
	void Finish(Common::UniqueFunction<void>&& apply);
	// Render thread only: moves what the workers finished into `backlog`.
	void TakeFinished();

	ShaderWorkers                             workers;
	std::mutex                                done_mutex;
	std::condition_variable                   done_cv;
	std::vector<Common::UniqueFunction<void>> done;
	// Render thread only: finished jobs taken from `done` that PumpAsync's time budget has not
	// applied yet.
	std::deque<Common::UniqueFunction<void>> backlog;
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
	// Render thread only: pipelines being created on workers.
	uint32_t pipelines_building = 0;

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
	uint64_t last_report_ms   = 0;
	uint64_t reported_skipped = 0;
	bool     was_busy         = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEASYNC_H_
