#include "graphics/host_gpu/renderer/pipeline/pipelineAsync.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/recoverable.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <xxhash.h>

// --async-shader-compile: stages and pipelines the cache cannot serve are made on worker threads,
// and the draws that need them are skipped until they are ready. Workers only run code that
// reads nothing but its own copy of the inputs. The cache itself, descriptor materialization and
// everything that reads guest memory stay on the render thread, which applies what the workers
// finished in PumpAsync.
//
// A stage takes two rounds. A worker translates the shader, which needs only the stage's static
// state. That translation waits until a draw names the resources it is compiled for, because
// reading them must happen in the draw. A second worker then emits and creates the module.
namespace Libs::Graphics {

PipelineCache::Async::~Async() {
	workers.Stop();
	uint32_t pending_done = static_cast<uint32_t>(backlog.size());
	backlog.clear();
	{
		std::scoped_lock lock(done_mutex);
		pending_done += static_cast<uint32_t>(done.size());
		done.clear();
	}
	if (pending_done != 0) {
		NoteShaderWorkerJobsFinished(pending_done);
	}
}

void PipelineCache::Async::Finish(Common::UniqueFunction<void>&& apply) {
	{
		std::scoped_lock lock(done_mutex);
		done.push_back(std::move(apply));
		NoteShaderWorkerJobsAdded(1);
	}
	done_cv.notify_all();
	std::scoped_lock notify_lock(notify_mutex);
	if (on_finish) {
		on_finish();
	}
}

uint64_t PipelineCache::FlightId(
    const ProgramKey& key, ShaderRecompiler::IR::PushData::Window push_data,
    const ShaderRecompiler::IR::ResourceSpecialization* specialization) {
	static const ShaderRecompiler::IR::ResourceSpecialization none {};
	ShaderRecompiler::IR::Writer                              writer;
	ShaderRecompiler::IR::IoAll(writer, key.stage, key.hash, key.user_data_count, key.static_state,
	                            push_data.begin, push_data.end, specialization != nullptr,
	                            specialization != nullptr ? *specialization : none);
	return XXH3_64bits(writer.bytes.data(), writer.bytes.size());
}

template <typename InputInfo>
void PipelineCache::RequestAsync(const ProgramKey& key, Program* program,
                                 const ShaderParams& params, const InputInfo& input_info,
                                 const ShaderRecompiler::CompileOptions& options,
                                 ShaderRecompiler::IR::PushData::Window push_data, AsyncMode mode,
                                 bool urgent_translate) {
	static_assert(std::is_same_v<InputInfo, ShaderVertexInputInfo> ||
	              std::is_same_v<InputInfo, ShaderPixelInputInfo> ||
	              std::is_same_v<InputInfo, ShaderComputeInputInfo>);
	auto*      async       = m_async.get();
	const auto translating = FlightId(key, {UINT32_MAX, UINT32_MAX}, nullptr);

	if (program != nullptr) {
		if (mode == AsyncMode::PrefetchOnly) {
			return;
		}
		// The program is known; a draw's resources decide which module it needs. If a translation
		// is waiting, compile it for them.
		if (const auto found = async->translations.find(key); found != async->translations.end()) {
			const auto compiling = FlightId(key, push_data, &program->specialization);
			if (const auto flying = async->in_flight.find(compiling);
			    flying != async->in_flight.end()) {
				// A draw wants it again: it is the scene on screen.
				flying->second.job->Touch();
				return;
			}
			auto translation = async->translations.extract(found);
			auto inputs      = std::move(translation.mapped().inputs);
			auto job = async->workers.Post([this, async, device = m_graphics.device, inputs,
			                                compiling, push_data,
			                                specialization = program->specialization,
			                                translated = std::move(translation.mapped().translated)]() mutable {
				std::optional<Common::RecoverableScope> scope;
				if (Config::KeepGoingEnabled()) {
					scope.emplace();
				}
				std::string fail_reason;
				const char* fail_file = "";
				int         fail_line = 0;
				bool        failed    = false;
				std::unique_ptr<Async::PendingStage> pending;
				try {
					auto compiled = CompileStage(device, inputs->key, inputs->options,
					                             std::move(*translated), specialization,
					                             push_data, {});
					pending = std::make_unique<Async::PendingStage>(device, std::move(compiled));
				} catch (const Common::RecoverableFailure& e) {
					failed      = true;
					fail_reason = e.what();
					fail_file   = e.file;
					fail_line   = e.line;
				} catch (const std::exception& e) {
					failed      = true;
					fail_reason = e.what();
				}
				async->Finish([this, inputs, compiling, push_data, failed,
				               fail_reason = std::move(fail_reason), fail_file, fail_line,
				               pending     = std::move(pending)]() mutable {
					m_async->in_flight.erase(compiling);
					if (failed) {
						if (Config::KeepGoingEnabled()) {
							RecordShaderFailure(inputs->key.stage, inputs->key.hash, inputs->code,
							                    fail_reason, fail_file, fail_line);
						} else {
							Log::WriteToConsoleAndLog(fmt::format(
							    "!!! SHADER DISPATCH SKIP (async compile failed) shader=0x{:016x}: {}\n",
							    inputs->key.hash, fail_reason));
						}
						m_program_cache.try_emplace(inputs->key).first->second.skip_dispatch = true;
						return;
					}
					auto& cached = m_program_cache.at(inputs->key);
					// Another draw's compilation may have made it first.
					if (cached.FindCandidate(pending->stage.module.specialization,
					                         push_data) != nullptr) {
						return;
					}
					pending->taken = true;
					AddModule(cached, inputs->key, std::move(pending->stage));
				});
			}, ShaderWorkers::Priority::Urgent);
			if (job) {
				async->in_flight.emplace(compiling, Async::Flight {std::move(job), false});
			}
			return;
		}
	}

	// Nothing to compile yet: translate the shader.
	if (const auto flying = async->in_flight.find(translating); flying != async->in_flight.end()) {
		if (urgent_translate) {
			async->workers.Promote(flying->second.job);
		} else {
			flying->second.job->Touch();
		}
		return;
	}
	auto inputs = std::make_shared<StageInputs>();
	inputs->key = key;
	inputs->info = input_info;
	auto& info   = std::get<InputInfo>(inputs->info);
	info.stage   = {};
	inputs->code.assign(params.code.begin(), params.code.end());
	inputs->back_code.assign(params.back_code.begin(), params.back_code.end());
	inputs->user_data         = params.user_data;
	inputs->profile           = SpirvProfile(m_graphics);
	inputs->options           = options;
	inputs->options.profile   = &inputs->profile;
	inputs->options.user_data = std::span(inputs->user_data).first(params.user_data_count);
	inputs->options.back_code = inputs->back_code;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		inputs->options.input_info.vertex = &info;
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		inputs->options.input_info.pixel = &info;
	} else {
		inputs->options.input_info.compute = &info;
	}

	auto job = async->workers.Post([this, async, inputs, translating] {
		std::optional<Common::RecoverableScope> scope;
		if (Config::KeepGoingEnabled()) {
			scope.emplace();
		}
		std::string fail_reason;
		const char* fail_file = "";
		int         fail_line = 0;
		bool        failed    = false;
		bool        sync_only = false;
		std::unique_ptr<ShaderRecompiler::TranslateResult> translated;
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan;
		try {
			if (inputs->key.stage == ShaderType::Compute) {
				// Decode before PrepareShaderSource so GPU-selected calls (s_swappc_b64) never
				// invoke PrepareCallTargets / Xbyak on a worker thread.
				ShaderRecompiler::Decoder::Program decoded;
				ShaderRecompiler::Decoder::DecodeProgram(inputs->code, decoded);
				if (decoded.has_swap_pc) {
					sync_only = true;
				} else {
					translated = std::make_unique<ShaderRecompiler::TranslateResult>(
					    ShaderRecompiler::TranslateProgram(decoded, inputs->options));
				}
			} else {
				translated = std::make_unique<ShaderRecompiler::TranslateResult>(
				    ShaderRecompiler::TranslateProgram(inputs->code, inputs->options));
			}
			// The resource plan reads only the translated program, which no other thread touches.
			// Its x86 walker is built with Xbyak, whose error state is per thread.
			if (translated != nullptr) {
				plan = std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
				    ShaderRecompiler::IR::ExtractResourcePlan(translated->program));
			}
		} catch (const Common::RecoverableFailure& e) {
			failed      = true;
			fail_reason = e.what();
			fail_file   = e.file;
			fail_line   = e.line;
		} catch (const std::exception& e) {
			failed      = true;
			fail_reason = e.what();
		}
		async->Finish([this, inputs, translating, failed, sync_only,
		               fail_reason = std::move(fail_reason), fail_file, fail_line,
		               translated  = std::move(translated),
		               plan        = std::move(plan)]() mutable {
			m_async->in_flight.erase(translating);
			if (sync_only) {
				m_async->sync_only_keys.insert(inputs->key);
				return;
			}
			if (failed) {
				if (Config::KeepGoingEnabled()) {
					RecordShaderFailure(inputs->key.stage, inputs->key.hash, inputs->code,
					                    fail_reason, fail_file, fail_line);
				} else {
					Log::WriteToConsoleAndLog(fmt::format(
					    "!!! SHADER DISPATCH SKIP (resource tracking failed) shader=0x{:016x}: {}\n",
					    inputs->key.hash, fail_reason));
				}
				m_program_cache.try_emplace(inputs->key).first->second.skip_dispatch = true;
				return;
			}
			if (!m_program_cache.contains(inputs->key)) {
				m_program_cache.emplace(inputs->key, Program {.resource_plan = std::move(*plan)});
			}
			m_async->translations.insert_or_assign(
			    inputs->key, Async::Translation {inputs, std::move(translated), AskClockMs()});
		});
	}, urgent_translate ? ShaderWorkers::Priority::Urgent : ShaderWorkers::Priority::Normal);
	if (job) {
		async->in_flight.emplace(translating, Async::Flight {std::move(job), true});
	}
}

template void PipelineCache::RequestAsync<ShaderVertexInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderVertexInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode,
    bool);
template void PipelineCache::RequestAsync<ShaderPixelInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderPixelInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode,
    bool);
template void PipelineCache::RequestAsync<ShaderComputeInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderComputeInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode,
    bool);

void PipelineCache::SetAsyncFinishCallback(Common::UniqueFunction<void>&& callback) {
	if (m_async == nullptr) {
		return;
	}
	std::scoped_lock lock(m_async->notify_mutex);
	m_async->on_finish = std::move(callback);
}

// Moves what the workers finished into the render thread's backlog.
void PipelineCache::Async::TakeFinished() {
	std::scoped_lock lock(done_mutex);
	for (auto& apply: done) {
		backlog.push_back(std::move(apply));
	}
	done.clear();
}

// The jobs applied here are the render-thread-bound steps only: the cache, the module and pipeline
// hand-off. Everything else ran on the workers. The backlog is applied within the time budget
// (--async-pump-budget-ms, at least one job per call), so a burst of finished jobs is spread over
// several calls instead of holding up the draw that called.
void PipelineCache::PumpAsync() {
	if (m_async == nullptr) {
		return;
	}
	const auto begin = std::chrono::steady_clock::now();
	m_async->TakeFinished();
	const auto now_us = [] {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                 std::chrono::steady_clock::now().time_since_epoch())
		                                 .count());
	};
	const auto applied = ApplyWithinBudget(
	    m_async->backlog, static_cast<uint64_t>(Config::AsyncPumpBudgetMs()) * 1000u, now_us);
	if (applied != 0) {
		NoteShaderWorkerJobsFinished(static_cast<uint32_t>(applied));
		m_async->pump_applied += static_cast<uint32_t>(applied);
	}
	const auto spent = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                             std::chrono::steady_clock::now() - begin)
	                                             .count());
	m_async->pump_us += spent;
	m_async->pump_worst_us = std::max(m_async->pump_worst_us, spent);
	m_async->ReportStatus();
}

void PipelineCache::WaitForAsync() {
	if (m_async == nullptr) {
		return;
	}
	if (m_async->backlog.empty()) {
		std::unique_lock lock(m_async->done_mutex);
		m_async->done_cv.wait(lock, [this] {
			return !m_async->done.empty() || m_async->workers.Stopped();
		});
	}
	m_async->TakeFinished();
	// The caller is blocked until something it waits for is ready: apply everything.
	const auto applied = ApplyWithinBudget(m_async->backlog, 0, [] { return uint64_t {0}; });
	if (applied != 0) {
		NoteShaderWorkerJobsFinished(static_cast<uint32_t>(applied));
	}
	m_async->ReportStatus();
}

bool PipelineCache::Async::NoteTargetDraw(uint64_t address) {
	const auto frame = CurrentFrame();
	auto&      seen  = targets[address];
	if (seen.frame != frame) {
		const auto age = frame - seen.frame;
		seen.seen      = age >= 8 ? 0 : static_cast<uint8_t>(seen.seen << age);
		seen.frame     = frame;
	}
	seen.seen |= 1u;
	if (targets.size() > 4096) {
		// Targets of scenes long gone.
		std::erase_if(targets,
		              [frame](const auto& entry) { return frame - entry.second.frame > 600; });
	}
	return std::popcount(static_cast<unsigned>(seen.seen & 0x3Fu)) >= 3;
}

// Whether the module writes buffers, images or memory through addresses, which other draws and
// dispatches may read: a draw with such a module is never skipped, because the writes are lost.
static bool WritesMemory(const ShaderModule& module) {
	if (module.info.has_address_writes) {
		return true;
	}
	const auto& info = module.info.info;
	return std::ranges::any_of(info.buffers, [](const auto& buffer) { return buffer.written; }) ||
	       std::ranges::any_of(info.images, [](const auto& image) { return image.written; });
}

void PipelineCache::Async::NoteStall(bool pipeline, std::chrono::steady_clock::time_point begin) {
	const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                          std::chrono::steady_clock::now() - begin)
	                                          .count());
	auto& cost = stalls.by_reason[static_cast<size_t>(reason)];
	(pipeline ? cost.pipelines : cost.shaders)++;
	cost.us += us;
	stalls.worst_us = std::max(stalls.worst_us, us);
}

void PipelineCache::Async::ReportStatus() {
	const auto now = AskClockMs();
	if (now - last_report_ms < 2000) {
		return;
	}
	const auto last_status_ms = last_report_ms;
	last_report_ms = now;
	// A translation no draw has come back for in a minute belongs to something no longer on
	// screen; a later draw translates it again.
	std::erase_if(translations,
	              [now](const auto& entry) { return now - entry.second.since > 60000; });
	size_t translating = 0;
	for (const auto& [id, flight]: in_flight) {
		translating += flight.translating ? 1u : 0u;
	}
	const size_t compiling = in_flight.size() - translating;
	uint64_t     oldest    = 0;
	for (const auto& [key, translation]: translations) {
		oldest = std::max(oldest, now - translation.since);
	}
	const auto frame   = CurrentFrame();
	size_t     redrawn = 0;
	for (const auto& [address, history]: targets) {
		// Drawn in the last six frames, and in at least three of them.
		redrawn += frame - history.frame < 6 &&
		                   std::popcount(static_cast<unsigned>(history.seen & 0x3Fu)) >= 3
		               ? 1u
		               : 0u;
	}
	const auto skipped_total = TotalDrawsSkippedForShaders();
	const auto skipped       = skipped_total - reported_skipped;
	reported_skipped         = skipped_total;
	const bool busy = !in_flight.empty() || !translations.empty() || pipelines_building != 0 ||
	                  skipped != 0 || !backlog.empty();
	if (busy) {
		const auto   finished = workers.FinishedJobs();
		const double window_s = static_cast<double>(std::max<uint64_t>(now - last_status_ms, 1)) / 1000.0;
		std::printf("Async: %zu translating, %zu compiling, %zu waiting (oldest %.0f s), %u pipelines, "
		            "%zu queued, %" PRIu64 " skipped/2s, %zu/%zu redrawn; workers %u/%u busy, "
		            "%.1f jobs/s, pump %.1f ms (worst %.1f ms, %u applied, %zu backlog)\n",
		            translating, compiling, translations.size(),
		            static_cast<double>(oldest) / 1000.0, pipelines_building, workers.Queued(),
		            skipped, redrawn, targets.size(), workers.Busy(), workers.ThreadCount(),
		            static_cast<double>(finished - reported_finished) / window_s,
		            static_cast<double>(pump_us) / 1000.0,
		            static_cast<double>(pump_worst_us) / 1000.0, pump_applied, backlog.size());
		reported_finished = finished;
	} else if (was_busy) {
		std::printf("Async: idle\n");
	}
	was_busy            = busy;
	pump_us             = 0;
	pump_worst_us       = 0;
	pump_applied        = 0;
	reported_finished   = workers.FinishedJobs();
	uint64_t stalled_us = 0;
	for (const auto& cost: stalls.by_reason) {
		stalled_us += cost.us;
	}
	if (stalled_us >= 20000) {
		static constexpr const char* names[] = {"small targets", "targets not yet redrawn",
		                                        "compute", "writes memory"};
		std::string text;
		for (size_t i = 0; i < stalls.by_reason.size(); i++) {
			const auto& cost = stalls.by_reason[i];
			if (cost.shaders + cost.pipelines == 0) {
				continue;
			}
			char part[160];
			std::snprintf(part, sizeof(part), "%s%s %.0f ms (%u shaders, %u pipelines)",
			              text.empty() ? "" : "; ", names[i], static_cast<double>(cost.us) / 1000.0,
			              cost.shaders, cost.pipelines);
			text += part;
		}
		std::printf("Async STALL: the render thread built %.0f ms in 2 s (worst %.0f ms): %s\n",
		            static_cast<double>(stalled_us) / 1000.0,
		            static_cast<double>(stalls.worst_us) / 1000.0, text.c_str());
	}
	stalls = {};
}

bool PipelineCache::EnsureGraphicsPipeline(const CommandBuffer&   command,
                                           const DrawRenderState& state,
                                           const DrawTargets&     targets) {
	// A draw that must not be skipped gets its pipeline from GetGraphicsPipeline, on this thread.
	if (m_async == nullptr || !state.deferrable) {
		return true;
	}
	std::array<const ShaderModule*, 4> stages {};
	const auto key = MakeGraphicsPipelineKey(command, state, targets, stages);
	if (std::ranges::any_of(stages, [](const ShaderModule* stage) {
		    return stage != nullptr && WritesMemory(*stage);
	    })) {
		m_async->reason = Async::StallReason::WritesMemory;
		return true;
	}
	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
	auto& pipeline        = iter->second;
	if (inserted) {
		pipeline.pending = true;
		m_async->pipelines_building++;
		// The map's nodes do not move, and the stages' modules sit in deques that only grow.
		m_async->workers.Post([this, async = m_async.get(), &graphics = m_graphics,
		                       driver_cache = m_driver_cache, &pipeline, key, stages] {
			std::optional<Common::RecoverableScope> scope;
			if (Config::KeepGoingEnabled()) {
				scope.emplace();
			}
			bool        failed = false;
			std::string fail_reason;
			try {
				Common::DebugCounters::ScopedTimer timer(
				    Common::DebugCounters::Counter::PipelinesCreated,
				    Common::DebugCounters::Counter::PipelineCreateNs,
				    Common::DebugCounters::Peak::PipelineCreateNs);
				CreatePipelineInternal(graphics, pipeline, key, stages, driver_cache);
			} catch (const std::exception& e) {
				failed      = true;
				fail_reason = e.what();
			}
			async->Finish([this, &pipeline, key, failed,
			               fail_reason = std::move(fail_reason)]() mutable {
				m_async->pipelines_building--;
				pipeline.pending = false;
				if (failed) {
					pipeline.failed = true;
					Log::WriteToConsoleAndLog(fmt::format(
					    "!!! GRAPHICS PIPELINE SKIP (async create failed): {}\n", fail_reason));
					return;
				}
				StorePipeline(key);
				CheckpointDriverCache();
			});
		}, ShaderWorkers::Priority::Urgent);
	}
	if (pipeline.failed) {
		return false;
	}
	if (pipeline.pending) {
		NoteDrawSkippedForShaders();
		return false;
	}
	return true;
}

bool PipelineCache::EnsureComputePipeline(const ShaderComputeInputInfo& input_info, bool& pending) {
	pending = false;
	if (m_async == nullptr) {
		return true;
	}
	const auto& module        = *input_info.stage.module;
	auto [iter, inserted]     = m_compute_pipelines.try_emplace(module.id);
	auto& pipeline            = iter->second;
	if (inserted) {
		pipeline.pending = true;
		m_async->pipelines_building++;
		// Map nodes stay stable and program module deques only grow while workers run.
		m_async->workers.Post([this, async = m_async.get(), &graphics = m_graphics,
		                       driver_cache = m_driver_cache, &pipeline, &module] {
			std::optional<Common::RecoverableScope> scope;
			if (Config::KeepGoingEnabled()) {
				scope.emplace();
			}
			bool        failed = false;
			std::string fail_reason;
			try {
				Common::DebugCounters::ScopedTimer timer(
				    Common::DebugCounters::Counter::PipelinesCreated,
				    Common::DebugCounters::Counter::PipelineCreateNs,
				    Common::DebugCounters::Peak::PipelineCreateNs);
				CreatePipelineInternal(graphics, pipeline, module, driver_cache);
			} catch (const std::exception& e) {
				failed      = true;
				fail_reason = e.what();
			}
			async->Finish([this, &pipeline, hash = module.info.shader_hash, failed,
			               fail_reason = std::move(fail_reason)]() mutable {
				m_async->pipelines_building--;
				pipeline.pending = false;
				if (failed) {
					pipeline.failed = true;
					Log::WriteToConsoleAndLog(fmt::format(
					    "!!! COMPUTE PIPELINE SKIP (async create failed) shader=0x{:016x}: {}\n",
					    hash, fail_reason));
					return;
				}
				CheckpointDriverCache();
			});
		}, ShaderWorkers::Priority::Urgent);
	}
	if (pipeline.failed) {
		return false;
	}
	if (pipeline.pending) {
		pending = true;
		return false;
	}
	return true;
}

} // namespace Libs::Graphics
