#include "graphics/host_gpu/renderer/pipeline/pipelineAsync.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/recoverable.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"

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

ShaderWorkers::ShaderWorkers(uint32_t threads) {
	for (uint32_t i = 0; i < threads; i++) {
		m_threads.emplace_back([this](std::stop_token stop) {
			std::unique_lock lock(m_mutex);
			// Once stop is requested the wait returns false only when the queue is empty.
			while (m_ready.wait(lock, stop,
			                    [this] { return !m_urgent_jobs.empty() || !m_jobs.empty(); })) {
				auto& queue    = m_urgent_jobs.empty() ? m_jobs : m_urgent_jobs;
				const auto job = std::move(queue.front());
				queue.pop();
				lock.unlock();
				job();
				lock.lock();
			}
		});
	}
}

void ShaderWorkers::Post(Common::UniqueFunction<void>&& job, Priority priority) {
	{
		std::scoped_lock lock(m_mutex);
		if (m_stopped) {
			return;
		}
		(priority == Priority::Urgent ? m_urgent_jobs : m_jobs).push(std::move(job));
	}
	m_ready.notify_one();
}

void ShaderWorkers::Stop() {
	{
		std::scoped_lock lock(m_mutex);
		m_stopped     = true;
		m_jobs        = {};
		m_urgent_jobs = {};
	}
	for (auto& thread: m_threads) {
		thread.request_stop();
	}
	// A running job finishes first.
	m_threads.clear();
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
                                 ShaderRecompiler::IR::PushData::Window push_data, AsyncMode mode) {
	static_assert(std::is_same_v<InputInfo, ShaderVertexInputInfo> ||
	              std::is_same_v<InputInfo, ShaderPixelInputInfo>);
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
			if (!async->in_flight.insert(compiling).second) {
				return;
			}
			auto translation = async->translations.extract(found);
			auto inputs      = std::move(translation.mapped().inputs);
			async->workers.Post([this, async, device = m_graphics.device, inputs, compiling,
			                     push_data, specialization = program->specialization,
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
			return;
		}
	}

	// Nothing to compile yet: translate the shader.
	if (!async->in_flight.insert(translating).second) {
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
	} else {
		inputs->options.input_info.pixel = &info;
	}

	async->workers.Post([this, async, inputs, translating] {
		std::optional<Common::RecoverableScope> scope;
		if (Config::KeepGoingEnabled()) {
			scope.emplace();
		}
		std::string fail_reason;
		const char* fail_file = "";
		int         fail_line = 0;
		bool        failed    = false;
		std::unique_ptr<ShaderRecompiler::TranslateResult> translated;
		try {
			translated = std::make_unique<ShaderRecompiler::TranslateResult>(
			    ShaderRecompiler::TranslateProgram(inputs->code, inputs->options));
		} catch (const Common::RecoverableFailure& e) {
			failed      = true;
			fail_reason = e.what();
			fail_file   = e.file;
			fail_line   = e.line;
		} catch (const std::exception& e) {
			failed      = true;
			fail_reason = e.what();
		}
		async->Finish([this, inputs, translating, failed,
		               fail_reason = std::move(fail_reason), fail_file, fail_line,
		               translated  = std::move(translated)]() mutable {
			m_async->in_flight.erase(translating);
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
			// ExtractResourcePlan builds the Xbyak SRT walker on the render thread so Xbyak's
			// global error state is never raced across worker threads.
			if (!m_program_cache.contains(inputs->key)) {
				auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated->program);
				m_program_cache.emplace(inputs->key, Program {.resource_plan = std::move(plan)});
			}
			m_async->translations.insert_or_assign(
			    inputs->key, Async::Translation {inputs, std::move(translated)});
		});
	});
}

template void PipelineCache::RequestAsync<ShaderVertexInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderVertexInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode);
template void PipelineCache::RequestAsync<ShaderPixelInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderPixelInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode);

void PipelineCache::PumpAsync() {
	if (m_async == nullptr) {
		return;
	}
	std::vector<Common::UniqueFunction<void>> done;
	{
		std::scoped_lock lock(m_async->done_mutex);
		done.swap(m_async->done);
	}
	for (auto& apply: done) {
		apply();
	}
}

bool PipelineCache::EnsureGraphicsPipeline(const CommandBuffer&   command,
                                           const DrawRenderState& state,
                                           const DrawTargets&     targets) {
	// A draw that must not be skipped gets its pipeline from GetGraphicsPipeline, on this thread.
	if (m_async == nullptr || !state.deferrable) {
		return true;
	}
	std::array<const ShaderModule*, 4> stages {};
	const auto key        = MakeGraphicsPipelineKey(command, state, targets, stages);
	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
	auto& pipeline        = iter->second;
	if (inserted) {
		pipeline.pending = true;
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

} // namespace Libs::Graphics
