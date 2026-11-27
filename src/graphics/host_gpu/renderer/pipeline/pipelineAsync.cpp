#include "graphics/host_gpu/renderer/pipeline/pipelineAsync.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/recoverable.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/asyncPlan.h"
#include "graphics/host_gpu/renderer/pipeline/renderActivity.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <fmt/format.h>
#include <optional>
#include <ranges>
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

namespace {
template <typename... Args>
void AsyncLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}
} // namespace

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

void PipelineCache::Async::Finish(Common::UniqueFunction<void>&& apply, uint64_t flight) {
	{
		std::scoped_lock lock(done_mutex);
		done.push_back(FinishedJob {flight, std::move(apply)});
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

// What the translated IR of a stage holds in memory: its instructions, in the program's value
// storage and in its blocks, at the size of a list node.
static uint64_t EstimateIrBytes(const ShaderRecompiler::TranslateResult& translated) {
	const auto& program = translated.program;
	uint64_t    count   = program.value_storage.size();
	for (const auto& block: program.block_storage) {
		count += block->Instructions().size();
	}
	return count * (sizeof(ShaderRecompiler::IR::Inst) + 2 * sizeof(void*)) +
	       translated.decoded_dump.size() + translated.cfg_dump.size();
}

template <typename InputInfo>
std::shared_ptr<PipelineCache::StageInputs>
PipelineCache::MakeStageInputs(const ProgramKey& key, const ShaderParams& params,
                               const InputInfo& input_info,
                               const ShaderRecompiler::CompileOptions& options) {
	auto inputs  = std::make_shared<StageInputs>();
	inputs->key  = key;
	inputs->info = input_info;
	auto& info   = std::get<InputInfo>(inputs->info);
	info.stage   = {};
	inputs->code.assign(params.code.begin(), params.code.end());
	inputs->back_code.assign(params.back_code.begin(), params.back_code.end());
	inputs->user_data         = params.user_data;
	inputs->shader_base       = params.Base();
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
	return inputs;
}

template <typename InputInfo>
void PipelineCache::RequestAsync(const ProgramKey& key, Program* program,
                                 const ShaderParams& params, const InputInfo& input_info,
                                 const ShaderRecompiler::CompileOptions& options,
                                 ShaderRecompiler::IR::PushData::Window push_data, AsyncMode mode,
                                 bool urgent_translate, bool user_data_known) {
	static_assert(std::is_same_v<InputInfo, ShaderVertexInputInfo> ||
	              std::is_same_v<InputInfo, ShaderPixelInputInfo> ||
	              std::is_same_v<InputInfo, ShaderComputeInputInfo>);
	auto*      async       = m_async.get();
	const auto translating = FlightId(key, {UINT32_MAX, UINT32_MAX}, nullptr);

	const bool has_program = program != nullptr;
	const auto found       = has_program ? async->translations.find(key) : async->translations.end();
	const auto compiling =
	    has_program ? FlightId(key, push_data, &program->specialization) : uint64_t {0};
	const auto compiling_flight = has_program ? async->in_flight.find(compiling) : async->in_flight.end();
	const auto action           = AsyncPlan::Decide({
	              .has_program           = has_program,
	              .prefetch_only         = mode == AsyncMode::PrefetchOnly,
	              .translation_ready     = found != async->translations.end(),
	              .module_job_in_flight  = compiling_flight != async->in_flight.end(),
	              .translation_in_flight = async->in_flight.contains(translating),
	          });
	if (mode != AsyncMode::PrefetchOnly) {
		// What the stall reporter says the render thread waits for.
		RenderActivity::NoteDetail(AsyncPlan::Name(action));
	}
	if (action == AsyncPlan::Action::Nothing) {
		return;
	}
	if (action == AsyncPlan::Action::JoinModuleJob) {
		// A draw wants it again: it is the scene on screen.
		async->Want(compiling, compiling_flight->second.job);
		return;
	}
	if (action == AsyncPlan::Action::PostModuleJob) {
		{
			// A translation is waiting: compile it for the draw's resources.
			found->second.asked = AskClockMs();
			auto translation = async->translations.extract(found);
			auto inputs      = std::move(translation.mapped().inputs);
			async->counters.module_jobs++;
			const char* reason = std::is_same_v<InputInfo, ShaderComputeInputInfo> ? "dispatch" : "draw";
			auto job = async->workers.Post([this, async, device = m_graphics.device, inputs,
			compiling, push_data, reason, post_time = std::chrono::steady_clock::now(),
			translate_us = translation.mapped().translate_us,
			plan_us = translation.mapped().plan_us,
			specialization = program->specialization,
			translated = std::move(translation.mapped().translated)]() mutable {
			const auto queue_wait_us = BuildTiming::ElapsedUs(post_time);
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
					                             compiled.steps.translate_us = translate_us;
					                             compiled.steps.plan_us = plan_us;
					                             compiled.steps.queue_wait_us = queue_wait_us;
					                             compiled.steps.total_us = BuildTiming::ElapsedUs(post_time);
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
				async->Finish([this, inputs, compiling, push_data, failed, reason,
				fail_reason = std::move(fail_reason), fail_file, fail_line,
				pending = std::move(pending)]() mutable {
					m_async->Settle(compiling);
					if (failed) {
						m_async->counters.module_failed++;
						if (Config::KeepGoingEnabled()) {
							RecordShaderFailure(inputs->key.stage, inputs->key.hash, inputs->code,
							                    fail_reason, fail_file, fail_line);
						} else {
							Log::WriteToConsoleAndLog(fmt::format(
							    "!!! SHADER DISPATCH SKIP (async compile failed) shader=0x{:016x}: {}\n",
							    inputs->key.hash, fail_reason));
						}
						auto& failed_program = m_program_cache.try_emplace(inputs->key).first->second;
						failed_program.skip_dispatch  = true;
						failed_program.logged_neg_hit = true;
						failed_program.failure_reason = fail_reason;
						AsyncLog(
						    "ShaderRecompiler {} translation end: hash=0x{:016x} outcome=failed ({})",
						    ShaderRecompiler::StageName(inputs->key.stage), inputs->key.hash,
						    fail_reason);
						StoreFailedProgram(inputs->key, fail_reason);
						return;
					}
					auto& cached = m_program_cache.at(inputs->key);
					// Another draw's compilation may have made it first.
					if (cached.FindCandidate(pending->stage.module.specialization,
					                         push_data) != nullptr) {
						return;
					}
					pending->taken = true;
					NoteBuild({.stage = ShaderRecompiler::StageName(inputs->key.stage),
					.hash = inputs->key.hash,
					.words = pending->stage.spirv.size(),
					.steps = pending->stage.steps,
					.where = "worker",
					.reason = reason});
					AddModule(cached, inputs->key, std::move(pending->stage));
					m_async->counters.modules_ready++;
					AsyncLog(
					"ShaderRecompiler {} translation end: hash=0x{:016x} outcome=emitted (async)",
					    ShaderRecompiler::StageName(inputs->key.stage), inputs->key.hash);
				}, compiling);
			}, ShaderWorkers::Priority::Urgent, /*waited=*/true);
			if (job) {
				async->wanted.insert(compiling);
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
		if (mode != AsyncMode::PrefetchOnly) {
			// A draw or dispatch is blocked on it now.
			async->Want(translating, flying->second.job);
		}
		return;
	}
	auto inputs = MakeStageInputs(key, params, input_info, options);

	async->counters.requested++;
	const bool speculate = AsyncPlan::MaySpeculate(
	    Config::AsyncSpeculativeModulesMode(), std::is_same_v<InputInfo, ShaderComputeInputInfo>,
	    has_program, user_data_known);
	// Whose user data the module is built from: the scan of the command stream's (lookahead), or
	// the dispatch's own.
	const bool        from_scan = speculate && mode == AsyncMode::PrefetchOnly && user_data_known;
	const char* const reason    = mode == AsyncMode::PrefetchOnly ? "prefetch"
	                              : std::is_same_v<InputInfo, ShaderComputeInputInfo> ? "dispatch"
	                                                                                  : "draw";
	auto job = async->workers.Post([this, async, inputs, translating, speculate, from_scan, reason,
	                                post_time = std::chrono::steady_clock::now(),
	                                device = m_graphics.device] {
		const auto queue_wait_us = BuildTiming::ElapsedUs(post_time);
		uint64_t   translate_us = 0;
		uint64_t   plan_us      = 0;
		std::optional<Common::RecoverableScope> scope;
		if (Config::KeepGoingEnabled()) {
			scope.emplace();
		}
		std::string fail_reason;
		const char* fail_file = "";
		int         fail_line = 0;
		bool        failed    = false;
		bool        sync_only = false;
		bool        speculated = false;
		bool        lost       = false;
		std::unique_ptr<ShaderRecompiler::TranslateResult> translated;
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan;
		std::unique_ptr<Async::PendingStage> pending;
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
		if (speculate && !failed && translated != nullptr && plan != nullptr) {
			// A compute shader whose resources follow from its user data alone has one module
			// whatever the dispatch is; build it now, so that it is stored at once and the
			// dispatch finds it. Anything that needs guest memory, the workgroup counts or the
			// image heap waits for its dispatch as before. The translation is consumed.
			try {
				const bool static_plan =
				    plan->resource_tracking_complete && !plan->capture_specialization_reads &&
				    plan->source_reads.empty() &&
				    std::ranges::none_of(plan->descriptor_sources, [](const auto& source) {
					    return source.indirect_descriptor &&
					           source.indirect_descriptor->workgroup_axis != UINT32_MAX;
				    });
				ShaderRecompiler::IR::ResourceSnapshot       snapshot;
				ShaderRecompiler::IR::ResourceSpecialization specialization;
				ShaderRecompiler::IR::SrtRuntime             runtime {
				    .user_data   = inputs->options.user_data,
				    .shader_base = inputs->shader_base,
				};
				if (static_plan &&
				    ShaderRecompiler::IR::MaterializeResources(*plan, runtime, snapshot,
				                                               specialization) &&
				    snapshot.specialization_reads.empty()) {
					auto compiled = CompileStage(device, inputs->key, inputs->options,
					                             std::move(*translated), specialization, {}, {});
					translated.reset();
					if (compiled.module.info.info.uses_image_heap) {
						// Its images come from the heap table, which only a dispatch can read.
						if (compiled.module.module) {
							device.destroyShaderModule(compiled.module.module, nullptr);
						}
						lost = true;
					} else {
						pending = std::make_unique<Async::PendingStage>(device, std::move(compiled));
						speculated = true;
					}
				}
			} catch (const Common::RecoverableFailure&) {
				// The dispatch tries again, with its own inputs.
			} catch (const std::exception&) {
			}
		}
		async->Finish([this, inputs, translating, failed, sync_only, speculated, lost, from_scan,
		               reason, translate_us, plan_us,
		               fail_reason = std::move(fail_reason), fail_file, fail_line,
		               translated  = std::move(translated),
		               plan        = std::move(plan),
		               pending     = std::move(pending)]() mutable {
			m_async->Settle(translating);
			if (sync_only) {
				m_async->sync_only_keys.insert(inputs->key);
				return;
			}
			if (failed) {
				m_async->counters.translate_failed++;
				if (Config::KeepGoingEnabled()) {
					RecordShaderFailure(inputs->key.stage, inputs->key.hash, inputs->code,
					                    fail_reason, fail_file, fail_line);
				} else {
					Log::WriteToConsoleAndLog(fmt::format(
					    "!!! SHADER DISPATCH SKIP (resource tracking failed) shader=0x{:016x}: {}\n",
					    inputs->key.hash, fail_reason));
				}
				auto& program          = m_program_cache.try_emplace(inputs->key).first->second;
				program.skip_dispatch  = true;
				program.logged_neg_hit = true;
				program.failure_reason = fail_reason;
				AsyncLog("ShaderRecompiler {} translation end: hash=0x{:016x} outcome=failed ({})",
				         ShaderRecompiler::StageName(inputs->key.stage), inputs->key.hash,
				         fail_reason);
				StoreFailedProgram(inputs->key, fail_reason);
				return;
			}
			m_async->counters.translated++;
			if (!m_program_cache.contains(inputs->key)) {
				m_program_cache.emplace(inputs->key, Program {.resource_plan = std::move(*plan)});
			}
			if (lost) {
				// The dispatch translates it again with its own resources.
				m_async->counters.speculative_lost++;
				return;
			}
			if (speculated) {
				auto& cached = m_program_cache.at(inputs->key);
				if (cached.FindCandidate(pending->stage.module.specialization, {}) == nullptr) {
					pending->taken = true;
					NoteBuild({.stage  = ShaderRecompiler::StageName(inputs->key.stage),
					           .hash   = inputs->key.hash,
					           .words  = pending->stage.spirv.size(),
					           .steps  = pending->stage.steps,
					           .where  = "worker",
					           .reason = reason});
					const auto& added = AddModule(cached, inputs->key, std::move(pending->stage));
					if (from_scan) {
						m_lookahead_modules.insert(added.id);
						m_prefetch.lookahead_modules_built++;
					}
				}
				m_async->counters.speculative++;
				m_async->counters.modules_ready++;
				AsyncLog("ShaderRecompiler {} translation end: hash=0x{:016x} outcome=emitted (async)",
				         ShaderRecompiler::StageName(inputs->key.stage), inputs->key.hash);
				return;
			}
			const auto bytes = EstimateIrBytes(*translated);
			m_async->translations.insert_or_assign(
			    inputs->key, Async::Translation {inputs, std::move(translated), AskClockMs(),
			                                     AskClockMs(), bytes, translate_us, plan_us});
		}, translating);
	}, urgent_translate ? ShaderWorkers::Priority::Urgent : ShaderWorkers::Priority::Normal,
	   mode == AsyncMode::On);
	if (job) {
		if (mode == AsyncMode::On) {
			async->wanted.insert(translating);
		}
		async->in_flight.emplace(translating, Async::Flight {std::move(job), true});
	}
}

template void PipelineCache::RequestAsync<ShaderVertexInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderVertexInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode,
    bool, bool);
template void PipelineCache::RequestAsync<ShaderPixelInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderPixelInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode,
    bool, bool);
template void PipelineCache::RequestAsync<ShaderComputeInputInfo>(
    const ProgramKey&, Program*, const ShaderParams&, const ShaderComputeInputInfo&,
    const ShaderRecompiler::CompileOptions&, ShaderRecompiler::IR::PushData::Window, AsyncMode,
    bool, bool);

// A compute program with GPU-selected calls is translated and linked on the render thread (it
// reads the callees from guest memory), but compiling the module from that translation takes the
// longest, and needs nothing but copies: it runs on a worker and the dispatch is suspended.
void PipelineCache::NoteDispatchSkipped(uint64_t waited_ms) {
	// The dispatch is not retried: its request is over.
	(void)m_requests.Begin(m_pending_compute_identity);
	const auto hash = m_pending_compute_hash;
	const auto use  = m_skipped_dispatches.Note(hash, waited_ms);
	if (use == 1) {
		std::printf("[skip] cs hash=0x%016" PRIx64 " waited=%" PRIu64 "ms use=#1\n", hash,
		            waited_ms);
	}
}

void PipelineCache::ReportPrefetch() const {
const auto& p = m_prefetch;
const auto registrations = ShaderComputePrefetchQueueStats();
std::printf("[prefetch] registrations: compute %" PRIu64 " (nonzero code offset %" PRIu64
") queued %" PRIu64 " dropped-queue-full %" PRIu64 "\n",
registrations.registered, registrations.nonzero_offset, registrations.queued,
registrations.queue_full);
	std::printf("[prefetch] scan/registration calls=%" PRIu64 " queued=%" PRIu64
	            " (with user data %" PRIu64 ") not-registered=%" PRIu64 " known=%" PRIu64
	            " in-flight=%" PRIu64 "; lookahead modules built=%" PRIu64 " used=%" PRIu64
	            " unused=%" PRIu64 "; served from store: %" PRIu64 " hashes, %" PRIu64
	            " uses; skipped dispatches: %" PRIu64 " hashes, %" PRIu64 " uses, %" PRIu64
	            " ms waited\n",
	            p.requested, p.queued, p.with_data, p.unmapped, p.known, p.in_flight,
	            p.lookahead_modules_built, p.lookahead_hits,
	            p.lookahead_modules_built - std::min(p.lookahead_hits, p.lookahead_modules_built),
	            m_served.Hashes(), m_served.Uses(), m_skipped_dispatches.Hashes(),
	            m_skipped_dispatches.Total(), m_skipped_dispatches.WaitedMs());
}

bool PipelineCache::PostCallProgramModule(const ProgramKey& key, Program& program,
                                          const ShaderParams&                     params,
                                          const ShaderComputeInputInfo&           input_info,
                                          const ShaderRecompiler::CompileOptions& options,
                                          ShaderRecompiler::IR::PushData::Window  push_data,
                                          ShaderRecompiler::TranslateResult&&     translated) {
	auto* async           = m_async.get();
	const auto compiling  = FlightId(key, push_data, &program.specialization);
	auto       inputs     = MakeStageInputs(key, params, input_info, options);
	async->counters.module_jobs++;
	auto job = async->workers.Post(
	    [this, async, device = m_graphics.device, inputs, compiling, push_data,
	     post_time = std::chrono::steady_clock::now(),
	     specialization = program.specialization,
	     function_code  = std::vector<uint32_t>(program.FunctionCode().begin(),
	                                            program.FunctionCode().end()),
	     translated = std::make_unique<ShaderRecompiler::TranslateResult>(std::move(translated))]() mutable {
		    const auto queue_wait_us = BuildTiming::ElapsedUs(post_time);
		    std::optional<Common::RecoverableScope> scope;
		    if (Config::KeepGoingEnabled()) {
			    scope.emplace();
		    }
		    std::string                          fail_reason;
		    bool                                 failed = false;
		    std::unique_ptr<Async::PendingStage> pending;
		    try {
			    auto compiled = CompileStage(device, inputs->key, inputs->options,
			                                 std::move(*translated), specialization, push_data,
			                                 function_code);
			    compiled.steps.queue_wait_us = queue_wait_us;
			    compiled.steps.total_us      = BuildTiming::ElapsedUs(post_time);
			    pending = std::make_unique<Async::PendingStage>(device, std::move(compiled));
		    } catch (const std::exception& e) {
			    failed      = true;
			    fail_reason = e.what();
		    }
		    async->Finish([this, inputs, compiling, push_data, failed,
		                   fail_reason = std::move(fail_reason),
		                   pending     = std::move(pending)]() mutable {
			    m_async->Settle(compiling);
			    auto& cached = m_program_cache.at(inputs->key);
			    if (failed) {
				    m_async->counters.module_failed++;
				    cached.skip_dispatch  = true;
				    cached.logged_neg_hit = true;
				    cached.failure_reason = fail_reason;
				    Log::WriteToConsoleAndLog(fmt::format(
				        "!!! SHADER DISPATCH SKIP (async compile failed) shader=0x{:016x}: {}\n",
				        inputs->key.hash, fail_reason));
				    return;
			    }
			    pending->taken = true;
			    NoteBuild({.stage  = ShaderRecompiler::StageName(inputs->key.stage),
			               .hash   = inputs->key.hash,
			               .words  = pending->stage.spirv.size(),
			               .steps  = pending->stage.steps,
			               .where  = "worker",
			               .reason = "dispatch"});
			    // Programs with calls are not stored: they depend on their callees' code.
			    AddModule(cached, inputs->key, std::move(pending->stage));
			    m_async->counters.modules_ready++;
			    AsyncLog("ShaderRecompiler {} translation end: hash=0x{:016x} outcome=emitted "
			             "(async, linked program)",
			             ShaderRecompiler::StageName(inputs->key.stage), inputs->key.hash);
		    }, compiling);
	    },
	    ShaderWorkers::Priority::Urgent, /*waited=*/true);
	if (!job) {
		return false;
	}
	async->wanted.insert(compiling);
	async->in_flight.emplace(compiling, Async::Flight {std::move(job), false});
	return true;
}

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
	for (auto& finished: done) {
		backlog.push_back(std::move(finished));
	}
	done.clear();
}

// Pipelines (flight 0) and the jobs a draw or dispatch is blocked on go first, in the order they
// finished; a blocked draw no longer waits behind the thousand prefetched translations that
// finished before its job did.
void PipelineCache::Async::PrioritizeBacklog() {
	if (backlog.size() < 2) {
		return;
	}
	std::stable_partition(backlog.begin(), backlog.end(), [this](const FinishedJob& job) {
		return job.flight == 0 || wanted.contains(job.flight);
	});
}

// The jobs applied here are the render-thread-bound steps only: the cache, the module and pipeline
// hand-off. Everything else ran on the workers. The backlog is applied within the time budget
// (--async-pump-budget-ms, at least one job per call), so a burst of finished jobs is spread over
// several calls instead of holding up the draw that called.
void PipelineCache::PumpAsync() {
	if (m_async == nullptr) {
		return;
	}
	RenderActivity::Scope activity("pump-async");
	const auto begin = std::chrono::steady_clock::now();
	m_async->TakeFinished();
	m_async->PrioritizeBacklog();
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
	if (const auto now = AskClockMs(); now - m_async->last_store_report_ms >= 30000) {
		m_async->last_store_report_ms = now;
		m_async->ReportStore(m_store.GetCounts());
		ReportLookups();
		ReportPrefetch();
	}
}

void PipelineCache::Async::ReportStore(const ShaderStoreWriter::Counts& counts) {
	std::printf("[store] written=%" PRIu64 " pending=%" PRIu64 " failed=%" PRIu64
	            " flushed=%" PRIu64 "\n",
	            counts.written, counts.pending, counts.failed, counts.flushed);
}

void PipelineCache::WaitForAsync() {
	if (m_async == nullptr) {
		return;
	}
	RenderActivity::Scope activity("wait-async", 0, "-", "waiting for a worker job");
	if (m_async->backlog.empty()) {
		std::unique_lock lock(m_async->done_mutex);
		m_async->done_cv.wait(lock, [this] {
			return !m_async->done.empty() || m_async->workers.Stopped();
		});
	}
	m_async->TakeFinished();
	m_async->PrioritizeBacklog();
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
	// Translations are kept while their shader may still be drawn; only over the IR budget are the
	// least recently requested dropped. The IR of a thousand translations takes the render thread
	// seconds to free, so the dropped ones are freed by a worker.
	std::vector<Translation> dropped;
	const auto budget = AsyncPlan::TakeOverBudget(
	    translations, now, static_cast<uint64_t>(Config::AsyncIrBudgetMb()) << 20, dropped);
	counters.expired += budget.count;
	if (!dropped.empty()) {
		RenderActivity::Scope activity("expire-translations");
		std::string hashes;
		for (size_t i = 0; i < std::min<size_t>(dropped.size(), 5); i++) {
			hashes += fmt::format(" 0x{:016x}", dropped[i].inputs->key.hash);
		}
		std::printf("[async] IR budget %u MiB exceeded: dropped %zu translations (%.0f MiB), the "
		            "newest of them requested %.1f s ago; kept %zu (%.0f MiB); first dropped:%s\n",
		            Config::AsyncIrBudgetMb(), budget.count,
		            static_cast<double>(budget.bytes) / 1048576.0,
		            static_cast<double>(budget.newest_age_ms) / 1000.0, budget.kept,
		            static_cast<double>(budget.kept_bytes) / 1048576.0, hashes.c_str());
		// If the workers are stopped the lambda, and with it the vector, is freed here.
		(void)workers.Post([dropped = std::move(dropped)]() mutable { dropped.clear(); });
	}
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
	const auto skipped_dispatches_now =
	skipped_dispatches != nullptr ? skipped_dispatches->TakePeriod() : uint64_t {0};
	const bool busy = !in_flight.empty() || !translations.empty() || pipelines_building != 0 ||
	skipped != 0 || !backlog.empty() || skipped_dispatches_now != 0;
	if (busy) {
		const auto   finished = workers.FinishedJobs();
		const double window_s = static_cast<double>(std::max<uint64_t>(now - last_status_ms, 1)) / 1000.0;
		std::printf("Async: %zu translating, %zu compiling, %zu waiting (oldest %.0f s), %u pipelines, "
		            "%zu queued, %" PRIu64 " skipped/2s, %zu/%zu redrawn; workers %u/%u busy, "
		            "%.1f jobs/s, pump %.1f ms (worst %.1f ms, %u applied, %zu backlog); "
		            "stages: requested %" PRIu64 ", translating %zu, translated %" PRIu64
		            " (failed %" PRIu64 "), module-compiling %zu of %" PRIu64 " jobs, modules ready %" PRIu64
		            " (speculative %" PRIu64 ", lost %" PRIu64 ", failed %" PRIu64
		            "), pipeline-compiling %u of %" PRIu64 " jobs, pipelines ready %" PRIu64
		            " (failed %" PRIu64 "), expired %" PRIu64 ", wanted %zu; skipped dispatches %" PRIu64
		            "/2s (%" PRIu64 " total); %s\n",
		            translating, compiling, translations.size(),
		            static_cast<double>(oldest) / 1000.0, pipelines_building, workers.Queued(),
		            skipped, redrawn, targets.size(), workers.Busy(), workers.ThreadCount(),
		            static_cast<double>(finished - reported_finished) / window_s,
		            static_cast<double>(pump_us) / 1000.0,
		            static_cast<double>(pump_worst_us) / 1000.0, pump_applied, backlog.size(),
		            counters.requested, translating, counters.translated, counters.translate_failed,
		            compiling, counters.module_jobs, counters.modules_ready, counters.speculative,
		            counters.speculative_lost, counters.module_failed, pipelines_building,
		            counters.pipeline_jobs, counters.pipelines_ready, counters.pipeline_failed,
		            counters.expired, wanted.size(), skipped_dispatches_now,
		            skipped_dispatches != nullptr ? skipped_dispatches->Total() : uint64_t {0},
		            build_totals != nullptr && !build_totals->Empty() ? build_totals->Text().c_str()
		            : "no builds");
		            reported_finished = finished;
	} else if (was_busy) {
		std::printf("Async: idle\n");
	}
	was_busy = busy;
	if (build_totals != nullptr) {
	*build_totals = {};
	}
	pump_us = 0;
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
		m_async->counters.pipeline_jobs++;
		// The map's nodes do not move, and the stages' modules sit in deques that only grow.
		m_async->workers.Post([this, async = m_async.get(), &graphics = m_graphics,
		driver_cache = m_driver_cache, &pipeline, key, stages,
		post_time = std::chrono::steady_clock::now(),
		hash = stages[0] != nullptr ? stages[0]->info.shader_hash : uint64_t {0}] {
		const auto queue_wait_us = BuildTiming::ElapsedUs(post_time);
		uint64_t pipeline_us = 0;
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
				const auto begin = std::chrono::steady_clock::now();
				CreatePipelineInternal(graphics, pipeline, key, stages, driver_cache);
				pipeline_us = BuildTiming::ElapsedUs(begin);
				} catch (const std::exception& e) {
				failed = true;
				fail_reason = e.what();
				}
				const auto total_us = BuildTiming::ElapsedUs(post_time);
				async->Finish([this, &pipeline, key, failed, hash, pipeline_us, queue_wait_us, total_us,
				fail_reason = std::move(fail_reason)]() mutable {
				m_async->pipelines_building--;
				pipeline.pending = false;
				if (failed) {
					m_async->counters.pipeline_failed++;
					pipeline.failed = true;
					Log::WriteToConsoleAndLog(fmt::format(
					    "!!! GRAPHICS PIPELINE SKIP (async create failed): {}\n", fail_reason));
					return;
				}
				m_async->counters.pipelines_ready++;
				NoteBuild({.stage = "pipeline-gfx",
				.hash = hash,
				.steps = {.pipeline_us = pipeline_us, .queue_wait_us = queue_wait_us, .total_us = total_us},
				.where = "worker",
				.reason = "draw"});
				StorePipeline(key);
				CheckpointDriverCache();
			});
		}, ShaderWorkers::Priority::Urgent, /*waited=*/true);
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
	RenderActivity::Note("dispatch-pipeline", module.info.shader_hash, "cs", "pipeline known");
	auto [iter, inserted]     = m_compute_pipelines.try_emplace(module.id);
	auto& pipeline            = iter->second;
	if (inserted) {
		pipeline.pending = true;
		m_async->pipelines_building++;
		m_async->counters.pipeline_jobs++;
		// Map nodes stay stable and program module deques only grow while workers run.
		m_async->workers.Post([this, async = m_async.get(), &graphics = m_graphics,
		driver_cache = m_driver_cache, &pipeline, &module,
		post_time = std::chrono::steady_clock::now()] {
		const auto queue_wait_us = BuildTiming::ElapsedUs(post_time);
		uint64_t pipeline_us = 0;
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
				const auto begin = std::chrono::steady_clock::now();
				CreatePipelineInternal(graphics, pipeline, module, driver_cache);
				pipeline_us = BuildTiming::ElapsedUs(begin);
				} catch (const std::exception& e) {
				failed = true;
				fail_reason = e.what();
				}
				const auto total_us = BuildTiming::ElapsedUs(post_time);
				async->Finish([this, &pipeline, hash = module.info.shader_hash, failed, pipeline_us,
				queue_wait_us, total_us, fail_reason = std::move(fail_reason)]() mutable {
				m_async->pipelines_building--;
				pipeline.pending = false;
				if (failed) {
					m_async->counters.pipeline_failed++;
					pipeline.failed = true;
					Log::WriteToConsoleAndLog(fmt::format(
					    "!!! COMPUTE PIPELINE SKIP (async create failed) shader=0x{:016x}: {}\n",
					    hash, fail_reason));
					return;
				}
				m_async->counters.pipelines_ready++;
				NoteBuild({.stage = "pipeline-cs",
				.hash = hash,
				.steps = {.pipeline_us = pipeline_us, .queue_wait_us = queue_wait_us, .total_us = total_us},
				.where = "worker",
				.reason = "dispatch"});
				CheckpointDriverCache();
				});
		}, ShaderWorkers::Priority::Urgent, /*waited=*/true);
	}
	if (pipeline.failed) {
		return false;
	}
	if (pipeline.pending) {
	pending = true;
	// The dispatch comes back until the pipeline is ready: one request, not many.
	m_requests.Retry(ProgramVariantKey());
	m_pending_compute_hash = module.info.shader_hash;
	m_pending_compute_identity = ProgramVariantKey();
	RenderActivity::NoteDetail("pipeline compiling");
		return false;
	}
	return true;
}

} // namespace Libs::Graphics
