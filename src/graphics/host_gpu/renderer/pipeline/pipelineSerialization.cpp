#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreCapability.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreFile.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreIdentity.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/recoverable.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <ranges>
#include <thread>
#include <unordered_set>
#include <numeric>
#include <xxhash.h>
#include <zstd.h>

// The shader store: a header, then zstd-compressed records of everything a run created first.
// Records are only appended, so a store cut short by a crash keeps every complete record.
namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

enum class Record : uint8_t {
	Module,
	GraphicsPipeline,
	TessellationLayout,
	FailedProgram,
	KeyIndex,
	UseOrder, // A module's first-use rank in the run that stored it, for the next load order.
};

// Explicit store format version. Bump manually only when recompiler output, SPIR-V profile, or
// store record layout changes so rebuilds across branches keep reusing compatible warm stores.
// 5: CompiledShaderInfo::subgroup_size, 6: KeyIndex, 7: compute keys hold lds_storage,
// 8: modules with calls (their callee code), 9: old-layout hash of a module's specialization,
// 10: UseOrder
constexpr uint64_t kShaderStoreFormatVersion = 10;

// What every record depends on besides its own fields.
std::vector<uint8_t> StoreHeader(const GraphicContext& graphics) {
	Writer header;
	IoAll(header, kShaderStoreFormatVersion);
	VisitShaderStoreProfile(SpirvProfile(graphics), [&](auto value) { IoAll(header, value); });
	IoAll(header, graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize);
	return std::move(header.bytes);
}

// Record fields, shared by writing and reading.
template <class S, class Key> void IoKey(S& io, Key& key) {
	auto& [stage, hash, user_data_count, static_state] = key;
	IoAll(io, stage, hash, user_data_count, static_state);
}

template <class S, class Module> void IoModule(S& io, Module& stage) {
	// The handle comes from the SPIR-V. Function code: the callee code of a module with calls.
	auto& [id, handle, info, specialization, function_code, rect_parameters, mesh_task,
	       legacy_hash, legacy_checked] = stage;
	IoAll(io, id, info, specialization, function_code, rect_parameters, mesh_task, legacy_hash);
}

} // namespace

void ShaderStoreWriter::Open(const std::filesystem::path& path, std::span<const uint8_t> header,
                             uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (kept == 0) {
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(kept);
		m_file.Seek(kept);
	}
}

ShaderStoreWriter::ShaderStoreWriter() {
	m_thread = std::jthread([this](std::stop_token stop) {
		std::unique_lock lock(m_mutex);
		// Once stop is requested the wait returns false only when the queue is empty.
		while (m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
			const auto job = std::move(m_jobs.front());
			m_jobs.pop();
			lock.unlock();
			job();
			lock.lock();
		}
	});
}

void ShaderStoreWriter::Append(std::vector<uint8_t>&& record) {
	m_appended.fetch_add(1, std::memory_order_relaxed);
	Post([this, record = std::move(record)] {
		const auto bytes = ShaderStoreFile::Encode(record);
		// A failed write leaves a partial record; the next run drops it and everything after it.
		const auto total   = static_cast<uint32_t>(bytes.size());
		uint32_t   written = 0;
		m_file.Write(bytes.data(), total, &written);
		(written == total ? m_written : m_failed).fetch_add(1, std::memory_order_relaxed);
		// A run that ends by an abort must lose nothing it had finished: the data is forced to the
		// disk as soon as the queue is empty, and at least every FlushIntervalMs while it is not.
		const auto now = std::chrono::steady_clock::now();
		if (QueueEmpty() || now - m_last_flush >= std::chrono::milliseconds(FlushIntervalMs)) {
			Flush();
		}
	});
}

void ShaderStoreWriter::Flush() {
	if (m_file.Flush()) {
		m_flushed.store(m_written.load(std::memory_order_relaxed), std::memory_order_relaxed);
	}
	m_last_flush = std::chrono::steady_clock::now();
}

bool ShaderStoreWriter::QueueEmpty() {
	std::scoped_lock lock(m_mutex);
	return m_jobs.empty();
}

void ShaderStoreWriter::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	m_ready.notify_one();
}

// One record of the store on its way into the cache: parsed and prepared on any thread, applied
// on the thread that owns the cache.
struct PipelineCache::StoreItem {
	size_t   offset      = 0; // Of its frame, from the first record.
	size_t   frame_bytes = 0;
	uint8_t  type        = 0; // Record.
	bool     failed      = false;
	uint32_t rank        = UINT32_MAX; // Modules: first-use rank of an earlier run.
	ProgramKey                                          key;
	std::vector<uint8_t>                                plan_bytes;
	std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan; // Modules without calls.
	ShaderModule                                        module;
	std::vector<uint32_t>                               spirv;
	uint64_t                                            spirv_words = 0;
	uint64_t                                            module_us   = 0;
	GraphicsPipelineKey                                 pipeline_key {};
	uint64_t                                            layout_key = 0;
	ShaderTessellationInputInfo                         layout;
	uint64_t                                            capability_version = 0;
	std::string                                         reason;
	uint64_t                                            module_id = 0; // UseOrder.
};

namespace {

using Clock = std::chrono::steady_clock;

uint64_t Nanoseconds(Clock::time_point since) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - since).count());
}

} // namespace

// Summed over the threads that did the work, so the parts compare whatever ran in parallel.
struct PipelineCache::WarmUpTimes {
	std::atomic<uint64_t> decode {0};
	std::atomic<uint64_t> plans {0};
	std::atomic<uint64_t> modules {0};
	std::atomic<uint64_t> pipelines {0};
};

// Decompresses and reads one frame; no cache state, no Vulkan.
void PipelineCache::ParseStoreItem(std::span<const uint8_t> frame, StoreItem& item) {
	std::vector<uint8_t> raw;
	size_t               frame_bytes = 0;
	auto                 bytes       = frame;
	if (ShaderStoreFile::Decode(bytes, true, raw, frame_bytes) != ShaderStoreFile::Status::Loaded) {
		item.failed = true;
		return;
	}
	Reader     reader {raw};
	auto       type     = Record::Module;
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	IoAll(reader, type);
	item.type = static_cast<uint8_t>(type);
	switch (type) {
		case Record::Module:
			IoKey(reader, item.key);
			IoAll(reader, item.plan_bytes);
			IoModule(reader, item.module);
			IoAll(reader, item.spirv);
			break;
		case Record::GraphicsPipeline: reader.Raw(&item.pipeline_key, sizeof(item.pipeline_key)); break;
		case Record::TessellationLayout: IoAll(reader, item.layout_key, item.layout); break;
		case Record::FailedProgram:
			IoKey(reader, item.key);
			IoAll(reader, item.capability_version, item.reason);
			break;
		case Record::KeyIndex:
			IoKey(reader, item.key);
			reader.ok = reader.ok && item.key.stage != ShaderType::Unknown &&
			            static_cast<uint32_t>(item.key.stage) <=
			                static_cast<uint32_t>(ShaderType::TessellationEvaluation);
			break;
		case Record::UseOrder: IoAll(reader, item.module_id, item.rank); break;
		default: reader.ok = false; break;
	}
	item.failed = !complete();
}

// The plan and the shader module of a module record; any thread.
void PipelineCache::PrepareStoreModule(StoreItem& item, bool keep, WarmUpTimes& times) const {
	if (item.failed || item.type != static_cast<uint8_t>(Record::Module)) return;
	// A module with calls keeps the plan of its callee code; it loads at the first use that
	// links the same code.
	if (item.module.function_code.empty()) {
		const auto begin = Clock::now();
		item.plan        = std::make_unique<ShaderRecompiler::IR::ResourcePlan>();
		if (!ShaderRecompiler::IR::LoadResourcePlan(item.plan_bytes, *item.plan)) {
			item.failed = true;
		}
		times.plans += Nanoseconds(begin);
		if (item.failed) return;
	}
	item.spirv_words = item.spirv.size();
	const auto begin = Clock::now();
	if (keep) {
		Common::RecoverableScope scope;
		try {
			item.module.module = CompileSPV(item.spirv, m_graphics.device);
		} catch (const Common::RecoverableFailure&) {
			item.failed = true;
		} catch (const std::exception&) {
			item.failed = true;
		}
	} else {
		item.module.module = CompileSPV(item.spirv, m_graphics.device);
	}
	const auto module_ns = Nanoseconds(begin);
	times.modules += module_ns;
	item.module_us = module_ns / 1'000u;
	item.spirv = {};
}

// Puts a parsed (and for modules prepared) record into the cache; owner thread only. False if
// the record cannot be used.
bool PipelineCache::ApplyStoreItem(StoreItem& item, std::vector<vk::ShaderModule>* orphans) {
	if (item.failed) return false;
	switch (static_cast<Record>(item.type)) {
		case Record::Module: {
			auto& module = item.module;
			NoteBuild({.stage = ShaderRecompiler::StageName(module.info.stage),
			           .hash = module.info.shader_hash,
			           .words = item.spirv_words,
			           .steps = {.module_us = item.module_us, .total_us = item.module_us},
			           .where = "worker",
			           .reason = "warmup"});
			const auto [program, inserted] = m_program_cache.try_emplace(item.key);
			const bool calls = !module.function_code.empty();
			if (inserted && !calls) {
				program->second.resource_plan = std::move(*item.plan);
			}
			if (std::ranges::any_of(program->second.modules,
			                        [&](const ShaderModule& other) { return other.id == module.id; })) {
				// This run already built it (a background load came late); the loader may still make
				// pipelines from the handle, so it goes with the loader.
				if (orphans != nullptr) {
					orphans->push_back(module.module);
				} else {
					m_graphics.device.destroyShaderModule(module.module, nullptr);
				}
				return true;
			}
			if (calls) {
				program->second.call_plans.insert_or_assign(
				    XXH3_64bits(module.function_code.data(),
				                module.function_code.size() * sizeof(uint32_t)),
				    std::move(item.plan_bytes));
			}
			// A module the old layout's key is known for waits for its first comparison.
			module.legacy_checked = module.legacy_hash == 0;
			program->second.legacy_unchecked += module.legacy_checked ? 0u : 1u;
			program->second.modules.push_back(std::move(module));
			return true;
		}
		case Record::GraphicsPipeline:
			// WarmUp creates the pipeline once every module is loaded.
			m_graphics_pipelines.try_emplace(item.pipeline_key);
			return true;
		case Record::TessellationLayout: m_tess_layouts.try_emplace(item.layout_key, item.layout); return true;
		case Record::FailedProgram:
			// A translator fix starts a new capability generation: a failure recorded by an
			// older translator is not evidence this one cannot translate the program.
			if (FailureCacheApplies(item.capability_version) && !Config::ShaderCacheRetryFailed()) {
				auto& program          = m_program_cache.try_emplace(item.key).first->second;
				program.skip_dispatch  = true;
				program.failure_reason =
				    item.reason.empty() ? "unspecified failure" : std::move(item.reason);
			}
			return true;
		case Record::KeyIndex:
			if (Config::ShaderKeyIndexEnabled()) {
				RecordIndexedKey(item.key, /*persist=*/false);
			}
			return true;
		case Record::UseOrder: {
			auto [rank, inserted] = m_use_rank.try_emplace(item.module_id, item.rank);
			if (!inserted) rank->second = std::min(rank->second, item.rank);
			m_next_use_rank = std::max(m_next_use_rank, item.rank + 1u);
			return true;
		}
		default: return false;
	}
}

void PipelineCache::DestroyPipeline(const Pipeline& pipeline) const {
	m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
}

// The part of the store left when --warmup-budget-s ran out, loaded on a thread of its own while
// the title runs; what it finishes is applied by DrainWarmUp on the render thread.
struct PipelineCache::WarmUpLoader {
	struct ReadyPipeline {
		bool                graphics = false;
		GraphicsPipelineKey key {};
		uint64_t            module_id = 0;
		uint64_t            hash      = 0;
		Pipeline            pipeline;
		uint64_t            pipeline_us = 0;
	};
	struct Ready {
		std::vector<std::unique_ptr<StoreItem>> items;
		std::vector<ReadyPipeline>              pipelines;
		bool                                    done = false;
	};

	std::mutex                                                 mutex;
	Ready                                                      ready;
	// Module records not yet applied or failed, per program.
	std::unordered_map<ProgramKey, uint32_t, ProgramKeyHash>   pending_keys;
	std::vector<ProgramKey>                                    urgent;
	// Render thread only: handles of modules the cache had already, destroyed when it ends.
	std::vector<vk::ShaderModule>                              orphans;
	// Compute modules loaded before the budget ran out whose pipelines it left.
	std::vector<uint64_t>                                      compute;
	// Modules in rank order; a module stays here (its copy goes to the cache) until the loader
	// ends, so the pipelines it makes can read it.
	std::vector<std::unique_ptr<StoreItem>>                    modules;
	std::vector<GraphicsPipelineKey>                           graphics;
	ModuleMap                                                  applied; // Modules loaded before.
	size_t                                                     total_modules     = 0;
	size_t                                                     total_pipelines   = 0;
	size_t                                                     drained_pipelines = 0;
	WarmUpTimes                                                times;
	Clock::time_point                                          start;
	std::jthread                                               thread;
};

void PipelineCache::WarmUpLoaderDelete::operator()(WarmUpLoader* loader) const {
	delete loader;
}

void PipelineCache::RunWarmUpLoader(std::stop_token stop) {
	auto&             loader  = *m_warmup;
	const auto        threads = Common::ParallelThreads();
	std::vector<bool> taken(loader.modules.size());
	size_t            next    = 0;
	ModuleMap         modules = loader.applied;
	std::vector<uint64_t> compute = loader.compute;
	const auto settle = [&](const ProgramKey& key) {
		if (const auto found = loader.pending_keys.find(key);
		    found != loader.pending_keys.end() && --found->second == 0) {
			loader.pending_keys.erase(found);
		}
	};
	while (!stop.stop_requested()) {
		// Programs a draw waits for first, then the rank order.
		std::vector<size_t> batch;
		{
			std::scoped_lock lock(loader.mutex);
			for (const auto& key: loader.urgent) {
				for (size_t i = 0; i < loader.modules.size(); ++i) {
					if (!taken[i] && loader.modules[i]->key == key) {
						batch.push_back(i);
						taken[i] = true;
					}
				}
			}
			loader.urgent.clear();
		}
		for (; batch.size() < threads * 2u && next < loader.modules.size(); ++next) {
			if (!taken[next]) {
				batch.push_back(next);
				taken[next] = true;
			}
		}
		if (batch.empty()) break;
		Common::ParallelFor(batch.size(), [&](size_t i) {
			PrepareStoreModule(*loader.modules[batch[i]], /*keep=*/true, loader.times);
		}, threads);
		std::scoped_lock lock(loader.mutex);
		for (const auto i: batch) {
			auto& item = *loader.modules[i];
			if (item.failed) {
				settle(item.key);
				continue;
			}
			// The cache takes a copy; the original stays for the pipelines made below.
			auto copy        = std::make_unique<StoreItem>();
			copy->type       = item.type;
			copy->key        = item.key;
			copy->module     = item.module;
			copy->plan       = std::move(item.plan);
			copy->plan_bytes = item.plan_bytes;
			copy->spirv_words = item.spirv_words;
			copy->module_us   = item.module_us;
			modules.emplace(item.module.id, &item.module);
			if (item.module.info.stage == ShaderType::Compute) compute.push_back(item.module.id);
			loader.ready.items.push_back(std::move(copy));
		}
		m_warmup_ready.store(true, std::memory_order_release);
	}
	// Compute pipelines first: a dispatch waits for its pipeline.
	std::vector<WarmUpLoader::ReadyPipeline> made;
	for (const auto id: compute) {
		const auto found = modules.find(id);
		made.push_back({.graphics = false,
		                .module_id = id,
		                .hash = found != modules.end() && found->second != nullptr
		                            ? found->second->info.shader_hash
		                            : 0});
	}
	for (const auto& key: loader.graphics) {
		uint64_t hash = 0;
		for (const auto id: key.stage_ids) {
			const auto found = modules.find(id);
			if (id != 0 && found != modules.end() && found->second != nullptr) {
				hash = found->second->info.shader_hash;
				break;
			}
		}
		made.push_back({.graphics = true, .key = key, .hash = hash});
	}
	Common::ParallelFor(made.size(), [&](size_t i) {
		if (stop.stop_requested()) return;
		auto&      entry = made[i];
		const auto begin = Clock::now();
		Common::RecoverableScope scope;
		try {
			if (entry.graphics) {
				std::array<const ShaderModule*, 4> stages {};
				for (size_t s = 0; s < stages.size(); s++) {
					const auto found = modules.find(entry.key.stage_ids[s]);
					if (found == modules.end()) return;
					stages[s] = found->second;
				}
				if (stages[0] == nullptr) return;
				CreatePipelineInternal(m_graphics, entry.pipeline, entry.key, stages, m_driver_cache);
			} else if (const auto found = modules.find(entry.module_id);
			           found != modules.end() && found->second != nullptr) {
				CreatePipelineInternal(m_graphics, entry.pipeline, *found->second, m_driver_cache);
			}
		} catch (const std::exception&) {
			entry.pipeline = {};
		}
		const auto pipeline_ns = Nanoseconds(begin);
		loader.times.pipelines += pipeline_ns;
		entry.pipeline_us = pipeline_ns / 1'000u;
	}, threads);
	std::scoped_lock lock(loader.mutex);
	for (auto& entry: made) {
		if (entry.pipeline.pipeline) loader.ready.pipelines.push_back(std::move(entry));
	}
	loader.ready.done = true;
	m_warmup_ready.store(true, std::memory_order_release);
}

bool PipelineCache::WarmUpPending(const ProgramKey& key) {
	if (m_warmup == nullptr) return false;
	std::scoped_lock lock(m_warmup->mutex);
	if (!m_warmup->pending_keys.contains(key)) return false;
	m_warmup->urgent.push_back(key);
	return true;
}

void PipelineCache::DrainWarmUp() {
	if (m_warmup == nullptr || !m_warmup_ready.exchange(false, std::memory_order_acquire)) return;
	auto&               loader = *m_warmup;
	WarmUpLoader::Ready ready;
	{
		std::scoped_lock lock(loader.mutex);
		std::swap(ready, loader.ready);
		for (const auto& item: ready.items) {
			if (const auto found = loader.pending_keys.find(item->key);
			    found != loader.pending_keys.end() && --found->second == 0) {
				loader.pending_keys.erase(found);
			}
		}
	}
	for (auto& item: ready.items) ApplyStoreItem(*item, &loader.orphans);
	loader.drained_pipelines += ready.pipelines.size();
	for (auto& entry: ready.pipelines) {
		NoteBuild({.stage = entry.graphics ? "pipeline-gfx" : "pipeline-cs",
		           .hash = entry.hash,
		           .steps = {.pipeline_us = entry.pipeline_us, .total_us = entry.pipeline_us},
		           .where = "worker",
		           .reason = "warmup"});
		auto& existing = entry.graphics ? m_graphics_pipelines.try_emplace(entry.key).first->second
		                                : m_compute_pipelines.try_emplace(entry.module_id).first->second;
		if (existing.pipeline || existing.pending) {
			DestroyPipeline(entry.pipeline); // This run made it meanwhile.
		} else {
			existing = entry.pipeline;
		}
	}
	if (ready.done) {
		loader.thread.join();
		Log::WriteToConsoleAndLog(fmt::format(
		    "[warmup] background load done after {} ms: modules={} pipelines={} plans={}ms "
		    "modules={}ms pipelines={}ms\n",
		    Nanoseconds(loader.start) / 1'000'000u, loader.total_modules, loader.drained_pipelines,
		    loader.times.plans.load() / 1'000'000u, loader.times.modules.load() / 1'000'000u,
		    loader.times.pipelines.load() / 1'000'000u));
		for (const auto handle: loader.orphans) m_graphics.device.destroyShaderModule(handle, nullptr);
		// The cache holds copies of the loaded modules and owns their handles.
		m_warmup.reset();
		CheckpointDriverCache();
	}
}

void PipelineCache::StopWarmUp() {
	if (m_warmup == nullptr) return;
	auto& loader = *m_warmup;
	loader.thread.request_stop();
	if (loader.thread.joinable()) loader.thread.join();
	// What the cache did not take yet is destroyed here.
	for (const auto& item: loader.ready.items) {
		m_graphics.device.destroyShaderModule(item->module.module, nullptr);
	}
	for (const auto& entry: loader.ready.pipelines) DestroyPipeline(entry.pipeline);
	for (const auto handle: loader.orphans) m_graphics.device.destroyShaderModule(handle, nullptr);
	m_warmup.reset();
}

void PipelineCache::WarmUp(const std::filesystem::path& path) {
	if (path.empty()) {
		return;
	}
	const auto header = StoreHeader(m_graphics);
	if (header.empty()) {
		return;
	}
	const auto           start = Clock::now();
	std::vector<uint8_t> bytes;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		bytes.resize(read);
	}
	// Before the title starts nothing else runs: every core loads.
	const auto threads = std::max<size_t>(1u, std::thread::hardware_concurrency());
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: compiling shaders and pipelines from {} ({:.1f} MB) with {} threads\n",
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0, threads));
	auto       records        = std::span<const uint8_t>(bytes);
	const bool keep           = Config::ShaderCacheKeepEnabled();
	const bool header_matches =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
	const bool load_store     = header_matches || (keep && records.size() >= header.size());
	const auto budget_ms      = s_warmup_budget_ms_for_tests >= 0
	                                ? static_cast<uint64_t>(s_warmup_budget_ms_for_tests)
	                                : uint64_t {Config::WarmUpBudgetSeconds()} * 1000u;
	const bool budget         = budget_ms != 0 || s_warmup_budget_ms_for_tests == 0;
	size_t     loaded_records = 0;
	size_t     dropped_records = 0;
	WarmUpTimes times;

	// Frames: the framing and the checksums, in file order. Without keep a damaged frame ends the
	// store; with it the frame is skipped.
	std::vector<std::unique_ptr<StoreItem>> items;
	size_t                                  kept_bytes = 0; // Records that stay, from the first.
	if (load_store) {
		records = records.subspan(header.size());
		size_t offset = 0;
		for (;;) {
			auto                 rest = records.subspan(offset);
			size_t               frame_bytes = 0;
			ShaderStoreFile::Frame frame;
			if (rest.size() < sizeof(frame)) break;
			std::memcpy(&frame, rest.data(), sizeof(frame));
			if (rest.size() - sizeof(frame) < frame.compressed_size) break;
			frame_bytes = sizeof(frame) + frame.compressed_size;
			const auto compressed = rest.subspan(sizeof(frame), frame.compressed_size);
			const bool valid = XXH3_64bits(compressed.data(), compressed.size()) == frame.checksum;
			if (!valid) {
				if (!keep) break;
				dropped_records++;
				offset += frame_bytes;
				continue;
			}
			auto item         = std::make_unique<StoreItem>();
			item->offset      = offset;
			item->frame_bytes = frame_bytes;
			items.push_back(std::move(item));
			offset += frame_bytes;
		}
		kept_bytes = offset;
	}
	const auto frames_ns = Nanoseconds(start);

	// Decode: decompress and read every frame in parallel.
	Common::ParallelFor(items.size(), [&](size_t i) {
		const auto begin = Clock::now();
		ParseStoreItem(records.subspan(items[i]->offset, items[i]->frame_bytes), *items[i]);
		times.decode += Nanoseconds(begin);
	}, threads);
	// Without keep (and without a budget, which skips instead) the first unreadable record ends
	// the store like a damaged frame.
	if (!keep && !budget) {
		for (size_t i = 0; i < items.size(); ++i) {
			if (items[i]->failed) {
				kept_bytes = items[i]->offset;
				items.resize(i);
				break;
			}
		}
	}

	// Everything but the modules is small and applied in file order; the use order ranks the
	// modules, the earliest first, then the unranked in file order.
	std::vector<std::unique_ptr<StoreItem>> modules;
	for (auto& item: items) {
		if (!item->failed && item->type == static_cast<uint8_t>(Record::Module)) {
			modules.push_back(std::move(item));
		} else if (ApplyStoreItem(*item)) {
			loaded_records++;
		} else {
			dropped_records++;
		}
	}
	items.clear();
	for (auto& item: modules) {
		const auto found = m_use_rank.find(item->module.id);
		item->rank       = found != m_use_rank.end() ? found->second : UINT32_MAX;
	}
	std::ranges::stable_sort(modules, {}, &StoreItem::rank);

	// Modules: plan and shader module per record on every core, in rank order, in chunks so a
	// budget can stop between them.
	const auto over_budget = [&] { return budget && Nanoseconds(start) / 1'000'000u >= budget_ms; };
	size_t     prepared    = 0;
	const auto chunk       = std::max<size_t>(threads * 4u, 64u);
	while (prepared < modules.size() && !over_budget()) {
		const auto count = std::min(chunk, modules.size() - prepared);
		Common::ParallelFor(count, [&](size_t i) {
			PrepareStoreModule(*modules[prepared + i], keep || budget, times);
		}, threads);
		for (size_t i = 0; i < count; ++i) {
			auto& item = *modules[prepared + i];
			if (ApplyStoreItem(item)) {
				loaded_records++;
			} else {
				dropped_records++;
			}
		}
		prepared += count;
	}
	const auto modules_left = modules.size() - prepared;

	// Pipelines of the loaded modules, in rank order of their modules.
	const auto module_map = MapModules();
	const auto rank_of    = [&](uint64_t id) {
		const auto found = m_use_rank.find(id);
		return found != m_use_rank.end() ? found->second : UINT32_MAX;
	};
	struct PipelineJob {
		uint32_t                     rank = UINT32_MAX;
		const GraphicsPipelineKey*   graphics = nullptr;
		Pipeline*                    pipeline = nullptr;
		const ShaderModule*          module   = nullptr;
		uint64_t                     hash = 0;
		uint64_t                     pipeline_us = 0;
	};
	std::vector<PipelineJob> pipeline_jobs;
	std::vector<GraphicsPipelineKey> deferred_graphics;
	for (auto& [key, pipeline]: m_graphics_pipelines) {
		bool     ready = true;
		uint64_t hash  = 0;
		uint32_t rank = UINT32_MAX;
		for (const auto id: key.stage_ids) {
			if (id == 0) continue;
			const auto found = module_map.find(id);
			ready = ready && found != module_map.end();
			if (hash == 0 && found != module_map.end() && found->second != nullptr) {
				hash = found->second->info.shader_hash;
			}
			rank  = std::min(rank, rank_of(id));
		}
		if (ready) {
			pipeline_jobs.push_back(
			    {.rank = rank, .graphics = &key, .pipeline = &pipeline, .hash = hash});
		} else if (budget && modules_left != 0) {
			deferred_graphics.push_back(key);
		}
	}
	for (const auto* module: module_map | std::views::values) {
		if (module != nullptr && module->info.stage == ShaderType::Compute) {
			pipeline_jobs.push_back({.rank = rank_of(module->id),
			                         .pipeline = &m_compute_pipelines[module->id],
			                         .module = module,
			                         .hash = module->info.shader_hash});
		}
	}
	std::ranges::stable_sort(pipeline_jobs, {}, &PipelineJob::rank);
	std::atomic<size_t> dropped_pipelines = 0;
	size_t              created = 0;
	while (created < pipeline_jobs.size() && !over_budget()) {
		const auto count = std::min(chunk, pipeline_jobs.size() - created);
		Common::ParallelFor(count, [&](size_t i) {
			auto&      job   = pipeline_jobs[created + i];
			const auto begin = Clock::now();
			const auto make  = [&] {
				if (job.graphics != nullptr) {
					CreateGraphicsPipeline(*job.pipeline, *job.graphics, module_map);
				} else {
					CreatePipelineInternal(m_graphics, *job.pipeline, *job.module, m_driver_cache);
				}
			};
			if (!keep && !budget) {
				make();
			} else {
				Common::RecoverableScope scope;
				try {
					make();
				} catch (const Common::RecoverableFailure&) {
					*job.pipeline = {};
					dropped_pipelines.fetch_add(1, std::memory_order_relaxed);
				} catch (const std::exception&) {
					*job.pipeline = {};
					dropped_pipelines.fetch_add(1, std::memory_order_relaxed);
				}
			}
			const auto pipeline_ns = Nanoseconds(begin);
			times.pipelines += pipeline_ns;
			job.pipeline_us = pipeline_ns / 1'000u;
		}, threads);
		for (size_t i = 0; i < count; ++i) {
			const auto& job = pipeline_jobs[created + i];
			if (job.pipeline->pipeline) {
				NoteBuild({.stage = job.graphics != nullptr ? "pipeline-gfx" : "pipeline-cs",
				           .hash = job.hash,
				           .steps = {.pipeline_us = job.pipeline_us, .total_us = job.pipeline_us},
				           .where = "worker",
				           .reason = "warmup"});
			}
		}
		created += count;
	}
	// Pipelines left by the budget go to the background loader with the modules.
	for (size_t i = created; i < pipeline_jobs.size(); ++i) {
		if (pipeline_jobs[i].graphics != nullptr) deferred_graphics.push_back(*pipeline_jobs[i].graphics);
	}
	// A key with a stage missing from the store keeps no pipeline; one left for the loader is
	// entered when it is made.
	std::erase_if(m_graphics_pipelines,
	              [](const auto& entry) { return entry.second.pipeline == nullptr; });
	std::erase_if(m_compute_pipelines,
	              [](const auto& entry) { return entry.second.pipeline == nullptr; });
	const auto pipelines_left = pipeline_jobs.size() - created;

	const auto dropped_pipeline_count = dropped_pipelines.load();
	// With a budget a record that cannot be read is skipped, so the store keeps every frame.
	if (bytes.empty() || !keep) {
		// A damaged record ends the store; the records before it stay.
		m_store.Open(path, header, header_matches ? header.size() + kept_bytes : 0);
	} else if (header_matches && dropped_records == 0 && dropped_pipeline_count == 0) {
		m_store.Open(path, header, header.size() + kept_bytes);
	}

	if (!m_graphics_pipelines.empty() || !m_compute_pipelines.empty()) {
		CheckpointDriverCache();
	}
	if (keep || dropped_records != 0 || dropped_pipeline_count != 0) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader cache: keep mode{} - loaded {} records, dropped {} ({} records, {} pipelines)\n",
		    header_matches ? "" : " (header mismatch)", loaded_records,
		    dropped_records + dropped_pipeline_count, dropped_records, dropped_pipeline_count));
	}
	if (Config::ShaderKeyIndexEnabled() && !m_key_index.empty()) {
		size_t indexed_keys = 0;
		for (const auto& keys: m_key_index | std::views::values) {
			indexed_keys += keys.size();
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader cache: indexed {} keys for {} hashes\n", indexed_keys, m_key_index.size()));
	}
	{
		// Per stage: programs with at least one module, and programs recorded as failed.
		constexpr size_t stages = static_cast<size_t>(ShaderType::TessellationEvaluation) + 1;
		std::array<size_t, stages> with_modules {};
		std::array<size_t, stages> failed {};
		for (const auto& [key, program]: m_program_cache) {
			const auto stage = static_cast<size_t>(key.stage);
			if (stage >= stages) {
				continue;
			}
			with_modules[stage] += program.modules.empty() ? 0u : 1u;
			failed[stage] += program.skip_dispatch ? 1u : 0u;
		}
		const auto text = [&](const auto& counts) {
			return fmt::format("vs={} ps={} cs={} gs={} ls={} hs={} tes={}",
			                   counts[static_cast<size_t>(ShaderType::Vertex)],
			                   counts[static_cast<size_t>(ShaderType::Pixel)],
			                   counts[static_cast<size_t>(ShaderType::Compute)],
			                   counts[static_cast<size_t>(ShaderType::Mesh)],
			                   counts[static_cast<size_t>(ShaderType::Local)],
			                   counts[static_cast<size_t>(ShaderType::TessellationControl)],
			                   counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		};
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader cache: loaded per stage: programs with modules {}; failed programs {}\n",
		    text(with_modules), text(failed)));
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: loaded {} programs, {} modules, {} pipelines in {} ms\n",
	    m_program_cache.size(), module_map.size() - 1,
	    m_graphics_pipelines.size() + m_compute_pipelines.size(), Nanoseconds(start) / 1'000'000u));
	// The parts summed over the threads that ran them; wall is the time before the title starts.
	Log::WriteToConsoleAndLog(fmt::format(
	    "[warmup] frames={}ms decode={}ms plans={}ms modules={}ms pipelines={}ms threads={} "
	    "wall={}ms ranked={}\n",
	    frames_ns / 1'000'000u, times.decode.load() / 1'000'000u, times.plans.load() / 1'000'000u,
	    times.modules.load() / 1'000'000u, times.pipelines.load() / 1'000'000u, threads,
	    Nanoseconds(start) / 1'000'000u, m_use_rank.size()));

	if (modules_left != 0 || pipelines_left != 0) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[warmup] budget {} s: the title starts after {} of {} modules and {} of {} pipelines; "
		    "the rest loads in the background\n",
		    budget_ms / 1000u, prepared, modules.size(), created, pipeline_jobs.size()));
		m_warmup.reset(new WarmUpLoader());
		auto& loader            = *m_warmup;
		loader.start            = Clock::now();
		loader.applied          = module_map;
		loader.graphics         = std::move(deferred_graphics);
		loader.total_modules    = modules_left;
		loader.total_pipelines  = pipelines_left;
		for (size_t i = prepared; i < modules.size(); ++i) {
			loader.pending_keys[modules[i]->key]++;
			loader.modules.push_back(std::move(modules[i]));
		}
		// Compute pipelines of modules loaded before the budget ran out that it left.
		for (size_t i = created; i < pipeline_jobs.size(); ++i) {
			if (pipeline_jobs[i].module != nullptr) loader.compute.push_back(pipeline_jobs[i].module->id);
		}
		loader.thread = std::jthread([this](std::stop_token stop) { RunWarmUpLoader(stop); });
	}
}

// Records are encoded here, since programs and modules change later.
void PipelineCache::StoreModule(const ProgramKey& key, const std::vector<uint8_t>& plan,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::Module);
	IoKey(record, key);
	auto plan_bytes = plan;
	IoAll(record, plan_bytes);
	IoModule(record, module);
	IoAll(record, spirv);
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::NoteModuleUse(const ShaderModule& module) {
	if (!m_store || module.id == 0) return;
	const auto [rank, inserted] = m_use_rank.try_emplace(module.id, m_next_use_rank);
	if (!inserted) return;
	m_next_use_rank++;
	Writer record;
	auto   id    = module.id;
	auto   value = rank->second;
	IoAll(record, Record::UseOrder, id, value);
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreFailedProgram(const ProgramKey& key, std::string_view reason) {
	if (!m_store) {
		return;
	}
	Writer      record;
	std::string text(reason);
	IoAll(record, Record::FailedProgram);
	IoKey(record, key);
	IoAll(record, kTranslatorCapabilityVersion, text);
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreKeyIndex(const ProgramKey& key) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::KeyIndex);
	IoKey(record, key);
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StorePipeline(const GraphicsPipelineKey& key) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::GraphicsPipeline);
	record.Raw(&key, sizeof(key));
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::TessellationLayout, key, layout);
	m_store.Append(std::move(record.bytes));
}

PipelineCache::ModuleMap PipelineCache::MapModules() const {
	ModuleMap modules {{0, nullptr}};
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			modules.emplace(module.id, &module);
		}
	}
	return modules;
}

bool PipelineCache::CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                           const ModuleMap& modules) {
	std::array<const ShaderModule*, 4> stages {};
	for (size_t i = 0; i < stages.size(); i++) {
		const auto found = modules.find(key.stage_ids[i]);
		if (found == modules.end()) {
			return false;
		}
		stages[i] = found->second;
	}
	if (stages[0] == nullptr) {
		return false;
	}
	CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	return true;
}

} // namespace Libs::Graphics
