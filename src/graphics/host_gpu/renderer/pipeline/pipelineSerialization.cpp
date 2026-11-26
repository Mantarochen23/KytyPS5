#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreCapability.h"
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
#include <xxhash.h>
#include <zstd.h>

// The shader store: a header, then zstd-compressed records of everything a run created first.
// Records are only appended, so a store cut short by a crash keeps every complete record.
namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

enum class Record : uint8_t { Module, GraphicsPipeline, TessellationLayout, FailedProgram, KeyIndex };

// Precedes each record's zstd frame.
struct Frame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	uint64_t checksum        = 0; // XXH3 of the compressed bytes.
};

// Explicit store format version. Bump manually only when recompiler output, SPIR-V profile, or
// store record layout changes so rebuilds across branches keep reusing compatible warm stores.
constexpr uint64_t kShaderStoreFormatVersion = 6; // 5: CompiledShaderInfo::subgroup_size, 6: KeyIndex

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
	// The handle comes from the SPIR-V; stored programs have no function code.
	auto& [id, handle, info, specialization, function_code, rect_parameters, mesh_task] = stage;
	IoAll(io, id, info, specialization, rect_parameters, mesh_task);
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
		std::vector<uint8_t> bytes(sizeof(Frame) + ZSTD_compressBound(record.size()));
		const auto size = ZSTD_compress(bytes.data() + sizeof(Frame), bytes.size() - sizeof(Frame),
		                                record.data(), record.size(), 3);
		EXIT_IF(ZSTD_isError(size));
		const Frame frame {.compressed_size = static_cast<uint32_t>(size),
		                   .raw_size        = static_cast<uint32_t>(record.size()),
		                   .checksum        = XXH3_64bits(bytes.data() + sizeof(Frame), size)};
		std::memcpy(bytes.data(), &frame, sizeof(frame));
		// A failed write leaves a partial record; the next run drops it and everything after it.
		const auto total   = static_cast<uint32_t>(sizeof(Frame) + size);
		uint32_t   written = 0;
		m_file.Write(bytes.data(), total, &written);
		(written == total ? m_written : m_failed).fetch_add(1, std::memory_order_relaxed);
	});
}

void ShaderStoreWriter::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	m_ready.notify_one();
}

void PipelineCache::WarmUp(const std::filesystem::path& path) {
	if (path.empty()) {
		return;
	}
	const auto header = StoreHeader(m_graphics);
	if (header.empty()) {
		return;
	}
	const auto           start = std::chrono::steady_clock::now();
	std::vector<uint8_t> bytes;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		bytes.resize(read);
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: compiling shaders and pipelines from {} ({:.1f} MB) with {} threads\n",
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    Common::ParallelThreads()));
	auto       records        = std::span<const uint8_t>(bytes);
	const bool keep           = Config::ShaderCacheKeepEnabled();
	const bool header_matches =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
	const bool load_store     = header_matches || (keep && records.size() >= header.size());
	size_t     loaded_records = 0;
	size_t     dropped_records = 0;
	if (load_store) {
		records = records.subspan(header.size());
		for (;;) {
			const auto status = LoadRecord(records, keep);
			if (status == LoadRecordStatus::Loaded) {
				loaded_records++;
			} else if (status == LoadRecordStatus::Failed && keep) {
				dropped_records++;
			} else {
				break;
			}
		}
	}

	const auto modules = MapModules();
	std::vector<Common::UniqueFunction<void>> jobs;
	jobs.reserve(m_graphics_pipelines.size() + modules.size());
	std::atomic<size_t> dropped_pipelines = 0;
	for (auto& node: m_graphics_pipelines) {
		jobs.emplace_back([this, &modules, &node, keep, &dropped_pipelines] {
			if (!keep) {
				CreateGraphicsPipeline(node.second, node.first, modules);
				return;
			}
			Common::RecoverableScope scope;
			try {
				CreateGraphicsPipeline(node.second, node.first, modules);
			} catch (const Common::RecoverableFailure&) {
				node.second.pipeline = nullptr;
				dropped_pipelines.fetch_add(1, std::memory_order_relaxed);
			} catch (const std::exception&) {
				node.second.pipeline = nullptr;
				dropped_pipelines.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	// A compute pipeline depends only on its module.
	for (const auto* module: modules | std::views::values) {
		if (module != nullptr && module->info.stage == ShaderType::Compute) {
			jobs.emplace_back([this, module, &pipeline = m_compute_pipelines[module->id], keep,
			                   &dropped_pipelines] {
				if (!keep) {
					CreatePipelineInternal(m_graphics, pipeline, *module, m_driver_cache);
					return;
				}
				Common::RecoverableScope scope;
				try {
					CreatePipelineInternal(m_graphics, pipeline, *module, m_driver_cache);
				} catch (const Common::RecoverableFailure&) {
					pipeline.pipeline = nullptr;
					dropped_pipelines.fetch_add(1, std::memory_order_relaxed);
				} catch (const std::exception&) {
					pipeline.pipeline = nullptr;
					dropped_pipelines.fetch_add(1, std::memory_order_relaxed);
				}
			});
		}
	}
	Common::ParallelFor(jobs.size(), [&](size_t i) { jobs[i](); });
	// A key with a stage missing from the store keeps no pipeline.
	std::erase_if(m_graphics_pipelines,
	              [](const auto& entry) { return entry.second.pipeline == nullptr; });
	std::erase_if(m_compute_pipelines,
	              [](const auto& entry) { return entry.second.pipeline == nullptr; });

	const auto dropped_pipeline_count = dropped_pipelines.load();
	if (bytes.empty() || !keep) {
		// A damaged record ends the store; the records before it stay.
		m_store.Open(path, header, header_matches ? bytes.size() - records.size() : 0);
	} else if (header_matches && dropped_records == 0 && dropped_pipeline_count == 0) {
		m_store.Open(path, header, bytes.size() - records.size());
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
	    m_program_cache.size(), modules.size() - 1,
	    m_graphics_pipelines.size() + m_compute_pipelines.size(),
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
	                                                          start)
	        .count()));
}

PipelineCache::LoadRecordStatus PipelineCache::LoadRecord(std::span<const uint8_t>& records,
                                                          bool keep_on_error) {
	Frame frame;
	if (records.size() < sizeof(frame)) {
		return LoadRecordStatus::EndOfStream;
	}
	std::memcpy(&frame, records.data(), sizeof(frame));
	if (records.size() - sizeof(frame) < frame.compressed_size) {
		return LoadRecordStatus::EndOfStream;
	}
	const auto compressed = records.subspan(sizeof(frame), frame.compressed_size);
	const auto advance    = [&] { records = records.subspan(sizeof(frame) + compressed.size()); };
	if (keep_on_error) {
		advance();
	}
	if (XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return LoadRecordStatus::Failed;
	}
	std::vector<uint8_t> raw(frame.raw_size);
	if (ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) != raw.size()) {
		return LoadRecordStatus::Failed;
	}
	Reader     reader {raw};
	auto       type     = Record::Module;
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	IoAll(reader, type);
	switch (type) {
		case Record::Module: {
			ProgramKey            key;
			std::vector<uint8_t>  plan;
			ShaderModule          module;
			std::vector<uint32_t> spirv;
			IoKey(reader, key);
			IoAll(reader, plan);
			IoModule(reader, module);
			IoAll(reader, spirv);
			if (!complete()) {
				return LoadRecordStatus::Failed;
			}
			const auto [program, inserted] = m_program_cache.try_emplace(std::move(key));
			if (inserted &&
			    !ShaderRecompiler::IR::LoadResourcePlan(plan, program->second.resource_plan)) {
				m_program_cache.erase(program);
				return LoadRecordStatus::Failed;
			}
			if (keep_on_error) {
				Common::RecoverableScope scope;
				try {
					module.module = CompileSPV(spirv, m_graphics.device);
				} catch (const Common::RecoverableFailure&) {
					if (inserted) {
						m_program_cache.erase(program);
					}
					return LoadRecordStatus::Failed;
				} catch (const std::exception&) {
					if (inserted) {
						m_program_cache.erase(program);
					}
					return LoadRecordStatus::Failed;
				}
			} else {
				module.module = CompileSPV(spirv, m_graphics.device);
			}
			program->second.modules.push_back(std::move(module));
			break;
		}
		case Record::GraphicsPipeline: {
			GraphicsPipelineKey key;
			reader.Raw(&key, sizeof(key));
			if (!complete()) {
				return LoadRecordStatus::Failed;
			}
			// WarmUp creates the pipeline once every record is loaded.
			m_graphics_pipelines.try_emplace(key);
			break;
		}
		case Record::TessellationLayout: {
			uint64_t                    key = 0;
			ShaderTessellationInputInfo layout;
			IoAll(reader, key, layout);
			if (!complete()) {
				return LoadRecordStatus::Failed;
			}
			m_tess_layouts.try_emplace(key, layout);
			break;
		}
		case Record::FailedProgram: {
			ProgramKey  key;
			uint64_t    capability_version = 0;
			std::string reason;
			IoKey(reader, key);
			IoAll(reader, capability_version, reason);
			if (!complete()) {
				return LoadRecordStatus::Failed;
			}
			// A translator fix starts a new capability generation: a failure recorded by an
			// older translator is not evidence this one cannot translate the program.
			if (FailureCacheApplies(capability_version) && !Config::ShaderCacheRetryFailed()) {
				auto& program          = m_program_cache.try_emplace(std::move(key)).first->second;
				program.skip_dispatch  = true;
				program.failure_reason = std::move(reason);
			}
			break;
		}
		case Record::KeyIndex: {
			ProgramKey key;
			IoKey(reader, key);
			if (!complete() || key.stage == ShaderType::Unknown ||
			    static_cast<uint32_t>(key.stage) >
			        static_cast<uint32_t>(ShaderType::TessellationEvaluation)) {
				return LoadRecordStatus::Failed;
			}
			if (Config::ShaderKeyIndexEnabled()) {
				RecordIndexedKey(key, /*persist=*/false);
			}
			break;
		}
		default: return LoadRecordStatus::Failed;
	}
	if (!keep_on_error) {
		advance();
	}
	return LoadRecordStatus::Loaded;
}

// Records are encoded here, since programs and modules change later.
void PipelineCache::StoreModule(const ProgramKey& key, const Program& program,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::Module);
	IoKey(record, key);
	IoAll(record, ShaderRecompiler::IR::SaveResourcePlan(program.resource_plan));
	IoModule(record, module);
	IoAll(record, spirv);
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
