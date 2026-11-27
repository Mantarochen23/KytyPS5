#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/pipeline/buildTiming.h"
#include "graphics/host_gpu/renderer/pipeline/flowPlan.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreDiff.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderParams;
class CommandBuffer;

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
} // namespace ShaderRecompiler

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

namespace ShaderRecompiler::Spirv {
struct Profile;
} // namespace ShaderRecompiler::Spirv

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
		uint8_t  format  = 0; // Guest buffer format.
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

// Everything a graphics pipeline is made of besides the stage modules its ids name. Aligned so
// that rendering.color_formats can be passed to Vulkan.
struct alignas(4) GraphicsPipelineKey {
	// VS, LS or mesh; HS; TES; PS. 0 marks an absent stage.
	std::array<uint64_t, 4>  stage_ids {};
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

// Keys are compared and hashed as bytes.
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

// A compiled stage with what pipeline creation needs from it besides the key.
struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
	// Indirect mesh draw: a task shader reads the records and launches the groups.
	bool                                         mesh_task = false;
};

// A draw's depth target and its first color_count color targets.
struct DrawTargets {
	RenderDepthInfo depth;
	RenderColorInfo colors[RENDER_COLOR_ATTACHMENTS_MAX];
	uint32_t        color_count = 0;
};

// A draw's topology and shader stages (from PipelineCache::GetGraphicsPrograms).
struct DrawRenderState {
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable = false;
	bool                                 ps_active                = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets                indirect_offsets;
	bool                                 indirect_draw = false;
	// Output: false when the draw must not be skipped while its shaders are built on workers,
	// because it draws a small cached target (glyphs, UI layers) the game may not redraw.
	bool                                 deferrable    = true;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
};

// Writes the shader store on its own thread, running jobs in the order they were posted.
class ShaderStoreWriter {
public:
	ShaderStoreWriter();
	// Keeps the first `kept` bytes of the store at `path`, or starts it over with `header`.
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	// Whether the store is open.
	explicit operator bool() const { return !m_file.IsInvalid(); }
	// Compresses `record` and appends it on the store thread, in call order.
	void Append(std::vector<uint8_t>&& record);
	void Post(Common::UniqueFunction<void>&& job);
	// Records appended, written to the file, and failed to write, for the status line. `flushed`
	// is how many of the written ones were forced to the disk.
	struct Counts {
		uint64_t written = 0;
		uint64_t pending = 0;
		uint64_t failed  = 0;
		uint64_t flushed = 0;
	};
	// The longest a written record waits for the disk while more records are queued behind it.
	static constexpr int FlushIntervalMs = 500;
	[[nodiscard]] Counts GetCounts() const {
		const auto appended = m_appended.load(std::memory_order_relaxed);
		const auto written  = m_written.load(std::memory_order_relaxed);
		const auto failed   = m_failed.load(std::memory_order_relaxed);
		return {written, appended - written - failed, failed,
		        m_flushed.load(std::memory_order_relaxed)};
	}
	// Runs the queued jobs and stops the thread; jobs posted later never run. Safe to call again.
	void Finish() { m_thread = {}; }

private:
	friend struct PipelineCacheTestAccess;

	// Store thread only: forces what is written to the disk.
	void Flush();
	[[nodiscard]] bool QueueEmpty();

	Common::File                             m_file;
	std::chrono::steady_clock::time_point    m_last_flush = std::chrono::steady_clock::now();
	std::atomic<uint64_t>                    m_flushed {0};
	std::atomic<uint64_t>                    m_appended {0};
	std::atomic<uint64_t>                    m_written {0};
	std::atomic<uint64_t>                    m_failed {0};
	std::atomic<uint64_t>                    m_dropped {0};
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Without a shader store path nothing is kept across runs.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Finishes the queued writes and stops the store thread, then writes the driver cache if it
	// changed.
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// The push constant range's stages, which every update names.
		vk::ShaderStageFlags    push_stages;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
		// A worker is still creating it; only the render thread reads or clears this.
		bool                    pending               = false;
		bool                    failed                = false;
	};

	static constexpr size_t MaxIndexedKeysPerHash = 4;

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Returns the indexed ProgramKeys previously recorded for `hash` (empty if none or disabled).
	[[nodiscard]] std::span<const ProgramKey> FindIndexedKeys(uint64_t hash) const;

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// Null skips the dispatch unless `pending` is non-null and set to true (building on a worker).
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info,
	                                      bool*                        pending = nullptr);
	// Port wave64-over-limit-32: oversized wave64 groups get 32-lane host subgroups.
	void LimitComputeSubgroup(ShaderComputeInputInfo& input_info) const;
	// Starts translating a compute shader discovered by PM4 lookahead on a worker thread.
	// `user_data` (null = unknown) are the compute user data the dispatch will have, as far as the
	// scan of the command stream could follow them; with them a shader whose resources follow from
	// its user data is built to its module ahead of the dispatch.
	bool PrefetchComputeProgram(const HW::CsStageRegisters& cs_regs,
	                            bool dispatch_thread_dimensions, bool async_compute,
	                            const HW::UserSgprInfo* user_data = nullptr);
	// A dispatch gave up waiting for its shader (--async-compute-wait-ms) and is skipped. Counts it
	// and logs the first skip of each shader. Render thread only.
	void NoteDispatchSkipped(uint64_t waited_ms);
	// The `[prefetch]` line.
	void ReportPrefetch() const;
	// The `[mem]` owners the cache can count (render thread, every 2 s); also drops on --mem-limit-gb.
	void AccountMemory();
	// Counters of what the scans and registrations queued for translation, printed every 30 s.
	struct PrefetchCounters {
		uint64_t requested   = 0; // calls
		uint64_t queued      = 0; // translation jobs posted
		uint64_t unmapped    = 0; // no registered shader at the address, or none usable
		uint64_t known       = 0; // program, translation or sync-only mark exists already
		uint64_t in_flight   = 0; // translation job already running
		uint64_t with_data   = 0; // queued with the user data of a scan
		uint64_t lookahead_modules_built = 0;
		uint64_t lookahead_hits          = 0;
	};
	// Registers a callback invoked on worker threads whenever an async shader/pipeline job
	// finishes, so the guest command processor wakes without polling.
	void SetAsyncFinishCallback(Common::UniqueFunction<void>&& callback);

	// With --async-shader-compile, starts the draw's pipeline on a worker if it has none and says
	// whether it is ready; the draw is skipped until it is. Otherwise true, and
	// GetGraphicsPipeline creates the pipeline.
	[[nodiscard]] bool EnsureGraphicsPipeline(const CommandBuffer& command,
	                                          const DrawRenderState& state,
	                                          const DrawTargets& targets);
	// With --async-shader-compile, starts a compute pipeline on a worker and reports `pending`
	// so the command processor can suspend without skipping the dispatch.
	[[nodiscard]] bool EnsureComputePipeline(const ShaderComputeInputInfo& input_info,
	                                         bool&                         pending);
	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                              const DrawTargets& targets);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque, so pipelines being made on workers keep valid pointers to their modules.
		std::deque<ShaderModule>                     modules;
		bool                                         skip_dispatch   = false;
		bool                                         logged_neg_hit  = false;
		std::string                                  failure_reason;

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         ShaderRecompiler::IR::PushData::Window  push_data,
		                         bool*                                   materialized          = nullptr,
		                         std::string*                            materialization_error = nullptr);
		// The module of a program without calls that was built for these inputs, if any.
		ShaderModule* FindCandidate(const ShaderRecompiler::IR::ResourceSpecialization& wanted,
		                            ShaderRecompiler::IR::PushData::Window              push_data);
	};

	// Variants of one source share a bucket; ProgramKey equality compares the state.
	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	bool                          m_driver_cache_dirty = false;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;
	std::unordered_map<uint64_t, ShaderTessellationInputInfo>                  m_tess_layouts;
	// Shader hash -> distinct ProgramKeys observed for next-boot lookup (--shader-key-index).
	std::unordered_map<uint64_t, std::vector<ProgramKey>>                      m_key_index;
	// Records everything first created if a shader store is open, and writes the driver cache.
	ShaderStoreWriter m_store;
	// What the module lookups ended in, for the `[store] lookups=` line.
	StoreDiff::LookupStats m_lookup_stats;
	// Counts a lookup that found no module, and describes the first few.
	void NoteLookupMiss(const ProgramKey& key, const Program* program,
	                    ShaderRecompiler::IR::PushData::Window push_data, bool first);
	// Prints the lookup counters.
	void ReportLookups() const;
	// One line per built module or pipeline, and the sums for the status line.
	void NoteBuild(BuildTiming::Build build);
	BuildTiming::Totals                    m_build_totals;
	// The steps and SPIR-V size of the last CompileModule, for the render thread's build line.
	BuildTiming::Steps                     m_last_compile_steps;
	uint64_t                               m_last_compile_words = 0;
	std::unordered_map<uint64_t, uint32_t> m_built_count;
	// "served from store" uses, per hash.
	FlowPlan::UseCounter     m_served;
	// A dispatch that waits is retried; the lookup counters count the request once.
	FlowPlan::RequestCounter m_requests;
	FlowPlan::SkipLedger     m_skipped_dispatches;
	// The hash of the compute shader the last GetComputeProgram call left pending.
	uint64_t                 m_pending_compute_hash     = 0;
	uint64_t                 m_pending_compute_identity = 0;
	PrefetchCounters         m_prefetch;
	// Modules built from the scan's user data, by program key hash and module id, until a dispatch
	// uses them; what is left at the end is a wrong guess.
	std::unordered_set<uint64_t> m_lookahead_modules;

	// A stage built from its translation, before the cache holds it.
	struct CompiledStage {
		ShaderModule          module;
		std::vector<uint32_t> spirv;
		// What the SPIR-V emission and the module creation took.
		BuildTiming::Steps    steps;
	};
	// What a worker needs to translate and compile one stage, copied from the draw that asked.
	struct StageInputs;
	// The worker threads and what they have finished (pipelineAsync.cpp); null unless
	// --async-shader-compile is on.
	struct Async;
	// How much of a stage a GetProgram call may leave to the workers.
	enum class AsyncMode {
		Off,
		// Null while the stage is not ready.
		On,
		// Only start the translation of a program the cache has never seen; always null.
		PrefetchOnly,
	};
	std::unique_ptr<Async> m_async;

	// Null skips the draw or dispatch. With --keep-going any compile failure lands here.
	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               ShaderRecompiler::IR::PushData::Window& push_data,
	                               AsyncMode                               mode = AsyncMode::Off);
	template <typename InputInfo>
	const ShaderModule* GetProgramImpl(const ShaderParams& params, InputInfo& input_info,
	                                   ShaderRecompiler::IR::PushData::Window& push_data,
	                                   AsyncMode                               mode);
	void RecordShaderFailure(const ShaderParams& params, std::string_view reason, const char* file,
	                         int line);
	void NoteBisectSkippedShader(uint64_t hash, ShaderType stage);
	void RecordShaderFailure(ShaderType stage, uint64_t hash, std::span<const uint32_t> code,
	                         std::string_view reason, const char* file, int line);
	void NoteSkippedUse(uint64_t hash);
	bool                                   m_lookup_key_ready = false;
	bool                                   m_last_deferred    = false;
	// Allocation-free identity of m_lookup_key for the bringup ledger's variant count.
	[[nodiscard]] uint64_t                 ProgramVariantKey() const;
	std::unordered_map<uint64_t, uint64_t> m_skipped_uses; // By shader hash.
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions       options,
	                            ShaderRecompiler::TranslateResult      translated,
	                            ShaderRecompiler::IR::PushData::Window push_data);
	// The part of CompileModule that touches no cache state, so a worker can run it.
	static CompiledStage
	CompileStage(vk::Device device, const ProgramKey& key,
	             ShaderRecompiler::CompileOptions                    options,
	             ShaderRecompiler::TranslateResult                   translated,
	             const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	             ShaderRecompiler::IR::PushData::Window              push_data,
	             std::span<const uint32_t>                           function_code);
	ShaderModule& AddModule(Program& program, const ProgramKey& key, CompiledStage compiled);
	// The pipeline's key, and in `stages` the modules it is made of.
	GraphicsPipelineKey MakeGraphicsPipelineKey(const CommandBuffer&   command,
	                                            const DrawRenderState& state,
	                                            const DrawTargets&     targets,
	                                            std::array<const ShaderModule*, 4>& stages);

	// pipelineAsync.cpp
	// Starts whatever work gets a stage the cache cannot serve; the caller then returns null.
	template <typename InputInfo>
	void RequestAsync(const ProgramKey& key, Program* program, const ShaderParams& params,
	                  const InputInfo& input_info, const ShaderRecompiler::CompileOptions& options,
	                  ShaderRecompiler::IR::PushData::Window push_data, AsyncMode mode,
	                  bool urgent_translate = false, bool user_data_known = true);
	// The inputs a worker needs to translate and compile one stage, copied from the call.
	template <typename InputInfo>
	std::shared_ptr<StageInputs> MakeStageInputs(const ProgramKey& key, const ShaderParams& params,
	                                             const InputInfo&                        input_info,
	                                             const ShaderRecompiler::CompileOptions& options);
	// Compiles the module of a translated compute program with GPU-selected calls on a worker.
	// False when the workers are stopped.
	bool PostCallProgramModule(const ProgramKey& key, Program& program, const ShaderParams& params,
	                           const ShaderComputeInputInfo&           input_info,
	                           const ShaderRecompiler::CompileOptions& options,
	                           ShaderRecompiler::IR::PushData::Window  push_data,
	                           ShaderRecompiler::TranslateResult&&     translated);
	// Applies the work the workers have finished. Render thread only.
	void            PumpAsync();
	// Sleeps on a condition variable until a worker finishes, then applies finished work.
	void            WaitForAsync();
	static uint64_t FlightId(const ProgramKey& key,
	                         ShaderRecompiler::IR::PushData::Window push_data,
	                         const ShaderRecompiler::IR::ResourceSpecialization* specialization);
	void InitializeDriverCache();
	// Marks the driver cache changed and, once per interval, has the store thread write it.
	void CheckpointDriverCache();
	void WriteDriverCache() const;

	// pipelineSerialization.cpp
	enum class LoadRecordStatus : uint8_t { Loaded, EndOfStream, Failed };
	void             WarmUp(const std::filesystem::path& path);
	LoadRecordStatus LoadRecord(std::span<const uint8_t>& records, bool keep_on_error);
	void StoreModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv);
	void StoreFailedProgram(const ProgramKey& key, std::string_view reason);
	bool RecordIndexedKey(const ProgramKey& key, bool persist);
	void StoreKeyIndex(const ProgramKey& key);
	void StorePipeline(const GraphicsPipelineKey& key);
	void StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	// Every module by id, with id 0 naming an absent stage.
	[[nodiscard]] ModuleMap MapModules() const;
	// False when a stage id of the key is not in `modules`.
	bool CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                            const ModuleMap& modules);
};

// The shader store of the running title; empty unless --shader-cache is on in a Release build
// without debug dumps, which change the SPIR-V.
std::filesystem::path ShaderStorePath();
// Host capabilities that change the emitted SPIR-V.
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);

// Stages are indexed like GraphicsPipelineKey::stage_ids.
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
