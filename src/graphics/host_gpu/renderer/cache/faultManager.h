#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/deviceFault.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>
#include <set>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class BufferCache;

class FaultManager {
	static constexpr size_t MaxPendingFaults = 8;

public:
	FaultManager(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~FaultManager();
	KYTY_CLASS_NO_COPY(FaultManager);

	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return &m_fault_buffer; }
	void                  ProcessFaultBuffer();
	[[nodiscard]] static std::vector<std::string> FormatNewBdaMisses(
	    std::span<const uint32_t>                       miss_words,
	    const GpuBreadcrumbTracker::ShaderHashSnapshot& shaders,
	    std::unordered_set<uint64_t>&                   seen_pages);

private:
	GraphicContext&                            m_graphics;
	CommandScheduler&                          m_scheduler;
	BufferCache&                               m_buffer_cache;
	Buffer                                     m_fault_buffer;
	Buffer                                     m_download_buffer;
	std::array<uint64_t, MaxPendingFaults>      m_fault_areas {};
	uint32_t                                   m_current_area          = 0;
	uint32_t                                   m_last_breadcrumb_op_id = 0;
	std::unordered_set<uint64_t>               m_seen_bda_miss_pages;
	std::set<std::pair<uint64_t, uint32_t>>    m_seen_loop_guard_trips;
	vk::DescriptorSetLayout                    m_fault_process_desc_layout = nullptr;
	vk::Pipeline                               m_fault_process_pipeline = nullptr;
	vk::PipelineLayout                         m_fault_process_pipeline_layout = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
