#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct GraphicContext;
enum class ShaderType;

enum class BreadcrumbOpKind : uint8_t {
	Unknown,
	Draw,
	DrawIndexed,
	DrawIndirect,
	DrawIndexedIndirect,
	DrawMesh,
	DrawMeshIndirect,
	Dispatch,
	DispatchIndirect,
};

struct ShaderStageBreadcrumb {
	ShaderType stage     = {};
	uint64_t   hash      = 0;
	uint64_t   module_id = 0;
};

struct OpBreadcrumb {
	uint32_t                             marker_id   = 0;
	BreadcrumbOpKind                     kind        = BreadcrumbOpKind::Unknown;
	uint32_t                             dim_x       = 0;
	uint32_t                             dim_y       = 0;
	uint32_t                             dim_z       = 0;
	uint8_t                              stage_count = 0;
	std::array<ShaderStageBreadcrumb, 4> stages {};
};

// Tracks per-submit draw/dispatch breadcrumbs on the CPU alongside a small host-visible GPU
// marker buffer so VK_ERROR_DEVICE_LOST can identify the in-flight submit, suspect operation,
// and shader hashes without storing guest paths or shader binaries.
class GpuBreadcrumbTracker {
public:
	GpuBreadcrumbTracker() = default;
	~GpuBreadcrumbTracker() = default;
	KYTY_CLASS_NO_COPY(GpuBreadcrumbTracker);

	void Init(GraphicContext& graphics);
	void Shutdown(GraphicContext& graphics);

	void     BeginSubmit(GraphicContext& graphics, vk::CommandBuffer cmd, uint64_t frame_num);
	uint32_t BeginOp(GraphicContext& graphics, vk::CommandBuffer cmd, OpBreadcrumb op);
	void     EndOp(GraphicContext& graphics, vk::CommandBuffer cmd, uint32_t marker_id);
	void     EndSubmitRecording(GraphicContext& graphics, vk::CommandBuffer cmd);
	void     MarkSubmitted(uint64_t tick, uint64_t known_gpu_tick);

	void ReportDeviceLost(GraphicContext& graphics, const char* operation, vk::Result result,
	                      uint64_t tick = 0, uint64_t known_gpu_tick = 0);

private:
	static constexpr size_t kMaxSubmits      = 16;
	static constexpr size_t kMaxOpsPerSubmit = 512;

	enum class SubmitState : uint8_t {
		Empty,
		Recording,
		Submitted,
		Completed,
	};

	struct SubmitRecord {
		uint32_t                  seq         = 0;
		uint64_t                  tick        = 0;
		uint64_t                  frame_num   = 0;
		uint32_t                  total_ops   = 0;
		uint32_t                  first_op_id = 0;
		uint32_t                  last_op_id  = 0;
		SubmitState               state       = SubmitState::Empty;
		std::vector<OpBreadcrumb> ops;
	};

	std::mutex                         m_mutex;
	std::array<SubmitRecord, kMaxSubmits> m_submits {};
	size_t                             m_active_slot       = kMaxSubmits;
	uint32_t                           m_next_submit_seq   = 0;
	uint32_t                           m_next_op_id        = 0;
	uint64_t                           m_last_known_tick   = 0;
	vk::Buffer                         m_marker_buffer     = nullptr;
	VmaAllocation                      m_marker_allocation = nullptr;
	volatile uint32_t*                 m_marker_mapped     = nullptr;
	bool                               m_emit_debug_labels = false;
	std::atomic_flag                   m_reported          = ATOMIC_FLAG_INIT;
};

void ReportDeviceLost(GraphicContext& graphics, const char* operation, vk::Result result,
                      uint64_t tick = 0, uint64_t known_gpu_tick = 0);
void RequireVulkanSuccess(GraphicContext& graphics, vk::Result result, const char* operation);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_
