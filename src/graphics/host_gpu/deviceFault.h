#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
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
	bool       uses_dma  = false;
};

struct OpBufferBreadcrumb {
	uint64_t                   guest_va         = 0;
	uint64_t                   vk_buffer        = 0;
	uint64_t                   gpu_va           = 0;
	uint64_t                   offset           = 0;
	uint64_t                   range            = 0;
	uint64_t                   buffer_size      = 0;
	uint16_t                   binding          = 0;
	uint16_t                   array_element    = 0;
	uint8_t                    set              = 0;
	uint8_t                    is_storage_array = 0;
	uint8_t                    word_count       = 0;
	std::array<uint32_t, 16>   words {};
};

struct OpBreadcrumb {
	static constexpr size_t kMaxBuffers = 32;

	uint32_t                                    marker_id        = 0;
	BreadcrumbOpKind                            kind             = BreadcrumbOpKind::Unknown;
	uint32_t                                    dim_x            = 0;
	uint32_t                                    dim_y            = 0;
	uint32_t                                    dim_z            = 0;
	uint8_t                                     stage_count      = 0;
	uint8_t                                     buffer_count     = 0;
	uint16_t                                    total_buffers    = 0;
	uint8_t                                     push_dword_count = 0;
	uint8_t                                     user_data_count  = 0;
	std::array<ShaderStageBreadcrumb, 4>        stages {};
	std::array<OpBufferBreadcrumb, kMaxBuffers> buffers {};
	std::array<uint32_t, 32>                    push_dwords {};
	std::array<uint32_t, 16>                    user_data_dwords {};
};

// Tracks per-submit draw/dispatch breadcrumbs on the CPU alongside a small host-visible GPU
// marker buffer so VK_ERROR_DEVICE_LOST can identify the in-flight submit, suspect operation,
// and shader hashes without storing guest paths or shader binaries.
class GpuBreadcrumbTracker {
public:
	GpuBreadcrumbTracker() = default;
	~GpuBreadcrumbTracker();
	KYTY_CLASS_NO_COPY(GpuBreadcrumbTracker);

	void Init(GraphicContext& graphics);
	// True when draws and dispatches record breadcrumbs and driver markers (see Init for why).
	[[nodiscard]] bool Active() const noexcept { return m_active.load(std::memory_order_relaxed); }
	void Shutdown(GraphicContext& graphics);

	struct ShaderHashSnapshot {
		uint32_t                           last_op_id   = 0;
		uint32_t                           submit_count = 0;
		bool                               truncated    = false;
		bool                               dma_filtered = false;
		std::vector<ShaderStageBreadcrumb> shaders;
	};

	void     BeginSubmit(GraphicContext& graphics, vk::CommandBuffer cmd, uint64_t frame_num);
	uint32_t BeginOp(GraphicContext& graphics, vk::CommandBuffer cmd, const OpBreadcrumb& op);
	void     EndOp(GraphicContext& graphics, vk::CommandBuffer cmd, uint32_t marker_id);
	void     EndSubmitRecording(GraphicContext& graphics, vk::CommandBuffer cmd);
	void     MarkSubmitted(uint64_t tick, uint64_t known_gpu_tick);
	[[nodiscard]] ShaderHashSnapshot CollectShaderHashesSince(uint32_t after_op_id,
	                                                          size_t   max_shaders = 64);

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

	std::atomic<bool>                  m_active {false};
	std::mutex                         m_mutex;
	std::array<SubmitRecord, kMaxSubmits> m_submits {};
	size_t                             m_active_slot       = kMaxSubmits;
	uint32_t                           m_next_submit_seq   = 0;
	uint32_t                           m_next_op_id        = 0;
	uint64_t                           m_last_known_tick   = 0;
	vk::Buffer                         m_marker_buffer     = nullptr;
	VmaAllocation                      m_marker_allocation = nullptr;
	volatile uint32_t*                 m_marker_mapped     = nullptr;
	bool                               m_emit_debug_labels        = false;
	bool                               m_device_fault_ext_enabled = false;
	VkDevice                           m_vk_device                = VK_NULL_HANDLE;
	PFN_vkGetDeviceFaultInfoEXT        m_vk_get_device_fault_info = nullptr;
	std::atomic_flag                   m_reported                 = ATOMIC_FLAG_INIT;
	std::atomic<bool>                  m_fault_shutdown {false};
	std::atomic<bool>                  m_fault_request {false};
	std::atomic<bool>                  m_fault_done {false};
	std::atomic<bool>                  m_report_finished {false};
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	int m_wake_pipe[2] = {-1, -1};
#else
	std::mutex              m_fault_mutex;
	std::condition_variable m_fault_cv;
#endif
	std::thread             m_fault_thread;
	std::array<char, 8192>  m_report_buf {};

	void WaitForDriverHangDump();
	void FaultWorkerLoop();
	void WakeFaultWorker();
};

void ReportDeviceLost(GraphicContext& graphics, const char* operation, vk::Result result,
                      uint64_t tick = 0, uint64_t known_gpu_tick = 0);
void RequireVulkanSuccess(GraphicContext& graphics, vk::Result result, const char* operation);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_
