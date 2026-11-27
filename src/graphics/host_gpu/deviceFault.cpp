#include "graphics/host_gpu/deviceFault.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/frameCost.h"
#include "common/hangDump.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {

namespace {

constexpr vk::DeviceSize kMarkerBufferSize     = 64;
constexpr vk::DeviceSize kOffsetSubmitStarted  = 0;
constexpr vk::DeviceSize kOffsetSubmitFinished = 4;
constexpr vk::DeviceSize kOffsetOpStarted      = 8;
constexpr vk::DeviceSize kOffsetOpFinished     = 12;

constexpr uint32_t kMaxFaultRecords = 64;

void WriteRawStderr(const char* data, size_t len) {
	while (len > 0) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const int written = ::_write(2, data, static_cast<unsigned int>(len));
#else
		const ssize_t written = ::write(STDERR_FILENO, data, len);
#endif
		if (written <= 0) {
			break;
		}
		data += written;
		len -= static_cast<size_t>(written);
	}
}

// Streaming fixed-buffer formatter: zero heap allocations, flushes directly to stderr.
class FixedStderrSink {
public:
	explicit FixedStderrSink(std::array<char, 8192>& buf): m_buf(buf) {}
	~FixedStderrSink() { Flush(); }

	void Flush() {
		if (m_pos > 0) {
			WriteRawStderr(m_buf.data(), m_pos);
			m_pos = 0;
		}
	}

	void Append(std::string_view sv) {
		while (!sv.empty()) {
			if (m_pos == m_buf.size()) {
				Flush();
			}
			const size_t chunk = std::min(sv.size(), m_buf.size() - m_pos);
			std::memcpy(m_buf.data() + m_pos, sv.data(), chunk);
			m_pos += chunk;
			sv.remove_prefix(chunk);
		}
	}

	template <typename... Args>
	void Printf(fmt::format_string<Args...> fmt_str, Args&&... args) {
		std::array<char, 1024> tmp;
		const auto res = fmt::format_to_n(tmp.data(), tmp.size(), fmt_str,
		                                  std::forward<Args>(args)...);
		const size_t len = std::min(res.size, tmp.size());
		Append({tmp.data(), len});
	}

private:
	std::array<char, 8192>& m_buf;
	size_t                  m_pos = 0;
};

[[nodiscard]] const char* OpKindName(BreadcrumbOpKind kind) {
	switch (kind) {
		case BreadcrumbOpKind::Draw: return "Draw";
		case BreadcrumbOpKind::DrawIndexed: return "DrawIndexed";
		case BreadcrumbOpKind::DrawIndirect: return "DrawIndirect";
		case BreadcrumbOpKind::DrawIndexedIndirect: return "DrawIndexedIndirect";
		case BreadcrumbOpKind::DrawMesh: return "DrawMesh";
		case BreadcrumbOpKind::DrawMeshIndirect: return "DrawMeshIndirect";
		case BreadcrumbOpKind::Dispatch: return "Dispatch";
		case BreadcrumbOpKind::DispatchIndirect: return "DispatchIndirect";
		case BreadcrumbOpKind::Unknown:
		default: return "Unknown";
	}
}

[[nodiscard]] const char* FaultAddressTypeName(VkDeviceFaultAddressTypeEXT type) {
	switch (type) {
		case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "None";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "ReadInvalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "WriteInvalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "ExecuteInvalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT:
			return "InstructionPointerUnknown";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT:
			return "InstructionPointerInvalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:
			return "InstructionPointerFault";
		default: return "Unknown";
	}
}

[[nodiscard]] std::string_view SafeDescription(const char* data, size_t max_len) {
	if (data == nullptr || max_len == 0) {
		return {};
	}
	size_t len = 0;
	while (len < max_len && data[len] != '\0') {
		++len;
	}
	return {data, len};
}

[[nodiscard]] bool RadvDebugEnabled() {
	const char* env = std::getenv("RADV_DEBUG");
	return env != nullptr && env[0] != '\0';
}

// The driver's hang debugging (RADV_DEBUG=hang, also inside a list) reads our markers.
[[nodiscard]] bool RadvHangDebugEnabled() {
	const char* env = std::getenv("RADV_DEBUG");
	return env != nullptr && std::strstr(env, "hang") != nullptr;
}

void FormatOpStages(FixedStderrSink& out, const OpBreadcrumb& op) {
	for (uint8_t i = 0; i < op.stage_count && i < op.stages.size(); ++i) {
		if (i != 0) {
			out.Append(" ");
		}
		out.Printf("{}=0x{:016x}", ShaderRecompiler::StageName(op.stages[i].stage),
		           op.stages[i].hash);
	}
}

void FormatOpLine(FixedStderrSink& out, const OpBreadcrumb& op) {
	out.Append(OpKindName(op.kind));
	out.Append(" ");
	switch (op.kind) {
		case BreadcrumbOpKind::Dispatch:
			out.Printf("{}x{}x{}", op.dim_x, op.dim_y, op.dim_z);
			break;
		case BreadcrumbOpKind::DispatchIndirect: out.Append("indirect"); break;
		case BreadcrumbOpKind::DrawMesh:
			out.Printf("groups={}x{}", op.dim_x, op.dim_y);
			break;
		case BreadcrumbOpKind::DrawMeshIndirect:
			out.Printf("indirect_count={}", op.dim_x);
			break;
		case BreadcrumbOpKind::DrawIndirect:
		case BreadcrumbOpKind::DrawIndexedIndirect:
			out.Printf("count={} draws={} stride={}", op.dim_x, op.dim_y, op.dim_z);
			break;
		default: out.Printf("count={} instances={}", op.dim_x, op.dim_y); break;
	}
	out.Append(" ");
	FormatOpStages(out, op);
}

[[nodiscard]] std::string FormatOpLineString(const OpBreadcrumb& op) {
	std::string out = OpKindName(op.kind);
	out += ' ';
	switch (op.kind) {
		case BreadcrumbOpKind::Dispatch:
			out += fmt::format("{}x{}x{}", op.dim_x, op.dim_y, op.dim_z);
			break;
		case BreadcrumbOpKind::DispatchIndirect: out += "indirect"; break;
		case BreadcrumbOpKind::DrawMesh:
			out += fmt::format("groups={}x{}", op.dim_x, op.dim_y);
			break;
		case BreadcrumbOpKind::DrawMeshIndirect:
			out += fmt::format("indirect_count={}", op.dim_x);
			break;
		case BreadcrumbOpKind::DrawIndirect:
		case BreadcrumbOpKind::DrawIndexedIndirect:
			out += fmt::format("count={} draws={} stride={}", op.dim_x, op.dim_y, op.dim_z);
			break;
		default: out += fmt::format("count={} instances={}", op.dim_x, op.dim_y); break;
	}
	for (uint8_t i = 0; i < op.stage_count && i < op.stages.size(); ++i) {
		out += fmt::format(" {}=0x{:016x}", ShaderRecompiler::StageName(op.stages[i].stage),
		                   op.stages[i].hash);
	}
	return out;
}

void QueryDeviceFaultExt(VkDevice device, bool enabled, PFN_vkGetDeviceFaultInfoEXT pfn) {
	std::array<char, 8192> buf {};
	FixedStderrSink        out(buf);
	out.Append("=== GPU DEVICE FAULT INFO ===\n");
	if (!enabled || device == VK_NULL_HANDLE || pfn == nullptr) {
		out.Printf("VK_EXT_device_fault: unavailable (enabled={} pfn={})\n",
		           enabled ? "true" : "false", pfn != nullptr ? "present" : "null");
		out.Append("=== END GPU DEVICE FAULT INFO ===\n");
		out.Flush();
		return;
	}

	VkDeviceFaultCountsEXT counts {};
	counts.sType       = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
	counts.pNext       = nullptr;
	const auto count_result = static_cast<vk::Result>(pfn(device, &counts, nullptr));
	if (count_result != vk::Result::eSuccess && count_result != vk::Result::eIncomplete) {
		out.Printf("VK_EXT_device_fault: count query failed ({})\n", vk::to_string(count_result));
		out.Append("=== END GPU DEVICE FAULT INFO ===\n");
		out.Flush();
		return;
	}

	const uint64_t vendor_binary_bytes = counts.vendorBinarySize;
	const uint32_t address_count       = std::min(counts.addressInfoCount, kMaxFaultRecords);
	const uint32_t vendor_count        = std::min(counts.vendorInfoCount, kMaxFaultRecords);

	std::array<VkDeviceFaultAddressInfoEXT, kMaxFaultRecords> addresses {};
	std::array<VkDeviceFaultVendorInfoEXT, kMaxFaultRecords>  vendors {};

	counts.addressInfoCount = address_count;
	counts.vendorInfoCount  = vendor_count;
	// Do not copy raw vendor binary dumps into the text log.
	counts.vendorBinarySize = 0;

	VkDeviceFaultInfoEXT info {};
	info.sType             = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
	info.pNext             = nullptr;
	info.pAddressInfos     = address_count != 0 ? addresses.data() : nullptr;
	info.pVendorInfos      = vendor_count != 0 ? vendors.data() : nullptr;
	info.pVendorBinaryData = nullptr;

	const auto info_result = static_cast<vk::Result>(pfn(device, &counts, &info));
	if (info_result != vk::Result::eSuccess && info_result != vk::Result::eIncomplete) {
		out.Printf("VK_EXT_device_fault: info query failed ({})\n", vk::to_string(info_result));
		out.Append("=== END GPU DEVICE FAULT INFO ===\n");
		out.Flush();
		return;
	}

	const auto desc = SafeDescription(info.description, sizeof(info.description));
	out.Printf("VK_EXT_device_fault: description=\"{}\" addresses={} vendor_infos={} "
	           "vendor_binary_bytes={}\n",
	           desc, counts.addressInfoCount, counts.vendorInfoCount, vendor_binary_bytes);

	const uint32_t reported_addresses = std::min(counts.addressInfoCount, address_count);
	if (counts.addressInfoCount == 0) {
		out.Append("  fault_addr: none (no per-process GPUVM fault reported by driver)\n");
	}
	for (uint32_t i = 0; i < reported_addresses; ++i) {
		const auto& addr = addresses[i];
		out.Printf("  fault_addr[{}]: type={} ({}) gpu_va=0x{:016x} precision=0x{:x}\n", i,
		           FaultAddressTypeName(addr.addressType), static_cast<int>(addr.addressType),
		           addr.reportedAddress, addr.addressPrecision);
	}

	const uint32_t reported_vendors = std::min(counts.vendorInfoCount, vendor_count);
	for (uint32_t i = 0; i < reported_vendors; ++i) {
		const auto& v     = vendors[i];
		const auto  vdesc = SafeDescription(v.description, sizeof(v.description));
		out.Printf("  fault_vendor[{}]: code=0x{:016x} data=0x{:016x} desc=\"{}\"\n", i,
		           v.vendorFaultCode, v.vendorFaultData, vdesc);
	}
	out.Append("=== END GPU DEVICE FAULT INFO ===\n");
	out.Flush();
}

} // namespace

GpuBreadcrumbTracker::~GpuBreadcrumbTracker() {
	m_fault_shutdown.store(true, std::memory_order_release);
	WakeFaultWorker();
	if (m_fault_thread.joinable()) {
		m_fault_thread.join();
	}
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	if (m_wake_pipe[0] >= 0) {
		::close(m_wake_pipe[0]);
		m_wake_pipe[0] = -1;
	}
	if (m_wake_pipe[1] >= 0) {
		::close(m_wake_pipe[1]);
		m_wake_pipe[1] = -1;
	}
#endif
}

void GpuBreadcrumbTracker::WakeFaultWorker() {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	if (m_wake_pipe[1] >= 0) {
		const uint8_t b = 1;
		(void)::write(m_wake_pipe[1], &b, 1);
	}
#else
	m_fault_cv.notify_one();
#endif
}

void GpuBreadcrumbTracker::FaultWorkerLoop() {
	for (;;) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		uint8_t b = 0;
		if (m_wake_pipe[0] >= 0) {
			(void)::read(m_wake_pipe[0], &b, 1);
		}
#else
		{
			std::unique_lock lk(m_fault_mutex);
			m_fault_cv.wait(lk, [this] {
				return m_fault_shutdown.load(std::memory_order_acquire) ||
				       m_fault_request.load(std::memory_order_acquire);
			});
		}
#endif
		if (m_fault_shutdown.load(std::memory_order_acquire)) {
			break;
		}
		if (m_fault_request.exchange(false, std::memory_order_acq_rel)) {
			QueryDeviceFaultExt(m_vk_device, m_device_fault_ext_enabled,
			                    m_vk_get_device_fault_info);
			m_fault_done.store(true, std::memory_order_release);
		}
	}
}

void GpuBreadcrumbTracker::Init(GraphicContext& graphics) {
	std::lock_guard lock(m_mutex);
	{
		const char* reason = nullptr;
		if (Config::GpuDebugEnabled()) {
			reason = "--gpu-debug";
		} else if (RadvHangDebugEnabled()) {
			reason = "RADV_DEBUG contains hang";
		} else if (Config::GpuFaultQueryEnabled()) {
			reason = "--gpu-fault-query";
		} else if (Config::GraphicsDebugDumpEnabled()) {
			reason = "--graphics-debug-dump";
		} else if (Config::GuestFaultReportOnlyEnabled()) {
			reason = "--guest-fault-report-only";
		}
		m_active.store(reason != nullptr, std::memory_order_relaxed);
		Log::WriteToConsoleAndLog(fmt::format(
		    "[gpu] breadcrumbs and driver markers per draw and dispatch: {}\n",
		    reason != nullptr ? fmt::format("on ({})", reason)
		                      : std::string("off (--gpu-debug turns them on for device-lost reports)")));
	}
	m_emit_debug_labels =
	    graphics.debug_utils_ext_enabled &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdInsertDebugUtilsLabelEXT != nullptr &&
	    (Config::GraphicsDebugDumpEnabled() || RadvDebugEnabled());

	m_vk_device                = static_cast<VkDevice>(graphics.device);
	m_device_fault_ext_enabled = graphics.device_fault_ext_enabled;
	if (m_device_fault_ext_enabled && m_vk_device != VK_NULL_HANDLE &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr != nullptr) {
		m_vk_get_device_fault_info = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(m_vk_device,
		                                                      "vkGetDeviceFaultInfoEXT"));
	} else {
		m_vk_get_device_fault_info = nullptr;
	}

	if (!m_fault_thread.joinable()) {
		m_fault_shutdown.store(false, std::memory_order_release);
		m_fault_request.store(false, std::memory_order_release);
		m_fault_done.store(false, std::memory_order_release);
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		if (m_wake_pipe[0] < 0 && ::pipe(m_wake_pipe) != 0) {
			m_wake_pipe[0] = -1;
			m_wake_pipe[1] = -1;
		}
		if (m_wake_pipe[0] >= 0) {
			m_fault_thread = std::thread(&GpuBreadcrumbTracker::FaultWorkerLoop, this);
		}
#else
		m_fault_thread = std::thread(&GpuBreadcrumbTracker::FaultWorkerLoop, this);
#endif
	}

	if (graphics.allocator == nullptr || m_marker_buffer != nullptr) {
		return;
	}

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = kMarkerBufferSize;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags =
	    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	alloc_info.usage          = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo alloc_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	const auto        result        = static_cast<vk::Result>(vmaCreateBuffer(
        graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &alloc_info,
        &native_buffer, &m_marker_allocation, &alloc_result));
	if (result != vk::Result::eSuccess || alloc_result.pMappedData == nullptr) {
		if (native_buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(graphics.allocator, native_buffer, m_marker_allocation);
		}
		m_marker_buffer     = nullptr;
		m_marker_allocation = nullptr;
		m_marker_mapped     = nullptr;
		LOGF("GpuBreadcrumbTracker: marker buffer allocation failed (%s); using CPU ring only\n",
		     vk::to_string(result).c_str());
		return;
	}

	m_marker_buffer = native_buffer;
	m_marker_mapped = static_cast<volatile uint32_t*>(alloc_result.pMappedData);
	std::memset(alloc_result.pMappedData, 0, static_cast<size_t>(kMarkerBufferSize));
}

void GpuBreadcrumbTracker::Shutdown(GraphicContext& graphics) {
	m_fault_shutdown.store(true, std::memory_order_release);
	WakeFaultWorker();
	if (m_fault_thread.joinable()) {
		m_fault_thread.join();
	}
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	if (m_wake_pipe[0] >= 0) {
		::close(m_wake_pipe[0]);
		m_wake_pipe[0] = -1;
	}
	if (m_wake_pipe[1] >= 0) {
		::close(m_wake_pipe[1]);
		m_wake_pipe[1] = -1;
	}
#endif
	std::lock_guard lock(m_mutex);
	if (m_marker_buffer != nullptr && graphics.allocator != nullptr) {
		vmaDestroyBuffer(graphics.allocator, m_marker_buffer, m_marker_allocation);
	}
	m_marker_buffer            = nullptr;
	m_marker_allocation        = nullptr;
	m_marker_mapped            = nullptr;
	m_vk_device                = VK_NULL_HANDLE;
	m_device_fault_ext_enabled = false;
	m_vk_get_device_fault_info = nullptr;
}

void GpuBreadcrumbTracker::BeginSubmit(GraphicContext& graphics, vk::CommandBuffer cmd,
                                       uint64_t frame_num) {
	if (!Active()) {
		return;
	}
	uint32_t   seq           = 0;
	vk::Buffer marker_buffer = nullptr;
	{
		std::lock_guard lock(m_mutex);
		seq = ++m_next_submit_seq;
		if (seq == 0) {
			seq = ++m_next_submit_seq;
		}
		const size_t slot = static_cast<size_t>(seq - 1u) % kMaxSubmits;
		m_active_slot     = slot;
		auto& rec         = m_submits[slot];
		rec.seq           = seq;
		rec.tick          = 0;
		rec.frame_num     = frame_num;
		rec.total_ops     = 0;
		rec.first_op_id   = 0;
		rec.last_op_id    = 0;
		rec.state         = SubmitState::Recording;
		rec.ops.clear();
		marker_buffer = m_marker_buffer;
	}

	if (marker_buffer != nullptr && cmd != nullptr) {
		if (graphics.buffer_marker_amd_enabled &&
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD != nullptr) {
			cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, marker_buffer,
			                         kOffsetSubmitStarted, seq);
		} else {
			cmd.fillBuffer(marker_buffer, kOffsetSubmitStarted, sizeof(uint32_t), seq);
		}
	}
}

uint32_t GpuBreadcrumbTracker::BeginOp(GraphicContext& graphics, vk::CommandBuffer cmd,
                                       const OpBreadcrumb& op_in) {
	if (!Active()) {
		return 0;
	}
	Common::FrameCost::Add(Common::FrameCost::Counter::BreadcrumbOps);
	OpBreadcrumb op = op_in;
	uint32_t   submit_seq    = 0;
	vk::Buffer marker_buffer = nullptr;
	bool       emit_label    = false;
	{
		std::lock_guard lock(m_mutex);
		op.marker_id = ++m_next_op_id;
		if (op.marker_id == 0) {
			op.marker_id = ++m_next_op_id;
		}
		if (m_active_slot < kMaxSubmits) {
			auto& rec  = m_submits[m_active_slot];
			submit_seq = rec.seq;
			if (rec.total_ops == 0) {
				rec.first_op_id = op.marker_id;
			}
			rec.last_op_id = op.marker_id;
			if (rec.ops.size() < kMaxOpsPerSubmit) {
				rec.ops.push_back(op);
			} else {
				rec.ops[rec.total_ops % kMaxOpsPerSubmit] = op;
			}
			++rec.total_ops;
		}
		marker_buffer = m_marker_buffer;
		emit_label    = m_emit_debug_labels;
	}

	if (cmd != nullptr) {
		if (marker_buffer != nullptr && graphics.buffer_marker_amd_enabled &&
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD != nullptr) {
			cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, marker_buffer,
			                         kOffsetOpStarted, op.marker_id);
		}
		if (emit_label) {
			const auto label_text =
			    fmt::format("submit={} op={} {}", submit_seq, op.marker_id, FormatOpLineString(op));
			vk::DebugUtilsLabelEXT label {};
			label.pLabelName = label_text.c_str();
			cmd.insertDebugUtilsLabelEXT(&label);
		}
	}

	return op.marker_id;
}

void GpuBreadcrumbTracker::EndOp(GraphicContext& graphics, vk::CommandBuffer cmd,
                                 uint32_t marker_id) {
	if (marker_id == 0 || cmd == nullptr || !graphics.buffer_marker_amd_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD == nullptr) {
		return;
	}
	vk::Buffer marker_buffer = nullptr;
	{
		std::lock_guard lock(m_mutex);
		marker_buffer = m_marker_buffer;
	}
	if (marker_buffer != nullptr) {
		cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, marker_buffer,
		                         kOffsetOpFinished, marker_id);
	}
}

void GpuBreadcrumbTracker::EndSubmitRecording(GraphicContext& graphics, vk::CommandBuffer cmd) {
	if (!Active()) {
		return;
	}
	uint32_t   seq           = 0;
	vk::Buffer marker_buffer = nullptr;
	{
		std::lock_guard lock(m_mutex);
		if (m_active_slot < kMaxSubmits) {
			seq = m_submits[m_active_slot].seq;
		}
		marker_buffer = m_marker_buffer;
	}

	if (seq != 0 && marker_buffer != nullptr && cmd != nullptr) {
		if (graphics.buffer_marker_amd_enabled &&
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD != nullptr) {
			cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, marker_buffer,
			                         kOffsetSubmitFinished, seq);
		} else {
			cmd.fillBuffer(marker_buffer, kOffsetSubmitFinished, sizeof(uint32_t), seq);
		}
	}
}

void GpuBreadcrumbTracker::MarkSubmitted(uint64_t tick, uint64_t known_gpu_tick) {
	std::lock_guard lock(m_mutex);
	m_last_known_tick = std::max(m_last_known_tick, known_gpu_tick);
	if (m_active_slot < kMaxSubmits) {
		auto& rec = m_submits[m_active_slot];
		rec.tick  = tick;
		rec.state = SubmitState::Submitted;
	}
	for (auto& rec: m_submits) {
		if (rec.state == SubmitState::Submitted && rec.tick != 0 && rec.tick <= m_last_known_tick) {
			rec.state = SubmitState::Completed;
		}
	}
}

GpuBreadcrumbTracker::ShaderHashSnapshot
GpuBreadcrumbTracker::CollectShaderHashesSince(uint32_t after_op_id, size_t max_shaders) {
	ShaderHashSnapshot snapshot {};
	std::lock_guard    lock(m_mutex);
	snapshot.last_op_id = m_next_op_id;
	if (after_op_id == m_next_op_id) {
		return snapshot;
	}

	std::array<size_t, kMaxSubmits> order {};
	size_t                          order_count = 0;
	for (size_t i = 0; i < kMaxSubmits; ++i) {
		const auto& rec = m_submits[i];
		if (rec.state != SubmitState::Empty && rec.total_ops != 0 && rec.last_op_id > after_op_id) {
			order[order_count++] = i;
		}
	}
	if (order_count == 0) {
		snapshot.truncated = true;
		return snapshot;
	}
	std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(order_count),
	          [&](size_t a, size_t b) { return m_submits[a].seq < m_submits[b].seq; });

	if (m_submits[order[0]].first_op_id > after_op_id + 1u) {
		snapshot.truncated = true;
	}

	std::vector<ShaderStageBreadcrumb> dma_shaders;
	std::vector<ShaderStageBreadcrumb> all_shaders;
	bool                               dma_overflow = false;
	bool                               all_overflow = false;
	const auto add_unique = [&](std::vector<ShaderStageBreadcrumb>& out, bool& overflow,
	                            const ShaderStageBreadcrumb& stage) {
		for (const auto& existing: out) {
			if (existing.stage == stage.stage && existing.hash == stage.hash) {
				return;
			}
		}
		if (out.size() >= max_shaders) {
			overflow = true;
			return;
		}
		out.push_back(stage);
	};

	for (size_t idx = 0; idx < order_count; ++idx) {
		const auto& rec = m_submits[order[idx]];
		++snapshot.submit_count;
		if (rec.total_ops > kMaxOpsPerSubmit) {
			snapshot.truncated = true;
		}
		const size_t op_count   = rec.ops.size();
		const size_t oldest_idx = rec.total_ops <= kMaxOpsPerSubmit
		                              ? 0u
		                              : static_cast<size_t>(rec.total_ops % kMaxOpsPerSubmit);
		for (size_t i = 0; i < op_count; ++i) {
			const auto& op = rec.ops[(oldest_idx + i) % op_count];
			if (op.marker_id <= after_op_id) {
				continue;
			}
			for (uint8_t si = 0; si < op.stage_count && si < op.stages.size(); ++si) {
				const auto& stage = op.stages[si];
				if (stage.hash == 0) {
					continue;
				}
				if (stage.uses_dma) {
					add_unique(dma_shaders, dma_overflow, stage);
				}
				add_unique(all_shaders, all_overflow, stage);
			}
		}
	}

	if (!dma_shaders.empty()) {
		snapshot.dma_filtered = true;
		snapshot.truncated |= dma_overflow;
		snapshot.shaders = std::move(dma_shaders);
	} else {
		snapshot.dma_filtered = false;
		snapshot.truncated |= all_overflow;
		snapshot.shaders = std::move(all_shaders);
	}
	return snapshot;
}

void GpuBreadcrumbTracker::ReportDeviceLost(GraphicContext& graphics, const char* operation,
                                            vk::Result result, uint64_t tick,
                                            uint64_t known_gpu_tick) {
	if (m_reported.test_and_set(std::memory_order_acq_rel)) {
		// Another thread owns the report and may be waiting for the driver's hang dump. The
		// caller usually ends the process next; hold it back until the first thread is done.
		if (Common::HangDump::DriverHangModeActive(std::getenv("RADV_DEBUG"))) {
			const auto limit = std::chrono::seconds(Config::HangDumpWaitSeconds() + 10u);
			const auto start = std::chrono::steady_clock::now();
			while (!m_report_finished.load(std::memory_order_acquire) &&
			       std::chrono::steady_clock::now() - start < limit) {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		}
		return;
	}

	FixedStderrSink out(m_report_buf);
	out.Append("=== GPU DEVICE LOST TRIAGE ===\n");

	const auto& props = graphics.physical_device_properties;
	out.Printf("operation={} result={} tick={} known_gpu_tick={}\n",
	           operation != nullptr ? operation : "?", vk::to_string(result), tick,
	           known_gpu_tick);
	out.Printf("device=\"{}\" vendor=0x{:04x} device_id=0x{:04x} driver=0x{:08x}\n",
	           SafeDescription(props.deviceName.data(), props.deviceName.size()), props.vendorID,
	           props.deviceID, props.driverVersion);

	// The guard records live in a device-local buffer that is unreadable once the device is lost:
	// a hang with the guard on therefore means no loop tripped before the stall (the records of
	// completed dispatches were already logged as [loop-guard] lines).
	out.Printf("debug_options: loop_guard_limit={} compute_loop_guard_limit={} buffer_probe_shaders={} loop_guard_records=unreadable_after_device_loss\n",
	           Config::LoopGuardLimit(), Config::ComputeLoopGuardLimit(), Config::BufferProbeHashes().size());

	// Signal-handler safe: never block on m_mutex if the interrupted thread holds it.
	std::unique_lock lock(m_mutex, std::try_to_lock);
	const bool       locked               = lock.owns_lock();
	uint32_t         gpu_submit_started   = 0;
	uint32_t         gpu_submit_finished  = 0;
	uint32_t         gpu_op_started       = 0;
	uint32_t         gpu_op_finished      = 0;
	bool             has_gpu_markers      = false;
	uint64_t         effective_known_tick = known_gpu_tick;

	if (m_marker_mapped != nullptr) {
		if (graphics.allocator != nullptr && m_marker_allocation != nullptr) {
			(void)vmaInvalidateAllocation(graphics.allocator, m_marker_allocation, 0,
			                              VK_WHOLE_SIZE);
		}
		gpu_submit_started  = m_marker_mapped[0];
		gpu_submit_finished = m_marker_mapped[1];
		gpu_op_started      = m_marker_mapped[2];
		gpu_op_finished     = m_marker_mapped[3];
		has_gpu_markers     = true;
	}

	if (has_gpu_markers) {
		out.Printf(
		    "gpu_markers: submit_started={} submit_finished={} op_started={} op_finished={} "
		    "amd_buffer_marker={}\n",
		    gpu_submit_started, gpu_submit_finished, gpu_op_started, gpu_op_finished,
		    graphics.buffer_marker_amd_enabled ? "true" : "false");
	} else {
		out.Append("gpu_markers: unavailable (CPU submit ring only)\n");
	}

	const bool has_op_suspect =
	    has_gpu_markers && graphics.buffer_marker_amd_enabled && gpu_op_started > gpu_op_finished;
	const bool has_submit_suspect =
	    has_gpu_markers && gpu_submit_started > gpu_submit_finished;

	std::array<uint64_t, 16> suspect_hashes {};
	size_t                   suspect_hash_count = 0;
	const auto               add_suspect_hashes = [&](const OpBreadcrumb& op) {
        for (uint8_t i = 0; i < op.stage_count && i < op.stages.size(); ++i) {
            const uint64_t h = op.stages[i].hash;
            if (h == 0) {
                continue;
            }
            bool seen = false;
            for (size_t j = 0; j < suspect_hash_count; ++j) {
                if (suspect_hashes[j] == h) {
                    seen = true;
                    break;
                }
            }
            if (!seen && suspect_hash_count < suspect_hashes.size()) {
                suspect_hashes[suspect_hash_count++] = h;
            }
        }
    };

	const auto is_suspect_op = [&](uint32_t marker_id) {
		return has_op_suspect && marker_id > gpu_op_finished && marker_id <= gpu_op_started;
	};

	if (!locked) {
		out.Append("submits: skipped (tracker mutex held by interrupted thread)\n");
	} else {
		effective_known_tick = std::max(effective_known_tick, m_last_known_tick);

		std::array<size_t, kMaxSubmits> order {};
		size_t                          order_count = 0;
		for (size_t i = 0; i < kMaxSubmits; ++i) {
			if (m_submits[i].state != SubmitState::Empty) {
				order[order_count++] = i;
			}
		}
		std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(order_count),
		          [&](size_t a, size_t b) { return m_submits[a].seq < m_submits[b].seq; });

		const auto is_in_flight = [&](const SubmitRecord& rec) {
			if (rec.state == SubmitState::Recording) {
				return true;
			}
			if (has_gpu_markers && gpu_submit_finished != 0 && rec.seq > gpu_submit_finished) {
				return true;
			}
			return rec.tick == 0 || rec.tick > effective_known_tick;
		};

		// Keep up to 2 completed submits before the first in-flight submit for context.
		size_t first_to_show = order_count;
		for (size_t i = 0; i < order_count; ++i) {
			if (is_in_flight(m_submits[order[i]])) {
				first_to_show = i > 2 ? i - 2 : 0;
				break;
			}
		}
		if (first_to_show == order_count && order_count != 0) {
			first_to_show = order_count > 4 ? order_count - 4 : 0;
		}

		if (order_count == 0) {
			out.Append("submits: none recorded\n");
		} else {
			out.Printf("recent_submits (showing {} of {} retained):\n",
			           order_count - first_to_show, order_count);
		}

		for (size_t idx = first_to_show; idx < order_count; ++idx) {
			const auto& rec       = m_submits[order[idx]];
			const bool  in_flight = is_in_flight(rec);
			const char* state_str = rec.state == SubmitState::Recording
			                            ? "recording"
			                            : (in_flight ? "in-flight" : "completed");
			const bool  suspect_submit = has_submit_suspect && rec.seq == gpu_submit_started;

			out.Printf("  submit seq={} tick={} frame={} state={} ops={} markers={}..{}{}\n",
			           rec.seq, rec.tick, rec.frame_num, state_str, rec.total_ops,
			           rec.first_op_id, rec.last_op_id,
			           suspect_submit ? " <<< SUSPECT SUBMIT" : "");

			if (rec.ops.empty()) {
				continue;
			}

			const size_t op_count   = rec.ops.size();
			const size_t oldest_idx = rec.total_ops <= kMaxOpsPerSubmit
			                              ? 0u
			                              : static_cast<size_t>(rec.total_ops % kMaxOpsPerSubmit);
			// Print ALL ops for in-flight submits; keep last 4 for completed submits.
			const size_t tail_keep  = in_flight ? op_count : 4u;
			const size_t tail_start = op_count > tail_keep ? op_count - tail_keep : 0u;
			bool         elided     = false;

			for (size_t i = 0; i < op_count; ++i) {
				const auto& op      = rec.ops[(oldest_idx + i) % op_count];
				const bool  suspect = is_suspect_op(op.marker_id);
				const bool  is_last = (i + 1 == op_count);
				const bool  show    = suspect || i >= tail_start ||
				                   (has_op_suspect && op.marker_id == gpu_op_finished);
				if (!show) {
					if (!elided) {
						out.Printf("    ... ({} earlier ops omitted)\n", tail_start);
						elided = true;
					}
					continue;
				}
				const char* tag = "";
				if (suspect) {
					tag = " <<< SUSPECT OP";
					add_suspect_hashes(op);
				} else if (has_op_suspect && op.marker_id == gpu_op_finished) {
					tag = " (last completed)";
				} else if (is_last && in_flight) {
					tag = " (last in submit)";
				}
				out.Printf("    [op={}] ", op.marker_id);
				FormatOpLine(out, op);
				out.Printf("{}\n", tag);

				const bool print_buffers =
				    suspect || (is_last && in_flight) || (!has_op_suspect && in_flight);
				if (print_buffers) {
					if (op.push_dword_count > 0) {
						out.Append("      push_dwords:");
						for (uint8_t pi = 0; pi < op.push_dword_count && pi < op.push_dwords.size();
						     ++pi) {
							out.Printf(" {:08x}", op.push_dwords[pi]);
						}
						out.Append("\n");
					}
					if (op.user_data_count > 0) {
						out.Append("      user_data:");
						for (uint8_t ui = 0;
						     ui < op.user_data_count && ui < op.user_data_dwords.size(); ++ui) {
							out.Printf(" {:08x}", op.user_data_dwords[ui]);
						}
						out.Append("\n");
					}
					for (uint8_t bi = 0; bi < op.buffer_count && bi < op.buffers.size(); ++bi) {
						const auto& b = op.buffers[bi];
						out.Printf(
						    "      buf[{}]: set={} binding={} elem={} guest_va=0x{:016x} "
						    "vk_buf=0x{:016x} gpu_va=0x{:016x} offset=0x{:x} range=0x{:x} "
						    "buffer_size=0x{:x}",
						    bi, b.set, b.binding, b.array_element, b.guest_va, b.vk_buffer,
						    b.gpu_va, b.offset, b.range, b.buffer_size);
						if (b.gpu_va != 0 && b.range != 0 && b.range != VK_WHOLE_SIZE) {
							out.Printf(" gpu_range=[0x{:016x}..0x{:016x})", b.gpu_va + b.offset,
							           b.gpu_va + b.offset + b.range);
						}
						out.Append("\n");
						if (b.word_count > 0) {
							out.Append("        words:");
							for (uint8_t wi = 0; wi < b.word_count && wi < b.words.size(); ++wi) {
								out.Printf(" {:08x}", b.words[wi]);
							}
							out.Append("\n");
						}
					}
					if (op.total_buffers > op.buffer_count) {
						out.Printf("      ... ({} more bound buffers omitted)\n",
						           op.total_buffers - op.buffer_count);
					}
				}
			}

			if (!has_op_suspect && (suspect_submit || (in_flight && idx + 1 == order_count))) {
				for (size_t i = tail_start; i < op_count; ++i) {
					add_suspect_hashes(rec.ops[(oldest_idx + i) % op_count]);
				}
			}
		}
	}

	if (suspect_hash_count != 0) {
		out.Append("bisect_hint: --skip-shaders ");
		for (size_t i = 0; i < suspect_hash_count; ++i) {
			if (i != 0) {
				out.Append(",");
			}
			out.Printf("0x{:016x}", suspect_hashes[i]);
		}
		out.Append("\n");
	}
	out.Append("=== END GPU DEVICE LOST TRIAGE ===\n");
	out.Flush();

	// With the driver's hang mode on, its dump is written by the driver's own submit thread; the
	// process must stay alive until that is finished. This comes before the optional fault query.
	WaitForDriverHangDump();

	// Run VK_EXT_device_fault only when explicitly requested via --gpu-fault-query /
	// KYTY_GPU_FAULT_QUERY=1, on the dedicated host worker thread after breadcrumbs are flushed.
	if (Config::GpuFaultQueryEnabled() && m_fault_thread.joinable()) {
		m_fault_done.store(false, std::memory_order_release);
		m_fault_request.store(true, std::memory_order_release);
		WakeFaultWorker();
		for (int i = 0; i < 200 && !m_fault_done.load(std::memory_order_acquire); ++i) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		if (!m_fault_done.load(std::memory_order_acquire)) {
			constexpr std::string_view kTimeoutMsg =
			    "=== GPU DEVICE FAULT INFO ===\nVK_EXT_device_fault: worker timed out\n"
			    "=== END GPU DEVICE FAULT INFO ===\n";
			WriteRawStderr(kTimeoutMsg.data(), kTimeoutMsg.size());
		}
	}
	m_report_finished.store(true, std::memory_order_release);
}

void GpuBreadcrumbTracker::WaitForDriverHangDump() {
	if (!Common::HangDump::DriverHangModeActive(std::getenv("RADV_DEBUG"))) {
		return;
	}
	const uint32_t seconds = Config::HangDumpWaitSeconds();
	const char*    home    = std::getenv("HOME");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const int pid = _getpid();
#else
	const int pid = static_cast<int>(::getpid());
#endif
	std::array<char, 8192> buf {};
	FixedStderrSink        out(buf);
	if (seconds == 0) {
		out.Printf("=== DRIVER HANG DUMP ===\nwait disabled (--hang-dump-wait 0); looking in {}/radv_dumps_{}_*\n"
		           "=== END DRIVER HANG DUMP ===\n",
		           home != nullptr ? home : ".", pid);
		out.Flush();
		return;
	}
	out.Printf("=== DRIVER HANG DUMP ===\nRADV_DEBUG has `hang`: waiting up to {} s for {}/radv_dumps_{}_*\n",
	           seconds, home != nullptr ? home : ".", pid);
	out.Flush();

	Common::HangDump::WaitPolicy policy;
	policy.timeout    = std::chrono::seconds(seconds);
	const auto result = Common::HangDump::WaitForDriverDump(
	    policy, [&] { return Common::HangDump::ScanDumpFolder(home, pid); },
	    [](std::chrono::milliseconds d) { std::this_thread::sleep_for(d); });
	out.Printf("driver_dump: outcome={} path={} files={} bytes={} last_chunk={} waited_ms={}\n"
	           "=== END DRIVER HANG DUMP ===\n",
	           Common::HangDump::WaitOutcomeName(result.outcome),
	           result.state.found ? result.state.path : "(none)", result.state.file_count,
	           result.state.total_bytes, result.state.has_last_chunk ? "yes" : "no",
	           result.waited.count());
	out.Flush();
}

void ReportDeviceLost(GraphicContext& graphics, const char* operation, vk::Result result,
                      uint64_t tick, uint64_t known_gpu_tick) {
	graphics.breadcrumbs.ReportDeviceLost(graphics, operation, result, tick, known_gpu_tick);
}

void RequireVulkanSuccess(GraphicContext& graphics, vk::Result result, const char* operation) {
	if (result == vk::Result::eErrorDeviceLost) {
		ReportDeviceLost(graphics, operation, result);
	}
	RequireVulkanSuccess(result, operation);
}

} // namespace Libs::Graphics
