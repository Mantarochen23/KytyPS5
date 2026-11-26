#include "graphics/host_gpu/deviceFault.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr vk::DeviceSize kMarkerBufferSize     = 64;
constexpr vk::DeviceSize kOffsetSubmitStarted  = 0;
constexpr vk::DeviceSize kOffsetSubmitFinished = 4;
constexpr vk::DeviceSize kOffsetOpStarted      = 8;
constexpr vk::DeviceSize kOffsetOpFinished     = 12;

constexpr uint32_t kMaxFaultRecords = 4096;

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

[[nodiscard]] const char* FaultAddressTypeName(vk::DeviceFaultAddressTypeEXT type) {
	switch (type) {
		case vk::DeviceFaultAddressTypeEXT::eNone: return "None";
		case vk::DeviceFaultAddressTypeEXT::eReadInvalid: return "ReadInvalid";
		case vk::DeviceFaultAddressTypeEXT::eWriteInvalid: return "WriteInvalid";
		case vk::DeviceFaultAddressTypeEXT::eExecuteInvalid: return "ExecuteInvalid";
		case vk::DeviceFaultAddressTypeEXT::eInstructionPointerUnknown:
			return "InstructionPointerUnknown";
		case vk::DeviceFaultAddressTypeEXT::eInstructionPointerInvalid:
			return "InstructionPointerInvalid";
		case vk::DeviceFaultAddressTypeEXT::eInstructionPointerFault:
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

[[nodiscard]] std::string FormatOpStages(const OpBreadcrumb& op) {
	std::string out;
	for (uint8_t i = 0; i < op.stage_count && i < op.stages.size(); ++i) {
		if (!out.empty()) {
			out += ' ';
		}
		out += fmt::format("{}=0x{:016x}", ShaderRecompiler::StageName(op.stages[i].stage),
		                   op.stages[i].hash);
	}
	return out;
}

[[nodiscard]] std::string FormatOpLine(const OpBreadcrumb& op) {
	std::string dims;
	switch (op.kind) {
		case BreadcrumbOpKind::Dispatch:
			dims = fmt::format("{}x{}x{}", op.dim_x, op.dim_y, op.dim_z);
			break;
		case BreadcrumbOpKind::DispatchIndirect: dims = "indirect"; break;
		case BreadcrumbOpKind::DrawMesh:
			dims = fmt::format("groups={}x{}", op.dim_x, op.dim_y);
			break;
		case BreadcrumbOpKind::DrawMeshIndirect:
			dims = fmt::format("indirect_count={}", op.dim_x);
			break;
		case BreadcrumbOpKind::DrawIndirect:
		case BreadcrumbOpKind::DrawIndexedIndirect:
			dims = fmt::format("count={} draws={} stride={}", op.dim_x, op.dim_y, op.dim_z);
			break;
		default: dims = fmt::format("count={} instances={}", op.dim_x, op.dim_y); break;
	}
	return fmt::format("{} {} {}", OpKindName(op.kind), dims, FormatOpStages(op));
}

void QueryDeviceFaultExt(const GraphicContext& graphics, std::string& report) {
	if (!graphics.device_fault_ext_enabled || graphics.device == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) {
		report += fmt::format("VK_EXT_device_fault: unavailable (enabled={})\n",
		                      graphics.device_fault_ext_enabled ? "true" : "false");
		return;
	}

	vk::DeviceFaultCountsEXT counts {};
	const auto               count_result = graphics.device.getDeviceFaultInfoEXT(&counts, nullptr);
	if (count_result != vk::Result::eSuccess && count_result != vk::Result::eIncomplete) {
		report += fmt::format("VK_EXT_device_fault: count query failed ({})\n",
		                      vk::to_string(count_result));
		return;
	}

	const uint64_t vendor_binary_bytes = counts.vendorBinarySize;
	const uint32_t address_count       = std::min(counts.addressInfoCount, kMaxFaultRecords);
	const uint32_t vendor_count        = std::min(counts.vendorInfoCount, kMaxFaultRecords);

	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(address_count);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(vendor_count);

	counts.addressInfoCount = address_count;
	counts.vendorInfoCount  = vendor_count;
	// Do not copy raw vendor binary dumps into the text log.
	counts.vendorBinarySize = 0;

	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos     = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos      = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = nullptr;

	const auto info_result = graphics.device.getDeviceFaultInfoEXT(&counts, &info);
	if (info_result != vk::Result::eSuccess && info_result != vk::Result::eIncomplete) {
		report += fmt::format("VK_EXT_device_fault: info query failed ({})\n",
		                      vk::to_string(info_result));
		return;
	}

	const auto desc = SafeDescription(info.description.data(), info.description.size());
	report += fmt::format(
	    "VK_EXT_device_fault: description=\"{}\" addresses={} vendor_infos={} "
	    "vendor_binary_bytes={}\n",
	    desc, counts.addressInfoCount, counts.vendorInfoCount, vendor_binary_bytes);

	const uint32_t reported_addresses =
	    std::min(counts.addressInfoCount, static_cast<uint32_t>(addresses.size()));
	for (uint32_t i = 0; i < reported_addresses; ++i) {
		const auto& addr = addresses[i];
		report += fmt::format("  fault_addr[{}]: type={} ({}) gpu_va=0x{:016x} precision=0x{:x}\n",
		                      i, FaultAddressTypeName(addr.addressType),
		                      static_cast<int>(addr.addressType), addr.reportedAddress,
		                      addr.addressPrecision);
	}

	const uint32_t reported_vendors =
	    std::min(counts.vendorInfoCount, static_cast<uint32_t>(vendors.size()));
	for (uint32_t i = 0; i < reported_vendors; ++i) {
		const auto& v     = vendors[i];
		const auto  vdesc = SafeDescription(v.description.data(), v.description.size());
		report += fmt::format(
		    "  fault_vendor[{}]: code=0x{:016x} data=0x{:016x} desc=\"{}\"\n", i,
		    v.vendorFaultCode, v.vendorFaultData, vdesc);
	}
}

} // namespace

void GpuBreadcrumbTracker::Init(GraphicContext& graphics) {
	std::lock_guard lock(m_mutex);
	m_emit_debug_labels =
	    graphics.debug_utils_ext_enabled &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdInsertDebugUtilsLabelEXT != nullptr &&
	    (Config::GraphicsDebugDumpEnabled() || RadvDebugEnabled());

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
	std::lock_guard lock(m_mutex);
	if (m_marker_buffer != nullptr && graphics.allocator != nullptr) {
		vmaDestroyBuffer(graphics.allocator, m_marker_buffer, m_marker_allocation);
	}
	m_marker_buffer     = nullptr;
	m_marker_allocation = nullptr;
	m_marker_mapped     = nullptr;
}

void GpuBreadcrumbTracker::BeginSubmit(GraphicContext& graphics, vk::CommandBuffer cmd,
                                       uint64_t frame_num) {
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
                                       OpBreadcrumb op) {
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
			    fmt::format("submit={} op={} {}", submit_seq, op.marker_id, FormatOpLine(op));
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

void GpuBreadcrumbTracker::ReportDeviceLost(GraphicContext& graphics, const char* operation,
                                            vk::Result result, uint64_t tick,
                                            uint64_t known_gpu_tick) {
	if (m_reported.test_and_set(std::memory_order_acq_rel)) {
		return;
	}

	std::string report;
	report.reserve(4096);
	report += "=== GPU DEVICE LOST TRIAGE ===\n";

	const auto& props = graphics.physical_device_properties;
	report += fmt::format("operation={} result={} tick={} known_gpu_tick={}\n",
	                      operation != nullptr ? operation : "?", vk::to_string(result), tick,
	                      known_gpu_tick);
	report += fmt::format("device=\"{}\" vendor=0x{:04x} device_id=0x{:04x} driver=0x{:08x}\n",
	                      SafeDescription(props.deviceName.data(), props.deviceName.size()),
	                      props.vendorID, props.deviceID, props.driverVersion);

	QueryDeviceFaultExt(graphics, report);

	uint32_t gpu_submit_started  = 0;
	uint32_t gpu_submit_finished = 0;
	uint32_t gpu_op_started      = 0;
	uint32_t gpu_op_finished     = 0;
	bool     has_gpu_markers     = false;

	std::vector<SubmitRecord> snapshot;
	uint64_t                  effective_known_tick = known_gpu_tick;
	{
		std::lock_guard lock(m_mutex);
		effective_known_tick = std::max(effective_known_tick, m_last_known_tick);
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
		for (const auto& rec: m_submits) {
			if (rec.state != SubmitState::Empty) {
				snapshot.push_back(rec);
			}
		}
	}

	std::ranges::sort(snapshot, {}, &SubmitRecord::seq);

	if (has_gpu_markers) {
		report += fmt::format(
		    "gpu_markers: submit_started={} submit_finished={} op_started={} op_finished={} "
		    "amd_buffer_marker={}\n",
		    gpu_submit_started, gpu_submit_finished, gpu_op_started, gpu_op_finished,
		    graphics.buffer_marker_amd_enabled ? "true" : "false");
	} else {
		report += "gpu_markers: unavailable (CPU submit ring only)\n";
	}

	const bool has_op_suspect =
	    has_gpu_markers && graphics.buffer_marker_amd_enabled && gpu_op_started > gpu_op_finished;
	const bool has_submit_suspect =
	    has_gpu_markers && gpu_submit_started > gpu_submit_finished;

	std::vector<uint64_t>        suspect_hashes;
	std::unordered_set<uint64_t> suspect_seen;
	const auto                   add_suspect_hashes = [&](const OpBreadcrumb& op) {
        for (uint8_t i = 0; i < op.stage_count && i < op.stages.size(); ++i) {
            const uint64_t h = op.stages[i].hash;
            if (h != 0 && suspect_seen.insert(h).second) {
                suspect_hashes.push_back(h);
            }
        }
    };

	const auto is_suspect_op = [&](uint32_t marker_id) {
		return has_op_suspect && marker_id > gpu_op_finished && marker_id <= gpu_op_started;
	};

	const auto is_in_flight = [&](const SubmitRecord& rec) {
		if (rec.state == SubmitState::Recording) {
			return true;
		}
		if (has_gpu_markers && gpu_submit_finished != 0 && rec.seq > gpu_submit_finished) {
			return true;
		}
		return rec.tick == 0 || rec.tick > effective_known_tick;
	};

	// Determine first submit index to print: keep up to 2 completed submits before the first
	// in-flight submit so the reader has context without flooding the log.
	size_t first_to_show = snapshot.size();
	for (size_t i = 0; i < snapshot.size(); ++i) {
		if (is_in_flight(snapshot[i])) {
			first_to_show = i > 2 ? i - 2 : 0;
			break;
		}
	}
	if (first_to_show == snapshot.size() && !snapshot.empty()) {
		first_to_show = snapshot.size() > 4 ? snapshot.size() - 4 : 0;
	}

	if (snapshot.empty()) {
		report += "submits: none recorded\n";
	} else {
		report += fmt::format("recent_submits (showing {} of {} retained):\n",
		                      snapshot.size() - first_to_show, snapshot.size());
	}

	for (size_t idx = first_to_show; idx < snapshot.size(); ++idx) {
		const auto& rec       = snapshot[idx];
		const bool  in_flight = is_in_flight(rec);
		const char* state_str = rec.state == SubmitState::Recording
		                            ? "recording"
		                            : (in_flight ? "in-flight" : "completed");
		const bool  suspect_submit = has_submit_suspect && rec.seq == gpu_submit_started;

		report += fmt::format("  submit seq={} tick={} frame={} state={} ops={} markers={}..{}{}\n",
		                      rec.seq, rec.tick, rec.frame_num, state_str, rec.total_ops,
		                      rec.first_op_id, rec.last_op_id,
		                      suspect_submit ? " <<< SUSPECT SUBMIT" : "");

		if (rec.ops.empty()) {
			continue;
		}

		// Reorder the per-submit ring into chronological order.
		std::vector<OpBreadcrumb> ordered;
		ordered.reserve(rec.ops.size());
		if (rec.total_ops <= kMaxOpsPerSubmit) {
			ordered.assign(rec.ops.begin(), rec.ops.end());
		} else {
			const size_t oldest = rec.total_ops % kMaxOpsPerSubmit;
			ordered.insert(ordered.end(), rec.ops.begin() + static_cast<std::ptrdiff_t>(oldest),
			               rec.ops.end());
			ordered.insert(ordered.end(), rec.ops.begin(),
			               rec.ops.begin() + static_cast<std::ptrdiff_t>(oldest));
		}

		// For completed submits show only the last 4 ops; for in-flight submits show any suspect
		// ops plus the last 16 ops.
		const size_t tail_keep  = in_flight ? 16u : 4u;
		const size_t tail_start = ordered.size() > tail_keep ? ordered.size() - tail_keep : 0u;
		bool         elided     = false;

		for (size_t i = 0; i < ordered.size(); ++i) {
			const auto& op      = ordered[i];
			const bool  suspect = is_suspect_op(op.marker_id);
			const bool  is_last = (i + 1 == ordered.size());
			const bool  show    = suspect || i >= tail_start ||
                               (has_op_suspect && op.marker_id == gpu_op_finished);
			if (!show) {
				if (!elided) {
					report += fmt::format("    ... ({} earlier ops omitted)\n", tail_start);
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
			report += fmt::format("    [op={}] {}{}\n", op.marker_id, FormatOpLine(op), tag);
		}

		// If no per-op GPU marker narrowed down the suspect, gather hashes from the suspect
		// submit or the last in-flight submit's tail for the bisect hint.
		if (!has_op_suspect && (suspect_submit || (in_flight && idx + 1 == snapshot.size()))) {
			for (size_t i = tail_start; i < ordered.size(); ++i) {
				add_suspect_hashes(ordered[i]);
			}
		}
	}

	if (!suspect_hashes.empty()) {
		std::string csv;
		const size_t limit = std::min<size_t>(suspect_hashes.size(), 16);
		for (size_t i = 0; i < limit; ++i) {
			if (i != 0) {
				csv += ',';
			}
			csv += fmt::format("0x{:016x}", suspect_hashes[i]);
		}
		report += fmt::format("bisect_hint: --skip-shaders {}\n", csv);
	}
	report += "=== END GPU DEVICE LOST TRIAGE ===\n";

	Log::WriteToConsoleAndLog(report);
	std::fflush(nullptr);
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
