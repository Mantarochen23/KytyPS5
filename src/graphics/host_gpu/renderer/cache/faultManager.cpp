#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/assert.h"
#include "common/frameCost.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults         = 1024;
constexpr size_t PageFaultAreaSize     = MaxPageFaults * sizeof(uint64_t);
constexpr size_t FaultDownloadAreaSize = PageFaultAreaSize + BufferCache::FAULT_TAIL_SIZE;
constexpr size_t MaxTrackedBdaMissPages = 4096;

std::string FormatShaderSuffix(const GpuBreadcrumbTracker::ShaderHashSnapshot& shaders) {
	std::string out;
	if (!shaders.shaders.empty()) {
		out += shaders.dma_filtered ? " dma_shaders=[" : " shaders=[";
		for (size_t i = 0; i < shaders.shaders.size(); ++i) {
			if (i != 0) {
				out += ' ';
			}
			out += fmt::format("{}=0x{:016x}",
			                   ShaderRecompiler::StageName(shaders.shaders[i].stage),
			                   shaders.shaders[i].hash);
		}
		out += ']';
	} else {
		out += " dma_shaders=[]";
	}
	if (shaders.truncated) {
		out += " truncated=1";
	}
	return out;
}

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::FAULT_BUFFER_SIZE),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * FaultDownloadAreaSize) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	m_fault_process_pipeline =
	    CreateComputePipeline(m_graphics.device, FAULT_BUFFER_PROCESS_SPV,
		                      m_fault_process_pipeline_layout, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

std::vector<std::string> FaultManager::FormatNewBdaMisses(
    std::span<const uint32_t> miss_words, const GpuBreadcrumbTracker::ShaderHashSnapshot& shaders,
    std::unordered_set<uint64_t>& seen_pages) {
	if (miss_words.size() < 2u || miss_words[0] == 0u) {
		return {};
	}
	const uint32_t count           = miss_words[0];
	const uint32_t available_slots = static_cast<uint32_t>((miss_words.size() - 2u) / 2u);
	const uint32_t recorded =
	    std::min<uint32_t>(count, std::min<uint32_t>(BufferCache::BDA_MISS_SLOT_COUNT,
	                                                 available_slots));
	std::vector<uint64_t> new_addresses;
	for (uint32_t i = 0; i < recorded; ++i) {
		const uint64_t lo      = miss_words[2u + i * 2u];
		const uint64_t hi      = miss_words[2u + i * 2u + 1u];
		const uint64_t address = lo | (hi << 32u);
		const uint64_t page    = address >> BufferCache::CACHING_PAGEBITS;
		if (seen_pages.contains(page)) {
			continue;
		}
		if (seen_pages.size() >= MaxTrackedBdaMissPages) {
			break;
		}
		seen_pages.insert(page);
		new_addresses.push_back(address);
	}
	if (new_addresses.empty()) {
		return {};
	}
	const std::string        shader_suffix = FormatShaderSuffix(shaders);
	std::vector<std::string> lines;
	lines.reserve(new_addresses.size());
	for (const uint64_t address: new_addresses) {
		lines.push_back(
		    fmt::format("[bda-miss] address=0x{:016x} count={}{}", address, count, shader_suffix));
	}
	return lines;
}

void FaultManager::ProcessFaultBuffer() {
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		Common::FrameCost::ScopedWaitReason wait_reason(Common::FrameCost::WaitReason::FaultBuf);
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * FaultDownloadAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	std::memset(mapped, 0, FaultDownloadAreaSize);
	m_download_buffer.Flush(offset, FaultDownloadAreaSize);

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead |
	                            vk::AccessFlagBits2::eShaderWrite |
	                            vk::AccessFlagBits2::eTransferRead;
	pre_barrier.buffer = m_fault_buffer.Handle();
	pre_barrier.offset = 0;
	pre_barrier.size   = m_fault_buffer.Size();

	vk::BufferMemoryBarrier2 clear_barrier {};
	clear_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	clear_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	clear_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	clear_barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
	clear_barrier.buffer        = m_fault_buffer.Handle();
	clear_barrier.offset        = BufferCache::FAULT_BITMAP_SIZE;
	clear_barrier.size          = BufferCache::FAULT_TAIL_SIZE;

	vk::BufferMemoryBarrier2 post_barrier {};
	post_barrier.srcStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	post_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	post_barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	post_barrier.buffer = m_fault_buffer.Handle();
	post_barrier.offset = 0;
	post_barrier.size   = m_fault_buffer.Size();

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, BufferCache::FAULT_BITMAP_SIZE},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0, writes);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 128; // Four words per invocation.
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);

	const vk::BufferCopy miss_copy {
	    BufferCache::FAULT_BITMAP_SIZE,
	    offset + PageFaultAreaSize,
	    BufferCache::FAULT_TAIL_SIZE,
	};
	command.copyBuffer(m_fault_buffer.Handle(), m_download_buffer.Handle(), 1, &miss_copy);
	dependency.pBufferMemoryBarriers = &clear_barrier;
	command.pipelineBarrier2(dependency);
	command.fillBuffer(m_fault_buffer.Handle(), BufferCache::FAULT_BITMAP_SIZE,
	                   BufferCache::FAULT_TAIL_SIZE, 0);

	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);

	auto shaders            = m_graphics.breadcrumbs.CollectShaderHashesSince(m_last_breadcrumb_op_id);
	m_last_breadcrumb_op_id = shaders.last_op_id;

	const auto area = m_current_area;
	m_scheduler.DeferOperation([this, mapped, offset, area, shaders = std::move(shaders)] {
		m_download_buffer.Invalidate(offset, FaultDownloadAreaSize);
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		// An overflowing scan keeps counting past the MaxPageFaults - 1 records the area holds.
		const auto count = std::min<uint64_t>(faults[0], MaxPageFaults - 1);
		for (uint32_t index = 1; index <= count; ++index) {
			const auto address = BufferCache::GuestAddress(faults[index]);
			fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			(void)m_buffer_cache.FindBuffer(start, end - start);
		});
		const auto* miss_words = std::bit_cast<const uint32_t*>(mapped + PageFaultAreaSize);
		for (const auto& line: FormatNewBdaMisses(
		         std::span<const uint32_t>(miss_words, BufferCache::BDA_MISS_AREA_DWORDS), shaders,
		         m_seen_bda_miss_pages)) {
			LOGF("%s\n", line.c_str());
		}
		const auto* guard_words = miss_words + BufferCache::BDA_MISS_AREA_DWORDS;
		for (const auto& line: FormatLoopGuardTrips(
		         std::span<const uint32_t>(guard_words, LoopGuardAreaDwords),
		         m_seen_loop_guard_trips)) {
			Log::WriteToConsoleAndLog((line + "\n").c_str());
		}
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics
