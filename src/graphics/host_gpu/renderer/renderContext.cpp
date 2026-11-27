#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/frameCost.h"
#include "common/logging/log.h"
#include "common/ports.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/faultStats.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <string>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_gpu(nullptr), m_graphics(graphics), m_render_executor(*this),
      m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_resolve_cache(*this), m_texture_cache(graphics, m_command_scheduler, m_page_manager,
	                                          m_buffer_cache, m_resolve_cache),
      m_image_heap(*this, m_page_manager), m_read_watch(m_page_manager),
      m_pipeline_cache(graphics, &m_image_heap, shader_store),
      m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	Log::WriteToConsoleAndLog(Config::PortsSummary() + "\n");
	FaultStats::ConfigureStorm(Config::FaultStormFaults(), Config::FaultStormFrames(),
	                           Config::FaultStormHold());
	FaultStats::StartReporter();
	if (const auto stamps = Config::MaterializeMemoStamps(); stamps != 0u) {
		m_read_watch.Configure(stamps, {.user      = this,
		                                .trackable = [](void* user, uint64_t address, uint64_t size) {
			                                auto& self = *static_cast<RenderContext*>(user);
			                                return self.IsMapped(address, size) &&
			                                       !self.m_texture_cache.IsRegionRegistered(address, size);
		                                },
		                                .gpu_holds = [](void* user, uint64_t address, uint64_t size) {
			                                return static_cast<RenderContext*>(user)
			                                    ->m_buffer_cache.HasGpuDirtyBytes(address, size);
		                                }});
	}
	GlobalMaterializationMemoSettings() = {
	    .entries = Config::MaterializeMemoEntries(),
	    .stamps  = Config::MaterializeMemoStamps() != 0u ? &m_read_watch : nullptr};
	// One unconditional line per debug option, so a log without findings still shows the option
	// was active (a hang can end the process before any other line is written).
	if (Config::BufferProbeEnabled()) {
		std::string hashes;
		for (const auto hash: Config::BufferProbeHashes()) {
			char text[32];
			std::snprintf(text, sizeof(text), "%s0x%016" PRIx64, hashes.empty() ? "" : ",", hash);
			hashes += text;
		}
		Log::WriteToConsoleAndLog(("[buffer-probe] enabled hashes=" + hashes + "\n").c_str());
	}
	if (Config::LoopGuardLimit() != 0 || Config::ComputeLoopGuardLimit() != 0) {
		char text[128];
		std::snprintf(text, sizeof(text), "[loop-guard] enabled limit=%u compute-limit=%u\n",
		              Config::LoopGuardLimit(), Config::ComputeLoopGuardLimit());
		Log::WriteToConsoleAndLog(text);
	}
	if (graphics.ray_query_enabled) {
		m_ray_tracing = std::make_unique<RayTracing>(*this);
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
	GlobalMaterializationMemoSettings() = {};
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

uint64_t RenderContext::CurrentFrameNum() const noexcept {
	return m_gpu != nullptr ? static_cast<uint64_t>(m_gpu->GetFrameNum()) : 0u;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	const auto started = FaultStats::NowNs();
	FaultStats::SetFrame(static_cast<uint32_t>(CurrentFrameNum()));
	FaultStats::g_hot_blocks.Note(fault_vaddr);
	(access == PageFaultAccess::Write ? FaultStats::g_counters.write_faults
	                                  : FaultStats::g_counters.read_faults)
	    .fetch_add(1, std::memory_order_relaxed);
	const auto finish = [started] {
		FaultStats::g_counters.handler_ns.fetch_add(FaultStats::NowNs() - started,
		                                            std::memory_order_relaxed);
	};
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		finish();
		return true;
	}
	if (access == PageFaultAccess::Write) {
		m_read_watch.Written(fault_vaddr, fault_size);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	finish();
	return true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_read_watch.Written(vaddr, size);
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	m_buffer_cache.InvalidateRuntimeTables(vaddr, size);
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			Common::FrameCost::ScopedWaitReason wait_reason(Common::FrameCost::WaitReason::Unmap);
			const auto                          tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_read_watch.Written(vaddr, size);
		m_buffer_cache.InvalidateRuntimeTables(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda(bool untracked_writes) {
	[[maybe_unused]] static const bool logged = [] {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		return true;
	}();
	// DMA/device-address shaders can write any range. Runtime buffer tables pass false because
	// their exact cached candidate union is stamped by RebindBuffers.
	if (untracked_writes) m_buffer_cache.NoteUntrackedWrite();
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	if (m_ray_tracing) {
		m_ray_tracing->RunGarbageCollector();
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	const auto ident = static_cast<uintptr_t>(event_id);
	auto*      data  = reinterpret_cast<void*>((static_cast<uintptr_t>(context_id) << 16u) | ident);
	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, ident, LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS, data);
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
