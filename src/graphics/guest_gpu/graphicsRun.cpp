#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/command_processor/waitDiagnostics.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/renderActivity.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static thread_local Pm4Execution* g_current_execution = nullptr;
static thread_local GuestGpu*     g_gpu_state         = nullptr;
// Prints what the command thread is doing when no frame was recorded for a while.
static std::unique_ptr<RenderStallReporter> g_stall_reporter;

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_renderer.GetPipelineCache().SetAsyncFinishCallback([this] {
		Common::LockGuard lock(m_queue_mutex);
		m_async_ready = true;
		m_work_available.Signal();
	});
	ShaderSetComputePrefetchNotifier([this] {
		Common::LockGuard lock(m_queue_mutex);
		m_prefetch_pending = true;
		m_work_available.Signal();
	});
	m_thread = std::jthread(ThreadRun, this);
	g_stall_reporter = std::make_unique<RenderStallReporter>(Config::RenderStallReportSeconds());
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	g_stall_reporter.reset();
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting        = false;
		m_stopping         = true;
		m_prefetch_pending = false;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_renderer.GetPipelineCache().SetAsyncFinishCallback({});
	ShaderSetComputePrefetchNotifier({});
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> commands) {
	if (commands.empty()) {
		return;
	}
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 0;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// process the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready.acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.reset();
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_addr               = 0;
	m_predicate_skip                   = false;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;
}

// Pops a pushed context, then resets the register state the operation bits name.
void CommandProcessor::ResetQueue(uint32_t op) {
	if ((op & Pm4::ResetQueueOp::PopContext) != 0 && m_saved_ctx.has_value()) {
		ApplyContextStateOperation(ContextStateOperation::Pop);
	}
	if ((op & Pm4::ResetQueueOp::CxState) != 0) {
		m_ctx.Reset();
	}
	if ((op & Pm4::ResetQueueOp::ShState) != 0) {
		m_sh_ctx.Reset();
	}
	if ((op & Pm4::ResetQueueOp::UcState) != 0) {
		m_ucfg.Reset();
	}
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_saved_ctx.has_value());
			m_ctx = *m_saved_ctx;
			m_saved_ctx.reset();
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			m_ctx.Reset();
			break;
	}
}

void CommandProcessor::BufferInit() {
	m_draws = 0;
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
	// The occlusion counters reach a guest that polls them without waiting for a later fence.
	m_renderer.GetOcclusionQueries().Publish();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	const T value = *addr;
	if (!TestWaitRegMemValue(value, ref, mask, func)) {
		NoteWaitPoll(addr, sizeof(T) == sizeof(uint64_t), func, ref, mask, value);
		SuspendPm4();
	}
}

uint32_t CommandProcessor::DiagnosticsSlot() const {
	return m_interrupt_event_id >= 0x20 ? static_cast<uint32_t>(m_interrupt_event_id - 0x20) + 1u
	                                    : 0u;
}

// For the stall reporter: which packet this queue polls and what it read. The buffer cache is asked
// about the range on the first poll and about once a second after (it is the same thread).
void CommandProcessor::NoteWaitPoll(const void* address, bool is64, uint32_t func, uint64_t ref,
                                    uint64_t mask, uint64_t value) {
	WaitDiag::WaitPacket packet;
	packet.is64      = is64;
	packet.func      = func;
	packet.address   = reinterpret_cast<uint64_t>(address);
	packet.mask      = mask;
	packet.reference = ref;
	packet.value     = value;
	if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
		const auto& cursor = g_current_execution->m_buffer_stack.back();
		packet.buffer      = reinterpret_cast<uint64_t>(cursor.commands.data());
		packet.offset_dw   = cursor.offset_dw;
	}
	const auto now   = WaitDiag::NowMs();
	auto&      diag  = WaitDiag::Global();
	const auto polls = diag.NotePoll(DiagnosticsSlot(), packet, now);
	if (polls == 1 || polls % 8192u == 0) {
		const auto size = is64 ? sizeof(uint64_t) : sizeof(uint32_t);
		if (GuestRange {packet.address, size}.Valid()) {
			auto& buffers          = m_renderer.GetBufferCache();
			packet.gpu_modified    = buffers.IsRegionGpuModified(packet.address, size) ? 1 : 0;
			packet.gpu_dirty       = buffers.HasGpuDirtyBytes(packet.address, size) ? 1 : 0;
			diag.NotePoll(DiagnosticsSlot(), packet, now);
		}
	}
}

void CommandProcessor::NoteLabelWrite(const void* address, uint64_t value, uint32_t size) {
	WaitDiag::Global().NoteLabel(reinterpret_cast<uint64_t>(address), value, size,
	                             DiagnosticsSlot(), WaitDiag::NowMs());
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t);

void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
	} else if (*counter == 0) {
		SuspendPm4();
	} else {
		(*counter)--;
	}
}

void CommandProcessor::WriteData(uint64_t dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}

	// A GDS destination is a 16-bit DWORD offset.
	if (dst_sel == 6) {
		for (uint32_t i = 0; i < dw_num; i++, dst += !write_one_address) {
			m_renderer.GetBufferCache().FillBuffer((dst & 0xffffu) * 4u, 4u, src[i], true);
		}
		return;
	}

	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = reinterpret_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		if (dw_num <= 4u) {
			NoteLabelWrite(const_cast<const uint32_t*>(output), src[i], sizeof(uint32_t));
		}
		*output = src[i];
		output += !write_one_address;
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::GetLodStats(volatile uint32_t* dst, uint32_t size, bool reset) {
	auto& counters = m_renderer.GetRenderExecutor().m_mip_stats;
	// Each of up to 256 2-dword slots after the 16-dword header holds the clamped sample count
	// (dword 0 bits 0-23), the slot id (bits 24-31), the unclamped sample count (dword 1 bits 0-23)
	// and the finest mip sampled (bits 24-27, 15 for none). A non-zero header dword 0 then marks
	// the report complete.
	if (dst != nullptr && size >= 64) {
		const auto slots = std::min((size - 64) / 8, 256u);
		for (uint32_t slot = 0; slot < slots; slot++) {
			const auto& counter = counters[slot];
			dst[16 + slot * 2]  = (slot << 24u) | uint32_t {counter.clamped};
			dst[16 + slot * 2 + 1] =
			    (uint32_t {counter.finest_level} << 24u) | uint32_t {counter.unclamped};
		}
		dst[0] = 1;
	}
	if (reset) {
		counters = {};
	}
}

void CommandProcessor::DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes) {
	if (num_bytes == 0) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::DrainComputePrefetches() {
	if (!Config::AsyncShaderCompileEnabled()) {
		(void)ShaderTakeComputePrefetches();
		return;
	}
	const auto batch = ShaderTakeComputePrefetches();
	if (batch.empty()) {
		return;
	}
	Common::LockGuard lock(m_renderer.GetMutex());
	auto&             cache = m_renderer.GetPipelineCache();
	for (const auto& req: batch) {
		cache.PrefetchComputeProgram(req.cs_regs, req.dispatch_thread_dimensions, false);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_state = gpu;
	if (const auto processor = Common::Thread::ReserveCoreForCurrentThread(); processor >= 0) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "GPU: command processor assigned to the core of logical processor {}.\n", processor));
	}

	for (;;) {
		Submission                                                   submission;
		Common::UniqueFunction<void>                                 command;
		std::vector<std::pair<uint32_t, std::span<const uint32_t>>>  pending_prefetches;
		bool                                                         should_stop  = false;
		bool                                                         run_prefetch = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 &&
			       !gpu->m_prefetch_pending && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				RenderActivity::Note("cp-no-work", 0, "-", "nothing submitted");
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else if (gpu->m_prefetch_pending) {
				gpu->m_prefetch_pending = false;
				gpu->m_processing       = true;
				run_prefetch            = true;
			} else {
				if (gpu->m_async_ready) {
					gpu->m_async_ready = false;
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
				}
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					if (Config::AsyncShaderCompileEnabled()) {
						for (uint32_t id = 0; id < QueueCount && pending_prefetches.size() < 16u;
						     id++) {
							for (auto& queued: gpu->m_queues[id]) {
								if (queued.type == SubmissionType::Commands && !queued.prefetched) {
									queued.prefetched = true;
									pending_prefetches.emplace_back(id, queued.commands);
									if (pending_prefetches.size() >= 16u) {
										break;
									}
								}
							}
						}
					}
					if (!pending_prefetches.empty()) {
						gpu->m_processing = true;
					} else {
						bool all_waiting_on_shaders = true;
						for (const auto& queue: gpu->m_queues) {
							if (!queue.empty() && !queue.front().execution.SuspendedForShader()) {
								all_waiting_on_shaders = false;
								break;
							}
						}
						gpu->m_processing = false;
						// When every blocked queue is waiting on async shader/pipeline workers,
						// sleep on the condition variable until Async::Finish signals
						// m_work_available (with a 50 ms safety fallback). Guest memory WaitRegMem
						// polls keep 100 us.
						const uint32_t wait_us = all_waiting_on_shaders ? 50000u : 100u;
						RenderActivity::Note("cp-blocked", 0, "-",
						                     all_waiting_on_shaders
						                         ? "every queue waits for a shader or pipeline"
						                         : "a queue waits on guest memory or a flip");
						if (!gpu->m_async_ready && !gpu->m_prefetch_pending) {
							gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, wait_us);
						}
						gpu->m_async_ready = false;
						for (auto& queue: gpu->m_queues) {
							if (!queue.empty()) {
								queue.front().blocked = false;
							}
						}
						continue;
					}
				} else {
					auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
					submission  = std::move(queue.front());
					queue.pop_front();
					gpu->m_submission_count--;
					gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
					gpu->m_processing = true;
				}
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state = nullptr;
			return;
		}
		if (run_prefetch) {
			RenderActivity::Note("cp-prefetch-drain", 0, "-", "queueing compute prefetches");
			gpu->DrainComputePrefetches();
			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}
		if (!pending_prefetches.empty()) {
			RenderActivity::Note("cp-prefetch-scan", 0, "-", "scanning submitted commands");
			for (const auto& [queue_id, cmds]: pending_prefetches) {
				gpu->GetProcessor(queue_id).PrefetchComputeShaders(cmds);
			}
			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			continue;
		}

		if (command) {
			RenderActivity::Note("cp-command", 0, "-", "host command");
			{
				Common::DebugCounters::ScopedTimer busy(
				    Common::DebugCounters::Counter::GpuThreadBusyNs);
				command();
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		RenderActivity::Note("cp-process", 0, "-", "submission");
		WaitDiag::Global().NoteRunning(submission.queue_id, WaitDiag::NowMs());
		const bool complete = [&] {
			Common::DebugCounters::ScopedTimer busy(
			    Common::DebugCounters::Counter::GpuThreadBusyNs);
			return gpu->Process(submission);
		}();

		if (submission.type == SubmissionType::Commands) {
			using State = WaitDiag::QueueState;
			const bool for_shader = !complete && submission.execution.SuspendedForShader();
			WaitDiag::Global().NoteQueue(submission.queue_id,
			                             complete ? State::Idle
			                                      : (for_shader ? State::SuspendedForShader
			                                                    : State::WaitingMemory),
			                             for_shader ? RenderActivity::Read().hash : 0,
			                             WaitDiag::NowMs());
		}
		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	auto& cp = GetProcessor(submission.queue_id);

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Commands:
			if (!submission.prefetched) {
				submission.prefetched = true;
				cp.PrefetchComputeShaders(submission.commands);
			}
			complete =
			    cp.Process(submission.execution, submission.commands) == Pm4ProcessResult::Complete;
			if (submission.execution.MadeProgress()) {
				cp.BufferFlush();
			}
			break;
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			// Age caches at the frame boundary, not once per command list.
			m_renderer.RunGarbageCollector();
			cp.BufferFlush();
			cp.Reset();
			// Like completion labels, the point has executed once the CP has submitted every
			// packet before it, without waiting for the host GPU.
			m_suspend_point_ready.release();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended            = false;
	execution.m_suspended_for_shader = false;
	execution.m_made_progress        = false;

	struct ExecutionScope {
		explicit ExecutionScope(Pm4Execution& execution) { g_current_execution = &execution; }
		~ExecutionScope() { g_current_execution = nullptr; }
	} execution_scope(execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::SuspendPm4ForShader() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended            = true;
	g_current_execution->m_suspended_for_shader = true;
}

void CommandProcessor::PrefetchComputeShaders(std::span<const uint32_t> commands) {
	if (!Config::AsyncShaderCompileEnabled() || commands.empty()) {
		return;
	}
	constexpr uint32_t MaxPacketsScanned = 131072u;
	constexpr size_t   MaxIbDepth        = 4u;
	constexpr size_t   MaxPrefetches     = 1024u;

	struct ScanCursor {
		std::vector<uint32_t>     owned;
		std::span<const uint32_t> external;
		uint32_t                  offset_dw = 0;

		[[nodiscard]] std::span<const uint32_t> Words() const {
			return owned.empty() ? external : std::span<const uint32_t>(owned);
		}
	};
	std::vector<ScanCursor> stack;
	stack.reserve(MaxIbDepth);
	stack.push_back({{}, commands, 0u});

	HW::CsStageRegisters               cs_regs = m_sh_ctx.GetCs().cs_regs;
	std::vector<ComputeShaderPrefetch> discovered;
	std::vector<uint32_t>              indirect_pairs;
	uint32_t                           packets_scanned = 0;

	while (!stack.empty() && packets_scanned < MaxPacketsScanned &&
	       discovered.size() < MaxPrefetches) {
		auto&      cursor = stack.back();
		const auto words  = cursor.Words();
		if (cursor.offset_dw >= words.size()) {
			stack.pop_back();
			continue;
		}
		packets_scanned++;
		const auto* const packet        = words.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(words.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			continue;
		}
		if (remaining_dw < 2u) {
			stack.pop_back();
			continue;
		}
		uint32_t packet_dw = KYTY_PM4_LEN(packet_header);
		if (packet_dw < 2u || packet_dw > remaining_dw) {
			stack.pop_back();
			continue;
		}
		const auto  opcode     = (packet_header >> 8u) & 0xffu;
		const auto* body       = packet + 1;
		const auto  payload_dw = packet_dw - 1u;

		switch (opcode) {
			case Pm4::IT_SET_SH_REG: {
				const auto reg_offset = body[0];
				if (reg_offset != Pm4::SH_NOP && reg_offset < Pm4::SH_NUM) {
					for (uint32_t i = 1u; i < payload_dw; i++) {
						ApplyComputeShRegister(cs_regs, reg_offset + (i - 1u), body[i]);
					}
				}
				break;
			}
			case Pm4::IT_SET_SH_REG_INDIRECT: {
				if (packet_dw == 5u) {
					const uint64_t addr = (static_cast<uint64_t>(body[0]) & 0xfffffffcu) |
					                      (static_cast<uint64_t>(body[1]) << 32u);
					const uint32_t count = body[3] & 0x3fffu;
					const uint64_t bytes = static_cast<uint64_t>(count) * 8u;
					if (addr != 0 && count != 0 && m_renderer.IsMapped(addr, bytes)) {
						indirect_pairs.resize(static_cast<size_t>(count) * 2u);
						if (LibKernel::Memory::TryReadBacking(addr, indirect_pairs.data(), bytes)) {
							for (uint32_t i = 0; i < count; i++) {
								const auto raw_offset = indirect_pairs[i * 2u];
								if (raw_offset == 0xffffffffu) {
									continue;
								}
								ApplyComputeShRegister(cs_regs, raw_offset,
								                       indirect_pairs[i * 2u + 1u]);
							}
						}
					}
				}
				break;
			}
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				if (payload_dw >= 2u && (KYTY_PM4_R(packet_header) == 1u || IsAsyncComputeQueue()) &&
				    body[0] == Pm4::SQ_THREAD_TRACE_USERDATA_2) {
					const auto tag = body[1] & 0xff000000u;
					if (tag == Pm4::USERDATA_TAG_FLIP && remaining_dw >= Pm4::FLIP_PACKET_DW) {
						packet_dw = Pm4::FLIP_PACKET_DW;
					} else if (tag == Pm4::USERDATA_TAG_WAIT_FLIP_DONE &&
					           remaining_dw >= Pm4::WAIT_FLIP_DONE_PACKET_DW) {
						packet_dw = Pm4::WAIT_FLIP_DONE_PACKET_DW;
					} else if (tag == Pm4::USERDATA_TAG_RESET_QUEUE &&
					           (body[1] & Pm4::ResetQueueOp::ShState) != 0) {
						cs_regs = {};
					}
				}
				break;
			}
			case Pm4::IT_INDIRECT_BUFFER: {
				if (packet_dw == 4u) {
					const uint64_t ib_addr =
					    (body[0] & ~0x3u) | (static_cast<uint64_t>(body[1]) << 32u);
					const uint32_t ib_dw = body[2] & 0xfffffu;
					const bool     chain = (body[2] & (1u << 20u)) != 0;
					cursor.offset_dw += packet_dw;
					const uint64_t ib_bytes = static_cast<uint64_t>(ib_dw) * sizeof(uint32_t);
					if (ib_addr != 0 && ib_dw != 0 && ib_dw <= 0x40000u &&
					    m_renderer.IsMapped(ib_addr, ib_bytes)) {
						std::vector<uint32_t> ib_copy(ib_dw);
						if (LibKernel::Memory::TryReadBacking(ib_addr, ib_copy.data(), ib_bytes)) {
							if (chain) {
								cursor.owned     = std::move(ib_copy);
								cursor.external  = {};
								cursor.offset_dw = 0u;
							} else if (stack.size() < MaxIbDepth) {
								stack.push_back({std::move(ib_copy), {}, 0u});
							}
						}
					}
					continue;
				}
				break;
			}
			case Pm4::IT_DISPATCH_DIRECT: {
				if (packet_dw == 5u && cs_regs.data_addr != 0) {
					const uint32_t mode = body[3];
					auto           spec = cs_regs;
					spec.wave_size      = Pm4::ComputeWaveSize(mode);
					const bool use_thread_dims =
					    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
					discovered.push_back({spec, use_thread_dims});
				}
				break;
			}
			case Pm4::IT_DISPATCH_INDIRECT: {
				if ((packet_dw == 3u || packet_dw == 4u) && cs_regs.data_addr != 0) {
					const uint32_t mode = (packet_dw == 4u) ? body[2] : body[1];
					auto           spec = cs_regs;
					spec.wave_size      = Pm4::ComputeWaveSize(mode);
					const bool use_thread_dims =
					    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
					discovered.push_back({spec, use_thread_dims});
				}
				break;
			}
			default: break;
		}
		cursor.offset_dw += packet_dw;
	}

	if (discovered.empty()) {
		return;
	}
	Common::LockGuard lock(m_renderer.GetMutex());
	auto&             cache         = m_renderer.GetPipelineCache();
	const bool        async_compute = IsAsyncComputeQueue();
	for (const auto& req: discovered) {
		cache.PrefetchComputeProgram(req.cs_regs, req.dispatch_thread_dimensions, async_compute);
	}
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool debug_dump = graphics_debug_dump_enabled();
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		RenderActivity::Note("pm4", 0, "-", "packets");
		auto& cursor = execution.m_buffer_stack.back();
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		RenderActivity::NotePacket(opcode);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (debug_dump) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     cursor.offset_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = cursor.offset_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n", offset, packet_header);
		}

		if (!g_cp_op_keeps_draw_window[opcode]) {
			m_renderer.GetBufferCache().EndDrawWindow();
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances      = num_instances;
	m_num_instances_addr = 0;
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_addr != 0) {
		m_num_instances      = *reinterpret_cast<const uint32_t*>(m_num_instances_addr);
		m_num_instances_addr = 0;
	}
	return m_num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address) {
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			m_renderer.GetOcclusionQueries().Publish();
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
}

// Submits the recorded work every N draws so the host GPU starts before a long buffer ends.
void CommandProcessor::CountDraw() {
	if (m_flush_draws == 0) {
		return;
	}
	if (m_draws == m_flush_draws) {
		GetScheduler().Flush();
		m_draws = 0;
	}
	m_draws++;
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	CountDraw();
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawIndex(CurrentBuffer(), args);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}
	return reinterpret_cast<const void*>(m_index_base_addr + first_index * index_size);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index, uint32_t draw_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
	write(registers.draw_index, draw_index);
}

// Writes the GS user-data slots the packet patches; false when the shaders cannot read the
// offsets from GPU records: a first-index or subdraw-index destination, or another stage's
// user data.
static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets) {
	if (registers.index_offset != Pm4::SH_NOP || registers.draw_index != Pm4::SH_NOP) {
		return false;
	}
	for (const auto [location, slot]: {std::pair {registers.vertex_offset, &offsets.vertex},
	                                   std::pair {registers.instance_offset, &offsets.instance}}) {
		if (location == Pm4::SH_NOP) {
			continue;
		}
		const uint32_t index = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
		if (index >= HW::UserSgprInfo::SGPRS_MAX) {
			return false;
		}
		*slot = static_cast<uint8_t>(index);
	}
	return true;
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	CountDraw();
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);
	// The guest records match Vulkan's indirect commands byte for byte.
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	const auto draw_count = count_addr == nullptr
	                            ? max_count_or_count
	                            : std::min(uint32_t {*count_addr}, max_count_or_count);
	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto args_addr = m_draw_indirect_args_base_addr + data_offset;
	// NUM_INSTANCES keeps the last draw's instance count.
	m_num_instances_addr = args_addr + static_cast<uint64_t>(draw_count - 1) * stride_in_bytes +
	                       offsetof(vk::DrawIndirectCommand, instanceCount);
	// The GPU reads the records if the shaders allow; the VS prolog removes Vulkan's bases
	// from the index VGPRs and copies them only to the user SGPRs the packet patches.
	DrawIndirectArgs records {args_addr, draw_count, stride_in_bytes};
	if (IndirectDrawOffsets(registers, records.offsets)) {
		if (!indexed && executor.DrawAuto(CurrentBuffer(), {.indirect = records})) {
			return;
		}
		if (indexed && m_index_buffer_size != 0 &&
		    executor.DrawIndex(CurrentBuffer(), {.index_count         = m_index_buffer_size,
		                                         .index_addr          = IndexAddress(0),
		                                         .index_type_and_size = m_index_type_and_size,
		                                         .indirect            = records})) {
			return;
		}
	}

	const uint32_t max_indices = m_index_buffer_size != 0 ? m_index_buffer_size : UINT32_MAX;
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto addr = args_addr + static_cast<uint64_t>(i) * stride_in_bytes;
		if (!indexed) {
			const auto& args = *reinterpret_cast<const vk::DrawIndirectCommand*>(addr);
			PatchIndirectDrawOffsets(*this, registers, args.firstVertex, args.firstInstance, 0, i);
			executor.DrawAuto(CurrentBuffer(), {.vertex_count   = args.vertexCount,
			                                    .instance_count = args.instanceCount,
			                                    .first_vertex   = args.firstVertex,
			                                    .first_instance = args.firstInstance,
			                                    .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		const auto& args = *reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(addr);
		PatchIndirectDrawOffsets(*this, registers, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);
		executor.DrawIndex(CurrentBuffer(), {.index_count = std::min(args.indexCount, max_indices),
		                                     .index_addr  = IndexAddress(args.firstIndex),
		                                     .instance_count      = args.instanceCount,
		                                     .index_type_and_size = m_index_type_and_size,
		                                     .base_vertex         = args.vertexOffset,
		                                     .first_instance      = args.firstInstance,
		                                     .offset_source = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	const bool allow_async = g_current_execution != nullptr;
	if (!m_renderer.GetRenderExecutor().DispatchDirect(CurrentBuffer(), thread_group_x,
	                                                   thread_group_y, thread_group_z, mode,
	                                                   IsAsyncComputeQueue(), allow_async)) {
		SuspendPm4ForShader();
	}
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	EXIT_NOT_IMPLEMENTED(IsAsyncComputeQueue());
	m_renderer.GetRenderExecutor().ProcessPssr(CurrentBuffer(), request);
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		vk::DispatchIndirectCommand args;
		LibKernel::Memory::ReadGuestMemory(args_addr, &args, sizeof(args));
		DispatchDirect(args.x, args.y, args.z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	const bool allow_async = g_current_execution != nullptr;
	if (!m_renderer.GetRenderExecutor().DispatchIndirect(CurrentBuffer(), args_addr, mode,
	                                                     IsAsyncComputeQueue(), allow_async)) {
		SuspendPm4ForShader();
	}
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	CountDraw();
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(const uint64_t* label) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(label);
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id,
                                  bool gl2_writeback) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	// QueuedInterrupt: kNone, kOnly, kAfterWriteConfirm, kDataAfterWriteConfirm, kSendIntCtx.
	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	// kSendIntCtx has no label, and kOnly writes none on graphics queues.
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	// A GL2 writeback makes GPU writes CPU-visible even without a signal.
	if (gl2_writeback) {
		WriteBackImages();
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	// Dumped occlusion counters are written before any end-of-pipe signal reaches the guest.
	m_renderer.GetOcclusionQueries().Publish();

	// A label beside GPU-written bytes goes to the backing store, so they are not downloaded.
	// Other labels go through the guest mapping: its fault keeps a clean cached copy current and
	// downloads GPU-written bytes under the label before the store.
	const auto write_label = [this, address](const void* value, uint64_t size) {
		const auto vaddr   = reinterpret_cast<uint64_t>(address);
		uint64_t   noted   = 0;
		std::memcpy(&noted, value, size);
		NoteLabelWrite(address, noted, static_cast<uint32_t>(size));
		auto&      buffers = m_renderer.GetBufferCache();
		if (!GuestRange {vaddr, size}.Valid() || !buffers.IsRegionGpuModified(vaddr, size) ||
		    buffers.HasGpuDirtyBytes(vaddr, size) ||
		    !LibKernel::Memory::TryWriteBacking(vaddr, value, size)) {
			std::memcpy(address, value, size);
		}
	};
	// EndOfPipeData: kNone, kImmediate32b, kImmediate64b, kRefClk; 5 reads GDS.
	switch (data_sel) {
		case 0: break;
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			write_label(&value, sizeof(value));
			break;
		}
		case 3: data = Sync::ReadReferenceClock(); [[fallthrough]];
		case 2: write_label(&data, sizeof(data)); break;
		case 5: {
			GetScheduler().Finish();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	// GuestGpu::Process's end-of-slice flush submits the work this interrupt waits on.
	if (interrupt) {
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
			    renderer.TriggerInterrupt(event_id, interrupt_context_id);
		    });
	}
}

void CommandProcessor::WriteBackImages() {
	// GPU-written images are copied back without holding any signal: a held signal exposes the host
	// GPU's speed. Guest accesses to them wait for the copies.
	if (m_renderer.GetTextureCache().WriteBackImages()) {
		GetScheduler().Flush();
	}
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks only order GPU work, which the renderer's resource
	// barriers already order.
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (auto& queries = m_renderer.GetOcclusionQueries(); queries.Enabled()) {
				queries.Dump(event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			// Without host occlusion queries, publish an always-visible result. The PS5 layout
			// contains one interleaved begin/end pair per DB, and bit 63 marks a result ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}
	m_renderer.GetOcclusionQueries().Publish();

	auto& command = CurrentBuffer();
	auto  request = Sync::PrepareVideoOutFlip(command, handle, index, flip_mode, flip_arg);
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_state != nullptr;
}

} // namespace Libs::Graphics
