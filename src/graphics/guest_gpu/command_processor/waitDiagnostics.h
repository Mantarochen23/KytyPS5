#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_WAITDIAGNOSTICS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_WAITDIAGNOSTICS_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::WaitDiag {

// What a command queue is waiting on, for the stall reporter. The command processor notes a wait
// packet each time it polls it, every label it writes and every GPU-to-guest readback; the
// reporter reads it only while no frame is recorded. Nothing here reads guest memory: the value
// shown is the one the command processor read at its last poll.

constexpr uint32_t QueueSlots   = 64;
constexpr uint32_t LabelRingLen = 32;
constexpr uint32_t ReadbackLen  = 32;
constexpr uint64_t NearBytes    = 4096;

enum class QueueState : uint32_t {
	Unused,
	Idle,
	Running,
	WaitingMemory,
	SuspendedForShader,
};

struct LabelWrite {
	uint64_t address = 0;
	uint64_t value   = 0;
	uint32_t size    = 0;
	uint32_t queue   = 0;
	uint64_t ms      = 0;
};

struct Readback {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t ms      = 0;
};

struct WaitPacket {
	bool     valid        = false;
	bool     is64         = false;
	uint32_t func         = 0;
	uint64_t address      = 0;
	uint64_t mask         = 0;
	uint64_t reference    = 0;
	uint64_t value        = 0; // read at the last poll
	uint64_t buffer       = 0; // address of the command buffer holding the packet
	uint32_t offset_dw    = 0;
	uint64_t polls        = 0;
	uint64_t first_ms     = 0;
	uint64_t last_ms      = 0;
	int      gpu_modified = -1; // -1 not asked, 0 no, 1 yes: a shader wrote the range
	int      gpu_dirty    = -1; // -1 not asked, 0 no, 1 yes: written bytes not read back yet
};

struct QueueInfo {
	QueueState state = QueueState::Unused;
	uint64_t   hash  = 0; // the shader a queue is suspended for
	uint64_t   since_ms = 0;
	WaitPacket wait;
};

[[nodiscard]] inline const char* CompareName(uint32_t func) {
	switch (func) {
		case 0: return "always";
		case 1: return "<";
		case 2: return "<=";
		case 3: return "==";
		case 4: return "!=";
		case 5: return ">=";
		case 6: return ">";
		default: return "?";
	}
}

[[nodiscard]] inline std::string QueueName(uint32_t slot) {
	char text[32];
	if (slot == 0) {
		std::snprintf(text, sizeof(text), "0 (draw)");
	} else {
		std::snprintf(text, sizeof(text), "0x%02x (compute)", 0x20u + slot - 1u);
	}
	return text;
}

__attribute__((format(printf, 1, 2))) inline std::string Fmt(const char* format, ...) {
	char    text[512];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	return text;
}

class Registry {
public:
	void NoteLabel(uint64_t address, uint64_t value, uint32_t size, uint32_t queue, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		m_labels[m_label_count % LabelRingLen] = {address, value, size, queue, ms};
		m_label_count++;
	}

	void NoteReadback(uint64_t address, uint64_t size, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		m_readbacks[m_readback_count % ReadbackLen] = {address, size, ms};
		m_readback_count++;
	}

	// One poll of a wait packet; returns how often this same packet was polled in a row.
	uint64_t NotePoll(uint32_t slot, const WaitPacket& packet, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		auto&            info = m_queues[slot % QueueSlots];
		auto&            wait = info.wait;
		const bool same = wait.valid && wait.buffer == packet.buffer &&
		                  wait.offset_dw == packet.offset_dw && wait.address == packet.address &&
		                  wait.reference == packet.reference;
		const auto polls     = same ? wait.polls + 1 : 1;
		const auto first_ms  = same ? wait.first_ms : ms;
		const auto modified  = same ? wait.gpu_modified : -1;
		const auto dirty     = same ? wait.gpu_dirty : -1;
		wait                 = packet;
		wait.valid           = true;
		wait.polls           = polls;
		wait.first_ms        = first_ms;
		wait.last_ms         = ms;
		wait.gpu_modified    = packet.gpu_modified >= 0 ? packet.gpu_modified : modified;
		wait.gpu_dirty       = packet.gpu_dirty >= 0 ? packet.gpu_dirty : dirty;
		return polls;
	}

	// A queue starts a slice; a queue already known to wait keeps its state and start time.
	void NoteRunning(uint32_t slot, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		auto&            info = m_queues[slot % QueueSlots];
		if (info.state == QueueState::Unused || info.state == QueueState::Idle) {
			info.state    = QueueState::Running;
			info.hash     = 0;
			info.since_ms = ms;
		}
	}

	void NoteQueue(uint32_t slot, QueueState state, uint64_t hash, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		auto&            info = m_queues[slot % QueueSlots];
		if (info.state != state || info.hash != hash) {
			info.since_ms = ms;
		}
		info.state = state;
		info.hash  = hash;
		if (state != QueueState::WaitingMemory) {
			info.wait.valid = false;
		}
	}

	// Lines for the stall reporter; empty when no queue waits on memory and none is known.
	[[nodiscard]] std::vector<std::string> Report(uint64_t now_ms) const {
		std::scoped_lock         lock(m_mutex);
		std::vector<std::string> lines;
		std::vector<uint64_t>    addresses;
		for (uint32_t slot = 0; slot < QueueSlots; slot++) {
			const auto& info = m_queues[slot];
			if (info.state == QueueState::Unused) {
				continue;
			}
			const double age = static_cast<double>(now_ms - std::min(now_ms, info.since_ms)) / 1000.0;
			switch (info.state) {
				case QueueState::Idle:
					lines.push_back(Fmt("[wait] queue %s: idle for %.1f s", QueueName(slot).c_str(), age));
					break;
				case QueueState::Running:
					lines.push_back(Fmt("[wait] queue %s: running for %.1f s", QueueName(slot).c_str(), age));
					break;
				case QueueState::SuspendedForShader:
					lines.push_back(Fmt("[wait] queue %s: suspended for shader 0x%016" PRIx64 " for %.1f s",
					                    QueueName(slot).c_str(), info.hash, age));
					break;
				case QueueState::WaitingMemory: {
					const auto& w = info.wait;
					if (!w.valid || now_ms - std::min(now_ms, w.last_ms) > 1500) {
						lines.push_back(Fmt("[wait] queue %s: blocked for %.1f s, not on a wait_reg_mem "
						                    "packet (flip, semaphore or predication)",
						                    QueueName(slot).c_str(), age));
						break;
					}
					const double wait_age =
					    static_cast<double>(now_ms - std::min(now_ms, w.first_ms)) / 1000.0;
					lines.push_back(Fmt(
					    "[wait] queue %s: wait_reg_mem%d on memory 0x%016" PRIx64 " mask 0x%" PRIx64
					    " compare (value & mask) %s 0x%" PRIx64 ", value at last poll 0x%" PRIx64
					    "; packet at 0x%016" PRIx64 " + %u dw; polled %" PRIu64 " times in %.1f s",
					    QueueName(slot).c_str(), w.is64 ? 64 : 32, w.address, w.mask,
					    CompareName(w.func), w.reference, w.value, w.buffer, w.offset_dw, w.polls,
					    wait_age));
					const auto yes_no = [](int value) {
						return value < 0 ? "unknown" : (value != 0 ? "yes" : "no");
					};
					lines.push_back(Fmt("[wait]   range written by a shader: %s; written bytes not read "
					                    "back yet: %s",
					                    yes_no(w.gpu_modified), yes_no(w.gpu_dirty)));
					addresses.push_back(w.address);
					break;
				}
				default: break;
			}
		}
		for (const auto address: addresses) {
			AppendNear(lines, address, now_ms);
		}
		return lines;
	}

	void Clear() {
		std::scoped_lock lock(m_mutex);
		m_queues         = {};
		m_labels         = {};
		m_readbacks      = {};
		m_label_count    = 0;
		m_readback_count = 0;
	}

private:
	static bool Near(uint64_t address, uint64_t size, uint64_t waited) {
		const auto lo = waited >= NearBytes ? waited - NearBytes : 0;
		return address + size > lo && address <= waited + NearBytes;
	}

	void AppendNear(std::vector<std::string>& lines, uint64_t waited, uint64_t now_ms) const {
		const auto count = std::min<uint64_t>(m_label_count, LabelRingLen);
		uint64_t   shown = 0;
		for (uint64_t i = 0; i < count; i++) {
			// Newest first.
			const auto& l = m_labels[(m_label_count - 1 - i) % LabelRingLen];
			if (!Near(l.address, l.size, waited)) {
				continue;
			}
			shown++;
			lines.push_back(Fmt("[wait]   label write 0x%016" PRIx64 " <- 0x%" PRIx64
			                    " (%u bytes) by queue %s, %.2f s ago%s",
			                    l.address, l.value, l.size, QueueName(l.queue).c_str(),
			                    static_cast<double>(now_ms - std::min(now_ms, l.ms)) / 1000.0,
			                    l.address == waited ? " [the waited address]" : ""));
		}
		if (shown == 0) {
			const auto oldest = count == 0 ? 0 : m_labels[(m_label_count - count) % LabelRingLen].ms;
			lines.push_back(Fmt("[wait]   no label write within 4 KiB of 0x%016" PRIx64
			                    " among the last %" PRIu64 " (%" PRIu64 " in all, oldest %.1f s ago)",
			                    waited, count, m_label_count,
			                    static_cast<double>(now_ms - std::min(now_ms, oldest)) / 1000.0));
		}
		const auto readbacks = std::min<uint64_t>(m_readback_count, ReadbackLen);
		uint64_t   covering  = 0;
		for (uint64_t i = 0; i < readbacks; i++) {
			const auto& r = m_readbacks[(m_readback_count - 1 - i) % ReadbackLen];
			if (r.address <= waited && waited < r.address + r.size) {
				covering++;
				lines.push_back(Fmt("[wait]   read back 0x%016" PRIx64 " + 0x%" PRIx64 ", %.2f s ago",
				                    r.address, r.size,
				                    static_cast<double>(now_ms - std::min(now_ms, r.ms)) / 1000.0));
				break;
			}
		}
		if (covering == 0) {
			lines.push_back(Fmt("[wait]   no recent read back covers 0x%016" PRIx64
			                    " (last %" PRIu64 " read backs of %" PRIu64 " in all)",
			                    waited, readbacks, m_readback_count));
		}
	}

	mutable std::mutex                    m_mutex;
	std::array<QueueInfo, QueueSlots>     m_queues {};
	std::array<LabelWrite, LabelRingLen>  m_labels {};
	std::array<Readback, ReadbackLen>     m_readbacks {};
	uint64_t                              m_label_count    = 0;
	uint64_t                              m_readback_count = 0;
};

[[nodiscard]] inline Registry& Global() {
	static Registry registry;
	return registry;
}

[[nodiscard]] inline uint64_t NowMs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace Libs::Graphics::WaitDiag

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_WAITDIAGNOSTICS_H_
