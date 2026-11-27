#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOWPLAN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOWPLAN_H_

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

// The decisions that keep frames flowing while shaders build, without Vulkan so they can be tested:
// how long a compute dispatch waits for its shader (--async-compute-wait-ms), what is counted when
// it gives up, and the once-per-hash accounting of cache hits.
namespace Libs::Graphics::FlowPlan {

enum class WaitAction : uint8_t {
	// Keep the queue suspended and try again later.
	Wait,
	// Stop waiting: the dispatch is skipped for this use.
	Skip,
};

// The wait of one suspended dispatch. Lives with the queue's execution state; the dispatch is
// retried until its shader is ready, so the first retry starts the clock.
struct DispatchWait {
	bool     waiting  = false;
	uint64_t since_ms = 0;

	struct Result {
		WaitAction action    = WaitAction::Wait;
		uint64_t   waited_ms = 0;
	};

	// The dispatch found its shader not ready. `limit_ms` 0 waits for ever.
	Result NotReady(uint32_t limit_ms, uint64_t now_ms) {
		if (!waiting) {
			waiting  = true;
			since_ms = now_ms;
		}
		const uint64_t waited = now_ms >= since_ms ? now_ms - since_ms : 0;
		if (limit_ms != 0 && waited >= limit_ms) {
			waiting = false;
			return {WaitAction::Skip, waited};
		}
		return {WaitAction::Wait, waited};
	}
	// The dispatch ran, or the queue moved past it.
	void Done() { waiting = false; }
};

// What was skipped. A use is one skipped dispatch of a shader.
class SkipLedger {
public:
	// Returns the use number of this shader (1 for the first skip, which is the one that is logged).
	uint64_t Note(uint64_t hash, uint64_t waited_ms) {
		auto& entry = m_by_hash[hash];
		entry.uses++;
		m_total++;
		m_period++;
		m_waited_ms += waited_ms;
		return entry.uses;
	}
	// Skips since the last call, for the per-2-s counter.
	uint64_t TakePeriod() {
		const auto value = m_period;
		m_period         = 0;
		return value;
	}
	[[nodiscard]] uint64_t Total() const { return m_total; }
	[[nodiscard]] uint64_t Hashes() const { return m_by_hash.size(); }
	[[nodiscard]] uint64_t WaitedMs() const { return m_waited_ms; }
	[[nodiscard]] uint64_t Uses(uint64_t hash) const {
		const auto found = m_by_hash.find(hash);
		return found == m_by_hash.end() ? 0 : found->second.uses;
	}

private:
	struct Entry {
		uint64_t uses = 0;
	};
	std::unordered_map<uint64_t, Entry> m_by_hash;
	uint64_t                            m_total     = 0;
	uint64_t                            m_period    = 0;
	uint64_t                            m_waited_ms = 0;
};

// Counts uses per hash; the first use of a hash is the one that is announced.
class UseCounter {
public:
	// Returns the use number, 1 for the first.
	uint64_t Note(uint64_t hash) {
		m_uses++;
		return ++m_by_hash[hash];
	}
	[[nodiscard]] uint64_t Hashes() const { return m_by_hash.size(); }
	[[nodiscard]] uint64_t Uses() const { return m_uses; }

private:
	std::unordered_map<uint64_t, uint64_t> m_by_hash;
	uint64_t                               m_uses = 0;
};

// A lookup is counted once per request, not once per retry of a suspended dispatch.
class RequestCounter {
public:
	// True when this call starts a request; false when it retries one that did not finish.
	bool Begin(uint64_t identity) { return m_open.erase(identity) == 0; }
	// The request ended without a result (it will be retried).
	void Retry(uint64_t identity) { m_open.insert(identity); }
	[[nodiscard]] size_t Open() const { return m_open.size(); }

private:
	std::unordered_set<uint64_t> m_open;
};

// The compute user data a scan of the command stream has seen set, without executing it. A value
// is only trusted if the scan itself saw it written, so a dispatch whose user data were set by an
// earlier command buffer is not guessed.
class UserDataTracker {
public:
	static constexpr uint32_t Slots = 16;

	// A write of `value` to the SH register `reg` (type bits in 0x70000000 are ignored); true when
	// it was one of the compute user data registers that start at `first_reg`.
	bool Note(uint32_t reg, uint32_t value, uint32_t first_reg) {
		reg &= ~0x70000000u;
		if (reg < first_reg || reg - first_reg >= Slots) {
			return false;
		}
		const auto slot = reg - first_reg;
		m_values[slot]  = value;
		m_mask |= 1u << slot;
		return true;
	}
	// The queue's shader state was reset.
	void Reset() { *this = UserDataTracker {}; }
	// Whether the first `count` slots were all written by the scan.
	[[nodiscard]] bool Covers(uint32_t count) const {
		if (count > Slots) {
			return false;
		}
		const uint32_t need = count == 32 ? ~0u : (1u << count) - 1u;
		return (m_mask & need) == need;
	}
	[[nodiscard]] uint32_t Value(uint32_t slot) const { return m_values[slot]; }
	[[nodiscard]] const uint32_t* Values() const { return m_values; }

private:
	uint32_t m_values[Slots] {};
	uint32_t m_mask = 0;
};

} // namespace Libs::Graphics::FlowPlan

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOWPLAN_H_
