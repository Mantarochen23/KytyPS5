#ifndef KYTY_COMMON_GUEST_PARK_H_
#define KYTY_COMMON_GUEST_PARK_H_

// Bookkeeping for guest threads parked after an unresolvable fault (--guest-fault-park).
//
// Add() runs inside the fault handler, so it takes no locks that other code can hold: a tiny spin
// lock that is only ever taken here, for a handful of instructions, by threads that are about to
// park for good. Count() and the totals are plain atomics.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

namespace Common::GuestPark {

struct Record {
	uint64_t pc        = 0;
	uint64_t address   = 0;
	int32_t  thread_id = 0;
	char     name[32]  = {};
};

class Registry {
public:
	static constexpr size_t kCapacity = 256;

	struct Result {
		uint32_t count;       // parked threads including this one
		bool     first_at_pc; // no earlier parked thread faulted at the same pc
		uint32_t at_pc;       // parked threads at this pc including this one
	};

	Result Add(uint64_t pc, uint64_t address, int32_t thread_id, const char* name) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
		}
		const auto stored = m_stored.load(std::memory_order_relaxed);
		uint32_t   at_pc  = 1;
		for (size_t i = 0; i < stored; i++) {
			at_pc += m_records[i].pc == pc ? 1u : 0u;
		}
		if (stored < kCapacity) {
			auto& record     = m_records[stored];
			record.pc        = pc;
			record.address   = address;
			record.thread_id = thread_id;
			std::strncpy(record.name, name != nullptr ? name : "", sizeof(record.name) - 1);
			m_stored.store(stored + 1, std::memory_order_relaxed);
		}
		// More than kCapacity parked threads are still counted, just no longer listed per pc.
		const auto count = m_count.fetch_add(1, std::memory_order_relaxed) + 1u;
		m_lock.clear(std::memory_order_release);
		return {count, at_pc == 1, at_pc};
	}

	[[nodiscard]] uint32_t Count() const { return m_count.load(std::memory_order_relaxed); }

private:
	std::atomic_flag      m_lock = ATOMIC_FLAG_INIT;
	std::atomic<uint32_t> m_count {0};
	std::atomic<size_t>   m_stored {0};
	Record                m_records[kCapacity] {};
};

inline Registry& Global() {
	static Registry registry;
	return registry;
}

// Blocks the calling thread for good. Sleeps in a loop so the thread stays killable and the
// process can exit around it; it holds no lock of its own.
[[noreturn]] inline void ParkForever() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

} // namespace Common::GuestPark

#endif /* KYTY_COMMON_GUEST_PARK_H_ */
