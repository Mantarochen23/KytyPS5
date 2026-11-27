#ifndef KYTY_COMMON_FAULT_REPORT_H_
#define KYTY_COMMON_FAULT_REPORT_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>

// Host-side context for a fatal fault: which module and symbol the pc and each return address are
// in, a backtrace of the faulting thread, the thread's name and what it was doing. The tables are
// built outside the handler; the handler itself allocates nothing, takes no lock of its own and
// writes through a caller-supplied sink (stdio in the emulator, a buffer in the test).
namespace Common::FaultReport {

// Notes what the calling host thread does, for the fault report. `tag` must be a string literal.
// With `name` the host thread is also given that name (at most 15 characters).
void SetThreadTag(const char* tag, const char* name = nullptr) noexcept;

// Sets a per-thread activity detail printed alongside the thread tag on a fault (for example the
// shader hash, stage and word count a worker is compiling). Passing nullptr or "" clears it.
// Allocation-free and signal-safe.
void SetThreadActivity(const char* detail) noexcept;

// RAII guard for SetThreadActivity; restores the previous activity on exit.
class ScopedActivity {
public:
	explicit ScopedActivity(const char* detail) noexcept;
	~ScopedActivity() noexcept;
	ScopedActivity(const ScopedActivity&)            = delete;
	ScopedActivity& operator=(const ScopedActivity&) = delete;

private:
	char m_previous[160] {};
	bool m_had_previous = false;
};

// Reads the loaded modules (and, once, the symbol table of the main executable). Call at startup
// and after a dlopen; faults at an address no table knows fall back to reading /proc/self/maps.
void RefreshModules();

// Remembers the calling thread's stack limits so the frame-pointer walk stays inside them.
void RegisterThreadStack();

struct Sink {
	void (*write)(void* user, const char* text) = nullptr;
	void* user                                  = nullptr;
};

struct Registers {
	uint64_t pc  = 0;
	uint64_t rsp = 0;
	uint64_t rbp = 0;
};

// "module+0xoffset [symbol+0xoffset]" or "not in a loaded module". Signal-safe.
void FormatAddress(uint64_t address, char* out, size_t size);

// Prints the thread name and tag, the pc, the frame-pointer chain, the unwinder backtrace and
// each stack word that points into code, every address as module+offset [symbol]. Signal-safe.
void WriteHostContext(const Registers& regs, const Sink& sink);

} // namespace Common::FaultReport

#endif /* KYTY_COMMON_FAULT_REPORT_H_ */
