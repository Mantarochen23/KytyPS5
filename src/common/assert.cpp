#include "common/assert.h"

#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>

#if defined(__linux__)
#include <execinfo.h>
#include <unistd.h>
#endif

namespace Common {

static thread_local RecoverHook t_recover_hook = nullptr;

RecoverHook SetThreadRecoverHook(RecoverHook hook) {
	auto* previous = t_recover_hook;
	t_recover_hook = hook;
	return previous;
}

[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line) {
	if (auto* hook = t_recover_hook) {
		hook(text, file, line);
	}
	Log::WriteToConsoleAndLog(fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL,
	                                      title, text, file, line));
#if defined(__linux__)
	{
		// Host call chain of the failing thread; resolve with addr2line -e kyty_emulator <offset>.
		void*      frames[48];
		const auto count = backtrace(frames, 48);
		std::fflush(nullptr);
		(void)!write(STDOUT_FILENO, "--- Host backtrace ---\n", 23);
		backtrace_symbols_fd(frames, count, STDOUT_FILENO);
	}
#endif
	if (const char* core = std::getenv("KYTY_FATAL_ABORT"); core != nullptr && core[0] == '1') {
		std::fflush(nullptr);
		std::abort(); // Leaves a core dump for the debugger; skips the orderly shutdown.
	}
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(321);
}

void DbgExitIfHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr), file,
	          line);
}

void DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

void DbgExitHandler(const char* file, int line, std::string_view text) {
	DbgReport("--- Error ---", text, file, line);
}

} // namespace Common
