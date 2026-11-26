#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <atomic>
#include <fmt/format.h>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLED g_print_name

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLE(flag) PRINT_NAME_ENABLED = flag;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_DEFINE(name) void name(Loader::SymbolDatabase* s)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_NAME(l, m)                                                                             \
	[[maybe_unused]] static thread_local bool PRINT_NAME_ENABLED = false;                          \
	static constexpr char                     g_library[]        = l;                              \
	static constexpr char                     g_module[]         = m;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_ADD(l, n, f, t) s->Add({n, l, g_module, t}, reinterpret_cast<uint64_t>(f), "" #f)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_OBJECT(n, f) LIB_ADD(g_library, n, f, Loader::SymbolType::Object)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f) LIB_ADD(g_library, n, f, Loader::SymbolType::Func)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC_IN(l, n, f) LIB_ADD(l, n, f, Loader::SymbolType::Func)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	if (PRINT_NAME_ENABLED) {                                                                      \
		Libs::PrintName(g_library, g_module, __func__);                                              \
	} else if (Libs::g_trace_calls) {                                                              \
		/* --trace-calls: calls 1-4 of each function, then every power of two, with the count. */   \
		static std::atomic<uint64_t> trace_calls {0};                                                \
		const auto trace_count = trace_calls.fetch_add(1, std::memory_order_relaxed) + 1;           \
		if (trace_count <= 4 || (trace_count & (trace_count - 1)) == 0) {                            \
			Libs::PrintName(g_library, g_module, __func__, trace_count);                             \
		}                                                                                            \
	}

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
inline bool g_trace_calls = false; // Set once at start from --trace-calls.

[[gnu::noinline]] inline void PrintName(const char* library, const char* module, const char* function,
                                        uint64_t call_count = 0) {
	if (call_count != 0) {
		// Straight to stdout: guest logging may be silent or throttled while tracing.
		std::printf("[%d][trace] %s::%s::%s() #%llu\n", Common::Thread::GetThreadIdUnique(), library,
		            module, function, static_cast<unsigned long long>(call_count));
		return;
	}
	if (!Log::IsSilent()) {
		const auto elapsed_ms      = static_cast<uint64_t>(Loader::Timer::GetTimeMs());
		const auto print_name_time = fmt::format(
		    "{:02}:{:02}:{:02}.{:03}", elapsed_ms / 3600000, (elapsed_ms / 60000) % 60,
		    (elapsed_ms / 1000) % 60, elapsed_ms % 1000);
		LOGF_COLOR(Log::Color::Cyan, "[%d][%s] %s::%s::%s()\n",
		           Common::Thread::GetThreadIdUnique(), print_name_time.c_str(), library, module,
		           function);
	}
}

void InitAll(Loader::SymbolDatabase* s);

} // namespace Libs
#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_ */
