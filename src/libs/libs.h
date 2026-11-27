#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/fpEnv.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <type_traits>
#include <vector>

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
#define LIB_FUNC(n, f)                                                                             \
	s->Add({n, g_library, g_module, Loader::SymbolType::Func},                                     \
	       Libs::MakeLibThunk<f, Libs::FixedString(g_library), Libs::FixedString("" #f)>(),       \
	       "" #f)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC_IN(l, n, f)                                                                       \
	s->Add({n, l, g_module, Loader::SymbolType::Func},                                             \
	       Libs::MakeLibThunk<f, Libs::FixedString(l), Libs::FixedString("" #f)>(), "" #f)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	if (PRINT_NAME_ENABLED) {                                                                      \
		Libs::PrintName(g_library, g_module, __func__);                                              \
	} else if (Libs::g_trace_calls && !Libs::g_trace_calls_filtered) {                             \
		/* --trace-calls: calls 1-4 of each function, then every power of two, with the count. */   \
		static std::atomic<uint64_t> trace_calls {0};                                                \
		const auto trace_count = trace_calls.fetch_add(1, std::memory_order_relaxed) + 1;           \
		if (Libs::ShouldLogPowerOfTwo(trace_count)) {                                               \
			Libs::PrintName(g_library, g_module, __func__, trace_count);                             \
		}                                                                                            \
	}

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
inline bool                     g_trace_calls          = false; // Set once at start from --trace-calls.
inline bool                     g_trace_calls_filtered = false; // True when --trace-calls=<lib,...> is active.
inline std::vector<std::string> g_trace_calls_filter;

constexpr size_t kCallRingSize = 256;

struct CallRecord {
	const char* library        = nullptr;
	const char* function       = nullptr;
	uint64_t    args[6]        = {};
	uint64_t    result         = 0;
	uint64_t    guest_ret_addr = 0;
	bool        completed      = false;
};

struct CallRing {
	std::array<CallRecord, kCallRingSize> entries {};
	size_t                                next  = 0;
	size_t                                count = 0;

	CallRecord& Push(const char* library, const char* function, const uint64_t (&args)[6],
	                 uint64_t guest_ret_addr) {
		auto& entry          = entries[next];
		entry.library        = library;
		entry.function       = function;
		std::copy_n(args, 6, entry.args);
		entry.result         = 0;
		entry.guest_ret_addr = guest_ret_addr;
		entry.completed      = false;
		next                 = (next + 1) % kCallRingSize;
		if (count < kCallRingSize) {
			count++;
		}
		return entry;
	}

	template <typename Fn>
	void ForEachOldestFirst(Fn&& fn) const {
		const size_t start = (count < kCallRingSize) ? 0 : next;
		for (size_t i = 0; i < count; i++) {
			fn(entries[(start + i) % kCallRingSize], i);
		}
	}
};

constexpr bool ShouldLogPowerOfTwo(uint64_t count) {
	return count != 0 && (count <= 4 || (count & (count - 1)) == 0);
}

const CallRing& GetThreadCallRing();
CallRecord&     RecordThreadCallEntry(const char* library, const char* function,
                                      const uint64_t (&args)[6], uint64_t guest_ret_addr);
bool            MatchesTraceFilter(const char* library);
bool            ShouldLogFailedCall(const void* fn_key, uint64_t result);
void            LogFilteredCall(const char* library, const char* function, const uint64_t (&args)[6],
                                uint64_t result);
void            LogFailedCall(const char* library, const char* function, const uint64_t (&args)[6],
                              uint64_t result);
// One `[fpenv]` line: the FP control state `env` differs from `expected` at `site` in a call of
// library::function (Common::FpControlFirstDeviation decides that it is the first on the thread).
void ReportCallFpEnv(Common::FpCheckSite site, const char* library, const char* function,
                     Common::FpEnv env, Common::FpEnv expected);
// A library function declared to return float or double returned: counted, and the first zero or
// non-finite result of each function is logged (`[fp-return]`): a stub that returns 0.0 to a
// caller that divides by it gives the guest an infinity.
void NoteFloatReturn(const void* fn_key, const char* library, const char* function, double value);
uint64_t FloatReturnCalls();

// Around every library call (default on, two control-word reads per call): the guest's state on
// entry, and whether the call returned with another one.
inline void CheckCallEntryFpEnv(const char* library, const char* function, Common::FpEnv env) {
	if (Common::FpControlFirstDeviation(Common::FpCheckSite::LibraryEntry, env)) {
		ReportCallFpEnv(Common::FpCheckSite::LibraryEntry, library, function, env,
		                Common::GuestFpEnv);
	}
}
inline void CheckCallExitFpEnv(const char* library, const char* function, Common::FpEnv before) {
	const auto after = Common::ReadFpEnv();
	if (Common::FpControlFirstDeviation(Common::FpCheckSite::LibraryChanged, after, before)) {
		ReportCallFpEnv(Common::FpCheckSite::LibraryChanged, library, function, after, before);
	}
}

template <size_t N>
struct FixedString {
	char data[N] {};
	consteval FixedString(const char (&str)[N]) {
		for (size_t i = 0; i < N; i++) {
			data[i] = str[i];
		}
	}
};

template <typename T>
void PushSysvIntArg(uint64_t (&out)[6], size_t& idx, const T& arg) {
	if (idx >= 6) {
		return;
	}
	if constexpr (std::is_pointer_v<T>) {
		out[idx++] = reinterpret_cast<uint64_t>(arg);
	} else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
		out[idx++] = static_cast<uint64_t>(arg);
	}
}

template <typename Ret>
bool IsFailureReturn(const Ret& value, uint64_t& raw_result) {
	if constexpr (std::is_pointer_v<Ret>) {
		raw_result = reinterpret_cast<uint64_t>(value);
		return value == nullptr;
	} else if constexpr (std::is_integral_v<Ret> && !std::is_same_v<Ret, bool>) {
		if constexpr (sizeof(Ret) <= 4) {
			const auto u32 = static_cast<uint32_t>(value);
			raw_result     = static_cast<uint64_t>(u32);
			return u32 >= 0x80000000u;
		} else {
			raw_result = static_cast<uint64_t>(value);
			return static_cast<int64_t>(raw_result) < 0 ||
			       (raw_result >= 0x80000000ull && raw_result <= 0xffffffffull);
		}
	} else {
		raw_result = 0;
		return false;
	}
}

template <auto Fn, FixedString Library, FixedString Name, typename Sig>
struct LibThunkHelper;

template <auto Fn, FixedString Library, FixedString Name, typename Ret, typename... Args>
struct LibThunkHelper<Fn, Library, Name, Ret (KYTY_SYSV_ABI*)(Args...)> {
	static KYTY_SYSV_ABI Ret Call(Args... args) {
		uint64_t raw_args[6] {};
		size_t   arg_idx = 0;
		(PushSysvIntArg(raw_args, arg_idx, args), ...);
		const auto guest_ret = reinterpret_cast<uint64_t>(__builtin_return_address(0));
		const auto fp_before = Common::ReadFpEnv();
		auto&      record    = RecordThreadCallEntry(Library.data, Name.data, raw_args, guest_ret);
		CheckCallEntryFpEnv(Library.data, Name.data, fp_before);
		const bool trace_all = g_trace_calls_filtered && MatchesTraceFilter(Library.data);
		if constexpr (std::is_void_v<Ret>) {
			Fn(args...);
			CheckCallExitFpEnv(Library.data, Name.data, fp_before);
			record.completed = true;
			if (trace_all) {
				LogFilteredCall(Library.data, Name.data, raw_args, 0);
			}
		} else {
			Ret      ret        = Fn(args...);
			CheckCallExitFpEnv(Library.data, Name.data, fp_before);
			if constexpr (std::is_floating_point_v<Ret>) {
				NoteFloatReturn(reinterpret_cast<const void*>(&Call), Library.data, Name.data,
				                static_cast<double>(ret));
			}
			uint64_t raw_result = 0;
			const bool failed   = IsFailureReturn(ret, raw_result);
			record.result       = raw_result;
			record.completed    = true;
			if (trace_all) {
				LogFilteredCall(Library.data, Name.data, raw_args, raw_result);
			}
			if (failed && ShouldLogFailedCall(reinterpret_cast<const void*>(&Call), raw_result)) {
				LogFailedCall(Library.data, Name.data, raw_args, raw_result);
			}
			return ret;
		}
	}

	static uint64_t Addr() { return reinterpret_cast<uint64_t>(&Call); }
};

template <auto Fn, FixedString Library, FixedString Name, typename Ret, typename... Args>
struct LibThunkHelper<Fn, Library, Name, Ret (KYTY_SYSV_ABI*)(Args..., ...)> {
	static uint64_t Addr() { return reinterpret_cast<uint64_t>(Fn); }
};

template <auto Fn, FixedString Library, FixedString Name>
uint64_t MakeLibThunk() {
	return LibThunkHelper<Fn, Library, Name, decltype(Fn)>::Addr();
}

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
