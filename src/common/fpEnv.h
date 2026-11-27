#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include "common/common.h"

#include <cstdint>

namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

// What every guest thread starts with: MXCSR flush-to-zero, denormals-are-zero, round to nearest,
// all exceptions masked; x87 64-bit precision, round to nearest, all exceptions masked.
inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

inline FpEnv ReadFpEnv() {
	FpEnv env {};
	asm volatile("stmxcsr %0\n\tfnstcw %1" : "=m"(env.mxcsr), "=m"(env.control) : : "memory");
	return env;
}

inline void WriteFpEnv(FpEnv env) {
	asm volatile("ldmxcsr %0\n\tfldcw %1" : : "m"(env.mxcsr), "m"(env.control) : "memory");
}

// The chosen part of the state, without what computation leaves behind: MXCSR without its six
// sticky exception flags (bits 0-5); the x87 control word's masks (0-5), precision (8-9) and
// rounding (10-11), without the reserved and infinity-control bits.
[[nodiscard]] constexpr uint32_t MxcsrControl(uint32_t mxcsr) {
	return mxcsr & 0xffc0u;
}
[[nodiscard]] constexpr uint16_t X87Control(uint16_t control) {
	return static_cast<uint16_t>(control & 0x0f3fu);
}
[[nodiscard]] constexpr bool SameFpControl(FpEnv a, FpEnv b) {
	return MxcsrControl(a.mxcsr) == MxcsrControl(b.mxcsr) &&
	       X87Control(a.control) == X87Control(b.control);
}

// Places the FP control state is checked at (one report per site and thread, default on).
enum class FpCheckSite : uint32_t {
	ThreadStart    = 0, // a guest thread about to enter its entry point
	ThreadCreator  = 1, // the creating thread's state differs from the one the new thread gets
	LibraryEntry   = 2, // guest code called a library function with a state other than GuestFpEnv
	LibraryChanged = 3, // a library function returned with another state than it was called with
	FaultReturn    = 4, // a fault handler changed the state it returns to the guest
	GuestFault     = 5, // guest code faulted with a state other than GuestFpEnv
	Submit         = 6, // a command buffer submitted with a state other than GuestFpEnv
	SignalHandler  = 7, // a guest signal handler would start with a state other than GuestFpEnv
	Count          = 8,
};

inline thread_local uint32_t t_fp_check_reported = 0;

// True the first time on this thread that `site` sees `env` differ from `expected` in its chosen
// part; the caller then logs. Costs one compare while the state is as expected.
[[nodiscard]] inline bool FpControlFirstDeviation(FpCheckSite site, FpEnv env,
                                                  FpEnv expected = GuestFpEnv) {
	if (SameFpControl(env, expected)) {
		return false;
	}
	const uint32_t bit = 1u << static_cast<uint32_t>(site);
	if ((t_fp_check_reported & bit) != 0) {
		return false;
	}
	t_fp_check_reported |= bit;
	return true;
}

[[nodiscard]] constexpr const char* FpCheckSiteName(FpCheckSite site) {
	switch (site) {
		case FpCheckSite::ThreadStart: return "thread-start";
		case FpCheckSite::ThreadCreator: return "thread-creator";
		case FpCheckSite::LibraryEntry: return "library-entry";
		case FpCheckSite::LibraryChanged: return "library-changed";
		case FpCheckSite::FaultReturn: return "fault-return";
		case FpCheckSite::GuestFault: return "guest-fault";
		case FpCheckSite::Submit: return "submit";
		case FpCheckSite::SignalHandler: return "signal-handler";
		default: return "?";
	}
}

} // namespace Common

#endif /* KYTY_COMMON_FP_ENV_H_ */
