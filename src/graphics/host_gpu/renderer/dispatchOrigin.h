#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHORIGIN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHORIGIN_H_

// Where the compute dispatch being processed came from, for the `[dispatch]` report of a refused
// dispatch (dispatchLimits.h): the guest words before its packet, the argument buffer of a
// thread-dimension indirect dispatch, and the guest thread that submitted the command buffer with
// its last library calls. The command processor sets it around one dispatch on its own thread.
// Header only, no renderer types, so the formatting is unit tested (tests/DispatchLimitsTests.cpp).

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace Libs::Graphics {

constexpr uint32_t DispatchSubmitterCalls  = 16;
constexpr uint32_t DispatchWindowDwords    = 64;

struct DispatchSubmitterCall {
	const char* library   = nullptr;
	const char* function  = nullptr;
	uint64_t    guest_ret = 0;
};

// Taken on the guest thread when it submits a command buffer.
struct DispatchSubmitter {
	int                   thread_id = -1; // guest thread unique id, -1 = not known
	uint32_t              mxcsr     = 0;  // its FP control state at the submit
	uint16_t              x87_cw    = 0;
	uint32_t              calls_num = 0;  // oldest first
	DispatchSubmitterCall calls[DispatchSubmitterCalls] {};
};

struct DispatchOrigin {
	const uint32_t*          packet    = nullptr; // guest address of the PM4 header, nullptr = unknown
	uint64_t                 args_addr = 0;       // argument buffer (indirect), 0 = direct packet
	int                      args_gpu_written = -1; // -1 not asked, 0 no, 1 yes (before the read)
	const DispatchSubmitter* submitter = nullptr;
};

// The origin of the dispatch the command processor of this thread is handing to the renderer.
inline thread_local const DispatchOrigin* t_dispatch_origin = nullptr;

class ScopedDispatchOrigin {
public:
	explicit ScopedDispatchOrigin(const DispatchOrigin* origin): m_previous(t_dispatch_origin) {
		t_dispatch_origin = origin;
	}
	~ScopedDispatchOrigin() { t_dispatch_origin = m_previous; }
	ScopedDispatchOrigin(const ScopedDispatchOrigin&)            = delete;
	ScopedDispatchOrigin& operator=(const ScopedDispatchOrigin&) = delete;

private:
	const DispatchOrigin* m_previous;
};

// `count` dwords as hex, 16 per line, each line prefixed by `prefix` and the dword index relative
// to the packet header (negative before it). `first_index` is the index of words[0].
[[nodiscard]] inline std::string FormatDwordWindow(const char* prefix, const uint32_t* words,
                                                   size_t count, int64_t first_index) {
	std::string out;
	char        text[32];
	for (size_t i = 0; i < count; i++) {
		if (i % 16 == 0) {
			if (i != 0) out += '\n';
			std::snprintf(text, sizeof(text), "%+" PRId64 ":", first_index + static_cast<int64_t>(i));
			out += prefix;
			out += text;
		}
		std::snprintf(text, sizeof(text), " %08" PRIx32, words[i]);
		out += text;
	}
	if (count != 0) out += '\n';
	return out;
}

// "lib::function@0x<ret>" oldest first, separated by ", ".
[[nodiscard]] inline std::string FormatSubmitterCalls(const DispatchSubmitter& submitter) {
	std::string out;
	char        text[32];
	const auto  num = submitter.calls_num < DispatchSubmitterCalls ? submitter.calls_num
	                                                               : DispatchSubmitterCalls;
	for (uint32_t i = 0; i < num; i++) {
		const auto& call = submitter.calls[i];
		if (i != 0) out += ", ";
		out += call.library != nullptr ? call.library : "?";
		out += "::";
		out += call.function != nullptr ? call.function : "?";
		std::snprintf(text, sizeof(text), "@0x%" PRIx64, call.guest_ret);
		out += text;
	}
	return out.empty() ? std::string("none") : out;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHORIGIN_H_ */
