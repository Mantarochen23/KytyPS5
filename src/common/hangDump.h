#ifndef KYTY_COMMON_HANG_DUMP_H_
#define KYTY_COMMON_HANG_DUMP_H_

// Waiting for the Vulkan driver's own GPU hang dump (Mesa RADV, RADV_DEBUG=hang).
//
// RADV writes `$HOME/radv_dumps_<pid>_<time>/` from inside the queue-submit path of the thread
// that observes the hang (radv_check_gpu_hangs): one file per chunk, in the order trace, pipeline,
// umr_waves, umr_ring, registers, bo_ranges, bo_history, addr_binding_report, vm_fault, app_info,
// gpu_info, dmesg; then it prints "GPU hang report saved successfully!" and abort()s. When another
// emulator thread sees VK_ERROR_DEVICE_LOST first and ends the process, the dump stays partial or
// never starts. The logic here is header-only and takes its probe, clock and sleep as callables so
// it can be unit-tested without a driver.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace Common::HangDump {

// True when a RADV_DEBUG-style comma/space separated list contains the `hang` flag.
inline bool DriverHangModeActive(const char* radv_debug) {
	if (radv_debug == nullptr) {
		return false;
	}
	std::string_view rest(radv_debug);
	while (!rest.empty()) {
		const auto sep   = rest.find_first_of(", \t:;");
		const auto token = rest.substr(0, sep);
		if (token == "hang") {
			return true;
		}
		if (sep == std::string_view::npos) {
			break;
		}
		rest.remove_prefix(sep + 1);
	}
	return false;
}

// The chunk RADV writes last; once it exists and nothing changes any more the dump is complete.
inline constexpr std::string_view kLastChunkFile = "dmesg.log";

struct DumpState {
	bool          found           = false; // a radv_dumps_<pid>_* folder exists
	std::string   path;                    // newest matching folder
	uint32_t      file_count      = 0;
	uint64_t      total_bytes     = 0;
	bool          has_last_chunk  = false;
	bool operator==(const DumpState&) const = default;
};

// Folder RADV creates: "<home or .>/radv_dumps_<pid>_<time>" (home from the HOME variable).
inline DumpState ScanDumpFolder(const char* home, int pid) {
	DumpState             state;
	namespace fs = std::filesystem;
	const fs::path        base   = (home != nullptr && home[0] != '\0') ? fs::path(home) : fs::path(".");
	const std::string     prefix = "radv_dumps_" + std::to_string(pid) + "_";
	std::error_code       ec;
	fs::file_time_type    newest {};
	for (fs::directory_iterator it(base, ec), end; !ec && it != end; it.increment(ec)) {
		std::error_code ec2;
		if (!it->is_directory(ec2) || it->path().filename().string().rfind(prefix, 0) != 0) {
			continue;
		}
		const auto t = it->last_write_time(ec2);
		if (!state.found || (!ec2 && t >= newest)) {
			state.found = true;
			state.path  = it->path().string();
			newest      = ec2 ? newest : t;
		}
	}
	if (!state.found) {
		return state;
	}
	for (fs::directory_iterator it(state.path, ec), end; !ec && it != end; it.increment(ec)) {
		std::error_code ec2;
		if (!it->is_regular_file(ec2)) {
			continue;
		}
		++state.file_count;
		const auto size = it->file_size(ec2);
		state.total_bytes += ec2 ? 0 : size;
		if (it->path().filename().string() == kLastChunkFile) {
			state.has_last_chunk = true;
		}
	}
	return state;
}

enum class WaitOutcome {
	Disabled,        // wait of 0 s requested
	Complete,        // last chunk present and the folder stopped changing
	StableIncomplete,// the folder stopped changing for a long time without the last chunk
	NotStarted,      // no dump folder appeared within the timeout
	TimedOut,        // the folder was still growing / incomplete at the timeout
};

inline const char* WaitOutcomeName(WaitOutcome o) {
	switch (o) {
	case WaitOutcome::Disabled: return "disabled";
	case WaitOutcome::Complete: return "complete";
	case WaitOutcome::StableIncomplete: return "stable-without-last-chunk";
	case WaitOutcome::NotStarted: return "not-started";
	case WaitOutcome::TimedOut: return "timed-out";
	}
	return "?";
}

struct WaitPolicy {
	std::chrono::milliseconds timeout            = std::chrono::seconds(60);
	std::chrono::milliseconds poll               = std::chrono::milliseconds(250);
	// Unchanged for this long with the last chunk present = complete (the driver closes the
	// file right before it aborts).
	std::chrono::milliseconds settle             = std::chrono::milliseconds(1000);
	// Unchanged for this long without the last chunk = the driver is done / not going to finish
	// (e.g. a Mesa version with another chunk order); report what exists.
	std::chrono::milliseconds settle_incomplete  = std::chrono::seconds(10);
};

struct WaitResult {
	WaitOutcome outcome = WaitOutcome::Disabled;
	DumpState   state;
	std::chrono::milliseconds waited {0};
};

// `probe` returns the current folder state, `sleep` advances time by the given amount. The
// elapsed time is counted from the sleeps, so tests need no real clock.
inline WaitResult WaitForDriverDump(const WaitPolicy&                                 policy,
                                    const std::function<DumpState()>&                 probe,
                                    const std::function<void(std::chrono::milliseconds)>& sleep) {
	WaitResult result;
	if (policy.timeout.count() <= 0) {
		result.state = probe();
		return result;
	}
	std::chrono::milliseconds elapsed {0};
	std::chrono::milliseconds unchanged {0};
	DumpState                 last = probe();
	for (;;) {
		if (last.found && last.has_last_chunk && unchanged >= policy.settle) {
			result.outcome = WaitOutcome::Complete;
			break;
		}
		if (last.found && !last.has_last_chunk && unchanged >= policy.settle_incomplete) {
			result.outcome = WaitOutcome::StableIncomplete;
			break;
		}
		if (elapsed >= policy.timeout) {
			result.outcome = last.found ? WaitOutcome::TimedOut : WaitOutcome::NotStarted;
			break;
		}
		const auto step = std::min(policy.poll, policy.timeout - elapsed);
		sleep(step);
		elapsed += step;
		const DumpState now = probe();
		unchanged           = (now == last) ? unchanged + step : std::chrono::milliseconds(0);
		last                = now;
	}
	result.state  = last;
	result.waited = elapsed;
	return result;
}

} // namespace Common::HangDump

#endif /* KYTY_COMMON_HANG_DUMP_H_ */
