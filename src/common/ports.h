#ifndef KYTY_COMMON_PORTS_H_
#define KYTY_COMMON_PORTS_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

// Fixes ported from other trees that change behaviour. Each one is on by default and can be turned
// off at run time with --disable-port <name[,name...]> (or "all"), so one build can test them one by
// one. Each commit names its source in a "Source:" line.
namespace Config {

enum class Port : uint32_t {
	Wave64OverLimit32, // run wave64 compute groups over the invocation limit in 32-lane subgroups
	Wave64LdsSync,     // order LDS between the halves of a wave64 compute program
	GuestLaneNumber,   // number lanes inside the guest wave when the host subgroup is wider
	GpuSelectedWrites, // keyed-table atomics; buffer writes do not block scalar branch lowering
	DispatchGuard,     // zero indirect dispatches whose GPU-written group counts are out of range
	LargeMemoryBudget, // grant the 1216 MiB larger guest memory budget without Pro-mode metadata
	CanonicalResources, // indirect resources specialize independently of table layout and encounter order
	Count
};

struct PortInfo {
	Port        port;
	const char* name;
	const char* source;
	// The port changes generated shader code: its state selects the shader store file.
	bool        shader_code;
};

inline constexpr PortInfo PORTS[] = {
    {Port::Wave64OverLimit32, "wave64-over-limit-32", "upstream PR 2567", true},
    {Port::Wave64LdsSync, "wave64-lds-sync", "Senaxx/KytyPS5@86151b7251", true},
    {Port::GuestLaneNumber, "guest-lane-number", "lordra9-svg/KytyPS5@85c70d0416", true},
    {Port::GpuSelectedWrites, "gpu-selected-writes", "wololo-wololo/VIcky-Blue@d134910670", true},
    {Port::DispatchGuard, "dispatch-guard", "wololo-wololo/VIcky-Blue@841cb9cac2", false},
    {Port::LargeMemoryBudget, "large-memory-budget", "wololo-wololo/VIcky-Blue@26d53f0fb", false},
    {Port::CanonicalResources, "canonical-resources", "this tree (shader store reuse across boots)", true},
};
static_assert(std::size(PORTS) == static_cast<size_t>(Port::Count));

// Checks a comma-separated list of port names ("all" names every port) and appends them to out.
inline bool ParseDisabledPorts(std::string_view text, std::vector<std::string>& out);
// Sets the disabled ports; the names must have passed ParseDisabledPorts.
inline void SetDisabledPorts(const std::vector<std::string>& names);

inline bool PortEnabled(Port port);
// Bit i: PORTS[i] is enabled and changes shader code. 0 when every such port is off.
inline uint32_t ShaderPortMask();
// "[ports] enabled: a,b disabled: c" for the start-up log.
inline std::string PortsSummary();

namespace PortsDetail {

inline constexpr uint32_t AllPorts = (1u << static_cast<uint32_t>(Port::Count)) - 1u;

inline std::atomic<uint32_t> g_enabled {AllPorts};

inline const PortInfo* FindPort(std::string_view name) {
	const auto* found = std::ranges::find(PORTS, name, &PortInfo::name);
	return found != std::end(PORTS) ? found : nullptr;
}

} // namespace PortsDetail

inline bool ParseDisabledPorts(std::string_view text, std::vector<std::string>& out) {
	while (!text.empty()) {
		const auto comma = text.find(',');
		const auto name  = text.substr(0, comma);
		text             = comma == std::string_view::npos ? std::string_view {} : text.substr(comma + 1);
		if (name.empty()) continue;
		if (name == "all") {
			for (const auto& port: PORTS) out.emplace_back(port.name);
		} else if (PortsDetail::FindPort(name) != nullptr) {
			out.emplace_back(name);
		} else {
			return false;
		}
	}
	return true;
}

inline void SetDisabledPorts(const std::vector<std::string>& names) {
	uint32_t enabled = PortsDetail::AllPorts;
	for (const auto& name: names) {
		if (const auto* port = PortsDetail::FindPort(name)) {
			enabled &= ~(1u << static_cast<uint32_t>(port->port));
		}
	}
	PortsDetail::g_enabled.store(enabled, std::memory_order_relaxed);
}

inline bool PortEnabled(Port port) {
	return (PortsDetail::g_enabled.load(std::memory_order_relaxed) & (1u << static_cast<uint32_t>(port))) != 0;
}

inline uint32_t ShaderPortMask() {
	uint32_t mask = 0;
	for (const auto& port: PORTS) {
		if (port.shader_code && PortEnabled(port.port)) mask |= 1u << static_cast<uint32_t>(port.port);
	}
	return mask;
}

inline std::string PortsSummary() {
	std::string enabled;
	std::string disabled;
	for (const auto& port: PORTS) {
		auto& list = PortEnabled(port.port) ? enabled : disabled;
		if (!list.empty()) list += ',';
		list += port.name;
	}
	return "[ports] enabled: " + (enabled.empty() ? std::string("none") : enabled) +
	       " disabled: " + (disabled.empty() ? std::string("none") : disabled);
}

} // namespace Config

#endif /* KYTY_COMMON_PORTS_H_ */
