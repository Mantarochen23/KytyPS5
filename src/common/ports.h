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
	IndirectFormattedStores, // allow general aligned formatted indirect buffer stores
	FiniteBufferDescriptors, // enumerate bounded GPU-selected buffer descriptors across select/phi/lane trees
	ImageHeapIncremental, // rediscover resident images only after relevant cache changes
	RuntimeTableCache, // reuse runtime-indexed buffer tables while their candidate backing is unchanged
	BarrierTracker,   // order guest dispatches with deferred, narrowed barriers instead of all-commands pairs
	GpuIndirectThreadDim, // compute thread-dimension indirect dispatch counts and LDS bounds on the GPU
	GpuDrawIndirectCount, // keep multi-draw indirect counts on the GPU via vkCmdDraw[Indexed]IndirectCount
	InPlaceReadback, // read finished GPU writes in place from host-visible device memory (resizable BAR)
	DetachedReadback, // read back submitted buffer data via a detached submit without flushing the open command buffer
	BatchedReadback, // co-download hot readback buffers and recent finished small GPU writes during forced sync readbacks
	Count
};

struct PortInfo {
	Port        port;
	const char* name;
	const char* source;
	// The port changes the code generated for a shader the translator already accepted without it
	// (it takes a different path for the same input): its state is part of the shader store identity
	// (the store file name). Off for a port that cannot change an accepted shader's output.
	bool        affects_codegen;
	// The port lets the translator accept shaders it rejected without the port and leaves every
	// other shader's output alone: not part of the store identity. Instead each stored failure
	// remembers which such ports were on, and is retried when one is on that was not.
	bool        widens_translation;
	// Off unless named by --enable-port: for a port nobody has run on a GPU yet.
	bool        default_off = false;
};

inline constexpr PortInfo PORTS[] = {
    {Port::Wave64OverLimit32, "wave64-over-limit-32", "upstream PR 2567", true, false},
    {Port::Wave64LdsSync, "wave64-lds-sync", "Senaxx/KytyPS5@86151b7251", true, false},
    {Port::GuestLaneNumber, "guest-lane-number", "lordra9-svg/KytyPS5@85c70d0416", true, false},
    {Port::GpuSelectedWrites, "gpu-selected-writes", "wololo-wololo/VIcky-Blue@d134910670", true, false},
    {Port::DispatchGuard, "dispatch-guard", "wololo-wololo/VIcky-Blue@841cb9cac2", false, false},
    {Port::LargeMemoryBudget, "large-memory-budget", "wololo-wololo/VIcky-Blue@26d53f0fb", false, false},
    {Port::CanonicalResources, "canonical-resources", "this tree (shader store reuse across boots)", true, false},
    {Port::IndirectFormattedStores, "indirect-formatted-stores", "local matfail2", false, true},
    {Port::FiniteBufferDescriptors, "finite-buffer-descriptors", "local matfail2", false, true},
    {Port::ImageHeapIncremental, "image-heap-incremental", "upstream PR 1289 (NeySlim)", false, false},
    {Port::RuntimeTableCache, "runtime-table-cache", "this tree (table1)", false, false},
    {Port::BarrierTracker, "barrier-tracker", "CarRamRod-963/KytyPS5@f80443cf88", false, false},
    {Port::GpuIndirectThreadDim, "gpu-indirect-thread-dim", "this tree (readback1)", false, false, true},
    {Port::GpuDrawIndirectCount, "gpu-draw-indirect-count", "this tree (readback1)", false, false},
    {Port::InPlaceReadback, "inplace-readback", "CarRamRod-963/KytyPS5@4c4300d87d", false, false, true},
    {Port::DetachedReadback, "detached-readback", "upstream PR 1254 (NeySlim)", false, false, true},
    {Port::BatchedReadback, "batched-readback", "upstream PR 2093 (MrBigBritches)", false, false, true},
};
static_assert(std::size(PORTS) == static_cast<size_t>(Port::Count));

// Checks a comma-separated list of port names ("all" names every port) and appends them to out.
inline bool ParseDisabledPorts(std::string_view text, std::vector<std::string>& out);
// Sets the disabled ports; the names must have passed ParseDisabledPorts.
// Ports with default_off are off unless named in `enabled`; naming a port in `names` turns it off.
inline void SetDisabledPorts(const std::vector<std::string>& names,
                             const std::vector<std::string>& enabled = {});

inline bool PortEnabled(Port port);
// Bit i: PORTS[i] is enabled and affects_codegen. 0 when every such port is off. Selects the store file.
inline uint32_t ShaderPortMask();
// Bit i: PORTS[i] is enabled and widens_translation.
inline uint32_t CapabilityPortMask();
// Port names of the bits set in mask, comma separated ("none" for 0).
inline std::string PortNamesOfMask(uint32_t mask);
// "[ports] enabled: a,b disabled: c" for the start-up log.
inline std::string PortsSummary();

namespace PortsDetail {

inline constexpr uint32_t AllPorts = (1u << static_cast<uint32_t>(Port::Count)) - 1u;

// The ports on when no option is given: every port except those marked default_off.
inline constexpr uint32_t DefaultPorts() {
	uint32_t mask = AllPorts;
	for (const auto& port: PORTS) {
		if (port.default_off) mask &= ~(1u << static_cast<uint32_t>(port.port));
	}
	return mask;
}

inline std::atomic<uint32_t> g_enabled {DefaultPorts()};

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

inline void SetDisabledPorts(const std::vector<std::string>& names,
                             const std::vector<std::string>& enabled_names) {
	uint32_t enabled = PortsDetail::DefaultPorts();
	for (const auto& name: enabled_names) {
		if (const auto* port = PortsDetail::FindPort(name)) {
			enabled |= 1u << static_cast<uint32_t>(port->port);
		}
	}
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
		if (port.affects_codegen && PortEnabled(port.port)) mask |= 1u << static_cast<uint32_t>(port.port);
	}
	return mask;
}

inline uint32_t CapabilityPortMask() {
	uint32_t mask = 0;
	for (const auto& port: PORTS) {
		if (port.widens_translation && PortEnabled(port.port)) mask |= 1u << static_cast<uint32_t>(port.port);
	}
	return mask;
}

inline std::string PortNamesOfMask(uint32_t mask) {
	std::string names;
	for (const auto& port: PORTS) {
		if ((mask & (1u << static_cast<uint32_t>(port.port))) == 0) continue;
		if (!names.empty()) names += ',';
		names += port.name;
	}
	return names.empty() ? std::string("none") : names;
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
