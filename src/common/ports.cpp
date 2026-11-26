#include "common/ports.h"

#include <algorithm>
#include <atomic>

namespace Config {

namespace {

constexpr uint32_t AllPorts = (1u << static_cast<uint32_t>(Port::Count)) - 1u;

std::atomic<uint32_t> g_enabled {AllPorts};

const PortInfo* FindPort(std::string_view name) {
	const auto* found = std::ranges::find(PORTS, name, &PortInfo::name);
	return found != std::end(PORTS) ? found : nullptr;
}

} // namespace

bool ParseDisabledPorts(std::string_view text, std::vector<std::string>& out) {
	while (!text.empty()) {
		const auto comma = text.find(',');
		const auto name  = text.substr(0, comma);
		text             = comma == std::string_view::npos ? std::string_view {} : text.substr(comma + 1);
		if (name.empty()) continue;
		if (name == "all") {
			for (const auto& port: PORTS) out.emplace_back(port.name);
		} else if (FindPort(name) != nullptr) {
			out.emplace_back(name);
		} else {
			return false;
		}
	}
	return true;
}

void SetDisabledPorts(const std::vector<std::string>& names) {
	uint32_t enabled = AllPorts;
	for (const auto& name: names) {
		if (const auto* port = FindPort(name)) {
			enabled &= ~(1u << static_cast<uint32_t>(port->port));
		}
	}
	g_enabled.store(enabled, std::memory_order_relaxed);
}

bool PortEnabled(Port port) {
	return (g_enabled.load(std::memory_order_relaxed) & (1u << static_cast<uint32_t>(port))) != 0;
}

uint32_t ShaderPortMask() {
	uint32_t mask = 0;
	for (const auto& port: PORTS) {
		if (port.shader_code && PortEnabled(port.port)) mask |= 1u << static_cast<uint32_t>(port.port);
	}
	return mask;
}

std::string PortsSummary() {
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
