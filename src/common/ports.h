#ifndef KYTY_COMMON_PORTS_H_
#define KYTY_COMMON_PORTS_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Fixes ported from other trees that change behaviour. Each one is on by default and can be turned
// off at run time with --disable-port <name[,name...]> (or "all"), so one build can test them one by
// one. The source of each is in docs/upstream/THIRD-PARTY.md.
namespace Config {

enum class Port : uint32_t {
	Wave64OverLimit32, // run wave64 compute groups over the invocation limit in 32-lane subgroups
	Wave64LdsSync,     // order LDS between the halves of a wave64 compute program
	GuestLaneNumber,   // number lanes inside the guest wave when the host subgroup is wider
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
};
static_assert(std::size(PORTS) == static_cast<size_t>(Port::Count));

// Checks a comma-separated list of port names ("all" names every port) and appends them to out.
bool ParseDisabledPorts(std::string_view text, std::vector<std::string>& out);
// Sets the disabled ports; the names must have passed ParseDisabledPorts.
void SetDisabledPorts(const std::vector<std::string>& names);

bool PortEnabled(Port port);
// Bit i: PORTS[i] is enabled and changes shader code. 0 when every such port is off.
uint32_t ShaderPortMask();
// "[ports] enabled: a,b disabled: c" for the start-up log.
std::string PortsSummary();

} // namespace Config

#endif /* KYTY_COMMON_PORTS_H_ */
