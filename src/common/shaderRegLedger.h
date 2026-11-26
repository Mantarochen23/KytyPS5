#ifndef KYTY_COMMON_SHADER_REG_LEDGER_H_
#define KYTY_COMMON_SHADER_REG_LEDGER_H_

#include <cstdint>

namespace Common::ShaderRegLedger {

// Coarse buckets shared by registration (guest binary type) and use (host stage), so a shader can
// be matched across the two paths without depending on either enum here.
enum class Stage : uint8_t {
	Compute = 0,
	Pixel,
	Geometry,
	Hull,
	Front,
	Unknown,
	Count
};

void Initialize();
void NoteCreateCall();
void NoteRegistration(uint64_t hash, Stage stage);
void NoteUse(uint64_t hash, Stage stage, uint64_t variant_key);
void PrintSummary();

struct Lifecycle {
	static constexpr const char* name               = "ShaderRegLedger";
	static constexpr auto        initialize         = Initialize;
	static constexpr auto        shutdown           = PrintSummary;
	static constexpr auto        emergency_shutdown = PrintSummary;
};

} // namespace Common::ShaderRegLedger

#endif /* KYTY_COMMON_SHADER_REG_LEDGER_H_ */
