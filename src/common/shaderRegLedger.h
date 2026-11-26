#ifndef KYTY_COMMON_SHADER_REG_LEDGER_H_
#define KYTY_COMMON_SHADER_REG_LEDGER_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

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
void Shutdown();
void NoteCreateCall();
void NoteRegistration(uint64_t hash, Stage stage);
struct DumpRegistrationInfo {
	std::string_view binary_stage;
	uint32_t         registered_size_bytes = 0;
	uint32_t         translated_size_bytes = 0;
	uint32_t         scratch_size_dwords   = 0;
	uint32_t         input_semantics       = 0;
	uint32_t         direct_resources      = 0;
	uint32_t         sharp_resources       = 0;
	uint32_t         eud_size_dwords       = 0;
	uint32_t         srt_size_dwords       = 0;
	bool             compute_input_known   = false;
	uint32_t         user_data_count       = 0;
	uint32_t         threads_num[3]        = {};
	uint32_t         lds_size_dwords       = 0;
	uint32_t         wave_size             = 64;
	uint32_t         float_mode            = 0xc0;
	bool             group_id[3]           = {};
	uint32_t         thread_ids_num        = 0;
	bool             tg_size_en            = false;
};
// The shared immutable copy is the same one retained by the shader map. The enqueue path never
// waits for queue space or the writer lock; a contended/full queue drops the dump and counts it.
void DumpRegistration(uint64_t hash, const DumpRegistrationInfo& info,
                      std::shared_ptr<const std::vector<uint32_t>> code);
void NoteUse(uint64_t hash, Stage stage, uint64_t variant_key);
void PrintSummary();

struct Lifecycle {
	static constexpr const char* name               = "ShaderRegLedger";
	static constexpr auto        initialize         = Initialize;
	static constexpr auto        shutdown           = Shutdown;
	static constexpr auto        emergency_shutdown = PrintSummary;
};

} // namespace Common::ShaderRegLedger

#endif /* KYTY_COMMON_SHADER_REG_LEDGER_H_ */
