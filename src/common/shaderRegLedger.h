#ifndef KYTY_COMMON_SHADER_REG_LEDGER_H_
#define KYTY_COMMON_SHADER_REG_LEDGER_H_

#include <cstdint>
#include <memory>
#include <string>
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
	uint32_t         workgroup_register    = 0;
	bool             dispatch_thread_dimensions = false;
};
// The shared immutable copy is the same one retained by the shader map. The enqueue path never
// waits for queue space or the writer lock; a contended/full queue drops the dump and counts it.
void DumpRegistration(uint64_t hash, const DumpRegistrationInfo& info,
                      std::shared_ptr<const std::vector<uint32_t>> code);
void NoteUse(uint64_t hash, Stage stage, uint64_t variant_key);
void PrintSummary();

// First-use records (--dump-registered-shaders): what the translator and the resource
// materialization read when a stage was first used. One record per (stage, hash), written to
// use.jsonl and one blob file. Guest addresses appear only inside the blob and the memory range
// list, which together are replayed by the offline tool as a private virtual memory.
struct UseRange {
	uint64_t address = 0;
	uint64_t bytes   = 0;
};
struct UseRecord {
	std::string           stage; // cs, ps, vs, ls, hs, tes, ms
	uint64_t              hash = 0;
	uint32_t              options_wave_size = 64;
	uint32_t              user_data_base    = 0;
	uint32_t              push_begin        = 0;
	uint32_t              push_end          = 0;
	uint32_t              workgroup_counts[3] = {};
	uint64_t              indirect_args_address = 0;
	uint64_t              shader_base           = 0;
	bool                  uses_image_heap       = false;
	bool                  memory_truncated      = false;
	std::vector<uint32_t> user_data;
	std::vector<uint8_t>  input_info; // Raw bytes of the stage input struct, pointers cleared.
	std::vector<uint32_t> back_code;
	std::vector<UseRange> ranges;
	std::vector<uint8_t>  memory; // The bytes of `ranges`, back to back.
};
// The use.jsonl line and the blob file bytes of a record (the writer thread uses the same code).
void SerializeUseRecord(const UseRecord& record, uint64_t order, std::string& line,
                        std::vector<uint8_t>& blob);
[[nodiscard]] bool UseDumpEnabled();
// True once per (stage, hash) while the use dump is enabled; the caller then builds the record.
[[nodiscard]] bool WantUse(uint64_t hash, std::string_view stage);
// Never blocks: a contended or full queue drops the record and counts it.
void DumpUse(UseRecord&& record);
// Counted reasons a wanted record was not produced (call targets, failed materialization, ...).
void NoteUseSkipped(std::string_view reason);

struct Lifecycle {
	static constexpr const char* name               = "ShaderRegLedger";
	static constexpr auto        initialize         = Initialize;
	static constexpr auto        shutdown           = Shutdown;
	static constexpr auto        emergency_shutdown = PrintSummary;
};

} // namespace Common::ShaderRegLedger

#endif /* KYTY_COMMON_SHADER_REG_LEDGER_H_ */
