#include "common/emulatorConfig.h"
#include "common/recoverable.h"
#include "common/shaderRegLedger.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <type_traits>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Json = nlohmann::ordered_json;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderPixelInputInfo;
using Libs::Graphics::ShaderType;
using Libs::Graphics::ShaderVertexInputInfo;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

struct Options {
	std::filesystem::path input;
	std::filesystem::path output;
	uint32_t              jobs       = 0;
	uint32_t              limit      = 0;
	uint32_t              timeout_ms = 20000;
	std::string           only;
	bool                  emit_spirv = false;
	bool                  self_test  = false;
	std::filesystem::path compare_log;
};

struct CorpusEntry {
	Json                  record;
	std::string           hash;
	std::string           stage;
	std::filesystem::path binary;
	// The first-use record of this (stage, hash), when the dump has one.
	std::optional<Json>   use;
	std::filesystem::path use_binary;
};

struct RunningChild {
#if !defined(_WIN32)
	pid_t pid = -1;
#endif
	size_t                                  index = 0;
	std::chrono::steady_clock::time_point deadline;
};

thread_local std::map<std::string, double>* g_pass_times = nullptr;

double Milliseconds(std::chrono::steady_clock::duration duration) {
	return std::chrono::duration<double, std::milli>(duration).count();
}

void PassTiming(const char* name, uint64_t nanoseconds) {
	if (g_pass_times != nullptr) {
		(*g_pass_times)[name] += static_cast<double>(nanoseconds) / 1.0e6;
	}
}

std::string RejectionClass(std::string_view shape) {
	const auto separator = shape.find_first_of(":\n");
	auto       value     = std::string(shape.substr(0, std::min(separator, size_t {64})));
	for (char& character: value) {
		if ((character >= 'a' && character <= 'z') ||
		    (character >= 'A' && character <= 'Z') ||
		    (character >= '0' && character <= '9')) {
			character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
		} else {
			character = '-';
		}
	}
	while (!value.empty() && value.back() == '-') value.pop_back();
	if (value.empty()) value = "unknown";
	return value;
}

bool ReadJsonLines(const std::filesystem::path& path, std::vector<Json>& records,
	               std::string& error) {
	std::ifstream input(path);
	if (!input) {
		error = "could not open " + path.string();
		return false;
	}
	std::string line;
	uint64_t    number = 0;
	while (std::getline(input, line)) {
		++number;
		if (line.empty()) continue;
		auto record = Json::parse(line, nullptr, false);
		if (record.is_discarded() || !record.is_object()) {
			error = "invalid JSON at index line " + std::to_string(number);
			return false;
		}
		records.push_back(std::move(record));
	}
	return true;
}

// use.jsonl is optional. The first record of a (stage, hash) wins; a malformed line is skipped
// (the writer appends whole lines, so only a killed run can leave one).
void AttachUseRecords(const Options& options, std::vector<CorpusEntry>& entries) {
	std::ifstream input(options.input / "use.jsonl");
	if (!input) return;
	std::map<std::pair<std::string, std::string>, Json> uses;
	std::string line;
	while (std::getline(input, line)) {
		if (line.empty()) continue;
		auto record = Json::parse(line, nullptr, false);
		if (record.is_discarded() || !record.is_object() || !record.contains("hash") ||
		    !record["hash"].is_string() || !record.contains("stage") ||
		    !record["stage"].is_string() || !record.contains("file") ||
		    !record["file"].is_string()) {
			continue;
		}
		const auto relative = std::filesystem::path(record["file"].get<std::string>());
		if (relative.is_absolute() || relative.string().find("..") != std::string::npos) continue;
		uses.emplace(std::pair {record["stage"].get<std::string>(),
		                        record["hash"].get<std::string>()},
		             std::move(record));
	}
	for (auto& entry: entries) {
		const auto found = uses.find({entry.stage, entry.hash});
		if (found == uses.end()) continue;
		entry.use        = found->second;
		entry.use_binary = options.input / found->second["file"].get<std::string>();
	}
}

bool LoadEntries(const Options& options, std::vector<CorpusEntry>& entries, std::string& error) {
	std::vector<Json> records;
	if (!ReadJsonLines(options.input / "index.jsonl", records, error)) return false;
	std::set<std::pair<std::string, std::string>> seen;
	for (auto& record: records) {
		if (!record.contains("hash") || !record["hash"].is_string() ||
		    !record.contains("stage") || !record["stage"].is_string() ||
		    !record.contains("file") || !record["file"].is_string()) {
			error = "index record is missing hash, stage, or file";
			return false;
		}
		auto hash  = record["hash"].get<std::string>();
		auto stage = record["stage"].get<std::string>();
		if (!options.only.empty() && stage != options.only) continue;
		if (!seen.emplace(hash, stage).second) continue;
		const auto relative = std::filesystem::path(record["file"].get<std::string>());
		if (relative.is_absolute() || relative.string().find("..") != std::string::npos) {
			error = "index contains an unsafe shader path";
			return false;
		}
		entries.push_back(
		    {.record = std::move(record), .hash = std::move(hash), .stage = std::move(stage),
		     .binary = options.input / relative});
	}
	AttachUseRecords(options, entries);
	std::ranges::sort(entries, {}, [](const CorpusEntry& entry) {
		return std::pair {entry.hash, entry.stage};
	});
	if (options.limit != 0 && entries.size() > options.limit) entries.resize(options.limit);
	return true;
}

std::optional<uint64_t> ParseHash(std::string_view text) {
	const std::string input(text);
	char*      end   = nullptr;
	const auto value = std::strtoull(input.c_str(), &end, 0);
	if (end == nullptr || *end != '\0') return {};
	return value;
}

std::vector<uint32_t> ReadBinary(const CorpusEntry& entry) {
	std::ifstream input(entry.binary, std::ios::binary | std::ios::ate);
	if (!input) throw std::runtime_error("could not open shader binary");
	const auto bytes = input.tellg();
	if (bytes <= 0 || static_cast<uint64_t>(bytes) % sizeof(uint32_t) != 0) {
		throw std::runtime_error("shader binary size is not a positive DWORD multiple");
	}
	std::vector<uint32_t> code(static_cast<size_t>(bytes) / sizeof(uint32_t));
	input.seekg(0);
	input.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(bytes));
	if (!input) throw std::runtime_error("could not read shader binary");
	return code;
}

template <typename Value>
Value JsonValue(const Json& object, const char* name, Value fallback) {
	if (!object.is_object() || !object.contains(name) || !object[name].is_number()) return fallback;
	return object[name].get<Value>();
}

void PopulateComputeInfo(const CorpusEntry& entry, ShaderComputeInputInfo& input) {
	if (!entry.record.contains("stage_input_info")) return;
	const auto& info = entry.record["stage_input_info"];
	if (info.contains("threads_num") && info["threads_num"].is_array() &&
	    info["threads_num"].size() == 3) {
		for (size_t axis = 0; axis < 3; axis++) input.threads_num[axis] = info["threads_num"][axis];
	}
	if (info.contains("group_id") && info["group_id"].is_array() &&
	    info["group_id"].size() == 3) {
		for (size_t axis = 0; axis < 3; axis++) input.group_id[axis] = info["group_id"][axis];
	}
	input.scratch_size_dwords = JsonValue(info, "scratch_size_dwords", 0u);
	input.lds_size_dwords     = JsonValue(info, "lds_size_dwords", 0u);
	input.wave_size           = JsonValue(info, "wave_size", 64u);
	input.float_mode          = JsonValue(info, "float_mode", uint8_t {0xc0});
	input.thread_ids_num      = static_cast<int>(JsonValue(info, "thread_ids_num", 0u));
	input.tg_size_en          = info.value("tg_size_en", false);
	// The workgroup registers follow the user SGPRs; a dump from before this field was recorded
	// has the same value in user_data_count (both come from COMPUTE_PGM_RSRC2.USER_SGPR).
	input.workgroup_register = static_cast<int>(
	    JsonValue(info, "workgroup_register", JsonValue(entry.record, "user_data_count", 0u)));
	input.dispatch_thread_dimensions = info.value("dispatch_thread_dimensions", false);
}

bool ComputeInputKnown(const CorpusEntry& entry) {
	return entry.record.contains("stage_input_info") &&
	       entry.record["stage_input_info"].value("known", false);
}

bool PlanNeedsRuntime(const Recompiler::IR::ResourcePlan& plan) {
	return !plan.descriptor_sources.empty() || !plan.srt_reads.empty() ||
	       !plan.info.buffers.empty() || !plan.info.images.empty() || !plan.info.samplers.empty();
}

void WriteJson(const std::filesystem::path& path, const Json& value) {
	std::ofstream output(path, std::ios::trunc);
	if (!output) throw std::runtime_error("could not create result file");
	output << value.dump() << '\n';
}

Json BaseResult(const CorpusEntry& entry) {
	return {{"hash", entry.hash},
	        {"stage", entry.stage},
	        {"outcome", "crashed"},
	        {"shape", ""},
	        {"phase_ms", Json::object()},
	        {"pass_ms", Json::object()},
	        {"blocks", 0},
	        {"loops", 0},
	        {"spirv_words", 0},
	        {"wave_size", 0},
	        {"total_ms", 0.0}};
}

// Text of a failure with the shader hash replaced, so equal shapes group together.
std::string NormalizeShape(std::string_view text) {
	std::string out;
	for (size_t index = 0; index < text.size();) {
		if (text.substr(index, 2) == "0x" && index + 18 <= text.size() &&
		    std::ranges::all_of(text.substr(index + 2, 16), [](char c) {
			    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
		    }) &&
		    (index + 18 == text.size() ||
		     std::isxdigit(static_cast<unsigned char>(text[index + 18])) == 0)) {
			out += "<hash>";
			index += 18;
			continue;
		}
		out += text[index++];
	}
	return out;
}

// What a translation failure means. The resource tracking of a stage depends on its stage input
// registers, which only a first-use record has, so without one it says nothing about the
// translator: it is reported as missing input, never as a rejection.
std::string ClassifyFailure(std::string_view shape, bool has_use_record) {
	if (!has_use_record && shape.starts_with("shader resource tracking")) {
		return "needs-runtime:use_record";
	}
	return "rejected:" + RejectionClass(shape);
}

std::string FailureText(const std::exception& failure) {
	std::string text = failure.what() != nullptr ? failure.what() : "";
	if (text.empty()) text = "exception without text";
	return text;
}

// The guest memory ranges a use record holds, replayed as the SRT walker's reader.
class ReplayMemory {
public:
	void Add(uint64_t address, std::span<const uint8_t> bytes) {
		Piece piece {address, std::vector<uint8_t>(bytes.begin(), bytes.end())};
		m_pending.push_back(std::move(piece));
	}
	// Merges overlapping and touching ranges so a read is served by one lookup.
	void Seal() {
		std::ranges::sort(m_pending, {}, &Piece::address);
		for (auto& piece: m_pending) {
			if (!m_runs.empty()) {
				auto& run = m_runs.back();
				const auto end = run.address + run.bytes.size();
				if (piece.address <= end) {
					const auto piece_end = piece.address + piece.bytes.size();
					if (piece_end > end) {
						run.bytes.insert(run.bytes.end(), piece.bytes.begin() + (end - piece.address),
						                 piece.bytes.end());
					}
					continue;
				}
			}
			m_runs.push_back(std::move(piece));
		}
		m_pending.clear();
	}
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values) {
		return static_cast<ReplayMemory*>(userdata)->Lookup(address, values);
	}
	uint64_t Misses() const { return m_misses; }
	std::string FirstMiss() const {
		char text[64];
		std::snprintf(text, sizeof(text), "0x%llx+%llu",
		              static_cast<unsigned long long>(m_first_miss_address),
		              static_cast<unsigned long long>(m_first_miss_bytes));
		return text;
	}

private:
	struct Piece {
		uint64_t             address = 0;
		std::vector<uint8_t> bytes;
	};
	bool Lookup(uint64_t address, std::span<uint32_t> values) {
		const auto bytes = static_cast<uint64_t>(values.size_bytes());
		auto       found = std::ranges::upper_bound(m_runs, address, {}, &Piece::address);
		if (found != m_runs.begin()) {
			--found;
			if (address >= found->address &&
			    address - found->address + bytes <= found->bytes.size()) {
				std::memcpy(values.data(), found->bytes.data() + (address - found->address), bytes);
				return true;
			}
		}
		if (m_misses++ == 0) {
			m_first_miss_address = address;
			m_first_miss_bytes   = bytes;
		}
		return false;
	}
	std::vector<Piece> m_pending;
	std::vector<Piece> m_runs;
	uint64_t           m_misses = 0;
	uint64_t           m_first_miss_address = 0;
	uint64_t           m_first_miss_bytes   = 0;
};

// One slice of a use record's blob file, or nothing when the reference is out of range.
bool ReadSlice(std::ifstream& blob, uint64_t file_size, const Json& reference,
	           std::vector<uint8_t>& bytes) {
	if (!reference.is_object() || !reference.contains("offset") || !reference.contains("bytes") ||
	    !reference["offset"].is_number_unsigned() || !reference["bytes"].is_number_unsigned()) {
		return false;
	}
	const auto offset = reference["offset"].get<uint64_t>();
	const auto count  = reference["bytes"].get<uint64_t>();
	if (offset > file_size || count > file_size - offset) return false;
	bytes.resize(count);
	blob.seekg(static_cast<std::streamoff>(offset));
	if (count != 0) blob.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
	return static_cast<bool>(blob);
}

struct UseInputs {
	std::vector<uint8_t>  input_info;
	std::vector<uint8_t>  memory;
	std::vector<uint8_t>  back_bytes;
	std::vector<uint32_t> user_data;
	std::vector<uint32_t> back_code;
	ReplayMemory          guest;
	uint32_t              wave_size      = 64;
	uint32_t              user_data_base = 0;
	uint32_t              push_begin     = 0;
	uint32_t              push_end       = 32;
	uint32_t              workgroup_counts[3] = {};
	uint64_t              indirect_args  = 0;
	uint64_t              shader_base    = 0;
	bool                  uses_image_heap = false;
	bool                  memory_truncated = false;
};

// Empty on success, otherwise the reason the record cannot be replayed.
std::string LoadUseInputs(const CorpusEntry& entry, UseInputs& inputs) {
	const auto& use = *entry.use;
	std::ifstream blob(entry.use_binary, std::ios::binary | std::ios::ate);
	if (!blob) return "use blob is missing";
	const auto file_size = static_cast<uint64_t>(blob.tellg());
	if (!ReadSlice(blob, file_size, use.value("input_info", Json {}), inputs.input_info) ||
	    !ReadSlice(blob, file_size, use.value("memory", Json {}), inputs.memory) ||
	    !ReadSlice(blob, file_size, use.value("back_code", Json {}), inputs.back_bytes)) {
		return "use blob is truncated";
	}
	if (inputs.back_bytes.size() % sizeof(uint32_t) != 0) return "use blob back code is misaligned";
	inputs.back_code.resize(inputs.back_bytes.size() / sizeof(uint32_t));
	if (!inputs.back_bytes.empty()) {
		std::memcpy(inputs.back_code.data(), inputs.back_bytes.data(), inputs.back_bytes.size());
	}
	if (use.contains("user_data") && use["user_data"].is_array()) {
		for (const auto& word: use["user_data"]) {
			if (!word.is_number_unsigned()) return "use user data is malformed";
			inputs.user_data.push_back(word.get<uint32_t>());
		}
	}
	uint64_t offset = 0;
	if (use.contains("ranges") && use["ranges"].is_array()) {
		for (const auto& range: use["ranges"]) {
			if (!range.is_array() || range.size() != 2) return "use memory ranges are malformed";
			const auto address = range[0].get<uint64_t>();
			const auto bytes   = range[1].get<uint64_t>();
			if (offset > inputs.memory.size() || bytes > inputs.memory.size() - offset) {
				return "use memory ranges exceed the blob";
			}
			inputs.guest.Add(address, std::span<const uint8_t>(inputs.memory).subspan(offset, bytes));
			offset += bytes;
		}
	}
	inputs.guest.Seal();
	inputs.wave_size       = JsonValue(use, "options_wave_size", 64u);
	inputs.user_data_base  = JsonValue(use, "user_data_base", 0u);
	if (use.contains("push_window") && use["push_window"].is_array() &&
	    use["push_window"].size() == 2) {
		inputs.push_begin = use["push_window"][0].get<uint32_t>();
		inputs.push_end   = use["push_window"][1].get<uint32_t>();
	}
	if (use.contains("workgroup_counts") && use["workgroup_counts"].is_array() &&
	    use["workgroup_counts"].size() == 3) {
		for (size_t axis = 0; axis < 3; axis++) {
			inputs.workgroup_counts[axis] = use["workgroup_counts"][axis].get<uint32_t>();
		}
	}
	inputs.indirect_args    = JsonValue(use, "indirect_args_address", uint64_t {0});
	inputs.shader_base      = JsonValue(use, "shader_base", uint64_t {0});
	inputs.uses_image_heap  = use.value("uses_image_heap", false);
	inputs.memory_truncated = use.value("memory_truncated", false);
	return {};
}

template <typename Info>
bool RestoreInputInfo(const UseInputs& inputs, Info& info) {
	static_assert(std::is_trivially_copyable_v<Info>);
	if (inputs.input_info.size() != sizeof(Info)) return false;
	std::memcpy(&info, inputs.input_info.data(), sizeof(Info));
	info.stage = {};
	return true;
}

void TranslateOne(const CorpusEntry& entry, const Options& options,
	              const std::filesystem::path& result_path) {
	Json result = BaseResult(entry);
	const auto total_begin = std::chrono::steady_clock::now();
	std::map<std::string, double> passes;
	const bool has_use = entry.use.has_value() && (entry.stage == "cs" || entry.stage == "ps");
	result["source"]   = has_use ? "use" : "registration";
	try {
		Config::Initialize();
		Config::Load(Config::ConfigOptions {});
		Common::RecoverableScope recoverable;
		auto code = ReadBinary(entry);

		Recompiler::Decoder::Program decoded;
		auto begin = std::chrono::steady_clock::now();
		Recompiler::Decoder::DecodeProgram(code, decoded);
		result["phase_ms"]["decode"] = Milliseconds(std::chrono::steady_clock::now() - begin);

		begin = std::chrono::steady_clock::now();
		auto graph = Recompiler::CFG::BuildGraph(decoded);
		result["phase_ms"]["cfg"] = Milliseconds(std::chrono::steady_clock::now() - begin);
		result["blocks"]          = graph.blocks.size();
		result["loops"]           = graph.natural_loops.size();

		if (entry.stage != "cs" && entry.stage != "ps") {
			result["outcome"] = entry.stage.find("front") != std::string::npos ||
			                            entry.stage.find("back") != std::string::npos
			                        ? "needs-runtime:paired_stage_code"
			                        : "needs-runtime:graphics_stage_input";
			result["shape"] = "registration does not contain the graphics linkage and stage state";
			result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
			WriteJson(result_path, result);
			return;
		}
		if (decoded.has_swap_pc) {
			// GPU-selected callees are linked from guest memory by the emulator.
			result["outcome"] = "needs-runtime:call_targets";
			result["shape"]   = "the shader calls code selected by the GPU; its callees are guest data";
			result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
			WriteJson(result_path, result);
			return;
		}

		ShaderComputeInputInfo compute {};
		ShaderPixelInputInfo   pixel {};
		UseInputs              use_inputs;
		std::vector<uint32_t>  user_data;
		Recompiler::CompileOptions compile;
		compile.dump_ir     = false;
		compile.shader_hash = ParseHash(entry.hash).value_or(0);
		if (has_use) {
			const auto problem = LoadUseInputs(entry, use_inputs);
			bool       restored = problem.empty();
			if (restored) {
				restored = entry.stage == "cs" ? RestoreInputInfo(use_inputs, compute)
				                                : RestoreInputInfo(use_inputs, pixel);
			}
			if (!restored) {
				result["outcome"] = problem.empty() ? "needs-runtime:input_layout" : "needs-runtime:use_record";
				result["shape"]   = problem.empty()
				                        ? "the use record's stage input struct has another size than this tool's"
				                        : problem;
				result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
				WriteJson(result_path, result);
				return;
			}
			user_data              = use_inputs.user_data;
			compile.wave_size      = use_inputs.wave_size;
			compile.user_data_base = use_inputs.user_data_base;
			compile.back_code      = use_inputs.back_code;
		} else {
			PopulateComputeInfo(entry, compute);
			if (entry.record.contains("user_data_count") &&
			    entry.record["user_data_count"].is_number_unsigned()) {
				user_data.resize(std::min(entry.record["user_data_count"].get<size_t>(), size_t {64}));
			}
			compile.wave_size = compute.wave_size;
		}
		compile.user_data = user_data;
		if (entry.stage == "cs") {
			compile.stage              = ShaderType::Compute;
			compile.input_info.compute = &compute;
			result["wave_size"]        = compile.wave_size;
		} else {
			compile.stage            = ShaderType::Pixel;
			compile.input_info.pixel = &pixel;
			result["wave_size"]      = pixel.wave_size;
			if (!has_use) compile.wave_size = pixel.wave_size;
		}

		// Without a use record the stage state of a pixel shader is a default; the translation
		// below is then only a timing probe and its failure says nothing about the translator.
		const bool probe = !has_use && (entry.stage == "ps" || !ComputeInputKnown(entry));
		Recompiler::TranslateResult translated;
		g_pass_times = &passes;
		Recompiler::SetPassTimingHook(PassTiming);
		begin = std::chrono::steady_clock::now();
		try {
			translated = Recompiler::TranslateProgram(code, compile);
		} catch (const std::exception& failure) {
			Recompiler::SetPassTimingHook(nullptr);
			g_pass_times = nullptr;
			for (const auto& [name, milliseconds]: passes) result["pass_ms"][name] = milliseconds;
			const auto text = FailureText(failure);
			result["shape"] = text;
			if (probe) {
				result["outcome"] = entry.stage == "ps" ? "needs-runtime:pixel_stage_input"
				                                        : "needs-runtime:compute_stage_input";
			} else {
				result["outcome"] = ClassifyFailure(text, has_use);
			}
			result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
			WriteJson(result_path, result);
			return;
		}
		result["phase_ms"]["translate"] = Milliseconds(std::chrono::steady_clock::now() - begin);

		begin     = std::chrono::steady_clock::now();
		auto plan = Recompiler::IR::ExtractResourcePlan(translated.program);
		result["phase_ms"]["resource_plan"] =
		    Milliseconds(std::chrono::steady_clock::now() - begin);
		Recompiler::SetPassTimingHook(nullptr);
		g_pass_times = nullptr;
		for (const auto& [name, milliseconds]: passes) result["pass_ms"][name] = milliseconds;

		Recompiler::IR::ResourceSpecialization specialization;
		if (has_use) {
			Recompiler::IR::ResourceSnapshot snapshot;
			Recompiler::IR::SrtRuntime       runtime {
			    .user_data                  = user_data,
			    .shader_base                = use_inputs.shader_base,
			    .userdata                   = &use_inputs.guest,
			    .read_specialization_memory = ReplayMemory::Read,
			    .workgroup_counts           = use_inputs.workgroup_counts,
			    .indirect_args_address      = use_inputs.indirect_args,
			};
			begin = std::chrono::steady_clock::now();
			const bool materialized =
			    Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
			result["phase_ms"]["materialize"] =
			    Milliseconds(std::chrono::steady_clock::now() - begin);
			if (!materialized) {
				if (use_inputs.guest.Misses() != 0 || use_inputs.memory_truncated) {
					result["outcome"] = "needs-runtime:guest_memory";
					result["shape"] = "a read missed the recorded ranges (first " +
					                  use_inputs.guest.FirstMiss() + ")" +
					                  (use_inputs.memory_truncated ? "; the record was truncated" : "");
				} else {
					result["outcome"] = "rejected:resource-materialization";
					result["shape"]   = "resource materialization failed with every read answered";
				}
				result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
				WriteJson(result_path, result);
				return;
			}
			if (use_inputs.uses_image_heap) {
				result["notes"] = "image heap kinds are not replayed; the module may differ";
			}
		} else if (entry.stage == "ps") {
			result["outcome"] = "needs-runtime:pixel_stage_input";
			result["shape"]   = "pixel stage state is not known at registration";
			result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
			WriteJson(result_path, result);
			return;
		} else if (!ComputeInputKnown(entry)) {
			result["outcome"] = "needs-runtime:compute_stage_input";
			result["shape"]   = "compute stage state is not known at registration";
			result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
			WriteJson(result_path, result);
			return;
		} else if (PlanNeedsRuntime(plan)) {
			result["outcome"] = "needs-runtime:user_data_values";
			result["shape"]   = "resource specialization consumes registration-time unknown values";
			result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
			WriteJson(result_path, result);
			return;
		}

		begin = std::chrono::steady_clock::now();
		const Recompiler::IR::PushData::Window push {use_inputs.push_begin, use_inputs.push_end};
		auto compiled = Recompiler::CompileProgram(std::move(translated), compile, specialization,
		                                           has_use ? push : Recompiler::IR::PushData::Window {});
		result["phase_ms"]["spirv"] = Milliseconds(std::chrono::steady_clock::now() - begin);
		result["spirv_words"]        = compiled.spirv.size();
		result["outcome"]            = "emitted";
		if (options.emit_spirv) {
			const auto folder = options.output / "spirv";
			std::filesystem::create_directories(folder);
			std::ofstream spirv(folder / (entry.stage + "_" + entry.hash.substr(2) + ".spv"),
			                    std::ios::binary | std::ios::trunc);
			spirv.write(reinterpret_cast<const char*>(compiled.spirv.data()),
			            static_cast<std::streamsize>(compiled.spirv.size() * sizeof(uint32_t)));
			if (!spirv) throw std::runtime_error("could not write SPIR-V output");
		}
	} catch (const Common::RecoverableFailure& failure) {
		Recompiler::SetPassTimingHook(nullptr);
		g_pass_times       = nullptr;
		const auto text    = FailureText(failure);
		result["shape"]   = text;
		result["outcome"] = ClassifyFailure(text, has_use);
	} catch (const std::exception& failure) {
		Recompiler::SetPassTimingHook(nullptr);
		g_pass_times       = nullptr;
		const auto text    = FailureText(failure);
		result["shape"]   = text;
		result["outcome"] = ClassifyFailure(text, has_use);
	}
	result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
	for (const auto& [name, milliseconds]: passes) result["pass_ms"][name] = milliseconds;
	WriteJson(result_path, result);
}

Json FailureResult(const CorpusEntry& entry, std::string outcome, std::string shape,
	               double elapsed_ms = 0.0) {
	auto result         = BaseResult(entry);
	result["outcome"] = std::move(outcome);
	result["shape"]   = std::move(shape);
	result["total_ms"] = elapsed_ms;
	return result;
}

std::vector<Json> RunChildren(const Options& options, const std::vector<CorpusEntry>& entries,
	                          std::string& error) {
	std::vector<Json> results(entries.size());
	const auto work = options.output / ".work";
	std::filesystem::create_directories(work);
	const uint32_t jobs = std::max(1u, options.jobs);

#if defined(_WIN32)
	error = "per-shader crash isolation is not implemented on this host";
	return {};
#else
	size_t                    next = 0;
	std::vector<RunningChild> running;
	while (next < entries.size() || !running.empty()) {
		while (next < entries.size() && running.size() < jobs) {
			const auto result_path = work / (std::to_string(next) + ".json");
			std::error_code ignored;
			std::filesystem::remove(result_path, ignored);
			const pid_t pid = ::fork();
			if (pid < 0) {
				error = "fork failed";
				return {};
			}
			if (pid == 0) {
				TranslateOne(entries[next], options, result_path);
				::_exit(0);
			}
			running.push_back({.pid = pid,
			                   .index = next,
			                   .deadline = std::chrono::steady_clock::now() +
			                               std::chrono::milliseconds(options.timeout_ms)});
			++next;
		}

		bool progressed = false;
		for (size_t slot = 0; slot < running.size();) {
			int   status = 0;
			auto& child  = running[slot];
			const auto waited = ::waitpid(child.pid, &status, WNOHANG);
			const bool timed_out = waited == 0 && std::chrono::steady_clock::now() >= child.deadline;
			if (timed_out) {
				::kill(child.pid, SIGKILL);
				while (::waitpid(child.pid, &status, 0) < 0 && errno == EINTR) {}
				results[child.index] = FailureResult(entries[child.index], "timeout",
				                                           "translation exceeded timeout");
			} else if (waited == child.pid) {
				const auto result_path = work / (std::to_string(child.index) + ".json");
				std::ifstream input(result_path);
				Json result = input ? Json::parse(input, nullptr, false) : Json {};
				if (result.is_discarded() || !result.is_object()) {
					const auto shape = WIFSIGNALED(status)
					                       ? "child terminated by signal " + std::to_string(WTERMSIG(status))
					                       : "child exited without a result";
					result = FailureResult(entries[child.index], "crashed", shape);
				}
				results[child.index] = std::move(result);
				std::error_code ignored;
				std::filesystem::remove(result_path, ignored);
			} else if (waited < 0) {
				results[child.index] =
				    FailureResult(entries[child.index], "crashed", "waitpid failed");
			} else {
				++slot;
				continue;
			}
			running.erase(running.begin() + static_cast<std::ptrdiff_t>(slot));
			progressed = true;
		}
		if (!progressed && !running.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	std::error_code ignored;
	std::filesystem::remove(work, ignored);
	return results;
#endif
}

void WriteSummary(const Options& options, const std::vector<Json>& results) {
	std::map<std::string, uint64_t> outcomes;
	std::map<std::string, std::map<std::string, uint64_t>> stages;
	std::map<std::string, std::vector<std::string>> rejection_examples;
	std::map<std::string, double> pass_totals;
	uint64_t compute_emitted = 0;
	for (const auto& result: results) {
		const auto outcome = result.value("outcome", "crashed");
		const auto stage   = result.value("stage", "unknown");
		outcomes[outcome]++;
		stages[stage][outcome]++;
		if (stage == "cs" && outcome == "emitted") compute_emitted++;
		if (outcome.starts_with("rejected:")) {
			auto& examples = rejection_examples[outcome.substr(9)];
			if (examples.size() < 3) examples.push_back(result.value("hash", "?"));
		}
		if (result.contains("pass_ms")) {
			for (const auto& [name, milliseconds]: result["pass_ms"].items()) {
				pass_totals[name] += milliseconds.get<double>();
			}
		}
	}

	std::ofstream summary(options.output / "summary.md", std::ios::trunc);
	if (!summary) throw std::runtime_error("could not write summary.md");
	summary << "# Offline shader corpus\n\n";
	summary << "Shaders: " << results.size() << "\n\n## Outcomes\n\n";
	for (const auto& [outcome, count]: outcomes) summary << "- " << outcome << ": " << count << '\n';
	summary << "\n## Outcomes by stage\n\n";
	for (const auto& [stage, counts]: stages) {
		summary << "- " << stage << ':';
		for (const auto& [outcome, count]: counts) summary << ' ' << outcome << '=' << count;
		summary << '\n';
	}
	summary << "\n## Rejection classes\n\n";
	std::vector<std::pair<std::string, uint64_t>> ranked;
	for (const auto& [outcome, count]: outcomes) {
		if (outcome.starts_with("rejected:")) ranked.emplace_back(outcome.substr(9), count);
	}
	std::ranges::sort(ranked, [](const auto& left, const auto& right) {
		return left.second != right.second ? left.second > right.second : left.first < right.first;
	});
	if (ranked.empty()) summary << "- none\n";
	for (const auto& [name, count]: ranked) {
		summary << "- " << name << ": " << count << " (examples:";
		for (const auto& hash: rejection_examples[name]) summary << ' ' << hash;
		summary << ")\n";
	}

	summary << "\n## Failure shapes (hash masked)\n\n";
	{
		std::map<std::string, uint64_t> shapes;
		for (const auto& result: results) {
			const auto outcome = result.value("outcome", "crashed");
			if (outcome.starts_with("rejected:") || outcome == "crashed" || outcome == "timeout" ||
			    outcome == "needs-runtime:use_record") {
				auto shape = NormalizeShape(result.value("shape", ""));
				if (shape.size() > 240) shape.resize(240);
				shapes[outcome + " | " + shape]++;
			}
		}
		std::vector<std::pair<std::string, uint64_t>> ranked_shapes(shapes.begin(), shapes.end());
		std::ranges::sort(ranked_shapes, [](const auto& left, const auto& right) {
			return left.second != right.second ? left.second > right.second : left.first < right.first;
		});
		if (ranked_shapes.empty()) summary << "- none\n";
		for (size_t index = 0; index < std::min(size_t {40}, ranked_shapes.size()); index++) {
			summary << "- " << ranked_shapes[index].second << " x " << ranked_shapes[index].first << '\n';
		}
	}
	uint64_t with_use = 0;
	for (const auto& result: results) with_use += result.value("source", "") == "use" ? 1 : 0;
	summary << "\nShaders replayed from a use record: " << with_use << "\n";

	auto slowest = results;
	std::ranges::sort(slowest, [](const Json& left, const Json& right) {
		return left.value("total_ms", 0.0) > right.value("total_ms", 0.0);
	});
	summary << "\nFully translatable compute shaders: " << compute_emitted << "\n";
	summary << "\n## Slowest shaders\n\n";
	for (size_t index = 0; index < std::min(size_t {30}, slowest.size()); index++) {
		const auto& result = slowest[index];
		std::string slowest_pass = "phase";
		double      slowest_ms   = 0.0;
		if (result.contains("pass_ms")) {
			for (const auto& [name, milliseconds]: result["pass_ms"].items()) {
				if (milliseconds.get<double>() > slowest_ms) {
					slowest_ms   = milliseconds.get<double>();
					slowest_pass = name;
				}
			}
		}
		summary << "- " << result.value("hash", "?") << " (" << result.value("stage", "?")
		        << "): " << result.value("total_ms", 0.0) << " ms; slowest pass "
		        << slowest_pass << " " << slowest_ms << " ms\n";
	}
	summary << "\n## Pass totals\n\n";
	if (pass_totals.empty()) summary << "- none\n";
	for (const auto& [name, milliseconds]: pass_totals) {
		summary << "- " << name << ": " << milliseconds << " ms\n";
	}
}

// One live translation line: "ShaderRecompiler <STAGE> translation end: hash=0x<16 hex> outcome=<text>".
struct LiveLine {
	std::string stage;
	std::string hash;
	bool        emitted = false;
};

std::optional<LiveLine> ParseLiveLine(std::string_view line) {
	constexpr std::string_view Marker = " translation end: hash=0x";
	const auto end = line.find(Marker);
	if (end == std::string_view::npos) return {};
	const auto head = line.rfind("ShaderRecompiler ", end);
	if (head == std::string_view::npos) return {};
	LiveLine parsed;
	const auto stage = line.substr(head + 17, end - head - 17);
	if (stage == "CS") parsed.stage = "cs";
	else if (stage == "PS") parsed.stage = "ps";
	else return {};
	const auto hash_at = end + Marker.size();
	if (line.size() < hash_at + 16) return {};
	parsed.hash = "0x" + std::string(line.substr(hash_at, 16));
	const auto outcome_at = line.find(" outcome=", hash_at + 16);
	if (outcome_at == std::string_view::npos) return {};
	const auto outcome = line.substr(outcome_at + 9);
	parsed.emitted     = outcome.starts_with("emitted") || outcome.starts_with("served from store");
	return parsed;
}

struct Agreement {
	uint64_t tool_total = 0, not_in_log = 0, both = 0, only_live = 0, only_tool = 0, neither = 0;
	std::map<std::string, uint64_t> only_live_by_tool_outcome;
};

std::map<std::string, Agreement> CompareWithLiveLog(const std::vector<std::string>& lines,
	                                                const std::vector<Json>& results) {
	std::map<std::pair<std::string, std::string>, bool> live;
	for (const auto& line: lines) {
		if (const auto parsed = ParseLiveLine(line)) {
			live[{parsed->stage, parsed->hash}] |= parsed->emitted;
		}
	}
	std::map<std::string, Agreement> stages;
	for (const auto& result: results) {
		const auto stage = result.value("stage", "");
		if (stage != "cs" && stage != "ps") continue;
		auto& agreement = stages[stage];
		agreement.tool_total++;
		const auto found = live.find({stage, result.value("hash", "")});
		if (found == live.end()) {
			agreement.not_in_log++;
			continue;
		}
		const auto outcome     = result.value("outcome", "crashed");
		const bool tool_emitted = outcome == "emitted";
		if (found->second && tool_emitted) agreement.both++;
		else if (found->second) {
			agreement.only_live++;
			agreement.only_live_by_tool_outcome[outcome]++;
		} else if (tool_emitted) agreement.only_tool++;
		else agreement.neither++;
	}
	return stages;
}

std::string AgreementText(const std::map<std::string, Agreement>& stages) {
	std::string text = "\n## Agreement with the live log (emitted in both / only live / only tool)\n\n";
	if (stages.empty()) text += "- no compute or pixel shader was translated\n";
	for (const auto& [stage, agreement]: stages) {
		const auto live_emitted = agreement.both + agreement.only_live;
		char line[256];
		std::snprintf(line, sizeof(line),
		              "- %s: tool %llu, not in log %llu; emitted in both %llu, only live %llu, "
		              "only tool %llu, neither %llu; fidelity %.1f%% of %llu live-emitted\n",
		              stage.c_str(), static_cast<unsigned long long>(agreement.tool_total),
		              static_cast<unsigned long long>(agreement.not_in_log),
		              static_cast<unsigned long long>(agreement.both),
		              static_cast<unsigned long long>(agreement.only_live),
		              static_cast<unsigned long long>(agreement.only_tool),
		              static_cast<unsigned long long>(agreement.neither),
		              live_emitted == 0 ? 0.0 : 100.0 * static_cast<double>(agreement.both) /
		                                            static_cast<double>(live_emitted),
		              static_cast<unsigned long long>(live_emitted));
		text += line;
		for (const auto& [outcome, count]: agreement.only_live_by_tool_outcome) {
			text += "  - only live, tool said " + outcome + ": " + std::to_string(count) + "\n";
		}
	}
	return text;
}

int Run(const Options& options) {
	std::error_code error_code;
	std::filesystem::create_directories(options.output, error_code);
	if (error_code) {
		std::fprintf(stderr, "shader_corpus_tool: could not create output directory: %s\n",
		             error_code.message().c_str());
		return 1;
	}
	std::vector<CorpusEntry> entries;
	std::string              error;
	if (!LoadEntries(options, entries, error)) {
		std::fprintf(stderr, "shader_corpus_tool: %s\n", error.c_str());
		return 1;
	}
	auto results = RunChildren(options, entries, error);
	if (!error.empty()) {
		std::fprintf(stderr, "shader_corpus_tool: %s\n", error.c_str());
		return 1;
	}
	std::ofstream jsonl(options.output / "results.jsonl", std::ios::trunc);
	if (!jsonl) return 1;
	for (const auto& result: results) jsonl << result.dump() << '\n';
	jsonl.close();
	WriteSummary(options, results);
	if (!options.compare_log.empty()) {
		std::ifstream log(options.compare_log);
		if (!log) {
			std::fprintf(stderr, "shader_corpus_tool: could not open the run log\n");
			return 1;
		}
		std::vector<std::string> lines;
		for (std::string line; std::getline(log, line);) {
			if (line.find(" translation end: hash=0x") != std::string::npos) lines.push_back(line);
		}
		const auto text = AgreementText(CompareWithLiveLog(lines, results));
		std::ofstream summary(options.output / "summary.md", std::ios::app);
		summary << text;
	}
	// The part of the summary a caller wants on its terminal; timings stay in the file.
	std::ifstream written(options.output / "summary.md");
	bool          skipping = false;
	for (std::string line; std::getline(written, line);) {
		if (line.starts_with("## ")) {
			skipping = line == "## Slowest shaders" || line == "## Pass totals";
		}
		if (!skipping) std::printf("%s\n", line.c_str());
	}
	return 0;
}

void WriteFixture(const std::filesystem::path& input, std::ofstream& index, std::string stage,
	              std::string hash, std::span<const uint32_t> code, bool compute_known) {
	const auto file = stage + "_" + hash.substr(2) + ".bin";
	std::ofstream binary(input / file, std::ios::binary | std::ios::trunc);
	binary.write(reinterpret_cast<const char*>(code.data()),
	             static_cast<std::streamsize>(code.size_bytes()));
	Json record {{"hash", hash},
	             {"stage", stage},
	             {"file", file},
	             {"words", code.size()},
	             {"user_data_count", compute_known ? Json(0) : Json(nullptr)},
	             {"stage_input_info",
	              {{"known", compute_known},
	               {"threads_num", {1, 1, 1}},
	               {"scratch_size_dwords", 0},
	               {"lds_size_dwords", 0},
	               {"wave_size", 64},
	               {"float_mode", 192},
	               {"group_id", {false, false, false}},
	               {"thread_ids_num", 1},
	               {"tg_size_en", false}}}};
	index << record.dump() << '\n';
}

// A use record as the emulator's writer produces it: same serializer, files in `input`.
void WriteUseFixture(const std::filesystem::path& input, std::ofstream& use_index,
	                 const std::string& stage, uint64_t hash, size_t info_bytes,
	                 std::span<const uint8_t> memory) {
	Common::ShaderRegLedger::UseRecord record;
	record.stage = stage;
	record.hash  = hash;
	if (stage == "cs") {
		ShaderComputeInputInfo info {};
		info.threads_num[0] = info.threads_num[1] = info.threads_num[2] = 1;
		info.wave_size                                                    = 64;
		record.input_info.resize(info_bytes);
		std::memcpy(record.input_info.data(), &info, std::min(info_bytes, sizeof(info)));
	}
	record.options_wave_size = 64;
	record.push_begin        = 0;
	record.push_end          = 32;
	record.ranges.push_back({0x1000, memory.size()});
	record.memory.assign(memory.begin(), memory.end());
	std::string          line;
	std::vector<uint8_t> blob;
	Common::ShaderRegLedger::SerializeUseRecord(record, 0, line, blob);
	const auto name = Json::parse(line)["file"].get<std::string>();
	std::ofstream file(input / name, std::ios::binary | std::ios::trunc);
	file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
	use_index << line << '\n';
}

std::string TestHash(uint64_t value) {
	char text[32];
	std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(value));
	return text;
}

std::string LiveText(const char* stage, uint64_t hash, const char* outcome) {
	return std::string("ShaderRecompiler ") + stage + " translation end: hash=" + TestHash(hash) +
	       " outcome=" + outcome;
}

bool Check(bool condition, const char* what) {
	if (!condition) std::fprintf(stderr, "shader_corpus_tool self-test: %s\n", what);
	return condition;
}

bool SelfTestUnits() {
	bool ok = true;
	ok &= Check(ClassifyFailure("shader resource tracking: hash=" + TestHash(1) + " pc=0x1 x", false) ==
	                "needs-runtime:use_record",
	            "a tracking failure without a use record must be missing input");
	ok &= Check(ClassifyFailure("shader resource tracking: hash=" + TestHash(1) + " pc=0x1 x", true) ==
	                "rejected:shader-resource-tracking",
	            "a tracking failure with a use record is a rejection");
	ok &= Check(ClassifyFailure("unsupported thing at pc 0x0", false) == "rejected:unsupported-thing-at-pc-0x0",
	            "an input-independent failure stays a rejection");
	ok &= Check(NormalizeShape("a hash=" + TestHash(0x123456789abcdefull) + " pc=0x00000010") ==
	                "a hash=<hash> pc=0x00000010",
	            "hash masking");
	{
		ReplayMemory memory;
		const std::array<uint8_t, 8> first {1, 0, 0, 0, 2, 0, 0, 0};
		const std::array<uint8_t, 8> second {3, 0, 0, 0, 4, 0, 0, 0};
		memory.Add(0x2000, second);
		memory.Add(0x1ffc, first);
		memory.Seal();
		std::array<uint32_t, 3> words {};
		ok &= Check(ReplayMemory::Read(&memory, 0x1ffc, words), "a read across merged ranges");
		ok &= Check(words[0] == 1 && words[1] == 2 && words[2] == 4, "merged range content");
		std::array<uint32_t, 1> far {};
		ok &= Check(!ReplayMemory::Read(&memory, 0x9000, far) && memory.Misses() == 1,
		            "a read outside the ranges is a counted miss");
	}
	{
		const std::vector<std::string> log {
		    "x " + LiveText("CS", 1, "emitted (async)"), LiveText("CS", 2, "failed (x)"),
		    LiveText("PS", 3, "served from store"), LiveText("CS", 4, "emitted")};
		std::vector<Json> results {
		    {{"stage", "cs"}, {"hash", TestHash(1)}, {"outcome", "emitted"}},
		    {{"stage", "cs"}, {"hash", TestHash(2)}, {"outcome", "emitted"}},
		    {{"stage", "ps"}, {"hash", TestHash(3)}, {"outcome", "needs-runtime:x"}},
		    {{"stage", "cs"}, {"hash", TestHash(4)}, {"outcome", "needs-runtime:y"}},
		    {{"stage", "cs"}, {"hash", TestHash(5)}, {"outcome", "emitted"}}};
		const auto stages = CompareWithLiveLog(log, results);
		const auto& cs    = stages.at("cs");
		ok &= Check(cs.both == 1 && cs.only_tool == 1 && cs.only_live == 1 && cs.not_in_log == 1,
		            "compute agreement counts");
		ok &= Check(stages.at("ps").only_live == 1, "pixel agreement counts");
	}
	return ok;
}

int SelfTest() {
#if defined(_WIN32)
	return 0;
#else
	if (!SelfTestUnits()) return 1;
	const auto root = std::filesystem::temp_directory_path() /
	                  ("kyty-shader-corpus-test-" + std::to_string(::getpid()));
	const auto input  = root / "input";
	const auto output = root / "output";
	std::filesystem::create_directories(input);
	std::ofstream index(input / "index.jsonl");
	std::ofstream use_index(input / "use.jsonl");
	constexpr std::array<uint32_t, 1> emitted {0xbf810000u};
	constexpr std::array<uint32_t, 2> rejected {0xb96af818u, 0xbf810000u};
	WriteFixture(input, index, "cs", TestHash(1), emitted, true);
	WriteFixture(input, index, "cs", TestHash(2), rejected, true);
	WriteFixture(input, index, "ps", TestHash(3), emitted, false);
	// Registered without compute state: only its use record lets the tool translate it.
	WriteFixture(input, index, "cs", TestHash(4), emitted, false);
	// Registered without compute state and used with a stage struct this tool cannot read.
	WriteFixture(input, index, "cs", TestHash(5), emitted, false);
	index.close();
	constexpr std::array<uint8_t, 16> memory {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	WriteUseFixture(input, use_index, "cs", 4, sizeof(ShaderComputeInputInfo), memory);
	WriteUseFixture(input, use_index, "cs", 5, sizeof(ShaderComputeInputInfo) - 4, memory);
	use_index.close();
	Options options {.input = input, .output = output, .jobs = 2, .timeout_ms = 5000};
	const int status = Run(options);
	std::vector<Json> results;
	std::string error;
	const bool read = ReadJsonLines(output / "results.jsonl", results, error);
	bool have_emitted = false, have_rejected = false, have_runtime = false;
	bool use_emitted = false, layout_not_rejected = false, no_false_rejection = true;
	for (const auto& result: results) {
		const auto outcome = result.value("outcome", "");
		const auto hash    = result.value("hash", "");
		have_emitted |= outcome == "emitted";
		have_rejected |= outcome.starts_with("rejected:");
		have_runtime |= outcome.starts_with("needs-runtime:");
		use_emitted |= hash == TestHash(4) && outcome == "emitted" &&
		               result.value("source", "") == "use";
		layout_not_rejected |= hash == TestHash(5) &&
		                       outcome == "needs-runtime:input_layout";
		if (hash == TestHash(3) && outcome.starts_with("rejected:")) no_false_rejection = false;
		if (result.value("shape", "").empty() && outcome != "emitted") no_false_rejection = false;
	}
	std::error_code ignored;
	std::filesystem::remove_all(root, ignored);
	if (status != 0 || !read || results.size() != 5 || !have_emitted || !have_rejected ||
	    !have_runtime || !use_emitted || !layout_not_rejected || !no_false_rejection) {
		std::fprintf(stderr,
		             "shader_corpus_tool self-test failed: count=%zu emitted=%d rejected=%d runtime=%d "
		             "use_emitted=%d layout=%d no_false_rejection=%d %s\n",
		             results.size(), have_emitted, have_rejected, have_runtime, use_emitted,
		             layout_not_rejected, no_false_rejection, error.c_str());
		return 1;
	}
	return 0;
#endif
}

void Usage() {
	std::printf("shader_corpus_tool --dir <dump dir> --out <report dir> [--jobs N] "
	            "[--only <stage>] [--limit N] [--timeout-ms N] [--emit-spirv] [--compare-log <run log>]\n");
}

bool ParseUnsigned(const char* text, uint32_t& value) {
	char*      end    = nullptr;
	const auto parsed = std::strtoul(text, &end, 10);
	if (end == text || *end != '\0' || parsed > UINT32_MAX) return false;
	value = static_cast<uint32_t>(parsed);
	return true;
}

std::optional<Options> ParseOptions(int argc, char** argv) {
	Options options;
	options.jobs = std::max(1u, std::thread::hardware_concurrency());
	for (int index = 1; index < argc; index++) {
		const std::string_view argument = argv[index];
		if (argument == "--self-test") {
			options.self_test = true;
			continue;
		}
		if (argument == "--emit-spirv") {
			options.emit_spirv = true;
			continue;
		}
		if (index + 1 >= argc) return {};
		const char* value = argv[++index];
		if (argument == "--dir") options.input = value;
		else if (argument == "--out") options.output = value;
		else if (argument == "--only") options.only = value;
		else if (argument == "--jobs") {
			if (!ParseUnsigned(value, options.jobs) || options.jobs == 0) return {};
		} else if (argument == "--compare-log") {
			options.compare_log = value;
		} else if (argument == "--limit") {
			if (!ParseUnsigned(value, options.limit)) return {};
		} else if (argument == "--timeout-ms") {
			if (!ParseUnsigned(value, options.timeout_ms) || options.timeout_ms == 0) return {};
		} else return {};
	}
	if (!options.self_test && (options.input.empty() || options.output.empty())) return {};
	return options;
}

} // namespace

int main(int argc, char** argv) {
	auto options = ParseOptions(argc, argv);
	if (!options) {
		Usage();
		return 1;
	}
	return options->self_test ? SelfTest() : Run(*options);
}
