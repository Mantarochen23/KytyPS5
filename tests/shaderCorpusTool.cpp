#include "common/emulatorConfig.h"
#include "common/recoverable.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
	uint32_t              spirv_profile = 0;
	std::string           only;
	bool                  emit_spirv = false;
	bool                  time_pipeline = false;
	bool                  self_test  = false;
};

struct CorpusEntry {
	Json                  record;
	std::string           hash;
	std::string           stage;
	std::filesystem::path binary;
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

bool AnalysisRequested(const Options& options) {
	return options.spirv_profile != 0 || options.time_pipeline;
}

std::filesystem::path AnalysisModulePath(const Options& options, size_t index) {
	return options.output / ".analysis-modules" / (std::to_string(index) + ".spv");
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

void WriteSpirv(const std::filesystem::path& path, std::span<const uint32_t> words) {
	std::filesystem::create_directories(path.parent_path());
	std::ofstream output(path, std::ios::binary | std::ios::trunc);
	if (!output) throw std::runtime_error("could not create SPIR-V output");
	output.write(reinterpret_cast<const char*>(words.data()),
	             static_cast<std::streamsize>(words.size_bytes()));
	if (!output) throw std::runtime_error("could not write SPIR-V output");
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

void TranslateOne(const CorpusEntry& entry, size_t entry_index, const Options& options,
	              const std::filesystem::path& result_path) {
	Json result = BaseResult(entry);
	const auto total_begin = std::chrono::steady_clock::now();
	std::map<std::string, double> passes;
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

		ShaderComputeInputInfo compute {};
		ShaderPixelInputInfo   pixel {};
		ShaderVertexInputInfo  vertex {};
		PopulateComputeInfo(entry, compute);
		std::vector<uint32_t> user_data;
		if (entry.record.contains("user_data_count") &&
		    entry.record["user_data_count"].is_number_unsigned()) {
			user_data.resize(std::min(entry.record["user_data_count"].get<size_t>(), size_t {64}));
		}

		Recompiler::CompileOptions compile;
		compile.dump_ir     = false;
		compile.shader_hash = ParseHash(entry.hash).value_or(0);
		compile.user_data   = user_data;
		if (entry.stage == "cs") {
			compile.stage              = ShaderType::Compute;
			compile.wave_size          = compute.wave_size;
			compile.input_info.compute = &compute;
			result["wave_size"]        = compute.wave_size;
		} else {
			compile.stage            = ShaderType::Pixel;
			compile.input_info.pixel = &pixel;
			result["wave_size"]      = pixel.wave_size;
		}

		g_pass_times = &passes;
		Recompiler::SetPassTimingHook(PassTiming);
		begin = std::chrono::steady_clock::now();
		auto translated = Recompiler::TranslateProgram(code, compile);
		result["phase_ms"]["translate"] = Milliseconds(std::chrono::steady_clock::now() - begin);

		begin     = std::chrono::steady_clock::now();
		auto plan = Recompiler::IR::ExtractResourcePlan(translated.program);
		result["phase_ms"]["resource_plan"] =
		    Milliseconds(std::chrono::steady_clock::now() - begin);
		Recompiler::SetPassTimingHook(nullptr);
		g_pass_times = nullptr;
		for (const auto& [name, milliseconds]: passes) result["pass_ms"][name] = milliseconds;

		if (entry.stage == "ps") {
			result["outcome"] = "needs-runtime:pixel_stage_input";
			result["shape"]   = "pixel stage state is not known at registration";
		} else if (!ComputeInputKnown(entry)) {
			result["outcome"] = "needs-runtime:compute_stage_input";
			result["shape"]   = "compute stage state is not known at registration";
		} else if (PlanNeedsRuntime(plan)) {
			result["outcome"] = "needs-runtime:user_data_values";
			result["shape"]   = "resource specialization consumes registration-time unknown values";
		} else {
			begin = std::chrono::steady_clock::now();
			auto compiled = Recompiler::CompileProgram(std::move(translated), compile, {});
			result["phase_ms"]["spirv"] = Milliseconds(std::chrono::steady_clock::now() - begin);
			result["spirv_words"]        = compiled.spirv.size();
			result["outcome"]            = "emitted";
			if (AnalysisRequested(options)) {
				WriteSpirv(AnalysisModulePath(options, entry_index), compiled.spirv);
			}
			if (options.emit_spirv) {
				const auto folder = options.output / "spirv";
				WriteSpirv(folder / (entry.stage + "_" + entry.hash.substr(2) + ".spv"),
				           compiled.spirv);
			}
		}
	} catch (const Common::RecoverableFailure& failure) {
		Recompiler::SetPassTimingHook(nullptr);
		g_pass_times       = nullptr;
		result["shape"]   = failure.what();
		result["outcome"] = "rejected:" + RejectionClass(failure.what());
	} catch (const std::exception& failure) {
		Recompiler::SetPassTimingHook(nullptr);
		g_pass_times       = nullptr;
		result["shape"]   = failure.what();
		result["outcome"] = "rejected:" + RejectionClass(failure.what());
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
				TranslateOne(entries[next], next, options, result_path);
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

struct SpirvInstruction {
	size_t   offset = 0;
	uint16_t opcode = 0;
	uint16_t words  = 0;
};

std::vector<uint32_t> ReadSpirv(const std::filesystem::path& path) {
	std::ifstream input(path, std::ios::binary | std::ios::ate);
	if (!input) throw std::runtime_error("could not open analysis SPIR-V module");
	const auto bytes = input.tellg();
	if (bytes < static_cast<std::streamoff>(5 * sizeof(uint32_t)) ||
	    static_cast<uint64_t>(bytes) % sizeof(uint32_t) != 0) {
		throw std::runtime_error("analysis SPIR-V module has an invalid size");
	}
	std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
	input.seekg(0);
	input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
	if (!input) throw std::runtime_error("could not read analysis SPIR-V module");
	return words;
}

std::vector<SpirvInstruction> ParseSpirv(std::span<const uint32_t> words) {
	if (words.size() < 5 || words.front() != spv::MagicNumber) {
		throw std::runtime_error("analysis module is not SPIR-V");
	}
	std::vector<SpirvInstruction> instructions;
	for (size_t offset = 5; offset < words.size();) {
		const auto count  = static_cast<uint16_t>(words[offset] >> spv::WordCountShift);
		const auto opcode = static_cast<uint16_t>(words[offset] & spv::OpCodeMask);
		if (count == 0 || offset + count > words.size()) {
			throw std::runtime_error("analysis module contains a malformed instruction");
		}
		instructions.push_back({offset, opcode, count});
		offset += count;
	}
	return instructions;
}

bool IsTypeOrConstant(uint16_t opcode) {
	return (opcode >= spv::OpTypeVoid && opcode <= spv::OpTypeForwardPointer) ||
	       (opcode >= spv::OpConstantTrue && opcode <= spv::OpSpecConstantOp) ||
	       opcode == spv::OpUndef;
}

bool IsDebugInstruction(uint16_t opcode) {
	switch (opcode) {
		case spv::OpSourceContinued:
		case spv::OpSource:
		case spv::OpSourceExtension:
		case spv::OpName:
		case spv::OpMemberName:
		case spv::OpString:
		case spv::OpLine:
		case spv::OpNoLine:
		case spv::OpModuleProcessed: return true;
		default: return false;
	}
}

bool IsControlScaffolding(uint16_t opcode) {
	switch (opcode) {
		case spv::OpPhi:
		case spv::OpLoopMerge:
		case spv::OpSelectionMerge:
		case spv::OpLabel:
		case spv::OpBranch:
		case spv::OpBranchConditional:
		case spv::OpSwitch: return true;
		default: return false;
	}
}

bool IsBoundsCheck(uint16_t opcode) {
	return (opcode >= spv::OpIEqual && opcode <= spv::OpFUnordGreaterThanEqual) ||
	       opcode == spv::OpSelect;
}

bool IsWaveInstruction(uint16_t opcode) {
	return opcode >= spv::OpGroupNonUniformElect && opcode <= spv::OpGroupNonUniformQuadSwap;
}

bool IsImageInstruction(uint16_t opcode) {
	return (opcode >= spv::OpSampledImage && opcode <= spv::OpImage) ||
	       opcode == spv::OpImageTexelPointer;
}

bool IsAtomicInstruction(uint16_t opcode) {
	return opcode >= spv::OpAtomicLoad && opcode <= spv::OpAtomicXor;
}

bool IsResourceStorage(uint32_t storage) {
	return storage == spv::StorageClassUniformConstant || storage == spv::StorageClassUniform ||
	       storage == spv::StorageClassStorageBuffer ||
	       storage == spv::StorageClassPhysicalStorageBuffer;
}

uint32_t SequenceHash(std::span<const SpirvInstruction> instructions) {
	uint32_t hash = 2166136261u;
	for (const auto& instruction: instructions) {
		hash ^= static_cast<uint32_t>(instruction.opcode) |
		        (static_cast<uint32_t>(instruction.words) << 16u);
		hash *= 16777619u;
	}
	return hash;
}

Json RepeatedSequences(const std::vector<SpirvInstruction>& instructions, size_t module_words) {
	struct Repeat {
		uint32_t hash         = 0;
		uint32_t window       = 0;
		uint32_t occurrences  = 0;
		uint64_t covered_words = 0;
	};
	std::vector<Repeat> repeats;
	for (const uint32_t window: {8u, 16u, 32u}) {
		if (instructions.size() < window) continue;
		std::unordered_map<uint32_t, std::vector<uint32_t>> starts;
		for (uint32_t start = 0; start + window <= instructions.size(); start++) {
			starts[SequenceHash(std::span(instructions).subspan(start, window))].push_back(start);
		}
		for (const auto& [hash, positions]: starts) {
			uint32_t occurrences = 0;
			uint32_t next         = 0;
			uint64_t covered      = 0;
			for (const auto start: positions) {
				if (occurrences != 0 && start < next) continue;
				uint64_t words = 0;
				for (uint32_t i = 0; i < window; i++) words += instructions[start + i].words;
				covered += words;
				occurrences++;
				next = start + window;
			}
			if (occurrences >= 2) repeats.push_back({hash, window, occurrences, covered});
		}
	}
	std::ranges::sort(repeats, [](const Repeat& left, const Repeat& right) {
		return left.covered_words != right.covered_words
		           ? left.covered_words > right.covered_words
		           : std::pair {left.window, left.hash} < std::pair {right.window, right.hash};
	});
	Json result = Json::array();
	for (size_t i = 0; i < std::min(size_t {20}, repeats.size()); i++) {
		const auto& repeat = repeats[i];
		result.push_back({{"sequence_id", repeat.hash},
		                  {"instruction_window", repeat.window},
		                  {"non_overlapping_occurrences", repeat.occurrences},
		                  {"covered_words", repeat.covered_words},
		                  {"module_share_percent",
		                   module_words == 0 ? 0.0
		                                     : 100.0 * static_cast<double>(repeat.covered_words) /
		                                           static_cast<double>(module_words)}});
	}
	return result;
}

Json ProfileSpirv(std::span<const uint32_t> words) {
	const auto instructions = ParseSpirv(words);
	std::unordered_map<uint32_t, uint32_t> pointer_types;
	std::unordered_map<uint32_t, uint32_t> pointer_values;
	std::unordered_set<uint32_t> constants;
	for (const auto& instruction: instructions) {
		const auto* operands = words.data() + instruction.offset;
		const auto  opcode   = instruction.opcode;
		if (opcode == spv::OpTypePointer && instruction.words >= 4) {
			pointer_types[operands[1]] = operands[2];
		} else if (opcode >= spv::OpConstantTrue && opcode <= spv::OpSpecConstantOp &&
		           instruction.words >= 3) {
			constants.insert(operands[2]);
		}
		if (instruction.words >= 3) {
			const auto type_it = pointer_types.find(operands[1]);
			if (type_it != pointer_types.end()) pointer_values[operands[2]] = type_it->second;
		}
		if (opcode == spv::OpVariable && instruction.words >= 4) {
			pointer_values[operands[2]] = operands[3];
		}
	}

	struct OpcodeCount { uint64_t instructions = 0; uint64_t words = 0; };
	std::map<uint16_t, OpcodeCount> opcodes;
	std::map<std::string, uint64_t> categories;
	uint64_t functions = 0;
	for (const auto& instruction: instructions) {
		auto& count = opcodes[instruction.opcode];
		count.instructions++;
		count.words += instruction.words;
		if (instruction.opcode == spv::OpFunction) functions++;

		const auto* operands = words.data() + instruction.offset;
		uint32_t pointer_storage = UINT32_MAX;
		if ((instruction.opcode == spv::OpLoad || IsAtomicInstruction(instruction.opcode)) &&
		    instruction.words >= 4) {
			if (const auto it = pointer_values.find(operands[3]); it != pointer_values.end()) {
				pointer_storage = it->second;
			}
		} else if (instruction.opcode == spv::OpStore && instruction.words >= 2) {
			if (const auto it = pointer_values.find(operands[1]); it != pointer_values.end()) {
				pointer_storage = it->second;
			}
		}

		bool indirect_descriptor = false;
		if ((instruction.opcode == spv::OpAccessChain ||
		     instruction.opcode == spv::OpInBoundsAccessChain ||
		     instruction.opcode == spv::OpPtrAccessChain) && instruction.words >= 5) {
			const auto storage = pointer_values.find(operands[2]);
			const bool resource = storage != pointer_values.end() && IsResourceStorage(storage->second);
			for (uint16_t index = 4; resource && index < instruction.words; index++) {
				indirect_descriptor |= !constants.contains(operands[index]);
			}
		}

		const char* category = "other";
		if (IsDebugInstruction(instruction.opcode)) category = "debug_names";
		else if (IsTypeOrConstant(instruction.opcode)) category = "constants_types";
		else if (indirect_descriptor) category = "indirect_descriptor_search";
		else if (pointer_storage == spv::StorageClassFunction ||
		         (instruction.opcode == spv::OpVariable && instruction.words >= 4 &&
		          operands[3] == spv::StorageClassFunction)) {
			category = "function_register_load_store";
		} else if (IsImageInstruction(instruction.opcode) || IsResourceStorage(pointer_storage)) {
			category = "resource_access_helpers";
		} else if (IsWaveInstruction(instruction.opcode)) category = "wave_lane_emulation";
		else if (IsBoundsCheck(instruction.opcode)) category = "bounds_checks";
		else if (IsControlScaffolding(instruction.opcode)) category = "phi_merge_scaffolding";
		categories[category] += instruction.words;
	}
	categories["module_header"] += 5;

	std::vector<std::pair<uint16_t, OpcodeCount>> ranked(opcodes.begin(), opcodes.end());
	std::ranges::sort(ranked, [](const auto& left, const auto& right) {
		return left.second.words != right.second.words ? left.second.words > right.second.words
		                                               : left.first < right.first;
	});
	Json histogram = Json::array();
	for (const auto& [opcode, count]: ranked) {
		histogram.push_back({{"opcode", opcode},
		                     {"name", spvOpcodeString(opcode)},
		                     {"instructions", count.instructions},
		                     {"words", count.words},
		                     {"module_share_percent",
		                      100.0 * static_cast<double>(count.words) /
		                          static_cast<double>(words.size())}});
	}
	Json attributed = Json::object();
	for (const auto& [name, count]: categories) {
		attributed[name] = {{"words", count},
		                    {"module_share_percent",
		                     100.0 * static_cast<double>(count) /
		                         static_cast<double>(words.size())}};
	}
	return {{"words", words.size()},
	        {"instructions", instructions.size()},
	        {"functions", functions},
	        {"attribution_method",
	         "exclusive opcode/pointer-storage lower bounds; comparisons and selects are counted as bounds checks"},
	        {"attributed_words", std::move(attributed)},
	        {"opcode_histogram", std::move(histogram)},
	        {"repeated_sequences", RepeatedSequences(instructions, words.size())}};
}

bool ValidateSpirv(std::span<const uint32_t> words, std::string& error) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position,
	                             const char* message) {
		error = "word " + std::to_string(position.index) + ": " + message;
	});
	return tools.Validate(words.data(), words.size());
}

std::vector<uint32_t> OptimizeSpirv(std::span<const uint32_t> words, bool size_recipe,
	                                double& milliseconds, std::string& error) {
	spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_3);
	optimizer.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position,
	                                 const char* message) {
		error = "word " + std::to_string(position.index) + ": " + message;
	});
	if (size_recipe) optimizer.RegisterSizePasses();
	else optimizer.RegisterPerformancePasses();
	std::vector<uint32_t> optimized;
	const auto begin = std::chrono::steady_clock::now();
	const bool valid = optimizer.Run(words.data(), words.size(), &optimized);
	milliseconds = Milliseconds(std::chrono::steady_clock::now() - begin);
	if (!valid) optimized.clear();
	return optimized;
}

struct PipelineMeasurement {
	double      module_ms   = 0.0;
	double      pipeline_ms = 0.0;
	std::string error;
};

class HeadlessVulkan {
public:
	~HeadlessVulkan() {
		if (m_device != nullptr) {
			if (m_layout != nullptr) m_device.destroyPipelineLayout(m_layout, nullptr);
			m_device.destroy(nullptr);
		}
		if (m_instance != nullptr) m_instance.destroy(nullptr);
	}

	bool Initialize(std::string& error) {
		static vk::detail::DynamicLoader loader;
		const auto get_instance_proc_addr =
		    loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		if (get_instance_proc_addr == nullptr) {
			error = "could not load the Vulkan loader";
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);

		vk::ApplicationInfo app {};
		app.pApplicationName = "shader_corpus_tool";
		app.apiVersion        = VK_API_VERSION_1_3;
		vk::InstanceCreateInfo instance_info {};
		instance_info.pApplicationInfo = &app;
		const auto instance_result = vk::createInstance(&instance_info, nullptr, &m_instance);
		if (instance_result != vk::Result::eSuccess) {
			error = "vkCreateInstance: " + vk::to_string(instance_result);
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(m_instance);

		uint32_t physical_count = 0;
		auto result = m_instance.enumeratePhysicalDevices(&physical_count, nullptr);
		if (result != vk::Result::eSuccess || physical_count == 0) {
			error = result != vk::Result::eSuccess
		            ? "vkEnumeratePhysicalDevices: " + vk::to_string(result)
		            : "no Vulkan physical device";
			return false;
		}
		std::vector<vk::PhysicalDevice> physical_devices(physical_count);
		result = m_instance.enumeratePhysicalDevices(&physical_count, physical_devices.data());
		if (result != vk::Result::eSuccess) {
			error = "vkEnumeratePhysicalDevices: " + vk::to_string(result);
			return false;
		}
		for (const auto physical: physical_devices) {
			uint32_t queue_count = 0;
			physical.getQueueFamilyProperties(&queue_count, nullptr);
			std::vector<vk::QueueFamilyProperties> queues(queue_count);
			physical.getQueueFamilyProperties(&queue_count, queues.data());
			for (uint32_t index = 0; index < queue_count; index++) {
				if ((queues[index].queueFlags & vk::QueueFlagBits::eCompute) !=
				    vk::QueueFlagBits::eCompute) continue;
				m_physical     = physical;
				m_queue_family = index;
				break;
			}
			if (m_physical != nullptr) break;
		}
		if (m_physical == nullptr) {
			error = "no Vulkan compute queue";
			return false;
		}

		vk::PhysicalDeviceVulkan11Features features11 {};
		vk::PhysicalDeviceVulkan12Features features12 {};
		vk::PhysicalDeviceVulkan13Features features13 {};
		features13.pNext = &features12;
		features12.pNext = &features11;
		vk::PhysicalDeviceFeatures2 features2 {};
		features2.pNext = &features13;
		m_physical.getFeatures2(&features2);

		const float priority = 1.0f;
		vk::DeviceQueueCreateInfo queue_info {};
		queue_info.queueFamilyIndex = m_queue_family;
		queue_info.queueCount       = 1;
		queue_info.pQueuePriorities = &priority;
		vk::DeviceCreateInfo device_info {};
		device_info.pNext                   = &features13;
		device_info.queueCreateInfoCount    = 1;
		device_info.pQueueCreateInfos       = &queue_info;
		device_info.pEnabledFeatures        = &features2.features;
		result = m_physical.createDevice(&device_info, nullptr, &m_device);
		if (result != vk::Result::eSuccess) {
			error = "vkCreateDevice: " + vk::to_string(result);
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(m_device);

		vk::PipelineLayoutCreateInfo layout_info {};
		result = m_device.createPipelineLayout(&layout_info, nullptr, &m_layout);
		if (result != vk::Result::eSuccess) {
			error = "vkCreatePipelineLayout: " + vk::to_string(result);
			return false;
		}
		return true;
	}

	PipelineMeasurement Measure(std::span<const uint32_t> words) const {
		PipelineMeasurement measurement;
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize = words.size_bytes();
		module_info.pCode    = words.data();
		vk::ShaderModule module = nullptr;
		auto begin = std::chrono::steady_clock::now();
		auto result = m_device.createShaderModule(&module_info, nullptr, &module);
		measurement.module_ms = Milliseconds(std::chrono::steady_clock::now() - begin);
		if (result != vk::Result::eSuccess) {
			measurement.error = "vkCreateShaderModule: " + vk::to_string(result);
			return measurement;
		}

		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline_info.stage.module = module;
		pipeline_info.stage.pName  = "main";
		pipeline_info.layout       = m_layout;
		vk::Pipeline pipeline      = nullptr;
		begin = std::chrono::steady_clock::now();
		result = m_device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
		measurement.pipeline_ms = Milliseconds(std::chrono::steady_clock::now() - begin);
		if (pipeline != nullptr) m_device.destroyPipeline(pipeline, nullptr);
		m_device.destroyShaderModule(module, nullptr);
		if (result != vk::Result::eSuccess) {
			measurement.error = "vkCreateComputePipelines: " + vk::to_string(result);
		}
		return measurement;
	}

private:
	vk::Instance       m_instance {};
	vk::PhysicalDevice m_physical {};
	vk::Device         m_device {};
	vk::PipelineLayout m_layout {};
	uint32_t           m_queue_family = 0;
};

struct EmittedModule {
	size_t   result_index = 0;
	uint64_t words        = 0;
};

std::vector<EmittedModule> EmittedModules(const std::vector<Json>& results) {
	std::vector<EmittedModule> modules;
	for (size_t index = 0; index < results.size(); index++) {
		if (results[index].value("outcome", "") == "emitted" &&
		    results[index].value("stage", "") == "cs") {
			modules.push_back({index, results[index].value("spirv_words", uint64_t {0})});
		}
	}
	std::ranges::sort(modules, [](const auto& left, const auto& right) {
		return left.words != right.words ? left.words < right.words
		                                 : left.result_index < right.result_index;
	});
	return modules;
}

std::vector<EmittedModule> QuantileSample(const std::vector<EmittedModule>& modules,
	                                      size_t count) {
	if (modules.size() <= count) return modules;
	std::vector<EmittedModule> sample;
	std::unordered_set<size_t>  selected;
	for (size_t index = 0; index < count; index++) {
		const auto position = static_cast<size_t>(std::llround(
		    static_cast<double>(index) * static_cast<double>(modules.size() - 1) /
		    static_cast<double>(count - 1)));
		if (selected.insert(position).second) sample.push_back(modules[position]);
	}
	return sample;
}

void WriteSpirvProfiles(const Options& options, const std::vector<Json>& results,
	                    const std::vector<EmittedModule>& modules) {
	if (options.spirv_profile == 0) return;
	std::ofstream output(options.output / "spirv-profile.jsonl", std::ios::trunc);
	if (!output) throw std::runtime_error("could not create spirv-profile.jsonl");
	const auto count = std::min<size_t>(options.spirv_profile, modules.size());
	for (size_t rank = 0; rank < count; rank++) {
		const auto& module = modules[modules.size() - 1 - rank];
		auto profile = ProfileSpirv(ReadSpirv(AnalysisModulePath(options, module.result_index)));
		profile["rank"]         = rank + 1;
		profile["corpus_index"] = module.result_index;
		profile["blocks"]       = results[module.result_index].value("blocks", 0u);
		profile["loops"]        = results[module.result_index].value("loops", 0u);
		output << profile.dump() << '\n';

		std::string top = "none";
		uint64_t top_words = 0;
		for (const auto& [name, value]: profile["attributed_words"].items()) {
			if (name != "other" && value["words"].get<uint64_t>() > top_words) {
				top       = name;
				top_words = value["words"].get<uint64_t>();
			}
		}
		std::printf("SPIRV_PROFILE rank=%zu words=%" PRIu64 " blocks=%u loops=%u functions=%" PRIu64
		            " top=%s top_words=%" PRIu64 "\n",
		            rank + 1, module.words, profile["blocks"].get<uint32_t>(),
		            profile["loops"].get<uint32_t>(), profile["functions"].get<uint64_t>(),
		            top.c_str(), top_words);
	}
}

void WritePipelineTimings(const Options& options, const std::vector<EmittedModule>& modules) {
	if (!options.time_pipeline) return;
#if !defined(_WIN32)
	::setenv("MESA_SHADER_CACHE_DISABLE", "true", 1);
#endif
	HeadlessVulkan vulkan;
	std::string device_error;
	const bool have_device = vulkan.Initialize(device_error);
	std::ofstream output(options.output / "pipeline-timings.jsonl", std::ios::trunc);
	if (!output) throw std::runtime_error("could not create pipeline-timings.jsonl");
	const auto sample = QuantileSample(modules, 20);
	if (!have_device) {
		output << Json({{"error", device_error}, {"sample_count", sample.size()}}).dump() << '\n';
		std::printf("PIPELINE_TIMING unavailable error=%s\n", device_error.c_str());
		return;
	}

	std::printf("PIPELINE_TIMING columns=sample,raw_words,raw_module_ms,raw_pipeline_ms,"
	            "size_words,size_opt_ms,size_pipeline_ms,perf_words,perf_opt_ms,perf_pipeline_ms\n");
	for (size_t sample_index = 0; sample_index < sample.size(); sample_index++) {
		const auto& module = sample[sample_index];
		auto raw = ReadSpirv(AnalysisModulePath(options, module.result_index));
		double size_opt_ms = 0.0, perf_opt_ms = 0.0;
		std::string size_error, perf_error, raw_validation, size_validation, perf_validation;
		auto size = OptimizeSpirv(raw, true, size_opt_ms, size_error);
		auto perf = OptimizeSpirv(raw, false, perf_opt_ms, perf_error);
		const bool raw_valid  = ValidateSpirv(raw, raw_validation);
		const bool size_valid = !size.empty() && ValidateSpirv(size, size_validation);
		const bool perf_valid = !perf.empty() && ValidateSpirv(perf, perf_validation);
		const auto raw_time   = raw_valid ? vulkan.Measure(raw) : PipelineMeasurement {};
		const auto size_time  = size_valid ? vulkan.Measure(size) : PipelineMeasurement {};
		const auto perf_time  = perf_valid ? vulkan.Measure(perf) : PipelineMeasurement {};

		Json record {{"sample", sample_index + 1}, {"corpus_index", module.result_index}};
		auto add_recipe = [&](const char* name, size_t words, double optimize_ms, bool valid,
		                      const std::string& optimize_error, const std::string& validation_error,
		                      const PipelineMeasurement& timing) {
			record[name] = {{"words", words},
			                {"optimize_ms", optimize_ms},
			                {"valid", valid},
			                {"module_ms", timing.module_ms},
			                {"pipeline_ms", timing.pipeline_ms},
			                {"error", !optimize_error.empty() ? optimize_error
			                          : !validation_error.empty() ? validation_error
			                                                        : timing.error}};
		};
		add_recipe("raw", raw.size(), 0.0, raw_valid, {}, raw_validation, raw_time);
		add_recipe("size", size.size(), size_opt_ms, size_valid, size_error, size_validation,
		           size_time);
		add_recipe("performance", perf.size(), perf_opt_ms, perf_valid, perf_error,
		           perf_validation, perf_time);
		output << record.dump() << '\n';
		std::printf("PIPELINE_TIMING sample=%zu raw=%zu,%.3f,%.3f size=%zu,%.3f,%.3f "
		            "perf=%zu,%.3f,%.3f valid=%d/%d/%d\n",
		            sample_index + 1, raw.size(), raw_time.module_ms, raw_time.pipeline_ms,
		            size.size(), size_opt_ms, size_time.pipeline_ms, perf.size(), perf_opt_ms,
		            perf_time.pipeline_ms, raw_valid, size_valid, perf_valid);
	}
}

void WriteAnalysis(const Options& options, const std::vector<Json>& results) {
	if (!AnalysisRequested(options)) return;
	const auto modules = EmittedModules(results);
	WriteSpirvProfiles(options, results, modules);
	WritePipelineTimings(options, modules);
	std::error_code ignored;
	std::filesystem::remove_all(options.output / ".analysis-modules", ignored);
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

	auto slowest = results;
	std::ranges::sort(slowest, [](const Json& left, const Json& right) {
		return left.value("total_ms", 0.0) > right.value("total_ms", 0.0);
	});
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
	summary << "\nFully translatable compute shaders: " << compute_emitted << "\n";
}

int Run(const Options& options) {
	std::error_code error_code;
	std::filesystem::create_directories(options.output, error_code);
	if (error_code) {
		std::fprintf(stderr, "shader_corpus_tool: could not create output directory: %s\n",
		             error_code.message().c_str());
		return 1;
	}
	if (AnalysisRequested(options)) {
		std::filesystem::remove_all(options.output / ".analysis-modules", error_code);
		error_code.clear();
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
	WriteAnalysis(options, results);
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

int SelfTest() {
#if defined(_WIN32)
	return 0;
#else
	const auto root = std::filesystem::temp_directory_path() /
	                  ("kyty-shader-corpus-test-" + std::to_string(::getpid()));
	const auto input  = root / "input";
	const auto output = root / "output";
	std::filesystem::create_directories(input);
	std::ofstream index(input / "index.jsonl");
	constexpr std::array<uint32_t, 1> emitted {0xbf810000u};
	constexpr std::array<uint32_t, 2> rejected {0xb96af818u, 0xbf810000u};
	WriteFixture(input, index, "cs", "0x0000000000000001", emitted, true);
	WriteFixture(input, index, "cs", "0x0000000000000002", rejected, true);
	WriteFixture(input, index, "ps", "0x0000000000000003", emitted, false);
	index.close();
	Options options {.input = input,
	                 .output = output,
	                 .jobs = 2,
	                 .timeout_ms = 5000,
	                 .spirv_profile = 1};
	const int status = Run(options);
	std::vector<Json> results;
	std::string error;
	const bool read = ReadJsonLines(output / "results.jsonl", results, error);
	bool have_emitted = false, have_rejected = false, have_runtime = false;
	for (const auto& result: results) {
		const auto outcome = result.value("outcome", "");
		have_emitted |= outcome == "emitted";
		have_rejected |= outcome.starts_with("rejected:");
		have_runtime |= outcome.starts_with("needs-runtime:");
	}
	if (std::ifstream summary(output / "summary.md"); summary) {
		std::string line;
		while (std::getline(summary, line)) std::printf("%s\n", line.c_str());
	}
	std::error_code ignored;
	std::filesystem::remove_all(root, ignored);
	const bool have_profile = std::filesystem::exists(output / "spirv-profile.jsonl");
	if (status != 0 || !read || results.size() != 3 || !have_emitted || !have_rejected ||
	    !have_runtime || !have_profile) {
		std::fprintf(stderr,
		             "shader_corpus_tool self-test failed: count=%zu emitted=%d rejected=%d "
		             "runtime=%d profile=%d %s\n",
		             results.size(), have_emitted, have_rejected, have_runtime, have_profile,
		             error.c_str());
		return 1;
	}
	return 0;
#endif
}

void Usage() {
	std::printf("shader_corpus_tool --dir <dump dir> --out <report dir> [--jobs N] "
	            "[--only <stage>] [--limit N] [--timeout-ms N] [--emit-spirv] "
	            "[--spirv-profile N] [--time-pipeline]\n");
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
		if (argument == "--time-pipeline") {
			options.time_pipeline = true;
			continue;
		}
		if (index + 1 >= argc) return {};
		const char* value = argv[++index];
		if (argument == "--dir") options.input = value;
		else if (argument == "--out") options.output = value;
		else if (argument == "--only") options.only = value;
		else if (argument == "--jobs") {
			if (!ParseUnsigned(value, options.jobs) || options.jobs == 0) return {};
		} else if (argument == "--limit") {
			if (!ParseUnsigned(value, options.limit)) return {};
		} else if (argument == "--timeout-ms") {
			if (!ParseUnsigned(value, options.timeout_ms) || options.timeout_ms == 0) return {};
		} else if (argument == "--spirv-profile") {
			if (!ParseUnsigned(value, options.spirv_profile) || options.spirv_profile == 0) return {};
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
