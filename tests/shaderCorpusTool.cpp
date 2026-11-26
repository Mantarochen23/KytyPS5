#include "common/emulatorConfig.h"
#include "common/recoverable.h"
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
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
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

void TranslateOne(const CorpusEntry& entry, const Options& options,
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
			if (options.emit_spirv) {
				const auto folder = options.output / "spirv";
				std::filesystem::create_directories(folder);
				std::ofstream spirv(folder / (entry.stage + "_" + entry.hash.substr(2) + ".spv"),
				                    std::ios::binary | std::ios::trunc);
				spirv.write(reinterpret_cast<const char*>(compiled.spirv.data()),
				            static_cast<std::streamsize>(compiled.spirv.size() * sizeof(uint32_t)));
				if (!spirv) throw std::runtime_error("could not write SPIR-V output");
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
	Options options {.input = input, .output = output, .jobs = 2, .timeout_ms = 5000};
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
	std::error_code ignored;
	std::filesystem::remove_all(root, ignored);
	if (status != 0 || !read || results.size() != 3 || !have_emitted || !have_rejected ||
	    !have_runtime) {
		std::fprintf(stderr,
		             "shader_corpus_tool self-test failed: count=%zu emitted=%d rejected=%d runtime=%d %s\n",
		             results.size(), have_emitted, have_rejected, have_runtime, error.c_str());
		return 1;
	}
	return 0;
#endif
}

void Usage() {
	std::printf("shader_corpus_tool --dir <dump dir> --out <report dir> [--jobs N] "
	            "[--only <stage>] [--limit N] [--timeout-ms N] [--emit-spirv]\n");
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
