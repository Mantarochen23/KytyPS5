#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREIDENTITY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREIDENTITY_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>

namespace Libs::Graphics {

// The SPIR-V profile fields that identify a shader store, in the order the store header has always
// written them. A debug option that changes the emitted SPIR-V (--loop-guard) is deliberately NOT
// part of it: adding it to the header made every boot with a different option value reject the
// whole store and rewrite it (0 programs loaded). Such an option selects its own store file
// instead (ShaderStoreSuffix), so each variant keeps and reloads its own entries.
template <class Profile, class Visit> void VisitShaderStoreProfile(const Profile& profile, Visit&& visit) {
	visit(profile.maintenance8);
	visit(profile.native_float_min_max);
	visit(profile.device_clock_period_ps);
	visit(profile.robust_buffer_access);
	visit(profile.r8_srgb_views);
	visit(profile.r8g8_srgb_views);
	visit(profile.relaxed_float);
	visit(profile.ray_query);
}

// File-name part that separates stores whose SPIR-V differs by a debug option. Empty = default.
// A limit for every stage selects ".guard<n>"; only the default compute-shader limit ".guardc<n>".
[[nodiscard]] inline std::string ShaderStoreSuffix(uint32_t loop_guard_limit,
                                                   uint32_t compute_loop_guard_limit = 0) {
	if (loop_guard_limit != 0) return ".guard" + std::to_string(loop_guard_limit);
	return compute_loop_guard_limit == 0 ? std::string {}
	                                     : ".guardc" + std::to_string(compute_loop_guard_limit);
}

// The store file name: title, the mask of ports that affect generated code (empty when 0), the debug
// suffix. Ports that only widen translation are not in it (see ports.h affects_codegen).
[[nodiscard]] inline std::string ShaderStoreFileName(std::string_view title_id, uint32_t codegen_ports,
                                                     std::string_view suffix) {
	char hex[16];
	std::snprintf(hex, sizeof(hex), "%x", codegen_ports);
	return std::string(title_id) + (codegen_ports != 0 ? ".ports" + std::string(hex) : std::string {}) +
	       std::string(suffix) + ".shaders";
}

struct ShaderStoreName {
	std::string title;
	uint32_t    ports = 0;
	std::string suffix; // ".guard<n>" or empty
};

// Splits "<title>[.ports<hex>][.guard<n>].shaders" (the inverse of ShaderStoreFileName); false for any
// other file name. A title id has no dots.
[[nodiscard]] inline bool ParseShaderStoreFileName(std::string_view name, ShaderStoreName& out) {
	constexpr std::string_view tail = ".shaders";
	if (name.size() <= tail.size() || name.substr(name.size() - tail.size()) != tail) {
		return false;
	}
	name.remove_suffix(tail.size());
	const auto dot = name.find('.');
	out            = {};
	out.title      = std::string(name.substr(0, dot));
	if (out.title.empty()) return false;
	auto rest = dot == std::string_view::npos ? std::string_view {} : name.substr(dot);
	if (rest.substr(0, 6) == ".ports") {
		rest.remove_prefix(6);
		size_t digits = 0;
		while (digits < rest.size() && std::isxdigit(static_cast<unsigned char>(rest[digits])) != 0) {
			const char c = rest[digits];
			out.ports    = out.ports * 16u + static_cast<uint32_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
			digits++;
		}
		if (digits == 0) return false;
		rest.remove_prefix(digits);
	}
	if (!rest.empty() && rest.substr(0, 6) != ".guard") return false;
	out.suffix = std::string(rest);
	return true;
}

// Number of whole record frames after `header_size` bytes (frame header: u32 compressed size,
// u32 raw size, u64 checksum; the checksum is not verified here). For the open-time log line.
[[nodiscard]] inline size_t CountShaderStoreRecords(std::span<const uint8_t> file, size_t header_size) {
	size_t count  = 0;
	size_t offset = header_size;
	while (offset + 16u <= file.size()) {
		uint32_t compressed = 0;
		std::memcpy(&compressed, file.data() + offset, sizeof(compressed));
		if (file.size() - offset - 16u < compressed) break;
		offset += 16u + compressed;
		count++;
	}
	return count;
}

// The same for a file on disk, reading only the 16-byte frame headers (seeking over the records).
[[nodiscard]] inline size_t CountShaderStoreRecordsInFile(const std::filesystem::path& path,
                                                          size_t header_size) {
	std::ifstream file(path, std::ios::binary);
	if (!file) return 0;
	file.seekg(0, std::ios::end);
	const auto size = static_cast<uint64_t>(file.tellg());
	size_t     count  = 0;
	uint64_t   offset = header_size;
	while (offset + 16u <= size) {
		uint32_t compressed = 0;
		file.seekg(static_cast<std::streamoff>(offset));
		file.read(reinterpret_cast<char*>(&compressed), sizeof(compressed));
		if (!file || size - offset - 16u < compressed) break;
		offset += 16u + compressed;
		count++;
	}
	return count;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREIDENTITY_H_
