#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREFILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREFILE_H_

#include <cstdint>
#include <span>
#include <vector>

// The framing of the shader store records: a frame header and a zstd frame per record. Kept apart
// from the cache so that writing and reloading can be tested without a Vulkan device.
namespace Libs::Graphics::ShaderStoreFile {

// Precedes each record's zstd frame.
struct Frame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	uint64_t checksum        = 0; // XXH3 of the compressed bytes.
};

// The frame header and the compressed `record`, ready to append.
[[nodiscard]] std::vector<uint8_t> Encode(std::span<const uint8_t> record);

enum class Status : uint8_t { Loaded, EndOfStream, Failed };

// Reads the record at the front of `records` into `raw`; `frame_bytes` is the size of its frame.
// With `skip_failed` `records` is stepped over the frame before it is checked, so a damaged record
// is skipped; otherwise the caller steps over it (records.subspan(frame_bytes)) once it has
// accepted the content, and a record it does not accept stays at the front.
[[nodiscard]] Status Decode(std::span<const uint8_t>& records, bool skip_failed,
                            std::vector<uint8_t>& raw, size_t& frame_bytes);

} // namespace Libs::Graphics::ShaderStoreFile

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREFILE_H_
