#include "graphics/host_gpu/renderer/pipeline/shaderStoreFile.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics::ShaderStoreFile {

std::vector<uint8_t> Encode(std::span<const uint8_t> record) {
	std::vector<uint8_t> bytes(sizeof(Frame) + ZSTD_compressBound(record.size()));
	const auto size = ZSTD_compress(bytes.data() + sizeof(Frame), bytes.size() - sizeof(Frame),
	                                record.data(), record.size(), 3);
	if (ZSTD_isError(size)) {
		std::fprintf(stderr, "shader store: compression failed\n");
		std::abort();
	}
	const Frame frame {.compressed_size = static_cast<uint32_t>(size),
	                   .raw_size        = static_cast<uint32_t>(record.size()),
	                   .checksum        = XXH3_64bits(bytes.data() + sizeof(Frame), size)};
	std::memcpy(bytes.data(), &frame, sizeof(frame));
	bytes.resize(sizeof(Frame) + size);
	return bytes;
}

Status Decode(std::span<const uint8_t>& records, bool skip_failed, std::vector<uint8_t>& raw,
              size_t& frame_bytes) {
	Frame frame;
	if (records.size() < sizeof(frame)) {
		return Status::EndOfStream;
	}
	std::memcpy(&frame, records.data(), sizeof(frame));
	if (records.size() - sizeof(frame) < frame.compressed_size) {
		return Status::EndOfStream;
	}
	const auto compressed = records.subspan(sizeof(frame), frame.compressed_size);
	frame_bytes           = sizeof(frame) + compressed.size();
	if (skip_failed) {
		records = records.subspan(frame_bytes);
	}
	if (XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return Status::Failed;
	}
	raw.resize(frame.raw_size);
	if (ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) != raw.size()) {
		return Status::Failed;
	}
	return Status::Loaded;
}

} // namespace Libs::Graphics::ShaderStoreFile
