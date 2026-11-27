#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/smallVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/resolveStamp.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <memory>
#include <map>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class ResolveCache;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
		bool          heap = false; // PS5: resolved for a resident image heap.
		// PS5: set by FindImage.
		ResolveStamp stamp;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache, ResolveCache& resolve_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(const RangeSet& ranges);
	// A content change can alter image discovery even when no registered image overlaps it.
	void TouchResolutionGeneration(uint64_t address, uint64_t size);
	// Writes a copy into whole levels of one layer of a GPU-written image straight into it, and
	// returns false for any other copy.
	[[nodiscard]] bool CopyToImage(uint64_t address, uint64_t source, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// With storage readback, a GL2 writeback makes GPU-written images visible to the CPU; returns
	// whether a download was recorded.
	[[nodiscard]] bool WriteBackImages();
	// The tick of the latest write-back pending over the range's pages, or 0 when none is.
	[[nodiscard]] uint64_t WriteBackTick(uint64_t address, uint64_t size);
	void RunGarbageCollector();

private:
	struct MetaDataInfo {
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
		// The depth image that last acquired this HTile range.
		ImageId owner;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	// Fields which decide an exact SameBacking or stencil-plane match.
	struct ExactImageKey {
		GuestRange             data;
		vk::Extent3D           extent;
		ImageSubresources      resources;
		uint32_t               samples         = 1;
		uint32_t               bytes_per_block = 0;
		Prospero::TileMode     tile_mode       = Prospero::TileMode::kLinear;
		ColorTransform         transform       = ColorTransform::None;
		vk::Format             pixel_format    = vk::Format::eUndefined;
		Prospero::BufferFormat guest_format    = Prospero::BufferFormat::kInvalid;
		Prospero::ImageType    type            = Prospero::ImageType::kColor2D;
		BindingType            binding         = BindingType::Texture;
		bool                   exact_format    = false;

		bool operator==(const ExactImageKey&) const = default;
	};
	struct ExactImageKeyHash {
		[[nodiscard]] size_t operator()(const ExactImageKey& key) const noexcept;
	};
	struct ExactImage {
		ImageId  id;
		uint64_t generation = 0;
		bool     sole        = false;
	};

	// A recorded image download whose guest pages stay inaccessible until it lands.
	struct WriteBack {
		GuestRange range;
		uint64_t   tick;
	};

	// PS5: guest addresses span 44 bits, so an entry keeps the whole image address.
	struct BucketEntry {
		uint64_t address;
		uint32_t size;
		ImageId  id;

		[[nodiscard]] bool Overlaps(uint64_t addr, size_t size) const noexcept {
			return address < addr + size && addr < address + this->size;
		}
	};

	struct alignas(64) Bucket {
		Common::SmallVector<BucketEntry, 3, uint32_t> entries;
	};
	static_assert(sizeof(Bucket) == 64);

	using ImagePageTable = MultiLevelPageTable<Bucket, 20, 44, 14>;
	using ResolutionPageTable =
	    MultiLevelPageTable<uint64_t, ImagePageTable::kPageBits,
	                        ImagePageTable::kAddressSpaceBits, ImagePageTable::kFirstLevelBits>;

	// Registered images per page of the image page table. Written under m_lock, read without it,
	// so that a write fault on a page no image covers (most of them) does not wait for the lock.
	// Ported from upstream PR 1284 (NeySlim).
	class RegisteredPageCounts final {
	public:
		RegisteredPageCounts(): m_leaves(std::make_unique<std::atomic<Leaf*>[]>(kLeafCount)) {}
		~RegisteredPageCounts() {
			for (size_t i = 0; i < kLeafCount; i++) {
				delete m_leaves[i].load(std::memory_order_relaxed);
			}
		}
		KYTY_CLASS_NO_COPY(RegisteredPageCounts);

		void Add(size_t page, int32_t delta) {
			auto& slot = m_leaves[page >> kLeafBits];
			auto* leaf = slot.load(std::memory_order_relaxed);
			if (leaf == nullptr) {
				leaf = new Leaf {};
				slot.store(leaf, std::memory_order_release);
			}
			(*leaf)[page & (kLeafSize - 1)].fetch_add(static_cast<uint32_t>(delta),
			                                          std::memory_order_release);
		}
		[[nodiscard]] bool Any(size_t first, size_t last_exclusive) const noexcept {
			for (auto page = first; page < last_exclusive; page++) {
				const auto* leaf = m_leaves[page >> kLeafBits].load(std::memory_order_acquire);
				if (leaf == nullptr) {
					// Skip to the next leaf.
					page = ((page >> kLeafBits) + 1) * kLeafSize - 1;
					continue;
				}
				if ((*leaf)[page & (kLeafSize - 1)].load(std::memory_order_acquire) != 0) {
					return true;
				}
			}
			return false;
		}

	private:
		static constexpr size_t kLeafBits  = 10;
		static constexpr size_t kLeafSize  = size_t {1} << kLeafBits;
		static constexpr size_t kLeafCount = ImagePageTable::kPageCount / kLeafSize;
		using Leaf                         = std::array<std::atomic<uint32_t>, kLeafSize>;

		std::unique_ptr<std::atomic<Leaf*>[]> m_leaves;
	};

	// Callers pass a nonempty, valid guest range.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		static constexpr bool RETURNS_BOOL =
		    std::is_same_v<std::invoke_result_t<Func, uint64_t>, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	// The walks below iterate the page entries in place: func must not insert, register,
	// unregister or free images. Caller holds m_lock.

	// Calls func once per image overlapping the region, from the first page both share; a func
	// returning true stops the walk.
	template <typename Func>
	void ForEachImageInRegion(uint64_t address, size_t size, Func&& func) {
		static constexpr bool BOOL_BREAK =
		    std::is_same_v<std::invoke_result_t<Func, ImageId, Image&>, bool>;
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
			return;
		}
		const uint64_t first_page = address >> ImagePageTable::kPageBits;
		ForEachPage(address, size, [&](uint64_t page) {
			const auto* bucket = m_image_page_table.Find(page);
			if (bucket == nullptr) {
				if constexpr (BOOL_BREAK) {
					return false;
				} else {
					return;
				}
			}
			for (const auto& entry: bucket->entries) {
				const uint64_t base_page = entry.address >> ImagePageTable::kPageBits;
				if (page != std::max(first_page, base_page)) {
					continue;
				}
				if (!entry.Overlaps(address, size)) {
					continue;
				}
				Image& image = m_slot_images[entry.id];
				if constexpr (BOOL_BREAK) {
					if (func(entry.id, image)) {
						return true;
					}
				} else {
					func(entry.id, image);
				}
			}
			if constexpr (BOOL_BREAK) {
				return false;
			}
		});
	}

	// Calls func for each image registered at address.
	template <typename Func>
	void ForEachImageWithAddress(uint64_t address, Func&& func) {
		const auto* bucket = m_image_page_table.Find(address >> ImagePageTable::kPageBits);
		if (bucket == nullptr) {
			return;
		}
		for (const auto& entry: bucket->entries) {
			if (entry.address == address) {
				func(entry.id, m_slot_images[entry.id]);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TouchResolutionGenerationLocked(
	    uint64_t address, uint64_t size,
	    Common::FrameCost::Counter site = Common::FrameCost::Counter::CacheBumpOther);
	void                      TouchResolutionGenerationLocked(
	    const RangeSet& ranges,
	    Common::FrameCost::Counter site = Common::FrameCost::Counter::CacheBumpOther);
	// Returns the ranges whose discovery state changed after `generation`, together with one
	// coherent generation that covers the returned records. ImageHeap is the sole consumer.
	// `incomplete` is set when records after `generation` were dropped to bound the journal.
	[[nodiscard]] uint64_t    TakeResolutionChangesSince(uint64_t generation,
	                                                     std::vector<GuestRange>& changes,
	                                                     bool* incomplete = nullptr);
	void                      TouchRegistrationsLocked(uint64_t address, uint64_t size);
	[[nodiscard]] bool        RegistrationsSinceLocked(uint64_t address, uint64_t size,
	                                                   uint64_t generation) const;
	[[nodiscard]] uint64_t    ResolutionGeneration() const noexcept {
		return m_resolution_generation.load(std::memory_order_relaxed);
	}
	// Reports whether either range changed after generation and returns one coherent current
	// generation for an entry that can now be skipped.
	[[nodiscard]] bool ResolutionChangedSince(GuestRange data, GuestRange metadata,
	                                          uint64_t generation, uint64_t& current);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged, bool exact_format);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	// Caller holds m_lock and ends rendering. Begins rendering `layers` layers of one color mip
	// through a `format` view, clearing them or, when `clear` is null, loading them; returns the
	// render area.
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               vk::CommandBuffer command);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	// The depth image's stencil plane record covering exactly range, or none.
	[[nodiscard]] ImageId FindStencilPlane(const Image& depth, GuestRange range) const;
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	[[nodiscard]] bool DownloadQueuedImages();

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	ResolveCache&                                     m_resolve_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	RegisteredPageCounts                              m_registered_pages;
	ResolutionPageTable                               m_resolution_page_table;
	struct ResolutionChange {
		GuestRange range;
		uint64_t   generation = 0;
	};
	static constexpr size_t                           MaxResolutionChanges = 1u << 16;
	std::vector<ResolutionChange>                     m_resolution_changes;
	uint64_t                                          m_resolution_journal_floor = 0;
	std::atomic<uint64_t>                             m_resolution_generation {1};
	std::unordered_map<ExactImageKey, ExactImage, ExactImageKeyHash> m_exact_images;
	ResolutionPageTable                               m_registration_page_table;
	uint64_t                                          m_registration_generation = 1;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::mutex                                        m_write_back_mutex;
	std::vector<WriteBack>                            m_write_backs;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t m_critical_gc_memory      = 3ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick                 = 0;
	bool     m_readback_linear_images  = false;
	bool     m_readback_tiled_images   = false;
	bool     m_readback_storage_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend class ResolveCache;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
