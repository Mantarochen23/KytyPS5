#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/frameCost.h"
#include "common/ports.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <iterator>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {

namespace IR = ShaderRecompiler::IR;

// Smaller tables leave after one idle frame, larger ones after LargeTableIdleFrames.
constexpr uint64_t MinTableBytes        = 64ull * 1024ull;
constexpr uint32_t LargeTableIdleFrames = 32;
constexpr uint32_t ElementMask          = (1u << IR::ImageHeapKindShift) - 1u;

template <typename Pred, typename Fn>
void ForEachRun(size_t begin, size_t end, Pred&& pred, Fn&& fn) {
	while (begin < end) {
		if (!pred(begin)) {
			++begin;
			continue;
		}
		auto run = begin + 1u;
		while (run < end && pred(run)) ++run;
		fn(begin, run);
		begin = run;
	}
}

// Grows [begin, end) to cover [first, last).
void Widen(size_t& begin, size_t& end, size_t first, size_t last) {
	const bool empty = begin == end;
	begin            = empty ? first : std::min(begin, first);
	end              = empty ? last : std::max(end, last);
}

// Probes at first + k * step whose first dword lies within a table of size bytes.
uint64_t Probes(uint64_t size, uint64_t first, uint64_t step) {
	return size >= first + 4u ? (size - first - 4u) / step + 1u : 0u;
}

uint64_t Hash(const IR::DescriptorValue& value) {
	return XXH3_64bits(value.dwords.data(), sizeof(value.dwords));
}

// Tables grow a little on most frames, so their slot words get room to double.
std::unique_ptr<Buffer> SlotBuffer(RenderContext& context, uint64_t size) {
	return std::make_unique<Buffer>(
	    context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
	        vk::BufferUsageFlagBits::eShaderDeviceAddress,
	    std::bit_ceil(size + 4u));
}

// Guest bytes of a table that its tracker pages [begin, end) cover.
std::pair<uint64_t, uint64_t> PageRange(uint64_t base, uint64_t size, size_t begin, size_t end) {
	const auto origin = Common::AlignDown(base, TRACKER_PAGE_SIZE);
	return {std::max(base, origin + begin * TRACKER_PAGE_SIZE),
	        std::min(base + size, origin + end * TRACKER_PAGE_SIZE)};
}

} // namespace

size_t ImageHeap::KeyHash::operator()(const Key& key) const noexcept {
	const auto binding = (static_cast<uint32_t>(key.dimension) << 2u) | (uint32_t(key.r128) << 1u) | uint32_t(key.depth_view);
	return XXH3_64bits_withSeed(key.value.dwords.data(), sizeof(key.value.dwords), binding);
}

ImageHeap::ImageHeap(RenderContext& context, PageManager& page_manager)
    : m_context(context), m_page_manager(page_manager) {
	auto& graphics = context.GetGraphics();
	vk::PhysicalDeviceVulkan12Properties limits {};
	vk::PhysicalDeviceProperties2        properties {};
	properties.pNext = &limits;
	graphics.physical_device.getProperties2(&properties);
	EXIT_NOT_IMPLEMENTED(std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages,
	                               limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
	                               limits.maxPerStageUpdateAfterBindResources}) <
	                     IR::ImageHeapCapacity);

	const vk::DescriptorBindingFlags binding_flags =
	    vk::DescriptorBindingFlagBits::ePartiallyBound |
	    vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	    vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = 1;
	flags_info.pBindingFlags = &binding_flags;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eSampledImage,
	                                              IR::ImageHeapCapacity,
	                                              vk::ShaderStageFlagBits::eAll, nullptr};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout) !=
	        vk::Result::eSuccess);
	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eSampledImage,
	                                        IR::ImageHeapCapacity};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	EXIT_IF(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool) !=
	        vk::Result::eSuccess);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &m_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&allocate, &m_set) != vk::Result::eSuccess);
	graphics.image_heap_layout = m_layout;
	Image::heap_layout_quiet.store(Config::PortEnabled(Config::Port::ImageHeapIncremental),
	                               std::memory_order_relaxed);
	// Slot word 0 selects element 0, which stays a null image.
	m_elements.emplace_back();
	m_writes.push_back({0u, {nullptr, nullptr, vk::ImageLayout::eShaderReadOnlyOptimal}});
	WriteDescriptors();
}

ImageHeap::~ImageHeap() {
	for (const auto& memory: m_memories) Unwatch(*memory);
	auto& graphics             = m_context.GetGraphics();
	graphics.image_heap_layout = nullptr;
	Image::heap_layout_quiet.store(false, std::memory_order_relaxed);
	graphics.device.destroyDescriptorPool(m_pool, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
}

bool ImageHeap::Acquire(const IR::ImageHeapRequest& request, IR::ImageHeapView& result) {
	KYTY_PROFILER_FUNCTION();
	const auto size = request.size & ~uint64_t {3};
	if (request.count == 0u || request.step == 0u ||
	    ((request.base | request.first | request.step) & 3u) != 0u) {
		return false;
	}
	auto memory = std::ranges::find_if(
	    m_memories, [&](const auto& candidate) { return candidate->base == request.base; });
	// A table follows its descriptor's size, as the first program of a draw acquires it.
	if (memory == m_memories.end() || ((*memory)->size != size && (*memory)->draw != m_draw)) {
		if (!m_context.IsMapped(request.base, size)) return false;
		if (memory == m_memories.end()) {
			// The guest republishes tables at new bases; a copy that idled a frame moves there,
			// unless it buried words, which a new table resolves.
			memory = std::ranges::find_if(m_memories, [&](const auto& candidate) {
				return candidate->size == size && candidate->idle != 0u &&
				       candidate->draw <= m_suspend_draw && candidate->dead.empty();
			});
		}
		if (memory == m_memories.end()) {
			std::scoped_lock lock {m_mutex};
			memory = m_memories.insert(m_memories.end(), std::make_unique<Memory>());
		}
		Resize(**memory, request.base, size);
	}
	// A draw refreshes a table and a view takes its changes once, so all programs of the draw see
	// one snapshot and one set of kinds.
	if ((*memory)->draw != m_draw) {
		if (!Refresh(**memory)) return false;
		(*memory)->draw = m_draw;
	}
	const auto root   = IR::ImageHeapResource(*request.root, 0u);
	auto&      views  = (*memory)->views;
	const auto bytes  = (*memory)->size;
	const auto probes = Probes(size, request.first, request.step);
	const auto limit  = request.count < probes ? request.count : UINT64_MAX;
	// Slots are indexed by table dword, so unbounded requests share the view of their lattice.
	const auto first = limit == UINT64_MAX ? request.first % request.step : request.first;
	auto       view  = std::ranges::find_if(views, [&](const auto& candidate) {
		return candidate->step == request.step && candidate->first == first &&
		       candidate->limit == limit && candidate->root == root;
	});
	if (view == views.end()) {
		const auto count   = std::min(limit, Probes(bytes, first, request.step));
		auto       created = std::make_unique<View>(View {.memory  = **memory,
		                                                  .step    = request.step,
		                                                  .first   = first,
		                                                  .count   = count,
		                                                  .limit   = limit,
		                                                  .root    = root});
		created->id = m_next_view++;
		m_views[created->id] = created.get();
		created->slots.resize(bytes / 4u + 1u);
		created->buffer = SlotBuffer(m_context, bytes);
		UpdateProbes(*created, 0u, bytes);
		// The first upload also clears the slots no probe selects.
		created->dirty_begin = 0;
		created->dirty_end   = created->slots.size();
		view = views.insert(views.end(), std::move(created));
	} else {
		// Empty after the first acquire in a draw, since the table refreshes once per draw.
		UpdateProbes(**view, (*view)->changed_begin, (*view)->changed_end);
		(*view)->changed_begin = (*view)->changed_end = 0;
	}
	if (std::ranges::find(m_active, view->get()) == m_active.end()) m_active.push_back(view->get());
	const auto& selected = **view;
	result = {.slots      = selected.buffer->BufferDeviceAddress(),
	          .slot_count = static_cast<uint32_t>(selected.slots.size() - 1u),
	          .kinds      = selected.kinds};
	return true;
}

bool ImageHeap::Refresh(Memory& memory) {
	// GPU writes leave no fault; reading the table downloads them.
	const bool gpu_dirty = m_context.GetBufferCache().HasGpuDirtyBytes(memory.base, memory.size);
	const auto pages     = memory.dirty_pages.size();
	m_runs.clear();
	{
		std::scoped_lock lock {m_mutex};
		// Watch before reading, so a write during the read faults the page again.
		ForEachRun(
		    0, pages, [&](size_t page) { return memory.dirty_pages[page] != 0u; },
		    [&](size_t begin, size_t end) {
			    std::fill(memory.dirty_pages.begin() + begin, memory.dirty_pages.begin() + end,
			              uint8_t {0});
			    Watch<true>(memory, begin, end);
			    m_runs.emplace_back(begin, end);
		    });
	}
	if (gpu_dirty) m_runs.assign(1u, {0u, pages});
	for (size_t run = 0; run < m_runs.size(); ++run) {
		const auto [address, end] = PageRange(memory.base, memory.size, m_runs[run].first,
		                                      m_runs[run].second);
		m_scratch.resize((end - address) / 4u);
		if (!LibKernel::Memory::TryReadBufferBacking(address, m_scratch.data(), end - address)) {
			std::scoped_lock lock {m_mutex};
			for (; run < m_runs.size(); ++run) {
				MarkDirty(memory, m_runs[run].first, m_runs[run].second);
			}
			return false;
		}
		const auto words = std::span(memory.words).subspan((address - memory.base) / 4u,
		                                                   m_scratch.size());
		size_t     first = 0;
		size_t     last  = words.size();
		while (first < last && words[first] == m_scratch[first]) ++first;
		while (first < last && words[last - 1u] == m_scratch[last - 1u]) --last;
		if (first == last) continue;
		std::copy(m_scratch.begin() + first, m_scratch.begin() + last, words.begin() + first);
		const auto offset = address - memory.base;
		for (auto& view: memory.views) {
			Widen(view->changed_begin, view->changed_end, offset + first * 4u, offset + last * 4u);
		}
	}
	return true;
}

void ImageHeap::MarkDirty(Memory& memory, size_t begin, size_t end) {
	ForEachRun(
	    begin, end, [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t first, size_t last) {
		    std::fill(memory.dirty_pages.begin() + first, memory.dirty_pages.begin() + last,
		              uint8_t {1});
		    Watch<false>(memory, first, last);
	    });
}

template <bool track>
void ImageHeap::Watch(const Memory& memory, size_t begin, size_t end) {
	const auto [address, limit] = PageRange(memory.base, memory.size, begin, end);
	m_page_manager.UpdatePageWatchers<track>(address, limit - address);
}

void ImageHeap::Unwatch(const Memory& memory) {
	ForEachRun(
	    0, memory.dirty_pages.size(), [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t begin, size_t end) { Watch<false>(memory, begin, end); });
}

void ImageHeap::InvalidateMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_mutex};
	for (auto& memory: m_memories) {
		// Watches cover whole pages, so a write next to the table must unwatch its page too.
		const auto origin = Common::AlignDown(memory->base, TRACKER_PAGE_SIZE);
		const auto begin  = std::max(address, origin);
		const auto end    = std::min(address + size,
		                             origin + memory->dirty_pages.size() * TRACKER_PAGE_SIZE);
		if (begin >= end) continue;
		MarkDirty(*memory, (begin - origin) / TRACKER_PAGE_SIZE,
		          (end - 1u - origin) / TRACKER_PAGE_SIZE + 1u);
	}
}

void ImageHeap::UpdateProbes(View& view, uint64_t begin, uint64_t end) {
	// A probe reads its descriptor from first + probe * step.
	const uint64_t bytes = view.root.r128 ? 16u : 32u;
	auto probe = begin < view.first + bytes ? 0u : (begin - view.first - bytes) / view.step + 1u;
	for (; probe < view.count && view.first + probe * view.step < end; ++probe) {
		UpdateProbe(view, probe);
	}
}

void ImageHeap::UpdateProbe(View& view, uint64_t probe) {
	const auto& words   = view.memory.words;
	const auto  offset  = static_cast<uint32_t>((view.first + probe * view.step) / 4u);
	Key         key {.value = {.dword_count = 8u}, .dimension = view.root.dimension, .r128 = view.root.r128};
	std::copy_n(words.begin() + offset,
	            std::min<size_t>(key.r128 ? 4u : 8u, words.size() - offset),
	            key.value.dwords.begin());
	// A buried word stays null in every view until its descriptor changes.
	auto& dead = view.memory.dead;
	if (const auto found = dead.find(offset); found != dead.end()) {
		if (found->second == Hash(key.value)) {
			Assign(view, offset, 0u);
			return;
		}
		dead.erase(found);
	}
	const auto kind = IR::ImageHeapKind(view.root, key.value);
	if (kind == 0u) {
		Assign(view, offset, 0u);
		return;
	}
	const auto format = DecodeNativeDescriptor<ShaderTextureResource>(key.value).Format();
	key.depth_view = SampledDepthCompareFormat(format, view.root.depth_compare).has_value();
	Assign(view, offset, Intern(view, kind, key));
}

void ImageHeap::Assign(View& view, size_t slot, uint32_t entry) {
	const auto previous = std::exchange(view.slots[slot], entry);
	Release(previous);
	if (previous != entry) {
		if ((entry & ElementMask) != 0u) {
			AddSlotRef(entry & ElementMask, view.id, static_cast<uint32_t>(slot));
		}
		Widen(view.dirty_begin, view.dirty_end, slot, slot + 1u);
		++m_descriptor_generation;
	}
}

uint32_t ImageHeap::Intern(View& view, uint32_t kind, const Key& key) {
	const auto arm = static_cast<uint32_t>(std::ranges::find(view.kind_keys, kind) -
	                                       view.kind_keys.begin());
	if (arm == view.kind_keys.size()) {
		view.kind_keys.push_back(kind);
		view.kinds.push_back(key.value);
	}
	auto [found, inserted] = m_index.try_emplace(key, 0u);
	if (inserted) {
		found->second = Allocate();
		auto& element = m_elements[found->second];
		element.key   = key;
		element.kind  = kind;
		m_pending.push_back(found->second);
	}
	++m_elements[found->second].refs;
	return arm << IR::ImageHeapKindShift | found->second;
}

void ImageHeap::Release(uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u || --m_elements[index].refs != 0u) return;
	auto& element = m_elements[index];
	Mark(index, State::Stale);
	auto& textures = m_context.GetTextureCache();
	if (auto* image = textures.m_slot_images.try_get(element.image);
	    image != nullptr && image->serial == element.serial) {
		image->heap_refs.fetch_sub(1u, std::memory_order_relaxed);
		image->tick_accessed_last = textures.m_gc_tick;
	}
	m_released.push_back(index);
}

// A released element leaves once the tick of the upload that unselects it ends, so no list may
// keep its index.
void ImageHeap::Retire() {
	if (m_released.empty()) return;
	const auto released = [&](uint32_t index) { return m_elements[index].refs == 0u; };
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	for (const auto index: m_released) {
		if (released(index)) SetRetry(index, RetryKind::None);
		m_retired.emplace_back(tick, index);
	}
	m_released.clear();
}

// Unlists the element, so its descriptor interns a new one.
void ImageHeap::Mark(uint32_t index, State state) {
	auto& element = m_elements[index];
	element.state = state;
	if (element.refs != 0u) m_marked.push_back(index);
	SetRetry(index, RetryKind::None);
	if (const auto found = m_index.find(element.key);
	    found != m_index.end() && found->second == index) {
		m_index.erase(found);
	}
}

void ImageHeap::SetRetry(uint32_t index, RetryKind retry) {
	auto& element = m_elements[index];
	if (element.retry == retry) return;
	element.retry = retry;
	element.retry_stamp = ++m_retry_stamp;
	if (retry == RetryKind::None) {
		m_retry_elements.erase(index);
		return;
	}
	m_retry_elements.insert(index);
	auto& textures = m_context.GetTextureCache();
	const auto index_range = [&](GuestRange range) {
		if (range.Empty()) return;
		textures.ForEachPage(range.address, range.size, [&](uint64_t page) {
			m_resolution_elements[page].push_back({index, element.retry_stamp});
			++m_resolution_refs;
		});
	};
	index_range(element.data);
	index_range(element.metadata);
}

// Calls fn with each selected element of an image in m_changes, sorted by serial; true if any.
template <typename Fn>
bool ImageHeap::ForEachChanged(Fn&& fn, uint64_t* visited) {
	bool found = false;
	for (const auto& change: m_changes) {
		const auto entries = m_image_elements.find(change.serial);
		if (entries == m_image_elements.end()) continue;
		for (const auto index: entries->second) {
			if (visited != nullptr) ++*visited;
			const auto& element = m_elements[index];
			if (element.refs != 0u && element.state == State::Live && element.image &&
			    element.serial == change.serial) {
				fn(index);
				found = true;
			}
		}
	}
	return found;
}

void ImageHeap::AddSlotRef(uint32_t index, uint64_t view, uint32_t slot) {
	auto& refs = m_elements[index].slot_refs;
	refs.push_back({view, slot});
	if (refs.size() <= 2u * static_cast<size_t>(m_elements[index].refs) + 32u) return;
	// Slots that moved away leave their entries; keep the ones that still select the element.
	std::ranges::sort(refs);
	refs.erase(std::ranges::unique(refs).begin(), refs.end());
	std::erase_if(refs, [&](const SlotRef& ref) {
		const auto found = m_views.find(ref.view);
		return found == m_views.end() || ref.slot >= found->second->slots.size() ||
		       (found->second->slots[ref.slot] & ElementMask) != index;
	});
}

// Moves the slots that select a marked element; pending draws keep reading those. Only the
// slots of elements marked since the last call are visited.
void ImageHeap::Repoint() {
	Common::FrameCost::Add(Common::FrameCost::Counter::HeapRepointCalls);
	m_repointing.clear();
	m_repointing.swap(m_marked);
	uint64_t visited = 0;
	for (const auto index: m_repointing) {
		auto refs = std::move(m_elements[index].slot_refs);
		m_elements[index].slot_refs.clear();
		for (const auto ref: refs) {
			const auto found = m_views.find(ref.view);
			if (found == m_views.end()) continue;
			auto& view = *found->second;
			if (ref.slot >= view.slots.size() || (view.slots[ref.slot] & ElementMask) != index) {
				continue;
			}
			++visited;
			const auto& element = m_elements[index];
			if (element.state == State::Buried) {
				view.memory.dead[ref.slot] = Hash(element.key.value);
				Assign(view, ref.slot, 0u);
			} else if (element.state == State::Stale) {
				const auto key  = element.key; // Intern may grow m_elements.
				const auto kind = element.kind;
				Assign(view, ref.slot, Intern(view, kind, key));
			}
		}
	}
	m_repoint_slots += visited;
	Common::FrameCost::Add(Common::FrameCost::Counter::HeapRepointSlots, visited);
}

// Page index entries of elements that no longer wait leave once they outnumber the live ones.
void ImageHeap::SweepResolutionIndex() {
	size_t kept = 0;
	for (auto page = m_resolution_elements.begin(); page != m_resolution_elements.end();) {
		std::erase_if(page->second, [&](const ElementRef& ref) {
			const auto& element = m_elements[ref.index];
			return element.retry == RetryKind::None || element.retry_stamp != ref.stamp;
		});
		kept += page->second.size();
		page = page->second.empty() ? m_resolution_elements.erase(page) : std::next(page);
	}
	m_resolution_refs = m_resolution_refs_floor = kept;
}

// Elements of images that changed identity or left stay in m_image_elements until swept.
void ImageHeap::SweepImageIndex() {
	size_t kept = 0;
	for (auto entry = m_image_elements.begin(); entry != m_image_elements.end();) {
		std::erase_if(entry->second, [&](uint32_t index) {
			const auto& element = m_elements[index];
			return element.state != State::Live || !element.image ||
			       element.serial != entry->first;
		});
		std::ranges::sort(entry->second);
		entry->second.erase(std::ranges::unique(entry->second).begin(), entry->second.end());
		kept += entry->second.size();
		entry = entry->second.empty() ? m_image_elements.erase(entry) : std::next(entry);
	}
	m_image_refs = m_image_refs_floor = kept;
}

void ImageHeap::TakeChanges() {
	m_changes.clear();
	{
		std::scoped_lock lock {Image::heap_mutex};
		std::swap(m_changes, Image::heap_changes);
		m_heap_change_generation =
		    Image::heap_change_generation.load(std::memory_order_acquire);
	}
	std::ranges::sort(m_changes, {}, &Image::HeapChange::serial);
	const auto [first, last] = std::ranges::unique(m_changes, {}, &Image::HeapChange::serial);
	m_changes.erase(first, last);
}

// The registered, unassociated image a change came from, if it still is one.
ImageId ImageHeap::CurrentImage(const Image::HeapChange& change) {
	auto&            textures = m_context.GetTextureCache();
	std::scoped_lock lock {textures.m_lock};
	ImageId          current;
	textures.ForEachImageWithAddress(change.data.address, [&](ImageId id, const Image& image) {
		if (image.serial == change.serial && !image.depth_id) current = id;
	});
	return current;
}

uint32_t ImageHeap::Allocate() {
	auto& scheduler = m_context.GetCommandScheduler();
	while (!m_retired.empty() && scheduler.IsFree(m_retired.front().first)) {
		m_free.push_back(m_retired.front().second);
		m_retired.pop_front();
	}
	if (m_free.empty()) {
		if (m_elements.size() == IR::ImageHeapCapacity) {
			EXIT("image heaps exceed %u resident images\n", IR::ImageHeapCapacity);
		}
		m_elements.emplace_back();
		return static_cast<uint32_t>(m_elements.size() - 1u);
	}
	const auto index = m_free.back();
	m_free.pop_back();
	m_elements[index] = {};
	return index;
}

void ImageHeap::Resize(Memory& memory, uint64_t base, uint64_t size) {
	{
		std::scoped_lock lock {m_mutex};
		Unwatch(memory);
		memory.base = base;
		memory.size = size;
		// The next refresh rereads the whole table.
		memory.dirty_pages.assign((Common::AlignUp(base + size, TRACKER_PAGE_SIZE) -
		                           Common::AlignDown(base, TRACKER_PAGE_SIZE)) /
		                              TRACKER_PAGE_SIZE,
		                          uint8_t {1});
	}
	memory.words.resize(size / 4u);
	std::erase_if(memory.dead, [&](const auto& word) { return word.first >= size / 4u; });
	auto& scheduler = m_context.GetCommandScheduler();
	for (auto& view: memory.views) {
		const auto count = std::min(view->limit, Probes(size, view->first, view->step));
		for (auto probe = count; probe < view->count; ++probe) {
			Assign(*view, (view->first + probe * view->step) / 4u, 0u);
		}
		view->count       = count;
		const auto slots  = view->slots.size();
		view->slots.resize(size / 4u + 1u);
		view->dirty_end   = std::min(view->dirty_end, view->slots.size());
		view->dirty_begin = std::min(view->dirty_begin, view->dirty_end);
		if (view->slots.size() * sizeof(uint32_t) > view->buffer->Size()) {
			if (scheduler.Active()) {
				scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
			}
			view->buffer = SlotBuffer(m_context, size);
			Widen(view->dirty_begin, view->dirty_end, 0u, view->slots.size());
		} else if (slots < view->slots.size()) {
			// Device words past the old end may hold stale slots.
			Widen(view->dirty_begin, view->dirty_end, slots, view->slots.size());
		}
	}
}

void ImageHeap::Rebind(RenderExecutor& executor) {
	KYTY_PROFILER_FUNCTION();
	Common::FrameCost::ScopedTimer frame_timer(Common::FrameCost::Counter::ImageHeapRebindNs);
	Common::FrameCost::Add(Common::FrameCost::Counter::ImageHeapRebindCalls);
	auto&      textures             = m_context.GetTextureCache();
	const auto cache_generation     = textures.ResolutionGeneration();
	const auto heap_change_generation =
	    Image::heap_change_generation.load(std::memory_order_acquire);
	const bool pending_clear = m_pending.empty();
	const bool descriptor_ok = m_descriptor_generation == m_rebound_descriptor_generation;
	const bool cache_ok      = cache_generation == m_cache_generation;
	const bool heap_ok       = heap_change_generation == m_heap_change_generation;
	if (Config::PortEnabled(Config::Port::ImageHeapIncremental) && pending_clear && descriptor_ok &&
	    cache_ok && heap_ok) {
		const auto skipped = static_cast<uint64_t>(m_retry_elements.size());
		++m_fast_rebinds;
		m_entries_skipped += skipped;
		Common::FrameCost::Add(Common::FrameCost::Counter::ImageHeapRebindFastCalls);
		Common::FrameCost::Add(Common::FrameCost::Counter::ImageHeapEntriesSkipped, skipped);
		return;
	}
	// Each failed condition counts, so one run names what keeps a rebind off the fast return.
	if (!pending_clear) Common::FrameCost::Add(Common::FrameCost::Counter::HeapMissPending);
	if (!descriptor_ok) Common::FrameCost::Add(Common::FrameCost::Counter::HeapMissDescriptorGen);
	if (!cache_ok) Common::FrameCost::Add(Common::FrameCost::Counter::HeapMissCacheGen);
	if (!heap_ok) Common::FrameCost::Add(Common::FrameCost::Counter::HeapMissHeapGen);
	uint64_t entries_visited  = 0;
	uint64_t entries_skipped  = 0;
	uint64_t entries_resolved = 0;
	// A changed image refreshes; one that changed identity needs new elements.
	TakeChanges();
	std::erase_if(m_changes, [&](const Image::HeapChange& change) {
		const auto id = CurrentImage(change);
		if (id) {
			textures.UpdateImage(id);
			const auto& image = textures.m_slot_images[id];
			if (const auto stencil = textures.FindStencilPlane(image, image.info.stencil)) {
				textures.UpdateImage(stencil);
			}
			m_away.emplace_back(change.serial, id);
		}
		return static_cast<bool>(id);
	});
	bool stale = ForEachChanged([&](uint32_t index) { Mark(index, State::Stale); }, &entries_visited);
	const auto resolve = [&](Element& element) {
		Common::FrameCost::ScopedTimer timer(Common::FrameCost::Counter::ResolveTextureNs);
		Common::FrameCost::Add(Common::FrameCost::Counter::ResolveTextureCalls);
		++m_resolve_calls;
		++entries_resolved;
		const auto& key = element.key;
		const auto resource = IR::ImageHeapResource(
		    {.dimension = key.dimension, .depth_compare = key.depth_view, .r128 = key.r128}, element.kind);
		auto binding = executor.ResolveTexture(resource, element.key.value);
		element.descriptor_hash = Hash(element.key.value);
		element.cache_generation = textures.ResolutionGeneration();
		element.data               = binding.desc.info.data;
		element.metadata           = binding.desc.info.metadata.range;
		return binding;
	};
	// A listed element whose descriptor resolves to another image is stale; true once it leaves
	// the list. A held element whose texture is unmapped stays null.
	const auto retry = [&](uint32_t index) {
		++entries_visited;
		auto& element = m_elements[index];
		if (element.state != State::Live) return true;
		const auto binding = resolve(element);
		if (binding.image_id == element.image &&
		    (!element.image || textures.m_slot_images[element.image].serial == element.serial)) {
			return !element.image && binding.desc.info.data.Empty();
		}
		Mark(index, State::Stale);
		stale = true;
		return true;
	};
	// One journal read turns the texture-cache generation into only the retry entries whose data
	// or metadata page changed. The page index may contain stale references, which stamps reject.
	bool journal_incomplete = false;
	m_cache_generation =
	    textures.TakeResolutionChangesSince(m_cache_generation, m_resolution_changes,
	                                        &journal_incomplete);
	m_retry_candidates.clear();
	if (Config::PortEnabled(Config::Port::ImageHeapIncremental) && !journal_incomplete) {
		m_changed_pages.clear();
		for (const auto range: m_resolution_changes) {
			textures.ForEachPage(range.address, range.size,
			                     [&](uint64_t page) { m_changed_pages.push_back(page); });
		}
		std::ranges::sort(m_changed_pages);
		m_changed_pages.erase(std::ranges::unique(m_changed_pages).begin(), m_changed_pages.end());
		const auto candidate_stamp = ++m_candidate_stamp;
		for (const auto page: m_changed_pages) {
			const auto indexed = m_resolution_elements.find(page);
			if (indexed == m_resolution_elements.end()) continue;
			for (const auto ref: indexed->second) {
				auto& element = m_elements[ref.index];
				if (element.retry == RetryKind::None || element.retry_stamp != ref.stamp ||
				    element.candidate_stamp == candidate_stamp) {
					continue;
				}
				element.candidate_stamp = candidate_stamp;
				m_retry_candidates.push_back(ref.index);
			}
			// Entries of elements that stopped waiting are dropped when their page is visited.
			const auto before = indexed->second.size();
			std::erase_if(indexed->second, [&](const ElementRef& ref) {
				const auto& element = m_elements[ref.index];
				return element.retry == RetryKind::None || element.retry_stamp != ref.stamp;
			});
			m_resolution_refs -= before - indexed->second.size();
		}
		entries_skipped = m_retry_elements.size() - m_retry_candidates.size();
	} else {
		m_retry_candidates.assign(m_retry_elements.begin(), m_retry_elements.end());
	}
	for (const auto index: m_retry_candidates) {
		if (retry(index)) SetRetry(index, RetryKind::None);
	}
	if (stale) Repoint();
	for (const auto index: m_pending) {
		++entries_visited;
		if (m_elements[index].refs == 0u) continue;
		auto binding = resolve(m_elements[index]);
		if (binding.image_id) {
			auto& image   = textures.GetImage(binding.image_id);
			auto& element = m_elements[index];
			// Pinned before the refresh, so a CPU write after it reports a change.
			image.heap_refs.fetch_add(1u, std::memory_order_relaxed);
			element.image      = binding.image_id;
			element.serial     = image.serial;
			binding.layout     = image.ReadLayout();
			binding.image_view = textures.FindTexture(binding.image_id, binding.desc);
			m_image_elements[image.serial].push_back(index);
			++m_image_refs;
			if (binding.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
				SetRetry(index, RetryKind::Dcc);
			}
			m_away.emplace_back(image.serial, binding.image_id);
		} else if (!binding.desc.info.data.Empty()) {
			SetRetry(index, RetryKind::Held);
		}
		m_writes.push_back({index, {nullptr, binding.image_view, binding.layout}});
	}
	m_pending.clear();
	WriteDescriptors();
	// Resolution itself can register or replace images. Those entries have already taken the
	// resulting state, matching the old end-of-rebind stamp, so acknowledge their journal records.
	m_cache_generation =
	    textures.TakeResolutionChangesSince(m_cache_generation, m_resolution_changes);
	m_rebound_descriptor_generation = m_descriptor_generation;
	// A generation already past the stored one makes the next rebind slow again: a bump by this
	// rebind's own work (or another thread's), named here so the heap line tells them apart.
	if (Image::heap_change_generation.load(std::memory_order_acquire) != m_heap_change_generation) {
		Common::FrameCost::Add(Common::FrameCost::Counter::HeapSelfBump);
	}
	if (textures.ResolutionGeneration() != m_cache_generation) {
		Common::FrameCost::Add(Common::FrameCost::Counter::CacheSelfBump);
	}
	if (m_resolution_refs > 2u * m_resolution_refs_floor + 65536u) SweepResolutionIndex();
	if (m_image_refs > 2u * m_image_refs_floor + 65536u) SweepImageIndex();
	m_entries_visited += entries_visited;
	m_entries_skipped += entries_skipped;
	m_entries_resolved += entries_resolved;
	Common::FrameCost::Add(Common::FrameCost::Counter::ImageHeapEntriesVisited, entries_visited);
	Common::FrameCost::Add(Common::FrameCost::Counter::ImageHeapEntriesSkipped, entries_skipped);
	Common::FrameCost::Add(Common::FrameCost::Counter::ImageHeapEntriesResolved, entries_resolved);
}

void ImageHeap::WriteDescriptors() {
	if (m_writes.empty()) return;
	std::vector<vk::WriteDescriptorSet> writes(m_writes.size());
	for (size_t write = 0; write < m_writes.size(); ++write) {
		writes[write].dstSet          = m_set;
		writes[write].dstArrayElement = m_writes[write].first;
		writes[write].descriptorCount = 1;
		writes[write].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[write].pImageInfo      = &m_writes[write].second;
	}
	m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
	                                                    writes.data(), 0, nullptr);
	m_writes.clear();
}

void ImageHeap::Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point,
                       vk::PipelineLayout layout) {
	KYTY_PROFILER_FUNCTION();
	for (auto* view: m_active) {
		if (view->dirty_begin == view->dirty_end) continue;
		const auto slots  = std::span(view->slots).subspan(view->dirty_begin,
		                                                   view->dirty_end - view->dirty_begin);
		auto&      stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		const auto offset = stream.Copy(slots.data(), slots.size_bytes(), sizeof(uint32_t));
		view->buffer->CopyFrom(buffer, stream, offset, view->dirty_begin * sizeof(uint32_t),
		                       slots.size_bytes());
		view->dirty_begin = view->dirty_end = 0;
	}
	m_active.clear();
	Retire();
	// Images that left the read layout since discovery, such as by its own uploads, return to it
	// now, freed ones too; the next discovery still takes their changes.
	{
		std::scoped_lock lock {Image::heap_mutex};
		m_changes.assign(Image::heap_changes.begin(), Image::heap_changes.end());
		// Layout-only changes need nothing but this return, so they leave the queue here instead
		// of waiting for a rebind that may not run.
		if (Image::heap_layout_quiet.load(std::memory_order_relaxed)) {
			std::erase_if(Image::heap_changes,
			              [](const Image::HeapChange& change) { return change.layout_only; });
		}
	}
	std::ranges::sort(m_changes, {}, &Image::HeapChange::serial);
	std::erase_if(m_changes, [&](const Image::HeapChange& change) {
		const auto id = CurrentImage(change);
		if (id) m_away.emplace_back(change.serial, id);
		return static_cast<bool>(id);
	});
	ForEachChanged([&](uint32_t index) {
		m_away.emplace_back(m_elements[index].serial, m_elements[index].image);
	});
	std::ranges::sort(m_away);
	m_away.erase(std::ranges::unique(m_away).begin(), m_away.end());
	// This draw's own bindings own the layouts of its attachments and storage images. An image
	// this draw's discovery freed lives until the tick ends, and its slots still select it.
	auto& textures = m_context.GetTextureCache();
	std::erase_if(m_away, [&](const std::pair<uint64_t, ImageId>& away) {
		auto* image = textures.m_slot_images.try_get(away.second);
		if (image == nullptr || image->serial != away.first ||
		    image->heap_refs.load(std::memory_order_relaxed) == 0u) {
			return true;
		}
		if (image->binding.is_target || image->binding.shader_write ||
		    image->binding.force_general) {
			return false;
		}
		image->Transit(image->ReadLayout(), vk::AccessFlagBits2::eShaderRead, std::nullopt,
		               buffer.Handle());
		return true;
	});
	buffer.Handle().bindDescriptorSets(bind_point, layout, IR::ImageHeapSet, 1, &m_set, 0,
	                                   nullptr);
}

void ImageHeap::UnmapMemory(uint64_t address, uint64_t size) {
	// The range's images are freed. Their words stay null, even once it is mapped again, until
	// their descriptors change; other changes wait for discovery.
	TakeChanges();
	const auto other = [&](const Image::HeapChange& change) {
		return !ImageRangeOverlaps(change.data, {address, size});
	};
	{
		std::scoped_lock lock {Image::heap_mutex};
		const auto before = Image::heap_changes.size();
		std::ranges::copy_if(m_changes, std::back_inserter(Image::heap_changes), other);
		if (Image::heap_changes.size() != before) {
			Image::heap_change_generation.fetch_add(1u, std::memory_order_release);
			Common::FrameCost::Add(Common::FrameCost::Counter::HeapBumpUnmap);
		}
	}
	std::erase_if(m_changes, other);
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		if (address >= memory->base + memory->size || memory->base >= address + size) {
			return false;
		}
		Drop(*memory);
		return true;
	});
	if (ForEachChanged([&](uint32_t index) { Mark(index, State::Buried); })) Repoint();
}

void ImageHeap::RunGarbageCollector() {
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		memory->idle = memory->draw > m_suspend_draw ? 0u : memory->idle + 1u;
		if (memory->idle < (memory->size < MinTableBytes ? 1u : LargeTableIdleFrames)) {
			return false;
		}
		Drop(*memory);
		return true;
	});
	m_suspend_draw = m_draw;
}

void ImageHeap::Drop(Memory& memory) {
	auto& scheduler = m_context.GetCommandScheduler();
	Unwatch(memory);
	for (auto& view: memory.views) {
		std::erase(m_active, view.get());
		m_views.erase(view->id);
		for (uint64_t probe = 0; probe < view->count; ++probe) {
			Release(view->slots[(view->first + probe * view->step) / 4u]);
		}
		if (scheduler.Active()) {
			scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
		}
	}
}

} // namespace Libs::Graphics
