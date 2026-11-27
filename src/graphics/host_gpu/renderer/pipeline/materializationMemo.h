#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATERIALIZATIONMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATERIALIZATIONMEMO_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

// One page of watched guest memory at the generation it had when a memo entry was recorded.
struct PageStamp {
	const void* page       = nullptr;
	uint64_t    generation = 0;
};

// Says whether guest memory a memo entry read has changed since, without reading it. A stamp stays
// current until a write is seen over its page (CPU fault, host I/O into the page, unmap); bytes only
// the GPU holds are reported by GpuHolds, since reading them downloads new content.
class ReadStamps {
public:
	// Watches the pages of the range and appends their stamps. False, with nothing appended, when a
	// page cannot be proven tracked (unmapped, image-backed, written too often).
	virtual bool Track(uint64_t address, uint64_t size, std::vector<PageStamp>& stamps) = 0;
	// True when none of the stamped pages was written since.
	virtual bool Unchanged(std::span<const PageStamp> stamps) = 0;
	// True when the range holds GPU-written bytes that have not been downloaded.
	virtual bool GpuHolds(uint64_t address, uint64_t size) = 0;

protected:
	~ReadStamps() = default;
};

// Records which resident image heap tables a materialization acquired and what it was given.
class RecordingImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	struct Acquired {
		ShaderRecompiler::IR::ImageHeapRequest request;
		uint64_t                               slots      = 0;
		uint32_t                               slot_count = 0;
		std::vector<ShaderRecompiler::IR::DescriptorValue> kinds;
	};

	explicit RecordingImageHeap(ShaderRecompiler::IR::ImageHeapSource* inner): m_inner(inner) {}
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView&          view) override;
	[[nodiscard]] bool Declined() const { return m_declined; }
	std::vector<Acquired>& Take() { return m_acquired; }

private:
	ShaderRecompiler::IR::ImageHeapSource* m_inner    = nullptr;
	std::vector<Acquired>                  m_acquired;
	bool                                   m_declined = false;
};

// Hashes every guest read a materialization walk makes, as the walk sees it. A memo entry is only
// valid if the bytes it later arms and hashes are the bytes the walk saw: a write between the walk's
// read and the arming would otherwise be recorded as the entry's content while the snapshot holds
// the older bytes, and every later lookup would hit on it.
class WalkReadHashes {
public:
	// The runtime with its reader wrapped; this object must outlive the walk.
	[[nodiscard]] ShaderRecompiler::IR::SrtRuntime Wrap(const ShaderRecompiler::IR::SrtRuntime& runtime);
	enum class State : uint8_t { NotRead, Stable, Unstable };
	// Stable: every read of this range returned the hash in `hash`.
	[[nodiscard]] State Find(uint64_t address, uint64_t size, uint64_t& hash) const;

private:
	static bool Read(void* user, uint64_t address, std::span<uint32_t> values);
	struct Seen {
		uint64_t hash     = 0;
		bool     unstable = false;
	};
	ShaderRecompiler::IR::SrtRuntime                      m_inner;
	std::map<std::pair<uint64_t, uint64_t>, Seen>         m_seen;
};

// Why a lookup did not reuse an entry; Hit when it did.
enum class MemoOutcome : uint8_t {
	Hit,
	Empty,        // no entry was ever stored (first use, or the plan changed)
	HeapPlan,     // the plan uses the resident image heap and the memo is single-entry
	ShaderBase,
	IndirectArgs,
	LegacyHash,
	UserData,
	WorkgroupCounts,
	Unreadable,   // a recorded range cannot be read now
	RangeChanged, // the key matched, a recorded range holds other bytes
	HeapChanged,  // the key and ranges matched, the resident heap hands out something else
	Count
};

struct MemoLookupStats {
	uint32_t ranges_stamped  = 0; // ranges validated by their stamps alone
	uint32_t ranges_compared = 0; // ranges read and hashed
	uint64_t bytes_compared  = 0;
	uint32_t entries_tried   = 0; // entries whose key matched
	bool     walk_changed    = false; // Store: a range held other bytes than the walk read
};

// What a program remembers of its last materializations. An entry is the full result for one set
// of inputs (user data, shader base, dispatch inputs, the bytes of every guest range the walk
// read, the heap tables acquired). A lookup that matches the inputs of an entry reads nothing for
// a range whose stamp is current; it reads and hashes the others, and keeps the entry when they
// hold the same bytes (the range is then tracked again).
class MemoizedMaterialization {
public:
	struct Read {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t hash    = 0;
		uint32_t first_page = 0; // into Entry::pages, when tracked
		uint32_t page_count = 0;
		bool     tracked    = false;
	};
	struct Entry {
		std::vector<uint32_t>        user_data;
		std::vector<uint32_t>        workgroup_counts;
		uint64_t                     shader_base           = 0;
		uint64_t                     indirect_args_address = 0;
		bool                         legacy_hash           = false;
		std::vector<Read>            reads;
		std::vector<PageStamp>       pages;
		std::vector<RecordingImageHeap::Acquired> heap;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint64_t                     specialization_hash = 0;
		uint64_t                     serial              = 0;
		uint64_t                     used                = 0;
	};

	// `entries` is how many results are kept (at least one); `stamps` the guest memory stamp source,
	// or null to compare every range.
	void Configure(uint32_t entries, ReadStamps* stamps);
	[[nodiscard]] uint32_t Capacity() const { return m_capacity; }
	void Clear();

	// The entry for these inputs, or null with the reason in `outcome`.
	Entry* Lookup(const ShaderRecompiler::IR::SrtRuntime& runtime, bool uses_image_heap,
	              MemoOutcome& outcome, MemoLookupStats& stats);
	// Records a materialization. Null when its inputs cannot be recorded (an unreadable range, a
	// heap table that was not acquired); the caller then has no entry for them.
	Entry* Store(const ShaderRecompiler::IR::SrtRuntime&          runtime,
	             const ShaderRecompiler::IR::ResourceSnapshot&       resources,
	             const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	             uint64_t specialization_hash, RecordingImageHeap& heap, bool uses_image_heap,
	             uint32_t& evicted, MemoLookupStats& stats, const WalkReadHashes* walk = nullptr);

	// Serial of the entry whose result the owner holds right now, 0 for none.
	uint64_t current = 0;
	// Test and report visible.
	[[nodiscard]] size_t   Size() const { return m_entries.size(); }
	[[nodiscard]] uint64_t ApproximateBytes() const;

	// Storing costs a read and a hash of every range the walk read (hundreds of KiB for a large
	// table walk). A program whose key changes on nearly every call (per-object user data) never
	// gets that back: its entry is evicted before the next lookup can hit. The gate counts the
	// lookups of a window; with fewer than MinHits hits in Window lookups, the next Cooldown
	// misses are not stored. Then it measures again, so a program that settles down is memoized.
	struct StoreGate {
		static constexpr uint32_t Window   = 64;
		static constexpr uint32_t MinHits  = 4;
		static constexpr uint32_t Cooldown = 1024;

		uint32_t lookups  = 0;
		uint32_t hits     = 0;
		uint32_t cooldown = 0;

		void Note(bool hit) {
			if (cooldown != 0) {
				cooldown--;
				return;
			}
			lookups++;
			hits += hit ? 1u : 0u;
			if (lookups >= Window) {
				if (hits < MinHits) cooldown = Cooldown;
				lookups = 0;
				hits    = 0;
			}
		}
		[[nodiscard]] bool Allows() const { return cooldown == 0; }
	};
	StoreGate store_gate;

private:
	bool KeyMatches(const Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                MemoOutcome& why) const;
	// Reads one range through the runtime's reader and hashes it; false when it cannot be read.
	bool HashRange(const ShaderRecompiler::IR::SrtRuntime& runtime, uint64_t address, uint64_t size,
	               uint64_t& hash);
	bool ValidateReads(Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                   MemoOutcome& why, MemoLookupStats& stats);
	bool ReplayHeap(const Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime);

	std::vector<std::unique_ptr<Entry>> m_entries;
	std::vector<uint32_t>               m_scratch;
	uint32_t                            m_capacity = 1;
	uint64_t                            m_tick     = 0;
	uint64_t                            m_serials  = 0;
	ReadStamps*                         m_stamps   = nullptr;
};

// How every program's memo is configured: --materialize-memo-entries and the stamp source of the
// render context (null without --materialize-memo-stamps). Set once at start-up.
struct MaterializationMemoSettings {
	uint32_t    entries = 1;
	ReadStamps* stamps  = nullptr;
};
MaterializationMemoSettings& GlobalMaterializationMemoSettings();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATERIALIZATIONMEMO_H_
