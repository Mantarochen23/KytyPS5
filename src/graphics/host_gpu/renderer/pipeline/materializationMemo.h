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

struct XXH3_state_s;

namespace Libs::Graphics {

// Frees the streaming hash state of a memo (xxhash is included by the .cpp only).
struct HashStateDeleter {
	void operator()(XXH3_state_s* state) const;
};

// One page of watched guest memory at the generation it had when a memo entry was recorded.
struct PageStamp {
	const void* page       = nullptr;
	uint64_t    generation = 0;
};

// Says whether guest memory a memo entry read has changed since, without reading it. A stamp stays
// current until a write is seen over its page (CPU fault, host I/O into the page, unmap); bytes only
// the GPU holds are reported by GpuHolds, since reading them downloads new content.
class ReadStamps: public ShaderRecompiler::IR::MemoryChangeSource {
public:
	// Watches the pages of the range and appends their stamps. False, with nothing appended, when a
	// page cannot be proven tracked (unmapped, image-backed, written too often).
	virtual bool Track(uint64_t address, uint64_t size, std::vector<PageStamp>& stamps) = 0;
	// True when none of the stamped pages was written since.
	virtual bool Unchanged(std::span<const PageStamp> stamps) = 0;
	// True when the range holds GPU-written bytes that have not been downloaded.
	virtual bool GpuHolds(uint64_t address, uint64_t size) = 0;
	// General memo implementations need not expose an opaque stamp. The renderer's guest watch does,
	// so large runtime tables can retain their enumeration outside the general memo.
	std::shared_ptr<ShaderRecompiler::IR::MemoryChangeStamp> Track(uint64_t, uint64_t) override {
		return {};
	}

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
	uint32_t ranges_small    = 0; // ranges compared by their stored bytes
	uint32_t ranges_large    = 0; // ranges over SmallRangeBytes, compared by hash
	uint64_t bytes_large     = 0;
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
		uint32_t byte_offset = 0; // into Entry::bytes, when has_bytes
		bool     has_bytes   = false;
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
		// The bytes of every range of at most SmallRangeBytes, compared directly.
		std::vector<uint8_t>         bytes;
		std::vector<PageStamp>       pages;
		std::vector<RecordingImageHeap::Acquired> heap;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint64_t                     specialization_hash = 0;
		uint64_t                     serial              = 0;
		uint64_t                     used                = 0;
	};

	// Ranges up to this size keep their bytes and are compared with memcmp, larger ones by hash.
	static constexpr uint64_t SmallRangeBytes = 256;
	// "Compare the whole user-data vector" for Revalidate.
	static constexpr size_t AllUserData = static_cast<size_t>(-1);

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

	// Checks one entry against these inputs outside the memo's own lookup: the key (only the first
	// `user_data_words` user-data words, the ones the walk reads), every recorded range, and the heap
	// tables. On a hit the entry is current. A call-site record holds the entry beyond its eviction.
	bool Revalidate(Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                size_t user_data_words, MemoOutcome& outcome, MemoLookupStats& stats);
	// The entry the owner holds the result of right now (`current`), or null.
	[[nodiscard]] std::shared_ptr<Entry> ShareCurrent() const;
	// Bumped by Clear: what an entry shared before it described is gone.
	[[nodiscard]] uint64_t Generation() const { return m_generation; }
	// What the entry holds, for memory accounting.
	static uint64_t ApproximateBytes(const Entry& entry);

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
	                MemoOutcome& why, size_t user_data_words = AllUserData) const;
	// Reads one range through the runtime's reader and hashes it; false when it cannot be read.
	bool HashRange(const ShaderRecompiler::IR::SrtRuntime& runtime, uint64_t address, uint64_t size,
	               uint64_t& hash);
	bool ValidateReads(Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                   MemoOutcome& why, MemoLookupStats& stats);
	bool ReplayHeap(const Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime);

	std::vector<std::shared_ptr<Entry>> m_entries;
	std::vector<uint32_t>               m_scratch;
	uint32_t                            m_capacity = 1;
	uint64_t                            m_tick     = 0;
	uint64_t                            m_serials  = 0;
	uint64_t                            m_generation = 0;
	// Streaming state for ranges hashed in pieces; made once, so a hit allocates nothing.
	std::unique_ptr<XXH3_state_s, HashStateDeleter> m_hash_state;
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
