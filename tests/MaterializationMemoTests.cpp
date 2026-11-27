#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MaterializationMemoTests: failed: %s\n", text);
    std::abort();
  }
}

constexpr uint64_t PageSize = 4096;

// Guest memory by 4 KiB page, and what reading it costs.
struct Memory {
  std::map<uint64_t, std::vector<uint32_t>> pages;
  uint32_t reads = 0;
  bool readable = true;

  void Write(uint64_t address, uint32_t value) {
    auto &page = pages[address / PageSize];
    page.resize(PageSize / 4, 0u);
    page[(address % PageSize) / 4] = value;
  }
  static bool Read(void *user, uint64_t address, std::span<uint32_t> values) {
    auto &self = *static_cast<Memory *>(user);
    ++self.reads;
    if (!self.readable) return false;
    for (size_t i = 0; i < values.size(); ++i) {
      const auto at = address + i * 4;
      const auto found = self.pages.find(at / PageSize);
      values[i] = found == self.pages.end() ? 0u : found->second[(at % PageSize) / 4];
    }
    return true;
  }
};

struct Stamps final : ReadStamps {
  struct Page {
    uint64_t generation = 1;
    bool watched = false;
  };
  std::map<uint64_t, Page> pages;
  std::set<uint64_t> untrackable; // Page indices.
  bool gpu_holds = false;
  uint32_t tracks = 0;

  bool Track(uint64_t address, uint64_t size, std::vector<PageStamp> &stamps) override {
    ++tracks;
    for (auto page = address / PageSize; page <= (address + size - 1) / PageSize; ++page) {
      if (untrackable.contains(page)) return false;
    }
    for (auto page = address / PageSize; page <= (address + size - 1) / PageSize; ++page) {
      auto &node = pages[page];
      node.watched = true;
      stamps.push_back({&node, node.generation});
    }
    return true;
  }
  bool Unchanged(std::span<const PageStamp> stamps) override {
    for (const auto &stamp : stamps) {
      const auto &node = *static_cast<const Page *>(stamp.page);
      if (node.generation != stamp.generation || !node.watched) return false;
    }
    return true;
  }
  bool GpuHolds(uint64_t, uint64_t) override { return gpu_holds; }
  // A write over the page: the fault bumps the generation and releases the watch.
  void Written(uint64_t address) {
    auto &node = pages[address / PageSize];
    if (node.watched) {
      ++node.generation;
      node.watched = false;
    }
  }
};

struct Heap final : IR::ImageHeapSource {
  uint64_t slots = 0x1000;
  uint32_t slot_count = 8;
  std::vector<IR::DescriptorValue> kinds{IR::DescriptorValue{.dwords = {1}, .dword_count = 8}};
  uint32_t acquires = 0;
  bool Acquire(const IR::ImageHeapRequest &, IR::ImageHeapView &view) override {
    ++acquires;
    view = {.slots = slots, .slot_count = slot_count, .kinds = kinds};
    return true;
  }
};

struct Inputs {
  std::vector<uint32_t> user_data{1, 2, 3, 4};
  std::vector<uint32_t> workgroups;
  uint64_t base = 0x40000;
  uint64_t indirect = 0;
  bool legacy = false;
  IR::ImageHeapSource *heap = nullptr;
  Memory *memory = nullptr;

  IR::SrtRuntime Runtime() const {
    IR::SrtRuntime runtime;
    runtime.user_data = user_data;
    runtime.shader_base = base;
    runtime.userdata = memory;
    runtime.read_specialization_memory = Memory::Read;
    runtime.workgroup_counts = workgroups;
    runtime.indirect_args_address = indirect;
    runtime.image_heap = heap;
    runtime.legacy_hash = legacy;
    return runtime;
  }
};

IR::ResourceSnapshot Snapshot(std::vector<std::pair<uint64_t, uint64_t>> reads) {
  IR::ResourceSnapshot snapshot;
  snapshot.specialization_reads = std::move(reads);
  snapshot.flattened_srt = {7, 8, 9};
  return snapshot;
}

MemoOutcome Lookup(MemoizedMaterialization &memo, const Inputs &inputs, bool heap_plan,
                   MemoLookupStats &stats) {
  MemoOutcome outcome = MemoOutcome::Count;
  stats = {};
  const auto *entry = memo.Lookup(inputs.Runtime(), heap_plan, outcome, stats);
  Check((entry != nullptr) == (outcome == MemoOutcome::Hit), "a hit returns its entry, a miss none");
  return outcome;
}

const void *Store(MemoizedMaterialization &memo, const Inputs &inputs,
                  std::vector<std::pair<uint64_t, uint64_t>> reads, bool heap_plan = false,
                  uint32_t *evicted = nullptr, IR::ImageHeapSource *source = nullptr) {
  RecordingImageHeap recorder(source != nullptr ? source : inputs.heap);
  if (heap_plan && source != nullptr) {
    IR::ImageHeapView view;
    IR::ImageHeapRequest request{.base = 0x9000, .size = 64, .step = 4, .first = 0, .count = 16};
    Check(recorder.Acquire(request, view), "heap acquire");
  }
  uint32_t dropped = 0;
  MemoLookupStats stats;
  return memo.Store(inputs.Runtime(), Snapshot(std::move(reads)), IR::ResourceSpecialization{}, 1,
                    recorder, heap_plan, evicted != nullptr ? *evicted : dropped, stats);
}

constexpr uint64_t Table = 0x10000;  // Descriptor table: one page.
constexpr uint64_t Block = 0x20000;  // User-data block: one page.
constexpr uint64_t Spans = 0x30F00;  // A range that crosses a page boundary.

void FillMemory(Memory &memory) {
  memory.Write(Table, 11);
  memory.Write(Block, 22);
  memory.Write(Spans, 33);
  memory.Write(Spans + 0x200, 44);
}

const std::vector<std::pair<uint64_t, uint64_t>> Reads{{Table, 64}, {Block, 32}, {Spans, 0x300}};

void TestHitWithoutChange() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, nullptr);
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Empty, "nothing stored yet");
  Check(Store(memo, inputs, Reads) != nullptr, "stored");
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Hit, "same inputs, same bytes: hit");
  Check(stats.ranges_compared == 3 && stats.ranges_stamped == 0, "without stamps every range is compared");
  memory.Write(Block, 23);
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::RangeChanged, "changed bytes: miss");
}

void TestStampsSkipReads() {
  Memory memory;
  FillMemory(memory);
  Stamps stamps;
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, &stamps);
  Check(Store(memo, inputs, Reads) != nullptr, "stored");
  const auto after_store = memory.reads;
  MemoLookupStats stats;
  for (int i = 0; i < 5; ++i) {
    Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Hit, "unchanged: hit");
    Check(stats.ranges_stamped == 3 && stats.ranges_compared == 0, "every range validated by its stamp");
  }
  Check(memory.reads == after_store, "a hit reads no guest memory");
}

// A write to each kind of tracked range: the table, the user-data block, the range over two pages
// (either page), and a write that leaves the bytes as they were.
void TestWriteToEachRange() {
  struct Case {
    uint64_t address;
    uint32_t value;
    MemoOutcome expected;
  };
  const Case cases[] = {{Table, 12, MemoOutcome::RangeChanged},
                        {Block, 23, MemoOutcome::RangeChanged},
                        {Spans, 34, MemoOutcome::RangeChanged},
                        {Spans + 0x200, 45, MemoOutcome::RangeChanged},
                        {Table, 11, MemoOutcome::Hit}};
  for (const auto &item : cases) {
    Memory memory;
    FillMemory(memory);
    Stamps stamps;
    Inputs inputs{.memory = &memory};
    MemoizedMaterialization memo;
    memo.Configure(1, &stamps);
    Store(memo, inputs, Reads);
    memory.Write(item.address, item.value);
    stamps.Written(item.address);
    MemoLookupStats stats;
    Check(Lookup(memo, inputs, false, stats) == item.expected, "a write is seen exactly when it changed bytes");
    if (item.expected == MemoOutcome::Hit) {
      // The range was compared once and is watched again.
      Check(stats.ranges_compared == 1 && stats.ranges_stamped == 2, "only the written range is read");
      Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Hit && stats.ranges_stamped == 3 &&
                stats.ranges_compared == 0,
            "after an equal rewrite the stamps are current again");
    }
  }
}

void TestUntrackedRangesAreCompared() {
  Memory memory;
  FillMemory(memory);
  Stamps stamps;
  stamps.untrackable.insert(Block / PageSize); // For example the guest stack.
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, &stamps);
  Store(memo, inputs, Reads);
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Hit && stats.ranges_stamped == 2 &&
            stats.ranges_compared == 1,
        "the range that cannot be tracked is compared, the others are stamped");
  memory.Write(Block, 99);
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::RangeChanged,
        "a change in an untracked range is found without any write notice");
}

void TestGpuHeldBytes() {
  Memory memory;
  FillMemory(memory);
  Stamps stamps;
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, &stamps);
  Store(memo, inputs, Reads);
  stamps.gpu_holds = true;
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Hit && stats.ranges_compared == 3,
        "bytes the GPU holds are read even when no page was written");
  memory.Write(Table, 5); // What a download would bring.
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::RangeChanged, "and a changed download is a miss");
}

void TestKeyMisses() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, nullptr);
  Store(memo, inputs, Reads);
  MemoLookupStats stats;
  auto changed = inputs;
  changed.user_data[2] = 77;
  Check(Lookup(memo, changed, false, stats) == MemoOutcome::UserData, "user data");
  changed = inputs;
  changed.base = 0x50000;
  Check(Lookup(memo, changed, false, stats) == MemoOutcome::ShaderBase, "shader base");
  changed = inputs;
  changed.indirect = 0x8000;
  Check(Lookup(memo, changed, false, stats) == MemoOutcome::IndirectArgs, "indirect arguments");
  changed = inputs;
  changed.legacy = true;
  Check(Lookup(memo, changed, false, stats) == MemoOutcome::LegacyHash, "legacy hash");
  changed = inputs;
  changed.workgroups = {1, 2, 3};
  Check(Lookup(memo, changed, false, stats) == MemoOutcome::WorkgroupCounts, "workgroup counts");
  Check(stats.ranges_compared == 0, "a key miss reads nothing");
}

void TestEntriesAlternate() {
  Memory memory;
  FillMemory(memory);
  Inputs a{.memory = &memory};
  Inputs b = a;
  b.user_data[0] = 100;
  MemoizedMaterialization single;
  single.Configure(1, nullptr);
  Store(single, a, Reads);
  MemoLookupStats stats;
  Check(Lookup(single, a, false, stats) == MemoOutcome::Hit, "one entry holds one draw");
  uint32_t evicted = 0;
  Store(single, b, Reads, false, &evicted);
  Check(evicted == 1 && Lookup(single, a, false, stats) == MemoOutcome::UserData,
        "the second draw evicts the first: a thrash");

  MemoizedMaterialization several;
  several.Configure(4, nullptr);
  Store(several, a, Reads);
  Store(several, b, Reads);
  for (int i = 0; i < 4; ++i) {
    Check(Lookup(several, a, false, stats) == MemoOutcome::Hit, "first draw hits");
    Check(Lookup(several, b, false, stats) == MemoOutcome::Hit, "second draw hits");
  }
  // Same inputs, different bytes in two moments: both can be remembered.
  const auto before = memory.reads;
  (void)before;
  Check(several.Size() == 2, "two entries");
}

void TestHeapPlans() {
  Memory memory;
  FillMemory(memory);
  Heap heap;
  Inputs inputs{.heap = &heap, .memory = &memory};
  const std::vector<std::pair<uint64_t, uint64_t>> reads{{Table, 64}, {0x9000, 64}};
  MemoizedMaterialization single;
  single.Configure(1, nullptr);
  MemoLookupStats stats;
  Check(Store(single, inputs, reads, true, nullptr, &heap) == nullptr, "a single entry never keeps a heap plan");

  MemoizedMaterialization memo;
  memo.Configure(2, nullptr);
  Check(Store(memo, inputs, reads, true, nullptr, &heap) != nullptr, "several entries keep a heap plan");
  heap.acquires = 0;
  memory.reads = 0;
  Check(Lookup(memo, inputs, true, stats) == MemoOutcome::Hit, "heap plan hits");
  Check(heap.acquires == 1, "a hit still acquires the heap table, which is the per-draw side effect");
  Check(stats.ranges_compared == 1, "the heap table itself is not read or hashed: only the other range");
  memory.Write(0x9000, 5);
  Check(Lookup(memo, inputs, true, stats) == MemoOutcome::Hit, "a write inside the heap table is the heap's to report");
  heap.slots = 0x2000;
  Check(Lookup(memo, inputs, true, stats) == MemoOutcome::HeapChanged, "other slots: miss");
  heap.slots = 0x1000;
  heap.kinds.push_back({});
  Check(Lookup(memo, inputs, true, stats) == MemoOutcome::HeapChanged, "other kinds: miss");
  Check(Lookup(single, inputs, true, stats) == MemoOutcome::Empty, "single entry, nothing stored");
  MemoizedMaterialization single_with_entry;
  single_with_entry.Configure(1, nullptr);
  Store(single_with_entry, inputs, reads);
  Check(Lookup(single_with_entry, inputs, true, stats) == MemoOutcome::HeapPlan, "a single entry reports the heap plan");
}

void TestUnreadable() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, nullptr);
  memory.readable = false;
  Check(Store(memo, inputs, Reads) == nullptr, "an unreadable range cannot be remembered");
  memory.readable = true;
  Check(Store(memo, inputs, Reads) != nullptr, "stored");
  memory.readable = false;
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Unreadable, "unreadable at lookup");
  memory.readable = true;
  const std::vector<std::pair<uint64_t, uint64_t>> odd{{Table, 6}};
  Check(Store(memo, inputs, odd) == nullptr, "a size that is not whole dwords cannot be remembered");
}

void TestClear() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(2, nullptr);
  Store(memo, inputs, Reads);
  memo.Clear();
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, false, stats) == MemoOutcome::Empty, "a changed plan drops every entry");
}

} // namespace

// A write that lands between the walk's read and the arming must not become the entry's content.
void TestWriteBetweenWalkAndStore() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  const std::vector<std::pair<uint64_t, uint64_t>> reads{{Table, 64}, {Block, 32}};
  const auto walk_and_store = [&](bool write_after_walk, bool write_during_walk, bool &changed) {
    MemoizedMaterialization memo;
    memo.Configure(2, nullptr);
    WalkReadHashes hashes;
    auto walk = hashes.Wrap(inputs.Runtime());
    std::vector<uint32_t> words(16);
    for (const auto &[address, size] : reads) {
      std::vector<uint32_t> data(size / 4);
      Check(walk.read_specialization_memory(walk.userdata, address, data), "walk read");
      if (write_during_walk) {
        memory.Write(address, 1000u);
        Check(walk.read_specialization_memory(walk.userdata, address, data), "walk read again");
      }
    }
    if (write_after_walk) memory.Write(Block, 77);
    RecordingImageHeap recorder(nullptr);
    uint32_t evicted = 0;
    MemoLookupStats stats;
    const auto *entry = memo.Store(inputs.Runtime(), Snapshot(reads), IR::ResourceSpecialization{}, 1,
                                   recorder, false, evicted, stats, &hashes);
    changed = stats.walk_changed;
    return entry != nullptr;
  };
  bool changed = false;
  Check(walk_and_store(false, false, changed) && !changed, "bytes as the walk read them: stored");
  Check(!walk_and_store(true, false, changed) && changed, "a write after the walk's read: not stored");
  FillMemory(memory);
  Check(!walk_and_store(false, true, changed) && changed, "a range that changed during the walk: not stored");
  FillMemory(memory);
  // Without the walk's hashes the old behaviour holds.
  MemoizedMaterialization plain;
  plain.Configure(2, nullptr);
  Check(Store(plain, inputs, reads) != nullptr, "no walk hashes: stored as before");
}

// A program whose memo keeps missing stops storing for Cooldown misses, then measures again; one
// that hits often enough keeps storing.
void TestStoreGate() {
  using Gate = Libs::Graphics::MemoizedMaterialization::StoreGate;
  Gate busy;
  for (uint32_t i = 0; i < Gate::Window; i++) {
    Check(busy.Allows(), "stores while measuring");
    busy.Note(i % 8 == 0); // 8 hits in 64: enough
  }
  Check(busy.Allows(), "enough hits keep storing");

  Gate missing;
  for (uint32_t i = 0; i < Gate::Window - 1; i++) missing.Note(i < Gate::MinHits - 1);
  Check(missing.Allows(), "not judged before the window is full");
  missing.Note(false);
  Check(!missing.Allows(), "too few hits: stores stop");
  for (uint32_t i = 0; i < Gate::Cooldown - 1; i++) missing.Note(false);
  Check(!missing.Allows(), "still cooling down one lookup before the end");
  missing.Note(false);
  Check(missing.Allows(), "measures again after the cooldown");
  // The new window starts from zero.
  for (uint32_t i = 0; i < Gate::Window; i++) missing.Note(true);
  Check(missing.Allows(), "a program that settled down is stored again");
}

int main() {
  TestStoreGate();
  TestHitWithoutChange();
  TestStampsSkipReads();
  TestWriteToEachRange();
  TestUntrackedRangesAreCompared();
  TestGpuHeldBytes();
  TestKeyMisses();
  TestEntriesAlternate();
  TestHeapPlans();
  TestUnreadable();
  TestClear();
  TestWriteBetweenWalkAndStore();
  std::puts("MaterializationMemoTests: ok");
  return 0;
}
