#include "graphics/host_gpu/renderer/pipeline/callSiteTable.h"

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
    std::fprintf(stderr, "CallSiteTableTests: failed: %s\n", text);
    std::abort();
  }
}

constexpr uint64_t PageSize = 4096;

struct Memory {
  std::map<uint64_t, std::vector<uint32_t>> pages;
  uint32_t reads = 0;

  void Write(uint64_t address, uint32_t value) {
    auto &page = pages[address / PageSize];
    page.resize(PageSize / 4, 0u);
    page[(address % PageSize) / 4] = value;
  }
  static bool Read(void *user, uint64_t address, std::span<uint32_t> values) {
    auto &self = *static_cast<Memory *>(user);
    ++self.reads;
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
  bool Track(uint64_t address, uint64_t size, std::vector<PageStamp> &stamps) override {
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
  bool GpuHolds(uint64_t, uint64_t) override { return false; }
  void Written(uint64_t address) {
    auto &node = pages[address / PageSize];
    if (node.watched) {
      ++node.generation;
      node.watched = false;
    }
  }
};

struct Inputs {
  std::vector<uint32_t> user_data{1, 2, 3, 4, 5, 6};
  uint64_t base = 0x40000;
  Memory *memory = nullptr;
  IR::SrtRuntime Runtime() const {
    IR::SrtRuntime runtime;
    runtime.user_data = user_data;
    runtime.shader_base = base;
    runtime.userdata = memory;
    runtime.read_specialization_memory = Memory::Read;
    return runtime;
  }
};

constexpr uint64_t Small = 0x10000;  // 64 bytes: compared by bytes.
constexpr uint64_t Large = 0x20000;  // 96 KiB: compared by hash, in several pieces.
constexpr uint64_t LargeSize = 96 << 10;
constexpr size_t ReadWords = 4;  // The walk reads the first four user-data words.

struct Fixture {
  Memory memory;
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  CallSiteTable table{4, 2};
  CallSiteKey key{.packet = 0x7000, .shader_hash = 0xaaaa, .stage = 1};
  int owner = 0;

  explicit Fixture(ReadStamps *stamps = nullptr) {
    memory.Write(Small, 11);
    memory.Write(Large, 22);
    memory.Write(Large + LargeSize - 4, 33);
    memo.Configure(1, stamps);
  }
  // Materializes for the current inputs the way a miss does, and records the call site.
  MemoizedMaterialization::Entry *Record(const CallSiteKey &at, void *module = nullptr) {
    RecordingImageHeap recorder(nullptr);
    uint32_t evicted = 0;
    MemoLookupStats stats;
    IR::ResourceSnapshot snapshot;
    snapshot.specialization_reads = {{Small, 64}, {Large, LargeSize}};
    snapshot.user_data.assign(inputs.user_data.begin(), inputs.user_data.end());
    auto *entry = memo.Store(inputs.Runtime(), snapshot, IR::ResourceSpecialization{}, 1, recorder, false,
                             evicted, stats);
    Check(entry != nullptr, "stored");
    memo.current = entry->serial;
    bool dropped = false;
    table.Store(at, &owner, memo.Generation(), memo.ShareCurrent(), module, dropped);
    return entry;
  }
  CallSiteOutcome Lookup(const CallSiteKey &at, CallSiteStats &stats, CallSiteTable::Record *&record) {
    stats = {};
    return table.Lookup(at, &owner, memo.Generation(), memo, inputs.Runtime(), ReadWords, record, stats);
  }
};

void TestHitAndUnreadWords() {
  Fixture f;
  int module = 0;
  f.Record(f.key, &module);
  f.memo.Clear();  // The program's memo is gone; the record's generation no longer matches.
  CallSiteStats stats;
  CallSiteTable::Record *record = nullptr;
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Stale, "a cleared memo makes records stale");

  Fixture g;
  g.Record(g.key, &module);
  Check(g.Lookup(g.key, stats, record) == CallSiteOutcome::Hit, "same inputs hit");
  Check(record->module == &module, "the module comes back");
  Check(stats.ranges_small == 1 && stats.ranges_large == 1, "one small, one large range compared");
  Check(stats.bytes_large == LargeSize, "the large range is counted");
  // A word the walk does not read changes (a vertex offset): still a hit, and the snapshot follows.
  g.inputs.user_data[5] = 99;
  Check(g.Lookup(g.key, stats, record) == CallSiteOutcome::Hit, "an unread word does not matter");
  Check(record->entry->resources.user_data[5] == 99 && record->entry->user_data[5] == 99,
        "the entry holds the current words");
  // A word the walk reads changes: miss by registers.
  g.inputs.user_data[2] = 7;
  Check(g.Lookup(g.key, stats, record) == CallSiteOutcome::Registers, "a read word misses");
  Check(stats.bytes_compared == 0, "a register miss reads no memory");
}

void TestSamePacketOtherRegisters() {
  Fixture f;
  f.Record(f.key);
  CallSiteStats stats;
  CallSiteTable::Record *record = nullptr;
  // A loop over one command buffer: the same packet again with other registers, base or shader.
  f.inputs.user_data[0] = 1000;
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Registers, "other registers miss");
  f.inputs.user_data[0] = 1;
  f.inputs.base = 0x50000;
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Registers, "other shader base misses");
  f.inputs.base = 0x40000;
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Hit, "back to the recorded inputs hits");
  auto other = f.key;
  other.shader_hash = 0xbbbb;
  Check(f.Lookup(other, stats, record) == CallSiteOutcome::NoRecord, "another shader is another site");
  other = f.key;
  other.packet += 0x40;
  Check(f.Lookup(other, stats, record) == CallSiteOutcome::NoRecord, "another packet is another site");
}

void TestChangedBytes() {
  Fixture f;
  f.Record(f.key);
  CallSiteStats stats;
  CallSiteTable::Record *record = nullptr;
  f.memory.Write(Small, 12);
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Ranges, "a changed byte in a small range misses");
  f.memory.Write(Small, 11);
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Hit, "restored bytes hit again");
  f.memory.Write(Large + LargeSize - 4, 34);  // In the last piece of the large range.
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Ranges, "a changed word in a large range misses");
  f.memory.Write(Large + LargeSize - 4, 33);
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Hit, "restored large range hits again");
}

void TestStampsSkipReads() {
  Stamps stamps;
  Fixture f(&stamps);
  f.Record(f.key);
  CallSiteStats stats;
  CallSiteTable::Record *record = nullptr;
  f.memory.reads = 0;
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Hit, "stamped hit");
  Check(f.memory.reads == 0 && stats.ranges_stamped == 2, "stamps spare every read");
  f.memory.Write(Small, 12);
  stamps.Written(Small);
  Check(f.Lookup(f.key, stats, record) == CallSiteOutcome::Ranges, "a written page falls back to the bytes");
}

void TestEviction() {
  Fixture f;
  CallSiteTable tiny(1, 2);  // One set, two ways.
  auto record_at = [&](uint64_t packet, bool &evicted) {
    f.Record({packet, 0xaaaa, 1});
    tiny.Store({packet, 0xaaaa, 1}, &f.owner, f.memo.Generation(), f.memo.ShareCurrent(), nullptr, evicted);
  };
  bool evicted = false;
  record_at(0x100, evicted);
  Check(!evicted, "first fits");
  record_at(0x200, evicted);
  Check(!evicted && tiny.Live() == 2, "second fits");
  CallSiteStats stats;
  CallSiteTable::Record *record = nullptr;
  // Touch the first so the second is the oldest.
  Check(tiny.Lookup({0x100, 0xaaaa, 1}, &f.owner, f.memo.Generation(), f.memo, f.inputs.Runtime(), ReadWords,
                    record, stats) == CallSiteOutcome::Hit, "first hits");
  record_at(0x300, evicted);
  Check(evicted && tiny.Live() == 2, "third evicts the oldest");
  Check(tiny.Lookup({0x200, 0xaaaa, 1}, &f.owner, f.memo.Generation(), f.memo, f.inputs.Runtime(), ReadWords,
                    record, stats) == CallSiteOutcome::NoRecord, "the evicted site is gone");
  Check(tiny.Lookup({0x100, 0xaaaa, 1}, &f.owner, f.memo.Generation(), f.memo, f.inputs.Runtime(), ReadWords,
                    record, stats) == CallSiteOutcome::Hit, "the touched site stayed");
  Check(tiny.Bytes() > 0, "bytes are counted");
  tiny.Clear();
  Check(tiny.Live() == 0 && tiny.Bytes() == 0, "clear empties");
}

}  // namespace

int main() {
  TestHitAndUnreadWords();
  TestSamePacketOtherRegisters();
  TestChangedBytes();
  TestStampsSkipReads();
  TestEviction();
  std::puts("CallSiteTableTests: ok");
  return 0;
}
