#include "kernel/physicalCoverage.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

using Libs::LibKernel::Memory::PhysicalCoverage;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PhysicalCoverageTests: failed: %s\n", message);
		std::abort();
	}
}

std::vector<std::pair<uint64_t, uint64_t>> Covered(const PhysicalCoverage& c, uint64_t start,
                                                   uint64_t size) {
	std::vector<std::pair<uint64_t, uint64_t>> out;
	c.ForEachCovered(start, size, [&](uint64_t first, uint64_t last) { out.emplace_back(first, last); });
	return out;
}

void TestBasics() {
	PhysicalCoverage c;
	Check(Covered(c, 0, 100).empty(), "empty covers nothing");
	c.Add(10, 20); // [10, 30)
	Check(c.CountAt(9) == 0 && c.CountAt(10) == 1 && c.CountAt(29) == 1 && c.CountAt(30) == 0,
	      "single range counts");
	auto v = Covered(c, 0, 100);
	Check(v.size() == 1 && v[0] == std::pair<uint64_t, uint64_t>(10, 30), "single range piece");
	v = Covered(c, 15, 5);
	Check(v.size() == 1 && v[0] == std::pair<uint64_t, uint64_t>(15, 20), "clipped to the query");

	c.Add(20, 20); // alias overlap: [20, 30) has 2, [30, 40) has 1
	Check(c.CountAt(25) == 2 && c.CountAt(35) == 1, "overlapping counts");
	v = Covered(c, 0, 100);
	Check(v.size() == 1 && v[0] == std::pair<uint64_t, uint64_t>(10, 40),
	      "pieces with different nonzero counts are reported as one");

	c.Remove(10, 20); // the first mapping goes; [20, 40) still covered once
	v = Covered(c, 0, 100);
	Check(v.size() == 1 && v[0] == std::pair<uint64_t, uint64_t>(20, 40), "alias keeps its part");
	c.Remove(20, 20);
	Check(Covered(c, 0, 100).empty() && c.BoundaryCount() == 0, "everything removed, map empty");
}

void TestGapsAndMerging() {
	PhysicalCoverage c;
	c.Add(0, 10);
	c.Add(20, 10);
	auto v = Covered(c, 0, 40);
	Check(v.size() == 2 && v[0] == std::pair<uint64_t, uint64_t>(0, 10) &&
	          v[1] == std::pair<uint64_t, uint64_t>(20, 30),
	      "gap between two ranges");
	c.Add(10, 10); // fills the gap: one run of count 1
	Check(c.BoundaryCount() == 2, "adjacent equal runs merge");
	v = Covered(c, 0, 40);
	Check(v.size() == 1 && v[0] == std::pair<uint64_t, uint64_t>(0, 30), "merged run");
	// Splitting one mapping in two (as SplitMapping does) leaves the coverage as it was.
	c.Remove(10, 10);
	c.Add(10, 5);
	c.Add(15, 5);
	Check(c.BoundaryCount() == 2 && Covered(c, 0, 40).size() == 1, "split mapping, same coverage");
}

// Random adds and removes of previously added ranges, against a per-address count.
void TestAgainstBruteForce() {
	constexpr uint64_t Space = 512;
	std::mt19937_64    rng(12345);
	for (int round = 0; round < 200; round++) {
		PhysicalCoverage                            c;
		std::vector<int64_t>                        counts(Space, 0);
		std::vector<std::pair<uint64_t, uint64_t>>  live;
		for (int step = 0; step < 300; step++) {
			if (!live.empty() && rng() % 3 == 0) {
				const auto index = rng() % live.size();
				const auto [s, n] = live[index];
				live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
				c.Remove(s, n);
				for (uint64_t a = s; a < s + n; a++) counts[a]--;
			} else {
				const uint64_t s = rng() % (Space - 1);
				const uint64_t n = 1 + rng() % std::min<uint64_t>(64, Space - s - 1);
				live.emplace_back(s, n);
				c.Add(s, n);
				for (uint64_t a = s; a < s + n; a++) counts[a]++;
			}
			for (uint64_t a = 0; a < Space; a++) {
				if (c.CountAt(a) != counts[a]) Check(false, "count differs from brute force");
			}
			const uint64_t qs = rng() % Space;
			const uint64_t qn = rng() % (Space - qs + 1);
			std::vector<std::pair<uint64_t, uint64_t>> expected;
			for (uint64_t a = qs; a < qs + qn; a++) {
				if (counts[a] <= 0) continue;
				if (!expected.empty() && expected.back().second == a) {
					expected.back().second++;
				} else {
					expected.emplace_back(a, a + 1);
				}
			}
			if (Covered(c, qs, qn) != expected) Check(false, "pieces differ from brute force");
		}
		for (const auto& [s, n]: live) c.Remove(s, n);
		Check(c.BoundaryCount() == 0, "all removed: no boundaries left");
	}
}

} // namespace

int main() {
	TestBasics();
	TestGapsAndMerging();
	TestAgainstBruteForce();
	std::printf("PhysicalCoverageTests: ok\n");
	return 0;
}
