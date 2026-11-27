#include "graphics/host_gpu/renderer/pipeline/asyncPlan.h"
#include "graphics/host_gpu/renderer/pipeline/createStats.h"
#include "graphics/host_gpu/renderer/pipeline/renderActivity.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace Libs::Graphics;
using AsyncPlan::Action;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AsyncPlanTests: failed: %s\n", message);
		std::abort();
	}
}

// A model of one stage's life in PipelineCache::RequestAsync, driven by the same decisions.
struct Stage {
	struct Translation {
		uint64_t since = 0;
		uint64_t asked = 0;
		uint64_t bytes = 0;
	};
	bool                      has_program = false; // in the program cache (translated or loaded)
	std::map<int, Translation> translations;        // zero or one entry, key 0
	bool                      translation_job = false;
	bool                      module_job      = false;
	bool                      speculated      = false;
	int                       translation_jobs_posted = 0;
	int                       module_jobs_posted      = 0;

	// A draw or dispatch asks; returns what it did.
	Action Request(bool prefetch_only, uint64_t now) {
		const auto action = AsyncPlan::Decide({.has_program           = has_program,
		                                       .prefetch_only         = prefetch_only,
		                                       .translation_ready     = has_program && translations.contains(0),
		                                       .module_job_in_flight  = module_job,
		                                       .translation_in_flight = translation_job});
		switch (action) {
			case Action::PostModuleJob:
				translations.erase(0); // the job consumes the translation
				module_job = true;
				module_jobs_posted++;
				break;
			case Action::PostTranslation:
				translation_job = true;
				translation_jobs_posted++;
				speculated = AsyncPlan::MaySpeculate(1, true, has_program, true);
				break;
			default: break;
		}
		(void)now;
		return action;
	}
	// The translation job's result is applied.
	void FinishTranslation(uint64_t now) {
		translation_job = false;
		has_program     = true;
		if (!speculated) {
			translations[0] = {now, now, 0};
		}
		speculated = false;
	}
	void FinishModule() { module_job = false; }
};

// Cold start: translate, wait for the translation, then compile the module for the resources.
void TestColdRequest() {
	Stage stage;
	Check(stage.Request(false, 0) == Action::PostTranslation, "first request posts a translation");
	Check(stage.Request(false, 10) == Action::JoinTranslation, "second request joins it");
	Check(stage.Request(true, 10) == Action::JoinTranslation, "a prefetch joins it too");
	Check(stage.translation_jobs_posted == 1, "one translation job");
	stage.speculated = false; // a dispatch-named request: this model makes no speculative module
	stage.FinishTranslation(100);
	Check(stage.Request(false, 110) == Action::PostModuleJob, "the translation yields a module job");
	Check(stage.Request(false, 120) == Action::JoinModuleJob, "later requests join the module job");
	Check(stage.module_jobs_posted == 1, "one module job");
	stage.FinishModule();
}

// A program loaded from the store has modules but no translation. A dispatch whose resources match
// none of its modules must translate once, without speculating again, then build its module.
void TestLoadedFromStore() {
	Stage stage;
	stage.has_program = true;
	Check(stage.Request(true, 0) == Action::Nothing, "a prefetch for a known program does nothing");
	Check(stage.Request(false, 0) == Action::PostTranslation,
	      "a dispatch with an unserved key posts a translation");
	Check(!stage.speculated, "no speculative module for a program that is already known");
	stage.FinishTranslation(50);
	Check(stage.translations.contains(0), "the translation is kept for the dispatch");
	Check(stage.Request(false, 60) == Action::PostModuleJob, "and the module is built from it");
	stage.FinishModule();
}

// A speculative module that did not serve the dispatch that came back must not repeat: the second
// translation is kept for the dispatch's own key.
void TestSpeculationEndsAfterOneRound() {
	Check(AsyncPlan::MaySpeculate(1, true, false, true), "first request may speculate");
	Check(!AsyncPlan::MaySpeculate(1, true, true, true), "a known program may not");
	Check(!AsyncPlan::MaySpeculate(1, false, false, true), "graphics stages may not");
	Check(!AsyncPlan::MaySpeculate(0, true, false, true), "off means off");
	// A prefetch from registration has no user data: its module would be built from zeros.
	Check(!AsyncPlan::MaySpeculate(1, true, false, false), "no user data, no module by default");
	Check(AsyncPlan::MaySpeculate(2, true, false, false), "the old behaviour can be turned on");

	Stage stage;
	Check(stage.Request(false, 0) == Action::PostTranslation, "request");
	Check(stage.speculated, "the first translation speculates");
	stage.FinishTranslation(10); // module built and stored; no translation kept
	Check(!stage.translations.contains(0), "a speculating translation is consumed");
	// The dispatch comes back: its key was not the speculative one.
	Check(stage.Request(false, 20) == Action::PostTranslation, "the dispatch translates again");
	Check(!stage.speculated, "this time without a speculative module");
	stage.FinishTranslation(30);
	Check(stage.Request(false, 40) == Action::PostModuleJob, "so the dispatch gets its module job");
}

// Translations are kept for as long as the memory budget allows, however old; over it the least
// recently requested go first, and a request for a dropped stage translates it again.
void TestBudgetedExpiry() {
	constexpr uint64_t MiB = 1048576;
	std::map<int, Stage::Translation> table;
	table[0] = {1000, 1000, 100 * MiB};
	table[1] = {1000, 50000, 100 * MiB}; // asked later
	table[2] = {20000, 20000, 100 * MiB};
	std::vector<Stage::Translation> dropped;
	constexpr uint64_t Hours = 3'600'000;
	auto none = AsyncPlan::TakeOverBudget(table, 10 * Hours, 300 * MiB, dropped);
	Check(none.count == 0 && table.size() == 3 && none.kept_bytes == 300 * MiB,
	      "a table within the budget keeps everything, however old");
	none = AsyncPlan::TakeOverBudget(table, 10 * Hours, 0, dropped);
	Check(none.count == 0 && table.size() == 3, "budget 0 means no limit");

	auto first = AsyncPlan::TakeOverBudget(table, 60000, 250 * MiB, dropped);
	Check(first.count == 1 && dropped.size() == 1 && !table.contains(0),
	      "one entry goes: the least recently requested");
	Check(first.bytes == 100 * MiB && first.kept == 2 && first.kept_bytes == 200 * MiB,
	      "and the totals say what went and what stays");
	Check(first.newest_age_ms == 59000, "the age is that of the dropped entry's last request");

	// Asking for an entry refreshes it: the next to go is the one nobody asked for.
	table[2].asked = 55000;
	auto second = AsyncPlan::TakeOverBudget(table, 60000, 150 * MiB, dropped);
	Check(second.count == 1 && table.size() == 1 && table.contains(2),
	      "the entry asked for last stays, the older request goes");

	// One bigger than the budget on its own goes too, after everything less recent.
	table[3] = {60000, 60000, 400 * MiB};
	const auto third = AsyncPlan::TakeOverBudget(table, 61000, 300 * MiB, dropped);
	Check(third.count == 2 && table.empty() && third.bytes == 500 * MiB,
	      "entries go in order of last request until the rest fits");

	// The waiter: translation gone while its dispatch was blocked.
	Stage stage;
	stage.has_program = true;
	Check(stage.Request(false, 70000) == Action::PostTranslation,
	      "a dispatch whose translation was dropped posts a new one");
	Check(stage.Request(false, 70050) == Action::JoinTranslation, "and then waits for that one");
	stage.FinishTranslation(71000);
	Check(stage.Request(false, 71050) == Action::PostModuleJob, "which reaches the module job");
}

// A module job already running for the same resources is joined, never posted twice, whatever
// else is in flight.
void TestModuleJobDedup() {
	Stage stage;
	stage.has_program = true;
	stage.translations[0] = {0, 0, 0};
	Check(stage.Request(false, 1) == Action::PostModuleJob, "posts");
	stage.translations[0] = {0, 0, 0}; // another translation arrives for other resources
	Check(stage.Request(false, 2) == Action::JoinModuleJob, "joins while running");
	Check(stage.module_jobs_posted == 1, "posted once");
}

void TestActivity() {
	RenderActivity::Note("a", 7, "cs", "x");
	const auto first = RenderActivity::Read();
	Check(std::string(first.phase) == "a" && first.hash == 7, "phase is recorded");
	RenderActivity::Note("a", 7, "cs", "x");
	Check(RenderActivity::Read().since_ms == first.since_ms, "the same phase keeps its start time");
	{
		RenderActivity::Scope scope("b", 9, "vs", "y");
		Check(std::string(RenderActivity::Read().phase) == "b", "the scope sets its phase");
	}
	Check(std::string(RenderActivity::Read().phase) == "a" && RenderActivity::Read().hash == 7,
	      "the scope restores the previous phase");
	const auto packets = RenderActivity::Read().packets;
	RenderActivity::NotePacket(0x15);
	Check(RenderActivity::Read().opcode == 0x15 && RenderActivity::Read().packets == packets + 1,
	      "packets are counted");
}

// A registration without user data does not guess a ProgramKey and translate ahead, even when
// another ProgramKey of the same hash exists in the cache (that mismatch leaves an unclaimed
// waiting translation); lookahead with user data and scanned dispatches always translate.
void TestRegistrationTranslatesOnlyWhatTheCacheKnows() {
	using AsyncPlan::TranslateAhead;
	Check(!TranslateAhead(true, false, false), "a registration of an unknown hash is only kept");
	Check(!TranslateAhead(true, false, true),
	      "a registration without user data does not translate a mismatched key for a cached hash");
	Check(TranslateAhead(true, true, false), "a registration with user data translates");
	Check(TranslateAhead(false, false, false), "a scanned dispatch always translates");
	Check(TranslateAhead(false, true, true), "a scanned dispatch with data translates");
}

// Unclaimed waiting translations cannot linger forever: once idle past the TTL they are retired
// cleanly, while entries recently asked for stay alive and dropped entries re-translate on demand.
void TestUnclaimedWaitingTranslationsRetire() {
	constexpr uint64_t MiB = 1048576;
	std::map<int, Stage::Translation> table;
	table[0] = {1'000, 1'000, 2 * MiB};   // finished at 1s, never asked again
	table[1] = {1'000, 25'000, 3 * MiB};  // refreshed at 25s
	table[2] = {20'000, 20'000, 4 * MiB}; // finished at 20s
	std::vector<Stage::Translation> retired;

	const auto disabled = AsyncPlan::TakeExpired(table, 100'000, 0, retired);
	Check(disabled.count == 0 && table.size() == 3 && retired.empty(),
	      "TTL 0 disables time-based retirement");

	const auto first = AsyncPlan::TakeExpired(table, 35'000, AsyncPlan::DefaultWaitingTtlMs, retired);
	Check(first.count == 1 && retired.size() == 1 && !table.contains(0),
	      "entries idle >= TTL retire while fresher ones stay");
	Check(first.bytes == 2 * MiB && first.kept == 2 && first.kept_bytes == 7 * MiB &&
	          first.newest_age_ms == 34'000,
	      "retirement reports retired and remaining totals");

	const auto second =
	    AsyncPlan::TakeExpired(table, 60'000, AsyncPlan::DefaultWaitingTtlMs, retired);
	Check(second.count == 2 && table.empty() && second.kept == 0 && second.kept_bytes == 0 &&
	          second.newest_age_ms == 35'000,
	      "all unclaimed entries eventually retire to zero waiting");

	// A stage whose waiting translation retired still rebuilds cleanly if a dispatch later arrives.
	Stage stage;
	stage.has_program = true;
	Check(stage.Request(false, 70'000) == Action::PostTranslation,
	      "a retired translation re-translates on a later dispatch");
	stage.FinishTranslation(71'000);
	Check(stage.Request(false, 71'100) == Action::PostModuleJob,
	      "and builds its module once the new translation arrives");
}

// The pipeline-create line separates a warm driver cache (calls under 10 ms, cache hits) from a
// compile (seconds each), and shows how many calls were inside the driver at once.
void TestCreateStats() {
	namespace CS = Libs::Graphics::CreateStats;
	Check(CS::Bucket(9'999) == 0 && CS::Bucket(10'000) == 1 && CS::Bucket(999'999) == 2 &&
	      CS::Bucket(1'000'000) == 3 && CS::Bucket(10'000'000) == 4, "bucket edges");
	CS::Totals totals;
	CS::Enter(totals);
	CS::Enter(totals);
	CS::Leave(totals, totals.compute, 4'000, true, true, 3'000'000);
	CS::Leave(totals, totals.compute, 2'500'000, true, false, 2'400'000'000);
	CS::Enter(totals);
	CS::Leave(totals, totals.graphics, 500, false, false, 0);
	const auto line = CS::Line(totals);
	Check(line.find("cs n=2 wall=2504ms driver=2403ms max=2500ms cache-hit=1/2 <10ms=1 <100ms=0 "
	                "<1s=0 <10s=1 >=10s=0") != std::string::npos, "compute figures");
	Check(line.find("gfx n=1") != std::string::npos && line.find("cache-hit=0/0") != std::string::npos,
	      "graphics figures");
	Check(line.find("in-driver now=0 peak=2") != std::string::npos, "concurrency");
}

// --defer-new-targets-on-stall-ms: draws are let through only once one frame has lasted the limit,
// and the edges are reported once.
void TestFrameStallTracker() {
	using Change = AsyncPlan::FrameStallTracker::Change;
	AsyncPlan::FrameStallTracker t;
	Change c = Change::None;
	// Off: never, however long the frame lasts.
	Check(!t.Update(5, 1'000, 0, &c) && c == Change::None, "off at the first draw");
	Check(!t.Update(5, 100'000, 0, &c) && c == Change::None, "off stays off");

	AsyncPlan::FrameStallTracker s;
	// Frames advance: no stall.
	Check(!s.Update(1, 0, 2'000, &c) && c == Change::None, "first draw");
	Check(!s.Update(2, 1'500, 2'000, &c) && c == Change::None, "new frame");
	Check(!s.Update(2, 3'499, 2'000, &c) && c == Change::None, "just under the limit");
	// The same frame for 2 s: the stall starts once.
	Check(s.Update(2, 3'500, 2'000, &c) && c == Change::Started, "stall starts at the limit");
	Check(s.Update(2, 9'000, 2'000, &c) && c == Change::None, "stall goes on");
	Check(s.deferred == 2 && s.deferred_total == 2, "draws counted while stalled");
	// The next frame ends it and resets the clock.
	Check(!s.Update(3, 9'100, 2'000, &c) && c == Change::Ended, "stall ends with the next frame");
	Check(s.deferred == 2, "the ended stall keeps its count for the log line");
	Check(!s.Update(3, 11'000, 2'000, &c) && c == Change::None, "new frame clock");
	Check(s.Update(3, 11'100, 2'000, &c) && c == Change::Started, "second stall");
	Check(s.deferred == 1 && s.deferred_total == 3, "second stall counted afresh");
	// A clock that goes backwards does not count as a stall.
	AsyncPlan::FrameStallTracker b;
	Check(!b.Update(1, 5'000, 2'000, &c), "first");
	Check(!b.Update(1, 1'000, 2'000, &c) && c == Change::None, "backwards clock");
}

} // namespace

int main() {
	TestFrameStallTracker();
	TestCreateStats();
	TestRegistrationTranslatesOnlyWhatTheCacheKnows();
	TestUnclaimedWaitingTranslationsRetire();
	TestColdRequest();
	TestLoadedFromStore();
	TestSpeculationEndsAfterOneRound();
	TestBudgetedExpiry();
	TestModuleJobDedup();
	TestActivity();
	std::printf("AsyncPlanTests: ok\n");
	return 0;
}
