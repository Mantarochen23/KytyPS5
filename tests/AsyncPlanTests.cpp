#include "graphics/host_gpu/renderer/pipeline/asyncPlan.h"
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

// A registration translates only when the cache already knows the hash or the lookahead had the
// user data; the scan of a dispatch about to run always translates. A model of the registration
// of N shaders of which a program dispatches a few shows what stays in memory.
void TestRegistrationTranslatesOnlyWhatTheCacheKnows() {
	using AsyncPlan::TranslateAhead;
	Check(!TranslateAhead(true, false, false), "a registration of an unknown hash is only kept");
	Check(TranslateAhead(true, false, true), "a registration of a hash the cache has translates");
	Check(TranslateAhead(true, true, false), "a registration with user data translates");
	Check(TranslateAhead(false, false, false), "a scanned dispatch always translates");
	Check(TranslateAhead(false, true, true), "a scanned dispatch with data translates");
	size_t translated = 0;
	for (int hash = 0; hash < 3854; hash++) {
		const bool in_store = hash % 100 == 0;
		translated += TranslateAhead(true, false, in_store) ? 1u : 0u;
	}
	Check(translated == 39, "only the registrations of stored hashes are translated ahead");
}

} // namespace

int main() {
	TestRegistrationTranslatesOnlyWhatTheCacheKnows();
	TestColdRequest();
	TestLoadedFromStore();
	TestSpeculationEndsAfterOneRound();
	TestBudgetedExpiry();
	TestModuleJobDedup();
	TestActivity();
	std::printf("AsyncPlanTests: ok\n");
	return 0;
}
