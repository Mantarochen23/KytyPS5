#include "graphics/host_gpu/renderer/pipeline/shaderWorkers.h"

#include "common/faultReport.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"

namespace Libs::Graphics {

ShaderWorkers::ShaderWorkers(uint32_t threads) {
	m_thread_count = threads;
	for (uint32_t i = 0; i < threads; i++) {
		m_threads.emplace_back([this, i](std::stop_token stop) {
			Common::Thread::LowerPriority();
			Common::FaultReport::SetThreadTag("shader worker", "shader-worker");
			std::unique_lock lock(m_mutex);
			// Once stop is requested the wait returns false only when no job may be taken.
			while (m_ready.wait(lock, stop, [this, i] { return CanPick(i); })) {
				const auto job = Pick(i);
				if (!job) {
					continue;
				}
				lock.unlock();
				{
					struct JobCounterGuard {
						ShaderWorkers& workers;
						~JobCounterGuard() {
							workers.m_busy.fetch_sub(1, std::memory_order_relaxed);
							workers.m_finished.fetch_add(1, std::memory_order_relaxed);
							NoteShaderWorkerJobsFinished(1);
						}
					} guard {*this};
					m_busy.fetch_add(1, std::memory_order_relaxed);
					job->run();
				}
				lock.lock();
			}
		});
	}
}

bool ShaderWorkers::CanPick(uint32_t) const {
	return !m_urgent_jobs.empty() || !m_jobs.empty();
}

ShaderWorkers::JobPtr ShaderWorkers::Pick(uint32_t index) {
	const auto now   = AskClockMs();
	const auto fresh = [now](const JobPtr& job) {
		return now - job->asked.load(std::memory_order_relaxed) < StaleAfterMs;
	};
	// Level one: jobs a draw or dispatch is blocked on, oldest first.
	{
		std::vector<JobPtr>* best_pool = nullptr;
		std::vector<JobPtr>::iterator best;
		for (auto* pool: {&m_urgent_jobs, &m_jobs}) {
			for (auto it = pool->begin(); it != pool->end(); ++it) {
				if (!(*it)->waited.load(std::memory_order_relaxed)) {
					continue;
				}
				if (best_pool == nullptr || (*it)->sequence < (*best)->sequence) {
					best_pool = pool;
					best      = it;
				}
			}
		}
		if (best_pool != nullptr) {
			auto job = std::move(*best);
			*best    = std::move(best_pool->back());
			best_pool->pop_back();
			return job;
		}
	}
	// Level two: urgent before normal, then by the draws' interest.
	for (auto* pool: {&m_urgent_jobs, &m_jobs}) {
		auto best = pool->end();
		for (auto it = pool->begin(); it != pool->end(); ++it) {
			if (best == pool->end()) {
				best = it;
				continue;
			}
			// Worker 0 takes the oldest. The others take a job a draw asked for lately before any
			// stale one, and among those the one asked for last.
			const bool better = index == 0
			    ? (*it)->sequence < (*best)->sequence
			    : (fresh(*it) != fresh(*best)
			           ? fresh(*it)
			           : (*it)->asked.load(std::memory_order_relaxed) >
			                 (*best)->asked.load(std::memory_order_relaxed));
			if (better) {
				best = it;
			}
		}
		if (best != pool->end()) {
			auto job = std::move(*best);
			// The order in the vector does not matter: the choice scans all of it.
			*best = std::move(pool->back());
			pool->pop_back();
			return job;
		}
	}
	return nullptr;
}

ShaderWorkers::JobPtr ShaderWorkers::Post(Common::UniqueFunction<void>&& run, Priority priority,
                                          bool waited) {
	auto job      = std::make_shared<Job>();
	job->waited.store(waited, std::memory_order_relaxed);
	job->run      = std::move(run);
	job->priority = priority;
	job->Touch();
	{
		std::scoped_lock lock(m_mutex);
		if (m_stopped) {
			return nullptr;
		}
		job->sequence = m_next_sequence++;
		(priority == Priority::Urgent ? m_urgent_jobs : m_jobs).push_back(job);
		NoteShaderWorkerJobsAdded(1);
	}
	m_ready.notify_one();
	return job;
}

void ShaderWorkers::MarkWaited(const JobPtr& job) {
	if (job) {
		job->Touch();
		job->waited.store(true, std::memory_order_relaxed);
	}
}

void ShaderWorkers::Promote(const JobPtr& job) {
	if (!job) {
		return;
	}
	MarkWaited(job);
	bool promoted = false;
	{
		std::scoped_lock lock(m_mutex);
		if (m_stopped || job->priority == Priority::Urgent) {
			return;
		}
		if (const auto it = std::ranges::find(m_jobs, job); it != m_jobs.end()) {
			job->priority = Priority::Urgent;
			m_urgent_jobs.push_back(std::move(*it));
			*it = std::move(m_jobs.back());
			m_jobs.pop_back();
			promoted = true;
		}
	}
	if (promoted) {
		m_ready.notify_one();
	}
}

size_t ShaderWorkers::Queued() {
	std::scoped_lock lock(m_mutex);
	return m_urgent_jobs.size() + m_jobs.size();
}

bool ShaderWorkers::Stopped() {
	std::scoped_lock lock(m_mutex);
	return m_stopped;
}

void ShaderWorkers::Stop() {
	uint32_t dropped = 0;
	{
		std::scoped_lock lock(m_mutex);
		m_stopped = true;
		dropped   = static_cast<uint32_t>(m_jobs.size() + m_urgent_jobs.size());
		m_jobs.clear();
		m_urgent_jobs.clear();
	}
	if (dropped != 0) {
		NoteShaderWorkerJobsFinished(dropped);
	}
	for (auto& thread: m_threads) {
		thread.request_stop();
	}
	// A running job finishes first.
	m_threads.clear();
}


} // namespace Libs::Graphics
