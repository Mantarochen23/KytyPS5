#include "FileTimeTestUtil.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/emulatorConfig.h"
#include "common/faultReport.h"
#include "common/file.h"
#include "common/frameCost.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/memAccount.h"
#include "common/ports.h"
#include "common/platform/sysDbg.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/cache/hotReadbackTable.h"
#include "graphics/host_gpu/renderer/cache/recentGpuWrites.h"
#include "libs/libs.h"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace {

namespace VirtualMemory = Common::VirtualMemory;

constexpr uint64_t PAGE = 0x10000;
constexpr uint64_t BASE = 0x0000'0600'0000'0000ull;

TEST(VirtualMemory, AllocFixedRefusesMappedPagesAndFreeReleasesOnlyItsRange) {
	ASSERT_TRUE(VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::ReadWrite));
	ASSERT_TRUE(VirtualMemory::AllocFixed(BASE + PAGE, PAGE, VirtualMemory::Mode::ReadWrite));

	EXPECT_FALSE(VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::ReadWrite));
	EXPECT_FALSE(VirtualMemory::AllocFixed(BASE - PAGE, 2 * PAGE, VirtualMemory::Mode::ReadWrite));

	ASSERT_TRUE(VirtualMemory::Free(BASE, PAGE));
	EXPECT_FALSE(VirtualMemory::AllocFixed(BASE + PAGE, PAGE, VirtualMemory::Mode::ReadWrite));
	ASSERT_TRUE(VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::ReadWrite));

	EXPECT_TRUE(VirtualMemory::Free(BASE, PAGE));
	EXPECT_TRUE(VirtualMemory::Free(BASE + PAGE, PAGE));
}

volatile uint64_t g_fault_rsp = 0;

// Resolves the write fault on BASE by making the page writable.
bool MakeBaseWritable(const Common::HostException::ExceptionInfo& info) {
	g_fault_rsp = info.rsp;
	return info.type == Common::HostException::ExceptionType::AccessViolation &&
	       info.access_violation_type == Common::HostException::AccessViolationType::Write &&
	       info.access_violation_vaddr == BASE &&
	       VirtualMemory::Protect(BASE, PAGE, VirtualMemory::Mode::ReadWrite);
}

bool RefuseFault(const Common::HostException::ExceptionInfo& /*info*/) {
	return false;
}

TEST(HostException, TheFirstHandlerStaysInstalledAndResolvesFaults) {
	EXPECT_EXIT(
	    {
		    namespace HostException = Common::HostException;
		    const bool installs     = HostException::InstallHandler(MakeBaseWritable) &&
			                          HostException::InstallHandler(MakeBaseWritable) &&
			                          !HostException::InstallHandler(RefuseFault);
		    const bool mapped = VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::Read);
		    auto*      value  = reinterpret_cast<volatile uint32_t*>(BASE);
		    *value            = 7;
		    // The faulting frame's stack pointer lies just below this local.
		    const auto local  = reinterpret_cast<uint64_t>(&value);
		    const bool rsp_ok = g_fault_rsp <= local && local - g_fault_rsp < 0x10000;
		    std::_Exit(installs && mapped && *value == 7 && rsp_ok ? 0 : 1);
	    },
	    testing::ExitedWithCode(0), "");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
volatile sig_atomic_t g_dispatched = 0;

// Refuses the fault if the guest signal it raises was delivered before the handler returned.
bool RaiseDispatchDuringFault(const Common::HostException::ExceptionInfo& info) {
	std::raise(Common::HostException::GuestSignalDispatchSignal());
	return g_dispatched == 0 && info.access_violation_vaddr == BASE &&
	       VirtualMemory::Protect(BASE, PAGE, VirtualMemory::Mode::ReadWrite);
}

TEST(HostException, GuestSignalWaitsUntilTheFaultHandlerReturns) {
	EXPECT_EXIT(
	    {
		    std::signal(Common::HostException::GuestSignalDispatchSignal(),
			            [](int) { g_dispatched = 1; });
		    const bool installs = Common::HostException::InstallHandler(RaiseDispatchDuringFault);
		    const bool mapped   = VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::Read);
		    *reinterpret_cast<volatile uint32_t*>(BASE) = 1;
		    std::_Exit(installs && mapped && g_dispatched == 1 ? 0 : 1);
	    },
	    testing::ExitedWithCode(0), "");
}
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
namespace {
int g_fault_pipe = -1;

size_t CountOf(const char* text, const char* needle) {
	size_t count = 0;
	for (const char* at = std::strstr(text, needle); at != nullptr; at = std::strstr(at + 1, needle)) {
		count++;
	}
	return count;
}

// Writes the host fault report to the pipe the parent reads, then ends the child.
bool ReportAndExit(const Common::HostException::ExceptionInfo& info) {
	Common::FaultReport::Registers regs {};
	regs.pc  = info.exception_address;
	regs.rsp = info.rsp;
	regs.rbp = info.rbp;
	Common::FaultReport::WriteHostContext(
	    regs, {[](void*, const char* text) { (void)!::write(g_fault_pipe, text, std::strlen(text)); },
	           nullptr});
	std::_Exit(0);
}
} // namespace

extern "C" __attribute__((noinline, used)) void KytyFaultReportTestNullWrite(volatile int* target) {
	*target = 1;
}

TEST(FaultReport, WorkerThreadFaultNamesModuleFunctionAndFrames) {
	int fds[2];
	ASSERT_EQ(::pipe(fds), 0);
	const pid_t child = ::fork();
	ASSERT_GE(child, 0);
	if (child == 0) {
		::close(fds[0]);
		g_fault_pipe = fds[1];
		Common::FaultReport::RefreshModules();
		Common::HostException::InstallHandler(ReportAndExit);
		std::thread worker([] {
			Common::FaultReport::SetThreadTag("fault test tag", "fault-test-wrk");
			Common::FaultReport::RegisterThreadStack();
			KytyFaultReportTestNullWrite(nullptr);
		});
		worker.join();
		std::_Exit(2);
	}
	::close(fds[1]);
	std::string text;
	char        buffer[4096];
	for (ssize_t n; (n = ::read(fds[0], buffer, sizeof(buffer))) > 0;) {
		text.append(buffer, static_cast<size_t>(n));
	}
	::close(fds[0]);
	int status = 0;
	::waitpid(child, &status, 0);
	const size_t frames = CountOf(text.c_str(), " bt[") + CountOf(text.c_str(), " fp[");
	EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "status " << status;
	EXPECT_NE(text.find("fault-test-wrk"), std::string::npos) << text;
	EXPECT_NE(text.find("doing: fault test tag"), std::string::npos) << text;
	EXPECT_NE(text.find("common_tests+0x"), std::string::npos) << text;
	EXPECT_NE(text.find("[KytyFaultReportTestNullWrite+0x"), std::string::npos) << text;
	EXPECT_GE(frames, 3u) << text;
}

TEST(FaultReport, ScopedActivityNamesShaderStageHashAndWords) {
	int fds[2];
	ASSERT_EQ(::pipe(fds), 0);
	const pid_t child = ::fork();
	ASSERT_GE(child, 0);
	if (child == 0) {
		::close(fds[0]);
		g_fault_pipe = fds[1];
		Common::FaultReport::RefreshModules();
		Common::HostException::InstallHandler(ReportAndExit);
		std::thread worker([] {
			Common::FaultReport::SetThreadTag("shader worker", "kyty-shdwk-0");
			Common::FaultReport::RegisterThreadStack();
			{
				Common::FaultReport::ScopedActivity transient("transient-step");
			}
			Common::FaultReport::ScopedActivity activity(
			    "pipeline-create stage=CS hash=0xaaaa000000000001 words=11545822");
			KytyFaultReportTestNullWrite(nullptr);
		});
		worker.join();
		std::_Exit(2);
	}
	::close(fds[1]);
	std::string text;
	char        buffer[4096];
	for (ssize_t n; (n = ::read(fds[0], buffer, sizeof(buffer))) > 0;) {
		text.append(buffer, static_cast<size_t>(n));
	}
	::close(fds[0]);
	int status = 0;
	::waitpid(child, &status, 0);
	EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "status " << status;
	EXPECT_NE(text.find("doing: shader worker [pipeline-create stage=CS hash=0xaaaa000000000001 "
	                    "words=11545822]"),
	          std::string::npos)
	    << text;
	EXPECT_EQ(text.find("transient-step"), std::string::npos) << text;
}
#endif

TEST(SysDbg, StackUsageReportsTheCallingThreadStackBase) {
	std::thread([] {
		const auto s = SysStackUsage();
		EXPECT_NE(s.reserved_addr, 0u);
		EXPECT_LT(s.reserved_addr, reinterpret_cast<uintptr_t>(&s));
	}).join();
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
constexpr int FATAL_EXIT_STATUS = 321;
#else
// POSIX keeps the low 8 bits of the exit status.
constexpr int FATAL_EXIT_STATUS = 321 & 0xff;
#endif

TEST(Assert, ExitMacrosEvaluateOnceAndExitWithTheFatalStatus) {
	int evaluations = 0;
	EXIT_IF(++evaluations == 0);
	EXIT_NOT_IMPLEMENTED(++evaluations == 0);
	EXPECT_EQ(evaluations, 2);

	EXPECT_EXIT(EXIT_IF(evaluations == 2), testing::ExitedWithCode(FATAL_EXIT_STATUS), "");
	EXPECT_EXIT(EXIT_NOT_IMPLEMENTED(true), testing::ExitedWithCode(FATAL_EXIT_STATUS), "");
	EXPECT_EXIT(EXIT("%d\n", evaluations), testing::ExitedWithCode(FATAL_EXIT_STATUS), "");
}

TEST(File, MissingFileStaysInvalidAndCreatedFileRoundTrips) {
	const auto path = std::filesystem::temp_directory_path() / "kyty_common_file_test.bin";
	std::filesystem::remove(path);

	Common::File missing;
	EXPECT_FALSE(missing.Open(path, Common::File::Mode::Read));
	EXPECT_TRUE(missing.IsInvalid());

	const uint32_t value = 0x12345678u;
	{
		Common::File out;
		ASSERT_TRUE(out.Create(path));
		out.Write(&value, sizeof(value));
		EXPECT_EQ(out.Size(), sizeof(value));
	}

	Common::File in;
	ASSERT_TRUE(in.Open(path, Common::File::Mode::Read));
	uint32_t read_value = 0;
	in.Read(&read_value, sizeof(read_value));
	EXPECT_EQ(read_value, value);
	EXPECT_TRUE(in.Seek(2));
	EXPECT_EQ(in.Tell(), 2u);
	in.Close();
	std::filesystem::remove(path);
}

TEST(File, NativeExistenceQueriesFollowTheEntryType) {
	const auto dir  = std::filesystem::temp_directory_path();
	const auto file = dir / "kyty_common_exists_test.bin";
	{
		Common::File out;
		ASSERT_TRUE(out.Create(file));
		const uint8_t byte = 1;
		out.Write(&byte, sizeof(byte));
	}

	EXPECT_TRUE(Common::File::IsDirectoryExisting(dir));
	EXPECT_TRUE(Common::File::IsDirectoryExisting(dir / ""));
	EXPECT_FALSE(Common::File::IsFileExisting(dir));
	EXPECT_EQ(Common::File::Size(dir), 0u);

	EXPECT_TRUE(Common::File::IsFileExisting(file));
	EXPECT_FALSE(Common::File::IsFileExisting(file / ""));
	EXPECT_FALSE(Common::File::IsDirectoryExisting(file));
	EXPECT_EQ(Common::File::Size(file), 1u);

	std::filesystem::remove(file);
	EXPECT_FALSE(Common::File::IsFileExisting(file));
}

TEST(File, GetTimesReportsExactNanoseconds) {
	const auto path = std::filesystem::temp_directory_path() / "kyty_common_time_test.bin";
	ASSERT_TRUE(Common::File().Create(path));
	// A multiple of 100 ns, the Windows file time unit.
	constexpr int64_t WRITE_NS = 1700000000123456700;
	SetUnixWriteTime(path, WRITE_NS);

	const auto times = Common::File::GetTimes(path);
	ASSERT_TRUE(times.has_value());
	EXPECT_EQ(times->write_ns, WRITE_NS);
	EXPECT_GT(times->access_ns, WRITE_NS);

	Common::File in(path, Common::File::Mode::Read);
	const auto   open_times = in.GetTimes();
	ASSERT_TRUE(open_times.has_value());
	EXPECT_EQ(open_times->write_ns, WRITE_NS);
	in.Close();
	std::filesystem::remove(path);
}

std::string ReadAll(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), {}};
}

TEST(Log, FileGetsPlainTextAndTheConsoleGetsEachLineOnce) {
	EXPECT_EXIT(
	    {
		    const auto dir =
		        std::filesystem::temp_directory_path() /
		        ("kyty_log_test_" +
				 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		    std::filesystem::create_directories(dir);
		    const auto console_file = dir / "console.txt";
		    std::freopen(console_file.string().c_str(), "wb", stdout);
		    Log::Write({}, "p");
		    const auto log_file = dir / "log.txt";
		    std::ofstream(log_file) << "previous run";
		    Config::Initialize();
		    Config::ConfigOptions options;
		    options.printf_direction   = Config::LogDirection::File;
		    options.printf_output_file = log_file;
		    Config::Load(options);
		    Log::Initialize();
		    Log::Write(Log::Color::Red, "a");
		    Log::WriteToConsoleAndLog("b");
		    Log::Flush();
		    const bool file_ok = ReadAll(log_file) == "ab";

		    options.printf_direction = Config::LogDirection::Console;
		    Config::Load(options);
		    Log::Initialize();
		    Log::Write(Log::Color::Red, "c");
		    Log::WriteToConsoleAndLog("d");
		    const bool console_ok = ReadAll(console_file) == "pb\x1b[31mc\x1b[0md";

		    options.printf_direction = Config::LogDirection::Silent;
		    Config::Load(options);
		    Log::Initialize();
		    const bool      silent_ok = Log::IsSilent();
		    std::error_code ec;
		    std::filesystem::remove_all(dir, ec);
		    std::_Exit(file_ok && console_ok && silent_ok ? 0 : 1);
	    },
	    testing::ExitedWithCode(0), "");
}

TEST(StringUtils, PathRoundTripsUtf8) {
	constexpr std::string_view Utf8 = "dir/\xc3\xa9-\xe6\x97\xa5.bin";
	EXPECT_EQ(Common::PathToString(Common::PathFromUtf8(Utf8)), Utf8);
	EXPECT_EQ(Common::PathToGenericString(Common::PathFromUtf8(Utf8)), Utf8);
}

TEST(Thread, MutexIsRecursiveAndCondVarTimesOutOrWakes) {
	Common::Mutex   mutex;
	Common::CondVar cond_var;
	mutex.Lock();
	mutex.Lock();
	mutex.Unlock();
	EXPECT_FALSE(cond_var.WaitFor(&mutex, 1000));

	bool        ready = false;
	std::thread signaller([&] {
		Common::LockGuard lock(mutex);
		ready = true;
		cond_var.Signal();
	});
	while (!ready) {
		cond_var.Wait(&mutex);
	}
	mutex.Unlock();
	signaller.join();
}

TEST(Thread, SleepsAtLeastTheRequestedTimeAndIdsAreUniquePerThread) {
	Common::InitializeThreads();
	EXPECT_TRUE(Common::Thread::IsMainThread());

	const auto start = std::chrono::steady_clock::now();
	Common::Thread::SleepMicro(2000);
	EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::microseconds(2000));

	const int id = Common::Thread::GetThreadIdUnique();
	EXPECT_EQ(Common::Thread::GetThreadIdUnique(), id);
	int  other_id   = id;
	bool other_main = true;
	std::thread([&] {
		other_id   = Common::Thread::GetThreadIdUnique();
		other_main = Common::Thread::IsMainThread();
	}).join();
	EXPECT_NE(other_id, id);
	EXPECT_FALSE(other_main);
}

#ifdef __linux__
unsigned ReadCpuValue(int cpu, const char* file) {
	unsigned value = 0;
	std::ifstream("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + file) >> value;
	return value;
}

TEST(Thread, ReservesTheFastestCoreAndRestoresTheProcessAffinity) {
	Common::InitializeThreads();
	int       processor = -1;
	cpu_set_t reserved {};
	std::thread([&] {
		processor = Common::Thread::ReserveCoreForCurrentThread();
		ASSERT_EQ(sched_getaffinity(0, sizeof(reserved), &reserved), 0);
	}).join();
	cpu_set_t others {};
	ASSERT_EQ(sched_getaffinity(0, sizeof(others), &others), 0);
	const auto rank = [](int cpu) {
		return std::tuple(ReadCpuValue(cpu, "acpi_cppc/highest_perf"),
		                  ReadCpuValue(cpu, "cpufreq/cpuinfo_max_freq"),
		                  ReadCpuValue(cpu, "topology/thread_siblings_list"));
	};
	if (processor < 0) {
		// No core is reserved only where fewer than eight cores report a performance or frequency.
		cpu_set_t cores {};
		for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
			if (CPU_ISSET(cpu, &others)) {
				const auto [performance, frequency, first] = rank(cpu);
				if (performance != 0 || frequency != 0) {
					CPU_SET(first, &cores);
				}
			}
		}
		EXPECT_LT(CPU_COUNT(&cores), 8);
		return;
	}
	EXPECT_TRUE(CPU_ISSET(processor, &reserved));
	for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
		if (CPU_ISSET(cpu, &reserved)) {
			EXPECT_EQ(std::get<2>(rank(cpu)), std::get<2>(rank(processor))) << cpu;
		}
		if (CPU_ISSET(cpu, &others)) {
			EXPECT_LT(rank(cpu), rank(processor)) << cpu;
		}
	}

	cpu_set_t restored {};
	std::thread([&] {
		Common::Thread::RestoreProcessAffinity();
		ASSERT_EQ(sched_getaffinity(0, sizeof(restored), &restored), 0);
	}).join();
	CPU_OR(&reserved, &reserved, &others);
	EXPECT_TRUE(CPU_EQUAL(&restored, &reserved));
}
#endif

TEST(Config, ParseTraceCallsOptionHandlesBareAndFiltered) {
	Config::ConfigOptions options;
	EXPECT_TRUE(Config::ParseTraceCallsOption("--trace-calls", options));
	EXPECT_TRUE(options.trace_calls_enabled);
	EXPECT_TRUE(options.trace_calls_filter.empty());

	EXPECT_TRUE(Config::ParseTraceCallsOption("--trace-calls=libA,libB", options));
	EXPECT_TRUE(options.trace_calls_enabled);
	ASSERT_EQ(options.trace_calls_filter.size(), 2u);
	EXPECT_EQ(options.trace_calls_filter[0], "libA");
	EXPECT_EQ(options.trace_calls_filter[1], "libB");

	// Bare --trace-calls clears a previous filter.
	EXPECT_TRUE(Config::ParseTraceCallsOption("--trace-calls", options));
	EXPECT_TRUE(options.trace_calls_filter.empty());

	EXPECT_FALSE(Config::ParseTraceCallsOption("--trace-calls=", options));
	EXPECT_FALSE(Config::ParseTraceCallsOption("--trace-calls=,,,", options));
	EXPECT_FALSE(Config::ParseTraceCallsOption("--other-flag", options));
}

TEST(Libs, PowerOfTwoRateLimitMatchesFirstFourAndPowersOfTwo) {
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(0));
	for (uint64_t n = 1; n <= 4; n++) {
		EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(n));
	}
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(5));
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(7));
	EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(8));
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(9));
	EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(16));
	EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(32));
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(33));
}

TEST(Libs, CallRingWrapsAroundAndIteratesOldestFirst) {
	Libs::CallRing ring;
	for (uint64_t i = 0; i < Libs::kCallRingSize + 10; i++) {
		const uint64_t args[6] = {i, i + 1, 0, 0, 0, 0};
		auto&          rec     = ring.Push("libkernel", "Func", args, 0x1000 + i);
		rec.result             = i * 2;
		rec.completed          = true;
	}
	EXPECT_EQ(ring.count, Libs::kCallRingSize);
	size_t seen = 0;
	ring.ForEachOldestFirst([&](const Libs::CallRecord& rec, size_t idx) {
		EXPECT_EQ(idx, seen);
		EXPECT_EQ(rec.args[0], 10u + seen);
		EXPECT_EQ(rec.result, (10u + seen) * 2u);
		EXPECT_EQ(rec.guest_ret_addr, 0x1000u + 10u + seen);
		seen++;
	});
	EXPECT_EQ(seen, Libs::kCallRingSize);
}

TEST(Config, ParseShaderHashListAndSkipLookup) {
	std::vector<uint64_t> hashes;
	EXPECT_TRUE(Config::ParseShaderHashList("", hashes));
	EXPECT_TRUE(hashes.empty());

	EXPECT_TRUE(Config::ParseShaderHashList(
	    "0x0123456789abcdef, FEDCBA9876543210; 42 # trailing comment\n0Xdeadbeef\n", hashes));
	ASSERT_EQ(hashes.size(), 4u);
	EXPECT_EQ(hashes[0], 0x0123456789abcdefull);
	EXPECT_EQ(hashes[1], 0xfedcba9876543210ull);
	EXPECT_EQ(hashes[2], 0x42ull);
	EXPECT_EQ(hashes[3], 0xdeadbeefull);

	const size_t before_invalid = hashes.size();
	EXPECT_FALSE(Config::ParseShaderHashList("0x1234,not_hex", hashes));
	EXPECT_FALSE(Config::ParseShaderHashList("0x1234567890abcdef0", hashes));
	EXPECT_FALSE(Config::ParseShaderHashList("0x", hashes));
	EXPECT_EQ(hashes.size(), before_invalid);

	Config::Initialize();
	Config::ConfigOptions opts;
	opts.skip_shader_hashes = {0xdeadbeefull, 0x0123456789abcdefull, 0xdeadbeefull};
	Config::Load(opts);
	if (const char* env = std::getenv("KYTY_SKIP_SHADERS"); env == nullptr || env[0] == '\0') {
		EXPECT_EQ(Config::GetSkipShaderHashes().size(), 2u);
	}
	EXPECT_TRUE(Config::ShouldSkipShader(0xdeadbeefull));
	EXPECT_TRUE(Config::ShouldSkipShader(0x0123456789abcdefull));
	EXPECT_FALSE(Config::ShouldSkipShader(0x1111ull));
	Config::Shutdown();
}

TEST(Config, ShaderCacheRetryFailedDefaultsOffAndTracksLoadedOption) {
	Config::Initialize();
	Config::ConfigOptions options;
	EXPECT_FALSE(options.shader_cache_retry_failed);
	Config::Load(options);
	EXPECT_FALSE(Config::ShaderCacheRetryFailed());

	options.shader_cache_retry_failed = true;
	Config::Load(options);
	EXPECT_TRUE(Config::ShaderCacheRetryFailed());
	Config::Shutdown();
}

TEST(Config, ShaderCacheKeepDefaultsOffAndTracksLoadedOption) {
	Config::Initialize();
	Config::ConfigOptions options;
	EXPECT_FALSE(options.shader_cache_keep_enabled);
	Config::Load(options);
	EXPECT_FALSE(Config::ShaderCacheKeepEnabled());

	options.shader_cache_keep_enabled = true;
	Config::Load(options);
	EXPECT_TRUE(Config::ShaderCacheKeepEnabled());
	Config::Shutdown();
}

TEST(Config, ShaderKeyIndexDefaultsOffAndTracksLoadedOption) {
	Config::Initialize();
	Config::ConfigOptions options;
	EXPECT_FALSE(options.shader_key_index_enabled);
	Config::Load(options);
	EXPECT_FALSE(Config::ShaderKeyIndexEnabled());

	options.shader_key_index_enabled = true;
	Config::Load(options);
	EXPECT_TRUE(Config::ShaderKeyIndexEnabled());
	Config::Shutdown();
}

TEST(Config, MaxSpirvWordsDefaultsOnAndTracksLoadedOption) {
	Config::Initialize();
	Config::ConfigOptions options;
	EXPECT_EQ(options.max_spirv_words, Config::ConfigOptions::DefaultMaxSpirvWords);
	EXPECT_EQ(options.max_spirv_words, 2'000'000u);
	Config::Load(options);
	EXPECT_EQ(Config::MaxSpirvWords(), 2'000'000u);

	options.max_spirv_words = 0u;
	Config::Load(options);
	EXPECT_EQ(Config::MaxSpirvWords(), 0u);

	options.max_spirv_words = 750'000u;
	Config::Load(options);
	EXPECT_EQ(Config::MaxSpirvWords(), 750'000u);
	Config::Shutdown();
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

TEST(Ports, DisablePortParsesNamesAndSelectsTheShaderStore) {
	std::vector<std::string> names;
	EXPECT_FALSE(Config::ParseDisabledPorts("no-such-port", names));
	names.clear();
	ASSERT_TRUE(Config::ParseDisabledPorts("wave64-lds-sync,,guest-lane-number", names));
	EXPECT_EQ(names, (std::vector<std::string> {"wave64-lds-sync", "guest-lane-number"}));
	Config::SetDisabledPorts({});
	EXPECT_TRUE(Config::PortEnabled(Config::Port::Wave64LdsSync));
	const auto all_on = Config::ShaderPortMask();
	EXPECT_NE(all_on, 0u);
	Config::SetDisabledPorts(names);
	EXPECT_FALSE(Config::PortEnabled(Config::Port::Wave64LdsSync));
	EXPECT_FALSE(Config::PortEnabled(Config::Port::GuestLaneNumber));
	EXPECT_TRUE(Config::PortEnabled(Config::Port::Wave64OverLimit32));
	EXPECT_NE(Config::ShaderPortMask(), all_on);
	EXPECT_NE(Config::PortsSummary().find("disabled: wave64-lds-sync,guest-lane-number"),
	          std::string::npos);
	names.clear();
	ASSERT_TRUE(Config::ParseDisabledPorts("all", names));
	Config::SetDisabledPorts(names);
	EXPECT_EQ(Config::ShaderPortMask(), 0u);
	EXPECT_NE(Config::PortsSummary().find("enabled: none"), std::string::npos);
	Config::SetDisabledPorts({});
}

namespace {

TEST(MemAccount, ProcStatusFieldsAreKilobytes) {
	const auto memory = Common::MemAccount::ParseProcStatus(
	    "Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\nVmSwap:\t     8 kB\nThreads:\t3\n");
	EXPECT_EQ(memory.rss, 1024u * 1024u);
	EXPECT_EQ(memory.peak_rss, 2048u * 1024u);
	EXPECT_EQ(memory.swap, 8u * 1024u);
	EXPECT_EQ(Common::MemAccount::ParseProcStatus("nothing here").rss, 0u);
}

TEST(MemAccount, GaugesAddAndSaturate) {
	auto& gauge = Common::MemAccount::Owner("test-owner-a");
	EXPECT_EQ(&gauge, &Common::MemAccount::Owner("test-owner-a"));
	gauge.Set(100, 2);
	gauge.Add(50, 1);
	EXPECT_EQ(gauge.Bytes(), 150u);
	EXPECT_EQ(gauge.Items(), 3u);
	gauge.Add(-1000, -10);
	EXPECT_EQ(gauge.Bytes(), 0u);
	EXPECT_EQ(gauge.Items(), 0u);
}

TEST(MemAccount, UnaccountedIsSigned) {
	EXPECT_EQ(Common::MemAccount::Unaccounted(10, 4), 6);
	EXPECT_EQ(Common::MemAccount::Unaccounted(4, 10), -6);
	const std::vector<Common::MemAccount::Entry> owners {{"a", 3, 0}, {"b", 4, 1}};
	EXPECT_EQ(Common::MemAccount::Counted(owners), 7u);
}

TEST(MemAccount, LineNamesEveryOwnerAndTheGap) {
	Common::MemAccount::Sample sample;
	sample.process = {.rss = 3ull << 30, .swap = 1ull << 30, .peak_rss = 4ull << 30};
	sample.owners  = {{"ir-table", 1ull << 30, 12}, {"store-queue", 0, 0}};
	sample.heap    = {.known = true, .in_use = 2ull << 30, .free_held = 512ull << 20};
	sample.external.emplace_back("vk", "device=100MiB");
	const auto line = Common::MemAccount::FormatLine(sample, "now");
	EXPECT_NE(line.find("[mem] now"), std::string::npos);
	EXPECT_NE(line.find("rss=3072MiB"), std::string::npos);
	EXPECT_NE(line.find("swap=1024MiB"), std::string::npos);
	EXPECT_NE(line.find("total=4096MiB"), std::string::npos);
	EXPECT_NE(line.find("peak-rss=4096MiB"), std::string::npos);
	EXPECT_NE(line.find("counted=1024MiB"), std::string::npos);
	EXPECT_NE(line.find("unaccounted=2048MiB"), std::string::npos);
	EXPECT_NE(line.find("ir-table=1024MiB(12)"), std::string::npos);
	EXPECT_NE(line.find("store-queue=0MiB"), std::string::npos);
	EXPECT_NE(line.find("heap-free-held=512MiB"), std::string::npos);
	EXPECT_NE(line.find("vk: device=100MiB"), std::string::npos);
	sample.owners = {{"big", 5ull << 30, 0}};
	EXPECT_NE(Common::MemAccount::FormatLine(sample, "x").find("unaccounted=-2048MiB"),
	          std::string::npos);
}

TEST(MemAccount, ProcStatusSplitsRss) {
	const auto memory = Common::MemAccount::ParseProcStatus(
	    "VmRSS:\t 100 kB\nRssAnon:\t 60 kB\nRssFile:\t 30 kB\nRssShmem:\t 10 kB\n");
	EXPECT_EQ(memory.rss_anon, 60u * 1024u);
	EXPECT_EQ(memory.rss_file, 30u * 1024u);
	EXPECT_EQ(memory.rss_shmem, 10u * 1024u);
}

TEST(MemAccount, MallocInfoGivesArenasAndSkipsTheTotals) {
	const char* xml =
	    "<malloc version=\"1\">\n<heap nr=\"0\">\n<sizes>\n<size from=\"17\" to=\"32\" total=\"32\" "
	    "count=\"1\"/>\n</sizes>\n<total type=\"fast\" count=\"2\" size=\"100\"/>\n<total type=\"rest\" "
	    "count=\"3\" size=\"900\"/>\n<system type=\"current\" size=\"4096\"/>\n<system type=\"max\" "
	    "size=\"8192\"/>\n</heap>\n<heap nr=\"3\">\n<total type=\"fast\" count=\"0\" size=\"0\"/>\n"
	    "<total type=\"rest\" count=\"1\" size=\"50\"/>\n<system type=\"current\" size=\"2048\"/>\n</heap>\n"
	    "<total type=\"fast\" count=\"2\" size=\"100\"/>\n<total type=\"mmap\" count=\"1\" size=\"7\"/>\n"
	    "</malloc>\n";
	const auto arenas = Common::MemAccount::ParseMallocInfo(xml);
	ASSERT_EQ(arenas.size(), 2u);
	EXPECT_EQ(arenas[0].nr, 0u);
	EXPECT_EQ(arenas[0].free_bytes, 1000u);
	EXPECT_EQ(arenas[0].system_bytes, 4096u);
	EXPECT_EQ(arenas[1].nr, 3u);
	EXPECT_EQ(arenas[1].free_bytes, 50u);
	EXPECT_EQ(arenas[1].system_bytes, 2048u);
	EXPECT_TRUE(Common::MemAccount::ParseMallocInfo("").empty());
}

TEST(MemAccount, ZramCountsSwapAtItsCompressedCost) {
	const auto zram = Common::MemAccount::ParseZramMmStat(
	    "8000 2000 2400 0 3000 5 0 0 0\n");
	EXPECT_TRUE(zram.known);
	EXPECT_EQ(zram.orig, 8000u);
	EXPECT_EQ(zram.compressed, 2400u);
	Common::MemAccount::ProcessMemory process;
	process.rss  = 1000;
	process.swap = 1000;
	EXPECT_EQ(Common::MemAccount::Footprint(process, zram), 1300u); // 1000 + 1000 * 2400/8000
	EXPECT_EQ(Common::MemAccount::Footprint(process, {}), 2000u);   // no zram: rss + swap
	EXPECT_FALSE(Common::MemAccount::ParseZramMmStat("garbage").known);
}

TEST(MemAccount, GrowthNamesTheBiggestHolders) {
	const std::vector<std::pair<std::string, uint64_t>> before {
	    {"heap-in-use", 100ull << 20}, {"rss-shmem(guest)", 50ull << 20}, {"ir-table", 10ull << 20}};
	const std::vector<std::pair<std::string, uint64_t>> after {
	    {"heap-in-use", 400ull << 20}, {"rss-shmem(guest)", 50ull << 20}, {"ir-table", 30ull << 20},
	    {"new-owner", 5ull << 20}};
	const auto text = Common::MemAccount::FormatGrowth(before, after, 2);
	EXPECT_EQ(text, "heap-in-use +300MiB, ir-table +20MiB");
	EXPECT_EQ(Common::MemAccount::FormatGrowth(after, before), "nothing");
}

TEST(MemAccount, LineShowsFootprintArenasAndGuest) {
	Common::MemAccount::Sample sample;
	sample.process = {.rss = 3ull << 30, .swap = 1ull << 30};
	sample.process.rss_shmem = 1ull << 30;
	sample.zram     = {.known = true, .orig = 4ull << 30, .compressed = 1ull << 30};
	sample.arenas   = {{0, 100ull << 20, 200ull << 20}, {1, 900ull << 20, 1000ull << 20}};
	sample.guest_committed = 16ull << 30;
	const auto line = Common::MemAccount::FormatLine(sample, "now");
	EXPECT_NE(line.find("footprint=3328MiB"), std::string::npos);
	EXPECT_NE(line.find("arenas=2 held=1000MiB top: #1=900MiB/1000MiB #0=100MiB/200MiB"),
	          std::string::npos);
	EXPECT_NE(line.find("guest: committed=16384MiB touched=1024MiB"), std::string::npos);
}

TEST(MemAccount, LimitGuardDropsThenStopsOnlyWhileOver) {
	using Action = Common::MemAccount::LimitGuard::Action;
	Common::MemAccount::LimitGuard off(0);
	EXPECT_EQ(off.Step(0, 1ull << 40), Action::None);
	Common::MemAccount::LimitGuard guard(100, 10'000);
	EXPECT_EQ(guard.Step(1000, 50), Action::None);
	EXPECT_EQ(guard.Step(2000, 150), Action::DropCaches);
	EXPECT_EQ(guard.Step(5000, 150), Action::None);
	EXPECT_EQ(guard.Step(11'999, 150), Action::None);
	EXPECT_EQ(guard.Step(12'000, 150), Action::Stop);
	// Back under: the next time over starts again with a drop.
	EXPECT_EQ(guard.Step(13'000, 90), Action::None);
	EXPECT_EQ(guard.Step(14'000, 150), Action::DropCaches);
	EXPECT_EQ(guard.Step(20'000, 100), Action::None);
}

TEST(FrameCost, RenderThreadAndReasonScopesNestAndRestore) {
	using namespace Common::FrameCost;
	EXPECT_FALSE(IsRenderThread());
	EXPECT_EQ(CurrentWaitReason(), WaitReason::Other);
	EXPECT_EQ(CurrentSubmitReason(), SubmitReason::Other);

	MarkRenderThread(true);
	EXPECT_TRUE(IsRenderThread());
	{
		ScopedWaitReason   outer_wait(WaitReason::Readback);
		ScopedSubmitReason outer_submit(SubmitReason::Slice);
		EXPECT_EQ(CurrentWaitReason(), WaitReason::Readback);
		EXPECT_EQ(CurrentSubmitReason(), SubmitReason::Slice);
		{
			ScopedWaitReason   inner_wait(WaitReason::Unmap);
			ScopedSubmitReason inner_submit(SubmitReason::Sync);
			EXPECT_EQ(CurrentWaitReason(), WaitReason::Unmap);
			EXPECT_EQ(CurrentSubmitReason(), SubmitReason::Sync);
		}
		EXPECT_EQ(CurrentWaitReason(), WaitReason::Readback);
		EXPECT_EQ(CurrentSubmitReason(), SubmitReason::Slice);
	}
	EXPECT_EQ(CurrentWaitReason(), WaitReason::Other);
	EXPECT_EQ(CurrentSubmitReason(), SubmitReason::Other);

	bool bg_render = true;
	auto bg_wait   = WaitReason::Readback;
	auto bg_submit = SubmitReason::Flip;
	std::thread([&] {
		bg_render = IsRenderThread();
		bg_wait   = CurrentWaitReason();
		bg_submit = CurrentSubmitReason();
	}).join();
	EXPECT_FALSE(bg_render);
	EXPECT_EQ(bg_wait, WaitReason::Other);
	EXPECT_EQ(bg_submit, SubmitReason::Other);

	MarkRenderThread(false);
	EXPECT_FALSE(IsRenderThread());
}

TEST(FrameCost, StepClockSamplesOnZeroAndRetriesWhenSkipped) {
	using Common::FrameCost::StepClock;
	uint32_t countdown = 2;
	uint64_t bucket    = 0;

	{
		StepClock c(countdown);
		EXPECT_FALSE(c.Active());
		c.AddTo(bucket);
		EXPECT_EQ(bucket, 0u);
		EXPECT_EQ(countdown, 1u);
	}
	{
		StepClock c(countdown);
		EXPECT_FALSE(c.Active());
		EXPECT_EQ(countdown, 0u);
	}
	// Skipped call: Active() is true, but Complete() is not called, so countdown stays 0.
	{
		StepClock c(countdown);
		EXPECT_TRUE(c.Active());
		EXPECT_EQ(countdown, 0u);
	}
	// Next call executes and completes: countdown resets to period - 1.
	{
		StepClock c(countdown);
		EXPECT_TRUE(c.Active());
		c.AddTo(bucket);
		c.Complete(8);
		EXPECT_FALSE(c.Active());
		EXPECT_EQ(countdown, 7u);
	}
}

TEST(FrameCost, FormatWindowSeparatesBackgroundGpuWaitsAndReportsArchFields) {
	using namespace Common::FrameCost;
	std::array<uint64_t, CounterCount> delta {};
	const auto set = [&](Counter c, uint64_t v) { delta[static_cast<size_t>(c)] = v; };

	set(Counter::Frames, 2);
	set(Counter::Draws, 200);
	set(Counter::Dispatches, 400);
	// 2 frames in 0.1 s => 50 ms/frame total.
	// Render thread: 10 ms/frame fence wait (6 ms readback + 4 ms unmap) + 2 ms/frame prio wait
	// => wait_gpu = 12.000 ms/frame, other = 38.000 ms/frame.
	// Background thread: 200 ms/frame wait_gpu_bg must NOT shrink other or inflate wait_gpu!
	set(Counter::GpuFenceWaitNs, 20'000'000);
	set(Counter::GpuPriorityOpWaitNs, 4'000'000);
	set(Counter::GpuFenceWaitBgNs, 400'000'000);
	set(Counter::WaitReadbackCalls, 6);
	set(Counter::WaitReadbackNs, 12'000'000);
	set(Counter::WaitUnmapCalls, 2);
	set(Counter::WaitUnmapNs, 8'000'000);
	set(Counter::WaitPriorityOpCalls, 4);
	set(Counter::SubmitsTotal, 10);
	set(Counter::SubmitSlice, 6);
	set(Counter::SubmitFlip, 2);
	set(Counter::SubmitSync, 2);
	set(Counter::DrawsDirect, 100);
	set(Counter::DrawsIndirectGpuPackets, 4);
	set(Counter::DrawsIndirectGpuSubdraws, 40);
	set(Counter::DrawsIndirectCpuPackets, 2);
	set(Counter::DrawsIndirectCpuSubdraws, 96);
	set(Counter::DrawsCoalesced, 12);
	set(Counter::DrawsSkipped, 8);
	set(Counter::DispatchesDirect, 50);
	set(Counter::DispatchesIndirect, 340);
	set(Counter::DispatchesIndirectThreadDim, 10);
	set(Counter::DispatchesGuard, 340);
	set(Counter::BvhDispatches, 16);
	set(Counter::BarriersAllCommands, 1200);
	set(Counter::DrawStepSamples, 4);
	set(Counter::DrawStepProgramsNs, 4000);
	set(Counter::DrawStepTargetsNs, 2000);
	set(Counter::DrawStepPipelineNs, 1200);
	set(Counter::DrawStepBindNs, 8000);
	set(Counter::DrawStepCommitNs, 6000);
	set(Counter::DrawStepRecordNs, 2800);
	set(Counter::DispatchStepSamples, 5);
	set(Counter::DispatchStepProgramNs, 5000);
	set(Counter::DispatchStepPipelineNs, 1500);
	set(Counter::DispatchStepBindNs, 10000);
	set(Counter::DispatchStepGuardNs, 7500);
	set(Counter::DispatchStepCommitNs, 6000);
	set(Counter::DispatchStepRecordNs, 9000);

	const auto lines = FormatWindow(delta, 0.1);
	EXPECT_NE(lines.summary.find("wait_gpu=12.000"), std::string::npos) << lines.summary;
	EXPECT_NE(lines.summary.find("other=38.000 total=50.000"), std::string::npos) << lines.summary;
	EXPECT_NE(lines.arch.find("[frame] arch wait_gpu_bg=200.000"), std::string::npos) << lines.arch;
	EXPECT_NE(lines.arch.find("wait_ms(rb=6.000 stream=0.000 unmap=4.000"), std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("prio=2.000 other=0.000)"), std::string::npos) << lines.arch;
	EXPECT_NE(lines.arch.find("wait_n(rb=3.00 stream=0.00 unmap=1.00"), std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("submits=5.00(slice=3.00 flip=1.00 susp=0.00 sync=1.00"),
	          std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("calls_draw(dir=50.00 ind_gpu=2.00/20.00 ind_cpu=1.00/48.00 "
	                          "ind_zero=0.00 coal=6.00 quad_ex=0.00 skip=4.00)"),
	          std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("calls_disp(dir=25.00 ind=170.00 ind_tdim=5.00 guard=170.00 "
	                          "oa_ex=0.00 skip=0.00 bvh=8.00)"),
	          std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("barriers_all=600.00"), std::string::npos) << lines.arch;
	EXPECT_NE(lines.arch.find("draw_ns(n=4 prog=1000 rt=500 pipe=300 bind=2000 commit=1500 rec=700)"),
	          std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("disp_ns(n=5 prog=1000 pipe=300 bind=2000 guard=1500 commit=1200 rec=1800)"),
	          std::string::npos)
	    << lines.arch;
}

TEST(FrameCost, RecordSubmitAndRenderWaitReasonAccumulateCounters) {
	using namespace Common::FrameCost;
	const auto before = SnapshotCounters();
	{
		ScopedWaitReason   wait_reason(WaitReason::StreamWrap);
		ScopedSubmitReason submit_reason(SubmitReason::Flip);
		RecordRenderWaitReason(12345);
		RecordSubmit();
	}
	const auto after = SnapshotCounters();
	EXPECT_EQ(after[static_cast<size_t>(Counter::WaitStreamCalls)] -
	              before[static_cast<size_t>(Counter::WaitStreamCalls)],
	          1u);
	EXPECT_EQ(after[static_cast<size_t>(Counter::WaitStreamNs)] -
	              before[static_cast<size_t>(Counter::WaitStreamNs)],
	          12345u);
	EXPECT_EQ(after[static_cast<size_t>(Counter::SubmitsTotal)] -
	              before[static_cast<size_t>(Counter::SubmitsTotal)],
	          1u);
	EXPECT_EQ(after[static_cast<size_t>(Counter::SubmitFlip)] -
	              before[static_cast<size_t>(Counter::SubmitFlip)],
	          1u);
}

TEST(FrameCost, ReadbackSiteScopesAndArchFormattingSplitProducers) {
	using namespace Common::FrameCost;
	EXPECT_EQ(CurrentReadbackSite(), ReadbackSite::Other);
	const auto before = SnapshotCounters();
	{
		ScopedWaitReason   wait_reason(WaitReason::Readback);
		ScopedReadbackSite outer(ReadbackSite::ThreadDim);
		EXPECT_EQ(CurrentReadbackSite(), ReadbackSite::ThreadDim);
		RecordRenderWaitReason(2'000'000);
		{
			ScopedReadbackSite inner(ReadbackSite::Scalar);
			EXPECT_EQ(CurrentReadbackSite(), ReadbackSite::Scalar);
			RecordRenderWaitReason(3'000'000);
		}
		EXPECT_EQ(CurrentReadbackSite(), ReadbackSite::ThreadDim);
		RecordRenderWaitReason(1'000'000);
	}
	EXPECT_EQ(CurrentReadbackSite(), ReadbackSite::Other);
	const auto after = SnapshotCounters();
	EXPECT_EQ(after[static_cast<size_t>(Counter::WaitRbThreadDimCalls)] -
	              before[static_cast<size_t>(Counter::WaitRbThreadDimCalls)],
	          2u);
	EXPECT_EQ(after[static_cast<size_t>(Counter::WaitRbThreadDimNs)] -
	              before[static_cast<size_t>(Counter::WaitRbThreadDimNs)],
	          3'000'000u);
	EXPECT_EQ(after[static_cast<size_t>(Counter::WaitRbScalarCalls)] -
	              before[static_cast<size_t>(Counter::WaitRbScalarCalls)],
	          1u);
	EXPECT_EQ(after[static_cast<size_t>(Counter::WaitRbScalarNs)] -
	              before[static_cast<size_t>(Counter::WaitRbScalarNs)],
	          3'000'000u);

	std::array<uint64_t, CounterCount> delta {};
	const auto set = [&](Counter c, uint64_t v) { delta[static_cast<size_t>(c)] = v; };
	set(Counter::Frames, 2);
	set(Counter::WaitRbThreadDimCalls, 4);
	set(Counter::WaitRbThreadDimNs, 6'000'000);
	set(Counter::WaitRbReadFaultCalls, 2);
	set(Counter::WaitRbReadFaultNs, 2'000'000);
	set(Counter::ReadbackInPlace, 10);
	set(Counter::ReadbackDetached, 6);
	set(Counter::ReadbackSync, 4);
	set(Counter::ReadbackBatchExtra, 8);
	set(Counter::DispatchIndirectThreadDimGpu, 12);
	set(Counter::DrawIndirectCountGpu, 14);
	const auto lines = FormatWindow(delta, 0.1);
	EXPECT_NE(lines.arch.find("rb_site(tdim=2.00/3.000 lds=0.00/0.000 scalar=0.00/0.000 "
	                          "heap=0.00/0.000 vattr=0.00/0.000 meta=0.00/0.000 "
	                          "rfault=1.00/1.000 wfault=0.00/0.000 inv=0.00/0.000 "
	                          "other=0.00/0.000)"),
	          std::string::npos)
	    << lines.arch;
	EXPECT_NE(lines.arch.find("rb_path(inplace=5.00 detached=3.00 sync=2.00 batch_ex=4.00 "
	                          "gpu_tdim=6.00 gpu_icount=7.00)"),
	          std::string::npos)
	    << lines.arch;
}

TEST(ReadbackBatchTables, HotTableEvictsOldestAndExpiresStaleTicks) {
	using namespace Libs::Graphics;
	HotReadbackTable table;
	for (uint64_t i = 0; i < MaxHotReadbackEntries; ++i) {
		table.Note(0x10000ull + i * 0x4000ull, 0x4000ull, i + 10u);
	}
	EXPECT_EQ(table.Size(), MaxHotReadbackEntries);
	// Refresh oldest entry so the second entry becomes oldest.
	table.Note(0x10000ull, 0x8000ull, 1000u);
	table.Note(0x9000000ull, 0x4000ull, 1001u);
	EXPECT_EQ(table.Size(), MaxHotReadbackEntries);
	uint64_t tick = 0;
	EXPECT_TRUE(table.TickOf(0x10000ull, tick));
	EXPECT_EQ(tick, 1000u);
	EXPECT_FALSE(table.TickOf(0x14000ull, tick));

	size_t kept = 0;
	table.ForEachHot(1001u + HotReadbackTicks, [&](uint64_t, uint64_t) { ++kept; });
	// Only entries at tick >= 1001 survive; tick 1000 is older than HotReadbackTicks.
	EXPECT_EQ(kept, 1u);
	EXPECT_EQ(table.Size(), 1u);
}

TEST(ReadbackBatchTables, RecentWriteRingDedupesNewestAndSplitsFinished) {
	using namespace Libs::Graphics;
	RecentGpuWriteRing ring;
	ring.Note(0x20000ull, 0u, 1u);
	ring.Note(0x20000ull, RecentWriteMaxBytes + 1u, 1u);
	EXPECT_EQ(ring.Size(), 0u);

	ring.Note(0x20100ull, 0x80ull, 5u);
	ring.Note(0x20100ull, 0x80ull, 7u); // same span refreshes tick without adding an entry
	EXPECT_EQ(ring.Size(), 1u);
	ring.Note(0x30000ull, 0x100ull, 12u);
	EXPECT_EQ(ring.Size(), 2u);

	std::vector<std::tuple<uint64_t, uint64_t, bool>> seen;
	ring.ForEachNewest(10u, [&](uint64_t addr, uint64_t size, bool finished) {
		seen.emplace_back(addr, size, finished);
		return true;
	});
	ASSERT_EQ(seen.size(), 2u);
	EXPECT_EQ(seen[0], std::make_tuple(0x30000ull, TRACKER_PAGE_SIZE, false));
	EXPECT_EQ(seen[1], std::make_tuple(0x20000ull, TRACKER_PAGE_SIZE, true));
}

} // namespace
