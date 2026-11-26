#include "FileTimeTestUtil.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/ports.h"
#include "common/platform/sysDbg.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "libs/libs.h"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
	EXPECT_TRUE(Config::PortEnabled(Config::Port::WaveIAddReduce));
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
