#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

enum class PssrUpscaler { Linear, DLSS, FSR };

// Values are the speaker types games read.
enum class AudioOutput { TV, AVReceiver, SoundBar, Headphones };

using Keymap = std::vector<std::string>;
using ControllerColor = std::array<uint8_t, 3>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr int     MAX_PLAYERS         = 4;
constexpr int32_t PLAYER_USER_ID_STEP = 0x100;
constexpr int32_t MAX_USER_ID =
    std::numeric_limits<int32_t>::max() - (MAX_PLAYERS - 1) * PLAYER_USER_ID_STEP;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id <= MAX_USER_ID && user_id != USER_ID_EVERYONE &&
	       user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	AudioOutput            audio_output                = AudioOutput::TV;
	bool                   audio_3d_enabled            = false;
	std::optional<ControllerColor> controller_color;
	uint32_t               controller_speaker_volume      = 50;
	uint32_t               controller_vibration_intensity = 100;
	// 0 shares player 1 with the first gamepad; 1..MAX_PLAYERS dedicates that player to the keyboard.
	uint32_t               keyboard_player                = 0;
	PresentMode            present_mode                = PresentMode::Mailbox;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   hdr_enabled                 = false;
	bool                   hide_cursor_enabled         = false;
	bool                   stretch_enabled             = false;
	bool                   splash_enabled              = true;
	bool                   vr_enabled                  = false;
	bool                   pro_enabled                 = false;
	PssrUpscaler           pssr_upscaler               = PssrUpscaler::Linear;
	std::filesystem::path  dlss_folder;
	uint32_t               dlss_preset                 = 0; // 0 is the model default
	std::filesystem::path  fsr_folder;
	bool                   amd_cpu_enabled             = false;
	bool                   guest_tsc_enabled           = false;
	bool                   network_connected           = false;
	bool                   psn_emulation_enabled       = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	bool                   keep_going_enabled          = false;
	bool                   trace_calls_enabled         = false;
	std::vector<std::string> trace_calls_filter;
	int32_t                auto_cross_seconds          = 0;
	std::vector<uint64_t>  skip_shader_hashes;
	bool                   shader_cache_enabled        = false;
	bool                   shader_cache_keep_enabled   = false;
	bool                   shader_cache_retry_failed   = false;
	bool                   async_shader_compile_enabled = true;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   readback_tiled_images       = false;
	bool                   readback_storage_images     = false;
	bool                   sync_raw_image_buffers      = false;
	uint32_t               flush_draws                 = 0; // 0 is off
	bool                   tessellation_enabled        = false;
	bool                   occlusion_queries_enabled   = false;
	bool                   relaxed_shader_float        = false;
	bool                   native_rt                   = false;
	bool                   lod_stats_enabled           = false;
	bool                   trophy_enabled              = true;
	uint32_t               trophy_duration_seconds     = 6;
	bool                   perf_overlay_enabled            = false;
	bool                   playgo_hack_enabled         = false;
	bool                   skip_notice_screen          = false;
	bool                   gpu_fault_query_enabled     = false;
	std::string            launch_activity;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
	// Ordinary boots need a non-null argv[0]; --guest-exec supplies the caller's arguments.
	std::vector<std::string> guest_arguments {""};
	bool                     bda_table_validate_enabled = false;
	// Seconds to wait for the driver's own hang dump after a device loss (RADV_DEBUG=hang); 0 = off.
	uint32_t                 hang_dump_wait_seconds     = 60;
	// Shader hashes whose dispatches get the --buffer-probe comparison of guest memory against the
	// host buffer contents (sorted, unique). Empty = off.
	std::vector<uint64_t>    buffer_probe_hashes;
	// --loop-guard: evaluated loop exits per invocation before every loop of a shader is left and
	// reported. 0 = off.
	uint32_t                 loop_guard_limit           = 0;
	// --disable-port: names from common/ports.h whose ported behaviour is turned off.
	std::vector<std::string> disabled_ports;
	bool                     shader_key_index_enabled   = false;
	// --async-pump-budget-ms: render-thread time per PumpAsync call for applying finished
	// shader jobs; 0 = unlimited.
	uint32_t                 async_pump_budget_ms       = 2;
	// --async-workers: shader worker threads; 0 = hardware threads minus four, at least two.
	uint32_t                 async_workers              = 0;
	// --async-speculative-modules: a finished compute translation whose resources need no guest
	// memory also gets its module on the worker (and so into the shader store). 0 = off.
	uint32_t                 async_speculative_modules  = 1;
	// --render-stall-report-s: when no frame was recorded for this many seconds, print one line
	// with what the render thread is doing, then again every that many seconds. 0 = off.
	uint32_t                 render_stall_report_s      = 5;
	// --async-ir-budget-mb: the memory finished shader translations that no draw has asked for
	// yet may hold before the least recently requested are dropped. 0 = no limit.
	uint32_t                 async_ir_budget_mb         = 2048;
	// --guest-fault-park: a guest thread that takes a fault nobody resolves is reported and parked
	// instead of ending the process.
	bool                     guest_fault_park           = false;
	// --guest-fault-report-only: on that fault, also dump guest code around the pc and the return
	// addresses into the run's log folder.
	bool                     guest_fault_report_only    = false;
	// --guest-clock-hold-on-stall <ms>: the guest clock stands still while no frame was recorded for
	// more than this long and the render side is busy with shaders. 0 = off.
	uint32_t                 guest_clock_hold_ms        = 0;
	// --dump-registered-shaders: copy each distinct registered shader and an index record here.
	// Empty = off.
	std::filesystem::path    dump_registered_shaders_folder;
	// --async-compute-wait-ms: a compute dispatch whose shader is not ready suspends its queue at
	// most this long, then is skipped for this use while the build goes on. 0 = wait for ever.
	uint32_t                 async_compute_wait_ms      = 0;
	// --async-lookahead: the scan of submitted commands also reads the compute user data, so that
	// shaders whose resources follow from it are built before their dispatch is reached.
	uint32_t                 async_lookahead            = 1;
	// --warmup-budget-s: the title starts after this many seconds of loading the shader store; the
	// rest loads in the background. 0 = load everything first.
	uint32_t                 warmup_budget_s            = 0;
	// --spirv-compact: run the measured conservative post-emission cleanup recipe. Off by default.
	bool                     spirv_compact_enabled = false;
	// --mem-limit-gb: when the process's RSS + swap passes this, caches are dropped and the memory
	// table printed; still over 10 s later, the run stops with a message. 0 = off.
	uint32_t                 mem_limit_gb               = 0;
};

void Load(const ConfigOptions& cfg);
bool ParseTraceCallsOption(std::string_view arg, ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
AudioOutput GetAudioOutput();
bool Audio3dEnabled();
const std::optional<ControllerColor>& GetControllerColor();
uint32_t GetControllerSpeakerVolume();
uint32_t GetControllerVibrationIntensity();
uint32_t GetKeyboardPlayer();
// Player 0 is the configured user; later players derive their IDs from it.
int32_t  GetPlayerUserId(int player);
// Returns -1 when the user ID does not belong to any player.
int      GetPlayerFromUserId(int32_t user_id);
// Players after the first are guests: they have no profile, trophies or NP account.
bool     IsGuestUserId(int32_t user_id);
PresentMode GetPresentMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     HdrEnabled();
bool     HideCursorEnabled();
bool     StretchEnabled();
bool     VrEnabled();
bool     Ps5ProEnabled();
PssrUpscaler GetPssrUpscaler();
const std::filesystem::path& GetDlssFolder();
uint32_t GetDlssPreset();
const std::filesystem::path& GetFsrFolder();
bool     AmdCpuEnabled();
bool     GuestTscEnabled();
bool     IsNetworkConnected();
bool     PsnEmulationEnabled();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
bool                   KeepGoingEnabled();
bool                   TraceCallsEnabled();
const std::vector<std::string>& TraceCallsFilter();
int32_t                AutoCrossSeconds();
bool                   ParseShaderHashList(std::string_view text, std::vector<uint64_t>& out);
bool                   ShouldSkipShader(uint64_t hash);
const std::vector<uint64_t>& GetSkipShaderHashes();
bool                   ShaderCacheEnabled();
bool                   ShaderCacheKeepEnabled();
bool                   ShaderCacheRetryFailed();
bool                   AsyncShaderCompileEnabled();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool ReadbackTiledImagesEnabled();
bool ReadbackStorageImagesEnabled();
bool SyncRawImageBuffersEnabled();
uint32_t GetFlushDraws();
bool TessellationEnabled();
bool OcclusionQueriesEnabled();
bool RelaxedShaderFloatEnabled();
bool NativeRtEnabled();
bool LodStatsEnabled();
bool TrophyEnabled();
uint32_t GetTrophyDurationSeconds();
bool PlayGoHackEnabled();
bool SkipNoticeScreen();
bool GpuFaultQueryEnabled();
// Activity ID of the one-shot launchActivity game intent; empty sends none.
const std::string& GetLaunchActivity();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();
const std::vector<std::string>& GetGuestArguments();
bool BdaTableValidateEnabled();
uint32_t HangDumpWaitSeconds();
bool     BufferProbeShader(uint64_t hash);
bool     BufferProbeEnabled();
const std::vector<uint64_t>& BufferProbeHashes();
uint32_t LoopGuardLimit();
bool ShaderKeyIndexEnabled();
uint32_t AsyncPumpBudgetMs();
uint32_t AsyncWorkers();
bool     AsyncSpeculativeModules();
uint32_t RenderStallReportSeconds();
uint32_t AsyncIrBudgetMb();
bool     GuestFaultParkEnabled();
bool     GuestFaultReportOnlyEnabled();
uint32_t GuestClockHoldMs();
const std::filesystem::path& GetRegisteredShaderDumpFolder();
uint32_t AsyncComputeWaitMs();
bool     AsyncLookaheadEnabled();
uint32_t AsyncSpeculativeModulesMode();
uint32_t WarmUpBudgetSeconds();
bool SpirvCompactEnabled();
uint32_t MemLimitGb();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
