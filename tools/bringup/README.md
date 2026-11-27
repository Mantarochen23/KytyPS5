# Bring-up run and triage

## What this is

This fork brings one program up on Linux with an AMD RDNA4 GPU and RADV. It is unfinished.
A run usually stops in a new place. The point of a run is to learn where and why.

`run.sh` starts one run in a closed sandbox and writes a log.
`triage.py` turns that log into a short summary.
Send the summary, not the log and not the program files.

## Build on Linux

Use a build with symbols. `Release` strips them and a backtrace becomes bare addresses.

    cmake -S . -B _Build/linux-debug -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKYTY_BUILD_LAUNCHER=OFF
    cmake --build _Build/linux-debug --target kyty_emulator --parallel 12

Test binaries are not built by default. Build one by name, for example:

    cmake --build _Build/linux-debug --target common_tests --parallel 12
    cmake --build _Build/linux-debug --target shader_corpus_tool --parallel 12
    python3 -m unittest discover -s tools/bringup/tests

`run.sh` uses `_Build/linux-debug` unless `KYTY_BUILD` says otherwise.

## One run

    tools/bringup/run.sh <label>
    KYTY_SECONDS=600 KYTY_PAD=host tools/bringup/run.sh <label> -- --loop-guard 0

The first argument is a short label. Anything after `--` goes to the emulator.
The run stops after 180 seconds by default.
The sandbox has no network but loopback, no host user folders, and no controller unless asked for.
It uses the Wayland socket and the GPU nodes. It binds the state and run folders as writable.
If another sandboxed run holds the GPU, `run.sh` refuses to start.
Machine-local settings are read from `<repo>/../bringup-local.env` when that file exists.

Every variable is optional except `KYTY_GAME`:

| Variable | What it does |
|---|---|
| `KYTY_GAME` | Folder with the program to run. Required. |
| `KYTY_BUILD` | Build folder. Default `<repo>/_Build/linux-debug`. |
| `KYTY_RUNS` | Parent folder for run folders. Default `<repo>/../runs`. |
| `KYTY_STATE` | Persistent state: stores, saves, shader failures. Default `<repo>/../state`. |
| `KYTY_SECONDS` | Stop after this many seconds. Default 180. `0` runs until the program exits. |
| `KYTY_PAD` | `host` passes controllers in. Default `none`; a drifting stick can write thousands of log lines a second. |
| `KYTY_PAD_VENDORS` | USB vendor ids allowed through when `KYTY_PAD=host`. |
| `KYTY_KEEP_GOING` | `0` turns off `--keep-going`; the first shader failure is then fatal. Default `1`. |
| `KYTY_FATAL_ABORT` | `1` aborts on a fatal error so a core is kept. Large. |
| `KYTY_AUDIO` | `none` disables sound. Default `host`, which passes the sound server socket in. |
| `KYTY_AUTO_CROSS` | Press a button for 0.15 s every 2 s for this many seconds from the first pad read. |
| `KYTY_PROFILE_AT` | Start `perf` at this second, or at `stall` when frames stop. |
| `KYTY_PROFILE_SECONDS` | Length of that profile. Default 20. |
| `KYTY_SHADER_CACHE` | `0` turns off `--shader-cache`. Default `1`, so later runs start from stored shaders. |
| `KYTY_TRACE` | `0` turns off `--trace-calls`. Default `1`. |
| `KYTY_UMR` | `1` dumps GPU rings and waves when frames stop. Needs root access. |
| `KYTY_SKIP_SHADERS` | Comma-separated shader hashes to skip, for bisecting a hang. |
| `RADV_DEBUG` | Driver flags. With `hang`, driver dump folders are kept in the run folder. |
| `KYTY_FORCE` | `1` starts even if another sandboxed run is live. |
| `XDG_RUNTIME_DIR` | Runtime folder that holds the Wayland socket. |
| `WAYLAND_DISPLAY` | Wayland socket name. |

Each run lands in `<KYTY_RUNS>/<UTC>-<label>/`:

- `run.txt`: label, times, commit, options, emulator hash, kernel and driver versions, exit code.
  Exit code 124 means the time limit stopped it. 65 means a fatal error.
- `kyty.log`: the full log, with `@@T <seconds>` markers added once per second.
- `triage.txt` and `triage.json`: the summary described below.
- `ShaderFailures/`: a copy of the accumulated failure ledger, when it exists.
- `kernel-gpu.txt` and `gpu-devcoredump.txt`: kernel messages, when GPU timeouts were seen.
- `radv/`, `perf.*`, `umr-*.txt`: only when those options were used.

## Debug options

By default `run.sh` adds `--printf-direction Console`, `--keep-going`, `--trace-calls` and
`--shader-cache`. `KYTY_KEEP_GOING=0`, `KYTY_TRACE=0` and `KYTY_SHADER_CACHE=0` remove them.
All the options below exist in `kyty_emulator --help`. They are off unless stated.

- `--auto-cross <seconds>`: press a button at intervals for unattended runs.
- `--trace-calls[=<lib[,lib]>]`: log library calls, or every call with arguments for named libraries.
- `--keep-going`: skip draws and dispatches whose shader fails, record them, keep running.
- `--skip-shaders <hashes|@file>`: skip draws and dispatches for listed hashes; repeatable.
- `--gpu-fault-query`: query the driver for fault details after a device loss.
- `--bda-table-validate`: check every page-table write against live buffers.
- `--hang-dump-wait <seconds>`: wait this long for the driver hang dump after a loss. Default 60.
- `--buffer-probe <hashes>`: compare guest buffer bytes with the host copy before each dispatch.
- `--loop-guard <n>`: after n loop exits an invocation leaves its loops and the loop is logged.
  Default: compute shaders only, n = 1048576. `0` is off.
- `--guest-fault-park`: park, instead of ending the run, a guest thread with an unresolved fault.
- `--guest-fault-report-only`: write guest code bytes around such a fault into the run folder.
- `--guest-clock-hold-on-stall <ms>`: hold the guest clock during a shader stall. Default 0.
- `--dump-registered-shaders <dir>`: copy distinct shaders and their metadata out through a bounded queue.
- `--async-compute-wait-ms <n>`: cap how long a dispatch waits for its shader. `0` waits for ever.
- `--shader-cache`: keep shaders and pipelines between runs.
- `--shader-cache-keep`: keep usable records from a mismatched store.
- `--shader-cache-retry-failed`: retry shaders that were cached as failures.
- `--no-async-shader-compile`: build shaders on the render thread instead of workers.
- `--vulkan-validation`: turn on Vulkan validation layers.
- `--gpu-assisted-validation`: bounds-check shader accesses on the GPU. Very slow; implies validation.
- `--shader-validation`: turn on shader validation.
- `--command-buffer-dump`: dump command buffers; `--command-buffer-dump-folder <path>` sets the folder.
- `--graphics-debug-dump`: dump graphics debug data, including shaders and IR.
- `--shader-log-direction <value>`: `Silent`, `Console` or `File`; `--shader-log-folder <path>` sets the folder.
- `--printf-direction <value>`: `Silent`, `Console` or `File` for guest prints.
- `--printf-output-file <path>`: where guest prints go when the direction is `File`.
- `--profile`: start the Tracy profiler.
- `--rd`: open a RenderDoc capture.
- `--perf-overlay`: show the on-screen performance panel.
- `--spirv-debug-printf`: enable debug prints from emitted modules.
- `--spirv-compact`: run conservative cleanup after module emission.
- `--async-pump-budget-ms <ms>`: render-thread time per step spent applying finished jobs. Default 2.
- `--async-workers <n>`: shader worker threads. Default 0, meaning cores minus four, at least two.
- `--async-speculative-modules <0|1|2>`: build modules for unrequested translations. Default 1.
- `--render-stall-report-s <n>`: report what the render thread does after n seconds without a frame. Default 5.
- `--async-ir-budget-mb <n>`: cap memory held by translations nobody has used. Default 2048.
- `--async-lookahead <0|1>`: read compute user data ahead of the dispatch. Default 1.
- `--warmup-budget-s <n>`: start the program after n seconds of store loading. Default 0.
- `--shader-key-index`: persist a hash to key index in the store. Default off.
- `--disable-port <names>`: turn off named ported fixes, or all of them.
- `--mem-limit-gb <n>`: print the memory table and drop caches above n GB; stop if still over after 10 s.
- `--heap-tune <n>`: cap allocator arenas, trim freed memory and use mapped blocks for large ones.
- `--driver-cache-cap-mb <n>`: ignore driver cache files larger than n MB.
- `--fault-storm-faults <n>`: stop watching a page after n faults in the window. `0` is off.
- `--fault-storm-frames <n>`: window for the fault storm rule. Default 8.
- `--fault-storm-hold <n>`: frames a marked page stays unwatched. Default 60.
- `--max-spirv-words <n>`: refuse modules larger than n words. Default 2000000.
- `--materialize-memo-entries <n>`: remembered resource plans per shader. Default 1.
- `--materialize-memo-stamps <n>`: watch pages read by a remembered plan. `0` is off.

## Log markers

`@@T` lines carry the seconds since the start. Every other line below comes from the code.
Read them in the log or in the triage sections that quote them.

- `[frame]`: five-second window of per-frame averages. `frames` shows progress; `wait_gpu`,
  `wait_guest` and `materialize` show where time goes. The `arch` line splits waits and call counts.
  `WARNING module-misses` means lookups fail after the store load.
- `[fault]`: five-second rates for write watches. Read `handler` and `lock-wait` as cost, and
  `top 64K blocks` as the hot pages. A high handler cost means the watch path is the problem.
- `[mem]`: process RSS, swap, peak and the sum counted by each owner. `unaccounted` is the part no
  owner explains. `--mem-limit-gb` prints the table, then stops if the total stays high.
- `[pipeline-create]`: driver pipeline builds. Read `max` and the `>=10s` bucket for slow builds;
  `cache-hit` shows how many needed no driver build.
- `[store]`: store load, per-miss lines, `lookups`/`hit`/`distinct-miss`, and `written`/`failed`.
  Many distinct misses mean stored work is not being reused.
- `[dispatch]`: a compute dispatch that was limited or refused. It gives the hash, arguments, group
  counts and action. The `origin` and submitter lines show where the command came from.
- `[loop-guard]`: an invocation used up its loop-exit budget. It names the shader, the loop and the
  number of invocations that tripped it.
- `[stall]`: no frame for a while. It names the render phase, shader and pending jobs, and says how
  many guest threads are parked. The verdict's longest stall comes from here.
- `[wait]`: a command queue is idle, running, suspended on a shader, or waiting on guest memory.
  The memory form gives the address, compared value and poll count.
- `[build]`: one shader build with translate, plan, emit, module, pipeline and wait times, module
  size and `repeat=#`. `WARNING` names a module that grew too large.
- `[park]`: a guest thread faulted again and was parked; the short line counts parked threads.
- `[shader-reg]`: the registration ledger at exit. It lists registered and used shaders per stage and
  the minimum lead time before first use.
- `[prefetch]`: registration and prefetch counts, lookahead modules built against used, and skipped
  dispatches. Low use means prefetch work was wasted.
- `[buffer-probe]`: bytes that differ between guest memory and the copy the GPU reads. The summary
  counts probed dispatches and mismatches.
- `[module-index] BUG`: a module in a program was not found in the store. Expect repeated translations.
- `[async]`: a budget was exceeded and unrequested translations were dropped.
- `[warmup]`: how long store loading took and when the program was allowed to start.
- `[bda-table-validate]`: a page-table invariant failed. It is fatal when the option is on.
- `[bda-miss]`: an address was not found among live buffers.
- `[clock-hold]`: the guest clock was held or restarted during a shader stall.
- `[sync-build]`: the render thread built or linked a shader stage, with the reason it could not wait.
- `[skip]`: a dispatch was skipped because its shader was late or its inputs tainted.
- `[fault-code]`: guest code bytes around a fault were written, or could not be written.
- `[fpenv]`: floating-point control state does not match what the caller expected.
- `[fp-return]`: a library call returned a non-finite number.
- `[ampr]`: an operation on an audio buffer, with its lifetime and history.
- `[dispatch-src]`: where a dispatch came from in the guest library.
- `[Vulkan]`: driver or validation messages.
- `[Trophy]` and `[Splash]`: overlay, sound and image problems.
- `[UDS]`: events posted through the user service.
- `[PS5]`: one line of detected program configuration.
- `[exit]`: what the process held when it ended and what was torn down.
- `[file]`, `[dir ]`, `[ok]`, `[fail]`: folder listings and open results in call traces.

## Triage summary

`run.sh` runs this after every run. To run it again by hand:

    python3 tools/bringup/triage.py <run>/kyty.log --json <run>/triage.json > <run>/triage.txt

It reads the log once, line by line, so a large log is not held in memory.
`triage.txt` starts with a numbered `VERDICT`. It leads with the frame timeline after loading,
then the longest stall, largest frame gap, slowest shader, repeated translations, store misses and
longest memory wait.

`FINDINGS` then has one section per subject:

- `TIMELINE`: frames per ten seconds, gaps, the first gap over 20 seconds and the frames after it.
- `STALLS`: for each no-frame gap, the render phases with their times and shaders.
- `QUEUES`: longest shader suspension per queue and memory waits of five seconds or more.
- `SHADERS`: outcomes per stage, the 20 slowest, shaders built more than once, and render-thread build totals.
- `ASYNC`: peak queued and waiting jobs, jobs per second, worst pump and final job stages.
- `BUILD RATE`: one row per minute with modules finished, median interval and peak busy workers.
  Five or more serial windows are flagged. With `[build]` lines it also totals phase time.
- `PARKED`: threads parked at a program counter and the parked-thread summaries.
- `FAULT COST`: fault rates, lock waits by caller and the hottest blocks.
- `HOLES`: skipped dispatches and failed stages grouped by failure text, with shader counts.
- `STORE`: store load lines, last store line, lookups, and misses grouped by the field that differed.
- `REJECTIONS`: shader failure classes ranked by shaders and uses, plus skipped-shader counts.
- `LEDGER`: registered and used per stage, and minimum lead time.
- `FRAME COST`: every `[frame]` line with its time.

After that come the summary lines: log size, shaders seen, new shaders per ten seconds, flip waits,
failure classes, other `!!!` lines, traced calls, unresolved stubs, device-lost triage and driver
hang dumps. `[RUN-ENDER]` prints the fatal block, or says there is none.

A section with no lines says `no data`. A line that cannot be parsed is skipped and counted as a
parse error. Each section is isolated, so one bad section prints `section failed: ...` and the rest
still appear. The same numbers are in `triage.json`.

## What to attach to an issue

Attach `triage.txt` and `run.txt`. They carry the verdict, the failure classes, the options and the
exit code. That is enough to compare runs.

Never attach program files, logs, dumps, stored modules or shader hashes. Ask for a named number or
a short line from the summary instead.
