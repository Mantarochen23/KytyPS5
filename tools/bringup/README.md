# bringup branch: working notes

Fork of KytyPS5 for bring-up work on Linux with AMD RDNA4 / RADV.
This file describes the run and triage tools and how the branch is organised.

## Branches

| Branch | What |
|---|---|
| `main` | upstream `KytyPS5/KytyPS5` main, untouched. Update with `git fetch origin && git merge --ff-only origin/main`. |
| `bringup` | integration branch = `main` + the commits below, in order. Rebase onto `main` after an upstream update. |
| `perf/amd-linux` | older note-only branch, not used by this work. |

Commits on `bringup` are kept one topic each, so a single one can be offered upstream:

1. `bringup: import community bring-up patch` - a community patch posted on the upstream tracker, unchanged.
   Contains two shortcuts that hide errors rather than fix them: an oversized storage buffer is
   clamped to `maxStorageBufferRange - 256`, and an unknown sampler descriptor reuses the last
   resolved one. Both print nothing useful today; treat pictures as suspect until they are replaced.
2. `build: enable C++ exceptions` - Linux build fix for (1).
3. `shader: decode and translate V_CMP_NLE_F16` - upstreamable as is.
4. `debug: --keep-going` and `tools/bringup` - this folder, the failure ledger, host backtrace.

Tag `first-loading-screen-2026-11-26` = the build that first showed the loading screen.

## Build (Bosie)

    cmake -S . -B _Build/linux-debug -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKYTY_BUILD_LAUNCHER=OFF
    cmake --build _Build/linux-debug --target kyty_emulator --parallel 12

Use `RelWithDebInfo` for work: the `Release` configuration links with `--strip-all`, so its
backtraces and profiles are bare addresses. `run.sh` defaults to `_Build/linux-debug`.

## Run

    tools/bringup/run.sh <label>            # 180 s, sandboxed, --keep-going, no controller
    KYTY_SECONDS=600 KYTY_PAD=host tools/bringup/run.sh play

Each run gets a folder `../runs/<utc>-<label>/`: `run.txt` (commit, options, exit code),
`kyty.log`, `guest-printf.log`, `triage.txt`, `triage.json`, `ShaderFailures/`.
The script refuses to start while another sandboxed run holds the GPU.

## Debug facilities

- `--keep-going`: a shader that fails anywhere between decode and module creation (any `EXIT`,
  `EXIT_IF`, `EXIT_NOT_IMPLEMENTED` on that path, or a thrown error) is recorded and every draw or
  dispatch using it is skipped. Log line `!!! SHADER SKIP (keep-going) shader=.. stage=.. at
  file:line: reason`; later uses are counted in `!!! SHADER SKIP USES` lines (powers of two).
  Ledger: `_ShaderFailures/failures.jsonl` plus the shader binary `<stage>_<hash>.bin` (game
  data: never leaves the machine).
  Mechanism: `common/recoverable.h` (`RecoverableScope`), used in `PipelineCache::GetProgram`.
  It can be put around other subsystems the same way, but only where a half-done operation can be
  thrown away.
- Fatal errors outside such a scope print `--- Host backtrace ---` (resolve with
  `addr2line -Cfe kyty_emulator <offset>`); `KYTY_FATAL_ABORT=1` leaves a core dump.
- GPU hang / `VK_ERROR_DEVICE_LOST` triage: `VK_EXT_device_fault` and `VK_AMD_buffer_marker` are
  enabled automatically when supported. Every submit records draw/dispatch breadcrumbs with shader
  hashes (plus `VK_EXT_debug_utils` command labels when `RADV_DEBUG` or `--graphics-debug-dump` is
  active). On `VK_ERROR_DEVICE_LOST` (`vkQueueSubmit`, `vkQueuePresentKHR`, `vkAcquireNextImageKHR`,
  `vkWaitSemaphores`, `vkGetSemaphoreCounterValue`, or `waitIdle`), the emulator queries
  `VK_EXT_device_fault` and prints `=== GPU DEVICE LOST TRIAGE ===` with fault addresses/vendor
  codes, GPU marker progress, recent/in-flight submits, suspect ops, shader hashes, and a
  `bisect_hint` line. `RADV_DEBUG=hang tools/bringup/run.sh <label>` also preserves RADV's
  `radv_dumps_*` folder in `<run>/radv/`.
- `--skip-shaders <hashes|@file>` / `KYTY_SKIP_SHADERS=<comma-separated hex hashes>`: skip draws
  and dispatches matching any listed shader hash (with or without `--keep-going`) to bisect a
  hanging shader in a few runs. Skipped uses are logged as `!!! SHADER SKIP (skip-shaders)` and
  summarized by `triage.py`.
- Existing upstream switches worth knowing: `--shader-log-direction File`, `--graphics-debug-dump`
  (dumps every shader and IR), `--vulkan-validation`, `--gpu-assisted-validation`,
  `--command-buffer-dump`, `--rd` (RenderDoc), `--profile` (Tracy), `--perf-overlay`.

## Triage summary

`run.sh` writes `triage.txt` and `triage.json` from one streaming pass over `kyty.log` (the real log is
~2 GB, so the script reads bytes line by line and decodes as latin-1; nothing is buffered per line):

    python3 tools/bringup/triage.py <run>/kyty.log --json <run>/triage.json > <run>/triage.txt
    python3 -m unittest discover -s tools/bringup/tests

`triage.txt` starts with a numbered `VERDICT`. It leads with the frame timeline after the loading phase:
the first frame gap >= 20 s ends loading, and every frame presented after it is listed with its time and
the gap before it. The rest of the verdict is the longest stall and what it waited on, the largest frame
gap, the slowest shader, repeated translations, store distinct misses and the longest memory wait. Then
comes a `FINDINGS` block with one section per subject:

- `TIMELINE`: frames per 10 s, last frame, every gap >= 3 s (start, length), the first gap >= 20 s and the
  frames after it.
- `STALLS`: for each no-frame gap, the render-thread phase sequence with seconds and the shader it waited
  on per phase.
- `QUEUES`: per queue, the longest single-shader suspension and which shader; every memory wait >= 5 s with
  address, compare, wanted/seen values, polls and time.
- `SHADERS`: translation outcomes per stage, the 20 slowest by summed phase `elapsed_ms` with their slowest
  phase, shaders translated more than once, and render-thread build totals/worst from `Async STALL:`.
  A "translated N times" entry counts real builds only (`outcome=emitted*`); `served from store` lines are
  reported as outcomes but are not translations.
- `ASYNC`: peak queued/waiting, min and median jobs/s while queued is non-zero, worst pump, final stages.
- `BUILD RATE`: one row per 60 s window with modules finished (CS `translation end` emitted), the median
  interval between them and the peak workers busy from `Async:`. A `serial build` is flagged when the median
  interval is over 1 s with at most one worker busy for five consecutive windows. With `[build]` lines it
  also totals translate/emit/module/pipeline ms and lists the ten largest builds by total. These lines are
  optional; their absence leaves only the summary.
- `PARKED`: each `[park] thread .. parked for good` line as thread id, pc as module+offset and time, plus the
  `[park] N guest threads parked` summaries.
- `HOLES`: dispatch skips and failed stages grouped by failure text with digits normalised, with the count
  of holes and of distinct shaders in each group.
- `STORE`: cache load numbers, the last `[store]` line, `lookups` total/hit/distinct-miss (plus the
  miss-key/miss-absent split), and `[store] miss #..` lines grouped by the field that differed.
- `REJECTIONS`: failure classes as before, plus the count of `!!! SHADER SKIP` lines.
- `LEDGER`: registered/used per stage and minimum lead time from `[shader-reg]`.
- `RUN-ENDER`: the fatal block, unchanged, last.

A section with no source lines prints `no data`, not zeros. The same numbers are in `triage.json` under
`verdict` and `findings`. Times are the `@@T` seconds in the run. Parsing is defensive: a line that cannot
be parsed is skipped (counted as `parse_errors` in the JSON), and each section is isolated so a failure
prints `section failed: <error>` in that section while the others still appear; the exit code stays 0.
`tools/bringup/tests/` holds hand-written snippets per section, the absent case, the section-isolation and
garbage-line cases, and a 2,000-line masked excerpt used as a regression run.

## The loop

1. `tools/bringup/run.sh <label>`.
2. Read `triage.txt`: the `VERDICT` first, then `FINDINGS` for the numbers behind it (frame gaps,
   stalls, queues, shader timing, async pipeline, store, ledger) and the failure classes (one class =
   one fix), ordered by shaders affected; the run-ender if the run died; new shaders per 10 s (still
   climbing = more classes to come).
3. Fix a class: prove the cause first (for an instruction, `llvm-mc --disassemble -triple=amdgcn
   -mcpu=gfx1013`), add a case to `tests/ShaderRecompilerComputeTests.cpp`, one commit.
4. Run again with the same state folder (warm pipeline cache) and compare `triage.json`.

Known gap: the GPU test harness needs `robustStorageBufferAccessSizeAlignment == 1`, which RADV
does not report, so the shader test cases cannot run on Bosie yet.

## Offline shader corpus

Add `--dump-registered-shaders <dir>` to one emulator run to copy every distinct registration into
`<dir>/<stage>_<hash>.bin` and append its metadata to `<dir>/index.jsonl`. The option is off by
default. Registration only hands immutable copies to a bounded queue: a full or contended queue is
counted as dropped in the shutdown ledger instead of stalling the registering thread.

Build and translate the dump without starting the program:

    cmake --build _Build/linux-debug --target shader_corpus_tool --parallel 12
    _Build/linux-debug/shader_corpus_tool --dir <dump> --out <report> --jobs 12

Optional filters are `--only <stage>`, `--limit <n>`, and `--timeout-ms <n>` (20 seconds by
default). `--emit-spirv` also writes emitted modules under `<report>/spirv/`. Each shader runs in a
separate child process; the parent bounds concurrency and kills only the timed-out child, so one
crash or timeout cannot end the corpus run.

`results.jsonl` is sorted by hash and contains the phase/pass timings, CFG size, wave size, SPIR-V
size, and one of `emitted`, `needs-runtime:<input>`, `rejected:<class>`, `crashed`, or `timeout`.
`summary.md` groups outcomes by stage, ranks rejection shapes with examples, lists the 30 slowest
translations and their slowest passes, totals pass time, and counts compute shaders which need no
runtime input. Graphics registrations normally need later pipeline state; fused stages also need
their paired bytecode. Compute registrations include their creation-time stage registers, but a
shader whose resource plan reads user-data values remains runtime-dependent.

### First-use records and fidelity

With the same `--dump-registered-shaders <dir>` option the run also writes, at the first use of each
(stage, hash) whose resource plan exists, one line in `<dir>/use.jsonl` and one blob
`<dir>/use_<stage>_<hash>.bin`: the user-data words, the raw stage input struct, the push window,
fused back code, and every guest memory range the walker read while materializing (bounded to 16 MiB
and 65,536 ranges per record; `memory_truncated` says so). The queue is bounded in count (128) and
bytes (256 MiB), never blocks the render thread, and the shutdown ledger prints
`use dump: written= dropped= queue_high_water= skipped:<reason>=<n>`. Programs with GPU-selected
calls and stages whose input struct is not trivially copyable are counted as skipped.

The tool attaches a record to its registration by (stage, hash) and replays it for compute and pixel
shaders: it restores the stage struct (a size mismatch is `needs-runtime:input_layout`), serves the
recorded memory to the walker, materializes, and emits SPIR-V (`source` is `use` in `results.jsonl`).
Without a record the stage state is not known, so a resource tracking failure is reported as
`needs-runtime:use_record` with the full text in `shape`, never as `rejected`; decode, CFG and
unsupported-instruction failures do not depend on the inputs and stay `rejected`. The summary ranks
the failure shapes with the hash masked. Image heap kinds are not replayed (`notes` flags such a
shader).

Fidelity: `--compare-log <run log>` reads the emulator's `translation end ... outcome=` lines and adds
per stage how many shaders were emitted in both, only live, only in the tool and in neither, and the
share of live-emitted shaders the tool also emitted.
