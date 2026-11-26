# bringup branch: working notes

Fork of KytyPS5 for bring-up work on Linux with AMD RDNA4 / RADV.
Everything here is local; nothing is published.

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
- Existing upstream switches worth knowing: `--shader-log-direction File`, `--graphics-debug-dump`
  (dumps every shader and IR), `--vulkan-validation`, `--gpu-assisted-validation`,
  `--command-buffer-dump`, `--rd` (RenderDoc), `--profile` (Tracy), `--perf-overlay`.

## The loop

1. `tools/bringup/run.sh <label>`.
2. Read `triage.txt`: failure classes (one class = one fix), ordered by shaders affected; the
   fatal block if the run died; new shaders per 10 s (still climbing = more classes to come).
3. Fix a class: prove the cause first (for an instruction, `llvm-mc --disassemble -triple=amdgcn
   -mcpu=gfx1013`), add a case to `tests/ShaderRecompilerComputeTests.cpp`, one commit.
4. Run again with the same state folder (warm pipeline cache) and compare `triage.json`.

Known gap: the GPU test harness needs `robustStorageBufferAccessSizeAlignment == 1`, which RADV
does not report, so the shader test cases cannot run on Bosie yet.
