# Handoff: state, findings, open tasks

Working rules for this repository: no program names, ids, file or package names, paths, addresses
inside the program, or shader binaries here or in commit messages. Error text, counts, emulator
source references and timings are fine. Branch from `bringup`, name branches `neon/<task>`, one
topic per commit, no force push to `bringup`.

Host for all measurements: AMD RDNA4 (RX 9070 XT), RADV (Mesa 26.2), Linux 7.4-rc, Wayland.

## Where it stands (2026-11-26, 8 runs)

The program boots, plays its intro videos with sound, shows loading screens, reads the pad.
About 45 s in (with `--auto-cross`) it requests a burst of more than 800 new shaders. From there:

1. **Frames stop, then the program ends itself 30 s later.** Its own watchdog fires after 30 s
   without a presented frame (same abort in 5 of 8 runs, always 30-31 s after the last flip).
   Measured cause of the frame stop: the GPU command thread translates the burst serially.
   `perf` over the stall: 73 % of emulator CPU time in `ShaderRecompiler::IR` passes
   (`Tracker::Run` 12 %, `ConstantPropagationPass` 12 %, `Value::operator==` + `Value::Resolve`
   14 %, `EliminateDeadCode` 3 %). The four phases that are timed in the log (CFG build,
   structurize, translate, SPIR-V emit) add up to 2.3 s of the 30 s; the rest is the untimed IR
   passes between "IR TranslateProgram" and "SPIR-V EmitProgram" in `ShaderRecompiler.cpp`.
   811 shaders were translated in those 30 s (about 37 ms each) and the burst was not finished.
2. **40 compute shaders are rejected** (with `--keep-going` they are skipped and counted):
   38 x `buffer descriptor is not a valid runtime value; GPU-selected access requires a scalar,
   raw 8/16-bit, raw DWORD x1/x2/x3/x4, or formatted X load` (ResourceTracking.cpp), 2 x
   `resource materialization failed`.
3. **GPU hang, once in 8 runs:** kernel `amdgpu: ring gfx_0.0.0 timeout`, process named, ring
   reset, then `vkQueueSubmit failed: ErrorDeviceLost`. The submitted work that hung is unknown.
4. **One other self-abort, once:** a failed check in the program right after a run of 64 KiB
   memory mappings, 42 s in. Not reproduced in three later runs.

Not yet known: whether pictures are correct once the burst is survived; what comes after.

## Open tasks

| Id | Task | Why | Size |
|---|---|---|---|
| T1 | Runtime descriptor-table mode for GPU-selected buffer descriptors (ResourceTracking, ResourceMaterialization, SPIR-V backend, renderer descriptors); vkd3d-proton's descriptor heap handling is the model | finding 2, 38 shaders | large, design first |
| T2 | Make `tests/ShaderRecompilerComputeTests.cpp` run on RADV (it requires `robustStorageBufferAccessSizeAlignment == 1`) | no shader test can run on the target machine | small |
| T3 | Decode-table audit against LLVM AMDGPU (`llvm-mc -triple=amdgcn -mcpu=gfx1013` as oracle): script, table of decoded/translated per opcode and encoding, implement mechanical twins | one missing compare opcode was found by crashing; find the rest in one pass | medium |
| T4 | Device-lost triage: name the last submitted draws/dispatches and shader hashes (VK_EXT_device_fault, RADV_DEBUG=hang, per-submit markers), and a switch to skip shaders from a hash list for bisection | finding 3 | medium |
| T5 | Shader translation off the command thread and/or a disk cache of translated SPIR-V + reflection keyed by shader hash and static key, written incrementally so every run gets further; also profile the IR passes named above for superlinear behaviour | finding 1, the current wall | large |
| T6 | Time the untimed IR passes per shader in the log (one line per pass over 5 ms) | makes finding 1 visible without perf | small |

Taken so far: none recorded here. Write your name and branch next to a task when you start.

## How to reproduce a finding

`tools/bringup/README.md` has the build line, `run.sh` and the switches. A run folder holds
`triage.txt` (failure classes, traced system calls, fatal block), `kernel-gpu.txt`, and with
`KYTY_PROFILE_AT=stall` a `perf.data` taken when frames stop. Run folders stay on the test machine;
ask for the triage text of a run rather than for its files.
