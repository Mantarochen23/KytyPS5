# Shader Cache, Async Compilation, and IR Pass Analysis (T5)

> **Build status note**: This branch was authored and audited on a Windows host without `clang++`,
> Linux headers, or a Vulkan/RADV runtime. No Linux build or runtime test results have been
> executed here; exact verification commands for the Linux/RADV test machine are listed at the end.

## 1. Problem Summary

During a cold-store boot, the emulator receives a burst of 800+ previously unseen shaders about
45 s into execution. On `bringup` prior to T5, the GPU command thread translates and compiles each
shader synchronously inside draw/dispatch execution:

- At roughly 37 ms average per shader, 811 translations consume ~30 s of command-thread wall time
  without completing a frame, tripping the guest's 30 s watchdog.
- Instrumented IR pass timings and `perf` samples over the stall identified the dominant CPU costs:
  - `ConstantPropagation`: 6,379 ms total
  - `TrackResources`: 6,224 ms total
  - `RemoveIdentities`: 565 ms total
  - `EliminateDeadCode`: 556 ms total
  - `Value::operator==` + `Value::Resolve`: 14% of total CPU samples (`perf`)

## 2. Synchronous Blocking Call Chain (Code Evidence)

Before async compilation, every draw or compute dispatch with a cold shader blocked the GPU command
thread end-to-end across IR translation, SPIR-V emission, and Vulkan pipeline creation:

- **Graphics draw path**:
  `CommandProcessor` -> `RenderExecutor` (`src/graphics/host_gpu/renderer/renderDraw.cpp:1189`) ->
  `PipelineCache::GetGraphicsPipeline` (`src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp:1341`)
  -> `ShaderRecompiler::CompileShader` -> `vkCreateShaderModule` -> `vkCreateGraphicsPipelines`.
- **Compute dispatch path**:
  `CommandProcessor::CpOpDispatchDirect` / `CpOpDispatchIndirect`
  (`src/graphics/guest_gpu/graphicsRun.cpp:1163`, `:1184`) ->
  `RenderExecutor::DispatchDirect` / `DispatchIndirect`
  (`src/graphics/host_gpu/renderer/renderCompute.cpp:369-380`, `:526-530`) ->
  `PipelineCache::GetComputePipeline` (`src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp:1387`)
  -> `ShaderRecompiler::CompileShader` -> `vkCreateShaderModule` -> `vkCreateComputePipelines`.

Graphics draws targeting swapchain/frame-redrawn render targets can optionally be deferred/skipped
while background workers build their pipelines. Compute dispatches, by contrast, cannot be safely
dropped because downstream dispatches, indirect draw arguments, and graphics passes observe their
storage-buffer and storage-image writes. Compute therefore requires **lossless** off-thread
compilation with prefetch and command-queue suspension rather than draw skipping.

## 3. Incremental On-Disk Shader Store (`--shader-cache`)

An incremental, append-only on-disk shader and pipeline store already exists in
`src/graphics/host_gpu/renderer/pipeline/pipelineSerialization.cpp`:

- `PipelineCache::WarmUp` (`:140`) and `PipelineCache::LoadRecord` (`:204`) replay previously
  saved shader modules, reflection metadata, tessellation layouts, and graphics pipeline keys at
  startup.
- `PipelineCache::StoreModule` (`:276`), `PipelineCache::StorePipeline` (`:290`), and
  `PipelineCache::StoreLayout` (`:300`) append each compiled artifact incrementally as soon as it is
  produced, so interrupted runs retain partial progress.
- `origin/bringup` already defaults the store on (`a3274921`) and fixes `RenderContext` member
  initialization order so `m_image_heap` exists before `PipelineCache::WarmUp` (`3cd92746`).

Because the on-disk store already solves warm-boot reloads, T5 avoids duplicating serialization
code in `pipelineSerialization.cpp` and focuses on surviving the **cold-store first boot** via
multi-threaded compilation, lossless async compute, presentation keepalive, and IR pass complexity
fixes.

## 4. Off-Thread Compilation, Async Compute, and Presentation Keepalive

### 4.1 Upstream PR #2119 Port (`KytyPS5/KytyPS5#2119`, `newblord:async_shader_compile`)

The first five commits on this branch port upstream PR #2119 cleanly onto `origin/bringup`
(`3ecafb91`), preserving original authorship and commit boundaries:

1. Worker pool (`ShaderWorkers`) building graphics shaders and pipelines off the command thread
   (`--async-shader-compile`, enabled by default).
2. Demand-driven priority promotion at below-normal worker thread priority.
3. Skipping draws only when the target is redrawn every frame.
4. Diagnostics for render-thread builds plus skipping small and G-buffer draws.
5. Bounding render area and layer count to the attachment's real image dimensions.

### 4.2 Lossless Async Compute, Prefetch, Keepalive, and Review Fixes

The follow-up commit extends the worker pipeline to compute shaders and closes several correctness
and responsiveness gaps:

- **Immutable shader code snapshots** (`shaderCompiler.h`, `shader.cpp`):
  `ShaderMapEntry` captures an immutable `std::shared_ptr<const std::vector<uint32_t>>` snapshot of
  shader words when parsed on the command/API thread. `TryPrepareComputeProgram` reads strictly
  from this snapshot so background workers never dereference mutable guest virtual memory.
- **Early compute prefetch**:
  - `AgcCreateShader` (`agc.cpp`) invokes `ShaderQueueComputePrefetch` (`shader.cpp:800`), drained
    by `GuestGpu::DrainComputePrefetches()` on the GPU thread into `PipelineCache::RequestAsync`
    (`AsyncMode::PrefetchOnly`).
  - `CommandProcessor::PrefetchComputeShaders` (`graphicsRun.cpp:685`) scans queued PM4 packets
    ahead of execution (using fault-free `TryReadBacking` for indirect SH registers and indirect
    buffers) to enqueue compute shader builds before the dispatch packet is reached.
- **Worker-side compute build & Xbyak guard** (`pipelineAsync.cpp`):
  Compute programs compile SPIR-V and create `vk::Pipeline` on worker threads and persist modules
  via `StoreModule`. Before recompiling on a worker, `Decoder::DecodeProgram` checks `has_swap_pc`
  and routes Xbyak-dependent compute shaders to `sync_only_keys` so JIT codegen stays on the
  command thread.
- **Lossless PM4 suspension** (`renderCompute.cpp:369`, `:526`, `graphicsRun.cpp:679`, `:1163`, `:1184`):
  When `allow_async` is true (guarded on `g_current_execution != nullptr` so direct unit-test calls
  in `ShaderRecompilerComputeTests` stay synchronous), `EnsureComputePipeline` requests/promotes
  the worker job and returns `pending = true`; `CommandProcessor` calls `SuspendPm4ForShader()` to
  yield the queue without dropping the dispatch and re-executes once the worker finishes.
- **Condition-variable waits replacing busy-wait loops**:
  `PipelineCache::WaitForAsync()` (`pipelineAsync.cpp:434`) sleeps on a `std::condition_variable`
  notified whenever a worker pushes to `m_finished`, replacing 1 ms sleep loops in
  `GetGraphicsPipeline` (`pipelineCache.cpp:1356`) and `GetComputePipeline`
  (`pipelineCache.cpp:1396`). `SetAsyncFinishCallback` wakes `GuestGpu`'s `m_async_ready` condvar
  immediately when any worker finishes (with a 50 ms fallback when queues are blocked only on
  shaders, and 100 us polling when waiting on `WaitRegMem`).
- **Presentation keepalive** (`skippedDraws.h:54`, `videoOut.cpp:830`, `:1163`, `:1174`):
  An atomic `g_shader_worker_jobs` counter tracks queued, running, and pending-finish worker jobs
  via `AreShaderWorkersBusy()`. While workers are busy, `PresentThread` repeats
  `PresentLastFrame()` to keep compositor flips alive, and `FlipQueue::Flip()` defers starting its
  3 s `MaxHold` timeout until workers finish.
- **Additional PR #2119 review fixes**:
  Fixed off-by-one target dimension checks (`rt.attrib2.width + 1u` / `height + 1u`) in G-buffer
  and small-target heuristics, and guarded `MaterializeResources` failures against re-deferral so
  an unmaterializable shader cannot loop indefinitely.

## 5. Superlinear IR Pass Analysis and Fixes

Code inspection of the passes named by `perf` and `TimeIrPass` identified multiple quadratic and
exponential paths that amplify sharply on large shaders (thousands of instructions and deep
exec-mask / descriptor DAGs):

| Hot spot (measured cost) | Root cause | Complexity before -> after | Files changed |
|---|---|---|---|
| `Value::operator==`, `Value::Resolve`, `Inst::HasUses/UseCount` (14% CPU in `perf`) | Out-of-line non-LTO calls per operand/use across all IR passes; recursive `Value::Resolve` | Cross-TU call overhead + recursion -> header-inlined + iterative | `ir/Value.h`, `ir/Value.cpp` |
| `EquivalentValue` (called heavily by `TrackResources`) | Allocated `std::vector` and linearly scanned visited pairs on every call, even trivial mismatches | $O(M^2)$ visited scan + heap alloc per call -> trivial fast-path + 16-entry inline array spilling to `unordered_set` | `ir/Program.cpp` |
| `RemoveIdentities` (565 ms) & `EliminateDeadCode` (556 ms) | Called `ReplaceUsesWith` / `Invalidate` -> `Inst::RemoveUse` per dead/identity node, doing linear scan + `vector::erase` on shared targets with $U$ uses | $O(U^2)$ per heavily shared target -> $O(V + E)$ single-pass stable `std::erase_if` compaction (preserving exact `Inst::Uses()` order) | `ir/passes/DeadCodeElimination.cpp` |
| `ConstantPropagationPass` (6,379 ms) | `FlushUsers()` ran an unbounded transitive BFS + fresh `unordered_set` over every downstream user per wave; copied `users` and `operands` vectors on every visit; `LaneMaskProjection::CanProject` re-walked non-projectable wave32 mask DAGs repeatedly | $O(W \cdot N)$ transitive flooding + repeated mask walks -> epoch-stamped `FlushUsers` traversing only pattern-inspected opcodes, mutation-driven `Replace`/`NotifyMutated` without per-visit copies, and mutation-epoch negative cache in `LaneMaskProjection` | `ir/passes/ConstantPropagation.cpp` |
| `TrackResources` (`Tracker::Run`, 6,224 ms) | Exponential $O(2^D)$ re-walks of diamond `LogicalAnd`/`LogicalOr` DAGs in `Implies`, `ImpliesNonzero`, `ImpliesIndexBelow32`, `EqualLocalValue`, `UnsignedBound` (plus repeated `SimplifyGuard` + `ImpliesNonzero` re-scans inside `Implies`); four nested $O(S \cdot N)$ / $O(S^2)$ loops in `PlanScalarReads`; $O(B^2)$ linear `pending` scans in `GuardedOnEntry`, `Reaches`, `Dominates` and uncached `InCycle`; scanning entire blocks past leading `Phi` nodes in `FoldBoundedLoopSelectors`, `MatchReversedImageRange`, `SelectsRange`, `BoundedSelectedIndex`; linear scans in `PhiWeb`, `UsesOnlyImageDescriptors`, `BitScanRanks`, `FiniteSelectorDomain`, `FindIndirectDescriptor`, `IsIndirectPlanningMemory`, `AddHandlePatch`, `AddMemoryPatch`, `read0 -> key`, and `std::map` for `m_descriptor_values` | Exponential / quadratic -> memoized $O(V)$ DAG predicates, $O(1)$ block-index visited arrays + cached `InCycle`, pre-indexed + bucketed `PlanScalarReads`, early `break` on first non-`Phi`, and hash-indexed lookups/deduplication | `ir/passes/ResourceTracking.cpp` |

All IR changes preserve `Inst::Uses()` ordering (asserted by `ScalarProvenanceTests`), IR
validation invariants (`ValidateProgram`), and exact fold / resource-tracking semantics.

## 6. Build and Verification Steps (Linux / RADV)

Run on the Linux/RADV test machine from a checkout of the async translation branch:

```bash
# 1. Build emulator and IR/recompiler test targets
cmake --build _Build/linux-debug --target \
  kyty_emulator \
  scalar_provenance_tests \
  resource_tracking_tests \
  resource_materialization_tests \
  shader_recompiler_compute_tests

# 2. Run unit test suites
ctest --test-dir _Build/linux-debug --output-on-failure \
  -R "scalar_provenance|resource_tracking|resource_materialization|shader_recompiler_compute"

# 3. Cold-store boot test (verify watchdog survival, per-pass timings, and async worker progress)
#    Remove or point to an empty shader cache directory, then run tools/bringup/run.sh.
#    Compare IR pass totals (TrackResources, ConstantPropagation, RemoveIdentities, EliminateDeadCode)
#    and verify frames stay alive through the 800+ shader burst.

# 4. Warm-store second boot test (verify incremental store reload + steady frame pacing)
#    Re-run tools/bringup/run.sh against the populated shader cache directory.
```
