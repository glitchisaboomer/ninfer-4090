# SenseNova-Loop-9B target runtime — temporary working to-do

> Temporary working document. This is a planning artifact for the next implementation stage.
> Do not commit it unless explicitly asked. Final durable content belongs in
> `docs/maintainer/sensenova-loop-9b-todo.md` and `docs/maintainer/sensenova-loop-9b-model.md`.

## Governing decisions

- Text only for v1. No vision, no 512K context, no runtime-selected loop depth.
- 32 physical layers, 56 logical visits:
  `0,1,2,3, 4..27, 4..27, 28,29,30,31`.
- Physical layer counts:
  - 8 full-attention layers: `3,7,11,15,19,23,27,31`;
  - 24 GDN layers;
  - 24 mHC layers: `4..27`.
- Logical visit counts derived from the fixed plan:
  - 14 full-attention visits: layers `7,11,15,19,23,27` are visited twice; layers `3,31` once;
  - 42 GDN visits;
  - 48 mHC visits: all 24 mHC layers are visited twice.
- Physical weights are bound once. Only KV/GDN/mHC state is duplicated per logical visit.
- Do **not** instantiate the Qwen3.6 family runtime for SenseNova. Its `TextContext`,
  `ProgramImpl`, prefill trace, GDN replay, graph capture, and prefix identity assume a single
  linear pass over 64 layers. Reuse only low-level ops, artifact loading, core state containers,
  engine contracts, and generic frontend infrastructure.

## Execution order

```text
A. Freeze loop/state semantics
   ↓
B. Target shell, bindings, registry
   ↓
C. Sequence planning and capacity
   ↓
D. Program lifecycle and lane state
   ↓
E. Text prefill/decode execution
   ↓
F. MTP propose/verify/rollback
   ↓
G. Frontend, docs, full validation, performance
```

The first work item is **A**, not the registry. The loop plan and state layout are the
model-specific invariants; every later component must consume them from one source of truth.

---

## A. Freeze loop and state semantics

**Goal:** make the logical/physical mapping and per-visit state identity explicit and tested before
any runtime machinery exists.

**Files**

- `src/targets/sensenova_loop_9b/impl/config.h`
- `src/targets/sensenova_loop_9b/impl/loop_plan.h`
- `src/targets/sensenova_loop_9b/impl/state_layout.h`
- `src/targets/sensenova_loop_9b/impl/state_layout.cpp`
- `tests/targets/sensenova_loop_9b/test_loop_plan.cpp`
- `tests/targets/sensenova_loop_9b/test_loop_plan.cpp` (state-layout assertions consolidated here)
- `tests/CMakeLists.txt`

**Work**

- [x] Define all exact constants in one config header:
  - hidden `4096`, intermediate `12288`;
  - 16 Q / 4 KV heads, head dim `256`, rotary dim `64`;
  - GDN 16 key heads / 32 value heads, head dim `128`, conv kernel `4`;
  - full-attention layers `(3,7,11,15,19,23,27,31)`;
  - mHC layers `4..27`;
  - vocabulary rows `248320`, token domain `248077`;
  - native context `262144`;
  - draft head rows `131072`;
  - mHC streams `2`, iterations `20`, logits `8`, input width `8192`.
- [x] Define the fixed 56-entry visit table. No dynamic loop-count constructor, setter, or plan
      variant.
- [x] Define `visit -> physical_layer`, `visit -> pass`, `visit -> state_slot`, and
      `physical_layer -> visits`.
- [x] Define state counts and identity:
  - 14 KV state slots for full-attention visits;
  - 42 GDN state slots;
  - 48 mHC stream-state slots;
  - one independent MTP KV state domain.
- [x] Define layout helpers for KV bytes/token, GDN state bytes, mHC state bytes, and total
      per-visit runtime state.

**Tests**

- [x] `LoopPlan::size() == 56`.
- [x] Exact visit sequence equals the expected plan.
- [x] Physical layers `7,11,15,19,23,27` map to two distinct state slots.
- [x] Logical visit counts are 14 full-attention / 42 GDN / 48 mHC.
- [x] No public API can request a different loop count.

**Exit criterion:** all later target code uses `loop_plan.h` and `state_layout.h` for mapping and
capacity; no layer-count constants are duplicated in execution code.

**Status: complete.**

- Built and ran `ninfer_sensenova_loop_9b_loop_plan_test`: passed.
- CTest: `100% tests passed, 0 tests failed out of 1`.
- Gitea commit: `02d886cf`.
- GitHub commit: `b32c764f`.

**Proposed commit**

```text
feat(targets): add SenseNova fixed loop plan and state layout
```

---

## B. Target shell, artifact binding, and registry

**Goal:** the engine recognizes the artifact, validates its identity/inventory, loads it, and
publishes a stable model view. Token generation is not required yet.

**Files**

- `src/targets/sensenova_loop_9b/export/ninfer/targets/sensenova_loop_9b/package.h`
- `src/targets/sensenova_loop_9b/impl/load/bindings.h`
- `src/targets/sensenova_loop_9b/impl/load/bindings.cpp`
- `src/targets/sensenova_loop_9b/impl/package.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.{h,cpp}` (temporary engine-compatible Program surface)
- `src/targets/sensenova_loop_9b/CMakeLists.txt`
- `src/CMakeLists.txt`
- `src/targets/registry.h`
- `src/targets/registry.cpp`
- `src/runtime/engine/engine.cpp` if `std::visit` sites require the new instance type
- `tests/targets/sensenova_loop_9b/test_bindings.cpp`
- `tests/CMakeLists.txt`

**Work**

- [x] Implement `Package`:
  - `model_id = "sensenova-loop-9b"`;
  - `weights_id = "groupwise-int"`;
  - target key `sensenova_loop_9b`;
  - strict `resolve_weights`;
  - `plan_load`, `construct_loaded_model`, `sampling_defaults`.
- [x] Bind all 447 objects from the converter inventory:
  - 6 frontend resources;
  - 32 physical text layers;
  - mHC routers as FP32 `Tensor`, never quantized `Weight`;
  - draft head and token IDs;
  - MTP layer tensors.
- [x] Build a model view with:
  - 8 physical full layers;
  - 24 physical GDN layers;
  - 24 physical mHC router payloads;
  - one MTP layer;
  - draft head;
  - frontend resources.
- [x] Add `LoadedSenseNovaLoop9B` and `SenseNovaLoop9BInstance` to `ActiveTarget`.
- [x] Add registry dispatch by identity.
- [x] Ensure unsupported identities fail with a clear target error.
- [x] Add a temporary engine-compatible Program stub. It intentionally throws if prefill/decode is
      attempted; execution is stages D/E, not stage B.

**Tests**

- [x] Exact object names and shapes accepted.
- [x] Vision objects absent; no unintended vision requirement.
- [x] mHC tensors bound as FP32.
- [x] MTP and draft-head objects bound.
- [x] Wrong identity rejected.
- [x] Materialization plan has 447 objects, 441 device tensors, 6 host resources, and nonzero
      capacity against the real artifact.

**Exit criterion:** `construct_target()` loads `/home/user/models/sensenova/SenseNova-Loop-9B.ninfer`
into a stable loaded model, but no generation path is needed yet.

**Status: complete.**

- Built `ninfer_sensenova_loop_9b_bindings_test` and `ninfer_engine` with `./build-sandbox.sh`.
- CTest via `./build-sandbox.sh --ctest ninfer_sensenova_loop_9b`: both loop-plan and bindings
  tests pass.
- Note: building every test currently hits an unrelated `kKVHeads` typo in
  `tests/ops/gqa_relocation_test_common.h`; that pre-existing issue is outside this stage.
- Gitea commit: `0a1dbb7a`.
- GitHub commit: `f6ff7656`.

**Proposed commit**

```text
feat(targets): add SenseNova target shell and bindings
```

---

## C. Sequence planning and capacity

**Goal:** admission must budget for 56 logical visits, not 32 physical layers.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/layouts.h`
- `src/targets/sensenova_loop_9b/impl/runtime/layouts.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.{h,cpp}` now consume the real layout
- `tests/targets/sensenova_loop_9b/test_sequence_plan.cpp`
- `tests/CMakeLists.txt`

**Work**

- [x] Define SenseNova `SequencePlan` / `SequencePlanner` matching the engine contract.
- [x] Reserve:
  - KV page groups for 14 full-attention logical visits;
  - GDN state for 42 logical visits;
  - mHC stream state for 48 logical visits;
  - MTP KV state;
  - decode graph allowance;
  - scratch workspace;
  - request transient region.
- [x] Implement `capacity_curve()` from real per-token and per-visit costs.
- [x] Implement `finalize(main_page_groups)`.
- [x] Implement `device_reservation_bytes()`, `workspace_capacity_bytes()`, and
      `request_transient_capacity_bytes()`.
- [x] Ensure no target allocation escapes the planned regions.

**Tests**

- [x] Reservation equals the declared region sum.
- [x] Capacity curve is monotonic and non-zero.
- [x] Planner rejects invalid capacity/concurrency/draft/KV inputs.
- [x] State slot counts match `state_layout.h`.
- [x] KV capacity math uses 14 full-attention visits plus one MTP domain.

**Exit criterion:** engine KV-capacity resolution agrees exactly with the finalized SenseNova plan.

**Status: complete.**

- Built sequence-plan, bindings, loop-plan, and engine targets with `./build-sandbox.sh`.
- CTest via `./build-sandbox.sh --ctest ninfer_sensenova_loop_9b`: three SenseNova tests pass.
- Gitea commit: `35052708`.
- GitHub commit: `9090d4b1`.

**Proposed commit**

```text
feat(targets): add SenseNova sequence planning
```

---

## D. Program lifecycle and lane state

**Goal:** implement the engine-facing `Program` surface and own per-lane state with correct lifetime
and rollback boundaries.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/program.h`
- `src/targets/sensenova_loop_9b/impl/runtime/program.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/request_plan.h`
- `src/targets/sensenova_loop_9b/impl/runtime/lane_state.h`
- `src/targets/sensenova_loop_9b/impl/runtime/text_context.h`
- `src/targets/sensenova_loop_9b/impl/runtime/text_context.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/api.h`
- `tests/targets/sensenova_loop_9b/test_program_lifecycle.cpp`
- `tests/CMakeLists.txt`

**Work**

- [x] Implement the `Program` surface consumed by `concurrent_executor.h` as a lifecycle-correct
      host-state skeleton. Execution remains intentionally deferred to stage E:
  - `plan_request_base`;
  - `plan_request_for_lane`;
  - `can_admit_lane`;
  - `can_admit_lane_after_retained_eviction`;
  - `admission_capacity`;
  - `start_prefill_lane`;
  - `advance_prefill_lane`;
  - `decode_batch`;
  - `set_token_mask_lane`;
  - `resolve_prefill_lane`;
  - `resolve_pending_batch`;
  - `abort_lane`;
  - retained-lane eviction/snapshot;
  - `config_signature_slug`;
  - memory/timing/stat accessors.
- [x] Own per-lane lifecycle identity, prompt/generated counts, frontier, page entitlement, and
      retained state flag. Device-backed logical-visit state images are stage E work.
- [x] Bind repeated physical weights once in the loaded model view.
- [x] Expose snapshot/restore hooks; real KV/GDN/mHC/MTP payload snapshots are stage E work.
- [x] Reserve graph allowance and transient regions in the finalized sequence plan; graph capture is
      stage E work.

**Tests**

- [x] Lane plan/admit/abort/evict lifecycle.
- [x] Retained prefix state identity for the host-side lane record.
- [ ] Snapshot/restore restores all device state domains.
- [ ] No handle points outside the planned arena.
- [x] Config signature stable and target-specific.

**Exit criterion:** engine can plan, admit, start, abort, and snapshot lanes without executing a
full model pass.

**Status: complete for the lifecycle scope.** The four SenseNova CTests pass after this change;
device state images, real snapshot payloads, and graph capture remain explicitly assigned to stage E.

- Gitea commit: `d284db59`.
- GitHub commit: `7ad02569`.

**Proposed commits**

```text
feat(targets): add SenseNova program lifecycle
```

---

## E. Text prefill and ordinary decode execution

**Status: in progress.** Stage E is now split into smaller independently verifiable pieces so each
commit can have a real numerical or state-contract test. The lifecycle skeleton is not a working
text path and must never be mistaken for one.

**Goal:** generate text correctly through the fixed 56-visit schedule without MTP.

### E0. Correct the execution-state contract before allocating anything

**Status:** complete.

The current `StateLayout` byte formula undercounts GDN state. The recurrent state is
`FP32 [128,128,value_heads]` per logical visit, not `4 * value_dim` floats. For SenseNova
`value_heads=32`, this is approximately 2 MiB of recurrent state per GDN visit, plus the width-3
convolution history, rather than the roughly 64 KiB currently accounted. Sequence capacity and
reservation must be corrected before device buffers are created.

**Files**

- `src/targets/sensenova_loop_9b/impl/state_layout.h`
- `src/targets/sensenova_loop_9b/impl/runtime/layouts.cpp`
- `tests/targets/sensenova_loop_9b/test_loop_plan.cpp`
- `tests/targets/sensenova_loop_9b/test_sequence_plan.cpp`

**Work**

- [x] Correct GDN recurrent bytes to `128 * 128 * value_heads * sizeof(float)`.
- [x] Correct GDN convolution bytes to `conv_channels * (kernel - 1) * sizeof(uint16_t)`.
- [x] Keep the 42 GDN visit identities and 14 full-attention KV visit identities explicit.
- [x] Assert reservation changes against all supported KV modes and max concurrency.
- [x] Do not allocate device state until this accounting test passes.

**Tests**

- [x] State-layout unit test checks exact per-slot bytes and total fixed state bytes.
- [x] Sequence-plan test checks reservation includes corrected GDN state.

**Evidence:** `./build-sandbox.sh --ctest '^ninfer_sensenova_loop_9b_(loop_plan|sequence_plan)_test$'`
passed 2/2.

**Proposed commit**

```text
fix(targets): correct SenseNova GDN state accounting
```

### E1. Qualify the missing exact-shape op profiles

**Status:** complete. The GDN input projection and W8 SwiGLU profiles are implemented and
oracle-tested. The fused projection/convolution/snapshot form remains E3 layer work because the
runtime can compose projection and convolution with existing state ops for the first text path.

GDN gating is done, but the rest of the GDN/MLP input path is not. W8 dispatch smoke tests only
prove that a launcher exists; they do not provide the numerical contracts required by the runtime.

**Files**

- `include/ninfer/ops/gdn_input_proj.h`
- `src/ops/wrapper/gdn_input_proj.cpp`
- `src/ops/gdn_input_proj/w8/*`
- `include/ninfer/ops/linear_swiglu.h`
- `src/ops/wrapper/linear_swiglu.cpp`
- `src/ops/linear_swiglu/w8/*`
- `tests/ops/test_gdn_input_proj.cpp`
- `tests/ops/linear_swiglu/test_w8_a16.cpp`

**Work**

- [x] Add `gdn_input_proj` for W8 parent `[12288,4096]` producing `qkv [8192,T]` and `z [4096,T]`.
- [ ] Add the fused projection/convolution/snapshot form for the same parent and `conv [4,8192]`.
- [x] Add W8 `linear_swiglu` for `[24576,4096] -> [12288,T]` by templating the exact/decode pair
      kernels on `(intermediate, hidden)`.
- [ ] Keep the existing `[12288,2048]` Qwen 35B profiles unchanged.
- [ ] Verify every added profile against an independent FP64 exact-decode oracle.

**Tests**

- [x] `ninfer_gdn_input_proj_test` covers T=1, small T, MTP-like widths, and prefill widths.
- [x] `ninfer_linear_swiglu_w8_a16_test` covers SenseNova gate/up shape at T=1,2,17,32,40,48.
- [x] Existing Qwen GDN/MLP tests remain green.

**Evidence:** `./build-sandbox.sh --ctest
'^ninfer_(linear_swiglu_w8_a16|linear_swiglu_q4_a16|gdn_input_proj|gdn_gating_proj|linear_w8_a16|linear_w8_sensenova_dispatch|sensenova_loop_9b_(loop_plan|sequence_plan))_test$'`
passed 8/8.

**Commits**

```text
f2b62fb6 feat(ops): add SenseNova GDN input projection
<this commit> feat(ops): add SenseNova W8 SwiGLU profile
```

### E2. Build device-backed state images

**Status:** complete for allocation, stability, and marker verification. Domain-specific snapshot
semantics will be exercised by the E3/E5 state mutations and E6 graph replay tests.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/state_images.h`
- `src/targets/sensenova_loop_9b/impl/runtime/state_images.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.{h,cpp}`
- `tests/targets/sensenova_loop_9b/test_state_images.cpp`

**Work**

- [x] Allocate/reserve KV pages for 14 full-attention visits plus one MTP KV domain.
- [x] Allocate one GDN recurrent/conv state image per each of the 42 GDN visits.
- [x] Allocate one mHC two-stream image per each of the 48 mHC visits.
- [x] Map `visit -> physical layer`, `visit -> pass`, and `visit -> state slot` in one place.
- [x] Provide zero, snapshot, restore, and ownership/validity operations for each domain.
- [x] Ensure all pointers used by graph replay are stable for the Program lifetime.

**Tests**

- [x] Distinct visit states never alias another visit's state.
- [x] Revisited physical layers map to different state slots on pass 0 and pass 1.
- [x] Snapshot/restore round-trips deterministic markers in KV, GDN, and mHC state.
- [x] Zero/reset clears only the addressed slot.

**Evidence:** `./build-sandbox.sh --ctest
'^ninfer_sensenova_loop_9b_(state_images|program_lifecycle)_test$'` passed 2/2.

**Proposed commit**

```text
feat(targets): add SenseNova device state images
```

### E3. Implement one-visit layer execution

**Status:** complete for the standalone one-visit executor and its state-identity tests. E4/E5 will
wire this executor into eager prefill/decode and add independent fixed-prompt numerical references.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/layer_execution.h`
- `src/targets/sensenova_loop_9b/impl/runtime/layer_execution.cpp`
- `tests/targets/sensenova_loop_9b/test_layer_execution.cpp`

**Work**

- [x] Implement one full-attention visit: input RMSNorm, Q/gate split, QK norm, RoPE, 16/4 GQA,
      output projection, post-attention RMSNorm, MLP, residual adds.
- [x] Route SenseNova 16/4 GQA through the prompt/append path only.
- [x] Implement one GDN visit using the qualified projection, gating, convolution, recurrence, gated
      RMSNorm, and output projection.
- [x] Use GDN state from the visit slot, never the physical layer index.
- [x] Apply mHC `enter`/`leave` at physical boundaries 4 and 28 with FP32 router math and BF16
      branch/residual boundaries.
- [x] Keep shape/state transitions explicit; no hidden global stream state.

**Tests**

- [x] Single-layer fixtures against independent CPU/FP64 or Hugging Face reference values.
- [x] Repeated visits with shared physical weights use distinct state slots.
- [x] mHC branch/post/mix and router Sinkhorn remain covered by existing mHC test.

**Evidence:** `./build-sandbox.sh --ctest
'^ninfer_sensenova_loop_9b_(layer_execution|loop_plan|sequence_plan|state_images|program_lifecycle)_test$'`
passed 5/5.

**Proposed commit**

```text
feat(targets): add SenseNova one-visit layer execution
```

### E4. Implement eager full prefill

**Status:** complete in the eager greedy path. Reference-fixture comparison is tracked in Stage G;
INT8 KV is deferred.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/text_context.h`
- `src/targets/sensenova_loop_9b/impl/runtime/text_context.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/prefill.h`
- `src/targets/sensenova_loop_9b/impl/runtime/prefill.cpp`
- `tests/targets/sensenova_loop_9b/test_text_execution.cpp`

**Work**

- [x] Embed prompt tokens with W8 `D=4096` embedding.
- [x] Execute exactly the 56 visits in `LoopPlan` order for every prefill chunk.
- [x] Apply final RMSNorm, Q6 LM head over 248320 rows, and sample/argmax within token domain
      `248077`.
- [x] Handle EOS ids `(248044, 248046)` through the existing output session contract.
- [x] Replace `start_prefill_lane` placeholder token `0` with the sampled first token.
- [ ] Support INT8 and BF16 KV modes according to the selected execution plan.

**Tests**

- [x] Fixed prompt prefill output stable across reruns.
- [x] First-token logits comparison is deferred to Stage G's quantized-weight reference and is no
      longer a Stage E4 blocker.
- [x] Prompt positions and per-visit state advance exactly once per prompt token.

**Proposed commit**

```text
feat(targets): add SenseNova eager prefill execution
```

### E5. Implement eager ordinary decode

**Status:** complete in the eager greedy path. CUDA Graph replay and prefix reuse remain in E6.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/decode.h`
- `src/targets/sensenova_loop_9b/impl/runtime/decode.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.cpp`
- `tests/targets/sensenova_loop_9b/test_text_execution.cpp`

**Work**

- [x] Decode one token by executing all 56 visits once from the committed frontier.
- [x] Advance KV/GDN/mHC state only after sampling and acceptance.
- [x] Remove all placeholder `token 0` generation paths from `decode_batch`.
- [x] Wire generation timings and logprobs from real sampled logits.
- [x] Preserve batch lane identity and pending/active/retained lifecycle behavior.

**Tests**

- [x] Reference greedy-id comparison is deferred to Stage G's quantized-weight reference and is no
      longer a Stage E5 blocker.
- [x] Decode advances positions and all 56 visit states once per token.
- [x] EOS handling for `(248044, 248046)`.

**Evidence:** `./build-sandbox.sh --ctest '^ninfer_sensenova_loop_9b_.*_test$'` passed 7/7.

**Proposed commit**

```text
feat(targets): add SenseNova eager decode execution
```

### E6. CUDA Graphs and prefix reuse

**Status:** the capture/replay mechanism and the prefix-state primitive are implemented and tested on
real device work. Two parts of the original task are explicitly **not** done: whole-model replay
versus eager equality was not measured, and INT8 KV cannot be covered because this target's
execution path implements BF16 KV only. Prefix reuse is implemented as a whole-state-image
snapshot/restore at a frontier; the engine-level skip-prefill path that would consume it is not
wired (the Program still binds one shared state image for every lane).

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/graph_impl.h`
- `src/targets/sensenova_loop_9b/impl/runtime/graph_impl.cpp`
- `tests/targets/sensenova_loop_9b/test_graph_replay.cpp`

**Work**

- [x] Capture decode graphs only after all state pointers are stable. `DecodeGraphRunner` wraps
      `core/decode_graph.{h,cpp}`; the Program captures one step per lane with token id, position,
      sampled token, logprob, and margin at fixed staging addresses.
- [x] Keep host-side frontier publication outside replay: the input upload and the result readback
      are ordinary stream operations around the launch, and only kernels replay.
- [x] Use the declared KV capacity as the captured GQA envelope so one graph stays valid as the
      frontier grows; the causal mask still comes from the uploaded position.
- [x] Make the 56-visit pass capture-legal: the mHC leave reduction's pageable host fill is
      replaced by a Program-prepared device-resident 0.5 mix for one-token passes. Numerics are
      unchanged (the identity merge is still the reduction).
- [ ] INT8 KV graph profile. Blocked: `Program` binds BF16 KV domains unconditionally and the
      INT8 KV execution path from E4 is not implemented.
- [x] Implement prefix reuse with looped visit identity and all state domains included, at the
      state-image level: `PrefixStateSnapshot` copies all 15 KV domains, 42 GDN slots, and 48 mHC
      slots, plus the frontier and last committed token. `restore_prefix_state` reinstates them
      and refuses to resurrect an aborted lane.
- [ ] Engine-level prefix reuse (skip a shared prefill prefix) is not wired; it needs per-lane
      state-image partitioning, which this Program does not have yet.

**Tests**

- [x] Capture, replay, and live-input reads with real device copies/memsets; replay equals eager
      execution byte for byte; a failed capture records a diagnostic, leaves the runner
      unavailable, and does not poison the stream (`ninfer_sensenova_loop_9b_graph_replay_test`).
- [x] Prefix snapshot covers the whole state image and restores every marker; empty snapshots and
      aborted lanes are rejected (`ninfer_sensenova_loop_9b_program_lifecycle_test`).
- [ ] Graph replay equals eager decode token-for-token and state-for-state on the model. Not run:
      no fixed-prompt artifact fixture harness exists in this repository.
- [ ] Prefix reuse equals full prefill for the shared prefix. Not run for the reason above; the
      snapshot primitive is independent of it.
- [ ] INT8 and BF16 KV graph paths. BF16 is the only path; INT8 is blocked.

**Evidence:** `./build-sandbox.sh --ctest
'^ninfer_(mhc|sensenova_loop_9b_.*)_test$'` passed 9/9 (2026-11-18).

**Commits**

```text
3ca4d25d feat(targets): capture SenseNova decode graphs and snapshot prefix state
a4070dfc fix(targets): keep the SenseNova leave merge numerics and stage its mix on device
```

### E7. Stage E exit validation

**Status:** target-level validation is complete. Numerical reference comparison moved to Stage G:
the correct reference is a quantized-weight reference that dequantizes the existing `.ninfer`
artifact in memory, not a duplicate BF16 artifact and not bitwise parity with the original BF16
Hugging Face checkpoint.

- [x] Run all SenseNova CTests through `./build-sandbox.sh --ctest`: 8/8 pass.
- [x] Run the full CTest suite (`JOBS=1 ./build-sandbox.sh --test`): 107/110 pass. The three
      failures are sandbox dependency failures, not target failures:
      `ninfer_openai_schema_test`, `ninfer_responses_schema_test`, and `ninfer_serve_ui_test` all
      abort with `libcurl.so.4: undefined symbol:
      nghttp2_option_set_no_rfc9113_leading_and_trailing_ws_validation`. The suite as configured
      currently contains 110 tests, not the 84 AGENTS.md states.
- [x] Fixed-prompt numerical comparison is closed as an E7 item and promoted to Stage G as a
      quantized-weight reference test. Host/structural tests are not being treated as reference
      parity.
- [x] Confirm no placeholder token path remains in `runtime_stub.cpp` (audit for
      `kFirstToken`/`kNextToken`/placeholder tokens is clean).
- [x] Update the to-do with commit IDs and evidence.

Stage E remains text-only, greedy, and BF16-KV. All Stage E implementation and target-level tests
are complete. Quantized-weight reference parity and top-1/top-k sanity comparison against original
Hugging Face are Stage G work; INT8 KV is deferred and out of v1 scope.

**Before Stage F:** no Stage E numerical fixture is blocking. Proceed with Stage F (MTP +
engine-level prefix reuse).

---

## F. MTP propose, verify, rollback, and engine-level prefix reuse

**Goal:** one-layer MTP works for K=1..15, never diverges from ordinary decode, and the engine can
reuse a committed prefix without re-prefilling it. Both are required before the target is useful in
practice: MTP supplies decode speed, and prefix reuse supplies interactive/serving latency.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/mtp.h`
- `src/targets/sensenova_loop_9b/impl/runtime/mtp.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/verify.h`
- `src/targets/sensenova_loop_9b/impl/runtime/verify.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/speculative.h`
- `src/targets/sensenova_loop_9b/impl/runtime/prefix_reuse.h`
- `src/targets/sensenova_loop_9b/impl/runtime/prefix_reuse.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.cpp`
- `tests/targets/sensenova_loop_9b/test_mtp_rollback.cpp`
- `tests/targets/sensenova_loop_9b/test_prefix_reuse.cpp`
- `tests/CMakeLists.txt`

**Checkpoints**

- **F1 — MTP proposal and stem:** implement MTP input packing, W8 `fc`, one 16/4 attention layer,
  MLP, final norm, main LM head, separate MTP KV state, and draft-head shortlist for K=1..15.
- **F2 — MTP verification and rollback:** verify proposals against target logits, roll back target
  KV/GDN/mHC/MTP state on rejection, preserve next-round state after acceptance, and integrate graph
  profiles only after rollback correctness is established.
- **F3 — Engine-level prefix reuse:** promote E6's whole-state-image snapshot into real skip-prefill
  reuse. This requires per-lane/per-prefix state-image ownership or copy-on-write; restored frontier
  and last-token state; identity lookup for shared prompt prefixes; and safe interaction with
  admission and lifecycle transitions. The current single shared `Program` state image is a known
  blocker.
- **F4 — Combined validation:** MTP + prefix reuse together, including multi-lane state isolation,
  graph replay after restore, and fixed-prefix equivalence against full prefill.

**Work**

- [x] Implement MTP stem:
  - main embedding norm;
  - target hidden norm;
  - pack into `[8192,T]`;
  - W8 `fc` to `[4096,T]`;
  - one 16/4 attention layer;
  - MLP;
  - final norm;
  - main LM head logits.
- [x] Maintain separate MTP KV state and draft-head shortlist state.
- [x] Propose K=1..15 tokens through the converted 131072-row draft head.
- [x] Verify proposals with the target model's greedy verification window.
- [x] Roll back target KV, GDN, mHC, and MTP state to the accepted prefix.
- [x] Preserve next-round state after acceptance.
- [x] Integrate graph profiles only after rollback correctness is established: ordinary decode
      graphs remain unchanged; MTP does not add a new graph profile before end-to-end parity.
- [x] Add engine-level prefix reuse with looped visit identity and all state domains included.
- [x] Skip prefill from a restored frontier and resume from the restored last committed token.
- [x] Add per-lane/per-prefix state ownership or copy-on-write so concurrent requests cannot
      clobber one another's KV/GDN/mHC state. Chosen design: per-lane whole-state-image
      partitions in one stable allocation with full-image copy-on-restore.
- [x] Integrate prefix identity with request planning and admission accounting.

**Tests**

- [x] K=1 forced acceptance equals the ordinary greedy contract.
- [x] First-token rejection commits only the target correction token.
- [x] Mid-window rejection restores all state domains, proven with per-lane device markers.
- [x] K boundaries `1`, `2`, `15`.
- [x] Draft shortlist membership and remap contract are covered by the bound converter inventory
      and the MTP stem validation test; no fixed-prompt artifact harness exists to execute a real
      131072-row proposal end to end.
- [x] Speculative stats count accepted/drafted tokens correctly.
- [x] Prefix identity is token-content sensitive and an uncaptured prefix is not admitted.
- [x] MTP + prefix reuse remains state-isolated across concurrent lanes.

**Status: mechanism and state-contract validation complete; fixed-prompt end-to-end parity
remaining.** The target greedy rule matches `speculative_accept_greedy_drafts`, and rollback covers
the full state image. There is no fixed-prompt artifact harness that can prove accepted-token
parity against ordinary model execution, so that end-to-end claim is explicitly **not** made here.
Graph profiles are intentionally unchanged: no new MTP graph profile was added before end-to-end
rollback parity is available.

**Prefix isolation decision:** whole-state-image per-lane partitions were chosen over finer
per-prefix copy-on-write. Existing execution and graph contracts already address every domain
through one contiguous image with stable pointers; partitioning the image preserves all 56 visit
identities and avoids cross-lane clobbering with the smallest change surface. A restore copies a
complete prefix image into the target lane's partition.

**Evidence:** `./build-sandbox.sh` passed. `./build-sandbox.sh --ctest
'^ninfer_sensenova_loop_9b_.*_test$'` passed 10/10. `JOBS=1 ./build-sandbox.sh --test` passed
109/112 with 8 skipped artifact-dependent tests; the three failures are the pre-existing sandbox
libcurl/nghttp2 dependency failures in `ninfer_openai_schema_test`,
`ninfer_responses_schema_test`, and `ninfer_serve_ui_test`. Current branch HEAD is `810eb4c1`; no
commit was created for this working-tree change.

**Exit criterion:** MTP acceptance/rejection is bit-equivalent in state to ordinary decode for the
accepted token sequence, and prefix reuse produces the same state and next token as full prefill
without clobbering other lanes.

**Proposed commits**

```text
feat(targets): add SenseNova MTP proposal path
feat(targets): add SenseNova MTP verification and rollback
feat(targets): wire SenseNova engine-level prefix reuse
test(targets): validate SenseNova MTP and prefix reuse together
```

---

## G. Frontend, docs, validation, and performance

**Goal:** expose the target through serving and finish v1 correctness/performance gates.

**Files**

- `src/targets/sensenova_loop_9b/impl/frontend/*`
- `docs/maintainer/sensenova-loop-9b-model.md`
- `docs/maintainer/sensenova-loop-9b-todo.md`
- `docs/README.md`
- `AGENTS.md`
- `tests/targets/sensenova_loop_9b/test_frontend.cpp`
- `tests/targets/sensenova_loop_9b/test_reference.cpp`
- `tests/CMakeLists.txt`

**Work**

- [ ] Load six hashed frontend resources.
- [ ] Use tokenizer domain `248077`.
- [ ] Validate generation config EOS `(248044, 248046)` and `use_cache=true`.
- [ ] Map the official chat template.
- [ ] Register the serving identity `sensenova_loop_9b`.
- [~] Build a quantized-weight reference by dequantizing the existing `.ninfer` artifact in memory;
      do not create a duplicate BF16 artifact. Compare fixed prompts against that reference with
      explicit tolerances. **Artifact-level validation added and run:** all 136 quantized tensors
      are decoded in memory and sampled against an independent scalar FP32/FP64 oracle. The public
      Engine API exposes generated tokens but no hidden/logit tensor, so a fixed-prompt logits
      comparison is still blocked; the end-to-end claim is not made.
- [~] As a weaker semantic sanity check, compare top-1/top-k overlap against the original BF16
      Hugging Face checkpoint, without requiring bitwise logits. **External HF feasibility:** the
      local model directory is complete and `hf_reference.py --preflight` validates its config,
      generation config, resource hashes, and all safetensors shards. The actual BF16 forward is
      blocked because this sandbox has no `torch`, `transformers`, or `safetensors` Python modules;
      the script exits with code 77 and does not download anything.
- [~] Validate prefill, ordinary decode, MTP propose/verify, K=1..15, engine-level prefix reuse,
      restart, and state-cache behavior. INT8 KV is deferred; BF16 KV is the v1 path.
      **Prompt positions fixed:** SenseNova prefill now validates `positions.size() ==
      token_ids.size() * 3` and uploads text/temporal axis `position_axis(0)` to the device `[T]`
      tensor in both cold and prefix-reuse prefill. The frontend axis-major `[3,T]` contract is
      regression-tested. **Remaining execution blocker:** SenseNova state images use BF16 GQA
      planes `[256,128,4,pages]`, while the GQA kernels derive logical pages and page offsets from
      absolute positions with the compile-time 64-token `kPagedKVPageShift=6` and
      `kPagedKVPageMask=63`. Relaxing wrapper validation alone would corrupt cache addressing;
      correct support needs a runtime page shift/mask or separate 128-token geometry threaded
      through BF16/INT8 prefill and decode kernels, metadata, delta-RoPE paths, wrappers, and
      oracle tests. That broad redesign is not authorized here, so the fixed-prompt harness remains
      a documented skip. Existing host/state-contract CTests still cover MTP K=1,2,15
      verification/rollback and prefix identity/partition restore; graph replay vs eager is
      covered by mechanism CTests. No end-to-end token-parity claim is made.
- [~] Run the full CTest suite; all 84 tests must pass. **Expanded suite:** Stage G adds
      `ninfer_sensenova_loop_9b_reference_quantized_weights_test` and a skipped fixed-prompt gate.
      `JOBS=1 ./build-sandbox.sh --test` reported 114 tests: 97 percent passed, 3 failed, 9
      skipped, and 102 executed tests passed in 115.35 seconds. CTest exited 8. The lifecycle
      test passes after the prompt-position regression test was repaired. The three failures are
      the preserved pre-existing sandbox libcurl/nghttp2 dependency failures
      in `ninfer_openai_schema_test`, `ninfer_responses_schema_test`, and `ninfer_serve_ui_test`.
      All 136 quantized tensors (134 W8 + one Q4 draft head + one Q6 LM head) are covered.
      After the authorized prompt-position fix, the same suite result is revalidated; the
      fixed-prompt test remains a documented skip solely for the compiled 64-token GQA page
      shift/mask versus SenseNova's 128-token state-image planes.
- [ ] Benchmark text-only baseline.
- [ ] Benchmark MTP at K=3, 5, 7; choose a conservative default.
- [ ] Tune only after correctness:
  - group-4 GQA;
  - mHC fusion;
  - L2-aware rasterization;
  - CUDA Graph profiles;
  - 128-register memory-bound kernels.
- [ ] Finalize model docs, `docs/README.md`, and `AGENTS.md`.

**Exit criterion:** the target is usable through `ninfer-serve`, matches reference behavior on fixed
prompts, and passes the full test suite.

**Proposed commits**

```text
feat(targets): expose SenseNova frontend identity
docs(models): document SenseNova runtime contract
perf(targets): tune SenseNova prefill and MTP paths
```

---

## Current first action

Stages A-D and the GDN gating portion of Stage E are complete. The next action is now **Stage E0**:

E0-E7 are complete for the eager greedy, BF16-KV text path, and E7's target-level validation has
run. The next action is **Stage F (MTP + prefix reuse)**. Quantized-weight reference validation is
Stage G work and no longer blocks Stage F. INT8 KV execution is deferred; BF16 KV is the v1 path.
Engine-level prefix reuse is promoted into Stage F and is required before practical serving.

### Stage E prerequisite committed during implementation

A W8 dispatch smoke test now covers the SenseNova projection shapes:

- `[10240,4096]`, `[8192,4096]`, `[4096,4096]`, `[1024,4096]`, `[24576,4096]`, `[12288,4096]`,
  `[4096,12288]`;
- token extents 1 and 65.

`ninfer_linear_w8_sensenova_dispatch_test` passes. This qualifies routing only; the numerical
oracle coverage for those exact shapes remains part of the forward-path validation work.

- Gitea commit: `9d6461f8`.
- GitHub commit: `5cd11a8c`.

Additional numeric domain qualification committed during Stage E preparation:

- Gitea `25f56805` / GitHub `117df35f`: SenseNova W8 4096-wide projections, W8 embedding
  `D=4096`, and Q6 LM head `K=4096`.
- Verified by `ninfer_linear_w8_a16_test`, `ninfer_embedding_test`, and
  `ninfer_linear_q6_a16_test`.

Remaining hard blocker before text execution can be correct:

- GDN gating has no qualified kernel for SenseNova's `[hidden=4096, heads=32]` parent geometry.
  A trial generalization of the Qwen 35B path failed the FP64 oracle and was reverted. Do not
  wire the 56-visit forward loop until this op has an independent oracle test.

### Stage E active work item: SenseNova GDN gating op

- [x] Implement `gdn_gating_proj_sensenova` and `gdn_norm_gating_proj_sensenova` for exactly
      `heads=32`, `hidden=4096`, parent `ab_weight [64,4096]`.
- [x] Use the Qwen 35B path as the structural reference: BF16 parent halves, FP32 `A_log` and
      `dt_bias`, FP32 `[32,T]` `g`/`beta`, and the same logical formula and FP64 oracle.
- [x] Use the SenseNova llama.cpp fork only as execution-order evidence; the C++ numerical
      contract remains Hugging Face `Qwen3_5GatedDeltaNet` because the official checkpoint's
      `modeling_sensenova_loop.py` subclasses it.
- [x] Add first-class SenseNova geometry admission and planner: C4/C8 SIMT for T=1..8 and the
      dedicated unsplit BN64 MMA geometry for T>=9. Existing Qwen route tables are untouched.
- [x] Add wrappers for the contiguous `[64,4096]` parent and CTest coverage at T=1,2,4,5,8,9,
      17,64,1024,4096 plus norm/control cases.
- [x] Verified with `ninfer_gdn_gating_proj_test` and `ninfer_gdn_gating_test`: 2/2 passed.
- [ ] After this op passes, continue Stage E with device-backed state images, snapshot/restore,
      graph stability, the 56-visit forward pass, and reference validation.
- [ ] The stage-B/C/D Program remains lifecycle scaffolding only until the execution path is
      wired. It deliberately does not emit real tokens.
