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

**Status:** in progress. The GDN input projection is complete; W8 SwiGLU remains.

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
- [ ] Add W8 `linear_swiglu` for `[24576,4096] -> [12288,T]`. A direct reuse of the generic
      rowsplit MMA epilogue failed to compile in this TU, and the current exact/decode pair kernels
      hard-code `[12288,2048]`. The next implementation must template those pair kernels on
      `(intermediate, hidden)` and qualify T=1..48 explicitly; do not fake the profile by routing
      through unrelated W8 linear plus external SiLU.
- [ ] Keep the existing `[12288,2048]` Qwen 35B profiles unchanged.
- [ ] Verify every added profile against an independent FP64 exact-decode oracle.

**Tests**

- [x] `ninfer_gdn_input_proj_test` covers T=1, small T, MTP-like widths, and prefill widths.
- [ ] `ninfer_linear_swiglu_w8_a16_test` covers SenseNova gate/up shape at equivalent extents.
- [ ] Existing Qwen GDN/MLP tests remain green.

**Evidence for completed GDN projection:** `./build-sandbox.sh --ctest
'^ninfer_(gdn_input_proj|linear_w8_sensenova_dispatch|linear_w8_a16)_test$'` passed 3/3.

**Proposed commits**

```text
feat(ops): add SenseNova GDN input projection
feat(ops): add SenseNova W8 SwiGLU profile
```

### E2. Build device-backed state images

**Status:** not started.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/state_images.h`
- `src/targets/sensenova_loop_9b/impl/runtime/state_images.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.{h,cpp}`
- `tests/targets/sensenova_loop_9b/test_state_images.cpp`

**Work**

- [ ] Allocate/reserve KV pages for 14 full-attention visits plus one MTP KV domain.
- [ ] Allocate one GDN recurrent/conv state image per each of the 42 GDN visits.
- [ ] Allocate one mHC two-stream image per each of the 48 mHC visits.
- [ ] Map `visit -> physical layer`, `visit -> pass`, and `visit -> state slot` in one place.
- [ ] Provide zero, snapshot, restore, and ownership/validity operations for each domain.
- [ ] Ensure all pointers used by graph replay are stable for the Program lifetime.

**Tests**

- [ ] Distinct visit states never alias another visit's state.
- [ ] Revisited physical layers map to different state slots on pass 0 and pass 1.
- [ ] Snapshot/restore round-trips deterministic markers in KV, GDN, and mHC state.
- [ ] Zero/reset clears only the addressed slot.

**Proposed commit**

```text
feat(targets): add SenseNova device state images
```

### E3. Implement one-visit layer execution

**Status:** not started.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/layer_execution.h`
- `src/targets/sensenova_loop_9b/impl/runtime/layer_execution.cpp`
- `tests/targets/sensenova_loop_9b/test_layer_execution.cpp`

**Work**

- [ ] Implement one full-attention visit: input RMSNorm, Q/gate split, QK norm, RoPE, 16/4 GQA,
      output projection, post-attention RMSNorm, MLP, residual adds.
- [ ] Route SenseNova 16/4 GQA through the prompt/append path only.
- [ ] Implement one GDN visit using the qualified projection, gating, convolution, recurrence, gated
      RMSNorm, and output projection.
- [ ] Use GDN state from the visit slot, never the physical layer index.
- [ ] Apply mHC `enter`/`leave` at physical boundaries 4 and 28 with FP32 router math and BF16
      branch/residual boundaries.
- [ ] Keep shape/state transitions explicit; no hidden global stream state.

**Tests**

- [ ] Single-layer fixtures against independent CPU/FP64 or Hugging Face reference values.
- [ ] Repeated visits with shared physical weights use distinct state slots.
- [ ] mHC branch/post/mix and router Sinkhorn remain covered by existing mHC test.

**Proposed commit**

```text
feat(targets): add SenseNova one-visit layer execution
```

### E4. Implement eager full prefill

**Status:** not started.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/text_context.h`
- `src/targets/sensenova_loop_9b/impl/runtime/text_context.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/prefill.h`
- `src/targets/sensenova_loop_9b/impl/runtime/prefill.cpp`
- `tests/targets/sensenova_loop_9b/test_text_execution.cpp`

**Work**

- [ ] Embed prompt tokens with W8 `D=4096` embedding.
- [ ] Execute exactly the 56 visits in `LoopPlan` order for every prefill chunk.
- [ ] Apply final RMSNorm, Q6 LM head over 248320 rows, and sample/argmax within token domain
      `248077`.
- [ ] Handle EOS ids `(248044, 248046)` through the existing output session contract.
- [ ] Replace `start_prefill_lane` placeholder token `0` with the sampled first token.
- [ ] Support INT8 and BF16 KV modes according to the selected execution plan.

**Tests**

- [ ] Fixed prompt prefill output stable across reruns.
- [ ] First-token logits compare against a Hugging Face reference fixture.
- [ ] Prompt positions and per-visit state advance exactly once per prompt token.

**Proposed commit**

```text
feat(targets): add SenseNova eager prefill execution
```

### E5. Implement eager ordinary decode

**Status:** not started.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/decode.h`
- `src/targets/sensenova_loop_9b/impl/runtime/decode.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime_stub.cpp`
- `tests/targets/sensenova_loop_9b/test_text_execution.cpp`

**Work**

- [ ] Decode one token by executing all 56 visits once from the committed frontier.
- [ ] Advance KV/GDN/mHC state only after sampling and acceptance.
- [ ] Remove all placeholder `token 0` generation paths from `decode_batch`.
- [ ] Wire generation timings and logprobs from real sampled logits.
- [ ] Preserve batch lane identity and pending/active/retained lifecycle behavior.

**Tests**

- [ ] Ordinary decode of a fixed prefix matches Hugging Face generated ids for greedy settings.
- [ ] Decode advances positions and all 56 visit states once per token.
- [ ] EOS handling for `(248044, 248046)`.

**Proposed commit**

```text
feat(targets): add SenseNova eager decode execution
```

### E6. CUDA Graphs and prefix reuse

**Status:** not started.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/graph_impl.h`
- `src/targets/sensenova_loop_9b/impl/runtime/graph_impl.cpp`
- `tests/targets/sensenova_loop_9b/test_graph_replay.cpp`

**Work**

- [ ] Capture decode graphs only after all state pointers are stable.
- [ ] Use stable graph profiles for BF16/INT8 KV and supported prefill/decode token extents.
- [ ] Keep host-side frontier publication outside replay unless it is address-stable.
- [ ] Implement prefix reuse with looped visit identity and all state domains included.

**Tests**

- [ ] Graph replay equals eager decode token-for-token and state-for-state.
- [ ] Prefix reuse equals full prefill for the shared prefix.
- [ ] INT8 and BF16 KV graph paths.

**Proposed commit**

```text
feat(targets): add SenseNova decode graphs and prefix reuse
```

### E7. Stage E exit validation

**Status:** not started.

- [ ] Run all SenseNova CTests through `./build-sandbox.sh --ctest`.
- [ ] Run the full CTest suite; all tests must pass or unrelated missing fixtures must be explicitly
      identified.
- [ ] Compare fixed prompts against Hugging Face for prefill and ordinary decode.
- [ ] Confirm no placeholder token path remains in `runtime_stub.cpp`.
- [ ] Update the to-do with commit IDs and evidence.

**Exit criterion:** ordinary text generation is numerically and behaviorally correct with fixed loop
semantics, with no MTP involved.

---

## F. MTP propose, verify, and rollback

**Goal:** one-layer MTP works for K=1..15 and never diverges from ordinary decode.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/mtp.h`
- `src/targets/sensenova_loop_9b/impl/runtime/mtp.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/verify.h`
- `src/targets/sensenova_loop_9b/impl/runtime/verify.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/speculative.h`
- `tests/targets/sensenova_loop_9b/test_mtp_rollback.cpp`
- `tests/CMakeLists.txt`

**Work**

- [ ] Implement MTP stem:
  - main embedding norm;
  - target hidden norm;
  - pack into `[8192,T]`;
  - W8 `fc` to `[4096,T]`;
  - one 16/4 attention layer;
  - MLP;
  - final norm;
  - main LM head logits.
- [ ] Maintain separate MTP KV state and draft-head shortlist state.
- [ ] Propose K=1..15 tokens through the converted 131072-row draft head.
- [ ] Verify proposals with the target model.
- [ ] Roll back target KV, GDN, mHC, and MTP state to the accepted prefix.
- [ ] Preserve next-round proposal state after acceptance.
- [ ] Integrate graph profiles only after rollback correctness is established.

**Tests**

- [ ] K=1 forced acceptance equals ordinary decode.
- [ ] First-token rejection equals ordinary decode of the accepted token only.
- [ ] Mid-window rejection restores all state domains.
- [ ] K boundaries `1`, `2`, `15`.
- [ ] Draft shortlist membership.
- [ ] Speculative stats count accepted/drafted tokens correctly.

**Exit criterion:** MTP acceptance and rejection paths are bit-equivalent in state to ordinary
decode for the accepted token sequence.

**Proposed commits**

```text
feat(targets): add SenseNova MTP proposal path
feat(targets): add SenseNova MTP verification and rollback
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
- [ ] Compare fixed prompts against the official Hugging Face reference.
- [ ] Validate prefill, ordinary decode, MTP propose/verify, K=1..15, INT8/BF16 KV, restart, and
      state-cache behavior.
- [ ] Run the full CTest suite; all 84 tests must pass.
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

E0 is complete. The next action is **Stage E1**:

1. Add the SenseNova W8 GDN input-projection profile `[12288,4096] -> qkv[8192,T], z[4096,T]`.
2. Add its projection/conv/snapshot form against an independent FP64 oracle.
3. Add SenseNova W8 SwiGLU `[24576,4096] -> [12288,T]`.
4. Keep the Qwen `[12288,2048]` profiles unchanged.
5. Build and run the affected op CTests with `./build-sandbox.sh`.
6. Commit each verified profile separately and mirror on the GitHub lineage.

Do not begin E2 until E1's exact-shape op profiles have numerical oracle tests.

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
