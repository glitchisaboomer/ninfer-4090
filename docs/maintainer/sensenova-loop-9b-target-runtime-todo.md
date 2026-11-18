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

**Goal:** generate text correctly through the fixed 56-visit schedule without MTP.

**Files**

- `src/targets/sensenova_loop_9b/impl/runtime/text_context.h`
- `src/targets/sensenova_loop_9b/impl/runtime/text_context.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/prefill.h`
- `src/targets/sensenova_loop_9b/impl/runtime/prefill.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/decode.h`
- `src/targets/sensenova_loop_9b/impl/runtime/decode.cpp`
- `src/targets/sensenova_loop_9b/impl/runtime/mhc_state.h`
- `src/targets/sensenova_loop_9b/impl/runtime/graph_impl.h`
- `tests/targets/sensenova_loop_9b/test_text_execution.cpp`
- `tests/targets/sensenova_loop_9b/test_graph_replay.cpp`
- `tests/CMakeLists.txt`

**Work**

- [ ] Execute visit schedule exactly once per generated token.
- [ ] Use 16/4 GQA for full-attention visits; only the prompt/append path for this geometry.
- [ ] Use GDN state slot from the visit, never the physical layer index.
- [ ] Apply mHC `enter`/`leave` stream-shape transitions at physical boundaries 4 and 28.
- [ ] Use FP32 mHC router math with BF16 branch/residual boundary.
- [ ] Apply per-visit RMSNorm/QK-norm weights bound from the physical layer.
- [ ] Sample through token domain `248077`; LM head has `248320` rows.
- [ ] Capture decode CUDA Graphs only after all state pointers are stable.
- [ ] Implement prefix reuse with looped state identity.

**Tests**

- [ ] Fixed prompt prefill output stable across reruns.
- [ ] Ordinary decode advances positions and all 56 visit states once per token.
- [ ] Graph replay equals eager decode.
- [ ] Prefix reuse equals full prefill for the shared prefix.
- [ ] INT8 and BF16 KV paths.
- [ ] EOS handling for `(248044, 248046)`.

**Exit criterion:** ordinary text generation is numerically and behaviorally correct with fixed loop
semantics.

**Proposed commits**

```text
feat(targets): add SenseNova fixed-plan text execution
feat(targets): add SenseNova decode graphs and prefix reuse
```

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

Start with **A**:

1. Create `src/targets/sensenova_loop_9b/impl/loop_plan.h`.
2. Create `src/targets/sensenova_loop_9b/impl/state_layout.h` and `.cpp`.
3. Add `tests/targets/sensenova_loop_9b/test_loop_plan.cpp` and `test_state_layout.cpp`.
4. Build only those tests with `./build-sandbox.sh`.
5. Commit as `feat(targets): add SenseNova fixed loop plan and state layout`.
6. Mirror the commit on the GitHub lineage with the required author identity.

Do not begin the registry or program work until the loop/state tests pass.
