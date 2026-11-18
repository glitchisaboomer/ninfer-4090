# SenseNova-Loop-9B implementation to-do

This is the work checklist for native text + MTP support for SenseNova-Loop-9B. Keep v1 scope
fixed: text only, one fixed two-pass loop, mHC, 16/4 GQA, 16/32 GDN, one-layer MTP, 262K native
context, no vision, no 512K context, no dynamic loop counts.

## Reference material

### Local model and official reference

- Model directory: `/home/user/models/sensenova/SenseNova-Loop-9B`
- Official config: `/home/user/models/sensenova/SenseNova-Loop-9B/config.json`
- Official modeling code: `modeling_sensenova_loop.py` in the model directory
- Official index: `model.safetensors.index.json`
- Official draft weights: `mtp.safetensors`

Useful locations in the official modeling code:

- `execution_plan()` at line 25: validates fixed loop config and returns the 56-entry plan.
- `sinkhorn()` at line 77: 20-pass log-space row/column normalization.
- `MHCRouter` at line 86: FP32 router parameters, logits, pre/post gates, mix, and merge.
- `enter()` at line 146 and `leave()` at line 155: residual stream shape at mHC boundaries.
- `_VisitCache` at line 173: one independent cache view per logical visit.
- `SenseNovaLoopTextModel.forward()` around line 207: the looped execution driver.
- `MHCDecoderLayer` at line 332: sublayer-level mHC application.
- `_configure_text_model()` at line 354: attaches mHC routers to physical layers 4..27.

### llama.cpp reference fork

- Repository: `https://github.com/GuideboardLabs/llama.cpp-sensenova-loop`
- Pinned commit: `a7f77a788893ef2cb17c82b9ac0cb1279bcd0918`
- Local clone used during review: `/tmp/llamacpp-sensenova`

Useful locations in the fork:

- `conversion/sensenova.py:34`: `execution_plan()` for 32 physical to 56 virtual blocks.
- `conversion/sensenova.py:50`: mHC kinds per virtual block.
- `conversion/sensenova.py:76`: strict rejection of non-identity mHC settings.
- `conversion/sensenova.py:100`: 32-physical-layer to 56-block conversion log.
- `conversion/sensenova.py:108`: stores 56 virtual blocks while setting MTP count.
- `conversion/sensenova.py:141`: looped weights are stored once and aliased on later visits.
- `src/models/sensenova-loop.cpp:25`: one block per visit plus residual stream routers.
- `src/models/sensenova-loop.cpp:149`: second loop visit owns no new tensors.
- `src/models/sensenova-loop.cpp:182`: re-entered blocks reuse the first visit's weights.
- `src/models/sensenova-loop.cpp:218`: expanded graph for all virtual blocks.
- `src/models/sensenova-loop.cpp:626`: one-layer Qwen3.5 MTP graph.
- `src/models/sensenova-loop.cpp:660`: MTP embedding falls back to main `tok_embd`.
- `src/models/sensenova-loop.cpp:787`: MTP head falls back to main output head.
- `ggml/src/ggml-cpu/ops.cpp:12526`: fused CPU mHC route.
- `ggml/src/ggml-cpu/ops.cpp:12607`: fused CPU mHC merge.
- `ggml/src/ggml-cuda/mhc.cu:1`: CUDA route/merge kernels.
- `ggml/include/ggml.h:579`: `GGML_OP_MHC_ROUTE` and `GGML_OP_MHC_MERGE`.
- `tests/test-backend-ops.cpp:5364`: mHC route test with strided stream views.
- `tests/test-backend-ops.cpp:5391`: mHC merge test.
- `tests/test-backend-ops.cpp:11157`: token counts 1, 7, and 512 for route tests.
- `tests/test-backend-ops.cpp:11169`: token counts 1, 7, and 512 for merge tests.

The fork is a reference for structure and for independent numerical checks. Do not copy its
all-FP32 residual approximation blindly. The Hugging Face boundary keeps branch/mix application at
BF16 while router math stays FP32.

## To-do

### 0. Branch and scope

- [x] Rename this branch to `feat/sensenova-loop-9b`.
- [x] Create matching branches on Gitea and GitHub for the implementation.
- [x] Keep v1 scope fixed:
  - text only;
  - fixed 32 physical / 56 logical layer plan;
  - mHC two-stream routing;
  - 16/4 GQA;
  - 16/32 GDN;
  - one-layer MTP;
  - 262K native context first;
  - no vision;
  - no 512K context;
  - no dynamic loop counts.

Dynamic loop counts mean runtime-selected loop depth. The official
`execution_plan()` requires `counts == [eval_count]`; SenseNova-Loop-9B uses `eval_count=2`, so
the only supported plan is:

```text
0,1,2,3, 4..27, 4..27, 28,29,30,31
```

### 1. Contract documentation

- [ ] Add `docs/maintainer/sensenova-loop-9b-model.md` with:
  - config assertions from the official `config.json`;
  - source prefix map from `model.safetensors.index.json`;
  - exact tensor shapes for text layers, GDN layers, mHC routers, and MTP;
  - fixed loop plan and per-visit cache identity;
  - mHC equations and precision boundary;
  - YaRN schedule;
  - EOS, tokenizer, and chat-template behavior.
- [ ] Update `docs/README.md` after the target exists.
- [ ] Update `AGENTS.md` only when the new identity is actually supported.

### 2. Converter

- [x] Add `tools/convert/sensenova_loop_9b/` with strict config validation.
- [x] Normalize `model.language_model.*` source prefixes.
- [x] Ignore vision tensors only through an explicit whitelist; reject unknown tensors.
- [x] Hard-fail on unexpected shapes or config fields instead of adapting silently.
- [x] Stream shards one at a time and avoid copying the 19 GB source tree.
- [x] Bind physical text weights once. The artifact stores one physical copy of each of the 32
      layers; the fixed loop is a runtime contract.
- [x] Convert mHC routers as FP32:
  - `mhc_attn.weight [8,8192]`, `mhc_attn.bias [8]`;
  - `mhc_ffn.weight [8,8192]`, `mhc_ffn.bias [8]`;
  - only for physical layers 4..27.
- [x] Convert MTP from `mtp.safetensors`:
  - assert `mtp_use_dedicated_embeddings=false`;
  - use the main embedding and main LM head;
  - map `mtp.fc.weight [4096,8192]`, norms, one 16/4 attention layer, and one 12288-wide MLP.
- [x] Add converter fixture tests using tiny generated fixtures, not the real checkpoint.

The converter has been run once against the licensed checkpoint with the fixed output path:

```bash
python -m tools.convert.sensenova_loop_9b.convert \
  --model /home/user/models/sensenova/SenseNova-Loop-9B \
  --out /home/user/models/sensenova/SenseNova-Loop-9B.ninfer \
  --device cuda
```

The result is a 9.2 GB artifact with 447 objects and a verified structural summary. Do not create a
second conversion output; rerun the same path only with explicit approval.

Converter commits:

- Gitea `a84ec296`: `feat(convert): add SenseNova-Loop-9B text and MTP converter`.
- GitHub `aead052a`: same change with the required author identity.

### 3. Ops

- [x] Add closed mHC ops:
  - route op: two-stream RMS norm, 8 FP32 logits, pre-gates, post-gates, 20 log-space Sinkhorn
    passes, routed branch;
  - merge op: stream mix and post-gated branch add.
- [x] Use the official `MHCRouter` as the correctness contract:
  - official modeling code `sinkhorn()` at line 77;
  - `MHCRouter.forward()` around line 86;
  - `MHCRouter.merge()` at line 138.
- [x] Use the fork's route/merge kernels as an engineering reference:
  - `ggml/src/ggml-cpu/ops.cpp:12526`;
  - `ggml/src/ggml-cpu/ops.cpp:12607`;
  - `ggml/src/ggml-cuda/mhc.cu:1`.
- [x] Add independent FP32/FP64 oracle tests. Match HF precision boundaries: FP32 router/math,
      BF16 branch and mix storage/application.
- [x] Test token extents 1, 7, and 512, with and without strided stream views. Mirror the fork's
      cases at `tests/test-backend-ops.cpp:11157` through `11174`.
- [x] Add/confirm GQA geometry 16 Q heads / 4 KV heads / head dim 256.
- [x] Add MTP proposal geometry for hidden 4096 and 16/4 draft attention.

Ops commits and verification:

- Gitea `75f602c6` / GitHub `a5fc09d0`: mHC route and merge; `ninfer_mhc_test` passes.
- Gitea `1516d1e3` / GitHub `fb9d0782`: 16/4 GQA and RoPE geometry; `ninfer_gqa_attention_test`
  and `ninfer_rope_test` pass.
- Gitea `0dfee239` / GitHub `ea0a8226`: MTP geometry; `ninfer_mtp_pack_test` passes.

The 16/4 GQA geometry runs through the prompt attention path. The small-T INT8 kernel does not
instantiate group size 4, so SenseNova is excluded from that route rather than silently using the
35B 16/2 geometry. Geometry resolution now uses both Q-head and KV-head counts.

### 4. Target runtime

- [ ] Add `src/targets/sensenova_loop_9b/` package:
  - config;
  - bindings;
  - variant;
  - loop plan;
  - sequence and request planning;
  - frontend resources;
  - KV/GDN/mHC state mapping.
- [ ] Implement the fixed loop plan from official `execution_plan()` at line 25.
- [ ] Keep independent KV/GDN state per logical visit, following official `_VisitCache` at line 173
      and the fork's per-virtual-block state at `src/models/sensenova-loop.cpp:218`.
- [ ] Bind repeated physical weights once; never duplicate looped weight bytes.
- [ ] Integrate mHC `enter`/`leave` shape changes at physical boundaries 4 and 28, following
      official `enter()` at line 146 and `leave()` at line 155.
- [ ] Bind MTP draft weights and allocate separate MTP KV/cache state.
- [ ] Verify speculative accept/reject rolls back target GDN/KV state and mHC stream state.
- [ ] Add target registry identity and artifact validation.

### 5. Build and tests

- [x] Add CMake entries for the new ops and tests.
- [x] Build the qualified op targets with `./build-sandbox.sh`.
- [x] Run the new op CTests: mHC, 16/4 GQA, RoPE 16/4, and MTP pack D=4096.
- [ ] Add the new target package, then run the full CTest suite.
- [x] Add tests completed so far:
  - config rejection for non-fixed loop counts;
  - mHC numerical oracle;
  - mHC strided stream views;
  - converter fixture tests;
  - MTP pack D=4096 test.
- [ ] Add tests still pending:
  - loop plan equals the expected 56 entries;
  - converter prefix normalization against real source keys;
  - converter strict shape rejection against a full fixture;
  - MTP binding shape validation;
  - prefix reuse with looped visits;
  - graph replay state correctness.

### 6. Correctness validation

- [x] Convert the real checkpoint to `.ninfer` without duplicating the 19 GB model directory.
- [x] Report expected artifact size before writing if conversion needs more than a few GB scratch.
- [ ] Compare against official Hugging Face modeling code or a served reference on fixed prompts.
- [ ] Validate:
  - prefill;
  - ordinary decode;
  - MTP propose/verify;
  - K=1..15;
  - INT8/BF16 KV;
  - restart and state-cache behavior.
- [ ] Do not call the target done if text output or MTP acceptance diverges from the reference.

### 7. Performance

- [ ] Benchmark text-only baseline.
- [ ] Benchmark MTP at K=3, 5, and 7; select a conservative default.
- [ ] Tune:
  - mHC launch fusion;
  - group-4 GQA;
  - L2-aware weight rasterization;
  - CUDA Graph profiles;
  - 128-register tuning for memory-bound kernels.

### 8. Final docs and lineage commits

- [ ] Update `AGENTS.md`, `docs/README.md`, model cards, and release notes.
- [ ] Commit implementation on the Gitea lineage.
- [ ] Graft the same content to the GitHub lineage with the required author identity.
- [ ] Run the protocol secret checks before pushing GitHub.
