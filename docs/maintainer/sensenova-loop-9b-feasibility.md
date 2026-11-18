# SenseNova-Loop-9B Feasibility and Integration Design

## Status and decision

SenseNova-Loop-9B **can be adapted to this engine**, but not by changing a configuration field or
by loading the existing Qwen3.6 artifact. It is a new target package in the same Qwen3.5 operator
family, plus two genuinely new target-level semantics:

1. a shared-weight **loop execution plan** with independent state per visit;
2. **mHC two-stream residual routing** inside the looped block, including FP32 log-space Sinkhorn
   normalization.

The existing GDN, GQA, RoPE, RMSNorm, MLP, MTP speculative-decoding machinery, scheduler, KV cache,
CUDA Graph, state-cache, artifact, and serving layers are reusable. The fixed one-pass 27B/35B
target schedules are not. For text serving with MTP this is a larger but well-bounded target
integration. Image/video input and 512K context remain separate work streams; MTP is part of the
first performance-complete target because decode speed is a primary product goal.

This document is a design study only. No engine code was changed.

## 1. Evidence and provenance

The canonical repository at `sensenova/SenseNova-Loop-9B` is gated, so its `config.json`,
`modeling_sensenova_loop.py`, and weight shards cannot be read anonymously from this machine.
The following public, ungated sources were used to derive the model contract:

- Hugging Face model metadata for `sensenova/SenseNova-Loop-9B`: architecture
  `SenseNovaLoopForConditionalGeneration`, `model_type: sensenova_loop`, BF16 parameter count
  9,409,812,208, 3,151,744 FP32 parameters, 11 shards, and a separate `mtp.safetensors`.
- The public `sensenova/SenseNova-Loop-9B-Preview` repository: `config.json`, `generation_config`,
  tokenizer files, weight index, and `modeling_sensenova_loop.py`.
- The public BF16 re-upload `nativ-community/SenseNova-Loop-9B-bf16`: config, weight index,
  tokenizer and the **same** `modeling_sensenova_loop.py` file hash as the Preview repository
  (`26e11d35…5b207d979a5b`). Its BF16 parameter total and FP32 router parameter total match the
  gated release metadata exactly.
- The public re-upload README and the gated release README, which agree that the release has 32
  physical decoder layers, loop interval `[4,28)`, two passes, two-stream mHC, and 56 layer
  executions.
- SGLang PR 47281 and vLLM PR 64832, both authored for this architecture, which independently
  confirm the execution plan, per-visit cache identity, FP32 mHC routers, and text-only support
  boundary of early engine integrations.
- `GuideboardLabs/llama.cpp-sensenova-loop`, a working llama.cpp fork at commit `a7f77a7`, which
  contains a complete text-only architecture implementation, GGUF converter changes, new GGML
  mHC CPU/CUDA ops, and backend op tests.

The derived config has now been revalidated against the licensed local checkpoint at
`/home/user/models/sensenova/SenseNova-Loop-9B`. It matches the public mirror exactly on the
registered fields: 32 layers, hidden 4096, intermediate 12288, 16/4 attention, 16/32 GDN,
`loop_settings={start:4,end:28,eval_count:2,counts:[2]}`, two-stream 20-iteration mHC,
`pre_scale=1.0`, `fixed_identity=false`, YaRN factor 2 over 262144 native context, and
`mtp_use_dedicated_embeddings=false`. The local `modeling_sensenova_loop.py` is byte-identical to
the public custom code. The official index uses `model.language_model.*` / `model.visual.*`
prefixes. The official `mtp.safetensors` contains the same 15 tensors as the public Preview draft,
with HF-linear orientation (`fc.weight` shape `[4096,8192]`). No blocking shape discrepancy was
found. The converter must still hard-fail on any unexpected config member or tensor shape instead
of silently adapting.

### 1.1 What the llama.cpp fork proves

The fork is the strongest available implementation evidence that text-only SenseNova-Loop is
practical. Its converter expands the 32-physical-layer checkpoint into a 56-block GGUF, writes one
copy of the looped weights, and lets the loader alias the second visit's layer records onto the
first visit's tensors. Each virtual block owns its own KV/GDN state. That is the same logical/physical
split recommended in this document, confirmed in production engine code.

It implements the 32-layer release shape (`n_embd=4096`, `n_layer=56`, type `LLM_TYPE_9B`), a
text-only Qwen3.5 hybrid trunk, 16/4 full attention, 16-key/32-value GDN, MTP draft graph code,
and a fused mHC route/merge op. It explicitly drops the Vision tower. Its converter supports both
the `model.language_model.*` and `model.layers.*` source-prefix conventions.

The fork's CUDA mHC op is materially cleaner than simply adding normal RMSNorm kernels: one
`GGML_OP_MHC_ROUTE` computes the two-stream RMS norm, eight FP32 logits, two `sigmoid` pre-gates,
two `2*sigmoid` post-gates, 20 alternating row/column log-space Sinkhorn passes, and the routed
branch in one launch; `GGML_OP_MHC_MERGE` mixes streams and adds the post-gated branch in a second
launch. This avoids what its authors call 192 CPU↔GPU hops per token. That is exactly the kind of
closed operation our Op layer should own rather than decomposing mHC into many generic kernels.

The fork also exposes useful restrictions and limitations:

- GGUF mHC is fixed to two streams, eight router logits, and a hardcoded 20 Sinkhorn iterations.
  The converter rejects `fixed_identity` and any `pre_scale != 1.0`; the runtime does not store
  iteration count or pre-scale metadata. These are acceptable for this exact checkpoint but not a
  general mHC implementation.
- The op keeps residual streams, router weights, gates, and mixing weights in FP32. This differs
  from the HF reference, which casts branch and mix to the residual dtype for `matmul` after FP32
  routing. The llama.cpp implementation is therefore a qualified approximate integration, not an
  exact match to the HF reference numeric boundary. Our engine's governing rule requires an
  independent FP32/FP64 oracle, so we should not copy this simplification blindly.
- The converter stores looped weights once and aliases visits at load time. Our artifact model can
  do the same with one set of source tensors and multiple logical state slots.
- The fork has backend-op tests for mHC route/merge at multiple token extents and strided versus
  contiguous stream views. That is a good template for our own CTest coverage.

No general release notes or proof-of-quality report are visible in the shallow clone, so "works"
should be read as a functioning implementation with tests, not as independently verified numerical
equality to HF.

## 2. Model contract

### 2.1 Backbone

The text backbone is Qwen3.5-family and is very close to the already supported Qwen3.6 targets:

| Property | SenseNova-Loop-9B | Existing 27B target |
|---|---:|---:|
| physical text layers | 32 | 64 |
| logical executions | 56 | 64 |
| hidden size | 4096 | 5120 |
| intermediate size | 12288 | 17408 |
| full-attention pattern | every 4th, zero-based 3,7,…,31 | every 4th, zero-based 3,7,…,63 |
| full-attention heads / KV heads | 16 / 4 | 24 / 4 |
| head dimension | 256 | 256 |
| rotary dimensions | 64, partial 0.25, theta 1e7 | 64, partial 0.25, theta 1e7 |
| GDN Q/K heads × dim | 16 × 128 | 16 × 128 |
| GDN V heads × dim | 32 × 128 | 48 × 128 |
| GDN convolution width | 4 | 4 |
| RMSNorm epsilon | 1e-6 | 1e-6 |
| tokenizer rows | 248077 addressable / 248320 matrix rows | same |
| native checkpoint context | 262144 | 262144 |
| release context via YaRN | 524288, factor 2 | target uses its own runtime scaling |

The release config uses `rope_type: yarn`, `factor: 2.0`, and
`original_max_position_embeddings: 262144`; the Preview config uses default RoPE. Our current
`compute_yarn_config()` is derived from the registered 27B runtime and uses a 1,048,576-token
native base. That is not the same schedule. A SenseNova target needs its own checkpoint-derived
YaRN table and attention-temperature policy, or must cap context at native 262K until the schedule
is independently verified.

Source tensor names in the public re-upload use the `language_model.model.*` prefix; the gated
release index uses `model.language_model.*`. That is a harness-dependent naming difference, not a
different model. The converter should accept the canonical release prefix and the known re-upload
prefix explicitly, normalize both to target object names, and reject any other prefix.

### 2.2 Loop plan

`execution_plan(config)` in the reference implementation is:

```text
plan = [0 .. start-1]
       + [start .. end-1] repeated eval_count times
       + [end .. num_layers-1]
```

For the registered checkpoint, `start=4`, `end=28`, `eval_count=2`, so:

```text
0 1 2 3 | 4 5 ... 27 | 4 5 ... 27 | 28 29 30 31
```

That is 56 executions but only 32 physical weight sets. The 24 repeated visits cover six physical
full-attention layers and 18 physical GDN layers, so the logical runtime needs:

- 14 full-attention KV state identities: eight physical full layers plus six loop-revisited full
  layers;
- 42 GDN state identities: 24 physical GDN layers plus 18 loop-revisited GDN layers;
- one set of physical weights, bound once, with multiple logical state slots.

The current `qwen3_6::DecoderState`, `SequencePlan`, and `Program` implement a one-to-one mapping
from logical layer index to layer weights and persistent state. That mapping is the central
architectural change. The llama.cpp fork avoids the problem by materializing an expanded 56-block
GGUF while storing looped weights once and aliasing later visits during load. We can use the same
concept without expanding persistent artifact tensors: bind one physical tensor set and allocate
56 logical state identities.

### 2.3 mHC residual routing

Within physical layers 4 through 27, the hidden state is split into two residual streams of width
4096 each (total width 8192). For each layer there are two routers: `mhc_attn` and `mhc_ffn`.

Each router has FP32 weight `[8, 8192]` and FP32 bias `[8]` for two streams:

```text
flat       = flatten(hidden [T,2,4096])
normalized = flat * rsqrt(mean(flat^2, last) + 1e-6)
logits     = FP32 linear(normalized, weight, bias)

pre        = sigmoid(logits[0:2])
post       = 2 * sigmoid(logits[2:4])
mix_logits = logits[4:8].reshape(2,2)
mix        = sinkhorn_log_space(mix_logits, iterations=20)

branch     = pre @ streams
branch     = branch * pre_scale             # configured 1.0
```

After the attention or FFN branch:

```text
mixed      = mix @ streams
update     = post[...,None] * branch[...,None,:]
streams    = mixed + update
```

At the first mHC layer the scalar 4096-wide hidden state is expanded by unsqueeze and broadcast;
at the last mHC layer (`physical == end-1`) streams are reduced by mean. Between those boundaries
the flattened 8192-wide representation is carried through the layer stack.

The crucial numeric boundary is that router parameters, normalization, logits, sigmoid, and
Sinkhorn are FP32. Stream mixing and branch application are performed in the residual dtype (BF16).
An implementation that casts router weights to BF16 or performs logsumexp in BF16 will not match
the reference.

### 2.4 Layer skeleton

Physical layers outside mHC follow the existing Qwen3.5/Qwen3.6 skeleton:

```text
h = RMSNorm(x, input_norm, effective scale = 1 + w)
x = x + mixer(h)
h = RMSNorm(x, post_norm, effective scale = 1 + w)
x = x + down(SiLU(gate(h)) * up(h))
```

Inside mHC, that skeleton is wrapped by router entry/merge at both the mixer and FFN stages.
Full-attention layers use zero-centered Q/K RMSNorm, 64-dim partial RoPE, a per-head output gate,
and GQA over 16 Q heads / 4 KV heads. GDN layers use the same gated-delta recurrence as our current
GDN Op with 16 key heads and 32 value heads.

### 2.5 Vision, MTP, and tokenizer

The checkpoint carries a 27-layer Qwen3.5 Vision tower (hidden 1152, 16 heads, patch 16, temporal
patch 2, spatial merge 2), a one-layer MTP draft (Preview config), and a 248320-row LM head. The
release API metadata is `image-text-to-text`; text generation alone is therefore a subset of the
checkpoint contract.

The tokenizer has 248044 BPE entries plus 33 added tokens through id 248076. EOS is `<|im_end|>`
(id 248046 in the public config) and the release generation config also lists 248044. The chat
template is Qwen-style `<|im_start|>/<|im_end|>` with thinking mode and image/video placeholders.
Our current frontend hard-codes `kTokenDomain = 248077`, which happens to match the tokenizer domain,
but it also hard-checks the 27B chat template and enabled thinking conventions. A new target needs
its own resource binding and template qualification.

## 3. Gap analysis against current engine

### 3.1 Reusable as-is or with narrow qualification

- `gated_delta_net`: supports arbitrary positive QK head counts with `value_heads % qk_heads == 0`;
  16/32 is already exercised by the 35B target.
- `gqa_attention`: currently registers geometries `24/4` group 6 and `16/2` group 8. SenseNova
  needs `16/4` group 4. This is a new compile-time geometry, not a new algorithm.
- `rope`: Text MRoPE `head_dim=256`, `rotary_dim=64` is generic, but optimized head geometries are
  registered as 24/4 and 16/2. Add 16/4 to the qualified profile matrix.
- `attn_input_proj`: the BF16 controller form is `[14336,5120]` with row counts
  `[6144,1024,6144,1024]`. SenseNova needs an 8192-row Q+gate projection plus separate K/V, so
  the target needs either a fused BF16 parent layout `[10240,4096]` with row counts
  `[4096,1024,4096,1024]`, or separate linear projections. The latter is simpler and leaves the
  fused Op untouched.
- Norms, SiLU/SwiGLU, residual-add, sampling, paged KV, D3D12 residency, DirectStorage state cache,
  CUDA Graph machinery, CLI, and HTTP serving are target-neutral.

### 3.2 New target-level work

1. **Artifact inventory and converter.** New model/weights identity, quantized formats and layouts,
   32 physical layers, 24 mHC routers × 2, loop metadata, tokenizer/chat resources, and source
   prefix normalization. The current converters validate exact Qwen configs and would reject this
   checkpoint.
2. **Binding and load plan.** Bind 4096-wide text tensors, 16/4 attention weights, 16/32 GDN
   weights, and FP32 mHC routers. Router weights must remain FP32 through materialization.
3. **Loop topology in SequencePlan / DecoderState.** Introduce a logical-to-physical layer map and
   logical-to-state-slot map. Weight binding follows physical index; KV/GDN state follows logical
   visit index. Prompt edit, prefix identity, state cache, and rollback must use the logical plan.
4. **mHC kernel/Op boundary.** Sinkhorn-20 and stream mixing are mathematically closed and could be
   admitted as an Op family, but they are target-specific in first integration. The clean boundary
   is a semantically closed `mhc_route`/`mhc_merge` contract with independent FP32 oracle tests;
   embedding it in the Program without a contract would violate the repository’s Op development
   rules.
5. **Shape changes in host layouts.** Hidden width 4096 and flattened mHC width 8192 change
   workspace, graph capture, state layout, and tensor slicing throughout `impl/runtime`. Existing
   Qwen targets instantiate templates with compile-time config; a new variant can reuse the shared
   templates without parameterizing the 27B target.
6. **Frontend.** New resources, thinking mode, EOS set, and image/video placeholder policy. If
   vision is deferred, the converter must still bind or validate the vision resources and the
   runtime must explicitly report that image input is unsupported rather than accepting and
   silently dropping media.
7. **YaRN/context.** Implement the checkpoint’s factor-2 schedule or restrict to 262144 native
   context. Do not reuse the 27B `compute_yarn_config` blindly.

### 3.3 Product charter conflict

`AGENTS.md` declares supported identities as only `qwen3.8-27b/groupwise-int` and
`qwen3.6-27b/groupwise-int`, explicitly excludes plugin architectures, and says targets own fixed
checkpoint profiles. Adding SenseNova is a deliberate expansion of the product contract, not a bug
fix. It should be approved and documented before code changes, with a new model card and maintainer
model reference.

## 4. Recommended architecture

### 4.1 Target layout

Create a sibling package, not a conditional branch inside `qwen3_6_27b`:

```text
src/targets/sensenova_loop_9b/
  export/ninfer/targets/sensenova_loop_9b/package.h
  impl/config.h
  impl/loop_plan.h
  impl/variant.{h,cpp}
  impl/package.cpp
  impl/load/bindings.{h,cpp}
  impl/variant.cpp
tools/convert/sensenova_loop_9b/
  convert.py
  inventory.py
  recipe.py
  verify.py
```

Register a new identity such as `sensenova-loop-9b/groupwise-int` and a new target key
`sensenova_loop_9b` in `src/targets/registry.cpp`. Keep the existing Qwen identities untouched.

### 4.2 Loop plan as a first-class target object

```cpp
struct LoopPlan {
    int physical_layers;
    std::vector<int> physical_for_visit;   // size 56
    std::vector<int> state_slot_for_visit; // size 56, unique per visit
    std::vector<int> visits_for_physical;  // reverse map for shared weights
    int mhc_first_visit;
    int mhc_last_visit;
};
```

Invariants to assert at construction:

- `physical_for_visit` follows the config’s exact `[0,start) + [start,end)*count + [end,N)` plan;
- `start < end <= N`, `count >= 1`, `counts == [count]`;
- mHC range lies wholly inside one loop phase, as required by the reference implementation;
- every visit has a distinct state slot for both full KV and GDN state;
- physical weights are read-only after load and are never rebound per visit.

### 4.3 mHC numerical contract

Define a closed Op with an independent FP32/FP64 oracle:

```text
inputs:  hidden BF16 [S, D] or [S,2,D], router FP32 [8,2D], bias FP32 [8]
outputs: branch BF16 [S,D], post FP32 [S,2], mix FP32 [S,2,2]
```

Separate merge:

```text
inputs: streams BF16 [S,2,D], branch BF16 [S,D], post FP32 [S,2], mix FP32 [S,2,2]
output: streams BF16 [S,2,D]
```

Required oracle checks:

- `normalized` uses FP32 mean-square over `2D` with epsilon `1e-6`;
- 20 alternating row/column `logsumexp` normalizations in FP32;
- sigmoid and `post = 2*sigmoid` in FP32;
- branch and mix matrix-multiply results in BF16 storage after FP32 accumulation;
- first/last-stream expansion and mean reduction exactly match reference;
- `pre_scale` is consumed as a positive finite FP32 scalar.

The llama.cpp fork demonstrates a faster single-kernel route/merge structure, but its residual
math is FP32 throughout while HF uses BF16 storage for branch/mix application. We should expose a
closed two-op contract and may implement it as one fused route kernel plus one merge kernel, but
the oracle must model the HF boundary. If exact HF parity proves incompatible with acceptable sm_89
performance, publish a named qualified profile instead of silently redefining correctness.

### 4.4 State and cache ownership

Extend the existing layout planner so state allocation iterates over logical visits while tensor
binding iterates over physical layers:

```text
weight[physical] -> shared across visits
KV[visit]        -> independent cache identity
GDN[visit]       -> independent recurrent/conv identity
router[physical] -> shared across visits
```

Prefix reuse, mid-prompt edit, DirectStorage snapshots, and replay/rollback must key on the logical
plan. A reused prefix must restore both visits of a looped physical layer; restoring only the
second visit silently produces wrong output. This is the highest-risk correctness area.

### 4.5 Attention geometry

Add `GqaSenseNovaGeometry = GqaGeometry<16,4,...>` and qualify:

- scale `1/sqrt(256) = 0.0625`;
- 16 Q heads, 4 KV heads, group size 4;
- head dim 256, rotated dim 64;
- BF16 and INT8-G64 caches if both are offered;
- decode, prefill, KV append, prefix relocation, and INT8 scale layouts.

Because group size 4 changes decode row tiling and split policy, do not assume the 24/4 group-6
tuning is valid. Benchmark grid occupancy on 128 SMs and select split scales empirically.

### 4.6 Quantization

The existing Q4/Q5/Q6/W8 formats are a reasonable starting point for a groupwise-int artifact, but
the converter must define exact row counts for:

- attention Q+gate `[8192,4096]` and K/V `[1024,4096]` each;
- GDN QKV `[8192,4096]`, Z `[4096,4096]`, A/B `[32,4096]`;
- MLP gate/up `[24576,4096]`, down `[4096,12288]`;
- embedding and LM head `[248320,4096]`;
- mHC routers FP32 `[8,8192]` plus bias FP32 `[8]`.

Do not reuse 27B scale divisors blindly. Quantized outputs must be qualified against an
independent FP64 oracle from the represented BF16/FP32 source values.

## 5. Implementation phases and verification

### Phase 0 — contract lock

- Obtain an licensed local copy of the release checkpoint and reconcile its config/index against
  the public mirror facts in section 1.
- Write `docs/maintainer/sensenova-loop-9b-model.md` with exact shapes, formulas, state counts,
  YaRN schedule, EOS/chat template, and source prefix mapping.
- Update `AGENTS.md`, `docs/README.md`, model cards, and release notes to add the supported
  identity. Do this only after explicit scope approval.

### Phase 1 — numerical primitives and converter

- Add mHC Op contracts and CTests with FP32/FP64 oracles, including strided stream views and
  token extents 1/7/512 as in the llama.cpp backend op tests.
- Add GQA 16/4 geometry tests for decode/prefill/KV append and both cache formats.
- Add GDN 16/32 conformance for the same token extents used by the target.
- Add converter inventory, strict config validation, source-prefix normalization, and artifact
  round-trip tests.

Exit criterion: every primitive is correct independently of the full model.

### Phase 2 — target execution

- Add package, config, binding, variant, sequence planner integration, loop plan, and mHC wiring.
- Text-only greedy and sampling decode.
- Prefill chunking, prompt edit, prefix identity, concurrency, CUDA Graph capture, and state-cache
  transactions with 56 logical visits.

Exit criterion: deterministic text output matches the reference implementation on a small set of
prompts, with independent state per visit verified by direct state inspection.

### Phase 3 — MTP and hardening

- Convert and bind the release `mtp.safetensors` draft: 4096-wide stem, 16/4 one-layer attention,
  12288-wide MLP, and full-vocab proposal through the main LM head.
- Validate dedicated draft embeddings if present in the licensed release; the public Preview uses
  the main embedding table and has `mtp_use_dedicated_embeddings=false`.
- Add draft KV state and graph profiles to the 56-visit target plan.
- INT8/BF16 KV quality sweep and 262K/512K context tests.
- Performance tuning for group-4 GQA, mHC kernels, L2-aware weight rasterization, and graph profiles.
- Vision preprocessing, 3D patch convolution, Vision tower, mRoPE positions, and image scattering
  remain phase 4 if required.

## 6. Test plan

Per repository rules, every new feature or critical change requires CTests, and all registered tests
must pass. Minimum additions:

1. `test_sensenova_loop_plan.cpp` — plan construction, validation, mHC phase boundaries, and
   logical/physical mapping.
2. `test_sensenova_mhc.cpp` — router and merge against independent FP64 oracle, including 20
   Sinkhorn iterations, FP32 router parameters, and strided two-stream views.
3. GQA 16/4 geometry tests in the existing attention conformance suite.
4. GDN 16/32 tests in the existing GDN suite, including snapshot/replay if speculative decoding is
   in scope.
5. `test_sensenova_load_plan.cpp` — artifact binding with no device required (skip code 77 when
   the checkpoint fixture is absent).
6. Converter Python tests — strict config rejection, source-prefix normalization, tensor shapes,
   FP32 router preservation, resource bytes, and deterministic artifact output.
7. Engine real-model tests — prompt edit, prefix identity, state save/restore, and decode
   determinism with 56 visits, using skip code 77 when large weights are unavailable.
8. Serving contract tests — EOS `<|im_end|>`, thinking-template behavior, and explicit rejection
   of image input if vision is deferred.

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Gated release config differs from public mirror | Wrong shapes or semantics | Strict converter validation against licensed local checkpoint; no silent fallback |
| Wrong loop state identity | Silent quality collapse | Logical state slots; targeted prefix/edit/snapshot tests |
| mHC precision drift | Output divergence | FP32 router contract and independent oracle; no BF16 router cast |
| Group-4 GQA performance regression | Low throughput | New geometry with measured split policy; do not reuse group-6 tuning |
| YaRN mismatch beyond 262K | Long-context quality failure | Implement checkpoint-specific schedule or cap context |
| Hidden-width assumptions in shared runtime | Compile/runtime failure | New target template instantiation; avoid parameterizing 27B target in-place |
| Long decode is 56 layers, not 64 | Different perf/state budget | Re-run capacity curves and graph profiles; do not copy 27B plan thresholds |
| MTP draft mismatch or missing licensed draft weights | No speculative speedup or wrong drafts | Validate public Preview draft now and strict-check the release `mtp.safetensors`; keep ordinary decode as fallback |
| Vision scope creep | Large unstable change | Explicit text-only first phase; reject unsupported media clearly |
| VRAM pressure at long context | Capacity failure | Recompute KV/GDN capacity curves for 14 KV visits and 42 GDN visits; use 8-bit KV and compact state pools |

## 8. Feasibility verdict

- **Text-only BF16 + groupwise-int target:** feasible with focused work. Most heavy numerical
  machinery already exists. Main new code is loop/mHC target semantics, a 16/4 GQA geometry, a
  4096-wide converter/binding package, and tests. The llama.cpp fork confirms that a text-only
  version is implementable without solving vision or 512K context first.
- **Full image/video parity:** feasible but substantially larger. It requires the Qwen3.5 Vision
  path, 3D patch convolution, mRoPE position generation for mixed text/image/video, and frontend
  resource work.
- **512K context:** feasible only after implementing the release’s factor-2 YaRN schedule and
  validating KV/GDN state capacity; 262K native is the safer first milestone.
- **MTP:** feasible and strategically worthwhile. The canonical gated release and the public
  Preview both ship a separate `mtp.safetensors`; the Preview copy is public and contains exactly
  one 4096-wide Qwen3.5-style draft layer (`fc [8192,4096]`, 16/4 attention, 12288-wide MLP,
  norms, and a full-vocab projection via the main LM head). That shape maps to our existing MTP
  runtime almost directly. The main extra work is converter support, dedicated-embedding handling,
  draft KV state for the 56-visit target, and spec-decode validation. Treat MTP as part of the first
  performance-complete target rather than an optional late add-on.
- **External runtime fallback:** lowest engineering cost. vLLM/SGLang already have architecture
  support, but they do not provide this engine’s artifact format, D3D12 residency, DirectStorage
  state cache, and sm_89-specialized kernels.

Recommendation: if native support is pursued, implement text+MTP first behind a new target
identity, prove numerical correctness with the mHC oracle and loop-state tests, then decide whether
vision justifies the additional target surface.
