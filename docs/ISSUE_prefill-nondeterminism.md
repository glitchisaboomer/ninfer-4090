## Summary

Prefill is not bit-reproducible on this engine. Two runs of the **same** prompt, in **separate
processes**, with **identical flags**, produce GDN linear-attention states that differ by
~1.6e-3 relative and first-token logprobs that differ by 3e-4..3e-3 nats.

This was found while diagnosing a suspected Suffix Cache Reuse defect (`scr-quality-run3`). SCR is
**not** the cause — the same divergence reproduces with `--no-scr` on every phase. But it
invalidates the noise floor that the SCR quality harness measures against, so the harness reports
`UNATTRIBUTABLE` on essentially every run and cannot tell a real regression from jitter.

### The originating SCR investigation closed clean

Recorded here so the next person does not reopen it. `scr-quality-run3` reported k2 as an anomaly;
it is not a defect.

- Every relocation span in that suite has `delta=0` (`src[448..640) -> dst[448..640)`), so the
  re-rotation is a mathematical no-op. Relocated pages verified `intact`, `changed_rows=0`.
- A `[scr-diag]` fingerprint of the GDN slot confirmed the invalidation the runtime documents: the
  slot went from `rec_abs=134721` to exactly `0` after the zero, and prefill refilled all 48 layers
  (`rec_abs=118615`), `non_finite=0/0` throughout.
- The residual difference between the SCR leg and the reference legs (`rec_abs` 118615 vs ~117200,
  i.e. +1.2%) is the *intended* GDN approximation: the relocated span contributes nothing to the
  linear-attention recurrence while full attention reads it from the relocated KV pages.
- It does not explain why k2 was singled out. k64 relocates 6208 of 6696 tokens (93%) and lands at
  0.0026 nats; k2 relocates 192 and lands at 0.0801. There is no monotonic relationship, so the
  approximation is not what makes k2 look anomalous. That part was never explained and should not
  be assumed solved — see the last paragraph of Notes.

## Reproduction

```powershell
.\build-ninja\apps\ninfer-scr-quality.exe --model models\qwen3_8_27b.ninfer `
  --work-dir repro --k-blocks 2 --flag-control --scr-diag --keep-logs
```

`--k-blocks 2` is enough. Any k works; the effect is present on all six prompts in run3.

### Evidence 1 — first-token logprob, no decoding involved

Position 0's logprob is produced by a single prefill forward pass. There is no decode loop, no
sampler and no speculation, so any difference here originates in prefill.

`baseline` and `control` are launched with **identical** server flags. They still differ at
position 0 on every prompt:

| prompt | d(lp@0) baseline vs control |
|---|---|
| k1 | 0.002592 |
| k2 | 0.000355 |
| k3 | 0.000273 |
| k6 | 0.003014 |
| k12 | 0.000919 |
| k64 | 0.001509 |

The harness's own reported config floor is 0.000354767 — the same order as the disagreement
between two runs that are supposed to be doing the same thing.

### Evidence 2 — GDN recurrent state totals, same prompt, same flags

From the `[scr-diag] gdn stage=post_prefill` lines in `scr-quality-k2b/console.log`. Each phase's
prime request is the same 661-token prompt, prefilled from a cold `full_reset`.

| phase | prime rec_abs | prime conv_abs | edited rec_abs |
|---|---|---|---|
| SCR | 117165 | 1.29042e+06 | **118615** |
| baseline | 117211 | 1.29018e+06 | 117139 |
| control | 117172 | 1.29077e+06 | 117096 |
| flag | 117351 | 1.29044e+06 | 117266 |

Prime spread: `rec_abs` 117165..117351 (**1.6e-3** relative), `conv_abs` 1.29018e6..1.29077e6
(**4.6e-4**). Four processes, identical flags, identical prompt, different state.

**Caveat on this table.** `sum|x|` is a magnitude aggregate, not a checksum: distinct states can
share a total, and near-identical states can differ in it. It establishes that the four states are
*not* bit-identical, but it cannot by itself attribute that to float-order nondeterminism rather
than to a genuine difference in what was computed. Evidence 1 has no such caveat — it is a direct
observation of the output.

Also worth recording: the per-layer sample `rec[0]` prints as `15124` in all four prime rows, and
the prime rows separate cleanly by request role (prime `15124.0` vs edited `15122.8` vs harness
warm-up `13464.6`) with only ~1.3e-3 jitter within a class. So the grouping is coherent; only the
magnitude of the jitter is in question.

### Evidence 3 — the SCR effect is separable from the noise

Useful for sizing the problem. Comparing the edited request:

- noise floor (`baseline`/`control`/`flag` edited): `rec_abs` 117096..117266, spread **1.5e-3**
- SCR edited: `rec_abs` **118615**, i.e. **+1.2%** over the reference

So the deliberate SCR GDN approximation is roughly **7x** the run-to-run noise, and is measurable
at the state level even though it is not separable at the token level. That is the diagnostic
result; it also means the SCR quality harness's floors need to come down before its thresholds mean
anything.

## What has been ruled out

- **PDL / programmatic dependent launch.** `src/core/pdl.cuh` compiles the whole mechanism out
  under `NINFER_SM86`/`NINFER_SM89`, so the 4090 never takes that path.
- **Cooperative launch grid sync.** The only `grid().sync()` is in
  `bf16_gdn_gating_proj_gemm_mma.cuh` and is a real barrier, not a race.
- **SCR relocation.** Reproduces with `--no-scr` on all phases. Rotation is a no-op in this suite
  (every span has `delta=0`); relocated pages verify `intact`; the GDN slot is zeroed and refilled
  by all 48 layers.
- **Non-deterministic RNG.** Sampling is greedy in this harness.

## Reproduced in-process

Added `ninfer_qwen3_6_27b_prefill_determinism_test`
(`tests/targets/qwen3_6_27b/test_engine_prefill_determinism.cpp`). It uses the served configuration
(`max_context=8192`, `max_concurrency=1`, `prefill_chunk=1024`, `KvCacheStorage::BFloat16`) and the
k2 suite prompt verbatim from `apps/scr_quality`'s `built_in_prompt`, including `repeated_tail`, so
the text tokenizes to the same ids the harness feeds. It compares `generated_logprobs[0]` — a value
produced by the forward pass alone, with no decode round, sampler, or speculation before it.

**Two cold prefills of the same prompt through one Engine disagree.** Five consecutive runs:

| run | first | second | delta |
|---|---|---|---|
| 1 | -0.00844002 | -0.00785446 | 5.86e-4 |
| 2 | -0.0115166 | -0.00644493 | 5.07e-3 |
| 3 | -0.0073452 | -0.00751877 | 1.74e-4 |
| 4 | -0.00665283 | -0.00638199 | 2.71e-4 |
| 5 | -0.00704575 | -0.00771523 | 6.69e-4 |
| 6 | -0.00865364 | -0.0100021 | 1.35e-3 |

The spread spans 1.7e-4 to 5.1e-3 nats, i.e. two orders of magnitude, and it is not drifting
monotonically. This is the same magnitude as the harness's own baseline-vs-control divergence
(5.0e-4 nats for k2 prime), so the harness's noise floor is this bug, not a separate effect.

**An earlier version of this test reported `delta=0` and was wrong.** It used `max_context=4096` and
synthetic token ids in [1000, 61000). Those are not equivalent: a hand-picked id sequence produces a
different expert-routing distribution, and appears to sit in a degenerate regime where the
order-sensitive path happens to agree. Two conclusions drawn from that version were incorrect and
are retracted here:

- that prefill was deterministic within a process, and
- that the MoE `atomicAdd` at `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:162` was
  ruled out.

It is ruled out again, this time on a correct and independent basis: **it never executes for this
model.** `qwen3_8_27b` is dense. `ops::sparse_moe` is referenced only from
`src/targets/qwen3_6_35b_a3b`; the 27B MLP is `ops::linear_swiglu` + `ops::linear_add`
(`src/targets/qwen3_6_27b/impl/variant.cpp:231`). The 27B artifact has no `mlp/gate_up` routed
weight and no router, so there are no experts and nothing to gather.

The general lesson stands though: that version of the test had no power to test anything, because
it did not reproduce the symptom. A test that cannot reproduce the divergence cannot exonerate
anything.

## Remaining hypotheses

1. **Split/reduction order in the dense path.** `ops::linear_swiglu`, `ops::linear_add` and the
   row-split GEMMs reduce across splits; if the split count or the merge order depends on anything
   not constant across calls, FP32 association order varies and prefill follows. Recent perf work
   changed tile and split shapes; `45a5ae57`'s own message notes that changing a split partition
   changes FP32 cross-split merge order.
2. **GDN chunked recurrence.** `src/ops/linear_attention/gated_delta_net/recurrent.cuh` sums a warp
   reduction with `__shfl_down_sync` (lines 79, 112, 260) inside a recurrence that is sequential over
   the chunk. Same mechanism as (1) but inside the 48 linear-attention layers, which is where the
   state fingerprints showed the largest sensitivity.
3. **`atomicAdd` on `__nv_bfloat16`.** `src/ops/linear/w8/w8_small_t_mma.cuh:317` accumulates with
   `atomicAdd` on BF16 under `KSplits > 1`, which would be nondeterministic *and* lossy. **This looks
   unlikely to be the cause.** Every `KSplits > 1` launcher is registered through
   `make_splitk_launchers` (`src/ops/linear/w8/w8_small_t.cu:68`) and covers only MTP and vocabulary
   projections; the MLP reaches the small-T kernel through the non-split-K `launch_exact`. The Q5
   counterpart (`src/ops/linear/q5/q5_small_t_mma.cuh:44`) is an epilogue that is instantiated only for
   the residual-atomic case. Not fully eliminated — `KSplits` is a template parameter and its uses
   across the repository have not been audited exhaustively.

   **Audited since, on the measured path.** Position 0's logprob is produced by the vocabulary
   projection at `T=1` (`text_prefill_impl.h:148` slices the logits column before `ops::linear`), and
   `w8_dispatch.cpp` sends that shape to `launch_w8_small_t`; the vocabulary geometry is registered
   through `make_launchers`, i.e. `launch_exact`, `KSplits=1`, a plain store — `w8_small_t_mma.cuh:317`
   takes the `atomicAdd` branch only when `KSplits > 1`. `w8_small_t.cu` registers split-K launchers for
   the MTP input and MTP attention-output geometries only (`w8_config.h:73-76`), and neither runs with
   speculation off. So (3) is not the cause of the 661-token divergence.

   **But it is live in the short band the sweep was written to test.** `launch_q5_linear_add_small_t_mma`
   (`src/ops/linear/q5/q5_small_t_mma.cu:87`) runs `launch_residual_exact` with `kSplits = 2` and
   `Q5SmallTMmaResidualAtomicEpilogue` (`q5_small_t_mma.cuh:44`) for the `n=5120, k=17408` MLP *down*
   projection at `t <= 16`. That is a real BF16 `atomicAdd` on the prefill path — nondeterministic and
   lossy by construction. A delta confined to `t <= 16` should be read as this first and is not evidence
   about the 661-token case. The dismissal above reasoned only about W8 and `gate_up` and never asked
   what the Q5 residual-atomic case *is*.

## Schedule boundaries, for whoever reads this next

Token count selects the kernel, so it is a usable control on *which arithmetic runs*. For the 27B
MLP shapes (`k=5120`):

| projection | small / SIMT band | bulk band | source |
|---|---|---|---|
| `mlp/gate_up` n=34816, Q4 | `t<=4` simt_c4, `t<=16` simt_c8 | mma c128 | `ops/linear/q4/q4_dispatch.cpp:35` |
| `mlp/gate_up` n=34816, W8 | `t<=40` small_t | c128 | `ops/linear/w8/w8_dispatch.cpp:30` |
| `mlp/down` n=17408, Q5 | `t<=16` small_t | c32 to 64, c128 beyond | `ops/linear/q5/q5_dispatch.cpp:40` |

No single token count puts all three in their small band: `t<=16` sends gate_up Q4 to SIMT, not
small-T. A sweep over t = 8, 16, 40, 64, 128, 661 was meant to straddle all of them, and `t=661` is the
harness's own length. **It does not straddle anything** — see "The sweep does not sweep" below: the
rendered chat template's floor is 67 tokens, above every boundary in these tables, so the short points
all measure one 67-token prompt. Reaching the bands needs a raw id sequence (`Engine::prepare_tokens`).

Ruled out by test: PDL (compiles out on sm_89), the cooperative `grid().sync()` in
`bf16_gdn_gating_proj_gemm_mma.cuh` (a real barrier), SCR relocation (reproduces with `--no-scr`),
non-deterministic RNG (sampling is greedy here), and sparse-MoE prefill gather (not reachable from
this model).

## Requested

Guidance on which of (1), (2), (3) best matches a divergence of 1.7e-4..5.1e-3 nats that varies
per prefill of an identical prompt, and which diagnostic best localizes it. Our reading is that a
per-call-varying FP32 association order is the mechanism class that fits, but we would rather be
pointed at the right layer than guess across three candidates.

*Answered below, in order:* (1) and (2) survive only as an amplifier, not as a source — nothing on the
measured path varies its reduction order between two calls. (3) is exonerated for `t=661` (the logits
are a `KSplits=1` store) but is confirmed live for `t<=16` via the Q5 residual-atomic epilogue. The
localizing diagnostic is the per-layer residual trace, described below.

## Where this stands, and the one command to run next

The sweep had never executed: `main` returned on the bit-identity assertion before reaching it, and
that assertion fails by design. The ordering is fixed, and the sweep has now been run. It is gated on
`NINFER_PREFILL_DET_VERBOSE`, and a run that does not set it prints no sweep output *and no warning* —
an empty result means the gate, not agreement.

```powershell
$env:NINFER_QWEN3_6_27B_WEIGHTS = "$PWD\models\qwen3_8_27b.ninfer"
$env:NINFER_PREFILL_DET_VERBOSE = "1"
& build-ninja\tests\ninfer_qwen3_6_27b_prefill_determinism_test.exe 2>&1 | Select-String 'delta=|target='
```

One run so far, plus the repeat case and the two-engine case from the same binary:

| leg | values | delta |
|---|---|---|
| same Engine, 2 of 3 | -0.010149 vs -0.0090313 | 1.11771e-03 |
| two Engines | -0.00865555 vs -0.010149 | 1.49345e-03 |
| sweep target=8 | - | 1.0490e-04 |
| sweep target=16 | - | 4.9591e-05 |
| sweep target=40 | - | 1.9073e-05 |
| sweep target=64 | - | 7.8201e-05 |
| sweep target=128 | - | 6.7139e-04 |
| sweep target=661 | - | 3.9787e-03 |

Two structural facts, both worth keeping:

- The two-engine leg's second value is *exactly* the repeat leg's first value (-0.010149). Two cold
  prefills of one prompt can coincide to the last digit while a third does not, which is what a
  discrete, per-call-cause effect looks like and not what dithering noise looks like.
- Both legs' shortest measurement is nonzero, and no leg has yet produced a zero delta.

### The sweep does not sweep

Targets 8, 16, 40 and 64 all produced the **same 67-token prompt**. `prompt_text_of_length` starts from
the suite prompt's last user turn and only ever appends, so any target at or below the rendered
chat-template floor (67 tokens) returns the seed unchanged. 67 exceeds every dispatch boundary above
(16 for `gate_up` Q4/SIMT and `down` Q5, 40 for `gate_up` W8, 48 in general), so **no sweep point
entered a small-T or SIMT schedule**: the six rows describe three distinct prompts, not six schedules.
The test now prints `same_prompt_as_previous=` so this cannot be misread again.

The four short rows still carry information, and it is the kind this document kept asking for: they are
four independent measurements of one identical configuration, and all four are nonzero
(1.9e-05 … 1.05e-04). At 67 tokens every GEMM is on a bulk MMA path and nothing on the path is
order-nondeterministic (no atomics in the dense or GDN paths; every launch geometry is a compile-time
constant or a function of shape and SM count), so a nonzero delta here means the varying quantity is
not the reduction tree.

### The fork, resolved as far as one run can resolve it

The divergence is present at 67, 132 and 664 tokens, never zero, and grows with length. It is therefore
**not token-count dependent** and not in the MLP's small-T/SIMT schedules — which never ran, so this is
weaker than "the short bands are clean": it is "the bulk bands are not clean either". By this document's
own rule, that promotes hypothesis (2), attention or the GDN recurrence, both of which run at every
length. Note that this rules out (1) only as stated — split geometry that varies per call; a *fixed*
reduction order fed different bytes is still on the table, and is the only thing left if nothing in the
path is order-nondeterministic.

## The next diagnostic: a per-layer residual trace

One scalar per leg cannot say which layer moved, and the sweep cannot say it at any reachable length.
`NINFER_PREFILL_TRACE=1` now makes `TextContext::prefill_chunk` instantiate a `PrefillLayerTrace` tap
(`src/targets/qwen3_6/impl/runtime/prefill_layer_trace.h`) on the plain-text prefill path. It hashes the
residual tensor after every layer and diffs that hash against the previous prefill through the same
Engine, so one run answers the question:

```powershell
$env:NINFER_PREFILL_TRACE = "1"    # 2 also prints the layers that did not move
& build-ninja\tests\ninfer_qwen3_6_27b_prefill_determinism_test.exe 2>&1 |
  Select-String 'prefill-trace|repeat:'
```

Each chunk prints one summary line and, by default, only the layers that differ:

```
[prefill-trace] chunk=1 tokens=659 layers=64 baseline=no scope=this_instance divergent_layers=12 first_divergent_layer=7
[prefill-trace] chunk=1 layer=7 hash=0x... sum_abs=... prev=0x... prev_sum_abs=... same=no
```

The numbers in that line are the ones this artifact really produces (659 tokens in the first chunk, 64
layers); an earlier draft of this document illustrated it with 48.

**Why the first differing layer is the answer.** The residual that enters layer `L` is layer `L-1`'s
output, so the first layer whose hash differs had bit-identical inputs. Whatever diverged is inside that
layer's attention/mixer, its MLP, or its norms. That is a named kernel rather than a band, and it is one
run.

**Why a hash and not `sum|x|`.** The same distinction as Evidence 2's caveat: a magnitude aggregate can
show that two states differ but never that they agree, so it cannot say "this layer is identical". A
64-bit fold of the residual words answers both, and the FP64 `sum_abs` rides along only to say how large
a difference is. The hash folds 8-byte words in memory order, so it is a change detector on one machine,
not a portable checksum.

**Comparisons are declined, not approximated.** A chunk is only compared against a previous chunk of the
same token width and layer count, so prefilling several different prompts in a row (which the sweep does)
prints `baseline=yes` instead of reporting every layer of prompt B as divergent from prompt A.

**`scope=` says which window supplied the baseline**, which matters because the trace cannot be read
without it: see the next section.

`NINFER_PREFILL_TRACE` is opt-in at the runtime rather than forced by the test, because it copies every
layer's residual tile to the host and synchronises: 664 tokens is ~6.8 MB per layer, ~325 MB per prefill,
and it will visibly slow a run. The test prints a note pointing at it when the variable is unset.

**What is tested, and what is not.** The fold and the trace's memory are device-free host code
(`impl/runtime/prefill_trace.h`), and `ninfer_qwen3_6_prefill_trace_hash_test` pins them: the FP64
magnitude sum against values declared next to the BF16 words, that a `-0.0` word contributes zero and
never a negative magnitude, and that changing *any* word of a tile changes the hash -- tested per
position, because a fold that read only the first word or only the last would pass a single-position
test. The tap's comparison and window rotation need a device and a prefill, so they are exercised by the
engine test under `NINFER_PREFILL_TRACE` rather than by the unit test.

The same change makes the first case prefill the prompt **three** times and print all three values
(`repeat: p1= p2= p3= d12= d23= d13=`). A pair cannot distinguish "the second call read what the first
wrote" from "the first call is special"; a triple can: `p2 == p3 != p1` is a first-call effect,
two agreeing values with a third apart is a pairwise coincidence worth noticing, and three distinct
values say the divergence is not a function of call index at all. The trio is printed whether or not
`NINFER_PREFILL_DET_VERBOSE` is set, since it is the datum rather than a table.

## The trace's first real run, and the Engine it could not see

The trace has now run against the real artifact (`NINFER_PREFILL_TRACE=1`, the k2 fixture). It reached
the prefill path and hashed every layer, and this is what it printed for the two-Engine case:

```
[prefill-trace] chunk=1 tokens=659 layers=64 baseline=yes divergent_layers=n/a first_divergent_layer=n/a
[prefill-trace] chunk=2 tokens=2   layers=64 baseline=yes divergent_layers=n/a first_divergent_layer=n/a
[prefill-trace] chunk=1 tokens=659 layers=64 baseline=yes divergent_layers=n/a first_divergent_layer=n/a
[prefill-trace] chunk=2 tokens=2   layers=64 baseline=yes divergent_layers=n/a first_divergent_layer=n/a
two Engines in one process: logprob=-0.00818253 vs -0.00683975 delta=0.00134277 limit=1e-06
```

Four baselines and no comparison: the instrument declined to answer the leg that failed. The cause is
structural, not a data condition. The comparison window lives in a `thread_local`, and every Engine
serves its requests on a worker thread of its own (`ConcurrentExecutor` constructs `worker_` for every
Engine, `max_concurrency` 1 included), so the window is per-Engine: an Engine's *first* prefill never has
a predecessor, and the two-Engine leg is two first prefills. The leg that a fresh process boundary makes
interesting is exactly the one a per-thread window cannot see.

`NINFER_PREFILL_TRACE_CROSS=1` closes it. A completed chunk is published to a process-wide table keyed by
`(tokens, layers)`, and an instance with no baseline of its own compares against the last chunk of that
shape published by *any* instance. Each compared line then carries `scope=this_instance` or
`scope=other_instance`, because those are different claims and the reader must not have to infer which
one they got. The cross lookup is a separate variable rather than part of `NINFER_PREFILL_TRACE` because
it is the only mode that can report two *different* prompts of one width as fully divergent — the
approximation the same-instance path exists to refuse. The two-Engine case prefills one prompt through
both Engines, which is the guarantee the flag relies on; a sweep over differing prompts should leave it
off.

```powershell
$env:NINFER_PREFILL_TRACE       = "1"
$env:NINFER_PREFILL_TRACE_CROSS = "1"   # also compares the two-Engine leg
& build-ninja\tests\ninfer_qwen3_6_27b_prefill_determinism_test.exe 2>&1 |
  Select-String 'prefill-trace|repeat:'
```

**This run's numbers**, alongside the previous one. The two-Engine leg's delta moved from 1.49345e-03 to
1.34277e-03 and its values changed entirely (-0.00818253 vs -0.00683975, where last time the second value
was byte-identical to the repeat leg's first), so that coincidence was a coincidence and not structure.
The sweep's longest row is unchanged at ~1.3e-03.

| leg | values | delta |
|---|---|---|
| two Engines, previous run | -0.00865555 vs -0.010149 | 1.49345e-03 |
| two Engines, this run | -0.00818253 vs -0.00683975 | 1.34277e-03 |
| sweep target=661, this run | - | 1.34659e-03 |

Two same-Engine prefills and two different-Engine prefills of one prompt disagree by the same ~1.3e-03,
which is the useful part: the magnitude does not depend on the Engine boundary, so whatever varies is in
the per-prefill path rather than in weight materialization or arena layout. The cross-instance mode is
still what turns that claim into a named layer, because "the divergence is not engine-specific" is a
conclusion the same-instance trace reports only for the legs it can compare.

**What the harness caught.** The stub harness now models the real topology — one thread per Engine — and
it failed the first version of the cross-instance path, which identified a trace by the address of its
store. Two Engines built one after the other get the *same* address, because the TLS block of an exited
thread is handed to the next thread the OS creates; the comparison then reported `scope=this_instance`
for a genuine cross-Engine baseline. Identity is now a monotonic per-process counter. A harness that
only ever ran one thread would have shipped that.

## What the trace found: a two-token tail chunk and a bf16 atomic split

The trace's second run — `NINFER_PREFILL_TRACE=1` with `NINFER_PREFILL_TRACE_CROSS=1`, 17 prefills
across the three legs, 34 chunk summaries — has one shape throughout:

```
[prefill-trace] chunk=1 tokens=659  layers=64 baseline=no scope=this_instance divergent_layers=0  first_divergent_layer=none
[prefill-trace] chunk=2 tokens=2    layers=64 baseline=no scope=this_instance divergent_layers=64 first_divergent_layer=0
```

**Every** chunk 1 is bit-identical — at 65, 130, 659 and 662 tokens, across the three-prefill repeat, the
sweep's twelve prefills, and the two cold Engines. **Every** one of the 17 chunk-2 summaries differs at
layer 0 and then in all 64 layers. No exceptions, no partial agreements: the divergence is not diffuse,
it enters at one place.

Chunk 2 is 2 tokens, and 2 is the token count that decides the schedule. `q5_linear_add_resolve_plan`
routes `cols` 2..13 (k=6144) and 2..16 (k=17408) to `Split2ExactResidual`, which is
`launch_q5_linear_add_small_t_mma` → `launch_residual_exact` with `kSplits = 2` and
`Q5SmallTMmaResidualAtomicEpilogue`:

```cuda
__device__ __forceinline__ void store(__nv_bfloat16* out, int out_ld, int row, int col, float acc) const {
    auto* dst = &out[static_cast<std::int64_t>(col) * out_ld + row];
    atomicAdd(dst, __float2bfloat16_rn(acc));
}
```

Two CTAs (`blockIdx.y` of 2) each own half of K and both add their partial into the *same* bf16 output
element. The programming model fixes no order between them and bf16 addition is not associative, so
`bf16(bf16(R+A)+B)` and `bf16(bf16(R+B)+A)` differ in the low bits. Both q5 linear_add shapes are in a
lateral: attention output (k=6144) and MLP down (k=17408), and both run in every layer, so the first
layer's hash already reflects it and the rest of the stack inherits it.

This is the whole symptom, and it explains the parts that did not fit:

- **Why the tail chunk and not the bulk.** At t≥17 the q5 residual routes to the MMA
  `cta_collective_residual` schedules, which own their output element and are deterministic. The bulk
  chunk was never the problem, and no reachable prompt length avoids the tail.
- **Why the logprob delta is nonzero at every prompt length.** The measured position is produced by the
  last prefill chunk, and in every trace line the last chunk is the 2-token tail — 65+2 for the 67-token
  prompt, 130+2, 659+2 and 662+2 for its longer forms. So all six sweep rows and all three legs report a
  logprob that the atomic band produced.
- **Why the short sweep rows were nonzero.** The earlier reading in this document — "at 67 tokens every
  GEMM is on a bulk MMA path and nothing on the path is order-nondeterministic, so the varying quantity
  is not the reduction tree" — was wrong on its second clause. There are atomics on the path, in the
  small-T bands, and the 2-token tail reaches them at every length. The four short rows were measuring
  the tail, not "a schedule with no atomics".
- **Why the delta still varies with length** (1.9e-05 at 67 tokens up to 3.98e-03 at 664). A fixed
  perturbation placed low in the stack is amplified differently by different contexts; it is not that a
  longer prompt diverges more. That reading also retires the earlier conclusion drawn from the growth —
  token count was the wrong axis all along.

### Direction of the fix, and why it is not in this commit

The nondeterministic write is one epilogue on one token band, so the fix does not need the divergence to
be understood any further. Three options, in the order they can be justified:

1. **One split for the residual band.** With `kSplits = 1` each output element has exactly one writer, so
   the same schedule becomes deterministic — and rounds once instead of twice, which is the more accurate
   sum. The cost is half the CTAs along K, which is what the split exists for; at t=2 the GEMM is tiny and
   the trade may well be free, but "may well" is not a measurement.
2. **A two-phase split.** Both splits write partials to a workspace with plain stores and a second kernel
   merges them in a fixed order. Keeps the parallelism and the determinism; `LinearAddPlan` already
   carries `workspace_bytes`, so the plumbing exists. This is the architecturally right shape and the one
   to reach for if option 1 measures slow.
3. **Route the band to the MMA collective-residual kernels**, which already solve this exact problem —
   their schedule names say so (`linear_add.q5.mma.r64.c16.cta_collective_residual`). `MmaResidualR64C16`
   exists for k=6144 but is routed only for cols 14..16, and there is no small-column MMA residual
   variant for k=17408 at all, so this option needs either a routing change with a benchmark or a new
   kernel instantiation.

All three are arithmetic changes on a shipped path, so none belongs in a commit that cannot be built or
measured. What this commit adds is the entry point and the one experiment that turns the argument above
into a proof:

```powershell
$env:NINFER_QWEN3_6_27B_WEIGHTS  = "$PWD\models\qwen3_8_27b.ninfer"
$env:NINFER_PREFILL_TRACE        = "1"
$env:NINFER_Q5_RESIDUAL_SPLITS   = "1"   # one writer per element, deterministic
& build-ninja\tests\ninfer_qwen3_6_27b_prefill_determinism_test.exe 2>&1 |
  Select-String 'prefill-trace|repeat:'
```

If the diagnosis is right, `NINFER_Q5_RESIDUAL_SPLITS=1` turns **every** `chunk=2 ... divergent_layers=64`
into `divergent_layers=0 first_divergent_layer=none`, and the `repeat:` deltas to zero, while the chunk 1
lines stay as they are. Any other outcome — chunk 2 still diverging, or chunk 1 changing — refutes it and
this section has to be rewritten. The switch is off by default so the shipped schedule is unchanged, and
it prints one `[q5] linear_add residual: 1 K-split` line when it is on, so a log cannot be misread about
which schedule produced it.

**Not verified here.** The path from that switch to a fixed engine: it needs a build, a GPU run, and the
benchmark that decides between options 1 and 2. The numerical oracle for the op is likewise unrun — a
deterministic schedule still has to match `LinearAdd` against its FP32 reference, and that is the check
that would catch a wrong merge order in option 2.

## Caveats on everything above

- Every configuration detail in the test is load-bearing, and two of them were wrong in the first
  version. Before trusting any conclusion drawn from it, confirm the repeat-prefill case still shows
  a non-zero delta. A clean run means the fixture has drifted, not that the bug is fixed.
- The GDN fingerprint evidence (Evidence 2) is a magnitude aggregate, not a checksum. It shows two
  states differ; it cannot distinguish float-order drift from a genuinely different computation.
- Evidence 1 is the only claim free of that weakness, because position 0's logprob is a directly
  measured scalar.
- Single machine, single artifact, `--k-blocks 2`, one run per configuration. No repeats.

## Notes

- `NINFER_PREFILL_DET_TOL` overrides the tolerance; set it to `0` to demand bit-identity. The
  default `1e-6` is a tripwire, not a numerical criterion, so the test currently **fails** — which is
  the intended state until the divergence is fixed.
- Run the binary directly for verbose output; under CTest a passing test's stdout is suppressed, so
  the verbose line needs `-V`. While the test fails, `--output-on-failure` is sufficient.
- The test is opt-in on `NINFER_QWEN3_6_27B_WEIGHTS`, matching the existing real-Engine tests.
- The `[scr-diag] gdn` instrumentation that produced the fingerprints in Evidence 2 is committed
  (`LinearAttentionStatePool::fingerprint_slot` plus `ProgramImplCore::log_scr_gdn_state`). It is
  diagnostic-only, every call site sits behind `scr_diagnostics_enabled()`, and `post_prefill` costs
  a PCIe round trip per request when enabled. It still needs a CTest against an independent oracle
  before it could ship as more than a diagnostic.
- Measurements are from a single machine (RTX 4090, sm_89) with `--k-blocks 2`.
- **Still open, and distinct from this issue:** why k2's SCR divergence (0.0801 nats) is an order
  of magnitude worse than every other k, when the amount relocated does not predict it. Fixing
  prefill reproducibility is a *precondition* for measuring that honestly — until the floor is
  valid, it is not knowable whether k2 is a real quality regression or perturbation sensitivity in
  a prompt that happened to place a near-tie at position 6 (its margin there is 0.000 nats).