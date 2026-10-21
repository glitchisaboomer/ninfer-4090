# Decode Optimization Findings

Single-GPU decode performance plan and diagnostic catalogue for Qwen3.8-27B on the RTX 4090
(`sm_89`, AD102, 128 SMs, 72 MB L2, 24 GB GDDR6X).

Baseline configuration under study: `--max-concurrency 1`, `--max-context 131144`,
`--kv-dtype rk4v4-e8`, `--spec mtp --draft-tokens 3 --lm-head-draft`, CUDA Graph enabled.

Governing constraint (AGENTS.md): mathematical fidelity first — every floating-point operator must
match an independent FP32/FP64 oracle. Nothing in this plan ships without oracle verification.

This document merges the earlier working plan (formerly a separate root-level `speedup_todo.md`)
with the earlier diagnostic catalogue. The plan sections state the analysis; the diagnostic
sections state unproven hypotheses. Neither is a live profile.

---

## Workload and evidence status

| Item | Current interpretation |
|---|---|
| GPU / model | RTX 4090; Qwen3.8-27B through the `qwen3_6_27b` path |
| Weights / context | Groupwise-int artifact; reported measurements around 40K context |
| KV cache / MTP | `rk4v4-e8`; K=3 with `--lm-head-draft` |
| Throughput | About 95 tok/s with MTP reported; the older 42 tok/s no-MTP comparison is not a matched verified baseline |
| Acceptance | Workload-dependent: the ~50% report and the documented AIME results (about 73–83%) refer to different workload/campaign conditions; neither should be generalized without matching fixtures and acceptance counters by position |

The 95 tok/s figure, the estimated bandwidth parity, and every acceptance-derived ceiling below
are **reported findings, not independently reproduced by this document**. Use the verification
gates before treating them as established or using them to close an optimization path. Published
RTX 5090 results do not establish RTX 4090 behavior.

---

## Plan status summary

**Reported facts that close out most of this plan:**

- **95 t/s average at 40K context**, which size-adjusts to **~94% of an equally-efficient engine**
  (prediction 100.7 t/s from issue #96's cold bench).
- **Base decode runs at 86% of the 4090's 1.008 TB/s peak** — i.e. at the memory wall.
- **MTP acceptance ≈ 50%**, which caps the greedy chain at **2.0 tokens/round**.

| # | Item | Expected | Status |
|---|---|---|---|
| 0 | Fix `$ExtraArgs` splatting in serving script | correctness | **TODO** |
| 0 | Confirm `text/draft_head` in artifact | gates 1.1 | **TODO** |
| 1.1 | `--lm-head-draft` | 10-15% | **Already applied** |
| 1.2 | MTP hidden copy-back → ping-pong buffers | 0.013% at K=5 | **RETIRED (measured)** — a copy node costs ~1.5 µs, linear in `K-1`: 4.05 µs at K=5 against a 31.31 ms round, and ~22 µs at K=15 against ~52 ms. Payload is 10 KiB (~10 ns), so the cost is launch overhead, not bandwidth. Below the round-time noise floor; no implementation exists in `src/` |
| 1.3 / 1.4 | GDN decode grid sizing | **RETIRED (measured)** | Production grid is 384 blocks × 128 threads = exactly 3.0 waves on 128 SMs. Prototype compared 4 grid/block variants under batch=1 with a bit-exactness gate: no candidate beat production (6–47% slower at T=16) |
| 2 | MTP verify attention at width 6 | ~+5% at K=5 | **MECHANISM FOUND (NCU, 40K)** — width 6 costs +72% per call (93.6 → 160.6 µs) because the third Q row tile pushes registers 128 → 238, dropping block residency to 1 and occupancy to 12.4%. DRAM falls 52% → 29% while instructions rise only 15%: latency-hiding collapse, not extra work. Fix is to restore 2-block residency at width 6 |
| 1.5 | INT8 group-64 activation prefill (from `sergiuszm/ninfer-4090`) | **+71% prefill / -7.6% decode** | **Open — different axis (TTFT)** |
| 2 | Cost × acceptance curve vs K | decision | **MEASURED at four context depths. Short context peaks at K=5 (+10% over K=3); at 44.5K, the campaign's operating point, K=3 and K=5 are a tie within noise; at 150K K=5 loses to K=3 by 11%. Width is not the lever: the marginal cost of a draft column at 44.5K is 51% MTP proposal and 42% attention** |
| 3 | Scan-based GDN verify | ~+6% ceiling | **RETIRED at the reported acceptance regime; reopen only per Gate B** |
| 4 | Tree attention | — | **RETIRED** — trees need high q |
| 5 | GDN state dtype (BF16/FP16) | 0.9% | **RETIRED** — below noise |
| — | L2 policy window, KV/MLP graph overlap | <1% | **Shelved** |
| **6** | **Raise per-position q** (self-speculative / Jacobi / lookahead) | only reported path past ~101 t/s | **DEPRIORITIZED — q is high at short context (0.74 at position 6) but decays with depth (0.61 by position 3 at 150K). Adaptive K is the measured win, not raising q** |

**Bottom line (updated after four Gate B runs and a 44.5K kernel profile):** the optimized draft head
is enabled. The roadmap's premise that base decode is at the memory wall and that acceptance caps the
chain at ~2.0 tokens/round is **falsified on this hardware** — measured per-position acceptance is
0.74–0.92 and tokens/round reaches 3.6–4.0. Two roadmap items are retired with measurements (GDN grid
sizing; the scan, whose cost model was wrong in kind). Wider K pays at short context (K=5, +66% over
K=1) and is a **tie at 44.5K**, the campaign's operating point, **losing only at 150K** — so the
production default does not change on width. The measured marginal cost of a draft column at 44.5K is
51% MTP proposal and 42% attention; both are reducible without touching K. See Gate B for the numbers
of record.

---

## The load-bearing architectural fact

**GDN verify is serial in K.** The MTP verify path calls `gated_delta_net_replay_record` →
`recurrent_record_kernel` → `recurrent_bf16_body`, which steps all K+1 columns through a serial
`for (token = 0; token < valid; ++token)` recurrence
(`src/ops/linear_attention/gated_delta_net/recurrent.cuh:733`). There is no column dimension in
the decode grid, and no scan kernel in the verify path.

Consequence: the critical path carries **48 GDN layers × (K+1) serial transitions** per round.
Layers cannot overlap (layer *i+1* needs layer *i*'s output), and within a layer the recurrence is
inherently sequential. Full-attention layers (16 of 64) verify in parallel; the other 48 do not.

This penalizes **all** K-scaling, not just tree attention — every extra draft token adds a serial
transition per GDN layer whether speculation is a chain or a tree.

Rough floor (one transition of one layer moves ~6 MiB = 3 MiB read + 3 MiB write):

| K | Transitions | Bandwidth floor | Latency-bound estimate |
|---|---|---|---|
| 3 | 48×4 = 192 | ~1.1 ms | ~2–3 ms (~10% of round) |
| 15 | 48×16 = 768 | ~4.6 ms | ~10–14 ms (~half the round) |

---

## Reported baseline: memory-wall parity

Measured **95 t/s average at 40K context** on the production artifact (16.62 GiB of weights,
~15.87 GiB read per decode token).

Size-adjusted prediction from `iamwavecut/ninfer-all` issue #96's cold-prefix bench (same model,
same quant, no speculation bonus): their 151.9 t/s at 44K × (10.52 / 15.87) = **100.7 t/s**
predicted for our artifact.

| | |
|---|---|
| Measured | 95 t/s |
| Predicted for an equally-efficient engine | 100.7 t/s |
| **Parity** | **~94%** |

**Reported conclusion: no engine-side deficit to chase.** Base decode is at **86% of peak
bandwidth** — essentially the memory wall. If confirmed by the Gate A/B measurements, further gains
must come from tokens-per-round.

---

## Reported acceptance: q ≈ 0.50 and the chain ceiling

> **Superseded by measurement.** The q ≈ 0.50 premise below is the roadmap's original claim. Three
> Gate B runs on the target hardware measured per-position q directly: **0.74–0.86 at short
> context**, decaying to **0.61 by position 3 at 150K context**. The chain does *not* saturate at
> 2.0 tokens/round — it reaches 3.6–4.0 at K=5–7 short-context, and 3.6 at K=5 long-context. The
> correct, measured answer is under **Gate B**; this section is retained to show what was believed
> and why it was wrong.

For a greedy chain, the expected accepted drafts is bounded:

```
E[accepted drafts] = sum_{i=1..K} q^i      (draft i needs all prior drafts accepted)
```

| K | q=0.50 tokens/round | q=0.70 | q=0.85 |
|---|---|---|---|
| 1 | 1.500 | 1.700 | 1.850 |
| 3 | 1.875 | 1.533+1=2.533 | 3.187 |
| 5 | 1.969 | 2.941 | 4.152 |
| 15 | 2.000 | 3.322 | 6.172 |

**At q=0.50 the chain saturates at exactly 2.0 tokens/round.** K=3 already delivers 1.875, which
is **94% of the theoretical maximum**. Widening K=3 → 15 buys **+6.7%**, and that is the ceiling
*even with a free, zero-cost scan*.

Cross-check against the reported 95 t/s: 95 / 1.875 = **50.7 t/s base**, which at 15.87 GiB/token
is 0.86 TB/s — 86% of peak. The two measurements are mutually consistent, which is reassuring, but
both rest on the same unreproduced inputs.

### Consequences (conditional on the reported q)

| Plan item | Verdict |
|---|---|
| Scan-based GDN verify (Phase 3) | **RETIRED on both counts.** The chain-saturation premise (q≈0.50 → 2.0 tokens/round) is falsified by measurement: reasoning per-position q is 0.74–0.86 and reaches 3.89 tokens/round at K=7. And the cost premise was wrong in kind (see the cost-curve section). **A scan would remove ~1 µs/column/layer of serial cost — i.e. almost nothing.** Not worth building. |
| Wider K (5, 7, 15) | **MEASURED, and the answer depends on context.** At short context K=5 wins (+66% over K=1, +10% over K=3). At 150K context **K=3 wins and K=5 is worse than K=3** — per-position acceptance decays with depth, so extra drafted columns stop paying for their round time. K=7 loses to the local optimum in every regime. The roadmap rejected wider K on a 2–14 ms serial-cost estimate that was ~15–19x too high per column; the real cost of K=3→5 is ~2.7 ms/round. **Net: no change to the production default; adaptive K is the actionable item.** |
| Tree attention | **RETIRED.** A tree helps only when q is *high* (many independent correct paths). At q=0.50 branching adds GDN serial work without adding accepted tokens. |
| AR re-propose batching | **Retired with the scan.** It existed to serve wider speculation. |
| GDN grid sizing | **Retired (measured).** Grid is already 3.0 exact waves; prototype found no faster variant. |

### What would actually move the needle at q=0.50

1. **Raise per-position q** — self-speculative decoding (draft from the same model's intermediate
   layers), Jacobi/iterative decoding, or lookahead. These attack the acceptance rate itself
   rather than the chain width.
2. **Reduce base cost** — not possible at 86% of peak bandwidth.
3. **Batching for aggregate throughput** — C=1 wastes the opportunity; admission is by capacity
   entitlement, so concurrency costs no context.

---

## Acceptance regime: reasoning is NOT the low-acceptance case

**Superseded by measurement.** This section recorded the roadmap's cross-campaign figures. The
Gate B run on this machine measured per-position acceptance directly and found the same direction:
reasoning is a *high*-acceptance workload. See "Measured result" under Gate B for the numbers of
record. The original figures are kept below for provenance.

Earlier cross-campaign figures (`docs/performance.md:230-250`, thinking **enabled** for the
long-reasoning corpus):

| Fixture | MTP acceptance | MTP tokens/round |
|---|---|---|
| `long_decode_aime26_01` | **82.8% ± 3.4%** | 3.48 |
| `long_decode_aime26_15` | **72.7% ± 1.4%** | 3.18 |
| `long_decode_aime26_30` | **80.1% ± 2.7%** | 3.40 |

The low-acceptance categories are the thinking-**disabled** cross-scenario fixtures: Story
**38.0%**, Translation 67.6%, Code 70.3%, Structured 89.1%.

**Reasoning (chain-of-thought) is the most formulaic token stream there is** — LaTeX scaffolding,
step numbering, repeated syntactic frames — so the MTP head predicts it *better* than free-form
prose. Do not assume a reasoning workload implies low acceptance; measure it.

Caveats that could break the analogy for a specific workload:

- AIME math is the most formulaic reasoning; long-horizon agentic reasoning with tool calls and
  JSON is structurally less predictable.
- Long context degrades acceptance: issue #96 measured 0.724 (145 tok) → 0.603 (133K) on real
  agent traffic.
- The tok/s in the table above are from a different artifact/hardware campaign; transfer the
  acceptance *percentages*, not the throughput.

**Resolved by measurement.** The ~50% figure and the 73–83% AIME figures are reconciled by
the Gate B run: they were never the same quantity. The ~50% is close to the *aggregate*
`accepted/drafted` metric (measured 0.489 for free-form at K=3), while the AIME figures and the
Gate B per-position numbers (0.74–0.83 for reasoning) measure per-position q. They differ by
roughly 1.4x by construction, and they also differ by workload and context. See "Which q is which"
under Gate B.

---

## Verification roadmap (execution checklist)

This section turns the items above into reproducible evidence gates. No builds or model runs are
implied by this plan; commands below are run only when explicitly scheduled. Use the same RTX 4090,
driver, artifact checksum, power/clock policy, context, corpus, seed, graph mode, and server
settings for each paired comparison. Record the git SHA, artifact SHA-256, command line, GPU
clocks/temperature, and raw JSON/JSONL/CSV under `profiles/bench/<date>-decode-speedup/`.

### Gate A — Establish valid inputs and a trustworthy baseline

1. **Serving-script argument forwarding.** Inspect/fix `scripts/run-qwen38-c1-reduced.ps1`; run it
   with a harmless diagnostic extra argument and confirm the launched process receives it
   (PowerShell transcript or process command line). Then run the normal launch and confirm the
   effective server options in startup/request logs. This is script correctness, not a throughput
   claim; retain the invocation and log as evidence.
2. **Draft-head artifact.** Run the documented artifact inspector on the exact artifact; retain its
   object inventory and verify names, shapes, and encodings against the table below. Also confirm
   startup reports the optimized proposal head active. Missing/mismatched objects block any
   `--lm-head-draft` comparison; do not interpret silent fallback as a win.
3. **Paired baseline / optimized head.** On the same fixed prompt and generation length, run at
   least 5 paired repetitions with MTP K=3, once with the full proposal head and once with
   `--lm-head-draft`. Record output tokens/s, engine tokens/s, rounds, drafted/accepted counts,
   acceptance by position, and per-round latency. Require greedy output equality (or explain a
   documented expected difference); compare medians and spread, not a best run. Report both total
   round throughput and acceptance so the head's speed/quality tradeoff is visible.

**Gate:** artifact and option are confirmed, and paired runs are stable enough to interpret. Do not
advance based on an unverified flag or a single run.

### Gate B — Resolve the workload acceptance/cost curve (decides whether the scan stays retired)

**Script:** `tools/bench/run_gate_b_acceptance_sweep.py` automates this entire gate. See the section
below for usage.

#### Which `q` is which

Two different quantities are both loosely called "acceptance", and conflating them produces wrong
conclusions:

| Quantity | Definition | At true per-position q=0.50, K=3 |
|---|---|---|
| **per-position q_i** | `accepted_at_i / drafted_at_i` | 0.500 at every position |
| **aggregate acceptance** | `accepted_tokens / drafted_tokens` | 0.292 |

The aggregate is much lower than per-position q, because a column only counts as *drafted* in the
rounds that reached it, yet the denominator is the full `rounds × K`. The roadmap's tokens/round
table (`sum q^i`) is built from **per-position q**, so that is the quantity this gate must extract.
The script derives it with exact denominators: `drafted_at_0 = rounds`, and
`drafted_at_i = accepted_at_{i-1}`.

This derivation was verified by reproducing the roadmap's own table exactly from synthetic data:
per-position q=0.50 at K=1/3/15 gives 1.500 / 1.875 / 2.000 tokens per round, and q=0.85 at K=3
gives 3.187 — matching the published figures. So a real run reporting **aggregate** acceptance of
0.29 at K=3 is consistent with per-position q=0.50, not evidence of lower acceptance. This matters
because the roadmap's central contradiction ("~50%" vs "73–83%") may be partly a unit mismatch
rather than a workload mismatch.

1. Select and freeze a representative real Qwen3.8 reasoning prompt/corpus, sampling mode, seed,
   context length (include the target 40K case and at least one longer-context case), and output
   budget. Include a lower-acceptance control such as story/free-form text; do not substitute one
   fixture's result for another workload.
2. Sweep K = 1, 3, 5, and 15 if supported by the serving runtime. Collect at least 5 runs per
   point. Report requested-output throughput and engine throughput separately, plus rounds,
   drafted/accepted tokens, acceptance histogram by proposal position, and wall/GPU time. Aggregate
   acceptance as accepted-at-position / drafted-at-position; do not infer q from a single scalar
   acceptance field. Keep all raw request logs.
3. Measure the actual GDN verify cost with the existing public GDN benchmark for supported snapshot
   widths, then correlate with end-to-end per-round latency. Do not treat isolated Op timing as the
   whole-model cost: verify is repeated across layers and includes surrounding projections.
4. Plot incremental accepted tokens versus incremental round time for each K. A wider K or scan is
   considered only if the real workload retains substantial acceptance at later positions and the
   extra accepted tokens can exceed the measured extra verify cost. Otherwise record the negative
   result and close the scan/tree branch for that workload.

**Decision rule:** the scan is not reopened from a theoretical acceptance rate. Reopen only with
reproducible high later-position acceptance *and* end-to-end evidence that serial verify cost
consumes enough round time to make extra accepted positions profitable.

#### Measured result (RTX 4090, 27B artifact, batch=1)

Measured with `tools/bench/run_gate_b_acceptance_sweep.py`. Conditions: `--max-context 4096`,
`--kv-dtype rk4v4-e8`, `--lm-head-draft`, `--max-concurrency 1`, greedy, seed 57004, 512 output
tokens, three prompts per workload per K (four for reasoning). Raw data:
`profiles/bench/gate-b/`.

**Per-position acceptance q_i = accepted_at_i / drafted_at_i** (exact denominators):

| Workload | K=1 | K=2 | K=3 | K=5 | K=7 |
|---|---|---|---|---|---|
| reasoning | 0.832 | 0.860 / 0.786 | 0.858 / 0.796 / 0.784 | 0.838 / 0.768 / 0.756 / 0.754 / 0.736 | 0.815 … 0.743 (7 positions) |
| freeform | 0.720 | 0.703 / 0.642 | 0.706 / 0.654 / 0.649 | 0.719 / 0.567 / 0.582 / 0.687 / 0.779 | 0.684 … 0.619 |
| structured | 0.855 | 0.840 / 0.806 | 0.876 / 0.842 / 0.815 | 0.809 / 0.755 / 0.797 / 0.824 / 0.851 | 0.794 … 0.848 |

**Decode throughput and round cost:**

| Workload | K=1 | K=2 | K=3 | K=5 | K=7 |
|---|---|---|---|---|---|
| reasoning tok/s | 84.3 | 111.0 | 128.1 | **135.0** | 133.4 |
| reasoning tok/round | 1.832 | 2.536 | 3.078 | 3.605 | 3.891 |
| reasoning ms/round | 21.81 | 22.90 | 24.08 | 26.75 | 29.25 |
| freeform tok/s | 79.2 | 93.9 | **102.2** | 99.0 | 88.2 |
| structured tok/s | 85.3 | 109.2 | 133.1 | **136.0** | 132.7 |

#### Conclusions — the roadmap's q ≈ 0.50 premise is falsified on this machine

1. **Per-position acceptance is high, not ~0.50.** On reasoning it is 0.83 at position 0 and still
   **0.74 at position 6**. The roadmap's "chain saturates at 2.0 tokens/round at q=0.50" does not
   apply: measured tokens/round at K=7 is **3.89**, not 2.0.
2. **The decay is slow, not collapsing.** The roadmap's own Gate B decision rule says a slow decay
   means wider speculation pays. It does. Reasoning peaks at **K=5 (+60% over K=1)** and K=7 is
   within 1.2% of that peak.
3. **The cost model correction and the acceptance measurement agree.** K=3 → K=5 costs only ~2.7 ms
   per round of extra GDN verify (26.75 − 24.08), matching the corrected ~1 µs/column/layer model
   rather than the roadmap's ~8 ms estimate. Wider K is affordable *and* productive.
4. **K=7 is past the optimum.** Round time keeps rising (+2.5 ms from K=5 to K=7) while
   tokens/round gains flatten, so K=5 is the operating point for reasoning and structured, K=3 for
   free-form prose. This is the adaptive-width case the roadmap described as a research lead.
5. **The ~50% figure was almost certainly the aggregate metric, not per-position q.** At K=3 the
   measured aggregate (`accepted/drafted`) is 0.695 for reasoning and 0.489 for free-form — the
   latter is very close to the roadmap's "~50%". So the roadmap's contradiction resolves as a unit
   mismatch plus a workload difference, not a hardware difference. See the "Which q is which" note
   above.

#### Second run: longer outputs (`--max-context 262144 --output-tokens 1024`)

Raw data: `profiles/bench/gate-b-64k/`. Per-position q and throughput were both re-measured.

| Workload | K=1 | K=3 | K=5 | K=7 |
|---|---|---|---|---|
| reasoning tok/s | 86.7 | 127.6 | **140.9** | 134.9 |
| reasoning tok/round | 1.886 | 3.078 | 3.789 | 3.961 |
| freeform tok/s | 81.1 | **109.8** | 106.0 | 100.1 |
| structured tok/s | 87.4 | 122.6 | **136.5** | 135.4 |

Reasoning at K=5 improved from 135.0 to 140.9 tok/s with the longer output, and per-position q was
essentially unchanged (reasoning K=5: 0.847/0.802/0.800/0.765/0.728 vs
0.838/0.768/0.756/0.754/0.736), so the acceptance measurement is stable across output lengths. **At
the time this read as "the K=5 peak survives"; the third run at 150K context overturned it.**

**Important limitation: this run did NOT test long context.** `--max-context 262144` only raised the
ceiling; the prompts were ~90 prefill tokens, so the deepest context any request reached was
prefill + 1024 output ≈ **1,194 tokens**, not 40K. Raising the window does not exercise long-context
behaviour — the *prompts* have to be long.

The script now prints a **"Context depth actually reached"** table (server-reported
`computed_prefill_tokens` + generated tokens) precisely so this cannot be missed again: a run whose
`final_ctx` column is ~1K is not a long-context run no matter what `--max-context` was set to.

Related correction: the `longctx` workload's padding calibration was initially stated as "~110
tokens per repeat", which was wrong. The paragraph is 607 chars / 106 words, i.e. roughly **140
tokens per repeat**, so `--longctx-repeats 260` lands near 36K tokens rather than 39.5K. The estimate
is approximate by nature — calibrate `--longctx-repeats` against the reported `prefill_tok`, not
against a chars-based rule of thumb.

#### Caveats on the first two runs (short context)

- **Long context was untested in these two runs.** Both used ~90-token prompts. Issue #96 measured
  acceptance falling 0.724 → 0.603 going from 145 to 133K context, so the 40K regime could look
  materially worse — **and it did, which is why the third run exists below.**
- **Three to four prompts per point.** Enough to establish the trend, not enough for a confidence
  interval. Per-request spread is large (reasoning/K=3 tokens/round ranges 2.57–3.52 in the first
  run), so treat the peak *location* (K=5) as more reliable than the exact peak *value*.
- **Both runs hit the output cap** (512 and 1024 tokens respectively), so neither measures
  steady-state behaviour past its cap.
- **Same machine, same artifact, all three workloads, both runs** — which is what makes the workload
  comparison meaningful, unlike the roadmap's cross-campaign figures.

#### Third run: genuine long context (150,591-token prompts) — **the conclusion flips**

Raw data: `profiles/bench/gate-b-64k/` (this run overwrote the previous one in place; the longctx
points carry `computed_prefill_tokens = 150591`, far deeper than the campaign's 40K target).

| K | q0 | q1 | q2 | q3 | q4 | q5 | q6 | tok/round | dec tok/s | ms/round |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.851 / 0.872 | — | — | — | — | — | — | 1.851 / 1.872 | **70.3 / 70.9** | 26.4 |
| 3 | 0.841 | 0.783 | 0.772 | — | — | — | — | 3.009 | **99.9** | 30.2 |
| 5 | 0.876 | 0.741 | 0.776 | 0.690 | 0.724 | — | — | 3.628 | 99.1 | 36.7 |
| 7 | 0.830 | 0.713 | 0.719 | 0.610 | 0.720 | 0.722 | 0.795 | 3.536 | 88.9 | 39.9 |

**K=5 no longer wins. At 150K context the peak moves to K=3**, and K=7 is clearly worse than K=3
(88.9 vs 99.9 tok/s, −11%).

Why: per-position acceptance decays much faster at depth. At K=5, q falls 0.876 → 0.690 by position
3, so the extra drafted columns mostly get rejected — tokens/round still rises (3.628 vs 3.009) but
round time rises faster (36.7 vs 30.2 ms), and throughput goes *down*. At short context the same K=5
was the optimum because the decay was shallow enough to pay for the extra columns.

**This is exactly the failure mode the roadmap's Gate B was designed to catch**, and it is why the
long-context run was mandatory: the short-context measurement pointed at K=5, and it would have been
wrong for the campaign's actual operating point.

#### Decomposition of the K=3 → K=5 round-time gap (the "+20% is locked behind 6.5 ms" question)

The observation that motivated this: at 150K context, K=5 delivers **+20.6% tokens/round** but costs
**+21.5% ms/round**, so the two cancel (99.1 vs 99.9 tok/s). If the 6.50 ms of extra round time were
removable, K=5 would give ~120 tok/s, **+20% over K=3**. Both candidate sources were measured.

**#2 Attention first — ruled out.** `ninfer_causal_softmax_attention_bench`, entry `append`, geometry
`d256-h24-kv4` (the 27B shape), 150K context, cold L2:

| Verify width (cols) | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| µs per layer | 421.9 | 420.9 | 427.0 | 428.0 | 428.0 | 455.7 | 463.9 |

Essentially flat. Attention is **KV-history-read-bound**, so extra query columns are nearly free.
Across the 16 full-attention layers, K=3 → K=5 adds **0.10 ms — 2% of the 6.50 ms**.

**#1 Rejected-column waste — real, but not where it was expected.** The mechanism was verified in
source: `mtp_decode_batch_body` (`src/targets/qwen3_6/impl/runtime/mtp_impl.h:161-184`) runs
`mtp_forward_decode_batch` **once per draft step at width = 1**, so every drafted column — accepted or
rejected — pays a full MTP-layer pass. From the real tensor shapes in
`src/targets/qwen3_6_27b/impl/load/bindings.cpp` and the `W8G32_F16S` encoding (34 B / 32 elem):

| Component | Bytes |
|---|---|
| MTP layer (input_proj, QKGV, attn out, MLP up/down) | 430.4 MiB |
| draft head, 131072×5120 Q4G64 | 340.0 MiB |
| **Total per proposal step** | **770.4 MiB ≈ 0.93 ms at 86% of peak** |

At K=5, 150K context, 662 of 1403 drafted columns were rejected (47%), so wasted proposal time is
~2.19 ms/round — **6% of the 36.65 ms round**. Real, but not the answer.

**Where the 6.50 ms actually is.** Fitting ms/round against K gives a marginal cost of **2.34 ms per
draft column**, against measured per-column costs of:

| Source | Per column | Evidence |
|---|---|---|
| GDN recurrence | 0.042 ms | cost-curve prototype, ~0.88 µs/col/layer × 48 |
| Full-attention layers | 0.025 ms | `ninfer_causal_softmax_attention_bench`, above |
| MTP proposal step | 0.930 ms | tensor sizes × 86% of peak |
| **Accounted for** | **~1.00 ms** | |
| **Observed** | **2.34 ms** | **~1.34 ms (57%) unattributed at op level** |

> **Superseded by the Nsight profile below.** This op-level decomposition under-counted attention and
> the MTP proposal. The profile shows the marginal cost is dominated by `w8_small_t_mma_kernel`
> (MTP proposal + output head, cost grows via call count) and by attention, which nearly doubles per
> call at wider verify width despite the isolated benchmark showing it flat. The `+0.042 ms` GDN
> figure below is correct; the attention figure is not a valid proxy for the real path.

Two measurements pin down the rest of the structure:

- **The body weights are NOT re-read per column.** `ninfer_q4_linear_swiglu_bench` on the
  `[34816,5120] → [17408,T]` profile (5.645 GiB, 68% of body traffic) is flat in T: 144.4 µs at T=1
  vs 146.4 µs at T=8. A T-wide quantized GEMM reads the weight once for all T columns, so batching
  K+1 columns into one call makes the MLP free with respect to K. This rules out the obvious
  hypothesis that wider K multiplies weight traffic.
- **The remaining ~1.34 ms/column is unmeasured.** It is consistent with per-column work in the
  target-model verify that was not isolated here: KV append, RoPE, norms, the GDN input
  projection/conv, the output head, and sampling. No evidence yet for which dominates.

#### Fourth run: 44.5K context, the campaign's operating point

44,546 prompt tokens, `qwen3_8_27b.ninfer`, 256 output tokens, greedy, seed 57004. Round counts,
tok/rd, ms/rd and acceptance come from the profiled request; tok/s lists every sample at this depth
(K=3 ×3, K=5 ×2). Raw data: `profiles/verify-profile/`.

| K | rounds | tok/rd | ms/rd | tok/s (all samples) | agg accept | q by position |
|---|---|---|---|---|---|---|
| 1 | 132 | 1.939 | 25.34 | 76.5 | 0.924 | 0.924 |
| 3 | 76 | 3.368 | 28.32 | **119.0 / 114.8 / 110.7** | 0.788 | 0.908 / 0.826 / 0.912 |
| 5 | 73 | 3.507 | 31.31 | 115.2 / 112.0 | 0.508 | 0.877 / 0.734 / 0.787 / 0.622 / 0.478 |

**K=3 and K=5 are a tie at 44.5K** (spread within K exceeds the gap); both 46–55% over K=1. The same
shape on `swift1_5_27b` (45,435-token prompts) also ties: K=3 119.2 / 105.4, K=5 118.1 / 118.1. The
extra columns do earn tokens (+0.14 tok/rd) but cost **1.50–1.94 ms per draft column** across sample
pairs.

Aggregated kernel time for those two requests (Nsight, prefill excluded, per round):

| Component | K=3 µs/rd | K=5 µs/rd | Δ | share of Δ |
|---|---|---|---|---|
| Body GEMM (weights) | 18,829 | 18,969 | +140 | 4% |
| Attention (`gqa_attention_decode_i8_tiled`) | 2,124 | 3,688 | +1,564 | 42% |
| MTP layer (W8G32, 100 µs/call) | 1,525 | 2,589 | +1,064 | 29% |
| Draft head (Q4G64, 380 µs/call) | 1,154 | 1,966 | +812 | 22% |
| Output head, rmsnorm, GDN | 2,722 | 2,907 | +185 | 5% |
| Other (rope, state passing) | 509 | 454 | −55 | −1% |
| **Total GPU** | **26,863** | **30,573** | **+3,710** | |

The body is amortized (+0.7% for two more columns). **MTP proposal (layer + head) is 51% of the
marginal cost, attention 42%.** Attention runs 19.4 → 21.7 calls/rd at 96.0 → 155.6 µs/call, i.e. it
gets 62% slower per call with width, against an isolated benchmark that reports it flat.

Action plan, in order:

1. **Find attention's width scaling** (42% of the delta, no artifact change). `ncu` on
   `gqa_attention_decode_i8_tiled` at verify width 4 vs 6 at 44.5K: compare achieved occupancy, L2
   hit rate, and the mask/page-layout path against the isolated bench. Success = per-call time flat
   in width; worth ~+5% at K=5.

   **Mechanism found (NCU, width 4 vs 6, at depth).** Profiled on the production verify kernel with
   `tools/bench/profile_attention_width_ncu.py --launch-skip 128`, capturing a launch between the
   request's submit and completion at `prompt_tokens 40736`:

   | metric | width 4 (K=3) | width 6 (K=5) | delta |
   |---|---|---|---|
   | specialization | `<4, 8, 2, 32>` | `<6, 6, 1, 32>` | — |
   | grid × block | 256 × 256 | 256 × 192 | block −25% |
   | **duration** | **93.60 µs** | **160.64 µs** | **+71.6%** |
   | **registers/thread** | **128** | **238** | **+86%** |
   | block limit (registers) | 2 | **1** | −50% |
   | **theoretical occupancy** | **33.3%** | **12.5%** | **−62.5%** |
   | **achieved occupancy** | **31.25%** | **12.41%** | **−60.3%** |
   | eligible warps/scheduler | 0.60 | 0.28 | −53.3% |
   | no eligible | 61.9% | 75.3% | +21.6% |
   | **DRAM throughput** | **52.4%** | **29.0%** | **−44.6%** |
   | L2 hit rate | 35.9% | 32.4% | −9.6% |
   | executed instructions | 44.9 M | 51.7 M | +15.2% |

   This reproduces the Nsys per-call figure independently: **96.0 → 155.6 µs (+62%)** there,
   **93.6 → 160.6 µs (+71.6%)** here. The isolated `causal_softmax_attention_bench` reporting it flat
   was the wrong proxy, and that question is now settled.

   The cause is a **third Q row tile**: `TokenTile × GroupSize` is 24 rows at width 4 (2 row tiles)
   but 36 at width 6 (3 row tiles), so the launcher selects `<6, 6, 1, 32>` instead of
   `<4, 8, 2, 32>`. At 238 registers `Block Limit Registers` falls to 1 and occupancy — theoretical
   and achieved alike — collapses to ~12%: one 192-thread block per SM. The kernel is
   **memory-bound at width 4 and starved of warps at width 6**: DRAM utilisation falls 52% → 29%
   while duration rises 72%, and executed instructions rise only 15%, so the extra time is not extra
   work — it is the loss of latency hiding. It is the repository's own `__launch_bounds__` trade
   from `AGENTS.md` landing on the wrong side: 128 registers buy occupancy at width 4, and the extra
   row tile spends it.

   **A warmup capture gets the bottleneck wrong, not just the µs.** Two earlier runs (skip 0 and
   skip 16) profiled the server's warmup instead — `max_tokens=4` over 16 full-attention layers puts
   ~64 matching launches before the request, so a skip of 16 lands inside it. That capture showed a
   16-block grid at DRAM 1.4% and read as *latency-bound*; at real depth the same kernel runs at 52%
   of DRAM peak. `Grid Size` is the discriminator (16 blocks vs 256), and the tool now warns when the
   captured grid is too small for the requested depth rather than reporting numbers for the wrong
   launch.

   **Next experiment, now targeted:** the width-6 specialization already concedes
   `MinBlocksPerSm = 1`. Cutting its register demand enough to restore 2-block residency — fewer
   live accumulators per row tile, or a serialized third row tile — is the specific change that would
   flatten the width curve; nothing in the memory path needs to move. At 29% DRAM against 52% for
   width 4, there is headroom for a second resident block to exploit. Measure with
   `--launch-skip 128` and check `Achieved Occupancy` and `DRAM Throughput` move together.
2. **Halve the proposal bytes** (51% of the delta). 770 MiB/column today; Q4G64 MTP layer + a
   65,536-row draft head gives 385 MiB (~880 → ~450 µs/column), ~+5% at K=3 and ~+8% at K=5. Gate on
   per-position acceptance at 44.5K within ~1% of the current curve.
3. **Repeat the measurement before choosing K.** Three requests per K and a second prompt at 44.5K:
   q₀ varies ±3% between runs that should be identical, wider than the K=3/K=5 gap, so do not adopt
   adaptive K on the current data.
4. **Keep K=3 as the production default** until 1–2 land.

#### Short-context profile (superseded)

An earlier short-context run (`--no-cuda-graph`, ~100-token prompts) was the first attempt at this
split. Its per-kernel rows mixed prefill into decode and mistook the MTP layer for the output head, so
the analysis has been dropped; the 44.5K trace above is the authoritative split. Raw data for both
runs: `profiles/verify-profile/`.

#### How to profile the verify path (beginner instructions)

`tools/bench/profile_verify_path.py` does this end to end. It is pure Python, so it runs on Windows
with no extra tooling.

```bat
python tools\bench\profile_verify_path.py ^
    --server build\apps\Release\ninfer-serve.exe ^
    --model  E:\models\qwen3_8_27b.ninfer
```

or in PowerShell:

```powershell
python tools/bench/profile_verify_path.py `
    --server build/apps/Release/ninfer-serve.exe `
    --model  E:/models/qwen3_8_27b.ninfer
```

It launches the server under Nsight Systems once per K, sends a warmup request and then one measured
request, and prints a table of GPU kernel time per kernel name sorted by how much it grew between the
two K values. The top of that table is where the extra round time goes.

**Why short context is fine, and better.** The extra cost is per-drafted-column work inside verify —
KV append, RoPE, norms, GDN input projection/conv, output head, sampling — none of which depends on
context depth. A short prompt means prefill is instant, the trace has far fewer kernels, and
attention (already measured flat in K) becomes negligible, so the per-column work stands out *more*
clearly. The default is `--max-context 4096` with ~100-token prompts. Attention's share grows with
depth, so profile at the context you care about before trusting its percentages.

**One necessary flag: `--no-cuda-graph`.** Verify normally runs inside a captured CUDA Graph, which
nsys reports as a single opaque node — that would hide exactly what we want to see. The script passes
`--no-cuda-graph` so every kernel is visible. Absolute timings are inflated by launch overhead as a
result; the K=3-vs-K=5 *delta* is what carries the information.

If nsys is not on `PATH`, the script probes the usual Windows install locations
(`C:\Program Files\NVIDIA Corporation\Nsight Systems <version>\target-windows-x64\nsys.exe`);
override with `--nsys` if yours lives elsewhere.

Useful variations:

```bat
rem Compare more widths, or different ones
python tools\bench\profile_verify_path.py --server ... --model ... --ks 3,5,7

rem Longer generation so the steady-state verify dominates the trace
python tools\bench\profile_verify_path.py --server ... --model ... --output-tokens 1024

rem One long-context confirmation run
python tools\bench\profile_verify_path.py --server ... --model ... --max-context 65536 --output-tokens 1024
```

For detail on a single kernel (occupancy, memory throughput, stall reasons) use Nsight Compute on the
kernel name that tops the table:

```bat
ncu --kernel-name regex:<kernel_name> --launch-count 1 <the same server command line>
```

What to look for: kernels whose **count** grows with K are per-column work (that is the shape we
expect for the unexplained 1.34 ms). Kernels whose **time** grows but whose count is flat are getting
slower per call, which would point at cache or occupancy effects instead. A kernel that appears only
at the higher K — like the MTP proposal loop, which is absent at K=1 — is proposal-side cost.

Copy the `profiles\verify-profile` directory back into the repo to continue the analysis.

#### Final recommendation

| Regime | Best K | Gain vs K=1 | Gain vs current K=3 baseline |
|---|---|---|---|
| Short context (~1.2K, 512–1024 tok out) | **K=5** | +66% | +10% |
| **44.5K (the campaign's operating point)** | **K=3 / K=5 tie** | K=3 +55%, K=5 +46% | **0% — within run-to-run noise** |
| Long context (150K) | **K=3** (already the baseline) | +42% | **0% — no change** |

**Do not change the production default.** The campaign runs at `--max-context 131144`, and at both
depths inside that window K=3 is either optimal (150K) or tied (44.5K). The K=5 win exists only in the
short-context regime, which is not the deployment. Width is not where the remaining time is: at 44.5K
a draft column costs 1.50–1.94 ms, split 51% MTP proposal and 42% attention, and both are reducible
without touching K.

Also actionable: short-context deployments get the K=5 win as a one-flag change (`--draft-tokens 5`,
+10%), and wider K (7/10/15) is rejected at every measured depth. Adaptive width (K from measured
context) needs captured K-specific graphs, so it waits until items 1–2 of the action plan above land.
2. **If any deployment actually runs short contexts**, `--draft-tokens 5` is a one-flag +10%.
3. **Wider K (7, 10, 15) is rejected at all measured context depths.** K=7 is worse than the local
   optimum in every regime, so the roadmap's instinct there was right even though its reasoning
   (a 2–14 ms serial-cost estimate) was wrong.

#### Caveat on the long-context measurement

- **One prompt, one request per K.** No spread, no confidence interval. The direction (K=3 > K=5 >
  K=7 at depth) is clear and consistent with the per-position decay, but the exact crossover point
  is not pinned. A campaign at 40K — the actual target — sits between the two regimes measured here
  and could favour either K.
- **150K is deeper than the campaign's 40K target.** That gap has since been closed by the fourth run
  above (44,546 prompt tokens), which found K=3 and K=5 tied. It is still one prompt per K, so it
  should be repeated with more prompts before any adaptive policy is locked in.
- Two K=1 server instances ran (a duplicate point), which is harmless — both agree at ~70.6 tok/s.

#### Running the sweep

`tools/bench/run_gate_b_acceptance_sweep.py` runs the whole gate: one server per point (the memory
plan and execution graphs are startup-fixed, so K cannot change in-process), the same prompts at
every K, then it parses the structured request log and derives per-position acceptance.

```bash
# Windows, from the repo root
python tools/bench/run_gate_b_acceptance_sweep.py \
    --server build/apps/Release/ninfer-serve.exe \
    --model  E:/models/qwen3_8_27b.ninfer \
    --out    profiles/bench/gate-b \
    --workloads reasoning,freeform,structured \
    --ks 1,2,3,5,7,10,15

# Longer context, if VRAM allows (this is the case the campaign targets)
python tools/bench/run_gate_b_acceptance_sweep.py \
    --server build/apps/Release/ninfer-serve.exe \
    --model  E:/models/qwen3_8_27b.ninfer \
    --out    profiles/bench/gate-b-40k \
    --max-context 65536 --output-tokens 1024
```

Every point is held fixed except K: `--max-concurrency 1`, `--kv-dtype rk4v4-e8`,
`--lm-head-draft`, `temperature 0`, `seed 57004`, same prompts. Override any of these with flags.

Outputs, under `--out`:

| File | Contents |
|---|---|
| `gate_b_points.json` | every point in full: counters, histograms, derived per-position q, decode tok/s, ms/round, and the exact server command line |
| `gate_b_summary.csv` | one row per point, spreadsheet-ready |
| `server.requests.jsonl` | the raw, append-mode structured request log (full precision, no rounding) |
| `server-logs/<workload>-k<K>.log` | each server's stderr, for diagnosing startup or OOM |

The console prints the per-position table with the Gate B decision rule already applied
("collapses after early positions" vs "slow decay across positions"), but the JSON/CSV are the
artifacts of record — do not re-type numbers from the console.

Copy the `profiles/bench/gate-b*` directories back into the repo to continue the analysis.

#### What to look for

- **`q_0` near 1.0 with `q_1` already low** → chain-limited. The roadmap's conclusion holds and wider
  K is dead regardless of the corrected cost model.
- **Slow decay (`q_3`+ still above ~0.3)** → the corrected cost model makes wider K genuinely
  attractive now, because the round cost of K=3→15 is only ~0.54 ms. This is the case the roadmap
  could not see, because it believed the cost was ~8 ms.
- **`tokens_per_round` rising with K** → confirm it rises faster than `ms_per_round`. The corrected
  model says ms/round grows ~49 µs per extra column, so almost any acceptance gain clears the bar.
- **A single aggregate number** → not enough. The histogram is the decision input.

### Gate C — Free / low-risk performance candidates

For any code change, the full CTest suite must run and pass before the change is considered
complete. Until then, use these focused suites as fast iteration signals; the full suite remains
the final acceptance gate. All tests live in `tests/` and are registered by `tests/CMakeLists.txt`.

| Candidate | Focused CTest targets to run | Evidence to retain |
|---|---|---|
| GDN launch/grid sizing — **retired by measurement** | `ninfer_gated_delta_net_test`, `ninfer_gated_delta_net_replay_record_test`, `ninfer_gdn_replay_fold_test`, `ninfer_gdn_input_proj_conv_snapshot_test`, `ninfer_gdn_input_proj_conv_record_test`, `ninfer_gdn_replay_records_test` | Not applicable — no change proposed. The prototype's negative result is the evidence; rerun these only if a new grid hypothesis appears |
| INT8 group-64 activation prefill | `ninfer_linear_q4_a16_test`, `ninfer_linear_q5_a16_test`, `ninfer_linear_w8_a16_test`, `ninfer_linear_add_bf16_a16_test`, `ninfer_linear_add_w8_a16_test`, plus new oracle tests | Paired prefill (`pp`) and decode (`tg`) benchmarks, new FP32/FP64 oracle test results |
| State dtype, L2 policy window, KV/MLP overlap (shelved items) | Op tests for the touched kernels (`ninfer_gqa_kv_page_delta_rotated_test`, `ninfer_gqa_kv_page_delta_i8_test`, `ninfer_test_e8_codec` where applicable) | Microbenchmark plus end-to-end A/B; state dtype also needs long-context drift evidence |
| Program/MTP policy changes (adaptive width, acceptance research) | `ninfer_mtp_round_test`, `ninfer_mtp_pack_test`, `ninfer_speculative_round_test`, `ninfer_serve_options_test`, `ninfer_request_log_test`, `ninfer_bench_support_test`, `ninfer_qwen3_6_runtime_mechanisms_test` | Acceptance-by-position traces for each fixed K, paired greedy-equivalence outputs, policy simulation results |
| Artifact/loader changes (draft head, prefill quant) | `ninfer_artifact_reader_test`, `ninfer_artifact_materialization_test`, `ninfer_qwen3_6_27b_load_plan_test`, `ninfer_bench_support_test` | Artifact inspector output, loader log, artifact SHA-256 |

`ninfer_add_op_test` registrations set `SKIP_RETURN_CODE 77`, so a skipped op test means no
compatible GPU was found; it is not a pass. Several engine and artifact tests are also skip-gated;
treat every skip as a blocked gate and rerun on the target RTX 4090.

- **GDN launch/grid sizing (prototyped and retired):** the production grid is already exactly
  3.0 waves (384 blocks on 128 SMs), so there is no wave-alignment defect. A prototype compared four
  grid/block variants under batch=1 across T=2..16 with a bit-exactness gate, and **no candidate
  beat production** (6–47% slower at the stable T=16 point; the one faster-looking variant was not
  bit-exact and is invalid). No production change is proposed. See the item above for the table.
  Do not revisit without a new hypothesis that is not about block count or wave count.
- **INT8 group-64 activation prefill:** identify the exact upstream change and its mathematical
  contract before porting. Establish public benchmark and end-to-end prefill/decode baselines with
  matched prompts. Add independent FP32/FP64 oracle tests for quantization, dequantization,
  scaling, and boundary/group-tail cases, plus regression coverage for supported shapes. Measure
  TTFT/prefill and decode separately (the expected decode regression matters); land only if the
  measured prefill benefit justifies the decode cost for the intended serving profile.
- **L2 policy window / overlap / state dtype (shelved items):** leave shelved absent a new measured
  bottleneck. Any reconsideration needs a scoped microbenchmark and end-to-end A/B; state dtype
  additionally requires long-context oracle/error-drift validation, not one-step parity.

### Gate D — Research candidates that change per-position acceptance

**Deprioritized, but for a different reason than first recorded.** The roadmap believed per-position
q was the binding constraint at ~0.50. Measurement shows q is high at short context (0.74–0.86) but
decays with depth (0.61 by position 3 at 150K), so raising q is not the binding constraint either —
**choosing K to match context depth is.** That makes this a scheduling/config problem rather than a
proposal-quality problem.

Keep the protocol below for the record. It is the right process for any future proposal-quality
work, and it would apply if the adaptive-K route is exhausted.

For self-speculation, Jacobi, and lookahead, do a minimal algorithm/architecture prototype only
after documenting where it enters Program and how it interacts with the serial GDN recurrence,
CUDA Graph address stability, and fallback behavior. First measure proposal quality on the frozen
corpus without optimizing kernels: per-position acceptance, accepted tokens/round, proposal cost,
verify cost, and net throughput. Compare against current MTP and prompt-lookup fallback under the
same workload. Require correctness tests for greedy equivalence and an independent numerical oracle
for any changed FP operator. A candidate advances only if the measured net throughput gain is
repeatable and exceeds a predeclared practical threshold (suggested: >3% with confidence interval
excluding zero); otherwise document the result and stop before kernel work.

### Verification artifacts and closeout

For every gate retain: (1) raw outputs, not only a summary; (2) a small table of median and spread;
(3) exact software/hardware/artifact identity; (4) correctness-test and oracle results for code
changes; and (5) a written decision: proceed, defer, or retire, with the evidence. Changes to
runtime, ops, or artifact handling require corresponding CTests per repository policy. Do not run
build or CTest commands until explicitly requested; when implementation is authorized, run the
mandated full CTest suite before calling the work complete.

---

## Pre-flight details

### Serving-script `$ExtraArgs` pass-through

`scripts/run-qwen38-c1-reduced.ps1` places `$ExtraArgs` on a separate line with a blank line
before it, so PowerShell parses it as an independent statement rather than an argument to
`& $Server`. Extra flags never reach `ninfer-serve.exe`.

Fix — continue the same command and splat:

```powershell
    --scr --scr-mode segmented --request-log-jsonl E:\out.jsonl `
    @ExtraArgs
```

### Draft-head artifact objects

```bash
python -m tools.artifact.inspect models\qwen3_8_27b.ninfer --objects | findstr draft
```

Expect both:

| Object | Shape | Format |
|---|---|---|
| `text/draft_head` | `[131072, 5120]` | Q4G64 |
| `text/draft_head_token_ids` | `[131072]` | I32 |

If absent, `--lm-head-draft` silently no-ops (`proposal_head_` stays null and `proposal_argmax`
falls through to the full LM head) — the flag's win would be unclaimed.

### A/B baseline capture

Same fixture, same seed, `--spec mtp --draft-tokens 3`, with and without `--lm-head-draft`.
Capture net tok/s, `acceptance_rate`, `acceptance_length`, `accepted_per_position`.

---

## Free-win details

### `--lm-head-draft` — already applied

The serving script already passes it, and the artifact resolves to
`WeightsProfile::GroupwiseIntW8Endpoints` (qwen3.8-27b), whose endpoints are `W8G32_F16S`.

| | Default (full head) | `--lm-head-draft` |
|---|---|---|
| Matrix | `text/output_head`, 248320×5120, W8G32 | `text/draft_head`, 131072×5120, Q4G64 |
| Read per proposal | ~1.18 GiB | ~0.33 GiB |

Verification always uses the full head, so generated text is unchanged (greedy: bit-identical).
The flag trades slightly lower acceptance for markedly faster rounds — measure the product, not
one factor.

### MTP hidden copy-back — not worth implementing (measured)

`mtp_impl.h:183` copies the hidden state back after every draft step:

```cpp
CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                           cudaMemcpyDeviceToDevice, state.execution.device.stream));
```

A ping-pong buffer pair would delete that copy. `bench/ops/mtp_hidden_copy_bench.cu` isolates the
removable graph-node cost by capturing two chains at the production shape (`hidden=5120`, BF16,
batch=1, so 10 KiB per step) and comparing graph topology and replay time.

**The payload is not the problem.** 10,240 B at 86% of 1 TB/s is ~10 ns. What a copy node costs is
fixed launch overhead. Measured on the RTX 4090 (`--repeats 60000 --samples 20`, clock ramped to
2,670 MHz first):

| K | steps | copy-back µs | ping-pong µs | delta µs | graph nodes (copy/ping) |
|---|---|---|---|---|---|
| 1 | 0 | 0.1356 | 0.1341 | +0.00 | 0 / 0 |
| 3 | 2 | 7.1526 | 6.8688 | +0.28 | 4/2 vs 2/0 |
| 5 | 4 | 11.0968 | 7.0494 | +4.05 | 8/4 vs 4/0 |
| 7 | 6 | 13.9271 | 9.1395 | +4.79 | 12/6 vs 6/0 |
| 15 | 14 | 34.5911 | 12.7392 | +21.85 | 28/14 vs 14/0 |

~1.5 µs per copy node, linear in `K-1` as expected. **The node counts are the reproducible result;
the microseconds are order-of-magnitude only.** The default `--repeats 500` is too short to resolve
a 0.13 µs/launch workload — short runs reported ping-pong *slower* than copy-back at K=3 and K=5,
which is physically impossible — and the WSL/`dxgkrnl` submission path plus the P8→P2 clock ramp add
jitter of the same magnitude. Repeat runs at K=15 gave 21.9 / 42.7 / 44.3 µs.

**Verdict: do not implement ping-pong.** Against the 31.31 ms K=5 round measured at 44.5K context
(§ fourth run above), 4.05 µs is **0.013%**. Even the pessimistic K=15 delta is ~0.04% of a ~52 ms
round. The measured marginal cost of a draft column is 1.50–1.94 ms, four orders of magnitude
larger. A measurable delta here would only have justified an end-to-end A/B; there is not one.

There is no ping-pong implementation in `src/` to A/B — `grep -rn ping_pong src/` is empty. The bench
compares two chains captured for the purpose, so its numbers are a proxy, not a speedup.

**Confirmed end-to-end, and closed.** `ninfer_qwen3_6_27b_mtp_round_bench` on `qwen3_8_27b.ninfer`
at `--draft-tokens 3` reports `mtp_round_mean_ms` 24.2044 with `mtp_round_min_ms` 24.0845 and
`mtp_round_max_ms` 24.2762. The within-run spread is **192 µs**, so the 4.05 µs copy node is
**0.017% of the round** — nearly three orders of magnitude inside the instrument's own noise. An
effect this far below the noise floor is not worth an implementation, and this is the measurement
that says so rather than the arithmetic alone. Two caveats if the figure is ever quoted: that run
logged `fallback_steps 4` of 32 rounds, so its mean blends MTP rounds with native-decode rounds and
is **not** comparable to the 24.08 ms K=3 reasoning row above (which licenses 3.08 tokens/round
against this run's 1.87); and `mtp_round_mean_ms` averages both regimes together. Note also the
instrument cannot resolve what it was built to find — that is the finding, not a shortcoming of
the run.

### GDN decode grid sizing — retired by measurement

Production verify grid (`src/ops/linear_attention/gated_delta_net/recurrent.cu`, record and
snapshot launchers): `dim3(v_heads, batch, kStateDim / kBlockDv)` = `dim3(48, 1, 8)` with
`kBlockDv = kNumWarps * kDvPerWarp = 4 * 4 = 16` and block = `kWarpSize * kNumWarps` = 128 threads.

**384 blocks / 128 SMs = exactly 3.0 full waves.** Warps per SM = 12, i.e. 25% of the 48-warp
capacity. The grid is already wave-aligned, so there is no trailing straggler wave and no
occupancy problem to fix. An earlier revision of this document claimed 96 blocks; that was an
arithmetic error (it used `kBlockDv = 32`) and is corrected here. The original 384-block figure
in the plan was correct.

Alternatives, for the record (48 value heads, 128 SMs):

| kDvPerWarp | kNumWarps | blocks | waves | warps/SM | capacity |
|---|---|---|---|---|---|
| 4 | 4 (production) | 384 | 3.00 | 12 | 25% |
| 2 | 4 | 768 | 6.00 | 24 | 50% |
| 1 | 4 | 1536 | 12.00 | 48 | 100% |
| 8 | 4 | 192 | 1.50 | 6 | 12% |

Deeper splitting is still available in principle, but it buys warps, not waves, and this kernel is
latency-bound on the serial recurrence rather than on SM occupancy.

#### Prototype result (measured, RTX 4090, batch=1)

A sandbox prototype (`bench/ops/gdn_verify_grid_prototype.cu`, local-only, not committed) measured
the production `gated_delta_net_replay_record` Op against four grid/block variants over the full
public verify envelope T=2..16, with every candidate required to reproduce the production output
**bit-exactly** before its timing was considered. Warm and cold-L2 (256 MiB eviction) timings were
both taken; `rel_min` below is `candidate_cold_min / production_cold_min`.

| Strategy | blocks × threads | waves | rel_min (T=16) | rel_min range over T=2..16 | Verdict |
|---|---|---|---|---|---|
| production dv4/w4 | 384 × 128 | 3.00 | 0.941x | reference | — |
| dv2/w4 | 768 × 128 | 6.00 | 1.101x | 0.86x – 1.37x | no win |
| dv1/w4 | 1536 × 128 | 12.00 | 1.471x | 1.10x – 1.47x | clearly worse |
| dv2/w8 | 384 × 256 | 3.00 | 1.094x | 0.86x – 1.18x | no win |
| dv8/w4 | 192 × 128 | 1.50 | — | — | **REJECTED, not bit-exact** |

**Result: no candidate beats production.** At T=16 — the most stable point, where the kernel is
long enough (~17 µs cold) that host-scheduling noise is small — every bit-exact candidate is 6% to
47% slower. The T=16 numbers were reproducible across repeated runs. Short widths (T=2..5) are
too noisy to distinguish anything, and show no consistent win either.

Interpretation: adding warps per SM does not improve latency hiding here, and halving the rows
per warp doubles the warp-shuffle reduction traffic without adding useful parallelism. The
`dv8/w4` variant is not merely slower, it is **wrong**: `kDvPerWarp = 8` exceeds the 4-element
value pack, so it silently reads out of range. That failure is exactly why the bit-exactness gate
belongs in the harness.

**Conclusion: item 1.3/1.4 is retired with a measurement, not an assumption.** The production grid
is already optimal among the strategies tested. No production code change is proposed or made.


---

## Decision-gate detail

### Cost curve
GDN kernel time vs `--draft-tokens {1, 3, 5, 15}`. Use `bench/ops/gated_delta_net_bench.cu
--snapshot --sweep` for the kernel, plus end-to-end runs at each K to catch layer-sequential
compounding.

### Benefit curve — read the *shape*, not just the mean
Accepted-tokens-per-round vs K. Counters already exist and are emitted:
`speculative_stats.accepted_per_position` (`src/targets/qwen3_6/impl/runtime/program_impl.h:3159,
3321`), surfaced by both the serve request log and
`bench/targets/qwen3_6_27b/ninfer_bench_support.cpp:218-237`.

**The `accepted_per_position` histogram is the decisive signal**, more than the mean rate:

- **High at positions 0-1, collapses at 2+** → chain-limited. K=3 is already near-optimal and the
  scan buys little. Wider speculation cannot help.
- **Slow decay across positions** → the scan is the gate. Wider speculation pays and serial GDN is
  what caps it.

Run it on the *actual* reasoning workload, not a generic fixture — the whole point is that
reasoning acceptance is workload-dependent and counter-intuitive (see the acceptance-regime
section above).

### ROI
The scan is worth building only if acceptance *p* is high enough that wider speculation would
actually be accepted.

- Acceptance high (>80%) **and** GDN cost already crushing K=3 → **build the scan**.
- GDN cost tolerable at K=3 **and** acceptance low → temper the goal; bank free wins and use
  batching for aggregate throughput.

---

## Measured cost curve: the roadmap's serial-K model was wrong in kind

The roadmap's load-bearing claim was that GDN verify is serial in K with a per-transition cost of
~6 MiB, giving a bandwidth floor of ~1.1 ms at K=3 and ~4.6 ms at K=15, and a latency-bound round
cost of 2–3 ms and 10–14 ms respectively. It used that estimate to retire the scan.

A sandbox prototype (`bench/ops/gdn_verify_cost_curve.cu`, local-only, not committed) measured the
production `gated_delta_net_replay_record` Op at the 27B geometry (Hq=16, Hv=48, batch=1) over the
full public verify envelope K=2..16, cold-L2, minimum of 500 runs. Results were bit-reproducible
across repeated runs.

| K | kernel µs (1 layer) | ns/column | 48-layer sum µs |
|---|---|---|---|
| 2 | 10.24 | 5120 | 491 |
| 3 | 11.26 | 3755 | 541 |
| 4 | 12.29 | 3072 | 590 |
| 8 | 16.38 | 2048 | 786 |
| 16 | 22.53 | 1408 | 1081 |

**The marginal cost of one extra draft column is ~1.0 µs per layer, i.e. ~49 µs per round across 48
layers.** The roadmap's model implied ~15–19 µs per layer per column (6 MiB at 1.008 TB/s), which is
15–19x the measured marginal cost.

### Why the model was wrong

The roadmap assumed the recurrent state (3 MiB read + 3 MiB write per layer) is touched once per
*drafted column*. It is not. `recurrent_bf16_body` loads the state tile once, before the
`for (token = 0; token < valid; ++token)` loop, and holds it in registers across all K columns,
writing it back once at the end. So the state traffic is **fixed per layer, independent of K**, and
K only adds the small per-column q/k/v/gate reads.

| Model | Cost per round (48 layers) |
|---|---|
| roadmap (state re-read per column) | K=3: 2–3 ms, K=15: 10–14 ms |
| measured | K=3: 0.54 ms, K=15: 1.08 ms |

### What this changes

1. **The serial-K penalty used to retire the scan does not exist in the assumed form.** The scan's
   value proposition was removing a 2–14 ms serial cost. The real GDN verify cost is 0.5–1.1 ms per
   round. This does not automatically reopen the scan — the chain-saturation argument at q≈0.50
   still stands on its own — but the *cost* justification for retiring it was wrong, and any future
   decision must use measured costs.
2. **Wider K is cheaper than believed.** Going K=3 → K=15 costs ~0.54 ms of extra GDN verify per
   round, not ~8 ms. Whether that is worth it still depends entirely on acceptance by position
   (Gate B), which remains unmeasured.
3. **Do not treat this as a throughput win.** It only corrects a cost model. The acceptance
   measurement in Gate B is still the gating unknown, and the base decode is still at the memory
   wall.

---

## Retired items, kept for the record

### Scan-based GDN verify
**Retired on two independent grounds, both measured.**

The roadmap retired this on the premise that at q≈0.50 the greedy chain saturates at 2.0
tokens/round, so removing the serial-K penalty buys at most +6%. Both halves of that argument fail
under measurement:

1. **The acceptance premise is false.** Reasoning per-position q measures 0.74–0.86 and reaches
   **3.89 tokens/round at K=7**, not 2.0. The chain does not saturate.
2. **The cost premise is false in kind.** The serial penalty was modelled as ~6 MiB of state traffic
   *per drafted column*. The state is actually loaded once per layer and held in registers across
   all columns, so the marginal serial cost is ~1 µs per column per layer (~49 µs per round). A scan
   would remove almost nothing.

So the scan is retired, but for the opposite reason from the original: not "the penalty is
irrelevant because the chain saturates", but "the penalty barely exists, so there is nothing to
remove". Widening K captures the available gain directly, without a kernel project.

The associativity analysis remains correct and is kept for the record: the delta-net recurrence
`S_t = S_{t-1}(α_t I − β_t k_t k_t^T) + β_t v_t k_t^T` is associatively scannable, and the low-rank
composition of those rank-1-modified-identity transforms is the **WY representation already
present in prefill** (`src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh`, driven
by `launch_chunked`, `src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp:246`). The WY
chunk is `kChunkSize = 64` (`common.h:8`), so at K ≤ 15 a scan would be a single-chunk WY build.
A new short-block scan would need per-token outputs, replay/state semantics, and numerical-oracle
testing.

The AR re-propose batching (`mtp_impl.h:161-184`, `width = 1` per AR step through the 0.336 GiB
MTP layer) is retired with it — it existed to serve wider speculation.

### Tree attention
Shelved until the scan lands. The serial recurrence makes a wider tree strictly worse: GDN verify
cost is linear in K, so the "free compute budget" argument for wide trees collapses.

Supporting detail: the verify mask is a linear chain (`speculative_prepare_verify_inputs_kernel`,
`src/ops/kernel/speculative_round.cuh:45`) and the attention kernel has no mask tensor / parent
indices / DAG input — the `Masked` template parameter only handles the ragged column tail. The
existing `bidirectional_gqa_attention` op is full block-bidirectional, which cannot express a
multi-branch dependency.

### GDN state dtype (BF16/FP16)
**0.9% of per-token traffic, below measurement noise.** GDN state is 144 MiB per slot across 48
layers → read+write = 288 MiB = 0.28 GiB/token. FP16 saves 0.14 GiB of 16.15 GiB total = 0.9%.
Not worth the recurrent-accumulator drift risk (BF16/FP16 state can compound error over long
contexts, and would need FP32/FP64 oracle validation over long contexts, not single-step diff).
The fork's `--gdn-state-fp16` flag exists, but the arithmetic does not justify porting it.

### L2 `cudaAccessPolicyWindow` for the MTP head
Marginal — already applied to the proposal head and the recurrent state pool
(`text_context.h:170-171`, `text_context_impl.h:272-275`); 0.336 GiB does not fit 72 MB anyway.

### KV-append / MLP graph overlap
Independent nodes already captured in the graph, but low single-digit t/s.

### Early exit in the verify kernel
Rejected. Compute is not the bottleneck — the weight read is, and it happens either way. A
device-side branch would break graph capture and address stability; a mid-round host sync would
forfeit the graph. Would save milliseconds out of ~21 ms.

---

## Research direction: raise per-position q

At q ≈ 0.50 the only lever that changes the regime is the **acceptance rate itself**, not the chain
width. Candidates, in rough order of promise for a hybrid GDN + full-attention model:

1. **Self-speculative decoding** — draft from an early-exit layer of the *same* model rather than
   a separate MTP head. No extra weights, and the draft distribution is closer to the target, so
   q rises. Our 48 GDN + 16 full-attention layer split means early exit lands in a GDN layer,
   whose serial recurrence makes multi-layer exit awkward — measure before committing.
2. **Jacobi / iterative decoding** — refine a block of K guesses in place over a few iterations.
   Attracts because it converts serial dependency into parallelism, but the GDN recurrence still
   has to be evaluated over the block, so it does not escape the serial-GDN cost by itself.
3. **Lookahead decoding** — n-gram guessing from the generated suffix, verified in one pass. Zero
   extra weights and no MTP head at all. Weakened by our existing `find_prompt_lookup_draft`
   fallback (`program_impl.h:1410-1414`), which already exploits exactly this signal — so the
   incremental gain over the current fallback is the open question.

All three still pay the serial GDN verify cost, so none of them escape the memory wall. The win
comes entirely from a higher q, and each needs its own acceptance measurement on the real reasoning
workload before any kernel work starts.

---

## Diagnostic hypotheses (subordinate to the plan)

The following are profiling leads, not active optimization commitments. The reported baseline
analysis above suggests near-memory-wall decode efficiency; therefore do not presume that these
areas contain a material engine-side deficit. Reopen a lead only when a matched profile isolates
meaningful cost and the verification gates above can be met.

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **Quantized weight reads and small-T linear kernels** | Each target-model decode/verification pass executes projections and MLPs across 64 layers. At small token widths, weight traffic, weight decoding, and the selected Q4/Q5 routes may limit throughput. | [Profile and tune the small-T linear path](#follow-up-1-profile-small-t-linear-and-weight-cost) |
| **Long-context GQA/KV reads** | The 16 full-attention layers read historical K/V, compute attention scores and softmax, then reduce values. At 44K, cache traffic and attention work may be substantial; the split policy can also affect utilization. | [Measure GQA cost across context lengths](#follow-up-2-measure-gqa-and-kv-cost) |
| **GDN recurrent verification and state replay/fold** | The 48 GDN layers process the verified token positions and preserve speculative state for the accepted prefix. Recurrent transitions across positions are sequential; record and fold add work beyond ordinary width-one decode. | [Attribute GDN record, recurrence, and fold cost](#follow-up-3-measure-gdn-verification-and-fold) |
| **Serial MTP draft generation or a mismatched draft width** | Each draft depends on the preceding draft's hidden state, so proposal steps are sequential. A larger K costs more drafting and verification; it only helps if the extra committed tokens offset that work. | [Sweep K and assess adaptive width](#follow-up-4-tune-mtp-draft-width-then-test-adaptive-width) |
| **Target output head and token-selection work** | Verification needs target logits, while the optimized draft head only reduces proposal-head work. Output-head projections and sampling may still contribute, particularly when several positions are verified. | [Measure target logits and proposal-head cost](#follow-up-5-measure-output-heads-and-sampling) |
| **V unpack/dequantization and BF16 P·V staging** | The current kernel expands each packed V tile to BF16 for P·V. This work grows with context and could matter, but global KV reads or other attention work may dominate instead. | [Profile P·V staging before prototyping FP8](#follow-up-6-profile-pv-staging) |
| **Round orchestration, synchronization, or WDDM residency** | Graph replay reduces launch overhead but does not remove copies, bookkeeping, or all synchronization. Memory pressure under Windows/WDDM can also cause stalls or variability. | [Check GPU gaps, round overhead, and memory headroom](#follow-up-7-check-orchestration-and-residency) |

## Follow-up options for later sessions

These are investigation plans, not claims that a change will be faster. Preserve the same prompt,
sampling, artifact, context, and output conditions when comparing end-to-end committed tokens/sec.

### Follow-up 1: Profile small-T linear and weight cost

Profile the target verification and ordinary decode shapes. Compare operator timings with the
dispatches selected in [`q4_dispatch.cpp`](../src/ops/linear/q4/q4_dispatch.cpp) and
[`q5_dispatch.cpp`](../src/ops/linear/q5/q5_dispatch.cpp). Use the public linear benchmark for
exact shapes only as a diagnostic; accept a dispatch or kernel change only if end-to-end `tg`
improves.

### Follow-up 2: Measure GQA and KV cost

Compare fixed context lengths such as 8K, 16K, and 44K, with MTP on and off. Attribute time to the
GQA kernels and inspect the split policy in
[`gqa_attention_decode.cu`](../src/ops/launcher/gqa_attention_decode.cu). If profiling supports a
kernel change, validate it against the independent attention oracle and the same generation
workload.

### Follow-up 3: Measure GDN verification and fold

This is diagnostic only unless Gate B shows serial GDN verify is binding. Separate
projection/record, recurrent verification, and accepted-prefix fold costs; the GDN snapshot
benchmark is not an exact proxy for replay-record.

### Follow-up 4: Tune MTP draft width, then test adaptive width

Use K=1, 3, and 5 first; compare committed tok/s, round time, and accepted drafts by position. The
public runtime supports up to K=15, but the retained Qwen3.6-27B target round benchmark currently
covers only K=1..5. Its GDN snapshot sweep is not an end-to-end K sweep or an exact replay-record
timing.

Adaptive width is a research lead, not an active commitment; measure position-wise acceptance and
fixed-K cost first as required by Gate B. **Crit** is the reference inference project
([inference index](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/inference/README.md));
the relevant engine example is specifically its Qwen3.8 implementation in
[crit-hub](https://github.com/crit-labs/crit-hub), not the generic
[`inference/mtp-decode` toolkit](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/inference/mtp-decode/README.md).
In the [Qwen3.8 NVFP4 engine implementation](https://github.com/crit-labs/crit-hub/blob/0b72fd0ee3fc03ca1c0fb853d0dd1021c1617117/engines/rtx5090/qwen3.8-27b%40NVFP4/mtp7-fp8attn/engine/src/engine.cpp),
Crit-hub uses recent acceptance and estimated round cost to choose draft width rather than always
doing a fixed number of draft steps. The generic Crit MTP entry also includes an
`AdaptiveDraftBudget` that explores candidate widths and minimizes an EWMA of round time per
published token; this is a useful policy pattern, but its proposer is token-history copying, not a
learned MTP head. An aggregate acceptance figure (including the reported ~50%) is not enough to
select a policy; first measure acceptance by position and timings for each fixed K. Crit-hub's
RTX 5090/NVFP4 results are policy inspiration, not evidence of a gain on this RTX
4090/groupwise-int workload.

Implementation needs to account for graph shapes: the current decode graph and proposal loop are
organized around configured K. A practical first design could select among a small set of captured
K-specific graph profiles; for concurrent requests, per-batch width choices or bucketing also need
consideration. Adaptive width is worth implementing only if fixed-K results and an online policy
simulation predict a measurable increase in committed tok/s after its selection/graph overhead.

### Follow-up 5: Measure output heads and sampling

Measure target-logit production separately from optimized draft proposal. The user already has
`--lm-head-draft` enabled, so this is not an untested optimization in the reported baseline. The
canonical Qwen3.8 converter inventory contains `text/draft_head` and `text/draft_head_token_ids`;
the loader requires these artifact objects when the optimized head is used. A
full-versus-optimized comparison is useful only if quantifying its contribution is desired; do not
assume a percentage gain.

### Follow-up 6: Profile P·V staging

The current `rk4v4-e8` decode kernel does **not** unpack the whole cache to FP16/BF16. Packed 4-bit
codes remain in global memory. The kernel unpacks K/V tiles into INT8 staging; QK uses INT8 Tensor
Core MMA with group scales, while V is dequantized into BF16 staging for the existing BF16 P·V MMA.
See [`gqa_attention_decode_i8.cuh`](../src/ops/kernel/gqa_attention_decode_i8.cuh).

The BF16 P·V staging is an unverified bottleneck hypothesis. Crit's
[FP8 KV attention entry](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/inference/fp8-kv-attention/README.md)
is a concrete reference: its **decode** kernel reads an E4M3 KV cache, dequantizes tiles, and
reuses the BF16 Tensor Core attention path. The entry's optional direct FP8 Tensor Core attention
path is specifically for **prefill**, not decode. It demonstrates a scale-aware P·V method
(rescaling probabilities and the accumulator across tiles), but not a ready-made FP8 Tensor Core
decode win. It is not a drop-in for NInfer: the entry notes that paged/shared caches need adapted
loads, while NInfer uses paged KV and rotated/group-quantized codes. Its E4M3 cache, per-head-vector
scales, and precision are also different from NInfer's `rk4v4-e8` representation. An on-chip FP8
tile path might reduce conversion or staging cost, but would not reduce global cache reads while
the stored cache remains 4-bit. Profile first; if P·V is material, adapt a narrow kernel and test
numerical fidelity, greedy exactness, and generated-token quality.

### Follow-up 7: Check orchestration and residency

Use a system timeline to look for GPU idle gaps, synchronization, or excessive per-round work;
check available VRAM after model load and during long-context decode. CUDA Graphs are already
supported, so graph support itself is not a missing feature. Investigate graph replay, round
orchestration, or WDDM budget behavior only if timings or memory telemetry point there.

---

## FP8 KV and mixed-precision context

FP8 could change attention accuracy and MTP acceptance, but higher acceptance is not guaranteed.
NInfer currently exposes BF16, INT8-G64, and packed rotated integer/E8 cache formats; there is no
FP8 KV dtype or FP8 append/prefill/decode path. Implementing FP8 KV would require cache
layout/scales, quantization, attention kernels, dispatch/workspace changes, MTP support, and
numerical-oracle tests.

FP8 also does not reduce KV storage versus INT8: both use one byte per element before scales. At
262,144 tokens, the main cache for 16 full-attention layers, 4 KV heads, head dimension 256, and
both K and V is **8 GiB raw** at one byte per element:

```text
2 × 16 × 4 × 256 × 262,144 bytes = 8 GiB
```

That excludes scales, page metadata, the separate MTP cache, weights, recurrent state, and
workspaces. The artifact is about 18.21 GB on disk (see its
[manifest](../model-cards/Qwen3.8-27B-NInfer/artifact-manifest.json)); actual GPU-resident model
bytes depend on load options. If INT8 does not fit at full context alongside runtime allocations,
ordinary full K+V FP8 is unlikely to fit either. K8V4 would use about 6 GiB raw for the main cache
at 262K, versus about 4 GiB for K4V4 and 8 GiB for K8V8, before overhead.

Ada Tensor Cores can accelerate suitable FP8 matrix operations, but FP8 storage does not
automatically use them. An FP8 attention path still needs operand scaling/layout, softmax, and
masks; decode has few query rows and may be limited by reading the long KV history rather than
matrix-multiply throughput. A mixed K8V4 path is a kernel-and-quality experiment, not an assumed
speedup or a known full-context-safe option.

---

## Using the Crit kernel guidance

Crit's
[Kernels README](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/kernels/README.md)
recommends profiling the actual engine before writing or importing a kernel, checking
formats/shapes/ABI, and measuring decode and speculative verification separately. It also stresses
column-count exactness: a speculative verify kernel must preserve the one-token greedy result for
each column, not merely match an aggregate tolerance. The
[Crit Eval kernel profile](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/docs/eval/README.md#kernel-profile)
uses CUPTI to report phase time, GPU idle time, kernel calls and widths; its documented runner
requires Linux/WSL and CUDA CUPTI, so availability in this Windows-target workflow must be checked
before relying on it. Crit's upstream kernels and results are investigation leads, not evidence of
speed or compatibility on NInfer's RTX 4090 path.

---

## The 120 t/s claim — how much is size, how much is engine

Community reports of ~120-190 t/s decode on a 4090 (`iamwavecut/ninfer-all` issue #96) use a
**different model quantization**. Decode is weight-read-bound, so quantization is a first-order
variable. This section decomposes the reported gap into its size and engine components.

### Verified byte accounting (our artifact)

From `src/targets/qwen3_6_27b/impl/load/bindings.cpp:121-169` (exact shapes and formats) and
`src/artifact/storage_layouts.cpp:36-44` (exact encoding):

- `Q4G64_F16S`: group 64, low 32 B, high 0 B, scale 2 B/group = **34 B / 64 elem**
- `Q5G64_F16S`: group 64, low 32 B, high 8 B, scale 2 B/group = **42 B / 64 elem**
- `W8G32_F16S`: group 32, low 32 B, high 0 B, scale 2 B/group = **34 B / 32 elem**

| Component | Shape | Format | Count | GiB |
|---|---|---|---|---|
| `mlp/gate_up` | 34816x5120 | Q4G64 | x64 | 5.645 |
| `mlp/down` | 5120x17408 | Q5G64 | x64 | 3.486 |
| `gdn/value_z` | 12288x5120 | Q5G64 | x48 | 1.846 |
| `gdn/output` | 5120x6144 | Q5G64 | x48 | 0.923 |
| `gdn/query_key` | 4096x5120 | Q4G64 | x48 | 0.498 |
| `attn/gate_value` | 7168x5120 | Q5G64 | x16 | 0.359 |
| `attn/output` | 5120x6144 | Q5G64 | x16 | 0.308 |
| `attn/query_key` | 7168x5120 | Q4G64 | x16 | 0.291 |
| **Body subtotal** | 24.33B params | | | **13.354** |
| `token_embedding` | 248320x5120 | W8G32 | | 1.258 |
| `output_head` | 248320x5120 | W8G32 | | 1.258 |
| MTP layer (5 tensors) | | W8G32 | | 0.420 |
| draft head | 131072x5120 | Q4G64 | | 0.332 |
| **Grand total** | | | | **16.62** |

Reconciles with the observed 17.7 GiB artifact (the ~1.1 GiB remainder is norms, conv, MTP extras,
and 256-byte pool alignment).

### Their side

Reported total: **12.1 GB = 11.27 GiB** including MTP. Back-solving the body, assuming comparable
endpoints:

| | Theirs (IQ3_S / GSQ-RCO) | Ours (groupwise-int) | Ratio |
|---|---|---|---|
| Total weights | 11.27 GiB | 16.62 GiB | 1.48x |
| Body | ~8.00 GiB | 13.35 GiB | **1.67x** |
| Body effective bpw | ~2.8 | ~4.7 | |
| Weight read / decode token | ~10.5 GiB | ~15.9 GiB | **1.51x** |

### The decomposition

Using their own cold-prefix bench (256 output tokens, 3 runs/depth, **no speculation bonus**, same
model and same quant) and scaling purely by weight-read ratio:

| Depth | Their decode | Size-adjusted for our artifact |
|---|---|---|
| short (145 tok prompt) | 187.5 t/s | ~124 t/s |
| mid (44,006 tok) | 151.9 t/s | ~101 t/s |
| deep (133,007 tok) | 135.8 t/s | ~90 t/s |

**Answer to "how much is size, how much is engine":**

- **Size accounts for ~1.5x** of the gap — if our engine were byte-for-byte as efficient as theirs,
  our artifact would land around 90-124 t/s on their bench, versus their 135-187.
- **The residual is unquantified.** Whether it is real engine depends entirely on what our own
  baseline actually is, which is not yet measured (Gate A/B). Until then the honest statement is:
  size explains ~1.5x, and the remainder is unmeasured rather than attributable.

Important caveat on the 2.8 bpw back-solve: it is **circular**. Their body size was derived by
subtracting assumed endpoints from their reported total, so it cannot independently confirm their
bpw. Treat the 1.67x body ratio as an estimate in the 1.4-1.7x range, not a measurement.

### Secondary factors in those commands

- `--gdn-state-fp16` — halves GDN state traffic. This is exactly the shelved state-dtype item, and
  the fork ships it as an opt-in flag (`src/models/qwen3_5/load.cpp:226`), i.e. a known quality
  trade, not a free win. Both reporters use it. Would need our own long-context oracle validation.
- `--spec dflash2 --draft-tokens 5` — the 484 t/s peak is explicitly attributed to "very high
  speculative acceptance". The second reporter measured acceptance falling 0.607 -> 0.482 going
  K=3 -> K=5 at 120K context, exactly as the serial-GDN analysis predicts.
- The first reporter's `--kv-dtype rk8v4` is *worse* than our `rk4v4-e8` (8-bit K vs 4-bit E8 K).
  The second switched to `rk4v4` and measured ~31% smaller KV at the same speed, +0.10% ppl —
  putting them at roughly parity with us on KV, not ahead.

### Conclusion

The 120 t/s figure is **not** evidence that our decode path is 2x off. Roughly 1.5x of it is
quantization. The remainder is unmeasured and requires our own baseline to attribute. The thread's
reporter states this himself: "these are not controlled speedup claims... the model quantization,
context lengths, speculation settings, and test conditions are not identical."

If the residual turns out to be large after we measure our baseline, the two candidate explanations
to investigate are (a) their `--gdn-state-fp16`, and (b) genuine engine differences in the
attention/GEMM routes — the fork carries a calibrated 4090 device profile
(`src/runtime/engine/device_profiles.json`) that our repo does not have.

---

## Upstream audit findings

The headline negative result: **upstream's GDN verify is the same serial recurrence.** Its
`qwen4_exp` model uses `gated_delta_net_replay_record` for verify
(`src/models/qwen4_exp/execution/gdn.cpp:47-69`), backed by
`src/ops/linear_attention/gated_delta_net/kernels/transition.cuh:228` —
`for (std::int32_t token = 0; token < valid; ++token)`. No scan. So the load-bearing architectural
constraint above is not a local shortfall; it is the shared design across both repositories. There
is no scan to port.

Upstream targets **sm_120 (Blackwell 2.0)**. Ports are not 1:1 — Ada has no native FP8 tensor-core
path of the same shape, different SM/wave geometry (128 vs Blackwell's count), and different
shared-memory and cache characteristics. Any candidate must be re-derived for sm_89 and re-verified
against the FP32/FP64 oracle before it ships. Upstream compiles **only** for `120a` and hard-errors
otherwise (`CMakeLists.txt:6-12`), so there is no sm_89 fallback to recover.

### Not portable to sm_89

| Upstream feature | Why it does not port |
|---|---|
| FP8 E4M3 KV cache + attention | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...e4m3.e4m3.f32.ue8m0` (`src/ops/common/cuda/mma.cuh:68`) — Blackwell-only block-scaled FP8 MMA, no sm_89 equivalent |
| `cvt.rn.bf16x2.e4m3x2` fast dequant | PTX 9.2 on sm_120a only (`src/ops/common/quant/fp8.cuh:26`); sm_89 must use the FP16 bridge |
| MoE expert-weight cache priorities | qwen4-exp MoE; not applicable to our dense 27B targets |

### Worth a look, port-effort permitting

- **`qsa` (query self-attention / sparse GQA indexer)** — `src/ops/softmax_attention/qsa/`. A
  top-k key indexer that selects a sparse key subset before attention. We have no equivalent.
  Could cut attention traffic at long context (our `--max-context 131144`). Needs an independent
  look before judging whether the sparsity is safe under the numerical contract.
- **`ple`** — `src/ops/ple/`, a 4-stream gated conv+residual block (2560 features, 9 history).
  Model-specific to qwen4-exp; not applicable to our targets.

### Conclusion

Upstream does not contain a missed win for our roadmap. It validates the serial-GDN diagnosis
rather than contradicting it, and its FP8 advantage is architecture-bound.

---

## MTP verify attention at 6 query tokens: what the occupancy work bought

The width-6 verify attention kernel (§ MTP verify width) carries 50% more query rows than width 4
but ran 72% slower per call, which looked like room to recover. The gap traced to a shared-memory
allocation that capped the kernel at one resident block per SM. Removing it recovered a real but
modest 6.6%, and the profile then showed the residual gap was never an inefficiency at all.

**The change.** The V tile was staged in shared memory as int8 codes and then read back on the very
next line to be widened to bf16 for the MMA. It was a pure pass-through: written once, read once,
never referenced again. `gqa_kv_dequant_i8x8_from` already documents that its source pointer may be
global or shared, so the cache read and the widening became one step and the intermediate region
disappeared. At 6 query tokens that region was 8 KiB of a 52.8 KiB block; per-block shared memory
fell to 44.7 KiB and two blocks fit the 102.4 KiB budget with 13 KiB spare. The K tile stays int8
because its dequant scale factors out of the QK contraction; V's does not factor out of P·V, which
is why V has to be bf16 and K does not.

The arena size is now produced by one `gqa_attention_decode_i8_arena_bytes<KeyBlock>()` helper used
by both the kernel's region offsets and the launcher's `cudaFuncSetAttribute`, so the two cannot
disagree about how large it is — previously each had its own `4 * Bc * D` literal.

**The result, and what it does not explain.** Registers went 238 → 168 with zero spills, and
`No Eligible` improved from 60.6% to 72.8%. Duration went 160.64 → 150.11 µs, a 6.6% gain. But DRAM
throughput *fell*, 29.0% → 31.9%, against a prediction that it would exceed 45%: occupancy was a
symptom, not the bottleneck. `Warp Cycles Per Issued Instruction` is 10.9 with 0.34 eligible warps
per scheduler, i.e. warps are issuing about once every 11 cycles — fed, not starved. The 72.8%
`No Eligible` is arithmetic, not a resource stall: one block of 6 warps cannot fill four schedulers.

**The residual gap is the work itself.** Per verified token the cost is 23.40 µs at width 4 against
25.02 µs at width 6, about 7%. Padded PV work per query is `Br × D`, with `Br` rounding 24 rows to
32 and 36 rows to 48 — 25% padding either way, and 36/24 = 1.50 while the measured duration ratio
is 1.604. Width 6 is very nearly linear in padded MMA work. Both widths pay the same 25% row
padding, so the 7% is the smaller per-warp column share (`PVNtPerWarp` 8 → 16 as
`ConsumerWarpsPerTile` drops 4 → 2), not a fixable stall.

Closing that 7% would mean a `RowTiles`-specialized geometry to avoid padding 36 rows into 48 — a
structural rewrite of the tiling for one of sixteen layers. Not worth it. **The kernel is done:**
keep the commit for its 6.6%, and look for MTP speed elsewhere.

A correction for the record: `Gqa27Geometry = GqaGeometry<24, 4, 1>` gives `GroupSize = 24/4 = 6`,
not 4. Register and accumulator figures in the earlier width-6 analysis that used 4 are wrong; the
conclusions hold, the arithmetic in them does not.

## Improving MTP draft acceptance

Acceptance is a better lever than any remaining kernel work. At the observed 30% acceptance the
engine does a 6-token verify to bank ~2.5 tokens per round, so most of the verify cost is discarded
work. The rule itself is a plain longest-prefix match — `speculative_accept_greedy_drafts_kernel`
advances while `target_tokens[a] == drafts[a]` and emits the target's own next token as the bonus
— so at temperature 0 acceptance is entirely a property of the draft model agreeing with the
target, not of anything in the acceptance path. Four levers, in rough order of value per unit of
risk:

1. **Set draft width from acceptance, not from a constant.** Acceptance decays with position: the
   first draft is usually right, later ones rarely are. Rounds beyond the expected accepted length
   pay full verify cost for tokens that are almost surely discarded. Earlier work already found
   adaptive width beats a fixed one (Follow-up 4); the measured 30% at K=5 says the fixed point is
   set too wide. Track per-position acceptance and set `--draft-tokens` to the expected accepted
   prefix plus one.

2. **Check the draft head's context.** The draft shares the target's KV but consumes a different
   hidden state (`--lm-head-draft`). If the draft head is fed the pre-norm continuation hidden
   state rather than the same normalization the target's LM head sees, agreement suffers in a way
   that looks exactly like a modelling problem. Cheap to inspect, and it is the kind of mismatch
   that no amount of kernel tuning can compensate for.

3. **Improve the sampling agreement.** At nonzero temperature the acceptance test resamples against
the target distribution with a penalty overlay for already-accepted drafts. Temperature and
`top_p`/`top_k` disagreement between draft and target silently lowers acceptance; at greedy it is
irrelevant, which is why the 30% figure on a greedy request is the number to chase first.

4. **Measure acceptance per position before changing anything.** The round benchmark already logs
   `accepted_draft_tokens`, `drafted_tokens`, and `speculative_rounds`. Per-position histograms turn
   the four levers above from guesses into a decision: if position 0 accepts 95% and position 4
   accepts 5%, the answer is lever 1 alone and the other three are noise.

What not to do: keep raising K. Verify cost is linear in K while accepted tokens are not, so a
wider draft is negative once per-position acceptance falls below the break-even rate — which the
30% aggregate already suggests it has.

## Caveats

- The reported 95/42 tok/s and acceptance figures are user-provided, not independently verified
  here.
- Findings are source-based hypotheses and reported measurements; no live profile or new benchmark
  was run for this document.
- RTX 5090 performance data is context, not a substitute for the same workload on the RTX 4090.
- Prefill optimization is out of scope except for the INT8 group-64 activation item, which is a
  TTFT-axis trade and must be judged separately from decode.
