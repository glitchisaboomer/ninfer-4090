# Suffix Cache Reuse (SCR) — Feature Spec

**Status:** Implemented/verified at the op level (see [Verification Status](#5-verification-status)).

SCR reuses a previous turn's KV cache for surviving tokens after an edit, avoiding a full re-prefill. It's an approximation (computed under the pre-edit context), so it's opt-in and off by default.
We implement SCR to make prefill more efficient for context language models (CLMs).

---

## 1. Concepts

| Term | Meaning |
| --- | --- |
| **Relocation** | Making a surviving token range readable at new prompt positions after an edit shifts later tokens. |
| **Delta plane** | Per-logical-page RoPE phase shift (`int32` positions). A page with delta $d$ holds keys rotated at position $d$ away from where they're read. |
| **Packed range plane** | Per-logical-page relocated interval $[begin, end)$, packed as 6-bit begin + 7-bit end into a `uint32` (`0xffffffff` = no interval). |
| **Whole-page delta** | Relocated interval covers an entire page (one query rotation describes all keys). |
| **Partial page (off-grid)** | Strict subset of a page (partially relocated); needs boundary-aware logic. |
| **Re-pointing** | Relocating by updating the block table to point to a different physical page. No payload copying. |
| **Copied relocation** | Relocating by copying payload into new pages (used when destination pages are occupied or source is off-grid). |
| **ScrRelocationMode** | `Tail` (single contiguous run; runtime active) or `Segmented` (disjoint runs; planner only). |

---

## 2. Public Surface

* **`include/ninfer/types.h`**:
* `ScrRelocationMode { Tail, Segmented }`
* `ExecutionOptions::enable_suffix_cache_reuse` (opt-in, default `false`)
* `ExecutionOptions::suffix_reuse_max_spans` (default `6`)
* `ExecutionOptions::scr_mode` (default `Tail`)


* **`src/targets/qwen3_6/impl/runtime/prompt_edit.h`** (Planner, host-only):
* `plan_tail_relocation(...)`, `plan_shifted_relocations(...)`, `plan_copied_page_remap(...)`, pool demand pricing helpers.


* **`src/core/paged_kv_delta_layout.h`** (Arithmetic, host/device):
* `pack_rope_delta_page_range`, `plan_rope_delta_page_ranges`, `encode_rope_delta_page_ranges`.


* **`src/core/paged_kv_cache.h`** (Storage):
* `PagedKVPool::rope_deltas()`, `rope_delta_ranges()`, `PagedKVAllocation::copy_pages_from_host(...)`.


* **`src/ops/`** (`sm_89` kernels):
* `gqa_attention_decode_bf16.cuh`, `gqa_attention_prefill_bf16.cuh`, and INT8 variants (`_i8.cuh`).

**Reporting.** A relocated run is served from resident KV, so it is a cache hit on the wire even
though it is not a prefix hit: `usage.prompt_tokens_details.cached_tokens` (Chat Completions, in the
non-streaming body and the streaming usage chunk) and `usage.input_tokens_details.cached_tokens`
(Responses) report the ordinary prefix hit **plus** the relocated run, and the result never exceeds
`prompt_tokens`. The two mechanisms stay separable for measurement: the request JSONL log carries
`prefix_cache_hit_tokens`, `prefix_relocated_tokens`, and `prefix_reuse_path`; `/metrics` exposes
`ninfer:prefix_cache_hit_tokens_total` and `ninfer:prefix_relocated_tokens_total`; and the `timings`
block reports the same combined count in `cache_n` with `prompt_n` net of it.



---

## 3. Relocation & Invariants

Keys aren't re-written with new rotations; they keep their original source rotation, and the delta plane records the offset so the read path can counter-rotate the query.

* Planner publishes `delta = dst - src` and interval $[begin, end)$ per destination page.
* **Partial page:** Interval is a strict subset. `is_partial` drives the boundary-aware score path.
* Planes publish/clear together (one relocation or none per page).
* Spans never share a logical destination page. Understating the delta plane silently acts like no relocation happened.

**Planner gate.** The gate builds one diff per request and feeds it to whichever selector the mode names, so the detector bounds every plan. `diff_prompt_edit` matches whole 64-token blocks and sees only content on the same page grid in both prompts, so an off-grid edit leaves it nothing but the leading prefix. With `NINFER_SCR_SOURCE_SHIFT` set the gate uses `diff_prompt_edit_shifted` (all-offset) instead, and either mode falls back to `plan_shifted_relocations` -- whose destination may sit mid-page -- when its own selector expresses nothing, rather than dropping to a cold prefill; unset, the gate is byte-identical to the aligned code. Both selectors then pass through one selection step that keeps the longest survivors up to `--scr-max-spans` and enforces the packed plane's one-interval-per-destination-page rule, so a plan the runtime refuses is a tripwire rather than a reachable outcome. `NINFER_SCR_DIAG` adds one `[scr-diag] plan …` line per request covering both modes -- what the detector fed the selector, what it kept, what the copied path recovered, and every span it dropped with a stable reason -- and a refused plan prints the rule that refused it along with its spans and ranges, so neither mode can decline silently. The shifted detector anchors at the page size, so survivals shorter than a page stay invisible to it.

**Linear-attention layers.** A linear-attention layer keeps one fixed-size recurrent state for the whole sequence rather than per-token entries, so there is nothing to relocate and no RoPE to undo: the relocated run is simply not forwarded, and the ranges the prefill covers advance the state instead. The state is **kept, not rebuilt**. The authors' design "snapshots the recurrent state before the edit and continues from that snapshot", which their reference implements as `KVREUSE_SSM_MODE=fork` (its default): restore the state saved at the end of the previous prompt. The resident lane state is exactly that snapshot. Zeroing it would instead drop the reused prefix's contribution from the 48 of 64 linear-attention layers in qwen3.8-27b — an approximation far wider than the one SCR accepts and bounds with $K$. (An early reset here targeted degenerate output that a later fix attributed to the prefill's base position, not this state.)

**Second-token row at the reuse frontier.** The layer that predicts a second token reads a row at position $p$ paired with token $p+1$, so the row at `base - 1` — the last reused position — has to be written from the hidden at `base - 1` and the embedding of the first new token. No cache holds that hidden: a lane keeps one, at the previous sequence's last position, which is `base - 1` only when the prompt reaches the frontier. The runtime therefore forwards that single token over the reused prefix and writes the row from the result. The forward is a real prefill of that position: it writes Text KV there and advances the recurrent summary. Its Text KV write is saved and restored around it, so the reused prefix is left exactly as the relocation found it. The recurrent summary stays advanced, because it is already the previous prompt's end state (the fork above) and one more token sits inside that approximation; restoring it would cost a second full state image per lane. A moved run's own last row is closed the other way, by handing each run's last block back to the prefill, which costs one block of reuse per run. Neither costs a wrong accepted token: drafts are verified against the base model.

---

## 4. Relocation Routes

1. **Re-pointing (Default):** Updates block table to point to an existing physical page. Zero copy cost, requires free destination pages.
2. **Copied Relocation:** Copies payload into acquired pool pages. Handles occupied destinations, off-grid sources, and a mapping that is not the same length as the resident one — a deleted region shortens the prompt, so a surviving run's source pages lie above the new mapping's end; they pay for a destination unless a copy reads them, in which case they are released after the copy has run.

---

## 5. Verification Status

Verified on RTX 4090 (`sm_89`, CUDA 13.3, GCC 11, Linux):

* `ninfer_gqa_attention_test`, `ninfer_gqa_kv_page_delta_test` (BF16), `_i8_test`, `_rotated_test`, `ninfer_scr_relocation_integration_test` all pass.
* CTest: 100/100 pass (8 skipped: the weight-dependent tests, no model weights here).
* **Accuracy:** Partial-page relative L2 vs direct prefill is $\sim 1.86\times 10^{-3}$ to $3.42\times 10^{-3}$ (well under the $2.0\times 10^{-2}$ limit).
* **Cost:** Partial-page decode is $1.007\times$ the baseline (essentially noise).

**Not covered:** Windows/MSVC builds; prefill timing for a page a kept run only partly fills;
contexts well beyond 20k.

### What a mid-prompt edit costs, measured at `228e10b5` (2026-10-18)

`ninfer_qwen3_6_27b_multi_span_edit_test` was run on the host (RTX 4090) with five of the six KV
formats (`rk2v4-e8` was not run), on the route that copies a kept run into new pages because it no
longer starts on a 64-token block boundary. Each row reports two numbers: how many of the eight
generated tokens match a run that reused nothing, and how far the first token's log-probability
moved against that run, in nats.

| KV format | one deep cut | three small cuts | three deep cuts |
| --- | --- | --- | --- |
| `bf16` | 8/8 · 0.061 | 8/8 · 0.006 | 8/8 · 0.004 |
| `int8` | 8/8 · 0.054 | 8/8 · 0.005 | 8/8 · 0.002 |
| `rk8v4` | 8/8 · 0.040 | 8/8 · 0.002 | 8/8 · 0.011 |
| `rk4v4` | 8/8 · 0.058 | 8/8 · 0.005 | 8/8 · 0.005 |
| `rk4v4-e8` | 8/8 · 0.047 | 8/8 · 0.003 | 8/8 · 0.007 |

* Every run kept its reused tokens rather than quietly recomputing the whole prompt, and every run
  matched the cold result on all eight tokens.
* Ten repeats of the one-deep-cut case on `rk4v4-e8` landed between 0.021 and 0.046 nats, so the
  spread between runs is about 0.03 nats and the token counts never moved. Any regression budget
  has to sit above that spread; 0.12 nats clears it and still catches the 0.2–1.1 nats failures
  that were recorded before the fix below.
* The run the edit kept started 27 tokens into a 64-token block, so its destination held a
  partly-used block at each end with 195 full blocks between them. That is the hardest shape for
  this feature, and it is the one the quantized formats were previously reported to get wrong.
  Of a 20,109-token prompt the reuse covered 12,544 tokens, with 27 tokens before it and the last
  114 recomputed.
* The quantized formats were not worse than `bf16`. `bf16` had the largest first-token shift of the
  five (0.061, against 0.047 for `rk4v4-e8`), so there is no gap left between `bf16` and the
  quantized formats for a kernel to explain.
* The second-token layer (MTP) drafted 5–6 tokens and had 4 accepted in every run, and the kept
  state was carried forward rather than cleared, which is what §3 asks for. The 2026-10-17 live
  session's collapse to 0% acceptance does not reproduce here.

These runs all took the copying route. The route that re-points whole blocks was measured earlier
and is not re-measured here; `NINFER_MULTISPAN_ALIGNED=1` selects it for a future run.

### Prefill is not bit-reproducible

Two cold prefills of the same prompt do not agree, whether they run in one Engine or in two. Six
prompts differed by 0.0003 to 0.003 nats at position 0 with identical server flags, and the number
moves from run to run. It has nothing to do with reuse: the same disagreement reproduces with reuse
switched off.

The cause was localized with a per-layer residual trace to the Q5 residual epilogue, which splits its
K dimension across two CTAs that both add into the same bf16 output element, so two otherwise
identical prefills can round differently in whichever order those CTAs land.
`NINFER_Q5_RESIDUAL_SPLITS=1` selects the single-writer schedule and makes prefills bit-identical
(three prefills of one prompt, zero divergent layers, confirmed on hardware 2026-10-11). It is an
override rather than the default because halving the K CTAs at `t <= 16` has not been benchmarked. A second, decode-side source is the w8 split-K
atomics in the MTP path, so with speculation on, decode rounds are not reproducible either; that one
costs pacing, not correctness, because drafts are verified against the base model.

For every measurement in this document it is the floor, and it decides how the numbers are read:

* Read the first-token log-probability difference, not the agreement count. A ~1e-3 nats difference
  is enough to flip a near-tie, which is why the counts move between runs (and between the 15 runs
  above) while the deltas stay put.
* Compare a hash of a chunk, not a sum of magnitudes. A magnitude total can show two states differ,
  but never that they agree; `prefill_trace.h` folds a hash for exactly that reason.
* `NINFER_PREFILL_TRACE=1` traces one chunk per layer, but only against a baseline the same Engine
  recorded, so two Engines cold-prefilling one prompt are invisible to it. `NINFER_PREFILL_TRACE_CROSS=1`
  publishes each completed chunk to a process-wide table keyed by shape and compares against the last
  one from any Engine, which is the only view that sees that pair.
* `ninfer_qwen3_6_27b_prefill_determinism_test` is the tripwire. Its default tolerance is `1e-6`, so
  it **fails** while the divergence exists: that is the intended state, not a broken test.
  `NINFER_PREFILL_DET_TOL` overrides it, and `0` demands bit-identity.

Still open: the k2 fixture's reuse divergence is an order of magnitude worse than the other k values
(0.0801 nats) while the amount relocated does not predict it. A reproducible prefill floor is a
precondition for diagnosing that — until it exists, a real quality regression and sensitivity in a
prompt that happens to place a near-tie look the same.

**Superseded note.** Until this revision, this section recorded the opposite result — the quantized
formats at 0/8 to 3/8 matching, 0.18 to 1.16 nats off, blamed on the quantized prefill kernel —
and §6 named fixing that kernel as the next task. That measurement was taken at 15:24 on the same
day, five and a half hours before `228e10b5` rewrote the quantized *decode* kernel (the query scale
tile moved off the tile it was overwriting, and a missing barrier was added). Nothing re-ran the
engine test afterwards, so the note described a build that no longer existed. Its reasoning did not
hold on its own either: it argued the prefill kernel must be at fault because `bf16` uses a
different prefill kernel, but `bf16` also uses a different decode kernel, so that observation could
not separate the two — and the quantized decode kernel was the one that was actually broken at the
time. The numbers above replace it.

---

## 6. Deferred Work

* BF16 prefill uses a fragile `INT32_MIN` sentinel to roll back in-place rotations (INT8 avoids this by staging a separate shifted rotary group).
* `ScrRelocationMode::Segmented` floors a span that overruns the last whole block before the limit, so up to 63 tokens per such run are re-prefilled even with `NINFER_SCR_SOURCE_SHIFT` set, where the copied path could take the partial block. The fallback does not rescue it while another span was selected; `[scr-diag] plan` reports it as `trimmed to whole blocks before the rewrite boundary`.
* Oracles test single dispatch shapes (8 decode, 128 prefill); other kernel variants (`TokenTile`, `MultiBatch`, `Masked`) inherit the boundary path without end-to-end test coverage.

---

## 7. INT8 Arbitrary Relocation

INT8 destinations aren't restricted to whole pages anymore.

* The rotary region (first 64 dims, group 0) gets an extra group-wide Q tile on partial pages.
* Counter-rotated and Hadamard-rotated to match the target frame.
* Adds one extra `mma_s8` per key block on partial pages, picking between shifted and base groups.

---

## 8. Test Suite Reference

| Test | Purpose |
| --- | --- |
| `tests/ops/test_gqa_kv_page_delta.cpp` | Delta oracle (whole/partial, both routes, negative controls; a run longer than a page as the control for the multi-page shape). |
| `tests/ops/test_gqa_kv_page_delta_i8.cpp` | INT8 aligned/off-grid tests, including a run longer than a page (partial head, whole pages, partial tail). |
| `tests/ops/test_gqa_kv_page_delta_rotated.cpp` | Rotated variants (`rk8v4`, `rk4v4`, `rk4v4-e8`), including the multi-page off-grid run. |
| `tests/ops/test_scr_relocation_integration.cpp` | End-to-end cache bytes check. |
| `tests/test_paged_kv_delta_layout.cpp` | Packed range encoding & planner math. |
| `tests/test_paged_kv_copied_relocation.cpp` | Copier oracle (`relocate_copied_pages`, device): a published mapping shorter than the resident one, the equal-length control, a growing mapping whose new destinations come from the pool, and the refusals. |
| `tests/targets/qwen3_6/test_prompt_edit.cpp` | Detector/selector planning, including the off-grid planner gate. |
| `tests/ops/test_scr_copied_relocation.cpp` | Copied-relocation oracle. |
| `tests/test_scr_e2e_shape.py` + `tools/scr_e2e_check.py` | Host-only relocation fixtures and plan-shape checks: Scope A, interleaved (two survivors), and the off-grid edit that forces the copied route. |
| `tests/targets/qwen3_6_27b/test_engine_prefill_determinism.cpp` | Prefill reproducibility (needs weights). |
| `tests/targets/qwen3_6_27b/test_engine_mid_prompt_edit.cpp` | What a single-span mid-prompt edit costs the retained recurrent state (~1,800 tokens, ~600 deleted; needs weights). |
| `tests/targets/qwen3_6_27b/test_engine_multi_span_edit.cpp` | The multi-span counterpart: one deep cut, three shallow cuts, and three deep cuts (~10k deleted) in one process, so the retained-state and multi-span-plan effects can be told apart. Reports agreement, first-token logprob delta and MTP counts; fails only if the reuse leg stops relocating (needs weights). Measured at `228e10b5`: every KV format matches the cold run on all three cases on the copying route — see §5. `NINFER_MULTISPAN_CASE` runs one case on its own; case A is the one-span probe. |
