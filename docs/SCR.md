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
* CTest: 98/98 pass (6 tests skipped due to missing model weights).
* **Accuracy:** Partial-page relative L2 vs direct prefill is $\sim 1.86\times 10^{-3}$ to $3.42\times 10^{-3}$ (well under the $2.0\times 10^{-2}$ limit).
* **Cost:** Partial-page decode is $1.007\times$ the baseline (essentially noise).

**Not covered:** Weights-dependent tests, `Segmented` runtime mode, prefill timing for boundary paths, long-context scaling, MSVC builds.

### The partly filled page reads back wrong on the quantized formats (measured 2026-10-18)

`ninfer_qwen3_6_27b_multi_span_edit_test` was run on the host (RTX 4090) with `bf16`, `int8`,
`rk4v4` and `rk4v4-e8` KV, on both relocation routes. Numbers are greedy agreement over the eight
generated tokens and the first-token logprob difference against a cold prefill of the edited
prompt, in nats.

| route | bf16 | int8 | rk4v4 | rk4v4-e8 |
| --- | --- | --- | --- | --- |
| aligned (whole pages) | 8/8 · 0.01 | — | — | 8/8 · 0.001–0.03 |
| shifted (run starts off a page) | 8/8 · 0.01 | 0/8 · 0.18–0.79 | 0/8–3/8 · 0.33–1.16 | 0/8–8/8 · 0.33–1.07 |

* bf16 is correct on the shifted route with the same plan, the same physical pages and the same
  per-page shift metadata the other formats read.
* The quantized formats are correct on the aligned route, which never leaves a page holding only
  part of a relocated run.
* Every format that fails (`int8`, `rk4v4`, `rk4v4-e8`) is served by the quantized prefill kernel
  (`gqa_attention_prefill_i8.cuh`); bf16, which is correct, is served by its own kernel. The fault
  is therefore there, in the handling of a page a relocated run only partly fills — not in the
  planner, the page copy, or the published shift.
* The agreement counts move from run to run because prefill is not bit-reproducible
  (`test_engine_prefill_determinism`): the choice is already marginal, so the ~1e-3 noise flips
  it. The first-token difference stays well above that floor, so read it, not the count.
* `tests/ops/test_gqa_kv_page_delta_i8.cpp` and `test_gqa_kv_page_delta_rotated.cpp` pass, so the
  shapes those oracles build are handled; the multi-span shifted plan builds a partly filled page
  they do not cover.

---

## 6. Deferred Work

* The quantized prefill kernel (`gqa_attention_prefill_i8.cuh`) reads a page that a relocated run
  only partly fills incorrectly, on every format it serves (`int8`, `rk4v4`, `rk4v4-e8`). bf16's
  own kernel is correct. Measured 2026-10-18; see §5. This is the next task.
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
| `tests/ops/test_gqa_kv_page_delta.cpp` | Delta oracle (whole/partial, both routes, negative controls). |
| `tests/ops/test_gqa_kv_page_delta_i8.cpp` | INT8 aligned/off-grid tests. |
| `tests/ops/test_gqa_kv_page_delta_rotated.cpp` | Rotated variants (`rk8v4`, `rk4v4`, `rk4v4-e8`). |
| `tests/ops/test_scr_relocation_integration.cpp` | End-to-end cache bytes check. |
| `tests/test_paged_kv_delta_layout.cpp` | Packed range encoding & planner math. |
| `tests/test_paged_kv_copied_relocation.cpp` | Copier oracle (`relocate_copied_pages`, device): a published mapping shorter than the resident one, the equal-length control, a growing mapping whose new destinations come from the pool, and the refusals. |
| `tests/targets/qwen3_6/test_prompt_edit.cpp` | Detector/selector planning, including the off-grid planner gate. |
| `tests/ops/test_scr_copied_relocation.cpp` | Copied-relocation oracle. |
| `tests/test_scr_e2e_shape.py` + `tools/scr_e2e_check.py` | Host-only relocation fixtures and plan-shape checks: Scope A, interleaved (two survivors), and the off-grid edit that forces the copied route. |
| `tests/targets/qwen3_6_27b/test_engine_prefill_determinism.cpp` | Prefill reproducibility (needs weights). |
| `tests/targets/qwen3_6_27b/test_engine_mid_prompt_edit.cpp` | What a single-span mid-prompt edit costs the retained recurrent state (~1,800 tokens, ~600 deleted; needs weights). |
| `tests/targets/qwen3_6_27b/test_engine_multi_span_edit.cpp` | The multi-span counterpart: one deep cut, three shallow cuts, and three deep cuts (~10k deleted) in one process, so the retained-state and multi-span-plan effects can be told apart. Reports agreement, first-token logprob delta and MTP counts; fails only if the reuse leg stops relocating (needs weights). Measured 2026-10-18: passes on the aligned route for every KV format and on the shifted route for bf16, fails on the shifted route for the quantized formats — see §5. |
