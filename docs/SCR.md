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



---

## 3. Relocation & Invariants

Keys aren't re-written with new rotations; they keep their original source rotation, and the delta plane records the offset so the read path can counter-rotate the query.

* Planner publishes `delta = dst - src` and interval $[begin, end)$ per destination page.
* **Partial page:** Interval is a strict subset. `is_partial` drives the boundary-aware score path.
* Planes publish/clear together (one relocation or none per page).
* Spans never share a logical destination page. Understating the delta plane silently acts like no relocation happened.

**Planner gate.** The gate builds one diff per request and feeds it to whichever selector the mode names, so the detector bounds every plan. `diff_prompt_edit` matches whole 64-token blocks and sees only content on the same page grid in both prompts, so an off-grid edit leaves it nothing but the leading prefix. With `NINFER_SCR_SOURCE_SHIFT` set the gate uses `diff_prompt_edit_shifted` (all-offset) instead, and `Segmented` falls back to `plan_shifted_relocations` after an inexpressible plan rather than dropping to a cold prefill; unset, the gate is byte-identical to the aligned code. `NINFER_SCR_DIAG` adds one `[scr-diag] segmented … survivors=… plan_spans=… shift=…` line per request, so segmented can no longer decline silently. The shifted detector anchors at the page size, so survivals shorter than a page stay invisible to it.

---

## 4. Relocation Routes

1. **Re-pointing (Default):** Updates block table to point to an existing physical page. Zero copy cost, requires free destination pages.
2. **Copied Relocation:** Copies payload into acquired pool pages. Handles occupied destinations and off-grid sources.

---

## 5. Verification Status

Verified on RTX 4090 (`sm_89`, CUDA 13.3, GCC 11, Linux):

* `ninfer_gqa_attention_test`, `ninfer_gqa_kv_page_delta_test` (BF16), `_i8_test`, `_rotated_test`, `ninfer_scr_relocation_integration_test` all pass.
* CTest: 97/97 pass (6 tests skipped due to missing model weights).
* **Accuracy:** Partial-page relative L2 vs direct prefill is $\sim 1.86\times 10^{-3}$ to $3.42\times 10^{-3}$ (well under the $2.0\times 10^{-2}$ limit).
* **Cost:** Partial-page decode is $1.007\times$ the baseline (essentially noise).

**Not covered:** Weights-dependent tests, `Segmented` runtime mode, prefill timing for boundary paths, long-context scaling, MSVC builds.

---

## 6. Deferred Work

* BF16 prefill uses a fragile `INT32_MIN` sentinel to roll back in-place rotations (INT8 avoids this by staging a separate shifted rotary group).
* `ScrRelocationMode::Segmented` lacks a runtime path.
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
| `tests/targets/qwen3_6/test_prompt_edit.cpp` | Detector/selector planning, including the off-grid planner gate. |
| `tests/ops/test_scr_copied_relocation.cpp` | Copied-relocation oracle. |
| `tests/targets/qwen3_6_27b/test_engine_prefill_determinism.cpp` | Prefill reproducibility (needs weights). |
