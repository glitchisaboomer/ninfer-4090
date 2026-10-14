# Suffix Cache Reuse (SCR) — feature specification

**Status: implemented and verified at the op level.** See §5 for what that covers and what it does not.

SCR reuses the KV cache of a previous turn's prompt for the tokens that survive an edit, instead of
re-prefilling the whole prompt. It is an approximation — reused states were computed under the
pre-edit context — so it is opt-in and off by default.

This document is the single reference for the feature. The design documents that used to live in
`docs/` were retired when this was written; source comments that used to point at them point here.

---

## 1. Concepts

| Term | Meaning |
|---|---|
| **Relocation** | Making a surviving token range readable at its new prompt positions after an edit shifts later tokens. |
| **Delta plane** | Per-logical-page RoPE phase shift, `int32` positions, indexed like the block table. A page with delta `d` holds keys rotated at a position `d` away from where they are now read. |
| **Packed range plane** | Per-logical-page relocated interval `[begin, end)` in token offsets, packed 6-bit begin + 7-bit end into one `uint32`; `0xffffffff` means no interval. |
| **Whole-page delta** | The relocated interval covers an entire page, so one rotation of the query describes every key in it. |
| **Partial page (off-grid)** | The interval is a strict subset of the page: part of its tokens were relocated and part were not, so no single whole-page shift describes either half. |
| **Re-pointing** | Relocating by editing the block table so a logical page names a different physical page. No payload moves. |
| **Copied relocation** | Relocating by copying the payload into acquired pages. Needed when the destination pages are already occupied. |
| **ScrRelocationMode** | `Tail` (one contiguous surviving run — the only mode the runtime acts on) or `Segmented` (disjoint runs; planner only). |

## 2. Public surface

`include/ninfer/types.h`:

- `ScrRelocationMode { Tail, Segmented }`
- `ExecutionOptions::enable_suffix_cache_reuse` — opt-in, default `false`
- `ExecutionOptions::suffix_reuse_max_spans` — bounds the approximation introduced at once, default 6
- `ExecutionOptions::scr_mode` — default `Tail`

`src/targets/qwen3_6/impl/runtime/prompt_edit.h` (planner, host-only):

- `plan_tail_relocation(...)` — the single contiguous run (`Tail` mode)
- `plan_shifted_relocations(..., destination_may_shift = false)` — general selector (`Segmented`).
  Passing `destination_may_shift = true` allows destinations off the 64-token page grid.
- `plan_copied_page_remap(...)` — the block-table remap for a copied relocation
- `shifted_relocation_pages_from_pool(...)` / `remap_pages_from_pool(...)` — pool demand pricing

`src/core/paged_kv_delta_layout.h` (pure arithmetic, host- and device-callable):

- `pack_rope_delta_page_range(begin, end)` / `rope_delta_page_range_{is_empty,begin,end,is_partial}`
- `plan_rope_delta_page_ranges(spans, mapping_pages)` → per-page `{begin, end, delta}`
- `encode_rope_delta_page_ranges(...)` → the packed plane

`src/core/paged_kv_cache.h` (device storage):

- `PagedKVPool::rope_deltas()` / `rope_delta_ranges()` — the two planes, sized from and parallel to
  the block table, never independently.
- `PagedKVAllocation::copy_pages_from_host(...)` — the copier

`src/ops/` (kernels, `sm_89`):

- `gqa_attention_decode_bf16.cuh`, `gqa_attention_prefill_bf16.cuh` — boundary-aware score path
- The INT8 equivalents follow the same delta-plane contract.

## 3. How a relocation is expressed

One span of surviving tokens is relocated from source positions `[src, src+tokens)` to destination
positions `[dst, dst+tokens)`. The keys are **not** re-rotated on write: they keep the rotation of
the source position they were computed at, and the delta plane records the offset so the read path
can counter-rotate the query back into the keys' frame.

Per destination page the planner publishes:

- `delta = dst - src` (uniform across the span, because the destination grid is whole pages and the
  source is one contiguous run), and
- the interval `[begin, end)` the relocated tokens occupy in that page.

A page is **partial** when its interval is a strict subset of the page. `is_partial` is the single
definition of "the boundary-aware score path applies"; every kernel takes it from
`paged_kv_delta_layout.h` rather than re-deriving it, because a derivation that forgets the emptiness
guard mistakes an ordinary page for a partial one — a silent failure in which both halves rotate
plausibly.

The planes are published and cleared together, and a page carries either no delta or one
relocation's. Invariants:

- Spans returned by the planner never share a logical destination page.
- One interval per page — which is why a caller that merges, splits or otherwise combines spans
  downstream has to preserve that property.
- A delta plane that understates the block table reads as "relocation silently did nothing" rather
  than as an error; `rope_delta_layout_is_supported` rejects it.

## 4. The two relocation routes

**Re-pointing** (default) — the destination logical page is re-pointed at a surviving physical page.
No payload moves, so it costs nothing per page, but it needs the destination pages to be free.

**Copied relocation** — the payload is copied into pages acquired from the pool. Required when the
destination pages are occupied, and it is the only route that can relocate an off-grid source: a
contiguous run starting mid-page occupies two physical pages at offsets the block table alone cannot
express. The copy is a `PagedKVAllocation` method inside a real `PagedKVPool`; the pool's sources are
held for the duration of the copy and returned after, and `plan_copied_page_remap` prices the
acquire demand.

## 5. Verification status

Verified on the RTX 4090 (`sm_89`, CUDA 13.3, GCC 11), Linux host:

| Suite | Result |
|---|---|
| `ninfer_gqa_attention_test` | pass |
| `ninfer_gqa_kv_page_delta_test` (BF16, whole-page + partial-page) | pass |
| `ninfer_gqa_kv_page_delta_i8_test` | pass |
| `ninfer_gqa_kv_page_delta_rotated_test` (`rk4v4-e8`) | pass |
| `ninfer_scr_relocation_integration_test` | pass |
| Full CTest suite | 97/97 (5 real-model tests skip without weights) |

Measured on the partial-page oracle — the branch's newest path — the delta-plane route reads
**2.25e-03** (decode) and **1.86e-03** (prefill) relative L2 against a directly prefilled cache,
against a 2.0e-02 limit, with the negative controls separating 2×–12×.

**Not covered:**

- The five real-model tests need model weights and skip here. SCR is verified at the op/oracle
  level, not end-to-end through the 27B/35B engine.
- `Segmented` mode is planner-only; the runtime acts on `Tail` alone.
- INT8 and packed layouts still quantize destinations to whole pages — the packed interval path is
  BF16-only.
- Performance is unmeasured.
- The Windows/MSVC build is the official gate; this verification is Linux/GCC.

## 6. Known deferred work

- The "is this page partial" predicate is re-derived from the packed value at five call sites
  instead of having one definition.
- The prefill boundary recompute rolls back an in-place `q_s` rotation via an `INT32_MIN` sentinel;
  it works but is fragile against changes to the reload early-out.
- `gqa_attention_decode_i8.cuh` threads a range pointer it never reads.
- `ScrRelocationMode::Segmented` has no runtime path.

---

## 7. Next task: INT8 arbitrary (partial-page) relocation

**Not started.** The BF16 path supports arbitrary token offsets (§3); the INT8 path does not. Today an
INT8 destination is quantized to whole pages, so up to 63 tokens at each end of a surviving run are
re-prefilled rather than relocated — a run with no whole destination page cannot be relocated at all
in that mode.

The gap, concretely:

1. **The range plane never reaches the INT8 kernels.** `launch_tc_partial_i8`
   (`src/ops/launcher/gqa_attention_decode.cu`) takes `rope_page_deltas` but no range pointer, and
   `gqa_attention_decode_i8.cuh` reads neither the packed interval nor
   `rope_delta_page_range_is_partial`. The dispatch macro even casts
   `cache.rope_delta_ranges.data` on the way in, so the plane is available at the call site and
   dropped on the floor.
2. **No boundary-aware score path in the INT8 kernels.** The BF16 kernels
   (`gqa_attention_{decode,prefill}_bf16.cuh`) recompute QK scores with the shifted query for the
   keys inside a partial page's interval and select per key; the INT8 equivalents have no analogue.
   The rotated variants add a wrinkle: the host-side 64-point Hadamard means the shifted-query
   recompute has to happen in the same rotated frame the plane is quantized in.
3. **The planner still quantizes.** `plan_shifted_relocations(..., destination_may_shift = true)`
   trims to whole pages for quantized destinations. With the kernels able to read an interval, that
   branch can return the full token interval and let the plane describe the partial page.
4. **The prefill (A1) route is unverified for INT8 partial pages.** The BF16 prefill path carries the
   `INT32_MIN` rollback sentinel noted in §6; whatever the INT8 path does instead should be settled
   before it is trusted.

Definition of done, mirroring how the BF16 path was pinned:

- `ninfer_gqa_kv_page_delta_i8_test` and `ninfer_gqa_kv_page_delta_rotated_test` gain partial-page
  blocks that measure against the same 2.0e-02 equivalence limit the BF16 blocks use, with negative
  controls that still separate.
- The oracle scenario must place the relocated run slot-by-slot into the destination physical pages
  (as `place_unaligned_relocation` in `test_gqa_kv_page_delta.cpp` does for BF16) — an append at the
  source positions plus a block-table alias does **not** produce a readable off-grid cache, which is
  what made the BF16 fixture silently wrong.
- Full CTest suite green, including the five real-model tests on the Windows box.

Risks worth pricing first: the INT8 kernels are tuned for occupancy on `sm_89`, and the
boundary-aware path adds a second QK recompute per partial tile. Whether that costs anything
measurable on the decode path is unknown and should be measured before the design is settled.

---

## 8. Tests

| Test | What it pins |
|---|---|
| `tests/ops/test_gqa_kv_page_delta.cpp` | The delta-plane oracle, whole-page and partial-page, both attention routes, plus negative controls. |
| `tests/ops/test_gqa_kv_page_delta_i8.cpp` | Same scenario on a plain INT8 plane. |
| `tests/ops/test_gqa_kv_page_delta_rotated.cpp` | Rotated variants (`rk8v4`, `rk4v4-e8`) with a host port of the 64-point Hadamard. |
| `tests/ops/test_scr_relocation_integration.cpp` | Every cache byte written by `gqa_kv_append`, read through the public entry points. |
| `tests/test_paged_kv_delta_layout.cpp` | The packed range encoding and the planner arithmetic, off-machine. |
| `tests/ops/test_scr_copied_relocation.cpp` | The copied-relocation oracle (mainline): bit-identical moved bytes, pool accounting, refusal cases. |
| `tests/targets/qwen3_6_27b/test_engine_prefill_determinism.cpp` | Prefill reproducibility (needs weights). |

The oracle tests share one scenario and one set of helpers
(`tests/ops/gqa_relocation_test_common.h`), so a scenario change moves all of them together — which
is why the shared BF16 conformance criterion is calibrated once, in that header.
