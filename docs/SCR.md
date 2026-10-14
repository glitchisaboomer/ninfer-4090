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
- `gqa_attention_decode_i8.cuh`, `gqa_attention_prefill_i8.cuh` — the same path on the quantized
  tiles, where the frame has to be chosen *before* quantization and only the rotary 64-dimension
  group has two of them

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
| `ninfer_gqa_kv_page_delta_i8_test` (whole-page + partial-page, both routes) | pass |
| `ninfer_gqa_kv_page_delta_rotated_test` (`rk8v4`, `rk4v4`, `rk4v4-e8`; whole-page + partial-page) | pass |
| `ninfer_scr_relocation_integration_test` | pass |
| Full CTest suite | 97/97 (6 tests skip without model weights: the two `*_real_test`s, both `*load_plan_test`s, `prefill_determinism`, `*prefix_real_test`) |

Measured on the partial-page oracle — the branch's newest paths — the delta-plane route reads
**2.25e-03** (decode) and **1.86e-03** (prefill) relative L2 against a directly prefilled cache,
against a 2.0e-02 limit, with the negative controls separating 2×–12×.

On the INT8 route the same comparison, with the relocated run starting mid-page so each end of it
lands inside a destination page, reads **3.42e-03** (plain `int8-g64`, decode) and **1.50e-03**
(prefill); `rk8v4` 4.20e-03 / 2.51e-03, `rk4v4` 1.20e-02 / 9.03e-03, `rk4v4-e8` 1.73e-02 / 1.32e-02,
against the same 2.0e-02 limit and against 9×–12× separation in the negative controls. The aligned
blocks read the same numbers to three digits (e.g. 3.414e-03 aligned against 3.418e-03 off-grid),
which is what the boundary path being an exact per-key frame choice rather than an approximation
predicts.

Costed on the RTX 4090 at a 4096-token context (T=8, `int8-g64`, 1000 iterations × 5 interleaved
passes, minimum taken): an all-zero plane set 0.0639 ms/op on the pre-batch route and 0.0647 ms on
this one; a whole-page relocation 0.0656 ms; the partial-page relocation 0.0651 ms — **1.007×** the
no-plane baseline, i.e. inside the run-to-run spread of the same configuration. The extra work is
confined to the partial pages (at most two per relocated span), and staging the shifted group only
re-runs on a page transition that changes a frame.

**Not covered:**

- The six tests that need model weights skip here. SCR is verified at the op/oracle level, not
  end-to-end through the 27B/35B engine.
- `Segmented` mode is planner-only; the runtime acts on `Tail` alone.
- Prefill timing for the boundary path is unmeasured; only the decode op was costed. The prefill
  kernel is also the one whose arena grew, so its occupancy is worth a measurement before the
  feature is trusted on the 35B shapes.
- Long-context cost is unmeasured: the decode figures above are for a single 4096-token context with
  one relocated span.
- The Windows/MSVC build is the official gate; this verification is Linux/GCC.

## 6. Known deferred work

- The BF16 prefill boundary recompute rolls back an in-place `q_s` rotation via an `INT32_MIN`
  sentinel; it works but is fragile against changes to the reload early-out. The INT8 prefill does
  not need the sentinel — it stages the shifted rotary group beside the base tile — so if the BF16
  kernel is ever revisited, that is the shape to copy.
- `ScrRelocationMode::Segmented` has no runtime path.
- Each oracle exercises one dispatch shape per route (8 decode tokens, 128 prefill tokens), so the
  other instantiations of the same kernel body — other `TokenTile`s, `MultiBatch`, `Masked` — carry
  the boundary path but are not exercised end to end. Every codec variant is covered, each at that
  one shape.

---

## 7. INT8 arbitrary (partial-page) relocation

**Implemented and verified at the op level.** An INT8 destination is no longer quantized to whole
pages: the planner returns a run's real token interval for every dtype, and both INT8 attention
routes read the packed interval and choose the query's frame per key, as BF16 does.

How the frame is carried. Rotation touches only the rotary region, which is the first 64 dimensions
of the quantized tile — group 0 — so a partial page needs one extra group-wide Q tile rather than a
second full-width one. Each INT8 kernel stages that tile in the frame of the position its page's
relocated keys were computed at, beside the base tile:

- `gqa_attention_kv_quant.cuh` owns the tile: 32 b16 lanes per row, the four-unit form of the b16
  swizzle (`(col >> 3) ^ (row & 3)`), with `gqa_kv_shift_store_i8` and `gqa_kv_shift_swz` as the one
  definition of its layout for the store and for every `ldmatrix` that reads it.
- Group 0 is counter-rotated *and* Hadamard-rotated for the rotated codecs, in that order, matching
  `stage_and_quantize_q`. The shifted group is quantized in the frame it is read in, so the shifted
  product is a real QK product in the rotated basis and not a re-rotation of a quantized tile.
- The score path takes one extra group-0 `mma_s8` per key block on a partial page and then selects
  per key from the packed interval: keys inside it use the shifted group with the scale that group
  quantized to, keys outside it use the base tile. Groups 1..3 are shared, so this is one extra
  group product on one page's keys rather than a second QK recompute.
- A page transition re-stages only the frame that changed: `current_delta` tracks the base tile,
  `current_shift` the shifted one, and a partial page's base frame is the unrotated one — its keys
  outside the interval never moved.
- Neither INT8 kernel rotates Q in place, so neither needs the `INT32_MIN` rollback sentinel the BF16
  prefill path carries (§6): the shifted group is a separate tile and the base tile is left alone.

The four gaps this section used to list are closed: the range pointer reaches `launch_tc_partial_i8`
and is threaded into the kernel; both INT8 kernels have the boundary-aware score path; the planner's
`destination_may_shift = true` branch no longer trims by dtype (`request_plan_impl.h`); and the A1
route is covered by the same blocks and the same limit as A3.

The decode launcher now requests dynamic shared memory for every INT8 configuration, not only the
ones whose K/V arena is dynamic, because the shifted group lives there. That is a residency change
on a kernel tuned for occupancy on `sm_89`, which is why the cost below is measured against the
pre-batch route and not against a same-binary control.

Definition of done, and where each part landed:

- Partial-page blocks in the two INT8 oracles, judged against the same 2.0e-02 limit as the BF16
  blocks, with negative controls that separate — done; numbers in §5.
- A scenario that places the run into the destination physical pages — done, as
  `place_unaligned_relocation_i8` in the shared header. It is one production `append_chunk` at the
  destination positions rather than the BF16 placer's byte-level copy, because the INT8 fill
  quantizes each row on its own: writing the source-phase rows at the destination slots produces
  exactly the codes and scales a page copy would have moved there.
- Full CTest suite green — 97/97 on this batch (§5), on the Linux/GCC host. The Windows/MSVC box
  and the real-model tests still have to run before the feature is called end-to-end verified.

Cost, measured (§5): the partial-page decode reads 1.007× the no-plane baseline and 0.992× the
whole-page case at a 4096-token context, inside the spread between runs of one configuration. The
occupancy question the dynamic-smem change raised is answered by the same numbers — 0.0647 ms here
against 0.0639 ms on the pre-batch route, 1.3%, at the edge of the noise. What is *not* priced is the
prefill route, whose arena grew by a whole tile, nor a long context carrying many relocated spans at
once.

---

## 8. Tests

| Test | What it pins |
|---|---|
| `tests/ops/test_gqa_kv_page_delta.cpp` | The delta-plane oracle, whole-page and partial-page, both attention routes, plus negative controls. |
| `tests/ops/test_gqa_kv_page_delta_i8.cpp` | Both scenarios (aligned and off-grid) on a plain INT8 plane, on both attention routes, each with its own negative control. |
| `tests/ops/test_gqa_kv_page_delta_rotated.cpp` | The same two scenarios on the rotated variants (`rk8v4`, `rk4v4`, `rk4v4-e8`), each with a host port of the 64-point Hadamard. |
| `tests/ops/test_scr_relocation_integration.cpp` | Every cache byte written by `gqa_kv_append`, read through the public entry points. |
| `tests/test_paged_kv_delta_layout.cpp` | The packed range encoding and the planner arithmetic, off-machine. |
| `tests/ops/test_scr_copied_relocation.cpp` | The copied-relocation oracle (mainline): bit-identical moved bytes, pool accounting, refusal cases. |
| `tests/targets/qwen3_6_27b/test_engine_prefill_determinism.cpp` | Prefill reproducibility (needs weights). |

The oracle tests share one scenario and one set of helpers
(`tests/ops/gqa_relocation_test_common.h`), so a scenario change moves all of them together — which
is why the shared BF16 conformance criterion is calibrated once, in that header.
