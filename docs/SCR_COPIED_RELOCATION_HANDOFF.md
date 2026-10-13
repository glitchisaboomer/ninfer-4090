# Handoff: the copied-relocation device oracle — ladder item 3

For a model picking this up cold. This is ladder item 3 of
`docs/SCR_ALIGNMENT_FREE_RELOCATION.md`: *relocate an off-grid source by payload copy and compare the
destination pages against a freshly prefilled cache, bit for bit — that is the oracle the copy must
match, and it is also the test that catches an aliasing bug.*

**Nothing is broken.** The host arithmetic is green on every level, and the one hardware run that
has happened passed. What is open is narrower and more serious than "broken": **the copied
relocation has never once executed on a GPU.** The device copy was written blind (no nvcc, no GPU)
in `a1c27b26` / `cec7a172` / `8ef3a7bd`, and the run that "passed" on 2026-10-13 did not reach it —
see *Read this first*. This document specifies the test that closes that gap.

Build and run happen on the Windows/CUDA box; the Linux sandbox has no nvcc and can only
type-check.

**Status (2026-10-13): implemented, never executed.** `tests/ops/test_scr_copied_relocation.cpp`
exists and is registered in `tests/CMakeLists.txt`, and it is the form this document specifies
(seven assertion groups, the real planner chain, the `{bf16, rk4v4-e8} x {PageMajor, HeadMajor}`
matrix with the one correction recorded under *The pool*). What has been verified, and how:

- the scenario's arithmetic — spans, mapping, deltas, the 8 moves, `from_pool = 7`, held = 7 — was
  run against the real planner chain on a host build (`prompt_edit.cpp` and
  `paged_kv_delta_layout.h` are CUDA-free), and the test asserts those numbers;
- the TU compiles clean (`g++ -std=c++20 -Wall -Wextra -fsyntax-only` against the repo headers with
  minimal CUDA stubs, since this sandbox has no toolkit);
- **nothing in it has run.** There is no CUDA device here, so no byte comparison, no attention
  number and no control separation exists yet. The gate below still stands: Phase 4d cannot be
  committed until this binary is green on the Windows box.

---

## Read this first: what actually ran on hardware

The 2026-10-13 e2e run (`--kv-dtype rk4v4-e8 --scr-mode segmented --shape interleaved`,
`NINFER_SCR_SOURCE_SHIFT=1`) engaged relocation, saved 256 of 990 prompt tokens, and produced
output identical to `--no-scr`. It did **not** exercise the copy. Its `[scr-diag]` line says so:

```
span{src=704 dst=576 tok=64 delta=-128 ...} span{src=960 dst=768 tok=192 delta=-192 ...}
remap[...,11,...,15,16,17,...]      <- kept pages, not acquire slots
```

704 and 960 are both exact multiples of the 64-token page. The e2e fixture *aligns by
construction*: `tools/scr_e2e_check.py`'s `solve_alignment` pads each region until the survivors
land on the grid (it prints "block aligned" per region, and the two edits differ by 192 = 3
blocks). With every span grid-aligned, `plan_page_remap` returns a re-pointing mapping, the
`remap.empty() && scr_source_shift_enabled()` branch in `program_impl.h` is not taken, and
`relocate_copied_pages` is never called.

So the shifted *detector* and *selector* ran; the *copier* did not. On the shape the feature exists
for — an edit whose token delta is not a multiple of 64 — the e2e harness still cannot reach the
copy at all, because the fixture refuses to produce that shape. Until this test exists, the
copied path has exactly zero execution anywhere. That is the gap.

## State of play

| Item | Status |
|---|---|
| Detector at any offset (`diff_prompt_edit_shifted`) | done; host-tested, `tests/targets/qwen3_6/test_prompt_edit.cpp` green |
| Selectors (`plan_shifted_relocations`) and copy mapping (`plan_copied_page_remap`) | done; host-tested in the same file |
| Pool pricing with sources held out (`shifted_relocation_pages_from_pool`) | done; host-tested |
| Move decomposition and substitution arithmetic (`plan_relocated_page_moves`, `plan_relocated_page_substitution`, `rope_delta_for_dest_page`, `relocated_dest_is_page_aligned`) | done; host-tested, `tests/test_paged_kv_delta_layout.cpp` green |
| Device copy (`PagedKVAllocation::relocate_copied_pages`, `src/core/paged_kv_cache.cu:752`) and its runtime wiring (`program_impl.h`, the `copying` branch) | **written blind — never compiled before 2026-10-13's fix, never executed** |
| Runtime branch that engages the copier | compiles; reached only by an off-grid span, which no fixture produces |
| Hardware e2e, flag on | passed, but re-pointed; the copier was not on the path |
| **Device oracle for the copy** (`tests/ops/test_scr_copied_relocation.cpp`) | **written and registered; arithmetic host-verified and TU syntax-clean; never run — no GPU on the authoring machine** |

---

## What to write

A new file, `tests/ops/test_scr_copied_relocation.cpp`, registered beside the existing integration
test in `tests/CMakeLists.txt`:

```cmake
# The copied-relocation oracle: an off-grid source relocated by payload copy, held bit-for-bit
# against the pages it was copied from and read back through the public attention entry points.
# Device-only (a PagedKVPool and a CUDA stream), gated off by ninfer_add_op_test's SKIP_RETURN_CODE.
ninfer_add_op_test(ninfer_scr_copied_relocation_test
  SOURCES ops/test_scr_copied_relocation.cpp
  LIBRARIES ninfer_ops ninfer_engine)
```

`ninfer_engine` is required and not optional: the planner chain the scenario drives
(`prompt_edit.cpp`: `diff_prompt_edit_shifted`, `plan_shifted_relocations`,
`plan_copied_page_remap`) compiles into `ninfer_engine`, not `ninfer_ops`, so a
`ninfer_ops`-only registration fails to link. `prompt_edit.h` is CUDA-free by design, which is
what lets a device test include it at all.

Why a new file instead of extending `test_scr_relocation_integration.cpp`: that test is BF16-only
and hand-builds its own plane tensors through `gqa_relocation_test_common.h`'s `Cache`. The copier
is a method on `PagedKVAllocation` inside a real `PagedKVPool`, so this harness must be
pool-backed — a different shape, not an extension of the old one. **Do not refactor the shared
header** to make it fit; the two standing oracle tests must not drift for this. Reuse what is
reusable by inclusion: the FP64 oracle pattern (`ideal_attention_with_delta`,
`expect_conformant` / `expect_control_outside` / `expect_control_separates`,
`kConformanceRotatedI8`) is a template over `image() / k_logical / v_logical`, so a ~40-line
pool-backed adapter in the new file satisfies it without touching the header.

### The pool

Ground truth for the plane set is `src/targets/qwen3_6/impl/state/decoder_state.cpp` (the loop
that fills `PagedKVPoolSpec::planes`). Per layer, quantized:

| plane | dtype | leading extent | heads |
|---|---|---|---|
| k codes | `DType::U8` (packed) | `head_dim / 2` = 128 | `kv_heads` = 4 |
| v codes | `DType::U8` (packed) | `head_dim / 2` = 128 | 4 |
| k scale | `DType::FP16` | `head_dim / quant_group` = 4 | 4 |
| v scale | `DType::FP16` | 4 | 4 |

BF16 variant: two planes, `DType::BF16`, leading `head_dim`. Text pools are `PageMajor`;
`DFlash`'s is `HeadMajor`. **Run the whole matrix `{bf16, rk4v4-e8} × {PageMajor, HeadMajor}`** —
the copier reads its strides out of the plane `Tensor` instead of deriving them precisely so both
orders work (`paged_kv_cache.cu`, the `page_major` branch), and that code is one of the written-blind
review items named in `a1c27b26`'s message.

**Correction (implemented, 2026-10-13): the attention matrix is PageMajor only.** The Ops address
pages page-major — `paged_kv_page_head_offset` (`src/ops/kernel/paged_kv_address.cuh:20-25`) hardcodes
`head + HeadExtent * physical_page` — so a `HeadMajor` plane neither passes `validate_cache`'s shape
and contiguity rules (`{leading, page, heads, pages}` with contiguous strides) nor can be written by
`ops::gqa_kv_append`. HeadMajor cells therefore run groups 1-4, 6 and 7, and their seed comes from a
scratch `PageMajor` pool of the same spec through the same writer, transplanted page by page: the
pool's page image is `[head][page_offset][leading]` in both orders, so every cache byte still
originates from the production fill path. The copier's order-independence — the thing the matrix
exists to prove — is exactly what the byte groups exercise; the kernel route cannot see a HeadMajor
plane in production either. The pool sizing that fell out of the verified arithmetic: 22 owned
pages, 9 named by the mapping, 13 acquire slots, 7 of them read by the copies, so `from_pool = 7`
and the tight pool is 29 groups against the main cell's 30.

Build it with `plan_paged_kv_pool(LayoutBuilder&, PagedKVPoolSpec{page_group_count,
logical_page_capacity, table_rows = 1, plane_order, planes})`, construct `PagedKVPool(DeviceSpan,
layout)`, then `pool.reserve(entitlement)` → `PagedKVAllocation`,
`allocation.materialize_tokens(...)`, `allocation.bind_row(0, stream)` — bind **before** any
append, it publishes the block table and clears the delta row.

### The scenario

The shipped scenario is a 1408-token (22-page) prompt whose first 576 tokens are prefix reuse, with
three survivors driven through the real detector and selector from two synthetic prompts (`P R1 S1
R2 S2 R3 S3` → `P S1 E2 S2 E3 S3 E4`, region ids chosen so no two regions share a token, so a run
cannot grow past its own region). The leading span is prefix reuse and the selector skips it; the
other three carry the three shapes:

| span | source_begin | dest_begin | tokens | why it is there |
|---|---|---|---|---|
| S1 | 741 (= 11·64 + 37, off-grid) | 576 | 128 | two-move destination pages |
| S2 | 960 (= 15·64, on the grid) | 768 | 64 | the single-move branch |
| S3 | 1216 | 896 | 192 | three destination pages |

`plan_relocated_page_moves` then issues 8 moves; S1's destination pages each split across a source
page boundary:

| dest page | move 1: from | move 2: from |
|---|---|---|
| 9 | page 11, offset 37, 27 tokens | page 12, offset 0, 37 tokens |
| 10 | page 12, offset 37, 27 tokens | page 13, offset 0, 37 tokens |

and S2/S3 are one move per page (12 ← 15, 14 ← 19, 15 ← 20, 16 ← 21). The mapping keeps pages 0-8,
acquires the other 13; the deltas are −165 on pages 9-10, −192 on page 12 and −320 on pages 14-16,
`kRopeDeltaNone` elsewhere; `from_pool` is 7. All of these numbers were verified against the real
planner chain on a host build before being written down here, and the test asserts them, so a planner
change is a failure rather than a silent re-shape.

Build the mapping and deltas with the **real** planner pair, not a hand-built array:
`qwen3_6::detail::plan_copied_page_remap(current_pages, spans, prefix_reuse_tokens, prompt_tokens)`
for the mapping and `ninfer::rope_delta_for_dest_page` per destination page for the deltas (that is
exactly what `program_impl.h`'s `copying` branch does — mirror it; there the third argument is the
local `base`, which *is* the prefix-reuse frontier). `prompt_edit.h` is CUDA-free
and host-buildable by design, so including it from this test is sanctioned; driving one case
through `diff_prompt_edit_shifted` on two token sequences is worth it, because it makes the test
cover detector → selector → copier as one chain.

Seed the source run through the production write path: `ops::gqa_kv_append` over a
`PagedKVLayerView` built from `pool.plane(i)` tensors plus `allocation.block_table()`. Every cache
byte comes from the writer, exactly as the old test's contract requires. After the copy, the pages no
span covers (the runtime's prefill ranges) are written through the same path, so the copied cache
differs from a directly prefilled one only in what the relocation itself moved.

### The assertions, in order

1. **Bytes, with no tolerance at all.** Before the copy, host-copy the source pages; after it,
   host-copy the destination pages (`pool.copy_page_to_host(plane_index, page_id, dst, stream)` per
   plane per page — sync before trusting the buffer). For every move, the destination page's
   `[dest_offset, dest_offset + move.tokens)` bytes, per head, must be **bit-identical** to the
   source page's `[source_offset, source_offset + move.tokens)`. This is the copy's whole
   contract: the payload moved, nothing was re-derived. It is also the "bit for bit" the ladder
   asks for, and it is what a wrong physical page, a wrong token offset, a wrong plane or a stale
   scale looks like. **All four planes, including both scale planes** — a stale scale is a
   plausible wrong KV, the worst failure class there is.
2. **Nothing else moved.** Bytes outside the moved ranges in the destination pages, and every page
   the mapping keeps, unchanged.
3. **What was published.** `allocation.page_ids()` equals the mapping with acquire slots filled;
   `pool.rope_delta_row(0)` read back equals the deltas (per destination page:
   `dest_begin - source_begin`, `kRopeDeltaNone` everywhere else); the block table row is
   consistent. Mapping and deltas are published in one call — that coupling is the thing that
   makes a half-relocation impossible, so assert both sides of it.
4. **Pool accounting.** After the call, the pages `held_for_copy` are back in the pool
   (`pool.free_pages()` grew by the held count) and no destination page is still free. The
   drain-before-release ordering is a correctness property, not hygiene: the pool is shared and
   another lane takes a released page on its own stream with nothing ordering against the copies.
5. **Operational equivalence.** Attention over the copied cache — `gqa_attention_cached` through a
   view carrying `rope_deltas` and `rope_inv_freq`, the same public route the served runtime uses
   — against a cache directly prefilled at the destination, judged by
   `ideal_attention_with_delta`: the FP64 oracle for exactly this case (keys quantized at the
   *source* RoPE phase, read at the *destination* positions with the published per-page delta). For
   `rk4v4-e8` use the anchored pair — `expect_conformant` at `kConformanceRotatedI8` plus
   `expect_control_outside` on a zeroed delta plane — **not** the multiplicative
   `expect_control_separates`: on a coarse (int4) plane the correct path's own error is dominated
   by encode noise and the separation collapses, which is why the rotated profile exists. Keep the
   multiplicative control for the BF16 variant. An A1 (append-and-attend) pass over the copied page
   is cheap and worth it — that is the route prefill itself uses.
6. **Aliasing, through the public API.** Size the pool tight so the relocation's surplus *is* the
   copy's source set: without the substitution's protection the acquire slots would be filled from
   the very pages being read, which is an overlapping `cudaMemcpy2DAsync` and undefined. The design
   chose reservation, not a staging kernel — `plan_relocated_page_substitution`'s protected-page
   rule — so the device assertion is that the published destinations never name a protected page
   **and** assertion 1 still holds byte-for-byte under that tight configuration. The arithmetic is
   host-tested; what is untested is that it holds for real page ids through the real call, which is
   the only place an aliasing bug could still live.
7. **Refusals.** `deltas` not parallel to `mapping`, empty `spans`, a mapping past the entitlement,
   and a pool that cannot afford `from_pool` — each must throw (`std::invalid_argument` /
   `std::bad_alloc`) **before any page moves**. That is the never-half-publish property this whole
   feature's error handling rests on.

### What is already covered — do not redo

| Surface | Test |
|---|---|
| Move decomposition, substitution, destination-coordinate delta predicate | `tests/test_paged_kv_delta_layout.cpp` (host, green) |
| Shifted detector, shifted selector, copied mapping, pool pricing | `tests/targets/qwen3_6/test_prompt_edit.cpp` (host, green) |
| Fixture and plan-shape arithmetic | `tests/test_scr_e2e_shape.py` (22 green) |
| Re-pointing relocation equivalence (aligned path) | `ninfer_scr_relocation_integration_test`, `ninfer_gqa_kv_page_delta{,_i8,_rotated}_test` (device, green) |

### Traps

- **The e2e pass is not evidence about the copier.** If the `[scr-diag]` span line shows a
  grid-aligned `src` and a `remap` with kept pages, the copy did not run — regardless of the flag.
- **`rk4v4-e8` is four planes with three different leading extents.** Derive the expected bytes per
  plane from the plane's own `Tensor` strides (`nb[1]` token bytes, `nb[2]`/`nb[3]` head stride),
  the same fields the copier reads; never from a shared constant.
- **Do not predict physical page ids.** Derive every expectation from `allocation.page_ids()` and
  the pool's visible state, before and after.
- **The copier drains the stream it is given.** Pass the test's own stream and sync on it; the
  copies are `cudaMemcpy2DAsync` on that stream and the host read-back must order after them.
- **The mapping must not resize.** Same size as the current mapping — the copier throws otherwise.
- **A one-move page is a real case** (`source_token % 64 == 0`), and it is the *only* case in which
  the address arithmetic's second branch is skipped. Cover both.

### Commands (Windows host, from `E:\playground\ninfer-4090`)

```powershell
ninja -C build-ninja -j 32
ctest --test-dir build-ninja -R ninfer_scr_copied_relocation_test --output-on-failure
ctest --test-dir build-ninja --output-on-failure
```

`ninfer_add_op_test` sets `SKIP_RETURN_CODE 77`, so a machine without a GPU skips instead of
failing — keep the `cudaFree(nullptr)` guard at `main`'s top, as the sibling test does.

### Done means

- All seven assertion groups pass; groups 1-4, 6 and 7 run on both plane orders, and group 5 (the
  attention route) runs on PageMajor — the Ops cannot address a HeadMajor plane at all, which is a
  property of the kernels, not of the copier (see *The pool*).
- The negative control breaks the criterion on every variant that has one.
- The full suite is green (85 tests, 84 plus this one), per the repository rule.
- Only then is the ladder's misaligned-edit rung meaningful — and note it needs a *misaligned* e2e
  shape the current script cannot produce (`solve_alignment` exists to prevent exactly that). Until
  such a mode exists, this device test is the only execution the copied relocation will ever have.
