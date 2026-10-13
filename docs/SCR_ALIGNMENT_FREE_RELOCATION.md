# Dropping the source alignment requirement from SCR relocation

Segmented and tail relocation currently require a surviving run to sit on the 64-token KV page grid
in **both** prompts. That is what makes the feature fire on roughly one edit in 64: `diff_prompt_edit`
matches whole blocks by hash, so an edit whose token delta is not a multiple of 64 shifts every later
block off its boundary and leaves the detector *nothing* to match. The harnesses were built around
this: `ninfer-scr-quality` balances its edits and **throws** on a misaligned one
(`apps/scr_quality/main.cpp:858`, flagged at `:836`), and the e2e script's alignment search pads
prompts until the survivors land on the grid (`tools/scr_e2e_check.py`, `solve_alignment`).

The question this document answers: can that be removed? **Yes, half of it** — and the half that can
go is the half that matters.

## The finding, measured

`tests/test_scr_e2e_shape.py`, section *"What it would take to stop requiring the source to be page
aligned"*, ports both halves of the pair and measures them on the shape every real edit has — a run
of 320 identical tokens sitting below an inserted span of `delta` tokens:

| `delta` (tokens inserted above the run) | shipped | proposed |
|---|---|---|
| 0 | 0 (no relocation: nothing changed) | 0 |
| 1 .. 63 (not a page multiple) | **0 of 320** | 256 of 320 (mean 257.0) |
| 64, 128, ... (page multiple) | 320 of 320 | 320 of 320, and the *same span* |

So the cliff is not a tax of a few tokens: for 63 of every 64 edit lengths the shipped path moves
nothing at all. Under the proposed pair the recovery is `run − (up to 63 at each end)` for **every**
shift, and the aligned case collapses onto exactly today's span, so the existing zero-copy fast path
survives untouched.

## The insight: only the *destination* has to be page granular

A page is what the allocation, the KV entitlement, the block table, the prefill accounting and the
per-page RoPE delta plane are all expressed in, so the destination must stay whole pages, and a page
cannot hold both relocated and prefilled tokens (it could carry no single delta). **The source has no
such requirement.** Nothing in the delta plane, the attention kernels or the accounting looks at where
a destination page's payload *came from*.

The shipped design re-points a destination page at the source page already holding its content, and
that is the only reason both ends must be aligned. Replace the re-point with a **copy**, and the
source is free: a destination page's content is one contiguous source range, which reaches into at
most two source pages — so at most two device copies per page, one uniform shift for all 64 tokens.
`rope_delta_for_span` already returns exactly that shift (`dest_begin - source_begin`, constant over
the span), so the delta plane does not change shape.

The price is that the payload moves: for a relocated token that is 64 KB on `bf16` and ~16 KB on
`rk4v4-e8`, so 10k relocated tokens is ~160 MB — order of 0.2 ms of device-to-device copy on the 4090,
against seconds of re-prefill for the same tokens.

## The pieces

1. **Detector — `prompt_edit.cpp`, `diff_prompt_edit`.** Anchoring on whole blocks cannot find a
   run that sits off the grid, by construction. Anchor on a short n-gram taken at *every* offset
   (index the previous prompt once), then grow each hit to its true boundaries and confirm it by
   exact token comparison. Keep the existing invariants: monotone walk, no token reused, pairwise
   disjoint runs, `kMaxCandidateProbes` cap on repeated content, a collision can only lose a match.
   Cost stays O(previous + current) with a larger constant (the test's port uses a 16-token anchor;
   a rolling hash makes the index O(N) rather than 16N).

2. **Planner — `plan_relocations` / `plan_tail_relocation`.** Replace the block-alignment gate with
   destination quantization: `dest_begin = ceil(run.new_begin / 64) * 64`, `dest_end = dest_begin +
   floor((min(run_end, limit) - dest_begin) / 64) * 64`, drop the run if no whole page remains. The
   source offset follows the run's own shift (`old_begin + (dest_begin - new_begin)`) and is
   deliberately *not* aligned. The line-per-run loss is capped at 63 tokens at each end, so a run of
   a few hundred tokens relocates all but its fringes.

3. **Copy plan — new, beside `plan_page_remap`.** One entry per destination page:
   `{source_token, dest_token, tokens}` plus its split into at most two segments
   `{source_page, source_offset, dest_offset, tokens}`. Relocated destination pages become **acquire**
   slots (a page whose content must move can never keep its identity); prefix pages keep theirs;
   everything else acquires. `remap_pages_from_pool`'s arithmetic still prices it unchanged, because
   the acquire/keep pattern is expressed the same way.

   **Trap — aliasing.** The pages the relocation copies *from* become surplus and are handed back to
   the pool, and the acquires are filled from exactly the dropped pages first (`reassign_pages`'
   documented order). A destination page can therefore end up being the same physical page as one of
   its own copy sources, with overlapping ranges — `cudaMemcpy`/`cudaMemcpyAsync` are undefined on
   overlapping regions, and the failure is a silently half-copied page. Handle it one of two ways:
   stage the page through shared memory in the copy kernel (load both source segments, `__syncthreads()`,
   write — correct even when source and destination are the same page), or reserve the destination
   pages from the pool *before* returning the sources and refuse the relocation when that reservation
   cannot be met. Prefer the kernel: no extra VRAM, and it doubles as the per-dtype copy path.

4. **Copy kernel — `src/ops/`.** One kernel per dtype family taking the block table, so it can turn
   logical pages into physical ones itself. `bf16` is a plain contiguous move. The quantized layouts
   are **not**: a page's payload is interleaved by `(group, kv_head, page_offset)` —
   `k_scale_pages` is shaped `{groups, kPagedKVPageSize, kv_heads, physical_pages}`
   (`ops/wrapper/gqa_attention.cpp:110`) — so a token sub-range of a page is a strided move for every
   `(group, kv_head)` pair, and the scale planes travel with the codes they belong to. `rk4v4-e8` is
   an `I8` layout and needs no special case beyond the strides.

5. **Runtime — `program_impl.h`, `start_prefill_lane`.** Keep the delta write, the acquire pricing, the
   remap publish, the GDN slot invalidation and the prefill range walk. Add: enqueue the copies on the
   same stream *before* `bind_sequence_kv` (the bind publishes the mapping prefill and attention read),
   and replace the "a relocated span is not page-aligned" assertion — the invariant becomes "the
   destination is whole pages and `tokens` is a whole number of pages; the source is free". The MTP
   allocation gets the same copy plan against its own page table; DFlash stays gated as it is today.

6. **Delta helper — `src/core/paged_kv_delta_layout.h`.** The value does not change
   (`dest_begin - source_begin`, uniform per span), but the containment test does: it currently checks
   a page index against the *source* token range (its parameter is named `logical_page` and is called
   with the source page). It has to test the *destination* page against the destination range, since
   the plane is indexed by destination page. `relocated_span_is_page_aligned` splits into a
   destination-page predicate and a page-multiple-of-`tokens` predicate; `tests/test_paged_kv_delta_layout.cpp`
   pins both.

7. **Tooling that should stop fencing off the case.** `apps/scr_quality/main.cpp`'s `NOT_PAGE_ALIGNED`
   throw exists to stop a run in which SCR never engaged from being read as a passing run; once the
   gate is gone, balanced edits are no longer required and realistic prompts become measurable.
   `tools/scr_e2e_check.py`'s `check_relocation_plan` must accept an off-grid `src` while still
   demanding aligned `dst`/`tok` pages.

## The other half: freeing the destination as well

### Implementation status

The first destination-free implementation uses one packed relocated interval per logical page beside
the existing page delta. BF16 decode and prefill attention use the page delta for whole-page spans and
recompute QK scores with the shifted query only for keys inside a mixed page's interval. The resulting
scores join the existing softmax/PV update, so K/V storage and the ordinary-page route are unchanged.
Copied remaps acquire every destination page touched and move only the span's token ranges.

This path is currently enabled for BF16 KV only. INT8 and packed layouts continue to quantize the
destination to whole pages until their decode and prefill kernels implement the same boundary score
handling. Plans with multiple independent relocated intervals in one logical page are refused because
the packed table intentionally carries only one interval there. The CUDA numerical oracle covers both
aligned and mixed destination pages, but must still be run on the RTX 4090 before enabling this for
default traffic.

How to extend it to the i8 family is specified below — *Extending the boundary score path to INT8 and
the packed layouts* — including why it is bounded rather than prohibitive, the one place it is
structurally harder than the bf16 version, and a trap in the decode kernels' page handling that
reads like a bug and is not.

A copy can write anywhere, so the destination is not free for a *copy* reason: a page carries exactly
one RoPE delta, and a page holding both relocated and prefilled tokens could carry no single shift.
Freeing the destination therefore means either making the delta finer or doing without it:

1. **Sub-page deltas.** A segment granularity (say 16 tokens) or, cheaper, a breakpoint table — at
   most two boundary pages per run — that the attention kernel consults before applying the page's
   shift. The plane stays small (one int32 per 16 tokens is 8 KB over a 32k context); the cost is a
   branch per page read and a second counter-rotation of the Q rotary pair inside the boundary pages,
   in every attention kernel (bf16 and the whole i8 family).
2. **Re-rotate the fringe during the copy.** No hot-loop change, but it means decoding the codes,
   rotating and re-encoding — the quantization drift the delta plane was introduced to remove, and
   the path `gqa_kv_rerotate` was deleted with. Only the ≤126 fringe tokens of a run would pay it.
3. **Token-granular KV for the relocated region.** This throws away the paging the entitlement, the
   block table and the prefill accounting are all expressed in. Not a real option.

Measured residual cost of *not* doing any of it (`tests/test_scr_e2e_shape.py`,
`test_destination_quantization_costs_only_the_two_edge_pages`), for a run at an arbitrary offset and
every shift in `0..63`:

| surviving run | mean relocated | fraction |
|---|---|---|
| 64 | 1 | 0.02 |
| 128 | 64 | 0.50 |
| 320 | 253 | 0.79 |
| 1024 | 946 | 0.92 |
| 4096 | 3970 | 0.97 |

That is the same fringe the shipped path already trims on an *aligned* edit, so the source-side
relaxation neither adds nor removes it. For the runs that matter it is 3–8% of the run; it is fatal
only to a run shorter than about two pages (a one-page run lands nowhere with probability 0.98).
**So do the source side first** — it is the difference between 0% and ~97% — then read the surviving
run lengths off the `[scr-diag]` spans in real traffic. If they are typically far above 256 tokens,
stop here; if short survivors dominate, mechanism 1 is the one to build.

### Extending the boundary score path to INT8 and the packed layouts

BF16 is live and INT8 is deferred. This is what the deferred half costs.

**Why it is bounded rather than prohibitive.** RoPE never touches the whole head. The i8 kernels gate
the rotation on `grp == 0` (`src/ops/kernel/gqa_attention_decode_i8.cuh:431`), because group 0 is the
only one covering dims `[0, 64)` — the rotary region. With `kGqaKvQuantHeadDim = 256` and
`kGqaKvQuantGroup = 64` there are four groups (`src/ops/kernel/gqa_attention_kv_quant.cuh:21-23`), so
group 0 is 2 of the 8 `GroupKc` iterations in the score loop: **25% of the QK MMA**. The bf16 fix-up
shape therefore carries over at a quarter of its apparent cost — re-run with a shifted Q, splice the
scores per key — not at double it.

**Why it is nonetheless harder than bf16.** The bf16 path can copy its A operand into registers
(`af_q`) and rotate the copy, because that operand *is* the query. On i8 it is not: Q has already been
rotated, Hadamard-transformed, `amax`-quantized and split into codes plus per-group scales
(`gqa_attention_decode_i8.cuh:416-442`). An int8 code cannot be rotated. So the shifted variant needs
`stage_and_quantize_q` run again into a **second** staged operand with its own scales, and the MMAs re-run
against the same K. Two consequences:

- `q_scale_tmp` has no free slot. It is `reinterpret_cast<float*>(p_s)`
  (`gqa_attention_decode_i8.cuh:113`) — the Q scale array is aliased onto the P-matrix arena, and P is
  written later in the same key block. The second scale array must come from somewhere that does not
  collide, not from `p_s`.
- The score is accumulated **per group** and is live and partial throughout
  (`score[nt][x] += q_scale_r0[g] * ka * c`). A splice needs the baseline score, the baseline group-0
  score and the shifted group-0 score at the same time, which is ~32 floats over the current 16 in a
  kernel that has already had one register fight (the `q_scale_r0` comment records moving a
  `__shfl_sync` to an LDS broadcast to escape an `-rdc` ABI stack frame). **Hold those accumulators in
  shared memory, not registers** — `p_s` is idle during the score phase — and re-measure.

**A trap that reads like a bug and is not.** `issue_kv_tile` takes a single `physical_page` for the
whole tile and indexes with `key & kPagedKVPageMask` (`gqa_attention_decode_i8.cuh:340-400`), and the
advance only fires on `(k0 & 63) == 0`. With `Bc == 32` that looks like it must read the wrong page
for any tile starting at page offset 32. It cannot: `first_tile = (split_start / Bc) * Bc` is
`Bc`-aligned by construction, and both `Bc` values in use (32 and 64) divide `kPagedKVPageSize`, so
every key tile lies wholly inside one logical page — a tile from offset 32 with `Bc=32` covers offsets
32..63 and stays in its page. Simulating both the bf16 synchronous and the i8 prefetch forms of the
advance over every `(window, splits, Bc)` up to 4200 tokens gives 1.6M cases and zero mismatches.
**Do not add a per-key page lookup to fix this.** It would cost a `block_table` load and a dependent
issue per key on the hot decode path to defend against an impossible case, and on a memory-bound kernel
that is a real regression for no correctness gain.

What that invariant *does* mean is that the boundary gate needs no page arithmetic of its own: the
range plane is per logical page, the tile is inside one page, and the gate is a comparison of
`key & kPagedKVPageMask` against the packed interval — the same
`rope_delta_page_range_is_partial` predicate the bf16 kernels now use.

**Shared-memory budget.** Group 0 only needs duplicating, not the whole head, so the second operand is
`Br * kGqaKvQuantGroup` int8 plus two extra scale slots per row:

| decode config | current | +second group 0 | total |
|---|---|---|---|
| `<6,1,32,true>` | 47.7 KB | +3.4 KB | 51.1 KB |
| `<8,2,32,false>` | 52.8 KB | +4.5 KB | 57.2 KB |
| `<6,1,64,true>` | 83.2 KB | +3.4 KB | 86.6 KB |

Ceiling is the sm_89 per-block shared-memory opt-in limit. The `<6,1,64,true>` shape
(`src/ops/launcher/gqa_attention_decode.cu:179`, reached above an 8198-token implementation window) is
the one that fits least comfortably, and the first thing to check with
`cudaDevAttrMaxSharedMemoryPerBlockOptin` before starting.

**Prefill is the easier half.** `kGqaPrefillI8Bc = 64` equals the page size
(`src/ops/kernel/gqa_attention_prefill_i8.cuh:25`), so a key block is exactly one logical page and the
page question does not arise. The range table is already delivered: `gqa_attention_prompt_attention_launch`
builds `GqaPrefillDirectMetadata` with `cache.rope_delta_ranges.data` and dispatches by head count to
bf16 *or* i8 (`src/ops/launcher/gqa_attention_prefill.cu:159`), so the i8 prefill kernel receives the
pointer and ignores it. What it needs is the same two-lambda split the bf16 kernel gained: an explicit
`load_and_rotate_q(delta)` so the boundary recompute can undo its in-place rotation by an ordinary
reload at angle 0 rather than a sentinel. Its two page-transition sites are
`gqa_attention_prefill_i8.cuh:648` and `:838`, mirroring bf16's two loops.

**Gates to change, both in `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`.**
`delta_path_supports_dtype` at `:21` must stop returning true for `DType::I8` before the kernels can
consume it (it is what currently stops the planner admitting a plan it cannot execute), and the
`kv_dtype == DType::BF16` test at `:343` must widen once decode and prefill are both done. Leave the
former true and the planner will produce mixed-page spans that read wrong scores silently; leave the
latter narrow and nothing changes.

**Checks required before this half ships.** Distinct from the document-level *Verification
ladder* at the foot of this file, whose items 1-8 still apply unchanged.

1. *Compile-side, no GPU:* `nvcc --ptxas-options=-v` on the existing kernel for a baseline register and
   spill count, then again after the change. This is the only measurement that can change the design —
   if spills appear, the accumulators move to shared memory as above. Get it before writing the
   feature, not after.
2. *Host:* `tests/test_paged_kv_delta_layout.cpp` already pins the encoding and the predicate, so the
   i8 work adds nothing there.
3. *Device:* extend `tests/ops/test_gqa_kv_page_delta_i8.cpp` with the unaligned scenario the bf16
   equivalent gained — the harness, its `Int8Cache` fixture and its FP64 anchor all already exist, so
   this is a port rather than new scaffolding. Keep the negative control: zeroed deltas on the
   relocated scenario must still fail loudly.
4. *Hardware:* the numerical oracle over mixed destination pages on every quantized layout
   (`int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`), then decode throughput on the 4090 to confirm
   the 25%-extra-QK-on-rare-pages reading. Per `AGENTS.md` this is a gate on enabling the path for
   default traffic, not on building it.

Until all four are done, INT8 and the packed layouts stay on whole-page destinations, which is the
status line above. The measured cost of that status quo is the fringe table in the previous section,
so it is a known and bounded hold, not a silent one.

## What this does not fix

- **The pre-edit head is still prefilled.** When SCR fires, the plan's reuse path is a `FullReset`, so
  `reuse_base` is 0 and the region before the first survivor is recomputed even though it is identical
  and resident (`plan_relocations` skips the leading run on the assumption that prefix reuse covers
  it). Fixing that is a separate change: a token-exact prefix frontier plus a linear-state snapshot at
  it — the 48 Gated-DeltaNet layers cannot be rewound to an earlier token.
- **The relocated run stays an approximation for the linear layers.** Their state is one recurrent
  summary, so it cannot be moved with the KV; the runtime zeroes the slot and rebuilds from the head
  prefill. More alignment-free relocation does not change that.

## Verification ladder

### The device copy

1. Host: the ports in `tests/test_scr_e2e_shape.py` (detector, selector, tiling, copy segments, one
   delta per page) — already in place, no hardware needed: `python3 -m pytest
   tests/test_scr_e2e_shape.py` from the repo root.
2. Host: `tests/targets/qwen3_6/test_prompt_edit.cpp` for the detector and selector, and
   `tests/test_paged_kv_delta_layout.cpp` for the split predicates.
3. Device: `tests/ops/test_scr_copied_relocation.cpp`, specified in full by
   `docs/SCR_COPIED_RELOCATION_HANDOFF.md` — an off-grid source relocated by payload copy, held
   bit-for-bit against the pages it was copied from and read back through the public attention entry
   points, across `{bf16, rk4v4-e8} x {PageMajor, HeadMajor}`. It is a new pool-backed file, not an
   extension of `test_scr_relocation_integration.cpp`: the copier is a `PagedKVAllocation` method
   inside a real `PagedKVPool`, and the existing integration test is BF16-only with hand-built
   planes. Until it exists the copied path has zero execution anywhere: the e2e fixture aligns by
   construction, so `relocate_copied_pages` is never reached on hardware.

### The interleaved fixture, segmented against tail

The two-deletion fixture (`[P R1 S1 R2 S2] -> [P R1' S1 R2' S2]`) is implemented and host-tested; what
remains is the hardware comparison that separates segmented relocation from tail.

4. Host: `tests/test_scr_e2e_shape.py` (22 tests) already pins the interleaved fixture's shape
   against ports of the real detector, both selectors, and the plan-shape and tiling checks the
   `--scr-diag` plan is read back through. No hardware needed.
5. Hardware, on the Windows box from the repo root, capturing raw output:

   ```powershell
   python tools\scr_e2e_check.py --exe build-ninja\apps\ninfer-serve.exe `
     --base-args ".\models\qwen3_8_27b.ninfer --kv-dtype bf16 --max-context 32768" `
     --scr-mode segmented --shape interleaved --scr-diag
   ```

   Pass criteria, all of them:

   - the plan line reports **two spans and three ranges**: `span{...} span{...} ranges=3 [0,a)
     [b,c) [d,prompt)` with the tiling exact;
   - the savings line accounts exactly: `computed == prompt - relocated`, where
     `relocated == |S1| + |S2| - 64`. The tail survivor loses exactly its final 64-token block to
     the final-token rule (prompt logits need a real forward pass); the middle survivor, which ends
     at a replacement that is itself prefilled, is unaffected;
   - `RESULT: outputs are identical`, exit 0.
6. The same fixture in tail mode: it must relocate **strictly less** — tail selects one contiguous
   run, segmented relocates both survivors. Then step 5 again with `--scr-max-spans 1`: segmented
   must collapse onto tail's choice, which proves the cap applies to the interleaved plan rather
   than to a single run. And the flag control: `--scr-max-spans 0` must relocate nothing, so a
   difference between two runs is relocation and not the flags.
7. Repeat step 5 per quantized dtype (`int8`, then `rk4v4-e8`) — the relocation path is the same
   code for every dtype the planner admits. **Trap:** the `[scr-diag]` post-prefill integrity line
   is still BF16-only — it decodes the K plane as `uint16` BF16 words, so on a quantized cache it
   reports `before=nan after=nan`, `non_finite` counts and `summary{0/3 intact}` even when the
   relocation is correct. The outputs decide correctness; that line is diagnostic only and wants
   either a dtype-aware decode or an explicit "not checked for this dtype".

### Still owed: the misaligned edit end to end

8. The alignment-free claim itself has no end-to-end run, and the e2e script cannot yet produce it:
   `solve_alignment` (`tools/scr_e2e_check.py`) pads every region until the survivors land on the
   grid — the interleaved shape runs five such searches — so an edit whose token delta is not a
   multiple of 64 never reaches the server. Add a mode for the deliberately misaligned edit (and
   teach `check_relocation_plan` to accept an off-grid `src` while still demanding aligned
   `dst`/`tok` pages) before the copy is exercised through the served path. Until then, ladder
   item 3's device test is the only execution the copied relocation will ever have.
