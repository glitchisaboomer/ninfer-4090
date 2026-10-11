# SCR with quantized KV — per-page RoPE delta

Companion to `docs/SCR_ROADMAP.md` and `docs/SCR_RELOCATION_PATHS.md`.

**Status: design only. Not implemented. The delicate part (Step 4) cannot be validated by reading —
it needs a GPU run and `compute-sanitizer`.**

### What has been verified here, and how

Every claim below that could be checked without a CUDA toolchain was checked, on a machine with no
`nvcc`, no GPU and no `cmake`. The results are recorded rather than summarised, because one of them
falsified an earlier draft of this document.

| Claim | Verdict |
|---|---|
| Every `file:line` anchor | **verified**, by printing the named line of each cited file |
| Rotor pairing is `(e, e+32)`, from the kernel not the header | **verified** — `kGqaRerotateHalf = 32` (`gqa_kv_rerotate.cuh:31`), applied at `:59-68` as `sincosf(delta * freq.v[pair])` over `(pair, pair + kGqaRerotateHalf)` |
| `RerotateFreq` is a by-value 32-float struct | **verified** (`include/ninfer/ops/gqa_kv_rerotate.h`) |
| `RelocatedSpan` carries the old and new positions | **verified** — fields are `source_begin`, `dest_begin`, `tokens` (`prompt_edit.h:87-93`). An earlier draft of this doc wrote `src_begin`, which does not exist |
| `reassign_pages` rejects in four ways, all before any mutation | **verified** (`paged_kv_cache.cu:547-573`: invalid allocation, empty mapping, exceeds entitlement, names an unowned page) |
| SCR is gated to `SpeculativeBackend::None` | **was verified** (`request_plan_impl.h:259`), and is no longer true: the gate now admits `SpeculativeBackend::Mtp` as well, because the runtime relocates the backend allocation from the same spans (see Step 3). DFlash remains gated |
| YaRN only rescales above 1048576 | **verified** (`yarn.h:16-26`) |
| The three tests this plan extends all exist | **verified** |
| Delta-plane memory cost is negligible | **verified by arithmetic** — see §5, 0.5–256 KiB |
| **Steps 1–3 are host-only** | **FALSIFIED — see the box below** |

---

## 0. Why this exists

*Motivation, as of the start of this work. `SCR_RELOCATION_PATHS.md` then limited SCR to an
uncompressed BF16 Text K plane, because the one relocation primitive it had, `ops::gqa_kv_rerotate`,
required one: `validate_text_k_cache` (`src/ops/wrapper/gqa_kv_rerotate.cpp:33`) rejected every
quantized layout, and the request planner had no matching guard, so `--kv-dtype rk4v4-e8 --scr`
threw instead of degrading. All of that is now gone from the code — see Step 3's **Landed** note and
Step 5's audit status — and this section is kept as the record of why the change was made.*

The consequence, from that document's own table, is that SCR is available only where it matters least:

| kv-dtype | KiB/token | KV @ 65k | SCR |
|---|---|---|---|
| bf16 | 64 | 4.0 GiB | yes |
| int8 | 32 | 2.0 GiB | no |
| rk4v4-e8 | 16 | 1.0 GiB | no |
| rk2v4-e8 | 12 | 0.8 GiB | no |

A 24 GB card therefore supports SCR to roughly 65k context while the quantized formats are what buy
160k–500k. Since a re-prefilled suffix is charged a full weight pass on this hardware, the contexts
where re-prefill is most expensive are exactly the ones SCR cannot serve.

That document recommends implementing `int8` K re-rotation next, on the grounds that re-rotating an
`rk4v4-e8` key is a lattice-automorphism problem with no known cheap encoding. **> **Falsified: "Steps 1–3 are host-only and testable without nvcc."**
>
> An earlier draft of this document led with that, and the sequencing table scored Steps 1–3 as
> "needs GPU: no, independently verifiable: yes". It is false. `src/core/paged_kv_cache.h:6` includes
> `<cuda_runtime_api.h>` directly, so the header Step 1 modifies does not even *compile* with plain
> `g++` — tested here:
>
> ```
> src/core/paged_kv_cache.h:6:10: fatal error: cuda_runtime_api.h: No such file or directory
> ```
>
> Steps 1–3 need a **CUDA toolchain**, though not a GPU. That is a real constraint — it means the
> pool-layout work cannot be finished or tested on a machine like the one this was written on, which
> is the same limitation that made the SCR wiring itself unverifiable for a while.
>
> **The fix, and it is worth doing rather than noting:** extract the delta plane's shape and sizing
> arithmetic — which is pure integer arithmetic over `table_rows`, `logical_page_capacity` and
> `sizeof(std::int32_t)` and touches no CUDA type at all — into a CUDA-free header
> (`src/core/paged_kv_delta_layout.h`), exactly as `q5_linear_add_catalog.h` was extracted from
> `q5_linear_add_plan.cpp` (commit `20620185`) so that routing policy could be tested where there is
> no toolchain. Then the arithmetic, its validation and its invariants are testable here, and
> `paged_kv_cache.{h,cu}` keeps only the `Tensor` binding and the device calls. Step 1a below is that
> extraction, and it is the difference between this roadmap being partly verifiable off-machine and
> not at all.

This roadmap
supersedes that recommendation.** The lattice problem is only unavoidable if you insist on writing the
rotated key back into the quantized plane. You do not have to, and the roadmap's own preferred
option — §"Path A: modify attention kernels to accept a per-block `position_ids` / RoPE offset map" —
was the right idea all along; it was skipped in favour of physical re-rotation.

Implement Path A, and the quantization question disappears.

---

## 1. The mechanism

### 1.1 The problem, stated as arithmetic

RoPE is a composition of independent per-plane 2-D rotations. A key prefilled at position `n_old` is
stored as `R(n_old) · k_raw`. Relocation moves its token to `n_new`, so attention must read it as if it
had been prefilled there — as `R(n_new) · k_raw`. The difference is a rotation by

```
delta = n_new - n_old            (constant across a whole relocated span)
```

`ops::gqa_kv_rerotate`'s contract states the per-plane form it applies, and it is the authority for
the layout used here. Use the named constant, not the literal:

```cpp
inline constexpr int kGqaRerotateHalf = 32;  // rotary_dim 64, so 32 rotated element planes.
                                             // src/ops/kernel/gqa_kv_rerotate.cuh:31
```

Entry `e` of `inv_freq[32]` governs rotary elements `e` and `e + kGqaRerotateHalf`, so

```
c_e = cos(delta * inv_freq[e]);  s_e = sin(delta * inv_freq[e])
K[p][:, e]                 <- K[p][:, e] * c_e - K[p][:, e+kGqaRerotateHalf] * s_e
K[p][:, e+kGqaRerotateHalf] <- K[p][:, e+kGqaRerotateHalf] * c_e + K[p][:, e] * s_e
```

and the implementation computes it as `sincosf(delta * freq.v[pair])` over the pair
`(pair, pair + kGqaRerotateHalf)`, one warp per logical position
(`src/ops/kernel/gqa_kv_rerotate.cuh:59-68`).

Three things in that contract matter for this design:

- **The pairing is `(e, e + 32)`, not `(e, e+1)`.** A rotary pair therefore spans 32 head dimensions.
  This is the fact that decides between the two implementations below, and it is the reason a naive
  "rotate the decoded group" plan does not work: a pair is not inside an 8-dimensional E8 group.
- **The FP32 `sincosf` convention matches `rope.cuh` deliberately**, and the kernel's header says the
  pairing is the one `apply_rope_head` uses (`gqa_kv_rerotate.cuh:12`). Step 4 must match
  `apply_rope_head`'s convention for the query side too — a query rotated with a different association
  will compose incorrectly and nothing will report it as an error.
- **The table is a required argument, not module state.** `rope.cuh`'s `kTextRopeInvFrequency` is a
  `static __device__ __constant__` with internal linkage, so every TU that includes `rope.cuh` owns a
  private copy and one `cudaMemcpyToSymbol` writes only its own. The `RerotateFreq` by-value struct in
  `include/ninfer/ops/gqa_kv_rerotate.h` exists precisely to avoid that. **Reuse it; do not re-derive
  the table inside the attention kernels.**

### 1.2 Two ways to apply it

**(A) Counter-rotate the query, per page. Recommended.**

Instead of rotating the key by `R(delta)`, rotate the already-RoPE'd query tile by `R(-delta)` for the
keys of that page. The score is then the score the key would have produced had it been prefilled at
`n_new`. The kernel already resolves `block_table[...]` per page inside the K loop
(`src/ops/kernel/gqa_attention_decode_i8.cuh:403` and again at `:632`), so the extra work is one int32
load and one rarely-taken branch per 64-token page, plus one 32-plane rotation of the Q tile per
relocated page.

Why this and not (B):

- **Dtype-agnostic.** It never touches the K plane, so `int8`, `rk4v4-e8`, `rk2v4-e8` and BF16 all work
  from one change, and `ops::gqa_kv_rerotate` can eventually be deleted rather than extended per
  format.
- **Exact.** FP32 rotations on a value that was already FP32 — no re-encoding error added on top of
  the E8 approximation already present.
- **No new staging.** Nothing is written back, so no workspace, no extra memory traffic beyond the
  delta plane itself.
- **The pairing is local.** `(e, e+32)` is trivially addressable inside a contiguous Q tile. On the K
  side it straddles four 8-dimensional E8 groups, which is what makes (B) awkward below.

**(B) Rotate the decoded key to FP16 before staging. Fallback.**

`gqa_attention_decode_i8.cuh:355-372` already decodes E8 to int8 and writes it into shared memory
(`e8_root_decode_8d_int8` → `k_i8[...]`, swizzled by `gqa_small_t_tc_swz`). Injecting the rotation
between decode and the shared store is mechanically possible, but:

- the staged type is `int8`, and a rotation by an arbitrary angle is not integral, so the tile would
  have to become FP16/BF16 — more shared memory and a different MMA operand type;
- the `(e, e+32)` pairing reaches across groups, so two decoded groups must be live at once;
- the E8 decode is composed with the Hadamard undo — `if constexpr (RotateK)
  gqa_kv_hadamard64(kv0, kv1, FullMask)` at `:221` — so the rotation must go *after* it, and getting
  that order wrong produces silently wrong scores.

Reach for (B) only if the Q-side counter-rotation turns out to be blocked, which would be a finding
worth writing up. **Do not implement both.**

### 1.3 Where the delta lives

A per-`(table_row, logical_page)` int32 plane beside the block table.

**Not** per physical page. A page is recycled between sequences; a delta stored against a physical page
would be inherited by whatever request acquires it next. The block table is already per-row and
per-sequence — readers resolve it as `block_tables + table_row * table_stride`
(`src/ops/kernel/gqa_attention_decode_i8.cuh:139-140`) — which is exactly the lifetime a relocation
needs.

`RelocatedSpan` (`src/targets/qwen3_6/impl/runtime/prompt_edit.h:87-93`) already carries the span's old
and new positions — `source_begin`, `dest_begin`, `tokens` — so `delta` is a subtraction at the call
site, not new state. Note the field is `source_begin`; there is no `src_begin`.

---

## 2. Change set, in dependency order

Steps 1–3 are host-side and individually testable. Step 4 is the kernel work. Do not start Step 4
until 1–3 are green.

### Step 1a — extract the arithmetic into a CUDA-free header

`src/core/paged_kv_delta_layout.h` (new). **This step exists so that Step 1b is testable without a
CUDA toolchain**, and it is the reason the sequencing table below no longer claims Steps 1–3 are
host-only as written.

Put in it, as `constexpr`/`inline` functions over plain integers:

- `rope_delta_logical_pages(max_context)` = `ceil(max_context / kPagedKVPageSize)`;
- `rope_delta_plane_bytes(table_rows, logical_pages)` = `table_rows * logical_pages * sizeof(int32_t)`;
- `validate_rope_delta_layout(logical_pages, table_rows)` — the same pair of checks
  `PagedKVPool::PagedKVPool` performs on the block table (`paged_kv_cache.cu:99-103`): `dtype == I32`,
  `ne[0] == logical_page_capacity`, `ne[1] == table_rows`, plus the delta plane's own rule that it be
  no smaller than the block table it parallels;
- `rope_delta_for_span(source_begin, dest_begin, logical_page)` — the per-page delta of one
  `RelocatedSpan`, with the span's own boundary rules (a page partially covered by a span must not
  receive a delta; spans are 64-aligned, so assert it).

Keep this header free of `Tensor`, `cuda_runtime_api.h` and anything else CUDA-shaped, the way
`q5_linear_add_catalog.h` is. Then register a host-only CTest through `ninfer_add_host_test` — the
function added with that commit — so it runs on a machine with no toolchain.

**Already done, and verified here.** Step 1a is implemented as `src/core/paged_kv_delta_layout.h`
with `tests/test_paged_kv_delta_layout.cpp`, registered through `ninfer_add_host_test`. It builds
clean under `-Wall -Wextra` with plain `g++` and passes:

- page rounding closed against `ceil(len/64)` for every length in `[1, 320]`, with the 64 / 65 /
  128 boundaries pinned explicitly;
- the `0.5 KiB / 16 KiB / 128 KiB / 256 KiB` figures from §5, computed from the code's own constants;
- every rejection path: non-positive extents, a plane that understates the block table beside it, and
  an int32 overflow;
- the per-page delta rule across a real span — inside pages, pages just outside, and the half-page case
  that must take no delta.

Two bugs in this arithmetic were found by that test *before* it was committed, which is the argument
for the extraction rather than for the test alone:

1. `relocated_span_is_page_aligned(0, 0)` returned **true**, because `0 % 64 == 0`. An empty span
   relocates nothing, so accepting it would let a caller believe it had. Now rejected.
2. The first draft of the test's int32 case used `0x7ffffff0`, which is *inside* the range — the
   unrepresentable case was not actually reached. Now pins both bounds.

Negative controls, so the assertions are known to bite: dropping the *upper-bound* check on the
span-to-page containment fails **4 runtime checks** naming the pages that wrongly take a delta, and an
off-by-one in the page rounding fails a `static_assert` at compile time. `kPagedKVPageSize` was moved
into this header from `paged_kv_cache.h` and stays in `namespace ninfer`, where it lived, because
`layouts_impl.h`, `request_plan_impl.h` and `dflash_context_impl.h` all name it unqualified and a
nested namespace would have broken every one of them — caught by reading, since the CUDA side cannot
be compiled here.

### Step 1b — bind the plane in the pool

`src/core/paged_kv_cache.h` / `.cu`. **Needs a CUDA toolchain, not a GPU.**

1. Add to `PagedKVPoolSpec` and `PagedKVPoolLayout` a `rope_deltas` region, sized from Step 1a's
   helper so the arithmetic has one definition. Model it on the existing `block_tables` region and its
   `Tensor` binding in `PagedKVPool::PagedKVPool` (`paged_kv_cache.cu:92-153`), which validates
   `dtype == I32` and `ne[0] == logical_page_capacity && ne[1] == table_rows` — the delta plane must
   carry the identical shape and route its validation through Step 1a's helper rather than a second
   hand-written copy of the same three comparisons.
2. Expose `Tensor rope_delta_row(std::int32_t row)`, mirroring
   `PagedKVPool::block_table_row` (`paged_kv_cache.h:123`).
3. Extend `total_storage_bytes()`/`metadata_bytes()` so the plane is accounted for.

**Invariant to assert here:** the delta plane is at least as large as the block table and both are
inside the arena. A layout that fits one and not the other is a silent OOM later.

**Verify:** extend `tests/test_paged_kv_reassign_pages.cpp` with a shape check. Needs a build.

### Step 2 — publish the two planes atomically

`src/core/paged_kv_cache.cu`

`reassign_pages` (`:547`), `publish_mapping` (`:636`) and the `publish_range` helper
(`:641-649`) currently write only `[0, mapped_page_count)` of the block-table row and leave the rest
untouched forever. The relocation branch in `program_impl.h` calls `reassign_pages` **before**
`bind_sequence_kv`, so the remap is what publishes the row the attention kernels read.

1. Extend the relocation variant to write the block table **and** the delta plane from the same
   `cudaStream_t`, so no kernel can ever observe one without the other.
2. Non-relocated pages must get `0`. Do this explicitly; do not assume a cleared plane, because a row is
   re-acquired across requests.
3. Add the two-planes-agree assertion that `advance_prefill` already performs for
   `schedule_state.text_kv_base` vs `staged.cursor`: same pattern, same rationale — two copies of one
   position, guarded by a throw rather than by reading whichever is convenient.
4. Clear the row on `acquire_row` so a recycled row cannot inherit a previous request's deltas.

**Invariant:** `delta[row][p] != 0` iff logical page `p` of row `row` was relocated, and `delta` is
constant across the pages of one relocated span.

**Verify:** reuse the reassign-pages oracle test's shape — permutation, grow, shrink,
shrink-one-region-while-growing-another, and the four rejections that must leave the mapping
untouched — adding, for each, "and the delta plane matches the mapping".

### Step 3 — write deltas instead of re-rotating (runtime)

`src/targets/qwen3_6/impl/runtime/program_impl.h`

The `SuffixRelocation` branch (`:754`, and the `apply_relocation` decision at `:489-493`) currently
loops over full-attention layers calling `gqa_kv_rerotate` (see `:925`, which already passes
`text_rope_inv_freq` — the table is in scope at this call site, which is why Step 4 needs no new
plumbing for it).

1. Replace that loop with a per-page delta upload: for each `RelocatedSpan`, compute
   `delta = span.dest_begin - span.source_begin` once, then write it to the delta plane for every logical
   page the span owns.
2. Keep `gqa_kv_rerotate` callable until Step 4 lands, then delete it and its wrapper validation — two
   mechanisms for one job is how the BF16-only restriction survived.
3. Remove the BF16-only gate: `validate_text_k_cache` and add the planner-side guard the handoff notes
   is missing (`request_plan_impl.h:254` does not check the KV dtype), so a quantized `--kv-dtype`
   plans a relocation it can honour instead of one that throws.
   **Landed.** `validate_text_k_cache` and `ops::gqa_kv_rerotate` are gone, the runtime writes the
   delta plane per span (`program_impl.h`, `rope_delta_for_span`) with no dtype condition, and
   `request_plan_impl.h`'s `delta_path_supports_dtype` admits `BF16` and `I8` — which is every
   quantized layout. `tools/scr_e2e_check.py` carried the stale BF16-only claim and now accepts
   `bf16 / int8 / rk8v4 / rk4v4 / rk4v4-e8` and refuses `rk2v4-e8`; the per-dtype runs are owed.

   **Also landed — the second allocation (MTP).** Text is not the only KV a served run reads. An MTP
   run keeps a second allocation in its own pool (`sequence.kv->backend`, `decoder->mtp_cache()`),
   whose logical pages cover the same tokens as Text's while its physical pages are numbered
   independently. `start_prefill_lane` now derives a second `plan_page_remap` from the *same* spans,
   against that allocation's own page table, and publishes the **same** `deltas` vector — the shift
   is a function of the token positions, and MTP logical page `p` covers exactly the tokens Text
   logical page `p` covers. The planner admits `SpeculativeBackend::Mtp` and asserts the staged MTP
   prefill that extends the relocated prefix over the two prefilled ranges. **Nothing in `src/ops/`
   changed**: MTP attention is the same public `ops::gqa_attention` entry the A1 case already covers,
   reached through `batch_layer_view`/`layer_view`, and both attach the pool's own delta row
   (`decoder_state.cpp:145,161`). DFlash stays gated (its context frontier is carried separately and
   its cache is filled through `kv_cache_append_prefix`). **Owed:** the two-block relocation test for
   the second allocation (`tests/targets/qwen3_6/test_prompt_edit.cpp`,
   `test_plan_page_remap_second_allocation` checks the mapping half on the host; the page-level half
   and the draft-acceptance comparison need hardware).

**Verify:** the existing `tests/ops/test_scr_relocation_integration.cpp` routes writes through
`gqa_kv_append` and reads through the real attention kernels; run it once per KV dtype, not just BF16.

### Step 4 — honour the delta at attention time (kernel work)

`src/ops/kernel/gqa_attention_decode_i8.cuh`, `gqa_attention_prefill_i8.cuh`,
`gqa_attention_decode_bf16.cuh`, and their launchers

1. Add two parameters to every affected kernel: `const std::int32_t* rope_page_deltas` and
   `RerotateFreq freq`, passed by value as the rerotate Op already does. Thread them through the
   launcher (`src/ops/launcher/gqa_attention_decode.cu`) and through the runtime call sites that
   already hold `ProgramImplCore::text_rope_inv_freq` (`program.h:460`).
2. Beside each `block_table[...]` read inside the K loop (`:403`, `:632`), read
   `rope_page_deltas[logical_page]`. Zero is the overwhelmingly common case and must cost one
   predictable branch, not a divergent one — hoist it to page granularity, which the loop already has.
3. On a nonzero delta, apply `R(-delta)` to the Q tile **once for that page** before its MMAs: for each
   plane `e` in `[0,32)`, rotate the pair `(e, e+32)` with `c_e`, `s_e` from `freq`. Use the query's
   position from the kernel's existing `pos` parameter to confirm the sign, and assert the convention
   against the rerotate oracle rather than trusting the algebra.
4. Leave the K path completely untouched. This is the whole point.

**Ordering trap:** if you end up in Step 1.2 (B), the rotation goes after the Hadamard undo at `:221`,
on the reconstructed vector. Under (A) it is irrelevant — say so in a comment so the next reader does
not go looking for an interaction that does not exist.

**Verify:** the oracle test in §3, plus `compute-sanitizer --tool racecheck` and `--tool initcheck`,
because the delta plane is a new read that most readers do not yet know about.

**Phase 4a Completion Gate (Must pass before starting 4b):**
1. **Spill Check:** Run `ptxas -v` on `gqa_attention_prefill_bf16_kernel`. The nested `cp.async` Q reload inside the phase loop must not push register pressure past the spilling threshold. If it spills, the common case (no relocation) regresses to pay for the rare case.
2. **Regression Gate:** Run `test_scr_relocation_integration`. It must pass unchanged. This proves the `nullptr` delta path is perfectly neutral and introduces no silent data corruption.

**Phase 4b Execution Sequence:**
1. Write `test_gqa_kv_page_delta.cpp` with the strict non-zero delta scenario described in §3.
2. Run it against the current unwired (`nullptr`) launchers. **It must fail with massive L2 error.**
3. Wire the runtime delta plane (e.g., `cache.rope_deltas().data`) into the launchers in `gqa_attention_decode.cu` and `gqa_attention_prefill.cu`.
4. Run the oracle test again. **It must now pass.**

### Step 5 — readers that are not decode attention

Anything that reads the K plane must honour the delta or it will produce silently wrong scores with no
type error to catch it:

- prefill attention (`gqa_attention_prefill_bf16.cuh`, `gqa_attention_prefill_i8.cuh`
  `gqa_attention_prefill_common.cuh`);
- the MTP attention path, which a relocated sequence now reaches: it is the same public
  `ops::gqa_attention` entry the A1 case covers, and its views attach the pool's own delta row, so it
  counter-rotates per key page exactly like Text. What no test here covers is the *pairing* — that the
  Text and backend allocations were relocated from the same spans;
- the DFlash attention path, still unreachable: SCR requires `SpeculativeBackend::None` or `Mtp`;
- `prepare_ragged_prefix` / `kv_cache_append_prefix`, which read previously written K;
- any diagnostic that fingerprints the K plane — reusing the existing `[scr-diag]` page fingerprints is
  the right instinct, but they must be taken *after* the operations that consume the pages, which is
  the lesson `SCR_PHASE_5_5_HANDOFF.md` §4 records.

**Audit status.** Every reader above was checked against the code rather than assumed:

| reader | state |
|---|---|
| decode attention, i8 and bf16 | honours the delta, and the delta anchor equals the reference anchor to 1e-5 |
| prefill attention, i8 and bf16 | honours it. The A1 oracle proves it rather than inspecting it: there the relocated page carries `delta = -384` while the *query* page carries none, so the check can only pass if the delta is read per key page |
| MTP attention | reachable since the second-allocation relocation landed, and honours the delta for free (same Op, delta row attached by the view). **No cell in the ladder runs MTP with SCR**, so this is reasoned from the call shape, not measured |
| DFlash attention | unreachable — the planner requires `SpeculativeBackend::None` or `SpeculativeBackend::Mtp` (`request_plan_impl.h`) |
| `prepare_ragged_prefix`, `kv_cache_append_prefix` | DFlash-only call sites (`dflash_impl.h`), so unreachable with a relocated page for the same reason |
| `[scr-diag]` fingerprints | present; the ordering rule above still needs confirming against the emit sites |

So the wiring this step asks for is done by the earlier steps, and the speculative-backend gate is no
longer the mechanism that keeps a relocated sequence away from a second cache: the runtime derives both
mappings from the same spans and throws if the second cannot be expressed, rather than relying on the
planner to have excluded it. What is left is hardening — the fingerprints' ordering — and measurement:
the MTP e2e comparison below.

---

## 3. Test plan

The precedent is already in the repository and should be copied rather than reinvented:
`tests/targets/qwen3_6/test_gqa_kv_rerotate.cpp` compares a re-rotated element against a **direct
prefill at the destination**, with the default table and a caller-installed scaled table, and reports
errors in BF16 ulp of the operand pair because a rotation can cancel toward zero. Its sibling
`tests/ops/test_scr_relocation_integration.cpp` routes every byte through `gqa_kv_append` and every
read through the production attention kernels, against a permuted block table, and carries a negative
control in which skipping the re-rotation registers 2.3e-1 relative L2 versus 7.2e-4 when done.

1. **`tests/ops/test_gqa_kv_page_delta.cpp`** — the new oracle. Seed a K plane, publish a block table
   with a nonzero delta on some pages, run the real decode and prefill kernels, and compare against a
   direct prefill at the destination. Negative control: delta ignored by the kernel must show a large
   error, or the test is blind. Run it for **every dtype**, with `rk4v4-e8` first — that is the
   configuration this change exists for and the one nobody has ever executed.

**The Phase 4b Oracle Test Trap:**
Because the unwired Phase 4a path passes `nullptr` for the delta plane, and `nullptr` is the mathematically correct behavior for a *non-relocated* sequence, a naive oracle test might accidentally pass against the unwired kernel. The test scenario *must* explicitly construct a sequence that requires a non-zero delta on at least one page (e.g., a relocated span). The test must assert that the L2 error against the FP64 oracle exceeds a threshold that a working null path cannot reach. If the test does not fail loudly and unambiguously on the unwired `nullptr` path, the harness is blind and will silently accept a broken wiring in Phase 4b.
2. **Delta-plane lifecycle test.** Acquire a row, relocate, release, re-acquire, and assert the row's
   deltas are zero before anything writes them. This is the recycling trap, and it is the failure mode
   most likely to ship silently.
3. **Pricing rule for the tolerance.** The existing E8 path is already an approximation by
   documentation, not by accident: `gqa_attention_decode_i8.cuh:265-272` records that the i4/int8 codes
   cannot hold the `D8+0.5` coset bit and that the half is collapsed by `rintf()` deliberately. So do
   **not** measure the delta path against the current output and call agreement success. Measure it
   against the FP32/FP64 oracle, and state the allowance in ulp of the operand pair.
4. **End-to-end** — extend `tools/scr_e2e_check.py` to run once per KV dtype and refuse to report a
   pass unless the server logs `reuse=suffix_relocation` *and* the KV dtype is the quantized one under
   test.

---

## 4. Invariants to enforce in code, not in prose

Each of these has already cost this project time once. Put them behind a throw.

1. `text_kv_base` vs `staged.cursor` already has one; give the block table and the delta plane the
   same treatment.
2. `delta[row][p] == 0` for every page of a page range that was not relocated.
3. `delta` is constant across the pages of one relocated span, and equals
   `span.dest_begin - span.src_begin`.
4. Row acquire clears the delta row.
5. The `inv_freq` used by the attention kernels is the same array the forward RoPE used — not
   `rope.cuh`'s compile-time default. This is the trap `SCR_RELOCATION_PATHS.md` already documents for
   the rerotate Op: it was latent because `compute_yarn_config` only scales above the native context,
   but a >1 M-token configuration would have rotated with the wrong angles.
6. A reader that resolves a page must read that page's delta in the same expression where it reads the
   block table. Two lookups at different points in the loop can disagree.

---

## 5. Sequencing, effort, and what it unlocks

| Step | Needs GPU? | Needs CUDA toolchain? | Effort | Independently verifiable |
|---|---|---|---|---|
| 1a extract arithmetic | no | **no** — plain `g++` | small | yes, here |
| 1b bind plane in pool | no | yes | small | on the build box |
| 2 atomic publish | no | yes | small | on the build box |
| 3 write deltas, drop BF16 gate | no | yes | small | on the build box |
| 4 attention kernels | **yes** | yes | **the real work** | only on device |
| 5 remaining readers | yes | yes | medium | only on device |

Step 1a is deliberately chosen so that *something* in this roadmap is testable without a toolchain —
the arithmetic, the validation and the per-page delta rule, which is where an off-by-one in `ceil` or
in a span boundary would otherwise hide until a GPU run. Everything from 1b onward needs a CUDA build
but not a GPU, so it can be finished in parallel with getting machine time. Step 4 changes the
quantized attention inner loop, which is the most delicate arithmetic in the engine; budget it in days
and expect `compute-sanitizer` to find something.

What it buys, in order of value:

- SCR becomes usable with `rk4v4-e8`, which is the serving configuration, removing the ~65k context
  ceiling and putting the feature in the regime where re-prefill actually costs something.
- `ops::gqa_kv_rerotate` and its BF16-only validation are deleted, taking the planner guard gap with
  them.
- The relocation stops writing to the cache at all, so the CUDA-graph address-stability argument and the
  `prepare_graphs` page-zeroing logic get simpler rather than more complex.

### The delta plane's cost, computed rather than asserted

From the code's own constants — `kPagedKVPageSize = 64` (`paged_kv_cache.h:15`), `logical_page_capacity
= ceil(max_context / 64)` (`layouts_impl.h:98`), `table_rows = max_concurrency` in `[1, 8]`
(`layouts_impl.h:169`, `types.h:79`) — the plane is `table_rows x logical_pages x 4` bytes:

| max_context | concurrency | logical pages | delta plane |
|---|---|---|---|
| 8192 | 1 | 128 | 0.5 KiB |
| 32768 | 4 | 512 | 8 KiB |
| 65536 | 4 | 1024 | 16 KiB |
| 131072 | 4 | 2048 | 32 KiB |
| 262144 | 8 | 4096 | 128 KiB |
| 524288 | 8 | 8192 | 256 KiB |

0.0031% of the BF16 Text KV cache at 65k. So the memory cost is settled; the remaining cost is one
int32 load beside the block-table load that already happens per page, and that one is worth a
benchmark rather than an argument.

---

## 6. Amendments owed when this lands

- `docs/SCR_RELOCATION_PATHS.md`: the "Recommended order: (1) planner guard, (2) int8 K re-rotation,
  (3) E8 work only if (2) pays off" section is superseded by this roadmap; §"SCR is currently limited
  to BF16 KV" and its memory table should be rewritten rather than left to contradict the code.
- `docs/SCR_ROADMAP.md` §3.4 records Path A as the target and Path B as what shipped. Mark Path A
  implemented and delete the B-era caveats.
- `docs/ISSUE_prefill-nondeterminism.md` is unaffected, but the prefill non-determinism must be fixed
  before any quality measurement taken through a quantized path can be attributed — the two work
  streams are independent and both are prerequisites for a real Phase 5.5 number.
