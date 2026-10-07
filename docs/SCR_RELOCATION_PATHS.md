# SCR relocation paths

Companion to `docs/SCR_ROADMAP.md`. That document describes the phases; this one pins down the two
concrete ways the runtime can apply a relocation plan, exactly where each attaches, and what has to
be decided before either is written.

Status: **Path A is written but has never been compiled.** `start_prefill_lane` replaces the common
frontier trim with the page remap, re-rotates the relocated run once the block table is published, and
stages the plan's two prefilled ranges, which the chunk walk skips the relocated run between. The
whole change set was prepared without a CUDA toolchain, so none of that is verified. Path B is still
unimplemented, and any plan without ranges falls back to ordinary prefix reuse.

The two open design questions are now settled: Path A is the target, and the 27B's linear layers are
handled by snapshot/restore. What remains is runtime wiring, which needs a machine with `nvcc` - see
"Verification that is still owed".

Already built and pushed:

- planning: `plan_relocations` in `src/targets/qwen3_6/impl/runtime/prompt_edit.{h,cpp}`, selecting
  block-aligned surviving spans and dropping any destination ordinary prefix reuse already covers.
  `RequestPlanImpl::relocated_spans` carries them.
- page remap: `PagedKVAllocation::reassign_pages` (`src/core/paged_kv_cache.cu`). Re-points logical
  pages at surviving physical pages with no KV copy and no second allocator. Surplus pages are
  reused as the newly acquired pages, so a remap that shrinks one region while growing another needs
  no free pool pages for the growth, and pages are acquired before anything is released.
- rotary adjustment: `gqa_kv_rerotate` (`include/ninfer/ops/gqa_kv_rerotate.h`). Re-rotates already
  cached Text K in place by `new_position - old_position`. Text geometry only.

## The problem both paths have to solve

Cached K is rotated by the forward RoPE at the position a token occupied when it was prefilled, and
a relocated token carries the rotation of its *old* position. Two consequences:

1. The relocation is only valid if the runtime also re-rotates K for every relocated position. This
   is why `gqa_kv_rerotate` exists and why it must run after the block table is republished (it
   resolves a page through the block table at the *new* position).
2. The runtime must not trim the resident pages above the reuse frontier, because the pages holding
   the relocated tail live above it. Today every reuse branch calls `trim_sequence_kv(sequence,
   base, ...)` before proceeding, which would drop exactly the pages a relocation needs.

And the structural problem: `start_prefill_lane` produces **one** staged prefill whose token range
is contiguous, `[reuse_base, prompt_tokens)`, subsequently walked by a cursor. Relocation removes a
run of tokens from the middle of that range, so the range stops being contiguous. Everything below
is about how each path handles that.

### Block alignment makes this unavoidable, even for reasoning stripping

Surviving spans are 64-token aligned (they hash whole `kPromptEditBlockTokens` blocks, matching
`kPagedKVPageSize`). Neither `reuse_base` (an execution frontier, an arbitrary token count) nor
`prompt_tokens - 1` is generally block aligned. So a relocated region is
`[B, L)` with `B`, `L` 64-aligned, and the tokens in `[reuse_base, B)` and `[L, prompt_tokens)`
still have to be prefilled. That is **two** prefilled ranges, not one, as soon as either boundary
falls mid-block.

This is the single most important thing to internalise: "relocate the suffix" does not reduce to
"prefill the head and stop". Even the paper's 68%-of-benefit case (reasoning-token stripping,
`[P][R][A] -> [P][A]`) has a mid-block boundary at the `P`/`A` junction unless `|P|` happens to be a
multiple of 64.

## Path A - tail relocation (Scope A, reasoning-token stripping)

**Applies when** the relocated spans form a single contiguous run, i.e. for chosen spans sorted by
destination, consecutive spans are adjacent (`next.dest_begin == previous.dest_end`). Compare Path B,
which allows several disjoint runs.

The run does *not* have to reach the end of the prompt, and it does not have to start at the reuse
frontier. Whatever is left over simply gets prefilled, which makes the prefilled set

```
  [prefix_reuse_tokens, run.dest_begin)  union  [run.dest_end, prompt_tokens)
```

and that is **two ranges, not one** whenever either boundary falls mid-block. Since spans are 64
block aligned while `prefix_reuse_tokens` and `prompt_tokens` generally are not, the two-range shape
is the normal case, not the exception. Do not plan around a single prefilled range: the A1 case
(the run happens to end on the last block and the frontier happens to be block aligned) is a special
case worth measuring but not worth designing for.

The final position is still always prefilled, for the reason above: prompt logits can only come from
a real forward pass, because only KV is cached and never the hidden state. So `prompt_tokens - 1` is
always inside the trailing range.

### Why not relocate to the very end

Relocation works at page granularity, and a page holds 64 tokens whose KV must stay contiguous at
their logical positions. Relocating a partial trailing block would leave a page holding both
relocated and prefilled tokens, which the page remap cannot express. So the relocated run always ends
on a 64 boundary at or below `prompt_tokens`, and the leftover block is prefilled.

### Step list

In `start_prefill_lane`, on `ReusePath::SuffixRelocation` with `relocated_spans` non-empty:

### The relocation is not a reuse branch

A first reading suggests this belongs in the `AppendAtFrontier` reuse branch, beside the other reuse
paths. It does not, because `start_prefill_lane` runs this for *every* path once the branches are done:

```
  trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
  bind_sequence_kv(sequence);
  materialize_sequence_kv(sequence, prompt_tokens, backend_materialized);
```

That `trim_sequence_kv(sequence, base, ...)` keeps `base` tokens, so it discards exactly the pages above
the frontier - which is where the whole surviving run lives. The relocation therefore has to *replace*
that trim, not sit before it, and `reassign_pages` has to run before `bind_sequence_kv` so binding
publishes the remapped block-table row. The rotary fix-up has to run after that publish, because the Op
resolves a page through the block table at the new position.

One thing that does *not* need changing: the ledger. The same common block assigns
`prompt.token_ids` to `sequence.ledger` after every path, so the relocated tokens are already recorded
and no path sets it explicitly.

The plan needs to carry the two prefilled boundaries, not just the spans: `head_begin`
(= `prefix_reuse_tokens`), `head_end` (= the run's first destination), `tail_begin` (= the run's
last destination end) and `tail_end` (= `prompt_tokens`). `RequestPlanImpl` has the spans today and
would gain the boundaries.

1. Instead of `trim_sequence_kv(sequence, base, ...)`, compute the pages the emitted sequence still
   needs and call `reassign_pages` with a mapping whose relocated slots name the surviving pages and
   whose replaced slots are `kPagedKVAquirePage`. This both keeps the surviving pages and publishes
   the new block-table row, which step 2 depends on. Note the pages above the frontier are exactly
   the ones the existing branches trim away, so the relocation branch must be taken *instead of*
   that trim, not after it.
2. For every Text full-attention layer, call `gqa_kv_rerotate(positions, rope_deltas, layer_view,
   stream)` with `positions` = the new logical positions of the relocated tokens and
   `rope_deltas` = `new_position - old_position` (negative, for a left shift). One call per layer,
   because the op takes a single layer's `PagedKVLayerView`.
3. Leave the linear state alone (see above). Do not let the prefilled ranges rebuild it, and do not
   copy a slot into it. If the sequence's linear state currently represents the pre-edit context,
   which it does, retaining it is the whole action.
4. Prefill the head range, then the trailing range. Two ranges, so the staged prefill and the
   chunking walk need at least the ability to run a second range after the first completes.
5. Report `relocated_prompt_tokens` and set `prefix_reuse_path` to `SuffixRelocation`.

### Linear-attention layers (decided: snapshot/restore)

16 of the 27B hybrid's 84 layers are full attention; the other 48 are Gated DeltaNet, whose state is
a single recurrent summary rather than per-token KV. Relocation as described only repairs the full
attention layers, so something has to be decided for the linear ones.

**Decision: keep the pre-edit linear state for the relocated suffix** (snapshot/restore), matching
the paper's "linear layers snapshot recurrent state, not per-token KV relocation". The suffix is not
re-processed by the linear layers at all.

Why not the alternatives:

- *Advancing the state over the tail* looks faithful but is self-defeating: a Gated DeltaNet state
  updates from each token's residual stream at that layer, and that residual stream comes from the
  preceding layers including the full-attention ones. Advancing over the tail therefore requires the
  tail's attention outputs over the full context, which is exactly the prefill work SCR exists to
  skip. It would leave the hybrid saving essentially nothing.
- *Caching per-token linear-layer inputs* would make advancing cheap and faithful, but it is a new
  per-token memory feature across all 48 linear layers and is out of scope here.

Mechanically this is nearly free, because the machinery already exists: `LinearStateSlots` plus
`decoder->linear_attention.copy_slot(...)`, which the turn-checkpoint restore path already uses to
save and restore this exact state. Path A must simply not disturb the linear state across the
relocation, instead of leaving it to be rebuilt by the prefilled ranges.

The resulting state corresponds to the pre-edit context, which is the SCR approximation for this
target family. Measuring it is the point of Phase 5.5; if the divergence exceeds the 0.05-nat
threshold, the fallback is to disable SCR rather than to silently keep an unmeasured state.

## Path B - general segmented prefill

Needed when relocated spans are interleaved anywhere in the prompt, i.e. the general CLM context
edit. The planner already expresses this; only the runtime cannot consume it.

What has to change:

- the staged prefill carries a **list of segments** rather than one contiguous range, where each
  segment is `[begin, end)` of tokens to prefill and the gaps are relocated;
- the scheduler's chunking walks segments in order and carries a `(segment_index, offset)` cursor
  instead of a single token cursor, so a chunk never straddles a segment boundary arbitrarily - it
  either stops at the boundary or spans it with a known irregularity;
- `sequence.text_kv_valid` becomes "KV is valid for every position in `[0, prompt_tokens)`"
  immediately after relocation, which is true, rather than "prefilled prefix length". Several checks
  compare `text_kv_valid` against the frontier and would otherwise fire spuriously;
- `sequence.ledger` has to hold the full new prompt from the start, since relocated positions are
  never re-appended during prefill;
- the block table must map the whole prompt before any segment runs, for the same reason as step 1
  of Path A.

Anchor points for that work, all in `src/targets/qwen3_6/impl/runtime/`:

- `program_impl.h`, `start_prefill_lane` - the reuse branches around the `AppendAtFrontier` case, and
  the staged `Prefill` construction at the end of the function (`host_input_consumed_pending`,
  `.reuse`, `.mtp_bridge`);
- `program.h` - the staged prefill record (`PrefillStepResult`'s cursor fields) and
  `RequestPlanImpl::relocated_spans`;
- the chunking walk further down `program_impl.h` that consumes the staged prefill
  (`host_input_consumed` / cursor bookkeeping around the `PrefillStepResult` staging and the
  `text_kv_valid` updates);
- `request_plan_impl.h`, `projected_service_work` - a segmented prefill changes the quantum
  accounting because a chunk can now cover less than `prefill_chunk` tokens.

Path B dominates the diff size and cannot be validated by reading; it should be written on a machine
with `nvcc` available.

## Switch for benchmarking

Wired: the choice reaches the plan through `ScrRelocationMode` on `ExecutionOptions`, mirrored into
`ResolvedExecutionOptions`, `ServeOptions` and `RequestBasePlanImpl`, and surfaced as `--scr-mode
tail|segmented`. The default is `tail`, and `tail` is the only mode the runtime can act on, so
`segmented` currently produces a plan the runtime ignores exactly as `tail` does. The remaining items
below are what still has to happen:

- have the planner reject a plan whose `relocated_spans` do not satisfy the selected mode's
  condition, falling back to `AppendAtFrontier` rather than relocating partially. `tail` already gets
  this from `plan_tail_relocation`; `segmented` has no such check yet;
- report the mode in the server-start record and `prefix_relocated_tokens` in the completion record,
  which is already wired.

The planner-side mode check is host-testable (`tests/targets/qwen3_6/test_prompt_edit.cpp` already
covers `plan_relocations`), so the mode split can be verified without CUDA even though the two
runtimes cannot.

## SCR produces garbage on both supported models: the GDN layers make it unworkable

This is the finding that ends the current phase. With the planner fixed (see below), SCR engages and
reports `reuse=suffix_relocation`, the relocated run is planned correctly, `reassign_pages` and
`gqa_kv_rerotate` run, and the output is degenerate: the model emits an empty `content`, loops the
maintenance log verbatim in `reasoning_content`, and runs to the `max_tokens` ceiling (observed:
10240 tokens) instead of the one-sentence answer the reference path produces in 81.

The cause is that the relocation branch in `program_impl.h` never touches
`decoder->linear_attention`. Compare the two branches: the turn-checkpoint path explicitly calls
`decoder->linear_attention.copy_slot(...)` to restore the recurrent state, and the relocation path
has zero references to it.

| model | full-attention layers | GDN layers |
|---|---|---|
| qwen3.8-27b | 16 | **48** |
| qwen3.6-35b-a3b | 10 | **30** |

So 48 of 64 layers (75%) keep a recurrent state that was summarised over the *old* prompt, including
the deleted middle, while the prompt no longer contains it. The 16 full-attention layers are
relocated correctly, which is why the plumbing is provably sound and the output is still garbage.

**This is architectural, not an implementation defect.** A linear-attention state is a running
recurrence: after tokens `[0, N)` it holds a sum over `t` of decayed `k_t v_t^T` products. Deleting a
middle segment changes the decay products of every later token, and the `k_t`/`v_t` themselves were
produced by hidden states computed *with* that middle present. There is therefore no way to derive
the post-deletion state from the pre-deletion one; recovering it means replaying the recurrence over
the surviving tokens, which is precisely the forward pass SCR exists to avoid. Page remapping and
RoPE re-rotation work because KV is per-token and position-addressed. Recurrent state is neither.

Neither supported model is a pure-attention transformer, so there is no configuration where this
path is correct today. Until a mechanism exists for relocating or invalidating recurrent state,
`--scr` should be treated as unimplemented for these architectures rather than experimental.

### Two blockers fixed along the way

**The planner made SCR unreachable.** The gate was `!base->turn_rewrite_boundary`, but
`chat_template.cpp` always records a boundary when `add_generation_prompt` is set, that option
defaults to true, and `translate.cpp` sets it true for every chat request. Nothing sets it false, so
the gate was always false and the whole path -- `diff_prompt_edit`, `plan_tail_relocation`,
`reassign_pages`, `gqa_kv_rerotate`, the prefill-lane wiring -- never executed, despite every
component being proven correct by unit tests. The boundary marks where the assistant's reasoning is
rewritten, so it bounds how far a relocated span may reach rather than forbidding relocation.
`plan_tail_relocation` now takes it as an optional parameter and trims a span back to the last whole
block ending before it. Before the fix every e2e run reported `full_reset`; after it, relocation is
planned and executed.

**The planner has no guard for quantized KV.** `gqa_kv_rerotate` requires an uncompressed BF16 Text
KV plane, so a quantized `--kv-dtype` plans a relocation that then throws at runtime instead of
degrading to `AppendAtFrontier`. This is the cheapest outstanding fix and is not done.

## Verification status

Verified on an RTX 4090 (`sm_89`, CUDA 13.x, MSVC 2022, sm_89 build, 88-test suite green):

1. `reassign_pages` against the real pool -- **done**, `tests/test_paged_kv_reassign_pages.cpp`.
   Permutation, grow from the pool, shrink, reclaiming a whole entitlement, the shrink-one-region
   while growing-another case served from the surplus, four rejections that must leave the mapping
   and pool untouched, and the republished block-table row. Each case re-reads the KV payload of
   every page it maps and confirms the bytes still belong to that physical page, which is the
   property that distinguishes re-pointing from copying.
2. `gqa_kv_rerotate` against an FP64 oracle -- **done**,
   `tests/targets/qwen3_6/test_gqa_kv_rerotate.cpp`. A re-rotated element is compared against a
   direct prefill at the destination, with the default table and with a caller-installed scaled
   table, through both entry points, plus zero-delta identity, the untouched tail outside `[0, 64)`,
   and six validation rejections. Errors are measured in BF16 ulp of the operand pair rather than
   of the result, because a rotation can cancel toward zero; the allowance is 4 ulp against a
   measured worst case of 1.79, while a sign-flipped variant costs 4222.
3. `gqa_kv_rerotate` against the production write and read paths -- **done**,
   `tests/ops/test_scr_relocation_integration.cpp`. The Op's own oracle test seeds the K plane
   through `paged_kv_element_offset`, which is the same addressing the Op uses, so it cannot detect
   the Op and the cache writer disagreeing about that convention. This test writes every cache byte
   with `gqa_kv_append` and reads through `gqa_attention_cached` and `gqa_attention`, against a
   permuted block table. A relocated cache comes out operationally indistinguishable from a directly
   prefilled one (7.2e-4 relative L2) while a skipped re-rotation registers at 2.3e-1, so the check
   is not blind.
4. End-to-end relocation wiring in `program_impl.h` -- **wiring defect found and fixed**, see below.
   `tools/scr_e2e_check.py` drives a live server over prompts that share a prefix and a suffix and
   differ in the middle, and refuses to report a pass unless the server logs `reuse=suffix_relocation`.
5. Phase 5.5 quality harness -- **not started**. `eval/ninfer_eval` has no `--scr` support`.

### The defect: the chunk's base position was captured before the cursor skipped the relocated run

Symptom: every relocation produced a single stop token and empty content, while `--no-scr` answered
correctly, and it did so for a relocated run of **one** 64-token block out of 3864 (1.7% of the
prompt). Scaling the run to 2, 5 and 10 blocks changed nothing, which ruled out the stale-state
approximation and pointed at the sequencing.

Cause: `advance_prefill` builds its `schedule::PrefillContext` from `staged.cursor` on entry, and that
becomes `PrefillContext::text_kv_base` -- the base position the chunk prefills at, which seeds
`positions`, sets `visible`, and decides which page `gqa_kv_append` writes. Further down the same
invocation, `skip_relocated_runs` advances `staged.cursor` across the relocated run. On the single
entry where that skip fires, the two diverged, and the chunk prefilled a whole relocated run early.
With a one-block run that is 64 tokens, so the 24-token tail landed at positions `[3776, 3800)`
instead of `[3840, 3864)`: it overwrote the first 24 positions of the relocated page, the prompt's
real last 24 positions were never written at all, and the sample was taken from the hidden state of
position 3799, which the prompt no longer ends at.

Two copies of one position, with nothing downstream comparing them. The fix propagates the skip into
`text_kv_base`, and `advance_prefill` now asserts the two agree before dispatching a chunk, so a
future divergence throws instead of quietly prefilling at the wrong positions.

What found it: fingerprinting each relocated destination page **after** the prefill that consumes it.
The existing diagnostic's before/after pair bracketed only the re-rotation, so a page that prefill
subsequently overwrote read as perfectly healthy -- which is why every component test passed while
the feature produced nothing. A per-row fingerprint then reported `changed_rows=96 of 256`, and 96 is
exactly 24 tokens x 4 KV heads, which is not a coincidence and does not need interpreting further.

Per-chunk tracing reported `cursor=3840 nominal=24` and looked clean, because it printed the
post-skip cursor. It now prints `base` alongside `cursor` and derives `visible` from `base`, so the
divergence shows as `base=3776 cursor=3840` instead of hiding behind a plausible-looking extent.

### A latent defect that the unit tests could not have caught

`rope.cuh` declares `kTextRopeInvFrequency` as `static __device__ __constant__`, which is internal
linkage: every translation unit that includes `rope.cuh` gets a private copy in its own device
module. `set_text_rope_frequencies` reaches `cudaMemcpyToSymbol` from `rope.cu` and writes only
that copy, so the Op was reading its own module's compile-time defaults rather than the table the
forward RoPE used. The table is now a required argument (`RerotateFreqTable`, carried into the
kernel by value as `RerotateFreq`), which is what made the oracle test above possible.

This is **not reachable in production on this hardware**. `compute_yarn_config` only scales above
`kNativeContext = 1048576`, and the 27B artifact's weights occupy 16.96 GiB of a 24 GiB card, so no
configurable context reaches that threshold. The defect was real but latent; the fix is
architectural (the Op no longer depends on mutable module-global state) rather than a correctness
rescue.

## SCR is currently limited to BF16 KV, and that caps it below useful context

`gqa_kv_rerotate` requires an uncompressed BF16 Text KV plane. `validate_text_k_cache` rejects every
quantized layout (`int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`), and **the request planner has
no matching guard**, so a quantized `--kv-dtype` plans a relocation that then throws at runtime
instead of degrading to `AppendAtFrontier`. A planner-side guard is the cheapest fix and is not
done yet.

For `qwen3_8_27b` the arithmetic is: 2 (K and V) x 16 full-attention layers x 4 KV heads x 256
head_dim, against 16.96 GiB of `groupwise-int` weights on a 23.99 GiB card.

| kv-dtype | KiB/token | KV @65k | KV @131k | KV @262k | SCR |
|---|---|---|---|---|---|
| bf16 | 64 | 4.0 GiB | 8.0 GiB (over) | 16 GiB (over) | yes |
| int8 | 32 | 2.0 GiB | 4.0 GiB | 8.0 GiB (over) | no |
| rk4v4-e8 | 16 | 1.0 GiB | 2.0 GiB | 4.0 GiB | no |
| rk2v4-e8 | 12 | 0.8 GiB | 1.5 GiB | 3.0 GiB | no |

So on this card SCR works only up to roughly 65k context, while the quantized formats are what buy
the 160k-500k configurations in the README. That is backwards for a memory-constrained device: the
feature is available only in the regime with the least memory pressure to relieve.

### Why quantized K is the blocker, and why V is not

RoPE acts on K only. The Op's own contract already states it: *"This Op deliberately does not touch
V: RoPE does not act on V, so relocated V needs no adjustment."* V therefore needs no work in any
format, and only K blocks a quantized layout.

Re-rotation of BF16 K is a multiply-add per rotary plane. Re-rotation of a quantized K is not a
rotation at all:

- **`int8` (channel-quantized).** Dequantize, rotate in FP32, requantize with the same group scales.
  Tractable. It adds one rounding of quantization error to relocated keys, which is measurable
  against the Phase 5.5 threshold and would roughly double the usable SCR context to ~131k. This is
  the highest-value item.
- **`rk4v4-e8` / `rk2v4-e8`.** K is a 4-bit (or 2-bit) Conway-Sloane `E8` lattice code. Re-encoding
  after an arbitrary rotation is a lattice-automorphism problem: find the codeword nearest to the
  rotated vector, which is generally not the rotation of the original codeword. This is a research
  problem, not an implementation detail, and there is no known cheap encoding. Deferred.

Both need a decision on error accounting before implementation: whether a relocated quantized key is
allowed to carry more error than a key prefilled at that position, and if so how much, measured
against the same < 0.05 nat divergence threshold Phase 5.5 already defines.

Recommended order: (1) the planner-side guard so `--scr` is safe to leave enabled across
config changes, (2) `int8` K re-rotation, (3) the E8 work only if (2) shows the approach pays off.

### A note on how SCR is triggered, for anyone writing tests

`prefix_matches` requires the resident prompt to be a strict prefix of the new one, so ordinary
prefix reuse never fires for an in-the-middle edit. SCR handles that case by diffing blocks
directly: `diff_prompt_edit` indexes every whole 64-token block of the resident prompt by hash and
finds surviving spans anywhere in the new prompt.

The consequence for testing is that **the surviving tail must sit at the same 64-token boundary in
both prompts**, which means the two edits must differ in length by a whole number of blocks. That
cannot be arranged by counting characters. `tools/scr_e2e_check.py` uses the server's own
`/v1/messages/count_tokens` to measure both prompts and search a padding until the delta is block
aligned; without that, SCR silently declines to engage and reports `full_reset`.
