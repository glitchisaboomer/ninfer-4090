# SCR relocation paths

Companion to `docs/SCR_ROADMAP.md`. That document describes the phases; this one pins down the two
concrete ways the runtime can apply a relocation plan, exactly where each attaches, and what has to
be decided before either is written.

Status: **the primitives exist and are committed; neither path is wired into the runtime yet.** A
`ReusePath::SuffixRelocation` plan is currently applied as ordinary prefix reuse, so generated output
is unchanged (`program_impl.h`, `start_prefill_lane`).

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

## Verification that is still owed

Everything in Path A and Path B below the plan is unverified by the build: this change set was
prepared on a host with no `nvcc`, `cmake`, or `ninja`, so `gqa_kv_rerotate`, the `reassign_pages`
integration, and the `src/CMakeLists.txt` entries have never been compiled. The only CUDA-side code
exercised at all is the `reassign_pages` accounting, through a host harness that compiles the real
function body against a mock pool.

On a CUDA machine, in order:

1. `reassign_pages` against the real pool: permutation, grow, shrink, and the shrink-and-grow case
   that needs no free pages.
2. `gqa_kv_rerotate` against a forward pass: prefill a token at position `p`, relocate it to `p'`,
   and compare the cached K elementwise against prefilling the same token at `p'` directly. The
   expectation is agreement except for one extra BF16 rounding. This is the check that
   `kTextRopeInvFrequency` pairing and the `sincosf` argument order are right.
3. Only then the end-to-end paths, with the Phase 5.5 quality harness (50 prompts, mean next-token
   logprob divergence < 0.05 nats, greedy agreement on the first 64 tokens > 95%).
