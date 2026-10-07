# SCR relocation paths

Companion to `docs/SCR_ROADMAP.md`. That document describes the phases; this one pins down the two
concrete ways the runtime can apply a relocation plan, exactly where each attaches, and what has to
be decided before either is written.

Status: **the primitives exist and are committed; neither path is wired into the runtime yet.** A
`ReusePath::SuffixRelocation` plan is currently applied as ordinary prefix reuse, so generated output
is unchanged (`program_impl.h`, `start_prefill_lane`).

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

**Applies when** the relocated spans tile a contiguous run that reaches the end of the prompt, so the
only prefilled region is the head. Concretely, for chosen spans sorted by destination:

- the first span's `dest_begin` is block aligned,
- consecutive spans are adjacent (`next.dest_begin == previous.dest_end`, no gap),
- the last span's `dest_end` is the last 64-aligned boundary at or below `prompt_tokens`.

Two sub-cases:

**A1 - the tail ends exactly on the relocated region.** Prefill the head range once, then relocate,
then run a single-token forward at position `prompt_tokens - 1`. The extra forward is not overhead
specific to SCR: it is the first decode step. Prompt logits have to come from a real forward pass
because only KV is cached, never the hidden state, so *no* design can sample the next token purely
from relocated state.

**A2 - a partial block is left after the relocated region.** Prefill the head, relocate, prefill the
partial trailing block, then take logits from that last chunk. Two prefilled ranges. This is the
common case, and it is why A1 alone is not enough.

Which of A1/A2 applies is decided by `prompt_tokens mod 64` and the alignment of the surviving
spans; it cannot be decided at plan time without also knowing the final block layout.

### Step list

In `start_prefill_lane`, on `ReusePath::SuffixRelocation` with `relocated_spans` non-empty:

1. Instead of `trim_sequence_kv(sequence, base, ...)`, compute the pages the emitted sequence still
   needs and call `reassign_pages` with a mapping whose relocated slots name the surviving pages and
   whose replaced slots are `kPagedKVAquirePage`. This both keeps the surviving pages and publishes
   the new block-table row, which step 2 depends on.
2. For every Text full-attention layer, call `gqa_kv_rerotate(positions, rope_deltas, layer_view,
   stream)` with `positions` = the new logical positions of the relocated tokens and
   `rope_deltas` = `new_position - old_position` (negative, for a left shift). One call per layer,
   because the op takes a single layer's `PagedKVLayerView`.
3. Prefill the head `[reuse_base, first_dest_begin)` and, for A2, the trailing partial block.
4. Report `relocated_prompt_tokens` and set `prefix_reuse_path` to `SuffixRelocation`.

### Linear-attention caveat (blocking, specific to Qwen3.6-27B)

16 of the 27B hybrid's 84 layers are full attention; the other 48 are Gated DeltaNet, whose state is
a single recurrent summary rather than per-token KV. Relocation as described only repairs the full
attention layers. For the linear layers the state is whatever the sequence already held, and if the
runtime prefills only the head and the trailing block, the recurrent state never sees the interior of
the relocated tail. That is a *much* larger approximation than the paper describes - the paper's
scheme snapshots and restores the recurrent state rather than silently skipping tokens.

So Path A is not complete for the 27B target until one of these is chosen:

- process the relocated tail through the linear layers only (state update without KV write), which
  costs a partial forward and eats into the savings but keeps the state faithful;
- accept the stale state and treat it as part of the SCR approximation, documented and measured;
- restrict SCR to full-attention-only models.

This needs a decision before Path A is written, and it is not a detail that can be settled by
reading the kernels.

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

To bench the paths against each other, the choice has to reach the plan:

- add an execution option (e.g. `ExecutionOptions::scr_mode`) and mirror it into
  `ResolvedExecutionOptions`, `ServeOptions`, `RequestBasePlanImpl`;
- surface it as `--scr-mode tail|segmented` in `serve_options.cpp`, validated like the other
  enumerated flags;
- default to `tail`, and have the planner reject a plan whose `relocated_spans` do not satisfy the
  selected mode's condition, falling back to `AppendAtFrontier` rather than relocating partially;
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
