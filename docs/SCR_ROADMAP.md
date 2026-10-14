# SCR_ROADMAP for ninfer-4090

Purpose: roadmap for adding Suffix Cache Reuse (SCR)-style reuse to ninfer-4090's
Qwen3.6-family runtime, based on the paper `2609.37725v1` and a targeted read of:

- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`
- `src/targets/qwen3_6/impl/runtime/prefix_identity.cpp`
- `src/serve/serve_options.cpp`

This roadmap is scoped to where SCR would attach in the existing reuse/prefill path,
not yet a full implementation plan.

--------------------------------------------------------------------------------
## 1. What SCR means here
--------------------------------------------------------------------------------

From the paper:

- Standard prefix reuse: after an edit such as `[A B C] -> [A B' C]`, only `A` is
  reused. `B'` and `C` are re-prefilled.
- SCR: reuse cached KV for the surviving suffix `C` at its new position, with key
  rotary positions adjusted for the new location. Only `B'` and any newly appended
  tokens are prefilled.
- SCR is explicitly an approximation: reused `C` was computed under the old context.
- The paper also covers linear-attention / recurrent-state layers differently from
  token-level KV layers, and caps relocated spans per edit via `K`.

For ninfer-4090, the practical win is:
- avoid full re-prefill of surviving suffix after a CLM-style context edit
- and/or avoid re-prefill after reasoning-token stripping in chat-style turn transitions

--------------------------------------------------------------------------------
## 2. Where the current reuse/prompt path lives
--------------------------------------------------------------------------------

Current path is prefix-oriented, not edit-aware.

### 2.1 Request planning and reuse decision
Primary file:
- `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`

Key reuse logic:
- `allow_prefix_reuse` gating
- reuse path selection among:
  - `ReusePath::FullReset`
  - `ReusePath::AppendAtFrontier`
  - `ReusePath::RestoreTurnCheckpoint`
  - reuse from disk checkpoint path
- reuse is based on prefix identity match against retained sequence state, especially:
  - `sequence.ledger`
  - `sequence.prefix_identity`
  - `sequence.execution_frontier`

So today there is a notion of:
- retained lane state
- a frontier up to which the resident prefix is reusable
- a prefix identity and ledger used for matching

This is the right neighborhood for SCR, but SCR needs a different primitive:
- not just "match prefix from position 0"
- but "find spans in the new prompt that survived an edit relative to the previous prompt"

### 2.2 Prompt identity and prefix matching
Primary file:
- `src/targets/qwen3_6/impl/runtime/prefix_identity.cpp`
- header: `src/targets/qwen3_6/impl/runtime/prefix_identity.h`

Current matching includes:
- token ids
- token types
- position axes
- vision item metadata for the matched prefix region

This is the place where SCR would need a new matching/alignment view:
- match surviving suffix spans, not only a single prefix length
- preserve old position/rope information for the reused spans
- decide which surviving spans are eligible for reuse

### 2.3 Prefill startup and reuse materialization
Primary file:
- `src/targets/qwen3_6/impl/runtime/program_impl.h`

Important sections read:
- constructor / state init
- `start_prefill_lane(...)`
- reuse branches:
  - `ReusePath::FullReset`
  - disk checkpoint restore path
  - `ReusePath::AppendAtFrontier`
  - turn-checkpoint reuse path
- KV reservation/bind/trim/materialize helpers:
  - `reserve_sequence_kv`
  - `resize_sequence_kv_entitlement`
  - `bind_sequence_kv`
  - `unbind_sequence_kv`
  - `materialize_sequence_kv`
  - `trim_sequence_kv`
- sequence state fields:
  - `sequence.ledger`
  - `sequence.prefix_identity`
  - `sequence.text_kv_valid`
  - `sequence.mtp_kv_valid`
  - `sequence.dflash_context_frontier`
  - `sequence.execution_frontier`
  - `sequence.rope_delta`

What this shows:
- reused prefix is currently materialized through existing KV allocation + validity
  tracking and ledger/prefix_identity assignment
- the runtime already knows how to retain, trim, and resume from a frontier
- it does not currently have a reuse path for "splice reused suffix KV into a new
  prompt after an edit"

### 2.4 Serve-layer controls
Primary file:
- `src/serve/serve_options.cpp`
- also relevant: `src/serve/generation_service.*`, request log, metrics paths

Relevant existing controls:
- `--no-prefix-reuse` to disable prefix reuse
- prompt cache / disk cache options
- reuse reporting through metrics/logging paths

For SCR, the serve layer would need to:
- expose an edit-aware request shape or session context bridge
- allow the harness / client to express "this turn is based on the previous context
  with these edits"
- report SCR-specific reuse outcomes, not just prefix reuse

--------------------------------------------------------------------------------
## 3. Where SCR pieces would attach
--------------------------------------------------------------------------------

### 3.1 New abstraction: edited-context request
Attach point:
- request preparation / request plan input path
- likely new structure describing:
  - previous prompt/context
  - current prompt/context
  - surviving spans with old positions and new positions
  - removed/replaced spans
  - newly appended tokens

Why:
- SCR requires diffing the new prompt against the session's previous prompt.
- The current `PreparedPromptData` is not expressed as "previous context + edit".

### 3.2 New primitive: surviving-span detection
Attach point:
- new module near or beside `prefix_identity.*`
- or a new `prompt_diff` / `edit_analysis` module

Responsibility:
- find maximal surviving spans after an edit
- map them from old token positions to new token positions
- identify which surviving spans can reuse cached KV
- identify which spans must be prefilled

Algorithmic constraint (must be satisfied before implementation):
- The diff must run on CPU before dispatch.
- Time complexity must be O(N) in prompt length.
- Token-level Longest Common Subsequence is O(N^2) and is unacceptable for 32K+ contexts.
- Use block-level matching: hash contiguous token blocks (e.g. 16 or 64 tokens) and match by block hash.
- This aligns with existing PagedAttention-style block granularity.
- Block-boundary handling:
  - When an edit splits a block, accept the loss and skip partial blocks.
  - This wastes up to (block_size - 1) tokens per edit boundary but keeps complexity at O(N).
  - Only revisit this choice if Phase 5.5 shows quality degradation.

This is one of the biggest pieces because today the engine only needs
prefix matching.

### 3.3 New reuse path: suffix span relocation
Attach point:
- `request_plan_impl.h` reuse selection
- `program_impl.h` prefill startup and KV setup

What changes:
- current reuse paths set `reuse_base` and resume from a frontier
- SCR would add a reuse mode where:
  - prefix up to edit point may still be reused as today
  - changed/replaced span is prefilled
  - surviving suffix spans are reused from old KV but relocated to new positions
  - any newly appended tokens are prefilled

VRAM allocator integration (must be decided before implementation):
- Session-private relocated slots must not fragment the 24 GB VRAM pool.
- Option A: represent relocated spans as virtual entries in the existing block table with a per-block RoPE offset field. No secondary allocator needed.
- Option B: maintain a secondary session-private pool with explicit eviction. Risk: fragmentation and OOM under concurrent requests.
- Prefer Option A.
- If Option B is required, add:
  - an LRU eviction policy for relocated slots
  - a hard cap on per-session relocated block count

ReusePath enum interaction:
- SCR composes with `AppendAtFrontier`: prefix reuse up to edit + suffix relocation.
- If the planning code cannot express this composition, add a new enum value
  `ReusePath::SuffixRelocation` as a mutually exclusive alternative.

This likely needs:
- a way to describe multiple reused spans, not just one `reuse_base`
- position/rope adjustment for reused keys in the new layout
- staging area for relocated reused KV separate from normal prefix cache

### 3.4 KV staging and rotary adjustment
Attach points:
- KV allocation/bind/materialize helpers in `program_impl.h`
- rope/position handling already present in the runtime
- execution/prefill schedule paths that feed KV into attention

Why:
- SCR reuses cached KV for surviving tokens but at new positions.
- In this runtime, that means:
  - either adjust rope/position metadata for reused entries before use
  - or copy/transform reused KV into a form usable at the new positions
- reused suffix must not corrupt the normal retained-prefix semantics

This is likely the highest-risk area because it touches how prefill/attention sees
the context layout.

#### Kernel & Memory Strategy
On RTX 4090, memory bandwidth (~1 TB/s) is the primary bottleneck, not FLOPs.
Physically copying and re-rotating Keys for relocated spans may cost more bandwidth
than the prefill compute saved.

Before implementation, decide between two paths:
- Path A: modify attention kernels to accept a per-block `position_ids` / RoPE offset map. Zero extra memory traffic; higher kernel complexity.
- Path B: copy and re-rotate Keys into a staging buffer. Lower kernel complexity; adds memory-bandwidth cost proportional to relocated span length.

Default to Path A unless kernel modifications are blocked by existing abstractions.
Document the decision and its bandwidth budget in (the design note produced by Section 7).

RoPE for queries and keys:
- RoPE applies to both queries and keys.
- Path A must handle K-side re-rotation via per-block position offset AND Q-side position assignment for `B'` prefill.
- Extend `sequence.rope_delta` from scalar to per-block offset array.

Decode-phase attention:
- Decode-phase attention kernels must consume the same per-block position map as prefill.
- Add validation item to Phase 5: "Verify decode-phase attention uses same position map."

### 3.5 Hybrid/recurrent-layer handling if applicable
Attach point:
- wherever recurrent/linear-attention state is managed for sequences
- e.g. GDN/replay state handling paths already referenced in `program_impl.h`

Why:
- Qwen3.6-27B has 48 linear-attention layers (Gated DeltaNet) and 16 full-attention layers.
- For linear-attention layers, SCR does not relocate per-token KV.
- Instead, snapshot the recurrent state before the edit and continue from that snapshot.
- The newly inserted tokens (`B'`) are not processed by linear-attention layers;
  they only affect full-attention layers.

GDN checkpoint granularity:
- If checkpoints exist only at turn boundaries, mid-turn edits force linear-attention
  layers to either resume from a stale checkpoint (losing intra-turn state) or re-run
  from scratch.
- Audit GDN checkpoint granularity before Phase 3.
- Document the accuracy cost of stale checkpoint vs re-running.

Whether this matters in ninfer-4090 depends on which attention/state types are in
the target build being used.

### 3.6 Serve-layer / harness integration
Attach point:
- `src/serve/` request translation and session handling
- request log / metrics paths

Why:
- The paper's SCR is driven by context edits or reasoning-token stripping between turns.
- ninfer-4090 would need a way to pass "previous context + edit" into the engine,
  not only a brand-new prompt string.
- Metrics should distinguish:
  - prefix reuse
  - SCR suffix reuse
  - fallback to full prefill

--------------------------------------------------------------------------------
## 4. Suggested implementation order
--------------------------------------------------------------------------------

### Phase 0: choose scope
Pick one of:
- A. reasoning-token-stripping suffix reuse only
- B. General CLM-style context edit suffix reuse
- C. both, but behind a feature flag

Recommendation:
- Start with Scope A.
- Paper evidence: Appendix B, Figure 14 shows that on BrowseComp-Plus with
  Qwen3.6-27B, 5.3 of the 7.8 percentage-point SCR savings come from
  reasoning-token stripping, not from CLM context edits.
- That means Scope A captures ~68% of SCR's total benefit with zero harness API changes.
- For ninfer-4090, this is also the easiest first win if the harness already emits
  turn transitions where prior reasoning tokens are removed and the rest is preserved.

### Phase 1: represent the edit
- Add an edited-context representation for the chosen scope.
- For A: "previous tokens minus removed block, preserved suffix unchanged".
- For B: explicit surviving spans with old/new positions.

### Phase 2: detect surviving spans
- Add diff/alignment logic.
- Preserve:
  - old token ids
  - old positions/rope
  - old prefix identity metadata where relevant
  - vision item constraints if multimodal

### Phase 3: add SCR reuse path in planning
- Extend reuse selection so a plan can include:
  - prefix reuse up to edit point if applicable
  - surviving suffix reuse spans
  - prefill for replaced/new tokens

### Phase 4: materialize reused suffix KV safely
- Add staging for relocated reused KV.
- Add rotary/position adjustment for reused keys in new positions.
- Ensure normal prefix cache and retained state are not corrupted.

### Phase 5: validate attention composition
- Verify prefill/attention can consume:
  - reused prefix
  - newly prefilled changed/appended tokens
  - relocated reused suffix
- This is where correctness work should concentrate.
- Verify decode-phase attention kernels consume same per-block position map as prefill and produce correct logits.

### Phase 5.5: approximation validation
SCR is an approximation. Reused suffix KV was computed under the old prefix.
On smaller consumer models, this may degrade output quality more than on large
cloud models.

Run 50 prompts through both:
- full-prefill path
- SCR path

Measure:
- mean next-token logprob divergence, with a < 0.05 nats threshold
- greedy-decode agreement on the first 64 tokens, with a > 95% threshold

Report per-prompt results.

If the threshold is exceeded:
- reduce `K`, or
- disable SCR for that session

#### Measured status (2026-10-08)

The harness is `build-ninja/apps/ninfer-scr-quality.exe` (`apps/scr_quality/`). Four phases per prompt:
`scr` (`--scr`), `baseline` (`--no-scr --no-prefix-reuse`), `control` (the baseline repeated), and, with
`--flag-control`, a fourth phase running the SCR flags with relocation off (`--scr-max-spans 0`).
Scoring is `evaluate_scr_quality`, so the two thresholds above exist in exactly one place. The
`ATTRIBUTION` table prints the measurement against both floors — the run-to-run control and the
flag-only configuration — and the run exits non-zero when the control is dirty.

What the first controlled run established:

- The token stream is reproducible: the control agreed with itself on every token at every position
  (`parted@ = 64`, agreement 1.0, all six prompts), so the early splits at token 6-8 belong to the
  reuse path, not to run-to-run noise.
- The logprobs are not. The control's own `d(first)` ran 3e-4 to 1.8e-3 nats, so a divergence is only
  attributable against the floor, never against zero.
- Subtracting the floor leaves two signals that reproduce across independent runs: `K=2` (192
  relocated tokens) at 0.0747/0.0733, about 178x the floor, and `K=6` (576 relocated) at
  0.0201/0.0213, about 40x. `K=3`'s signal changes sign between runs, which is what noise does.
- `K=2` is unexplained. It relocates the *least* context and shows the *most* perturbation, the
  signature of a near-tie being tipped rather than of accumulated staleness. `ops::logprob_of_choice`
  now also emits the decision margin, so a large `d(first)` beside a small margin is a coin flip and
  a large margin is a moved decision. Do not call `K=2` a defect until the margin says what the
  competing logits were.

Note the tension the numbers expose, whichever way `K=2` resolves: the thresholds are a mean
divergence under 0.05 nats and greedy agreement over 95% of 64 tokens. If the `K=2` signal is real,
the second bound is already violated by a perturbation the first permits — a finding about the
thresholds, not about SCR.

Still owed: the 50-prompt run on the chat-shaped suite (a real system prompt plus multi-turn history,
not the e2e filler, which measures SCR where it does nothing useful), a regime with a long reused
prefix rather than `base=0/cache_hit=0`, the per-dtype runs laddered in
`docs/SCR_ALIGNMENT_FREE_RELOCATION.md`, and the `K=2` margin investigation.

### Phase 6: metrics, fallback, and caps
- Add SCR-specific reuse reporting.
- Add fallback to full prefill if staging fails or reuse precondition fails.
- Hardcode `K = 6` as the initial relocated-span cap.
  - The paper's sensitivity analysis (Figure 13) shows task accuracy is robust across
    `K in {1,2,3,6,12,64}` and cache-reuse gains saturate by `K=6`.
  - Make `K` configurable via `serve_options`.
  - Do not expose tuning until correctness validation passes.

  *(Done: `--scr-max-spans <N>`, default 6 — `serve_options`' `suffix_reuse_max_spans`. It caps the
  number of spans, not the relocated token count; the paper's K is a token count, so a token cap is
  still unbuilt. `--scr-max-spans 0` leaves `--scr` set while switching relocation off, which is what
  the quality harness's configuration control uses.)*

--------------------------------------------------------------------------------
## 5. Concrete anchor points in current code
--------------------------------------------------------------------------------

- reuse decision: `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`
- prefix identity / matching: `src/targets/qwen3_6/impl/runtime/prefix_identity.*`
- prefill startup + reuse branches: `src/targets/qwen3_6/impl/runtime/program_impl.h`
  - especially `start_prefill_lane(...)` and its reuse branches
- KV lifecycle helpers: `program_impl.h`
  - `reserve_sequence_kv`
  - `trim_sequence_kv`
  - `materialize_sequence_kv`
  - `bind_sequence_kv`
  - `unbind_sequence_kv`
- serve controls: `src/serve/serve_options.cpp`
- reuse reporting: request log / metrics paths under `src/serve/`

--------------------------------------------------------------------------------
## 6. Biggest risks
--------------------------------------------------------------------------------

1. Prompt representation mismatch
   - current engine is built around one prompt + prefix reuse
   - SCR needs edit-aware context representation

2. Relocated KV correctness
   - reusing suffix KV at new positions requires careful rope/position handling
   - wrong handling here would silently produce wrong attention

3. Mixed-context prefill/attention layout
   - new prompt would be prefix + new tokens + relocated suffix
   - existing paths may not be written for that shape

4. Scope creep
   - full SCR for arbitrary edits is more work than the paper's easiest win
   - reasoning-token stripping case is narrower and may be a better first target

--------------------------------------------------------------------------------
## 7. Recommended next step
--------------------------------------------------------------------------------

Read the exact SCR-influencing code sections once more with this mapping in hand,
then decide:
- whether to start with reasoning-token stripping suffix reuse, or
- whether to go straight for general CLM edit reuse

After that, the next concrete artifact would be a small design note describing:
- the edited-context data structure
- the surviving-span representation
- the new reuse path enum/fields
- the staging/rotation approach
- the first-scope fallback behavior
