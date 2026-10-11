# Handoff: SCR Phase 5.5 — quality validation

For a model picking this up cold. Written after the Suffix Cache Reuse wiring was found and fixed;
the remaining work is *measurement*, not correctness.

Branch: `feat/rtx-4090-sm89-native`. Everything below is pushed. Build and run happen on the user's
Windows/CUDA machine (`E:\playground\ninfer-4090`); the development sandbox has no `nvcc`, so
changes can be type-checked there but not compiled or run.

---

## 1. Where things stand

### Done and verified on hardware

| Area | Commit | Note |
|---|---|---|
| SCR relocation wiring fixed | `316ff9ba` | the actual bug, see §4 |
| relocation integration test | `00f94f1e` | green, proves the Op is correct |
| `ops::logprob_of_choice` | `85581d25` | green, numerics verified |
| post-prefill integrity diagnostic | `50274e81` | kept, gated behind `NINFER_SCR_DIAG` |

`ctest` is green except for real-model tests that skip without artifacts. `--scr` now produces output
identical to `--no-scr` on the e2e check.

### The remaining work

Two pieces, in this order. Do not start the second before the first is green on hardware.

- **Stage 1A — runtime plumbing.** Make the engine emit per-token logprobs. Detailed in §3.
- **Stage 0 — the harness.** Drive both reuse paths over a prompt suite and score the divergence.
  Detailed in §3.

### Deliberately out of scope

The user chose **1A** (diagnostic-only logprob emission) over exposing `logprobs` on the served API.
Adding OpenAI/Anthropic `logprobs` response shapes is a separate, independently valuable feature. Do
not widen the serve surface as part of this work.

Two smaller gaps are also known and deferred, in case they come up:

- ~~The planner's SCR gate does not check the KV dtype, so a quantized `--kv-dtype` plans a
  relocation that `validate_text_k_cache` then rejects.~~ **Resolved.** The gate now exists
  (`request_plan_impl.h`'s `delta_path_supports_dtype`, which admits BF16 and I8) and
  `validate_text_k_cache` is gone: relocation publishes a per-page RoPE delta instead of
  re-rotating the stored K, so quantized layouts are available. What is owed is the per-dtype run of
  `tools/scr_e2e_check.py` (it accepts `bf16`, `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8` and refuses
  `rk2v4-e8`). See `docs/SCR_QUANTIZED_KV_RELOCATION.md`.
- There is no cap on the relocated **token** count. `--scr-max-spans` (default 6) caps the *number of
  spans*, and tail mode produces exactly one span, so it never binds. The paper's `K` in Figure 13 is
  a token count. The roadmap says hardcode `K = 6`, configurable but unexposed until correctness
  validation passes.

---

## 1a. The measurement is built; it has never yet produced an attributable number

`ninfer-scr-quality` exists and runs, but treat its output as **unproven** until the null control comes
back clean. Three successive defects were found by reading its output, and each one had produced a
confident wrong number rather than an obvious failure:

1. **The suite could not exercise the feature.** `diff_prompt_edit` finds surviving spans by hashing
   whole 64-token pages. The built-in suite's revision changed the prompt's token count by a few
   tokens, so no page boundary after the edit realigned and nothing survived. Five of six prompts
   reported `full_reset`. Fixed by making the edit length-balanced *and* by having the harness refuse
   a non-page-aligned edit (`f7119708`).

2. **The scorer measured its own divergence feedback.** `evaluate_scr_prompt` compared logprobs
   index-aligned over `min(len_a, len_b)` — "shared prefix" was implemented as *the shorter run*.
   Both runs emit 64 tokens, so the bound never bound, and past the first flipped token it
   differenced logprobs of two *different* tokens. The header for that file already said this must
   not happen. Fixed in `036a266c`.

3. **There is no noise floor.** Two runs of one configuration are not guaranteed bit-identical, so a
   divergence without a control is unattributable. This is what the third phase is for.

Symptom of (2) that gave it away: greedy agreement was *non-monotone* in the amount of reused context
(0.125, 0.109, 0.094, 0.094, 0.750, 0.750 for K = 1…64). Index-aligned agreement scores whatever
fraction of the drift happened to re-synchronize. **A sweep whose x-axis is reused-context size cannot
read as a curve if the y-axis measures coincidence.**

The corrected run then produced this, which is *not* a result, only a reason to want the control:

| K | relocated | parted@ | d(first) | d(mean) |
|---|---|---|---|---|
| 1 | 64 | 8 | 0.0021 | 0.0257 |
| 2 | 192 | 8 | 0.0751 | 0.1692 |
| 3 | 256 | 6 | 0.0043 | 0.0041 |
| 6 | 576 | 6 | 0.0206 | 0.0410 |
| 12 | 1152 | 6 | 0.00018 | 0.0028 |
| 64 | 6208 | 48 | 0.0033 | 0.0233 |

`d(first)` has **no relationship to how much context was reused** — the largest perturbation is at
K=2, the smallest at K=12. A relocation error that grows with the reused region does not look like
that. A run-to-run floor of comparable size does. Hence the control.

### The control is clean on tokens, and its logprobs still moved

The 2026-10-08 control agreed with itself on **every token at every position** (`parted@ = 64`,
agreement 1.0, all six prompts). So the early splits at token 6–8 are reproducible and belong to the
reuse path — the nondeterminism hypothesis is dead.

Its logprobs were **not** identical, though. `d(first)` ran 3e-4 to 1.8e-3 nats and `d(max)` reached
0.080. An earlier version of the harness printed "the run-to-run floor is zero" on the strength of the
token agreement; that sentence was false, and the control's own numbers contradicted it directly
underneath. Greedy choice is insensitive to a perturbation the logprob metric is not.

Subtracting the floor leaves **two of six prompts carrying signal**, reproducing across independent runs:

| K | relocated | signal run 1 | signal run 2 | × floor |
|---|---|---|---|---|
| 1 | 64 | 0.00033 | 0.00106 | 0.6 |
| 2 | 192 | 0.07465 | 0.07329 | **178** |
| 3 | 256 | 0.00270 | −0.00080 | 1.7 |
| 6 | 576 | 0.02006 | 0.02128 | **40** |
| 12 | 1152 | −0.00012 | 0.00062 | 2.0 |
| 64 | 6208 | 0.00177 | 0.00163 | 1.1 |

K=3's signal changes *sign* between runs, which is what noise does. So the earlier reading had it
backwards: the apparent K=2 anomaly is the most robust signal in the sweep, and K=12's pristine
0.00018 was the noise.

K=2 is nonetheless unexplained. It relocates the *least* context of any interesting prompt and shows the
*most* perturbation. That inversion is the signature of a near-tie being tipped rather than of
accumulated staleness — which is why the decision margin was added (see below) and why K=2 must not be
called a defect until the margin says what the competing logits were.

### What the margin settles

`ops::logprob_of_choice` now also emits `margin[t]`, the gap in nats between the best and second-best
logit. `d(first)` is the logprob of the *argmax*, so a near-tie turns a few thousandths of a nat into a
large apparent divergence: the perturbation is real but the decision it overturned was decided by a
hair. The margin separates "flipped a coin" from "moved a decision", and it rides in the same per-thread
reduction state, so it costs one compare and one select per streamed logit and no second pass.

The scorer reports `first_token_margin` — the margin at position 0, paired with the
`first_token_divergence` it explains. A small margin with a large divergence is a coin flip; a large
margin with a large divergence is a moved decision.

The floor itself is reported per prompt rather than subtracted from the thresholds.
`evaluate_scr_prompt_against_control` fills `floor_*` and `signal_*` and leaves `ok` exactly as
`evaluate_scr_prompt` set it: the roadmap's bounds are stated against the measurement, and quietly
scoring a noise-corrected quantity instead would mean the score no longer means what the roadmap says.
What a given signal-to-floor ratio is *worth* is a threshold question and belongs with the other
thresholds.

### How to read the control

- **Control clean** (`parted@ == logprobs_available` everywhere, `attributable: true`, exit 0): the
  *token stream* is reproducible, so `parted@` of 6–8 is a real finding about relocation worth chasing
  hard. It does **not** mean the logprobs matched — read `d(first)` against the `floor` column the
  harness now prints, never against zero.
- **Control dirty** (`attributable: false`, exit 3): two identical configurations disagreed, so
  `d(first)` has no floor to subtract and says nothing about SCR. The work is then determinism, not
  quality. The report is still written, because it is the evidence for *why* it is unattributable.

Note the second-order tension this exposes, whichever way the control comes back: the roadmap's two
thresholds are `mean divergence < 0.05 nats` and `greedy agreement > 95% over 64 tokens`. The first
is a budget loose enough to permit perturbations that will eventually flip a greedy decode; the second
requires *zero* flips in 64. If `d(first)` turns out to be real, those two bounds cannot both be
satisfied on this model, and that is a finding about the thresholds rather than about SCR.

---

## 2. Environment

```powershell
git pull
ninja -C build-ninja -j 32
ctest --test-dir build-ninja -R "logprob_of_choice|scr_relocation" --output-on-failure
```

The user's build is MSVC 2022 + CUDA 13.x, `-DCMAKE_CUDA_ARCHITECTURES=89`, `-DBUILD_TESTING=ON`, and
`NVCC_PREPEND_FLAGS=--split-compile=0`. A restricted `PATH=` (system32/Windows/Wbem only) in their
build script is a **deliberate WDDM workaround — do not "fix" it.**

Adding a file to a `CMakeLists.txt` source list triggers a CMake reconfigure. With
`--split-compile=0` a resulting CUDA rebuild of this project is genuinely slow — tens of minutes is
normal. Do not mistake it for a hang.

### Type-checking without a CUDA toolchain

The sandbox has stub CUDA headers at `/home/user/projects/_cuda_check/stub/` that let plain g++
parse runtime TUs. Both model variants must be checked — they instantiate the same headers with
different `TextConfig`:

```bash
cd /home/user/projects/ninfer-4090
g++ -std=c++20 -fsyntax-only -DNINFER_SM89 \
  '-DNINFER_QWEN36_VARIANT=::ninfer::targets::qwen3_6_27b::detail::Variant' \
  -DNINFER_QWEN36_RUNTIME_NS=qwen3_6_27b_runtime \
  -I src -I include -I src/targets/qwen3_6_27b/export -I src/targets/qwen3_6/export \
  -I src/targets/qwen3_6/impl -I /home/user/projects/_cuda_check/stub \
  /home/user/projects/_cuda_check/vc_q27b.cpp
```

Swap `qwen3_6_27b` → `qwen3_6_35b_a3b` for the second variant. This catches scope errors, wrong
qualified names, and missing includes — it caught three real mistakes while writing this work.

---

## 3. The work

### Stage 1A: make the engine emit per-token logprobs

`ops::logprob_of_choice` exists and is verified. It needs no changes. What is missing is that nothing
calls it.

Contract, for reference:

```cpp
// out[t] = float(logits[chosen[t],t]) - logsumexp_{0 <= v < valid_rows} float(logits[v,t])
void logprob_of_choice(const Tensor& logits, const Tensor& chosen, std::int32_t valid_rows,
                       Tensor& out, cudaStream_t stream);
```

`logits` is contiguous BF16 `[physical_rows, T]` with the vocabulary contiguous inside each column —
the same layout `ops::sample` and `ops::argmax` already consume, so nothing needs reshaping.

**Call sites.** Only two, because SCR is gated to the plain decode backend and MTP
(`request_plan_impl.h`) — the default backend is among them, so the e2e path uses these:

1. `src/targets/qwen3_6/impl/runtime/decode_impl.h:43` — ordinary decode, inside
   `ordinary_batch_body`, which is **CUDA-graph captured**. This is fine: the Op writes to the egress
   buffer's stable device address, which is what graph capture requires.
2. `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1184` and `:1187` — the prefill sample and
   its `ops::argmax` fallback, under `if (is_last)`. `logits` there is `matrix_window(io_.logits, 1)`,
   so `[rows, 1]`.

Skip the MTP/DFlash sites (`mtp_impl.h`, `dflash_impl.h`, `text_prefill_impl.h:152`) — out of scope
for this probe. DFlash is still not exercised by SCR; MTP is, since the second-allocation relocation
landed, so an MTP probe would now be a third call site rather than a dead one.

**Egress.** Add one field beside the existing token array in
`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:40`:

```cpp
struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
    std::array<float, kMaximumConcurrency> sampled_logprobs{};   // new
};
```

The host already copies this whole POD each round at `decode_impl.h:45`, so the value arrives for
free. `valid_rows` is `TextConfig::token_domain`, not `logits.ne[0]` — rows past the token domain are
padding and must not enter the sum.

**Accumulation.** `program_impl.h:2486` does `sequence.ledger.push_back(token)` per lane per round.
That is the companion point: append the logprob to a per-request vector there. For the prefill's
first token, take it from the prefill site instead so the sequence has no gap.

**API surface.** Add beside `generated_token_ids` in `GenerationResult`
(`include/ninfer/types.h:409`):

```cpp
std::vector<float> generated_logprobs;
```

Populated in `src/runtime/engine/concurrent_executor.h:455`, where `generated_token_ids` is already
assigned. No wire-protocol change, no HTTP schema change; the serve layer simply ignores it.

**Make it unconditional, not flag-gated.** Cost is one extra ~304 KB read per decode step against
~17 GB of weight traffic — roughly 2 ppm. The reason to prefer unconditional is structural, not
performance: a conditional inside a captured CUDA graph forces separate graph variants and adds a
divergence risk for no measurable gain. If someone later wants it gated, that is a deliberate change
with a measurement attached, not a default.

### Stage 0: the harness

The **scoring half already exists and is unit-tested**: `scr_quality.{h,cpp}` in
`src/targets/qwen3_6/impl/runtime/`, tested by `tests/targets/qwen3_6/test_scr_quality.cpp`. It
implements the roadmap's exact thresholds and `evaluate_scr_prompt` **fails when the logprobs are
absent**, by design. Do not reimplement it, and do not re-specify the thresholds in another language
— one source of truth.

What is missing is the producer. Write it in **C++**, not Python: `evaluate_scr_quality` is C++ and
a Python harness would either duplicate the thresholds or parse text. `GenerationResult` gives the
harness both paths' logprobs and tokens directly, with no HTTP layer in between.

Shape it as a new backend on the `eval/ninfer_eval` `EvaluationBackend` protocol
(`eval/ninfer_eval/backends/base.py`), registered in `registry.py`. That is the roadmap's stated home
for Phase 5.5 and the plugin point already exists. The existing `evalscope` backend is a task
benchmark, so bolt SCR comparison onto it and the abstraction will not fit.

A run is, per prompt:

1. Send the **pre-edit** prompt to prime the cache.
2. Send the edited prompt with SCR enabled.
3. Send the edited prompt with SCR disabled.
4. Fill one `ScrQualitySample` from the two results and call `evaluate_scr_prompt`.

Two requirements on the suite that are easy to get wrong:

- **Chat-shaped, not synthetic.** The existing `tools/scr_e2e_check.py` uses filler sentences to get
  token counts to land on 64-token boundaries. That is the right tool for testing the wiring and the
  wrong tool for measuring quality. The suite should be a real system prompt plus multi-turn history,
  where the edit is a revised earlier message — that is the actual CLM use case. Support both a
  built-in deterministic generator (so the harness is self-contained and reproducible) and
  `--prompts <file>` for real traffic.
- **Sweep K.** The paper's Figure 13 grid is `K ∈ {1,2,3,6,12,64}` blocks, and that is the most
  valuable output: agreement-vs-K is a *curve*, and it is what tells you where to put the cap. A
  single binary "did it match" is nearly useless — one pass tells you nothing about the limit.

Then report with `format_scr_quality_json`, which emits one JSON line per prompt plus an aggregate in
a fixed key order so two runs diff.

**Honest limitation to state in any report.** This measures **argmax robustness**, not distribution
shift. High greedy agreement means SCR does not flip decisions; it does not mean the logits are
unperturbed. The logprob metric bounds the second, and the two together are what the roadmap asked
for.

**Also measure the regime that matters.** The e2e check that verified the fix ran with `base=0` and
`cache_hit=0`, so SCR reused 64 of 3864 tokens — it was saving about 1.7%. The configuration that
justifies the feature is a long reused prefix with an edit invalidating a long suffix. A suite that
does not exercise that measures SCR where it does nothing useful, and reports a meaningless speedup.

### The harness as it exists now (2026-10-08)

```powershell
build-ninja\apps\ninfer-scr-quality.exe --model .\models\qwen3_8_27b.ninfer `
  --work-dir scr-quality-run2 --keep-logs --scr-diag --flag-control --max-tokens 512
```

Four phases, each a fresh server on its own port, one prompt pair at a time:

| phase | flags | what its comparison is |
| --- | --- | --- |
| `scr` | `--scr --scr-mode tail` | the measurement |
| `baseline` | `--no-scr --no-prefix-reuse` | the reference |
| `control` | same as baseline | the **run-to-run floor**: what two identical runs disagree by |
| `flag` (opt-in) | same as `scr` plus `--scr-max-spans 0` | the **configuration floor**: what the SCR flags alone cost, with relocation switched off |

The two floors exist because the phases being compared differ in more than relocation. `control`
shares the baseline's flags and cannot see the configuration; `flag` carries the SCR flags and
relocates nothing, and the harness **asserts that it relocated no token on any request** (`0` is
accepted by `--scr-max-spans` for exactly this). The row that isolates relocation is `scr` against
`flag` — identical flags on both sides — which the `ATTRIBUTION` table prints per prompt beside both
floors. Read the measurement against the larger of the two, never against zero.

Everything the phases print — the harness's own sections, the runtime's `[scr-diag]` lines (written
inside the spawned server), and the server's banner, whose `graphs=` budget is how a phase-dependent
CUDA-graph capture set becomes visible — is teed to `<work-dir>/console.log` **and** the terminal.
The tee works at the descriptor level, not on a stream buffer, because the children inherit fd 1 and
2: a streambuf tee would silently drop every line a child writes. `--log -` turns it off.

With `thinking` on, a 64-token budget cuts every reply off mid-thought (`finish_reason=length`,
`content` empty), so `--max-tokens 512` is what makes "did both paths answer the same" answerable at
all. The `outputs` section diffs `content` and `reasoning` separately and prints the window at the
first differing character.

---

## 4. The bug that was found, and what it teaches

Worth reading because the failure modes are the reusable part.

**Symptom.** Every relocation produced a single stop token and empty content, while `--no-scr`
answered correctly — and it did so for a relocated run of **one** 64-token block out of 3864 (1.7%
of the prompt). Scaling to 2, 5 and 10 blocks changed nothing.

**Cause.** `advance_prefill` builds its `schedule::PrefillContext` from `staged.cursor` on entry
(`program_impl.h:2175`), which becomes `PrefillContext::text_kv_base` — the base position a chunk
prefills at. Further down *the same invocation*, `skip_relocated_runs` advances `staged.cursor`
across the relocated run (`program_impl.h:2216`). On the single entry where the skip fired, the two
diverged. With a one-block run the 24-token tail landed at positions `[3776, 3800)` instead of
`[3840, 3864)`: it overwrote the first 24 positions of the relocated page, the prompt's real last 24
positions were never written at all, and the sample came from the hidden state of position 3799,
which the prompt no longer ends at.

**Why nothing caught it.** Every component was correct, and individually verified. The diagnostic
fingerprinted the relocated page *across the re-rotation only*, so a page the prefill then
overwrote read as perfectly healthy — the thermometer was read before the fire. The fix was to
fingerprint after the prefill that consumes the pages.

Two lessons that cost real time, both worth carrying:

1. **Check the state at the moment it is consumed, not at the moment it is produced.** A diagnostic
   bracketing stage N cannot see damage done at stage N+1.
2. **Self-consistent tests cannot detect a convention disagreement.** The Op's own oracle test seeded
   the K plane with `paged_kv_element_offset` — the same addressing helper the Op uses — so it could
   only verify the rotation *given* a convention. The integration test routes every byte through
   `gqa_kv_append` and every read through the attention kernels and cleared the Op. When a test seeds
   through the same helper the code under test uses, ask who could disagree.

**Why "1 block" and "10 blocks" behaved identically.** This was a *positional* error, not a
*magnitude* error. Whether you relocate one page or ten, the tail lands too early by the size of the
relocated run — wrong either way. So damage is flat in relocated size. When something does not scale
with the quantity you would expect it to scale with, that is evidence the fault is positional, and
it is worth saying so explicitly rather than re-theorising "too much stale data".

---

## 5. Traps in this codebase

Each of these cost time during this work.

**`__shfl_*_sync` masks must be reached by every named lane.** An early `return` above the shuffle,
with a full mask, deadlocks the kernel — the sync waits on threads that exited. Guard only the *use*
of the partner value, never the shuffle itself. This shipped once and hung a test for an hour.
Corollary: a log-sum-exp merge is **not** idempotent (`merge(x,x)` doubles the sum), so unlike a
maximum reduction it needs the use guarded as well. `src/ops/kernel/argmax.cuh` is the reference for
the correct shape.

**Arithmetic simulation validates arithmetic, not concurrency.** A host-side simulation of a kernel
executed every branch for every lane and therefore agreed with the oracle both before and after a
deadlock fix. It is excellent for sizing tolerances and worthless for divergence bugs — those need a
device or `compute-sanitizer`. Do not over-trust it.

**An oracle must reduce the values the device actually reads.** A test that generated FP64 inputs,
staged BF16-rounded copies, and built its oracle from the originals was measuring the caller's
storage rounding, not the kernel. With logits near 5, a BF16 half-ulp is ~0.01 nats against a result
of magnitude ~9 — enough to fail a 2e-5 tolerance. Round identically before reducing.

**Diagnostic values are easy to derive from the wrong variable.** The chunk trace printed
`cursor=3840 nominal=24 visible=3864` and looked perfectly healthy while the kernel was using
`base=3776` and `visible=3800`, because `visible` was computed from the cursor rather than the base
the kernels use. When a diagnostic looks clean but the system is broken, suspect the diagnostic
first.

**`schedule_state.text_kv_base` and `staged.cursor` are two copies of one position.** They now cannot
diverge — `advance_prefill` throws if they do. If you add another consumer of the prefill base, put
it behind that invariant rather than reading the cursor.

**Bash is not a tool.** Use `shell`. And in Python, `str.replace` silently no-ops when the anchor does
not match; assert the match count before trusting any scripted edit. Three patches no-opped this way.

**`tensor.ne[]` mutation does not change a Tensor's identity.** A test that copies a correctly-typed
view and expects the wrapper to reject it tests nothing. Construct the wrong dtype explicitly.

---

## 6. References

- `docs/SCR_ROADMAP.md` — phases, thresholds, the Figure 13 `K` grid.
- `docs/SCR_RELOCATION_PATHS.md` — how relocation works and its verification status; §"Verification
  status" now records the root cause above.
- `tools/scr_e2e_check.py` — A/B harness that verifies the wiring. `--tail-blocks` sizes the
  relocated run in whole 64-token blocks; `--fact-in` places the answer in the prefix or the
  relocated run. Useful for bisecting, wrong for measuring quality.
- `tests/ops/test_scr_relocation_integration.cpp` — routes relocation through the production write
  and read paths, with a negative control. The model for what an integration test here should look
  like.
- `/home/user/projects/_cuda_check/` — sandbox-only, not in the repo: CUDA stubs, the two variant TUs
  for type-checking, and the float32 kernel simulations used to size tolerances.