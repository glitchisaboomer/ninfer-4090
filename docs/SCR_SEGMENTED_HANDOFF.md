# Handoff: segmented relocation — proving the interleaved path

For a model picking this up cold. Written after segmented mode was made executable and smoke-tested on
hardware. **Nothing is broken.** The open work is that segmented mode has never been *distinguished*
from tail mode, so every run so far proves only "it did not break" and none proves what it is for.

Branch: `feat/scr-step2-atomic-publish`. Build and run happen on the user's Windows/CUDA box; the
development sandbox has no `nvcc`, so changes can be type-checked there but not compiled or run.

---

## Read this first: what the problem actually is

An edit that leaves survivors in **two or more separated places** is the case segmented mode exists
for. `plan_tail_relocation` selects **one contiguous run** — with interleaved survivors they are not
contiguous, so it relocates only the longest one and re-prefills the rest. `plan_relocations`
(the segmented selector) relocates **every** survivor, up to `--scr-max-spans`, and the runtime
prefills one range per gap.

The problem: **both harnesses' prompt suites have exactly one surviving run after the edit.** One
candidate, one run, one head and one tail. The two selectors then return the *same* span, the same
derived ranges, and the same savings — which is exactly what the two hardware smoke runs of
2026-10-12 produced (both: `span{src=896 dst=576 tok=192 delta=-320} ranges=2 [0,576) [768,798)`,
both: `computed=606`, both: identical output). So the segmented path is exercised end to end — it
engages, it saves the work it claims, it answers identically — and the *interleaved* claim, the only
thing that separates it from tail, is still unmeasured.

The work is therefore **a fixture**, not a code change: a prompt whose edit leaves two separated
survivors, and the two runs that show the modes differing on it.

## State of play

| Item | Status |
|---|---|
| Segmented plan → runtime wiring (`44d183de`) | done; `relocation_prefill_ranges` derives the ranges from the spans, the staged walk consumes any number of them, `relocation_prefill_splits` charges one extra quantum per further range, inexpressible plans are refused rather than half-applied |
| Selector limits (final token, turn boundary, trim-not-drop) | done; same rules `plan_tail_relocation` enforces, host-tested in `tests/targets/qwen3_6/test_prompt_edit.cpp` |
| Accounting bug (a downgraded plan reporting reuse it never performed) | fixed in `44d183de`; the FullReset downgrade clears the relocated fields |
| Savings assertion + `--scr-mode` in the e2e script (`803dc4ec`) | done; asserts `computed_prefill_tokens == prompt - relocated` exactly, per request |
| Hardware smoke, both modes | passes, identical output, work matches the plan — but on a **one-survivor** suite, so it does not separate the modes |
| Interleaved fixture | **not written** — this is the work |
| `[scr-diag]` integrity line on quantized dtypes | **broken and unrelated to relocation** — see Traps |

## Part 0 — gather these facts before writing anything

Run them and confirm each. If a fact disagrees with what is written here, STOP and report; do not
adapt this document to the checkout.

1. The selector's contract, and that it now takes the prompt length and the turn boundary:

```bash
sed -n '96,130p' src/targets/qwen3_6/impl/runtime/prompt_edit.h
```

Expected: `plan_relocations(diff, resident_kv_tokens, prefix_reuse_tokens, prompt_tokens,
turn_rewrite_boundary, max_spans)`, with the leading span excluded, the source inside resident KV,
the destination at or after the frontier, and no span reaching `min(prompt_tokens - 1, boundary)`.

2. The three helpers that turn spans into an executable plan:

```bash
grep -n "relocation_prefill_ranges\|relocation_prefill_splits\|relocation_is_expressible" \
  src/targets/qwen3_6/impl/runtime/prompt_edit.h
```

3. That the staged walk is range-list driven, not two-range:

```bash
sed -n '239,254p' src/targets/qwen3_6/impl/runtime/program.h
grep -n "relocated_prefill_ranges" src/targets/qwen3_6/impl/runtime/request_plan_impl.h
```

4. The e2e script's live prompt builder and the alignment search it uses — **note there are two
`request_prompt` definitions and only the later one is live** (line 93 is a shadowed older
revision; see Traps):

```bash
grep -n "def request_prompt\|def solve_alignment" tools/scr_e2e_check.py
sed -n '263,315p' tools/scr_e2e_check.py
```

5. That the harnesses can already ask for segmented:
   - e2e script: `--scr-mode tail|segmented` (pass 1's flag, `803dc4ec`).
   - quality harness: `--server-args "... --scr-mode segmented"` — the harness hardcodes `tail`
     and appends your extra args after it, and the server parses left-to-right with last-one-wins
     (`serve_options.cpp:278`).

## Part 1 — the fixture

The e2e script is the right home: it owns the tokenizer-driven alignment search and the savings
assertion. Its prompts are deliberately Scope A (`[P R S] -> [P S]`, the survivor running to the
prompt end) because a prompt that grows, or that trails a question after the survivor, exercises
different planner branches and can silently decline to engage. The interleaved shape keeps that
discipline: **two deletions, two survivors, the second one carries the fact and ends the prompt.**

```
prompt A:  [P R1 S1 R2 S2]
prompt B:  [P R1' S1 R2' S2]      R1' and R2' are fresh text of comparable length
```

`S1` is a middle survivor (does not run to the prompt end), `S2` is the tail survivor and carries the
fact the question asks about. On this fixture:

- tail mode relocates **one** run (the longer of S1, S2, after the final-token trim) and prefills
  everything else;
- segmented mode relocates **both** (up to `--scr-max-spans`, default 6), prefilling three ranges.

That difference — same fixture, two runs, `relocated=` strictly larger under segmented — is the test.

### The alignment arithmetic, generalized

`diff_prompt_edit` matches whole 64-token blocks by hash, so a survivor is only recognised if it
begins on a block boundary **in both prompts**. `solve_alignment()` currently pads two targets; the
interleaved shape needs five, all measured with the server's own tokenizer through `count_tokens()`:

| boundary | where it is | pad what |
|---|---|---|
| `len(P)` | where S1 begins in **both** prompts | one search, as today |
| `len(P)+len(R1)` | where S1 begins in prompt A | search R1 |
| `len(P)+len(R1')` | where S1 begins in prompt B | search R1' |
| `len(P)+len(R1)+len(S1)+len(R2)` | where S2 begins in prompt A | search R2 |
| `len(P)+len(R1')+len(S1)+len(R2')` | where S2 begins in prompt B | search R2' |

Because every region is block-aligned, `len(A) - len(B)` is a multiple of 64 automatically — which is
what the quality harness's `NOT_PAGE_ALIGNED` guard checks for the scr phase.

Suggested shape in the script: `--shape scope-a|interleaved` selecting the builder, with
`request_prompt(variant, pads)` taking the five pad counts. Keep the padding deterministic (repeat a
fixed filler sentence, as `_pad` does) so two runs reproduce the same prompts.

## Part 2 — verification ladder

Run in this order, on the Windows box, and capture the raw output of each.

1. Host tests first — they cover the machinery this fixture exercises:

```powershell
ctest --test-dir build-ninja -R ninfer_qwen3_6_prompt_edit_test --output-on-failure
```

2. The interleaved fixture, segmented, bf16 first (bf16 because the `[scr-diag]` integrity line is
only meaningful there — see Traps). `--shape interleaved` is the flag Part 1 asks you to add; until
it exists this run falls back to the built-in Scope-A shape and proves nothing about interleaving.

```powershell
python tools\scr_e2e_check.py --exe build-ninja\apps\ninfer-serve.exe `
  --base-args ".\models\qwen3_8_27b.ninfer --kv-dtype bf16 --max-context 32768" `
  --scr-mode segmented --shape interleaved --scr-diag
```

3. The same fixture, tail mode. **It must relocate strictly less** than step 2.

4. Step 2 again with `--scr-max-spans 1`: segmented must then relocate only the longer survivor —
   which is the same choice tail mode makes, and it proves the cap is applied to the interleaved
   plan rather than to a single run.

5. Repeat step 2 on one quantized dtype (`int8`, then `rk4v4-e8`) — the relocation path is the same
   code for every dtype the planner admits, and this is the only shape where a mis-tiled range would
   show up as wrong output rather than as a clean refusal.

**Pass criteria for step 2, all of them:**

- the plan line reports **two spans and three ranges**:
  `span{...} span{...} ranges=3 [0,a) [b,c) [d,prompt)` with the tiling exact;
- the savings line accounts exactly:
  `computed == prompt - relocated` where `relocated == |S1| + |S2| - 64`. S2 loses exactly its final
  64-token block to the final-token rule (the last prompt token must be prefilled, because prompt
  logits can only come from a real forward pass), no matter how many blocks S2 spans. S1 is
  unaffected — it ends mid-prompt, at a replacement that is itself prefilled.
- `RESULT: outputs are identical`, exit 0;
- the flag phase equivalent — run the same server with `--scr-max-spans 0` and confirm it relocated
  nothing, so a difference between the two is relocation and not the flags.

The savings assertion is the detector for a mis-tiled plan: ranges that overlap a span, miss a gap,
or double-claim a destination all break `computed == prompt - relocated`, and a span that cannot be
laid out at all is refused by `relocation_is_expressible` before the runtime sees it.

## Traps

- **The built-in suites cannot separate the modes.** Every prompt in both harnesses has one surviving
  run. Two passing runs on them is the expected, uninformative result.
- **Dead code in the e2e script — resolved.** `tools/scr_e2e_check.py` used to define `request_prompt`
  twice, the earlier revision (with `MIDDLE_A`/`middle_b`/`intro_text`) unreachable behind the live
  one, along with two duplicate `ALIGNMENT_BUDGET` definitions. All of it was removed with the
  interleaved fixture, so this file now has one of each. If you are reading an older revision of the
  script, read from `request_prompt` down.
- **The `[scr-diag]` integrity line is meaningless on a quantized cache.** It decodes the K plane as
  BF16 words; on `int8`/`rk4v4-e8` the plane holds codes, so it reports `before=nan after=nan`,
  `non_finite=1294`, `range=[-3.4e38, 3.4e38]` and `summary{0/3 intact}` even when the relocation is
  correct. The bf16 run of 2026-10-12 read `3/3 intact` on the same machinery. Diagnostic-only —
  the outputs are what decide correctness — but a red verdict inside a green run trains readers to
  ignore the line. Either make it dtype-aware (`log_scr_relocation` in
  `src/targets/qwen3_6/impl/runtime/program_impl.h`) or have it print "not checked for this dtype".
- **A misaligned edit engages nothing.** If a pad search fails, the diff finds no surviving block and
  the run reports `full_reset`; the e2e script's engagement check fails loudly on that, but the
  alignment search itself prints only a warning when it cannot align in its attempt budget. Check
  the printed padding before believing a result.
- **Do not "fix" the selectors.** They are host-tested and correct; the tail planner's one-run
  contract and the segmented planner's N-run contract are both intended. The gap is coverage, not
  behaviour.

## Definition of done

A single fixture, two modes, one observable difference: on the same prompt, tail relocates one
survivor and segmented relocates both, both produce output identical to `--no-scr`, both account
exactly in the savings line, and `--scr-max-spans 1` collapses segmented onto tail's choice. At that
point the segmented claim — interleaved survivors relocated in one pass — is measured rather than
reasoned, and the remaining work is the quality harness's per-dtype sweep, which is a separate
documented item and deliberately out of scope here.
