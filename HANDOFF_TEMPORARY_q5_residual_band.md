# TEMPORARY — DELETE THIS FILE AFTER THE GITEA RUN PASSES

**This file is temporary scaffolding.** It exists only to hand this change to whoever runs the build
and the CTest suite on a machine with `nvcc`. Once `ninfer_linear_add_q5_residual_band_test` and the
rest of the suite are green, delete this file (`git rm HANDOFF_TEMPORARY_q5_residual_band.md`) and
commit that on its own. Do not merge it. Do not treat it as documentation.

---

## What this commit is, in one paragraph

The Q5 `linear_add` residual epilogue writes its output with `atomicAdd` on `bf16` from two CTAs that
each own half of K. Nothing fixes the order of those two additions and bf16 addition is not
associative, so two otherwise identical prefills can leave the residual stream differing in its low
bits. That is the cause of the non-reproducible prefill documented in
`docs/ISSUE_prefill-nondeterminism.md`, and it is reachable at **every** prompt length because the
last prefill chunk is a short tail that lands in the small-T band.

The shipped default is still the nondeterministic one: `NINFER_Q5_RESIDUAL_SPLITS=1`
(`src/ops/linear/q5/q5_small_t_mma.cu:65-76`) is an **override, off by default**, not a fix. This
commit does not change that. It moves the routing policy into a CUDA-free header so it can be tested
without a toolchain, and adds the test that pins it. The decision to flip the default needs a
benchmark and is deliberately left open.

## Files

| File | Change |
|---|---|
| `src/ops/linear_add/q5/q5_linear_add_catalog.h` | **new.** The route tables, the support inventory, the schedule names, and the closedness check, moved out of the CUDA translation unit. Adds `q5_linear_add_schedule_owns_output_element()`, which states in code that exactly one schedule writes an output element from more than one place. Pure code motion, no behaviour change. |
| `src/ops/linear_add/q5/q5_linear_add_plan.h` | Includes the catalog; keeps only the CUDA-typed execution surface. |
| `src/ops/linear_add/q5/q5_linear_add_plan.cpp` | Delegates to the catalog. Net −81 lines. |
| `tests/ops/linear_add/test_q5_residual_band.cpp` | **new.** Host-only CTest. |
| `tests/CMakeLists.txt` | Adds `ninfer_add_host_test()` — no link libraries, strict FP options — and registers the test with it. |

## What was verified here, and what was not

Verified on the sandbox (Linux, `g++` only, **no nvcc, no GPU, no cmake**):

- the new test builds clean under `-Wall -Wextra` and **passes**;
- `ninfer_qwen3_6_scr_quality_test`, `ninfer_qwen3_6_prompt_edit_test` and
  `ninfer_qwen3_6_prefill_trace_hash_test` still build and pass, so moving the catalog did not break
  host-side logic that includes it;
- a **negative control**: widening the k=6144 order-dependent band from cols 2..13 to 2..16 makes the
  test fail with 8 errors, each naming exactly what changed. The test has teeth;
- the experiment was fully restored afterwards; `git status` shows only the five intended files.

**Not verified, because this machine cannot do it:**

- `q5_linear_add_plan.cpp` is a CUDA translation unit and was **never compiled here**. The refactor is
  mechanical, but it is unbuilt.
- The new `ninfer_add_host_test()` function and its registration were **never configured by CMake**;
  there is no `cmake` on this machine.
- No kernel behaviour, stream ordering, or atomic ordering was exercised. The test pins *which
  schedules are reachable* and *states the determinism of each*; it cannot show the two CTAs actually
  arriving in a different order, because that is a hardware scheduling property.

## What to run

```powershell
git pull
ninja -C build-ninja -j 32     # expect a full CUDA rebuild; tens of minutes is normal with --split-compile=0
```

Then the new test on its own, then the whole suite:

```powershell
ctest --test-dir build-ninja -R ninfer_linear_add_q5_residual_band_test --output-on-failure
ctest --test-dir build-ninja --output-on-failure
```

## What a pass looks like

```
q5 linear_add schedule catalog and residual band
part 1: catalog is total over cols 1..64
part 2: routes are pinned, and the order-dependent band is exactly what they are
part 3: the two-CTAs/one-element write, sized
  residual ~1..4   samples=20000 disagreements=142 (0.71%)  worst=3.125e-02 nats  bf16 step=3.125e-02  below the 0.05-nat roadmap bound
  residual ~4..16  samples=20000 disagreements=57  (0.28%)  worst=1.250e-01 nats  bf16 step=1.250e-01  ABOVE  the 0.05-nat roadmap bound
all checks passed
```

Note that **the suite count goes from 84 to 85.** AGENTS.md says "all 84 CTests must pass"; that
sentence needs updating to 85 as part of this change or a follow-up, but a CI gate pinned to 84 will
fail on the new test being added, not on anything being wrong.

## If something fails

| Symptom | Meaning | Action |
|---|---|---|
| CMake configure fails in `tests/CMakeLists.txt` | the new `ninfer_add_host_test()` function | read the error; the function is 15 lines and copies `ninfer_add_test` |
| `ninfer_linear_add_q5_residual_band_test` fails to compile | the catalog move; `q5_linear_add_plan.h` is included by `.cu` files that relied on the enum living there | the enum and both structs now come from the catalog header via `q5_linear_add_plan.h`, so any file that included the plan header still sees them; look for a file that included the *`.cpp`*'s anonymous namespace by accident |
| The new test fails at runtime | someone changed the route table | the failure output names the shape and both schedules; restore the table or update the expectations deliberately, with a benchmark beside it |
| Any other test regressed | the catalog move | the moved content is a verbatim copy; diff `q5_linear_add_catalog.h` against the `.cpp` history of this commit |

## The finding worth carrying forward

Part 3 of the test is the reason this commit exists as a test rather than as a fix.

A single re-ordered bf16 atomic step moves a logit by one bf16 step of the residual magnitude:
**0.031 nats at a residual of 1–4, and 0.125 nats at 4–16**, on roughly 0.5% of random K-splits.

The unexplained K2 anomaly in `docs/ISSUE_prefill-nondeterminism.md` is **0.0801 nats**, which is
inside that range. K2 therefore does **not** need a second, unknown cause: it is within reach of the
write this commit documents. The same numbers also mean the roadmap's `< 0.05 nats` threshold is not
reachable while this epilogue exists at all — which makes it a threshold problem, not an SCR problem.

## After the run

1. Confirm the suite is green.
2. Delete this file: `git rm HANDOFF_TEMPORARY_q5_residual_band.md`.
3. Decide the actual fix with a benchmark: either default `q5_residual_single_split()` to true, or
   route cols 2..13 / 2..16 to a schedule that owns its output element. Whichever is chosen, the
   test's pinned expectations must be updated in the same commit.
4. Update the "all 84 CTests" sentence in AGENTS.md to 85, or fold it into (2).
