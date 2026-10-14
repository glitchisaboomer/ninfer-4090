# Handoff: scr-free-relocation (`feat/scr-free-destination`)

**Status of this document**: handoff instructions for whoever picks up the
destination-free relocation work. Nothing here is speculative — every claim is
either verified locally (dry-run merge, file-level diff) or explicitly marked
as unverified.

---

## 1. Branch identity

- **Canonical branch name**: `feat/scr-free-destination`. There is no branch
  named `scr-free-relocation` on either remote; this document uses the name
  from the request for the same work stream.
- **Remotes**: published to both `origin` (private Gitea mirror) and
  `github` (`github.com/glitchisaboomer/ninfer-4090`). The two remotes carry
  identical content on deliberately different lineages: **Gitea holds the
  original commits**; **GitHub holds the identity-rewritten lineage** (every
  `agent@agent.com` commit re-authored as `guardiancreeperjp@gmail.com`,
  names preserved, trees untouched). New commits follow the same split —
  `agent <agent@agent.com>` on Gitea, `glitchisaboomer
  <guardiancreeperjp@gmail.com>` on GitHub — so a commit on one remote has no
  counterpart SHA on the other.
- **Branch head** (both remotes): the commit that adds this document.
- **Divergence point**: `ae371669` *docs: retire the two superseded SCR
  handoffs* — the commit directly below the oracle work. Nine commits exist on
  this branch since then (the WIP in §2). `feat/scr-step2-atomic-publish` has
  since been **merged into `feat/rtx-4090-sm89-native`** (Gitea PR #2, merge
  commit `2e68327c`; GitHub lineage `d5e93acf`), so its content is reachable
  from mainline, and mainline — not the feature branch — is this branch's
  merge source.

## 2. State of this branch (read before touching anything)

This branch is a checkpoint of in-flight work. Per its WIP commit (`8a46b404`),
it has **not been built and has not been run against the CTest suite or the
CUDA numerical oracle on the RTX 4090**. Treat it as unverified code.

What it implements: a per-logical-page packed relocated interval beside the
existing RoPE delta plane, so a copied SCR relocation can target arbitrary
token offsets instead of whole 64-token pages. BF16 decode and prefill
attention use the page delta for whole-page spans and recompute QK scores with
the shifted query only for keys inside a mixed page's interval.

Scope limits carried by the branch:

- **BF16 KV only.** INT8 and packed layouts still quantize destinations to
  whole pages.
- Deferred cleanup (correctness-neutral, fragile to future edits):
  1. The "is this page partial" predicate is re-derived from the packed value
     at five call sites instead of having one definition.
  2. The prefill boundary recompute rolls back an in-place `q_s` rotation via
     an `INT32_MIN` sentinel; it works but is fragile against changes to the
     reload early-out.
  3. `gqa_attention_decode_i8.cuh` threads a range pointer it never reads.

## 3. What must be merged from `feat/scr-step2-atomic-publish`

The nine commits below were the content of `feat/scr-step2-atomic-publish`.
They are no longer a branch to merge *from*: PR #2 merged them into
`feat/rtx-4090-sm89-native`, so **merging mainline into this branch is the
operation this handoff is about**. They deliver one thing — **the
copied-relocation device oracle** — plus four rounds of fixes to its own
harness, none of which has ever executed on a device.

### 3.1 The deliverable

| Commit | Subject |
|---|---|
| `9c5fa5b` | test(scr): order the harness's uploads against the stream it runs on |
| `cd5a752c` | test: check the harness's own write path before judging the relocation |
| `44abe78a` | test(scr): judge each side of the stored-key reading on its own |
| `e2600f54` | test(scr): report the stored-key deviation per logical page |
| `7e028f45` | test(scr): measure the copied cache's stored keys against the direct prefill |
| `aa0e590b` | test(scr): judge the copied-relocation control against the correct reading |
| `8389d210` | fix(test): index the copied-relocation oracle image by physical page |
| `5b1e7474` | docs: correct the copied-relocation handoff where the code said otherwise |
| `be11d38b` | test(scr): add the copied-relocation device oracle |

*SHAs shown are the Gitea lineage; GitHub's rewritten lineage carries
different SHAs for the same commits (branch tip `0d789e96`).*

### 3.2 Files it changes (complete list)

```
docs/SCR_COPIED_RELOCATION_HANDOFF.md    (rewritten; stop-and-report corrections)
tests/CMakeLists.txt                     (registers the new target)
tests/README.md                          (test inventory entry)
tests/ops/test_scr_copied_relocation.cpp (new, ~2050 lines)
```

Zero overlap with any file this branch touches. A dry-run merge
(`git merge --no-commit --no-ff` in an isolated worktree) completed with **no
conflicts**.

### 3.3 Why this merge is not optional

This is the verification path for this branch, not merely a nice-to-have:

1. **The oracle drives the same planner chain this branch modifies.** The test
   drives `diff_prompt_edit_shifted` → `plan_shifted_relocations` →
   `plan_copied_page_remap`, and this branch rewrites `prompt_edit.cpp`,
   `prompt_edit.h` and `request_plan_impl.h`. A textual merge is clean, but
   only the oracle holds the planner's output against independently verified
   geometry (three spans / 9 kept pages / 13 acquire slots / 8 moves /
   `from_pool` 7), so a semantic divergence introduced on this branch fails
   there instead of passing silently.
2. **The copied path itself is only exercised through this oracle.** The device
   copy was written at `a1c27b26` blind; the runtime branch that engages it is
   reached only by an off-grid span, and the e2e fixture aligns by
   construction — without this merge, nothing on this branch runs the code it
   changes.
3. **The suite grows from 84 to 85 CTests**; the AGENTS.md gate is defined
   against this binary.

## 4. Merge procedure

Run from a checkout of this branch:

```bash
git checkout feat/scr-free-destination
git merge --no-ff feat/rtx-4090-sm89-native
```

**This is no longer the clean merge it was on 2026-10-14.** That dry-run
predated PR #2; merging today's mainline conflicts in **21 files**, because
both sides have since moved the same code. Current conflict set (dry-run
against the Gitea lineage):

- add/add: `docs/SCR_ALIGNMENT_FREE_RELOCATION.md`,
  `docs/SCR_COPIED_RELOCATION_HANDOFF.md`, `src/core/paged_kv_delta_layout.h`,
  `tests/ops/test_gqa_kv_page_delta.cpp`,
  `tests/test_paged_kv_delta_layout.cpp`
- content: `src/core/paged_kv_cache.cu`, `src/core/paged_kv_cache.h`,
  `src/ops/kernel/gqa_attention_decode_bf16.cuh`,
  `src/ops/kernel/gqa_attention_prefill_bf16.cuh`,
  `src/ops/kernel/gqa_attention_prefill_common.cuh`,
  `src/ops/launcher/gqa_attention_decode.cu`,
  `src/ops/launcher/gqa_attention_prefill.cu`,
  `src/targets/qwen3_6/impl/runtime/program_impl.h`,
  `src/targets/qwen3_6/impl/runtime/prompt_edit.cpp`,
  `src/targets/qwen3_6/impl/runtime/prompt_edit.h`,
  `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`,
  `src/targets/qwen3_6/impl/state/decoder_state.cpp`,
  `tests/CMakeLists.txt`, `tests/README.md`,
  `tests/targets/qwen3_6/test_prompt_edit.cpp`, `tests/test_kv_cache.cpp`

The oracle TU itself (`tests/ops/test_scr_copied_relocation.cpp`) and its
CMake registration add cleanly. The add/add cluster means files this branch
created were independently created or superseded on mainline — each needs a
keep-both / keep-one decision, not a textual merge. Suggested commit message
once resolved:

```
test(scr): merge the copied-relocation oracle into the free-destination branch
```

## 5. Post-merge gate (Windows / RTX 4090 box)

This merge is the **first device execution** of the oracle — none of the nine
step2 commits has ever run on a GPU. All of the following are required, in
order, and any failure is a stop-and-report, not a work-around:

1. Configure and build with the usual `build-ninja/` CMake + Ninja flow.
   Note the new target links **both `ninfer_ops` and `ninfer_engine`** — the
   planner chain compiles into the latter, and a `ninfer_ops`-only line fails
   to link.
2. Run the full CTest suite: all **85** tests, zero failures.
3. Run `ninfer_scr_copied_relocation_test` specifically and read its output.
   It is device-only and gated by `SKIP_RETURN_CODE`; on a machine without a
   pool/stream it skips rather than fails — confirm it actually executed.
4. Expect groups 1–4 (bit-identical moved bytes on all four plane orders) to
   be the load-bearing assertions for this branch's changes; the attention
   groups (5–7) additionally pin the delta-plane publish/clear ordering this
   branch altered.

## 6. Known open items after the merge

- The three deferred cleanup items in §2 remain open on this branch
  (single definition of the partial-page predicate, remove the `INT32_MIN`
  rollback sentinel, drop the unread i8 range pointer).
- INT8 / packed-layout destinations still quantize to whole pages; the packed
  interval work is BF16-only.
- The prefill nondeterminism investigation (`docs/ISSUE_prefill-nondeterminism.md`)
  predates this branch and is untouched by the merge.
