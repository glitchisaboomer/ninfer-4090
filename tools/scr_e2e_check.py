#!/usr/bin/env python3
"""
End-to-end check for Suffix Cache Reuse (SCR).

Verifies the one thing the unit tests cannot: that a relocated KV suffix produces the same
output as re-prefilling it. Runs the server twice -- once with --scr, once with --no-scr --
and compares.

The prompts are built so that request B shares a long prefix AND a long suffix with request A,
differing only in the middle. A prefills everything; B hits the prefix cache, hits an edit,
and finds the tail still resident after the edit point, so SCR relocates it by publishing a
per-page RoPE delta instead of re-prefilling it.

The comparison is worthless unless SCR actually engaged, so this script fails loudly when the
server does not report reuse=suffix_relocation rather than reporting a vacuous pass.

It also checks that the engagement was real work: the request log's `computed_prefill_tokens` is the
engine's own count of the prompt tokens its prefill chunks ran, and for a relocated request that must
be the prompt minus the relocated run. A relocation that is planned but never applied still answers
correctly -- the re-prefilled pages hold what the relocation would have served -- so the only thing
that catches it is the work. `--scr-mode tail|segmented` selects which relocation shape to check, and
`--shape scope-a|interleaved` selects the fixture it runs on: the built-in Scope-A prompt has one
surviving run, which both shapes relocate identically, so `interleaved` supplies a prompt whose edit
leaves two separated survivors -- the one case that distinguishes them.

Run it once per supported --kv-dtype (`bf16`, `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`). The relocation
is a delta on the page, not a rewrite of the codes, so every layout whose attention kernels honour
that delta has to be checked; BF16 alone no longer exercises the path.

Usage (from the repo root):
    python tools/scr_e2e_check.py --model "qwen3_8_27b.ninfer" \
        --base-args "... --kv-dtype rk4v4-e8 ..."

Standard library only: no pip install.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def request_body(prompt, max_tokens, think):
    # Thinking is disabled by default. This model reasons at length, so a thinking run can burn the
    # whole token budget and never emit content, which turns the comparison into a test of how long
    # the model talked rather than of what it retrieved. With thinking off, both passes must emit the
    # answer directly and the KV contents decide whether it is right.
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if not think:
        body["reasoning_effort"] = "none"
    return body


def post_json(port, path, payload, timeout):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def count_tokens(port, prompt, max_tokens, timeout, think):
    """Token count for `prompt` under the server's own tokenizer and chat template."""
    payload = request_body(prompt, max_tokens, think)
    result = post_json(port, "/v1/messages/count_tokens", payload, timeout)
    return int(result["input_tokens"])


def post_completion(port, prompt, max_tokens, timeout, think):
    result = post_json(port, "/v1/chat/completions", request_body(prompt, max_tokens, think), timeout)
    choice = result["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return content, reasoning, result["usage"], choice.get("finish_reason")


BLOCK_TOKENS = 64

# The shape the planner's own unit tests prove viable is Scope A: [P R S] -> [P S]. The edit
# *deletes* a middle region R, and the surviving region S runs to the very end of the prompt. A
# test that instead grows the prompt, or that trails a question after S, exercises different
# planner branches and can silently decline to engage.
#
# So: PREFIX is a fixed question plus briefing, MIDDLE is deleted by request B, and TAIL carries
# the fact the question asks about and ends the prompt. B must answer identically to A, and it can
# only do that if the relocated tail KV is correct.
QUESTION = (
    "Read the maintenance log below and answer in one sentence: how many minutes was the "
    "approved rollback window?\n\n"
)

PREFIX_FILLER = [
    "The briefing that follows is a long operational record.",
    "It is repeated many times so that it occupies whole KV blocks.",
    "Read it carefully before answering the question at the start.",
    "Do not speculate beyond what the record actually states.",
]

# How many times the prefix block is repeated. This sets the *relocated fraction*, which is the
# quantity that actually governs whether SCR works. The paper reuses 7.8% of prompt tokens beyond
# prefix-cache hits (5.3 points from reasoning stripping, 2.5 from context edits). A short prefix
# with a long surviving tail relocates most of the prompt, a far harsher regime than anything the
# paper measured, and the stale-state error scales with the relocated fraction rather than with the
# absolute token count.
PREFIX_REPEATS = 4

MIDDLE_FILLER = [
    "SECTION ONE. The migration moved the primary datastore onto a replicated cluster.",
    "Consensus settings, leader election, and replica placement were all reviewed.",
    "Nothing in this section affects the rollback window.",
    "It is removed by the edit that request B applies.",
]

# Where the answer fact lives. "tail" puts it only in the surviving run, which is the harsh case:
# the linear-attention layers cannot see a relocated run at all, so the answer is unreachable for
# 48 of 64 layers. "prefix" puts it before the edit, where both the relocated KV and the rebuilt
# linear-attention state can reach it, which separates "relocation is broken" from "the
# approximation is too coarse at this relocated fraction".
FACT_PLACEMENT = "tail"

# Which fixture to run. `scope-a` is the built-in [P R S] -> [P S] prompt, whose edit leaves one
# surviving run -- the shape tail relocation is built for, and the one that cannot tell the two
# relocation modes apart. `interleaved` is the two-deletion fixture defined with the interleaved
# regions below, whose edit leaves two separated survivors, which is the only case where segmented
# relocation does something tail relocation does not.
SHAPE = "scope-a"

# The fact the question asks about. At the very end of the prompt when placed in the tail, so it is
# the surviving region.
TAIL_SENTENCES = [
    "The approved rollback window is fourteen minutes.",
    "Any rollback exceeding that window requires a change advisory.",
    "Two rollbacks were performed during the observation period.",
    "Both completed inside the approved window without incident.",
    "The verification suite ran after every rollout.",
    "Throughput stayed within the agreed band throughout.",
    "No customer-visible incident was attributed to these changes.",
    "The maintenance record closes with the operator sign-off.",
]

# The same fact, placed in the prefix instead so it survives the edit without being relocated.
PREFIX_FACT = (
    "The approved rollback window is fourteen minutes. "
    "Any rollback exceeding that window requires a change advisory.\n\n"
)


def _join(sentences, repeats):
    return " ".join(sentences * repeats)


PREFIX_PAD = "The record below is reproduced verbatim and in full."


def _pad(sentences, units):
    """Appends `units` copies of a dedicated filler to grow the segment deterministically.

    The filler is unique to the prefix so that padding cannot create a block that also occurs in
    the middle, which would give diff_prompt_edit a spurious survivor to latch onto.
    """
    return _join(sentences, 1) + " " + _join([PREFIX_PAD], units)


# The alignment shift, and why it exists. A pad is one filler's token count, and a walk over whole
# pad units can only reach the boundaries that count can reach: an even count per unit shares a
# factor with the 64-token block, which leaves half of them unreachable however long the search
# runs, and the fixture would then fail for a reason that has nothing to do with relocation. That
# was tolerable with two searches and is not with five. Appending a few copies of a one-word
# filler shifts the total by a small odd amount alongside every unit, so every boundary stays
# reachable whatever the filler's own count turns out to be. A single common word is one token in
# any BPE vocabulary, and four of them are far too few tokens to form a block of their own.
ALIGNMENT_SHIFT_WORD = "the"
ALIGNMENT_SHIFT_SPAN = 4

def _filler(word, units):
    """`units` copies of `word`, or nothing at all when `units` is zero."""
    if units <= 0:
        return ""
    return " " + _join([word], units)


def prefix_text(pad):
    """P, the shared prefix, padded by `pad` = (units, shift)."""
    units, shift = pad
    fact = PREFIX_FACT if FACT_PLACEMENT == "prefix" else ""
    return (QUESTION + fact + _join(PREFIX_FILLER, PREFIX_REPEATS) + _pad(MIDDLE_FILLER, units) +
            _filler(ALIGNMENT_SHIFT_WORD, shift))


def middle_text(pad):
    """Scope A's R1: the region prompt A carries and prompt B deletes, padded by `pad`."""
    units, shift = pad
    return (" " + _join(MIDDLE_FILLER, 1) + " " + _join(MIDDLE_FILLER[1:], 3 + units) +
            _filler(ALIGNMENT_SHIFT_WORD, shift))


# Size of the relocated run. TAIL_BLOCKS requests a whole number of 64-token blocks, which is the
# decisive knob: under the stale-state approximation a single block should be nearly harmless and
# the damage should grow with the count, whereas one block that is already fatal points at something
# structural rather than proportional to how much stale KV is reintroduced. TAIL_REPEATS is the
# fallback when TAIL_BLOCKS is unset.
TAIL_BLOCKS = None
TAIL_REPEATS = 3


def tail_text():
    sentences = TAIL_SENTENCES if FACT_PLACEMENT == "tail" else TAIL_SENTENCES[2:]
    repeats = TAIL_REPEATS
    if TAIL_BLOCKS is not None:
        per_repeat = max(1, len(sentences) * 11)
        repeats = max(1, -(-TAIL_BLOCKS * 64 // per_repeat))
    return " " + _join(sentences, repeats)


# --- The interleaved fixture ---------------------------------------------------------------
# Scope A's edit leaves one surviving run, which is exactly the shape `plan_tail_relocation`
# selects: one contiguous run, prefilled around on both sides. An edit that leaves survivors in
# two or more separated places is the case segmented mode exists for, and no prompt in this file
# or in the quality harness produces one -- every fixture has exactly one run after its edit, so
# both selectors return the same span, the same derived ranges, and the same savings, and no run
# has ever told the two apart. This shape does.
#
#     prompt A:  [P R1 S1 R2 S2]
#     prompt B:  [P R1' S1 R2' S2]      R1' and R2' are fresh text of comparable length
#
# S1 is a middle survivor: it does not run to the prompt end, so it is not the tail. S2 is the
# tail survivor and carries the fact the question asks about, so the answer is only reachable if
# the relocated KV is right. Two deletions, two survivors, the second one carries the fact and
# ends the prompt -- the Scope A discipline, because a prompt that grows or that trails a
# question after its last survivor exercises different planner branches and can silently decline
# to engage.
#
# Every region gets its own filler. diff_prompt_edit matches whole 64-token blocks by hash, so a
# sentence that appears in two regions can hand the diff a survivor that is not the one under
# test, and the plan then relocates the wrong text.
INNER_SENTENCES = [
    "The operator roster lists everyone credentialed for the maintenance window.",
    "Escalation runs through the duty engineer and then the service owner.",
    "Handovers are recorded together with the outstanding items carried forward.",
    "Nothing in this roster affects the approved rollback window.",
]

# R1 and R2 are the regions prompt A carries and prompt B replaces; R1' and R2' are the
# replacements. Comparable length, deliberately different text, so neither is a survivor.
FIRST_REMOVED_SENTENCES = [
    "The datastore migration moves the primary cluster to a three node consensus deployment.",
    "Leader election settings were reviewed alongside replica placement.",
    "This region is deleted by the edit the second request applies.",
]

FIRST_REPLACEMENT_SENTENCES = [
    "The storage tier migration moves the archive cluster to a five node quorum deployment.",
    "Replica placement and failure domain spread were reviewed together.",
    "This region replaces the first deleted region when the edit is applied.",
]

SECOND_REMOVED_SENTENCES = [
    "The release train publishes changes on a fixed fortnightly cadence.",
    "Freeze windows are announced two weeks in advance.",
    "This region is also removed by the edit.",
]

SECOND_REPLACEMENT_SENTENCES = [
    "The deployment pipeline publishes changes on a fixed weekly cadence.",
    "Change freezes are announced four weeks in advance.",
    "This region replaces the second deleted region when the edit is applied.",
]

# One filler per region, appended in whole units to grow that region deterministically. Each is
# unique to its region, for the block-collision reason above: padding is not allowed to invent a
# survivor, and no region may share a sentence with another.
INNER_PAD = "The roster entry is reproduced in full here."
FIRST_PAD = "The first removed region is padded out in full here."
FIRST_ALT_PAD = "The first replacement region is padded out in full here."
SECOND_PAD = "The second removed region is padded out in full here."
SECOND_ALT_PAD = "The second replacement region is padded out in full here."

# How many times each region's own sentences repeat. Roughly two blocks for the middle survivor
# and a couple of blocks per gap, so neither survivor dwarfs the other and `--scr-max-spans 1`
# has a real choice to make between them.
INNER_REPEATS = 2
FIRST_REPEATS = 1
SECOND_REPEATS = 1


def _region(sentences, filler, pad, repeats=1):
    """One region of the interleaved fixture: its sentences, then its padding.

    `pad` is a `(units, shift)` pair -- one unit is one copy of the region's own filler, and the
    shift appends the alignment words on top. The leading space is the separator from the region
    before it, the same join discipline `middle_text` and `tail_text` use.
    """
    units, shift = pad
    return " " + _join(sentences, repeats) + " " + _join([filler], units) + _filler(
        ALIGNMENT_SHIFT_WORD, shift)


def inner_text():
    """S1, the middle survivor: identical in both prompts, so it survives both deletions."""
    return _region(INNER_SENTENCES, INNER_PAD, (0, 0), INNER_REPEATS)


def first_removed_text(pad):
    """R1: the first removed region, present only in prompt A."""
    return _region(FIRST_REMOVED_SENTENCES, FIRST_PAD, pad, FIRST_REPEATS)


def first_replacement_text(pad):
    """R1': its replacement, present only in prompt B."""
    return _region(FIRST_REPLACEMENT_SENTENCES, FIRST_ALT_PAD, pad, FIRST_REPEATS)


def second_removed_text(pad):
    """R2: the second removed region, present only in prompt A."""
    return _region(SECOND_REMOVED_SENTENCES, SECOND_PAD, pad, SECOND_REPEATS)


def second_replacement_text(pad):
    """R2': its replacement, present only in prompt B."""
    return _region(SECOND_REPLACEMENT_SENTENCES, SECOND_ALT_PAD, pad, SECOND_REPEATS)


# How each shape's two prompts read, shown next to the request that runs them.
PROMPT_A_SHAPE = {
    "scope-a": "PREFIX + MIDDLE + TAIL, primes the KV cache",
    "interleaved": "P + R1 + S1 + R2 + S2, primes the KV cache",
}
PROMPT_B_SHAPE = {
    "scope-a": "PREFIX + TAIL: the middle is deleted, the tail survives",
    "interleaved": "P + R1' + S1 + R2' + S2: both regions are replaced, both survivors survive",
}


def request_prompt(variant, pads):
    """The prompt for `variant`, every region padded by `pads`.

    `pads` is five `(units, shift)` pairs in prompt order -- P, R1, R1', R2, R2'. Scope A has no
    interleaved regions, so it reads only the first two and leaves the rest at zero; the interleaved
    shape reads all five. One signature for both, so no caller has to know which shape is live.
    """
    pads = tuple(pads) + ((0, 0),) * (max(0, 5 - len(pads)))
    if SHAPE == "interleaved":
        head = prefix_text(pads[0])
        if variant == "a":
            return (head + first_removed_text(pads[1]) + inner_text() +
                    second_removed_text(pads[3]) + tail_text())
        return (head + first_replacement_text(pads[2]) + inner_text() +
                second_replacement_text(pads[4]) + tail_text())
    # Scope A: A = PREFIX + MIDDLE + TAIL, B = PREFIX + TAIL. R1 is the middle.
    head = prefix_text(pads[0])
    if variant == "a":
        return head + middle_text(pads[1]) + tail_text()
    return head + tail_text()


# How many pad units one alignment search may try before giving up, and how many shifts it tries
# inside each. Sixty was enough for the two original searches; the interleaved shape runs five of
# them and each spends one measurement per shift, so the budget covers a far wider range of
# reachable boundaries than the bare scan did. Every pad is a deterministic pair of counts, so two
# runs on the same shape reproduce the same prompts.
ALIGNMENT_BUDGET = 80


def _search_alignment(port, label, why, build, max_tokens, timeout, think):
    """Grow one region until the text it builds lands on a 64-token boundary.

    `build` takes a pad pair and returns the whole concatenation that has to be aligned, so what
    is measured is the real text rather than a sum of per-region estimates. Returns the pad that
    aligned it, or None when the budget ran out -- which the caller must treat as a failure, because
    a misaligned edit engages nothing and the run would report full_reset.

    One caveat on what "aligned" means here. `count_tokens` measures the whole rendered prompt,
    which is the build text *plus* the generation prompt the template appends after it, while the
    engine's block grid runs over the prompt itself. So a build whose count is a multiple of the
    block is not necessarily a build whose region boundary sits on the grid: the boundary lands
    `generation_prompt_tokens % 64` short of one. Both prompts share the offset, so the diff still
    finds the survivors, but it finds them from the first *whole* block inside each one and drops up
    to a block of their heads. That is why the fixtures keep their survivors at several blocks
    rather than one: a survivor barely longer than a block can lose its match entirely to this
    constant offset, which is exactly what a fixture tuned to the last block looks like when it
    silently declines to engage.
    """
    for units in range(0, ALIGNMENT_BUDGET):
        for shift in range(0, ALIGNMENT_SHIFT_SPAN):
            length = count_tokens(port, build((units, shift)), max_tokens, timeout, think)
            if length % BLOCK_TOKENS == 0:
                print(f"    {label} padded with {units} unit(s) + {shift} shift word(s) "
                      f"-> {length} tokens, block aligned ({why})")
                return (units, shift)
    print(f"    WARNING: could not block-align {label} in {ALIGNMENT_BUDGET} attempts "
          f"x {ALIGNMENT_SHIFT_SPAN} shifts")
    return None


def _alignment_targets(pads):
    """The boundaries that must each land on a 64-token block, and how to build the text that
    reaches each one.

    diff_prompt_edit matches whole blocks by hash on a grid anchored at position 0 in both prompts,
    so a survivor is only recognised where it *begins* at a block boundary in both. Scope A has one
    survivor, hence two boundaries. The interleaved shape has two survivors in two prompts, hence
    five: len(P), len(P)+len(R1), len(P)+len(R1'), and len(P)+len(R1/R1')+len(S1)+len(R2/R2') for
    where the tail survivor begins in each prompt. P is padded on its own so the two removed-region
    searches stay independent of each other rather than of the prefix.
    """
    if SHAPE == "interleaved":
        return [
            ("P", "the leading span begins in both prompts",
             lambda pad: prefix_text(pad)),
            ("R1", "S1 begins in prompt A",
             lambda pad: prefix_text(pads[0]) + first_removed_text(pad)),
            ("R1'", "S1 begins in prompt B",
             lambda pad: prefix_text(pads[0]) + first_replacement_text(pad)),
            ("R2", "S2 begins in prompt A",
             lambda pad: (prefix_text(pads[0]) + first_removed_text(pads[1]) + inner_text() +
                          second_removed_text(pad))),
            ("R2'", "S2 begins in prompt B",
             lambda pad: (prefix_text(pads[0]) + first_replacement_text(pads[2]) + inner_text() +
                          second_replacement_text(pad))),
        ]
    return [
        ("PREFIX", "the surviving run begins in both prompts",
         lambda pad: prefix_text(pad)),
        ("MIDDLE", "the survivor begins in prompt A",
         lambda pad: prefix_text(pads[0]) + middle_text(pad)),
    ]


def solve_alignment(port, max_tokens, timeout, think):
    """Pad the live shape's regions until every survivor starts on a 64-token boundary.

    Returns the five pads in prompt order -- P, R1, R1', R2, R2', each a `(units, shift)` pair --
    or None when any of them could not be aligned. The searches are printed in full because a
    misaligned edit is silent: the diff simply finds no surviving block and the run reports
    full_reset, which the engagement check would then fail on with nothing to say about why.
    """
    print("  aligning the surviving spans to 64-token boundaries...")

    pads = [(0, 0)] * 5
    for index, (label, why, build) in enumerate(_alignment_targets(pads)):
        pad = _search_alignment(port, label, why, build, max_tokens, timeout, think)
        if pad is None:
            return None
        pads[index] = pad

    tokens_a = count_tokens(port, request_prompt("a", pads), max_tokens, timeout, think)
    tokens_b = count_tokens(port, request_prompt("b", pads), max_tokens, timeout, think)
    if SHAPE == "interleaved":
        _report_interleaved(port, pads, tokens_a, tokens_b, max_tokens, timeout, think)
        return pads

    # The run is everything in prompt B after the shared prefix. Measuring the prefix directly is
    # the only sound way to get this: the sentinel this used to split on ("SECTION ONE") does not
    # occur in these prompts at all, so the split returned the whole string and the estimate was
    # wrong by the size of the prompt (it reported 647 of 856 tokens relocated where the applied
    # plan relocated 128). That error also drove the "harsher regime than the paper" warning below,
    # which then argued against a run that was actually inside the paper's regime.
    prefix_only = count_tokens(port, prefix_text(pads[0]), max_tokens, timeout, think)
    tail_tokens = tokens_b - prefix_only
    share = 100 * tail_tokens // max(tokens_b, 1)
    print(f"    prompt A = {tokens_a} tokens, prompt B = {tokens_b} tokens, "
          f"MIDDLE = {tokens_a - tokens_b} tokens ({(tokens_a - tokens_b) // BLOCK_TOKENS} blocks)")
    # Approximate by construction: the tokenisation of the assembled prompt can differ from the
    # per-segment counts by a constant, and the applied plan drops the last partial page. Read
    # `span{... tok=...}` from the --scr-diag output for the number that was actually relocated.
    print(f"    relocated run is about {tail_tokens} tokens (~{tail_tokens // BLOCK_TOKENS} blocks), "
          f"roughly {share}% of the prompt (estimate; the applied plan is authoritative)")
    print("    (the paper reuses ~8% of prompt tokens beyond prefix-cache hits; a share far above "
          "that is a much harsher regime than it measured)")
    return pads


def _report_interleaved(port, pads, tokens_a, tokens_b, max_tokens, timeout, think):
    """Print the interleaved fixture's own arithmetic, so a run can be checked by hand.

    Two survivors means two numbers to reconcile against the plan, and the pass criteria are written
    in terms of both: segmented relocates S1 in full plus S2 minus its final block, and tail mode
    relocates the longer of the two only. The measurements are of concatenations, so they carry the
    usual boundary-merge slack -- the applied plan and the engine's own count stay authoritative.
    """
    # Measure each region as the difference of two concatenations, so a boundary merge inside one
    # region cannot be attributed to another.
    after_r1 = count_tokens(
        port, prefix_text(pads[0]) + first_removed_text(pads[1]), max_tokens, timeout, think)
    after_s1 = count_tokens(
        port, prefix_text(pads[0]) + first_removed_text(pads[1]) + inner_text(),
        max_tokens, timeout, think)
    after_r2 = count_tokens(
        port, prefix_text(pads[0]) + first_removed_text(pads[1]) + inner_text() +
        second_removed_text(pads[3]), max_tokens, timeout, think)
    inner_tokens = after_s1 - after_r1
    tail_tokens = tokens_a - after_r2
    inner_blocks = inner_tokens // BLOCK_TOKENS
    tail_blocks = tail_tokens // BLOCK_TOKENS

    delta = tokens_a - tokens_b
    print(f"    prompt A = {tokens_a} tokens, prompt B = {tokens_b} tokens, "
          f"the two edits differ by {delta} tokens ({delta // BLOCK_TOKENS} blocks)")
    if delta % BLOCK_TOKENS != 0:
        print(f"    WARNING: prompt A and prompt B differ by {delta} tokens, which is not a whole "
              f"number of blocks; a per-dtype sweep's alignment guard would reject this fixture")
    print(f"    S1 is about {inner_tokens} tokens (~{inner_blocks} blocks), S2 about "
          f"{tail_tokens} tokens (~{tail_blocks} blocks)")
    # S2 loses its final block to the final-token rule whatever its length: the prompt ends with the
    # question S2 carries, so the last prompt token must be prefilled. S1 ends mid-prompt, at a
    # replacement that is itself prefilled, so it is unaffected. The trim only bites when the
    # rewrite boundary falls inside S2's last whole block -- if it sits past it, add 64 back, which
    # is why the estimate says "about" and the applied plan decides.
    segmented = inner_blocks * BLOCK_TOKENS + (tail_blocks - 1) * BLOCK_TOKENS
    single = max(inner_blocks, tail_blocks - 1) * BLOCK_TOKENS
    print(f"    segmented should relocate about {segmented} tokens (S1 whole, S2 minus its final "
          f"block); tail mode about {single} (the longer survivor alone)")
    print("    (estimates; the applied plan and the engine's own prefill count are authoritative)")


def wait_for_health(port, proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited early with code {proc.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return True, "ok"
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False, "timed out waiting for /health"


DONE_RE = re.compile(r"\[req (?P<id>[^\]]+)\] done .*?prompt=(?P<prompt>\d+) gen=(?P<gen>\d+) "
                     r"cache=(?P<cache>\d+) reuse=(?P<reuse>\w+)")


def read_request_records(path):
    """Per-request records from a `--request-log-jsonl` file the server wrote.

    The console lines carry the reuse path but not the work, and the work is what the savings
    assertion needs: `computed_prefill_tokens` is the engine's own count of the prompt tokens its
    prefill chunks ran, which for a relocated request is the prompt minus the relocated run.
    """
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a still-flushing server
            if record.get("event") != "request_done":
                continue
            result = record.get("result", {})
            records.append(
                {
                    "id": record.get("request", {}).get("request_id"),
                    "prompt_tokens": int(result.get("prompt_tokens", 0)),
                    "completion_tokens": int(result.get("completion_tokens", 0)),
                    "computed_prefill_tokens": int(result.get("computed_prefill_tokens", 0)),
                    "cache_hit": int(result.get("prefix_cache_hit_tokens", 0)),
                    "relocated": int(result.get("prefix_relocated_tokens", 0)),
                    "reuse": result.get("prefix_reuse_path"),
                }
            )
    return records


def parse_done_lines(text):
    records = []
    for line in text.splitlines():
        match = DONE_RE.search(line)
        if match:
            records.append(
                {
                    "id": match.group("id"),
                    "prompt_tokens": int(match.group("prompt")),
                    "gen_tokens": int(match.group("gen")),
                    "cache_hit": int(match.group("cache")),
                    "reuse": match.group("reuse"),
                }
            )
    return records


def scr_flags(args):
    """Pass 1's flags: relocation on, in the mode under test, with the cap when one was asked for.

    The cap goes after `--scr-mode` on purpose. The server parses left to right with last-one-wins,
    and the caller's own `--base-args` come earlier still, so an explicit flag always wins over
    whatever the base configuration says.
    """
    flags = ["--scr", "--scr-mode", args.scr_mode]
    if args.scr_max_spans is not None:
        flags += ["--scr-max-spans", str(args.scr_max_spans)]
    return flags


# `[scr-diag] prompt_tokens=798 base=0 rope_freq[0]=1e-06 rope_freq[17]=1e-05
#  span{src=896 dst=576 tok=192 delta=-320 src_pages=14..14 dst_pages=9..9}
#  ranges=2 [0,576) [768,798) remap[...]`
SPAN_RE = re.compile(r"span\{src=(?P<src>\d+) dst=(?P<dst>\d+) tok=(?P<tok>\d+) "
                     r"delta=(?P<delta>-?\d+)")
RANGE_ITEM_RE = re.compile(r"\[(?P<begin>\d+),(?P<end>\d+)\)")


def parse_relocation_plans(log_text):
    """Every applied relocation plan the `--scr-diag` output printed, in the order they ran.

    The runtime prints one line per request that relocated anything, and it is the only place the
    spans and the derived ranges are visible at all -- the request log carries the token counts but
    not the geometry. So this is where a plan's *shape* can be checked, as opposed to its
    accounting, which the savings assertion covers from the request log.
    """
    plans = []
    for line in log_text.splitlines():
        if not line.lstrip().startswith("[scr-diag]") or "prompt_tokens=" not in line:
            continue
        prompt = re.search(r"prompt_tokens=(\d+)", line)
        base = re.search(r"\bbase=(\d+)", line)
        ranges_header = re.search(r"\branges=(\d+)", line)
        if not (prompt and base and ranges_header):
            continue
        # Everything before `remap`, so the page mapping's own brackets can never be read as a
        # prefilled range.
        ranges_part = line.split("ranges=", 1)[1].split(" remap", 1)[0]
        spans = [{"src": int(m.group("src")), "dst": int(m.group("dst")),
                  "tok": int(m.group("tok"))} for m in SPAN_RE.finditer(line)]
        ranges = [(int(m.group("begin")), int(m.group("end")))
                  for m in RANGE_ITEM_RE.finditer(ranges_part)]
        plans.append({
            "prompt": int(prompt.group(1)),
            "base": int(base.group(1)),
            "spans": spans,
            "ranges": ranges,
            "ranges_reported": int(ranges_header.group(1)),
        })
    return plans


def _intervals(pairs):
    """`[(a, b), ...]` as `[a,b) [c,d)`, or `(none)` when there is nothing to show."""
    return " ".join(f"[{begin},{end})" for begin, end in pairs) or "(none)"


def check_relocation_plan(plan, expected_spans):
    """Structural checks on one applied plan; returns a list of complaints.

    The planner's own contract is that the prefilled ranges and the relocated spans tile the suffix
    `[base, prompt)` exactly -- a token is prefilled or relocated but never unaccounted -- and that
    every span is whole blocks in both prompts. That tiling is what keeps the summary's relocated
    count honest, so it is checked directly rather than inferred from the savings line, which only
    sees the totals. `expected_spans` is the count the mode claims: two separated survivors for
    segmented on the interleaved fixture, one run everywhere else.
    """
    problems = []
    label = f"plan(prompt={plan['prompt']}, base={plan['base']})"

    if len(plan["ranges"]) != plan["ranges_reported"]:
        problems.append(f"{label}: reported {plan['ranges_reported']} ranges but printed "
                        f"{len(plan['ranges'])}")
    if len(plan["spans"]) != expected_spans:
        problems.append(f"{label}: relocated {len(plan['spans'])} span(s), expected "
                        f"{expected_spans} -- {'segmented did not separate the two survivors, so ' if expected_spans > 1 else ''}"
                        f"the plan is not the shape this run claims to measure")

    # Tile the suffix with the ranges and the spans taken together, in prompt order.
    pieces = sorted([(begin, end, "range") for begin, end in plan["ranges"]] +
                    [(span["dst"], span["dst"] + span["tok"], "span") for span in plan["spans"]])
    cursor = plan["base"]
    for begin, end, kind in pieces:
        if begin < cursor:
            problems.append(f"{label}: {kind} [{begin},{end}) overlaps what precedes it at "
                            f"{cursor}")
        elif begin > cursor:
            problems.append(f"{label}: {kind} [{begin},{end}) leaves [{cursor},{begin}) "
                            f"unaccounted for")
        cursor = max(cursor, end)
    if cursor != plan["prompt"]:
        problems.append(f"{label}: ranges and spans cover up to {cursor}, not the prompt end "
                        f"{plan['prompt']}")

    for span in plan["spans"]:
        if span["src"] % BLOCK_TOKENS or span["dst"] % BLOCK_TOKENS or span["tok"] % BLOCK_TOKENS:
            problems.append(f"{label}: span src={span['src']} dst={span['dst']} "
                            f"tok={span['tok']} is not whole 64-token blocks")
        if span["dst"] + span["tok"] >= plan["prompt"]:
            problems.append(f"{label}: span dst={span['dst']} tok={span['tok']} reaches the "
                            f"final prompt token, which must always be prefilled")
    return problems


def run_config(args, extra_flags, label):
    """Launches the server with `extra_flags`, runs requests A then B, returns B's text and log."""
    port = args.port
    log_path = os.path.join(tempfile.gettempdir(), f"ninfer_scr_{label}.log")
    requests_path = os.path.join(tempfile.gettempdir(), f"ninfer_scr_{label}_requests.jsonl")
    for stale in (log_path, requests_path):
        if os.path.exists(stale):
            os.remove(stale)

    # The caller's own arguments (artifact path, --kv-dtype, --max-context, ...) come first and the
    # flag under test is appended, so the script does not need to know how they normally launch.
    # The request log is what the savings check reads: the console lines report the reuse path but
    # not the work, and only the work can distinguish "relocated" from "re-prefilled anyway".
    command = ([args.exe] + args.base_args_list + extra_flags +
               ["--port", str(port), "--request-log-jsonl", requests_path])
    print(f"  launching: {' '.join(extra_flags) or '(defaults)'}")
    print(f"  log: {log_path}")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    environment = dict(os.environ)
    if args.scr_diag:
        environment["NINFER_SCR_DIAG"] = "1"

    # The server log is redirected to a file rather than a pipe: on Windows a pipe buffer fills up
    # while we are waiting on HTTP and the server blocks, which looks like a hang.
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, cwd=args.cwd,
            env=environment, creationflags=creationflags
        )
        try:
            healthy, why = wait_for_health(port, proc, args.startup_timeout)
            if not healthy:
                print(f"  server not healthy: {why}", file=sys.stderr)
                return None, []

            pads = solve_alignment(port, args.max_tokens, args.request_timeout, args.think)
            if pads is None:
                print("  alignment failed: no survivor starts on a block boundary, so the edit "
                      "would engage nothing. Aborting rather than reporting a full_reset run as a "
                      "result.", file=sys.stderr)
                return None, []

            print(f"  request A ({PROMPT_A_SHAPE[SHAPE]})...")
            _, _, usage_a, _ = post_completion(port, request_prompt("a", pads),
                                               args.max_tokens, args.request_timeout, args.think)
            time.sleep(1.0)

            print(f"  request B ({PROMPT_B_SHAPE[SHAPE]})...")
            content_b, reasoning_b, usage_b, finish_b = post_completion(
                port, request_prompt("b", pads), args.max_tokens,
                args.request_timeout, args.think)
            time.sleep(1.0)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        records = parse_done_lines(handle.read())

    return {
        "content": content_b,
        "reasoning": reasoning_b,
        "usage": usage_b,
        "usage_a": usage_a,
        "finish": finish_b,
        "request_records": read_request_records(requests_path),
    }, records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", default=os.path.join("build-ninja", "apps", "ninfer-serve.exe"),
                        help="path to ninfer-serve")
    parser.add_argument("--base-args", required=True,
                        help="arguments you normally pass to ninfer-serve, e.g. "
                             r'"models\qwen3_8_27b.ninfer --kv-dtype bf16 --max-context 65536". '
                             "The script appends the flag under test and --port.")
    parser.add_argument("--cwd", default=".", help="working directory for the server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--fact-in", choices=("tail", "prefix"), default="tail",
                        help="where the answer fact sits. 'tail' puts it only in the relocated run, "
                             "which the linear-attention layers cannot see at all -- the harsh case. "
                             "'prefix' puts it before the edit, so both the relocated KV and the "
                             "rebuilt linear-attention state can reach it. If 'prefix' passes and "
                             "'tail' fails, relocation is working and the approximation is simply "
                             "too coarse at this relocated fraction.")
    parser.add_argument("--prefix-repeats", type=int, default=4,
                        help="length of the shared prefix, in repeats of the prefix block. This is "
                             "the knob for the relocated fraction, which is what governs whether "
                             "the stale-state approximation is tolerable. The paper reuses 7.8%% of "
                             "prompt tokens beyond prefix-cache hits; raise this until the reported "
                             "share lands in that range.")
    parser.add_argument("--tail-blocks", type=int, default=None,
                        help="size the relocated run to this many 64-token blocks. The decisive "
                             "experiment: a single block that is already fatal points at something "
                             "structural, while damage that grows with the count is the stale-state "
                             "approximation.")
    parser.add_argument("--tail-repeats", type=int, default=3,
                        help="length of the relocated run, in repeats of the tail block. The knob for "
                             "bisecting a mechanical fault, which fails at any size, from an "
                             "approximation that only bites once the run is a large share of the "
                             "prompt. The script prints the resulting token count and share.")
    parser.add_argument("--scr-mode", choices=("tail", "segmented"), default="tail",
                        help="relocation shape to pass to --scr (default: tail). 'segmented' "
                             "exercises the interleaved-survivor path, whose savings differ from "
                             "the one contiguous run 'tail' uses.")
    parser.add_argument("--shape", choices=("scope-a", "interleaved"), default="scope-a",
                        help="prompt fixture to run (default: scope-a). 'scope-a' is the built-in "
                             "[P R S] -> [P S] prompt, whose edit leaves one surviving run and so "
                             "cannot tell the two relocation modes apart. 'interleaved' builds "
                             "[P R1 S1 R2 S2] -> [P R1' S1 R2' S2], whose edit leaves two "
                             "separated survivors: segmented relocates both, tail only the longer "
                             "one, and that difference is the only thing that distinguishes them.")
    parser.add_argument("--scr-max-spans", type=int, default=None,
                        help="pass --scr-max-spans through to the --scr server (default: the "
                             "server's own). 1 collapses segmented mode onto the single longest "
                             "survivor, which is the choice tail mode makes, and so proves the cap "
                             "bites on the interleaved plan rather than on one run. 0 is the "
                             "flag-phase control -- the same server and fixture with relocation "
                             "switched off -- which must relocate nothing and still answer "
                             "identically, so anything else is the flags rather than relocation.")
    parser.add_argument("--scr-diag", action="store_true",
                        help="set NINFER_SCR_DIAG=1 on the server so it dumps the applied relocation "
                             "plan, the page remap, the bound block-table row, and whether the "
                             "relocated K plane is finite.")
    parser.add_argument("--think", action="store_true",
                        help="keep thinking enabled. Off by default: this model reasons at length, "
                             "so a thinking run can burn the whole token budget without ever "
                             "emitting content, which tests how long the model talked rather than "
                             "what it retrieved.")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()

    if not os.path.exists(args.exe):
        print(f"error: server executable not found: {args.exe}", file=sys.stderr)
        print("       pass --exe with the full path, or run from the repo root", file=sys.stderr)
        return 2

    # Split on whitespace. Deliberately not shlex: on Windows a backslash path would be mangled by
    # POSIX-style escape handling, and the server arguments here never need quoting.
    args.base_args_list = args.base_args.split()
    global FACT_PLACEMENT, TAIL_REPEATS, PREFIX_REPEATS, TAIL_BLOCKS, SHAPE
    FACT_PLACEMENT = args.fact_in
    TAIL_REPEATS = args.tail_repeats
    PREFIX_REPEATS = args.prefix_repeats
    TAIL_BLOCKS = args.tail_blocks
    SHAPE = args.shape

    artifact = next((a for a in args.base_args_list if a.lower().endswith(".ninfer")), None)
    if artifact is None:
        print("error: --base-args does not contain a .ninfer artifact path", file=sys.stderr)
        return 2
    if not os.path.exists(os.path.join(args.cwd, artifact)):
        print(f"error: model artifact not found: {artifact} (cwd {args.cwd})", file=sys.stderr)
        return 2

    print("SCR end-to-end check")
    print()

    # The relocation writes a per-page RoPE delta rather than re-rotating the stored K, so the
    # planner admits every KV dtype whose attention kernels honour that delta:
    # request_plan_impl.h's delta_path_supports_dtype returns BF16 or I8, and every quantized layout
    # below is I8. rk2v4-e8 is the one the engine cannot stand behind yet -- its K read path is a
    # different codec (e8_root_decode_8d_int8) whose oracle is not written -- so this script refuses
    # it instead of reporting a pass nobody can attribute. (The planner's own comment claims it too,
    # because the delta mechanism is codec-agnostic; the refusal here is about verification, not
    # about whether the rotation would be applied.)
    supported_kv_dtypes = ("bf16", "int8", "rk8v4", "rk4v4", "rk4v4-e8")
    kv_dtype = next((args.base_args_list[i + 1].lower() for i, a in enumerate(args.base_args_list)
                     if a.lower() == "--kv-dtype" and i + 1 < len(args.base_args_list)), None)
    if kv_dtype is None:
        print("error: --base-args must contain --kv-dtype <dtype>; supported: "
              + ", ".join(supported_kv_dtypes), file=sys.stderr)
        return 2
    if kv_dtype not in supported_kv_dtypes:
        print(f"error: --kv-dtype {kv_dtype} is not checked by this script; supported: "
              + ", ".join(supported_kv_dtypes), file=sys.stderr)
        return 2
    # Recorded in the output so a per-dtype sweep is distinguishable after the fact; the dtype itself
    # arrives through --base-args, so what is printed here is what the server was actually given.
    print(f"KV dtype under test: --kv-dtype {kv_dtype}")
    print(f"Relocation shape under test: --scr-mode {args.scr_mode}")
    print(f"Fixture under test: --shape {args.shape}")
    if args.scr_max_spans is not None:
        print(f"Span cap under test: --scr-max-spans {args.scr_max_spans}")
    print()
    print(f"Pass 1: {' '.join(scr_flags(args))}")
    scr_result, scr_records = run_config(args, scr_flags(args), "scr")
    if scr_result is None:
        return 1

    print()
    print("Pass 2: --no-scr (reference)")
    ref_result, ref_records = run_config(args, ["--no-scr"], "noscr")
    if ref_result is None:
        return 1

    print()
    print("=" * 72)
    print("per-request records")
    for label, records in (("--scr", scr_records), ("--no-scr", ref_records)):
        for record in records:
            print(f"  {label:<9} req={record['id']} prompt={record['prompt_tokens']} "
                  f"gen={record['gen_tokens']} cache_hit={record['cache_hit']} "
                  f"reuse={record['reuse']}")

    problems = []

    # A request that threw leaves no completion record, so surface the server's own error text
    # rather than making the user go read the log file.
    log_texts = {}
    for label in ("scr", "noscr"):
        log_path = os.path.join(tempfile.gettempdir(), f"ninfer_scr_{label}.log")
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            log_text = handle.read()
        log_texts[label] = log_text
        # The runtime's own account of what it did, emitted when NINFER_SCR_DIAG was set.
        diagnostic = [line for line in log_text.splitlines() if "[scr-diag]" in line]
        if diagnostic:
            print()
            print(f"runtime relocation diagnostic ({label}):")
            for line in diagnostic:
                print(f"  {line.strip()}")
        for needle, hint in (
            ("not supported", "the KV plane layout is one the relocation path refuses"),
            ("exceeds KV cache capacity", "lower --max-context for this run"),
            ("exceeds the page entitlement", "lower --max-context for this run"),
        ):
            if needle in log_text:
                problems.append(f"[{label}] server log contains '{needle}': {hint}")

    # --- The applied plan's shape ------------------------------------------------------------
    # The savings assertion below proves the prefill work matches the plan; this proves the plan is
    # the shape the mode claims to measure, which nothing else can see: the request log carries the
    # relocated token count but not the spans it came from. `--scr-diag` is the only place that
    # geometry is printed, so on the interleaved fixture the whole point of the run -- two
    # separated survivors, one prefilled range per gap -- is unobservable without it.
    expected_spans = 2 if (args.shape == "interleaved" and args.scr_mode == "segmented") else 1
    plans = parse_relocation_plans(log_texts.get("scr", ""))
    if args.scr_diag:
        if not plans:
            # `--scr-max-spans 0` is the flag-phase control: it is supposed to relocate nothing, so
            # a missing plan line is its expected output rather than a finding.
            if args.scr_max_spans != 0:
                problems.append("--scr-diag was set but no applied relocation plan was printed, so "
                                "the span and range geometry could not be checked")
        else:
            print("applied relocation plan(s):")
            for plan in plans:
                print(f"  prompt={plan['prompt']} base={plan['base']} "
                      f"spans={len(plan['spans'])} ranges={len(plan['ranges'])}")
                print(f"    relocated: {_intervals([(s['dst'], s['dst'] + s['tok']) for s in plan['spans']])}")
                print(f"    prefilled: {_intervals(plan['ranges'])}")
                problems.extend(check_relocation_plan(plan, expected_spans))
    elif args.shape == "interleaved":
        print("(pass --scr-diag to check the plan's shape: span count, range count, and the "
              "tiling between them)")

    if len(scr_records) < 2:
        problems.append(
            f"expected 2 completion records in the --scr run, saw {len(scr_records)}; "
            "check the server log for errors")
    else:
        second = scr_records[-1]
        if args.scr_max_spans == 0:
            # The flag-phase control: the same server, the same fixture, the one flag that differs
            # is the cap. The expected outcome is the inverse of every other run -- relocation
            # switched off, nothing reused -- so the engagement check is turned around rather than
            # reporting a vacuous pass as a failure.
            if second["reuse"] == "suffix_relocation":
                problems.append(
                    "--scr-max-spans 0 still reported suffix_relocation; the cap is not being "
                    "honoured, so the flag-phase control proves nothing")
            print("  control run: relocation is switched off, so nothing may be reused")
        elif second["reuse"] != "suffix_relocation":
            problems.append(
                f"the --scr run reported reuse={second['reuse']}, not suffix_relocation. "
                "SCR did not engage, so this comparison would be vacuous. "
                "Make the shared tail longer (>= a few 64-token blocks) and check that no "
                "turn-checkpoint path is taking precedence.")
        # cache_hit is prefix reuse only. For this edit shape the prefix does not match (the middle is
        # deleted, so the resident prompt is not a prefix of the new one), so reuse_base is 0 and
        # cache_hit is legitimately 0 even when relocation saved hundreds of tokens. The relocated
        # count is reported separately as prefix_relocated_tokens in the JSON completion record.

    if len(ref_records) < 2:
        problems.append(f"expected 2 completion records in the --no-scr run, saw {len(ref_records)}")
    else:
        second = ref_records[-1]
        if second["reuse"] == "suffix_relocation":
            problems.append("--no-scr still reported suffix_relocation; the flag is not being honoured")

    # --- Savings -------------------------------------------------------------
    # The reuse path says a run was relocated; this is the check that the work actually happened.
    # A relocation that is planned but never applied -- the staged walk prefilling the run it was
    # supposed to skip -- still produces correct output, because the re-prefilled pages hold the
    # same tokens the relocation would have served. Nothing else in this script would notice: the
    # engagement check above reads the plan, and the comparison below reads the answer. Only the
    # engine's own count of prefilled prompt tokens distinguishes the two, and that count is
    # `computed_prefill_tokens` in the request log, which is the engine's chunk accounting rather
    # than a figure derived from the plan.
    #
    # The invariant is exact, not a tolerance: the planner's ranges tile the prompt with the
    # relocated runs, so the prefill computes precisely the prompt minus the relocated tokens. A
    # mismatch in either direction is a finding -- too many means the run was re-prefilled, too few
    # means the plan claims reuse it did not perform.
    print()
    print("Savings (the engine's own count of prefilled prompt tokens)")
    scr_requests = scr_result.get("request_records", [])
    ref_requests = ref_result.get("request_records", [])
    if not scr_requests:
        problems.append(
            "the --scr server wrote no request records; enable --request-log-jsonl support and "
            "check for a permission or path problem")
    else:
        for record in scr_requests:
            prompt = record["prompt_tokens"]
            relocated = record["relocated"] if record["reuse"] == "suffix_relocation" else 0
            cache = record["cache_hit"] if record["reuse"] != "suffix_relocation" else 0
            expected = prompt - relocated - cache
            computed = record["computed_prefill_tokens"]
            skipped = prompt - computed
            print(f"  req {record['id']}: prompt={prompt} reuse={record['reuse']} "
                  f"relocated={record['relocated']} computed={computed} "
                  f"({100 * skipped // max(prompt, 1)}% of the prompt not prefilled)")
            # The engine's own relocated count, read even when the reuse path is not relocation:
            # a plan that downgraded itself must clear it, or a summary reports reuse that never
            # ran. `--scr-max-spans 0` is the case that must report exactly zero.
            if args.scr_max_spans == 0 and record["relocated"] != 0:
                problems.append(
                    f"request {record['id']} reported {record['relocated']} relocated tokens with "
                    f"--scr-max-spans 0; the cap is not being honoured, so the flag-phase control "
                    f"proves nothing")
            if computed != expected:
                detail = (f"request {record['id']} ({record['reuse']}): the prefill computed "
                          f"{computed} prompt tokens, the plan accounts for {expected} "
                          f"(prompt {prompt}, relocated {record['relocated']}, cache hit "
                          f"{record['cache_hit']})")
                if computed > expected:
                    problems.append(
                        detail + ". More was prefilled than the plan relocated: the relocated run is "
                        "being re-prefilled, so the reported reuse saves nothing.")
                else:
                    problems.append(
                        detail + ". Less was prefilled than the plan accounts for: the plan reports "
                        "reuse it did not perform.")
    if ref_requests:
        for record in ref_requests:
            expected = record["prompt_tokens"] - record["cache_hit"]
            if record["computed_prefill_tokens"] != expected:
                problems.append(
                    f"the --no-scr reference prefilled {record['computed_prefill_tokens']} tokens "
                    f"for a {record['prompt_tokens']}-token prompt with a "
                    f"{record['cache_hit']}-token cache hit; expected {expected}")
    else:
        problems.append("the --no-scr server wrote no request records")

    text_scr = scr_result["content"].strip()
    text_ref = ref_result["content"].strip()
    reason_scr = scr_result["reasoning"].strip()
    reason_ref = ref_result["reasoning"].strip()

    print()
    print(f"request B finish_reason: --scr={scr_result['finish']}  "
          f"--no-scr={ref_result['finish']}")
    print("request B, --scr:")
    print(f"  content   : {text_scr!r}")
    if reason_scr:
        print(f"  reasoning : {reason_scr[:200]!r}")
    print("request B, --no-scr:")
    print(f"  content   : {text_ref!r}")
    if reason_ref:
        print(f"  reasoning : {reason_ref[:200]!r}")
    print()
    print(f"usage --scr    : {scr_result['usage']}")
    print(f"usage --no-scr : {ref_result['usage']}")

    if not text_scr and not reason_scr:
        problems.append(
            "request B produced neither content nor reasoning; raise --max-tokens so the model has "
            "room to emit an answer")

    print()
    if text_scr == text_ref and reason_scr == reason_ref:
        print("RESULT: outputs are identical.")
    else:
        if text_scr != text_ref:
            problems.append("content differs between --scr and --no-scr")
        if reason_scr != reason_ref:
            problems.append("reasoning differs between --scr and --no-scr")

    if problems:
        print("RESULT: FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if args.scr_max_spans == 0:
        print("RESULT: CONTROL PASSED (relocation switched off: nothing reused, the work was a "
               "cold prefill, and the output still matched --no-scr)")
    else:
        print("RESULT: PASSED (SCR engaged and produced identical output)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
