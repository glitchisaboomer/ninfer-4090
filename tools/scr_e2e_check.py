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
that catches it is the work. `--scr-mode tail|segmented` selects which relocation shape to check.

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

# A KV page is 64 tokens, and the tail has to survive block alignment, so keep it well over
# one block. ~400 tokens of tail gives the planner several aligned blocks to relocate.
TAIL_SENTENCES = [
    "The maintenance log for the deployment records every change in order.",
    "Each entry lists the operator, the timestamp, and a short justification.",
    "Rollbacks were performed twice, both times within the same maintenance window.",
    "The verification suite ran after every rollout without exception.",
    "Throughput stayed within the agreed band during the whole observation period.",
    "No customer-visible incidents were attributed to these changes.",
]


def build_text(sentences, repeats):
    return " ".join(sentences * repeats)


def intro_text(repeats):
    return build_text(
        [
            "You are given a long technical briefing and must answer precisely.",
            "Read every section carefully before producing the final answer.",
            "The briefing describes an infrastructure migration in detail.",
            "Pay attention to the ordering of the steps and the stated constraints.",
            "Do not speculate beyond what the briefing actually says.",
        ],
        repeats,
    )



MIDDLE_A = (
    " SECTION ONE. The migration moves the primary datastore from a single node "
    "cluster to a three node cluster with a consensus protocol. "
    "All writes are routed through the new leader."
)


def middle_b(padding_words):
    """The same shape as MIDDLE_A but padded so its token count differs by a multiple of 64."""
    words = MIDDLE_A.split()
    # Swap in a different cluster size, then pad deterministically to the requested length.
    body = " ".join(words[:8] + ["five"] + words[9:12] + ["node"] + words[13:])
    filler = " ".join(["Additionally the rollout proceeds in careful stages."] * padding_words)
    return body + " " + filler


def request_prompt(variant, padding_words=0):
    """INTRO + MIDDLE + TAIL, where only the middle differs between variants."""
    middle = MIDDLE_A if variant == "a" else middle_b(padding_words)
    return (
        intro_text(6)
        + middle
        + "\n\n"
        + tail_text()
        + "\n\nIn one sentence, what number of nodes does the target cluster have?"
    )


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


def prefix_text(units):
    fact = PREFIX_FACT if FACT_PLACEMENT == "prefix" else ""
    return QUESTION + fact + _join(PREFIX_FILLER, PREFIX_REPEATS) + _pad(MIDDLE_FILLER, units)


def middle_text(units):
    return " " + _join(MIDDLE_FILLER, 1) + " " + _join(MIDDLE_FILLER[1:], 3 + units)


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
def request_prompt(variant, prefix_units=0, middle_units=0):
    """A = PREFIX + MIDDLE + TAIL, B = PREFIX + TAIL. Only A carries the middle."""
    head = prefix_text(prefix_units)
    if variant == "a":
        return head + middle_text(middle_units) + tail_text()
    return head + tail_text()


def solve_alignment(port, max_tokens, timeout, think):
    """Pad PREFIX and MIDDLE so TAIL starts on a 64-token boundary in both prompts.

    diff_prompt_edit matches whole 64-token blocks by hash, so TAIL is only recognised as a
    surviving run if it begins at a block boundary in the previous prompt as well as the current
    one. That needs len(PREFIX) and len(PREFIX)+len(MIDDLE) both divisible by 64. Token counts
    cannot be predicted from characters, so measure with the server's own tokenizer and search.
    """
    print("  aligning the surviving tail to a 64-token boundary...")

    prefix_units = 0
    for units in range(0, 60):
        # What has to be block-aligned is len(PREFIX) itself: that is where the surviving run
        # begins in prompt B, and the block grid is anchored at position 0 in both prompts.
        # Measuring the whole prompt B aligns prompt B, which is a different number, and it only
        # agrees when TAIL happens to be a whole number of blocks -- which is exactly the `--fact-in
        # prefix` case that failed to align at all.
        length = count_tokens(port, prefix_text(units), max_tokens, timeout, think)
        if length % BLOCK_TOKENS == 0:
            prefix_units = units
            print(f"    PREFIX padded with {units} unit(s) -> {length} tokens, block aligned")
            break
    else:
        print("    WARNING: could not block-align PREFIX in 60 attempts")
        return None, 0

    middle_units = 0
    for units in range(0, 80):
        # Same rule on the other side: len(PREFIX)+len(MIDDLE) is where the run begins in prompt A.
        length = count_tokens(port, prefix_text(prefix_units) + middle_text(units),
                              max_tokens, timeout, think)
        if length % BLOCK_TOKENS == 0:
            middle_units = units
            print(f"    MIDDLE padded with {units} unit(s) -> {length} tokens, block aligned")
            break
    else:
        print("    WARNING: could not block-align MIDDLE in 80 attempts")
        return prefix_units, None

    tokens_b = count_tokens(port, request_prompt("b", prefix_units), max_tokens, timeout, think)
    tokens_a = count_tokens(port, request_prompt("a", prefix_units, middle_units), max_tokens, timeout, think)
    # The run is everything in prompt B after the shared prefix. Measuring the prefix directly is
    # the only sound way to get this: the sentinel this used to split on ("SECTION ONE") does not
    # occur in these prompts at all, so the split returned the whole string and the estimate was
    # wrong by the size of the prompt (it reported 647 of 856 tokens relocated where the applied
    # plan relocated 128). That error also drove the "harsher regime than the paper" warning below,
    # which then argued against a run that was actually inside the paper's regime.
    prefix_only = count_tokens(port, prefix_text(prefix_units), max_tokens, timeout, think)
    tail_tokens = tokens_b - prefix_only
    share = 100 * tail_tokens // max(tokens_b, 1)
    print(f"    prompt A = {tokens_a} tokens, prompt B = {tokens_b} tokens, "
          f"MIDDLE = {tokens_a - tokens_b} tokens ({(tokens_a - tokens_b) // BLOCK_TOKENS} blocks)")
    # Approximate by construction: the tokenisation of the assembled prompt can differ from the
    # per-segment counts by a constant, and the applied plan drops the last partial page. Read
    # `span{... tok=...}` from the --scr-diag output for the number that was actually relocated.
    print(f"    relocated run is about {tail_tokens} tokens (~{tail_tokens // BLOCK_TOKENS} blocks), "
          f"roughly {share}% of the prompt (estimate; the applied plan is authoritative)")
    print(f"    (the paper reuses ~8% of prompt tokens beyond prefix-cache hits; a share far above "
          f"that is a much harsher regime than it measured)")
    return prefix_units, middle_units


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

            prefix_units, middle_units = solve_alignment(port, args.max_tokens,
                                                        args.request_timeout, args.think)

            print("  request A (PREFIX + MIDDLE + TAIL, primes the KV cache)...")
            _, _, usage_a, _ = post_completion(port, request_prompt("a", prefix_units or 0,
                                                                  middle_units or 0),
                                              args.max_tokens, args.request_timeout, args.think)
            time.sleep(1.0)

            print("  request B (PREFIX + TAIL: the middle is deleted, the tail survives)...")
            content_b, reasoning_b, usage_b, finish_b = post_completion(
                port, request_prompt("b", prefix_units or 0), args.max_tokens,
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
    global FACT_PLACEMENT, TAIL_REPEATS, PREFIX_REPEATS, TAIL_BLOCKS
    FACT_PLACEMENT = args.fact_in
    TAIL_REPEATS = args.tail_repeats
    PREFIX_REPEATS = args.prefix_repeats
    TAIL_BLOCKS = args.tail_blocks

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
    print()
    print(f"Pass 1: --scr --scr-mode {args.scr_mode}")
    scr_result, scr_records = run_config(args, ["--scr", "--scr-mode", args.scr_mode], "scr")
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
    for label in ("scr", "noscr"):
        log_path = os.path.join(tempfile.gettempdir(), f"ninfer_scr_{label}.log")
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            log_text = handle.read()
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

    if len(scr_records) < 2:
        problems.append(
            f"expected 2 completion records in the --scr run, saw {len(scr_records)}; "
            "check the server log for errors")
    else:
        second = scr_records[-1]
        if second["reuse"] != "suffix_relocation":
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

    print("RESULT: PASSED (SCR engaged and produced identical output)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
