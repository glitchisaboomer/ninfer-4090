#!/usr/bin/env python3
"""
End-to-end check for Suffix Cache Reuse (SCR).

Verifies the one thing the unit tests cannot: that a relocated KV suffix produces the same
output as re-prefilling it. Runs the server twice -- once with --scr, once with --no-scr --
and compares.

The prompts are built so that request B shares a long prefix AND a long suffix with request A,
differing only in the middle. A prefills everything; B hits the prefix cache, hits an edit,
and finds the tail still resident after the edit point, so SCR relocates and re-rotates it
instead of re-prefilling.

The comparison is worthless unless SCR actually engaged, so this script fails loudly when the
server does not report reuse=suffix_relocation rather than reporting a vacuous pass.

Usage (from the repo root):
    python tools/scr_e2e_check.py --model "qwen3_8_27b.ninfer"

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


# The surviving tail. diff_prompt_edit matches whole 64-token blocks by hash, so this text has to sit
# at a 64-token boundary in both prompts. That requires the two middles to differ in length by a
# multiple of 64 tokens, which cannot be arranged by counting characters, so count_prompt_tokens
# below measures it against the server's own tokenizer.
def tail_text():
    return build_text(TAIL_SENTENCES, 12)


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
    message = result["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return content, reasoning, result["usage"]


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

MIDDLE_FILLER = [
    "SECTION ONE. The migration moved the primary datastore onto a replicated cluster.",
    "Consensus settings, leader election, and replica placement were all reviewed.",
    "Nothing in this section affects the rollback window.",
    "It is removed by the edit that request B applies.",
]

# The fact the question asks about, at the very end of the prompt so it is the surviving tail.
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
    return QUESTION + _join(PREFIX_FILLER, 4) + _pad(MIDDLE_FILLER, units)


def middle_text(units):
    return " " + _join(MIDDLE_FILLER, 1) + " " + _join(MIDDLE_FILLER[1:], 3 + units)


def tail_text():
    return " " + _join(TAIL_SENTENCES, 3)


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
        length = count_tokens(port, request_prompt("b", units), max_tokens, timeout, think)
        if length % BLOCK_TOKENS == 0:
            prefix_units = units
            print(f"    PREFIX padded with {units} unit(s) -> {length} tokens, block aligned")
            break
    else:
        print("    WARNING: could not block-align PREFIX in 60 attempts")
        return None, 0

    middle_units = 0
    for units in range(0, 80):
        length = count_tokens(port, request_prompt("a", prefix_units, units), max_tokens, timeout, think)
        if length % BLOCK_TOKENS == 0:
            middle_units = units
            print(f"    MIDDLE padded with {units} unit(s) -> {length} tokens, block aligned")
            break
    else:
        print("    WARNING: could not block-align MIDDLE in 80 attempts")
        return prefix_units, None

    tokens_b = count_tokens(port, request_prompt("b", prefix_units), max_tokens, timeout, think)
    tokens_a = count_tokens(port, request_prompt("a", prefix_units, middle_units), max_tokens, timeout, think)
    print(f"    prompt A = {tokens_a} tokens, prompt B = {tokens_b} tokens, "
          f"MIDDLE = {tokens_a - tokens_b} tokens ({(tokens_a - tokens_b) // BLOCK_TOKENS} blocks)")
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
    if os.path.exists(log_path):
        os.remove(log_path)

    # The caller's own arguments (artifact path, --kv-dtype, --max-context, ...) come first and the
    # flag under test is appended, so the script does not need to know how they normally launch.
    command = [args.exe] + args.base_args_list + extra_flags + ["--port", str(port)]
    print(f"  launching: {' '.join(extra_flags) or '(defaults)'}")
    print(f"  log: {log_path}")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    # The server log is redirected to a file rather than a pipe: on Windows a pipe buffer fills up
    # while we are waiting on HTTP and the server blocks, which looks like a hang.
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, cwd=args.cwd, creationflags=creationflags
        )
        try:
            healthy, why = wait_for_health(port, proc, args.startup_timeout)
            if not healthy:
                print(f"  server not healthy: {why}", file=sys.stderr)
                return None, []

            prefix_units, middle_units = solve_alignment(port, args.max_tokens,
                                                        args.request_timeout, args.think)

            print("  request A (PREFIX + MIDDLE + TAIL, primes the KV cache)...")
            _, _, usage_a = post_completion(port, request_prompt("a", prefix_units or 0,
                                                                middle_units or 0),
                                            args.max_tokens, args.request_timeout, args.think)
            time.sleep(1.0)

            print("  request B (PREFIX + TAIL: the middle is deleted, the tail survives)...")
            content_b, reasoning_b, usage_b = post_completion(
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

    artifact = next((a for a in args.base_args_list if a.lower().endswith(".ninfer")), None)
    if artifact is None:
        print("error: --base-args does not contain a .ninfer artifact path", file=sys.stderr)
        return 2
    if not os.path.exists(os.path.join(args.cwd, artifact)):
        print(f"error: model artifact not found: {artifact} (cwd {args.cwd})", file=sys.stderr)
        return 2

    print("SCR end-to-end check")
    print()

    # gqa_kv_rerotate validates the Text KV plane and rejects every quantized layout, and the
    # planner has no matching guard, so a quantized --kv-dtype plans a relocation that then throws.
    if not any(a.lower() == "--kv-dtype" and args.base_args_list[i + 1].lower() == "bf16"
               for i, a in enumerate(args.base_args_list) if i + 1 < len(args.base_args_list)):
        print("WARNING: --kv-dtype bf16 was not found in --base-args.")
        print("         The relocation Op requires an uncompressed BF16 Text KV plane and rejects")
        print("         int8 / rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8. The planner has no guard for")
        print("         this, so it will plan a relocation and the request will then fail.")
        print()
    print("Pass 1: --scr --scr-mode tail")
    scr_result, scr_records = run_config(args, ["--scr", "--scr-mode", "tail"], "scr")
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
        for needle, hint in (
            ("must be BF16", "SCR requires --kv-dtype bf16; quantized KV planes are rejected"),
            ("not supported", "the KV plane layout is one the relocation Op refuses"),
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

    text_scr = scr_result["content"].strip()
    text_ref = ref_result["content"].strip()
    reason_scr = scr_result["reasoning"].strip()
    reason_ref = ref_result["reasoning"].strip()

    print()
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
