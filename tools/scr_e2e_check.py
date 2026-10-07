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


def middle_text(variant):
    if variant == "a":
        return (
            " SECTION ONE. The migration moves the primary datastore from a single node "
            "cluster to a three node cluster with a consensus protocol. "
            "All writes are routed through the new leader."
        )
    return (
        " SECTION ONE. The migration moves the primary datastore from a single node "
        "cluster to a five node cluster with a consensus protocol. "
        "All reads are served from the nearest replica."
    )


def request_prompt(variant):
    """INTRO + MIDDLE + TAIL, where only the middle differs between variants."""
    return (
        intro_text(6)
        + middle_text(variant)
        + "\n\n"
        + build_text(TAIL_SENTENCES, 12)
        + "\n\nIn one sentence, what number of nodes does the target cluster have?"
    )


def post_completion(port, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"], payload["usage"]


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

            print("  request A (primes the KV cache)...")
            post_completion(port, request_prompt("a"), args.max_tokens, args.request_timeout)
            time.sleep(1.0)

            print("  request B (shares prefix and tail, differs in the middle)...")
            text_b, usage_b = post_completion(
                port, request_prompt("b"), args.max_tokens, args.request_timeout
            )
            time.sleep(1.0)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        records = parse_done_lines(handle.read())

    return {"text": text_b, "usage": usage_b}, records


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
    parser.add_argument("--max-tokens", type=int, default=48)
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
        elif second["cache_hit"] == 0:
            problems.append("the --scr run reported cache_hit=0, so the shared prefix was not reused")

    if len(ref_records) < 2:
        problems.append(f"expected 2 completion records in the --no-scr run, saw {len(ref_records)}")
    else:
        second = ref_records[-1]
        if second["reuse"] == "suffix_relocation":
            problems.append("--no-scr still reported suffix_relocation; the flag is not being honoured")

    text_scr = scr_result["text"].strip()
    text_ref = ref_result["text"].strip()

    print()
    print("request B, --scr:")
    print(f"  {text_scr!r}")
    print("request B, --no-scr:")
    print(f"  {text_ref!r}")
    print()
    print(f"usage --scr    : {scr_result['usage']}")
    print(f"usage --no-scr : {ref_result['usage']}")

    print()
    if text_scr == text_ref:
        print("RESULT: outputs are identical.")
    else:
        problems.append("outputs differ between --scr and --no-scr")

    if problems:
        print("RESULT: FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("RESULT: PASSED (SCR engaged and produced identical output)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
