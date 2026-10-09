#!/usr/bin/env python3
"""One line per (phase, prompt, request): what it generated and its first-token state.

The per-position table answers "how did the streams differ"; this answers "what did each run
actually do", which is the question a per-position diff cannot: a pair that parts at position 6
behind a 0.0-nat margin and a pair that parts behind a 3-nat margin look identical in a delta
column and are different findings.

Prints, per phase and prompt: the prime and edited completion lengths, the position-0 logprob
and margin of each, and the reuse path.  Length is here because a run that stops early bounds
every comparison made against it -- the harness compares only the overlap, so a short reply
silently shortens the evidence.

Usage:
    python tools/bench/run3_phase_summary.py scr-quality-run3
"""

from __future__ import annotations

import argparse
import json
import pathlib


PHASES = ("scr", "baseline", "control", "flag")


def load(path: pathlib.Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and json.loads(line).get("event") == "request_done":
                records.append(json.loads(line))
    return records


def summarize(record: dict, kind: str) -> str:
    result = record["result"]
    generated = result["generated"]
    logprobs = generated.get("logprobs") or []
    margins = generated.get("margins") or []
    tokens = generated.get("token_ids") or []
    first_lp = f"{logprobs[0]:+.5f}" if logprobs else "n/a"
    first_marg = f"{margins[0]:.3f}" if margins else "n/a"
    # The head of the reply is the cheapest way to see that two runs of one prompt are on
    # different rails, and it is readable without a tokenizer.
    head = " ".join(str(t) for t in tokens[:6])
    return (f"gen={result['completion_tokens']:>4} prompt={result['prompt_tokens']:>5} "
            f"lp0={first_lp} marg0={first_marg:>7} reloc={result['prefix_relocated_tokens']:>5} "
            f"path={result['prefix_reuse_path']:<18} tok[0:6]={head}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--k-blocks", default="1,2,3,6,12,64")
    parser.add_argument("--prompts-per-k", type=int, default=1)
    args = parser.parse_args()

    phases = {}
    for phase in PHASES:
        path: pathlib.Path = args.run_dir / f"{phase}.jsonl"
        if path.exists():
            phases[phase] = load(path)

    names = []
    for blocks in [int(v) for v in args.k_blocks.split(",")]:
        for variant in range(args.prompts_per_k):
            names.append(f"chat-k{blocks}-p{variant}")

    for index, name in enumerate(names):
        print(f"\n{name}")
        for phase, records in phases.items():
            if index * 2 + 1 >= len(records):
                continue
            prime = records[index * 2]
            edited = records[index * 2 + 1]
            print(f"  {phase:<9} prime  {summarize(prime, 'prime')}")
            print(f"  {phase:<9} edited {summarize(edited, 'edited')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())