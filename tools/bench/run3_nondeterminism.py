#!/usr/bin/env python3
"""Locate where run-to-run nondeterminism enters, from request logs alone.

The run3 report is UNATTRIBUTABLE because the control leg chose different tokens than the baseline
leg on three of six prompts.  That makes every floor in the run unusable, and it is a bigger problem
than the k2 anomaly the run was opened for.  This narrows it with the one observation the logs
already carry:

  position 0's logprob is the logits row produced by the *last prefill chunk* -- one forward pass,
  no decoding, no sampler, no speculation.  Two runs that agree at position 0 and part later are
  showing decode-stage nondeterminism.  Two runs that already differ at position 0 are showing
  prefill nondeterminism, which is a much tighter search space and points at the prompt path rather
  than at the sampling loop.

For each prompt and each adjacent pair of phases, prints the first position at which the two token
streams part and the magnitude of the logprob difference up to it.  A pair that parts at 0 with a
tiny delta is not the same finding as a pair that runs 100 identical tokens and then parts.

Usage:
    python tools/bench/run3_nondeterminism.py scr-quality-run3
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

PHASES = ("baseline", "control", "flag", "scr")


def load(path: pathlib.Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and json.loads(line).get("event") == "request_done":
                out.append(json.loads(line))
    return out


def series(record: dict):
    generated = record["result"]["generated"]
    return (generated.get("token_ids") or [], generated.get("logprobs") or [],
            generated.get("margins") or [])


def margin_at_tie(margins: list, index: int, threshold: float) -> bool:
    return index < len(margins) and margins[index] is not None and margins[index] <= threshold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--tie-margin", type=float, default=0.25,
                        help="a margin at or below this counts as a coin flip (default 0.25 nats)")
    parser.add_argument("--k-blocks", default="1,2,3,6,12,64")
    args = parser.parse_args()

    phases = {}
    for phase in PHASES:
        path: pathlib.Path = args.run_dir / f"{phase}.jsonl"
        if path.exists():
            phases[phase] = load(path)
    if "baseline" not in phases or "control" not in phases:
        raise SystemExit("need baseline.jsonl and control.jsonl")

    names = [f"chat-k{b}-p0" for b in [int(v) for v in args.k_blocks.split(",")]]
    count = min(len(records) for records in phases.values()) // 2

    print("Run-to-run nondeterminism, phase pair by phase pair (edited request).\n")
    print(f"{'prompt':<13} {'pair':<20} {'parted@':>8} {'agreed':>7} {'d(mean leading)':>17} "
          f"{'margin@part':>12} {'tie?':>5}")

    tie_count = 0
    total = 0
    for index in range(count):
        for left, right in itertools.combinations(PHASES, 2):
            if left not in phases or right not in phases:
                continue
            left_tokens, left_logprobs, left_margins = series(phases[left][index * 2 + 1])
            right_tokens, right_logprobs, right_margins = series(phases[right][index * 2 + 1])
            overlap = min(len(left_tokens), len(right_tokens))
            if overlap == 0:
                continue

            parted = next((i for i in range(overlap) if left_tokens[i] != right_tokens[i]), None)
            agreed = parted if parted is not None else overlap

            deltas = []
            for i in range(agreed):
                a, b = left_logprobs[i], right_logprobs[i]
                if a is not None and b is not None:
                    deltas.append(abs(a - b))
            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

            margin = "-"
            tie = ""
            if parted is not None:
                value = (left_margins[parted] if parted < len(left_margins)
                         else right_margins[parted] if parted < len(right_margins) else None)
                if value is not None:
                    margin = f"{value:.3f}"
                    if margin_at_tie(left_margins, parted, args.tie_margin) or \
                       margin_at_tie(right_margins, parted, args.tie_margin):
                        tie = "yes"
                        tie_count += 1
            total += 1
            print(f"{names[index]:<13} {left + ' vs ' + right:<20} "
                  f"{parted if parted is not None else '-':>8} {agreed:>7} {mean_delta:>17.6f} "
                  f"{margin:>12} {tie:>5}")

    print(f"\n{tie_count}/{total} pairs parted at a margin <= {args.tie_margin} nats.")
    print("A 'parted@0' row means prefill itself differed: the two runs never agreed on a single")
    print("token, so the divergence is upstream of sampling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())