#!/usr/bin/env python3
"""Position-0 logprob across phases, for every prompt and both requests.

Position 0's logprob is produced by one prefill forward pass: no decode, no sampler, no
speculation.  If two phases that were given identical flags disagree there, the divergence is in
prefill, and every downstream token difference is a consequence rather than a cause.

`baseline` and `control` run the same flags in the same process, so they are the cleanest pair
available; `flag` adds `--scr-max-spans 0`, which changes configuration and not numerics, and so
is the cleanest pair that spans a configuration change.

Usage:
    python tools/bench/run3_position0.py scr-quality-run3
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--k-blocks", default="1,2,3,6,12,64")
    args = parser.parse_args()

    phases = {}
    for phase in PHASES:
        path: pathlib.Path = args.run_dir / f"{phase}.jsonl"
        if path.exists():
            phases[phase] = load(path)

    count = min(len(records) for records in phases.values()) // 2
    blocks = [int(v) for v in args.k_blocks.split(",")]

    print("Position-0 logprob and token, per phase. Baseline/control/flag share flags except")
    print("`flag` also sets --scr-max-spans 0. A nonzero delta in the baseline-vs-control column")
    print("is prefill nondeterminism, because position 0 involves no decoding at all.\n")

    for which, label in ((0, "prime"), (1, "edited")):
        print(f"--- {label} request ---")
        header = f"{'prompt':<13}"
        for phase in PHASES:
            if phase in phases:
                header += f" {phase + ' lp@0':>14} {'tok':>6}"
        print(header)
        for index in range(count):
            row = f"chat-k{blocks[index]}-p0".ljust(13)
            for phase in PHASES:
                if phase not in phases:
                    continue
                generated = phases[phase][index * 2 + which]["result"]["generated"]
                logprobs = generated.get("logprobs") or []
                tokens = generated.get("token_ids") or []
                lp = logprobs[0] if logprobs else None
                tok = tokens[0] if tokens else None
                row += f" {(lp if lp is not None else float('nan')):>14.6f} {tok:>6}"
            print(row)

        print("\n  pairwise deltas at position 0:")
        print(f"  {'pair':<24} {'d lp@0':>12} {'token differs':>14}")
        for index in range(count):
            for left, right in itertools.combinations(PHASES, 2):
                if left not in phases or right not in phases:
                    continue
                a = phases[left][index * 2 + which]["result"]["generated"]
                b = phases[right][index * 2 + which]["result"]["generated"]
                la = (a.get("logprobs") or [None])[0]
                lb = (b.get("logprobs") or [None])[0]
                if la is None or lb is None:
                    continue
                ta = (a.get("token_ids") or [None])[0]
                tb = (b.get("token_ids") or [None])[0]
                print(f"  {('k%d ' % blocks[index]) + left + ' vs ' + right:<24} "
                      f"{abs(la - lb):>12.6f} {str(ta != tb):>14}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())