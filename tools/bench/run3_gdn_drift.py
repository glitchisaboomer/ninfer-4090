#!/usr/bin/env python3
"""Compare the GDN and MoE fingerprints between two phases of the same run.

`post_prefill` reports per-layer `sum|x|` of the GDN conv and recurrent state. That is a
deterministic function of the state, so two legs that prefilled the same prompt from the same reset
must agree on it exactly. Where they do not, the divergence happened during prefill -- there is no
sampling, no decode and no speculation involved.

That turns the fingerprint from "a number to compare" into a localiser: the first layer index
where two legs disagree names the layer whose recurrence differed, which is the first thing worth
looking at in the kernel.

Usage:
    python tools/bench/run3_gdn_drift.py scr-quality-k2b
"""

from __future__ import annotations

import argparse
import pathlib
import re

LINE = re.compile(
    r"gdn stage=(\w+) .*?conv_samples \[([^\]]*)\] rec_samples \[([^\]]*)\]"
)


def parse(path: pathlib.Path) -> list[tuple[str, list[float], list[float]]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        stage = match.group(1)
        conv = [float(v) for v in match.group(2).split(",")]
        rec = [float(v) for v in match.group(3).split(",")]
        out.append((stage, conv, rec))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()

    path = args.run_dir / "console.log"
    if not path.exists():
        raise SystemExit(f"no console.log in {args.run_dir}")

    entries = parse(path)
    print(f"{len(entries)} post-prefill / invalidate fingerprint lines\n")

    print("Every leg that prefilled the same prompt from a cold reset. Two legs must agree")
    print("exactly on the GDN state; the deltas below are prefill nondeterminism, measured")
    print("before any token is sampled.\n")

    baseline = [e for e in entries if e[0] == "post_prefill"]
    if not baseline:
        raise SystemExit("no post_prefill entries")

    ref_conv, ref_rec = baseline[0][1], baseline[0][2]
    print(f"reference = first post_prefill leg: rec_samples {ref_rec}\n")

    header = f"{'#':>3} {'stage':<12} {'rec[0]':>12} {'d vs ref':>12} {'rec[1]':>11} {'d vs ref':>12}"
    print(header)
    for index, (stage, conv, rec) in enumerate(baseline):
        deltas = [abs(rec[i] - ref_rec[i]) for i in range(min(len(rec), len(ref_rec)))]
        print(f"{index:>3} {stage:<12} {rec[0]:>12.4f} {deltas[0]:>12.6f} "
              f"{rec[1]:>11.4f} {deltas[1]:>12.6f}")

    print("\nInterpretation:")
    print("  d == 0 for every leg      -> GDN recurrence is deterministic; the drift is elsewhere")
    print("  d > 0 and constant-ish      -> prefill is nondeterministic in the GDN layers themselves")
    print("  d > 0 only in the SCR leg  -> the drift is specific to the relocation path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())