#!/usr/bin/env python3
"""Correlate the run3 divergence with the surviving-span geometry of each prompt.

The console's ATTRIBUTION table answers "is this bigger than the floor"; it cannot answer
"is this explained by *where* the reuse landed".  This recomputes, from the same request logs,
the two geometric quantities a relocated run actually depends on:

  rotated span vs. page-aligned tail   -- does the re-rotation cover the whole span, or does the
                                          last partial page leave a rotated-but-not-realigned row?
  rotation delta per K                 -- the RoPE delta the relocated K is re-rotated by.

The first is a hypothesis test with a concrete prediction: a run whose delta is 0 (the edit is
token-neutral) should re-rotate nothing, and any divergence it shows is coming from somewhere
else.  The second separates "rotation went wrong" from "rotation was a no-op that should not have
been".

Usage:
    python tools/bench/run3_geometry.py scr-quality-run3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

PAGE = 64


def load(path: pathlib.Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and json.loads(line).get("event") == "request_done":
                out.append(json.loads(line))
    return out


SPAN_RE = re.compile(r"span\{src=(\d+) dst=(\d+) tok=(\d+)")


def parse_scr_diag(console: pathlib.Path) -> list[dict]:
    """Every relocation plan the runtime actually printed, per request.

    Only the SCR phase prints these, and only when it relocated; a request that ran full_reset
    has no line, so the list is keyed by order-of-appearance rather than by request id.
    """
    plans = []
    for line in console.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[scr-diag] prompt_tokens=" not in line:
            continue
        entry = {"raw_spans": line.count("span{"), "spans": []}
        for src, dst, tok in SPAN_RE.findall(line):
            entry["spans"].append({"src": int(src), "dst": int(dst), "tok": int(tok)})
        match = re.search(r"head\{(\d+),(\d+)\} tail\{(\d+),(\d+)\}", line)
        if match:
            entry["head"] = [int(match.group(1)), int(match.group(2))]
            entry["tail"] = [int(match.group(3)), int(match.group(4))]
        plans.append(entry)
    return plans


def geometry(plan: dict) -> str:
    notes = []
    for span in plan["spans"]:
        delta = span["dst"] - span["src"]
        end = span["dst"] + span["tok"]
        last_page_end = ((end - 1) // PAGE + 1) * PAGE
        notes.append(
            f"src[{span['src']}..{span['src'] + span['tok']}) -> dst[{span['dst']}..{end}) "
            f"tok={span['tok']} delta={delta} "
            f"page_aligned_end={end % PAGE == 0} "
            f"rotates_into_page={last_page_end - end} row(s) past span end"
        )
    if "head" in plan:
        notes.append(f"head={plan['head']} (re-prefilled) tail={plan['tail']} (re-prefilled)")
    return "; ".join(notes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()

    console = args.run_dir / "console.log"
    plans = parse_scr_diag(console) if console.exists() else []
    scr = load(args.run_dir / "scr.jsonl")

    names = [f"chat-k{b}-p0" for b in (1, 2, 3, 6, 12, 64)]

    print("Relocation geometry actually applied, per SCR request that relocated:\n")
    plan_index = 0
    for index, name in enumerate(names):
        print(f"{name}")
        for kind in ("prime", "edited"):
            record = scr[index * 2 + (0 if kind == "prime" else 1)]["result"]
            relocated = record["prefix_relocated_tokens"]
            if plan_index < len(plans) and (relocated or index > 0):
                plan = plans[plan_index]
                if plan["raw_spans"] == relocated or (plan["raw_spans"] > 0 and relocated > 0):
                    print(f"  {kind:<7} relocated={relocated:<6} {geometry(plan)}")
                    plan_index += 1
                    continue
            print(f"  {kind:<7} relocated={relocated:<6} (no relocation line)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())