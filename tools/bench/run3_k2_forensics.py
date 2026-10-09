#!/usr/bin/env python3
"""Read-only forensics over a scr-quality run directory.

Answers the question the run3 console cannot: *which token positions* moved, by how much,
and which phase produced each side of the pair.  The console prints one flip per prompt; this
prints the whole prefix, per position, for every prompt and every phase pair, so a single
outlier position (or a whole class of them) is visible instead of inferred.

Pairs printed, all aligned by the edited request of each prompt:
  measurement      scr        vs baseline
  control          baseline   vs control      (the noise floor: same flags, two runs)
  config_floor     flag       vs baseline      (SCR flags on, relocation off)
  relocation_only  scr        vs flag          (identical flags: relocation's own effect)

Usage:
    python tools/bench/run3_k2_forensics.py scr-quality-run3 [--limit 10] [--prompt k2]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


def load_completions(path: pathlib.Path) -> list[dict]:
    """Every ``request_done`` record of one phase's request log, in write order.

    The log interleaves ``request_start``, ``throughput`` and ``request_done``, so records are
    filtered by ``event`` rather than by position.
    """
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "request_done":
                records.append(record)
    return records


def edited_records(records: list[dict]) -> list[dict]:
    """The edited request of every prompt pair, positionally.

    Each prompt is sent as [prime, edited], and the phases are sequential and single-concurrency,
    so pairing by even index is what the harness itself does.  A wrong count is reported rather
    than silently truncating: a mis-paired table would point at the wrong prompt.
    """
    if len(records) % 2 != 0:
        raise SystemExit(f"odd completion count {len(records)}: [prime, edited] pairing is broken")
    return records[1::2]


def prompt_names(count: int, k_blocks: list[int], per_k: int) -> list[str]:
    names = []
    for blocks in k_blocks:
        for variant in range(per_k):
            names.append(f"chat-k{blocks}-p{variant}")
    return names[:count]


def token_series(record: dict) -> tuple[list, list, list]:
    generated = record["result"]["generated"]
    return (
        generated.get("token_ids") or [],
        generated.get("logprobs") or [],
        generated.get("margins") or [],
    )


def reuse_of(record: dict) -> str:
    result = record["result"]
    return (f"prompt={result['prompt_tokens']} relocated={result['prefix_relocated_tokens']} "
            f"hit={result['prefix_cache_hit_tokens']} path={result['prefix_reuse_path']}")


def fmt(value) -> str:
    if value is None:
        return "     n/a"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "     n/a"
    return f"{value:8.5f}"


def token_text(token_id) -> str:
    return f"{token_id:>7}" if isinstance(token_id, int) else "      ?"


def compare(label: str, left: dict, right: dict, left_name: str, right_name: str,
            name: str, limit: int) -> None:
    left_tokens, left_logprobs, left_margins = token_series(left)
    right_tokens, right_logprobs, right_margins = token_series(right)
    overlap = min(len(left_tokens), len(right_tokens), len(left_logprobs), len(right_logprobs))
    if overlap == 0:
        print(f"\n  {name}: no comparable tokens")
        return

    deltas = []
    for index in range(overlap):
        left_value = left_logprobs[index]
        right_value = right_logprobs[index]
        if left_value is None or right_value is None:
            continue
        deltas.append((index, abs(left_value - right_value), left_value, right_value))

    max_delta = max((item[1] for item in deltas), default=0.0)
    first_token = deltas[0][1] if deltas else 0.0
    mean_delta = sum(item[1] for item in deltas) / len(deltas) if deltas else 0.0
    flips = [index for index in range(overlap) if left_tokens[index] != right_tokens[index]]
    first_flip = flips[0] if flips else None

    print(f"\n  {name}   [{label}]")
    print(f"    {left_name}: {reuse_of(left)}")
    print(f"    {right_name}: {reuse_of(right)}")
    print(f"    comparable={overlap}  d(first)={first_token:.6f}  d(mean)={mean_delta:.6f}  "
          f"d(max)={max_delta:.6f}  first_flip={first_flip if first_flip is not None else '-'}  "
          f"flips={len(flips)}/{overlap}")
    print(f"    {'pos':>4}  {'tok ' + left_name:>7}  {'tok ' + right_name:>7}  "
          f"{'lp ' + left_name:>9}  {'lp ' + right_name:>9}  {'|delta|':>9}  "
          f"{'marg ' + left_name:>9}  {'marg ' + right_name:>9}  flip")

    window = range(min(limit, overlap))
    for index in window:
        _, delta, left_value, right_value = deltas[index]
        flip = "*" if left_tokens[index] != right_tokens[index] else " "
        print(f"    {index:>4}  {token_text(left_tokens[index])}  {token_text(right_tokens[index])}  "
              f"{fmt(left_value)}  {fmt(right_value)}  {fmt(delta)}  "
              f"{fmt(left_margins[index] if index < len(left_margins) else None)}  "
              f"{fmt(right_margins[index] if index < len(right_margins) else None)}  {flip}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=10,
                        help="positions printed per prompt (default 10)")
    parser.add_argument("--prompt", default=None,
                        help="substring filter, e.g. k2")
    parser.add_argument("--k-blocks", default="1,2,3,6,12,64")
    parser.add_argument("--prompts-per-k", type=int, default=1)
    args = parser.parse_args()

    run_dir: pathlib.Path = args.run_dir
    phases = {}
    for phase in ("scr", "baseline", "control", "flag"):
        path = run_dir / f"{phase}.jsonl"
        if not path.exists():
            print(f"note: {path} missing, skipping the {phase} phase", file=sys.stderr)
            continue
        phases[phase] = edited_records(load_completions(path))

    if "scr" not in phases or "baseline" not in phases:
        raise SystemExit("need at least scr.jsonl and baseline.jsonl")

    names = prompt_names(len(phases["scr"]), [int(v) for v in args.k_blocks.split(",")],
                         args.prompts_per_k)

    for phase, records in phases.items():
        print(f"{phase:<9} edited records: {len(records)}")

    pairs = [
        ("measurement", "scr", "baseline", "scr", "base"),
        ("control", "baseline", "control", "base2", "base"),
        ("config_floor", "flag", "baseline", "flag", "base"),
        ("relocation_only", "scr", "flag", "scr", "flag"),
    ]

    for index, name in enumerate(names):
        if args.prompt and args.prompt not in name:
            continue
        print(f"\n{'=' * 100}\n{name}\n{'=' * 100}")
        for label, left_phase, right_phase, left_name, right_name in pairs:
            if left_phase not in phases or right_phase not in phases:
                continue
            if index >= len(phases[left_phase]) or index >= len(phases[right_phase]):
                continue
            compare(label, phases[left_phase][index], phases[right_phase][index],
                    left_name, right_name, name, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())