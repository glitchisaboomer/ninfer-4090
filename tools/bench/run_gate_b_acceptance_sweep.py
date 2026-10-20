#!/usr/bin/env python3
"""Gate B acceptance/cost sweep for the Qwen3.8-27B RTX 4090 decode-speedup campaign.

Measures the one thing an op-level sandbox cannot: **acceptance by proposal position** across MTP
draft widths, plus the end-to-end per-round cost that goes with it. This is the gating unknown for
deciding whether wider speculation is worth revisiting after the corrected GDN cost model.

What it produces, per (workload, K) point:

  * `accepted_per_position` histogram  -> the shape of the decay, not just the mean
  * derived per-position acceptance q_i -> accepted_at_i / drafted_at_i, with exact denominators
  * tokens per round, rounds, drafted/accepted totals
  * decode tok/s and per-round wall time, from the full-precision request log

It starts and stops one server per point (execution graphs and the memory plan are startup-fixed,
so K cannot be changed in-process), waits for /health, issues the same requests at every K, then
parses the structured request log. It never reads generated text.

Usage (from the repo root, with the artifact present):

    python tools/bench/run_gate_b_acceptance_sweep.py \
        --server build/apps/Release/ninfer-serve.exe \
        --model  E:/models/qwen3_8_27b.ninfer \
        --out    profiles/bench/gate-b

Requirements: Python 3.10+, no third-party packages.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Workloads. Reasoning is the headline case; the free-form control is deliberately included because
# the roadmap's central unresolved contradiction is that AIME reasoning measured 73-83% acceptance
# while another campaign reported ~50%. Both must be measured on the same machine to settle it.
#
# Prompts are FIXED across K so the only variable is the draft width.
# ---------------------------------------------------------------------------

WORKLOADS: dict[str, dict[str, Any]] = {
    "reasoning": {
        "description": "Long-horizon chain-of-thought reasoning (the campaign target).",
        "prompts": [
            "Solve this step by step. Let n be the smallest positive integer such that "
            "n^2 + 3n + 5 is divisible by 77. Find n and justify each step.",
            "A sequence is defined by a_1 = 1 and a_{n+1} = a_n + 2n + 1 for n >= 1. "
            "Derive a closed form for a_n, prove it by induction, then compute a_100 modulo 1000.",
            "Prove that for any prime p > 3, p^2 - 1 is divisible by 24. Show each divisibility "
            "step separately and state which theorem you use at each point.",
            "Ten people sit around a table. Each shakes hands with exactly three others, and no "
            "two pairs shake hands twice. Determine whether this is possible, and if so give an "
            "explicit arrangement; otherwise prove impossibility.",
        ],
    },
    "freeform": {
        "description": "Free-form prose control. The roadmap measured Story acceptance as low as "
                       "38%, so this is the low-acceptance arm.",
        "prompts": [
            "Write a short story about a lighthouse keeper who discovers something unexpected "
            "washed ashore one winter morning. Keep it under two hundred words.",
            "Describe the smell of a city street just after rain, in the voice of someone who has "
            "just moved there and does not yet know anyone.",
            "Tell me about a meal that went wrong in an interesting way, as if telling a friend "
            "over coffee.",
        ],
    },
    "structured": {
        "description": "Highly regular output. The roadmap measured Structured acceptance at 89%, "
                       "so this is the high-acceptance arm and the case where wider K may pay.",
        "prompts": [
            "Return a JSON array of twelve objects. Each object has keys id (integer 1..12), "
            "name, and group, where group cycles through alpha, beta, gamma.",
            "Produce a markdown table with columns Rank, Country, Capital, Population for the ten "
            "most populous countries. Use exact formatting with no extra commentary.",
            "List the first thirty prime numbers as a comma-separated sequence, one per line, "
            "with no other text.",
        ],
    },
    # The campaign targets a 40K working context, and issue #96 measured acceptance falling
    # 0.724 -> 0.603 going from 145 to 133K context. A long-context arm is therefore essential:
    # raising --max-context alone does nothing, because the PROMPTS have to be long for the model
    # to ever sit at depth. These prompts are padded to ~40K tokens with in-distribution prose.
    "longctx": {
        "description": "~40K-token prompts, matching the campaign's working context. This is the "
                       "regime where issue #96 measured acceptance degradation.",
        "padding_paragraph": (
            "The harbor at Villefranche fills each morning with the same small traffic: two "
            "fishing boats working the shelf beyond the breakwater, a water taxi shuttling "
            "visitors between the town and the villas on Cap Ferrat, and a Coast Guard cutter "
            "that comes in whenever the mistral pushes past thirty knots. The boats are old and "
            "the crews know each other, so most of the morning passes in hand signals and the "
            "occasional shout about bait, weather, or the price of diesel. By nine the water goes "
            "flat and silver, the swimmers arrive, and the fishermen move further out where the "
            "depth makes the sonar worth reading. "
        ),
        # Calibration: this paragraph is 607 chars / 106 words. Against a Qwen BPE tokenizer that
        # is roughly 140 tokens per repeat, so 260 repeats lands near 36K tokens. Do NOT trust a
        # chars//4 rule here: it overestimates by ~8%. The script reports the server's true
        # computed_prefill_tokens for each request, so calibrate against that instead.
        "padding_repeats": 260,
        "task": (
            "After the notes above, solve this step by step. Let n be the smallest positive "
            "integer such that n^2 + 3n + 5 is divisible by 77. Find n and justify each step."
        ),
        "prompts": [],  # built at runtime by build_longctx_prompts()
    },
}


def build_longctx_prompts(repeats: int = 260) -> list[str]:
    """Pad the long-context workload to roughly 36K tokens.

    The paragraph is 607 chars / 106 words, which is roughly 140 Qwen tokens per repeat, so 260
    repeats lands near 36K tokens before the chat template and task. Treat this as an approximation
    only: the server reports the exact `computed_prefill_tokens` for every request, and the summary
    table prints it, so calibrate `--longctx-repeats` against that measured value rather than against
    this estimate. Raise `repeats` for a deeper context, subject to --max-context and VRAM.
    """
    workload = WORKLOADS["longctx"]
    paragraph = workload["padding_paragraph"]
    return [paragraph * repeats + workload["task"]]

DEFAULT_KS = (1, 2, 3, 5, 7, 10, 15)

# Fixed across every point so K is the only variable.
DEFAULT_MAX_CONTEXT = 4096
DEFAULT_OUTPUT_TOKENS = 512
DEFAULT_SEED = 57004


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def build_command(args: argparse.Namespace, k: int, port: int, campaign_log: Path) -> list[str]:
    command = [
        str(args.server),
        str(args.model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-context", str(args.max_context),
        "--max-concurrency", "1",
        "--kv-dtype", args.kv_dtype,
        "--request-log-jsonl", str(campaign_log),
    ]
    if args.kv_capacity:
        command += ["--kv-capacity", str(args.kv_capacity)]
    if k > 0:
        command += ["--spec", "mtp", "--draft-tokens", str(k)]
        if args.lm_head_draft:
            command += ["--lm-head-draft"]
    if args.extra_server_args:
        command += list(args.extra_server_args)
    return command


def check_server_executable(server: Path) -> None:
    """Catch the common unexecutable-path mistake with an actionable message."""
    if os.name != "nt" and server.is_file() and not os.access(server, os.X_OK):
        raise RuntimeError(
            f"server is not executable: {server}\n"
            f"  fix with: chmod +x {server}"
        )


def wait_ready(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with status {process.returncode}; "
                "check the artifact path, GPU, and server flags"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"server health timeout after {timeout:.0f}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=20)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=10)
        except Exception:
            pass


def post_chat(port: int, prompt: str, max_tokens: int, seed: int,
              timeout: float) -> dict[str, Any]:
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Request-log parsing. The log is append-mode JSONL; byte offsets let each point read only its own
# events even when several points share one campaign file.
# ---------------------------------------------------------------------------

def read_new_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size <= offset:
        return [], size
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    events: list[dict[str, Any]] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A partially flushed trailing line; it will be re-read on the next poll.
            continue
    return events, size


def speculative_block(event: dict[str, Any]) -> dict[str, Any]:
    """Locate the speculative counters in a request_done event.

    Schema-v8 puts them under `speculative`; be defensive about nesting so the script keeps working
    if the field moves.
    """
    for key in ("speculative", "speculative_stats", "spec"):
        block = event.get(key)
        if isinstance(block, dict):
            return block
    return {}


def per_position(event: dict[str, Any]) -> list[int]:
    values = speculative_block(event).get("accepted_per_position")
    if isinstance(values, list):
        return [int(v) for v in values]
    return []


# ---------------------------------------------------------------------------
# One measurement point
# ---------------------------------------------------------------------------

def run_point(args: argparse.Namespace, workload_name: str, workload: dict[str, Any],
              k: int, port: int, campaign_log: Path) -> dict[str, Any]:
    tag = f"{workload_name}/k{k}"
    log(f"--- {tag}: starting server ---")

    check_server_executable(Path(args.server))

    server_log_dir = args.out / "server-logs"
    server_log_dir.mkdir(parents=True, exist_ok=True)
    server_log = server_log_dir / f"{workload_name}-k{k}.log"
    command = build_command(args, k, port, campaign_log)

    log_offset = campaign_log.stat().st_size if campaign_log.exists() else 0

    with server_log.open("wb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, port, args.startup_timeout)
            log(f"{tag}: server ready, issuing {len(workload['prompts'])} prompts")

            responses = []
            for index, prompt in enumerate(workload["prompts"]):
                started = time.monotonic()
                try:
                    response = post_chat(port, prompt, args.output_tokens, args.seed,
                                         args.request_timeout)
                except Exception as error:  # noqa: BLE001 - report and continue with other prompts
                    log(f"{tag}: prompt {index} failed: {error}")
                    continue
                elapsed = time.monotonic() - started
                usage = response.get("usage", {}) or {}
                completion = usage.get("completion_tokens")
                responses.append({
                    "prompt_index": index,
                    "completion_tokens": completion,
                    "wall_seconds": elapsed,
                    "client_tok_s": (completion / elapsed) if completion and elapsed > 0 else None,
                })
                log(f"{tag}: prompt {index} -> {completion} tokens in {elapsed:.1f}s")

            # Give the server a moment to flush the final request_done events.
            time.sleep(2.0)
        finally:
            stop_server(process)
            try:
                process.wait(timeout=15)
            except Exception:
                pass

    events, log_offset = read_new_events(campaign_log, log_offset)
    done = [e for e in events if e.get("event") == "request_done"]

    totals = {"rounds": 0, "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0}
    position_histogram: list[int] = []
    draft_windows: list[int] = []
    decode_seconds = 0.0
    completion_tokens = 0
    ttft_seconds: list[float] = []
    prefill_seconds: list[float] = []
    prefill_tokens_per_request: list[int] = []
    finish_reasons: list[str] = []

    for event in done:
        block = speculative_block(event)
        totals["rounds"] += int(block.get("rounds", 0) or 0)
        totals["drafted_tokens"] += int(block.get("drafted_tokens", 0) or 0)
        totals["accepted_tokens"] += int(block.get("accepted_tokens", 0) or 0)
        totals["fallback_steps"] += int(block.get("fallback_steps", 0) or 0)
        if "draft_window" in block:
            draft_windows.append(int(block["draft_window"]))

        histogram = per_position(event)
        if histogram:
            if len(histogram) > len(position_histogram):
                position_histogram.extend([0] * (len(histogram) - len(position_histogram)))
            for index, value in enumerate(histogram):
                position_histogram[index] += value

        timings = event.get("timings_seconds") or {}
        if isinstance(timings, dict):
            decode_seconds += float(timings.get("decode", 0.0) or 0.0)
            prefill_seconds.append(float(timings.get("prefill", 0.0) or 0.0))
            ttft_seconds.append(float(timings.get("ttft", 0.0) or 0.0))
        # Token counts live under `result` in schema v8, not at the event top level.
        result = event.get("result") or {}
        if isinstance(result, dict):
            completion_tokens += int(result.get("completion_tokens", 0) or 0)
            prefill_tokens_per_request.append(int(result.get("computed_prefill_tokens", 0) or 0))
            finish_reasons.append(str(result.get("finish_reason", "") or ""))

    # Exact denominators. Position 0 is drafted in every round. Position i is drafted only in rounds
    # where every earlier position was accepted, so accepted_at_{i-1} is its exact denominator —
    # NOT rounds, and not rounds*K.
    drafted_at_position = [0] * len(position_histogram)
    if position_histogram:
        drafted_at_position[0] = totals["rounds"]
        for index in range(1, len(position_histogram)):
            drafted_at_position[index] = position_histogram[index - 1]

    per_position_q = [
        (position_histogram[i] / drafted_at_position[i]) if drafted_at_position[i] else None
        for i in range(len(position_histogram))
    ]

    return {
        "tag": tag,
        "workload": workload_name,
        "k": k,
        "lm_head_draft": bool(args.lm_head_draft),
        "kv_dtype": args.kv_dtype,
        "max_context": args.max_context,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "server_command": command,
        "requests_completed": len(done),
        "totals": totals,
        "position_histogram": position_histogram,
        "drafted_at_position": drafted_at_position,
        "per_position_acceptance": per_position_q,
        "tokens_per_round": (
            (totals["accepted_tokens"] + totals["rounds"]) / totals["rounds"]
            if totals["rounds"] else None
        ),
        "aggregate_acceptance": (
            totals["accepted_tokens"] / totals["drafted_tokens"]
            if totals["drafted_tokens"] else None
        ),
        "decode_seconds_total": decode_seconds,
        "decode_tok_s": (completion_tokens / decode_seconds) if decode_seconds > 0 else None,
        "completion_tokens_total": completion_tokens,
        "ms_per_round": (decode_seconds * 1000.0 / totals["rounds"]) if totals["rounds"] else None,
        "ttft_seconds": ttft_seconds,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_request": prefill_tokens_per_request,
        "finish_reasons": finish_reasons,
        "client_responses": responses,
        "draft_windows_seen": sorted(set(draft_windows)),
        "server_log": str(server_log),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(points: list[dict[str, Any]]) -> None:
    print()
    print("Summary")
    print("=" * 104)
    header = (f"{'workload':<14}{'K':>3}{'reqs':>6}{'rounds':>8}{'drafted':>9}{'accepted':>9}"
              f"{'q_agg':>8}{'tok/rnd':>9}{'dec t/s':>9}{'ms/rnd':>8}")
    print(header)
    print("-" * len(header))
    for point in points:
        print(f"{point['workload']:<14}{point['k']:>3}{point['requests_completed']:>6}"
              f"{point['totals']['rounds']:>8}{point['totals']['drafted_tokens']:>9}"
              f"{point['totals']['accepted_tokens']:>9}"
              f"{fmt(point['aggregate_acceptance']):>8}{fmt(point['tokens_per_round']):>9}"
              f"{fmt(point['decode_tok_s'], '.1f'):>9}{fmt(point['ms_per_round'], '.2f'):>8}")


def print_context_report(points: list[dict[str, Any]]) -> None:
    """Report the context depth each point actually reached.

    Raising --max-context does NOT make a workload long-context: the prompts have to be long. This
    table makes the real depth visible so a short-prompt run is never mistaken for a long-context
    one. Values are the server's own computed_prefill_tokens plus the generated tokens.
    """
    print()
    print("Context depth actually reached (server-reported)")
    print("=" * 104)
    print("prefill_tok is the server's computed_prefill_tokens per request; final_ctx adds that")
    print("point's generated tokens. If these are small, the run is NOT long-context regardless of")
    print("--max-context.")
    print()
    header = (f"{'tag':<20}{'reqs':>5}{'prefill_tok':>26}{'gen_tok/req':>12}"
              f"{'final_ctx':>11}{'finish':>18}")
    print(header)
    print("-" * len(header))
    for point in points:
        prefill = point.get("prefill_tokens_per_request") or []
        gen = point["completion_tokens_total"] / point["requests_completed"] \
            if point["requests_completed"] else 0.0
        if prefill:
            prefill_text = f"{min(prefill)}-{max(prefill)}"
            final = max(prefill) + gen
        else:
            prefill_text = "n/a"
            final = 0.0
        reasons = sorted(set(point.get("finish_reasons") or []))
        finish_text = ",".join(reasons) if reasons else "n/a"
        print(f"{point['tag']:<20}{point['requests_completed']:>5}{prefill_text:>26}"
              f"{gen:>12.0f}{final:>11.0f}{finish_text:>18}")


def print_position_table(points: list[dict[str, Any]]) -> None:
    width = max((len(p["position_histogram"]) for p in points), default=0)
    print()
    print("Per-position acceptance   q_i = accepted_at_i / drafted_at_i")
    print("  position 0 is drafted in every round; position i is drafted only when all earlier")
    print("  positions were accepted, so accepted_at_{i-1} is the exact denominator.")
    print()
    header = (f"{'workload':<14}{'K':>3}  "
              + "  ".join(f"q{i:<5}" for i in range(width))
              + "   tok/round  dec_tok/s")
    print(header)
    print("-" * len(header))
    for point in points:
        row = f"{point['workload']:<14}{point['k']:>3}  "
        for i in range(width):
            if i < len(point["per_position_acceptance"]):
                row += f"{fmt(point['per_position_acceptance'][i]):<7}"
            else:
                row += f"{'-':<7}"
        row += f"  {fmt(point['tokens_per_round']):<9} {fmt(point['decode_tok_s'], '.1f'):<9}"
        print(row)


def print_decision(points: list[dict[str, Any]]) -> None:
    """Apply the roadmap's Gate B decision rule to the measured data."""
    print()
    print("Gate B decision rule applied")
    print("=" * 104)
    print("The rule: a wider K (or a scan) pays only if acceptance stays high at LATER positions AND")
    print("the extra accepted tokens exceed the extra verify cost. With the corrected cost model")
    print("(~1.0 us per extra column per GDN layer, ~49 us per round across 48 layers), the cost side")
    print("is cheap, so the decision now rests almost entirely on the decay shape.")
    print()

    by_workload: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_workload.setdefault(point["workload"], []).append(point)

    for workload in sorted(by_workload):
        group = sorted(by_workload[workload], key=lambda p: p["k"])
        print(f"{workload}:")
        for point in group:
            histogram = point["per_position_acceptance"]
            if not histogram:
                print(f"    K={point['k']:<2} no position data")
                continue
            head = histogram[:3]
            tail = histogram[3:]
            collapsed = (not tail) or all(v is not None and v < 0.15 for v in tail)
            shape = "collapses after early positions" if collapsed else "slow decay across positions"
            verdict = "wider K unlikely to pay" if collapsed else "wider K worth evaluating"
            head_text = ",".join(fmt(v) for v in head)
            print(f"    K={point['k']:<2} q0..2 = {head_text:<28} tail={'yes' if tail else 'none'}"
                  f"  {shape}  -> {verdict}")
        print()


def write_outputs(args: argparse.Namespace, points: list[dict[str, Any]]) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": "gate-b-acceptance-cost-sweep",
        "measured_at_unix": time.time(),
        "host": {
            "platform": sys.platform,
            "server": str(args.server),
            "model": str(args.model),
            "gpu": os.environ.get("NINFER_GPU", "unknown"),
        },
        "fixed_conditions": {
            "max_context": args.max_context,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "kv_dtype": args.kv_dtype,
            "max_concurrency": 1,
            "lm_head_draft": bool(args.lm_head_draft),
        },
        "points": points,
    }
    json_path = args.out / "gate_b_points.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote {json_path}")

    csv_path = args.out / "gate_b_summary.csv"
    columns = ["workload", "k", "requests_completed", "rounds", "drafted_tokens",
               "accepted_tokens", "aggregate_acceptance", "tokens_per_round", "decode_tok_s",
               "ms_per_round", "max_prefill_tokens", "max_final_context", "finish_reasons",
               "position_histogram", "per_position_acceptance"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for point in points:
            handle.write(",".join([
                point["workload"],
                str(point["k"]),
                str(point["requests_completed"]),
                str(point["totals"]["rounds"]),
                str(point["totals"]["drafted_tokens"]),
                str(point["totals"]["accepted_tokens"]),
                fmt(point["aggregate_acceptance"], ".6f"),
                fmt(point["tokens_per_round"], ".6f"),
                fmt(point["decode_tok_s"], ".3f"),
                fmt(point["ms_per_round"], ".4f"),
                str(max(point.get("prefill_tokens_per_request") or [0])),
                str(int(max(point.get("prefill_tokens_per_request") or [0])
                        + (point["completion_tokens_total"] / point["requests_completed"]
                           if point["requests_completed"] else 0))),
                "|".join(sorted(set(point.get("finish_reasons") or []))),
                "|".join(str(v) for v in point["position_histogram"]),
                "|".join(fmt(v, ".6f") for v in point["per_position_acceptance"]),
            ]) + "\n")
    log(f"wrote {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate B acceptance/cost sweep for Qwen3.8-27B decode speedup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--server", required=True, type=Path, help="path to ninfer-serve executable")
    parser.add_argument("--model", required=True, type=Path, help="path to the .ninfer artifact")
    parser.add_argument("--out", type=Path, default=Path("profiles/bench/gate-b"),
                        help="campaign output directory")
    parser.add_argument("--workloads", default="reasoning,freeform,structured",
                        help="comma-separated subset of: " + ",".join(WORKLOADS))
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                        help="comma-separated MTP draft widths, e.g. 1,3,5,15")
    parser.add_argument("--kv-dtype", default="rk4v4-e8",
                        help="KV cache dtype passed to the server")
    parser.add_argument("--kv-capacity", type=int, default=0,
                        help="optional explicit --kv-capacity (0 = omit)")
    parser.add_argument("--longctx-repeats", type=int, default=260,
                        help="paragraph repeats used to pad the longctx workload (~110 tokens each)")
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lm-head-draft", action="store_true", default=True,
                        help="pass --lm-head-draft (default on, matches the campaign baseline)")
    parser.add_argument("--no-lm-head-draft", dest="lm_head_draft", action="store_false",
                        help="disable the optimized proposal head")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--extra-server-args", nargs=argparse.REMAINDER, default=[],
                        help="additional raw server flags appended to every launch")
    args = parser.parse_args(argv)

    if not args.server.exists():
        parser.error(f"server executable not found: {args.server}")
    if not args.model.exists():
        parser.error(f"model artifact not found: {args.model}")

    names = [n.strip() for n in args.workloads.split(",") if n.strip()]
    unknown = [n for n in names if n not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workload(s): {','.join(unknown)}; choose from {','.join(WORKLOADS)}")
    args.workload_names = names

    k_values = []
    for piece in args.ks.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if not 1 <= value <= 15:
            parser.error(f"K must be in [1,15], got {value}")
        k_values.append(value)
    if not k_values:
        parser.error("no K values selected")
    args.k_values = k_values
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    campaign_log = args.out / "server.requests.jsonl"

    log(f"server      : {args.server}")
    log(f"model       : {args.model}")
    log(f"workloads   : {','.join(args.workload_names)}")
    log(f"K values    : {args.k_values}")
    log(f"request log : {campaign_log}")

    points: list[dict[str, Any]] = []
    port = args.port
    for workload_name in args.workload_names:
        workload = WORKLOADS[workload_name]
        if not workload["prompts"] and "padding_paragraph" in workload:
            workload = dict(workload)
            workload["prompts"] = build_longctx_prompts(args.longctx_repeats)
            words = len(workload["prompts"][0].split())
            log(f"{workload_name}: built {len(workload['prompts'])} prompt(s), "
                f"{words} words (~{int(words / 0.75)} tokens, approximate)")
        for k in args.k_values:
            try:
                point = run_point(args, workload_name, workload, k, port, campaign_log)
            except Exception as error:  # noqa: BLE001 - keep the campaign going
                log(f"point {workload_name}/k{k} failed: {error}")
                continue
            points.append(point)
            if point["requests_completed"] == 0:
                log(f"WARNING: {point['tag']} produced no request_done events; "
                    f"check {point['server_log']}")
            port += 1

    print_summary(points)
    print_context_report(points)
    print_position_table(points)
    print_decision(points)
    write_outputs(args, points)

    print()
    log("Done. Copy the output directory back into the repo to continue the analysis.")
    return 0 if points else 1


if __name__ == "__main__":
    raise SystemExit(main())
