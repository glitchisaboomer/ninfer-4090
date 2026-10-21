#!/usr/bin/env python3
"""Measure achieved DRAM bandwidth of the fused Q4 SwiGLU decode kernel with Nsight Compute.

One question, one kernel. `docs/optimization-findings.md` carries two numbers that cannot both
be true:

  * `ninfer_q4_linear_swiglu_bench` reports 144.4 us for the `[34816,5120] -> [17408,T]` profile,
    i.e. 94,699,520 bytes in 144.4 us = 655.8 GB/s = 65.1% of the RTX 4090's 1008 GB/s peak.
  * The decode-aggregate roofline in the same document (17.90 GB of weights in 21.20 ms per
    round) implies 844 GB/s = 84%.

`mlp/gate_up` is 5.645 GiB across 64 layers = 34% of all weight bytes, so the difference decides
whether q4/q5 small-T has ~11% of decode recoverable (65% -> 84% on this op saves 2.09 ms of a
21.20 ms round) or nothing at all. This tool measures the op where it actually runs.

Example (Windows, from the repository root):
  python tools/bench/profile_swiglu_bandwidth_ncu.py ^
      --server build/apps/Release/ninfer-serve.exe ^
      --model E:/models/qwen3_8_27b.ninfer

Example (WSL/Linux):
  python3 tools/bench/profile_swiglu_bandwidth_ncu.py \
      --server build/apps/ninfer-serve --model /models/qwen3_8_27b.ninfer

Why the warmup trap from `profile_attention_width_ncu.py` does not apply here: that tool must
skip the server's `max_tokens=4` warmup because the attention launch geometry changes with context
depth, so a warmup capture measures a different grid. This kernel has a fixed grid (2176 blocks =
17408/8 rows per CTA), fixed bytes (94.7 MB), and no dependence on context depth or on which
specific round it belongs to, and NCU flushes L2 before every replay pass. Any matching launch is
therefore an equivalent measurement, so `--launch-skip 0` is safe. The captured grid is still
checked and reported, so an unexpected geometry is visible rather than assumed away.

This is an attribution tool, not a benchmark: NCU replay changes absolute timings. It reports
achieved bandwidth for one op, which is what the roofline question needs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.bench import profile_verify_path as shared
except ModuleNotFoundError:
    import profile_verify_path as shared  # type: ignore[no-redef]

try:
    from tools.bench.profile_attention_width_ncu import find_ncu
except ModuleNotFoundError:
    from profile_attention_width_ncu import find_ncu  # type: ignore[no-redef]


# The template's epilogue parameter is what identifies the fused SwiGLU instantiation: the plain
# draft-head / attention-projection / GDN-projection launches all name `q4_small_t_mma_kernel` but
# default their epilogue to `Q4SmallTMmaStoreEpilogue`. Matching on the kernel name alone would
# capture whichever of the four ran first. `--kernel-name-base demangled` is required for the
# template argument to appear in the string at all.
SWIGLU_KERNEL_FILTER = "regex:Q4SwiGluSmallTEpilogue"

# `mlp/gate_up`: 34816 output rows (gate and up interleaved), k=5120, Q4G64 = 34 B per 64 elements.
GATE_UP_ROWS = 34816
HIDDEN = 5120
# Q4G64_F16S: 32 B low nibbles + 2 B FP16 scale per 64 elements.
Q4_BYTES_PER_64 = 34
LAYERS = 64
# Rows per CTA in `Q4SwiGluSmallTRows`: 8 gate rows plus 8 up rows, one launch writes 17408 rows.
SWIGLU_ROWS_PER_CTA = 8
EXPECTED_GRID = (GATE_UP_ROWS // 2) // SWIGLU_ROWS_PER_CTA  # 2176

DEFAULT_PEAK_GBPS = 1008.0
# The two competing hypotheses, as fractions of peak, for the comparison table.
HYPOTHESIS_OP_BENCH = 0.651
HYPOTHESIS_AGGREGATE = 0.840

TIME_TO_SECONDS = {
    "ns": 1e-9, "nsecond": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
    "us": 1e-6, "usec": 1e-6, "usecond": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ms": 1e-3, "msec": 1e-3, "msecond": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
}
BYTES = {
    "byte": 1.0, "bytes": 1.0,
    "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
    "kibibyte": 1024.0, "mibibyte": 1024.0**2, "gibibyte": 1024.0**3,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure achieved DRAM bandwidth of the fused Q4 SwiGLU decode kernel.")
    parser.add_argument("--server", type=Path, required=True, help="path to ninfer-serve")
    parser.add_argument("--model", type=Path, required=True, help="path to the .ninfer artifact")
    parser.add_argument("--ncu", help="path to ncu; auto-detected when omitted")
    parser.add_argument("--out", type=Path, default=Path("profiles/swiglu-bandwidth-ncu"))
    parser.add_argument("--port", type=int, default=8497)
    parser.add_argument("--k", type=int, default=3,
                        help="MTP draft tokens; verify width becomes K+1 (default 3 -> 4 columns)")
    # Context is irrelevant to this kernel's cost: it reads weights and a T-wide activation slice,
    # never the KV cache. A short prompt keeps prefill out of the trace and the report small.
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=64,
                        help="generation length; each round launches the kernel 64 times")
    parser.add_argument("--seed", type=int, default=57004)
    parser.add_argument("--kv-dtype", default="rk4v4-e8")
    parser.add_argument("--launch-skip", type=int, default=0,
                        help="matching launches to skip (default 0). Any decode launch measures "
                             "the same bytes and geometry, and NCU flushes L2 per pass, so this "
                             "does not need to be tuned past the warmup")
    parser.add_argument("--launch-count", type=int, default=1,
                        help="matching launches to profile (default 1). The summary reports the "
                             "first captured launch; use >1 only as a stability check, the "
                             "runs cost a full replay each")
    parser.add_argument("--peak-gbps", type=float, default=DEFAULT_PEAK_GBPS,
                        help="device DRAM peak in GB/s for the percentage column (RTX 4090: 1008)")
    parser.add_argument("--target-pct", type=float, default=84.0,
                        help="roofline fraction to project against (default 84, the doc aggregate)")
    parser.add_argument("--round-ms", type=float, default=21.20,
                        help="published decode round time, used only for the indicative gain")
    parser.add_argument("--set", dest="metric_set", default="basic",
                        choices=("basic", "detailed", "full"),
                        help="NCU metric set (default basic: LaunchStats + SpeedOfLight, one "
                             "replay pass). 'full' is 45 passes and overflows NCU's save/restore "
                             "buffer on a 16.67 GiB model")
    parser.add_argument("--metrics", default="",
                        help="extra raw metrics; the bytes-read counter is added automatically")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--flush-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the NCU command and exit without launching anything")
    args = parser.parse_args(argv)

    if not args.server.is_file():
        parser.error(f"server not found: {args.server}")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if not (1024 <= args.port <= 65535):
        parser.error("--port must be in [1024,65535]")
    if args.k < 1 or args.output_tokens < 1 or args.launch_count < 1:
        parser.error("--k, --output-tokens and --launch-count must be positive")
    if args.launch_skip < 0:
        parser.error("--launch-skip must be nonnegative")
    if args.peak_gbps <= 0 or not (0 < args.target_pct <= 100):
        parser.error("--peak-gbps must be positive and --target-pct in (0,100]")
    return args


def swiglu_bytes_per_layer() -> int:
    """Bytes this kernel reads for one layer: the whole 34816x5120 Q4G64 gate/up matrix."""
    return (GATE_UP_ROWS * HIDDEN // 64) * Q4_BYTES_PER_64


def build_ncu_command(args: argparse.Namespace, ncu: Path, report_stem: Path,
                      request_log: Path) -> list[str]:
    metrics = [part.strip() for part in args.metrics.split(",") if part.strip()]
    # The bytes-read counter is the measurement, not an option: without it the tool can report a
    # duration but not a bandwidth. `dram__throughput.avg.pct_of_peak_sustained_elapsed` comes
    # free inside the `basic` set (SpeedOfLight) and is kept as an independent cross-check, because
    # a disagreement between it and bytes/duration means the wrong device or clock is in play.
    for required in ("dram__bytes_read.sum", "gpu__time_duration.sum"):
        if required not in metrics:
            metrics.append(required)

    command: list[str] = [
        str(ncu),
        "--set", args.metric_set,
        "--metrics", ",".join(metrics),
        "--kernel-name-base", "demangled",
        "--kernel-name", SWIGLU_KERNEL_FILTER,
        "--launch-count", str(args.launch_count),
    ]
    if args.launch_skip > 0:
        command += ["--launch-skip", str(args.launch_skip)]
    command += [
        "--force-overwrite",
        "--export", str(report_stem),
        "--",
        str(args.server), str(args.model),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", str(args.max_context), "--max-concurrency", "1",
        "--kv-dtype", args.kv_dtype,
        # Required: inside a captured graph NCU sees one opaque node and can match no kernel.
        "--no-cuda-graph",
        "--spec", "mtp", "--draft-tokens", str(args.k),
        "--lm-head-draft",
        "--request-log-jsonl", str(request_log),
    ]
    return command


def to_float(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"n/a", "na", "-", "inf", "nan"}:
        return None
    match = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def convert(value: float | None, unit: str, table: dict[str, float]) -> float | None:
    if value is None:
        return None
    factor = table.get(unit.strip().lower())
    if factor is None:
        return None
    return value * factor


def import_report(ncu: Path, report: Path, csv_path: Path) -> list[dict[str, Any]]:
    """Export the saved report to CSV and return rows with their unit column intact.

    `profile_attention_width_ncu.export_report` is deliberately not reused: it collapses each row
    to `{metric name: value string}` and drops `Metric Unit`, and this tool's entire output is a
    unit-bearing rate. Guessing whether a duration arrived in ns or usec would silently produce a
    bandwidth off by 1000x.
    """
    completed = subprocess.run([str(ncu), "--import", str(report), "--csv"],
                               capture_output=True, check=False, timeout=900)
    text = completed.stdout.decode("utf-8", errors="replace")
    if completed.returncode != 0 or not text.strip():
        detail = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(f"ncu --import failed (rc={completed.returncode}): "
                           f"{(detail[0] if detail else 'no output')}")
    csv_path.write_text(text, encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(text.splitlines()):
        name = (row.get("Metric Name") or "").strip()
        if not name:
            continue
        rows.append({
            "kernel": (row.get("Kernel Name") or "").strip(),
            "section": (row.get("Section Name") or "").strip(),
            "name": name,
            "unit": (row.get("Metric Unit") or "").strip(),
            "value": to_float(row.get("Metric Value") or ""),
        })
    return rows


def find_row(rows: list[dict[str, Any]], *patterns: str) -> dict[str, Any] | None:
    """Return the first row whose metric name matches any pattern, tolerating NCU version drift."""
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for row in rows:
            if regex.search(row["name"]):
                return row
    return None


def find_launch_attribute(rows: list[dict[str, Any]], *patterns: str) -> str | None:
    """Launch geometry (grid/block) arrives as a pseudo-metric row in the CSV export."""
    row = find_row(rows, *patterns)
    if row is None:
        return None
    return f"{row['value']:g}" if isinstance(row["value"], float) else str(row["value"])


def stop_profiler(process: subprocess.Popen[bytes], port: int) -> None:
    """Interrupt NCU so it flushes the report, then confirm the server is gone."""
    if process.poll() is None:
        try:
            if os.name == "nt":
                ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                if ctrl_break is not None:
                    os.kill(process.pid, ctrl_break)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (OSError, ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            shared.log("NCU did not exit after interrupt; terminating its process tree "
                       "(report may be incomplete).")
            shared._kill_tree(process.pid)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    shared._kill_tree(process.pid)
    for _ in range(60):
        occupied, _ = shared.probe_once(port, timeout=1.0)
        if not occupied:
            return
        time.sleep(0.25)
    raise RuntimeError(f"server still answers on port {port} after NCU shutdown")


def ncu_tail(log_path: Path, lines: int = 20) -> str:
    """NCU writes its own failures to the redirected stdout, never to this process."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    interesting = [line for line in text if "==ERROR==" in line or "ERR_" in line]
    if not interesting:
        interesting = text[-lines:]
    return "\n  " + "\n  ".join(interesting[-lines:])


def report(args: argparse.Namespace, rows: list[dict[str, Any]], csv_path: Path,
           log_path: Path, request_log: Path, depth: dict[str, Any] | None,
           completion: int) -> dict[str, Any]:
    time_row = find_row(rows, r"^gpu__time_duration\.sum$", r"duration")
    byte_row = find_row(rows, r"^dram__bytes_read\.sum$", r"DRAM Bytes Read{1}")
    pct_row = find_row(rows, r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed",
                       r"DRAM Throughput")
    grid = find_launch_attribute(rows, r"^Grid Size$", r"grid_size")
    block = find_launch_attribute(rows, r"^Block Size$", r"block_size")

    duration_s = convert(time_row["value"], time_row["unit"], TIME_TO_SECONDS) if time_row else None
    bytes_read = convert(byte_row["value"], byte_row["unit"], BYTES) if byte_row else None
    counter_pct = pct_row["value"] if pct_row else None

    if duration_s is None or bytes_read is None:
        missing = []
        if duration_s is None:
            missing.append("duration (gpu__time_duration.sum)")
        if bytes_read is None:
            missing.append("bytes read (dram__bytes_read.sum)")
        raise RuntimeError(f"report is missing {' and '.join(missing)}; inspect {csv_path}"
                           + ncu_tail(log_path))

    gbps = bytes_read / duration_s / 1e9
    fraction = gbps / args.peak_gbps

    expected_bytes = swiglu_bytes_per_layer()
    # 8% covers the scale/metadata rows and the swiglu epilogue's own activation reads; a wilder
    # discrepancy means the captured launch is a different shape (e.g. the draft head at k=2048).
    bytes_ok = abs(bytes_read - expected_bytes) <= 0.08 * expected_bytes
    grid_ok = grid is not None and abs(float(grid) - EXPECTED_GRID) < 1

    op_ms = LAYERS * duration_s * 1e3
    target = args.target_pct / 100.0
    projected_op_ms = op_ms * (fraction / target) if fraction > 0 else op_ms
    new_round = args.round_ms - op_ms + projected_op_ms
    gain = (args.round_ms / new_round - 1.0) if new_round > 0 else 0.0

    # The counter is an independent second opinion on the same quantity. Printing both without
    # comparing them makes it decorative: a wrong device, downclocked memory, or a stale export
    # shows up as the two disagreeing, and that has to be loud to be useful.
    cross_check = "counter cross-chk : unavailable"
    counter_delta: float | None = None
    if counter_pct is not None:
        counter_delta = abs(counter_pct - fraction * 100.0)
        cross_check = (f"counter cross-chk : {counter_pct:.2f}% "
                       f"(SpeedOfLight DRAM throughput)")
        if counter_delta > 5.0:
            cross_check += (f"\n                    WARNING: {counter_delta:.1f} points away from "
                            f"bytes/duration -- check device, clocks, or a stale report")

    if fraction < 0.72:
        verdict = ("HEADROOM -- consistent with the 65% op-bench reading; q4/q5 small-T has "
                   "recoverable bandwidth")
    elif fraction < 0.80:
        verdict = ("AMBIGUOUS -- between the two published figures; re-measure with --launch-count 3")
    else:
        verdict = "AT ROOFLINE -- the doc's 84% aggregate holds; close the lead"

    lines = [
        "",
        f"kernel            : {SWIGLU_KERNEL_FILTER}  ({time_row['kernel'] or 'n/a'})",
        f"captured grid     : {grid or 'unknown'} blocks, block {block or 'unknown'}"
        + ("" if grid_ok else f"  WARNING: expected {EXPECTED_GRID}"),
        f"prompt depth      : "
        + (f"{depth['prompt_tokens']} tokens, final context {depth['final_context']}"
           if depth else "unavailable"),
        f"completion tokens : {completion}",
        "",
        f"bytes read        : {bytes_read:,.0f} B"
        + ("" if bytes_ok else f"  WARNING: expected ~{expected_bytes:,} B"),
        f"kernel duration   : {duration_s * 1e6:.2f} us",
        f"achieved bandwidth: {gbps:.1f} GB/s = {fraction * 100:.2f}% of {args.peak_gbps:.0f} GB/s",
        cross_check,
        "",
        "hypotheses:",
        f"  op bench    : {HYPOTHESIS_OP_BENCH * 100:5.1f}%  {HYPOTHESIS_OP_BENCH * args.peak_gbps:6.1f} GB/s",
        f"  aggregate   : {HYPOTHESIS_AGGREGATE * 100:5.1f}%  {HYPOTHESIS_AGGREGATE * args.peak_gbps:6.1f} GB/s",
        f"  measured    : {fraction * 100:5.1f}%  {gbps:6.1f} GB/s",
        "",
        f"verdict           : {verdict}",
        "",
        f"indicative decode : {LAYERS} layers x {duration_s * 1e6:.2f} us = {op_ms:.2f} ms of a "
        f"{args.round_ms:.2f} ms round",
        f"                    at {args.target_pct:.0f}% this op would take {projected_op_ms:.2f} ms"
        f" -> round {new_round:.2f} ms -> {gain * 100:+.1f}% decode",
        "                    (indicative only: assumes the op's time scales linearly with "
        "achieved bandwidth)",
        "",
        f"report : {csv_path.with_suffix('.ncu-rep')}",
        f"raw csv: {csv_path}",
        "",
    ]
    print("\n".join(lines))

    summary = {
        "kernel_filter": SWIGLU_KERNEL_FILTER,
        "kernel": time_row["kernel"],
        "grid": grid,
        "block": block,
        "grid_expected": EXPECTED_GRID,
        "bytes_read": bytes_read,
        "bytes_expected": expected_bytes,
        "duration_us": duration_s * 1e6,
        "achieved_gbps": gbps,
        "fraction_of_peak": fraction,
        "peak_gbps": args.peak_gbps,
        "counter_pct": counter_pct,
        "counter_delta_points": counter_delta,
        "verdict": verdict,
        "op_ms_per_round": op_ms,
        "round_ms": args.round_ms,
        "target_pct": args.target_pct,
        "indicative_gain_pct": gain * 100,
        "prompt_tokens": depth["prompt_tokens"] if depth else None,
        "completion_tokens": completion,
        "k": args.k,
        "verify_width": args.k + 1,
        "launch_skip": args.launch_skip,
        "launch_count": args.launch_count,
        "request_log": str(request_log),
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        ncu = find_ncu(args.ncu)
    except SystemExit:
        # A dry run must be inspectable on a machine without Nsight Compute installed: the point
        # of `--dry-run` is to read the command line, not to require the profiler.
        if not args.dry_run:
            raise
        ncu = Path("ncu")
        shared.log("ncu not found; --dry-run uses the literal 'ncu' in the printed command")

    report_stem = args.out / "swiglu_gate_up"
    report_path = report_stem.with_suffix(".ncu-rep")
    csv_path = report_stem.with_suffix(".csv")
    log_path = args.out / "ncu-server.log"
    request_log = args.out / "server.requests.jsonl"
    if report_path.exists():
        report_path.unlink()

    command = build_ncu_command(args, ncu, report_stem, request_log)
    (args.out / "profile.json").write_text(json.dumps({
        "ncu_command": command,
        "kernel_filter": SWIGLU_KERNEL_FILTER,
        "expected_bytes_per_layer": swiglu_bytes_per_layer(),
        "expected_grid": EXPECTED_GRID,
        "max_context": args.max_context,
        "output_tokens": args.output_tokens,
        "k": args.k,
    }, indent=2) + "\n", encoding="utf-8")

    shared.log(f"ncu    : {ncu}")
    shared.log(f"server : {args.server}")
    shared.log(f"model  : {args.model}")
    shared.log(f"out    : {args.out}")
    shared.log(f"filter : {SWIGLU_KERNEL_FILTER}")
    if args.dry_run:
        print(" ".join(command))
        return 0

    occupied, why = shared.probe_once(args.port, timeout=2.0)
    if occupied:
        raise RuntimeError(f"port {args.port} is already serving /health ({why}); "
                           f"choose another --port")

    request_offset = request_log.stat().st_size if request_log.exists() else 0
    creationflags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                     if os.name == "nt" else 0)
    depth: dict[str, Any] | None = None
    completion = 0
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != "nt"),
                                   creationflags=creationflags)
        shared._LIVE_PIDS.add(process.pid)
        try:
            if not shared.wait_ready(args.port, args.startup_timeout):
                if process.poll() is not None:
                    raise RuntimeError(f"NCU/server exited ({process.returncode}); "
                                       f"see {log_path}{ncu_tail(log_path)}")
                raise RuntimeError(f"server did not become ready within {args.startup_timeout}s")
            # Context length is irrelevant to this kernel, so a short deterministic prompt keeps
            # prefill out of the trace; the byte count and geometry are what matter.
            prompt = ("Explain in detail how a hash table resolves collisions, with pros and cons "
                      "of chaining versus open addressing.")
            shared.log(f"sending one {args.output_tokens}-token request")
            response = shared.post_chat(args.port, prompt, args.output_tokens, args.seed,
                                        args.request_timeout)
            completion = int((response.get("usage") or {}).get("completion_tokens") or 0)
            shared.log(f"request returned {completion} tokens")
            time.sleep(args.flush_seconds)
            depth = shared.read_reached_depth(request_log, request_offset)
        finally:
            stop_profiler(process, args.port)
            shared._LIVE_PIDS.discard(process.pid)

    if not report_path.exists() or report_path.stat().st_size < 50_000:
        raise RuntimeError(
            f"NCU report missing or suspiciously small: {report_path}. A filter that matches "
            f"nothing produces exactly this. Try --k 1 (verify width 2) or --k 5 (width 6) to "
            f"confirm the SwiGLU small-T path is reached at this width, and check {log_path}"
            + ncu_tail(log_path))
    shared.log(f"saved {report_path} ({report_path.stat().st_size:,} bytes)")

    rows = import_report(ncu, report_path, csv_path)
    summary = report(args, rows, csv_path, log_path, request_log, depth, completion)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if abs(summary["fraction_of_peak"] - HYPOTHESIS_OP_BENCH) < 0.05:
        print("Interpretation: this reproduces the 65% op-bench figure in situ, so the aggregate "
              "84% in docs/optimization-findings.md is an overestimate and the weight stream is "
              "leaving ~19% of peak unused. That is the ~11% decode lead.")
    elif abs(summary["fraction_of_peak"] - HYPOTHESIS_AGGREGATE) < 0.05:
        print("Interpretation: this reproduces the 84% aggregate, so the 144.4 us op-bench figure "
              "was unrepresentative (cold/unwarmed or a different launch). q4/q5 small-T is at the "
              "roofline; close the lead and look at batch amortization instead.")
    else:
        print("Interpretation: neither published figure is reproduced. Record this as the number "
              "of record before choosing a direction; check that clocks are not downclocked and "
              "that the grid matches the expected 2176 blocks.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        shared.log("interrupted")
        shared._cleanup_all()
        raise SystemExit(130) from None
    except Exception as error:  # noqa: BLE001
        shared.log(f"ERROR: {error}")
        shared._cleanup_all()
        raise SystemExit(1) from error
