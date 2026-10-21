#!/usr/bin/env python3
"""Capture one production MTP target-verify attention kernel with Nsight Compute.

For K=3/5 this selects the width-4/6 INT8 tiled kernel specialization seen in the
Nsight Systems traces. It launches one server per K with CUDA Graphs disabled, sends a
long-context request, and asks NCU to profile the first matching kernel launch. The
request is real engine traffic; only the selected kernel is replay-profiled.

Example (Windows, from repository root):
  python tools/bench/profile_attention_width_ncu.py \\
      --server build/apps/Release/ninfer-serve.exe \\
      --model E:/models/qwen3_8_27b.ninfer

This is an attribution/profile tool, not a benchmark: NCU replay changes timings. Compare
occupancy/resource/stall metrics and use the existing end-to-end measurements for latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.bench import profile_verify_path as shared
except ModuleNotFoundError:
    import profile_verify_path as shared  # type: ignore[no-redef]


DEFAULT_NCU_ROOTS = [
    Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute"),
    Path(r"C:\Program Files (x86)\NVIDIA Corporation\Nsight Compute"),
    Path(r"C:\NVIDIA\Nsight Compute"),
]
KERNEL_SPECIALIZATIONS = {
    3: r"regex:.*gqa_attention_decode_i8_tiled_kernel.*\(int\)4, \(int\)8, \(int\)2,",
    5: r"regex:.*gqa_attention_decode_i8_tiled_kernel.*\(int\)6, \(int\)6, \(int\)1,",
}


def find_ncu(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise SystemExit(f"ERROR: --ncu path does not exist: {explicit}")
    found = shutil.which("ncu") or shutil.which("ncu.exe")
    if found:
        return Path(found)
    for root in DEFAULT_NCU_ROOTS:
        if root.exists():
            versions = sorted((p for p in root.iterdir() if p.is_dir()),
                              key=lambda p: p.name, reverse=True)
            for version in versions:
                for relative in (Path("ncu.exe"), Path("target-windows-x64") / "ncu.exe"):
                    candidate = version / relative
                    if candidate.exists():
                        return candidate
    raise SystemExit("ERROR: ncu not found; pass --ncu <path-to-ncu.exe>")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the first production width-4/6 MTP verify attention launch with NCU.")
    parser.add_argument("--server", type=Path, required=True, help="path to ninfer-serve.exe")
    parser.add_argument("--model", type=Path, required=True, help="path to the .ninfer artifact")
    parser.add_argument("--ncu", help="path to ncu.exe; auto-detected when omitted")
    parser.add_argument("--out", type=Path, default=Path("profiles/attention-width-ncu"))
    parser.add_argument("--ks", default="3,5", help="subset of 3,5; default: both")
    parser.add_argument("--port", type=int, default=8493)
    parser.add_argument("--max-context", type=int, default=50000)
    parser.add_argument("--longctx-repeats", type=int, default=320,
                        help="Gate B padding repeats (~140 tokens each); tune to desired depth")
    parser.add_argument("--min-context", type=int, default=40000,
                        help="fail if server-reported prompt depth is below this value")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=57004)
    parser.add_argument("--kv-dtype", default="rk4v4-e8")
    parser.add_argument("--launch-count", type=int, default=1,
                        help="matching specialization launches to profile per K (default 1)")
    # The server warms up before it listens, with a one-word prompt (`GenerationService::warmup`
    # sends "hi" for 4 tokens). That warmup launches the same specialized kernel, so with
    # `--launch-count 1` and no skip NCU profiles the WARMUP -- a ~16-block grid at minimum
    # splits -- and reports it as if it were the long-context request. Skipping the warmup's
    # attention launches is what makes the captured launch the one this tool claims to measure.
    parser.add_argument("--launch-skip", type=int, default=0,
                        help="matching launches to skip before profiling; set past the server's "
                             "warmup (its attention launch count, typically 16-32 for 16 "
                             "full-attention layers) or use --launch-skip-auto to size it")
    parser.add_argument("--set", dest="metric_set", default="full",
                        help="Nsight Compute metric set (default: full)")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--flush-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    ks: list[int] = []
    try:
        ks = [int(part.strip()) for part in args.ks.split(",") if part.strip()]
    except ValueError:
        parser.error("--ks must be a comma-separated subset of 3,5")
    if not ks or len(set(ks)) != len(ks) or any(k not in KERNEL_SPECIALIZATIONS for k in ks):
        parser.error("--ks must contain unique values from 3,5")
    args.ks = ks
    if not args.server.is_file():
        parser.error(f"server not found: {args.server}")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.port < 1024 or args.port + len(ks) - 1 > 65535:
        parser.error("--port and all K-specific ports must be in [1024,65535]")
    if args.max_context <= args.min_context or args.min_context < 1:
        parser.error("require --max-context > --min-context >= 1")
    if args.longctx_repeats < 1 or args.output_tokens < 1 or args.launch_count < 1:
        parser.error("--longctx-repeats, --output-tokens, and --launch-count must be positive")
    if args.launch_skip < 0:
        parser.error("--launch-skip must be nonnegative")
    return args


def stop_profiler(process: subprocess.Popen[bytes], port: int) -> None:
    """Interrupt NCU gracefully so it flushes .ncu-rep, then verify the server stopped."""
    if process.poll() is None:
        if os.name == "nt":
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                try:
                    os.kill(process.pid, ctrl_break)
                except OSError:
                    pass
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            shared.log("NCU did not exit after interrupt; terminating its process tree (report may be incomplete).")
            shared._kill_tree(process.pid)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    # NCU may leave the target server running after it detaches; clean up only its process tree.
    shared._kill_tree(process.pid)
    for _ in range(60):
        occupied, _ = shared.probe_once(port, timeout=1.0)
        if not occupied:
            return
        time.sleep(0.25)
    raise RuntimeError(f"server still answers on port {port} after NCU shutdown")


def run_point(args: argparse.Namespace, ncu: Path, k: int, port: int) -> dict[str, Any]:
    tag = f"k{k}"
    directory = args.out / tag
    directory.mkdir(parents=True, exist_ok=True)
    # Self-describing: the verify width is `k + 1`, and naming the report after it alone made
    # `attention_width4` mean K=3 in one directory and get re-imported as K=5 in another. Both the
    # K and the width belong in the name so an import cannot pick the wrong file.
    report_stem = directory / f"attention_k{k}_width{k + 1}"
    report = report_stem.with_suffix(".ncu-rep")
    log_path = directory / "ncu-server.log"
    request_log = directory / "server.requests.jsonl"
    if report.exists():
        report.unlink()

    occupied, why = shared.probe_once(port, timeout=2.0)
    if occupied:
        raise RuntimeError(f"port {port} is already serving /health ({why}); choose another --port")

    # The regex matches the exact width/warp/min-block specialization identified in the Nsys
    # traces. With a nonzero `--launch-skip` this skips the server's warmup launches, so the
    # captured launch belongs to the long-context request this tool sends.
    kernel_regex = KERNEL_SPECIALIZATIONS[k]
    command = [
        str(ncu), "--set", args.metric_set,
        "--kernel-name-base", "demangled",
        "--kernel-name", kernel_regex,
        "--launch-count", str(args.launch_count),
    ]
    if args.launch_skip > 0:
        command += ["--launch-skip", str(args.launch_skip)]
    command += [
        "--force-overwrite",
        "--export", str(report_stem),
        "--", str(args.server), str(args.model),
        "--host", "127.0.0.1", "--port", str(port),
        "--max-context", str(args.max_context), "--max-concurrency", "1",
        "--kv-dtype", args.kv_dtype, "--no-cuda-graph",
        "--spec", "mtp", "--draft-tokens", str(k), "--lm-head-draft",
        "--request-log-jsonl", str(request_log),
    ]
    request_offset = request_log.stat().st_size if request_log.exists() else 0
    metadata_path = directory / "profile.json"
    metadata_path.write_text(json.dumps({
        "k": k,
        "verify_width": k + 1,
        "kernel_filter": kernel_regex,
        "metric_set": args.metric_set,
        "launch_count": args.launch_count,
        "launch_skip": args.launch_skip,
        "max_context": args.max_context,
        "longctx_repeats": args.longctx_repeats,
        "output_tokens": args.output_tokens,
        "ncu_command": command,
    }, indent=2) + "\\n", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) if os.name == "nt" else 0
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != "nt"), creationflags=creationflags)
        shared._LIVE_PIDS.add(process.pid)
        shared.log(f"K={k}: NCU report -> {report}")
        shared.log(f"K={k}: kernel filter -> {kernel_regex}")
        try:
            if not shared.wait_ready(port, args.startup_timeout):
                if process.poll() is not None:
                    raise RuntimeError(f"NCU/server exited ({process.returncode}); see {log_path}")
                raise RuntimeError(f"server did not become ready within {args.startup_timeout}s")

            prompt = shared.build_longctx_prompt(args.longctx_repeats)
            shared.log(f"K={k}: sending one {args.longctx_repeats}-repeat long-context request")
            response = shared.post_chat(port, prompt, args.output_tokens, args.seed,
                                        args.request_timeout)
            completion = int((response.get("usage") or {}).get("completion_tokens") or 0)
            shared.log(f"K={k}: request returned {completion} tokens")
            time.sleep(args.flush_seconds)
            depth = shared.read_reached_depth(request_log, request_offset)
            if depth is None:
                raise RuntimeError(f"no completed request record in {request_log}")
            shared.log(f"K={k}: prompt depth {depth['prompt_tokens']}, final context "
                       f"{depth['final_context']}")
            if depth["prompt_tokens"] < args.min_context:
                raise RuntimeError(f"reached only {depth['prompt_tokens']} prompt tokens; "
                                   f"need >= {args.min_context}; tune --longctx-repeats")
        finally:
            stop_profiler(process, port)
            shared._LIVE_PIDS.discard(process.pid)

    if not report.exists() or report.stat().st_size < 100_000:
        raise RuntimeError(
            f"NCU report missing or suspiciously small: {report}; inspect {log_path}" +
            ncu_tail(log_path))
    shared.log(f"K={k}: saved {report} ({report.stat().st_size:,} bytes)")

    # The report is a binary whose bulk is raw per-metric counters, and nothing in this repository
    # reads it -- it exists for the Nsight Compute GUI. Export the text view beside it so the
    # interesting rows survive when the report itself is too large to move off the profiling box.
    csv_path = report_stem.with_suffix(".csv")
    exported = export_report(ncu, report_stem, csv_path)
    if exported:
        shared.log(f"K={k}: exported {csv_path} ({csv_path.stat().st_size:,} bytes)")

    # Verify the captured launch is the long-context one. `--launch-skip` is a guess about how many
    # matching launches the server's warmup performs, and getting it wrong is silent: the request
    # still runs at full depth, so the request log looks correct while NCU profiled a 3-token
    # warmup. The warmup launches a minimal-split grid, so an unexpectedly small grid is the
    # signature of that mistake. `splits` grows with the window, so the real verify launch of a
    # 40K request is (KVHeads, 64, batch) -- hundreds of blocks, not tens.
    grid = exported.get("Grid Size")
    if grid:
        blocks = 1
        for factor in re.findall(r"\d+", grid.replace(",", "")):
            blocks *= int(factor)
        expected_floor = 4 * max(1, int(args.min_context // 1024))
        shared.log(f"K={k}: captured launch grid {grid} = {blocks} blocks")
        if blocks < expected_floor:
            shared.log(f"  WARNING: only {blocks} blocks for a {args.min_context}-token request "
                       f"(expected >= {expected_floor}). This looks like a warmup launch -- raise "
                       f"--launch-skip (warmup runs max_tokens=4 over 16 full-attention layers, so "
                       f"~64 matching launches precede the request).")
    else:
        shared.log(f"K={k}: grid size unavailable; check --launch-skip by hand")
    return {"k": k, "depth": depth, "completion_tokens": completion,
            "report": str(report), "csv": str(csv_path) if exported else None,
            "command": command}


# Metrics the width comparison is actually decided on. Extracting only these keeps the text export
# small enough to commit or paste, which the full `--set full` dump is not. `block size` and
# `grid size` stay because they are how a warmup launch is told apart from the real request: the
# warmup sends "hi" and lands a minimal-split grid, which is the mistake this tool made once.
FOCUS_METRIC_PATTERN = (
    "occupancy|warps?_|stall|hit rate|sectors|eligible|throughput|duration|registers"
    "|shared memory|spill|instructions|cycles|block size|grid size"
)


def ncu_tail(log_path: Path, lines: int = 20) -> str:
    """Return the tail of an NCU run log for an error message.

    NCU writes its own failures -- `ERR_NVGPUCTRPERM`, `Could not open report file`, an empty
    `0 kernels` match -- to the redirected stdout, never to this process. Without this the only
    symptom of a failed profile is a missing report, which is indistinguishable from a slow flush.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    interesting = [line for line in text if "==ERROR==" in line or "ERR_" in line]
    if not interesting:
        interesting = text[-lines:]
    return "\n  " + "\n  ".join(interesting[-lines:])


def export_report(ncu: Path, report_stem: Path, csv_path: Path) -> dict[str, str]:
    """Export a saved report to CSV with `ncu --import`, keeping only the focus metrics.

    Failure is reported and swallowed: the report itself is still valid and the caller may be
    running on a machine without a working `--import`. Returns the kept metric/value pairs so the
    caller can check the launch it captured.
    """
    raw = csv_path.with_name(csv_path.stem + ".full.csv")
    try:
        completed = subprocess.run(
            [str(ncu), "--import", str(report_stem.with_suffix(".ncu-rep")), "--csv"],
            capture_output=True, check=False, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as error:
        shared.log(f"  CSV export skipped: {error}")
        return {}
    text = completed.stdout.decode("utf-8", errors="replace")
    if completed.returncode != 0 or not text.strip():
        detail = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
        shared.log(f"  CSV export failed (rc={completed.returncode}): "
                   f"{(detail[0] if detail else 'no stderr')}")
        return {}
    raw.write_text(text, encoding="utf-8")
    try:
        columns = list(csv.DictReader(text.splitlines()))
        pattern = re.compile(FOCUS_METRIC_PATTERN, re.IGNORECASE)
        kept = [row for row in columns if pattern.search(row.get("Metric Name") or "")]
    except (csv.Error, StopIteration) as error:
        shared.log(f"  CSV export kept unparsed: {error}")
        raw.replace(csv_path)
        return {}
    if not kept:
        shared.log("  CSV export matched no focus metrics; keeping the full export")
        raw.replace(csv_path)
        return {}
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    raw.unlink(missing_ok=True)
    shared.log(f"  kept {len(kept)} of {len(columns)} metric rows")
    return {row["Metric Name"]: row["Metric Value"] for row in kept
            if (row.get("Metric Name") or "").strip()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ncu = find_ncu(args.ncu)
    args.out.mkdir(parents=True, exist_ok=True)
    shared.log(f"ncu    : {ncu}")
    shared.log(f"server : {args.server}")
    shared.log(f"model  : {args.model}")
    shared.log(f"K      : {args.ks}; each point uses a separate server")
    shared.log(f"output : {args.out}")
    try:
        for index, k in enumerate(args.ks):
            run_point(args, ncu, k, args.port + index)
    except KeyboardInterrupt:
        shared.log("interrupted")
        shared._cleanup_all()
        return 130
    except Exception as error:  # noqa: BLE001
        shared.log(f"ERROR: {error}")
        shared._cleanup_all()
        return 1

    print("\nProfile both reports in Nsight Compute and compare LaunchStats, Occupancy, Scheduler, "
          "Memory Workload, and Warp State sections.")
    print(f"Reports: {args.out / 'k3'} and {args.out / 'k5'}")
    print("NCU timings are replay-instrumented; use them for bottleneck attribution, not end-to-end speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
