#!/usr/bin/env python3
"""Find where the extra round time goes at K=5 vs K=3. Written for a total beginner.

THE QUESTION
    At long context, K=5 gives +20.6% more tokens per round than K=3, but also costs
    +21.5% more time per round, so throughput is a wash (99.1 vs 99.9 tok/s). If the extra
    ~6.5 ms per round could be removed, K=5 would be roughly +20% faster than K=3.
    This tool finds out WHERE that 6.5 ms goes.

START SHORT, THEN CONFIRM AT DEPTH
    A short prompt is the right FIRST run: prefill is instant, the trace has few kernels, and
    per-column work stands out. Use it to find WHICH kernels scale with K.

    It is not sufficient on its own. A short-context trace accounts for ~1.5 ms of the ~3.25 ms
    that each extra column costs at 150K, so roughly 2 ms per column is only visible at depth.
    Attention in particular is NOT flat in K inside the real verify path, even though the
    isolated op benchmark reports it as flat -- it nearly doubles per call. A short run therefore
    cannot tell you how the cost splits at the deployment context (--max-context 131144).

    So run short first, then repeat at the real depth:

        python tools\\bench\\profile_verify_path.py --server ... --model ... ^
            --max-context 40000 --longctx-repeats 280

    To re-read profiles that were copied in from another machine, profile nothing:

        python tools\\bench\\profile_verify_path.py --analyze-only

    (with --out pointing at the copied directory, and optionally --ks 3,5).

    The summary prints the context depth the request actually reached, taken from the server's
    own computed_prefill_tokens, plus the tokens-per-repeat that implies. Calibrate
    --longctx-repeats against that measured number -- never against a chars/4 estimate.

HOW TO RUN (Windows, from the repo root, in cmd or PowerShell)

    python tools\\bench\\profile_verify_path.py ^
        --server build\\apps\\Release\\ninfer-serve.exe ^
        --model  E:\\models\\qwen3_8_27b.ninfer

or in PowerShell:

    python tools/bench/profile_verify_path.py `
        --server build/apps/Release/ninfer-serve.exe `
        --model  E:/models/qwen3_8_27b.ninfer

WHAT YOU GET
    profiles\\verify-profile\\k3\\ and ...\\k5\\, each with a .nsys-rep you can open by
    double-clicking it in the Nsight Systems GUI. The script also prints a side-by-side
    kernel-time table automatically, sorted by how much each kernel grew, and the context
    depth each request actually reached.

IMPORTANT: by default the server is run with --no-cuda-graph. Verify normally runs inside a
captured CUDA Graph, which nsys would show as one opaque node, hiding exactly what we want.
Turning the graph off makes every kernel visible. Absolute timings will be inflated by launch
overhead, but the K=3-vs-K=5 DIFFERENCE is still valid -- and that is what we care about.

GRAPH-ON MODE
    That launch inflation is fatal to one specific question: is a kernel launch-bound or
    compute-bound? With graphs off, a kernel called 26,000 times looks expensive because of
    dispatch, and optimising its arithmetic would be wasted work. Pass

        --cuda-graph --cuda-graph-trace=node

    to leave graphs ENABLED while nsys still records every kernel inside each graph separately.
    Per-kernel GPU time stays readable and the dispatch cost is realistic. Compare a graph-on
    and a graph-off trace of the same request: a kernel whose time collapses is launch-bound;
    one that barely moves is genuinely compute-bound and worth optimising.

If nsys is not on PATH, the script probes the usual Windows install locations. Point it at
yours with --nsys if it still cannot find it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

DEFAULT_NSYS_LOCATIONS = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems",
    r"C:\Program Files (x86)\NVIDIA Corporation\Nsight Systems",
    r"C:\NVIDIA\Nsight Systems",
]

DEFAULT_PROMPT = (
    "Solve step by step: find the smallest positive integer n such that "
    "n^2 + 3n + 5 is divisible by 77."
)

# The Gate B paragraph is 607 chars / 106 words and lands near 140 Qwen tokens per repeat, so
# 280 repeats is roughly 39K tokens. That is an ESTIMATE: the server's computed_prefill_tokens is
# the number of record, which is why every run prints the depth it actually reached.
APPROX_TOKENS_PER_REPEAT = 140


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_longctx_prompt(repeats: int) -> str:
    """Pad the prompt to depth with the same in-distribution prose the Gate B sweep uses.

    The paragraph and its calibration note live in run_gate_b_acceptance_sweep.py so the two
    campaigns cannot drift apart. Imported lazily so this script still runs standalone.
    """
    try:
        from tools.bench.run_gate_b_acceptance_sweep import build_longctx_prompts
    except ModuleNotFoundError:
        # Running as a plain script: tools/bench is sys.path[0], not the repo root.
        from run_gate_b_acceptance_sweep import build_longctx_prompts  # type: ignore[no-redef]
    return build_longctx_prompts(repeats)[0]


def read_reached_depth(request_log: Path, start_offset: int = 0) -> dict[str, int] | None:
    """Report how deep the last completed request actually ran.

    Depth is `prompt_tokens`: the length of the context the decode ran against. It is NOT
    `computed_prefill_tokens`, which counts prefill *work*. The warmup request primes the very
    same prompt, so the measured request reuses that prefix and computes only a couple of prefill
    tokens while still decoding at full depth -- treating prefill work as depth reports a 2-token
    context for a 45K-token run. Both numbers are returned so the reuse is visible rather than
    silently misread.

    `start_offset` skips records written by an earlier run. The server log is append-mode, so a
    directory reused across runs would otherwise report the previous run's depth.
    """
    if not request_log.exists():
        return None
    try:
        raw = request_log.read_bytes()
    except OSError:
        return None

    last: dict[str, Any] | None = None
    for line in raw[start_offset:].decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "request_done":
            last = event
    if last is None:
        return None

    result = last.get("result") or {}
    if not isinstance(result, dict):
        return None

    def as_int(value: object) -> int:
        # A malformed field must not fail an entire profiled run after the fact.
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    prompt = as_int(result.get("prompt_tokens", 0))
    prefill = as_int(result.get("computed_prefill_tokens", 0))
    completion = as_int(result.get("completion_tokens", 0))
    # `prompt_tokens` is the context length. Fall back to prefill work only when the server did
    # not report it, which old schema versions do not.
    depth = prompt if prompt > 0 else prefill
    return {
        "prompt_tokens": prompt,
        "prefill_work_tokens": prefill,
        "reused_tokens": max(0, prompt - prefill),
        "completion_tokens": completion,
        "final_context": depth + completion,
    }


# ---------------------------------------------------------------------------
# Locating nsys
# ---------------------------------------------------------------------------

def find_nsys(explicit: str | None) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise SystemExit(f"ERROR: --nsys path does not exist: {explicit}")

    found = shutil.which("nsys")
    if found:
        return Path(found)

    # Windows: probe the usual install roots and pick the newest version.
    for root in DEFAULT_NSYS_LOCATIONS:
        base = Path(root)
        if not base.exists():
            continue
        versions = sorted(
            (p for p in base.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        for version_dir in versions:
            for relative in (
                Path("target-windows-x64") / "nsys.exe",
                Path("windows-desktop-q64") / "nsys.exe",
            ):
                candidate = version_dir / relative
                if candidate.exists():
                    return candidate
            # Some installs put the exe directly in the version directory.
            direct = version_dir / "nsys.exe"
            if direct.exists():
                return direct

    return None


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def wait_ready(port: int, timeout: float) -> bool:
    """Poll /health until it answers 200. Never raises on connection errors.

    A refused connection just means the server has not opened its socket yet, which is
    the normal case while the model loads. Any other HTTP status is also not fatal yet.
    """
    deadline = time.monotonic() + timeout
    last_note = 0.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            # Expected while loading. Note progress occasionally so a hang is visible.
            now = time.monotonic()
            if now - last_note >= 10.0:
                remaining = deadline - now
                log(f"    still waiting for /health on port {port} "
                    f"({remaining:.0f}s left)...")
                last_note = now
            time.sleep(0.5)
    return False


def probe_once(port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Single /health probe. Returns (ok, human-readable reason).

    NOTE: on a machine where WSL and Windows share 127.0.0.1, a request sent to
    127.0.0.1 can land on a server running in the OTHER environment. The
    --port-occupied pre-check in run_point catches that case before it can
    contaminate a measurement.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            return (response.status == 200,
                    f"HTTP {response.status}" if response.status != 200 else "ok")
    except urllib.error.HTTPError as error:
        return False, f"HTTP {error.code} {error.reason}"
    except urllib.error.URLError as error:
        return False, f"{type(error.reason).__name__}: {error.reason}"
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {error}"


def post_chat(port: int, prompt: str, max_tokens: int, seed: int,
              timeout: float) -> dict[str, Any]:
    """Send one chat request and return the parsed response."""
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


def _dump_log(server_log: Path, lines: int = 40) -> None:
    """Print the tail of the server log so a failure is self-explaining."""
    if not server_log.exists():
        log(f"    (no server log at {server_log})")
        return
    try:
        content = server_log.read_text(errors="replace").splitlines()
    except OSError as error:
        log(f"    (could not read {server_log}: {error})")
        return
    if not content:
        log(f"    (server log is empty: {server_log})")
        return
    log(f"    last {min(lines, len(content))} lines of {server_log}:")
    for line in content[-lines:]:
        print(f"      {line}")


# PIDs of servers we started, so an unexpected exit still cleans up.
_LIVE_PIDS: set[int] = set()


# nsys writes its report when it EXITS. Force-killing it truncates the .nsys-rep, which is how a
# profiled point ends up with a report that carries no kernel data at all.
GRACEFUL_STOP_SECONDS = 30.0

# After finalising, nsys reports the signal that interrupted it as its exit status: 130 on
# POSIX (128 + SIGINT) and 0xC000013A on Windows. Both are normal ends of a profiled point.
CLEAN_STOP_CODES = {0, 128 + int(signal.SIGINT), 0xC000013A, -1073741510}


def _clean_stop(returncode: int | None) -> bool:
    return returncode is None or returncode in CLEAN_STOP_CODES


def _interrupt_tree(pid: int) -> bool:
    """Ask nsys (and its server child) to stop on their own so the report can be finalised.

    Returns True when the request was delivered; the caller still has to check whether the
    process honoured it, because this is a request and not a guarantee.
    """
    if os.name == "nt":
        # CTRL_BREAK reaches nsys and its child only because they were started in their own
        # process group (CREATE_NEW_PROCESS_GROUP); otherwise it would hit this script too.
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is None:
            return False
        try:
            os.kill(pid, ctrl_break)
            return True
        except OSError:
            return False
    # SIGINT is console Ctrl-C, which nsys treats as "stop and write the report".
    try:
        os.killpg(os.getpgid(pid), signal.SIGINT)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_tree(pid: int) -> None:
    """Kill a process AND its children.

    This matters a great deal here. We launch `nsys.exe`, and nsys launches
    `ninfer-serve.exe` as a CHILD. Terminating only nsys orphans the server, which keeps
    ~16.7 GiB of model weights resident on the GPU. After a couple of orphans the next run
    fails with out-of-memory, and the failure looks like a script bug rather than a leak.
    """
    if os.name == "nt":
        # /T = tree, /F = force. nsys spawns the server as a child, so /T is required.
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=30)
        return
    # POSIX: we start each server in its own session (start_new_session=True), so the
    # whole group can be signalled at once.
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    # Give it a moment, then force.
    for _ in range(20):
        try:
            os.killpg(os.getpgid(pid), 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def stop(process: subprocess.Popen[bytes], port: int | None = None) -> bool:
    """Stop the server and everything it spawned. Returns True if the port came free."""
    pid = process.pid
    _LIVE_PIDS.discard(pid)

    if process.poll() is None:
        # Graceful first: nsys finalises the report during its own shutdown. A hard kill here
        # costs the entire trace, which is a much worse outcome than a slow teardown.
        if _interrupt_tree(pid):
            log(f"    asking the profiler to stop so it can finalise its report "
                f"(up to {GRACEFUL_STOP_SECONDS:.0f}s)...")
            try:
                process.wait(timeout=GRACEFUL_STOP_SECONDS)
            except Exception:  # noqa: BLE001
                log("    it did not exit on request; killing the process tree")
        if process.poll() is None:
            try:
                _kill_tree(pid)
            except Exception as error:  # noqa: BLE001
                log(f"    (warning: could not kill process tree for pid {pid}: {error})")
            try:
                process.wait(timeout=25)
            except Exception:  # noqa: BLE001
                try:
                    process.kill()
                    process.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass

    # Always kill the tree even if the parent already exited: nsys may have died while its
    # child server kept running (that is the exact leak we are guarding against).
    try:
        _kill_tree(pid)
    except Exception:  # noqa: BLE001
        pass

    if port is None:
        return True

    # Verify the port actually came free. If it did not, the server is still holding VRAM
    # and the next point will either fail to bind or OOM.
    for _ in range(40):
        occupied, _why = probe_once(port, timeout=1.0)
        if not occupied:
            return True
        time.sleep(0.25)

    log(f"    WARNING: port {port} is STILL serving /health after shutdown.")
    log("             A server is probably still holding GPU memory.")
    log(f"             Kill it before continuing:  taskkill /F /IM ninfer-serve.exe")
    return False


def _cleanup_all() -> None:
    """Best-effort teardown of every server we started (runs on any exit path)."""
    for pid in list(_LIVE_PIDS):
        try:
            _kill_tree(pid)
        except Exception:  # noqa: BLE001
            pass
        _LIVE_PIDS.discard(pid)


# ---------------------------------------------------------------------------
# Kernel-time comparison
# ---------------------------------------------------------------------------

def sqlite_kernel_rows(sqlite_path: Path) -> int | None:
    """Count CUDA kernel rows in an exported report.

    0 means the file exported but the report carried no kernel data -- a stale, truncated, or
    interrupted .nsys-rep. None means the file could not be opened at all.
    """
    try:
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        connection.close()


def read_kernel_times(report: Path,
                      nsys: Path | None = None) -> tuple[dict[str, tuple[float, int]] | None, str]:
    """Sum GPU kernel time per kernel name from an nsys report.

    IMPORTANT: a .nsys-rep is NOT a sqlite database -- it is nsys's own binary format.
    It must first be exported with `nsys stats` (or `nsys export`) to produce a .sqlite
    file that can be queried. This function does that export automatically.

    Returns ({kernel_name: (total_ms, count)}, "") on success, or (None, reason) so the caller
    can say what actually went wrong instead of guessing.
    """
    if not report.exists():
        return None, f"{report.name} is missing"
    if report.stat().st_size == 0:
        return None, f"{report.name} is empty (0 bytes)"

    # Prefer a sibling .sqlite, but re-export when it is missing, older than the report, or
    # unusable -- and never silently trust a 0-kernel export.
    sqlite_path = report.with_suffix(".sqlite")
    needs_export = (
        not sqlite_path.exists()
        or sqlite_path.stat().st_mtime < report.stat().st_mtime
        or not sqlite_kernel_rows(sqlite_path)
    )
    if needs_export:
        exported, detail = export_to_sqlite(report, nsys)
        if exported is None:
            return None, (f"{report.name} ({report.stat().st_size:,} bytes) produced no kernel "
                          f"data: {detail}")
        sqlite_path = exported

    try:
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        return None, f"could not open {sqlite_path.name}: {error}"
    try:
        rows = connection.execute(
            """
            SELECT s.value AS name, SUM(k.end - k.start) AS total_ns, COUNT(*) AS n
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            GROUP BY s.value
            """
        ).fetchall()
    except sqlite3.Error as error:
        return None, f"{sqlite_path.name} could not be queried: {error}"
    finally:
        connection.close()
    if not rows:
        return None, f"{sqlite_path.name} contains 0 kernels"
    return {name: (total_ns / 1.0e6, count) for name, total_ns, count in rows}, ""


def export_to_sqlite(report: Path, nsys: Path | None = None) -> tuple[Path | None, str]:
    """Export a .nsys-rep to .sqlite so it can be queried.

    A failed export must never take a good export down with it. The existing .sqlite is moved
    aside first and only replaced once the new one demonstrably contains CUDA kernel rows:
    `nsys stats --force-export` will cheerfully overwrite a valid 15 MB export with an empty
    one derived from a truncated .nsys-rep, which loses the analysis with no warning.

    Returns (sqlite_path, "") on success, or (None, reason) -- reason carries nsys's own
    explanation, e.g. "SKIPPED: ... does not contain CUDA kernel data".
    """
    tool = nsys if nsys is not None and nsys.exists() else find_nsys(None)
    if tool is None:
        return None, "nsys not found; cannot export the report"

    target = report.with_suffix(".sqlite")
    backup = target.with_name(target.name + ".keep")
    if target.exists():
        try:
            target.replace(backup)
        except OSError as error:
            return None, f"could not set the existing {target.name} aside: {error}"

    try:
        completed = subprocess.run(
            [str(tool), "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum",
             str(report)],
            capture_output=True, timeout=300, text=True,
        )
    except Exception as error:  # noqa: BLE001
        if backup.exists():
            backup.replace(target)
        return None, f"export failed: {error}"

    notes = [line.strip()
             for line in f"{completed.stdout}\n{completed.stderr}".splitlines()
             if line.strip().lower().startswith(("skipped", "warning", "error", "failed"))]
    detail = "; ".join(notes[:3])

    exported_rows = sqlite_kernel_rows(target) if target.exists() else 0
    if exported_rows:
        backup.unlink(missing_ok=True)
        return target, detail

    if backup.exists():
        kept_rows = sqlite_kernel_rows(backup) or 0
        target.unlink(missing_ok=True)
        backup.replace(target)
        restored = (f"previous {target.name} restored, {kept_rows} kernels" if kept_rows
                    else f"previous {target.name} was empty too")
    else:
        restored = "no previous export to restore"
    reason = detail or f"nsys exited with status {completed.returncode}"
    return None, f"{reason} ({restored})"


def find_report(directory: Path) -> Path | None:
    for pattern in ("*.nsys-rep", "*.sqlite"):
        matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
        if matches:
            return matches[-1]
    return None


def print_comparison(out_dir: Path, ks: Sequence[int], nsys: Path | None = None,
                     cuda_graph: bool = False) -> None:
    data: dict[int, dict[str, tuple[float, int]]] = {}
    reasons: dict[int, str] = {}
    for k in ks:
        report = find_report(out_dir / f"k{k}")
        if report is None:
            data[k] = {}
            reasons[k] = f"nothing to read in {out_dir / f'k{k}'}"
            continue
        kernels, reason = read_kernel_times(report, nsys)
        data[k] = kernels or {}
        if reason:
            reasons[k] = reason

    usable = [k for k in ks if data[k]]
    missing = [k for k in ks if not data[k]]

    if missing and usable:
        # One dead report must not hide the points that did capture correctly.
        log(f"no kernel data for K={missing} (table below uses K={usable} only):")
        for k in missing:
            log(f"  k{k}: {reasons.get(k, 'unknown reason')}")
        log("")

    if not usable:
        log(f"could not read kernel data for K={missing}:")
        for k in missing:
            log(f"  k{k}: {reasons.get(k, 'unknown reason')}")
        log("")
        log("An empty .nsys-rep (a few tens of KB) means the trace never captured GPU work:")
        log("the profiled process was killed before nsys flushed the report, or the run did not")
        log("reach the GPU at all. Re-run that point; do not re-export it, there is nothing there.")
        log("")
        log("GPU kernel tracing being blocked has a different signature -- the report is full")
        log("size but has no kernel rows. On WSL, check:  cat /proc/sys/kernel/perf_event_paranoid")
        log("  A value of 2 blocks it. On native Windows this is usually fine.")
        log("")
        log("Fallback A -- CPU-side API counts. This DOES work even when GPU kernel")
        log("tracing is blocked, and it is still informative: per-round cudaLaunchKernel")
        log("counts tell you whether the extra cost is more launches or slower kernels.")
        log("Normalize by rounds (tokens / tokens-per-round) before comparing K values!")
        for k in ks:
            report = find_report(out_dir / f"k{k}")
            if report:
                log(f"    nsys stats --report cuda_api_sum {report}")
        log("")
        log("Fallback B -- open the report in the GUI. Even without kernel-exec rows the")
        log("CUDA API row shows the launch pattern over time:")
        for k in ks:
            report = find_report(out_dir / f"k{k}")
            if report:
                log(f"    {report}")
        log("")
        log("Fallback C -- if GPU kernel tracing is unavailable, use Nsight Compute on a")
        log("single kernel instead; ncu does not need perf_event access the same way:")
        log("    ncu --set full --launch-count 1 --kernel-name regex:<name> <server cmd>")
        return

    names = sorted(set().union(*[set(data[k]) for k in usable]))
    rows = []
    for name in names:
        times = [data[k].get(name, (0.0, 0))[0] for k in usable]
        counts = [data[k].get(name, (0.0, 0))[1] for k in usable]
        # With a single point there is no delta to sort by, so sort by absolute time.
        key = times[-1] - times[0] if len(usable) > 1 else times[-1]
        rows.append((key, name, times, counts))
    rows.sort(reverse=True)

    print()
    print("=" * 110)
    print("  Kernel GPU time by K  (sorted by how much it grew)")
    print("=" * 110)
    print()
    print("Read the top of this list: those kernels are where the extra round time goes.")
    if cuda_graph:
        print("The server ran with CUDA Graphs ON, so launch overhead is realistic and per-kernel")
        print("time is close to what production pays. The DELTA between K values is still the")
        print("quantity to read; absolute totals now include graph-replay dispatch.")
    else:
        print("Absolute numbers are inflated by launch overhead because --no-cuda-graph is on;")
        print("the DELTA between K values is the meaningful quantity. Re-run with")
        print("--cuda-graph --cuda-graph-trace=node to judge whether a kernel is launch-bound.")
    print()
    header = (f"{'kernel':<52}"
              + "".join(f"{'K=' + str(k):>15}" for k in usable)
              + f"{'delta':>12}")
    print(header)
    print("-" * len(header))
    for delta, name, times, counts in rows:
        cells = ""
        for t, c in zip(times, counts):
            cells += f"{t:>8.2f}ms x{c:<5}"
        print(f"{name[:50]:<52}{cells}{delta:>+10.2f}ms")
    print()
    print("Names are truncated to 50 characters here; the GUI shows full names.")


# ---------------------------------------------------------------------------
# One measurement point
# ---------------------------------------------------------------------------

def run_point(args: argparse.Namespace, k: int, nsys: Path, port: int,
              out_dir: Path) -> dict[str, Any]:
    tag = f"k{k}"
    directory = out_dir / tag
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / f"verify_{tag}"
    # Needed for one reason: it carries the server's own computed_prefill_tokens, which is how
    # this script proves the run actually reached the context depth it claims. One small write per
    # request; the trace records only CUDA and NVTX, so it does not perturb the kernels.
    request_log = directory / "server.requests.jsonl"

    command = [
        str(nsys), "profile",
        # cuda + nvtx only: portable across nsys builds. 'osrt' is Linux-only and some
        # Windows builds reject it outright (and 'wddm' is not accepted by older ones).
        "--trace=cuda,nvtx",
    ]
    # Without an explicit mode, nsys records a CUDA Graph as one opaque node and every kernel
    # inside it loses its identity -- which is exactly what this tool exists to measure. 'node'
    # keeps each kernel visible while the graph still runs as a graph, so launch overhead stays
    # realistic instead of being inflated by graph-off dispatch.
    if args.cuda_graph_trace:
        command.append(f"--cuda-graph-trace={args.cuda_graph_trace}")
    command += [
        "--sample=none",
        "--cpuctxsw=none",
        "--stats=false",
        "--force-overwrite=true",
        f"--output={report}",
        "--",
        str(args.server),
        str(args.model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-context", str(args.max_context),
        "--max-concurrency", "1",
        "--kv-dtype", args.kv_dtype,
    ]
    if not args.cuda_graph:
        command.append("--no-cuda-graph")
    command += [
        "--spec", "mtp",
        "--draft-tokens", str(k),
        "--lm-head-draft",
        "--request-log-jsonl", str(request_log),
    ]

    log(f"--- {tag}: starting the server under nsys ---")
    log(f"    report -> {report}.nsys-rep")

    # If something is already listening on this port, the new server will fail to bind and
    # the health probe would talk to the OLD server. Fail early and clearly instead.
    occupied, why = probe_once(port, timeout=3.0)
    if occupied:
        raise RuntimeError(
            f"port {port} is already serving /health ({why}). Another ninfer-serve is "
            "still running -- possibly in WSL, since WSL and Windows share 127.0.0.1.\n"
            f"       Kill it, or pick a different port with --port {port + 100}."
        )

    server_log = directory / "server.log"
    # The request log is append-mode, so a reused output directory also holds records from
    # earlier runs. Only bytes written after this offset belong to this run.
    log_offset = request_log.stat().st_size if request_log.exists() else 0
    with server_log.open("wb") as handle:
        # start_new_session puts the child in its own process group on POSIX so the whole
        # tree (nsys -> ninfer-serve) can be signalled together. On Windows we use taskkill /T.
        # Windows: a dedicated process group is what makes CTRL_BREAK reach nsys (and nothing
        # else), so the profiler can shut down cleanly and write its report.
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"), creationflags=creationflags,
        )
        _LIVE_PIDS.add(process.pid)
        try:
            log(f"{tag}: waiting for the model to load (~10 s expected)...")
            if not wait_ready(port, args.startup_timeout):
                # Distinguish "nsys or the server exited" from "it is just slow".
                if process.poll() is not None:
                    raise RuntimeError(
                        f"the profiler/server process exited with status "
                        f"{process.returncode} before serving /health. "
                        f"See {server_log}"
                    )
                raise RuntimeError(
                    f"server did not become ready within {args.startup_timeout:.0f}s. "
                    f"See {server_log} for the reason"
                )
            log(f"{tag}: ready.")

            try:
                log(f"{tag}: warmup request (settles caches and workspaces, not measured)")
                try:
                    post_chat(port, args.prompt, 32, args.seed, args.request_timeout)
                except Exception as error:  # noqa: BLE001
                    log(f"{tag}: warmup failed ({error}); continuing anyway")

                log(f"{tag}: measured request -- this is the one in the trace")
                started = time.monotonic()
                response = post_chat(port, args.prompt, args.output_tokens, args.seed,
                                     args.request_timeout)
                elapsed = time.monotonic() - started
                usage = response.get("usage", {}) or {}
                completion = usage.get("completion_tokens") or 0

                # A stray request from another environment would show up as an implausibly
                # fast response for the token count we asked for. Flag it rather than
                # silently reporting a contaminated measurement.
                if completion and elapsed > 0 and completion / elapsed > 2000:
                    log(f"{tag}: WARNING {completion} tokens in {elapsed:.2f}s looks too fast;")
                    log(f"{tag}:          a request may have been served by a DIFFERENT server")
                    log(f"{tag}:          (WSL and Windows share 127.0.0.1). Rerun with a")
                    log(f"{tag}:          unique --port if the report looks wrong.")

                log(f"{tag}: generated {completion} tokens in {elapsed:.1f}s "
                    f"({completion / elapsed:.1f} tok/s)")

                # Let nsys flush the report before we shut the server down.
                time.sleep(args.flush_seconds)

                depth = read_reached_depth(request_log, log_offset)
                if depth is None:
                    log(f"{tag}: WARNING no request_done event in {request_log};")
                    log(f"{tag}:          context depth is UNKNOWN for this point.")
                else:
                    log(f"{tag}: depth reached: {depth['prompt_tokens']} prompt tokens "
                        f"({depth['prefill_work_tokens']} prefilled, "
                        f"{depth['reused_tokens']} reused) + {depth['completion_tokens']} "
                        f"generated = {depth['final_context']} tokens")
            except KeyboardInterrupt:
                log(f"{tag}: interrupted. Shutting the server down...")
                raise
            except Exception as error:  # noqa: BLE001
                log(f"{tag}: {type(error).__name__}: {error}")
                _dump_log(server_log)
                raise
        finally:
            freed = stop(process, port=port)
            if not freed:
                log(f"{tag}: the server did not release port {port}; GPU memory may still be")
                log(f"{tag}: allocated. Stopping the campaign so we do not OOM on the next point.")
                raise RuntimeError(
                    f"server on port {port} did not shut down cleanly"
                )
            if not _clean_stop(process.returncode):
                log(f"{tag}: server exit status {process.returncode}")
                _dump_log(server_log, lines=20)

    # Validate the trace now, not at the analysis step: by then the next K has already spent its
    # GPU time, and an empty report silently turns into "nsys cannot make a sqlite" later.
    recorded = report.with_suffix(".nsys-rep")
    kernel_calls: int | None = None
    if not recorded.exists():
        log(f"{tag}: WARNING nsys wrote no report at {recorded}")
    else:
        kernels, reason = read_kernel_times(recorded, nsys)
        if kernels is None:
            log(f"{tag}: WARNING the trace holds no usable kernel data: {reason}")
            log(f"{tag}:          re-run this point; nsys most likely never finalised the report")
            log(f"{tag}:          because the profiled process was force-killed at shutdown.")
        else:
            kernel_calls = sum(count for _, count in kernels.values())
            log(f"{tag}: trace ok: {kernel_calls} kernel launches in {recorded.name} "
                f"({recorded.stat().st_size:,} bytes)")

    return {
        "tag": tag,
        "k": k,
        "completion_tokens": completion,
        "wall_seconds": elapsed,
        "client_tok_s": completion / elapsed if elapsed > 0 else None,
        "depth": depth,
        "kernel_calls": kernel_calls,
        "server_log": str(directory / "server.log"),
        "command": command,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the speculative verify path at two MTP widths and compare.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --help at the top of this file for a full beginner explanation.",
    )
    parser.add_argument("--server", type=Path, default=None,
                        help="path to ninfer-serve.exe (required unless --analyze-only)")
    parser.add_argument("--model", type=Path, default=None,
                        help="path to the .ninfer artifact (required unless --analyze-only)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="re-read the reports and request logs already under --out; start no "
                             "server and profile nothing")
    parser.add_argument("--out", type=Path, default=Path("profiles/verify-profile"),
                        help="output directory")
    parser.add_argument("--ks", default="3,5",
                        help="comma-separated MTP draft widths to compare, e.g. 3,5")
    parser.add_argument("--nsys", default=None,
                        help="path to nsys.exe; probed automatically if omitted")
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--output-tokens", type=int, default=256,
                        help="tokens to generate in the measured request")
    parser.add_argument("--max-context", type=int, default=4096,
                        help="server --max-context; short is fine and faster")
    parser.add_argument("--kv-dtype", default="rk4v4-e8")
    parser.add_argument("--seed", type=int, default=57004)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--longctx-repeats", type=int, default=0,
                        help="pad the prompt with the Gate B prose repeated N times to reach depth "
                             "(~140 tokens per repeat); 0 keeps --prompt as-is")
    parser.add_argument("--startup-timeout", type=float, default=600.0,
                        help="seconds to wait for the model to load")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--flush-seconds", type=float, default=5.0,
                        help="seconds to wait after the request before shutting down")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="leave CUDA Graphs ENABLED in the server (default: off). Graph-off "
                             "inflates launch overhead, which flatters high-launch-count kernels, "
                             "so use this when the question is whether a kernel is launch-bound "
                             "or compute-bound. Pair it with --cuda-graph-trace=node or the whole "
                             "graph appears as one opaque node")
    parser.add_argument("--cuda-graph-trace", default=None, choices=("node", "graph"),
                        help="nsys --cuda-graph-trace mode. 'node' records every kernel inside a "
                             "captured graph separately, which is what keeps per-kernel GPU time "
                             "readable while the graph runs; 'graph' collapses each graph to a "
                             "single entry. Ignored when the server runs with graphs disabled")
    args = parser.parse_args(argv)

    if args.analyze_only:
        if not args.out.exists():
            parser.error(f"--analyze-only needs an existing --out directory: {args.out}")
    else:
        if args.server is None or args.model is None:
            parser.error("--server and --model are required unless --analyze-only is given")
        if not args.server.exists():
            parser.error(f"server not found: {args.server}")
        if not args.model.exists():
            parser.error(f"model not found: {args.model}")
    if args.longctx_repeats < 0:
        parser.error("--longctx-repeats cannot be negative")

    if args.longctx_repeats > 0:
        # Build the padded prompt once, before any server starts, so every K gets identical
        # input text and the only variable stays the draft width.
        args.prompt = build_longctx_prompt(args.longctx_repeats)
        estimated = args.longctx_repeats * APPROX_TOKENS_PER_REPEAT
        if estimated > args.max_context:
            print(f"WARNING: --longctx-repeats {args.longctx_repeats} is roughly {estimated} "
                  f"tokens, beyond --max-context {args.max_context}.", file=sys.stderr)
            print(f"  Raise it:  --max-context {int(estimated * 1.1)}", file=sys.stderr)
            print("  The estimate is rough; the server will reject the request if it does "
                  "not fit, and the summary prints the depth actually reached.", file=sys.stderr)

    k_values = []
    for piece in args.ks.split(","):
        piece = piece.strip()
        if piece:
            k_values.append(int(piece))
    if len(k_values) < 2:
        parser.error("need at least two K values to compare")
    args.k_values = k_values
    return args


def print_depth(points: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    """State the depth each point actually reached, and calibrate --longctx-repeats."""
    print()
    print("=" * 110)
    print("  Context depth actually reached (prompt_tokens from the server's request log)")
    print("=" * 110)
    print("  prompt = context the decode ran against; prefill = tokens it actually recomputed")
    print(f"{'K':>4}{'prompt':>12}{'prefill':>10}{'generated':>12}{'final_ctx':>12}")
    print("-" * 52)
    deepest_prompt = 0
    deepest_final = 0
    reuse_note = False
    for point in points:
        depth = point.get("depth") or {}
        prompt = int(depth.get("prompt_tokens", 0) or 0)
        prefill = int(depth.get("prefill_work_tokens", 0) or 0)
        generated = int(depth.get("completion_tokens", 0) or 0)
        final = int(depth.get("final_context", 0) or 0)
        deepest_prompt = max(deepest_prompt, prompt)
        deepest_final = max(deepest_final, final)
        if prompt > 0 and prefill < prompt // 2:
            reuse_note = True
        print(f"{point['k']:>4}{prompt:>12}{prefill:>10}{generated:>12}{final:>12}")

    if reuse_note:
        print()
        print("  Note: prefill is small because the warmup request primed the same prompt and the")
        print("  prefix was reused. The decode still ran at the full `prompt` depth -- that is the")
        print("  number that matters, and it is why computed_prefill_tokens is not used as depth.")

    if args.longctx_repeats > 0 and deepest_prompt > 0:
        per_repeat = deepest_prompt / args.longctx_repeats
        print()
        print(f"  {deepest_prompt} prompt tokens / {args.longctx_repeats} repeats = "
              f"~{per_repeat:.1f} tokens per repeat (includes the task sentence).")
        print(f"  For a different depth D, set --longctx-repeats to round(D / {per_repeat:.1f}).")
    elif deepest_final < 8192 and args.max_context >= 16384:
        print()
        print(f"  WARNING: the deepest request reached only {deepest_final} tokens.")
        print("           --max-context moves the ceiling; it does not move the prompts.")
        print("           Per-column cost at 150K is roughly 2x what this trace can show, so")
        print("           treat the split above as a SHORT-context result and confirm at depth")
        print("           with --longctx-repeats.")


def run_analyze_only(args: argparse.Namespace, nsys: Path | None) -> int:
    """Re-read an output directory that was copied in, without profiling anything."""
    log(f"analysing existing reports under {args.out} (no server started)")
    points: list[dict[str, Any]] = []
    for k in args.k_values:
        directory = args.out / f"k{k}"
        request_log = directory / "server.requests.jsonl"
        depth = read_reached_depth(request_log)
        points.append({"k": k, "depth": depth})
        if depth is None:
            log(f"k{k}: no request log with a completed request in {directory}")
        # A copied directory can pair a fresh request log with an older trace, and reading that
        # trace as if it belonged to this run is how a short-context table gets mistaken for a
        # deep-context one. Print both timestamps rather than guessing at a threshold.
        report = find_report(directory)
        if report is not None and request_log.exists():
            trace_at = time.strftime("%m-%d %H:%M", time.localtime(report.stat().st_mtime))
            log_at = time.strftime("%m-%d %H:%M", time.localtime(request_log.stat().st_mtime))
            log(f"k{k}: trace {report.name} ({trace_at},{report.stat().st_size:,} bytes) | "
                f"request log {log_at}")
            log(f"k{k}: if those timestamps are far apart, the table below is from a different run")
    print_depth(points, args)
    print_comparison(args.out, args.k_values, nsys, args.cuda_graph)
    readable = [k for k in args.k_values
                if (report := find_report(args.out / f"k{k}")) is not None
                and sqlite_kernel_rows(report.with_suffix(".sqlite"))]
    return 0 if readable else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Whatever happens -- normal exit, Ctrl-C, or an unhandled error -- do not leave a
    # server holding GPU memory.
    import atexit
    atexit.register(_cleanup_all)

    def _on_sigterm(signum, _frame):  # noqa: ANN001
        log("terminating; shutting down any running server...")
        _cleanup_all()
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass  # not available in every environment

    nsys = find_nsys(args.nsys)
    if nsys is None:
        if not args.analyze_only:
            print("ERROR: nsys not found.", file=sys.stderr)
            print("  Install Nsight Systems, or point at it explicitly with --nsys.", file=sys.stderr)
            print("  Typical Windows path:", file=sys.stderr)
            print(r"    C:\Program Files\NVIDIA Corporation\Nsight Systems <version>"
                  r"\target-windows-x64\nsys.exe", file=sys.stderr)
            return 1
        log("nsys not found; using the .sqlite exports already in --out")

    if args.analyze_only:
        return run_analyze_only(args, nsys)

    args.out.mkdir(parents=True, exist_ok=True)

    log(f"nsys   : {nsys}")
    log(f"server : {args.server}")
    log(f"model  : {args.model}")
    log(f"K      : {args.k_values}")
    log(f"output : {args.out}")
    ports = [args.port + i for i in range(len(args.k_values))]
    log(f"ports  : {ports}")
    if any(prt < 1024 or prt > 65535 for prt in ports):
        raise SystemExit("ERROR: --port must be in [1024, 65535]")
    if args.port == 8399:
        log("        (default 8399; if a WSL server is already using it, pass --port 8500)")

    points = []
    port = args.port
    interrupted = False
    for k in args.k_values:
        try:
            points.append(run_point(args, k, nsys, port, args.out))
        except KeyboardInterrupt:
            log(f"stopped by you after k{k}.")
            interrupted = True
            break
        except Exception as error:  # noqa: BLE001 - keep going for the other K
            log(f"point k{k} failed: {error}")
        port += 1

    if interrupted:
        _cleanup_all()
        log("Stopped. Server processes have been terminated and GPU memory released.")
        log("Rerun with a different --port if a server is still stuck.")
        return 130

    print()
    print("=" * 110)
    print("  Per-point client throughput (not the profiler's numbers)")
    print("=" * 110)
    print(f"{'K':>4}{'tokens':>10}{'wall_s':>10}{'client tok/s':>16}")
    print("-" * 40)
    for point in points:
        print(f"{point['k']:>4}{point['completion_tokens']:>10}{point['wall_seconds']:>10.1f}"
              f"{point['client_tok_s'] or 0:>16.1f}")

    print_depth(points, args)
    print_comparison(args.out, args.k_values, nsys, args.cuda_graph)

    print()
    print("=" * 110)
    print("  Next steps")
    print("=" * 110)
    print()
    print("Start here: check the depth table above. If final_ctx is around 1K, this is a")
    print("   short-context run whatever --max-context said, and the split will NOT hold at")
    print("   150K. Re-run with --longctx-repeats before believing it at depth.")
    print()
    print("1. Look at the kernel table. The kernels at the top grew the most from the")
    print("   lower K to the higher K -- that is where the extra round time goes.")
    print()
    print("2. For detail on ONE kernel (occupancy, memory throughput, stall reasons),")
    print("   run Nsight Compute on it:")
    print(f"     {nsys.parent / 'ncu.exe'}")
    print("   or, if ncu is on PATH:")
    print("     ncu --kernel-name regex:<kernel_name> --launch-count 1 "
          "<your server command line>")
    print()
    print("3. To explore visually, open a report in the Nsight Systems GUI:")
    for k in args.k_values:
        report = find_report(args.out / f"k{k}")
        if report:
            log(f"     {report}")
    print()
    print("   In the GUI, zoom into one speculative round on the 'CUDA GPU Kernel' row")
    print("   to see the per-column pattern.")
    print()
    print("Copy the whole output directory back into the repo to continue the analysis.")
    return 0 if points else 1


if __name__ == "__main__":
    raise SystemExit(main())
