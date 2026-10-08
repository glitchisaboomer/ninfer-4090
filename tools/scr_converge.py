#!/usr/bin/env python3
"""
Check whether a --no-scr request's generated token_id sequence converges
across repeated runs, so we can tell whether k2's reasoning trace is stable
on the control before we attribute any SCR-side divergence to anything.

The harness logs each request as one `ninfer_serve_request_log` (schema v9)
JSONL record. Multiple event kinds live in the same file (server_start,
request_start, throughput, request_done, ...). This script picks out the
request_done records and compares their generated token_id sequences.

request_done records carry:
  result.generated.token_ids  (the decoded token ids, in order)
  result.generated.logprobs   (logprob of each generated token; same length)
  result.generated.margins     (decision margin at each generated position)
  result.prompt_tokens        (prefill token count)
  result.completion_tokens
  result.computed_prefill_tokens
  result.prefix_relocated_tokens / prefix_reuse_path / prefix_cache_hit_tokens
  result.finish_reason
  request.request_id
  request.sampling.seed

Usage:
  python3 tools/scr_converge.py run1.jsonl [run2.jsonl ...]

Each file may contain multiple request_done records (one per request the
server handled). Pass one file per repeated run. The script reports:
  - a per-record summary (prompt/completion/relocated/reuse/seed/length)
  - per (file, request_id) convergence
  - cross-file convergence for the same request_id/prompt_tokens appearing
    in more than one file

To isolate k2 specifically, pass JSONL files that each contain only the k2
request (one request_done record per file), or filter after the fact by
prompt_tokens/completion_tokens/request_id.

To run on the target machine for k2 specifically:
  - Start one ninfer-serve with --no-scr, same model/weights/max-tokens as
    the suite.
  - Send ONLY the k2 prompt (the qwen3.8-27b chat prompt whose run3
    produced chat-k2-p0 / prompt_tokens=661).
  - Save that server's request_log_jsonl (one request_done record).
  - Repeat 4-8 times (fresh server per run; if you want seed control,
    use the same seed across runs instead of the suite's per-request seeds).
  - Pass each run's JSONL file to this script.

If you only care about token_id convergence (what the harness actually
captures and what decoded text is deterministic in), this script is enough.
If you also want decoded UTF-8 text, you need the qwen3_6 tokenizer assets
(tokenizer.json etc.) which are not present as loadable Python here; the C++
Tokenizer can decode, but is not wired into this script.
"""

import json
import sys
from pathlib import Path

FIELD_GENERATED_TOKEN_IDS = "result.generated.token_ids"
FIELD_GENERATED_LOGPROBS  = "result.generated.logprobs"
FIELD_GENERATED_MARGINS   = "result.generated.margins"
FIELD_PROMPT_TOKENS       = "result.prompt_tokens"
FIELD_COMPLETION_TOKENS   = "result.completion_tokens"
FIELD_PREFILL_TOKENS      = "result.computed_prefill_tokens"
FIELD_RELOCATED           = "result.prefix_relocated_tokens"
FIELD_REUSE_PATH          = "result.prefix_reuse_path"
FIELD_CACHE_HIT           = "result.prefix_cache_hit_tokens"
FIELD_FINISH              = "result.finish_reason"
FIELD_REQUEST_ID          = "request.request_id"
FIELD_SEED                = "request.sampling.seed"


def load_request_done_records(paths):
    """Return list of (path, record) for every request_done line across paths."""
    records = []
    for path in paths:
        p = Path(path)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (obj.get("artifact_type") == "ninfer_serve_request_log"
                    and obj.get("event") == "request_done"):
                records.append((p.name, obj))
    return records


def pick(obj, keys, default=None):
    """Walk nested dotted key into obj, returning default on any miss."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _show_divergence(grp, label):
    """Print first divergent pair within `grp` (list of (path,tids,lp,mg))."""
    tids = [g[1] for g in grp]
    best_prefix = 0
    best_pair = None
    for i in range(len(grp)):
        for j in range(i + 1, len(grp)):
            a, b = grp[i][1], grp[j][1]
            k = 0
            limit = min(len(a), len(b))
            while k < limit and a[k] == b[k]:
                k += 1
            if k > best_prefix:
                best_prefix = k
                best_pair = (i, j)
    if not tids:
        return
    min_len = min(len(t) for t in tids)
    if min_len:
        print("    shared prefix: %d of %d token_ids (%.1f%%)"
              % (best_prefix, min_len, 100.0 * best_prefix / min_len), flush=True)
    if best_pair:
        i, j = best_pair
        a, b = grp[i][1], grp[j][1]
        k = 0
        limit = min(len(a), len(b))
        while k < limit and a[k] == b[k]:
            k += 1
        print("    first divergence between %s and %s at token position %d"
              % (grp[i][0], grp[j][0], k), flush=True)
        lo = max(0, k - 6); hi = min(min(len(a), len(b)), k + 6)
        print("      %-16s tokens[%d:%d] = %s" % (grp[i][0], lo, hi, a[lo:hi]), flush=True)
        print("      %-16s tokens[%d:%d] = %s" % (grp[j][0], lo, hi, b[lo:hi]), flush=True)
        lpi, mgi = grp[i][2], grp[i][3]
        lpj, mgj = grp[j][2], grp[j][3]
        if lpi and lpj and k < len(lpi) and k < len(lpj):
            print("      logprob %s=%.6g  %s=%.6g"
                  % (grp[i][0], lpi[k], grp[j][0], lpj[k]), flush=True)
        if mgi and mgj and k < len(mgi) and k < len(mgj):
            print("      margin %s=%.4f  %s=%.4f"
                  % (grp[i][0], mgi[k], grp[j][0], mgj[k]), flush=True)


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python3 tools/scr_converge.py run1.jsonl [run2.jsonl ...]")
        sys.exit(2)

    records = load_request_done_records(paths)
    if not records:
        print("no request_done records found in:", ", ".join(paths))
        print("  (expected each file to contain ninfer_serve_request_log event=request_done lines)")
        sys.exit(2)

    print("loaded %d request_done record(s) from %d file(s)" % (
        len(records), len({p for p, _ in records})), flush=True)

    rows = []
    for path, r in records:
        token_ids = pick(r, FIELD_GENERATED_TOKEN_IDS)
        if token_ids is None:
            continue
        rows.append((path, token_ids,
                     pick(r, FIELD_GENERATED_LOGPROBS),
                     pick(r, FIELD_GENERATED_MARGINS),
                     pick(r, FIELD_PROMPT_TOKENS),
                     pick(r, FIELD_COMPLETION_TOKENS),
                     pick(r, FIELD_PREFILL_TOKENS),
                     pick(r, FIELD_RELOCATED),
                     pick(r, FIELD_REUSE_PATH),
                     pick(r, FIELD_CACHE_HIT),
                     pick(r, FIELD_FINISH),
                     pick(r, FIELD_SEED),
                     pick(r, FIELD_REQUEST_ID)))

    if not rows:
        print("  (no request_done records with result.generated.token_ids; nothing to converge-test.)")
        print("\nnote: the harness request_done record carries result.generated.token_ids "
              "(schema v9). If you expected token_ids but none were found, check that each "
              "JSONL request_done line has a result.generated.token_ids array.")
        sys.exit(0)

    def s(x):
        return x if x is not None else "?"

    print("\n=== per-record summary ===")
    for path, tids, lp, mg, pt, ct, pf, rel, rp, ch, fr, seed, rid in rows:
        print("  %-16s req=%s  prompt=%s  prefill=%s  completion=%s  "
              "relocated=%s  reuse=%s  cache_hit=%s  finish=%s  seed=%s  "
              "len(token_ids)=%d"
              % (path, s(rid), s(pt), s(pf), s(ct), s(rel),
                 s(rp), s(ch), s(fr), s(seed), len(tids)), flush=True)

    # Convergence per (file, request_id).
    keyrows = {}
    for path, tids, lp, mg, pt, ct, pf, rel, rp, ch, fr, seed, rid in rows:
        k = (path, rid)
        keyrows.setdefault(k, []).append((path, tids, lp, mg))

    print("\n=== convergence per (file, request_id) ===")
    for (path, rid), grp in keyrows.items():
        tids = [g[1] for g in grp]
        ident = len({tuple(t) for t in tids}) == 1
        print("  %-16s req=%s  records=%d  identical_within=%s  len=%d"
              % (path, s(rid), len(grp), ident, len(tids[0])), flush=True)
        if not ident:
            _show_divergence(grp, "%s/%s" % (path, rid))

    # Cross-file convergence for the same request_id.
    by_rid = {}
    for path, tids, lp, mg, pt, ct, pf, rel, rp, ch, fr, seed, rid in rows:
        by_rid.setdefault(rid, []).append((path, tids, lp, mg))

    cross_file = {rid: grp for rid, grp in by_rid.items()
                  if len({p for p, *_ in grp}) > 1}
    print("\n=== cross-file convergence (same request_id across files) ===")
    if not cross_file:
        print("  no request_id appeared in more than one input file; nothing cross-file to compare.")
    else:
        any_identical = True
        for rid, grp in sorted(cross_file.items()):
            tids = [g[1] for g in grp]
            ident = len({tuple(t) for t in tids}) == 1
            print("  req=%s  files=%d  identical=%s  lens=%s"
                  % (s(rid), len(grp), ident, [len(t) for t in tids]), flush=True)
            if not ident:
                any_identical = False
                _show_divergence(grp, "req %s" % s(rid))
        print("\n  overall cross-file identical = %s" % any_identical, flush=True)

    # Cross-file convergence for the same prompt_tokens (useful when request_id
    # differs between runs but the prompt is the same).
    by_pt = {}
    for path, tids, lp, mg, pt, ct, pf, rel, rp, ch, fr, seed, rid in rows:
        by_pt.setdefault(pt, []).append((path, tids, lp, mg))

    cross_pt = {pt: grp for pt, grp in by_pt.items()
                if len({p for p, *_ in grp}) > 1}
    if cross_pt:
        print("\n=== cross-file convergence (same prompt_tokens across files) ===")
        any_identical = True
        for pt, grp in sorted(cross_pt.items()):
            tids = [g[1] for g in grp]
            ident = len({tuple(t) for t in tids}) == 1
            print("  prompt_tokens=%s  files=%d  identical=%s  lens=%s"
                  % (s(pt), len(grp), ident, [len(t) for t in tids]), flush=True)
            if not ident:
                any_identical = False
                _show_divergence(grp, "prompt_tokens %s" % s(pt))
        print("\n  overall cross-file identical (by prompt_tokens) = %s"
              % any_identical, flush=True)

    # Token-count convergence (cheap signal that the runs even produced the
    # same number of tokens).
    counts = [len(t) for _, t, *_ in rows]
    if len({counts}) == 1:
        print("\n[converged] all runs produced the same number of tokens (%d)" % counts[0])
    else:
        print("\n[diverged] token counts differ: %s" % counts)

    print("\nnote: if you want decoded UTF-8 text (not just token ids), the project has a "
          "C++ qwen3_6 tokenizer (src/targets/qwen3_6/impl/frontend/tokenizer.{h,cpp}) but no "
          "loadable Python tokenizer assets here.  The harness captures token_ids, which is the "
          "object decoded text is deterministic in, so token_id convergence is the right first check.")


if __name__ == "__main__":
    main()
