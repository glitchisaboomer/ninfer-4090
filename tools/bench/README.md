# tools/bench

Offline helper for the `ninfer_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v11. The flattened summary and schema-v3 matrix
manifest carry native names from the report: selected target, canonical `weights_id`, artifact,
load/read/upload/staging values, Engine memory arenas including request transient and CUDA Graph
allowance, per-test planned logical and allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

`run_serve_corpus.py` runs both registered targets and both published MTP0/MTP3 suites when both
artifacts are supplied. Pass one `--artifact` to select a single target and `--mode mtp0` or
`--mode mtp3` to run only that suite. The 35B-A3B-only `--mode dflash7` route runs the same
decode corpus with DFlash block=8 (`k=7`) and the optimized proposal head. Add
`--sampling greedy` to force exact argmax while retaining the same fixtures and repetition count.
Its schema-v5 result and flattened summaries retain the canonical `weights_id` received from the
schema-v8 serving startup record. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## MTP acceptance sweep

`run_gate_b_acceptance_sweep.py` answers one question that op-level benchmarks cannot: **how does
acceptance decay across proposal positions, and what does each extra draft token cost in round
time?** It starts one server per `(workload, K)` point — the memory plan and execution graphs are
startup-fixed, so `K` cannot change in-process — issues the same prompts at every K, and derives
per-position acceptance from the full-precision structured request log.

```bash
python tools/bench/run_gate_b_acceptance_sweep.py \
    --server build/apps/Release/ninfer-serve.exe \
    --model  E:/models/qwen3_8_27b.ninfer \
    --out    profiles/bench/gate-b \
    --workloads reasoning,freeform,structured --ks 1,2,3,5,7,10,15
```

Four built-in workloads span the acceptance range the decode campaign cares about: `reasoning`
(chain-of-thought, the campaign target), `freeform` (low-acceptance prose control), `structured`
(highly regular output), and `longctx` (~39K-token padded prompts, for the campaign's 40K working
context — issue #96 measured acceptance falling 0.724 to 0.603 going from 145 to 133K context, so
this arm is required before changing a production default). Everything except K is pinned:
`--max-concurrency 1`, `--kv-dtype rk4v4-e8`, `--lm-head-draft`, `temperature 0`, fixed seed,
identical prompts.

Note that raising `--max-context` alone does not exercise long context: the prompts themselves must
be long, or the model never sits at depth. The summary prints a **"Context depth actually
reached"** table from the server's own `computed_prefill_tokens`, so a short-prompt run cannot be
mistaken for a long-context one — check the `final_ctx` column. Pad `longctx` deeper with
`--longctx-repeats N` (roughly 140 tokens per repeat; treat that as an approximation and calibrate
against the reported `prefill_tok`).

Per-position acceptance is derived with exact denominators — `drafted_at_0 = rounds` and
`drafted_at_i = accepted_at_{i-1}` — rather than assuming a flat histogram. This matters because the
aggregate `accepted_tokens / drafted_tokens` is a different and much smaller quantity than
per-position q (at per-position q=0.50, K=3 the aggregate is 0.29), and conflating them inverts
conclusions about whether wider speculation can pay.

Outputs under `--out`: `gate_b_points.json` (full record), `gate_b_summary.csv`, the raw append-mode
`server.requests.jsonl`, and one server log per point. Rationale and interpretation live in
[`../docs/optimization-findings.md`](../docs/optimization-findings.md) under Gate B.

## Verify-path profiling

`profile_verify_path.py` answers "where does the extra round time at a wider draft width go?" by
running the server under Nsight Systems once per K and comparing GPU kernel time per kernel. Pure
Python, so it works on Windows with no shell.

```bat
python tools\bench\profile_verify_path.py ^
    --server build\apps\Release\ninfer-serve.exe ^
    --model  E:\models\qwen3_8_27b.ninfer
```

Start short: a short prompt keeps prefill instant and makes the per-column kernels stand out. It does
not, however, size the deployment regime. A short trace accounts for roughly half of the per-column
cost measured at 150K, and attention nearly doubles per call inside the real verify path even though
the isolated op benchmark reports it as flat in K. For a deep run, pad the prompt and raise the window:

```bat
python tools\bench\profile_verify_path.py ^
    --server ... --model ... --max-context 40000 --longctx-repeats 280
```

Every run prints the context depth each request actually reached and calibrates tokens-per-repeat
from it — `--max-context` alone moves the ceiling, not the prompts. Depth is `prompt_tokens`, the
context the decode ran against, and *not* `computed_prefill_tokens`: the warmup request primes the
same prompt, so the measured request reuses the prefix and prefills almost nothing while still
decoding at full depth. Both numbers are printed.

To re-read profiles copied in from another machine, profile nothing:

```bat
python tools\bench\profile_verify_path.py --analyze-only --out profiles\verify-profile
```

It starts no server, reports each point's trace and request-log timestamps so a stale trace cannot
pass for the current run, and never re-exports an empty `.nsys-rep` over a working `.sqlite`.

The script passes `--no-cuda-graph` because verify otherwise runs inside a
captured CUDA Graph that nsys reports as one opaque node; the K-vs-K delta is the meaningful quantity,
not the absolute times. Full beginner walkthrough, Windows install paths for nsys, and how to read the
output are in [`../docs/optimization-findings.md`](../docs/optimization-findings.md) under Gate B.

## MTP verify-attention Nsight Compute profile

`profile_attention_width_ncu.py` captures the first target-verify attention kernel matching the
width-4 specialization for K=3 or width-6 specialization for K=5. It starts one server per K with
CUDA Graphs disabled, sends a padded ~45K-context request, and saves an `.ncu-rep` plus server and
request logs. NCU's default `full` metric set is intentionally limited to one matching launch per K;
its replay timing is for attribution, not latency comparison.

```bat
python tools\bench\profile_attention_width_ncu.py ^
    --server build\apps\Release\ninfer-serve.exe ^
    --model E:\models\qwen3_8_27b.ninfer
```

Outputs default to `profiles/attention-width-ncu/k3/` and `k5/`, with reports named
`attention_k<K>_width<W>.ncu-rep` so an import cannot pick the wrong file. Verify the request log
reports the intended prompt depth, then compare `LaunchStats`, `Occupancy`, `SchedulerStats`,
`MemoryWorkloadAnalysis`, and `WarpStateStats` in Nsight Compute. If the padding lands below 40K,
increase `--longctx-repeats`; use `--ncu` if `ncu.exe` is not on PATH or in a standard install
location. `--launch-count 16` profiles all 16 target full-attention launches, but takes longer.

**Pass `--launch-skip` or you will profile the warmup.** The server warms up before it listens, and
that warmup (`GenerationService::warmup` sends `"hi"` with `max_tokens = 4`) launches the same
specialized kernel once per full-attention layer per decode round — roughly **64 matching launches**
for 16 layers, not 16. `--launch-count 1` with no skip, or with a skip smaller than that, captures a
`Grid (4,4,1)` = 16-block warmup and reports it as if it were the long-context request. A run that
does this still looks healthy: the request log shows the intended depth, because the request did
run; NCU simply profiled something else earlier.

The reliable check is **`Grid Size`**, which grows with the window: a 3-token warmup gives 16 blocks,
a 40K request gives `(4,64,1)` = 256. The script now computes the captured grid from the exported
CSV and warns when it is smaller than the requested depth implies, naming `--launch-skip` as the
fix. If it warns, raise the skip (128 is a safe overshoot past the warmup) and re-run; the request
itself always runs at full depth, so nothing else about the run needs to change.

Every point also writes `<report>.csv` beside the report, produced by `ncu --import --csv` and
trimmed to occupancy, register, stall, hit-rate, spill, instruction, cycle, block-size, and grid-size
rows — about 40% of the full export. That is the file to move off the profiling machine: the binary
report's bulk is raw counters, and nothing in this repository reads it. If `--import` fails the
error is reported and the report is kept.

**`--set full` is 45 replay passes and it does not always finish.** On the 27B model the resident
weights leave too little device memory for NCU's save-and-restore buffer, so the run prints
`Backing up device memory in system memory. Kernel replay might be slow` and may stop after a
fraction of the passes. The report is still written and looks like a success, but the metrics those
passes would have collected are simply absent — no stall reasons, no instruction or spill counters.
That reads exactly like the export dropping them, so compare the `N passes` line in the NCU log
between runs before trusting a missing metric.

Name the sections you need instead of asking for the whole set. `--sections` and `--metrics` skip
`--set` entirely and collect only what is asked for, in a handful of passes:

```bat
python tools\bench\profile_attention_width_ncu.py ^
    --server build\apps\Release\ninfer-serve.exe ^
    --model E:\models\qwen3_8_27b.ninfer ^
    --ks 5 --launch-skip 128 --sections SchedulerStats,WarpStateStats,InstructionStats
```

`SchedulerStats` gives warp cycles per issued instruction and eligible/issued ratios;
`WarpStateStats` gives the per-reason stall breakdown. Together they separate a kernel that is
waiting on memory from one that is simply doing more work.

Do not pass a comma-joined metric list through a shell line continuation. A `^` that fails to join
turns every metric after the first into its own argv token, and NCU reports the second one as
`'<metric>' does not exist or is not an executable` because it treats it as the application to
launch. `--metrics` is passed as one argument by the script, so the quoting is the script's problem,
not yours.

## Concurrent serving benchmark

`run_serve_concurrency.py` measures two separate concurrency properties through real loopback
Chat Completions requests:

- `decode-saturation` submits one long-decode wave and uses only complete one-second intervals in
  which every decode round has exactly the configured batch size. Ramp-up, prefill, and drain
  intervals are excluded.
- `corpus-makespan` shuffles the existing mode-specific corpus once with the fixed seed `20260811`,
  then runs that same order with exactly `N` persistent client workers. A worker submits the next
  request only after its current response completes, and makespan ends when the final response has
  been read. Request bodies are sent in shuffled-order sequence while response waits remain fully
  concurrent, removing client-thread arrival races without serializing inference.

Each concurrency point starts a fresh server because its execution graphs and memory plan are
startup-fixed. Prefix reuse is disabled, startup and warmup are outside both measurements, and the
runner writes per-point JSON, raw serving JSONL, and combined JSON/CSV/Markdown summaries.
The point report records the shuffle seed, dispatch method, shuffled position, and canonical corpus
position for every request.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.
