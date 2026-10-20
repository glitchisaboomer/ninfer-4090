# Decode Optimization Findings

## Workload and baseline

| Item | Reported setting/result |
|---|---|
| GPU / model | RTX 4090; Qwen3.8-27B through the `qwen3_6_27b` path |
| Weights / context | Groupwise-int artifact; approximately 44K context |
| KV cache | `rk4v4-e8` |
| MTP | K=3; optimized proposal head (`--lm-head-draft`) enabled |
| Throughput | About 95 tok/s with MTP, versus 42 tok/s without MTP |
| Acceptance | About 50% |

The MTP result is about 2.26× the MTP-disabled throughput. Acceptance rate alone does not show whether K=3 is optimal. This is a source review and hypothesis list, not a live profile; the published [`performance.md`](performance.md) results are primarily from an RTX 5090 and do not establish the best path for this workload.

## Possible decode bottlenecks

Each table contains one possible cause, an explanation, and a link to a possible follow-up session. None is confirmed without profiling.

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **Quantized weight reads and small-T linear kernels** | Each target-model decode/verification pass executes projections and MLPs across 64 layers. At small token widths, weight traffic, weight decoding, and the selected Q4/Q5 routes may limit throughput. | [Profile and tune the small-T linear path](#follow-up-1-profile-small-t-linear-and-weight-cost) |

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **Long-context GQA/KV reads** | The 16 full-attention layers read historical K/V, compute attention scores and softmax, then reduce values. At 44K, cache traffic and attention work may be substantial; the split policy can also affect utilization. | [Measure GQA cost across context lengths](#follow-up-2-measure-gqa-and-kv-cost) |

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **GDN recurrent verification and state replay/fold** | The 48 GDN layers process the verified token positions and preserve speculative state for the accepted prefix. Recurrent transitions across positions are sequential; record and fold add work beyond ordinary width-one decode. | [Attribute GDN record, recurrence, and fold cost](#follow-up-3-measure-gdn-verification-and-fold) |

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **Serial MTP draft generation or a mismatched draft width** | Each draft depends on the preceding draft's hidden state, so proposal steps are sequential. A larger K costs more drafting and verification; it only helps if the extra committed tokens offset that work. | [Sweep K and assess adaptive width](#follow-up-4-tune-mtp-draft-width) |

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **Target output head and token-selection work** | Verification needs target logits, while the optimized draft head only reduces proposal-head work. Output-head projections and sampling may still contribute, particularly when several positions are verified. | [Measure target logits and proposal-head cost](#follow-up-5-measure-output-heads-and-sampling) |

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **V unpack/dequantization and BF16 P·V staging** | The current kernel expands each packed V tile to BF16 for P·V. This work grows with context and could matter, but global KV reads or other attention work may dominate instead. | [Profile P·V staging before prototyping FP8](#follow-up-6-profile-pv-staging) |

| Possible cause | Explanation | Possible solution in a later session |
|---|---|---|
| **Round orchestration, synchronization, or WDDM residency** | Graph replay reduces launch overhead but does not remove copies, bookkeeping, or all synchronization. Memory pressure under Windows/WDDM can also cause stalls or variability. | [Check GPU gaps, round overhead, and memory headroom](#follow-up-7-check-orchestration-and-residency) |

## Follow-up options for later sessions

These are investigation plans, not claims that a change will be faster. Preserve the same prompt, sampling, artifact, context, and output conditions when comparing end-to-end committed tokens/sec.

### Follow-up 1: Profile small-T linear and weight cost

Profile the target verification and ordinary decode shapes. Compare operator timings with the dispatches selected in [`q4_dispatch.cpp`](../src/ops/linear/q4/q4_dispatch.cpp) and [`q5_dispatch.cpp`](../src/ops/linear/q5/q5_dispatch.cpp). Use the public linear benchmark for exact shapes only as a diagnostic; accept a dispatch or kernel change only if end-to-end `tg` improves.

### Follow-up 2: Measure GQA and KV cost

Compare fixed context lengths such as 8K, 16K, and 44K, with MTP on and off. Attribute time to the GQA kernels and inspect the split policy in [`gqa_attention_decode.cu`](../src/ops/launcher/gqa_attention_decode.cu). If profiling supports a kernel change, validate it against the independent attention oracle and the same generation workload.

### Follow-up 3: Measure GDN verification and fold

Separate projection/record, recurrent verification, and accepted-prefix fold costs. The GDN snapshot benchmark is not an exact proxy for the replay-record path used by speculative verification. A scan-based verifier is a larger follow-up: the existing WY path operates on complete 64-token chunks, whereas MTP widths are shorter and use the recurrent tail. A new short-block scan would need per-token outputs, replay/state semantics, and numerical-oracle testing.

### Follow-up 4: Tune MTP draft width, then test adaptive width

Use K=1, 3, and 5 first; compare committed tok/s, round time, and accepted drafts by position. The public runtime supports up to K=15, but the retained Qwen3.6-27B target round benchmark currently covers only K=1..5. Its GDN snapshot sweep is not an end-to-end K sweep or an exact replay-record timing.

A Crit-style adaptive policy is a strong candidate for this serial-draft cost. **Crit** is the reference inference project ([inference index](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/inference/README.md)); the relevant engine example is specifically its Qwen3.8 implementation in [crit-hub](https://github.com/crit-labs/crit-hub), not the generic [`inference/mtp-decode` toolkit](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/inference/mtp-decode/README.md). In the [Qwen3.8 NVFP4 engine implementation](https://github.com/crit-labs/crit-hub/blob/0b72fd0ee3fc03ca1c0fb853d0dd1021c1617117/engines/rtx5090/qwen3.8-27b%40NVFP4/mtp7-fp8attn/engine/src/engine.cpp), Crit-hub uses recent acceptance and estimated round cost to choose draft width rather than always doing a fixed number of draft steps. The generic Crit MTP entry also includes an `AdaptiveDraftBudget` that explores candidate widths and minimizes an EWMA of round time per published token; this is a useful policy pattern, but its proposer is token-history copying, not a learned MTP head. The user's aggregate ~50% acceptance at K=3 is not enough to select a policy; first measure acceptance by position and timings for each fixed K. Crit-hub's RTX 5090/NVFP4 results are policy inspiration, not evidence of a gain on this RTX 4090/groupwise-int workload.

Implementation needs to account for graph shapes: the current decode graph and proposal loop are organized around configured K. A practical first design could select among a small set of captured K-specific graph profiles; for concurrent requests, per-batch width choices or bucketing also need consideration. Adaptive width is worth implementing only if fixed-K results and an online policy simulation predict a measurable increase in committed tok/s after its selection/graph overhead.

### Follow-up 5: Measure output heads and sampling

Measure target-logit production separately from optimized draft proposal. The user already has `--lm-head-draft` enabled, so this is not an untested optimization in the reported baseline. The canonical Qwen3.8 converter inventory contains `text/draft_head` and `text/draft_head_token_ids`; the loader requires these artifact objects when the optimized head is used. A full-versus-optimized comparison is useful only if quantifying its contribution is desired; do not assume a percentage gain.

### Follow-up 6: Profile P·V staging

The current `rk4v4-e8` decode kernel does **not** unpack the whole cache to FP16/BF16. Packed 4-bit codes remain in global memory. The kernel unpacks K/V tiles into INT8 staging; QK uses INT8 Tensor Core MMA with group scales, while V is dequantized into BF16 staging for the existing BF16 P·V MMA. See [`gqa_attention_decode_i8.cuh`](../src/ops/kernel/gqa_attention_decode_i8.cuh).

The BF16 P·V staging is an unverified bottleneck hypothesis. Crit's [FP8 KV attention entry](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/inference/fp8-kv-attention/README.md) is a concrete reference: its **decode** kernel reads an E4M3 KV cache, dequantizes tiles, and reuses the BF16 Tensor Core attention path. The entry's optional direct FP8 Tensor Core attention path is specifically for **prefill**, not decode. It demonstrates a scale-aware P·V method (rescaling probabilities and the accumulator across tiles), but not a ready-made FP8 Tensor Core decode win. It is not a drop-in for NInfer: the entry notes that paged/shared caches need adapted loads, while NInfer uses paged KV and rotated/group-quantized codes. Its E4M3 cache, per-head-vector scales, and precision are also different from NInfer's `rk4v4-e8` representation. An on-chip FP8 tile path might reduce conversion or staging cost, but would not reduce global cache reads while the stored cache remains 4-bit. Profile first; if P·V is material, adapt a narrow kernel and test numerical fidelity, greedy exactness, and generated-token quality.

### Follow-up 7: Check orchestration and residency

Use a system timeline to look for GPU idle gaps, synchronization, or excessive per-round work; check available VRAM after model load and during long-context decode. CUDA Graphs are already supported, so graph support itself is not a missing feature. Investigate graph replay, round orchestration, or WDDM budget behavior only if timings or memory telemetry point there.

## FP8 KV and mixed-precision context

FP8 could change attention accuracy and MTP acceptance, but higher acceptance is not guaranteed. NInfer currently exposes BF16, INT8-G64, and packed rotated integer/E8 cache formats; there is no FP8 KV dtype or FP8 append/prefill/decode path. Implementing FP8 KV would require cache layout/scales, quantization, attention kernels, dispatch/workspace changes, MTP support, and numerical-oracle tests.

FP8 also does not reduce KV storage versus INT8: both use one byte per element before scales. At 262,144 tokens, the main cache for 16 full-attention layers, 4 KV heads, head dimension 256, and both K and V is **8 GiB raw** at one byte per element:

```text
2 × 16 × 4 × 256 × 262,144 bytes = 8 GiB
```

That excludes scales, page metadata, the separate MTP cache, weights, recurrent state, and workspaces. The artifact is about 18.21 GB on disk (see its [manifest](../model-cards/Qwen3.8-27B-NInfer/artifact-manifest.json)); actual GPU-resident model bytes depend on load options. If INT8 does not fit at full context alongside runtime allocations, ordinary full K+V FP8 is unlikely to fit either. K8V4 would use about 6 GiB raw for the main cache at 262K, versus about 4 GiB for K4V4 and 8 GiB for K8V8, before overhead.

Ada Tensor Cores can accelerate suitable FP8 matrix operations, but FP8 storage does not automatically use them. An FP8 attention path still needs operand scaling/layout, softmax, and masks; decode has few query rows and may be limited by reading the long KV history rather than matrix-multiply throughput. A mixed K8V4 path is a kernel-and-quality experiment, not an assumed speedup or a known full-context-safe option.

## Using the Crit kernel guidance

Crit's [Kernels README](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/kernels/README.md) recommends profiling the actual engine before writing or importing a kernel, checking formats/shapes/ABI, and measuring decode and speculative verification separately. It also stresses column-count exactness: a speculative verify kernel must preserve the one-token greedy result for each column, not merely match an aggregate tolerance. The [Crit Eval kernel profile](https://github.com/crit-labs/crit/blob/1c9e0b265da8a0cf28b07a444516853091b896f8/docs/eval/README.md#kernel-profile) uses CUPTI to report phase time, GPU idle time, kernel calls and widths; its documented runner requires Linux/WSL and CUDA CUPTI, so availability in this Windows-target workflow must be checked before relying on it. Crit's upstream kernels and results are investigation leads, not evidence of speed or compatibility on NInfer's RTX 4090 path.

## Caveats

- The reported 95/42 tok/s and acceptance are user-provided, not independently verified here.
- Findings are source-based hypotheses; no live profile or new benchmark was run for this document.
- RTX 5090 performance data is context, not a substitute for the same workload on the RTX 4090.
- Prefill optimization is intentionally out of scope.
