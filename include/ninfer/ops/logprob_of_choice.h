#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes the log-probability of an already-chosen token, per column:
 *
 *   out[t] = float(logits[chosen[t],t]) - logsumexp_{0 <= v < valid_rows} float(logits[v,t])
 *
 * `logits` is contiguous BF16 [physical_rows,T] with the vocabulary contiguous inside each column,
 * `chosen` is contiguous I32 [T], and `out` is contiguous FP32 [T]. 1 <= valid_rows <=
 * physical_rows; physical rows [valid_rows,physical_rows) are padding and do not participate.
 * `chosen[t]` must be in [0,valid_rows), which is what a sampler over the same domain guarantees.
 *
 * The reduction is a numerically stable log-sum-exp: the maximum is subtracted before exponentiating
 * and added back afterwards, so a vocabulary whose logits share a large offset -- which a trained
 * model's last layer routinely does -- does not overflow. The Op streams the column once, combining
 * each thread's partial state with the running-maximum rule rather than making two passes.
 *
 * This exists because Suffix Cache Reuse has to be *measured*, not merely observed to produce
 * plausible text: Phase 5.5 of docs/SCR_ROADMAP.md scores the divergence of the two reuse paths in
 * nats, and the argmax of a column says nothing about how far the runner-up was.
 *
 * `out` must not overlap `logits` or `chosen`. The Op has no workspace and changes no state other
 * than writing all of `out`.
 */
void logprob_of_choice(const Tensor& logits, const Tensor& chosen, std::int32_t valid_rows,
                       Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops