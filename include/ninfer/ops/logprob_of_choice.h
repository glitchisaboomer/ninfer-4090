#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes the log-probability of an already-chosen token, per column, and the column's decision
 * margin:
 *
 *   out[t]    = float(logits[chosen[t],t]) - logsumexp_{0 <= v < valid_rows} float(logits[v,t])
 *   margin[t] = max_v float(logits[v,t]) - max_{u != argmax_v} float(logits[u,t])
 *
 * `margin[t]` is the gap in nats between the best and second-best logit, which is also the gap between
 * their log-probabilities because the log-sum-exp subtracts out of both. It is what makes a measured
 * logprob difference interpretable: a difference of 0.005 nats means nothing when the two leading
 * candidates were 0.002 nats apart, and means a great deal when they were 4 nats apart. `chosen[t]` is
 * not consulted for it -- the margin is a property of the column, not of the pick.
 *
 * `logits` is contiguous BF16 [physical_rows,T] with the vocabulary contiguous inside each column,
 * `chosen` is contiguous I32 [T], and `out` and `margin` are contiguous FP32 [T]. 1 <= valid_rows <=
 * physical_rows; physical rows [valid_rows,physical_rows) are padding and do not participate.
 * `chosen[t]` must be in [0,valid_rows), which is what a sampler over the same domain guarantees.
 *
 * A column with a single valid row has no runner-up, so `margin[t]` is +inf: there was no second
 * candidate, and that is different from a column that was merely confident.
 *
 * The reduction is a numerically stable log-sum-exp: the maximum is subtracted before exponentiating
 * and added back afterwards, so a vocabulary whose logits share a large offset -- which a trained
 * model's last layer routinely does -- does not overflow. The Op streams the column once, combining
 * each thread's partial state with the running-maximum rule rather than making two passes; the
 * runner-up rides along in that same state rather than costing a second pass.
 *
 * This exists because Suffix Cache Reuse has to be *measured*, not merely observed to produce
 * plausible text: Phase 5.5 of docs/SCR.md scores the divergence of the two reuse paths in
 * nats, and the argmax of a column says nothing about how far the runner-up was.
 *
 * `out` and `margin` must not overlap `logits`, `chosen`, or each other. The Op has no workspace and
 * changes no state other than writing all of `out` and `margin`.
 */
void logprob_of_choice(const Tensor& logits, const Tensor& chosen, std::int32_t valid_rows,
                       Tensor& out, Tensor& margin, cudaStream_t stream);

} // namespace ninfer::ops