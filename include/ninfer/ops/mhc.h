#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one SenseNova mHC residual router.
 *
 * This contract is fixed to two residual streams and eight router logits. The
 * residual tensor has shape [D,2,T] in ninfer order: D is the contiguous
 * feature extent, stream stride nb[1] may include padding, and token stride
 * nb[2] addresses two stream rows. nb[0] must be sizeof(BF16). weight is FP32
 * [2*D,8] with contiguous rows of length 2*D, bias is FP32 [8]. branch is BF16
 * [D,T], post is FP32 [2,T], and mix is FP32 [2,2,T] with destination stream as
 * the fastest axis and source stream as the second axis.
 *
 * For token t, the oracle forms the stream-major flat vector x[s*D+d] from the
 * represented BF16 residual. It computes, in FP64 from the represented inputs:
 *
 *   inv  = 1 / sqrt(sum_s,d x[s,d]^2 / (2*D) + 1e-6)
 *   logit[k] = sum_i weight[k,i] * x_flat[i] * inv + bias[k]
 *   pre[s] = sigmoid(logit[s])
 *   post[s] = 2 * sigmoid(logit[2+s])
 *   mix = exp(sinkhorn_20(logit[4:8] as [2,2]))
 *
 * Sinkhorn alternates row and column logsumexp subtraction for `iterations`
 * passes. pre is rounded to BF16 before forming `branch[d]`, matching the
 * Hugging Face boundary where pre, mix, and post are cast to the residual dtype
 * before application. post and mix are returned as FP32; merge applies their
 * BF16-rounded values.
 *
 * `pre_scale` must be positive and finite. It scales the branch before BF16
 * storage. iterations must be positive. Inputs and outputs must not overlap.
 */
void mhc_route(const Tensor& residual, const Tensor& weight, const Tensor& bias,
               int streams, int iterations, float pre_scale, Tensor& branch,
               Tensor& post, Tensor& mix, cudaStream_t stream);

/**
 * Applies one SenseNova mHC residual merge.
 *
 * residual is BF16 [D,2,T] with unit feature stride and may be padded between
 * streams. branch is BF16 [D,T]. post is FP32 [2,T]. mix is FP32 [2,2,T] with
 * destination stream fastest and source stream second. out is contiguous BF16
 * [D,2,T]. Inputs and out must not overlap.
 *
 * The oracle rounds post and mix to BF16, then computes for destination s:
 *
 *   mixed[s,d] = sum_u mix[s,u,t] * residual[u,d,t]
 *   update[s,d] = post[s,t] * branch[d,t]
 *   out[s,d,t] = BF16(mixed[s,d] + update[s,d])
 */
void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post,
               const Tensor& mix, int streams, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
