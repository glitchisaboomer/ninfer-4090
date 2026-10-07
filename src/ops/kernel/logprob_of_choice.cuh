#pragma once

// Implements: include/ninfer/ops/logprob_of_choice.h
// Match: contiguous BF16 [physical_rows,T] logits with the vocabulary contiguous inside a column,
// I32 [T] chosen ids, FP32 [T] out, 1 <= valid_rows <= physical_rows.
// Algorithm assumptions: one CTA owns one column and streams it once, carrying a running
// (max, sum-of-exp) pair per thread and combining pairs with the associative form of log-sum-exp, so
// the vocabulary is read exactly once and never exponentiated at an unshifted logit.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// One column per CTA. The vocabulary is the reduction axis, so the block width is the only lever on
// parallelism; a decode step's weights dominate this kernel by orders of magnitude, so trading CTAs
// for simplicity here is not measurable next to a 64-layer forward pass.
inline constexpr int kLogprobBlock = 512;

// 16-byte transactions, the house rule for reading a decoded plane. Eight bf16 per load.
inline constexpr int kLogprobVecElems = 8;

/**
 * A partial log-sum-exp, held as the running maximum and the sum of exp(logit - max) that goes with
 * it. Accumulating in this form rather than as a plain sum is what keeps the result finite when the
 * logits carry a large shared offset, and it composes associatively across threads.
 */
struct LogsumexpState {
    float max;
    float sum;
};

__device__ __forceinline__ void logsumexp_reset(LogsumexpState& state) {
    state.max = -CUDART_INF_F;
    state.sum = 0.0F;
}

__device__ __forceinline__ void logsumexp_push(LogsumexpState& state, float value) {
    if (value > state.max) {
        // The new maximum retires the old scale, so rescale before adding.
        state.sum = state.sum * __expf(state.max - value) + 1.0F;
        state.max = value;
    } else {
        state.sum += __expf(value - state.max);
    }
}

// Associative combine: adopt the larger maximum and rebase the other partial onto it.
__device__ __forceinline__ LogsumexpState logsumexp_merge(LogsumexpState a, LogsumexpState b) {
    const float max = a.max > b.max ? a.max : b.max;
    // An empty partial is -inf with a zero sum; exp(-inf - max) is 0, so it contributes nothing and
    // needs no special case beyond not producing NaN when both are empty.
    const float a_scaled = (a.max == -CUDART_INF_F) ? 0.0F : a.sum * __expf(a.max - max);
    const float b_scaled = (b.max == -CUDART_INF_F) ? 0.0F : b.sum * __expf(b.max - max);
    return LogsumexpState{max, a_scaled + b_scaled};
}

__device__ __forceinline__ float logsumexp_value(LogsumexpState state) {
    return state.max + __logf(state.sum);
}

// Combines one shuffle step, guarding only the *use* of the partner's value.
//
// The shuffles must be reached by every lane. Their mask names all 32, and a lane that has already
// left the warp makes the sync undefined rather than merely redundant -- on Volta+ independent thread
// scheduling the barrier waits on threads that exited, so an early return here deadlocks the kernel.
// Argmax gets this right for the same reason, though it can afford to leave the use unguarded because
// a maximum is idempotent; a log-sum-exp merge is not, so the use is guarded too.
__device__ __forceinline__ void logsumexp_shfl_step(LogsumexpState& state, int offset) {
    const int lane      = threadIdx.x & 31;
    const float partner_max = __shfl_down_sync(0xffffffffu, state.max, offset);
    const float partner_sum = __shfl_down_sync(0xffffffffu, state.sum, offset);
    if (lane >= offset) { return; }
    state = logsumexp_merge(state, LogsumexpState{partner_max, partner_sum});
}

__device__ __forceinline__ LogsumexpState logsumexp_warp_reduce(LogsumexpState state) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { logsumexp_shfl_step(state, offset); }
    return state;
}

__device__ __forceinline__ LogsumexpState logsumexp_block_reduce(LogsumexpState state) {
    constexpr int kWarps = (kLogprobBlock + 31) >> 5;
    __shared__ float maxima[kWarps];
    __shared__ float sums[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    state = logsumexp_warp_reduce(state);
    if (lane == 0) {
        maxima[warp] = state.max;
        sums[warp]    = state.sum;
    }
    __syncthreads();

    // Only warp 0 reads the per-warp totals back, and every thread in it must reach the load, so the
    // early return below is deliberately after the barrier rather than before it.
    if (warp != 0) { return state; }
    state = logsumexp_warp_reduce(LogsumexpState{lane < kWarps ? maxima[lane] : -CUDART_INF_F,
                                                  lane < kWarps ? sums[lane] : 0.0F});
    return state;
}

__launch_bounds__(kLogprobBlock) __global__
    void logprob_of_choice_kernel(const __nv_bfloat16* __restrict__ logits,
                                  const std::int32_t* __restrict__ chosen,
                                  std::int32_t valid_rows, std::int32_t physical_rows,
                                  float* __restrict__ out) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    LogsumexpState state;
    logsumexp_reset(state);

    // The vectorised body needs both a 16-byte-aligned column start and an even tail inside the
    // valid domain. A vocabulary whose row count is not a multiple of eight takes the scalar path
    // rather than reading a whole vector past valid_rows.
    const bool vectorised =
        ((physical_rows % kLogprobVecElems) == 0) && ((valid_rows % kLogprobVecElems) == 0);

    if (vectorised) {
        const int vectors     = valid_rows / kLogprobVecElems;
        const int4* typed     = reinterpret_cast<const int4*>(logits + base);
        for (int i = static_cast<int>(threadIdx.x); i < vectors; i += kLogprobBlock) {
            const int4 packed = typed[i];
            const __nv_bfloat16* lanes =
                reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
            for (int e = 0; e < kLogprobVecElems; ++e) {
                logsumexp_push(state, __bfloat162float(lanes[e]));
            }
        }
    } else {
        for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows;
             v += kLogprobBlock) {
            logsumexp_push(state, __bfloat162float(logits[base + v]));
        }
    }

    state = logsumexp_block_reduce(state);
    if (threadIdx.x != 0) { return; }

    // The chosen id indexes the same valid domain the reduction covered, so the logit it reads is one
    // this column already accounted for.
    const std::int32_t pick = chosen[t];
    out[t] = __bfloat162float(logits[base + pick]) - logsumexp_value(state);
}

} // namespace ninfer::ops