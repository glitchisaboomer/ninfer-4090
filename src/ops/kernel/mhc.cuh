#pragma once

// ninfer::ops - SenseNova mHC residual router kernels.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {
namespace mhc_detail {

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float logsumexp2(float a, float b) {
    const float maximum = fmaxf(a, b);
    return maximum + logf(expf(a - maximum) + expf(b - maximum));
}

__device__ __forceinline__ float block_sum(float value, float* reductions) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) { reductions[warp] = value; }
    __syncthreads();
    const int warps = static_cast<int>(blockDim.x) / kWarpSize;
    float total = lane < warps ? reductions[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) { reductions[0] = total; }
    }
    __syncthreads();
    return reductions[0];
}

template <int Block>
__launch_bounds__(Block) __global__
    void mhc_route_kernel(const char* residual, std::int64_t residual_stream_stride,
                          std::int64_t residual_token_stride, const float* weight,
                          const float* bias, int d, int iterations, float pre_scale,
                          __nv_bfloat16* branch, float* post, float* mix, int tokens) {
    static_assert(Block % kWarpSize == 0);
    __shared__ float reductions[Block / kWarpSize];

    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int width = 2 * d;

    float sum_squares = 0.0f;
    float dots[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { dots[k] = 0.0f; }

    for (int index = tid; index < width; index += Block) {
        const int stream = index >= d ? 1 : 0;
        const int column = index - stream * d;
        const auto* value_ptr = reinterpret_cast<const __nv_bfloat16*>(
            residual + static_cast<std::int64_t>(token) * residual_token_stride +
            static_cast<std::int64_t>(stream) * residual_stream_stride +
            static_cast<std::int64_t>(column) * sizeof(__nv_bfloat16));
        const float value = __bfloat162float(*value_ptr);
        sum_squares += value * value;
#pragma unroll
        for (int k = 0; k < 8; ++k) { dots[k] += weight[k * width + index] * value; }
    }

    sum_squares = block_sum(sum_squares, reductions);
#pragma unroll
    for (int k = 0; k < 8; ++k) { dots[k] = block_sum(dots[k], reductions); }

    const float inverse = rsqrtf(sum_squares / static_cast<float>(width) + 1.0e-6f);
    float logits[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { logits[k] = dots[k] * inverse + bias[k]; }

    const float pre0  = sigmoid(logits[0]);
    const float pre1  = sigmoid(logits[1]);
    const float post0 = 2.0f * sigmoid(logits[2]);
    const float post1 = 2.0f * sigmoid(logits[3]);

    float m00 = logits[4];
    float m01 = logits[5];
    float m10 = logits[6];
    float m11 = logits[7];
#pragma unroll 2
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float row = logsumexp2(m00, m01);
        m00 -= row;
        m01 -= row;
        row = logsumexp2(m10, m11);
        m10 -= row;
        m11 -= row;
        float column = logsumexp2(m00, m10);
        m00 -= column;
        m10 -= column;
        column = logsumexp2(m01, m11);
        m01 -= column;
        m11 -= column;
    }

    if (tid == 0) {
        post[static_cast<std::int64_t>(token)]         = post0;
        post[static_cast<std::int64_t>(token) + tokens] = post1;
        const std::int64_t base = static_cast<std::int64_t>(token) * 4;
        mix[base]     = expf(m00); // destination 0, source 0
        mix[base + 1] = expf(m10); // destination 1, source 0
        mix[base + 2] = expf(m01); // destination 0, source 1
        mix[base + 3] = expf(m11); // destination 1, source 1
    }

    const auto pre0_bf16 = __float2bfloat16_rn(pre0);
    const auto pre1_bf16 = __float2bfloat16_rn(pre1);
    for (int column = tid; column < d; column += Block) {
        const auto* x0 = reinterpret_cast<const __nv_bfloat16*>(
            residual + static_cast<std::int64_t>(token) * residual_token_stride +
            static_cast<std::int64_t>(column) * sizeof(__nv_bfloat16));
        const auto* x1 = reinterpret_cast<const __nv_bfloat16*>(
            residual + static_cast<std::int64_t>(token) * residual_token_stride +
            residual_stream_stride + static_cast<std::int64_t>(column) * sizeof(__nv_bfloat16));
        float value = __bfloat162float(pre0_bf16) * __bfloat162float(*x0) +
                      __bfloat162float(pre1_bf16) * __bfloat162float(*x1);
        value *= pre_scale;
        branch[static_cast<std::int64_t>(token) * d + column] = __float2bfloat16_rn(value);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void mhc_merge_kernel(const char* residual, std::int64_t residual_stream_stride,
                          std::int64_t residual_token_stride, const __nv_bfloat16* branch,
                          const float* post, const float* mix, int d, __nv_bfloat16* out) {
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);

    const std::int64_t token_offset = static_cast<std::int64_t>(token);
    const int tokens = gridDim.x;
    const float post0 = __bfloat162float(__float2bfloat16_rn(post[token_offset]));
    const float post1 = __bfloat162float(__float2bfloat16_rn(post[token_offset + tokens]));
    const std::int64_t mix_base = token_offset * 4;
    const float m00 = __bfloat162float(__float2bfloat16_rn(mix[mix_base]));
    const float m10 = __bfloat162float(__float2bfloat16_rn(mix[mix_base + 1]));
    const float m01 = __bfloat162float(__float2bfloat16_rn(mix[mix_base + 2]));
    const float m11 = __bfloat162float(__float2bfloat16_rn(mix[mix_base + 3]));

    for (int column = tid; column < d; column += Block) {
        const auto* r0 = reinterpret_cast<const __nv_bfloat16*>(
            residual + token_offset * residual_token_stride +
            static_cast<std::int64_t>(column) * sizeof(__nv_bfloat16));
        const auto* r1 = reinterpret_cast<const __nv_bfloat16*>(
            residual + token_offset * residual_token_stride + residual_stream_stride +
            static_cast<std::int64_t>(column) * sizeof(__nv_bfloat16));
        const float x0 = __bfloat162float(*r0);
        const float x1 = __bfloat162float(*r1);
        const float b  = __bfloat162float(branch[token_offset * d + column]);
        const std::int64_t out_base = token_offset * 2 * d + column;
        out[out_base]       = __float2bfloat16_rn(m00 * x0 + m01 * x1 + post0 * b);
        out[out_base + d]   = __float2bfloat16_rn(m10 * x0 + m11 * x1 + post1 * b);
    }
}

} // namespace mhc_detail
} // namespace ninfer::ops
