// Implements: include/ninfer/ops/logprob_of_choice.h
// Match: validated contiguous BF16 [physical_rows,T] logits, I32 [T] chosen ids, FP32 [T] out.
// Algorithm assumptions: one CTA per column, vocabulary streamed once with a running
// (max, sum-of-exp) state per thread. No scratch and no workspace: the reduction axis is the
// vocabulary, so a second kernel to combine partials would need a buffer for no measurable gain
// against a decode step's weight traffic.
#include "ops/launcher/logprob_of_choice.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/logprob_of_choice.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void logprob_of_choice_launch(const Tensor& logits, const Tensor& chosen, std::int32_t valid_rows,
                              Tensor& out, cudaStream_t stream) {
    logprob_of_choice_kernel<<<static_cast<std::uint32_t>(logits.ne[1]), kLogprobBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(chosen.data), valid_rows, logits.ne[0],
        static_cast<float*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail