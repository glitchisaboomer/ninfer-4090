#pragma once

// ninfer::ops::detail - private launch prototype for logprob_of_choice.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void logprob_of_choice_launch(const Tensor& logits, const Tensor& chosen, std::int32_t valid_rows,
                              Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail