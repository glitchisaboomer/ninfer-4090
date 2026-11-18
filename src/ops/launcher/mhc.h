#pragma once

// ninfer::ops::detail - private launch prototypes for mHC.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mhc_route_launch(const Tensor& residual, const Tensor& weight, const Tensor& bias,
                      int streams, int iterations, float pre_scale, Tensor& branch,
                      Tensor& post, Tensor& mix, cudaStream_t stream);

void mhc_merge_launch(const Tensor& residual, const Tensor& branch, const Tensor& post,
                      const Tensor& mix, int streams, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
