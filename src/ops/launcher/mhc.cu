// ninfer::ops - mHC launcher.
#include "ops/launcher/mhc.h"

#include "core/device.h"
#include "ops/kernel/mhc.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

} // namespace

void mhc_route_launch(const Tensor& residual, const Tensor& weight, const Tensor& bias,
                      int /*streams*/, int iterations, float pre_scale, Tensor& branch,
                      Tensor& post, Tensor& mix, cudaStream_t stream) {
    const int d    = residual.ne[0];
    const int tokens = residual.ne[2];
    constexpr int kBlock = 256;
    mhc_detail::mhc_route_kernel<kBlock><<<static_cast<unsigned int>(tokens), kBlock, 0, stream>>>(
        static_cast<const char*>(residual.data), residual.nb[1], residual.nb[2],
        static_cast<const float*>(weight.data), static_cast<const float*>(bias.data), d,
        iterations, pre_scale, static_cast<__nv_bfloat16*>(branch.data),
        static_cast<float*>(post.data), static_cast<float*>(mix.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void mhc_reduce_launch(const Tensor& residual, int /*streams*/, Tensor& out, cudaStream_t stream) {
    const int d      = residual.ne[0];
    const int tokens = residual.ne[2];
    constexpr int kBlock = 256;
    mhc_detail::mhc_reduce_kernel<kBlock><<<static_cast<unsigned int>(tokens), kBlock, 0, stream>>>(
        static_cast<const char*>(residual.data), residual.nb[1], residual.nb[2], d,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void mhc_merge_launch(const Tensor& residual, const Tensor& branch, const Tensor& post,
                      const Tensor& mix, int /*streams*/, Tensor& out, cudaStream_t stream) {
    const int d      = residual.ne[0];
    const int tokens = residual.ne[2];
    constexpr int kBlock = 256;
    mhc_detail::mhc_merge_kernel<kBlock><<<static_cast<unsigned int>(tokens), kBlock, 0, stream>>>(
        static_cast<const char*>(residual.data), residual.nb[1], residual.nb[2],
        static_cast<const __nv_bfloat16*>(branch.data), static_cast<const float*>(post.data),
        static_cast<const float*>(mix.data), d, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
