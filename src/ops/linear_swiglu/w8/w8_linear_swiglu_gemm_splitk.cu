#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int Intermediate, int Hidden>
struct W8SwiGluExactTGeometry {
    static constexpr int kIntermediate = Intermediate;
    static constexpr int kHidden       = Hidden;
};
using QwenGeometry    = W8SwiGluExactTGeometry<6144, 2048>;
using SenseNovaGeometry = W8SwiGluExactTGeometry<12288, 4096>;
constexpr int kRowsPerCta   = 8;
constexpr int kFirstExactT  = 2;
constexpr int kLastExactT   = 48;
using ProjectionLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry>
struct W8SwiGluExactTRows {
    static constexpr int kOutputRowsPerCta = kRowsPerCta;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kRowsPerCta - 1)) +
               (local_row >= kRowsPerCta ? Geometry::kIntermediate : 0);
    }
};

struct W8SwiGluExactTEpilogue {
    __nv_bfloat16* out;
    int rows;

    template <int ActiveCols>
    __device__ __forceinline__ void store_pair(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * rows + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * rows + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

template <class Geometry, int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr auto ScaleAccess = W8SmallTMmaScaleAccess::Shared;
    using WeightGeometry = W8LinearGeometry<2 * Geometry::kIntermediate, Geometry::kHidden>;
    using Schedule =
        std::conditional_t<(ActiveCols <= 32), W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>,
                           W8SmallTMmaSchedule<4, TileCols, 4, ScaleAccess>>;
    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data),
                                            Geometry::kIntermediate};
    const W8SwiGluExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data),
                                          Geometry::kIntermediate};
    const W8SwiGluExactTRows<Geometry> row_policy{};
    w8_small_t_mma_kernel<WeightGeometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SwiGluExactTEpilogue, W8SwiGluExactTRows<Geometry>, true>
        <<<Geometry::kIntermediate / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Geometry, kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers<QwenGeometry>(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});
constexpr auto kSenseNovaLaunchers =
    make_launchers<SenseNovaGeometry>(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

void w8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    if (x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("W8 LinearSwiGLU exact split-K requires T=2..48");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void w8_linear_swiglu_sensenova_splitk_exact_t_launch(const Tensor& x, const Weight& w,
                                                      Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("W8 LinearSwiGLU SenseNova exact split-K requires T=2..48");
    }
    kSenseNovaLaunchers[x.ne[1] - kFirstExactT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
