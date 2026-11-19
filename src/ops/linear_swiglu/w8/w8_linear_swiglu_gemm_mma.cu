#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"



namespace ninfer::ops::detail {
namespace {

template <int Intermediate, int Hidden>
struct W8SwiGluGeometry {
    static constexpr int kIntermediate = Intermediate;
    static constexpr int kHidden       = Hidden;
};
using QwenGeometry     = W8SwiGluGeometry<6144, 2048>;
using SenseNovaGeometry = W8SwiGluGeometry<12288, 4096>;

template <class Geometry, class Schedule, bool Full>
void launch_variant(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kIntermediate;
    static_assert((kIntermediate % (Schedule::BM / 2)) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const dim3 grid(kIntermediate / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(w.qdata),
                                                 static_cast<const std::uint8_t*>(w.scales), output,
                                                 2 * kIntermediate, Geometry::kHidden, x.ne[1],
                                                 Geometry::kHidden);
}

template <class Geometry, class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<Geometry, Schedule, true>(x, w, out, stream);
    } else {
        launch_variant<Geometry, Schedule, false>(x, w, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_sensenova(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<Geometry, Schedule>(x, w, out, stream);
}

} // namespace

void w8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c32_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 32, 64, 16, 3>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c48_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 48, 64, 16, 3>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c80_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w,
    Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    launch_route<QwenGeometry, Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_sensenova_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                    cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                    cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                    cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                    cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                    cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                     cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                     cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>>(
        x, w, out, stream);
}
void w8_linear_swiglu_sensenova_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                     cudaStream_t stream) {
    launch_sensenova<SenseNovaGeometry, W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>>(
        x, w, out, stream);
}

} // namespace ninfer::ops::detail
