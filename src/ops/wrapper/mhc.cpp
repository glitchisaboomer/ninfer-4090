// ninfer::ops - mHC wrapper: public API validation and launcher dispatch.
#include "ninfer/ops/mhc.h"

#include "ops/launcher/mhc.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("mhc: ") + message); }
}

std::int64_t product(const Tensor& t) {
    std::int64_t result = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) { throw std::invalid_argument("mhc: tensor dimensions must be nonnegative"); }
        result *= t.ne[d];
    }
    return result;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    if (a.data == nullptr || b.data == nullptr) { return false; }
    const auto* ab = static_cast<const std::uint8_t*>(a.data);
    const auto* bb = static_cast<const std::uint8_t*>(b.data);
    const auto a_begin = reinterpret_cast<std::uintptr_t>(ab);
    const auto a_end = a_begin + static_cast<std::uintmax_t>(a.numel()) * dtype_size(a.dtype);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(bb);
    const auto b_end = b_begin + static_cast<std::uintmax_t>(b.numel()) * dtype_size(b.dtype);
    return a_begin < b_end && b_begin < a_end;
}

void require_residual(const Tensor& residual, int streams) {
    require(residual.dtype == DType::BF16, "residual must be BF16");
    require(residual.ne[0] > 0 && residual.ne[1] == streams && residual.ne[2] > 0 &&
                residual.ne[3] == 1,
            "residual must be [D,2,T]");
    require(residual.nb[0] == static_cast<std::int64_t>(dtype_size(DType::BF16)),
            "residual feature stride must be unit");
    require(residual.data != nullptr, "residual data must be non-null");
}

} // namespace

void mhc_route(const Tensor& residual, const Tensor& weight, const Tensor& bias,
               int streams, int iterations, float pre_scale, Tensor& branch,
               Tensor& post, Tensor& mix, cudaStream_t stream) {
    require(streams == 2, "only two residual streams are supported");
    require(iterations > 0, "iterations must be positive");
    require(std::isfinite(pre_scale) && pre_scale > 0.0f, "pre_scale must be positive and finite");
    require_residual(residual, streams);
    const int d = residual.ne[0];
    const int tokens = residual.ne[2];

    require(weight.dtype == DType::FP32 && bias.dtype == DType::FP32,
            "weight and bias must be FP32");
    require(product(weight) == static_cast<std::int64_t>(8) * 2 * d && weight.ne[3] == 1 &&
                weight.ne[1] == 8 && weight.ne[2] == 1,
            "weight must be FP32 [2*D,8] in ninfer order");
    require(weight.nb[0] == static_cast<std::int64_t>(sizeof(float)) && weight.is_contiguous(),
            "weight must have unit feature stride and be contiguous");
    require(product(bias) == 8 && bias.ne[0] == 8 && bias.ne[1] == 1 && bias.ne[2] == 1 &&
                bias.ne[3] == 1,
            "bias must be FP32 [8]");
    require(branch.dtype == DType::BF16 && branch.ne[0] == d && branch.ne[1] == tokens &&
                branch.ne[2] == 1 && branch.ne[3] == 1 && branch.is_contiguous(),
            "branch must be contiguous BF16 [D,T]");
    require(post.dtype == DType::FP32 && post.ne[0] == 2 && post.ne[1] == tokens &&
                post.ne[2] == 1 && post.ne[3] == 1 && post.is_contiguous(),
            "post must be contiguous FP32 [2,T]");
    require(mix.dtype == DType::FP32 && mix.ne[0] == 2 && mix.ne[1] == 2 &&
                mix.ne[2] == tokens && mix.ne[3] == 1 && mix.is_contiguous(),
            "mix must be contiguous FP32 [2,2,T]");
    require(weight.data != nullptr && bias.data != nullptr && branch.data != nullptr &&
                post.data != nullptr && mix.data != nullptr,
            "input and output data must be non-null");
    require(!overlaps(residual, branch) && !overlaps(residual, post) &&
                !overlaps(residual, mix) && !overlaps(weight, branch) &&
                !overlaps(weight, post) && !overlaps(weight, mix),
            "inputs and outputs must not overlap");

    detail::mhc_route_launch(residual, weight, bias, streams, iterations, pre_scale, branch,
                             post, mix, stream);
}

void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post,
               const Tensor& mix, int streams, Tensor& out, cudaStream_t stream) {
    require(streams == 2, "only two residual streams are supported");
    require_residual(residual, streams);
    const int d = residual.ne[0];
    const int tokens = residual.ne[2];

    require(branch.dtype == DType::BF16 && branch.ne[0] == d && branch.ne[1] == tokens &&
                branch.ne[2] == 1 && branch.ne[3] == 1 && branch.is_contiguous(),
            "branch must be contiguous BF16 [D,T]");
    require(post.dtype == DType::FP32 && post.ne[0] == 2 && post.ne[1] == tokens &&
                post.ne[2] == 1 && post.ne[3] == 1 && post.is_contiguous(),
            "post must be contiguous FP32 [2,T]");
    require(mix.dtype == DType::FP32 && mix.ne[0] == 2 && mix.ne[1] == 2 &&
                mix.ne[2] == tokens && mix.ne[3] == 1 && mix.is_contiguous(),
            "mix must be contiguous FP32 [2,2,T]");
    require(out.dtype == DType::BF16 && out.ne[0] == d && out.ne[1] == 2 &&
                out.ne[2] == tokens && out.ne[3] == 1 && out.is_contiguous(),
            "out must be contiguous BF16 [D,2,T]");
    require(branch.data != nullptr && post.data != nullptr && mix.data != nullptr &&
                out.data != nullptr,
            "input and output data must be non-null");
    require(!overlaps(residual, out) && !overlaps(branch, out) && !overlaps(post, out) &&
                !overlaps(mix, out),
            "inputs and out must not overlap");

    detail::mhc_merge_launch(residual, branch, post, mix, streams, out, stream);
}

void mhc_reduce(const Tensor& residual, int streams, Tensor& out, cudaStream_t stream) {
    require(streams == 2, "only two residual streams are supported");
    require_residual(residual, streams);
    const int d      = residual.ne[0];
    const int tokens = residual.ne[2];

    require(out.dtype == DType::BF16 && out.ne[0] == d && out.ne[1] == tokens &&
                out.ne[2] == 1 && out.ne[3] == 1 && out.is_contiguous(),
            "out must be contiguous BF16 [D,T]");
    require(out.data != nullptr, "out data must be non-null");
    require(!overlaps(residual, out), "input and out must not overlap");

    detail::mhc_reduce_launch(residual, streams, out, stream);
}

} // namespace ninfer::ops
