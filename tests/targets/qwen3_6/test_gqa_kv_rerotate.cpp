// Oracle test for gqa_kv_rerotate: a re-rotated K element must equal the element a forward prefill at
// the destination position would have written, up to the single extra BF16 rounding the Op's
// contract allows.
//
// The oracle is built from first principles in FP64 on the host. `forward_rope_oracle` rotates a raw
// element pair by `position * freq`; the Op instead rotates an already-rotated pair by `delta * freq`.
// A rotation is a product of per-plane 2-D rotations whose angle is linear in position, so the two
// agree exactly in real arithmetic:
//
//   (a*c_P - b*s_P) * c_D - (b*c_P + a*s_P) * s_D  ==  a*c_{P+D} - b*s_{P+D}
//
// Any disagreement beyond BF16 rounding is therefore an implementation defect, not a modelling
// choice. The `freq` table is supplied per call rather than read from module-global state, which is
// what lets this test pin a caller-installed table that differs from the built-in default -- the
// regime the runtime actually runs in once a YaRN-scaled context capacity is configured.

#include "ninfer/ops/gqa_kv_rerotate.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kRotaryPairs = 32; // rotary_dim 64, so 32 rotated element planes.
constexpr std::int32_t kRotaryElems = 2 * kRotaryPairs;
constexpr std::int32_t kPageSize    = 64;
constexpr std::int32_t kKvHeads     = 4;

using FreqTable = std::array<float, kRotaryPairs>;

// The default Text inverse-frequency table: theta 1e7 over 32 rotary planes. Matches the literals in
// src/ops/kernel/rope.cuh so a default run reproduces the production default exactly.
FreqTable default_inv_freq() {
    FreqTable freq{};
    for (std::int32_t i = 0; i < kRotaryPairs; ++i) {
        freq[static_cast<std::size_t>(i)] =
            static_cast<float>(std::pow(1.0e7, -2.0 * static_cast<double>(i) / 64.0));
    }
    return freq;
}

// A stand-in for compute_yarn_config with scale_factor > 1, the regime where the table the runtime
// installs stops matching the compile-time defaults. Reproducing the shape rather than calling
// yarn.h keeps this test independent of that header while exercising the same failure mode.
FreqTable yarn_scaled_inv_freq() {
    FreqTable freq  = default_inv_freq();
    constexpr double kPi     = 3.14159265358979323846;
    constexpr double kNative = 1048576.0;
    constexpr double kRlow   = kNative / 32.0;
    const double scale       = 4.0; // a 4M capacity against a 1M native context
    for (std::int32_t i = 0; i < kRotaryPairs; ++i) {
        const double base       = std::pow(1.0e7, -2.0 * static_cast<double>(i) / 64.0);
        const double wavelength = (2.0 * kPi) / base;
        double gamma            = 0.0;
        if (wavelength > kNative) {
            gamma = 1.0;
        } else if (wavelength > kRlow) {
            gamma = (wavelength - kRlow) / (kNative - kRlow);
        }
        freq[static_cast<std::size_t>(i)] =
            static_cast<float>(base * ((1.0 - gamma) + gamma / scale));
    }
    return freq;
}

// Forward Text RoPE for one position, in FP64, over a row of head_dim elements. Elements
// [kRotaryElems, head_dim) carry no rotation and are copied unchanged.
void forward_rope_oracle(const double* raw, std::int32_t position, const FreqTable& freq,
                         double* out) {
    for (std::int32_t e = 0; e < kHeadDim; ++e) { out[e] = raw[e]; }
    for (std::int32_t pair = 0; pair < kRotaryPairs; ++pair) {
        const double theta = static_cast<double>(position) * static_cast<double>(freq[pair]);
        const double c     = std::cos(theta);
        const double s     = std::sin(theta);
        const double a     = raw[pair];
        const double b     = raw[pair + kRotaryPairs];
        out[pair]                = a * c - b * s;
        out[pair + kRotaryPairs] = b * c + a * s;
    }
}

// One BF16 ulp at `x`.
double bf16_ulp(double x) {
    if (x == 0.0) { return std::ldexp(1.0, -133); }
    return std::ldexp(1.0, static_cast<int>(std::ilogb(std::fabs(x))) - 7);
}

// The tolerance is measured in BF16 ulp of the *operand* pair, not of the result. A rotation can
// cancel its output toward zero, in which case the result's own ulp collapses and any ratio
// measured against it explodes even though the absolute error is bounded by the operand magnitude.
// Sweeping positions up to 131072 with deltas of +/-4096 gives a worst case of 1.79 ulp against an
// FP64 oracle, of which most is FP32 argument reduction in sincos at large angles rather than
// rounding. A sign error or a swapped pair in this Op costs roughly 4222 ulp on the same data, so
// the allowance separates correct from broken by about three orders of magnitude.
constexpr double kUlpAllowance = 4.0;

// The BF16 ulp of the operand pair, used to normalize an element's error.
double operand_ulp(double a, double b) {
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return bf16_ulp(scale == 0.0 ? 1.0 : scale);
}

double to_double(__nv_bfloat16 v) { return static_cast<double>(__bfloat162float(v)); }

__nv_bfloat16 to_bf16(double v) { return __float2bfloat16(static_cast<float>(v)); }

// Element index inside k_pages shaped (head_dim, page_size, kv_heads, physical_pages).
std::size_t k_index(std::int32_t physical_page, std::int32_t kv_head, std::int32_t page_offset,
                    std::int32_t leading) {
    return ((static_cast<std::size_t>(physical_page) * kKvHeads + kv_head) * kPageSize + page_offset) *
               kHeadDim +
           leading;
}

// A device-resident Text K plane plus its block table. The block table is identity over the logical
// pages, and a host mirror is kept because the test resolves destination pages on the host.
struct Plane {
    __nv_bfloat16* k_pages            = nullptr;
    std::int32_t* block_table         = nullptr;
    std::int32_t physical_pages       = 0;
    std::int32_t logical_pages        = 0;
    std::vector<std::int32_t> table_host;

    explicit Plane(std::int32_t physical, std::int32_t logical)
        : physical_pages(physical), logical_pages(logical),
          table_host(static_cast<std::size_t>(logical)) {
        const std::size_t k_elems =
            static_cast<std::size_t>(kHeadDim) * kPageSize * kKvHeads * static_cast<std::size_t>(physical);
        CUDA_CHECK(cudaMalloc(&k_pages, k_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(k_pages, 0, k_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&block_table, sizeof(std::int32_t) * table_host.size()));
        for (std::int32_t i = 0; i < logical; ++i) {
            table_host[static_cast<std::size_t>(i)] = i % physical;
        }
        CUDA_CHECK(cudaMemcpy(block_table, table_host.data(), sizeof(std::int32_t) * table_host.size(),
                              cudaMemcpyHostToDevice));
    }
    ~Plane() {
        if (k_pages != nullptr) { (void)cudaFree(k_pages); }
        if (block_table != nullptr) { (void)cudaFree(block_table); }
    }
    Plane(const Plane&)            = delete;
    Plane& operator=(const Plane&) = delete;

    [[nodiscard]] std::int32_t physical_page_of(std::int32_t position) const {
        return table_host[static_cast<std::size_t>(position / kPageSize)];
    }

    [[nodiscard]] ninfer::PagedKVLayerView view() const {
        ninfer::PagedKVLayerView v;
        v.k_pages =
            ninfer::Tensor(k_pages, ninfer::DType::BF16, {kHeadDim, kPageSize, kKvHeads, physical_pages});
        v.block_table = ninfer::Tensor(block_table, ninfer::DType::I32, {logical_pages, 1, 1, 1});
        v.head_dim     = kHeadDim;
        v.num_kv_heads = kKvHeads;
        v.dtype        = ninfer::DType::BF16;
        return v;
    }

    // Writes `value` for every head of `position`.
    void write_row(std::int32_t position, const double* value) {
        const std::int32_t phys = physical_page_of(position);
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t e = 0; e < kHeadDim; ++e) {
                k_pages[k_index(phys, head, position % kPageSize, e)] = to_bf16(value[e]);
            }
        }
    }

    // Reads back the row of one head at `position`.
    void read_row(std::int32_t position, std::int32_t head, double* value) const {
        const std::int32_t phys = physical_page_of(position);
        for (std::int32_t e = 0; e < kHeadDim; ++e) {
            value[e] = to_double(k_pages[k_index(phys, head, position % kPageSize, e)]);
        }
    }
};

// Relocate one logical position and compare every head against a direct prefill at the destination.
// Reports the worst BF16-ulp distance over the rotary region and checks that the rest of the row was
// left byte-identical.
void check_relocation(Plane& plane, std::int32_t src_position, std::int32_t dst_position,
                      const std::vector<std::array<double, kHeadDim>>& raw, const FreqTable& freq,
                      const char* label) {
    const std::int32_t delta = dst_position - src_position;

    // Seed the cache with what a forward prefill at src_position leaves behind, rounded once to BF16.
    std::array<double, kHeadDim> rotated{};
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        forward_rope_oracle(raw[static_cast<std::size_t>(head)].data(), src_position, freq,
                            rotated.data());
        plane.write_row(src_position, rotated.data());
    }

    ninfer::ops::gqa_kv_rerotate_run(dst_position, 1, delta,
                                     std::span<const float, kRotaryPairs>(freq), plane.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    double worst = 0.0;
    std::array<double, kHeadDim> got{};
    std::array<double, kHeadDim> oracle{};
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        forward_rope_oracle(raw[static_cast<std::size_t>(head)].data(), dst_position, freq,
                            oracle.data());
        plane.read_row(dst_position, head, got.data());
        for (std::int32_t e = 0; e < kHeadDim; ++e) {
            if (e >= kRotaryElems) {
                // Outside [0, 64) the Op must leave the row byte-identical.
                const double untouched = to_double(to_bf16(rotated[e]));
                if (got[e] != untouched) {
                    fail(std::string(label) + ": element " + std::to_string(e) +
                         " outside [0,64) was modified");
                    return;
                }
                continue;
            }
            const std::int32_t p = e < kRotaryPairs ? e : e - kRotaryPairs;
            const double& a = raw[static_cast<std::size_t>(head)][static_cast<std::size_t>(p)];
            const double& b =
                raw[static_cast<std::size_t>(head)][static_cast<std::size_t>(p + kRotaryPairs)];
            worst = std::max(worst, std::fabs(got[e] - oracle[e]) / operand_ulp(a, b));
        }
    }
    char buf[256];
    if (worst > kUlpAllowance) {
        std::snprintf(buf, sizeof(buf), "%s: worst error %.2f BF16 ulp (allowance %.1f)", label, worst,
                      kUlpAllowance);
        fail(buf);
    } else {
        std::snprintf(buf, sizeof(buf), "%s: ok (worst %.2f BF16 ulp)", label, worst);
        std::printf("  %s\n", buf);
    }
}

// The vector entry point must agree with the run entry point on the same relocation.
void check_vector_entry_point(Plane& plane, std::int32_t src_position, std::int32_t dst_position,
                              const std::vector<std::array<double, kHeadDim>>& raw,
                              const FreqTable& freq) {
    const std::int32_t delta = dst_position - src_position;

    std::array<double, kHeadDim> rotated{};
    forward_rope_oracle(raw[0].data(), src_position, freq, rotated.data());
    plane.write_row(src_position, rotated.data());

    std::int32_t* d_pos   = nullptr;
    std::int32_t* d_delta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_delta, sizeof(std::int32_t)));
    const std::int32_t h_pos[1]   = {dst_position};
    const std::int32_t h_delta[1] = {delta};
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta, h_delta, sizeof(std::int32_t), cudaMemcpyHostToDevice));

    ninfer::ops::gqa_kv_rerotate(ninfer::Tensor(d_pos, ninfer::DType::I32, {1, 1, 1, 1}),
                                 ninfer::Tensor(d_delta, ninfer::DType::I32, {1, 1, 1, 1}), freq,
                                 plane.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    std::array<double, kHeadDim> oracle{};
    std::array<double, kHeadDim> got{};
    forward_rope_oracle(raw[0].data(), dst_position, freq, oracle.data());
    plane.read_row(dst_position, 0, got.data());

    double worst = 0.0;
    for (std::int32_t e = 0; e < kRotaryElems; ++e) {
        const std::int32_t p = e < kRotaryPairs ? e : e - kRotaryPairs;
        const double& a      = raw[0][static_cast<std::size_t>(p)];
        const double& b      = raw[0][static_cast<std::size_t>(p + kRotaryPairs)];
        worst = std::max(worst, std::fabs(got[e] - oracle[e]) / operand_ulp(a, b));
    }
    char buf[256];
    if (worst > kUlpAllowance) {
        std::snprintf(buf, sizeof(buf), "vector entry point: worst error %.2f BF16 ulp (allowance %.1f)",
                      worst, kUlpAllowance);
        fail(buf);
    } else {
        std::snprintf(buf, sizeof(buf), "vector entry point: ok (worst %.2f BF16 ulp)", worst);
        std::printf("  %s\n", buf);
    }
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_delta));
}

// A zero delta is the identity: re-rotating by 0 must reproduce the stored BF16 value exactly, with no
// second rounding at all.
void check_zero_delta(Plane& plane, std::int32_t position,
                      const std::vector<std::array<double, kHeadDim>>& raw, const FreqTable& freq) {
    std::array<double, kHeadDim> rotated{};
    forward_rope_oracle(raw[0].data(), position, freq, rotated.data());
    plane.write_row(position, rotated.data());

    ninfer::ops::gqa_kv_rerotate_run(position, 1, 0, std::span<const float, kRotaryPairs>(freq),
                                     plane.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    std::array<double, kHeadDim> got{};
    plane.read_row(position, 0, got.data());
    const int before = failures;
    for (std::int32_t e = 0; e < kRotaryElems; ++e) {
        if (got[e] != to_double(to_bf16(rotated[e]))) {
            fail("zero delta is not the identity at element " + std::to_string(e));
            break;
        }
    }
    if (failures == before) { std::printf("  zero delta: ok (identity)\n"); }
}

// Geometry and argument rejection. These are pure host checks but run here so a rejected relocation
// can never be mistaken for a silent one.
template <typename Fn>
void expect_throw(Fn fn, const char* label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    fail(std::string(label) + ": expected a rejection");
}

void check_validation(Plane& plane) {
    const FreqTable freq = default_inv_freq();

    ninfer::PagedKVLayerView base = plane.view();

    expect_throw(
        [&] {
            ninfer::ops::gqa_kv_rerotate_run(0, 0, 0, std::span<const float, kRotaryPairs>(freq), base,
                                             nullptr);
        },
        "run with tokens == 0");
    expect_throw(
        [&] {
            const std::int32_t capacity = plane.logical_pages * kPageSize;
            ninfer::ops::gqa_kv_rerotate_run(capacity - 1, 2, 0,
                                             std::span<const float, kRotaryPairs>(freq), base,
                                             nullptr);
        },
        "run past the cache capacity");
    expect_throw(
        [&] {
            ninfer::PagedKVLayerView bad = base;
            bad.head_dim                 = 128;
            ninfer::ops::gqa_kv_rerotate_run(0, 1, 0, std::span<const float, kRotaryPairs>(freq), bad,
                                             nullptr);
        },
        "run with head_dim != 256");
    expect_throw(
        [&] {
            ninfer::PagedKVLayerView bad = base;
            bad.dtype                    = ninfer::DType::I8;
            ninfer::ops::gqa_kv_rerotate_run(0, 1, 0, std::span<const float, kRotaryPairs>(freq), bad,
                                             nullptr);
        },
        "run with a non-BF16 cache");
    expect_throw(
        [&] {
            ninfer::PagedKVLayerView bad = base;
            bad.packed_k                 = true;
            ninfer::ops::gqa_kv_rerotate_run(0, 1, 0, std::span<const float, kRotaryPairs>(freq), bad,
                                             nullptr);
        },
        "run against a packed K plane");
    expect_throw(
        [&] {
            ninfer::PagedKVLayerView bad       = base;
            bad.num_kv_heads                  = 8;
            bad.k_pages.ne[2]                 = 8;
            ninfer::ops::gqa_kv_rerotate_run(0, 1, 0, std::span<const float, kRotaryPairs>(freq), bad,
                                             nullptr);
        },
        "run with 8 KV heads");
    if (failures == 0) { std::printf("  validation rejections: ok\n"); }
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }

    std::mt19937 rng(1234);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<std::array<double, kHeadDim>> raw(static_cast<std::size_t>(kKvHeads));
    for (auto& head_row : raw) {
        for (double& v : head_row) { v = normal(rng); }
    }

    Plane plane(8, 16);

    // Source and destination deliberately sit in different logical pages, so the Op has to resolve
    // the destination through the block table instead of reusing the source page.
    constexpr std::int32_t kSrc = 130;
    constexpr std::int32_t kDst = 66;

    const FreqTable defaults = default_inv_freq();
    const FreqTable yarn     = yarn_scaled_inv_freq();

    std::printf("gqa_kv_rerotate oracle checks\n");

    check_relocation(plane, kSrc, kDst, raw, defaults, "default table");
    check_vector_entry_point(plane, kSrc, kDst, raw, defaults);
    check_zero_delta(plane, kSrc, raw, defaults);
    // The regression guard: this fails whenever the Op reads a frequency table the caller did not
    // install, which is the state the YaRN path puts the runtime in.
    check_relocation(plane, kSrc, kDst, raw, yarn, "yarn-scaled table");
    check_validation(plane);

    if (failures != 0) {
        std::printf("gqa_kv_rerotate oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("gqa_kv_rerotate oracle checks passed\n");
    return 0;
}
