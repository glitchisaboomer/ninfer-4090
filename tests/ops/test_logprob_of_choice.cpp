// Oracle test for logprob_of_choice: the emitted log-probability must equal the FP64 log-softmax
// entry of the chosen token, computed from the same BF16 logits the Op reads.
//
// The contract is a mathematical identity, so the oracle is built from first principles rather than
// from the Op:
//
//   logsumexp(v) = m + log(sum_v exp(logits[v] - m)),  m = max_v logits[v]
//   out[t]       = logits[chosen[t]] - logsumexp
//
// The two properties this file exists to catch are the ones a running-maximum reduction gets wrong
// silently. A missed rescale when a new maximum arrives diverges, which is why the test feeds a
// vocabulary whose maximum is *not* on the first thread's element and includes a large shared offset,
// the shape a trained model's final layer actually produces. A vocabulary whose largest logit sits
// at index 0 would pass under a broken implementation that never rebases.
//
// The scalar and vectorised paths are both exercised: the kernel selects between them on
// `valid_rows % 8`, so a row count that is not a multiple of eight takes a different code path than
// one that is, and both have to agree with the oracle.

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/logprob_of_choice.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

double to_double(__nv_bfloat16 v) { return static_cast<double>(__bfloat162float(v)); }

__nv_bfloat16 to_bf16(double v) { return __float2bfloat16(static_cast<float>(v)); }

// The value the staging below writes past a column's own length. A column shorter than valid_rows is
// a request to reduce a padded plane, not a licence to read past the fixture: the padding is part of
// what the Op reads, so the oracle has to reduce it too.
constexpr double kStagingPadding = -30.0;

// The exact BF16 plane the Op reduces for column `t`: the column's own values followed by the
// staging padding, truncated to valid_rows. Building this by indexing `columns[t][v]` for
// v < valid_rows reads past the fixture when the column is shorter than valid_rows, which is
// precisely the near-tie case below, so the length has to be honoured here rather than assumed.
std::vector<double> staged_column(const std::vector<std::vector<double>>& columns, std::size_t t,
                                  std::int32_t valid_rows) {
    const std::vector<double>& source = columns[t];
    std::vector<double> column;
    column.reserve(static_cast<std::size_t>(valid_rows));
    for (std::int32_t v = 0; v < valid_rows; ++v) {
        const double value = v < static_cast<std::int32_t>(source.size())
                                 ? source[static_cast<std::size_t>(v)]
                                 : kStagingPadding;
        column.push_back(to_double(to_bf16(value)));
    }
    return column;
}

// FP64 log-softmax of one column, evaluated from the exact values the Op will read.
double oracle_logprob(const std::vector<double>& column, std::int32_t chosen) {
    double peak = -std::numeric_limits<double>::infinity();
    for (const double value : column) { peak = std::max(peak, value); }
    double total = 0.0;
    for (const double value : column) { total += std::exp(value - peak); }
    return column[static_cast<std::size_t>(chosen)] - (peak + std::log(total));
}

// The Op accumulates in FP32 with __expf/__logf, whose error grows with the magnitude of the
// arguments. Tolerance is relative to the result, and the allowance covers the accumulated rounding
// of a ~150k-term FP32 sum at the offsets this test uses. A missing rescale is off by orders of
// magnitude, so the separation is wide either way.
constexpr double kTolerance = 2.0e-5;

// FP64 top-two gap of one column, from the exact values the Op will read: the largest value minus the
// largest strictly-smaller one. Returns +inf for a single valid row, which is what the Op emits there.
double oracle_margin(const std::vector<double>& column, std::int32_t valid_rows) {
    double top = -std::numeric_limits<double>::infinity();
    double next = -std::numeric_limits<double>::infinity();
    for (std::int32_t v = 0; v < valid_rows; ++v) {
        const double value = column[static_cast<std::size_t>(v)];
        if (value > top) {
            next = top;
            top  = value;
        } else if (value > next) {
            next = value;
        }
    }
    return top - next;
}

struct Columns {
    __nv_bfloat16* device_logits  = nullptr;
    std::int32_t* device_chosen   = nullptr;
    float* device_out            = nullptr;
    float* device_margin          = nullptr;
    std::vector<float> host_out;
    std::vector<float> host_margin;
    std::int32_t physical_rows   = 0;
    std::int32_t count           = 0;

    Columns(const std::vector<std::vector<double>>& columns, std::int32_t physical_rows_in)
        : physical_rows(physical_rows_in), count(static_cast<std::int32_t>(columns.size())) {
        const std::size_t elems =
            static_cast<std::size_t>(physical_rows) * static_cast<std::size_t>(count);
        // Physical rows past the column's own length are padded with a value far below anything the
        // oracle sees, so a reduction that ignores valid_rows lands on a visibly wrong total.
        std::vector<__nv_bfloat16> staging(elems);
        for (std::int32_t t = 0; t < count; ++t) {
            const std::vector<double>& column = columns[static_cast<std::size_t>(t)];
            for (std::int32_t v = 0; v < physical_rows; ++v) {
                staging[static_cast<std::size_t>(t) * static_cast<std::size_t>(physical_rows) +
                        static_cast<std::size_t>(v)] =
                    v < static_cast<std::int32_t>(column.size())
                        ? to_bf16(column[static_cast<std::size_t>(v)])
                        : to_bf16(kStagingPadding);
            }
        }
        CUDA_CHECK(cudaMalloc(&device_logits, elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(device_logits, staging.data(), elems * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&device_chosen,
                              static_cast<std::size_t>(count) * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&device_out, static_cast<std::size_t>(count) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&device_margin, static_cast<std::size_t>(count) * sizeof(float)));
        host_out.assign(static_cast<std::size_t>(count), 0.0F);
        host_margin.assign(static_cast<std::size_t>(count), 0.0F);
    }
    Columns(const Columns&)            = delete;
    Columns& operator=(const Columns&) = delete;
    ~Columns() {
        if (device_logits != nullptr) { (void)cudaFree(device_logits); }
        if (device_chosen != nullptr) { (void)cudaFree(device_chosen); }
        if (device_out != nullptr) { (void)cudaFree(device_out); }
        if (device_margin != nullptr) { (void)cudaFree(device_margin); }
    }

    [[nodiscard]] ninfer::Tensor logits() const {
        return ninfer::Tensor(device_logits, ninfer::DType::BF16, {physical_rows, count});
    }
    [[nodiscard]] ninfer::Tensor chosen() const {
        return ninfer::Tensor(device_chosen, ninfer::DType::I32, {count});
    }
    [[nodiscard]] ninfer::Tensor out() const {
        return ninfer::Tensor(device_out, ninfer::DType::FP32, {count});
    }
    [[nodiscard]] ninfer::Tensor margin() const {
        return ninfer::Tensor(device_margin, ninfer::DType::FP32, {count});
    }
    void read_back() {
        CUDA_CHECK(cudaMemcpy(host_out.data(), device_out, host_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_margin.data(), device_margin,
                              host_margin.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

// Runs the Op over `columns`, padded to physical_rows, and compares every emitted value to the FP64
// oracle for the chosen id.
void check(const std::vector<std::vector<double>>& columns,
           const std::vector<std::int32_t>& picks, std::int32_t physical_rows, std::int32_t valid_rows,
           const std::string& label) {
    if (static_cast<std::int32_t>(columns.size()) != static_cast<std::int32_t>(picks.size())) {
        fail(label + ": test built mismatched columns and picks");
        return;
    }

    Columns state(columns, physical_rows);
    CUDA_CHECK(cudaMemcpy(state.device_chosen, picks.data(),
                          static_cast<std::size_t>(picks.size()) * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));

    ninfer::Tensor out = state.out();
    ninfer::Tensor margin = state.margin();
    ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), valid_rows, out, margin, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    state.read_back();

    double worst         = 0.0;
    std::int32_t worst_t = -1;
    for (std::size_t t = 0; t < state.host_out.size(); ++t) {
        // The oracle must reduce the values the Op actually reads, which are the BF16-rounded ones
        // staged into the plane above -- not the FP64 values the test generated. The numerical
        // contract is explicit that BF16 inputs are interpreted after their storage boundary, and the
        // rounding is not a rounding error of the Op: it is the caller's storage decision, and it moves
        // the answer by up to half a BF16 ulp of the chosen logit, which for a logit near 5 is ~0.01
        // nats against a result of magnitude ~9. Measuring the rounding would bury the reduction's own
        // error, which is what this file exists to bound.
        const std::vector<double> column = staged_column(columns, t, valid_rows);
        const double expected = oracle_logprob(column, picks[t]);
        const double error    =
            std::fabs(state.host_out[t] - expected) / std::max(1.0, std::fabs(expected));
        if (error > worst) {
            worst   = error;
            worst_t = static_cast<std::int32_t>(t);
        }
    }

    char buf0[256];
    // The margin is a difference of two logits, so it is exact in FP32 -- no reduction, no exp, and
    // both operands are BF16 values widened without loss. Compared exactly, not with a tolerance: a
    // merge that dropped the losing partial's maximum would still land near the right answer and would
    // pass any loose bound.
    double margin_worst = 0.0;
    std::int32_t margin_worst_t = -1;
    for (std::size_t t = 0; t < state.host_margin.size(); ++t) {
        const std::vector<double> column = staged_column(columns, t, valid_rows);
        const double expected = oracle_margin(column, valid_rows);
        const double got      = state.host_margin[t];
        if (expected > 1.0e30) {
            // A single valid row has no runner-up; the Op must say so rather than report a confident 0.
            if (!std::isinf(got) || got < 0.0) {
                std::snprintf(buf0, sizeof(buf0), "%s: single-candidate column reported margin %g",
                              label.c_str(), got);
                fail(buf0);
                return;
            }
            continue;
        }
        const double error = std::fabs(got - expected);
        if (error > margin_worst) {
            margin_worst   = error;
            margin_worst_t = static_cast<std::int32_t>(t);
        }
    }
    if (margin_worst != 0.0) {
        std::snprintf(buf0, sizeof(buf0),
                      "%s: worst margin error %.3e at column %d (must be exact)",
                      label.c_str(), margin_worst, margin_worst_t);
        fail(buf0);
        return;
    }

    char buf[256];
    if (worst > kTolerance) {
        std::snprintf(buf, sizeof(buf), "%s: worst relative error %.3e at column %d (allowance %.1e)",
                      label.c_str(), worst, worst_t, kTolerance);
        fail(buf);
        return;
    }
    std::snprintf(buf, sizeof(buf), "%s: ok (worst relative error %.3e)", label.c_str(), worst);
    std::printf("  %s\n", buf);
}

template <typename Fn>
void expect_throw(Fn fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    fail(label + ": expected a rejection");
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }

    std::mt19937 rng(4242u);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::printf("logprob_of_choice oracle checks\n");

    // The production shape: a full vocabulary with a large shared offset, so the running maximum is
    // large and positive and an unshifted exp would overflow.
    {
        constexpr std::int32_t kRows = 4096;
        constexpr std::int32_t kCols = 3;
        std::vector<std::vector<double>> columns(static_cast<std::size_t>(kCols));
        for (auto& column : columns) {
            column.resize(static_cast<std::size_t>(kRows));
            for (double& value : column) { value = 40.0 + normal(rng); }
        }
        // The argmax sits away from index 0 and from any single thread's first element, so a
        // reduction that never rebases onto a new maximum cannot pass.
        for (std::int32_t t = 0; t < kCols; ++t) {
            auto& column = columns[static_cast<std::size_t>(t)];
            const std::size_t peak = static_cast<std::size_t>(1234 + 977 * t) % static_cast<std::size_t>(kRows);
            column[peak] = 60.0;
        }
        const std::vector<std::int32_t> picks{1234, 2211, 3188};
        check(columns, picks, kRows, kRows, "full vocabulary with a large offset");
    }

    // The vectorised path, and one where the padding would be visible if it leaked into the sum.
    {
        constexpr std::int32_t kRows   = 1024;
        constexpr std::int32_t kValid  = 1024;
        constexpr std::int32_t kCols   = 5;
        std::vector<std::vector<double>> columns(static_cast<std::size_t>(kCols));
        for (auto& column : columns) {
            column.resize(static_cast<std::size_t>(kRows));
            for (double& value : column) { value = normal(rng); }
        }
        const std::vector<std::int32_t> picks{0, 511, 1023, 7, 900};
        check(columns, picks, kRows, kValid, "vectorised path, exact fit");
    }

    // valid_rows below physical_rows: the padding holds -30, which would dominate an unguarded sum.
    {
        constexpr std::int32_t kRows  = 1031;
        constexpr std::int32_t kValid = 1000;
        constexpr std::int32_t kCols  = 4;
        std::vector<std::vector<double>> columns(static_cast<std::size_t>(kCols));
        for (auto& column : columns) {
            column.resize(static_cast<std::size_t>(kRows));
            for (double& value : column) { value = 5.0 + normal(rng); }
        }
        const std::vector<std::int32_t> picks{0, 1, 999, 500};
        check(columns, picks, kRows, kValid, "scalar path, padded rows excluded");
    }

    // A single row: every column is a point mass, so the log-probability is exactly zero and any
    // reduction that divides by a wrong total shows up immediately.
    {
        std::vector<std::vector<double>> columns{{3.5}, {-9.25}, {0.0}};
        const std::vector<std::int32_t> picks{0, 0, 0};
        check(columns, picks, 1, 1, "single-row vocabulary is exactly zero");
    }

    // A near-tie at the top, which is the whole reason the margin is reported: the winner is decided by
    // a fraction of a nat, so any perturbation of that size legitimately swaps it.
    {
        std::vector<std::vector<double>> columns{
            {10.0, 9.999, 1.0, 0.5},
            {10.0, 9.996, 2.0, 0.25},
            {4.0, 3.5, 3.0, 2.5},
            // The two leading values are equal, so the margin is exactly zero and must read as a tie
            // rather than as whichever one the reduction happened to visit last.
            {7.25, 7.25, 0.0, -1.0},
            // The maximum appears more than once: the runner-up is the other copy of it, so the gap is
            // zero even though the top value is strictly greater than everything else in the column.
            {6.5, 6.5, 6.5, 0.0},
            // Descending, so the runner-up is the *second* element and a merge that only tracked the
            // per-thread maximum would report the tail instead.
            {9.0, 8.0, 7.0, 6.0, 5.0, 4.0},
        };
        const std::vector<std::int32_t> picks{0, 0, 0, 0, 0, 0};
        check(columns, picks, 6, 6, "near-ties, equal maxima, descending order");
    }

    // Two valid rows out of many, so the runner-up is found in a domain where most values are padding.
    {
        constexpr std::int32_t kRows  = 1031;
        constexpr std::int32_t kValid = 2;
        std::vector<std::vector<double>> columns{{12.0, 4.0}, {5.0, 5.0}, {-1.0, 8.0}};
        const std::vector<std::int32_t> picks{0, 1, 1};
        check(columns, picks, kRows, kValid, "two valid rows, margin excludes padding");
    }

    // A tiny vocabulary, so the block reduction combines mostly empty partials.
    {
        std::vector<std::vector<double>> columns{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}};
        const std::vector<std::int32_t> picks{6};
        check(columns, picks, 7, 7, "vocabulary smaller than one block");
    }

    // Structural rejections, so a malformed request cannot be mistaken for a computed zero.
    if (failures == 0) {
        std::vector<std::vector<double>> columns{{0.0, 1.0, 2.0}};
        Columns state(columns, 3);
        const std::int32_t pick = 1;
        CUDA_CHECK(cudaMemcpy(state.device_chosen, &pick, sizeof(pick), cudaMemcpyHostToDevice));

        ninfer::Tensor good_out   = state.out();
        ninfer::Tensor good_margin = state.margin();
        ninfer::Tensor bad_logits = state.logits();
        bad_logits.ne[1]          = 2;

        expect_throw([&] {
            ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), 0, good_out, good_margin,
                                           nullptr);
        }, "valid_rows == 0");
        expect_throw([&] {
            ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), 4, good_out, good_margin,
                                           nullptr);
        }, "valid_rows past the physical rows");
        // Each case must hand the Op a tensor of the *wrong* dtype. Copying a correctly typed view and
        // expecting a rejection would test nothing, which is what the first version of these cases did.
        expect_throw([&] {
            ninfer::Tensor wrong(state.device_out, ninfer::DType::BF16, {1});
            ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), 3, wrong, good_margin,
                                           nullptr);
        }, "out of the wrong dtype");
        expect_throw([&] {
            ninfer::Tensor wrong(state.device_margin, ninfer::DType::BF16, {1});
            ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), 3, good_out, wrong,
                                           nullptr);
        }, "margin of the wrong dtype");
        expect_throw([&] {
            ninfer::Tensor wrong(state.device_chosen, ninfer::DType::FP32, {1});
            ninfer::ops::logprob_of_choice(state.logits(), wrong, 3, good_out, good_margin, nullptr);
        }, "chosen of the wrong dtype");
        expect_throw([&] {
            ninfer::Tensor wrong(state.device_logits, ninfer::DType::FP32, {3, 1});
            ninfer::ops::logprob_of_choice(wrong, state.chosen(), 3, good_out, good_margin, nullptr);
        }, "logits of the wrong dtype");
        expect_throw([&] {
            ninfer::ops::logprob_of_choice(bad_logits, state.chosen(), 3, good_out, good_margin,
                                           nullptr);
        }, "chosen length not matching T");
        expect_throw([&] {
            ninfer::Tensor wrong(state.device_margin, ninfer::DType::FP32, {2});
            ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), 3, good_out, wrong,
                                           nullptr);
        }, "margin length not matching T");
        // The two outputs are written by the same thread in the same kernel, so aliasing them would
        // silently clobber one rather than fail. Pointed at the same storage on purpose here.
        expect_throw([&] {
            ninfer::Tensor aliased(state.device_out, ninfer::DType::FP32, {1});
            ninfer::ops::logprob_of_choice(state.logits(), state.chosen(), 3, good_out, aliased,
                                           nullptr);
        }, "margin aliasing out");

        if (failures == 0) { std::printf("  validation rejections: ok\n"); }
    }

    if (failures != 0) {
        std::printf("logprob_of_choice oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("logprob_of_choice oracle checks passed\n");
    return 0;
}