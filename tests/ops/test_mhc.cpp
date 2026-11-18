#include "ninfer/ops/mhc.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-6f;
constexpr int kIterations = 20;

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::uint16_t round_bf16(float value) { return f32_to_bf16(value); }
float bf16_value(float value) { return bf16_to_f32(round_bf16(value)); }

void logsumexp2(double& a, double& b) {
    const double maximum = std::max(a, b);
    const double shift = maximum + std::log(std::exp(a - maximum) + std::exp(b - maximum));
    a -= shift;
    b -= shift;
}

struct Oracle {
    std::vector<double> branch;
    std::vector<double> post;
    std::vector<double> mix;
};

Oracle mhc_oracle(const std::vector<float>& residual_rounded,
                  const std::vector<float>& weight,
                  const std::vector<float>& bias,
                  int d, int tokens, float pre_scale, bool reversed_stream_stride) {
    Oracle oracle;
    oracle.branch.resize(static_cast<std::size_t>(d) * tokens);
    oracle.post.resize(static_cast<std::size_t>(2) * tokens);
    oracle.mix.resize(static_cast<std::size_t>(4) * tokens);

    for (int token = 0; token < tokens; ++token) {
        std::vector<double> flat(static_cast<std::size_t>(2) * d);
        double sum_squares = 0.0;
        for (int stream = 0; stream < 2; ++stream) {
            for (int column = 0; column < d; ++column) {
                const int logical_stream = reversed_stream_stride ? 1 - stream : stream;
                const std::size_t index = static_cast<std::size_t>(token) * 2 * d +
                                          static_cast<std::size_t>(logical_stream) * d + column;
                const double value = static_cast<double>(residual_rounded[index]);
                flat[static_cast<std::size_t>(stream) * d + column] = value;
                sum_squares += value * value;
            }
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(2 * d) + kEps);
        double logits[8];
        for (int k = 0; k < 8; ++k) {
            double acc = 0.0;
            for (std::size_t i = 0; i < flat.size(); ++i) {
                acc += static_cast<double>(weight[static_cast<std::size_t>(k) * flat.size() + i]) *
                       flat[i];
            }
            logits[k] = acc * inverse + static_cast<double>(bias[k]);
        }
        const double pre0 = bf16_value(static_cast<float>(sigmoidf(static_cast<float>(logits[0]))));
        const double pre1 = bf16_value(static_cast<float>(sigmoidf(static_cast<float>(logits[1]))));
        const double post0 = 2.0 * static_cast<double>(sigmoidf(static_cast<float>(logits[2])));
        const double post1 = 2.0 * static_cast<double>(sigmoidf(static_cast<float>(logits[3])));
        oracle.post[token] = post0;
        oracle.post[tokens + token] = post1;

        double m00 = logits[4];
        double m01 = logits[5];
        double m10 = logits[6];
        double m11 = logits[7];
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            logsumexp2(m00, m01);
            logsumexp2(m10, m11);
            logsumexp2(m00, m10);
            logsumexp2(m01, m11);
        }
        oracle.mix[token * 4] = std::exp(m00);
        oracle.mix[token * 4 + 1] = std::exp(m10);
        oracle.mix[token * 4 + 2] = std::exp(m01);
        oracle.mix[token * 4 + 3] = std::exp(m11);

        for (int column = 0; column < d; ++column) {
            const double value = pre0 * flat[column] + pre1 * flat[d + column];
            oracle.branch[token * d + column] =
                static_cast<double>(bf16_value(static_cast<float>(value * pre_scale)));
        }
    }
    return oracle;
}

DeviceBuffer make_residual(const std::vector<float>& values, int d, int tokens,
                           bool strided, bool reversed_stream_stride) {
    const std::size_t elements = values.size();
    const std::size_t stride_words = strided ? (2 * d + 32) : (2 * d);
    const std::size_t allocation_words = static_cast<std::size_t>(tokens) * stride_words + 8;
    std::vector<std::uint16_t> staged(allocation_words, 0);
    std::vector<std::uint16_t> words(elements);
    for (std::size_t i = 0; i < elements; ++i) { words[i] = round_bf16(values[i]); }
    if (!strided) {
        std::copy(words.begin(), words.end(), staged.begin());
    } else {
    auto* base = staged.data();
    for (int token = 0; token < tokens; ++token) {
        for (int stream = 0; stream < 2; ++stream) {
            const int logical_stream = reversed_stream_stride ? 1 - stream : stream;
            const std::size_t source = static_cast<std::size_t>(token) * 2 * d +
                                       static_cast<std::size_t>(logical_stream) * d;
            const std::size_t offset = static_cast<std::size_t>(token) * (2 * d + 32) +
                                       static_cast<std::size_t>(stream) * d;
            std::memcpy(base + offset, words.data() + source,
                        d * sizeof(std::uint16_t));
        }
    }
    }
    DeviceBuffer host(allocation_words * sizeof(std::uint16_t));
    host.copy_from_host(staged.data(), staged.size() * sizeof(std::uint16_t));
    return host;
}

Tensor residual_view(DeviceBuffer& buffer, int d, int tokens, bool strided,
                     bool reversed_stream_stride) {
    (void)reversed_stream_stride;
    Tensor tensor(buffer.p, DType::BF16, {d, 2, tokens});
    if (strided) {
        tensor.nb[1] = static_cast<std::int64_t>(d) * sizeof(std::uint16_t);
        tensor.nb[2] = static_cast<std::int64_t>(2 * d + 32) * sizeof(std::uint16_t);
    }
    return tensor;
}

int run_case(const char* label, int d, int tokens, std::uint32_t seed, bool strided,
             bool reversed_stream_stride) {
    const std::size_t residual_count = static_cast<std::size_t>(2) * d * tokens;
    const std::size_t weight_count = static_cast<std::size_t>(8) * 2 * d;
    std::vector<float> residual(residual_count), weight(weight_count), bias(8);
    fill_uniform(residual, seed, -2.0f, 2.0f);
    fill_uniform(weight, seed + 1U, -0.08f, 0.08f);
    fill_uniform(bias, seed + 2U, -0.5f, 0.5f);
    round_to_bf16(residual);

    const Oracle expected = mhc_oracle(residual, weight, bias, d, tokens, 1.0f,
                                       reversed_stream_stride);
    DeviceBuffer d_residual = make_residual(residual, d, tokens, strided, reversed_stream_stride);
    DeviceBuffer d_weight = to_device_f32(weight);
    DeviceBuffer d_bias = to_device_f32(bias);
    DeviceBuffer d_branch(static_cast<std::size_t>(d) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_post(static_cast<std::size_t>(2) * tokens * sizeof(float));
    DeviceBuffer d_mix(static_cast<std::size_t>(4) * tokens * sizeof(float));
    DeviceBuffer d_out(residual_count * sizeof(std::uint16_t));

    Tensor residual_tensor = residual_view(d_residual, d, tokens, strided, reversed_stream_stride);
    Tensor weight_tensor(d_weight.p, DType::FP32, {2 * d, 8});
    Tensor bias_tensor(d_bias.p, DType::FP32, {8});
    Tensor branch_tensor(d_branch.p, DType::BF16, {d, tokens});
    Tensor post_tensor(d_post.p, DType::FP32, {2, tokens});
    Tensor mix_tensor(d_mix.p, DType::FP32, {2, 2, tokens});
    Tensor out_tensor(d_out.p, DType::BF16, {d, 2, tokens});

    ops::mhc_route(residual_tensor, weight_tensor, bias_tensor, 2, kIterations, 1.0f,
                   branch_tensor, post_tensor, mix_tensor, nullptr);
    cuda_synchronize();

    int failures = 0;
    const PointwiseCriterion bf16_criterion{1.0e-4, 3.0e-3};
    const PointwiseCriterion fp32_criterion{2.0e-6, 2.0e-6};
    failures += verify_pointwise(std::string(label) + " branch",
                                 from_device_bf16(d_branch.p, expected.branch.size()),
                                 expected.branch, bf16_criterion);
    failures += verify_pointwise(std::string(label) + " post",
                                 from_device_f32(d_post.p, expected.post.size()),
                                 expected.post, fp32_criterion);
    failures += verify_pointwise(std::string(label) + " mix",
                                 from_device_f32(d_mix.p, expected.mix.size()),
                                 expected.mix, fp32_criterion);

    ops::mhc_merge(residual_tensor, branch_tensor, post_tensor, mix_tensor, 2, out_tensor, nullptr);
    cuda_synchronize();

    std::vector<double> merge_expected(residual_count);
    for (int token = 0; token < tokens; ++token) {
        const float post0 = bf16_value(static_cast<float>(expected.post[token]));
        const float post1 = bf16_value(static_cast<float>(expected.post[tokens + token]));
        const float m00 = bf16_value(static_cast<float>(expected.mix[token * 4]));
        const float m10 = bf16_value(static_cast<float>(expected.mix[token * 4 + 1]));
        const float m01 = bf16_value(static_cast<float>(expected.mix[token * 4 + 2]));
        const float m11 = bf16_value(static_cast<float>(expected.mix[token * 4 + 3]));
        for (int column = 0; column < d; ++column) {
            const int logical0 = reversed_stream_stride ? 1 : 0;
            const int logical1 = reversed_stream_stride ? 0 : 1;
            const float x0 = residual[(static_cast<std::size_t>(token) * 2 + logical0) * d + column];
            const float x1 = residual[(static_cast<std::size_t>(token) * 2 + logical1) * d + column];
            const float branch = static_cast<float>(
                bf16_value(static_cast<float>(expected.branch[token * d + column])));
            merge_expected[token * 2 * d + column] =
                static_cast<double>(bf16_value(m00 * x0 + m01 * x1 + post0 * branch));
            merge_expected[token * 2 * d + d + column] =
                static_cast<double>(bf16_value(m10 * x0 + m11 * x1 + post1 * branch));
        }
    }
    failures += verify_pointwise(std::string(label) + " merge",
                                 from_device_bf16(d_out.p, merge_expected.size()),
                                 merge_expected, bf16_criterion);
    return failures;
}

// The leave-boundary reduction replaces an identity merge (zero branch, zero post, uniform 0.5
// mix). It must be bit-identical to that merge, not merely inside the BF16 tolerance, because the
// 56-visit pass previously produced the merged bytes.
int run_reduce_case(const char* label, int d, int tokens, std::uint32_t seed, bool strided) {
    const std::size_t residual_count = static_cast<std::size_t>(2) * d * tokens;
    std::vector<float> residual(residual_count);
    fill_uniform(residual, seed, -2.0f, 2.0f);
    round_to_bf16(residual);

    DeviceBuffer d_residual = make_residual(residual, d, tokens, strided, false);
    DeviceBuffer d_out(static_cast<std::size_t>(d) * tokens * sizeof(std::uint16_t));
    Tensor residual_tensor = residual_view(d_residual, d, tokens, strided, false);
    Tensor out_tensor(d_out.p, DType::BF16, {d, tokens});

    ops::mhc_reduce(residual_tensor, 2, out_tensor, nullptr);
    cuda_synchronize();

    std::vector<double> expected(static_cast<std::size_t>(d) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < d; ++column) {
            const double x0 = residual[(static_cast<std::size_t>(token) * 2) * d + column];
            const double x1 = residual[(static_cast<std::size_t>(token) * 2 + 1) * d + column];
            expected[static_cast<std::size_t>(token) * d + column] =
                static_cast<double>(bf16_value(static_cast<float>((x0 + x1) / 2.0)));
        }
    }
    const PointwiseCriterion bf16_criterion{1.0e-4, 3.0e-3};
    int failures = verify_pointwise(std::string(label) + " reduce",
                                    from_device_bf16(d_out.p, expected.size()), expected,
                                    bf16_criterion);

    // Independent byte-level parity with the identity merge the Op replaces.
    DeviceBuffer d_mix(static_cast<std::size_t>(4) * tokens * sizeof(float));
    const std::vector<float> half_mix(static_cast<std::size_t>(4) * tokens, 0.5f);
    d_mix.copy_from_host(half_mix.data(), half_mix.size() * sizeof(float));
    DeviceBuffer d_branch(static_cast<std::size_t>(d) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_post(static_cast<std::size_t>(2) * tokens * sizeof(float));
    d_branch.fill(0);
    d_post.fill(0);
    DeviceBuffer d_merge(static_cast<std::size_t>(2) * d * tokens * sizeof(std::uint16_t));
    Tensor branch_tensor(d_branch.p, DType::BF16, {d, tokens});
    Tensor post_tensor(d_post.p, DType::FP32, {2, tokens});
    Tensor mix_tensor(d_mix.p, DType::FP32, {2, 2, tokens});
    Tensor merge_tensor(d_merge.p, DType::BF16, {d, 2, tokens});
    ops::mhc_merge(residual_tensor, branch_tensor, post_tensor, mix_tensor, 2, merge_tensor,
                   nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> reduce_bits(static_cast<std::size_t>(d) * tokens);
    std::vector<std::uint16_t> merge_bits(reduce_bits.size());
    d_out.copy_to_host(reduce_bits.data(), reduce_bits.size() * sizeof(std::uint16_t));
    d_merge.copy_to_host(merge_bits.data(), merge_bits.size() * sizeof(std::uint16_t),
                         static_cast<std::size_t>(d) * tokens * sizeof(std::uint16_t));
    if (reduce_bits != merge_bits) {
        std::cerr << "FAIL: " << label << " reduce is not bit-identical to the identity merge\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("mhc d4096 t1", 4096, 1, 9001U, false, false);
    failures += run_case("mhc d4096 t1 strided", 4096, 1, 9002U, true, false);
    failures += run_case("mhc d4096 t7 strided", 4096, 7, 9003U, true, false);
    failures += run_case("mhc d4096 t512 strided", 4096, 512, 9004U, true, false);
    failures += run_case("mhc d1000 t1 view", 1000, 1, 9005U, true, true);
    failures += run_case("mhc d1000 t7 view", 1000, 7, 9006U, true, true);
    failures += run_case("mhc d1000 t512 view", 1000, 512, 9007U, true, true);
    failures += run_reduce_case("mhc_reduce d4096 t1", 4096, 1, 9101U, false);
    failures += run_reduce_case("mhc_reduce d4096 t1 strided", 4096, 1, 9102U, true);
    failures += run_reduce_case("mhc_reduce d4096 t7 strided", 4096, 7, 9103U, true);
    failures += run_reduce_case("mhc_reduce d1000 t17 view", 1000, 17, 9104U, true);
    std::cout << (failures ? "FAIL" : "OK") << " mhc\n";
    return failures ? 1 : 0;
}
