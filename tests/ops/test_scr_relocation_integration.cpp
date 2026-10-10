// Integration test for Suffix Cache Reuse relocation: a re-rotated page must be indistinguishable
// from a page a direct prefill at the destination would have written, judged by the real attention
// Op reading the real cache.
//
// The component test (tests/targets/qwen3_6/test_gqa_kv_rerotate.cpp) seeds the K plane by hand using
// paged_kv_element_offset, the same addressing the Op under test uses. That makes it a test of the
// rotation arithmetic *given* an addressing convention; it cannot detect the Op and the cache writer
// disagreeing about that convention. This file closes that gap: every byte of cache content is
// written by ops::gqa_kv_append, and every read goes through ops::gqa_attention_cached or
// ops::gqa_attention -- kernels that were not written with the Op in mind.
//
// Method:
//   1. One random raw K row per token, per KV head.
//   2. K_source    = bf16(rope(raw, src + t))  appended at positions [src, src+T)
//      K_reference = bf16(rope(raw, dst + t))  appended at positions [dst, dst+T)
//      V           = identical in both, so K is the only thing that differs.
//   3. A correct re-rotation by delta = dst - src makes the two caches equivalent, so the assertion is
//      that attention over the relocated cache is indistinguishable from attention over the directly
//      prefilled one. A re-rotation applied twice, or to the wrong pages, changes the K rows and shows
//      up in that comparison.
//   4. The directly prefilled cache is additionally held to the FP64 attention oracle at the published
//      conformance criterion, which anchors the harness itself: a failure there means the reader, the
//      oracle, or this test's cache construction disagree, independently of any relocation.
//   5. The block table is a non-identity permutation and the destination logical page is re-pointed at
//      the source's physical page before the append, which is exactly what reassign_pages produces.
//      The Op therefore has to resolve the page through the table rather than reuse a page number.
//   6. A negative control runs the same comparison with the re-rotation skipped and requires the error
//      to clear the pass limit by a wide factor, so a blind harness reports itself instead of passing.
//
// If this file passes, the relocation Op is consistent with the production write path and the
// production read path, and any remaining SCR failure lies in how the runtime sequences the two.
//
// Shared scenario, oracle, and threshold definitions live in gqa_relocation_test_common.h so this test
// and the delta-plane oracle test cannot drift.

#include "ops/gqa_relocation_test_common.h"
#include "ninfer/ops/gqa_kv_rerotate.h"

using namespace ninfer;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

// 1. The Op's addressing must agree with the production write path. The component test cannot see
//    this, because it seeds and reads through the Op's own helper.
void check_layout_agreement(const Scenario& scenario) {
    const FreqTable freq = default_inv_freq();
    Cache cache           = make_reference_cache(scenario);
    const Cache::HostImage img = cache.image();

    std::array<double, kHeadDim> raw{};
    std::array<double, kHeadDim> oracle{};
    double worst                 = 0.0;
    std::int32_t worst_head      = -1;
    std::int32_t worst_position = -1;
    for (std::int32_t t = 0; t < kRelocatedRun; ++t) {
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                raw[static_cast<std::size_t>(d)] = static_cast<double>(__bfloat162float(
                    scenario.raw_k[act_index(head, t, kKvHeads, d)]));
            }
            forward_rope(raw.data(), scenario.dst + t, freq, oracle.data());
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::int32_t pair = d < kRotaryPairs ? d : d - kRotaryPairs;
                const double scale =
                    std::max(std::fabs(raw[static_cast<std::size_t>(pair)]),
                             std::fabs(raw[static_cast<std::size_t>(pair + kRotaryPairs)]));
                const double ulp = bf16_ulp(scale == 0.0 ? 1.0 : scale);
                const double err =
                    std::fabs(cache.k_logical(img, scenario.dst + t, head, d) -
                              oracle[static_cast<std::size_t>(d)]) /
                    ulp;
                if (err > worst) {
                    worst           = err;
                    worst_head      = head;
                    worst_position = scenario.dst + t;
                }
            }
        }
    }
    // The row was written as bf16(rope(.)), so it reproduces the FP64 oracle exactly up to the host
    // rounding chain fp64 -> fp32 -> bf16. That chain reaches exactly 1.0 operand-BF16 ulp in the
    // worst case, where the rotation lands its result on twice the operand magnitude, so a limit at
    // 1.0 would fail on correct data. A layout disagreement is not a perturbation at all: it permutes
    // the rotary pairs, which costs 505 ulp on the same data. 4.0 keeps the separation the component
    // test uses while clearing the rounding chain.
    constexpr double kLayoutUlpLimit = 4.0;
    if (worst > kLayoutUlpLimit) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "write-path layout: gqa_kv_append stores a row %.2f BF16 ulp from the FP64 RoPE "
                      "oracle (head %d, position %d), over the %.1f limit, so gqa_kv_rerotate's "
                      "addressing convention is not the cache's",
                      worst, worst_head, worst_position, kLayoutUlpLimit);
        fail(buf);
    } else {
        std::printf("  write-path layout: ok (worst %.2f BF16 ulp, limit %.1f)\n", worst,
                    kLayoutUlpLimit);
    }
}

// 2. The Op must rotate exactly the destination physical page, and nothing else.
void check_targeting(const Scenario& scenario, std::int32_t delta) {
    const FreqTable freq = default_inv_freq();
    Cache cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    const std::vector<std::uint16_t> k_before = cache.k_plane();
    const std::vector<std::uint16_t> v_before = cache.v_plane();

    ops::gqa_kv_rerotate_run(scenario.dst, kRelocatedRun, delta,
                             std::span<const float, kRotaryPairs>(freq), cache.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    const std::vector<std::uint16_t> k_after = cache.k_plane();
    const std::vector<std::uint16_t> v_after = cache.v_plane();

    const int before_failures                 = failures;
    const std::int32_t expected_physical      = scenario.table[static_cast<std::size_t>(kDestPage)];
    const std::size_t span                    = cache.elements_per_page();
    int changed_pages                         = 0;
    for (std::int32_t page = 0; page < cache.physical_pages; ++page) {
        const std::size_t base = span * static_cast<std::size_t>(page);
        bool changed           = false;
        for (std::size_t i = base; i < base + span; ++i) {
            if (k_before[i] != k_after[i]) {
                changed = true;
                break;
            }
        }
        if (!changed) { continue; }
        ++changed_pages;
        if (page != expected_physical) {
            fail("targeting: physical page " + std::to_string(page) +
                 " was rotated, but the destination logical page resolves to " +
                 std::to_string(expected_physical));
        }
    }
    if (changed_pages != 1) {
        fail("targeting: " + std::to_string(changed_pages) +
             " physical pages changed, expected exactly the destination page " +
             std::to_string(expected_physical));
    }
    if (v_before != v_after) {
        fail("targeting: the Op modified the V plane; re-rotation must leave V byte-identical");
    }
    if (failures == before_failures) {
        std::printf("  targeting: ok (only physical page %d rotated, V untouched)\n",
                    expected_physical);
    }
}

// 3. The relocated cache must be operationally indistinguishable from a directly prefilled one when
//    read by A3. This is the assertion the component test cannot make: it reads the plane the way the
//    Op writes it and never asks whether the attention kernel agrees.
void check_attention_equivalence(const Scenario& scenario) {
    const FreqTable freq     = default_inv_freq();
    const std::int32_t delta = scenario.dst - scenario.src;
    const std::uint32_t visible = static_cast<std::uint32_t>(kQueryBase + kQueryTokens);

    Cache relocated(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(relocated, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);
    ops::gqa_kv_rerotate_run(scenario.dst, kRelocatedRun, delta,
                             std::span<const float, kRotaryPairs>(freq), relocated.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    Cache reference = make_reference_cache(scenario);

    const AttentionRun got_reference =
        run_cached_attention(reference, scenario.q, kQueryBase, kQueryTokens, visible);
    const AttentionRun got_relocated =
        run_cached_attention(relocated, scenario.q, kQueryBase, kQueryTokens, visible);

    // Anchors the harness: the directly prefilled cache is plain prefill, so the conformance criterion
    // applies to it unchanged and a failure here means the reader, the oracle, or this test's cache
    // construction disagree, independently of any relocation.
    if (!expect_conformant(measure(got_reference.out, ideal_attention(scenario.q, reference, kQueryBase,
                                                                kQueryTokens)),
                      "A3 over a directly prefilled page")) {
        ++failures;
    }

    const Error correct = measure(got_relocated.out, got_reference.out);
    if (!expect_equivalent(correct, "A3 over a relocated page")) {
        ++failures;
    }

    // Negative control: a skipped re-rotation is the failure this file exists to catch. The K then
    // still carries the source positions' rotation.
    Cache stale(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(stale, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);
    ops::gqa_kv_rerotate_run(scenario.dst, kRelocatedRun, 0,
                             std::span<const float, kRotaryPairs>(freq), stale.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const AttentionRun got_stale =
        run_cached_attention(stale, scenario.q, kQueryBase, kQueryTokens, visible);
    if (!expect_control_separates(correct, measure(got_stale.out, got_reference.out),
                             "negative control: a skipped re-rotation")) {
        ++failures;
    }
}

// 4. The same check through A1, which is the append-and-attend route the prefill itself uses, so the
//    relocated page is read by the same kernel that wrote the chunk sitting after it.
void check_append_attention_equivalence(const Scenario& scenario) {
    const FreqTable freq     = default_inv_freq();
    const std::int32_t delta = scenario.dst - scenario.src;
    const std::int32_t tail_first = kDestPage * kPageSize + kRelocatedRun;
    constexpr std::int32_t kTailTokens = 8;
    const std::uint32_t visible = static_cast<std::uint32_t>(tail_first + kTailTokens);

    const std::size_t tail_kv =
        static_cast<std::size_t>(kHeadDim) * kKvHeads * kTailTokens;
    const std::size_t tail_q =
        static_cast<std::size_t>(kHeadDim) * kQHeads * kTailTokens;
    std::mt19937 rng(991u);
    std::uniform_real_distribution<double> unit(-0.25, 0.25);
    std::vector<__nv_bfloat16> tail_k(tail_kv);
    std::vector<__nv_bfloat16> tail_v(tail_kv);
    std::vector<__nv_bfloat16> tail_q_data(tail_q);
    for (std::size_t i = 0; i < tail_kv; ++i) {
        tail_k[i] = __float2bfloat16(static_cast<float>(unit(rng)));
        tail_v[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }
    for (std::size_t i = 0; i < tail_q; ++i) {
        tail_q_data[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }

    Cache relocated(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(relocated, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);
    ops::gqa_kv_rerotate_run(scenario.dst, kRelocatedRun, delta,
                             std::span<const float, kRotaryPairs>(freq), relocated.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    Cache reference = make_reference_cache(scenario);

    const AttentionRun got_relocated =
        run_append_attention(relocated, tail_q_data, tail_k, tail_v, tail_first, kTailTokens, visible);
    const AttentionRun got_reference =
        run_append_attention(reference, tail_q_data, tail_k, tail_v, tail_first, kTailTokens, visible);

    if (!expect_conformant(measure(got_reference.out,
                              ideal_attention(tail_q_data, reference, tail_first, kTailTokens)),
                      "A1 over a directly prefilled page")) {
        ++failures;
    }
    const Error correct = measure(got_relocated.out, got_reference.out);
    if (!expect_equivalent(correct, "A1 over a relocated page")) {
        ++failures;
    }

    Cache stale(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(stale, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);
    ops::gqa_kv_rerotate_run(scenario.dst, kRelocatedRun, 0,
                             std::span<const float, kRotaryPairs>(freq), stale.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const AttentionRun got_stale =
        run_append_attention(stale, tail_q_data, tail_k, tail_v, tail_first, kTailTokens, visible);
    if (!expect_control_separates(correct, measure(got_stale.out, got_reference.out),
                             "negative control: a skipped re-rotation through A1")) {
        ++failures;
    }
}


} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }

    const FreqTable freq = default_inv_freq();
    const Scenario scenario = build_scenario(freq);

    std::printf("SCR relocation integration checks\n");
    std::printf("  run of %d tokens: src %d -> dst %d, delta %d, block table permuted\n",
                kRelocatedRun, scenario.src, scenario.dst, scenario.dst - scenario.src);

    check_layout_agreement(scenario);
    check_targeting(scenario, scenario.dst - scenario.src);
    check_attention_equivalence(scenario);
    check_append_attention_equivalence(scenario);

    if (failures != 0) {
        std::printf("SCR relocation integration checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("SCR relocation integration checks passed\n");
    return 0;
}