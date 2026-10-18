// Oracle test for the per-page RoPE delta machinery: a cache whose K carries source-position RoPE,
// read through attention kernels that receive a correct delta plane, produces output indistinguishable
// from a cache whose K was directly prefilled at the destination positions.
//
// This test calls detail:: launchers directly with a real delta plane. The production path
// (public API -> detail:: launcher) still passes nullptr until Phase 4c; the test's own negative
// control (all-zero deltas on the relocated scenario) is what proves the harness is not blind.
//
// This test and test_scr_relocation_integration.cpp use the same scenario, the same helpers, and the
// same FP64 anchor. If the integration test passes and this one fails, the bug is in the kernel's Q
// counter-rotation machinery. If both fail, the bug is in the scenario construction or the FP64
// reference. This separation is the reason both tests exist.
#include "ops/gqa_relocation_test_common.h"
#include "ops/launcher/gqa_attention_test_hooks.h"

using namespace ninfer;
using namespace ninfer::ops;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

// The detail::gqa_attention_cached_small_t_launch dispatch only accepts TOKENS in [1, 8].
// 16 would hit the default branch and throw. 8 is the largest token count the small-t
// bf16 kernel supports, and it is enough to exercise the Q counter-rotation path.
constexpr std::int32_t kDecodeTokens = 8;

// Helper: run the prefill A1 path through the detail:: launcher with the supplied batch view. This is
// the append-and-attend route, so the cache is mutated by the call; the test builds a fresh cache for
// each run.
AttentionRun run_prefill_attention_detail(Cache& cache, const std::vector<__nv_bfloat16>& q,
                                          const std::vector<__nv_bfloat16>& k,
                                          const std::vector<__nv_bfloat16>& v,
                                          std::int32_t first_position, std::int32_t tokens,
                                          std::uint32_t visible, const PagedKVBatchLayerView& batch_view) {
    DeviceBuffer dq = upload(q);
    DeviceBuffer dk = upload(k);
    DeviceBuffer dv = upload(v);
    DeviceBuffer dtable = DeviceBuffer(sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable.copy_from_host(&table_row, sizeof(table_row));
    DeviceBuffer dout = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor ttable(dtable.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    detail::gqa_attention_prompt_launch(tq, tk, tv, tp, Tensor{}, ttable, kAttentionScale,
                                        batch_view, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

void check_decode_attention_with_delta(const Scenario& scenario,
                                           const Criterion& anchor = kConformanceBf16) {

    // 0. Setup assertion: the destination page table entry is non-identity, so the zero-delta
    //    negative control remains meaningful for both aligned and mixed-page scenarios.
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq = default_inv_freq();
    const std::int32_t decode_base = scenario.dst + scenario.run - kDecodeTokens;
    const std::uint32_t visible = static_cast<std::uint32_t>(decode_base + kDecodeTokens);

    // 1. Build a cache with K at source positions (no physical re-rotation)

    Cache cache(kLogicalPages, scenario.table, kKvHeads);
    place_unaligned_relocation(cache, scenario);


    // 2. Build page deltas and packed interval bounds for every destination page touched.
    DeltaMetadata delta_metadata = make_delta_metadata(scenario);
    DeviceBuffer delta_buf = upload_i32(delta_metadata.deltas);
    DeviceBuffer range_buf(delta_metadata.ranges.size() * sizeof(std::uint32_t));
    range_buf.copy_from_host(delta_metadata.ranges.data(), delta_metadata.ranges.size() * sizeof(std::uint32_t));

    // 3. Build the view with the delta plane and freq table attached
    PagedKVLayerView view = cache.view();
    view.rope_deltas = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {kLogicalPages});
    view.rope_inv_freq = freq; // std::array<float, 32>


    // 4. Run attention WITH deltas via detail:: launcher (should match reference)
    auto got_relocated = run_cached_attention_view(cache, scenario.q, decode_base, kDecodeTokens,
                                                   visible, view);


    // 5. Build reference: direct prefill at destination (K carries dst RoPE)
    Cache reference = make_reference_cache(scenario);

    auto got_reference =
        run_cached_attention(reference, scenario.q, decode_base, kDecodeTokens, visible);


    // 5b. Anchor the reference itself to the FP64 ideal attention oracle. If this fails, the bug is
    //     in the scenario/reference construction, not the delta machinery.

    if (!expect_conformant(
            measure(got_reference.out,
                    ideal_attention(scenario.q, reference, decode_base, kDecodeTokens), anchor),
            "A3 over a directly prefilled page (anchors the harness)", anchor)) {
        ++failures;
    }

    // 6. Assert equivalence between delta-path and direct-prefill reference
    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           "A3 with delta plane vs direct prefill")) {
        ++failures;
    }

    // 7. NEGATIVE CONTROL: run with all-zero deltas (kernel applies no rotation)
    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    view.rope_deltas = Tensor(zero_buf.p, DType::I32, {kLogicalPages});
    auto got_stale = run_cached_attention_view(cache, scenario.q, decode_base, kDecodeTokens, visible,
                                               view);

    // 8. Assert the control FAILS loudly
    if (!expect_control_separates(
            measure(got_relocated.out, got_reference.out),
            measure(got_stale.out, got_reference.out),
            "negative control: zero deltas on decode path")) {
        ++failures;
    }
}

void check_prefill_attention_with_delta(const Scenario& scenario) {
    // 0. Setup assertion (identical to decode)
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq      = default_inv_freq();
    constexpr std::int32_t kPrefillTokens = 128;
    const std::int32_t prefill_base = scenario.dst + scenario.run;
    const std::uint32_t visible =
        static_cast<std::uint32_t>(prefill_base + kPrefillTokens);

    // Prefill path uses a Q sized for kPrefillTokens, not kQueryTokens. The reference must use
    // the same Q at the same positions, or the comparison is meaningless.
    const std::size_t prefill_q_elems = static_cast<std::size_t>(kHeadDim) *
                                        static_cast<std::size_t>(kQHeads) *
                                        static_cast<std::size_t>(kPrefillTokens);
    std::vector<__nv_bfloat16> prefill_q(prefill_q_elems);
    {
        std::mt19937 q_rng(17u);
        std::uniform_real_distribution<double> q_unit(-0.25, 0.25);
        for (std::size_t i = 0; i < prefill_q_elems; ++i) {
            prefill_q[i] = __float2bfloat16(static_cast<float>(q_unit(q_rng)));
        }
    }

    // 1. Build a cache with K at source positions (no physical re-rotation)
    Cache cache(kLogicalPages, scenario.table, kKvHeads);
    place_unaligned_relocation(cache, scenario);

    // 2. Build page deltas and packed interval bounds for every destination page touched.
    DeltaMetadata delta_metadata = make_delta_metadata(scenario);
    DeviceBuffer delta_buf = upload_i32(delta_metadata.deltas);
    DeviceBuffer range_buf(delta_metadata.ranges.size() * sizeof(std::uint32_t));
    range_buf.copy_from_host(delta_metadata.ranges.data(),
                             delta_metadata.ranges.size() * sizeof(std::uint32_t));

    // 3. Build the batch view with the delta plane and freq table attached
    PagedKVLayerView layer_view = cache.view();
    layer_view.rope_deltas  = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    layer_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {kLogicalPages});
    layer_view.rope_inv_freq = freq;

    PagedKVBatchLayerView prefill_batch_view = cache.batch_view();
    prefill_batch_view.rope_deltas  = layer_view.rope_deltas.view({kLogicalPages, 1});
    prefill_batch_view.rope_delta_ranges =
        layer_view.rope_delta_ranges.view({kLogicalPages, 1});
    prefill_batch_view.rope_inv_freq = layer_view.rope_inv_freq;

    // 4. Build the tail K/V chunk the prefill call writes at kPrefillBase
    const std::size_t tail_kv = static_cast<std::size_t>(kHeadDim) *
                                static_cast<std::size_t>(kKvHeads) *
                                static_cast<std::size_t>(kPrefillTokens);
    std::mt19937 rng(991u);
    std::uniform_real_distribution<double> unit(-0.25, 0.25);
    std::vector<__nv_bfloat16> tail_k(tail_kv);
    std::vector<__nv_bfloat16> tail_v(tail_kv);
    for (std::size_t i = 0; i < tail_kv; ++i) {
        tail_k[i] = __float2bfloat16(static_cast<float>(unit(rng)));
        tail_v[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }

    // 5. Run prefill attention WITH deltas
    auto got_relocated = run_prefill_attention_detail(cache, prefill_q, tail_k, tail_v,
                                                      prefill_base, kPrefillTokens, visible,
                                                      prefill_batch_view);

    // 6. Reference: direct prefill at the destination, then the same tail chunk
    Cache reference = make_reference_cache(scenario);
    append_chunk(reference, tail_k, tail_v, prefill_base, kPrefillTokens);
    auto got_reference =
        run_cached_attention(reference, prefill_q, prefill_base, kPrefillTokens, visible);

    // 6b. Anchor the reference to the FP64 oracle
    if (!expect_conformant(
            measure(got_reference.out,
                    ideal_attention(prefill_q, reference, prefill_base, kPrefillTokens)),
            "A1 over a directly prefilled page (anchors the harness)")) {
        ++failures;
    }

    // 7. Compare delta-path against reference
    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           "A1 with delta plane vs direct prefill")) {
        ++failures;
    }

    // 8. NEGATIVE CONTROL: zero deltas on a second fresh cache
    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);

    Cache stale_cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(stale_cache, scenario.k_src, scenario.v, scenario.src, scenario.run);

    PagedKVLayerView stale_layer_view = stale_cache.view();
    stale_layer_view.rope_deltas  = Tensor(zero_buf.p, DType::I32, {kLogicalPages});
    stale_layer_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {kLogicalPages});
    stale_layer_view.rope_inv_freq = freq;

    PagedKVBatchLayerView stale_batch_view = stale_cache.batch_view();
    stale_batch_view.rope_deltas  = stale_layer_view.rope_deltas.view({kLogicalPages, 1});
    stale_batch_view.rope_delta_ranges =
        stale_layer_view.rope_delta_ranges.view({kLogicalPages, 1});
    stale_batch_view.rope_inv_freq = stale_layer_view.rope_inv_freq;

    auto got_stale = run_prefill_attention_detail(stale_cache, prefill_q, tail_k, tail_v,
                                                  prefill_base, kPrefillTokens, visible,
                                                  stale_batch_view);

    if (!expect_control_separates(measure(got_relocated.out, got_reference.out),
                                  measure(got_stale.out, got_reference.out),
                                  "negative control: zero deltas on prefill path")) {
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
    const Scenario unaligned_scenario = build_unaligned_scenario(freq);
    // A run longer than a page: whole relocated pages sit between the two partial ends. The BF16
    // prefill path is the control that must keep agreeing with the FP64 oracle on it, which pins any
    // op-level failure of the same shape to the quantized kernel rather than to the fixture.
    const Scenario long_scenario = build_unaligned_long_scenario(freq);

    std::printf("GQA KV page-delta oracle checks\n");
    std::printf("  run of %d tokens: src %d -> dst %d, delta %d, block table permuted\n", kRelocatedRun,
                scenario.src, scenario.dst, scenario.dst - scenario.src);
    std::printf("  long off-grid run: src %d -> dst %d, delta %d, %d tokens over dst pages %d..%d\n",
                long_scenario.src, long_scenario.dst, long_scenario.dst - long_scenario.src,
                long_scenario.run, long_scenario.dst / kPageSize,
                (long_scenario.dst + long_scenario.run - 1) / kPageSize);

    check_decode_attention_with_delta(scenario);
    check_prefill_attention_with_delta(scenario);
    check_decode_attention_with_delta(unaligned_scenario);
    check_prefill_attention_with_delta(unaligned_scenario);
    check_decode_attention_with_delta(long_scenario, kConformanceBf16MultiPage);
    check_prefill_attention_with_delta(long_scenario);

    if (failures != 0) {
        std::printf("GQA KV page-delta oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("GQA KV page-delta oracle checks passed\n");
    return 0;
}
