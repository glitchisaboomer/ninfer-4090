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

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

// Sign convention: delta = dst - src. The kernel applies R(-delta) to Q, which is equivalent to the
// key having been at dst instead of src. This matches the Phase 4a comment in gqa_attention_decode.cuh
// and the rerotate oracle in tests/targets/qwen3_6/test_gqa_kv_rerotate.cpp.
constexpr std::int32_t kDelta = (kDestPage * kPageSize) - (kSourcePage * kPageSize); // = -384

// Helper: run the decode A3 path through the detail:: launcher with the supplied view.
AttentionRun run_decode_attention_detail(const Cache& cache, const std::vector<__nv_bfloat16>& q,
                                         std::int32_t first_position, std::int32_t tokens,
                                         std::uint32_t visible, const PagedKVLayerView& view) {
    DeviceBuffer dq   = upload(q);
    DeviceBuffer dout = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));

    const ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQHeads, DType::BF16, envelope, 1, tokens, tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    Tensor partial_acc(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor partial_m(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor partial_l(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    partial_acc.fill(0);
    partial_m.fill(0);
    partial_l.fill(0);

    detail::gqa_attention_cached_small_t_launch(tq, tp, kAttentionScale, view, envelope,
                                                 partial_acc, partial_m, partial_l, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

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

void check_decode_attention_with_delta(const Scenario& scenario) {
    // 0. Setup assertion: the scenario is only meaningful if the block table is non-identity at
    //    kDestPage. If a future refactor of build_scenario makes the table identity, the negative
    //    control becomes meaningless (zero deltas would be correct).
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq = default_inv_freq();
    const std::int32_t delta = scenario.dst - scenario.src; // = kDelta = -384
    const std::uint32_t visible = static_cast<std::uint32_t>(kQueryBase + kQueryTokens);

    // 1. Build a cache with K at source positions (no physical re-rotation)
    Cache cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    // 2. Build the delta plane: all zeros except kDestPage
    std::vector<std::int32_t> delta_host(kLogicalPages, 0);
    delta_host[kDestPage] = delta;
    DeviceBuffer delta_buf = upload_i32(delta_host);

    // 3. Build the view with the delta plane and freq table attached
    PagedKVLayerView view = cache.view();
    view.rope_deltas = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    view.rope_inv_freq = freq; // std::array<float, 32>

    // 4. Run attention WITH deltas via detail:: launcher (should match reference)
    auto got_relocated = run_decode_attention_detail(cache, scenario.q, kQueryBase, kQueryTokens,
                                                    visible, view);

    // 5. Build reference: direct prefill at destination (K carries dst RoPE)
    Cache reference = make_reference_cache(scenario);
    auto got_reference =
        run_cached_attention(reference, scenario.q, kQueryBase, kQueryTokens, visible);

    // 5b. Anchor the reference itself to the FP64 ideal attention oracle. If this fails, the bug is
    //     in the scenario/reference construction, not the delta machinery.
    expect_conformant(
        measure(got_reference.out, ideal_attention(scenario.q, reference, kQueryBase, kQueryTokens)),
        "A3 over a directly prefilled page (anchors the harness)");

    // 6. Assert equivalence between delta-path and direct-prefill reference
    expect_equivalent(measure(got_relocated.out, got_reference.out),
                      "A3 with delta plane vs direct prefill");

    // 7. NEGATIVE CONTROL: run with all-zero deltas (kernel applies no rotation)
    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    view.rope_deltas = Tensor(zero_buf.p, DType::I32, {kLogicalPages});
    auto got_stale = run_decode_attention_detail(cache, scenario.q, kQueryBase, kQueryTokens, visible,
                                                 view);

    // 8. Assert the control FAILS loudly
    expect_control_separates(
        measure(got_relocated.out, got_reference.out),
        measure(got_stale.out, got_reference.out),
        "negative control: zero deltas on decode path");
}

void check_prefill_attention_with_delta(const Scenario& scenario) {
    // 0. Setup assertion (identical to decode)
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq = default_inv_freq();
    const std::int32_t delta = scenario.dst - scenario.src; // = kDelta = -384

    // Prefill path uses kPrefillTokens = 128 (two full Br tiles) so the call reaches the prefill
    // kernel rather than the small-T decode kernel.
    constexpr std::int32_t kPrefillTokens = 128;
    constexpr std::int32_t kPrefillBase = kDestPage * kPageSize;
    const std::uint32_t visible =
        static_cast<std::uint32_t>(kPrefillBase + kPrefillTokens);

    // 1. Build a cache with K at source positions (no physical re-rotation)
    Cache cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    // 2. Build the delta plane: all zeros except kDestPage
    std::vector<std::int32_t> delta_host(kLogicalPages, 0);
    delta_host[kDestPage] = delta;
    DeviceBuffer delta_buf = upload_i32(delta_host);

    // 3. Build the view with the delta plane and freq table attached.
    //    Prefill path uses the batch view. The delta plane is one row of the {logical_pages, T} table
    //    the prefill path reads for this single-sequence case.
    PagedKVLayerView layer_view = cache.view();
    layer_view.rope_deltas = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    layer_view.rope_inv_freq = freq;

    // Prefill path reads deltas through the batch metadata that parallels block_tables.
    // For a single sequence that is a {logical_pages, 1} slice of the same delta plane.
    PagedKVBatchLayerView prefill_batch_view = prefill_cache.batch_view();
    prefill_batch_view.k_pages = prefill_layer_view.k_pages;
    prefill_batch_view.v_pages = prefill_layer_view.v_pages;
    prefill_batch_view.k_scale_pages = prefill_layer_view.k_scale_pages;
    prefill_batch_view.v_scale_pages = prefill_layer_view.v_scale_pages;
    prefill_batch_view.block_tables = prefill_layer_view.block_table.view({kLogicalPages, 1});
    prefill_batch_view.head_dim = prefill_layer_view.head_dim;
    prefill_batch_view.num_kv_heads = prefill_layer_view.num_kv_heads;
    prefill_batch_view.dtype = prefill_layer_view.dtype;
    prefill_batch_view.quant_group = prefill_layer_view.quant_group;
    prefill_batch_view.packed_v = prefill_layer_view.packed_v;
    prefill_batch_view.rotate_k = prefill_layer_view.rotate_k;
    prefill_batch_view.rotate_v = prefill_layer_view.rotate_v;
    prefill_batch_view.packed_k = prefill_layer_view.packed_k;
    prefill_batch_view.e8_lattice = prefill_layer_view.e8_lattice;
    prefill_batch_view.e8_root = prefill_layer_view.e8_root;
    // Prefill path carries its own delta-table pointer in the metadata the launcher builds from the
    // batch view. Mirror the delta plane there as a single-row slice.
    prefill_batch_view.rope_deltas = prefill_layer_view.rope_deltas.view({kLogicalPages, 1});
    prefill_batch_view.rope_inv_freq = prefill_layer_view.rope_inv_freq;

    // 4. Run prefill attention WITH deltas via detail:: launcher
    //    Prefill K/V are the same as the source run's tail so the prefill kernel writes into the same
    //    relocated page region. Use a distinct tail chunk so the call exercises the fill + attention
    //    path end to end.
    const std::size_t tail_kv =
        static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(kPrefillTokens);
    std::mt19937 rng(991u);
    std::uniform_real_distribution<double> unit(-0.25, 0.25);
    std::vector<__nv_bfloat16> tail_k(tail_kv);
    std::vector<__nv_bfloat16> tail_v(tail_kv);
    for (std::size_t i = 0; i < tail_kv; ++i) {
        tail_k[i] = __float2bfloat16(static_cast<float>(unit(rng)));
        tail_v[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }

    Cache prefill_cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(prefill_cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    PagedKVLayerView prefill_layer_view = prefill_cache.view();
    prefill_layer_view.rope_deltas = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    prefill_layer_view.rope_inv_freq = freq;

    prefill_batch_view = prefill_cache.batch_view();
    prefill_batch_view.k_pages = prefill_layer_view.k_pages;
    prefill_batch_view.v_pages = prefill_layer_view.v_pages;
    prefill_batch_view.k_scale_pages = prefill_layer_view.k_scale_pages;
    prefill_batch_view.v_scale_pages = prefill_layer_view.v_scale_pages;
    prefill_batch_view.block_tables = prefill_layer_view.block_table.view({kLogicalPages, 1});
    prefill_batch_view.head_dim = prefill_layer_view.head_dim;
    prefill_batch_view.num_kv_heads = prefill_layer_view.num_kv_heads;
    prefill_batch_view.dtype = prefill_layer_view.dtype;
    prefill_batch_view.quant_group = prefill_layer_view.quant_group;
    prefill_batch_view.packed_v = prefill_layer_view.packed_v;
    prefill_batch_view.rotate_k = prefill_layer_view.rotate_k;
    prefill_batch_view.rotate_v = prefill_layer_view.rotate_v;
    prefill_batch_view.packed_k = prefill_layer_view.packed_k;
    prefill_batch_view.e8_lattice = prefill_layer_view.e8_lattice;
    prefill_batch_view.e8_root = prefill_layer_view.e8_root;
    prefill_batch_view.rope_deltas = prefill_layer_view.rope_deltas.view({kLogicalPages, 1});
    prefill_batch_view.rope_inv_freq = prefill_layer_view.rope_inv_freq;

    auto got_relocated = run_prefill_attention_detail(prefill_cache, scenario.q, tail_k, tail_v,
                                                      kPrefillBase, kPrefillTokens, visible,
                                                      prefill_batch_view);

    // 5. Build reference: direct prefill at destination (K carries dst RoPE)
    Cache reference = make_reference_cache(scenario);
    append_chunk(reference, tail_k, tail_v, kPrefillBase, kPrefillTokens);
    auto got_reference =
        run_cached_attention(reference, scenario.q, kQueryBase, kQueryTokens, visible);

    // 5b. Anchor the reference itself to the FP64 ideal attention oracle.
    expect_conformant(
        measure(got_reference.out, ideal_attention(scenario.q, reference, kQueryBase, kQueryTokens)),
        "A1 over a directly prefilled page (anchors the harness)");

    // 6. Assert equivalence between delta-path and direct-prefill reference
    expect_equivalent(measure(got_relocated.out, got_reference.out),
                      "A1 with delta plane vs direct prefill");

    // 7. NEGATIVE CONTROL (prefill-specific): zero deltas on the prefill path. The prefill kernel has
    //    its own reload_and_rotate_q lambda with its own __syncthreads structure, so a bug in the
    //    prefill reload path would not be caught by a decode-only control.
    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);

    Cache stale_prefill_cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(stale_prefill_cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    PagedKVLayerView stale_layer_view = stale_prefill_cache.view();
    stale_layer_view.rope_deltas = Tensor(zero_buf.p, DType::I32, {kLogicalPages});
    stale_layer_view.rope_inv_freq = freq;

    PagedKVBatchLayerView stale_batch_view = stale_prefill_cache.batch_view();
    stale_batch_view.k_pages = stale_layer_view.k_pages;
    stale_batch_view.v_pages = stale_layer_view.v_pages;
    stale_batch_view.k_scale_pages = stale_layer_view.k_scale_pages;
    stale_batch_view.v_scale_pages = stale_layer_view.v_scale_pages;
    stale_batch_view.block_tables = stale_layer_view.block_table.view({kLogicalPages, 1});
    stale_batch_view.head_dim = stale_layer_view.head_dim;
    stale_batch_view.num_kv_heads = stale_layer_view.num_kv_heads;
    stale_batch_view.dtype = stale_layer_view.dtype;
    stale_batch_view.quant_group = stale_layer_view.quant_group;
    stale_batch_view.packed_v = stale_layer_view.packed_v;
    stale_batch_view.rotate_k = stale_layer_view.rotate_k;
    stale_batch_view.rotate_v = stale_layer_view.rotate_v;
    stale_batch_view.packed_k = stale_layer_view.packed_k;
    stale_batch_view.e8_lattice = stale_layer_view.e8_lattice;
    stale_batch_view.e8_root = stale_layer_view.e8_root;
    stale_batch_view.rope_deltas = stale_layer_view.rope_deltas.view({kLogicalPages, 1});
    stale_batch_view.rope_inv_freq = stale_layer_view.rope_inv_freq;

    auto got_stale = run_prefill_attention_detail(stale_prefill_cache, scenario.q, tail_k, tail_v,
                                                  kPrefillBase, kPrefillTokens, visible,
                                                  stale_batch_view);

    // 8. Assert the control FAILS loudly
    expect_control_separates(
        measure(got_relocated.out, got_reference.out),
        measure(got_stale.out, got_reference.out),
        "negative control: zero deltas on prefill path");
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }

    const FreqTable freq = default_inv_freq();
    const Scenario scenario = build_scenario(freq);

    std::printf("GQA KV page-delta oracle checks\n");
    std::printf("  run of %d tokens: src %d -> dst %d, delta %d, block table permuted\n", kRelocatedRun,
                scenario.src, scenario.dst, scenario.dst - scenario.src);

    check_decode_attention_with_delta(scenario);
    check_prefill_attention_with_delta(scenario);

    if (failures != 0) {
        std::printf("GQA KV page-delta oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("GQA KV page-delta oracle checks passed\n");
    return 0;
}
