// Integration test for Suffix Cache Reuse relocation: a delta-plane-relocated
// page must be indistinguishable from a page a direct prefill at the
// destination would have written, judged by the real attention Ops reading
// the real cache.
//
// The oracle test (tests/ops/test_gqa_kv_page_delta.cpp) drives the detail::
// launchers directly with a hand-built delta plane, which makes it a test of
// the counter-rotation arithmetic given an addressing convention; it cannot
// detect the public entry points and the cache writer disagreeing about that
// convention. This file closes that gap: every byte of cache content is
// written by ops::gqa_kv_append, and every read goes through
// ops::gqa_attention_cached or ops::gqa_attention -- kernels that were not
// written with the delta plane in mind.
//
// Method:
//   1. One random raw K row per token, per KV head.
//   2. K_source    = bf16(rope(raw, src + t))  appended at positions [src, src+T)
//      K_reference = bf16(rope(raw, dst + t))  appended at positions [dst, dst+T)
//      V           = identical in both, so K is the only thing that differs.
//   3. A relocation re-points the destination logical page at the source's
//      physical page -- already done to the block table by build_scenario,
//      exactly what reassign_pages publishes -- and writes delta = dst - src
//      for that page on the delta plane. The K bytes stay exactly as the
//      source prefill wrote them; the attention kernels counter-rotate Q per
//      page from the delta plane, so the assertion is that attention over the
//      relocated cache is indistinguishable from attention over the directly
//      prefilled one. A delta plane that names the wrong pages, or zeroes the
//      destination page, changes the scores and shows up in that comparison.
//   4. The directly prefilled cache is additionally held to the FP64 attention
//      oracle at the published conformance criterion, which anchors the harness
//      itself: a failure there means the reader, the oracle, or this test's
//      cache construction disagree, independently of any relocation.
//   5. A negative control runs the same comparison with an all-zero delta
//      plane -- a relocation that published no deltas -- and requires the error
//      to clear the pass limit by a wide factor, so a blind harness reports
//      itself instead of passing.
//
// If this file passes, the relocation delta plane is consistent with the
// production write path and the production read path, and any remaining SCR
// failure lies in how the runtime sequences the two.
//
// Shared scenario, oracle, and threshold definitions live in
// gqa_relocation_test_common.h so this test and the delta-plane oracle test
// cannot drift.

#include "ops/gqa_relocation_test_common.h"

using namespace ninfer;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

// 1. The production write path must store rows the way the FP64 RoPE oracle
//    describes them. The delta-plane reader and the writer have to agree on
//    the cache's addressing convention; the oracle test seeds and reads
//    through its own helpers, so it cannot see a disagreement between
//    ops::gqa_kv_append and the attention kernels. This check anchors the
//    writer directly against the oracle.
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
    // the rotary pairs, which costs 505 ulp on the same data. 4.0 keeps the separation the oracle
    // test uses while clearing the rounding chain.
    constexpr double kLayoutUlpLimit = 4.0;
    if (worst > kLayoutUlpLimit) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "write-path layout: gqa_kv_append stores a row %.2f BF16 ulp from the FP64 RoPE "
                      "oracle (head %d, position %d), over the %.1f limit, so the cache writer's "
                      "addressing convention is not the one the delta-plane reader uses",
                      worst, worst_head, worst_position, kLayoutUlpLimit);
        fail(buf);
    } else {
        std::printf("  write-path layout: ok (worst %.2f BF16 ulp, limit %.1f)\n", worst,
                    kLayoutUlpLimit);
    }
}

// The delta plane a one-page relocation publishes: every logical page is
// ordinary except the destination page, which carries the full dst - src
// shift. This is the plane `PagedKVAllocation::reassign_pages` writes for a
// relocation that moves one page-aligned run.
struct DeltaPlane {
    DeviceBuffer buffer;
};

[[nodiscard]] DeltaPlane make_delta_plane(std::int32_t delta) {
    std::vector<std::int32_t> host(kLogicalPages, 0);
    host[static_cast<std::size_t>(kDestPage)] = delta;
    DeltaPlane plane;
    plane.buffer = upload_i32(host);
    return plane;
}

// A single-sequence layer view with the delta plane and freq table attached,
// the shape `PagedKVCache::layer_view` hands the decode path.
[[nodiscard]] PagedKVLayerView delta_view(const Cache& cache, const DeltaPlane& plane,
                                               const FreqTable& freq) {
    PagedKVLayerView view = cache.view();
    view.rope_deltas   = Tensor(plane.buffer.p, DType::I32, {kLogicalPages});
    view.rope_inv_freq = freq;
    return view;
}

// The batched equivalent, the shape `PagedKVCache::batch_layer_view` hands
// the prefill path: one table row, so the plane is [logical_pages, 1].
[[nodiscard]] PagedKVBatchLayerView delta_batch_view(const Cache& cache,
                                                          const DeltaPlane& plane,
                                                          const FreqTable& freq) {
    PagedKVBatchLayerView batch = cache.batch_view();
    batch.rope_deltas   = Tensor(plane.buffer.p, DType::I32, {kLogicalPages, 1});
    batch.rope_inv_freq = freq;
    return batch;
}

// A1 (append-and-attend) through the public gqa_attention entry point, with
// an explicit batch view so the delta plane and freq table can be attached.
// Mirrors run_append_attention in the shared header.
AttentionRun run_append_attention_view(Cache& cache,
                                          const std::vector<__nv_bfloat16>& q,
                                          const std::vector<__nv_bfloat16>& k,
                                          const std::vector<__nv_bfloat16>& v,
                                          std::int32_t first_position, std::int32_t tokens,
                                          std::uint32_t visible,
                                          const PagedKVBatchLayerView& batch_view) {
    append_chunk(cache, k, v, first_position, tokens);

    DeviceBuffer dq     = upload(q);
    DeviceBuffer dk     = upload(k);
    DeviceBuffer dv     = upload(v);
    DeviceBuffer dtable = DeviceBuffer(sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable.copy_from_host(&table_row, sizeof(table_row));
    DeviceBuffer dout = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));

    const ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQHeads, DType::BF16, envelope, 1,
                                                           tokens, tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor ttable(dtable.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable, kAttentionScale, batch_view,
                       envelope, workspace, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

// 2. A delta-plane relocation must be operationally indistinguishable from a
//    directly prefilled cache when read by A3 through the public entry
//    point. The relocated cache's K still carries the source positions'
//    rotation -- exactly what a surviving page holds after reassign_pages --
//    and the attention kernel counter-rotates Q by the published per-page
//    delta. This is the assertion the oracle test makes on the detail::
//    launcher, exercised here on the public route the served runtime uses.
void check_attention_equivalence(const Scenario& scenario) {
    const FreqTable freq     = default_inv_freq();
    const std::int32_t delta = scenario.dst - scenario.src;
    const std::uint32_t visible = static_cast<std::uint32_t>(kQueryBase + kQueryTokens);

    Cache relocated(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(relocated, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);
    const DeltaPlane plane = make_delta_plane(delta);
    const PagedKVLayerView view = delta_view(relocated, plane, freq);

    Cache reference = make_reference_cache(scenario);

    const AttentionRun got_reference =
        run_cached_attention(reference, scenario.q, kQueryBase, kQueryTokens, visible);
    const AttentionRun got_relocated = run_cached_attention_view(
        relocated, scenario.q, kQueryBase, kQueryTokens, visible, view);

    // Anchors the harness: the directly prefilled cache is plain prefill, so the conformance criterion
    // applies to it unchanged and a failure here means the reader, the oracle, or this test's cache
    // construction disagree, independently of any relocation.
    if (!expect_conformant(measure(got_reference.out, ideal_attention(scenario.q, reference, kQueryBase,
                                                                 kQueryTokens)),
                      "A3 over a directly prefilled page")) {
        ++failures;
    }

    const Error correct = measure(got_relocated.out, got_reference.out);
    if (!expect_equivalent(correct, "A3 over a delta-plane relocated page")) {
        ++failures;
    }

    // Negative control: an all-zero delta plane is a relocation that published
    // no deltas. The K still carries the source positions' rotation, so the
    // scores must diverge by orders of magnitude.
    const DeltaPlane none        = make_delta_plane(0);
    const PagedKVLayerView stale = delta_view(relocated, none, freq);
    const AttentionRun got_stale = run_cached_attention_view(
        relocated, scenario.q, kQueryBase, kQueryTokens, visible, stale);
    if (!expect_control_separates(correct, measure(got_stale.out, got_reference.out),
                                 "negative control: a relocation that published no deltas")) {
        ++failures;
    }
}

// 3. The same check through A1, which is the append-and-attend route the
//    prefill itself uses, so the relocated page is read by the same kernel
//    that wrote the chunk sitting after it.
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
    const DeltaPlane plane = make_delta_plane(delta);
    const PagedKVBatchLayerView batch = delta_batch_view(relocated, plane, freq);

    Cache reference = make_reference_cache(scenario);

    const AttentionRun got_relocated = run_append_attention_view(
        relocated, tail_q_data, tail_k, tail_v, tail_first, kTailTokens, visible, batch);
    const AttentionRun got_reference = run_append_attention(
        reference, tail_q_data, tail_k, tail_v, tail_first, kTailTokens, visible);

    if (!expect_conformant(measure(got_reference.out,
                                  ideal_attention(tail_q_data, reference, tail_first, kTailTokens)),
                          "A1 over a directly prefilled page", kConformanceIntegrationA1Bf16)) {
        ++failures;
    }
    const Error correct = measure(got_relocated.out, got_reference.out);
    if (!expect_equivalent(correct, "A1 over a delta-plane relocated page")) {
        ++failures;
    }

    // Negative control on a fresh cache: the source rotation survives and no
    // delta is published, so the relocated page's keys are read at the wrong
    // angles.
    Cache stale(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(stale, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);
    const DeltaPlane none          = make_delta_plane(0);
    const PagedKVBatchLayerView stale_batch = delta_batch_view(stale, none, freq);
    const AttentionRun got_stale = run_append_attention_view(
        stale, tail_q_data, tail_k, tail_v, tail_first, kTailTokens, visible, stale_batch);
    if (!expect_control_separates(correct, measure(got_stale.out, got_reference.out),
                                 "negative control: a relocation that published no deltas through A1")) {
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
    check_attention_equivalence(scenario);
    check_append_attention_equivalence(scenario);

    if (failures != 0) {
        std::printf("SCR relocation integration checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("SCR relocation integration checks passed\n");
    return 0;
}
