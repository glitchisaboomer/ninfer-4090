// Oracle test for the per-page RoPE delta machinery on the INT8-G64 cache path. Same
// scenario, same delta plane, same negative control as test_gqa_kv_page_delta.cpp, but the
// cache holds signed int8 codes and half per-group scales and the FP64 anchor uses the
// INT8 conformance criterion.
//
// The counter-rotation is applied to Q at the load site of each i8 attention kernel, before
// quantization, so the quantized Q matches what a directly-prefilled query at the source
// position would have produced. For the plain Int8Group64 variant there is no K Hadamard;
// the rotated variants (rk4v4-e8) interpose one and are handled by
// test_gqa_kv_page_delta_rotated.cpp.
//
// Both scenarios run on both routes: the aligned one, where the relocated run covers whole
// destination pages, and the off-grid one, where each end of the run falls inside a page. The
// second is the only case in which a destination page holds relocated and unrelocated keys at
// once, so it is the one that pins the per-key query-frame choice in both kernels.
#include "ops/gqa_relocation_test_common.h"

using namespace ninfer;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

std::string named(const std::string& what, const char* scenario_name) {
    return labelled(what, scenario_name);
}

// Print what the delta plane actually carries for the destination page, not just what the fixture
// asked for. The two can only disagree if the planner refused the span, and a fixture that silently
// publishes an all-zero plane is the failure this whole test exists to catch.
void print_large_delta_scenario(const char* name, const Scenario& scenario) {
    const DeltaMetadata metadata = make_delta_metadata(scenario);
    const std::int32_t pages     = scenario.logical_pages;
    const std::size_t page       = static_cast<std::size_t>(scenario.dst / kPageSize);
    const std::uint32_t range    = metadata.ranges[page];
    std::printf("  %s: src %d -> dst %d, delta %d, plane[%zu] = %d, slots [%u, %u), %d pages\n",
                name, scenario.src, scenario.dst, scenario.dst - scenario.src, page,
                metadata.deltas[page], rope_delta_page_range_begin(range),
                rope_delta_page_range_end(range), pages);
    if (metadata.deltas[page] != scenario.dst - scenario.src) {
        fail("the delta plane does not carry the scenario's shift for the destination page");
    }
}

// The flag combination for KvCacheStorage::Int8Group64, confirmed in Part 0a: expected all
// false. These constants feed both Int8Cache constructions and the diagnostic printf in
// main, so a stale value is visible in the ctest output. If Part 0a reports otherwise,
// change them here and report the deviation.
constexpr bool kPackedV    = false;
constexpr bool kRotateK    = false;
constexpr bool kRotateV    = false;
constexpr bool kPackedK    = false;
constexpr bool kE8Lattice  = false;
constexpr bool kE8Root     = false;

// The small-t decode dispatch only accepts 1..8 query tokens; 8 is enough to exercise the
// Q counter-rotation path. The prefill route is a 128-token width, which resolves to
// GqaAttentionRoute::Prompt.
constexpr std::int32_t kDecodeTokens  = 8;
constexpr std::int32_t kPrefillTokens = 128;

void check_decode_attention_with_delta(const Scenario& scenario, const char* scenario_name) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq         = default_inv_freq();
    const std::int32_t base      = scenario.dst + scenario.run - kDecodeTokens;
    const std::uint32_t visible  = static_cast<std::uint32_t>(base + kDecodeTokens);
    const DeltaMetadata metadata = make_delta_metadata(scenario);
    const std::int32_t pages     = scenario.logical_pages;

    DeviceBuffer delta_buf = upload_i32(metadata.deltas);
    DeviceBuffer range_buf(metadata.ranges.size() * sizeof(std::uint32_t));
    range_buf.copy_from_host(metadata.ranges.data(),
                             metadata.ranges.size() * sizeof(std::uint32_t));

    Int8Cache cache(pages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                    kPackedK, kE8Lattice, kE8Root);
    place_unaligned_relocation_i8(cache, scenario);

    PagedKVLayerView view = cache.view();
    view.rope_deltas       = Tensor(delta_buf.p, DType::I32, {pages});
    view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages});
    view.rope_inv_freq     = freq;

    auto got_relocated = run_cached_attention_view_i8(cache, scenario.q, base, kDecodeTokens,
                                                      visible, view);

    Int8Cache reference(pages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                        kPackedK, kE8Lattice, kE8Root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, scenario.run);
    PagedKVLayerView ref_view = reference.view();
    auto got_reference = run_cached_attention_view_i8(reference, scenario.q, base, kDecodeTokens,
                                                      visible, ref_view);

    // FP64 anchor: enabled because Int8Group64 has rotate_k = false, so the oracle needs no
    // Hadamard stage. kConformanceInt8 goes to BOTH calls -- measure() uses it for the gross
    // thresholds and expect_conformant for relative L2 -- so the anchor is judged against the
    // INT8 profile end to end, not a BF16/INT8 hybrid. A failure here is in the scenario or
    // the Int8Cache image decode, not in the delta machinery.
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(scenario.q, reference, base, kDecodeTokens),
                                   kConformanceInt8),
                           named("A3 over a directly prefilled int8 page", scenario_name),
                           kConformanceInt8)) {
        ++failures;
    }

    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           named("A3 int8 delta plane vs direct prefill", scenario_name))) {
        ++failures;
    }

    // NEGATIVE CONTROL: the same run in the same destination slots, read with an all-zero delta
    // plane. The codes are byte-identical to the correct cache's -- the placement is the same
    // append -- so the skipped counter-rotation is the only difference this measures, and on a
    // partial page it stays the only difference: the ranges are still published, and a zero page
    // delta selects the same unrotated frame for both halves of the page.
    std::vector<std::int32_t> zero_deltas(pages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    Int8Cache stale(pages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                    kPackedK, kE8Lattice, kE8Root);
    place_unaligned_relocation_i8(stale, scenario);
    PagedKVLayerView stale_view = stale.view();
    stale_view.rope_deltas       = Tensor(zero_buf.p, DType::I32, {pages});
    stale_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages});
    stale_view.rope_inv_freq     = freq;
    auto got_stale = run_cached_attention_view_i8(stale, scenario.q, base, kDecodeTokens,
                                                  visible, stale_view);

    // expect_equivalent and expect_control_separates read only relative_l2, so these two
    // measure() calls keep the BF16 default for the gross field without affecting meaning.
    if (!expect_control_separates(measure(got_relocated.out, got_reference.out),
                                  measure(got_stale.out, got_reference.out),
                                  named("negative control: int8 decode with zero deltas",
                                        scenario_name))) {
        ++failures;
    }
}

void check_prefill_attention_with_delta(const Scenario& scenario, const char* scenario_name) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq        = default_inv_freq();
    const std::int32_t base     = scenario.dst + scenario.run;
    const std::uint32_t visible = static_cast<std::uint32_t>(base + kPrefillTokens);
    const std::int32_t pages    = scenario.logical_pages;

    // The prefill path uses a Q sized for kPrefillTokens, not kQueryTokens. The reference must use
    // the same Q at the same positions, or the comparison is meaningless.
    const std::size_t prefill_q_elems =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
        static_cast<std::size_t>(kPrefillTokens);
    std::vector<__nv_bfloat16> prefill_q(prefill_q_elems);
    {
        std::mt19937 q_rng(17u);
        std::uniform_real_distribution<double> q_unit(-0.25, 0.25);
        for (std::size_t i = 0; i < prefill_q_elems; ++i) {
            prefill_q[i] = __float2bfloat16(static_cast<float>(q_unit(q_rng)));
        }
    }

    const DeltaMetadata metadata = make_delta_metadata(scenario);
    DeviceBuffer delta_buf       = upload_i32(metadata.deltas);
    DeviceBuffer range_buf(metadata.ranges.size() * sizeof(std::uint32_t));
    range_buf.copy_from_host(metadata.ranges.data(),
                             metadata.ranges.size() * sizeof(std::uint32_t));

    Int8Cache cache(pages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                    kPackedK, kE8Lattice, kE8Root);
    place_unaligned_relocation_i8(cache, scenario);

    PagedKVBatchLayerView batch_view = cache.batch_view();
    batch_view.rope_deltas       = Tensor(delta_buf.p, DType::I32, {pages, 1});
    batch_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages, 1});
    batch_view.rope_inv_freq     = freq;

    // The tail chunk the prefill call writes past the run. In the off-grid scenario its first page
    // is the run's second destination page, so that page ends up holding relocated keys in its
    // leading slots and freshly prefilled ones after them -- the mixed page a boundary-aware prefill
    // has to read.
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

    auto got_relocated = run_prefill_attention_i8(prefill_q, tail_k, tail_v, base, kPrefillTokens,
                                                  visible, batch_view);

    // Reference: a direct prefill at the destination, then the same tail chunk.
    Int8Cache reference(pages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                        kPackedK, kE8Lattice, kE8Root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, scenario.run);
    auto got_reference = run_prefill_attention_i8(prefill_q, tail_k, tail_v, base, kPrefillTokens,
                                                  visible, reference.batch_view());

    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(prefill_q, reference, base, kPrefillTokens),
                                   kConformanceInt8),
                           named("A1 over a directly prefilled int8 page (anchors the harness)",
                                 scenario_name),
                           kConformanceInt8)) {
        ++failures;
    }

    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           named("A1 int8 delta plane vs direct prefill", scenario_name))) {
        ++failures;
    }

    // NEGATIVE CONTROL: the same run in the same destination slots with an all-zero delta plane,
    // through the same route, which also writes the same tail chunk into it.
    std::vector<std::int32_t> zero_deltas(pages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    Int8Cache stale(pages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                    kPackedK, kE8Lattice, kE8Root);
    place_unaligned_relocation_i8(stale, scenario);
    PagedKVBatchLayerView stale_view = stale.batch_view();
    stale_view.rope_deltas       = Tensor(zero_buf.p, DType::I32, {pages, 1});
    stale_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages, 1});
    stale_view.rope_inv_freq     = freq;

    auto got_stale = run_prefill_attention_i8(prefill_q, tail_k, tail_v, base, kPrefillTokens,
                                              visible, stale_view);

    if (!expect_control_separates(measure(got_relocated.out, got_reference.out),
                                  measure(got_stale.out, got_reference.out),
                                  named("negative control: zero deltas on prefill path",
                                        scenario_name))) {
        ++failures;
    }
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77;
    }

    const FreqTable freq     = default_inv_freq();
    const Scenario scenario  = build_scenario(freq);
    const Scenario unaligned = build_unaligned_scenario(freq);

    std::printf("GQA KV page-delta INT8 oracle checks\n");
    std::printf("  flags: packed_v=%d rotate_k=%d rotate_v=%d packed_k=%d e8_lattice=%d e8_root=%d\n",
                static_cast<int>(kPackedV), static_cast<int>(kRotateK),
                static_cast<int>(kRotateV), static_cast<int>(kPackedK),
                static_cast<int>(kE8Lattice), static_cast<int>(kE8Root));
    std::printf("  run of %d tokens: src %d -> dst %d, delta %d, block table permuted\n",
                kRelocatedRun, scenario.src, scenario.dst, scenario.dst - scenario.src);
    std::printf("  off-grid run: src %d -> dst %d, delta %d, dst slots %d..%d of the page\n",
                unaligned.src, unaligned.dst, unaligned.dst - unaligned.src,
                unaligned.dst % kPageSize, kPageSize - 1);

    check_decode_attention_with_delta(scenario, "aligned");
    check_prefill_attention_with_delta(scenario, "aligned");
    check_decode_attention_with_delta(unaligned, "off-grid");
    check_prefill_attention_with_delta(unaligned, "off-grid");

    // Deltas in the thousands: the two values production published (2026-10-17 session, -3021 and
    // -9340). The counter-rotation argument is delta * inv_freq, whose pair 0 is delta itself, so
    // these are the only cases whose angle is large enough for range reduction to matter, and the
    // off-grid one asks it of a page holding two frames at once.
    const Scenario delta_3021  = build_large_delta_scenario(freq, -3021);
    const Scenario delta_9340  = build_large_delta_scenario(freq, -9340);
    const Scenario delta_split = build_large_delta_unaligned_scenario(freq, -9340);
    print_large_delta_scenario("large delta, aligned", delta_3021);
    print_large_delta_scenario("large delta, aligned", delta_9340);
    print_large_delta_scenario("large delta, off-grid", delta_split);

    check_decode_attention_with_delta(delta_3021, "large delta, aligned");
    check_prefill_attention_with_delta(delta_3021, "large delta, aligned");
    check_decode_attention_with_delta(delta_9340, "large delta, aligned");
    check_prefill_attention_with_delta(delta_9340, "large delta, aligned");
    check_decode_attention_with_delta(delta_split, "large delta, off-grid");
    check_prefill_attention_with_delta(delta_split, "large delta, off-grid");

    // The engine's shape: a run longer than a page, so whole relocated pages sit between the two
    // partial ends. The one-page off-grid scenario above never produces a whole relocated page, so
    // only this one exercises the partial -> whole -> partial transitions in the score path.
    const Scenario long_run = build_unaligned_long_scenario(freq);
    std::printf("  long off-grid run: src %d -> dst %d, delta %d, %d tokens over dst pages %d..%d\n",
                long_run.src, long_run.dst, long_run.dst - long_run.src, long_run.run,
                long_run.dst / kPageSize, (long_run.dst + long_run.run - 1) / kPageSize);
    check_decode_attention_with_delta(long_run, "off-grid, multi-page");
    check_prefill_attention_with_delta(long_run, "off-grid, multi-page");

    if (failures != 0) {
        std::printf("GQA KV page-delta INT8 oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("GQA KV page-delta INT8 oracle checks passed\n");
    return 0;
}
