// Oracle test for the per-page RoPE delta machinery on the INT8-G64 cache path. Same
// scenario, same delta plane, same negative control as test_gqa_kv_page_delta.cpp, but the
// cache holds signed int8 codes and half per-group scales and the FP64 anchor uses the
// INT8 conformance criterion.
//
// The counter-rotation is applied to Q at the load site of each i8 attention kernel, before
// quantization, so the quantized Q matches what a directly-prefilled query at the source
// position would have produced. For the plain Int8Group64 variant there is no K Hadamard;
// the rotated variants (rk4v4-e8) interpose one and are handled in Phase 4d-3.
#include "ops/gqa_relocation_test_common.h"

using namespace ninfer;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
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

constexpr std::int32_t kDelta = (kDestPage * kPageSize) - (kSourcePage * kPageSize); // = -384

constexpr std::int32_t kDecodeTokens = 8;
constexpr std::int32_t kDecodeBase =
    kDestPage * kPageSize + kRelocatedRun - kDecodeTokens;
constexpr std::uint32_t kDecodeVisible =
    static_cast<std::uint32_t>(kDecodeBase + kDecodeTokens);

void check_decode_attention_with_delta(const Scenario& scenario) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const FreqTable freq      = default_inv_freq();
    const std::int32_t delta  = scenario.dst - scenario.src; // = kDelta = -384
    const std::uint32_t visible = kDecodeVisible;

    Int8Cache cache(kLogicalPages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                    kPackedK, kE8Lattice, kE8Root);
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    std::vector<std::int32_t> delta_host(kLogicalPages, 0);
    delta_host[kDestPage] = delta;
    DeviceBuffer delta_buf = upload_i32(delta_host);

    PagedKVLayerView view = cache.view();
    view.rope_deltas       = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    view.rope_inv_freq     = freq;

    auto got_relocated = run_cached_attention_view_i8(cache, scenario.q, kDecodeBase,
                                                     kDecodeTokens, visible, view);

    Int8Cache reference(kLogicalPages, scenario.table, kKvHeads, kPackedV, kRotateK, kRotateV,
                        kPackedK, kE8Lattice, kE8Root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, kRelocatedRun);
    PagedKVLayerView ref_view = reference.view();
    auto got_reference = run_cached_attention_view_i8(reference, scenario.q, kDecodeBase,
                                                     kDecodeTokens, visible, ref_view);

    // FP64 anchor: enabled because Int8Group64 has rotate_k = false, so the oracle needs no
    // Hadamard stage. kConformanceInt8 goes to BOTH calls — measure() uses it for the gross
    // thresholds and expect_conformant for relative L2 — so the anchor is judged against the
    // INT8 profile end to end, not a BF16/INT8 hybrid. A failure here is in the scenario or
    // the Int8Cache image decode, not in the delta machinery.
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(scenario.q, reference, kDecodeBase,
                                                   kDecodeTokens),
                                   kConformanceInt8),
                           "A3 over a directly prefilled int8 page", kConformanceInt8)) {
        ++failures;
    }

    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           "A3 int8 delta plane vs direct prefill")) {
        ++failures;
    }

    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    PagedKVLayerView stale_view = cache.view();
    stale_view.rope_deltas  = Tensor(zero_buf.p, DType::I32, {kLogicalPages});
    stale_view.rope_inv_freq = freq;
    auto got_stale = run_cached_attention_view_i8(cache, scenario.q, kDecodeBase,
                                                 kDecodeTokens, visible, stale_view);

    // expect_equivalent and expect_control_separates read only relative_l2, so these two
    // measure() calls keep the BF16 default for the gross field without affecting meaning.
    if (!expect_control_separates(measure(got_relocated.out, got_reference.out),
                                  measure(got_stale.out, got_reference.out),
                                  "negative control: int8 decode with zero deltas")) {
        ++failures;
    }
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77;
    }

    const FreqTable freq    = default_inv_freq();
    const Scenario scenario = build_scenario(freq);

    std::printf("GQA KV page-delta INT8 oracle checks\n");
    std::printf("  flags: packed_v=%d rotate_k=%d rotate_v=%d packed_k=%d e8_lattice=%d e8_root=%d\n",
                static_cast<int>(kPackedV), static_cast<int>(kRotateK),
                static_cast<int>(kRotateV), static_cast<int>(kPackedK),
                static_cast<int>(kE8Lattice), static_cast<int>(kE8Root));
    std::printf("  run of %d tokens: src %d -> dst %d, delta %d, block table permuted\n",
                kRelocatedRun, scenario.src, scenario.dst, scenario.dst - scenario.src);

    check_decode_attention_with_delta(scenario);

    if (failures != 0) {
        std::printf("GQA KV page-delta INT8 oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("GQA KV page-delta INT8 oracle checks passed\n");
    return 0;
}
