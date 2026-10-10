// Oracle test for the per-page RoPE delta machinery on the rotated INT8 cache paths: rk8v4
// (full-width int8 K, packed int4 V, K and V Hadamard-rotated) and rk4v4-e8 (packed int4 K and V,
// same rotations). Same scenario, same delta plane, same negative control as
// test_gqa_kv_page_delta_i8.cpp, plus the two things the plain-int8 test defers:
//
//   * a packed-int4 plane model in Int8Cache, because both rotated variants store V as nibble pairs
//     and rk4v4-e8 stores K the same way. The E8 projection is encode-side only -- the read branch of
//     every PackedK variant is gqa_kv_unpack_i4x16 followed by a scale multiply
//     (src/ops/kernel/gqa_attention_decode_i8.cuh:377-383) -- so no lattice codec is needed here;
//   * a host port of the 64-point Hadamard, because the rotated variants store H-rotated codes and
//     the FP64 oracle reads raw BF16 queries. H is orthonormal, so the oracle can inverse-rotate the
//     decoded plane instead of rotating Q, and the two are equal in exact arithmetic.
//
// rk2v4-e8 is out of scope (Phase 4d-4): its K read path is e8_root_decode_8d_int8, a real codec
// port, and codec_of refuses it loudly rather than decoding it as something else.
#include "ops/gqa_relocation_test_common.h"

using namespace ninfer;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

std::string named(const std::string& what, const char* variant) {
    return what + " [" + variant + "]";
}

// The runtime flag combinations, selected by KvCacheStorage in layouts_impl.h. This table is the
// single source for the Int8Cache constructors, the codec_of dispatch and the per-row diagnostic,
// so a stale value shows up in the ctest output instead of silently changing what is tested.
struct Variant {
    const char* name;
    bool packed_v;
    bool rotate_k;
    bool rotate_v;
    bool packed_k;
    bool e8_lattice;
    bool e8_root;
};

constexpr Variant kVariants[] = {
    {"rk8v4", true, true, true, false, false, false},
    {"rk4v4-e8", true, true, true, true, true, false},
    {"rk4v4", true, true, true, true, false, false},
};

constexpr std::int32_t kDelta = (kDestPage * kPageSize) - (kSourcePage * kPageSize); // = -384

constexpr std::int32_t kDecodeTokens = 8;
constexpr std::int32_t kDecodeBase =
    kDestPage * kPageSize + kRelocatedRun - kDecodeTokens;
constexpr std::uint32_t kDecodeVisible =
    static_cast<std::uint32_t>(kDecodeBase + kDecodeTokens);

// Part 2b's end-to-end packed-plane check, run on rk8v4's V plane: packed int4, Hadamard-rotated,
// scale = amax / 7, no clamp, so the encoder's per-element error is bounded by half an int4 step.
//
// The hard bound is a theorem, not a tuning knob. The oracle applies an orthonormal H to the decoded
// values, so the residual is |H(epsilon)| <= ||epsilon||_2 <= sqrt(64) * 0.5 * scale = 4 * scale.
// A correct decode can therefore never exceed 4 x scale, while the two ways to be wrong are both far
// above it: a swapped nibble parity and a missing inverse-Hadamard stage both measured a worst case
// of 14 x scale and 19 x scale respectively over 40k synthetic groups (the same 40k measured a worst
// case of 1.47 x scale and a mean of 0.75 x scale for the correct decode). The advisory value is the
// top of the observed correct envelope: exceeding it is recorded for the 4d-4 handoff, not failed.
constexpr double kPackedResidualUnits         = 4.0;
constexpr double kPackedResidualAdvisoryUnits = 1.5;

// ------------------------------------------------------------------------------------------------
// Part 2b: self-checks on the shared host helpers, run before any cache is built.

void check_hadamard_self_checks() {
    // 1. DC gain. The final stage scales by 0.125 and a 64-point +/-1 Hadamard has 64 in every DC
    //    entry, so a constant group maps to [8c, 0, ...]. A y[0] of c here would mean the port
    //    carries an extra factor of 1/8, which would bias every rotated score by that factor.
    std::array<double, 64> dc{};
    dc.fill(1.0);
    hadamard64_group(dc.data());
    double rest = 0.0;
    for (std::size_t i = 1; i < dc.size(); ++i) { rest = std::max(rest, std::fabs(dc[i])); }
    std::printf("  self-check DC gain: y[0]=%.17g max|y[i>0]|=%.3e\n", dc[0], rest);
    if (std::fabs(dc[0] - 8.0) > 1e-12 || rest > 1e-12) {
        fail("Hadamard DC gain is not 8: the port does not match gqa_kv_hadamard64");
    }

    std::mt19937 rng(4242u);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::array<double, 64> a{};
    for (double& x : a) { x = unit(rng); }

    // 2. Involution: H(H(x)) == x. The oracle's construction and the production inverse rotation of
    //    the V output both rely on the same transform undoing itself.
    std::array<double, 64> twice = a;
    hadamard64_group(twice.data());
    hadamard64_group(twice.data());
    double involution = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        involution = std::max(involution, std::fabs(twice[i] - a[i]));
    }
    std::printf("  self-check involution: max |H(H(x)) - x| = %.3e\n", involution);
    if (involution > 1e-12) { fail("the Hadamard port is not involutory"); }

    // 3. Orthogonality: <Ha, Hb> == <a, b>. This is what makes the oracle's "inverse-rotate the
    //    decoded plane" equal to the kernel's "rotate the query".
    std::array<double, 64> b{}, ha = a, hb{};
    for (double& x : b) { x = unit(rng); }
    hb = b;
    hadamard64_group(ha.data());
    hadamard64_group(hb.data());
    double lhs = 0.0;
    double rhs = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        lhs += ha[i] * hb[i];
        rhs += a[i] * b[i];
    }
    std::printf("  self-check orthogonality: <Ha,Hb>=%.17g <a,b>=%.17g\n", lhs, rhs);
    if (std::fabs(lhs - rhs) > 1e-12) { fail("the Hadamard port is not orthonormal"); }
}

void check_nibble_order() {
    // The fill kernel packs codes with gqa_kv_pack_i4: the low code in the low nibble, the high code
    // in the high nibble, each masked to four bits, and gqa_kv_unpack_i4 sign-extends the nibble.
    // 0x78 is (-8, +7) and 0x87 is (+7, -8), so a swapped parity convention cannot pass both; 0xFF is
    // (-1, -1), which separates sign extension from an eight-biased offset. (0xF8 and 0x8F, proposed
    // during review, decode to (-8, -1) and (-1, -8) -- 0xF is -1, not +7 -- and do not discriminate.)
    const std::int8_t b78_lo = unpack_i4(0x78u, 0);
    const std::int8_t b78_hi = unpack_i4(0x78u, 1);
    const std::int8_t b87_lo = unpack_i4(0x87u, 0);
    const std::int8_t b87_hi = unpack_i4(0x87u, 1);
    const std::int8_t bff_lo = unpack_i4(0xFFu, 0);
    const std::int8_t bff_hi = unpack_i4(0xFFu, 1);
    std::printf("  self-check nibble order: 0x78 -> (%d, %d), 0x87 -> (%d, %d), 0xFF -> (%d, %d)\n",
                static_cast<int>(b78_lo), static_cast<int>(b78_hi), static_cast<int>(b87_lo),
                static_cast<int>(b87_hi), static_cast<int>(bff_lo), static_cast<int>(bff_hi));
    if (b78_lo != -8 || b78_hi != 7 || b87_lo != 7 || b87_hi != -8 || unpack_i4(0x00u, 0) != 0 ||
        bff_lo != -1 || bff_hi != -1) {
        fail("packed int4 nibble order does not match gqa_kv_unpack_i4");
    }
}

// ------------------------------------------------------------------------------------------------
// Part 2b item 4, end-to-end form: decode a just-written packed V plane and compare it against the
// FP64 source the fill kernel quantized. Run on rk8v4, whose V encoder is a plain round-to-nearest
// int4 quantizer with scale = amax / 7 and no clamp, so the pre-Hadamard error bound of 0.5 * scale
// per element holds exactly. Do not run it on rk4v4-e8's planes: the encode-side E8 projection and
// its documented half-coset collapse are not modeled by the oracle, so the residual there is wider by
// construction and that variant's gate is its anchor instead.

void check_packed_plane_decode(const Scenario& scenario, const Variant& variant, bool k_plane) {
    Int8Cache cache(kLogicalPages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    // No delta plane here, and none is involved in the write path either: a delta only ever rotates
    // Q, so the codes this reads are exactly what the append in step 1 wrote.
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    constexpr std::int32_t head  = 0;
    constexpr std::int32_t token = 0;
    constexpr std::int32_t group = 1; // dims [64, 128)
    const std::int32_t position  = scenario.src + token;
    const std::int32_t page      = cache.physical_page_of(position);
    const std::int32_t offset    = position % kPageSize;

    const std::size_t row =
        (static_cast<std::size_t>(page) * kKvHeads + head) * kPageSize + offset;
    const std::int32_t leading  = k_plane ? cache.k_leading() : cache.v_leading();
    const DeviceBuffer& codes   = k_plane ? cache.k_codes : cache.v_codes;
    const DeviceBuffer& scales  = k_plane ? cache.k_scale : cache.v_scale;

    std::vector<std::uint8_t> code_bytes(
        static_cast<std::size_t>(leading) * kPageSize * kKvHeads *
        static_cast<std::size_t>(cache.physical_pages));
    codes.copy_to_host(code_bytes.data(), code_bytes.size());

    const std::size_t scale_elems = static_cast<std::size_t>(kQuantGroups) * kPageSize * kKvHeads *
                                    static_cast<std::size_t>(cache.physical_pages);
    std::vector<__half> scale_host(scale_elems);
    scales.copy_to_host(scale_host.data(), scale_host.size() * sizeof(__half));
    const double scale =
        static_cast<double>(__half2float(scale_host[row * kQuantGroups + group]));

    std::array<double, 64> decoded{};
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        const std::int32_t d = group * kQuantGroup + i;
        const std::size_t byte_idx = row * static_cast<std::size_t>(leading) +
                                     static_cast<std::size_t>(d / 2);
        decoded[static_cast<std::size_t>(i)] =
            static_cast<double>(unpack_i4(code_bytes[byte_idx], d & 1)) * scale;
    }
    hadamard64_group(decoded.data());

    const std::vector<__nv_bfloat16>& source_plane = k_plane ? scenario.k_src : scenario.v;
    double residual = 0.0;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        const std::int32_t d = group * kQuantGroup + i;
        const double source =
            static_cast<double>(__bfloat162float(source_plane[act_index(head, token, kKvHeads, d)]));
        residual = std::max(residual, std::fabs(decoded[static_cast<std::size_t>(i)] - source));
    }
    const double units = scale > 0.0 ? residual / scale : residual;
    std::printf("  [%s] packed %s plane cross-check: residual %.3e = %.3f x group scale %.3e\n",
                variant.name, k_plane ? "K" : "V", residual, units, scale);
    if (units > kPackedResidualUnits) {
        // Above 4 x scale the decode cannot be correct: H is orthonormal, so the residual is bounded
        // by ||epsilon||_2 <= 4 * scale no matter how the encoder distributed its error.
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "[%s] packed %s plane residual is %.2f x the group scale, over the %.2f that an "
                      "orthonormal inverse Hadamard proves a correct decode cannot reach; the nibble "
                      "order or the leading extent is wrong (measured wrong-decode floors: 14 x scale "
                      "for a swapped parity, 19 x scale for a missing rotation)",
                      variant.name, k_plane ? "K" : "V", units, kPackedResidualUnits);
        fail(buf);
    } else if (units > kPackedResidualAdvisoryUnits) {
        std::printf("  NOTE: [%s] packed %s plane residual %.2fx the group scale is above the advisory "
                    "%.2fx but inside the proven-correct envelope. The nibble assertions passed, so "
                    "this is the encoder's error distribution through the inverse Hadamard, not a "
                    "parity bug -- do not rearrange unpack_i4, just record the number.\n",
                    variant.name, k_plane ? "K" : "V", units, kPackedResidualAdvisoryUnits);
    }
}

// ------------------------------------------------------------------------------------------------
// A3: read-only attention over an already populated cache, per variant.

void check_decode_attention_with_delta(const Scenario& scenario, const Variant& variant,
                                       const FreqTable& freq) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const std::int32_t delta    = scenario.dst - scenario.src; // = kDelta = -384
    const std::uint32_t visible = kDecodeVisible;

    Int8Cache cache(kLogicalPages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    std::vector<std::int32_t> delta_host(kLogicalPages, 0);
    delta_host[kDestPage] = delta;
    DeviceBuffer delta_buf = upload_i32(delta_host);

    PagedKVLayerView view = cache.view();
    view.rope_deltas       = Tensor(delta_buf.p, DType::I32, {kLogicalPages});
    view.rope_inv_freq     = freq;

    auto got_relocated = run_cached_attention_view_i8(cache, scenario.q, kDecodeBase,
                                                      kDecodeTokens, visible, view);

    Int8Cache reference(kLogicalPages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                        variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, kRelocatedRun);
    PagedKVLayerView ref_view = reference.view();
    auto got_reference = run_cached_attention_view_i8(reference, scenario.q, kDecodeBase,
                                                      kDecodeTokens, visible, ref_view);

    // FP64 anchor, enabled for every rotated variant. kConformanceInt8 goes to BOTH calls -- measure()
    // uses it for the gross thresholds and expect_conformant for relative L2 -- so the anchor is
    // judged against the INT8 profile end to end. This is the check that validates the host Hadamard
    // and the packed-plane decode: a wrong parity, extent or DC gain fails here and nowhere else.
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(scenario.q, reference, kDecodeBase,
                                                   kDecodeTokens),
                                   kConformanceInt8),
                           named("A3 over a directly prefilled int8 page", variant.name),
                           kConformanceInt8)) {
        ++failures;
    }

    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           named("A3 int8 delta plane vs direct prefill", variant.name))) {
        ++failures;
    }

    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    PagedKVLayerView stale_view = cache.view();
    stale_view.rope_deltas  = Tensor(zero_buf.p, DType::I32, {kLogicalPages});
    stale_view.rope_inv_freq = freq;
    auto got_stale = run_cached_attention_view_i8(cache, scenario.q, kDecodeBase,
                                                  kDecodeTokens, visible, stale_view);

    // expect_equivalent and expect_control_separates read only relative_l2, so these two measure()
    // calls keep the BF16 default for the gross field without affecting what they assert.
    if (!expect_control_separates(measure(got_relocated.out, got_reference.out),
                                  measure(got_stale.out, got_reference.out),
                                  named("negative control: int8 decode with zero deltas",
                                        variant.name))) {
        ++failures;
    }
}

// ------------------------------------------------------------------------------------------------
// A1: the append-and-attend prefill route through the public entry point. The batch view carries the
// delta plane and the frequency table, and ops::gqa_attention forwards both into the prompt launcher
// it shares with the detail:: hook (src/ops/wrapper/gqa_attention.cpp:435-455 ->
// gqa_attention_prompt_launch). Going through the public route makes this stronger than the BF16
// prefill test, which enters at the detail:: hook: validation and route resolution are exercised too.
// Width 128 with batch 1 resolves to GqaAttentionRoute::Prompt (gqa_attention.cpp:344-356).

// The Op reads the cache through the view, so this takes the view alone: an Int8Cache parameter would
// be unused, and the batch view is what carries the delta plane, the frequency table and the flags.
AttentionRun run_prefill_attention_i8(const std::vector<__nv_bfloat16>& q,
                                      const std::vector<__nv_bfloat16>& k,
                                      const std::vector<__nv_bfloat16>& v,
                                      std::int32_t first_position, std::int32_t tokens,
                                      std::uint32_t visible,
                                      const PagedKVBatchLayerView& batch_view) {
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
        ops::gqa_attention_workspace_capacity_bytes(kQHeads, DType::I8, envelope, 1, tokens,
                                                    tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor ttable(dtable.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable, kAttentionScale, batch_view, envelope,
                       workspace, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

void check_prefill_attention_with_delta(const Scenario& scenario, const Variant& variant,
                                        const FreqTable& freq) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    constexpr std::int32_t kPrefillTokens = 128;
    constexpr std::int32_t kPrefillBase   = (kDestPage + 2) * kPageSize; // = 320; two pages past dst
    const std::uint32_t visible =
        static_cast<std::uint32_t>(kPrefillBase + kPrefillTokens);
    const std::int32_t delta = scenario.dst - scenario.src; // = kDelta = -384

    // The prefill path uses a Q sized for kPrefillTokens, not kQueryTokens. The reference must use
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
    Int8Cache cache(kLogicalPages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    // 2. Build the delta plane: all zeros except kDestPage
    std::vector<std::int32_t> delta_host(kLogicalPages, 0);
    delta_host[kDestPage] = delta;
    DeviceBuffer delta_buf = upload_i32(delta_host);

    // 3. Build the batch view with the delta plane and freq table attached
    PagedKVBatchLayerView batch_view = cache.batch_view();
    batch_view.rope_deltas           = Tensor(delta_buf.p, DType::I32, {kLogicalPages, 1});
    batch_view.rope_inv_freq         = freq;

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
    auto got_relocated = run_prefill_attention_i8(prefill_q, tail_k, tail_v, kPrefillBase,
                                                  kPrefillTokens, visible, batch_view);

    // 6. Reference: direct prefill at the destination, then the same tail chunk
    Int8Cache reference(kLogicalPages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                        variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, kRelocatedRun);
    auto got_reference = run_prefill_attention_i8(prefill_q, tail_k, tail_v, kPrefillBase,
                                                  kPrefillTokens, visible,
                                                  reference.batch_view());

    // 6b. Anchor the reference to the FP64 oracle. Same criterion plumbing as the decode path.
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(prefill_q, reference, kPrefillBase,
                                                   kPrefillTokens),
                                   kConformanceInt8),
                           named("A1 over a directly prefilled int8 page", variant.name),
                           kConformanceInt8)) {
        ++failures;
    }

    // 7. Compare delta-path against reference
    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           named("A1 with delta plane vs direct prefill", variant.name))) {
        ++failures;
    }

    // 8. NEGATIVE CONTROL: zero deltas on a second fresh cache
    std::vector<std::int32_t> zero_deltas(kLogicalPages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);

    Int8Cache stale_cache(kLogicalPages, scenario.table, kKvHeads, variant.packed_v,
                          variant.rotate_k, variant.rotate_v, variant.packed_k,
                          variant.e8_lattice, variant.e8_root);
    append_chunk(stale_cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun);

    PagedKVBatchLayerView stale_batch_view = stale_cache.batch_view();
    stale_batch_view.rope_deltas            = Tensor(zero_buf.p, DType::I32, {kLogicalPages, 1});
    stale_batch_view.rope_inv_freq          = freq;

    auto got_stale = run_prefill_attention_i8(prefill_q, tail_k, tail_v, kPrefillBase,
                                              kPrefillTokens, visible, stale_batch_view);

    if (!expect_control_separates(measure(got_relocated.out, got_reference.out),
                                  measure(got_stale.out, got_reference.out),
                                  named("negative control: zero deltas on prefill path",
                                        variant.name))) {
        ++failures;
    }
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }

    for (const Variant& variant : kVariants) {
        if (variant.e8_root) {
            std::printf("variant table must not request the Phase 4d-4 codec: %s\n", variant.name);
            return 1;
        }
        const KvCodec codec = codec_of(variant.packed_v, variant.rotate_k, variant.rotate_v,
                                       variant.packed_k, variant.e8_lattice, variant.e8_root);
        if (codec == KvCodec::I8G64) {
            std::printf("variant %s must not select the plain int8 codec\n", variant.name);
            return 1;
        }
    }

    const FreqTable freq    = default_inv_freq();
    const Scenario scenario = build_scenario(freq);

    std::printf("GQA KV page-delta rotated INT8 oracle checks\n");
    for (const Variant& variant : kVariants) {
        const KvCodec codec = codec_of(variant.packed_v, variant.rotate_k, variant.rotate_v,
                                       variant.packed_k, variant.e8_lattice, variant.e8_root);
        std::printf("  variant %-9s packed_v=%d rotate_k=%d rotate_v=%d packed_k=%d "
                    "e8_lattice=%d e8_root=%d -> codec=%s\n",
                    variant.name, static_cast<int>(variant.packed_v),
                    static_cast<int>(variant.rotate_k), static_cast<int>(variant.rotate_v),
                    static_cast<int>(variant.packed_k), static_cast<int>(variant.e8_lattice),
                    static_cast<int>(variant.e8_root),
                    codec == KvCodec::RK8V4 ? "RK8V4" : "RK4V4E8");
    }
    std::printf("  run of %d tokens: src %d -> dst %d, delta %d, block table permuted\n",
                kRelocatedRun, scenario.src, scenario.dst, scenario.dst - scenario.src);

    check_hadamard_self_checks();
    check_nibble_order();
    // The two packed encoders without the E8 projection: rk8v4's V plane and rk4v4's K plane. The
    // rk4v4-e8 row's planes deliberately get no such bound -- its encode-side projection is not
    // modeled by the oracle, so that row's gate is its anchor instead.
    check_packed_plane_decode(scenario, kVariants[0], false); // rk8v4, V
    check_packed_plane_decode(scenario, kVariants[2], true);  // rk4v4, K

    for (const Variant& variant : kVariants) {
        check_decode_attention_with_delta(scenario, variant, freq);
        check_prefill_attention_with_delta(scenario, variant, freq);
    }

    if (failures != 0) {
        std::printf("GQA KV page-delta rotated INT8 oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("GQA KV page-delta rotated INT8 oracle checks passed\n");
    return 0;
}
