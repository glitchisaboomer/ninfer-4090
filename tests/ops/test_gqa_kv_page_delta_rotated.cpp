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
//     decoded plane instead of rotating Q, and the two are equal in exact arithmetic;
//   * an FP64 anchor for the delta path itself. ideal_attention_with_delta rotates each decoded key
//     back by its page's delta -- the key-side form of the kernel's Q counter-rotation -- so the
//     relocated run is compared against mathematics rather than only against a second cache. The
//     cache-against-cache equivalence is kept, but it cannot carry the discrimination on a coarse
//     (int4) K plane: there the correct path's error against a directly prefilled cache is itself of
//     the order of that plane's encode error, which is not separable from a zero-delta run. The
//     anchor rotates the *decoded* key, so it models that encode error exactly and still sees a
//     skipped rotation at ~5.7e-2 against a 5.0e-3 criterion.
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

// A label carries both axes this suite varies: which codec variant, and which scenario shape.
std::string named(const std::string& what, const Variant& variant, const char* scenario_name) {
    return labelled(what, std::string(variant.name) + ", " + scenario_name);
}

// The small-t decode dispatch only accepts 1..8 query tokens. The prefill route is a 128-token
// width, which resolves to GqaAttentionRoute::Prompt.
constexpr std::int32_t kDecodeTokens  = 8;
constexpr std::int32_t kPrefillTokens = 128;

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
    const std::int32_t pages = scenario.logical_pages;
    Int8Cache cache(pages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    // No delta plane here, and none is involved in the write path either: a delta only ever rotates
    // Q, so the codes this reads are exactly what the append in step 1 wrote.
    append_chunk(cache, scenario.k_src, scenario.v, scenario.src, scenario.run);

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
                                       const char* scenario_name, const FreqTable& freq) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    const std::int32_t base      = scenario.dst + scenario.run - kDecodeTokens;
    const std::uint32_t visible  = static_cast<std::uint32_t>(base + kDecodeTokens);
    const DeltaMetadata metadata = make_delta_metadata(scenario);
    const std::int32_t pages     = scenario.logical_pages;

    Int8Cache cache(pages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    place_unaligned_relocation_i8(cache, scenario);

    DeviceBuffer delta_buf = upload_i32(metadata.deltas);
    DeviceBuffer range_buf(metadata.ranges.size() * sizeof(std::uint32_t));
    range_buf.copy_from_host(metadata.ranges.data(),
                             metadata.ranges.size() * sizeof(std::uint32_t));

    PagedKVLayerView view = cache.view();
    view.rope_deltas       = Tensor(delta_buf.p, DType::I32, {pages});
    view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages});
    view.rope_inv_freq     = freq;

    auto got_relocated = run_cached_attention_view_i8(cache, scenario.q, base, kDecodeTokens,
                                                      visible, view);

    // The FP64 reference for the relocated path itself: the same cache, with every relocated key
    // rotated back by its page's delta (the key-side form of the kernel's Q counter-rotation).
    // Anchoring here is what lets a correct relocation be told apart from a skipped rotation at the
    // criterion's own resolution: the oracle rotates the *decoded* key, so it carries the plane's
    // encode error exactly and the comparison does not inherit the coarse-int4 noise that swamps the
    // cache-against-cache equivalence below. It reads the interval plane too, because on a partial
    // page only the keys inside the interval are the ones that moved.
    const std::vector<double> delta_reference = ideal_attention_with_delta(
        scenario.q, cache, base, kDecodeTokens, metadata.deltas, metadata.ranges, freq);

    // kConformanceRotatedI8 goes to BOTH calls -- measure() uses it for the gross thresholds and
    // expect_conformant for relative L2 -- so the delta path is judged against the rotated profile
    // end to end. This is the check that validates the counter-rotation: a skipped rotation lands at
    // ~5.7e-2 and fails here and nowhere else.
    if (!expect_conformant(measure(got_relocated.out, delta_reference, kConformanceRotatedI8),
                           named("A3 delta plane against the FP64 delta oracle", variant,
                                 scenario_name),
                           kConformanceRotatedI8)) {
        ++failures;
    }

    Int8Cache reference(pages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                        variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, scenario.run);
    PagedKVLayerView ref_view = reference.view();
    auto got_reference = run_cached_attention_view_i8(reference, scenario.q, base, kDecodeTokens,
                                                      visible, ref_view);

    // FP64 anchor of the directly prefilled page, under the same rotated profile. This is the check
    // that validates the host Hadamard and the packed-plane decode: a wrong parity, extent or DC gain
    // fails here and nowhere else.
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(scenario.q, reference, base, kDecodeTokens),
                                   kConformanceRotatedI8),
                           named("A3 over a directly prefilled int8 page", variant,
                                 scenario_name),
                           kConformanceRotatedI8)) {
        ++failures;
    }

    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           named("A3 int8 delta plane vs direct prefill", variant,
                                 scenario_name))) {
        ++failures;
    }

    // NEGATIVE CONTROL: the same run in the same destination slots with an all-zero delta plane. The
    // codes are byte-identical to the correct cache's -- the placement is the same append -- so the
    // skipped counter-rotation is the only difference this measures, and on a partial page it stays
    // the only one: the ranges are still published, and a zero page delta selects the same unrotated
    // frame for both halves of the page.
    std::vector<std::int32_t> zero_deltas(pages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);
    Int8Cache stale(pages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    place_unaligned_relocation_i8(stale, scenario);
    PagedKVLayerView stale_view = stale.view();
    stale_view.rope_deltas       = Tensor(zero_buf.p, DType::I32, {pages});
    stale_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages});
    stale_view.rope_inv_freq     = freq;
    auto got_stale = run_cached_attention_view_i8(stale, scenario.q, base, kDecodeTokens, visible,
                                                  stale_view);

    // The control is anchored on the same delta oracle as the correct run, so what it measures is the
    // one difference that matters -- the missing delta plane on byte-identical codes. It asserts the
    // skipped rotation breaks the criterion the correct run is judged by, which holds for every
    // variant here; the multiplicative form does not, because the correct path's error against a
    // directly prefilled cache is itself O(the int4 K plane's encode error).
    if (!expect_control_outside(measure(got_stale.out, delta_reference, kConformanceRotatedI8),
                                named("negative control: int8 decode with zero deltas", variant,
                                      scenario_name),
                                kConformanceRotatedI8)) {
        ++failures;
    }
}

// ------------------------------------------------------------------------------------------------
// A1: the append-and-attend prefill route through the public entry point. The batch view carries the
// delta plane and the frequency table, and ops::gqa_attention forwards both into the prompt launcher
// it shares with the detail:: hook (src/ops/wrapper/gqa_attention.cpp:435-455 ->
// gqa_attention_prompt_launch). Going through the public route makes this stronger than the BF16
// prefill test, which enters at the detail:: hook: validation and route resolution are exercised too.
// Width 128 with batch 1 resolves to GqaAttentionRoute::Prompt (gqa_attention.cpp:344-356). The
// route itself is `run_prefill_attention_i8` in the shared header, so the BF16 and INT8 suites enter
// the same way and a change to the entry point moves both together.

void check_prefill_attention_with_delta(const Scenario& scenario, const Variant& variant,
                                        const char* scenario_name, const FreqTable& freq) {
    if (scenario.table[kDestPage] == kDestPage) {
        fail("block table must be non-identity at kDestPage for the negative control to be valid");
        return;
    }

    // The tail starts where the run ends, so in the off-grid scenario its first page is the run's
    // second destination page: that page then holds relocated keys in its leading slots and freshly
    // prefilled ones after them, which is the mixed page a boundary-aware prefill has to read.
    const std::int32_t base     = scenario.dst + scenario.run;
    const std::uint32_t visible = static_cast<std::uint32_t>(base + kPrefillTokens);
    const DeltaMetadata metadata = make_delta_metadata(scenario);
    const std::int32_t pages     = scenario.logical_pages;

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

    // 1. Build a cache holding the run at the destination slots, still rotated at the source
    //    positions (no physical re-rotation).
    Int8Cache cache(pages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                    variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    place_unaligned_relocation_i8(cache, scenario);

    // 2. Build both planes: the per-page delta and the interval each page's run occupies
    DeviceBuffer delta_buf = upload_i32(metadata.deltas);
    DeviceBuffer range_buf(metadata.ranges.size() * sizeof(std::uint32_t));
    range_buf.copy_from_host(metadata.ranges.data(),
                             metadata.ranges.size() * sizeof(std::uint32_t));

    // 3. Build the batch view with both planes and the freq table attached
    PagedKVBatchLayerView batch_view = cache.batch_view();
    batch_view.rope_deltas           = Tensor(delta_buf.p, DType::I32, {pages, 1});
    batch_view.rope_delta_ranges = Tensor(range_buf.p, DType::I32, {pages, 1});
    batch_view.rope_inv_freq     = freq;

    // 4. Build the tail K/V chunk the prefill call writes at the run's end
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
    auto got_relocated = run_prefill_attention_i8(prefill_q, tail_k, tail_v, base, kPrefillTokens,
                                                  visible, batch_view);

    // 5b. Anchor the relocated path to the FP64 delta oracle. The call above has just written the tail
    // chunk into `cache`, so the cache now holds exactly the codes the run read: the source-phase run
    // plus the tail. Rotating each relocated key back by its page's delta is what a skipped rotation
    // cannot survive -- and the tail sits outside every interval, so it must pass through untouched,
    // which is exactly what a partial page's plane says.
    const std::vector<double> delta_reference = ideal_attention_with_delta(
        prefill_q, cache, base, kPrefillTokens, metadata.deltas, metadata.ranges, freq);
    if (!expect_conformant(measure(got_relocated.out, delta_reference, kConformanceRotatedI8),
                           named("A1 delta plane against the FP64 delta oracle", variant,
                                 scenario_name),
                           kConformanceRotatedI8)) {
        ++failures;
    }

    // 6. Reference: direct prefill at the destination, then the same tail chunk
    Int8Cache reference(pages, scenario.table, kKvHeads, variant.packed_v, variant.rotate_k,
                        variant.rotate_v, variant.packed_k, variant.e8_lattice, variant.e8_root);
    append_chunk(reference, scenario.k_dst, scenario.v, scenario.dst, scenario.run);
    auto got_reference = run_prefill_attention_i8(prefill_q, tail_k, tail_v, base, kPrefillTokens,
                                                  visible, reference.batch_view());

    // 6b. Anchor the reference to the FP64 oracle. Same criterion plumbing as the decode path.
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(prefill_q, reference, base, kPrefillTokens),
                                   kConformanceRotatedI8),
                           named("A1 over a directly prefilled int8 page", variant,
                                 scenario_name),
                           kConformanceRotatedI8)) {
        ++failures;
    }

    // 7. Compare delta-path against reference
    if (!expect_equivalent(measure(got_relocated.out, got_reference.out),
                           named("A1 with delta plane vs direct prefill", variant,
                                 scenario_name))) {
        ++failures;
    }

    // 8. NEGATIVE CONTROL: the same run in the same destination slots with an all-zero delta plane,
    // through the same route, which also writes the same tail chunk into it.
    std::vector<std::int32_t> zero_deltas(pages, 0);
    DeviceBuffer zero_buf = upload_i32(zero_deltas);

    Int8Cache stale_cache(pages, scenario.table, kKvHeads, variant.packed_v,
                          variant.rotate_k, variant.rotate_v, variant.packed_k,
                          variant.e8_lattice, variant.e8_root);
    place_unaligned_relocation_i8(stale_cache, scenario);

    PagedKVBatchLayerView stale_batch_view = stale_cache.batch_view();
    stale_batch_view.rope_deltas            = Tensor(zero_buf.p, DType::I32, {pages, 1});
    stale_batch_view.rope_delta_ranges =
        Tensor(range_buf.p, DType::I32, {pages, 1});
    stale_batch_view.rope_inv_freq = freq;

    auto got_stale = run_prefill_attention_i8(prefill_q, tail_k, tail_v, base, kPrefillTokens,
                                              visible, stale_batch_view);

    // Same anchored control as the decode path: `stale_cache` holds byte-identical codes (the same
    // source-phase run and the same tail chunk), so against `delta_reference` the missing delta plane
    // is the only difference being measured.
    if (!expect_control_outside(measure(got_stale.out, delta_reference, kConformanceRotatedI8),
                                named("negative control: zero deltas on prefill path", variant,
                                      scenario_name),
                                kConformanceRotatedI8)) {
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

    const FreqTable freq     = default_inv_freq();
    const Scenario scenario  = build_scenario(freq);
    const Scenario unaligned = build_unaligned_scenario(freq);
    const Scenario delta_3021  = build_large_delta_scenario(freq, -3021);
    const Scenario delta_9340  = build_large_delta_scenario(freq, -9340);
    const Scenario delta_split = build_large_delta_unaligned_scenario(freq, -9340);
    // A run longer than a page, so whole relocated pages sit between the two partial ends -- the
    // only shape that drives the partial -> whole -> partial transitions in the score path.
    const Scenario long_run = build_unaligned_long_scenario(freq);

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
    std::printf("  off-grid run: src %d -> dst %d, delta %d, dst slots %d..%d of the page\n",
                unaligned.src, unaligned.dst, unaligned.dst - unaligned.src,
                unaligned.dst % kPageSize, kPageSize - 1);
    for (const Scenario* large : {&delta_3021, &delta_9340, &delta_split}) {
        const DeltaMetadata metadata = make_delta_metadata(*large);
        const std::size_t page       = static_cast<std::size_t>(large->dst / kPageSize);
        const std::uint32_t range    = metadata.ranges[page];
        std::printf("  large delta: src %d -> dst %d, plane[%zu] = %d, slots [%u, %u), %d pages\n",
                    large->src, large->dst, page, metadata.deltas[page],
                    rope_delta_page_range_begin(range), rope_delta_page_range_end(range),
                    large->logical_pages);
        if (metadata.deltas[page] != large->dst - large->src) {
            fail("the delta plane does not carry the scenario's shift for the destination page");
        }
    }

    check_hadamard_self_checks();
    check_nibble_order();
    // The two packed encoders without the E8 projection: rk8v4's V plane and rk4v4's K plane. The
    // rk4v4-e8 row's planes deliberately get no such bound -- its encode-side projection is not
    // modeled by the oracle, so that row's gate is its anchor instead.
    check_packed_plane_decode(scenario, kVariants[0], false); // rk8v4, V
    check_packed_plane_decode(scenario, kVariants[2], true);  // rk4v4, K

    for (const Variant& variant : kVariants) {
        check_decode_attention_with_delta(scenario, variant, "aligned", freq);
        check_prefill_attention_with_delta(scenario, variant, "aligned", freq);
        check_decode_attention_with_delta(unaligned, variant, "off-grid", freq);
        check_prefill_attention_with_delta(unaligned, variant, "off-grid", freq);
    }

    // Deltas in the thousands, which is the range production published (2026-10-17 session: -3021
    // and -9340). Every other scenario here moves a run by six pages, where the rotation argument
    // stays in the hundreds; pair 0 of the frequency table is 1.0, so the argument *is* the delta
    // and 9340 radians is a different numerical regime. The off-grid case additionally puts the
    // extra group-0 tile the partial page reads into a frame thousands of tokens away.
    for (const Variant& variant : kVariants) {
        check_decode_attention_with_delta(delta_3021, variant, "large delta 3021", freq);
        check_prefill_attention_with_delta(delta_3021, variant, "large delta 3021", freq);
        check_decode_attention_with_delta(delta_9340, variant, "large delta 9340", freq);
        check_prefill_attention_with_delta(delta_9340, variant, "large delta 9340", freq);
        check_decode_attention_with_delta(delta_split, variant, "large delta 9340, off-grid",
                                          freq);
        check_prefill_attention_with_delta(delta_split, variant, "large delta 9340, off-grid",
                                           freq);
    }

    // The multi-page off-grid run, on every variant: a partial head, whole interior pages, and a
    // partial tail, which the one-page off-grid scenario above cannot build.
    std::printf("  long off-grid run: src %d -> dst %d, delta %d, %d tokens over dst pages %d..%d\n",
                long_run.src, long_run.dst, long_run.dst - long_run.src, long_run.run,
                long_run.dst / kPageSize, (long_run.dst + long_run.run - 1) / kPageSize);
    for (const Variant& variant : kVariants) {
        check_decode_attention_with_delta(long_run, variant, "off-grid, multi-page", freq);
        check_prefill_attention_with_delta(long_run, variant, "off-grid, multi-page", freq);
    }

    if (failures != 0) {
        std::printf("GQA KV page-delta rotated INT8 oracle checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("GQA KV page-delta rotated INT8 oracle checks passed\n");
    return 0;
}
