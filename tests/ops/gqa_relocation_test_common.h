// Shared test infrastructure for SCR relocated-K attention oracle tests.
//
// Both test_gqa_kv_page_delta.cpp and test_scr_relocation_integration.cpp use the same scenario,
// the same FP64 oracle, and the same thresholds. This header is the single source for that
// infrastructure so the two tests cannot drift.
#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/gqa_kv_rerotate.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

// qwen3_6_27b Text geometry: 24 query heads over 4 KV heads, head_dim 256, 64-token pages.
constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kQHeads      = 24;
constexpr std::int32_t kKvHeads     = 4;
constexpr std::int32_t kQGroup      = kQHeads / kKvHeads;
constexpr std::int32_t kPageSize    = kPagedKVPageSize;
constexpr std::int32_t kRotaryPairs = 32;
constexpr float kAttentionScale     = 0.0625F; // 1/sqrt(head_dim), the runtime's Text scale.

// INT8-G64 quantization geometry, mirroring kGqaKvQuantGroup in
// src/ops/kernel/gqa_attention_kv_quant.cuh without including that device header.
constexpr std::int32_t kQuantGroup  = 64;
constexpr std::int32_t kQuantGroups = kHeadDim / kQuantGroup; // = 4

using FreqTable = std::array<float, kRotaryPairs>;

// The activation layout, matching the fill kernel's src_off = d + head_dim * (head + heads * token).
[[nodiscard]] inline std::size_t act_index(std::int32_t head, std::int32_t token, std::int32_t heads,
                                           std::int32_t d) {
    return (static_cast<std::size_t>(head) +
            static_cast<std::size_t>(heads) * static_cast<std::size_t>(token)) *
               kHeadDim +
           static_cast<std::size_t>(d);
}

// The cache plane layout, matching paged_kv_element_offset<256, KvHeads>.
[[nodiscard]] inline std::size_t plane_index(std::int32_t physical_page, std::int32_t kv_head,
                                             std::int32_t page_offset, std::int32_t d,
                                             std::int32_t kv_heads) {
    return (static_cast<std::size_t>(kv_head) +
            static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(physical_page)) *
                   static_cast<std::size_t>(kPageSize) * kHeadDim +
           static_cast<std::size_t>(page_offset) * kHeadDim + static_cast<std::size_t>(d);
}

inline FreqTable default_inv_freq() {
    FreqTable freq{};
    for (std::int32_t i = 0; i < kRotaryPairs; ++i) {
        freq[static_cast<std::size_t>(i)] =
            static_cast<float>(std::pow(1.0e7, -2.0 * static_cast<double>(i) / 64.0));
    }
    return freq;
}

// Forward Text RoPE for one 256-element row at one position, in FP64. Elements [64, 256) carry no
// rotation.
inline void forward_rope(const double* raw, std::int32_t position, const FreqTable& freq, double* out) {
    for (std::int32_t d = 0; d < kHeadDim; ++d) { out[d] = raw[d]; }
    for (std::int32_t pair = 0; pair < kRotaryPairs; ++pair) {
        const double theta = static_cast<double>(position) * static_cast<double>(freq[pair]);
        const double c     = std::cos(theta);
        const double s     = std::sin(theta);
        const double a     = raw[pair];
        const double b     = raw[pair + kRotaryPairs];
        out[pair]                = a * c - b * s;
        out[pair + kRotaryPairs] = b * c + a * s;
    }
}

inline double bf16_ulp(double x) {
    if (x == 0.0) { return std::ldexp(1.0, -133); }
    return std::ldexp(1.0, static_cast<int>(std::ilogb(std::fabs(x))) - 7);
}

// A conformance criterion, matching the shape of ReductionCriterion in
// tests/ops/test_gqa_attention.cpp so the two files agree on field names and semantics.
struct Criterion {
    double relative_l2;
    double gross_absolute;
    double gross_relative_to_max_reference;
};

// The BF16 criterion, matching kAttentionBf16Criterion in test_gqa_attention.cpp.
constexpr Criterion kConformanceBf16{2.8e-3, 1.0e-3, 2.7e-3};

// The INT8 criterion, matching kAttentionInt8Criterion in test_gqa_attention.cpp.
constexpr Criterion kConformanceInt8{3.15e-3, 1.1e-3, 2.2e-3};

// The rotated int8 profile: the same cache with a per-64-group Hadamard interposed on K and V. It
// differs from kConformanceInt8 by exactly one term the FP64 oracle cannot model -- the production
// output is staged through bf16 *before* the output inverse Hadamard
// (gqa_kv_inverse_rotate_output_kernel, used by both the decode and the prefill launcher), where a
// plain int8 cache writes bf16 once. That term is variant-independent, and so is the measurement:
// the A3 reference anchor of tests/ops/test_gqa_kv_page_delta_rotated.cpp reads 3.750e-3 (rk8v4),
// 3.777e-3 (rk4v4-e8) and 3.757e-3 (rk4v4) -- three digits agreeing across an int8 vs a packed int4
// K plane and across an E8-projected vs a plain encoder, which is what rules out a decode, extent or
// parity error (those scale with the codec) and attributes the excess to the shared staging term.
// kConformanceInt8 is left untouched so the plain path keeps its own profile; 5.0e-3 is the measured
// maximum plus a third of margin, and the gross thresholds are carried over unchanged so only the
// L2 side of the profile moves.
constexpr Criterion kConformanceRotatedI8{5.0e-3, 1.1e-3, 2.2e-3};

struct Error {
    double relative_l2 = 0.0;
    double gross       = 0.0;
};

// Relative L2 over the whole tensor, plus a gross-error count so a few catastrophic elements cannot
// hide inside an otherwise small L2. The criterion supplies the gross-error thresholds; the existing
// call sites pass kConformanceBf16 by default.
[[nodiscard]] inline Error measure(const std::vector<double>& got,
                                  const std::vector<double>& reference,
                                  const Criterion& criterion = kConformanceBf16) {
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    double max_ref    = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double err = got[i] - reference[i];
        sum_sq_err += err * err;
        sum_sq_ref += reference[i] * reference[i];
        max_ref         = std::max(max_ref, std::fabs(reference[i]));
    }
    Error error;
    error.relative_l2 =
        sum_sq_ref > 0.0 ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double err = std::fabs(got[i] - reference[i]);
        if (err > criterion.gross_absolute &&
            err > criterion.gross_relative_to_max_reference * max_ref) {
            error.gross += 1.0;
        }
    }
    return error;
}

// One BF16 paged K/V cache with a host-mirrored block table.
struct Cache {
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer table;
    std::vector<std::int32_t> table_host;
    std::int32_t logical_pages = 0;
    std::int32_t physical_pages = 0;

    Cache(std::int32_t logical, const std::vector<std::int32_t>& mapping, std::int32_t kv_heads)
        : table_host(mapping), logical_pages(logical) {
        for (const std::int32_t p : table_host) { physical_pages = std::max(physical_pages, p + 1); }
        const std::size_t elems = static_cast<std::size_t>(kHeadDim) *
                                  static_cast<std::size_t>(kPageSize) *
                                  static_cast<std::size_t>(kv_heads) *
                                  static_cast<std::size_t>(physical_pages);
        k = DeviceBuffer(elems * sizeof(__nv_bfloat16));
        v = DeviceBuffer(elems * sizeof(__nv_bfloat16));
        k.fill(0);
        v.fill(0);
        table = DeviceBuffer(table_host.size() * sizeof(std::int32_t));
        table.copy_from_host(table_host.data(), table_host.size() * sizeof(std::int32_t));
    }
    Cache(const Cache&)            = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&)                 = default;
    Cache& operator=(Cache&&)      = default;

    [[nodiscard]] std::size_t elements_per_page() const {
        return static_cast<std::size_t>(kHeadDim) * kPageSize * kKvHeads;
    }

    [[nodiscard]] std::int32_t physical_page_of(std::int32_t position) const {
        return table_host[static_cast<std::size_t>(position / kPageSize)];
    }

    [[nodiscard]] PagedKVLayerView view(std::int32_t kv_heads = kKvHeads) const {
        PagedKVLayerView result;
        result.k_pages =
            Tensor(k.p, DType::BF16, {kHeadDim, kPageSize, kv_heads, physical_pages});
        result.v_pages =
            Tensor(v.p, DType::BF16, {kHeadDim, kPageSize, kv_heads, physical_pages});
        result.block_table  = Tensor(table.p, DType::I32, {logical_pages});
        result.head_dim     = kHeadDim;
        result.num_kv_heads = kv_heads;
        result.dtype        = DType::BF16;
        return result;
    }

    [[nodiscard]] PagedKVBatchLayerView batch_view() const {
        const PagedKVLayerView direct = view();
        return PagedKVBatchLayerView{.k_pages       = direct.k_pages,
                                     .v_pages       = direct.v_pages,
                                     .k_scale_pages = Tensor{},
                                     .v_scale_pages = Tensor{},
                                     .block_tables  = direct.block_table.view({logical_pages, 1}),
                                     .head_dim      = direct.head_dim,
                                     .num_kv_heads  = direct.num_kv_heads,
                                     .dtype         = direct.dtype,
                                     .quant_group   = 0};
    }

    // A whole physical plane in raw BF16 bits, so a caller can prove the Op touched only one page.
    [[nodiscard]] std::vector<std::uint16_t> k_plane() const {
        std::vector<std::uint16_t> out(elements_per_page() * static_cast<std::size_t>(physical_pages));
        k.copy_to_host(out.data(), out.size() * sizeof(std::uint16_t));
        return out;
    }

    [[nodiscard]] std::vector<std::uint16_t> v_plane() const {
        std::vector<std::uint16_t> out(elements_per_page() * static_cast<std::size_t>(physical_pages));
        v.copy_to_host(out.data(), out.size() * sizeof(std::uint16_t));
        return out;
    }

    // Both planes as FP64 in physical layout, resolved to logical positions on demand.
    struct HostImage {
        std::vector<double> k;
        std::vector<double> v;
    };

    [[nodiscard]] HostImage image() const {
        const std::size_t elems = elements_per_page() * static_cast<std::size_t>(physical_pages);
        std::vector<std::uint16_t> k_bits(elems);
        std::vector<std::uint16_t> v_bits(elems);
        k.copy_to_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
        v.copy_to_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
        HostImage out{std::vector<double>(elems), std::vector<double>(elems)};
        for (std::size_t i = 0; i < elems; ++i) {
            __nv_bfloat16 kk;
            __nv_bfloat16 vv;
            std::memcpy(&kk, &k_bits[i], sizeof(kk));
            std::memcpy(&vv, &v_bits[i], sizeof(vv));
            out.k[i] = static_cast<double>(__bfloat162float(kk));
            out.v[i] = static_cast<double>(__bfloat162float(vv));
        }
        return out;
    }

    [[nodiscard]] double k_logical(const HostImage& img, std::int32_t position, std::int32_t head,
                                   std::int32_t d) const {
        return img.k[plane_index(physical_page_of(position), head, position % kPageSize, d, kKvHeads)];
    }

    [[nodiscard]] double v_logical(const HostImage& img, std::int32_t position, std::int32_t head,
                                   std::int32_t d) const {
        return img.v[plane_index(physical_page_of(position), head, position % kPageSize, d, kKvHeads)];
    }
};

// The three plane models the oracle can decode, selected by the runtime flag combination rather
// than by the storage enum: the oracle's input is a cache view, and the view carries flags.
// `e8_lattice` is deliberately absent from this list -- the E8 projection is encode-side, and the
// read branch of every PackedK variant is gqa_kv_unpack_i4x16 followed by a scale multiply.
enum class KvCodec {
    I8G64,   // full-width int8 K and V, no rotation
    RK8V4,   // full-width int8 K, packed int4 V, K and V Hadamard-rotated
    RK4V4E8, // packed int4 K and V, K and V Hadamard-rotated
};

// Deliberately not constexpr and deliberately not optional<KvCodec>: the e8_root case throws, and a
// throwing body is ill-formed in a constant expression, while a nullopt is something a caller can
// ignore. A variant the oracle cannot decode has to be loud.
[[nodiscard]] inline KvCodec codec_of(bool packed_v, bool rotate_k, bool rotate_v, bool packed_k,
                                      bool e8_lattice, bool e8_root) {
    if (e8_root) {
        // rk2v4-e8's K read path is e8_root_decode_8d_int8: root table + axis table + radius scale
        // beside the packed layout at paged_kv_page_head_offset<64, KVHeads>. That host port is
        // Phase 4d-4; never approximate it with the packed-int4 model below.
        throw std::invalid_argument("codec_of: the rk2v4-e8 oracle is Phase 4d-4");
    }
    // Mirror both wrapper invariants (src/ops/wrapper/gqa_attention.cpp:64-69) so an illegal flag
    // set fails at cache construction instead of aborting mid-run inside gqa_attention_cached.
    if (e8_lattice && !packed_k) {
        throw std::invalid_argument("codec_of: E8 lattice requires packed K");
    }
    if (rotate_v && !packed_v) {
        throw std::invalid_argument("codec_of: rotate_v requires packed V");
    }
    if (packed_k) { return KvCodec::RK4V4E8; }
    if (packed_v || rotate_k || rotate_v) { return KvCodec::RK8V4; }
    return KvCodec::I8G64;
}

// Matches gqa_kv_unpack_i4 exactly: the low nibble is the even dim, the high nibble is the odd dim,
// and the value is a 4-bit two's-complement sign extension, not a biased offset.
[[nodiscard]] inline std::int8_t unpack_i4(std::uint8_t packed, std::int32_t parity) {
    const unsigned nibble = parity ? (packed >> 4) : (packed & 0x0Fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// Host port of gqa_kv_hadamard64 (src/ops/kernel/gqa_attention_kv_quant.cuh:75-91). Device lane L
// holds group elements [L] and [L + 32]; each butterfly stage reads the pre-stage values of lane
// L ^ offset, because __shfl_xor_sync is simultaneous across lanes; and the final stage folds the two
// registers with a 0.125 scale, which leaves the orthonormal transform with a DC gain of 8
// (64 * 0.125) -- a constant group [c, c, ...] maps to [8c, 0, ...]. H is symmetric and involutory
// (H^2 = I), which is why this one function both applies and undoes the rotation; the production V
// path relies on exactly that in gqa_kv_inverse_rotate_output_kernel.
inline void hadamard64_group(double* y) {
    std::array<double, 64> prev{};
    for (int offset = 1; offset < 32; offset <<= 1) {
        std::copy(y, y + 64, prev.begin());
        for (int lane = 0; lane < 32; ++lane) {
            const int  src = lane ^ offset;
            const bool hi  = (lane & offset) != 0;
            y[lane]      = hi ? prev[src] - prev[lane] : prev[lane] + prev[src];
            y[lane + 32] = hi ? prev[src + 32] - prev[lane + 32]
                              : prev[lane + 32] + prev[src + 32];
        }
    }
    for (int lane = 0; lane < 32; ++lane) {
        const double a = y[lane];
        const double b = y[lane + 32];
        y[lane]        = (a + b) * 0.125;
        y[lane + 32]   = (a - b) * 0.125;
    }
}

// Undo the per-64-group Hadamard the write path applies before quantization, on one decoded plane
// whose leading index is d (so each row of kHeadDim elements holds kQuantGroups contiguous groups).
inline void inverse_hadamard_plane(std::vector<double>& plane) {
    static_assert(kQuantGroup == 64, "the host Hadamard port is the 64-point transform");
    static_assert(kQuantGroups * kQuantGroup == kHeadDim, "groups must tile the head dim");
    std::array<double, 64> group{};
    for (std::size_t row = 0; row + static_cast<std::size_t>(kHeadDim) <= plane.size();
         row += static_cast<std::size_t>(kHeadDim)) {
        for (std::int32_t g = 0; g < kQuantGroups; ++g) {
            const std::size_t base = row + static_cast<std::size_t>(g) * kQuantGroup;
            for (std::int32_t i = 0; i < kQuantGroup; ++i) {
                group[static_cast<std::size_t>(i)] = plane[base + static_cast<std::size_t>(i)];
            }
            hadamard64_group(group.data());
            for (std::int32_t i = 0; i < kQuantGroup; ++i) {
                plane[base + static_cast<std::size_t>(i)] = group[static_cast<std::size_t>(i)];
            }
        }
    }
}

// One INT8 paged K/V cache with a host-mirrored block table, covering every int8 storage variant
// whose decode this oracle models. Codes and per-group half scales are separate device buffers,
// matching the shape PagedKVCache::layer_view produces for DType::I8. Both code planes are byte
// storage; whether a plane is full width (DType::I8, leading extent kHeadDim) or packed (DType::U8
// nibble pairs, leading extent kHeadDim / 2) is exactly what the wrapper validates at
// src/ops/wrapper/gqa_attention.cpp:80-86, so the view published below declares that, and image()
// reads each plane with the matching host element type.
struct Int8Cache {
    DeviceBuffer k_codes;   // byte plane, [k_leading(), kPageSize, kKvHeads, physical_pages]
    DeviceBuffer v_codes;   // byte plane, [v_leading(), kPageSize, kKvHeads, physical_pages]
    DeviceBuffer k_scale;   // half plane, [kQuantGroups, kPageSize, kKvHeads, physical_pages]
    DeviceBuffer v_scale;   // half plane, [kQuantGroups, kPageSize, kKvHeads, physical_pages]
    DeviceBuffer table;
    std::vector<std::int32_t> table_host;
    std::int32_t logical_pages  = 0;
    std::int32_t physical_pages = 0;
    KvCodec codec = KvCodec::I8G64;
    bool packed_v = false, rotate_k = false, rotate_v = false, packed_k = false;
    bool e8_lattice = false, e8_root = false;

    Int8Cache(std::int32_t logical, const std::vector<std::int32_t>& mapping,
              std::int32_t kv_heads, bool packed_v_in, bool rotate_k_in, bool rotate_v_in,
              bool packed_k_in, bool e8_lattice_in, bool e8_root_in)
        : table_host(mapping), logical_pages(logical),
          codec(codec_of(packed_v_in, rotate_k_in, rotate_v_in, packed_k_in, e8_lattice_in,
                         e8_root_in)),
          packed_v(packed_v_in), rotate_k(rotate_k_in), rotate_v(rotate_v_in),
          packed_k(packed_k_in), e8_lattice(e8_lattice_in), e8_root(e8_root_in) {
        for (const std::int32_t p : table_host) {
            physical_pages = std::max(physical_pages, p + 1);
        }
        const std::size_t elements_per_k_plane =
            static_cast<std::size_t>(k_leading()) * kPageSize * kv_heads *
            static_cast<std::size_t>(physical_pages);
        const std::size_t elements_per_v_plane =
            static_cast<std::size_t>(v_leading()) * kPageSize * kv_heads *
            static_cast<std::size_t>(physical_pages);
        const std::size_t scale_elements =
            static_cast<std::size_t>(kQuantGroups) * kPageSize * kv_heads *
            static_cast<std::size_t>(physical_pages);
        k_codes  = DeviceBuffer(elements_per_k_plane);
        v_codes  = DeviceBuffer(elements_per_v_plane);
        k_scale  = DeviceBuffer(scale_elements * sizeof(__half));
        v_scale  = DeviceBuffer(scale_elements * sizeof(__half));
        k_codes.fill(0);
        v_codes.fill(0);
        k_scale.fill(0);
        v_scale.fill(0);
        table = DeviceBuffer(table_host.size() * sizeof(std::int32_t));
        table.copy_from_host(table_host.data(), table_host.size() * sizeof(std::int32_t));
    }
    Int8Cache(const Int8Cache&)            = delete;
    Int8Cache& operator=(const Int8Cache&) = delete;
    Int8Cache(Int8Cache&&)                 = default;
    Int8Cache& operator=(Int8Cache&&)      = default;

    // Plane extent and dtype per plane, derived from the codec the constructor selected. These must
    // match what validate_cache / validate_batch_cache require (gqa_attention.cpp:79-86): a packed
    // plane is U8 with leading extent kHeadDim / 2, a full-width plane is the code dtype with
    // leading extent kHeadDim.
    [[nodiscard]] std::int32_t k_leading() const {
        return codec == KvCodec::RK4V4E8 ? kHeadDim / 2 : kHeadDim;
    }
    [[nodiscard]] std::int32_t v_leading() const {
        return codec == KvCodec::I8G64 ? kHeadDim : kHeadDim / 2;
    }
    [[nodiscard]] DType k_dtype() const {
        return codec == KvCodec::RK4V4E8 ? DType::U8 : DType::I8;
    }
    [[nodiscard]] DType v_dtype() const {
        return codec == KvCodec::I8G64 ? DType::I8 : DType::U8;
    }

    [[nodiscard]] PagedKVLayerView view(std::int32_t kv_heads = kKvHeads) const {
        PagedKVLayerView result;
        result.k_pages = Tensor(k_codes.p, k_dtype(),
                                {k_leading(), kPageSize, kv_heads, physical_pages});
        result.v_pages = Tensor(v_codes.p, v_dtype(),
                                {v_leading(), kPageSize, kv_heads, physical_pages});
        result.k_scale_pages = Tensor(k_scale.p, DType::FP16,
                                      {kQuantGroups, kPageSize, kv_heads, physical_pages});
        result.v_scale_pages = Tensor(v_scale.p, DType::FP16,
                                      {kQuantGroups, kPageSize, kv_heads, physical_pages});
        result.block_table  = Tensor(table.p, DType::I32, {logical_pages});
        result.head_dim     = kHeadDim;
        result.num_kv_heads = kv_heads;
        result.dtype        = DType::I8;
        result.quant_group  = kQuantGroup;
        result.packed_v     = packed_v;
        result.rotate_k     = rotate_k;
        result.rotate_v     = rotate_v;
        result.packed_k     = packed_k;
        result.e8_lattice   = e8_lattice;
        result.e8_root      = e8_root;
        return result;
    }

    // The multi-sequence form of the same planes, mirroring Cache::batch_view(): identical plane
    // tensors and flags, the block table already reshaped to one row, and rope_deltas left unset for
    // the caller to attach (which is what the prefill path needs to carry the delta plane).
    [[nodiscard]] PagedKVBatchLayerView batch_view(std::int32_t kv_heads = kKvHeads) const {
        const PagedKVLayerView direct = view(kv_heads);
        return PagedKVBatchLayerView{.k_pages       = direct.k_pages,
                                     .v_pages       = direct.v_pages,
                                     .k_scale_pages = direct.k_scale_pages,
                                     .v_scale_pages = direct.v_scale_pages,
                                     .block_tables  = direct.block_table.view({logical_pages, 1}),
                                     .rope_deltas   = Tensor{},
                                     .rope_inv_freq = direct.rope_inv_freq,
                                     .head_dim      = direct.head_dim,
                                     .num_kv_heads  = direct.num_kv_heads,
                                     .dtype         = direct.dtype,
                                     .quant_group   = direct.quant_group,
                                     .packed_v      = direct.packed_v,
                                     .rotate_k      = direct.rotate_k,
                                     .rotate_v      = direct.rotate_v,
                                     .packed_k      = direct.packed_k,
                                     .e8_lattice    = direct.e8_lattice,
                                     .e8_root       = direct.e8_root};
    }

    // Decode the code planes with their half per-group scales, exactly per the contract in
    // include/ninfer/ops/gqa_attention.h lines 24-32:
    //   decode[i] = FP32(code[i]) * FP32(scale[group(i)])
    // A full-width plane holds signed int8 codes; a packed plane holds int4 nibble pairs, which must
    // go through unpack_i4 and never through a direct cast. A rotated variant's codes hold
    // Hadamard-rotated values, so the decoded plane is then passed through the inverse Hadamard one
    // 64-dim group at a time -- which is what makes this oracle's raw-BF16 query comparable to the
    // kernel's rotated-and-quantized one. The returned image is [physical page][kv head][page
    // offset][d], the same plane order k_logical / v_logical index into below.
    struct HostImage {
        std::vector<double> k;
        std::vector<double> v;
    };

    [[nodiscard]] HostImage image() const {
        const std::size_t elems = static_cast<std::size_t>(kHeadDim) * kPageSize * kKvHeads *
                                  static_cast<std::size_t>(physical_pages);
        const std::size_t scale_elems =
            static_cast<std::size_t>(kQuantGroups) * kPageSize * kKvHeads *
            static_cast<std::size_t>(physical_pages);
        const bool k_full = k_leading() == kHeadDim;
        const bool v_full = v_leading() == kHeadDim;

        // Each plane is read with the element type its published view declares, so the host vector's
        // type carries the interpretation: a plain static_cast<double> is only ever correct for a
        // signed full-width plane, and a packed plane only ever goes through unpack_i4.
        std::vector<std::int8_t>  k_i8;
        std::vector<std::int8_t>  v_i8;
        std::vector<std::uint8_t> k_u8;
        std::vector<std::uint8_t> v_u8;
        if (k_full) {
            k_i8.resize(elems);
            k_codes.copy_to_host(k_i8.data(), elems * sizeof(std::int8_t));
        } else {
            k_u8.resize(elems / 2);
            k_codes.copy_to_host(k_u8.data(), k_u8.size() * sizeof(std::uint8_t));
        }
        if (v_full) {
            v_i8.resize(elems);
            v_codes.copy_to_host(v_i8.data(), elems * sizeof(std::int8_t));
        } else {
            v_u8.resize(elems / 2);
            v_codes.copy_to_host(v_u8.data(), v_u8.size() * sizeof(std::uint8_t));
        }
        std::vector<__half> k_scale_host(scale_elems);
        std::vector<__half> v_scale_host(scale_elems);
        k_scale.copy_to_host(k_scale_host.data(), scale_elems * sizeof(__half));
        v_scale.copy_to_host(v_scale_host.data(), scale_elems * sizeof(__half));

        HostImage out{std::vector<double>(elems), std::vector<double>(elems)};
        // The element index runs [d, page_offset, kv_head, physical_page] in the plane, and the group
        // index is d / kQuantGroup. This matches paged_kv_element_offset for the code plane (leading
        // dim kHeadDim, or kHeadDim / 2 when packed) and for the scale plane (leading dim
        // kQuantGroups).
        for (std::int32_t page = 0; page < physical_pages; ++page) {
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                for (std::int32_t offset = 0; offset < kPageSize; ++offset) {
                    const std::size_t row =
                        (static_cast<std::size_t>(page) * kKvHeads + head) * kPageSize + offset;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        const std::size_t elem_idx = row * kHeadDim + static_cast<std::size_t>(d);
                        const std::int32_t group   = d / kQuantGroup;
                        const std::size_t scale_idx =
                            row * kQuantGroups + static_cast<std::size_t>(group);
                        const double ks = static_cast<double>(__half2float(k_scale_host[scale_idx]));
                        const double vs = static_cast<double>(__half2float(v_scale_host[scale_idx]));
                        // Packed planes are indexed with their own leading extent, kHeadDim / 2, so
                        // d / 2 lands on the byte holding dims {d & ~1, d | 1}.
                        out.k[elem_idx] =
                            k_full ? static_cast<double>(
                                         k_i8[row * static_cast<std::size_t>(k_leading()) +
                                              static_cast<std::size_t>(d)]) *
                                         ks
                                   : static_cast<double>(unpack_i4(
                                         k_u8[row * static_cast<std::size_t>(k_leading()) +
                                              static_cast<std::size_t>(d / 2)],
                                         d & 1)) *
                                         ks;
                        out.v[elem_idx] =
                            v_full ? static_cast<double>(
                                         v_i8[row * static_cast<std::size_t>(v_leading()) +
                                              static_cast<std::size_t>(d)]) *
                                         vs
                                   : static_cast<double>(unpack_i4(
                                         v_u8[row * static_cast<std::size_t>(v_leading()) +
                                              static_cast<std::size_t>(d / 2)],
                                         d & 1)) *
                                         vs;
                    }
                }
            }
        }
        if (rotate_k) { inverse_hadamard_plane(out.k); }
        if (rotate_v) { inverse_hadamard_plane(out.v); }
        return out;
    }

    [[nodiscard]] std::int32_t physical_page_of(std::int32_t position) const {
        return table_host[static_cast<std::size_t>(position / kPageSize)];
    }

    [[nodiscard]] std::size_t plane_index(std::int32_t physical_page, std::int32_t kv_head,
                                          std::int32_t page_offset, std::int32_t d) const {
        return ((static_cast<std::size_t>(physical_page) * kKvHeads + kv_head) * kPageSize +
                page_offset) * kHeadDim +
               d;
    }

    [[nodiscard]] double k_logical(const HostImage& img, std::int32_t position,
                                   std::int32_t head, std::int32_t d) const {
        return img.k[plane_index(physical_page_of(position), head, position % kPageSize, d)];
    }

    [[nodiscard]] double v_logical(const HostImage& img, std::int32_t position,
                                   std::int32_t head, std::int32_t d) const {
        return img.v[plane_index(physical_page_of(position), head, position % kPageSize, d)];
    }
};

// Append BF16 host data through the production i8 write path. The fill kernel quantizes on
// the fly; this test never quantizes by hand. Positions are sequential from first_position.
inline void append_chunk(Int8Cache& cache, const std::vector<__nv_bfloat16>& k,
                         const std::vector<__nv_bfloat16>& v, std::int32_t first_position,
                         std::int32_t tokens) {
    DeviceBuffer dk(k.size() * sizeof(__nv_bfloat16));
    DeviceBuffer dv(v.size() * sizeof(__nv_bfloat16));
    DeviceBuffer dp(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    dk.copy_from_host(k.data(), k.size() * sizeof(__nv_bfloat16));
    dv.copy_from_host(v.data(), v.size() * sizeof(__nv_bfloat16));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = first_position + t;
    }
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));

    ops::gqa_kv_append(Tensor(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                       Tensor(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                       Tensor(dp.p, DType::I32, {tokens}), cache.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
}

// Appends one chunk through the production write path. k/v are host activations in
// [head][token][d] order, positions are sequential from first_position.
inline void append_chunk(Cache& cache, const std::vector<__nv_bfloat16>& k,
                         const std::vector<__nv_bfloat16>& v, std::int32_t first_position,
                         std::int32_t tokens) {
    DeviceBuffer dk(k.size() * sizeof(__nv_bfloat16));
    DeviceBuffer dv(v.size() * sizeof(__nv_bfloat16));
    DeviceBuffer dp(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    dk.copy_from_host(k.data(), k.size() * sizeof(__nv_bfloat16));
    dv.copy_from_host(v.data(), v.size() * sizeof(__nv_bfloat16));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = first_position + t;
    }
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));

    ops::gqa_kv_append(Tensor(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                       Tensor(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                       Tensor(dp.p, DType::I32, {tokens}), cache.view(), nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
}

inline DeviceBuffer upload(const std::vector<__nv_bfloat16>& values) {
    DeviceBuffer buffer(values.size() * sizeof(__nv_bfloat16));
    buffer.copy_from_host(values.data(), values.size() * sizeof(__nv_bfloat16));
    return buffer;
}

inline DeviceBuffer upload_i32(const std::vector<std::int32_t>& values) {
    DeviceBuffer buffer(values.size() * sizeof(std::int32_t));
    buffer.copy_from_host(values.data(), values.size() * sizeof(std::int32_t));
    return buffer;
}

inline DeviceBuffer upload_positions(const std::vector<std::int32_t>& values) {
    DeviceBuffer buffer(values.size() * sizeof(std::int32_t));
    buffer.copy_from_host(values.data(), values.size() * sizeof(std::int32_t));
    return buffer;
}

inline std::vector<std::int32_t> sequential_positions(std::int32_t first, std::int32_t count) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(count));
    for (std::int32_t t = 0; t < count; ++t) { out[static_cast<std::size_t>(t)] = first + t; }
    return out;
}

inline std::vector<double> download_doubles(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<__nv_bfloat16> host(count);
    buffer.copy_to_host(host.data(), count * sizeof(__nv_bfloat16));
    std::vector<double> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(__bfloat162float(host[i]));
    }
    return out;
}

struct AttentionRun {
    std::vector<double> out;
};

// A3: read-only attention over an already populated cache.
inline AttentionRun run_cached_attention(const Cache& cache, const std::vector<__nv_bfloat16>& q,
                                         std::int32_t first_position, std::int32_t tokens,
                                         std::uint32_t visible) {
    DeviceBuffer dq   = upload(q);
    DeviceBuffer dout = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));

    const ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQHeads, DType::BF16, envelope, 1, tokens,
                                                   tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

// Same as run_cached_attention, but takes an explicit view so callers can attach
// rope_deltas and rope_inv_freq. Routes through the public gqa_attention_cached,
// which forwards the view's delta plane and freq table to the kernel.
inline AttentionRun run_cached_attention_view(const Cache& cache,
                                              const std::vector<__nv_bfloat16>& q,
                                              std::int32_t first_position, std::int32_t tokens,
                                              std::uint32_t visible,
                                              const PagedKVLayerView& view) {
    DeviceBuffer dq   = upload(q);
    DeviceBuffer dout = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));

    const ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQHeads, DType::BF16, envelope, 1, tokens,
                                                   tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention_cached(tq, tp, kAttentionScale, view, envelope, workspace, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

// Same as run_cached_attention_view, but for an Int8Cache. Workspace sizing uses DType::I8
// so the split count matches what gqa_attention_cached will actually dispatch.
inline AttentionRun run_cached_attention_view_i8(const Int8Cache& cache,
                                                 const std::vector<__nv_bfloat16>& q,
                                                 std::int32_t first_position,
                                                 std::int32_t tokens,
                                                 std::uint32_t visible,
                                                 const PagedKVLayerView& view) {
    DeviceBuffer dq   = upload(q);
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
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention_cached(tq, tp, kAttentionScale, view, envelope, workspace, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

// A1: the append-and-attend route the prefill uses. This is the same shape as the
// integration test's original helper: append a chunk through the production write path,
// then run public gqa_attention over the batch view.
inline AttentionRun run_append_attention(Cache& cache, const std::vector<__nv_bfloat16>& q,
                                         const std::vector<__nv_bfloat16>& k,
                                         const std::vector<__nv_bfloat16>& v,
                                         std::int32_t first_position, std::int32_t tokens,
                                         std::uint32_t visible) {
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
        ops::gqa_attention_workspace_capacity_bytes(kQHeads, DType::BF16, envelope, 1, tokens,
                                                   tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor ttable(dtable.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

// The attention core both FP64 oracles share, so the delta form cannot drift from the plain one:
// the BF16 inputs are taken as exact and the dot products, stable softmax and value reduction all
// run in FP64, per the numerical contract in gqa_attention.h. Keys are read from a
// [kv_head][position][d] plane the caller has already resolved to the reference position; V comes
// through the cache's own logical view.
template <typename CacheType>
inline std::vector<double> attention_against_key_plane(const std::vector<__nv_bfloat16>& q,
                                                       const CacheType& cache,
                                                       const typename CacheType::HostImage& img,
                                                       const std::vector<double>& key_plane,
                                                       std::int32_t first_position,
                                                       std::int32_t tokens) {
    const std::int32_t max_position = first_position + tokens - 1;
    std::vector<double> reference(q.size());
    std::vector<double> head_q(static_cast<std::size_t>(kHeadDim));
    std::vector<double> weights(static_cast<std::size_t>(max_position) + 1);
    const auto key_at = [&](std::int32_t x, std::int32_t kv_head, std::int32_t d) {
        return key_plane[(static_cast<std::size_t>(kv_head) * (max_position + 1) + x) * kHeadDim +
                         d];
    };

    for (std::int32_t q_head = 0; q_head < kQHeads; ++q_head) {
        const std::int32_t kv_head = q_head / kQGroup;
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::int32_t position = first_position + t;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                head_q[static_cast<std::size_t>(d)] = static_cast<double>(
                    __bfloat162float(q[act_index(q_head, t, kQHeads, d)]));
            }
            // Bottom-right causal alignment: query row t attends to keys [0, position].
            double peak = -std::numeric_limits<double>::infinity();
            for (std::int32_t x = 0; x <= position; ++x) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += head_q[static_cast<std::size_t>(d)] * key_at(x, kv_head, d);
                }
                const double score = static_cast<double>(kAttentionScale) * dot;
                weights[static_cast<std::size_t>(x)] = score;
                peak                                = std::max(peak, score);
            }
            double total = 0.0;
            for (std::int32_t x = 0; x <= position; ++x) {
                weights[static_cast<std::size_t>(x)] =
                    std::exp(weights[static_cast<std::size_t>(x)] - peak);
                total += weights[static_cast<std::size_t>(x)];
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (std::int32_t x = 0; x <= position; ++x) {
                    acc += (weights[static_cast<std::size_t>(x)] / total) *
                           cache.v_logical(img, x, kv_head, d);
                }
                reference[act_index(q_head, t, kQHeads, d)] = acc;
            }
        }
    }
    return reference;
}

// Ideal GQA attention in FP64 for a cache whose keys are read exactly as stored.
template <typename CacheType>
inline std::vector<double> ideal_attention(const std::vector<__nv_bfloat16>& q,
                                           const CacheType& cache,
                                           std::int32_t first_position, std::int32_t tokens) {
    const typename CacheType::HostImage img = cache.image();
    const std::int32_t max_position         = first_position + tokens - 1;
    std::vector<double> keys(static_cast<std::size_t>(kKvHeads) * (max_position + 1) * kHeadDim);
    for (std::int32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (std::int32_t x = 0; x <= max_position; ++x) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                keys[(static_cast<std::size_t>(kv_head) * (max_position + 1) + x) * kHeadDim + d] =
                    cache.k_logical(img, x, kv_head, d);
            }
        }
    }
    return attention_against_key_plane(q, cache, img, keys, first_position, tokens);
}

// The delta-plane oracle: the FP64 reference for a cache whose keys were quantized at the *source* RoPE
// phase and are read at the destination positions. The kernel counter-rotates Q by -delta at its load
// site (gqa_attention_decode_i8.cuh:428-433, group 0 only -- the rotary region is dims
// [0, 2 * kRotaryPairs)), and R is orthonormal, so the same equality can be evaluated on the key side
// and stay exact in FP64:
//
//     <R(-delta) q, K_src> == <q, R(delta) K_src>
//
// forward_rope(., +delta) is the exact inverse of the kernel's counter-rotation (it applies R(+delta)
// with the same per-pair convention build_scenario uses), and composing it with the stored R(src + t) key
// yields the R(dst + t) key a direct prefill would have written. That composition is what makes this
// the *anchored* form of the relocation check: it rotates the decoded key, so it models the encode
// error of the plane exactly, and a skipped rotation cannot hide behind a coarse plane's own noise the
// way it does when the reference is another cache.
//
// page_deltas is the view's rope_deltas plane, indexed by logical page, and is read per key -- the same
// plane the kernel indexes for the K tiles it processes.
template <typename CacheType>
inline std::vector<double> ideal_attention_with_delta(
    const std::vector<__nv_bfloat16>& q, const CacheType& cache, std::int32_t first_position,
    std::int32_t tokens, const std::vector<std::int32_t>& page_deltas, const FreqTable& freq) {
    const typename CacheType::HostImage img = cache.image();
    const std::int32_t max_position         = first_position + tokens - 1;
    std::vector<double> keys(static_cast<std::size_t>(kKvHeads) * (max_position + 1) * kHeadDim);
    std::array<double, kHeadDim> stored{};
    for (std::int32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (std::int32_t x = 0; x <= max_position; ++x) {
            double* row = &keys[(static_cast<std::size_t>(kv_head) * (max_position + 1) + x) *
                                kHeadDim];
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                stored[static_cast<std::size_t>(d)] = cache.k_logical(img, x, kv_head, d);
            }
            // delta == 0 is the identity in this convention, so a page without a delta passes
            // through unchanged and needs no branch.
            forward_rope(stored.data(), page_deltas[static_cast<std::size_t>(x / kPageSize)], freq,
                         row);
        }
    }
    return attention_against_key_plane(q, cache, img, keys, first_position, tokens);
}

// A correctly relocated cache must be operationally indistinguishable from a directly prefilled one.
// The K rows differ by at most the one extra BF16 rounding the re-rotation introduces, which the
// attention output cannot amplify into anything like this. A page whose K still carries the source
// positions' rotation is not a near miss, so the two populations are orders of magnitude apart; the
// negative control below measures that separation on the same data instead of assuming it.
constexpr double kEquivalenceLimit = 2.0e-2;

// The delta-plane oracle test inherits the integration test's observed control separation. The
// integration test observes ~300x separation with delta = -384 against the same reference. To keep the
// two tests' control semantics identical, the oracle test requires the zero-delta error to clear a
// floor computed from the correct error by the same multiplicative factor.
constexpr double kControlFactor = 10.0;

inline bool expect_equivalent(const Error& error, const std::string& label) {
    char buf[256];
    if (error.relative_l2 > kEquivalenceLimit) {
        std::snprintf(buf, sizeof(buf),
                      "%s: relative L2 %.3e from the directly prefilled cache, over the %.2e limit",
                      label.c_str(), error.relative_l2, kEquivalenceLimit);
        std::printf("  FAIL: %s\n", buf);
        return false;
    }
    std::snprintf(buf, sizeof(buf), "%s: ok (relative L2 %.3e, limit %.2e)", label.c_str(),
                  error.relative_l2, kEquivalenceLimit);
    std::printf("  %s\n", buf);
    return true;
}

inline bool expect_conformant(const Error& error, const std::string& label,
                              const Criterion& criterion = kConformanceBf16) {
    char buf[256];
    if (error.relative_l2 > criterion.relative_l2 || error.gross > 0.0) {
        std::snprintf(buf, sizeof(buf),
                      "%s: relative L2 %.3e (criterion %.2e) with %d gross errors; the harness "
                      "itself is not reading the cache the way the oracle describes it",
                      label.c_str(), error.relative_l2, criterion.relative_l2,
                      static_cast<int>(error.gross));
        std::printf("  FAIL: %s\n", buf);
        return false;
    }
    std::snprintf(buf, sizeof(buf), "%s: ok (relative L2 %.3e against the FP64 oracle)", label.c_str(),
                  error.relative_l2);
    std::printf("  %s\n", buf);
    return true;
}

// The anchored control: the correct run has to conform to the criterion and the control run has to
// break the same criterion, so the discrimination is stated at the resolution of the criterion
// itself rather than as a multiple of the correct path's own noise. This is what a variant with a
// coarse (int4) K plane needs: there the correct path's error against a directly prefilled cache is
// dominated by the encode error of a plane that was quantized at the source RoPE phase, so the
// multiplicative separation against a zero-delta run collapses even though the relocation is right.
// Pair it with `expect_conformant` on the same oracle and the two together say "the correct path
// conforms, the skipped rotation does not".
inline bool expect_control_outside(const Error& control, const std::string& label,
                                   const Criterion& criterion) {
    char buf[256];
    if (control.relative_l2 > criterion.relative_l2) {
        std::snprintf(buf, sizeof(buf),
                      "%s: ok (control relative L2 %.3e, %.0fx the %.2e criterion it must break%s)",
                      label.c_str(), control.relative_l2,
                      control.relative_l2 / criterion.relative_l2, criterion.relative_l2,
                      control.gross > 0.0 ? ", with gross errors" : "");
        std::printf("  %s\n", buf);
        return true;
    }
    std::snprintf(buf, sizeof(buf),
                  "%s: control only reached relative L2 %.3e, inside the %.2e criterion, so the "
                  "anchor it is paired with is not discriminating a skipped rotation at all",
                  label.c_str(), control.relative_l2, criterion.relative_l2);
    std::printf("  FAIL: %s\n", buf);
    return false;
}

// The multiplicative control, kept for the plain-int8 and BF16 delta tests, whose correct-path error
// is small enough for a 10x separation to be measurable. The control has to clear the limit by a wide
// factor, otherwise the limit is measuring nothing and the test would pass on a broken kernel.
inline bool expect_control_separates(const Error& correct, const Error& control, const std::string& label) {
    char buf[256];
    const double floor_error = std::max(correct.relative_l2 * kControlFactor, kEquivalenceLimit);
    if (control.relative_l2 <= floor_error) {
        std::snprintf(buf, sizeof(buf),
                      "%s: control only reached relative L2 %.3e, under the %.3e it had to clear, "
                      "so this harness cannot see a skipped rotation and the limit above proves nothing",
                      label.c_str(), control.relative_l2, floor_error);
        std::printf("  FAIL: %s\n", buf);
        return false;
    }
    std::snprintf(buf, sizeof(buf),
                  "%s: ok (control relative L2 %.3e versus %.3e correct, %.0fx the limit)", label.c_str(),
                  control.relative_l2, correct.relative_l2, control.relative_l2 / floor_error);
    std::printf("  %s\n", buf);
    return true;
}

// The scenario: a 64-token run that prefill wrote at `src` must, after re-pointing the destination
// logical page at the source's physical page, be what a direct prefill at `dst` would have written.
constexpr std::int32_t kLogicalPages = 16;
constexpr std::int32_t kSourcePage   = 9;
constexpr std::int32_t kDestPage     = 3;
constexpr std::int32_t kRelocatedRun = 64;
constexpr std::int32_t kQueryTokens  = 16;
// The queries sit at the end of the destination page, so every relocated key is visible.
constexpr std::int32_t kQueryBase = kDestPage * kPageSize + kRelocatedRun - kQueryTokens;

struct Scenario {
    std::vector<std::int32_t> table;
    std::int32_t src                  = kSourcePage * kPageSize;
    std::int32_t dst                  = kDestPage * kPageSize;
    std::vector<__nv_bfloat16> raw_k; // pre-RoPE, [head][token][d]
    std::vector<__nv_bfloat16> k_src; // bf16(rope(raw, src + t))
    std::vector<__nv_bfloat16> k_dst; // bf16(rope(raw, dst + t))
    std::vector<__nv_bfloat16> v;     // identical in both caches, so K is the only difference
    std::vector<__nv_bfloat16> q;
};

inline Scenario build_scenario(const FreqTable& freq) {
    Scenario scenario;
    // A non-identity permutation, so logical page N is not physical page N and the Op has to consult
    // the block table.
    scenario.table.resize(static_cast<std::size_t>(kLogicalPages));
    for (std::int32_t i = 0; i < kLogicalPages; ++i) {
        scenario.table[static_cast<std::size_t>(i)] = (i * 7 + 3) % kLogicalPages;
    }
    // reassign_pages: the destination logical page now points at the surviving physical page.
    scenario.table[static_cast<std::size_t>(kDestPage)] =
        scenario.table[static_cast<std::size_t>(kSourcePage)];

    const std::size_t elems =
        static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(kRelocatedRun);
    scenario.raw_k.resize(elems);
    scenario.k_src.resize(elems);
    scenario.k_dst.resize(elems);
    scenario.v.resize(elems);

    std::mt19937 rng(20261007u);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::array<double, kHeadDim> row{};
    std::array<double, kHeadDim> rotated{};
    for (std::int32_t t = 0; t < kRelocatedRun; ++t) {
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                row[static_cast<std::size_t>(d)] = normal(rng);
                scenario.raw_k[act_index(head, t, kKvHeads, d)] =
                    __float2bfloat16(static_cast<float>(row[static_cast<std::size_t>(d)]));
            }
            forward_rope(row.data(), scenario.src + t, freq, rotated.data());
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                scenario.k_src[act_index(head, t, kKvHeads, d)] =
                    __float2bfloat16(static_cast<float>(rotated[static_cast<std::size_t>(d)]));
            }
            forward_rope(row.data(), scenario.dst + t, freq, rotated.data());
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                scenario.k_dst[act_index(head, t, kKvHeads, d)] =
                    __float2bfloat16(static_cast<float>(rotated[static_cast<std::size_t>(d)]));
            }
        }
    }
    // V depends only on the run index, so the two caches carry byte-identical values once the K
    // relocation is correct. Without this the oracle would also be comparing stale V.
    std::uniform_real_distribution<double> unit(-0.25, 0.25);
    for (std::size_t i = 0; i < scenario.v.size(); ++i) {
        scenario.v[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }

    const std::size_t q_elems =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(kQueryTokens);
    scenario.q.resize(q_elems);
    for (std::size_t i = 0; i < q_elems; ++i) {
        scenario.q[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }
    return scenario;
}

// A cache holding what a direct prefill at the destination would have written, reachable as a value.
inline Cache make_reference_cache(const Scenario& scenario) {
    Cache cache(kLogicalPages, scenario.table, kKvHeads);
    append_chunk(cache, scenario.k_dst, scenario.v, scenario.dst, kRelocatedRun);
    return cache;
}
