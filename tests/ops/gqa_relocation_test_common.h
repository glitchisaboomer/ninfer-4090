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

// One INT8-G64 paged K/V cache with a host-mirrored block table. K/V codes and per-group
// half scales are separate device buffers, matching the shape PagedKVCache::layer_view
// produces for DType::I8. The flag combination (packed_v / rotate_k / rotate_v / packed_k /
// e8_lattice / e8_root) is the combination for KvCacheStorage::Int8Group64, confirmed in
// Part 0a; see layouts_impl.h around lines 711-728 and the constants in this test's
// constructor.
struct Int8Cache {
    DeviceBuffer k_codes;   // int8 plane, [kHeadDim, kPageSize, kKvHeads, physical_pages]
    DeviceBuffer v_codes;   // int8 plane, same layout; the uint8_t* in the fill kernel is
                            // a two's-complement write alias, not a different dtype
    DeviceBuffer k_scale;   // half plane,  [kQuantGroups, kPageSize, kKvHeads, physical_pages]
    DeviceBuffer v_scale;   // half plane,  [kQuantGroups, kPageSize, kKvHeads, physical_pages]
    DeviceBuffer table;
    std::vector<std::int32_t> table_host;
    std::int32_t logical_pages  = 0;
    std::int32_t physical_pages = 0;
    bool packed_v = false, rotate_k = false, rotate_v = false, packed_k = false;
    bool e8_lattice = false, e8_root = false;

    Int8Cache(std::int32_t logical, const std::vector<std::int32_t>& mapping,
              std::int32_t kv_heads, bool packed_v_in, bool rotate_k_in, bool rotate_v_in,
              bool packed_k_in, bool e8_lattice_in, bool e8_root_in)
        : table_host(mapping), logical_pages(logical), packed_v(packed_v_in),
          rotate_k(rotate_k_in), rotate_v(rotate_v_in), packed_k(packed_k_in),
          e8_lattice(e8_lattice_in), e8_root(e8_root_in) {
        for (const std::int32_t p : table_host) {
            physical_pages = std::max(physical_pages, p + 1);
        }
        const std::size_t elements_per_plane =
            static_cast<std::size_t>(kHeadDim) * kPageSize * kv_heads *
            static_cast<std::size_t>(physical_pages);
        const std::size_t scale_elements =
            static_cast<std::size_t>(kQuantGroups) * kPageSize * kv_heads *
            static_cast<std::size_t>(physical_pages);
        k_codes  = DeviceBuffer(elements_per_plane * sizeof(std::int8_t));
        v_codes  = DeviceBuffer(elements_per_plane * sizeof(std::int8_t));
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

    [[nodiscard]] PagedKVLayerView view(std::int32_t kv_heads = kKvHeads) const {
        PagedKVLayerView result;
        result.k_pages = Tensor(k_codes.p, DType::I8,
                                {kHeadDim, kPageSize, kv_heads, physical_pages});
        result.v_pages = Tensor(v_codes.p, DType::I8,
                                {kHeadDim, kPageSize, kv_heads, physical_pages});
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

    // Decode the int8 codes with their half scale, exactly per the contract in
    // include/ninfer/ops/gqa_attention.h lines 24-32:
    //   decode[i] = FP32(code[i]) * FP32(scale[group(i)])
    // Both K and V codes are signed; the returned image is [physical page][kv head][page
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
        std::vector<std::int8_t> k_codes_host(elems);
        std::vector<std::int8_t> v_codes_host(elems);
        std::vector<__half>      k_scale_host(scale_elems);
        std::vector<__half>      v_scale_host(scale_elems);
        k_codes.copy_to_host(k_codes_host.data(), elems * sizeof(std::int8_t));
        v_codes.copy_to_host(v_codes_host.data(), elems * sizeof(std::int8_t));
        k_scale.copy_to_host(k_scale_host.data(), scale_elems * sizeof(__half));
        v_scale.copy_to_host(v_scale_host.data(), scale_elems * sizeof(__half));

        HostImage out{std::vector<double>(elems), std::vector<double>(elems)};
        // The leading element index runs [d, page_offset, kv_head, physical_page] in the plane;
        // the group index is d / kQuantGroup. This matches paged_kv_element_offset for both
        // the code and scale planes when the leading dim is head_dim and groups respectively.
        for (std::int32_t page = 0; page < physical_pages; ++page) {
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                for (std::int32_t offset = 0; offset < kPageSize; ++offset) {
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        const std::size_t code_idx =
                            ((static_cast<std::size_t>(page) * kKvHeads + head) * kPageSize + offset) *
                                kHeadDim +
                            d;
                        const std::int32_t group = d / kQuantGroup;
                        const std::size_t scale_idx =
                            ((static_cast<std::size_t>(page) * kKvHeads + head) * kPageSize + offset) *
                                kQuantGroups +
                            group;
                        out.k[code_idx] = static_cast<double>(k_codes_host[code_idx]) *
                                          static_cast<double>(__half2float(k_scale_host[scale_idx]));
                        out.v[code_idx] = static_cast<double>(v_codes_host[code_idx]) *
                                          static_cast<double>(__half2float(v_scale_host[scale_idx]));
                    }
                }
            }
        }
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

// Ideal GQA attention in FP64: the BF16 inputs are taken as exact and dot products, stable softmax
// and the value reduction all run in FP64, per the numerical contract in gqa_attention.h.
template <typename CacheType>
inline std::vector<double> ideal_attention(const std::vector<__nv_bfloat16>& q,
                                           const CacheType& cache,
                                           std::int32_t first_position, std::int32_t tokens) {
    const typename CacheType::HostImage img = cache.image();
    const std::int32_t max_position    = first_position + tokens - 1;
    std::vector<double> reference(q.size());
    std::vector<double> head_q(static_cast<std::size_t>(kHeadDim));
    std::vector<double> weights(static_cast<std::size_t>(max_position) + 1);

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
                    dot += head_q[static_cast<std::size_t>(d)] *
                           cache.k_logical(img, x, kv_head, d);
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

// The control has to clear the limit by a wide factor, otherwise the limit is measuring nothing and
// the test would pass on a broken kernel.
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
