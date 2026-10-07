#pragma once

// Implements: include/ninfer/ops/gqa_kv_rerotate.h
//
// In-place re-rotation of already-cached K for the Text KV geometry. One warp owns one logical
// position and walks every KV head of that position; lane `pair` owns the rotary element pair
// (pair, pair + kGqaRerotateHalf).
//
// The inverse-frequency table and the FP32 sincos convention are taken from rope.cuh so that a
// re-rotation composes with the forward rotation the cached K already carries. The pairing is the
// one apply_rope_head uses: for every rotary element e in [0, 32), element e pairs with element
// e + 32 and uses frequency entry e.
//
// Only the Text paged plane is served, because that plane's element order is the one
// paged_kv_address.cuh describes. The DFlash paged plane is laid out head-major
// ([kv_head][physical_page][64][head_dim]) while paged_kv_element_offset assumes
// [physical_page][kv_head][page][head_dim], so DFlash needs its own addressing and is deliberately
// not accepted here.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaRerotateHeadDim = 256;
inline constexpr int kGqaRerotateHalf    = 32; // rotary_dim 64, so 32 rotated element planes.

// One warp per logical position, 32 rotary pairs per warp, KVHeads heads walked per pair.
template <int KVHeads>
__global__ void gqa_kv_rerotate_kernel(const std::int32_t* __restrict__ positions,
                                       const std::int32_t* __restrict__ rope_deltas,
                                       __nv_bfloat16* __restrict__ cache_k,
                                       const std::int32_t* __restrict__ block_table,
                                       std::int32_t tokens) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    const int lane = static_cast<int>(threadIdx.x) & 31;

    const int position               = positions[token];
    const std::int32_t delta         = rope_deltas[token];

    // One lane resolves the physical page and every lane reads it, so the block table is touched
    // once per position instead of once per rotary pair.
    int physical_page = 0;
    if (lane == 0) { physical_page = paged_kv_physical_page(block_table, position); }
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);

    // Only the rotary region [0, 64) of the 256-element row is touched; the remaining elements
    // carry no rotation and are left byte-identical.
    const int page_offset = position & kPagedKVPageMask;
    const int pair        = lane;
    float sine            = 0.0F;
    float cosine          = 0.0F;
    sincosf(static_cast<float>(delta) * kTextRopeInvFrequency[pair], &sine, &cosine);

    for (int kv_head = 0; kv_head < KVHeads; ++kv_head) {
        __nv_bfloat16* row =
            cache_k + paged_kv_element_offset<kGqaRerotateHeadDim, KVHeads>(physical_page, kv_head,
                                                                           page_offset, 0);
        const float first  = __bfloat162float(row[pair]);
        const float second = __bfloat162float(row[pair + kGqaRerotateHalf]);
        row[pair]          = __float2bfloat16_rn(first * cosine - second * sine);
        row[pair + kGqaRerotateHalf] = __float2bfloat16_rn(second * cosine + first * sine);
    }
}

} // namespace ninfer::ops
