#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: rotate already-cached K in place by a per-logical-position delta.
 *
 * This is the Suffix Cache Reuse relocation primitive. Cached K is rotated at write time for the
 * position a token occupied when it was prefilled. When an edit moves a surviving run of tokens to
 * new logical positions, their cached K still carries the old rotation and must be re-rotated
 * before attention can consume it at the new positions.
 *
 * Math / indexing:
 *   For every listed logical position p and rotary element e in [0, 32):
 *     c   = cos(delta[p] * freq[e])
 *     s   = sin(delta[p] * freq[e])
 *     x   = K[p][:, e]        y = K[p][:, e + 32]
 *     K'[p][:, e]      = x * c - y * s
 *     K'[p][:, e + 32] = y * c + x * s
 *
 *   `freq` is the Text inverse-frequency table the forward RoPE applied, and the pairing
 *   (e, e + 32) is the one the forward kernel rotated. RoPE is a composition of per-plane 2-D
 *   rotations whose angle is position * freq, so rotating an already-rotated element by
 *   (new_position - old_position) * freq yields that element rotated once at the new position. The
 *   caller therefore supplies rope_deltas[p] = new_position - old_position and this Op needs no
 *   knowledge of either position beyond the delta.
 *
 * Logical shapes:
 *   positions is a contiguous I32 vector [T] of distinct nonnegative logical cache positions.
 *   rope_deltas is a contiguous I32 vector [T], paired elementwise with positions.
 *   cache.k_pages is one BF16 Text paged K plane, cache.block_table is its contiguous I32
 *   logical-block row, and cache.head_dim/num_kv_heads select the registered geometry.
 *
 * Geometry:
 *   Text only: head_dim 256 with 4 or 2 KV heads. The DFlash/MTP paged plane is laid out
 *   head-major, which this Op's addressing does not describe, so a DFlash layer view is rejected
 *   rather than silently addressed wrongly. I8, packed, or lattice K planes are rejected as well.
 *
 * Numeric:
 *   The rotation is evaluated in FP32 from exact BF16 inputs and rounded once back to BF16, so a
 *   re-rotated element carries one more BF16 rounding than an element prefilled directly at that
 *   position. This Op deliberately does not touch V: RoPE does not act on V, so relocated V needs
 *   no adjustment.
 *
 * Effects:
 *   Overwrites elements [0, 64) of every addressed K row and leaves elements [64, 256) untouched.
 *   Reads no unrelated cache row, does not modify V or the block table, and owns no allocation,
 *   frontier, or commit authority.
 *
 * Ordering:
 *   The block table must already map each listed position to the page holding that token's K. The
 *   caller therefore publishes the relocation mapping before calling this Op.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void gqa_kv_rerotate(const Tensor& positions, const Tensor& rope_deltas, PagedKVLayerView cache,
                     cudaStream_t stream);

} // namespace ninfer::ops
