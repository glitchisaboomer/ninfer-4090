#pragma once

#include <array>
#include <cstddef>
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

// The Text inverse-frequency table both entry points take: 32 entries, entry `e` governing rotary
// elements `e` and `e + 32`. The caller passes the same table it gave the forward RoPE, which is
// not necessarily the built-in default: `compute_yarn_config` replaces it once the configured
// context capacity exceeds the model's native context.
using RerotateFreqTable = std::span<const float, 32>;

// The table in the form the kernel takes it.
//
// This is a by-value kernel argument rather than a read of `rope.cuh`'s `kTextRopeInvFrequency`. That
// symbol is a `static __device__ __constant__`, so it has internal linkage and every translation unit
// that includes `rope.cuh` gets a private copy in its own device module. The launcher writes the
// caller's table into `rope.cu`'s copy with cudaMemcpyToSymbol and nothing writes this module's copy,
// so a kernel in another TU would silently rotate with the compile-time defaults instead of the table
// the forward pass used. Passing the table explicitly makes the Op's numerical contract depend on its
// arguments alone, keeps it allocation-free, and is what lets the Op be checked against an
// independent oracle.
struct RerotateFreq {
    float v[32];
};

// Convert the std::array<float,32> stored on the view structs into the by-value kernel
// argument. A raw aggregate initializer cannot do this because float[32] and
// std::array<float,32> are different types.
inline RerotateFreq make_rerotate_freq(const std::array<float, 32>& src) {
    RerotateFreq f{};
    for (std::size_t i = 0; i < 32; ++i) { f.v[i] = src[i]; }
    return f;
}

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
 *   `inv_freq` is the Text inverse-frequency table the forward RoPE applied, and the pairing
 *   (e, e + 32) is the one the forward kernel rotated. RoPE is a composition of per-plane 2-D
 *   rotations whose angle is position * freq, so rotating an already-rotated element by
 *   (new_position - old_position) * freq yields that element rotated once at the new position. The
 *   caller therefore supplies rope_deltas[p] = new_position - old_position and this Op needs no
 *   knowledge of either position beyond the delta.
 *
 *   The table is a required argument rather than read from module-global state, because the Op must
 *   rotate by the frequencies the forward pass actually used. The forward Text RoPE's table is a
 *   mutable `__constant__` symbol that the caller overwrites at startup with a YaRN-scaled table
 *   whenever the context capacity exceeds the native context, and an Op that assumed the built-in
 *   defaults would silently rotate with the wrong angles in exactly that configuration.
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
void gqa_kv_rerotate(const Tensor& positions, const Tensor& rope_deltas, RerotateFreqTable inv_freq,
                     PagedKVLayerView cache, cudaStream_t stream);

/**
 * Op: the same re-rotation for one contiguous run of logical positions sharing a delta.
 *
 * A suffix relocation moves a run of surviving blocks together by a fixed position offset, so the run
 * is `[first_position, first_position + tokens)` and every one of its positions is rotated by the same
 * `delta`. Passing that run as scalars avoids staging per-token position and delta vectors on the
 * device for what is a uniform shift.
 *
 * `tokens` must be positive and the whole run must lie inside the cache capacity, and the block table
 * must already map every position of the run. `inv_freq` carries the same meaning as above.
 * Validation, geometry, numerics, effects, ordering, and workspace are otherwise the same as
 * gqa_kv_rerotate.
 */
void gqa_kv_rerotate_run(std::int32_t first_position, std::int32_t tokens, std::int32_t delta,
                         RerotateFreqTable inv_freq, PagedKVLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
