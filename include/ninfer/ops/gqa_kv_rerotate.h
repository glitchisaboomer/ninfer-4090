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

} // namespace ninfer::ops
