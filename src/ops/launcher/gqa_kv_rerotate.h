#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launches the in-place re-rotation for the Text KV geometry. `kv_heads` selects the instantiated
// head walk; the wrapper rejects every geometry that is not registered.
void gqa_kv_rerotate_launch(const Tensor& positions, const Tensor& rope_deltas,
                            PagedKVLayerView cache, std::int32_t kv_heads, cudaStream_t stream);

} // namespace ninfer::ops::detail
