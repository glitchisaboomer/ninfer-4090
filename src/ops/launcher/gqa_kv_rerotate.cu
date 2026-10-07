#include "ops/launcher/gqa_kv_rerotate.h"

#include "core/device.h"
#include "ops/kernel/gqa_kv_rerotate.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// One warp per logical position, matching the kernel's 32 rotary pairs.
constexpr int kRerotateThreads = 32;

template <int KVHeads>
void launch_mode(const Tensor& positions, const Tensor& rope_deltas, PagedKVLayerView cache,
                 cudaStream_t stream) {
    const std::int32_t tokens = positions.ne[0];
    auto* cache_k             = static_cast<__nv_bfloat16*>(cache.k_pages.data);
    const auto* pos           = static_cast<const std::int32_t*>(positions.data);
    const auto* deltas        = static_cast<const std::int32_t*>(rope_deltas.data);
    const auto* table         = static_cast<const std::int32_t*>(cache.block_table.data);
    gqa_kv_rerotate_kernel<KVHeads><<<tokens, kRerotateThreads, 0, stream>>>(pos, deltas, cache_k,
                                                                            table, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gqa_kv_rerotate_launch(const Tensor& positions, const Tensor& rope_deltas,
                            PagedKVLayerView cache, std::int32_t kv_heads, cudaStream_t stream) {
    if (kv_heads == 4) {
        launch_mode<4>(positions, rope_deltas, cache, stream);
        return;
    }
    if (kv_heads == 2) {
        launch_mode<2>(positions, rope_deltas, cache, stream);
        return;
    }
    throw std::invalid_argument("gqa_kv_rerotate: Text KV geometry requires 4 or 2 KV heads");
}

} // namespace ninfer::ops::detail
