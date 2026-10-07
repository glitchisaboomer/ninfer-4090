// ninfer::ops - in-place paged K re-rotation validation and text-geometry dispatch.
#include "ninfer/ops/gqa_kv_rerotate.h"

#include "ops/launcher/gqa_kv_rerotate.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

// Validates the Text paged K plane and returns its token capacity.
std::int32_t validate_text_k_cache(const PagedKVLayerView& cache, const char* op) {
    if (cache.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) +
                                    ": Text KV cache must be BF16 for suffix relocation");
    }
    if (cache.quant_group != 0 || cache.packed_k || cache.packed_v || cache.rotate_k ||
        cache.rotate_v || cache.e8_lattice || cache.e8_root) {
        throw std::invalid_argument(std::string(op) +
                                    ": packed, rotated, lattice, or quantized KV cache is not "
                                    "supported");
    }
    if (cache.head_dim != kHeadDim || (cache.num_kv_heads != 4 && cache.num_kv_heads != 2)) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV cache geometry");
    }
    if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity =
        static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }
    require_shape(cache.k_pages, kHeadDim, kPagedKVPageSize, cache.num_kv_heads, physical_pages, op,
                  "cache k pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");
    return static_cast<std::int32_t>(capacity);
}

void validate_positions_pair(const Tensor& positions, const Tensor& rope_deltas,
                             std::int32_t capacity, const char* op) {
    if (positions.dtype != DType::I32 || rope_deltas.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions and rope deltas must be I32");
    }
    const std::int32_t tokens = positions.ne[0];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(rope_deltas, tokens, 1, 1, 1, op, "rope deltas");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(rope_deltas, op, "rope deltas");
    if (tokens > capacity) {
        throw std::invalid_argument(std::string(op) + ": T exceeds KV cache capacity");
    }
}

} // namespace

void gqa_kv_rerotate(const Tensor& positions, const Tensor& rope_deltas, PagedKVLayerView cache,
                     cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_rerotate";
    const std::int32_t capacity = validate_text_k_cache(cache, op);
    validate_positions_pair(positions, rope_deltas, capacity, op);
    detail::gqa_kv_rerotate_launch(positions, rope_deltas, cache, cache.num_kv_heads, stream);
}

} // namespace ninfer::ops
