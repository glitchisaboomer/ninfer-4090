#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace ninfer {

struct LinearAttentionStatePoolSpec {
    std::uint32_t layers        = 0;
    std::int32_t conv_channels  = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t slot_count     = 1;
    DType conv_dtype            = DType::BF16;
};

struct LinearAttentionStatePoolLayout {
    LinearAttentionStatePoolSpec spec;
    std::vector<LayoutRegion> conv;
    std::vector<LayoutRegion> recurrent;

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        std::size_t sum = 0;
        for (const auto& r : conv) { sum += r.bytes; }
        for (const auto& r : recurrent) { sum += r.bytes; }
        return sum;
    }
};

struct LinearAttentionStateAllLayersView {
    Tensor conv_layer0;
    Tensor recurrent_layer0;
    std::int64_t conv_layer_stride_bytes      = 0;
    std::int64_t recurrent_layer_stride_bytes = 0;
    LinearAttentionStatePoolSpec spec;
};

// Per-layer magnitude summary of one Linear Attention state slot.
//
// A GDN slot cannot be relocated (it is one recurrent summary of the whole sequence, not a
// per-token cache), so a relocated request has to invalidate it. Whether that invalidation
// actually happened is invisible from outside the pool: nothing in the state itself records
// which prompt it was summarised over. This is the observation that makes the invalidation
// checkable, and it is deliberately coarse -- sum|x| per layer over the conv and recurrent
// halves separately, so "the zero did not run" and "the zero ran and prefill refilled it" are
// distinguishable from "the state is a plausible steady-state summary".
struct LinearAttentionStateFingerprint {
    std::vector<float> conv_abs_sum;      // one entry per layer
    std::vector<float> recurrent_abs_sum; // one entry per layer
    std::int64_t conv_non_finite         = 0;
    std::int64_t recurrent_non_finite     = 0;
};

[[nodiscard]] LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec);

/**
 * Fixed-capacity physical storage for model-level Linear Attention state images.
 *
 * One logical slot selects the same frontier across every layer's convolution and recurrent
 * component. The pool owns no slot roles, validity, request metadata, allocation policy, or CUDA
 * stream. Construction binds caller-owned backing without mutating it.
 */
struct LinearAttentionStatePool {
    std::vector<Tensor> conv;
    std::vector<Tensor> recurrent;
    LinearAttentionStatePoolSpec spec;

    LinearAttentionStatePool() = default;
    LinearAttentionStatePool(DeviceSpan backing, const LinearAttentionStatePoolLayout& layout);

    [[nodiscard]] std::uint32_t layer_count() const noexcept;
    [[nodiscard]] std::int32_t slot_count() const noexcept;
    [[nodiscard]] std::int64_t conv_slot_stride_elements() const noexcept;
    [[nodiscard]] std::int64_t recurrent_slot_stride_elements() const noexcept;
    [[nodiscard]] LinearAttentionStateAllLayersView all_layers_view() const;
    [[nodiscard]] Tensor conv_slot(std::uint32_t layer, std::int32_t slot) const;
    [[nodiscard]] Tensor recurrent_slot(std::uint32_t layer, std::int32_t slot) const;

    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);

    // sum|x| and non-finite counts per layer, for one slot. Diagnostics only: it reads every
    // byte of the slot over the PCIe bus and synchronises, so it is never on a request path.
    // `stream` must be the stream the writers were launched on -- a blocking read on the legacy
    // default stream does not drain a non-blocking one, and would report the pre-write state.
    [[nodiscard]] LinearAttentionStateFingerprint fingerprint_slot(
        std::int32_t slot, cudaStream_t stream) const;
};

} // namespace ninfer
