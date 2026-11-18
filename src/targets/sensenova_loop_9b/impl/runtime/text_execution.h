#pragma once

// Stage E4A: standalone text execution context skeleton for the fixed 56-visit
// SenseNova-Loop-9B plan.
//
// TextExecutionContext owns no memory. It holds, as non-owning handles, the
// four inputs every forward pass needs: the loaded model view, the CUDA
// stream, the stable state image base, and the per-request transient
// workspace — plus the KV capacity that addresses the state image's fixed
// GDN/mHC regions. The host-computable helpers below derive the stable state
// addresses of any visit from StateImageLayout, so execution and tests can
// reason about visit state identity without CUDA.
//
// No generation is implemented at this stage: embedding, the 56 layer visits,
// and sampling land in the later E4 stages.

#include <ninfer/targets/sensenova_loop_9b/package.h>
#include "targets/sensenova_loop_9b/impl/loop_plan.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"

#include "core/arena.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

// Non-owning execution context for one forward pass over the 56-visit loop.
//
// The state image is one contiguous region laid out by StateImageLayout: the
// 15 KV domains (14 text visits + 1 MTP) of `kv_capacity` tokens, then the
// fixed GDN and mHC slot regions. The helpers return stable addresses inside
// that image: identical across calls and safe for CUDA Graph replay.
class TextExecutionContext {
public:
    // Throws std::invalid_argument when `stream` or `state_image_base` is
    // null, when `kv_capacity` is zero, or when the workspace capacity is
    // zero. `model` is a reference and is always valid; the context requires
    // every handle to outlive it.
    TextExecutionContext(const RuntimeModelView& model, cudaStream_t stream,
                         std::byte* state_image_base, std::uint32_t kv_capacity,
                         WorkspaceArena& workspace);

    [[nodiscard]] const RuntimeModelView& model() const noexcept { return model_; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    [[nodiscard]] std::byte* state_image_base() const noexcept { return state_image_base_; }
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept { return kv_capacity_; }
    [[nodiscard]] WorkspaceArena& workspace() const noexcept { return workspace_; }

    // Host-computable total state image size for the stored capacity (this
    // target stores BF16 KV without scales, matching the Program's image
    // allocation).
    [[nodiscard]] std::size_t state_image_bytes() const noexcept {
        return StateImageLayout::total_bytes(kv_capacity_, false, false);
    }

    // Stable base of visit `visit`'s GDN state slot: [recurrent FP32 | conv
    // BF16], StateLayout::gdn_state_bytes_per_slot() bytes. The slot base is
    // the state image base plus the KV-region size plus the GDN slot offset.
    // Throws std::out_of_range for a visit that has no GDN slot
    // (>= StateImageLayout::kGdnSlots).
    [[nodiscard]] std::byte* gdn_state_base(std::size_t visit) const;

    // Stable base of visit `visit`'s mHC state slot: BF16 [8192],
    // StateLayout::mhc_state_bytes_per_slot() bytes. The mHC region follows
    // the full GDN region in the state image. Throws std::out_of_range for a
    // visit that has no mHC slot (>= StateImageLayout::kMhcSlots).
    [[nodiscard]] std::byte* mhc_state_base(std::size_t visit) const;

    // KV domain index for a visit, forwarded to
    // StateImageLayout::kv_domain_for_visit. Callers must gate on
    // LoopPlan::is_full_attention first; non-attention visits have no domain.
    [[nodiscard]] std::size_t kv_domain(std::size_t visit) const noexcept;

private:
    const RuntimeModelView& model_;
    cudaStream_t stream_;
    std::byte* state_image_base_;
    std::uint32_t kv_capacity_;
    WorkspaceArena& workspace_;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail