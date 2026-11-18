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
// Stage E4B adds the two ends of the forward pass -- the token embedding gather
// and the final norm/head/greedy pick for the last column -- so the 56 layer
// visits can be dropped between them without any lifecycle wiring. Sampling
// beyond the greedy argmax and the visit orchestration itself land in later E4
// stages.

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

    // Stage E4B: gathers one embedding row per prompt token into `hidden`.
    //
    // `ids` is contiguous I32 [T] device memory and `hidden` is contiguous BF16
    // [TextConfig::hidden,T] device memory; both must be non-null. T is the
    // prompt chunk width and must agree between the two. The gather runs
    // through ops::embedding on ctx.model().token_embedding, so the table's own
    // quantized encoding and vocabulary bound are the op's contract, not
    // re-derived here. No workspace and no state side effect.
    void embed_tokens(const Tensor& ids, Tensor& hidden) const;

    // Stage E4B: final RMSNorm + LM head + greedy token pick for the last column.
    //
    // `hidden` is contiguous BF16 [TextConfig::hidden,T]. Only column T-1 reaches
    // the head, because a prefill of T tokens yields one sampled token. That
    // column is normed with `final_norm` (zero-centered, epsilon
    // TextConfig::rms_epsilon), projected by `output_head` into a
    // workspace-resident BF16 [TextConfig::output_rows,1] logit column, and
    // reduced by ops::argmax over the first TextConfig::token_domain (248077)
    // rows. Argmax is exactly the greedy branch of ops::sample, and this stage
    // is greedy-only: sampling parameters and penalties arrive with the decode
    // stages.
    //
    // `sampled` is contiguous I32 [1] device memory. When `logprob` is non-null
    // the chosen token's log-probability is written there (contiguous FP32 [1]);
    // `margin` optionally receives the column's decision margin (contiguous
    // FP32 [1]) and requires `logprob`, since it is that measurement's
    // interpretability companion (see ops::logprob_of_choice). Both are read
    // from the same logit column the argmax consumed.
    //
    // The normed column, the logit column, and any margin scratch come from the
    // context workspace for the duration of the call, which is restored on
    // return. head_workspace_capacity_bytes() reports the requirement.
    void project_logits_and_sample(const Tensor& hidden, Tensor& sampled,
                                   Tensor* logprob = nullptr, Tensor* margin = nullptr) const;

    // Transient capacity project_logits_and_sample needs, independent of T: the
    // BF16 normed column [hidden,1], the BF16 logit column [output_rows,1], and,
    // when a logprob is measured without a margin destination, an FP32 [1]
    // margin scratch. Each region is rounded up to the DeviceArena allocation
    // alignment the head uses. The requirement assumes a fresh arena whose base
    // is aligned to that granularity, which is how the Program and request
    // workspaces are allocated; a borrowed misaligned arena may need up to one
    // alignment quantum more.
    [[nodiscard]] static std::size_t head_workspace_capacity_bytes(bool measure_logprob,
                                                                  bool measure_margin) noexcept;

private:
    const RuntimeModelView& model_;
    cudaStream_t stream_;
    std::byte* state_image_base_;
    std::uint32_t kv_capacity_;
    WorkspaceArena& workspace_;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail