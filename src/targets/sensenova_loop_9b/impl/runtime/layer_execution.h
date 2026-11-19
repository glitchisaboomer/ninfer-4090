#pragma once

// Stage E3: standalone one-visit layer execution for the fixed 56-visit SenseNova-Loop-9B plan.
//
// This module composes one logical visit (physical layer + pass) from the qualified op layer.
// It owns no device memory, no stream state, and no lane/Program state: every input, weight
// view, stable state pointer, stream, and workspace is explicit in `LayerVisitRequest`. The
// stage-E runtime (prefill/decode/graph) is the only caller that schedules the 56 visits in
// `LoopPlan` order; nothing here knows about neighbors.
//
// State identity follows `StateLayout` / `StateImageLayout`: GDN recurrent/convolution state is
// addressed through one stable slot per GDN visit, mHC two-stream state is published through one
// stable slot per mHC visit, and full-attention KV state is addressed through the caller-built
// paged views bound to one domain per full-attention visit. The physical layer index is never
// used to address state; the visit index is the sole state identity.
//
// Compositions (all norms are zero-centered, `1 + weight`, epsilon 1e-6, per the reference
// decoder skeleton; mHC routers compute in FP32 and apply their branch/mix/post in the BF16
// residual dtype through the `mhc_route`/`mhc_merge` ops):
//
// Full-attention visit:
//   [mhc enter: residual streams, or: carry = hidden_in]
//   ah        = rmsnorm(input, input_norm)                      (input = mHC branch0 when mHC)
//   qkv       = linear(ah, query_key_gate_value [10240,4096])   (q 4096 / k 1024 / gate 4096 / v 1024 rows)
//   qn, kn    = rmsnorm(q, query_norm), rmsnorm(k, key_norm)
//   rope(qn, kn, rotary_dim 64, theta 1e7, in place)
//   attn      = gqa_prompt/append (16/4, gqa_attention or gqa_attention_cached; never small-T INT8)
//   attn      *= sigmoid(gate)
//   mixer_out = linear(attn, attention.output)
//   [mHC: residual = mhc_merge(residual, mixer_out, post, mix)  :  carry += mixer_out]
//   mh        = rmsnorm(carry or merged streams, post_attention_norm)
//   ffn       = linear_swiglu_sensenova(mh, mlp_gate_up [24576,4096])
//   down      = linear(ffn, mlp_down [4096,12288])
//   [mHC: residual = mhc_merge(residual, down, post, mix)       :  carry += down]
//   [mhc leave: streams reduced by mean]
//   hidden_out = flattened/mean streams, or carry
//
// GDN visit: same skeleton with
//   control    = gdn_norm_gating_proj_sensenova(input, input_norm, a_b, A_log, dt_bias) -> h, g, beta
//   qkv, z     = gdn_input_proj_sensenova(h, query_key_value_z [12288,4096])
//   qkv_c      = causal_conv1d_silu(qkv, conv [8192,4], visit-slot conv state [8192,3])
//   q, k, v    = channel-range extraction (2048 / 2048 / 4096)
//   gdn_out    = gated_delta_net(q, k, v, g, beta, 1/sqrt(128), normalize_qk, visit-slot ssm state)
//   on         = gated_rmsnorm(gdn_out, gdn.norm, z)
//   mixer_out  = linear(on, gdn.output)
//
// Known qualification gap (reported, not worked around): `attn_input_proj` registers no
// fused Q/K/gate/V split profile for the 4096-wide W8 parent, so the split is composed from
// the registered, oracle-qualified W8 `linear` profile `[10240,4096]` plus contiguous views.
// The fused parent form remains an op-layer work item.

#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/loop_plan.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"
#include <ninfer/targets/sensenova_loop_9b/package.h>

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::targets::sensenova_loop_9b::detail {

// Typed planning/validation failure. Execution throws this for any request that violates the
// frozen visit plan, the stored weight profiles, the shape contracts of the composed ops, or
// the state-identity rules. The op layer's own `std::invalid_argument` propagates unchanged.
struct LayerVisitError : std::runtime_error {
    explicit LayerVisitError(std::string what) : std::runtime_error(std::move(what)) {}
};

enum class VisitKind {
    FullAttention,
    Gdn,
};

enum class MhcBoundary {
    None,   // interior visit
    Enter,  // physical layer MhcConfig::start (4): hidden [4096,T] expands to two equal streams
    Leave,  // physical layer MhcConfig::end - 1 (27): streams reduce to their mean [4096,T]
};

// Host-computable execution profile for one logical visit.
struct VisitExecutionProfile {
    std::uint32_t visit        = 0;
    int physical_layer         = 0;
    int pass                   = 0;
    VisitKind kind             = VisitKind::Gdn;
    bool is_mhc                = false;
    MhcBoundary mhc_boundary   = MhcBoundary::None;
    std::size_t state_slot     = 0;   // == visit: KV/GDN/mHC state identity for this visit
    std::uint32_t input_width  = TextConfig::hidden;   // 4096, or 8192 for carried mHC streams
    std::uint32_t output_width = TextConfig::hidden;   // 8192 when streams carry on, 4096 otherwise
};

// Resolves the frozen plan entry for `visit` (0..55). Throws LayerVisitError out of range.
[[nodiscard]] VisitExecutionProfile plan_visit(std::uint32_t visit);

// Compact KV-domain index for a full-attention visit: visits are assigned in visit order over
// the 14 full-attention visit identities (0..13); the single MTP domain occupies the final
// index `StateImageLayout::kKvDomains - 1` and is never aliased with a text visit. Revisited
// physical layers (7,11,15,19,23,27) receive distinct indices on pass 0 and pass 1.
// Throws LayerVisitError for non full-attention visits or out-of-range visits.
[[nodiscard]] std::size_t full_attention_domain_index(std::uint32_t visit);

// GDN visit-slot sub-layout: one `StateLayout::gdn_state_bytes_per_slot()` region holds the
// FP32 recurrent state [128,128,32] at offset 0, then the BF16 convolution history [8192,3].
// The offsets are the single source of truth for both execution and any state-cache mapping.
inline constexpr std::size_t kGdnRecurrentStateOffset = 0;
inline constexpr std::size_t kGdnConvStateOffset      = StateLayout::gdn_recurrent_bytes_per_slot();

struct GdnVisitStateViews {
    Tensor recurrent;  // FP32 [128,128,32], read and rewritten in place by the recurrence
    Tensor conv;       // BF16 [8192,3], oldest-to-newest history, updated in place
};

// Builds the two state views over one stable GDN visit slot base. Throws LayerVisitError on a
// null base. The region must be at least StateLayout::gdn_state_bytes_per_slot() bytes.
[[nodiscard]] GdnVisitStateViews gdn_state_views(const std::byte* slot_base);

// One-visit execution request. All device pointers are caller-owned and must be stable for the
// lifetime of the call (graph-capture safe: no internal allocation or stream switch).
struct LayerVisitRequest {
    // Identity. Must be a valid LoopPlan visit; the physical layer and pass are derived.
    std::uint32_t visit = 0;

    // Hidden state. BF16 contiguous. Shape is fixed by the visit profile: [4096,T] or [8192,T]
    // in, [4096,T] or [8192,T] out. hidden_out may alias hidden_in (in-place execution is
    // well defined); otherwise hidden_in is read-only.
    const Tensor* hidden_in  = nullptr;
    Tensor* hidden_out       = nullptr;

    // Absolute text positions, contiguous I32 [T]. Required for full-attention visits (GQA
    // causal indexing and RoPE); ignored for GDN visits (may stay null).
    const Tensor* positions = nullptr;

    // Full-attention GQA state. Exactly one form must be bound for a full-attention visit:
    //  - append form (A1): attention_kv_append view plus k/v BF16 [1024,T] and the B=1
    //    kv_table_rows I32 [1]; attention_valid_columns null (dense) or I32 [1];
    //  - cached form (A3): attention_kv_cached view only, k/v null;
    //  - state-image form: attention_kv_domain_base names one contiguous KV domain of
    //    attention_kv_domain_tokens tokens owned by the Program's state image, and the executor
    //    derives the A1 batch view over it (see state_images.h for the plane layout). This is the
    //    form the text forward uses: the executor appends the K/V it derives from its own
    //    projections, so the caller cannot pass a stale or absent payload. It requires
    //    attention_kv_block_table (I32 [pages], the identity logical-to-physical map) and
    //    attention_kv_table_rows (I32 [1] naming row 0); k/v and the cached view stay unbound.
    //
    // GDN visits must leave every GQA field null/empty.
    const Tensor* attention_k                   = nullptr;
    const Tensor* attention_v                   = nullptr;
    const Tensor* attention_kv_table_rows       = nullptr;
    const Tensor* attention_valid_columns       = nullptr;
    PagedKVBatchLayerView attention_kv_append;  // A1 batch view (k_pages.data null = unbound)
    PagedKVLayerView attention_kv_cached;       // A3 single-sequence view (k_pages.data null)
    ops::GqaExecutionEnvelope attention_envelope;

    // State-image KV domain (the third form above). All four are null/zero for the explicit
    // append and cached forms.
    const std::byte* attention_kv_domain_base   = nullptr;
    std::uint32_t attention_kv_domain_tokens    = 0;
    const Tensor* attention_kv_block_table      = nullptr;

    // GDN visit slot: stable base of StateLayout::gdn_state_bytes_per_slot() bytes holding
    // [recurrent FP32 | conv BF16] (see gdn_state_views). Required for GDN visits, null for
    // full-attention visits. Recurrent and conv state are mutated in place by the ops.
    const std::byte* gdn_state_base = nullptr;

    // GDN convolution weight in op form: contiguous BF16 [8192,4]. The artifact stores the same
    // bytes as kernel-major [4,8192]; setup reinterprets that storage without transposing it.
    // Required for GDN visits.
    const Tensor* gdn_convolution_weight = nullptr;

    // mHC router weights in op form (required for mHC visits, null otherwise):
    // contiguous FP32 [8192,8] weight rows and FP32 [8] biases. The stored artifact form is
    // [8,8192]; prepare once with transpose_mhc_router_weight.
    const Tensor* mhc_attn_weight = nullptr;
    const Tensor* mhc_attn_bias   = nullptr;
    const Tensor* mhc_ffn_weight  = nullptr;
    const Tensor* mhc_ffn_bias    = nullptr;

    // Device-resident uniform 0.5 mix for the leave reduction, contiguous FP32 [2,2,T]. When
    // bound it replaces the executor's host fill, keeping the visit loop free of host copies so
    // a one-token pass is stream-capture legal. Null means the executor stages the mix itself.
    const Tensor* mhc_half_mix = nullptr;

    // mHC state slot: stable BF16 [8192] region (StateLayout::mhc_state_bytes_per_slot()).
    // After an mHC visit it is published with the flattened two-stream state of the final
    // token of this visit (after the last router merge, before the leave reduction).
    // Required for mHC visits, null otherwise.
    std::byte* mhc_state_base = nullptr;

    // Physical-layer weight set, bound once and shared by every visit of that layer. The
    // is_full_attention/has_mhc flags are cross-checked against the visit identity.
    const TextLayerWeights* weights = nullptr;

    cudaStream_t stream   = nullptr;
    WorkspaceArena* workspace = nullptr;
};

// Caller-owned transient capacity required by one visit of kind `visit` for every T in
// [min_tokens, max_tokens]: the visit scratch layout plus the op-internal workspaces (GDN
// control projection, Gated DeltaNet recurrence, and the GQA small-T verify path for
// full-attention visits whose widths can reach the registered small-T maximum). The W8 A16
// projections require no caller workspace. Returns a capacity suitable for
// `DeviceArena(capacity_bytes)`.
[[nodiscard]] std::size_t layer_visit_workspace_capacity_bytes(std::uint32_t visit,
                                                               std::uint32_t min_tokens,
                                                               std::uint32_t max_tokens);

// Executes one logical visit. Validates the request against the frozen plan, the stored weight
// profiles, and the op shape contracts, then composes the ops. Throws LayerVisitError on any
// validation failure; op-layer errors propagate unchanged.
void execute_layer_visit(const LayerVisitRequest& request);

// mHC two-stream boundary helpers (explicit, no hidden stream state).
//
// mhc_enter_streams: [4096,T] BF16 -> [4096,2,T] BF16 materialized into `out`. Both streams start
// as the scalar hidden (the HuggingFace `enter` broadcast at the first mHC layer); the streams
// are separate storage so the in-place router merges of the visit can diverge them.
void mhc_enter_streams(const Tensor& hidden, Tensor& out, cudaStream_t stream);

// mhc_unflatten_streams: carried flattened [8192,T] BF16 -> [4096,2,T] stream view
// (the `enter` shape restore at interior mHC layers).
[[nodiscard]] Tensor mhc_unflatten_streams(const Tensor& hidden);

// mhc_flatten_streams: [4096,2,T] BF16 -> flattened [8192,T] view carried between mHC visits
// (the `leave` carry form at interior mHC layers).
[[nodiscard]] Tensor mhc_flatten_streams(const Tensor& merged);

// mhc_reduce_streams: [4096,2,T] BF16 -> [4096,T] BF16 mean over the streams, in the residual
// dtype, computed by mhc_merge with the exact half-identity mix and zero post/branch (0.0 and
// 0.5 are exact in BF16, so the result is bit-identical to the reference mean). This is the
// `leave` reduction at the last mHC layer.
[[nodiscard]] Tensor mhc_reduce_streams(const Tensor& merged, const Tensor* device_mix,
                                        WorkspaceArena& workspace, cudaStream_t stream);

// mhc_publish_state: copies the flattened [8192] stream state of the final token of
// `merged_streams` ([4096,2,T] contiguous) into the visit's mHC state slot.
void mhc_publish_state(const Tensor& merged_streams, std::byte* slot_base, cudaStream_t stream);

// Setup-time transpose helpers (source must be stream-stable; copies are stream-ordered).
// stored: FP32 [8,8192] logits-major router rows -> out: contiguous FP32 [8192,8] op form.
void transpose_mhc_router_weight(const Tensor& stored, Tensor& out, cudaStream_t stream);

} // namespace ninfer::targets::sensenova_loop_9b::detail
