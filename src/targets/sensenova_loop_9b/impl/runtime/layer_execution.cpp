// Stage E3: one-visit layer execution for the fixed 56-visit SenseNova-Loop-9B plan.
#include "targets/sensenova_loop_9b/impl/runtime/layer_execution.h"

#include "core/layout.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mhc.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

constexpr const char* kOp = "sensenova_layer_visit:";
constexpr float kGdnRecurrentScale = 0.08838834764831845F; // 1/sqrt(128)
constexpr float kAttentionScale    = 1.0F / std::sqrt(static_cast<float>(TextConfig::head_dim));
// The stored W8 parent is [q 4096 | k 1024 | gate 4096 | v 1024] = [10240,4096] rows.
constexpr std::int32_t kQueryKeyGateValueRows = 2 * (TextConfig::query_size + TextConfig::kv_size);
constexpr std::int32_t kGdnInputProjectionRows = 2 * TextConfig::key_dim + 2 * TextConfig::value_dim;
// Within one GDN visit slot the recurrent state precedes the convolution state (StateLayout).

[[noreturn]] void fail(std::string message) { throw LayerVisitError(kOp + message); }

void require(bool condition, std::string message) {
    if (condition) { return; }
    fail(std::move(message));
}

void require_tensor(const Tensor* tensor, DType dtype, const char* label) {
    require(tensor != nullptr, std::string(label) + " tensor is required");
    require(tensor->data != nullptr, std::string(label) + " data must be non-null");
    require(tensor->dtype == dtype, std::string(label) + " dtype mismatch");
    require(tensor->is_contiguous(), std::string(label) + " must be contiguous");
}

void require_weight_profile(const Weight& weight, QType qtype, QuantLayout layout, int n, int k,
                            const char* label) {
    require(weight.payload != nullptr, std::string(label) + " weight payload is required");
    require(weight.qtype == qtype, std::string(label) + " weight qtype mismatch");
    require(weight.layout == layout, std::string(label) + " weight layout mismatch");
    require(weight.ndim == 2 && weight.n == n && weight.k == k,
            std::string(label) + " weight profile mismatch");
}

void require_w8(const Weight& weight, int n, int k, const char* label) {
    require_weight_profile(weight, QType::W8G32_F16S, QuantLayout::RowSplit, n, k, label);
    require(weight.group_size == 32, std::string(label) + " W8 weight must use group size 32");
}

void require_fp32_vector(const Tensor& tensor, int rows, const char* label) {
    require_tensor(&tensor, DType::FP32, label);
    require(tensor.ne[0] == rows && tensor.ne[1] == 1 && tensor.ne[2] == 1 && tensor.ne[3] == 1,
            std::string(label) + " shape mismatch");
}

bool bound_kv_view(const PagedKVBatchLayerView& view) {
    return view.k_pages.data != nullptr && view.v_pages.data != nullptr &&
           view.block_tables.data != nullptr;
}

bool bound_kv_view(const PagedKVLayerView& view) {
    return view.k_pages.data != nullptr && view.v_pages.data != nullptr &&
           view.block_table.data != nullptr;
}

bool bound_kv_domain(const LayerVisitRequest& request) {
    return request.attention_kv_domain_base != nullptr;
}

// Derives the A1 batch view over one state-image KV domain. The domain already holds K then V in
// the page-major plane shape the paged GQA kernels read, so this is pure tensor arithmetic: no
// copy, no repacking, and the addresses are stable for the Program lifetime (graph replay safe).
PagedKVBatchLayerView kv_domain_batch_view(const LayerVisitRequest& request) {
    require(request.attention_kv_domain_base != nullptr,
            "state-image KV domain base is required");
    require(request.attention_kv_domain_tokens > 0,
            "state-image KV domain token extent must be positive");
    require(request.attention_kv_domain_tokens % StateImageLayout::kKvPageTokens == 0,
            "state-image KV domain must hold a whole number of pages");
    require(request.attention_kv_block_table != nullptr,
            "state-image KV domain requires the identity block table");
    require_tensor(request.attention_kv_block_table, DType::I32, "state-image block table");
    const std::size_t pages = StateImageLayout::kv_pages(request.attention_kv_domain_tokens);
    require(request.attention_kv_block_table->ne[0] == static_cast<int>(pages) &&
                request.attention_kv_block_table->ne[1] == 1 &&
                request.attention_kv_block_table->ne[2] == 1 &&
                request.attention_kv_block_table->ne[3] == 1,
            "state-image block table must be I32 [pages]");

    auto* base = reinterpret_cast<std::uint8_t*>(
        const_cast<std::byte*>(request.attention_kv_domain_base));
    const std::int32_t page_tokens = static_cast<std::int32_t>(StateImageLayout::kKvPageTokens);
    const std::int32_t kv_heads    = static_cast<std::int32_t>(StateLayout::kv_heads);
    const std::int32_t head_dim    = static_cast<std::int32_t>(StateLayout::head_dim);
    const std::int32_t page_count  = static_cast<std::int32_t>(pages);
    const std::size_t plane_bytes  =
        StateImageLayout::kv_plane_bytes(request.attention_kv_domain_tokens);

    PagedKVBatchLayerView view;
    view.k_pages     = Tensor(base, DType::BF16, {head_dim, page_tokens, kv_heads, page_count});
    view.v_pages     = Tensor(base + plane_bytes, DType::BF16,
                              {head_dim, page_tokens, kv_heads, page_count});
    view.block_tables = request.attention_kv_block_table->view({page_count, 1});
    view.head_dim     = head_dim;
    view.num_kv_heads = kv_heads;
    view.dtype        = DType::BF16;
    view.quant_group  = 0;
    return view;
}

void validate_kv_view_geometry(std::int32_t num_kv_heads, std::int32_t head_dim, DType dtype,
                               std::int32_t quant_group, const char* label) {
    require(num_kv_heads == TextConfig::kv_heads,
            std::string(label) + " must expose " + std::to_string(TextConfig::kv_heads) +
                " KV heads");
    require(head_dim == TextConfig::head_dim,
            std::string(label) + " head_dim must be " + std::to_string(TextConfig::head_dim));
    require(dtype == DType::BF16 || dtype == DType::I8,
            std::string(label) + " must be BF16 or INT8-G64");
    if (dtype == DType::BF16) {
        require(quant_group == 0, std::string(label) + " BF16 cache must not declare a quant group");
    } else {
        require(quant_group == 64, std::string(label) + " INT8 cache must use quant group 64");
    }
}

// The visit scratch layout, shared by execution and the capacity query. All allocations are
// offset-monotone inside the caller's top-level arena scope, so peak equals the total.
template <class Arena>
struct VisitBuffers {
    Tensor carry, ah, q, k, gate, v, qn, kn, attention_out, attn_out;
    Tensor gdn_h, gdn_g, gdn_beta, gdn_qkv, gdn_z, gdn_qkv_c, gdn_q, gdn_k, gdn_v, gdn_o, gdn_on;
    Tensor post_hidden, ffn_out, down_out;
    Tensor branch0, post0, mix0, branch1, post1, mix1, merged0, merged1;
    Tensor enter_residual;
    Tensor reduce_out, zero_branch, zero_post, half_mix;
};

template <class Arena>
VisitBuffers<Arena> allocate_visit_buffers(Arena& arena, const VisitExecutionProfile& profile,
                                            std::int32_t tokens) {
    VisitBuffers<Arena> b;
    b.ah = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    if (!profile.is_mhc) {
        b.carry = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    }
    if (profile.kind == VisitKind::FullAttention) {
        b.q = arena.alloc(DType::BF16, {TextConfig::query_size, tokens});
        b.k = arena.alloc(DType::BF16, {TextConfig::kv_size, tokens});
        b.gate = arena.alloc(DType::BF16, {TextConfig::query_size, tokens});
        b.v = arena.alloc(DType::BF16, {TextConfig::kv_size, tokens});
        b.qn = arena.alloc(DType::BF16, {TextConfig::head_dim, TextConfig::query_heads, tokens});
        b.kn = arena.alloc(DType::BF16, {TextConfig::head_dim, TextConfig::kv_heads, tokens});
        b.attention_out = arena.alloc(DType::BF16, {TextConfig::query_size, tokens});
        b.attn_out = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    } else {
        b.gdn_h = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
        b.gdn_g = arena.alloc(DType::FP32, {TextConfig::gdn_value_heads, tokens});
        b.gdn_beta = arena.alloc(DType::FP32, {TextConfig::gdn_value_heads, tokens});
        b.gdn_qkv = arena.alloc(DType::BF16, {TextConfig::convolution_dim, tokens});
        b.gdn_z = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
        b.gdn_qkv_c = arena.alloc(DType::BF16, {TextConfig::convolution_dim, tokens});
        b.gdn_q = arena.alloc(DType::BF16, {TextConfig::key_dim, tokens});
        b.gdn_k = arena.alloc(DType::BF16, {TextConfig::key_dim, tokens});
        b.gdn_v = arena.alloc(DType::BF16, {TextConfig::value_dim, tokens});
        b.gdn_o = arena.alloc(DType::BF16, {TextConfig::value_dim, tokens});
        b.gdn_on = arena.alloc(DType::BF16, {TextConfig::value_dim, tokens});
        b.attn_out = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    }
    b.post_hidden = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    b.ffn_out = arena.alloc(DType::BF16, {TextConfig::intermediate, tokens});
    b.down_out = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    if (profile.is_mhc) {
        b.branch0 = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
        b.post0 = arena.alloc(DType::FP32, {MhcConfig::streams, tokens});
        b.mix0 = arena.alloc(DType::FP32, {MhcConfig::streams, MhcConfig::streams, tokens});
        b.branch1 = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
        b.post1 = arena.alloc(DType::FP32, {MhcConfig::streams, tokens});
        b.mix1 = arena.alloc(DType::FP32, {MhcConfig::streams, MhcConfig::streams, tokens});
        b.merged0 = arena.alloc(DType::BF16, {TextConfig::hidden, MhcConfig::streams, tokens});
        b.merged1 = arena.alloc(DType::BF16, {TextConfig::hidden, MhcConfig::streams, tokens});
        if (profile.mhc_boundary == MhcBoundary::Enter) {
            b.enter_residual = arena.alloc(DType::BF16, {TextConfig::hidden, MhcConfig::streams,
                                                        tokens});
        }
        if (profile.mhc_boundary == MhcBoundary::Leave) {
            b.reduce_out = arena.alloc(DType::BF16, {TextConfig::hidden, MhcConfig::streams, tokens});
            b.zero_branch = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
            b.zero_post = arena.alloc(DType::FP32, {MhcConfig::streams, tokens});
            b.half_mix = arena.alloc(DType::FP32, {MhcConfig::streams, MhcConfig::streams, tokens});
        }
    }
    return b;
}

// W8G32_F16S RowSplit plane layout per weight row: group_size 8-bit codes, no high plane,
// one FP16 scale per 32-column group. A row subview keeps the parent k/padding and only
// offsets the planes by row_begin times the per-row plane strides.
Weight w8_row_subview(const Weight& parent, std::int32_t row_begin, std::int32_t row_count) {
    require(parent.qtype == QType::W8G32_F16S && parent.layout == QuantLayout::RowSplit,
            "w8 row subview requires W8G32_F16S RowSplit");
    require(parent.qdata != nullptr && parent.scales != nullptr,
            "w8 row subview requires populated planes");
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > parent.n) {
        fail("w8 row subview range out of bounds");
    }
    const std::int32_t groups_per_row = parent.padded_shape[1] /
                                        static_cast<std::int32_t>(parent.group_size);
    const std::size_t code_row = static_cast<std::size_t>(groups_per_row) * parent.group_size;
    const std::size_t scale_row = static_cast<std::size_t>(groups_per_row) * 2;
    Weight w = parent;
    w.qdata = static_cast<const std::uint8_t*>(parent.qdata) +
              static_cast<std::size_t>(row_begin) * code_row;
    w.scales = static_cast<const std::uint8_t*>(parent.scales) +
               static_cast<std::size_t>(row_begin) * scale_row;
    w.qhigh = nullptr;
    w.payload = w.qdata;
    w.n = row_count;
    w.shape[0] = row_count;
    return w;
}

Tensor run_full_attention(VisitBuffers<WorkspaceArena>& b, const LayerVisitRequest& request,
                        const Tensor& mixer_input, const TextLayerWeights& weights,
                        const Tensor& mhc_residual, bool has_mhc_residual, cudaStream_t stream,
                        WorkspaceArena& arena) {
    const std::int32_t T = mixer_input.ne[1];

    ops::rmsnorm(mixer_input, weights.input_norm, TextConfig::rms_epsilon, true, b.ah, stream);

    // The stored artifact parent is the single concatenated [10240,4096] W8 RowSplit weight
    // (q [0:4096), k [4096:5120), gate [5120:9216), v [9216:10240)). The GQA prompt path needs
    // contiguous per-projection outputs, and a row range of the [10240,T] linear output is not
    // contiguous, so each projection runs its own W8 A16 linear over the parent's row subview
    // (E4 may fuse these once a 16/4 attn_input_proj geometry exists).
    const Weight& parent = weights.attention.query_key_gate_value;
    const Weight w_q = w8_row_subview(parent, 0, TextConfig::query_size);
    const Weight w_k = w8_row_subview(parent, TextConfig::query_size, TextConfig::kv_size);
    const Weight w_gate = w8_row_subview(parent, TextConfig::query_size + TextConfig::kv_size,
                                         TextConfig::query_size);
    const Weight w_v = w8_row_subview(parent, kQueryKeyGateValueRows - TextConfig::kv_size,
                                      TextConfig::kv_size);
    ops::linear(b.ah, w_q, b.q, stream);
    ops::linear(b.ah, w_k, b.k, stream);
    ops::linear(b.ah, w_gate, b.gate, stream);
    ops::linear(b.ah, w_v, b.v, stream);

    Tensor q = b.q.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor k = b.k.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    Tensor gate = b.gate.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor v = b.v.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    Tensor qn = b.qn;
    Tensor kn = b.kn;

    ops::rmsnorm(q, weights.attention.query_norm, TextConfig::rms_epsilon, true, qn, stream);
    ops::rmsnorm(k, weights.attention.key_norm, TextConfig::rms_epsilon, true, kn, stream);
    ops::rope(*request.positions, TextConfig::rotary_dim, TextConfig::rope_theta, qn, kn, stream);

    Tensor attention_out = b.attention_out.view({TextConfig::head_dim, TextConfig::query_heads, T});
    const bool append = request.attention_k != nullptr || bound_kv_domain(request);
    if (append) {
        Tensor valid = request.attention_valid_columns != nullptr
                                 ? *request.attention_valid_columns
                                 : Tensor{};
        const PagedKVBatchLayerView cache = bound_kv_domain(request)
                                                ? kv_domain_batch_view(request)
                                                : request.attention_kv_append;
        ops::gqa_attention(qn, kn, v, *request.positions, valid,
                           *request.attention_kv_table_rows, kAttentionScale, cache,
                           request.attention_envelope, arena, attention_out, stream);
    } else {
        ops::gqa_attention_cached(qn, *request.positions, kAttentionScale,
                                  request.attention_kv_cached, request.attention_envelope, arena,
                                  attention_out, stream);
    }

    ops::sigmoid_mul(gate, attention_out, stream);
    Tensor attention_flat = b.attention_out.view({TextConfig::query_size, T});
    ops::linear(attention_flat, weights.attention.output, b.attn_out, stream);

    if (has_mhc_residual) {
        ops::mhc_merge(mhc_residual, b.attn_out, b.post0, b.mix0, MhcConfig::streams,
                       b.merged0, stream);
        return b.merged0;
    }
    ops::residual_add(b.attn_out, b.carry, stream);
    return Tensor{};
}

void run_gdn(VisitBuffers<WorkspaceArena>& b, const LayerVisitRequest& request,
             const Tensor& mixer_input, const TextLayerWeights& weights,
             GdnVisitStateViews& state, cudaStream_t stream, WorkspaceArena& arena) {
    const std::int32_t T = mixer_input.ne[1];

    ops::gdn_norm_gating_proj_sensenova(mixer_input, weights.input_norm,
                                        TextConfig::rms_epsilon, weights.gdn.a_b_projection,
                                        weights.gdn.a_log, weights.gdn.dt_bias, arena, b.gdn_h,
                                        b.gdn_g, b.gdn_beta, stream);

    ops::gdn_input_proj_sensenova(b.gdn_h, weights.gdn.query_key_value_z, b.gdn_qkv, b.gdn_z,
                                  stream);

    ops::causal_conv1d_silu(b.gdn_qkv, *request.gdn_convolution_weight, state.conv, b.gdn_qkv_c,
                            stream);

    ops::extract_bf16_columns(b.gdn_qkv_c, 0, b.gdn_q, stream);
    ops::extract_bf16_columns(b.gdn_qkv_c, TextConfig::key_dim, b.gdn_k, stream);
    ops::extract_bf16_columns(b.gdn_qkv_c, 2 * TextConfig::key_dim, b.gdn_v, stream);

    Tensor q = b.gdn_q.view({TextConfig::gdn_key_head_dim, TextConfig::gdn_key_heads, T});
    Tensor k = b.gdn_k.view({TextConfig::gdn_key_head_dim, TextConfig::gdn_key_heads, T});
    Tensor v = b.gdn_v.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    Tensor o = b.gdn_o.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    ops::gated_delta_net(q, k, v, b.gdn_g, b.gdn_beta, kGdnRecurrentScale,
                         /*normalize_qk=*/true, arena, state.recurrent, o, stream);

    Tensor zn = b.gdn_z.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    Tensor on = b.gdn_on.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    ops::gated_rmsnorm(o, weights.gdn.norm, zn, TextConfig::rms_epsilon, on, stream);
    ops::linear(b.gdn_on, weights.gdn.output, b.attn_out, stream);
}

// The FFN sublayer consumes the FFN router branch (branch1, or the carry for non-mHC visits),
// while the router merge folds its output into the two-stream residual carried across the visit.
Tensor run_ffn(VisitBuffers<WorkspaceArena>& b, const Tensor& ffn_input,
               const Tensor& merge_residual, const TextLayerWeights& weights,
               bool has_mhc_residual, cudaStream_t stream) {
    ops::rmsnorm(ffn_input, weights.post_attention_norm, TextConfig::rms_epsilon, true,
                 b.post_hidden, stream);
    ops::linear_swiglu_sensenova(b.post_hidden, weights.mlp_gate_up, b.ffn_out, stream);
    ops::linear(b.ffn_out, weights.mlp_down, b.down_out, stream);
    if (has_mhc_residual) {
        ops::mhc_merge(merge_residual, b.down_out, b.post1, b.mix1, MhcConfig::streams,
                       b.merged1, stream);
        return b.merged1;
    }
    ops::residual_add(b.down_out, b.carry, stream);
    return Tensor{};
}

void validate_request(const LayerVisitRequest& request, const VisitExecutionProfile& profile) {
    require(request.weights != nullptr, "weights are required");
    require(request.stream != nullptr, "stream is required");
    require(request.workspace != nullptr, "workspace is required");
    require(request.weights->is_full_attention == (profile.kind == VisitKind::FullAttention),
            "weights do not match the visit attention kind");
    require(request.weights->has_mhc == profile.is_mhc,
            "weights do not match the visit mHC membership");

    require_tensor(request.hidden_in, DType::BF16, "hidden_in");
    require(request.hidden_in->ne[0] == static_cast<int>(profile.input_width) &&
                request.hidden_in->ne[1] > 0 && request.hidden_in->ne[2] == 1 &&
                request.hidden_in->ne[3] == 1,
            "hidden_in shape must match the visit input width");
    const std::int32_t T = request.hidden_in->ne[1];
    require_tensor(request.hidden_out, DType::BF16, "hidden_out");
    require(request.hidden_out->ne[0] == static_cast<int>(profile.output_width) &&
                request.hidden_out->ne[1] == T && request.hidden_out->ne[2] == 1 &&
                request.hidden_out->ne[3] == 1,
            "hidden_out shape must match the visit output width");

    if (profile.kind == VisitKind::FullAttention) {
        require_tensor(request.positions, DType::I32, "positions");
        require(request.positions->ne[0] == T && request.positions->ne[1] == 1 &&
                    request.positions->ne[2] == 1 && request.positions->ne[3] == 1,
                "positions shape mismatch");
        require(request.attention_envelope.min_visible_keys > 0,
                "attention envelope min_visible_keys must be positive");
        require(request.attention_envelope.min_visible_keys <=
                    request.attention_envelope.max_visible_keys,
                "attention envelope min must not exceed max");
        require(request.attention_envelope.max_visible_keys >= static_cast<std::uint32_t>(T),
                "attention envelope is shorter than T");
        require(request.attention_envelope.max_visible_keys <=
                    ops::kGqaAttentionMaximumVisibleKeys,
                "attention envelope exceeds the visible-key maximum");

        const bool append = request.attention_k != nullptr || request.attention_v != nullptr;
        const bool cached = bound_kv_view(request.attention_kv_cached);
        const bool domain = bound_kv_domain(request);
        if (static_cast<int>(append) + static_cast<int>(cached) + static_cast<int>(domain) != 1) {
            fail("bind exactly one GQA form (append, cached, or state-image domain)");
        }
        if (domain) {
            require(request.attention_k == nullptr && request.attention_v == nullptr,
                    "state-image KV domain takes no caller K/V payload");
            require(request.attention_kv_block_table != nullptr,
                    "state-image KV domain requires the identity block table");
            require_tensor(request.attention_kv_table_rows, DType::I32,
                           "attention kv_table_rows");
            require(request.attention_kv_table_rows->ne[0] == 1 &&
                        request.attention_kv_table_rows->ne[1] == 1,
                    "attention kv_table_rows must be I32 [1] for B=1");
            const PagedKVBatchLayerView view = kv_domain_batch_view(request);
            validate_kv_view_geometry(view.num_kv_heads, view.head_dim, view.dtype,
                                      view.quant_group, "attention state-image view");
        } else if (append) {
            require(request.attention_k != nullptr && request.attention_v != nullptr,
                    "append attention requires both k and v");
            require_tensor(request.attention_k, DType::BF16, "attention k");
            require_tensor(request.attention_v, DType::BF16, "attention v");
            require(request.attention_k->ne[0] == TextConfig::kv_size &&
                        request.attention_k->ne[1] == T && request.attention_k->ne[2] == 1,
                    "attention k shape mismatch");
            require(request.attention_v->ne[0] == TextConfig::kv_size &&
                        request.attention_v->ne[1] == T && request.attention_v->ne[2] == 1,
                    "attention v shape mismatch");
            require_tensor(request.attention_kv_table_rows, DType::I32, "attention kv_table_rows");
            require(request.attention_kv_table_rows->ne[0] == 1 &&
                        request.attention_kv_table_rows->ne[1] == 1,
                    "attention kv_table_rows must be I32 [1] for B=1");
            if (request.attention_valid_columns != nullptr) {
                require_tensor(request.attention_valid_columns, DType::I32,
                               "attention valid_columns");
                require(request.attention_valid_columns->ne[0] == 1 &&
                            request.attention_valid_columns->ne[1] == 1,
                        "attention valid_columns must be I32 [1] for B=1");
            }
            require(bound_kv_view(request.attention_kv_append),
                    "append attention requires the batch KV view");
            validate_kv_view_geometry(request.attention_kv_append.num_kv_heads,
                                      request.attention_kv_append.head_dim,
                                      request.attention_kv_append.dtype,
                                      request.attention_kv_append.quant_group,
                                      "attention append view");
        } else if (cached) {
            require(request.attention_kv_table_rows == nullptr &&
                        request.attention_valid_columns == nullptr &&
                        request.attention_kv_block_table == nullptr &&
                        request.attention_kv_domain_tokens == 0,
                    "cached attention takes no table rows, valid columns, or domain state");
            validate_kv_view_geometry(request.attention_kv_cached.num_kv_heads,
                                      request.attention_kv_cached.head_dim,
                                      request.attention_kv_cached.dtype,
                                      request.attention_kv_cached.quant_group,
                                      "attention cached view");
        } else {
            fail("bind exactly one GQA form (append or cached)");
        }

        require_w8(request.weights->attention.query_key_gate_value, kQueryKeyGateValueRows,
                   TextConfig::hidden, "attention");
        require_w8(request.weights->attention.output, TextConfig::hidden, TextConfig::hidden,
                   "attention output");
        require_tensor(&request.weights->attention.query_norm, DType::BF16, "query_norm");
        require_tensor(&request.weights->attention.key_norm, DType::BF16, "key_norm");
        require(request.weights->attention.query_norm.ne[0] == TextConfig::head_dim &&
                    request.weights->attention.key_norm.ne[0] == TextConfig::head_dim,
                "QK norm weight shape mismatch");
    } else {
        if (request.positions != nullptr) {
            require_tensor(request.positions, DType::I32, "positions");
            require(request.positions->ne[0] == T, "positions shape mismatch");
        }
        require(request.attention_k == nullptr && request.attention_v == nullptr &&
                    request.attention_kv_table_rows == nullptr &&
                    request.attention_valid_columns == nullptr &&
                    !bound_kv_view(request.attention_kv_append) &&
                    !bound_kv_view(request.attention_kv_cached) &&
                    request.attention_kv_block_table == nullptr &&
                    request.attention_kv_domain_tokens == 0 &&
                    !bound_kv_domain(request),
                "GDN visits take no GQA inputs");

        require(request.gdn_state_base != nullptr, "GDN visits require the visit state slot");
        require(request.gdn_convolution_weight != nullptr,
                "GDN visits require the op-form convolution weight");
        require_tensor(request.gdn_convolution_weight, DType::BF16, "gdn convolution weight");
        require(request.gdn_convolution_weight->ne[0] == TextConfig::convolution_dim &&
                    request.gdn_convolution_weight->ne[1] == TextConfig::gdn_conv_kernel &&
                    request.gdn_convolution_weight->ne[2] == 1 &&
                    request.gdn_convolution_weight->ne[3] == 1,
                "gdn convolution weight must be contiguous BF16 [C,4]");
        require_w8(request.weights->gdn.query_key_value_z, kGdnInputProjectionRows,
                   TextConfig::hidden, "gdn input projection");
        require_weight_profile(request.weights->gdn.a_b_projection, QType::BF16_CTRL,
                               QuantLayout::Contiguous, 2 * TextConfig::gdn_value_heads,
                               TextConfig::hidden, "gdn a_b projection");
        require_w8(request.weights->gdn.output, TextConfig::hidden, TextConfig::hidden,
                   "gdn output");
        require_fp32_vector(request.weights->gdn.a_log, TextConfig::gdn_value_heads, "gdn A_log");
        require_fp32_vector(request.weights->gdn.dt_bias, TextConfig::gdn_value_heads,
                            "gdn dt_bias");
        require_tensor(&request.weights->gdn.norm, DType::BF16, "gdn norm");
        require(request.weights->gdn.norm.ne[0] == TextConfig::gdn_value_head_dim,
                "gdn norm weight shape mismatch");
    }

    if (profile.is_mhc) {
        require(request.mhc_state_base != nullptr, "mHC visits require the visit state slot");
        require_tensor(request.mhc_attn_weight, DType::FP32, "mhc attn weight");
        require_tensor(request.mhc_ffn_weight, DType::FP32, "mhc ffn weight");
        require_tensor(request.mhc_attn_bias, DType::FP32, "mhc attn bias");
        require_tensor(request.mhc_ffn_bias, DType::FP32, "mhc ffn bias");
        require(request.mhc_attn_weight->ne[0] == MhcConfig::input_width &&
                    request.mhc_attn_weight->ne[1] == MhcConfig::logits &&
                    request.mhc_ffn_weight->ne[0] == MhcConfig::input_width &&
                    request.mhc_ffn_weight->ne[1] == MhcConfig::logits,
                "mHC router weights must be FP32 [8192,8] in op form");
        require(request.mhc_attn_bias->ne[0] == MhcConfig::logits &&
                    request.mhc_ffn_bias->ne[0] == MhcConfig::logits,
                "mHC router biases must be FP32 [8]");
    } else {
        require(request.mhc_state_base == nullptr && request.mhc_attn_weight == nullptr &&
                    request.mhc_attn_bias == nullptr && request.mhc_ffn_weight == nullptr &&
                    request.mhc_ffn_bias == nullptr,
                "non mHC visits take no mHC inputs");
    }

    require_tensor(&request.weights->input_norm, DType::BF16, "input_norm");
    require_tensor(&request.weights->post_attention_norm, DType::BF16, "post_attention_norm");
    require(request.weights->input_norm.ne[0] == TextConfig::hidden &&
                request.weights->post_attention_norm.ne[0] == TextConfig::hidden,
            "norm weight shape mismatch");
    require_w8(request.weights->mlp_gate_up, 2 * TextConfig::intermediate, TextConfig::hidden,
               "mlp gate_up");
    require_w8(request.weights->mlp_down, TextConfig::hidden, TextConfig::intermediate,
               "mlp down");
}

} // namespace

VisitExecutionProfile plan_visit(std::uint32_t visit) {
    if (visit >= LoopPlan::kVisits) {
        fail("visit " + std::to_string(visit) + " is outside the fixed 56-visit plan");
    }
    VisitExecutionProfile profile;
    profile.visit          = visit;
    profile.physical_layer = LoopPlan::physical_layer(visit);
    profile.pass           = LoopPlan::pass(visit);
    profile.kind           = LoopPlan::is_full_attention(visit) ? VisitKind::FullAttention
                                                                : VisitKind::Gdn;
    profile.is_mhc         = LoopPlan::is_mhc(visit);
    if (profile.physical_layer == MhcConfig::start_physical) {
        profile.mhc_boundary = MhcBoundary::Enter;
    } else if (profile.physical_layer == MhcConfig::end_physical - 1) {
        profile.mhc_boundary = MhcBoundary::Leave;
    }
    profile.state_slot   = LoopPlan::state_slot(visit);
    const bool enter     = profile.mhc_boundary == MhcBoundary::Enter;
    profile.input_width  = (profile.is_mhc && !enter) ? MhcConfig::input_width
                                                      : TextConfig::hidden;
    profile.output_width = (profile.is_mhc && profile.mhc_boundary != MhcBoundary::Leave)
                               ? MhcConfig::input_width
                               : TextConfig::hidden;
    return profile;
}

std::size_t full_attention_domain_index(std::uint32_t visit) {
    if (visit >= LoopPlan::kVisits) {
        fail("visit " + std::to_string(visit) + " is outside the fixed 56-visit plan");
    }
    if (!LoopPlan::is_full_attention(visit)) {
        fail("visit " + std::to_string(visit) + " is not a full-attention visit");
    }
    std::size_t index = 0;
    for (std::size_t v = 0; v < visit; ++v) {
        if (LoopPlan::is_full_attention(v)) { ++index; }
    }
    if (index >= StateLayout::full_attention_state_slots) {
        fail("full-attention domain index exceeds the visit slot count");
    }
    return index;
}

GdnVisitStateViews gdn_state_views(const std::byte* slot_base) {
    require(slot_base != nullptr, "gdn visit state slot base is required");
    GdnVisitStateViews views;
    views.recurrent = Tensor(const_cast<std::byte*>(slot_base), DType::FP32,
                            {static_cast<int>(StateLayout::gdn_qk_head_dim),
                             static_cast<int>(StateLayout::gdn_qk_head_dim),
                             static_cast<int>(StateLayout::gdn_value_heads)});
    views.conv = Tensor(const_cast<std::byte*>(slot_base + kGdnConvStateOffset), DType::BF16,
                       {static_cast<int>(StateLayout::gdn_conv_channels),
                        static_cast<int>(StateLayout::gdn_conv_state_width)});
    return views;
}

std::size_t layer_visit_workspace_capacity_bytes(std::uint32_t visit, std::uint32_t min_tokens,
                                                 std::uint32_t max_tokens) {
    if (min_tokens == 0 || max_tokens < min_tokens) {
        fail("invalid token interval");
    }
    const VisitExecutionProfile profile = plan_visit(visit);
    WorkspaceLayoutBuilder layout;
    allocate_visit_buffers(layout, profile, static_cast<std::int32_t>(max_tokens));
    // DeviceArena scopes do not rewind allocations; every scratch tensor and every op-internal
    // workspace allocated during the call is live until the scope closes. Keep a conservative
    // multiplier for the scratch peak so the query is a true upper bound across the T interval.
    std::size_t total = layout.peak_bytes() * 4;
    if (profile.kind == VisitKind::Gdn) {
        total += ops::gdn_norm_gating_proj_workspace_capacity_bytes(
            TextConfig::gdn_value_heads, TextConfig::hidden,
            static_cast<std::int32_t>(min_tokens), static_cast<std::int32_t>(max_tokens));
        total += ops::gated_delta_net_workspace_capacity_bytes(
            TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true,
            static_cast<std::int32_t>(min_tokens), static_cast<std::int32_t>(max_tokens));
    } else {
        // The 16/4 SenseNova geometry takes the GQA small-T verify path for widths up to the
        // registered maximum and the zero-workspace prompt path beyond it; the interval query
        // below covers both for any request envelope inside [min_tokens, max_tokens].
        total += ops::gqa_attention_workspace_capacity_bytes(
            TextConfig::query_heads, TextConfig::kv_heads, DType::BF16,
            ops::GqaExecutionEnvelope{min_tokens, max_tokens},
            1, static_cast<std::int32_t>(min_tokens), static_cast<std::int32_t>(max_tokens));
    }
    // The W8 A16 projections and the fused GDN/FFN forms require no caller workspace.
    return total;
}

void execute_layer_visit(const LayerVisitRequest& request) {
    const VisitExecutionProfile profile = plan_visit(request.visit);
    validate_request(request, profile);
    const std::int32_t T = request.hidden_in->ne[1];
    const TextLayerWeights& weights = *request.weights;
    cudaStream_t stream             = request.stream;
    WorkspaceArena& arena           = *request.workspace;

    auto scope = arena.scope();
    VisitBuffers<WorkspaceArena> b = allocate_visit_buffers(arena, profile, T);

    Tensor mhc_residual;   // [4096,2,T] streams entering the visit (mHC visits only)
    Tensor ffn_residual;   // [4096,2,T] streams after the attention merge, or the carry
    Tensor mixer_input;
    if (profile.is_mhc) {
        if (profile.mhc_boundary == MhcBoundary::Enter) {
            mhc_enter_streams(*request.hidden_in, b.enter_residual, stream);
            mhc_residual = b.enter_residual;
        } else {
            mhc_residual = mhc_unflatten_streams(*request.hidden_in);
        }
        ops::mhc_route(mhc_residual, *request.mhc_attn_weight, *request.mhc_attn_bias,
                       MhcConfig::streams, MhcConfig::iterations, MhcConfig::pre_scale,
                       b.branch0, b.post0, b.mix0, stream);
        mixer_input  = b.branch0;
        ffn_residual = mhc_residual;
    } else {
        const cudaError_t status =
            cudaMemcpyAsync(b.carry.data, request.hidden_in->data, request.hidden_in->bytes(),
                            cudaMemcpyDeviceToDevice, stream);
        require(status == cudaSuccess, "hidden_in copy failed");
        mixer_input  = b.carry;
        ffn_residual = b.carry;
    }

    if (profile.kind == VisitKind::FullAttention) {
        if (profile.is_mhc) {
            ffn_residual = run_full_attention(b, request, mixer_input, weights, ffn_residual,
                                              true, stream, arena);
        } else {
            run_full_attention(b, request, mixer_input, weights, ffn_residual, false, stream,
                               arena);
        }
    } else {
        GdnVisitStateViews state = gdn_state_views(request.gdn_state_base);
        run_gdn(b, request, mixer_input, weights, state, stream, arena);
        if (profile.is_mhc) {
            ops::mhc_merge(ffn_residual, b.attn_out, b.post0, b.mix0, MhcConfig::streams,
                           b.merged0, stream);
            ffn_residual = b.merged0;
        }
    }
    Tensor final_streams = ffn_residual;
    if (profile.is_mhc) {
        ops::mhc_route(final_streams, *request.mhc_ffn_weight, *request.mhc_ffn_bias,
                       MhcConfig::streams, MhcConfig::iterations, MhcConfig::pre_scale,
                       b.branch1, b.post1, b.mix1, stream);
        final_streams = run_ffn(b, b.branch1, final_streams, weights, true, stream);
    } else {
        run_ffn(b, ffn_residual, ffn_residual, weights, false, stream);
    }

    if (profile.is_mhc) {
        if (profile.mhc_boundary == MhcBoundary::Leave) {
            Tensor reduced = mhc_reduce_streams(final_streams, arena, stream);
            const cudaError_t status =
                cudaMemcpyAsync(request.hidden_out->data, reduced.data, reduced.bytes(),
                                cudaMemcpyDeviceToDevice, stream);
            require(status == cudaSuccess, "hidden_out copy failed");
        } else {
            const cudaError_t status =
                cudaMemcpyAsync(request.hidden_out->data, final_streams.data,
                                final_streams.bytes(), cudaMemcpyDeviceToDevice, stream);
            require(status == cudaSuccess, "hidden_out copy failed");
        }
        mhc_publish_state(final_streams, request.mhc_state_base, stream);
    } else {
        const cudaError_t status =
            cudaMemcpyAsync(request.hidden_out->data, b.carry.data, b.carry.bytes(),
                            cudaMemcpyDeviceToDevice, stream);
        require(status == cudaSuccess, "hidden_out copy failed");
    }
}

void mhc_enter_streams(const Tensor& hidden, Tensor& out, cudaStream_t stream) {
    require(hidden.dtype == DType::BF16, "mhc enter input must be BF16");
    require(hidden.data != nullptr, "mhc enter input data is required");
    require(hidden.ne[0] == TextConfig::hidden && hidden.ne[1] > 0 && hidden.ne[2] == 1 &&
                hidden.ne[3] == 1 && hidden.is_contiguous(),
            "mhc enter input shape mismatch");
    require(out.dtype == DType::BF16 && out.data != nullptr, "mhc enter output is required");
    require(out.ne[0] == TextConfig::hidden && out.ne[1] == MhcConfig::streams &&
                out.ne[2] == hidden.ne[1] && out.ne[3] == 1 && out.is_contiguous(),
            "mhc enter output must be contiguous [4096,2,T]");
    const std::size_t stream_bytes = hidden.bytes();
    const std::uint8_t* src = static_cast<const std::uint8_t*>(hidden.data);
    std::uint8_t* dst = static_cast<std::uint8_t*>(out.data);
    const cudaError_t first = cudaMemcpyAsync(dst, src, stream_bytes, cudaMemcpyDeviceToDevice,
                                              stream);
    require(first == cudaSuccess, "mhc enter stream 0 copy failed");
    const cudaError_t second = cudaMemcpyAsync(dst + stream_bytes, src, stream_bytes,
                                               cudaMemcpyDeviceToDevice, stream);
    require(second == cudaSuccess, "mhc enter stream 1 copy failed");
}

Tensor mhc_unflatten_streams(const Tensor& hidden) {
    require(hidden.dtype == DType::BF16, "mhc unflatten input must be BF16");
    require(hidden.data != nullptr, "mhc unflatten input data is required");
    require(hidden.ne[0] == MhcConfig::input_width && hidden.ne[1] > 0 && hidden.ne[2] == 1 &&
                hidden.ne[3] == 1 && hidden.is_contiguous(),
            "mhc unflatten input must be contiguous BF16 [8192,T]");
    return hidden.view({TextConfig::hidden, MhcConfig::streams, hidden.ne[1]});
}

Tensor mhc_flatten_streams(const Tensor& merged) {
    require(merged.dtype == DType::BF16, "mhc flatten input must be BF16");
    require(merged.data != nullptr, "mhc flatten input data is required");
    require(merged.ne[0] == TextConfig::hidden && merged.ne[1] == MhcConfig::streams &&
                merged.ne[2] > 0 && merged.ne[3] == 1,
            "mhc flatten input must be [4096,2,T]");
    return merged.view({MhcConfig::input_width, merged.ne[2]});
}

Tensor mhc_reduce_streams(const Tensor& merged, WorkspaceArena& workspace, cudaStream_t stream) {
    require(merged.dtype == DType::BF16 && merged.data != nullptr,
            "mhc reduce input must be BF16");
    require(merged.ne[0] == TextConfig::hidden && merged.ne[1] == MhcConfig::streams &&
                merged.ne[2] > 0 && merged.ne[3] == 1,
            "mhc reduce input must be [4096,2,T]");
    const std::int32_t T = merged.ne[2];

    Tensor branch = workspace.alloc(DType::BF16, {TextConfig::hidden, T});
    Tensor post   = workspace.alloc(DType::FP32, {MhcConfig::streams, T});
    Tensor mix    = workspace.alloc(DType::FP32, {MhcConfig::streams, MhcConfig::streams, T});
    Tensor out    = workspace.alloc(DType::BF16, {TextConfig::hidden, MhcConfig::streams, T});
    Tensor reduced = workspace.alloc(DType::BF16, {TextConfig::hidden, T});
    const cudaError_t zero = cudaMemsetAsync(branch.data, 0, branch.bytes(), stream);
    require(zero == cudaSuccess, "mhc reduce zero-fill failed");
    const cudaError_t zero_post = cudaMemsetAsync(post.data, 0, post.bytes(), stream);
    require(zero_post == cudaSuccess, "mhc reduce zero-fill failed");
    std::vector<float> half(static_cast<std::size_t>(MhcConfig::streams) *
                                MhcConfig::streams * static_cast<std::size_t>(T),
                            0.5F);
    const cudaError_t fill = cudaMemcpyAsync(mix.data, half.data(), half.size() * sizeof(float),
                                             cudaMemcpyHostToDevice, stream);
    require(fill == cudaSuccess, "mhc reduce mix fill failed");

    ops::mhc_merge(merged, branch, post, mix, MhcConfig::streams, out, stream);
    const std::size_t stream_bytes = static_cast<std::size_t>(TextConfig::hidden) *
                                     static_cast<std::size_t>(T) * dtype_size(DType::BF16);
    const cudaError_t status =
        cudaMemcpyAsync(reduced.data, static_cast<const std::uint8_t*>(out.data) + stream_bytes,
                        stream_bytes, cudaMemcpyDeviceToDevice, stream);
    require(status == cudaSuccess, "mhc reduce copy failed");
    return reduced;
}

void mhc_publish_state(const Tensor& merged_streams, std::byte* slot_base, cudaStream_t stream) {
    require(slot_base != nullptr, "mhc state slot base is required");
    require(merged_streams.dtype == DType::BF16 && merged_streams.data != nullptr,
            "mhc publish input must be BF16");
    require(merged_streams.ne[0] == TextConfig::hidden &&
                merged_streams.ne[1] == MhcConfig::streams && merged_streams.ne[2] > 0 &&
                merged_streams.ne[3] == 1 && merged_streams.is_contiguous(),
            "mhc publish input must be contiguous [4096,2,T]");
    const std::int32_t T = merged_streams.ne[2];
    const std::size_t state_bytes =
        static_cast<std::size_t>(MhcConfig::input_width) * dtype_size(DType::BF16);
    const std::byte* token_state =
        static_cast<const std::byte*>(merged_streams.data) +
        (static_cast<std::size_t>(T) - 1) * state_bytes;
    const cudaError_t status =
        cudaMemcpyAsync(slot_base, token_state, state_bytes, cudaMemcpyDeviceToDevice, stream);
    require(status == cudaSuccess, "mhc state publish failed");
}

void transpose_mhc_router_weight(const Tensor& stored, Tensor& out, cudaStream_t stream) {
    require(stored.dtype == DType::FP32 && stored.data != nullptr && out.dtype == DType::FP32 &&
                out.data != nullptr,
            "mhc router transpose requires FP32 stored and output tensors");
    require(stored.ne[0] == MhcConfig::logits && stored.ne[1] == MhcConfig::input_width &&
                stored.ne[2] == 1 && stored.ne[3] == 1 && stored.is_contiguous(),
            "stored mHC router weight must be contiguous FP32 [8,8192]");
    require(out.ne[0] == MhcConfig::input_width && out.ne[1] == MhcConfig::logits &&
                out.ne[2] == 1 && out.ne[3] == 1 && out.is_contiguous(),
            "output mHC router weight must be contiguous FP32 [8192,8]");
    const std::size_t rows = static_cast<std::size_t>(MhcConfig::logits);
    const std::size_t cols = static_cast<std::size_t>(MhcConfig::input_width);
    std::vector<float> host(rows * cols);
    const cudaError_t down =
        cudaMemcpyAsync(host.data(), stored.data, host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    require(down == cudaSuccess, "mhc router transpose device-to-host failed");
    for (std::size_t logit = 0; logit < rows; ++logit) {
        for (std::size_t feature = 0; feature < cols; ++feature) {
            host[logit + feature * rows] = host[logit * cols + feature];
        }
    }
    const cudaError_t up =
        cudaMemcpyAsync(out.data, host.data(), host.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    require(up == cudaSuccess, "mhc router transpose host-to-device failed");
}

void transpose_gdn_convolution_weight(const Tensor& stored, Tensor& out, cudaStream_t stream) {
    require(stored.dtype == DType::BF16 && stored.data != nullptr && out.dtype == DType::BF16 &&
                out.data != nullptr,
            "gdn convolution transpose requires BF16 stored and output tensors");
    require(stored.ne[0] == TextConfig::gdn_conv_kernel &&
                stored.ne[1] == TextConfig::convolution_dim && stored.ne[2] == 1 &&
                stored.ne[3] == 1 && stored.is_contiguous(),
            "stored gdn convolution weight must be contiguous BF16 [4,C]");
    require(out.ne[0] == TextConfig::convolution_dim && out.ne[1] == TextConfig::gdn_conv_kernel &&
                out.ne[2] == 1 && out.ne[3] == 1 && out.is_contiguous(),
            "output gdn convolution weight must be contiguous BF16 [C,4]");
    const std::size_t channels = static_cast<std::size_t>(TextConfig::convolution_dim);
    const std::size_t kernel   = static_cast<std::size_t>(TextConfig::gdn_conv_kernel);
    std::vector<std::uint16_t> host(channels * kernel);
    const cudaError_t down = cudaMemcpyAsync(host.data(), stored.data,
                                             host.size() * sizeof(std::uint16_t),
                                             cudaMemcpyDeviceToHost, stream);
    require(down == cudaSuccess, "gdn convolution transpose device-to-host failed");
    for (std::size_t tap = 0; tap < kernel; ++tap) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            host[channel * kernel + tap] = host[tap * channels + channel];
        }
    }
    const cudaError_t up = cudaMemcpyAsync(out.data, host.data(),
                                           host.size() * sizeof(std::uint16_t),
                                           cudaMemcpyHostToDevice, stream);
    require(up == cudaSuccess, "gdn convolution transpose host-to-device failed");
}

} // namespace ninfer::targets::sensenova_loop_9b::detail