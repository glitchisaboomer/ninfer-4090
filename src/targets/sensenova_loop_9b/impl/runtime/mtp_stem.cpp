#include "targets/sensenova_loop_9b/impl/runtime/mtp_stem.h"

#include "core/device.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

constexpr std::size_t kAlign = 256;
constexpr std::int32_t kQueryKeyGateValueRows =
    2 * (TextConfig::query_size + TextConfig::kv_size);

std::size_t align_up(std::size_t bytes) noexcept {
    return ((bytes + kAlign - 1) / kAlign) * kAlign;
}

void require_shape(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
                   const char* label) {
    if (tensor.dtype != dtype || tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("mtp stem: ") + label +
                                    " must be a contiguous, non-null tensor of the right dtype");
    }
    const std::int32_t* dims = shape.begin();
    for (int i = 0; i < 4; ++i) {
        const std::int32_t expected = i < static_cast<int>(shape.size()) ? dims[i] : 1;
        if (tensor.ne[i] != expected) {
            throw std::invalid_argument(std::string("mtp stem: ") + label + " has the wrong shape");
        }
    }
}

// W8G32_F16S RowSplit plane layout: one row owns group_size codes plus one FP16
// scale per 32-column group. A row subview keeps the parent k/padding and only
// offsets the plane bases, matching the text layer executor's qualified route.
Weight w8_row_subview(const Weight& parent, std::int32_t row_begin, std::int32_t row_count) {
    if (parent.qtype != QType::W8G32_F16S || parent.layout != QuantLayout::RowSplit ||
        parent.qdata == nullptr || parent.scales == nullptr) {
        throw std::invalid_argument("mtp stem: W8 RowSplit parent is not bound");
    }
    const std::int32_t groups_per_row = parent.padded_shape[1] / parent.group_size;
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

PagedKVBatchLayerView mtp_kv_batch_view(std::byte* base, std::uint32_t kv_capacity,
                                        const Tensor& block_table) {
    std::byte* domain_base = base + StateImageLayout::kv_offset(
        StateLayout::full_attention_state_slots, kv_capacity, false, false);
    const std::size_t plane_bytes = StateImageLayout::kv_plane_bytes(kv_capacity);
    const std::int32_t pages = static_cast<std::int32_t>(StateImageLayout::kv_pages(kv_capacity));
    PagedKVBatchLayerView view;
    view.k_pages = Tensor(domain_base, DType::BF16,
                          {TextConfig::head_dim, static_cast<std::int32_t>(StateImageLayout::kKvPageTokens),
                           TextConfig::kv_heads, pages});
    view.v_pages = Tensor(domain_base + plane_bytes, DType::BF16,
                          {TextConfig::head_dim, static_cast<std::int32_t>(StateImageLayout::kKvPageTokens),
                           TextConfig::kv_heads, pages});
    view.block_tables = block_table.view({pages, 1});
    view.head_dim     = TextConfig::head_dim;
    view.num_kv_heads = TextConfig::kv_heads;
    view.dtype        = DType::BF16;
    view.quant_group  = 0;
    return view;
}

} // namespace

std::size_t mtp_stem_workspace_capacity_bytes(std::uint32_t tokens, std::uint32_t visible_keys) {
    if (tokens == 0 || visible_keys < tokens) {
        throw std::invalid_argument("mtp stem: invalid token or visible-key extent");
    }
    const auto t = static_cast<std::int64_t>(tokens);
    std::size_t bytes = 0;
    const auto add = [&](std::int64_t rows) { bytes += align_up(rows * t * sizeof(std::uint16_t)); };
    add(TextConfig::hidden);                  // embed
    add(MhcConfig::input_width);              // packed [e;h]
    add(TextConfig::hidden);                  // normed / fc output
    add(TextConfig::query_projection_rows);   // qkv
    add(TextConfig::query_size);              // qn
    add(TextConfig::kv_size);                 // kn
    add(TextConfig::query_size);              // gate
    add(TextConfig::kv_size);                 // v
    add(TextConfig::query_size);              // attention output
    add(TextConfig::hidden);                  // attention projection
    add(TextConfig::hidden);                  // carry
    add(TextConfig::hidden);                  // post norm
    add(TextConfig::intermediate);            // SwiGLU
    add(TextConfig::hidden);                  // down
    add(kDraftHeadRows);                      // shortlist logits
    bytes += ops::gqa_attention_workspace_capacity_bytes(
        TextConfig::query_heads, TextConfig::kv_heads, DType::BF16,
        ops::GqaExecutionEnvelope{tokens, visible_keys}, 1, static_cast<std::int32_t>(tokens),
        static_cast<std::int32_t>(tokens));
    return bytes;
}

void mtp_propose_tokens(const RuntimeModelView& model, cudaStream_t stream,
                        std::byte* mtp_state_base, std::uint32_t kv_capacity,
                        WorkspaceArena& workspace, const MtpStemBindings& bindings,
                        const Tensor& token_ids, const Tensor& target_hidden,
                        const Tensor& positions, std::uint32_t visible_keys,
                        Tensor& draft_ids, Tensor* draft_logits) {
    if (mtp_state_base == nullptr || model.mtp.attention.query_key_gate_value.qdata == nullptr ||
        model.draft_head.qdata == nullptr || model.draft_head_token_ids.data == nullptr) {
        throw std::invalid_argument("mtp stem: MTP weights, draft head, or state are not bound");
    }
    require_shape(token_ids, DType::I32, {token_ids.ne[0]}, "token ids");
    const std::int32_t tokens = token_ids.ne[0];
    if (tokens <= 0) { throw std::invalid_argument("mtp stem: empty proposal"); }
    require_shape(target_hidden, DType::BF16, {TextConfig::hidden, tokens}, "target hidden");
    require_shape(positions, DType::I32, {tokens}, "positions");
    require_shape(draft_ids, DType::I32, {tokens}, "draft ids");
    if (draft_logits != nullptr) {
        require_shape(*draft_logits, DType::BF16, {kDraftHeadRows, tokens}, "draft logits");
    }
    if (visible_keys < static_cast<std::uint32_t>(tokens) || visible_keys > kv_capacity ||
        kv_capacity % StateImageLayout::kKvPageTokens != 0) {
        throw std::invalid_argument("mtp stem: invalid visible-key envelope");
    }
    if (bindings.kv_block_table.data == nullptr || bindings.kv_table_rows.data == nullptr) {
        throw std::invalid_argument("mtp stem: KV page table is not bound");
    }

    const DeviceArena::Scope scope = workspace.scope();
    auto alloc = [&](DType dtype, std::initializer_list<std::int32_t> shape) {
        return workspace.alloc(dtype, shape);
    };
    Tensor embed;
    Tensor packed;
    Tensor normed;
    Tensor qkv;
    Tensor qn;
    Tensor kn;
    Tensor gate;
    Tensor v;
    Tensor attn;
    Tensor attn_out;
    Tensor carry;
    Tensor post_norm;
    Tensor ffn;
    Tensor down;
    Tensor shortlist_logits;
    try {
        embed = alloc(DType::BF16, {TextConfig::hidden, tokens});
        packed = alloc(DType::BF16, {MhcConfig::input_width, tokens});
        normed = alloc(DType::BF16, {TextConfig::hidden, tokens});
        qkv = alloc(DType::BF16, {TextConfig::query_projection_rows, tokens});
        qn = alloc(DType::BF16, {TextConfig::query_size, tokens});
        kn = alloc(DType::BF16, {TextConfig::kv_size, tokens});
        gate = alloc(DType::BF16, {TextConfig::query_size, tokens});
        v = alloc(DType::BF16, {TextConfig::kv_size, tokens});
        attn = alloc(DType::BF16, {TextConfig::query_size, tokens});
        attn_out = alloc(DType::BF16, {TextConfig::hidden, tokens});
        carry = alloc(DType::BF16, {TextConfig::hidden, tokens});
        post_norm = alloc(DType::BF16, {TextConfig::hidden, tokens});
        ffn = alloc(DType::BF16, {TextConfig::intermediate, tokens});
        down = alloc(DType::BF16, {TextConfig::hidden, tokens});
        shortlist_logits = alloc(DType::BF16, {kDraftHeadRows, tokens});
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "mtp stem: workspace is too small for one proposal step (needs " +
            std::to_string(mtp_stem_workspace_capacity_bytes(static_cast<std::uint32_t>(tokens),
                                                             visible_keys)) +
            " bytes)");
    }

    ops::embedding(token_ids, model.token_embedding, embed, stream);
    ops::rmsnorm(embed, model.mtp.embedding_norm, TextConfig::rms_epsilon, true, embed, stream);
    ops::rmsnorm(target_hidden, model.mtp.hidden_norm, TextConfig::rms_epsilon, true, normed,
                 stream);
    CUDA_CHECK(cudaMemcpyAsync(packed.data, embed.data, embed.bytes(), cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(packed.data) + embed.bytes(), normed.data,
                               normed.bytes(), cudaMemcpyDeviceToDevice, stream));
    ops::linear(packed, model.mtp.input_projection, normed, stream);
    ops::rmsnorm(normed, model.mtp.input_norm, TextConfig::rms_epsilon, true, normed, stream);

    const Weight& parent = model.mtp.attention.query_key_gate_value;
    ops::linear(normed, w8_row_subview(parent, 0, TextConfig::query_size), qn, stream);
    ops::linear(normed, w8_row_subview(parent, TextConfig::query_size, TextConfig::kv_size), kn,
                stream);
    ops::linear(normed, w8_row_subview(parent, TextConfig::query_size + TextConfig::kv_size,
                                       TextConfig::query_size), gate, stream);
    ops::linear(normed, w8_row_subview(parent, kQueryKeyGateValueRows - TextConfig::kv_size,
                                       TextConfig::kv_size), v, stream);

    Tensor q = qn.view({TextConfig::head_dim, TextConfig::query_heads, tokens});
    Tensor k = kn.view({TextConfig::head_dim, TextConfig::kv_heads, tokens});
    Tensor gate_heads = gate.view({TextConfig::head_dim, TextConfig::query_heads, tokens});
    Tensor v_heads = v.view({TextConfig::head_dim, TextConfig::kv_heads, tokens});
    Tensor qn_heads = qn.view({TextConfig::head_dim, TextConfig::query_heads, tokens});
    Tensor kn_heads = kn.view({TextConfig::head_dim, TextConfig::kv_heads, tokens});
    ops::rmsnorm(q, model.mtp.attention.query_norm, TextConfig::rms_epsilon, true, qn_heads, stream);
    ops::rmsnorm(k, model.mtp.attention.key_norm, TextConfig::rms_epsilon, true, kn_heads, stream);
    ops::rope(positions, TextConfig::rotary_dim, TextConfig::rope_theta, qn_heads, kn_heads,
              stream);

    Tensor attention_heads = attn.view({TextConfig::head_dim, TextConfig::query_heads, tokens});
    const PagedKVBatchLayerView mtp_kv =
        mtp_kv_batch_view(mtp_state_base, kv_capacity, bindings.kv_table_rows);
    ops::gqa_attention(qn_heads, kn_heads, v_heads, positions, Tensor{}, bindings.kv_table_rows,
                       kAttentionScale, mtp_kv,
                       ops::GqaExecutionEnvelope{static_cast<std::uint32_t>(tokens), visible_keys},
                       workspace, attention_heads,
                       stream);
    ops::sigmoid_mul(gate_heads, attention_heads, stream);
    ops::linear(attn, model.mtp.attention.output, attn_out, stream);
    CUDA_CHECK(cudaMemcpyAsync(carry.data, normed.data, normed.bytes(), cudaMemcpyDeviceToDevice,
                               stream));
    ops::residual_add(attn_out, carry, stream);
    ops::rmsnorm(carry, model.mtp.post_attention_norm, TextConfig::rms_epsilon, true, post_norm,
                 stream);
    ops::linear_swiglu_sensenova(post_norm, model.mtp.mlp_gate_up, ffn, stream);
    ops::linear(ffn, model.mtp.mlp_down, down, stream);
    ops::residual_add(down, carry, stream);
    ops::rmsnorm(carry, model.mtp.final_norm, TextConfig::rms_epsilon, true, normed, stream);

    Tensor& logits = draft_logits != nullptr ? *draft_logits : shortlist_logits;
    ops::linear(normed, model.draft_head, logits, stream);
    ops::argmax(logits, draft_ids, kDraftHeadRows, stream);
    ops::proposal_remap_token_ids(
        draft_ids, static_cast<const std::int32_t*>(model.draft_head_token_ids.data),
        kDraftHeadRows, stream);
}

} // namespace ninfer::targets::sensenova_loop_9b::detail
