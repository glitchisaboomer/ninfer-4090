#include "targets/sensenova_loop_9b/impl/runtime/text_execution.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/logprob_of_choice.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <new> // std::bad_alloc
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::sensenova_loop_9b::detail {

namespace {

// Start of the GDN region: immediately after all 15 KV domains of `kv_capacity`
// tokens (BF16, no scales — the only KV format this target stores).
constexpr std::size_t gdn_region_offset(std::uint32_t kv_capacity) noexcept {
    return StateImageLayout::total_kv_bytes(kv_capacity, false, false);
}

// Start of the mHC region: after the KV domains and the full GDN region
// (matching the Program's state image binding in runtime_stub).
constexpr std::size_t mhc_region_offset(std::uint32_t kv_capacity) noexcept {
    return gdn_region_offset(kv_capacity) +
           StateImageLayout::kGdnSlots * StateLayout::gdn_state_bytes_per_slot();
}

// DeviceArena::alloc's default alignment. The head stages its transient columns
// at that granularity so the Linear alignment contract is satisfied by
// construction.
constexpr std::size_t kArenaAlignment = 256;
constexpr std::size_t kBf16Bytes      = sizeof(std::uint16_t);

constexpr std::size_t round_up_to_arena(std::size_t bytes) noexcept {
    return ((bytes + kArenaAlignment - 1) / kArenaAlignment) * kArenaAlignment;
}

constexpr std::size_t head_normed_bytes() noexcept {
    return TextConfig::hidden * kBf16Bytes; // BF16 [hidden,1]
}

constexpr std::size_t head_logits_bytes() noexcept {
    return TextConfig::output_rows * kBf16Bytes; // BF16 [output_rows,1]
}

void require_fp32_scalar(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::FP32 || tensor.ne[0] != 1 || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("project_logits_and_sample: ") + label +
                                    " must be FP32 [1]");
    }
    if (tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("project_logits_and_sample: ") + label +
                                    " must be contiguous and non-null");
    }
}

} // namespace

TextExecutionContext::TextExecutionContext(const RuntimeModelView& model, cudaStream_t stream,
                                           std::byte* state_image_base,
                                           std::uint32_t kv_capacity, WorkspaceArena& workspace,
                                           TextExecutionBindings bindings,
                                           LayerVisitExecutor executor)
    : model_(model), stream_(stream), state_image_base_(state_image_base),
      kv_capacity_(kv_capacity), workspace_(workspace), bindings_(bindings),
      executor_(executor) {
    if (stream == nullptr) {
        throw std::invalid_argument("TextExecutionContext requires a non-null CUDA stream");
    }
    if (state_image_base == nullptr) {
        throw std::invalid_argument(
            "TextExecutionContext requires a non-null state image base");
    }
    if (kv_capacity == 0) {
        throw std::invalid_argument("TextExecutionContext requires a positive KV capacity");
    }
    if (workspace.capacity() == 0) {
        throw std::invalid_argument(
            "TextExecutionContext requires a workspace with positive capacity");
    }
    if (executor == nullptr) {
        throw std::invalid_argument("TextExecutionContext requires a visit executor");
    }
}

std::byte* TextExecutionContext::gdn_state_base(std::size_t visit) const {
    if (visit >= LoopPlan::kVisits || !LoopPlan::is_gdn(visit)) {
        throw std::out_of_range("SenseNova GDN visit is outside the plan");
    }
    return state_image_base_ + gdn_region_offset(kv_capacity_) +
           StateImageLayout::gdn_offset(StateImageLayout::gdn_slot_for_visit(visit));
}

std::byte* TextExecutionContext::mhc_state_base(std::size_t visit) const {
    if (visit >= LoopPlan::kVisits || !LoopPlan::is_mhc(visit)) {
        throw std::out_of_range("SenseNova mHC visit is outside the plan");
    }
    return state_image_base_ + mhc_region_offset(kv_capacity_) +
           StateImageLayout::mhc_offset(StateImageLayout::mhc_slot_for_visit(visit));
}

std::size_t TextExecutionContext::kv_domain(std::size_t visit) const noexcept {
    return StateImageLayout::kv_domain_for_visit(visit);
}

std::size_t TextExecutionContext::head_workspace_capacity_bytes(bool measure_logprob,
                                                                bool measure_margin) noexcept {
    std::size_t bytes = round_up_to_arena(head_normed_bytes()) +
                        round_up_to_arena(head_logits_bytes());
    if (measure_logprob && !measure_margin) {
        bytes += round_up_to_arena(sizeof(float));
    }
    return bytes;
}

void TextExecutionContext::embed_tokens(const Tensor& ids, Tensor& hidden) const {
    if (ids.dtype != DType::I32 || ids.ne[1] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("embed_tokens: ids must be I32 [T]");
    }
    if (ids.data == nullptr || !ids.is_contiguous()) {
        throw std::invalid_argument("embed_tokens: ids must be contiguous and non-null");
    }
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != ids.ne[0] ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument(
            "embed_tokens: hidden must be BF16 [4096,T] with T equal to the token count");
    }
    if (hidden.data == nullptr || !hidden.is_contiguous()) {
        throw std::invalid_argument("embed_tokens: hidden must be contiguous and non-null");
    }

    const Weight& table = model_.token_embedding;
    if (table.qdata == nullptr) {
        throw std::invalid_argument("embed_tokens: model token embedding is not bound");
    }
    if (table.ndim != 2 || table.shape[0] <= 0 || table.shape[1] != TextConfig::hidden) {
        throw std::invalid_argument("embed_tokens: token embedding must be [vocab,4096]");
    }

    ops::embedding(ids, table, hidden, stream_);
}

void TextExecutionContext::project_logits_and_sample(const Tensor& hidden, Tensor& sampled,
                                                     Tensor* logprob, Tensor* margin) const {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] <= 0 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument("project_logits_and_sample: hidden must be BF16 [4096,T]");
    }
    if (hidden.data == nullptr || !hidden.is_contiguous()) {
        throw std::invalid_argument(
            "project_logits_and_sample: hidden must be contiguous and non-null");
    }
    if (sampled.dtype != DType::I32 || sampled.ne[0] != 1 || sampled.ne[1] != 1 ||
        sampled.ne[2] != 1 || sampled.ne[3] != 1) {
        throw std::invalid_argument(
            "project_logits_and_sample: sampled must be I32 [1]");
    }
    if (sampled.data == nullptr || !sampled.is_contiguous()) {
        throw std::invalid_argument(
            "project_logits_and_sample: sampled must be contiguous and non-null");
    }

    if (margin != nullptr && logprob == nullptr) {
        throw std::invalid_argument(
            "project_logits_and_sample: margin requires a logprob destination");
    }
    if (logprob != nullptr) { require_fp32_scalar(*logprob, "logprob"); }
    if (margin != nullptr) {
        require_fp32_scalar(*margin, "margin");
        if (margin->data == logprob->data) {
            throw std::invalid_argument(
                "project_logits_and_sample: logprob and margin must not alias");
        }
    }

    if (model_.final_norm.dtype != DType::BF16 || model_.final_norm.data == nullptr ||
        model_.final_norm.ne[0] != TextConfig::hidden || model_.final_norm.ne[1] != 1 ||
        model_.final_norm.ne[2] != 1 || model_.final_norm.ne[3] != 1) {
        throw std::invalid_argument(
            "project_logits_and_sample: final norm is not bound as BF16 [4096]");
    }
    const Weight& head = model_.output_head;
    if (head.qdata == nullptr) {
        throw std::invalid_argument("project_logits_and_sample: output head is not bound");
    }
    if (head.n != TextConfig::output_rows || head.k != TextConfig::hidden) {
        throw std::invalid_argument(
            "project_logits_and_sample: output head must be [248320,4096]");
    }

    // The head's transient columns live in the context workspace and are released
    // when this call returns, so nothing the caller staged before it is lost.
    const DeviceArena::Scope scope = workspace_.scope();
    const bool measure_logprob     = logprob != nullptr;
    Tensor normed;
    Tensor logits;
    Tensor margin_scratch;
    try {
        normed = workspace_.alloc(DType::BF16, {TextConfig::hidden, 1});
        logits = workspace_.alloc(DType::BF16, {TextConfig::output_rows, 1});
        if (measure_logprob && margin == nullptr) {
            margin_scratch = workspace_.alloc(DType::FP32, {1});
        }
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "project_logits_and_sample: workspace capacity is too small (needs " +
            std::to_string(head_workspace_capacity_bytes(measure_logprob, margin != nullptr)) +
            " bytes)");
    }

    // Only the last column is projected: a prefill of T tokens produces one token.
    const Tensor last_column = hidden.slice(1, hidden.ne[1] - 1, 1);
    ops::rmsnorm(last_column, model_.final_norm, TextConfig::rms_epsilon, true, normed, stream_);
    ops::linear(normed, head, logits, stream_);
    ops::argmax(logits, sampled, TextConfig::token_domain, stream_);

    if (!measure_logprob) { return; }
    ops::logprob_of_choice(logits, sampled, TextConfig::token_domain, *logprob,
                           margin != nullptr ? *margin : margin_scratch, stream_);
}

std::size_t TextExecutionContext::forward_workspace_capacity_bytes(
    std::uint32_t max_tokens, std::uint32_t max_visible_keys) {
    if (max_tokens == 0 || max_visible_keys < max_tokens) {
        throw std::invalid_argument(
            "forward workspace: visible keys must cover the token extent");
    }
    // Hidden state plus the two mHC stream buffers, all sized by the pass width.
    std::size_t bytes = round_up_to_arena(TextConfig::hidden * max_tokens * kBf16Bytes);
    bytes += 2 * round_up_to_arena(MhcConfig::input_width * max_tokens * kBf16Bytes);

    // The visits run one at a time and each nests its own arena scope, so the worst single visit
    // bounds the rest. The executor's envelope promise is the longer of the token extent and the
    // causal history, which is what layer_visit_workspace_capacity_bytes prices.
    const std::uint32_t bound = std::max(max_tokens, max_visible_keys);
    std::size_t worst_visit   = 0;
    for (std::size_t visit = 0; visit < LoopPlan::kVisits; ++visit) {
        worst_visit = std::max(
            worst_visit, layer_visit_workspace_capacity_bytes(static_cast<std::uint32_t>(visit),
                                                             max_tokens, bound));
    }
    bytes += round_up_to_arena(worst_visit);
    bytes += head_workspace_capacity_bytes(true, true);
    return bytes;
}

void TextExecutionContext::forward_visits(Tensor& hidden, const Tensor& positions,
                                          std::uint32_t visible_keys) const {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] <= 0 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument("forward_visits: hidden must be BF16 [4096,T]");
    }
    if (hidden.data == nullptr || !hidden.is_contiguous()) {
        throw std::invalid_argument("forward_visits: hidden must be contiguous and non-null");
    }
    const std::int32_t tokens = hidden.ne[1];
    if (positions.dtype != DType::I32 || positions.ne[0] != tokens || positions.ne[1] != 1 ||
        positions.ne[2] != 1 || positions.ne[3] != 1) {
        throw std::invalid_argument("forward_visits: positions must be I32 [T]");
    }
    if (positions.data == nullptr || !positions.is_contiguous()) {
        throw std::invalid_argument("forward_visits: positions must be contiguous and non-null");
    }
    if (visible_keys < static_cast<std::uint32_t>(tokens) || visible_keys > kv_capacity_) {
        throw std::invalid_argument(
            "forward_visits: visible keys must cover T and fit the declared KV capacity");
    }
    if (kv_capacity_ % StateImageLayout::kKvPageTokens != 0) {
        throw std::invalid_argument(
            "forward_visits: KV capacity must be a whole number of state-image pages");
    }

    // Every asset a visit will read is checked here, once, so a missing preparation fails before
    // any executor sees a null pointer. In particular a full-attention visit is never handed a
    // null KV domain.
    const std::size_t pages = StateImageLayout::kv_pages(kv_capacity_);
    if (bindings_.kv_block_table.dtype != DType::I32 ||
        bindings_.kv_block_table.ne[0] != static_cast<std::int32_t>(pages) ||
        bindings_.kv_block_table.ne[1] != 1 || bindings_.kv_block_table.ne[2] != 1 ||
        bindings_.kv_block_table.ne[3] != 1 || bindings_.kv_block_table.data == nullptr ||
        !bindings_.kv_block_table.is_contiguous()) {
        throw std::invalid_argument(
            "forward_visits requires the identity KV block table as I32 [kv_pages]");
    }
    if (bindings_.kv_table_rows.dtype != DType::I32 || bindings_.kv_table_rows.ne[0] != 1 ||
        bindings_.kv_table_rows.ne[1] != 1 || bindings_.kv_table_rows.ne[2] != 1 ||
        bindings_.kv_table_rows.ne[3] != 1 || bindings_.kv_table_rows.data == nullptr ||
        !bindings_.kv_table_rows.is_contiguous()) {
        throw std::invalid_argument("forward_visits requires the B=1 KV table-row selector");
    }
    if (bindings_.gdn_convolution.size() < static_cast<std::size_t>(TextConfig::layers)) {
        throw std::invalid_argument(
            "forward_visits requires op-form GDN convolution weights per physical layer");
    }
    if (bindings_.mhc_router_weight.size() < static_cast<std::size_t>(2 * TextConfig::layers)) {
        throw std::invalid_argument(
            "forward_visits requires op-form mHC router weights per physical layer and branch");
    }

    const DeviceArena::Scope scope = workspace_.scope();
    Tensor streams_a;
    Tensor streams_b;
    try {
        streams_a = workspace_.alloc(DType::BF16, {MhcConfig::input_width, tokens});
        streams_b = workspace_.alloc(DType::BF16, {MhcConfig::input_width, tokens});
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "forward_visits: workspace capacity is too small (needs " +
            std::to_string(forward_workspace_capacity_bytes(static_cast<std::uint32_t>(tokens),
                                                           visible_keys)) +
            " bytes)");
    }

    Tensor* stream_current = &streams_a;
    Tensor* stream_next    = &streams_b;
    for (std::size_t visit = 0; visit < LoopPlan::kVisits; ++visit) {
        const VisitExecutionProfile profile = plan_visit(static_cast<std::uint32_t>(visit));
        const TextLayerWeights& weights     = model_.layers[profile.physical_layer];

        LayerVisitRequest request;
        request.visit     = static_cast<std::uint32_t>(visit);
        request.positions = &positions;
        request.weights   = &weights;
        request.stream    = stream_;
        request.workspace = &workspace_;
        request.attention_envelope =
            ops::GqaExecutionEnvelope{static_cast<std::uint32_t>(tokens), visible_keys};

        // The mHC boundary is the only place the carried width changes: enter widens 4096 -> 8192,
        // interior visits ping-pong between the two stream buffers, and leave narrows 8192 -> 4096.
        if (profile.mhc_boundary == MhcBoundary::Leave) {
            request.hidden_in  = stream_current;
            request.hidden_out = &hidden;
        } else if (profile.mhc_boundary == MhcBoundary::Enter) {
            request.hidden_in  = &hidden;
            request.hidden_out = stream_current;
        } else if (profile.is_mhc) {
            request.hidden_in  = stream_current;
            request.hidden_out = stream_next;
            std::swap(stream_current, stream_next);
        } else {
            request.hidden_in  = &hidden;
            request.hidden_out = &hidden;
        }

        if (profile.kind == VisitKind::FullAttention) {
            const std::size_t domain = kv_domain(visit);
            request.attention_kv_domain_base =
                static_cast<std::byte*>(state_image_base_) +
                StateImageLayout::kv_offset(domain, kv_capacity_, false, false);
            request.attention_kv_domain_tokens = kv_capacity_;
            request.attention_kv_block_table   = &bindings_.kv_block_table;
            request.attention_kv_table_rows    = &bindings_.kv_table_rows;
        } else {
            request.gdn_state_base          = gdn_state_base(visit);
            request.gdn_convolution_weight  = &bindings_.gdn_convolution[profile.physical_layer];
        }

        if (profile.is_mhc) {
            request.mhc_state_base      = mhc_state_base(visit);
            request.mhc_attn_weight     = &bindings_.mhc_router_weight[2 * profile.physical_layer];
            request.mhc_attn_bias       = &weights.mhc_attn.bias;
            request.mhc_ffn_weight      = &bindings_.mhc_router_weight[2 * profile.physical_layer + 1];
            request.mhc_ffn_bias        = &weights.mhc_ffn.bias;
        }

        executor_(request);
    }
}

void TextExecutionContext::forward_full(const Tensor& ids, const Tensor& positions,
                                        std::uint32_t visible_keys, Tensor& sampled,
                                        Tensor* logprob, Tensor* margin) const {
    if (ids.dtype != DType::I32 || ids.ne[0] <= 0 || ids.ne[1] != 1 || ids.ne[2] != 1 ||
        ids.ne[3] != 1) {
        throw std::invalid_argument("forward_full: ids must be I32 [T]");
    }
    if (positions.dtype != DType::I32 || positions.ne[0] != ids.ne[0]) {
        throw std::invalid_argument("forward_full: positions must be I32 [T] with one per id");
    }

    const DeviceArena::Scope scope = workspace_.scope();
    Tensor hidden;
    try {
        hidden = workspace_.alloc(DType::BF16, {TextConfig::hidden, ids.ne[0]});
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "forward_full: workspace capacity is too small (needs " +
            std::to_string(forward_workspace_capacity_bytes(
                static_cast<std::uint32_t>(ids.ne[0]), visible_keys)) +
            " bytes)");
    }

    embed_tokens(ids, hidden);
    forward_visits(hidden, positions, visible_keys);
    project_logits_and_sample(hidden, sampled, logprob, margin);
}

} // namespace ninfer::targets::sensenova_loop_9b::detail