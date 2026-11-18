#include "targets/sensenova_loop_9b/impl/runtime/text_execution.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/logprob_of_choice.h"
#include "ninfer/ops/rmsnorm.h"

#include <new> // std::bad_alloc
#include <stdexcept>
#include <string>

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
                                           std::uint32_t kv_capacity, WorkspaceArena& workspace)
    : model_(model), stream_(stream), state_image_base_(state_image_base),
      kv_capacity_(kv_capacity), workspace_(workspace) {
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
}

std::byte* TextExecutionContext::gdn_state_base(std::size_t visit) const {
    if (visit >= StateImageLayout::kGdnSlots) {
        throw std::out_of_range("SenseNova GDN visit slot is out of range");
    }
    return state_image_base_ + gdn_region_offset(kv_capacity_) +
           StateImageLayout::gdn_offset(visit);
}

std::byte* TextExecutionContext::mhc_state_base(std::size_t visit) const {
    if (visit >= StateImageLayout::kMhcSlots) {
        throw std::out_of_range("SenseNova mHC visit slot is out of range");
    }
    return state_image_base_ + mhc_region_offset(kv_capacity_) +
           StateImageLayout::mhc_offset(visit);
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

} // namespace ninfer::targets::sensenova_loop_9b::detail