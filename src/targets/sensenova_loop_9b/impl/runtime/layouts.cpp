#include "targets/sensenova_loop_9b/impl/runtime/layouts.h"

#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/load/bindings.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

bool supported_kv_mode(KvCacheStorage storage) {
    return storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Int8Group64;
}

std::size_t kv_bytes_per_page(KvCacheStorage storage, bool include_mtp) {
    const bool int8   = storage == KvCacheStorage::Int8Group64;
    const bool scales = int8;
    return checked_mul(StateLayout::total_kv_bytes_per_token(int8, scales, include_mtp),
                       kSenseNovaPageTokens, "sensenova_loop_9b: KV page bytes overflow");
}

} // namespace

void validate_sequence_inputs(const SequencePlanningInputs& inputs) {
    if (inputs.weights_profile != WeightsProfile::GroupwiseInt) {
        throw std::invalid_argument("sensenova_loop_9b: unsupported weights profile");
    }
    if (inputs.capacity == 0 || inputs.capacity > kNativeContext) {
        throw std::invalid_argument("sensenova_loop_9b: max_context exceeds native capacity");
    }
    if (inputs.max_concurrency == 0 || inputs.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("sensenova_loop_9b: invalid max_concurrency");
    }
    if (inputs.prefill_chunk == 0 || inputs.prefill_chunk > inputs.capacity) {
        throw std::invalid_argument("sensenova_loop_9b: invalid prefill chunk");
    }
    if (inputs.draft_window > kMaximumMtpDraftTokens) {
        throw std::invalid_argument("sensenova_loop_9b: draft window exceeds 15");
    }
    if (!supported_kv_mode(inputs.kv_cache)) {
        throw std::invalid_argument("sensenova_loop_9b: unsupported KV cache storage");
    }
}

runtime::SequenceCapacityCurve make_sequence_capacity_curve(const SequencePlanningInputs& inputs) {
    validate_sequence_inputs(inputs);
    const std::uint64_t logical_pages64 =
        1ULL + (static_cast<std::uint64_t>(inputs.capacity) - 1ULL) / kSenseNovaPageTokens;
    const std::uint64_t minimum64 =
        std::max<std::uint64_t>(logical_pages64, inputs.max_concurrency);
    const std::uint64_t maximum64 = checked_mul(minimum64, inputs.max_concurrency,
                                                "sensenova_loop_9b: page group count overflow");
    if (minimum64 > std::numeric_limits<std::uint32_t>::max() ||
        maximum64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("sensenova_loop_9b: page group count exceeds uint32");
    }
    const auto minimum = static_cast<std::uint32_t>(minimum64);
    const auto maximum = static_cast<std::uint32_t>(maximum64);
    const SequenceLayout minimum_layout = build_sequence_layout(inputs, minimum);
    const std::size_t stride =
        minimum < maximum ? kv_bytes_per_page(inputs.kv_cache, true) : std::size_t{0};
    return runtime::SequenceCapacityCurve{
        .main_page_tokens                     = kSenseNovaPageTokens,
        .minimum_main_page_groups             = minimum,
        .maximum_main_page_groups             = maximum,
        .minimum_device_reservation_bytes     = minimum_layout.device_reservation_bytes,
        .bytes_per_additional_main_page_group = stride,
    };
}

SequenceLayout build_sequence_layout(const SequencePlanningInputs& inputs,
                                     std::uint32_t main_page_groups) {
    if (main_page_groups == 0) { throw std::invalid_argument("sensenova_loop_9b: zero page groups"); }

    SequenceLayout layout;
    layout.inputs           = inputs;
    layout.capacity         = inputs.capacity;
    layout.kv_capacity      = main_page_groups * kSenseNovaPageTokens;
    layout.main_page_groups = main_page_groups;
    layout.prefill_chunk    = inputs.prefill_chunk;
    layout.draft_window     = inputs.draft_window;

    // State is deliberately reserved for all 56 logical visits: 14 full-attention KV views,
    // 42 GDN recurrent/conv slots, 48 mHC stream images, and one MTP KV domain.
    const std::size_t kv_bytes_per_group =
        checked_mul(main_page_groups, kv_bytes_per_page(inputs.kv_cache, true),
                    "sensenova_loop_9b: KV reservation overflow");
    layout.persistent_bytes = checked_add(StateLayout::total_fixed_state_bytes(),
                                          kv_bytes_per_group,
                                          "sensenova_loop_9b: fixed/KV overflow");
    constexpr std::size_t kWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;
    layout.workspace_bytes = kWorkspaceBytes;
    layout.graph_allowance_bytes =
        inputs.use_cuda_graph ? 16ULL * 1024ULL * 1024ULL : 0ULL;
    layout.request_transient_bytes = align_up(
        checked_mul(inputs.max_concurrency, 4096ULL,
                    "sensenova_loop_9b: request transient overflow"),
        256);
    layout.device_reservation_bytes = checked_add(
        checked_add(layout.persistent_bytes, layout.workspace_bytes,
                    "sensenova_loop_9b: persistent/workspace overflow"),
        checked_add(layout.graph_allowance_bytes, layout.request_transient_bytes,
                    "sensenova_loop_9b: graph/transient overflow"),
        "sensenova_loop_9b: sequence reservation overflow");
    return layout;
}

SequenceLayout finalize_sequence_layout(const SequencePlanningInputs& inputs,
                                        std::uint32_t main_page_groups) {
    const runtime::SequenceCapacityCurve curve = make_sequence_capacity_curve(inputs);
    if (main_page_groups < curve.minimum_main_page_groups ||
        main_page_groups > curve.maximum_main_page_groups) {
        throw std::invalid_argument("sensenova_loop_9b: page groups outside capacity curve");
    }
    return build_sequence_layout(inputs, main_page_groups);
}

} // namespace ninfer::targets::sensenova_loop_9b::detail
