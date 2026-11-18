#pragma once

#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include "runtime/contract/types.h"
#include <ninfer/targets/sensenova_loop_9b/package.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

inline constexpr std::uint32_t kSenseNovaPageTokens = 128;

struct SequencePlanningInputs {
    WeightsProfile weights_profile = WeightsProfile::GroupwiseInt;
    std::uint32_t capacity         = 0;
    std::uint32_t max_concurrency  = 1;
    std::uint32_t prefill_chunk    = 0;
    std::uint32_t draft_window     = 0;
    KvCacheStorage kv_cache        = KvCacheStorage::BFloat16;
    bool use_cuda_graph            = true;
    bool enable_prompt_cache       = false;
    int device                     = 0;
};

struct SequenceLayout {
    SequencePlanningInputs inputs;
    std::uint32_t capacity           = 0;
    std::uint32_t kv_capacity        = 0;
    std::uint32_t main_page_groups   = 0;
    std::uint32_t prefill_chunk      = 0;
    std::uint32_t draft_window       = 0;
    std::size_t persistent_bytes     = 0;
    std::size_t workspace_bytes      = 0;
    std::size_t graph_allowance_bytes = 0;
    std::size_t request_transient_bytes = 0;
    std::size_t device_reservation_bytes = 0;
};

[[nodiscard]] void validate_sequence_inputs(const SequencePlanningInputs& inputs);
[[nodiscard]] runtime::SequenceCapacityCurve
make_sequence_capacity_curve(const SequencePlanningInputs& inputs);
[[nodiscard]] SequenceLayout build_sequence_layout(const SequencePlanningInputs& inputs,
                                                   std::uint32_t main_page_groups);
[[nodiscard]] SequenceLayout finalize_sequence_layout(const SequencePlanningInputs& inputs,
                                                      std::uint32_t main_page_groups);

} // namespace ninfer::targets::sensenova_loop_9b::detail
