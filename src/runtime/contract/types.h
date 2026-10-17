#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

using ::ninfer::FinishReason;
using ::ninfer::KvCapacityMode;
using ::ninfer::KvCapacityPolicy;
using ::ninfer::OutputChannel;
using ::ninfer::ResolvedSamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

// Engine has already selected the registered model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    bool enable_suffix_cache_reuse        = false;
    std::uint32_t suffix_reuse_max_spans  = 6;
    ninfer::ScrRelocationMode scr_mode    = ninfer::ScrRelocationMode::Tail;
    std::vector<std::uint32_t> token_mask;
    bool disable_speculation = false;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    StructuredOutputOptions structured_output;
};

struct OutputDecision {
    std::uint32_t accepted_tokens = 0;
    FinishReason finish_reason    = FinishReason::None;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Complete request-lifetime ownership in the three independently exhausted admission domains.
// Values are already rounded to the physical allocation granularity by the target.
struct AdmissionResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    // Prompt tokens the plan expects to serve by relocating resident KV instead of
    // prefilling. Zero unless the plan selected PrefixReusePath::SuffixRelocation.
    std::uint32_t relocated_prompt_tokens = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    std::size_t transient_bytes           = 0;
    std::size_t transient_alignment       = 1;
    AdmissionResources admission;
    std::uint64_t service_work_quanta = 0;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    // Prompt tokens served by relocating a surviving span rather than by prefix reuse. Disjoint
    // from and additive with `reused_prompt_tokens`; reporting sums the two for the client-facing
    // cache hit. Carried separately because prefix reuse and Suffix Cache Reuse are indistinguishable
    // from `reused_prompt_tokens` alone, and a Suffix Cache Reuse measurement that cannot see its own
    // relocation reports nothing.
    std::uint32_t relocated_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::FullReset;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
    // Next-token logprob of each entry in `tokens`, in the same order, or empty when the target
    // produced the round without measuring one. A non-empty span always matches `tokens` in size.
    std::span<const float> logprobs;
    // Best-minus-second-best logit gap in nats for each entry, same order and same sizing rule as
    // `logprobs`. Empty means not measured, never that every margin was zero.
    std::span<const float> margins;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    // Same ragged layout as `tokens`: row r's logprobs start at r * row_stride. Empty means the
    // target did not measure this round's logprobs at all, never that they are zero.
    std::span<const float> logprobs;
    // Same ragged layout as `logprobs`.
    std::span<const float> margins;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    bool host_input_consumed              = false;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
