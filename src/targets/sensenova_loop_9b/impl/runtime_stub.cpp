#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

#include "core/device.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"
#include "targets/sensenova_loop_9b/impl/runtime/layouts.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

SequencePlanningInputs planning_inputs(const EngineOptions& options,
                                       WeightsProfile weights_profile) {
    return SequencePlanningInputs{
        .weights_profile = weights_profile,
        .capacity = options.max_context,
        .max_concurrency = options.max_concurrency,
        .prefill_chunk = std::min(options.prefill_chunk, options.max_context),
        .draft_window = options.speculative.draft_tokens,
        .kv_cache = options.kv_cache,
        .use_cuda_graph = options.use_cuda_graph,
        .enable_prompt_cache = options.enable_prompt_cache,
        .device = options.device,
    };
}

} // namespace

runtime::SequenceCapacityCurve make_curve(const EngineOptions& options) {
    return make_sequence_capacity_curve(planning_inputs(options, WeightsProfile::GroupwiseInt));
}

struct RequestBasePlan::Impl {
    runtime::RequestPlanSummary summary;
};

RequestBasePlan::RequestBasePlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;
const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

struct RequestPlan::Impl {
    runtime::RequestPlanSummary summary;
};

RequestPlan::RequestPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
RequestPlan::RequestPlan(RequestPlan&&) noexcept            = default;
RequestPlan& RequestPlan::operator=(RequestPlan&&) noexcept = default;
RequestPlan::~RequestPlan()                                 = default;
const runtime::RequestPlanSummary& RequestPlan::summary() const noexcept { return impl_->summary; }

struct SequencePlan::Impl {
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t max_concurrency           = 1;
    std::size_t device_reservation_bytes    = 0;
    std::size_t workspace_bytes             = 0;
    std::size_t request_transient_bytes     = 0;
};

SequencePlan::SequencePlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
SequencePlan::SequencePlan(SequencePlan&&) noexcept            = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;
SequencePlan::~SequencePlan()                                  = default;
std::uint32_t SequencePlan::capacity() const noexcept { return impl_->capacity; }
std::uint32_t SequencePlan::kv_capacity() const noexcept { return impl_->kv_capacity; }
std::uint32_t SequencePlan::max_concurrency() const noexcept { return impl_->max_concurrency; }
std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_->device_reservation_bytes;
}
std::size_t SequencePlan::workspace_capacity_bytes() const noexcept { return impl_->workspace_bytes; }
std::size_t SequencePlan::request_transient_capacity_bytes() const noexcept {
    return impl_->request_transient_bytes;
}

struct SequencePlanner::Impl {
    EngineOptions options;
    WeightsProfile weights_profile;
    runtime::SequenceCapacityCurve curve;
};

SequencePlanner SequencePlanner::create(const EngineOptions& options, WeightsProfile weights_profile) {
    auto impl                  = std::make_unique<Impl>();
    impl->options              = options;
    impl->weights_profile      = weights_profile;
    impl->curve                = make_curve(options);
    return SequencePlanner(std::move(impl));
}

SequencePlanner::SequencePlanner(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept            = default;
SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;
SequencePlanner::~SequencePlanner()                                     = default;
const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    return impl_->curve;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (main_page_groups < impl_->curve.minimum_main_page_groups ||
        main_page_groups > impl_->curve.maximum_main_page_groups) {
        throw std::invalid_argument("sensenova_loop_9b: invalid main page groups");
    }
    const SequenceLayout layout = finalize_sequence_layout(planning_inputs(impl_->options, impl_->weights_profile), main_page_groups);
    auto impl             = std::make_unique<SequencePlan::Impl>();
    impl->capacity        = layout.capacity;
    impl->kv_capacity     = layout.kv_capacity;
    impl->max_concurrency = layout.inputs.max_concurrency;
    impl->workspace_bytes = layout.workspace_bytes;
    impl->request_transient_bytes = layout.request_transient_bytes;
    impl->device_reservation_bytes = layout.device_reservation_bytes;
    return SequencePlan(std::move(impl));
}

struct Program::Impl {
    MemorySummary memory;
    runtime::AdmissionResources admission;
};

std::unique_ptr<Program> Program::create(std::uint32_t active_lanes, std::uint32_t kv_pages,
                                         std::uint32_t context, std::size_t reservation_bytes) {
    auto impl                            = std::make_unique<Impl>();
    impl->admission.active_lanes         = active_lanes;
    impl->admission.main_kv_pages        = kv_pages;
    impl->memory.max_context             = context;
    impl->memory.kv_capacity             = kv_pages;
    impl->memory.kv_capacity_page_groups = kv_pages;
    impl->memory.runtime_reservation_bytes = reservation_bytes;
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

Program::Program(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() noexcept                         = default;

RequestBasePlan Program::plan_request_base(const qwen3_6::PreparedPrompt&,
                                           const runtime::ResolvedExecutionOptions& options) {
    auto impl               = std::make_unique<RequestBasePlan::Impl>();
    impl->summary.prompt_tokens = 0;
    impl->summary.requested_output_tokens = options.requested_output_tokens;
    impl->summary.effective_output_tokens = options.requested_output_tokens;
    impl->summary.transient_bytes         = 4096;
    impl->summary.transient_alignment     = 256;
    impl->summary.admission               = impl_->admission;
    return RequestBasePlan(std::move(impl));
}

RequestPlan Program::plan_request_for_lane(std::uint32_t, const qwen3_6::PreparedPrompt&,
                                           const RequestBasePlan& base) {
    auto impl      = std::make_unique<RequestPlan::Impl>();
    impl->summary  = base.summary();
    return RequestPlan(std::move(impl));
}

bool Program::can_admit_lane(std::uint32_t lane, const RequestPlan&) const noexcept {
    return lane < impl_->admission.active_lanes;
}
bool Program::can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                     const RequestPlan&) const noexcept {
    return lane < impl_->admission.active_lanes;
}
runtime::AdmissionResources Program::admission_capacity() const noexcept { return impl_->admission; }

runtime::PrefillStepResult Program::start_prefill_lane(std::uint32_t, qwen3_6::PreparedPrompt&&,
                                                       RequestPlan&&, runtime::TransientRegion) {
    throw std::logic_error("sensenova_loop_9b: text execution is not implemented yet");
}
runtime::PrefillStepResult Program::advance_prefill_lane(std::uint32_t) {
    throw std::logic_error("sensenova_loop_9b: text execution is not implemented yet");
}
runtime::BatchedGeneratedRound Program::decode_batch(std::span<const std::uint32_t>,
                                                     std::span<const runtime::RoundBudget>) {
    throw std::logic_error("sensenova_loop_9b: text execution is not implemented yet");
}
void Program::set_token_mask_lane(std::uint32_t, std::span<const std::uint32_t>) {}
void Program::resolve_prefill_lane(std::uint32_t, bool) {}
void Program::resolve_pending_batch(std::span<const std::uint32_t>, std::span<const std::uint32_t>,
                                    std::span<const std::uint8_t>, std::span<const std::uint8_t>) {}
void Program::abort_lane(std::uint32_t) noexcept {}
bool Program::has_retained_lane(std::uint32_t) const noexcept { return false; }
std::uint32_t Program::retained_lane_depth(std::uint32_t) const noexcept { return 0; }
void Program::evict_retained_lane(std::uint32_t) noexcept {}
GenerationTimings Program::generation_timings_lane(std::uint32_t) const noexcept { return {}; }
SpeculativeStats Program::speculative_stats_lane(std::uint32_t) const noexcept { return {}; }
void Program::snapshot_lane_to_disk(std::uint32_t, DiskStateCache&) {}
void Program::snapshot_turn_checkpoint_to_disk(std::uint32_t, DiskStateCache&) {}
void Program::set_disk_state_cache(DiskStateCache*) noexcept {}
std::string Program::config_signature_slug() const { return "sensenova_loop_9b_stage_b"; }
MemorySummary Program::memory_summary() const noexcept { return impl_->memory; }
void Program::reset_memory_peaks() noexcept {}

} // namespace ninfer::targets::sensenova_loop_9b::detail
