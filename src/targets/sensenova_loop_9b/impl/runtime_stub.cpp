#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"
#include "targets/sensenova_loop_9b/impl/runtime/layouts.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <array>
#include <vector>
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

RequestPlan RequestPlanForTest::make() {
    auto impl = std::make_unique<RequestPlan::Impl>();
    impl->summary.admission = {1, 1, 0};
    impl->summary.service_work_quanta = 1;
    return RequestPlan(std::move(impl));
}

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

enum class Lifecycle : std::uint8_t { Free, Prefilling, Active, Pending, Retained };

struct LaneState {
    Lifecycle lifecycle = Lifecycle::Free;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t generated_tokens = 0;
    std::uint32_t frontier = 0;
    std::uint32_t page_entitlement = 0;
    std::vector<TokenId> licensed_tokens;
    ops::SamplingConfig sampling{};
    std::array<std::uint32_t, kMaximumConcurrency> token_mask{};
};

struct Program::Impl {
    DeviceBuffer state_images;
    std::size_t state_image_bytes = 0;
    std::size_t state_kv_bytes    = 0;
    std::size_t state_gdn_bytes   = 0;
    std::size_t state_mhc_bytes   = 0;
    MemorySummary memory;
    runtime::AdmissionResources admission;
    std::uint32_t max_concurrency = 1;
    std::uint32_t kv_capacity = 0;
    std::array<LaneState, kMaximumConcurrency> lanes;
    std::array<TokenId, kMaximumConcurrency> pending_tokens{};
    std::array<float, kMaximumConcurrency> pending_logprobs{};
    std::vector<TokenId> egress_tokens;
    std::vector<float> egress_logprobs;
    std::vector<std::int32_t> egress_row_counts;
    std::size_t egress_span = 1;
};

std::unique_ptr<Program> Program::create(std::uint32_t active_lanes, std::uint32_t kv_pages,
                                         std::uint32_t context, std::size_t reservation_bytes) {
    if (active_lanes == 0 || active_lanes > kMaximumConcurrency) {
        throw std::invalid_argument("sensenova_loop_9b: invalid active lane count");
    }
    auto impl                            = std::make_unique<Impl>();
    impl->admission.active_lanes         = active_lanes;
    impl->admission.main_kv_pages        = kv_pages;
    impl->memory.max_context             = context;
    impl->memory.kv_capacity             = kv_pages * kSenseNovaPageTokens;
    impl->memory.kv_capacity_page_groups = kv_pages;
    impl->memory.runtime_reservation_bytes = reservation_bytes;
    impl->max_concurrency                = active_lanes;
    impl->kv_capacity                    = kv_pages * kSenseNovaPageTokens;

    // Use the declared KV capacity as the fixed per-domain token extent. The region offsets are
    // derived from StateImageLayout, so every logical visit has a stable address for graph replay.
    const std::size_t kv_bytes = StateImageLayout::total_kv_bytes(
        impl->kv_capacity, false, false);
    const std::size_t fixed_bytes = StateImageLayout::fixed_bytes();
    impl->state_kv_bytes           = kv_bytes;
    impl->state_gdn_bytes          = StateImageLayout::kGdnSlots *
                                     StateLayout::gdn_state_bytes_per_slot();
    impl->state_mhc_bytes          = StateImageLayout::kMhcSlots *
                                     StateLayout::mhc_state_bytes_per_slot();
    impl->state_image_bytes        = kv_bytes + fixed_bytes;
    impl->state_images             = DeviceBuffer(impl->state_image_bytes);
    impl->state_images.fill(0);
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

LaneStateSnapshot Program::inspect_lane_for_test(const Program& program, std::uint32_t lane) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("lane is out of range"); }
    const LaneState& state = program.impl_->lanes[lane];
    return LaneStateSnapshot{
        .lane = lane,
        .occupied = state.lifecycle != Lifecycle::Free,
        .retained = state.lifecycle == Lifecycle::Retained,
        .prompt_tokens = state.prompt_tokens,
        .generated_tokens = state.generated_tokens,
        .frontier = state.frontier,
        .page_entitlement = state.page_entitlement,
    };
}

StateImageInspection Program::inspect_state_images_for_test(const Program& program) {
    const Impl& impl = *program.impl_;
    auto* base       = impl.state_images.p;
    return StateImageInspection{
        .base          = base,
        .bytes         = impl.state_image_bytes,
        .kv_bytes      = impl.state_kv_bytes,
        .gdn_bytes     = impl.state_gdn_bytes,
        .mhc_bytes     = impl.state_mhc_bytes,
        .mtp_kv_region = static_cast<std::uint8_t*>(base) +
                         StateImageLayout::kv_offset(StateLayout::full_attention_state_slots,
                                                     impl.kv_capacity, false, false),
        .gdn_region    = static_cast<std::uint8_t*>(base) + impl.state_kv_bytes,
        .mhc_region    = static_cast<std::uint8_t*>(base) + impl.state_kv_bytes +
                         impl.state_gdn_bytes,
    };
}

void Program::write_state_image_markers_for_test(Program& program, std::uint8_t value) {
    program.impl_->state_images.fill(value);
}

bool Program::verify_state_image_markers_for_test(const Program& program, std::uint8_t value) {
    std::vector<std::uint8_t> actual(program.impl_->state_image_bytes);
    program.impl_->state_images.copy_to_host(actual.data(), actual.size());
    for (const std::uint8_t byte : actual) {
        if (byte != value) { return false; }
    }
    return true;
}

Program::Program(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() noexcept                         = default;

RequestBasePlan Program::plan_request_base(const qwen3_6::PreparedPrompt& prompt,
                                           const runtime::ResolvedExecutionOptions& options) {
    const PromptSummary summary = prompt.summary();
    auto impl                   = std::make_unique<RequestBasePlan::Impl>();
    impl->summary.prompt_tokens = summary.prompt_tokens;
    impl->summary.requested_output_tokens = options.requested_output_tokens;
    impl->summary.effective_output_tokens = options.requested_output_tokens;
    impl->summary.transient_bytes         = 4096;
    impl->summary.transient_alignment     = 256;
    impl->summary.admission               = {1, 1, 0};
    const std::uint32_t chunks = summary.prompt_tokens == 0
                                     ? 1
                                     : (summary.prompt_tokens + impl_->memory.max_context - 1) /
                                           std::max(1U, impl_->memory.max_context);
    impl->summary.service_work_quanta = chunks + 1;
    return RequestBasePlan(std::move(impl));
}

RequestPlan Program::plan_request_for_lane(std::uint32_t lane, const qwen3_6::PreparedPrompt&,
                                           const RequestBasePlan& base) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    auto impl      = std::make_unique<RequestPlan::Impl>();
    impl->summary  = base.summary();
    impl->summary.admission = impl_->admission;
    return RequestPlan(std::move(impl));
}

bool Program::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    return lane < impl_->max_concurrency && plan.impl_ != nullptr &&
           (impl_->lanes[lane].lifecycle == Lifecycle::Free ||
            impl_->lanes[lane].lifecycle == Lifecycle::Retained);
}

bool Program::can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                     const RequestPlan& plan) const noexcept {
    return can_admit_lane(lane, plan);
}

runtime::AdmissionResources Program::admission_capacity() const noexcept { return impl_->admission; }

runtime::PrefillStepResult Program::start_prefill_lane(std::uint32_t lane,
                                                       qwen3_6::PreparedPrompt&& prompt,
                                                       RequestPlan&& plan,
                                                       runtime::TransientRegion) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (!can_admit_lane(lane, plan)) { throw std::logic_error("lane is not admissible"); }
    const PromptSummary info = prompt.summary();
    LaneState& state          = impl_->lanes[lane];
    state                     = LaneState{};
    state.lifecycle           = Lifecycle::Pending;
    state.prompt_tokens       = info.prompt_tokens;
    state.page_entitlement    = (info.prompt_tokens + kSenseNovaPageTokens - 1) /
                                kSenseNovaPageTokens;
    state.frontier            = info.prompt_tokens;

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens      = info.prompt_tokens;
    result.processed_prompt_tokens    = info.prompt_tokens;
    result.host_input_consumed        = true;
    result.complete                   = true;
    static constexpr TokenId kFirstToken = 0;
    result.round.tokens                = std::span<const TokenId>(&kFirstToken, 1);
    return result;
}

runtime::PrefillStepResult Program::advance_prefill_lane(std::uint32_t) {
    throw std::logic_error("sensenova_loop_9b: prefill completes in one staged step");
}

runtime::BatchedGeneratedRound Program::decode_batch(std::span<const std::uint32_t> lanes,
                                                     std::span<const runtime::RoundBudget> budgets) {
    if (lanes.empty() || lanes.size() > impl_->max_concurrency ||
        budgets.size() != lanes.size()) {
        throw std::invalid_argument("sensenova_loop_9b: invalid decode batch");
    }
    impl_->egress_tokens.clear();
    impl_->egress_logprobs.clear();
    impl_->egress_row_counts.clear();
    impl_->egress_span = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= impl_->max_concurrency || impl_->lanes[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("decode batch lane is not pending");
        }
        static constexpr TokenId kNextToken = 0;
        impl_->egress_tokens.push_back(kNextToken);
        impl_->egress_logprobs.push_back(0.0F);
        impl_->egress_row_counts.push_back(1);
        impl_->pending_tokens[lane]   = kNextToken;
        impl_->pending_logprobs[lane] = 0.0F;
    }
    return runtime::BatchedGeneratedRound{
        .tokens      = impl_->egress_tokens,
        .row_counts  = impl_->egress_row_counts,
        .row_stride  = 1,
        .logprobs    = impl_->egress_logprobs,
        .margins     = {},
    };
}

void Program::set_token_mask_lane(std::uint32_t lane, std::span<const std::uint32_t> mask) {
    if (lane >= impl_->max_concurrency) { return; }
    std::copy_n(mask.begin(), std::min<std::size_t>(mask.size(), impl_->lanes[lane].token_mask.size()),
                impl_->lanes[lane].token_mask.begin());
}

void Program::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    LaneState& state = impl_->lanes[lane];
    if (state.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("resolve_prefill_lane requires a pending prefill lane");
    }
    state.lifecycle = terminal ? Lifecycle::Retained : Lifecycle::Active;
}

void Program::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                    std::span<const std::uint32_t> accepted_tokens,
                                    std::span<const std::uint8_t> terminal,
                                    std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > impl_->max_concurrency ||
        accepted_tokens.size() != lanes.size() || terminal.size() != lanes.size() ||
        cancelled.size() != lanes.size()) {
        throw std::invalid_argument("sensenova_loop_9b: invalid pending resolution");
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
        LaneState& state = impl_->lanes[lane];
        if (state.lifecycle != Lifecycle::Pending) {
            throw std::logic_error("pending resolution lane is not pending");
        }
        state.lifecycle = cancelled[row] ? Lifecycle::Free
                          : terminal[row] ? Lifecycle::Retained
                                          : Lifecycle::Active;
        state.generated_tokens += accepted_tokens[row];
        state.frontier += accepted_tokens[row];
    }
}

void Program::abort_lane(std::uint32_t lane) noexcept {
    if (lane < impl_->max_concurrency) { impl_->lanes[lane] = LaneState{}; }
}

bool Program::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->max_concurrency && impl_->lanes[lane].lifecycle == Lifecycle::Retained;
}

std::uint32_t Program::retained_lane_depth(std::uint32_t lane) const noexcept {
    return has_retained_lane(lane) ? impl_->lanes[lane].frontier : 0;
}

void Program::evict_retained_lane(std::uint32_t lane) noexcept {
    if (has_retained_lane(lane)) { abort_lane(lane); }
}

GenerationTimings Program::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->max_concurrency ? GenerationTimings{} : GenerationTimings{};
}

SpeculativeStats Program::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->max_concurrency ? SpeculativeStats{} : SpeculativeStats{};
}

void Program::snapshot_lane_to_disk(std::uint32_t, DiskStateCache&) {}
void Program::snapshot_turn_checkpoint_to_disk(std::uint32_t, DiskStateCache&) {}
void Program::set_disk_state_cache(DiskStateCache*) noexcept {}
std::string Program::config_signature_slug() const {
    return "sensenova_loop_9b_h4096_56visits_ctx" + std::to_string(impl_->memory.max_context);
}
MemorySummary Program::memory_summary() const noexcept { return impl_->memory; }
void Program::reset_memory_peaks() noexcept {}

} // namespace ninfer::targets::sensenova_loop_9b::detail
