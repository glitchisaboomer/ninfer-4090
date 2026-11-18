#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"
#include "targets/sensenova_loop_9b/impl/runtime/layer_execution.h"
#include "targets/sensenova_loop_9b/impl/runtime/layouts.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"
#include "targets/sensenova_loop_9b/impl/runtime/text_execution.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <array>
#include <string>
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
    const RuntimeModelView* model = nullptr;
    cudaStream_t stream           = nullptr;
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
    std::vector<float> egress_margins;
    std::vector<std::int32_t> egress_row_counts;
    std::size_t egress_span = 1;

    // Execution assets prepared once from the stored artifact forms: the identity KV page map and
    // row selector, the op-form GDN convolution and mHC router weights, and the Program-owned
    // transient region eager decode runs in (the engine supplies a per-request region only for
    // prefill, and a decode step has no request-scoped allocation of its own).
    DeviceBuffer kv_aux;
    Tensor kv_block_table;  // I32 [kv_pages]
    Tensor kv_table_rows;   // I32 [1]
    DeviceBuffer visit_assets;
    std::array<Tensor, TextConfig::layers> gdn_conv{};
    std::array<Tensor, 2 * TextConfig::layers> mhc_router{};
    DeviceBuffer decode_workspace;
    std::size_t decode_workspace_bytes = 0;

    // Per-lane host-side execution results. Spans handed to the engine point at these arrays, which
    // are stable for the Program lifetime.
    struct LaneExecution {
        TokenId last_token = 0;
        std::array<TokenId, 1> prefill_tokens{};
        std::array<float, 1> prefill_logprobs{};
        std::array<float, 1> prefill_margins{};
        std::array<TokenId, 1> pending_tokens{};
        std::array<float, 1> pending_logprobs{};
        std::array<float, 1> pending_margins{};
        GenerationTimings timings{};
    };
    std::array<LaneExecution, kMaximumConcurrency> execution;
};

std::unique_ptr<Program> Program::create(std::uint32_t active_lanes, std::uint32_t kv_pages,
                                         std::uint32_t context, std::size_t reservation_bytes) {
    return create_impl(nullptr, nullptr, active_lanes, kv_pages, context, reservation_bytes);
}

std::unique_ptr<Program> Program::create(const RuntimeModelView& model, cudaStream_t stream,
                                         std::uint32_t active_lanes, std::uint32_t kv_pages,
                                         std::uint32_t context, std::size_t reservation_bytes) {
    return create_impl(&model, stream, active_lanes, kv_pages, context, reservation_bytes);
}

std::unique_ptr<Program> Program::create_impl(const RuntimeModelView* model, cudaStream_t stream,
                                              std::uint32_t active_lanes, std::uint32_t kv_pages,
                                              std::uint32_t context,
                                              std::size_t reservation_bytes) {
    if (active_lanes == 0 || active_lanes > kMaximumConcurrency) {
        throw std::invalid_argument("sensenova_loop_9b: invalid active lane count");
    }
    auto impl                            = std::make_unique<Impl>();
    impl->model  = model;
    impl->stream = stream;
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

    if (model != nullptr) {
        // Identity logical-to-physical page map for every KV domain, followed by the B=1 row
        // selector. Both are constant for the Program lifetime: every domain is one contiguous
        // sequence, so the same table describes all fifteen.
        const std::size_t pages = impl->kv_capacity / StateImageLayout::kKvPageTokens;
        impl->kv_aux = DeviceBuffer((pages + 1) * sizeof(std::int32_t));
        std::vector<std::int32_t> kv_host(pages + 1);
        for (std::size_t page = 0; page < pages; ++page) {
            kv_host[page] = static_cast<std::int32_t>(page);
        }
        kv_host[pages] = 0;
        impl->kv_aux.copy_from_host(kv_host.data(), kv_host.size() * sizeof(std::int32_t));
        impl->kv_block_table =
            Tensor(impl->kv_aux.p, DType::I32, {static_cast<std::int32_t>(pages)});
        impl->kv_table_rows =
            Tensor(static_cast<std::int32_t*>(impl->kv_aux.p) + pages, DType::I32, {1});

        // The op forms differ from the stored forms, so they are prepared once here rather than
        // transposed on every visit. The transposes are stream-ordered and drained before the
        // first forward can read them.
        const std::size_t conv_bytes = static_cast<std::size_t>(TextConfig::convolution_dim) *
                                       TextConfig::gdn_conv_kernel * sizeof(std::uint16_t);
        const std::size_t router_bytes = static_cast<std::size_t>(MhcConfig::input_width) *
                                         MhcConfig::logits * sizeof(float);
        impl->visit_assets = DeviceBuffer(conv_bytes * TextConfig::layers +
                                          router_bytes * 2 * TextConfig::layers);
        auto* assets_base = static_cast<std::uint8_t*>(impl->visit_assets.p);
        std::size_t asset_offset = 0;
        for (int layer = 0; layer < TextConfig::layers; ++layer) {
            Tensor& conv = impl->gdn_conv[layer];
            conv = Tensor(assets_base + asset_offset, DType::BF16,
                          {TextConfig::convolution_dim, TextConfig::gdn_conv_kernel});
            asset_offset += conv_bytes;
            if (!TextConfig::is_full_attention(layer)) {
                transpose_gdn_convolution_weight(model->layers[layer].gdn.convolution, conv,
                                                 impl->stream);
            }
        }
        for (int layer = 0; layer < TextConfig::layers; ++layer) {
            for (int branch = 0; branch < 2; ++branch) {
                Tensor& router = impl->mhc_router[2 * layer + branch];
                router = Tensor(assets_base + asset_offset, DType::FP32,
                                {MhcConfig::input_width, MhcConfig::logits});
                asset_offset += router_bytes;
                if (TextConfig::is_mhc_layer(layer)) {
                    transpose_mhc_router_weight(
                        branch == 0 ? model->layers[layer].mhc_attn.weight
                                    : model->layers[layer].mhc_ffn.weight,
                        router, impl->stream);
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(impl->stream));

        // Decode runs one token at a time from a Program-owned region sized for the widest causal
        // history the declared capacity allows.
        impl->decode_workspace_bytes =
            TextExecutionContext::forward_workspace_capacity_bytes(1, impl->kv_capacity);
        impl->decode_workspace = DeviceBuffer(impl->decode_workspace_bytes);
    }
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
    // Eager prefill runs the whole prompt in one forward, so the request needs the embedding
    // hidden state, the mHC stream buffers, the worst visit scratch, the head columns, and the
    // device copies of its ids/positions.
    const std::uint32_t plan_tokens =
        std::max<std::uint32_t>(1, summary.prompt_tokens);
    const std::size_t forward_bytes =
        TextExecutionContext::forward_workspace_capacity_bytes(plan_tokens, plan_tokens);
    const std::size_t staging_bytes =
        ((2ULL * plan_tokens * sizeof(std::int32_t) + 255ULL) / 256ULL) * 256ULL;
    impl->summary.transient_bytes     = forward_bytes + staging_bytes;
    impl->summary.transient_alignment = 256;
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
                                                       runtime::TransientRegion transient) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (!can_admit_lane(lane, plan)) { throw std::logic_error("lane is not admissible"); }
    if (impl_->model == nullptr) {
        throw std::logic_error("sensenova_loop_9b: text prefill requires a loaded model; "
                               "the test Program factory has none");
    }

    qwen3_6::PreparedPromptData data = qwen3_6::PreparedPromptAccess::take(std::move(prompt));
    if (data.token_ids.empty()) {
        throw std::invalid_argument("sensenova_loop_9b: prompt has no tokens");
    }
    if (data.token_ids.size() != data.positions.size()) {
        throw std::invalid_argument("sensenova_loop_9b: prompt positions do not match its tokens");
    }
    if (data.token_ids.size() > impl_->kv_capacity) {
        throw std::invalid_argument("sensenova_loop_9b: prompt exceeds the declared KV capacity");
    }
    if (data.has_media()) {
        throw std::logic_error("sensenova_loop_9b: vision prompts are not supported");
    }
    if (!transient) {
        throw std::invalid_argument("sensenova_loop_9b: text prefill requires a transient region");
    }
    const auto tokens = static_cast<std::uint32_t>(data.token_ids.size());

    WorkspaceArena workspace(DeviceSpan{transient.data, transient.size});
    const DeviceArena::Scope scope = workspace.scope();
    const auto started             = std::chrono::steady_clock::now();

    TokenId host_token  = 0;
    float host_logprob  = 0.0F;
    float host_margin   = 0.0F;
    try {
        const auto token_extent = static_cast<std::int32_t>(tokens);
        Tensor ids        = workspace.alloc(DType::I32, {token_extent});
        Tensor positions  = workspace.alloc(DType::I32, {token_extent});
        Tensor sampled    = workspace.alloc(DType::I32, {1});
        Tensor logprob    = workspace.alloc(DType::FP32, {1});
        Tensor margin     = workspace.alloc(DType::FP32, {1});
        CUDA_CHECK(cudaMemcpyAsync(ids.data, data.token_ids.data(),
                                   tokens * sizeof(TokenId), cudaMemcpyHostToDevice,
                                   impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(positions.data, data.positions.data(),
                                   tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   impl_->stream));

        const TextExecutionBindings bindings{impl_->kv_block_table, impl_->kv_table_rows,
                                             impl_->gdn_conv, impl_->mhc_router};
        TextExecutionContext context(*impl_->model, impl_->stream,
                                     static_cast<std::byte*>(impl_->state_images.p),
                                     impl_->kv_capacity, workspace, bindings);
        context.forward_full(ids, positions, tokens, sampled, &logprob, &margin);

        CUDA_CHECK(cudaMemcpyAsync(&host_token, sampled.data, sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_logprob, logprob.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_margin, margin.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "sensenova_loop_9b: request transient region is too small for text prefill");
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                               .count();

    LaneState& state       = impl_->lanes[lane];
    state                  = LaneState{};
    state.lifecycle        = Lifecycle::Pending;
    state.prompt_tokens    = tokens;
    state.page_entitlement = (tokens + kSenseNovaPageTokens - 1) / kSenseNovaPageTokens;
    state.frontier         = tokens;

    Impl::LaneExecution& execution = impl_->execution[lane];
    execution                      = Impl::LaneExecution{};
    execution.last_token          = host_token;
    execution.prefill_tokens[0]   = host_token;
    execution.prefill_logprobs[0] = host_logprob;
    execution.prefill_margins[0]  = host_margin;
    execution.timings.prefill_seconds     = seconds;
    execution.timings.first_token_seconds = seconds;
    execution.timings.total_seconds       = seconds;

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens   = tokens;
    result.processed_prompt_tokens = tokens;
    result.host_input_consumed     = true;
    result.complete                = true;
    result.round.tokens            = std::span<const TokenId>(execution.prefill_tokens);
    result.round.logprobs          = std::span<const float>(execution.prefill_logprobs);
    result.round.margins           = std::span<const float>(execution.prefill_margins);
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
