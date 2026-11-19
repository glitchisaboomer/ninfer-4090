#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"
#include "targets/sensenova_loop_9b/impl/runtime/graph_impl.h"
#include "targets/sensenova_loop_9b/impl/runtime/layer_execution.h"
#include "targets/sensenova_loop_9b/impl/runtime/mtp.h"
#include "targets/sensenova_loop_9b/impl/runtime/mtp_stem.h"
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
#include <span>
#include <unordered_map>
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
    bool use_cuda_graph                     = false;
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
bool SequencePlan::use_cuda_graph() const noexcept { return impl_->use_cuda_graph; }

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
    impl->use_cuda_graph           = layout.inputs.use_cuda_graph;
    return SequencePlan(std::move(impl));
}

enum class Lifecycle : std::uint8_t { Free, Prefilling, Active, Pending, Retained };

struct LaneState {
    Lifecycle lifecycle = Lifecycle::Free;
    std::uint32_t prompt_tokens = 0;
    std::vector<TokenId> prompt_token_ids;
    std::uint32_t generated_tokens = 0;
    std::uint32_t frontier = 0;
    std::uint32_t page_entitlement = 0;
    std::vector<TokenId> licensed_tokens;
    ops::SamplingConfig sampling{};
    std::array<std::uint32_t, kMaximumConcurrency> token_mask{};
};

// Stage F1/F2: one lane's MTP proposal and verification window. Drafts are produced by the
// device stem (or injected by a deterministic unit test); the target window is sampled by the
// target LM head. The pending state is cleared at every commit so a rejected round can never
// leak into the next proposal.
struct LaneMtpState {
    MtpRoundState round;
    MtpProposal pending;
    std::vector<TokenId> target_window;
    bool armed = false;
};

struct Program::Impl {
    const RuntimeModelView* model = nullptr;
    cudaStream_t stream           = nullptr;
    // Stage F3: one whole state image per lane. Every lane keeps a stable private partition,
    // so concurrent requests cannot clobber one another's KV/GDN/mHC/MTP state. A prefix snapshot
    // is copied into the target lane's partition on restore (copy-on-write at the image level).
    DeviceBuffer state_images;
    std::size_t state_image_bytes = 0;
    std::size_t state_image_stride = 0;
    std::size_t state_kv_bytes    = 0;
    std::size_t state_gdn_bytes   = 0;
    std::size_t state_mhc_bytes   = 0;
    MemorySummary memory;
    runtime::AdmissionResources admission;
    std::uint32_t max_concurrency = 1;
    std::uint32_t kv_capacity = 0;
    std::array<LaneState, kMaximumConcurrency> lanes;
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

    // Stage E6: fixed-address per-lane decode staging. The captured step reads the token id and
    // position from here, and the head's sampled token, logprob, and margin are read back after
    // replay. One 256-byte slot per lane keeps every address stable for the Program lifetime.
    bool use_cuda_graph = false;
    static constexpr std::size_t kLaneIoBytes = 256;
    DeviceBuffer decode_io;
    struct LaneIo {
        Tensor ids;
        Tensor positions;
        Tensor sampled;
        Tensor logprob;
        Tensor margin;
    };
    std::array<LaneIo, kMaximumConcurrency> lane_io{};
    std::array<DecodeGraphRunner, kMaximumConcurrency> decode_graphs;

    // Device-resident uniform 0.5 mix for the mHC leave reduction at T=1. Preparing it once
    // keeps a capturable decode pass free of the pageable host fill the staged merge needs.
    DeviceBuffer mhc_half_mix;
    Tensor mhc_half_mix_tensor;

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

    // Stage F1/F2: MTP window and real per-lane speculative counters.
    std::uint32_t draft_window = 0;
    std::array<LaneMtpState, kMaximumConcurrency> mtp;
    std::array<SpeculativeStats, kMaximumConcurrency> speculative_stats{};
    std::array<std::uint32_t, kMaximumConcurrency> speculative_licensed_counts{};
    struct SpeculativeEgress {
        std::array<TokenId, kMaximumMtpDraftTokens + 1> tokens{};
        std::array<float, kMaximumMtpDraftTokens + 1> logprobs{};
        std::array<float, kMaximumMtpDraftTokens + 1> margins{};
    };
    std::array<SpeculativeEgress, kMaximumConcurrency> speculative_egress{};

    // Stage F3: engine-level prefix identity. The key is the canonical token sequence hash of
    // a committed prompt; the value is a full state-image snapshot with its frontier and last
    // committed token. Prefix snapshots are independent of lane state and can be restored into
    // any admissible lane partition.
    struct EnginePrefix {
        PrefixStateSnapshot image;
    };
    std::unordered_map<std::uint64_t, EnginePrefix> engine_prefixes;
};

std::byte* Program::lane_state_base(Program::Impl& impl, std::uint32_t lane) {
    return static_cast<std::byte*>(impl.state_images.p) +
           static_cast<std::size_t>(lane) * impl.state_image_stride;
}

const std::byte* Program::lane_state_base(const Program::Impl& impl, std::uint32_t lane) {
    return static_cast<const std::byte*>(impl.state_images.p) +
           static_cast<std::size_t>(lane) * impl.state_image_stride;
}

std::unique_ptr<Program> Program::create(std::uint32_t active_lanes, std::uint32_t kv_pages,
                                         std::uint32_t context, std::size_t reservation_bytes) {
    return create_impl(nullptr, nullptr, active_lanes, kv_pages, context, reservation_bytes, false);
}

std::unique_ptr<Program> Program::create(const RuntimeModelView& model, cudaStream_t stream,
                                         std::uint32_t active_lanes, std::uint32_t kv_pages,
                                         std::uint32_t context, std::size_t reservation_bytes,
                                         bool use_cuda_graph) {
    return create_impl(&model, stream, active_lanes, kv_pages, context, reservation_bytes,
                       use_cuda_graph);
}

std::unique_ptr<Program> Program::create_impl(const RuntimeModelView* model, cudaStream_t stream,
                                              std::uint32_t active_lanes, std::uint32_t kv_pages,
                                              std::uint32_t context,
                                              std::size_t reservation_bytes,
                                              bool use_cuda_graph) {
    if (active_lanes == 0 || active_lanes > kMaximumConcurrency) {
        throw std::invalid_argument("sensenova_loop_9b: invalid active lane count");
    }
    auto impl                            = std::make_unique<Impl>();
    impl->model  = model;
    impl->stream = stream;
    impl->use_cuda_graph = use_cuda_graph;
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
    // One aligned whole-state partition per lane keeps every visit address stable and collision
    // free. The allocation itself is one DeviceBuffer, so no per-lane pointer changes at runtime.
    impl->state_image_stride = ((impl->state_image_bytes + 255ULL) / 256ULL) * 256ULL;
    impl->state_images       = DeviceBuffer(impl->state_image_stride * active_lanes);
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

        // Fixed-address staging for the captured decode step. The tensors are views into one
        // allocation, so no host pointer ever changes across capture, replay, or the eager path.
        impl->decode_io = DeviceBuffer(Impl::kLaneIoBytes * active_lanes);
        auto* io_base   = static_cast<std::uint8_t*>(impl->decode_io.p);
        for (std::uint32_t lane = 0; lane < active_lanes; ++lane) {
            auto* slot = io_base + static_cast<std::size_t>(lane) * Impl::kLaneIoBytes;
            Impl::LaneIo& io = impl->lane_io[lane];
            io.ids       = Tensor(slot, DType::I32, {1});
            io.positions = Tensor(slot + 4, DType::I32, {1});
            io.sampled   = Tensor(slot + 8, DType::I32, {1});
            io.logprob   = Tensor(slot + 12, DType::FP32, {1});
            io.margin    = Tensor(slot + 16, DType::FP32, {1});
        }

        impl->mhc_half_mix =
            DeviceBuffer(static_cast<std::size_t>(MhcConfig::streams) * MhcConfig::streams *
                         sizeof(float));
        const float half_mix[4] = {0.5F, 0.5F, 0.5F, 0.5F};
        impl->mhc_half_mix.copy_from_host(half_mix, sizeof(half_mix));
        impl->mhc_half_mix_tensor = Tensor(impl->mhc_half_mix.p, DType::FP32,
                                          {MhcConfig::streams, MhcConfig::streams, 1});
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

DecodeGraphInspection Program::inspect_decode_graph_for_test(const Program& program,
                                                             std::uint32_t lane) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("lane is out of range"); }
    DecodeGraphInspection inspection;
    inspection.configured = program.impl_->use_cuda_graph;
    inspection.attempted  = program.impl_->decode_graphs[lane].attempted();
    inspection.ready      = program.impl_->decode_graphs[lane].ready();
    inspection.diagnostic = program.impl_->decode_graphs[lane].diagnostic();
    return inspection;
}

void Program::seed_lane_for_test(Program& program, std::uint32_t lane, std::uint32_t frontier,
                                 std::uint32_t last_token) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("lane is out of range"); }
    LaneState& state = program.impl_->lanes[lane];
    state = LaneState{};
    state.lifecycle   = Lifecycle::Active;
    state.prompt_tokens = frontier;
    state.frontier    = frontier;
    program.impl_->execution[lane] = Impl::LaneExecution{};
    program.impl_->execution[lane].last_token = last_token;
}

void Program::seed_pending_lane_for_test(Program& program, std::uint32_t lane,
                                         std::uint32_t frontier, std::uint32_t last_token) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("lane is out of range"); }
    LaneState& state = program.impl_->lanes[lane];
    state = LaneState{};
    state.lifecycle = Lifecycle::Pending;
    state.prompt_tokens = frontier;
    state.frontier = frontier;
    program.impl_->execution[lane] = Impl::LaneExecution{};
    program.impl_->execution[lane].last_token = last_token;
}

void Program::set_committed_prompt_for_test(Program& program, std::uint32_t lane,
                                            std::span<const TokenId> tokens) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    LaneState& state = program.impl_->lanes[lane];
    state.prompt_tokens = static_cast<std::uint32_t>(tokens.size());
    state.prompt_token_ids.assign(tokens.begin(), tokens.end());
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
    if (data.positions.size() != data.token_ids.size() * 3) {
        throw std::invalid_argument("sensenova_loop_9b: prompt positions must contain three axes per token");
    }
    // SenseNova reads text RoPE from the temporal/text axis. The frontend stores axes
    // axis-major as [3,T], so axis 0 is the first T int32 values, not the whole vector.
    const std::span<const std::int32_t> text_positions = data.position_axis(0);
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
        CUDA_CHECK(cudaMemcpyAsync(positions.data, text_positions.data(),
                                   tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   impl_->stream));

        const TextExecutionBindings bindings{impl_->kv_block_table, impl_->kv_table_rows,
                                             impl_->gdn_conv, impl_->mhc_router,
                                             impl_->mhc_half_mix_tensor};
        TextExecutionContext context(*impl_->model, impl_->stream, lane_state_base(*impl_, lane),
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
    state.prompt_token_ids.assign(data.token_ids.begin(), data.token_ids.end());
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
    // The sampled id is published verbatim, including a generation-config EOS id; stopping is the
    // output session's decision (TextConfig::end_of_text_token / end_of_message_token).
    result.round.tokens            = std::span<const TokenId>(execution.prefill_tokens);
    result.round.logprobs          = std::span<const float>(execution.prefill_logprobs);
    result.round.margins           = std::span<const float>(execution.prefill_margins);
    return result;
}

runtime::PrefillStepResult
Program::start_prefill_lane_with_prefix(std::uint32_t lane, PrefixIdentity identity,
                                        qwen3_6::PreparedPrompt&& prompt, RequestPlan&& plan,
                                        runtime::TransientRegion transient) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (!plan_request_for_prefix(lane, identity, 0, plan)) {
        throw std::invalid_argument("sensenova_loop_9b: prefix identity is not reusable");
    }
    const Impl::EnginePrefix& prefix = impl_->engine_prefixes.at(identity.value);
    const std::uint32_t reused = prefix.image.token_count();

    qwen3_6::PreparedPromptData data = qwen3_6::PreparedPromptAccess::take(std::move(prompt));
    if (data.token_ids.empty() || reused >= data.token_ids.size()) {
        throw std::invalid_argument("sensenova_loop_9b: shared prefix must be shorter than the prompt");
    }
    if (data.positions.size() != data.token_ids.size() * 3) {
        throw std::invalid_argument("sensenova_loop_9b: prompt positions must contain three axes per token");
    }
    // SenseNova reads text RoPE from the temporal/text axis. The frontend stores axes
    // axis-major as [3,T], so axis 0 is the first T int32 values, not the whole vector.
    const std::span<const std::int32_t> text_positions = data.position_axis(0);
    if (data.token_ids.size() > impl_->kv_capacity) {
        throw std::invalid_argument("sensenova_loop_9b: prompt exceeds the declared KV capacity");
    }

    // Restore the full state image into this lane's private partition before running the suffix.
    // The lane is occupied for the duration of the restore, which is the lifecycle boundary the
    // E6 primitive enforces.
    LaneState& state = impl_->lanes[lane];
    state.lifecycle = Lifecycle::Prefilling;
    restore_prefix_state(lane, prefix.image);
    state.prompt_token_ids.assign(data.token_ids.begin(), data.token_ids.end());
    state.lifecycle = Lifecycle::Free;

    const std::uint32_t suffix_tokens = static_cast<std::uint32_t>(data.token_ids.size() - reused);
    WorkspaceArena workspace(DeviceSpan{transient.data, transient.size});
    const DeviceArena::Scope scope = workspace.scope();
    TokenId host_token = 0;
    float host_logprob = 0.0F;
    float host_margin  = 0.0F;
    try {
        Tensor ids = workspace.alloc(DType::I32, {static_cast<std::int32_t>(suffix_tokens)});
        Tensor positions = workspace.alloc(DType::I32, {static_cast<std::int32_t>(suffix_tokens)});
        Tensor sampled = workspace.alloc(DType::I32, {1});
        Tensor logprob = workspace.alloc(DType::FP32, {1});
        Tensor margin = workspace.alloc(DType::FP32, {1});
        CUDA_CHECK(cudaMemcpyAsync(ids.data, data.token_ids.data() + reused,
                                   suffix_tokens * sizeof(TokenId), cudaMemcpyHostToDevice,
                                   impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(positions.data, text_positions.data() + reused,
                                   suffix_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   impl_->stream));
        const TextExecutionBindings bindings{impl_->kv_block_table, impl_->kv_table_rows,
                                             impl_->gdn_conv, impl_->mhc_router,
                                             impl_->mhc_half_mix_tensor};
        TextExecutionContext context(*impl_->model, impl_->stream, lane_state_base(*impl_, lane),
                                     impl_->kv_capacity, workspace, bindings);
        context.forward_full(ids, positions, static_cast<std::uint32_t>(data.token_ids.size()),
                             sampled, &logprob, &margin);
        CUDA_CHECK(cudaMemcpyAsync(&host_token, sampled.data, sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_logprob, logprob.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_margin, margin.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "sensenova_loop_9b: request transient region is too small for prefix resume");
    }

    state = LaneState{};
    state.lifecycle = Lifecycle::Pending;
    state.prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
    state.prompt_token_ids = std::move(data.token_ids);
    state.frontier = state.prompt_tokens;
    state.page_entitlement =
        (state.frontier + kSenseNovaPageTokens - 1) / kSenseNovaPageTokens;
    Impl::LaneExecution& execution = impl_->execution[lane];
    execution = Impl::LaneExecution{};
    execution.last_token = host_token;
    execution.prefill_tokens[0] = host_token;
    execution.prefill_logprobs[0] = host_logprob;
    execution.prefill_margins[0] = host_margin;

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens = state.prompt_tokens;
    result.summary.reused_prompt_tokens = reused;
    result.summary.relocated_prompt_tokens = 0;
    result.summary.prefix_reuse_path = ninfer::PrefixReusePath::AppendAtSharedPrefix;
    result.processed_prompt_tokens = suffix_tokens;
    result.host_input_consumed = true;
    result.complete = true;
    result.round.tokens = std::span<const TokenId>(execution.prefill_tokens);
    result.round.logprobs = std::span<const float>(execution.prefill_logprobs);
    result.round.margins = std::span<const float>(execution.prefill_margins);
    return result;
}

runtime::PrefillStepResult Program::advance_prefill_lane(std::uint32_t) {
    throw std::logic_error("sensenova_loop_9b: prefill completes in one staged step");
}

void Program::decode_one(std::uint32_t lane) {
    LaneState& state = impl_->lanes[lane];
    if (state.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("decode requires a pending lane");
    }
    if (impl_->model == nullptr) {
        throw std::logic_error("sensenova_loop_9b: text decode requires a loaded model");
    }
    if (state.frontier >= impl_->kv_capacity) {
        throw std::logic_error("sensenova_loop_9b: decode frontier exceeds the KV capacity");
    }

    Impl::LaneIo& io = impl_->lane_io[lane];
    if (io.ids.data == nullptr) {
        throw std::logic_error("sensenova_loop_9b: decode staging is not prepared");
    }

    const auto started          = std::chrono::steady_clock::now();
    const TokenId input         = impl_->execution[lane].last_token;
    const std::int32_t position = static_cast<std::int32_t>(state.frontier);
    CUDA_CHECK(cudaMemcpyAsync(io.ids.data, &input, sizeof(TokenId), cudaMemcpyHostToDevice,
                               impl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(io.positions.data, &position, sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->stream));

    const TextExecutionBindings bindings{impl_->kv_block_table, impl_->kv_table_rows,
                                         impl_->gdn_conv, impl_->mhc_router,
                                         impl_->mhc_half_mix_tensor};
    // The captured body must be valid at every later frontier, so it runs with the envelope of the
    // declared KV capacity instead of this step's frontier. The causal mask itself is still read
    // from the uploaded position, and the decode workspace is sized for exactly this envelope.
    auto run_forward = [&] {
        WorkspaceArena workspace(DeviceSpan{impl_->decode_workspace.p,
                                            impl_->decode_workspace_bytes});
        TextExecutionContext context(*impl_->model, impl_->stream, lane_state_base(*impl_, lane),
                                     impl_->kv_capacity, workspace, bindings);
        context.forward_full(io.ids, io.positions, impl_->kv_capacity, io.sampled, &io.logprob,
                             &io.margin);
    };

    TokenId host_token = 0;
    float host_logprob = 0.0F;
    float host_margin  = 0.0F;
    try {
        bool launched = false;
        if (impl_->use_cuda_graph) {
            DecodeGraphRunner& graph = impl_->decode_graphs[lane];
            if (!graph.attempted()) { (void)graph.try_capture(impl_->stream, run_forward); }
            if (graph.ready()) {
                graph.launch(impl_->stream);
                launched = true;
            }
        }
        if (!launched) { run_forward(); }

        // Host frontier publication stays outside the captured region: only the kernels replay.
        CUDA_CHECK(cudaMemcpyAsync(&host_token, io.sampled.data, sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_logprob, io.logprob.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_margin, io.margin.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, impl_->stream));
        CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    } catch (const std::bad_alloc&) {
        throw std::invalid_argument(
            "sensenova_loop_9b: decode workspace is too small for one token");
    }

    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    Impl::LaneExecution& execution = impl_->execution[lane];
    execution.pending_tokens[0]    = host_token;
    execution.pending_logprobs[0]  = host_logprob;
    execution.pending_margins[0]   = host_margin;
    execution.timings.decode_seconds += seconds;
    execution.timings.total_seconds =
        execution.timings.prefill_seconds + execution.timings.decode_seconds;
}

void Program::configure_mtp_for_test(Program& program, std::uint32_t draft_window) {
    if (!valid_draft_window(draft_window)) {
        throw std::invalid_argument("sensenova_loop_9b: invalid MTP draft window");
    }
    program.impl_->draft_window = draft_window;
}

void Program::set_draft_window_for_test(Program& program, std::uint32_t draft_window) {
    configure_mtp_for_test(program, draft_window);
}

void Program::set_proposal_for_test(Program& program, std::uint32_t lane,
                                    std::span<const TokenId> drafts) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    LaneMtpState& mtp = program.impl_->mtp[lane];
    mtp.pending.valid = static_cast<std::uint32_t>(std::min<std::size_t>(
        drafts.size(), program.impl_->draft_window));
    mtp.pending.drafted.assign(drafts.begin(), drafts.begin() + mtp.pending.valid);
    mtp.armed = mtp.pending.valid != 0;
}

void Program::set_target_window_for_test(Program& program, std::uint32_t lane,
                                         std::span<const TokenId> target) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    program.impl_->mtp[lane].target_window.assign(target.begin(), target.end());
}

std::uint32_t Program::mtp_window_for_test(const Program& program, std::uint32_t lane) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return program.impl_->mtp[lane].round.draft_window();
}

void Program::write_lane_state_markers_for_test(Program& program, std::uint32_t lane,
                                                std::uint8_t value) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    std::vector<std::uint8_t> markers(program.impl_->state_image_bytes, value);
    CUDA_CHECK(cudaMemcpyAsync(lane_state_base(*program.impl_, lane), markers.data(),
                               markers.size(), cudaMemcpyHostToDevice, program.impl_->stream));
    CUDA_CHECK(cudaStreamSynchronize(program.impl_->stream));
}

bool Program::verify_lane_state_markers_for_test(const Program& program, std::uint32_t lane,
                                                 std::uint8_t value) {
    if (lane >= program.impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    std::vector<std::uint8_t> actual(program.impl_->state_image_bytes);
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), lane_state_base(*program.impl_, lane), actual.size(),
                               cudaMemcpyDeviceToHost, program.impl_->stream));
    CUDA_CHECK(cudaStreamSynchronize(program.impl_->stream));
    return std::all_of(actual.begin(), actual.end(), [value](std::uint8_t byte) {
        return byte == value;
    });
}

void Program::decode_one_speculative(std::uint32_t lane) {
    LaneState& state = impl_->lanes[lane];
    if (state.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("decode requires a pending lane");
    }
    LaneMtpState& mtp = impl_->mtp[lane];
    if (!mtp.armed || mtp.pending.valid == 0 || mtp.target_window.size() < mtp.pending.valid + 1U) {
        throw std::logic_error("sensenova_loop_9b: speculative decode requires a verified window");
    }

    // Snapshot the accepted-prefix state before the target pass. On rejection every domain --
    // KV, GDN, mHC, MTP and host frontier -- is restored from this image.
    PrefixStateSnapshot accepted = capture_prefix_state(lane);
    const MtpDecision decision = verify_greedy(mtp.pending, mtp.target_window);
    const std::uint32_t drafted = mtp.pending.valid;

    Impl::LaneExecution& execution = impl_->execution[lane];
    std::vector<TokenId> licensed;
    licensed.reserve(decision.licensed);
    for (std::uint32_t i = 0; i < decision.accepted; ++i) {
        licensed.push_back(mtp.pending.drafted[i]);
    }
    licensed.push_back(decision.next_token);

    // Commit the accepted target sequence. The restored snapshot is the authoritative state at
    // the accepted prefix; next-round proposal state is cleared only after the commit.
    restore_prefix_state(lane, accepted);
    mtp.round.reset(state.frontier, execution.last_token);
    mtp.round.propose(impl_->draft_window, std::move(mtp.pending));
    mtp.round.commit(decision);
    mtp.pending = MtpProposal{};
    mtp.target_window.clear();
    mtp.armed = false;

    const std::uint32_t committed = std::min<std::uint32_t>(decision.licensed,
                                                             impl_->kv_capacity - state.frontier);
    state.frontier += committed;
    state.generated_tokens += committed;
    execution.last_token = licensed.back();

    Impl::SpeculativeEgress& egress = impl_->speculative_egress[lane];
    std::copy_n(licensed.begin(), std::min<std::size_t>(licensed.size(), egress.tokens.size()),
                egress.tokens.begin());
    egress.logprobs.fill(0.0F);
    egress.margins.fill(0.0F);
    impl_->speculative_licensed_counts[lane] = static_cast<std::uint32_t>(licensed.size());

    SpeculativeStats& stats = impl_->speculative_stats[lane];
    stats.backend = SpeculativeBackend::Mtp;
    stats.enabled = true;
    stats.draft_window = impl_->draft_window;
    ++stats.rounds;
    stats.drafted_tokens += drafted;
    stats.accepted_tokens += decision.accepted;
    if (decision.accepted < drafted) { ++stats.fallback_steps; }
    if (stats.accepted_per_position.size() < impl_->draft_window) {
        stats.accepted_per_position.resize(impl_->draft_window, 0);
    }
    for (std::uint32_t i = 0; i < decision.accepted; ++i) {
        ++stats.accepted_per_position[i];
    }
    state.lifecycle = Lifecycle::Active;
}

runtime::BatchedGeneratedRound Program::decode_batch(std::span<const std::uint32_t> lanes,
                                                     std::span<const runtime::RoundBudget> budgets) {
    if (lanes.empty() || lanes.size() > impl_->max_concurrency ||
        budgets.size() != lanes.size()) {
        throw std::invalid_argument("sensenova_loop_9b: invalid decode batch");
    }
    (void)budgets; // one greedy token per lane; the round budget bounds acceptance, not the pass
    impl_->egress_tokens.clear();
    impl_->egress_logprobs.clear();
    impl_->egress_margins.clear();
    impl_->egress_row_counts.clear();
    impl_->egress_span = 1;
    for (const std::uint32_t lane : lanes) {
        if (lane >= impl_->max_concurrency) {
            throw std::out_of_range("decode batch lane is out of range");
        }
        const bool speculative = impl_->draft_window != 0 && impl_->mtp[lane].armed;
        if (speculative) {
            decode_one_speculative(lane);
        } else {
            decode_one(lane);
        }
        // Unfiltered, like prefill: an EOS id is a token the round licenses, not a token the target
        // suppresses. The output session maps it to StopToken.
        const Impl::LaneExecution& execution = impl_->execution[lane];
        const Impl::SpeculativeEgress& spec = impl_->speculative_egress[lane];
        const std::uint32_t row_count = speculative ? impl_->speculative_licensed_counts[lane] : 1;
        for (std::uint32_t token = 0; token < row_count; ++token) {
            impl_->egress_tokens.push_back(speculative ? spec.tokens[token]
                                                       : execution.pending_tokens[token]);
            impl_->egress_logprobs.push_back(speculative ? spec.logprobs[token]
                                                        : execution.pending_logprobs[token]);
            impl_->egress_margins.push_back(speculative ? spec.margins[token]
                                                       : execution.pending_margins[token]);
        }
        impl_->egress_row_counts.push_back(static_cast<std::int32_t>(row_count));
    }
    return runtime::BatchedGeneratedRound{
        .tokens     = impl_->egress_tokens,
        .row_counts = impl_->egress_row_counts,
        .row_stride = 1,
        .logprobs   = impl_->egress_logprobs,
        .margins    = impl_->egress_margins,
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
        const bool speculative =
            impl_->draft_window != 0 && impl_->speculative_licensed_counts[lane] > 0;
        const std::uint32_t committed =
            speculative ? impl_->speculative_licensed_counts[lane] : accepted_tokens[row];
        state.generated_tokens += committed;
        state.frontier += committed;
        // The lane's next decode step starts from the token the round actually
        // licensed, so a rejected round never advances the host frontier.
        if (accepted_tokens[row] > 0 || speculative) {
            impl_->execution[lane].last_token = speculative
                                                     ? impl_->speculative_egress[lane]
                                                           .tokens[committed - 1]
                                                     : impl_->execution[lane].pending_tokens[0];
        }
        impl_->speculative_licensed_counts[lane] = 0;
    }
}

void Program::abort_lane(std::uint32_t lane) noexcept {
    if (lane < impl_->max_concurrency) {
        impl_->lanes[lane] = LaneState{};
        impl_->mtp[lane] = LaneMtpState{};
        impl_->speculative_licensed_counts[lane] = 0;
    }
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

PrefixStateSnapshot Program::capture_prefix_state(std::uint32_t lane) const {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (impl_->state_images.p == nullptr || impl_->state_image_bytes == 0) {
        throw std::logic_error("sensenova_loop_9b: prefix snapshot requires state images");
    }
    PrefixStateSnapshot snapshot;
    snapshot.device_ = DeviceBuffer(impl_->state_image_bytes);
    CUDA_CHECK(cudaMemcpyAsync(snapshot.device_.p, lane_state_base(*impl_, lane),
                               impl_->state_image_bytes, cudaMemcpyDeviceToDevice, impl_->stream));
    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    snapshot.token_count_ = impl_->lanes[lane].frontier;
    snapshot.last_token_  = impl_->execution[lane].last_token;
    return snapshot;
}

void Program::restore_prefix_state(std::uint32_t lane, const PrefixStateSnapshot& snapshot) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (snapshot.device_.p == nullptr || snapshot.device_.bytes != impl_->state_image_bytes) {
        throw std::invalid_argument(
            "sensenova_loop_9b: prefix snapshot does not match this Program's state image");
    }
    if (impl_->lanes[lane].lifecycle == Lifecycle::Free) {
        throw std::logic_error("sensenova_loop_9b: prefix restore requires an occupied lane");
    }
    CUDA_CHECK(cudaMemcpyAsync(lane_state_base(*impl_, lane), snapshot.device_.p,
                               impl_->state_image_bytes, cudaMemcpyDeviceToDevice, impl_->stream));
    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    impl_->lanes[lane].frontier        = snapshot.token_count_;
    impl_->execution[lane].last_token  = snapshot.last_token_;
}

Program::PrefixIdentity Program::prefix_identity_for_test(std::span<const TokenId> tokens) {
    // FNV-1a over the exact token sequence. Deterministic, order sensitive, and independent of
    // lane or Program so the same prompt always resolves to the same cached prefix.
    std::uint64_t hash = 1469598103934665603ULL;
    for (TokenId token : tokens) {
        std::uint64_t value = token;
        for (int i = 0; i < 4; ++i) {
            hash ^= (value >> (8 * i)) & 0xffULL;
            hash *= 1099511628211ULL;
        }
    }
    return PrefixIdentity{hash};
}

Program::PrefixIdentity Program::capture_engine_prefix(std::uint32_t lane,
                                                      std::uint32_t prompt_token_count) {
    if (lane >= impl_->max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    LaneState& state = impl_->lanes[lane];
    if (prompt_token_count == 0 || prompt_token_count > state.prompt_tokens ||
        state.prompt_tokens == 0) {
        throw std::invalid_argument(
            "sensenova_loop_9b: engine prefix must be a committed prompt prefix");
    }
    // The identity is the exact prompt token sequence, not just its length: a different prompt of
    // the same length cannot safely reuse these visit state images.
    const std::span<const TokenId> identity_tokens(state.prompt_token_ids.data(),
                                                   prompt_token_count);
    const std::uint64_t key = prefix_identity_for_test(identity_tokens).value;
    PrefixStateSnapshot image = capture_prefix_state(lane);
    impl_->engine_prefixes[key] = Impl::EnginePrefix{std::move(image)};
    return PrefixIdentity{key};
}

bool Program::has_engine_prefix(PrefixIdentity identity) const noexcept {
    return impl_->engine_prefixes.find(identity.value) != impl_->engine_prefixes.end();
}

bool Program::plan_request_for_prefix(std::uint32_t lane, PrefixIdentity identity,
                                      std::uint32_t prompt_token_count,
                                      const RequestPlan& plan) const noexcept {
    (void)prompt_token_count; // the caller validates the concrete prompt against cached frontier
    const auto found = impl_->engine_prefixes.find(identity.value);
    return lane < impl_->max_concurrency && found != impl_->engine_prefixes.end() &&
           plan.impl_ != nullptr;
}

GenerationTimings Program::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->max_concurrency ? impl_->execution[lane].timings : GenerationTimings{};
}

SpeculativeStats Program::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->max_concurrency ? impl_->speculative_stats[lane] : SpeculativeStats{};
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
