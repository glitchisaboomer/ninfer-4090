#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/disk_state_cache.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/prompt_edit.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"
#include "targets/qwen3_6/impl/runtime/yarn.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData = qwen3_6::PreparedPromptData;

using ReusePath = ninfer::PrefixReusePath;

enum class TurnCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    CaptureNew,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

// Host-side relocation geometry for Suffix Cache Reuse. Defined with the edit detector so
// the selection rules stay testable without a device; this alias only re-homes it into the
// runtime namespace.
using RelocatedSpan = qwen3_6::detail::RelocatedSpan;
using PrefillRange   = qwen3_6::detail::PrefillRange;

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<std::uint32_t> turn_rewrite_boundary;
    bool allow_prefix_reuse = false;
    bool enable_suffix_cache_reuse = false;
    std::uint32_t suffix_reuse_max_spans = 0;
    ninfer::ScrRelocationMode scr_mode = ninfer::ScrRelocationMode::Tail;
    std::vector<std::uint32_t> token_mask;
    bool disable_speculation = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    // Populated only when `reuse == SuffixRelocation`. Empty means every token after
    // `reuse_base` is prefilled, which is the assumption every other reuse path relies on.
    std::vector<NINFER_QWEN36_RUNTIME_NS::RelocatedSpan> relocated_spans;
    // Every range the runtime prefills, in prompt order, with `relocated_spans` as the gaps between
    // them. Together they tile `[reuse_base, prompt_tokens)` exactly -- a token is prefilled or
    // relocated but never unaccounted -- which is what keeps `summary.relocated_prompt_tokens` and
    // the service-work projection honest, and why the runtime can serve the gaps from resident KV
    // without a second accounting pass. One range is the ordinary "prefill everything after the
    // frontier" shape and the two around a contiguous run; more than one is what interleaved
    // survivors need. Derived from the spans by `relocation_prefill_ranges`, so there is one source.
    std::vector<NINFER_QWEN36_RUNTIME_NS::PrefillRange> relocated_prefill_ranges;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::TurnCheckpointAction turn_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::TurnCheckpointAction::Drop;
    std::optional<std::uint32_t> turn_checkpoint_capture_frontier;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::filesystem::path disk_snapshot_path;
    std::vector<std::uint32_t> token_mask;
    bool disable_speculation = false;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct TurnCheckpoint {
    bool valid             = false;
    std::uint32_t frontier = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor turn_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    TurnCheckpoint turn_checkpoint;
    std::uint32_t last_disk_snapshot_tokens = 0;
};

// Diagnostic only, and only populated when NINFER_SCR_DIAG is set. One probe per relocated
// destination page, fingerprinted after the re-rotation and re-checked once the prefill that consumes
// those pages has finished. The relocation diagnostic's before/after pair brackets only the
// re-rotation, so without this nothing observes whether prefill then overwrote a page it was
// supposed to leave alone.
//
// The fingerprints are per row rather than per page. A page-level sum|x| says that a page changed
// without saying what wrote to it, and the two candidate writers look identical from a scalar: a
// re-rotation preserves sum|x| to within a fraction of a percent, while a prefilled token replaces
// four whole rows. Row granularity is what separates them, and it names the position and head.
struct RelocatedPageProbe {
    std::int32_t logical_page = -1;
    // Indexed kv_head * kPagedKVPageSize + page_offset, matching the plane's page-relative layout.
    std::vector<float> row_fingerprints;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    std::vector<std::uint32_t> initial_token_mask;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<std::uint32_t> turn_checkpoint_capture_frontier;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        // From the plan, not derivable from `base`: a relocated run is skipped rather than prefilled,
        // so it is neither the frontier nor anything the cursor passes through.
        std::uint32_t relocated_prompt_tokens = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool host_input_consumed_pending = false;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
        // Prefilled ranges in prompt order. Empty means the single range `[base, prompt_tokens)`,
        // which is what every reuse path except suffix relocation has. Suffix relocation fills this
        // with the plan's two ranges, leaving the relocated run in between to be skipped.
        std::vector<PrefillRange> prefill_ranges;

        std::vector<RelocatedPageProbe> relocated_page_probes;

        // End of the prefilled range containing `position`, or the prompt end when the caller only
        // has the default range.
        [[nodiscard]] std::uint32_t prefill_range_end(std::uint32_t position) const {
            for (const PrefillRange& range : prefill_ranges) {
                if (position >= range.begin && position < range.end) { return range.end; }
            }
            return prompt_tokens;
        }

        // A cursor that has completed a prefilled range resumes at the next one, skipping the
        // relocated run in between.
        [[nodiscard]] std::uint32_t skip_relocated_runs(std::uint32_t position) const {
            for (const PrefillRange& range : prefill_ranges) {
                if (position >= range.begin && position < range.end) { return position; }
                if (position < range.begin) { return range.begin; }
            }
            return position;
        }
    };

    std::optional<Prefill> prefill;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void set_token_mask_lane(std::uint32_t lane, std::span<const std::uint32_t> mask);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t retained_lane_depth(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;
    void snapshot_lane_to_disk(std::uint32_t lane, DiskStateCache& disk_cache);
    void snapshot_turn_checkpoint_to_disk(std::uint32_t lane, DiskStateCache& disk_cache);
    void set_disk_state_cache(DiskStateCache* cache) noexcept { disk_state_cache = cache; }
    [[nodiscard]] std::uint64_t model_identity_hash() const noexcept {
        constexpr std::uint64_t kFnv1aPrime       = 1099511628211ULL;
        constexpr std::uint64_t kFnv1aOffsetBasis = 14695981039346656037ULL;
        auto combine = [](std::uint64_t h, std::uint64_t val) noexcept -> std::uint64_t {
            return (h ^ val) * 1099511628211ULL;
        };

        std::uint64_t h = kFnv1aOffsetBasis;
        h = combine(h, static_cast<std::uint64_t>(model.token_embedding.payload_bytes));
        h = combine(h, static_cast<std::uint64_t>(model.token_embedding.shape[0]));
        h = combine(h, static_cast<std::uint64_t>(model.token_embedding.shape[1]));
        h = combine(h, static_cast<std::uint64_t>(capacity));
        h = combine(h, static_cast<std::uint64_t>(kv_capacity));
        h = combine(h, static_cast<std::uint64_t>(kv_dtype));
        h = combine(h, static_cast<std::uint64_t>(kv_quant_group));

        std::uint64_t flags = 0;
        if (kv_packed_k)   flags |= (1ULL << 0);
        if (kv_packed_v)   flags |= (1ULL << 1);
        if (kv_rotate_k)   flags |= (1ULL << 2);
        if (kv_rotate_v)   flags |= (1ULL << 3);
        if (kv_e8_lattice) flags |= (1ULL << 4);
        if (kv_e8_root)    flags |= (1ULL << 5);
        if (vision_enabled) flags |= (1ULL << 6);
        h = combine(h, flags);

        h = combine(h, static_cast<std::uint64_t>(speculative_backend));
        h = combine(h, static_cast<std::uint64_t>(draft_window));
        h = combine(h, static_cast<std::uint64_t>(proposal_head));
        return h;
    }

    [[nodiscard]] std::string config_signature_slug() const {
        std::string s;
        if (model.token_embedding.shape[1] == 5120) {
            s += "qwen3_8_27b";
        } else if (model.token_embedding.shape[1] == 2048) {
            s += "qwen3_6_35b_a3b";
        } else {
            s += "qwen_h" + std::to_string(model.token_embedding.shape[1]);
        }

        if (kv_dtype == DType::BF16) {
            s += "_bf16";
        } else if (kv_e8_root) {
            s += "_rk2v4-e8";
        } else if (kv_e8_lattice) {
            s += "_rk4v4-e8";
        } else if (kv_packed_k) {
            s += "_rk4v4";
        } else if (kv_rotate_v) {
            s += "_rk8v4";
        } else {
            s += "_int8";
        }

        if (speculative_backend == SpeculativeBackend::Mtp) {
            s += "_mtp_d" + std::to_string(draft_window);
            if (proposal_head == ProposalHead::Optimized) {
                s += "_lmhead";
            }
        } else if (speculative_backend == SpeculativeBackend::DFlash) {
            s += "_dflash_d" + std::to_string(draft_window);
        } else {
            s += "_none";
        }

        if (vision_enabled) {
            s += "_vision";
        }

        if (capacity >= 1000 && (capacity % 1000 == 0)) {
            s += "_ctx" + std::to_string(capacity / 1000) + "k";
        } else {
            s += "_ctx" + std::to_string(capacity);
        }

        return s;
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const bool kv_packed_v;
    const bool kv_rotate_k;
    const bool kv_rotate_v;
    const bool kv_packed_k;
    const bool kv_e8_lattice;
    const bool kv_e8_root;
    const ProposalHead proposal_head;
    const float attn_scale;
    const bool vision_enabled;

    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t text_kv_bytes;
    const std::size_t mtp_kv_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t dflash_kv_bytes;
    const std::size_t replay_records_bytes;
    const std::size_t graph_allowance_bytes;
    std::size_t graph_observed_bytes = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor token_masks;
    Tensor tail_hidden_store;
    Tensor turn_checkpoint_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    // Logprob of the token a completing prefill picked. Prefill has no decode egress to carry it,
    // so it is drained next to `host_tokens` and published through PrefillStepResult::round.logprobs.
    float host_logprob = 0.0f;
    // Decision margin beside `host_logprob`; see ops::logprob_of_choice.
    float host_margin = 0.0f;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    DiskStateCache* disk_state_cache          = nullptr;

    // The Text RoPE inverse-frequency table this program installed, kept so that every Op which has
    // to reproduce or undo that rotation is handed the same table. It is the YaRN-scaled table once
    // the configured context capacity exceeds the model's native context, so an Op that assumed the
    // built-in defaults would use the wrong angles in that configuration.
    std::array<float, 32> text_rope_inv_freq = compute_yarn_config(0).inv_freq;

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;

    // Suffix-relocation diagnostic. No-op unless NINFER_SCR_DIAG is set. Reports the applied plan,
    // the page remap, per-page fingerprints taken before and after the re-rotation, and whether the
    // relocated K plane is finite, since the relocation path has produced badly wrong output while
    // every component tests clean in isolation.
    void log_scr_relocation(const RequestPlanImpl& request_plan, std::uint32_t prompt_tokens,
                            std::uint32_t base, std::span<const std::int32_t> remap,
                            const qwen3_6::PagedKVCacheView& view,
                            std::span<const float> fingerprints_before, cudaStream_t stream) const;

    // `stream` is required and must be the stream the Kernels were launched on. A blocking
    // cudaMemcpy on the legacy default stream does not drain a non-blocking stream, so reading
    // without it returns whatever was resident before the launches -- which is how this diagnostic
    // reported a re-rotation as never having happened.
    [[nodiscard]] float scr_page_fingerprint(const PagedKVLayerView& layer,
                                             std::int32_t physical_page,
                                             cudaStream_t stream) const;

    // sum|x| per (kv_head, page_offset) row of one physical page. Strictly finer than
    // scr_page_fingerprint, which collapses the whole page to one number.
    [[nodiscard]] std::vector<float> scr_page_row_fingerprints(const PagedKVLayerView& layer,
                                                              std::int32_t physical_page,
                                                              cudaStream_t stream) const;

    // The other half of the relocation diagnostic, run once the prefill that consumes the relocated
    // pages has finished. log_scr_relocation brackets only the re-rotation, so it cannot see a
    // relocated page that prefill later overwrites -- which is the one failure mode the integration
    // test does not cover, because it exercises the Op and the kernels rather than the sequencing.
    // Reports each relocated destination page's fingerprint alongside its post-rotation value, so a
    // page that survived the prefill reads identical and a page that did not names the damage.
    void log_scr_relocation_integrity(const qwen3_6::PagedKVCacheView& view,
                                      std::span<const RelocatedPageProbe> probes,
                                      cudaStream_t stream) const;

    // One line per prefill chunk the relocation branch drives, so the trace shows the ranges, the
    // cursor skip across the relocated run, and the key extent each chunk makes visible to attention.
    // The relocated page surviving prefill intact does not by itself mean attention read it, and this
    // is the only place that states how far the visible extent reached.
    void log_scr_prefill_chunk(std::uint32_t base, std::uint32_t cursor, std::uint32_t nominal,
                               std::uint32_t range_end, std::uint32_t resumed,
                               const qwen3_6::PagedKVCacheView& view, cudaStream_t stream) const;

    // The GDN half of the relocation diagnostic.
    //
    // A Linear Attention slot is one recurrent summary of a whole sequence rather than a per-token
    // cache, so it cannot be relocated and has no RoPE to undo. The relocated run is simply not
    // forwarded; the state is kept. That follows the authors' design -- "snapshot the recurrent
    // state before the edit and continue from that snapshot" -- which their reference implements as
    // `KVREUSE_SSM_MODE=fork` (the default): restore the state saved at the end of the previous
    // prompt. program_impl.h therefore keeps the resident slot and lets only the ranges the prefill
    // covers advance it; it is not zeroed, because dropping the reused prefix's contribution from
    // 48 of 64 layers is a far wider approximation than SCR accepts and bounds with K.
    //
    // This reports, per stage, how much magnitude each layer's conv and recurrent halves carry, so
    // the state a stage ends with can be lined up against the cold-start control (`full_reset`) and
    // read for drift. `full_reset` is the only stage that takes a zero, so only it reports the
    // before/after pair; `fork` and `post_prefill` report the single retained magnitude. A non-zero
    // non-finite count means a kernel produced garbage, which is a different fault entirely.
    //
    // Reading the whole slot costs a full PCIe round trip per stage, which is why this stays behind
    // the same NINFER_SCR_DIAG switch as the KV probe rather than running unconditionally.
    void log_scr_gdn_state(const char* stage, bool zero_taken,
                           const LinearAttentionStateFingerprint& before,
                           const LinearAttentionStateFingerprint& after,
                           cudaStream_t stream) const;

    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
