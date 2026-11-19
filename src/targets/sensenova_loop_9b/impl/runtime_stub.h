#pragma once

// Stage B deliberately publishes a minimal engine-compatible Program. Execution and state
// ownership are added in stages D/E; this surface exists so the registry and Engine variant
// compile against a closed target package without pretending that generation is implemented.

#include <ninfer/targets/sensenova_loop_9b/package.h>
#include "core/arena.h"
#include "runtime/contract/types.h"
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace ninfer {
class DeviceContext;
class DiskStateCache;
} // namespace ninfer

namespace ninfer::targets::sensenova_loop_9b { struct Package; }

namespace ninfer::targets::sensenova_loop_9b::detail {

struct StateImageInspection {
    void* base                 = nullptr;
    std::size_t bytes          = 0;
    std::size_t kv_bytes       = 0;
    std::size_t gdn_bytes      = 0;
    std::size_t mhc_bytes      = 0;
    void* mtp_kv_region        = nullptr;
    void* gdn_region           = nullptr;
    void* mhc_region           = nullptr;
};

struct LaneStateSnapshot {
    std::uint32_t lane                    = 0;
    bool occupied                         = false;
    bool retained                         = false;
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t generated_tokens        = 0;
    std::uint32_t frontier                = 0;
    std::uint32_t page_entitlement        = 0;
};

// Diagnostic view of one lane's decode graph. `configured` is the plan's graph intent;
// `attempted` records that the lane tried to capture once; `ready` reports a replayable graph.
// A configured lane that attempted and is not ready fell back to eager decode, and
// `diagnostic` says why.
struct DecodeGraphInspection {
    bool configured = false;
    bool attempted  = false;
    bool ready      = false;
    std::string diagnostic;
};

// A prefix-boundary copy of every device state domain -- all 15 KV domains, all 42 GDN slots,
// and all 48 mHC slots -- together with the host identity needed to resume at that boundary.
// The copy is complete by construction: its byte count must equal the Program's whole state
// image, so no looped visit identity is left behind when it is restored.
class PrefixStateSnapshot {
public:
    PrefixStateSnapshot() noexcept = default;
    ~PrefixStateSnapshot()         = default;

    PrefixStateSnapshot(PrefixStateSnapshot&&) noexcept            = default;
    PrefixStateSnapshot& operator=(PrefixStateSnapshot&&) noexcept = default;
    PrefixStateSnapshot(const PrefixStateSnapshot&)                = delete;
    PrefixStateSnapshot& operator=(const PrefixStateSnapshot&)     = delete;

    [[nodiscard]] bool empty() const noexcept { return device_.bytes == 0; }
    [[nodiscard]] std::size_t bytes() const noexcept { return device_.bytes; }
    [[nodiscard]] std::uint32_t token_count() const noexcept { return token_count_; }
    [[nodiscard]] std::uint32_t last_token() const noexcept { return last_token_; }

private:
    DeviceBuffer device_;
    std::uint32_t token_count_ = 0;
    std::uint32_t last_token_  = 0;
    friend class Program;
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

private:
    struct Impl;
    explicit RequestBasePlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend class Program;
    friend class RequestPlanForTest;
};

class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(RequestPlan&)       = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

private:
    struct Impl;
    explicit RequestPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend class Program;
    friend class RequestPlanForTest;
};

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;
    [[nodiscard]] bool use_cuda_graph() const noexcept;

private:
    struct Impl;
    explicit SequencePlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend class SequencePlanner;
    friend class Program;
    friend struct ::ninfer::targets::sensenova_loop_9b::Package;
};

class SequencePlanner {
public:
    static SequencePlanner create(const EngineOptions& options, WeightsProfile weights_profile);
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

private:
    struct Impl;
    explicit SequencePlanner(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct ::ninfer::targets::sensenova_loop_9b::Package;
};

class RequestPlanForTest {
public:
    [[nodiscard]] static RequestPlan make();
};

class Program {
public:
    // Identity of a committed prompt prefix. The value is a deterministic hash of the full
    // token sequence, so a shared prefix cannot be confused with a different prompt of the
    // same length.
    struct PrefixIdentity {
        std::uint64_t value = 0;
    };

    [[nodiscard]] static LaneStateSnapshot inspect_lane_for_test(const Program& program,
                                                                 std::uint32_t lane);
    [[nodiscard]] static StateImageInspection inspect_state_images_for_test(
        const Program& program);
    static void write_state_image_markers_for_test(Program& program, std::uint8_t value);
    static bool verify_state_image_markers_for_test(const Program& program, std::uint8_t value);
    [[nodiscard]] static DecodeGraphInspection inspect_decode_graph_for_test(
        const Program& program, std::uint32_t lane);
    static void seed_lane_for_test(Program& program, std::uint32_t lane,
                                   std::uint32_t frontier, std::uint32_t last_token);
    static void seed_pending_lane_for_test(Program& program, std::uint32_t lane,
                                           std::uint32_t frontier, std::uint32_t last_token);
    static void set_committed_prompt_for_test(Program& program, std::uint32_t lane,
                                              std::span<const TokenId> tokens);
    // Stage F test hooks. They configure the target's MTP window and publish a deterministic
    // draft/target window for verification tests; production construction keeps MTP disabled
    // until an engine-supplied plan opts in.
    static void configure_mtp_for_test(Program& program, std::uint32_t draft_window);
    static void set_draft_window_for_test(Program& program, std::uint32_t draft_window);
    static void set_proposal_for_test(Program& program, std::uint32_t lane,
                                      std::span<const TokenId> drafts);
    static void set_target_window_for_test(Program& program, std::uint32_t lane,
                                           std::span<const TokenId> target);
    [[nodiscard]] static std::uint32_t mtp_window_for_test(const Program& program,
                                                           std::uint32_t lane);
    [[nodiscard]] static PrefixIdentity prefix_identity_for_test(
        std::span<const TokenId> prompt_tokens);
    static void write_lane_state_markers_for_test(Program& program, std::uint32_t lane,
                                                  std::uint8_t value);
    [[nodiscard]] static bool verify_lane_state_markers_for_test(const Program& program,
                                                                 std::uint32_t lane,
                                                                 std::uint8_t value);
    static std::unique_ptr<Program> create(std::uint32_t active_lanes, std::uint32_t kv_pages,
                                           std::uint32_t context, std::size_t reservation_bytes);
    static std::unique_ptr<Program> create(const RuntimeModelView& model, cudaStream_t stream,
                                           std::uint32_t active_lanes, std::uint32_t kv_pages,
                                           std::uint32_t context, std::size_t reservation_bytes,
                                           bool use_cuda_graph = false);
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestBasePlan plan_request_base(const qwen3_6::PreparedPrompt& prompt,
                                                    const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane, const qwen3_6::PreparedPrompt& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                              const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                qwen3_6::PreparedPrompt&& prompt,
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

    // Stage F engine-level prefix reuse. `capture_engine_prefix` records the whole state image
    // and host frontier under `identity`; `start_prefill_lane_with_prefix` restores a matching
    // committed prefix into the lane's private state partition, skips the shared prompt tokens,
    // and prefills only the suffix. Identity includes every looped visit and state domain because
    // the copied image covers all 15 KV domains, 42 GDN slots, and 48 mHC slots.
    [[nodiscard]] PrefixIdentity capture_engine_prefix(std::uint32_t lane,
                                                       std::uint32_t prompt_token_count);
    [[nodiscard]] bool has_engine_prefix(PrefixIdentity identity) const noexcept;
    [[nodiscard]] bool plan_request_for_prefix(std::uint32_t lane, PrefixIdentity identity,
                                               std::uint32_t prompt_token_count,
                                               const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    start_prefill_lane_with_prefix(std::uint32_t lane, PrefixIdentity identity,
                                   qwen3_6::PreparedPrompt&& prompt, RequestPlan&& plan,
                                   runtime::TransientRegion transient);

    // Stage E6 prefix reuse primitive. `capture_prefix_state` copies the whole state image at the
    // lane's current frontier into a fresh device buffer and records that frontier and the last
    // committed token. `restore_prefix_state` writes the image back and reinstates those two host
    // values, so the next decode step continues from the snapshot's boundary. Restoring into a
    // free (aborted) lane is rejected: a snapshot must never resurrect released state.
    [[nodiscard]] PrefixStateSnapshot capture_prefix_state(std::uint32_t lane) const;
    void restore_prefix_state(std::uint32_t lane, const PrefixStateSnapshot& snapshot);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;
    void snapshot_lane_to_disk(std::uint32_t lane, DiskStateCache& disk_cache);
    void snapshot_turn_checkpoint_to_disk(std::uint32_t lane, DiskStateCache& disk_cache);
    void set_disk_state_cache(DiskStateCache* cache) noexcept;
    [[nodiscard]] std::string config_signature_slug() const;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    struct Impl;
    explicit Program(std::unique_ptr<Impl> impl) noexcept;
    // Stage E5A: one eager decode step for a pending lane. Samples exactly one
    // token from the lane's last committed token and frontier, and publishes it
    // as that lane's pending token. Lane-visible state (last token, frontier,
    // generated count) advances only in resolve_pending_batch; the KV/GDN/mHC
    // state images advance when the forward runs, which is the eager path's
    // commit point until snapshots land with graph/prefix work.
    void decode_one(std::uint32_t lane);
    void decode_one_speculative(std::uint32_t lane);
    static std::byte* lane_state_base(Program::Impl& impl, std::uint32_t lane);
    static const std::byte* lane_state_base(const Program::Impl& impl, std::uint32_t lane);
    static std::unique_ptr<Program> create_impl(const RuntimeModelView* model, cudaStream_t stream,
                                                std::uint32_t active_lanes,
                                                std::uint32_t kv_pages, std::uint32_t context,
                                                std::size_t reservation_bytes,
                                                bool use_cuda_graph);
    std::unique_ptr<Impl> impl_;
    friend struct ::ninfer::targets::sensenova_loop_9b::Package;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail
