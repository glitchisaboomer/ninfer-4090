#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/prompt_edit.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

// The delta-plane relocation path requires the attention kernel to counter-rotate Q per page.
// Widen this set only when that dtype's attention kernel implements the per-page rotation:
// BF16 rotates the register A-fragment, the i8 kernels rotate the FP32 rotary pair at the
// Q-quantization site and re-quantize on each key-page boundary.
constexpr bool delta_path_supports_dtype(DType dtype) {
    // BF16 attention kernels counter-rotate Q at the fragment level; the i8
    // kernels counter-rotate the FP32 rotary pair before quantization. Both
    // paths are implemented. Every i8 variant (int8, rk8v4, rk4v4, rk4v4-e8,
    // rk2v4-e8) dispatches to the same kernels and inherits the support.
    return dtype == DType::BF16 || dtype == DType::I8;
}

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.patches.empty()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    base->enable_suffix_cache_reuse      = options.enable_suffix_cache_reuse;
    base->suffix_reuse_max_spans         = options.suffix_reuse_max_spans;
    base->scr_mode                       = options.scr_mode;
    base->token_mask                     = options.token_mask;
    base->disable_speculation            = options.disable_speculation;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_6::VisionControl>(qwen3_6::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_6::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.turn_rewrite_boundary) {
        const std::uint32_t candidate = *prompt.identity.turn_rewrite_boundary;
        if (candidate == 0 || candidate >= base->summary.prompt_tokens) {
            throw std::invalid_argument("turn rewrite boundary must lie inside the prompt");
        }
        base->turn_rewrite_boundary = candidate;
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->turn_rewrite_boundary ? 1ULL : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;
    plan->token_mask                  = base.token_mask;
    plan->disable_speculation         = base.disable_speculation;

    if (base.allow_prefix_reuse && prompt.identity.reusable) {
        if (sequence.retained) {
            const bool dflash_append_ready =
                speculative_backend != SpeculativeBackend::DFlash ||
                sequence.dflash_context_frontier == sequence.execution_frontier;
            if (sequence.execution_frontier != 0 && dflash_append_ready &&
                qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                                sequence.execution_frontier)) {
                plan->reuse      = ReusePath::AppendAtFrontier;
                plan->reuse_base = sequence.execution_frontier;
            } else if (sequence.turn_checkpoint.valid && sequence.turn_checkpoint.frontier != 0 &&
                       sequence.turn_checkpoint.frontier < prompt.token_ids.size() &&
                       qwen3_6::detail::prefix_matches(prompt, sequence.ledger,
                                                       sequence.prefix_identity,
                                                       sequence.turn_checkpoint.frontier)) {
                plan->reuse      = ReusePath::RestoreTurnCheckpoint;
                plan->reuse_base = sequence.turn_checkpoint.frontier;
            }
        }
        if (plan->reuse == ReusePath::FullReset && disk_state_cache && disk_state_cache->enabled()) {
            const std::uint64_t model_hash = model_identity_hash();
            auto disk_match = disk_state_cache->find_longest_matching_prefix(model_hash, prompt.token_ids);
            if (disk_match && disk_match->matched_tokens > 0 &&
                disk_match->matched_tokens < prompt.token_ids.size()) {
                plan->reuse              = ReusePath::RestoreDiskCheckpoint;
                plan->reuse_base         = disk_match->matched_tokens;
                plan->disk_snapshot_path = disk_match->file_path;
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool is_disk_restore = plan->reuse == ReusePath::RestoreDiskCheckpoint;
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && sequence.tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || sequence.mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready =
            (plan->reuse == ReusePath::RestoreTurnCheckpoint && decoder->mtp_cache() != nullptr &&
             sequence.mtp_kv_valid >= plan->reuse_base - 1) ||
            is_disk_restore;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse              = ReusePath::FullReset;
            plan->reuse_base         = 0;
            plan->disk_snapshot_path.clear();
        }
    }

    if (plan->reuse == ReusePath::RestoreTurnCheckpoint &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !sequence.kv || !sequence.kv->backend ||
         sequence.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    // --- Suffix Cache Reuse -----------------------------------------------------------
    // Reuse resident KV for regions that survived an in-the-middle edit instead of
    // re-prefilling them. Text-only prompts, because Vision items re-scatter columns and their
    // consumer spans are not what the page remap assumes. Both speculative backends whose KV is a
    // second allocation covering the same tokens at the same logical pages -- the plain decode
    // backend and MTP -- because the runtime relocates them from the same spans in one step (see
    // `start_prefill_lane`). DFlash still keeps the existing behaviour: its context frontier is
    // carried separately and its cache is filled through `kv_cache_append_prefix`, so relocating
    // Text pages out from under it is a different problem than a second page table.
    //
    // A turn-rewrite boundary does not disable relocation. Every chat request carries one, because
    // the chat template always appends a generation prompt, so rejecting prompts that have one would
    // make this path unreachable from the served API. The boundary only bounds how far a span may
    // reach, which `plan_tail_relocation` applies.
    if (base.allow_prefix_reuse && base.enable_suffix_cache_reuse &&
        base.suffix_reuse_max_spans != 0 &&
        (speculative_backend == SpeculativeBackend::None ||
         speculative_backend == SpeculativeBackend::Mtp) &&
        base.vision_control == nullptr && sequence.retained && sequence.kv &&
        prompt.identity.reusable &&
        delta_path_supports_dtype(kv_dtype) &&
        (plan->reuse == ReusePath::FullReset || plan->reuse == ReusePath::AppendAtFrontier)) {
        const qwen3_6::detail::PromptEditDiff diff = qwen3_6::detail::diff_prompt_edit(
            sequence.ledger, prompt.token_ids, qwen3_6::detail::kPromptEditBlockTokens);

        // Selection rules live with the detector so they stay unit-testable on the host. `reuse_base`
        // is the prefix-reuse frontier, so a destination below it is already covered by ordinary
        // prefix reuse. Both modes produce spans; the prefilled ranges are then derived from the
        // spans, so the plan always reaches the runtime in the same shape. Tail mode selects one
        // contiguous run, so the derivation reproduces exactly the head and tail that mode has
        // always reported; segmented mode may interleave survivors anywhere, which yields one range
        // per gap and is the only structural difference.
        // A relocation whose source is off the page grid cannot be re-pointed at all, only copied, so
        // both the selector below and the expressibility rule it is judged by have to know which grid
        // they are planning against. Off by default: with the flag unset this block is the code it was
        // before the copying path existed, and the runtime's copied branch stays unreachable.
        const bool source_may_shift = qwen3_6::detail::scr_source_shift_enabled();
        std::vector<RelocatedSpan> plan_spans;
        if (base.scr_mode == ninfer::ScrRelocationMode::Segmented) {
            plan_spans = qwen3_6::detail::plan_relocations(
                diff, sequence.text_kv_valid, plan->reuse_base,
                static_cast<std::uint32_t>(prompt.token_ids.size()), base.turn_rewrite_boundary,
                base.suffix_reuse_max_spans);
        } else {
            qwen3_6::detail::TailRelocationTrace trace;
            const qwen3_6::detail::TailRelocationPlan tail = qwen3_6::detail::plan_tail_relocation(
                diff, sequence.text_kv_valid, plan->reuse_base,
                static_cast<std::uint32_t>(prompt.token_ids.size()), base.suffix_reuse_max_spans,
                base.turn_rewrite_boundary, &trace);
            if (scr_diagnostics_enabled() && !tail.viable) {
                // A refusal the planner made silently is the hardest kind to diagnose: the suite
                // reports the path it took, never the reason it declined, and a run can relocate one
                // request out of twelve for reasons that look identical to not having tried.
                std::ostringstream declined;
                declined << "[scr-diag] declined prompt_tokens=" << trace.prompt_tokens
                         << " reuse_base=" << trace.prefix_reuse_tokens
                         << " resident_kv=" << trace.resident_kv_tokens
                         << " max_spans=" << trace.max_spans
                         << " limit=" << trace.limit
                         << " boundary=" << (trace.boundary_supplied ? "yes" : "no")
                         << " spans_seen=" << trace.spans_seen
                         << " candidates=" << trace.candidates.size();
                if (trace.early_exit != nullptr) { declined << " early_exit=" << trace.early_exit; }
                if (trace.declined != nullptr) { declined << " declined=" << trace.declined; }
                for (const auto& reject : trace.rejected) {
                    declined << " reject{" << reject.reason << " old=[" << reject.old_begin << ","
                             << reject.old_end << ") new_begin=" << reject.new_begin
                             << " tokens=" << reject.tokens << "}";
                }
                std::cerr << declined.str() << '\n';
                std::cerr.flush();
            }
            if (tail.viable) {
                plan_spans = tail.spans;
            } else if (source_may_shift) {
                // The re-pointing selector declined, and with the copied path enabled that is not the
                // end of the story. Every dtype that reaches here reads the packed boundary range:
                // the attention kernels that consume it -- BF16 already, and the INT8 ones, which
                // choose the query frame for the keys inside a page's interval per key -- can
                // therefore take the exact destination interval rather than a whole-page
                // quantization of it. The aligned plan and the hardware-verified re-pointing path
                // remain unchanged when either can express it.
                plan_spans = qwen3_6::detail::plan_shifted_relocations(
                    diff, sequence.text_kv_valid, plan->reuse_base,
                    static_cast<std::uint32_t>(prompt.token_ids.size()), base.turn_rewrite_boundary,
                    base.suffix_reuse_max_spans, /*destination_may_shift=*/true);
            }
        }

        // The prefilled ranges, derived from the spans rather than carried next to them: one source
        // for both modes, and the tiling with the spans is what makes the accounting below honest.
        const std::uint32_t prompt_tokens_count =
            static_cast<std::uint32_t>(prompt.token_ids.size());
        std::vector<qwen3_6::detail::PrefillRange> plan_ranges =
            qwen3_6::detail::relocation_prefill_ranges(plan_spans, plan->reuse_base,
                                                       prompt_tokens_count);
        if (!plan_spans.empty()) {
            // Overwriting `reuse` with SuffixRelocation here is what makes the MTP coupling the final
            // block's job rather than this block's: that block keys off the reuse path, and
            // SuffixRelocation is a fifth one it has to know about. Nothing in this block can ask
            // for the staged MTP prefill itself -- `prepare_mtp` is set further down, after the
            // turn-checkpoint decision, which may still downgrade this plan to a plain FullReset. The
            // assertion that the staged MTP prefill exists therefore lives after that block.
            //
            // A plan that is not expressible is refused rather than half-applied, and the refusal is
            // printed whether or not diagnostics are enabled: reaching it means a selector produced
            // spans the runtime cannot execute, which is a programming error rather than a prompt
            // shape, and a silent fallback would hide it while still reporting cold-prefill reuse.
            if (!qwen3_6::detail::relocation_is_expressible(plan_spans, plan_ranges, plan->reuse_base,
                                                            prompt_tokens_count, source_may_shift)) {
                std::cerr << "[scr] refused an inexpressible relocation plan: spans="
                          << plan_spans.size() << " ranges=" << plan_ranges.size()
                          << " reuse_base=" << plan->reuse_base
                          << " prompt_tokens=" << prompt_tokens_count << '\n';
                plan_spans.clear();
                plan_ranges.clear();
            }
        }
        if (!plan_spans.empty()) {
            std::uint64_t relocated = 0;
            for (const RelocatedSpan& span : plan_spans) { relocated += span.tokens; }
            plan->reuse                           = ReusePath::SuffixRelocation;
            plan->relocated_prefill_ranges        = std::move(plan_ranges);
            plan->relocated_spans                 = std::move(plan_spans);
            plan->summary.relocated_prompt_tokens = static_cast<std::uint32_t>(relocated);
        }
    }

    const std::optional<std::uint32_t> desired = base.turn_rewrite_boundary;
    const bool can_keep                        = desired && plan->reuse != ReusePath::FullReset &&
                          plan->reuse != ReusePath::RestoreDiskCheckpoint &&
                          sequence.turn_checkpoint.valid &&
                          sequence.turn_checkpoint.frontier == *desired &&
                          qwen3_6::detail::prefix_matches(prompt, sequence.ledger,
                                                          sequence.prefix_identity, *desired);
    if (!desired) {
        plan->turn_checkpoint_action = TurnCheckpointAction::Drop;
    } else if (can_keep) {
        plan->turn_checkpoint_action = TurnCheckpointAction::KeepExisting;
    } else if (plan->reuse == ReusePath::RestoreDiskCheckpoint) {
        plan->turn_checkpoint_action = TurnCheckpointAction::Drop;
    } else {
        if (*desired <= plan->reuse_base) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
            // The SCR block may have set relocation fields before this downgrade. A cold prefill
            // executes none of them, so leaving them set would report reused tokens the request
            // never reuses: `reusable_prompt_tokens` would count them, the service-work projection
            // would under-state the prefill that follows, and the telemetry would report a
            // relocation that never ran. Clear them here rather than trusting every consumer to
            // read `reuse` before the accounting.
            plan->relocated_prefill_ranges.clear();
            plan->relocated_spans.clear();
            plan->summary.relocated_prompt_tokens = 0;
        }
        plan->turn_checkpoint_action           = TurnCheckpointAction::CaptureNew;
        plan->turn_checkpoint_capture_frontier = desired;
    }

    plan->summary.reusable_prompt_tokens =
        plan->reuse_base + plan->summary.relocated_prompt_tokens;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (plan->reuse == ReusePath::RestoreTurnCheckpoint ||
                   plan->reuse == ReusePath::RestoreDiskCheckpoint) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = MtpBridgeMode::BeforeSuffix;
        } else if (plan->reuse == ReusePath::SuffixRelocation) {
            // The relocated run's MTP rows are re-pointed alongside Text's, so only the two prefilled
            // ranges need staging, and the walk's `prepare_mtp` places them on the same pages the
            // Text remap chose.
            //
            // No bridge, deliberately. The bridge writes the MTP row at the frontier from the base
            // model's hidden state at `base - 1`, and a mid-prompt edit's frontier is a position the
            // lane never retained: the one hidden state kept per lane, `sequence.tail_hidden`, is
            // the previous sequence's *last* position, which is `base - 1` only for the
            // AppendAtFrontier path. The MTP stem pairs row p with token p+1
            // (`plan_mtp_alignment_window`: `shifted_embedding_begin = chunk_begin + 1`), so the two
            // rows the edit moves -- the row at `base - 1` and the last row of the relocated run,
            // each paired with a pre-edit token -- stay as the relocation left them. That costs
            // draft quality at those positions and nothing else: drafts are verified against the
            // base model, so a mist-paired row can lose a draft but never become a wrong accepted
            // token. Recovering the true hidden is the open design question for this path; it is
            // recorded here so the omission reads as a decision rather than an oversight.
            plan->prepare_mtp = true;
            plan->mtp_bridge  = MtpBridgeMode::None;
        }
    }
    // The coupling, asserted where it can hold: the block above is what stages the MTP prefill, so a
    // relocated plan under MTP must have come through one of its arms. Without it the staged walk
    // prefills Text while MTP's rows stay unwritten, and the only thing that notices is the
    // end-of-prompt check.
    if (speculative_backend == SpeculativeBackend::Mtp &&
        plan->reuse == ReusePath::SuffixRelocation && !plan->prepare_mtp) {
        throw std::logic_error("suffix relocation plan has no staged MTP prefill");
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_6::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    // Extra prefill steps that no token count can predict. Vision uses and turn-checkpoint captures
    // each interrupt the chunk walk; so does a relocation, because it prefills several ranges with
    // the relocated runs skipped between them. The projection counts chunks over one contiguous
    // span, so `ceil((head+tail)/chunk)` undercounts a two-range prefill by up to one: each range's
    // last partial chunk is a whole step, and ceil(h/c) + ceil(t/c) exceeds ceil((h+t)/c) by at most
    // one. That generalises to one extra step per further range, which is what the count below says,
    // and it is not a rounding nicety: a 6699-token prompt reusing 3008 relocated tokens prefilled
    // 3691 tokens in six steps against a projected five, and the request died one quantum short at
    // 68 of 68. With interleaved survivors the range count is what the plan actually carries, so the
    // projection and the staged walk cannot disagree about how many prefills there are.
    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->turn_checkpoint_capture_frontier ? 1ULL : 0ULL) +
        qwen3_6::detail::relocation_prefill_splits(plan->relocated_prefill_ranges);
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->summary.reusable_prompt_tokens, prefill_chunk,
                               prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
