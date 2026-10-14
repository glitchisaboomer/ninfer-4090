#pragma once


// Host-side detection of token spans that survive a context edit.
//
// This is the Phase 1-2 half of Suffix Cache Reuse (SCR): it answers "which whole
// blocks of the new prompt are token-identical to blocks of the previous prompt, and
// where did they move to". It performs no KV or device work and holds no state, so the
// runtime can call it before dispatch and decide whether cached KV for a surviving
// region may be relocated instead of re-prefilled.
//
// Cost is O(previous.size() + current.size()). Matching is block granularity, aligned
// with kPagedKVPageSize, so an edit that splits a block forfeits the partial block;
// that wastes at most (block_tokens - 1) tokens per edit boundary.

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// Relocation diagnostics are off unless NINFER_SCR_DIAG is set, so a refusal costs nothing in
// production. Defined here rather than in the runtime because the planner needs it too: a relocation
// that is declined is exactly as interesting as one that is applied, and until this existed only the
// application was observable.
inline bool scr_diagnostics_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SCR_DIAG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

// Whether relocation may move a surviving run whose *source* is not on the page grid.
//
// Off by default. The path it selects copies the KV payload instead of re-pointing the page that
// already holds it, which is what lets a run survive an edit whose token delta is not a multiple of
// the 64-token page -- the shape of essentially every edit a user types, and one the aligned path
// can never relocate (see `docs/SCR.md`). The re-pointing path is
// hardware-verified and stays the default until this one has been through the same ladder, so a
// regression here cannot reach a default run.
inline bool scr_source_shift_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SCR_SOURCE_SHIFT");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

// Diff granularity. Matches kPagedKVPageSize so a surviving span maps onto whole KV
// pages and can be described by page-aligned relocation metadata.
inline constexpr std::uint32_t kPromptEditBlockTokens = 64;

// A contiguous run of tokens that is token-identical in both prompts, expressed in
// whole blocks. Ranges are half-open: [begin, begin + tokens).
struct SurvivingSpan {
    // Token index of the run in the previous prompt.
    std::uint32_t old_begin = 0;
    // Token index of the same run in the current prompt.
    std::uint32_t new_begin = 0;
    // Run length in tokens. A positive multiple of the block size when `diff_prompt_edit` produced
    // the span; the shifted detector reports the run's true length, which is any positive count.
    std::uint32_t tokens = 0;

    [[nodiscard]] std::uint32_t old_end() const noexcept { return old_begin + tokens; }
    [[nodiscard]] std::uint32_t new_end() const noexcept { return new_begin + tokens; }
};

struct PromptEditDiff {
    // Surviving spans ordered by increasing new_begin. Spans are pairwise disjoint in
    // both prompts, and both the old and the new begins are strictly increasing.
    std::vector<SurvivingSpan> spans;
    // Total tokens covered by `spans`. This is the token count that may skip re-prefill
    // if every span is relocated.
    std::uint32_t reusable_tokens = 0;

    [[nodiscard]] bool has_reuse() const noexcept { return !spans.empty(); }

    // Length of the reusable prefix, or 0 when the prompts do not begin with an
    // identical block run. This is the region the ordinary prefix-reuse path already
    // covers, so SCR only needs to relocate what follows it.
    [[nodiscard]] std::uint32_t leading_prefix_tokens() const noexcept {
        if (spans.empty() || spans.front().old_begin != 0 || spans.front().new_begin != 0) {
            return 0;
        }
        return spans.front().tokens;
    }

    // Tokens after the reusable prefix that survive the edit and would otherwise be
    // re-prefilled. This is the SCR-specific saving.
    [[nodiscard]] std::uint32_t relocated_tokens() const noexcept {
        return reusable_tokens - leading_prefix_tokens();
    }
};

// One surviving region that Suffix Cache Reuse would relocate instead of re-prefilling.
// `source_begin` indexes the previous prompt/resident KV; `dest_begin` indexes the current
// prompt. Both ranges are disjoint. The re-pointing path requires whole pages at both ends; the
// copied path accepts arbitrary token offsets at both ends and records each page's relocated
// interval.
//
// What bounds that freedom is the one interval a page can carry: at most one relocated span may
// cover a given logical page, because the packed range plane names a single [begin, end) shift per
// page and a page two spans met in could describe neither. Tokens prefilled into a page a span
// also reloads are not a violation -- they simply sit outside the span's interval and keep the
// ordinary-page route -- so a span may begin or end mid-page. `relocation_is_expressible`
// enforces the one-span-per-page rule and `plan_rope_delta_page_ranges` enforces it again, so a
// plan that breaks it is refused rather than silently truncated.
struct RelocatedSpan {
    std::uint32_t source_begin = 0;
    std::uint32_t dest_begin   = 0;
    std::uint32_t tokens       = 0;

    [[nodiscard]] std::uint32_t source_end() const noexcept { return source_begin + tokens; }
    [[nodiscard]] std::uint32_t dest_end() const noexcept { return dest_begin + tokens; }
};

// Turn a diff into the relocation geometry a runtime can act on.
//
// This is the general selector, for the segmented runtime described in
// `docs/SCR.md` as Path B: surviving spans interleaved anywhere in the prompt,
// rather than the one contiguous run `plan_tail_relocation` selects.
//
// A span is only relocatable when all of the following hold:
//
//   * It is not the leading span. The leading span is ordinary prefix reuse, which the existing
//     planner already expresses, so relocating it would double account the same tokens.
//   * Its source range is fully inside `resident_kv_tokens`, because resident KV for anything past
//     the resident frontier does not exist.
//   * Its destination range begins at or after `prefix_reuse_tokens`. A destination inside that
//     region is already covered by ordinary prefix reuse, and a span that straddles the boundary
//     would be claimed twice, so the whole span is dropped rather than split.
//   * Its destination ends before the last token a span may reach: the smaller of
//     `prompt_tokens - 1` and `turn_rewrite_boundary`. The final prompt token must always be
//     prefilled, because prompt logits can only come from a real forward pass. The boundary is the
//     first token whose KV is already stale, because the assistant's reasoning is about to be
//     rewritten there, and a span carrying tokens at or after it would reuse KV that is stale by
//     definition. A span overrunning either limit is trimmed back to the last whole block ending
//     before it rather than dropped, so a run keeps the part that is still valid.
//
// Trimming happens before the cap is applied, so a span shortened by a limit is compared at the
// length it would actually relocate.
//
// Of the survivors the `max_spans` longest are kept and the rest are left to be re-prefilled, which
// bounds how much approximation one edit introduces. The result is returned in prompt order.
//
// A `max_spans` of zero returns no spans.
[[nodiscard]] std::vector<RelocatedSpan> plan_relocations(const PromptEditDiff& diff,
                                                         std::uint32_t resident_kv_tokens,
                                                         std::uint32_t prefix_reuse_tokens,
                                                         std::uint32_t prompt_tokens,
                                                         std::optional<std::uint32_t> turn_rewrite_boundary,
                                                         std::uint32_t max_spans);

// The source-shifting selector: every survivor `diff_prompt_edit_shifted` found, left off-grid at the
// source. Destination pages are quantized unless `destination_may_shift` is true and the attention
// path consumes packed sub-page delta ranges.
//
// Same rules as `plan_relocations` for everything but the grid: the leading run is prefix reuse and
// is skipped, the source has to be resident, the destination has to start at or after the reuse
// frontier, and no span may reach the limit. With `destination_may_shift == false`, the destination
// is trimmed inward to whole pages. With it true, the full surviving token interval is returned;
// copied remap and boundary-aware attention handle mixed destination pages.
//
// When destinations are quantized, up to 63 tokens at each end are recomputed. A run with no whole
// destination page returns nothing in that mode; boundary-aware mode can relocate any nonempty run.
//
// In either mode the returned spans never share a logical destination page with each other. That is
// not a property of the selector passing the checks in `relocation_is_expressible` -- it is what
// lets the packed range plane name one interval per page at all, so a caller that merges, splits or
// otherwise combines these spans downstream has to preserve it.
[[nodiscard]] std::vector<RelocatedSpan> plan_shifted_relocations(
    const PromptEditDiff& diff, std::uint32_t resident_kv_tokens,
    std::uint32_t prefix_reuse_tokens, std::uint32_t prompt_tokens,
    std::optional<std::uint32_t> turn_rewrite_boundary, std::uint32_t max_spans,
    bool destination_may_shift = false);

// A range of prompt tokens the runtime prefills, `[begin, end)`.
struct PrefillRange {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;

    [[nodiscard]] bool empty() const noexcept { return begin >= end; }
    [[nodiscard]] std::uint32_t tokens() const noexcept { return end > begin ? end - begin : 0; }
};

// A tail-relocation plan: the runtime prefills `head` and `tail` and serves everything between them
// from resident KV.
//
// This is the shape the roadmap calls Scope A, and it is the widest plan the runtime can act on
// without prefilling a disjoint set of many ranges: the relocated spans are contiguous, so exactly
// two prefilled ranges remain.
// Why a surviving span could not be relocated. Stable labels: they appear in the diagnostic and are
// what a reader matches against to tell "the edit left nothing to reuse" from "the reuse was found
// and then thrown away by a limit".
struct TailRelocationReject {
    const char* reason = "";
    std::uint32_t old_begin = 0;
    std::uint32_t old_end   = 0;
    std::uint32_t new_begin = 0;
    std::uint32_t tokens    = 0;
};

// What the planner decided, whether or not it relocated anything. Filled on every call so that a
// decline is as inspectable as an application: the relocation diagnostic only prints when a relocation
// is applied, which left every refusal invisible and made a suite that relocated one request out of
// twelve indistinguishable from one that relocated nothing.
struct TailRelocationTrace {
    std::uint32_t prompt_tokens       = 0;
    std::uint32_t prefix_reuse_tokens = 0;
    std::uint32_t resident_kv_tokens  = 0;
    std::uint32_t max_spans           = 0;
    // The token no span may reach, and whether a turn-rewrite boundary supplied it. A boundary far
    // below the prompt end is the usual reason a long prompt relocates only a fraction of itself.
    std::uint32_t limit            = 0;
    bool boundary_supplied        = false;
    const char* early_exit        = nullptr;
    std::size_t spans_seen         = 0;
    std::vector<TailRelocationReject> rejected;
    std::vector<RelocatedSpan> candidates;
    // Set when candidates existed but none was selected.
    const char* declined = nullptr;
};

struct TailRelocationPlan {
    // False when no contiguous surviving run is usable; the caller must then treat the prompt as an
    // ordinary prefix-reuse append.
    bool viable = false;
    // Prefilled before the relocated run, `[prefix_reuse_tokens, spans.front().dest_begin)`.
    PrefillRange head;
    // The contiguous surviving run, in prompt order.
    std::vector<RelocatedSpan> spans;
    // Prefilled after the relocated run, `[spans.back().dest_end(), prompt_tokens)`. Never empty: the
    // final prompt token is always prefilled, because prompt logits can only come from a real
    // forward pass.
    PrefillRange tail;

    [[nodiscard]] std::uint32_t relocated_tokens() const noexcept {
        std::uint32_t total = 0;
        for (const RelocatedSpan& span : spans) { total += span.tokens; }
        return total;
    }
};

// Plan a tail relocation for an edited prompt.
//
// Selection is the longest contiguous run of surviving spans, capped at `max_spans` spans by taking
// the best window inside the run, so this returns the single most valuable relocation the runtime can
// act on rather than the sum of every survivor. The run may sit anywhere in the prompt: whatever is
// left on either side is prefilled, which is why the result carries two ranges rather than one.
//
// A span is a candidate only when all of the following hold:
//
//   * It is not the leading span, which is ordinary prefix reuse and would be counted twice.
//   * Its source range is fully inside `resident_kv_tokens`.
//   * Its destination begins at or after `prefix_reuse_tokens`.
//   * Its destination begins before `prompt_tokens - 1`, so the final prompt token is always
//     prefilled and the plan's `tail` range is never empty. A span reaching past that point is
//     trimmed back to the last whole block ending before it, rather than dropped, so a surviving
//     suffix that runs to the end of the prompt still relocates everything but its final block.
//   * Its destination ends at or before `turn_rewrite_boundary`, when one is supplied.
//
// The boundary is the first prompt token whose KV is stale because the assistant's reasoning is
// about to be rewritten there, so a relocated span may not carry a token at or after it. Every
// chat request carries a boundary, because the chat template always appends a generation prompt,
// which means the boundary sits at the end of the user content on a single-turn request. A span is
// trimmed back to the last whole block ending before the boundary rather than dropped, so the whole
// user content stays relocatable and only the region that genuinely has to be prefilled is left.
//
// Returns a non-viable plan when no candidate survives, when `max_spans` is zero, or when the prompt
// has nothing after the reuse frontier.
[[nodiscard]] TailRelocationPlan plan_tail_relocation(const PromptEditDiff& diff,
                                                      std::uint32_t resident_kv_tokens,
                                                      std::uint32_t prefix_reuse_tokens,
                                                      std::uint32_t prompt_tokens,
                                                      std::uint32_t max_spans,
                                                      std::optional<std::uint32_t> turn_rewrite_boundary =
                                                          std::nullopt,
                                                      TailRelocationTrace* trace = nullptr);

// Sentinel for a logical page with no resident physical page, so it has to come from the pool when a
// remap is applied. The value matches PagedKVAllocation's acquire sentinel; the runtime static_asserts
// that, because this header is deliberately host-buildable and cannot include the paged-cache header.
inline constexpr std::int32_t kRemapAcquirePage = -1;

// Build the logical-to-physical page mapping for a tail relocation.
//
// `current_pages` is the allocation's current mapping, one physical page per logical page of the
// resident prompt: page `i` holds tokens `[64i, 64i + 64)`. The result has one entry per logical page
// of the new prompt, either the physical page that should hold it or `kRemapAcquirePage` for a page
// that has no content to keep and that prefill will write into.
//
// Pages are only carried over where their content must survive, because the remap never copies data:
//
//   * every relocated block keeps the physical page that already holds it; and
//   * the reused prefix keeps its own pages, including the partially reused page straddling the
//     frontier, whose tail prefill overwrites in place.
//
// Everything else is left to acquire. The pages dropped from the replaced region are exactly what pays
// for those slots, through the surplus reuse in PagedKVAllocation::reassign_pages.
//
// Returns an empty vector when the relocation cannot be expressed, which the caller must treat as a
// fall back to ordinary prefix reuse:
//
//   * a relocated span is not 64-token aligned in both directions,
//   * a relocated block's source page does not exist, or its destination falls outside the prompt,
//   * the reused prefix reaches past the current mapping, or
//   * the mapping would name one physical page twice. Reassigning would reject that anyway, and the
//     overlap means a surviving region and the prefix disagree about which page holds it.
[[nodiscard]] std::vector<std::int32_t> plan_page_remap(std::span<const std::int32_t> current_pages,
                                                       std::span<const RelocatedSpan> spans,
                                                       std::uint32_t prefix_reuse_tokens,
                                                       std::uint32_t prompt_tokens);

// Build the page mapping for a *copied* relocation: the same shape as `plan_page_remap`, except that
// every destination page a span covers is left to acquire rather than pointed at the page holding
// its content.
//
// That is the whole difference, and it follows from the payload moving: with an off-grid source no
// destination page boundary lines up with a source page boundary, so a re-pointed page would hold
// the wrong tokens and the content has to be written into a fresh page instead. The sources are not
// named by the result, which is what drops them -- and why their physical pages must be kept out of
// the acquire substitution until the copy has run (`shifted_relocation_pages_from_pool` prices that).
//
// Returns an empty vector when the relocation cannot be expressed: a destination that is not a whole
// number of pages, a span reaching outside the prompt, or a prefix reaching past the current
// mapping. The caller treats that as a fall back to ordinary prefix reuse, as with `plan_page_remap`.
[[nodiscard]] std::vector<std::int32_t> plan_copied_page_remap(
    std::span<const std::int32_t> current_pages, std::span<const RelocatedSpan> spans,
    std::uint32_t prefix_reuse_tokens, std::uint32_t prompt_tokens);

// The pages `PagedKVAllocation::reassign_pages` would have to take from the pool rather than from
// the mapping's own dropped pages, for a mapping `plan_page_remap` built over `current_pages`.
//
// The allocation's documented arithmetic: an acquire slot is filled from a page the mapping drops
// before the pool is asked for anything, so `from_pool = acquires - min(acquires, surplus)`, with
// the surplus the currently owned pages the mapping does not name. Re-deriving it here is what lets
// a caller price *both* allocations of a paired relocation before publishing either: the Text
// reassign would otherwise succeed and the second one fail with std::bad_alloc after it, which
// leaves the two caches disagreeing about which physical page holds a logical one -- the split the
// pairing exists to prevent.
//
// `current_pages` is the allocation's present mapping, one physical page per logical page it maps
// now; `mapping` is the proposed one. Both are exactly what `plan_page_remap` takes and returns, so
// the result is the pool demand of applying that mapping: the free pages the pool must report, or
// the reassign is refused.
[[nodiscard]] std::uint32_t remap_pages_from_pool(std::span<const std::int32_t> current_pages,
                                                  std::span<const std::int32_t> mapping);

// The same price for a *copied* relocation, where the pages the copies read from must not be handed
// back as the destinations they are copied into.
//
// `reassign_pages` fills acquire slots from the pages the mapping drops before it asks the pool for
// anything, and that is exactly the aliasing this exists to prevent: if a destination page were the
// same physical page as one of a copy's sources, the copy would read and write overlapping ranges of
// one page, which the device copy APIs do not define. Counting a source occurrence as unusable
// rather than free is the whole difference; the result is the pool demand of applying the mapping,
// larger than `remap_pages_from_pool` reports, and it is the number the planner has to afford before
// anything is published -- the same price-first discipline the re-pointing path already follows.
//
// `source_pages` are *logical* pages of `current_pages` that the relocation reads, resolved against
// the current mapping, so a logical page the mapping does not name contributes nothing.
[[nodiscard]] std::uint32_t shifted_relocation_pages_from_pool(
    std::span<const std::int32_t> current_pages, std::span<const std::int32_t> mapping,
    std::span<const std::uint32_t> source_pages);

// The ranges of the prompt the runtime must prefill for `spans`, in prompt order: every token of
// `[prefix_reuse_tokens, prompt_tokens)` that no relocated span covers. With one contiguous run this
// is the head and the tail around it, which is exactly what `plan_tail_relocation` reports; with
// interleaved survivors it is one range per gap. Empty ranges are never returned, and together
// with `spans` the result tiles the whole suffix exactly -- so a token is prefilled or relocated
// but never unaccounted. That tiling is what keeps `summary.relocated_prompt_tokens` and the
// service-work projection honest, and why the final prompt token always lands inside a range: the
// planner forbids a span from reaching it.
//
// With no spans this is the single range `[prefix_reuse_tokens, prompt_tokens)`, the ordinary
// "prefill everything after the frontier" shape, so a caller that has one always has a plan.
[[nodiscard]] std::vector<PrefillRange>
relocation_prefill_ranges(std::span<const RelocatedSpan> spans, std::uint32_t prefix_reuse_tokens,
                          std::uint32_t prompt_tokens);

// The extra service-work quanta a multi-range prefill costs. The projection accounts for one
// range's worth of partial chunks, and every further range adds a whole step of its own, because a
// range's last partial chunk is a step the single-span formula does not count. Under-projecting is
// not cosmetic: a request whose real work exceeds its projection dies mid-prefill, which is how
// the two-range case was found.
[[nodiscard]] std::uint64_t relocation_prefill_splits(std::span<const PrefillRange> ranges);

// Whether a relocation plan is expressible at all. It is when every span is block aligned in both
// prompts, starts at or after the prefix frontier, ends before the final prompt token, and the
// reported ranges tile `[prefix_reuse_tokens, prompt_tokens)` together with the spans. A plan that
// fails any of this cannot be executed, so the planner must fall back to ordinary prefix reuse
// rather than report reuse it will not perform -- the failure this guards against is a summary that
// counts relocated tokens the runtime then re-prefills, which under-projects the work and reports
// reuse that never happened.
// `source_may_shift` selects which shape the spans may have: false is the re-pointing path, whose
// sources name pages that already hold the content and must therefore be whole pages; true is the
// copying path, where the source is a token offset like any other. Destinations are whole pages on
// both, because that is what a page table and a per-page delta can express.
[[nodiscard]] bool relocation_is_expressible(std::span<const RelocatedSpan> spans,
                                            std::span<const PrefillRange> ranges,
                                            std::uint32_t prefix_reuse_tokens,
                                            std::uint32_t prompt_tokens,
                                            bool source_may_shift = false);

// Detect the spans of `previous` that survive in `current`.
//
// Blocks are hashed and matched by hash, then confirmed by exact token comparison, so a
// hash collision can only cause a missed span, never a false match. Spans are chosen by
// a single monotone left-to-right pass that never reuses an old block, which keeps the
// result disjoint and the walk linear.
//
// Trailing tokens that do not fill a whole block are never reported as surviving.
// Matching blocks are the caller's concern: this function ignores token types,
// positions, and vision metadata, so a caller that reuses KV must still confirm its own
// identity constraints before relocating anything.
//
// Throws std::invalid_argument when block_tokens is zero, and std::overflow_error when
// the token counts do not fit uint32.
[[nodiscard]] PromptEditDiff diff_prompt_edit(std::span<const TokenId> previous,
                                              std::span<const TokenId> current,
                                              std::uint32_t block_tokens =
                                                  kPromptEditBlockTokens);

// Detect the spans of `previous` that survive in `current` at *any* offset.
//
// `diff_prompt_edit` matches whole blocks, so it can only find content that sits on the same 64-token
// grid in both prompts: an edit that changes the token count by anything other than a multiple of a
// page shifts every later block off its boundary and leaves it nothing to match. That is the shape
// of nearly every edit a user types, and why the aligned path relocates nothing on them.
//
// This anchors on a short n-gram taken at every offset instead, grows each hit to the run's true
// boundaries, and confirms it by exact token comparison, so a hash collision can only lose a match.
// The walk is monotone and never reuses a previous token, which keeps the runs disjoint in both
// prompts; a cap on how many candidate offsets one anchor may probe bounds the cost on prompts that
// repeat a phrase many times. Because only runs of at least `anchor` tokens can be anchored at all,
// runs shorter than that are not reported -- another cost of the finer granularity, and bounded by
// the anchor size rather than by the page.
//
// Same contract as `diff_prompt_edit` otherwise: token types, positions and vision metadata are the
// caller's concern, and a run touching the end of the prompt is reported as the tokens it holds.
[[nodiscard]] PromptEditDiff diff_prompt_edit_shifted(std::span<const TokenId> previous,
                                                      std::span<const TokenId> current,
                                                      std::uint32_t anchor_tokens = 16);

} // namespace ninfer::targets::qwen3_6::detail
