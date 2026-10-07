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
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

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
    // Run length in tokens. Always a positive multiple of the block size.
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
// prompt. Both ranges are block aligned and disjoint.
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
// `docs/SCR_RELOCATION_PATHS.md` as Path B, which is not implemented. The runtime currently plans
// with `plan_tail_relocation` instead, because it can only prefill a bounded number of ranges.
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
//
// Of the survivors the `max_spans` longest are kept and the rest are left to be re-prefilled, which
// bounds how much approximation one edit introduces. The result is returned in prompt order.
//
// A `max_spans` of zero returns no spans.
[[nodiscard]] std::vector<RelocatedSpan> plan_relocations(const PromptEditDiff& diff,
                                                         std::uint32_t resident_kv_tokens,
                                                         std::uint32_t prefix_reuse_tokens,
                                                         std::uint32_t max_spans);

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
                                                          std::nullopt);

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

} // namespace ninfer::targets::qwen3_6::detail
