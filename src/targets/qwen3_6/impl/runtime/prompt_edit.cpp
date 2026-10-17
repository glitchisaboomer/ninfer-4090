#include "targets/qwen3_6/impl/runtime/prompt_edit.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {
namespace {

// FNV-1a over the block's tokens. Any stable hash works; this one is cheap for the
// short blocks used here and needs no allocation.
constexpr std::uint64_t kHashOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kHashPrime  = 0x100000001b3ULL;

// Upper bound on hash candidates examined for one block of the current prompt. Prompt
// content that repeats a block many times can exceed this, and any surviving span we
// skip only costs an ordinary re-prefill, so the cap trades a bounded amount of reuse
// for a hard O(N) walk.
constexpr std::size_t kMaxCandidateProbes = 8;

[[nodiscard]] std::uint64_t hash_block(std::span<const TokenId> tokens, std::uint32_t first_block,
                                       std::uint32_t block_tokens) noexcept {
    std::uint64_t hash = kHashOffset;
    const std::uint32_t begin = first_block * block_tokens;
    for (std::uint32_t i = 0; i < block_tokens; ++i) {
        // Token ids are non-negative and fit int32; fold without sign extension so two
        // different ids never hash to the same value.
        hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[begin + i]));
        hash *= kHashPrime;
    }
    return hash;
}

// The same fold over an arbitrary window, for the shifted detector: anchoring on a short n-gram
// taken at every offset is what lets it match a run that is not on the block grid.
[[nodiscard]] std::uint64_t hash_tokens(std::span<const TokenId> tokens, std::uint32_t offset,
                                       std::uint32_t count) noexcept {
    std::uint64_t hash = kHashOffset;
    for (std::uint32_t i = 0; i < count; ++i) {
        hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[offset + i]));
        hash *= kHashPrime;
    }
    return hash;
}

[[nodiscard]] bool tokens_equal(std::span<const TokenId> left, std::uint32_t left_begin,
                                std::span<const TokenId> right, std::uint32_t right_begin,
                                std::uint32_t count) noexcept {
    for (std::uint32_t i = 0; i < count; ++i) {
        if (left[left_begin + i] != right[right_begin + i]) { return false; }
    }
    return true;
}

[[nodiscard]] bool blocks_equal(std::span<const TokenId> left, std::uint32_t left_block,
                                std::span<const TokenId> right, std::uint32_t right_block,
                                std::uint32_t block_tokens) noexcept {
    const std::uint32_t left_begin   = left_block * block_tokens;
    const std::uint32_t right_begin  = right_block * block_tokens;
    for (std::uint32_t i = 0; i < block_tokens; ++i) {
        if (left[left_begin + i] != right[right_begin + i]) { return false; }
    }
    return true;
}

[[nodiscard]] std::uint32_t checked_block_count(std::size_t tokens, std::uint32_t block_tokens,
                                                const char* label) {
    if (tokens > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit uint32 token indices");
    }
    return static_cast<std::uint32_t>(tokens) / block_tokens;
}

} // namespace

std::uint32_t relocation_limit(std::uint32_t prompt_tokens,
                               std::optional<std::uint32_t> turn_rewrite_boundary) {
    std::uint32_t limit = prompt_tokens > kMinimumForwardedTailTokens
                              ? prompt_tokens - kMinimumForwardedTailTokens
                              : 0;
    if (turn_rewrite_boundary && *turn_rewrite_boundary < limit) { limit = *turn_rewrite_boundary; }
    return limit;
}

void trim_relocated_run_tails(std::vector<RelocatedSpan>& spans, std::uint32_t tokens) {
    if (spans.empty() || tokens == 0) { return; }
    std::sort(spans.begin(), spans.end(), [](const RelocatedSpan& a, const RelocatedSpan& b) {
        return a.dest_begin < b.dest_begin;
    });

    std::vector<RelocatedSpan> kept;
    kept.reserve(spans.size());
    for (std::size_t i = 0; i < spans.size(); ++i) {
        // Only the last span of a run ends where the prompts diverge. Spans whose destinations touch
        // are one run, so the spans before that one have a trailing edge the edit did not replace and
        // nothing to give back.
        const bool ends_run =
            (i + 1 == spans.size()) || (spans[i].dest_end() != spans[i + 1].dest_begin);
        RelocatedSpan span = spans[i];
        if (ends_run) {
            if (span.tokens <= tokens) { continue; }
            span.tokens -= tokens;
        }
        kept.push_back(span);
    }
    spans = std::move(kept);
}

std::vector<RelocatedSpan> plan_relocations(const PromptEditDiff& diff,
                                           std::uint32_t resident_kv_tokens,
                                           std::uint32_t prefix_reuse_tokens,
                                           std::uint32_t prompt_tokens,
                                           std::optional<std::uint32_t> turn_rewrite_boundary,
                                           std::uint32_t max_spans) {
    std::vector<RelocatedSpan> candidates;
    if (max_spans == 0 || prompt_tokens == 0 || prompt_tokens <= prefix_reuse_tokens) {
        return candidates;
    }

    // The same limit the tail planner applies, for the same two reasons: the tail after the last
    // run is always prefilled, and the turn-rewrite boundary is the first token whose KV is stale.
    const std::uint32_t limit = relocation_limit(prompt_tokens, turn_rewrite_boundary);

    for (const SurvivingSpan& span : diff.spans) {
        // The leading span is ordinary prefix reuse, already expressed by the planner's
        // reuse base; relocating it would account for the same tokens twice.
        if (span.old_begin == 0 && span.new_begin == 0) { continue; }
        // Relocation needs the resident KV entry to still exist.
        if (span.old_end() > resident_kv_tokens) { continue; }
        // A destination at or below the prefix-reuse frontier is already covered by ordinary
        // prefix reuse. A span that starts below it and ends above it is dropped whole, because
        // splitting it would leave the reused part of the destination claimed by two paths.
        if (span.new_begin < prefix_reuse_tokens) { continue; }
        if (span.tokens == 0 || span.new_begin >= limit) { continue; }
        // Trimmed back to the last whole block ending before the limit. Trimming keeps the span
        // block aligned, and surviving spans are disjoint, so it cannot break a run's adjacency.
        const std::uint32_t room   = limit - span.new_begin;
        const std::uint32_t fit    = (room / kPromptEditBlockTokens) * kPromptEditBlockTokens;
        const std::uint32_t tokens = std::min(span.tokens, fit);
        if (tokens == 0) { continue; }
        candidates.push_back(RelocatedSpan{.source_begin = span.old_begin,
                                           .dest_begin   = span.new_begin,
                                           .tokens       = tokens});
    }

    // Longest survivors first so the cap discards the least useful reuse. Trimming happened above,
    // so a span shortened by a limit is compared at the length it would actually relocate.
    const std::size_t keep = std::min<std::size_t>(max_spans, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates.end(), [](const RelocatedSpan& a, const RelocatedSpan& b) {
                          return a.tokens > b.tokens;
                      });
    candidates.resize(keep);

    // The runtime consumes them in prompt order.
    std::sort(candidates.begin(), candidates.end(),
              [](const RelocatedSpan& a, const RelocatedSpan& b) {
                  return a.dest_begin < b.dest_begin;
              });
    return candidates;
}

TailRelocationPlan plan_tail_relocation(const PromptEditDiff& diff, std::uint32_t resident_kv_tokens,
                                       std::uint32_t prefix_reuse_tokens,
                                       std::uint32_t prompt_tokens, std::uint32_t max_spans,
                                       std::optional<std::uint32_t> turn_rewrite_boundary,
                                       TailRelocationTrace* trace) {
    TailRelocationPlan plan;
    if (trace != nullptr) {
        *trace = TailRelocationTrace{};
        trace->prompt_tokens       = prompt_tokens;
        trace->prefix_reuse_tokens = prefix_reuse_tokens;
        trace->resident_kv_tokens  = resident_kv_tokens;
        trace->max_spans           = max_spans;
    }
    if (max_spans == 0 || prompt_tokens == 0 || prompt_tokens <= prefix_reuse_tokens) {
        if (trace != nullptr) {
            trace->early_exit = (max_spans == 0)       ? "max_spans == 0"
                                : (prompt_tokens == 0) ? "empty prompt"
                                                      : "prompt is not longer than the prefix frontier";
        }
        return plan;
    }

    // No span may reach this token, because nothing at or after it can be served from resident KV.
    // The limit keeps the tail after the last run prefilled, and the turn-rewrite boundary, when
    // supplied, cuts it earlier still: the boundary is where the assistant's reasoning is rewritten
    // and every token before it stays valid.
    const std::uint32_t limit = relocation_limit(prompt_tokens, turn_rewrite_boundary);
    if (trace != nullptr) {
        trace->limit             = limit;
        trace->boundary_supplied = turn_rewrite_boundary.has_value();
        trace->spans_seen        = diff.spans.size();
    }

    std::vector<RelocatedSpan> candidates;
    for (const SurvivingSpan& span : diff.spans) {
        const TailRelocationReject observed{.old_begin = span.old_begin,
                                            .old_end   = span.old_end(),
                                            .new_begin = span.new_begin,
                                            .tokens    = span.tokens};
        const auto drop = [&](const char* reason) {
            if (trace != nullptr) {
                TailRelocationReject rejected = observed;
                rejected.reason              = reason;
                trace->rejected.push_back(rejected);
            }
        };
        // The leading span is ordinary prefix reuse, already expressed by the planner's reuse base.
        if (span.old_begin == 0 && span.new_begin == 0) { drop("leading span is prefix reuse"); continue; }
        // Relocation needs the resident KV entry to still exist.
        if (span.old_end() > resident_kv_tokens) {
            drop("source beyond the resident KV");
            continue;
        }
        // A destination at or below the prefix-reuse frontier is already covered by prefix reuse.
        if (span.new_begin < prefix_reuse_tokens) { drop("destination below the prefix frontier"); continue; }
        if (span.tokens == 0) { drop("empty span"); continue; }
        if (span.new_begin >= limit) {
            drop("destination at or beyond the rewrite boundary");
            continue;
        }
        // The span is trimmed back to the last whole block ending before the limit. Trimming keeps
        // the span block aligned, and only one span can be affected because surviving spans are
        // disjoint, so it cannot break a run's adjacency.
        const std::uint32_t room   = limit - span.new_begin;
        const std::uint32_t fit    = (room / kPromptEditBlockTokens) * kPromptEditBlockTokens;
        const std::uint32_t tokens = std::min(span.tokens, fit);
        if (tokens == 0) { drop("no whole block fits before the rewrite boundary"); continue; }
        if (tokens < span.tokens) { drop("trimmed to whole blocks before the rewrite boundary"); }
        candidates.push_back(RelocatedSpan{.source_begin = span.old_begin,
                                           .dest_begin   = span.new_begin,
                                           .tokens       = tokens});
    }
    if (trace != nullptr) { trace->candidates = candidates; }
    if (candidates.empty()) {
        if (trace != nullptr) { trace->declined = "no surviving span survived the limits"; }
        return plan;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const RelocatedSpan& a, const RelocatedSpan& b) {
                  return a.dest_begin < b.dest_begin;
              });

    // Only a contiguous run can be relocated without prefilling a range between survivors, so take
    // the most valuable fixed-size window across every maximal run. Ties keep the earliest run.
    const std::size_t cap = std::min<std::size_t>(max_spans, candidates.size());
    std::vector<RelocatedSpan> best;
    std::uint32_t best_total = 0;

    std::size_t run_begin = 0;
    while (run_begin < candidates.size()) {
        std::size_t run_end = run_begin + 1;
        while (run_end < candidates.size() &&
               candidates[run_end].dest_begin == candidates[run_end - 1].dest_end()) {
            ++run_end;
        }
        const std::size_t run_length = run_end - run_begin;
        const std::size_t window     = std::min(cap, run_length);

        std::uint32_t sum = 0;
        for (std::size_t i = 0; i < window; ++i) { sum += candidates[run_begin + i].tokens; }
        std::uint32_t best_sum = sum;
        std::size_t best_offset = 0;
        for (std::size_t offset = 1; offset + window <= run_length; ++offset) {
            sum -= candidates[run_begin + offset - 1].tokens;
            sum += candidates[run_begin + offset + window - 1].tokens;
            if (sum > best_sum) {
                best_sum    = sum;
                best_offset = offset;
            }
        }

        if (best_sum > best_total) {
            best_total = best_sum;
            const auto first = candidates.begin() + static_cast<std::ptrdiff_t>(run_begin + best_offset);
            best.assign(first, first + static_cast<std::ptrdiff_t>(window));
        }
        run_begin = run_end;
    }

    if (best.empty()) {
        if (trace != nullptr) { trace->declined = "candidates existed but no run was selected"; }
        return plan;
    }

    plan.viable = true;
    plan.head   = PrefillRange{.begin = prefix_reuse_tokens, .end = best.front().dest_begin};
    plan.spans  = std::move(best);
    plan.tail   = PrefillRange{.begin = plan.spans.back().dest_end(), .end = prompt_tokens};
    return plan;
}

std::vector<std::int32_t> plan_page_remap(std::span<const std::int32_t> current_pages,
                                         std::span<const RelocatedSpan> spans,
                                         std::uint32_t prefix_reuse_tokens,
                                         std::uint32_t prompt_tokens) {
    std::vector<std::int32_t> mapping;
    if (current_pages.empty() || spans.empty() || prompt_tokens == 0) { return mapping; }

    const std::uint32_t logical_pages =
        (prompt_tokens + kPromptEditBlockTokens - 1) / kPromptEditBlockTokens;
    mapping.assign(logical_pages, kRemapAcquirePage);

    // Relocated destinations claim their source page first, so the prefix pass below can see which
    // slots are already spoken for.
    std::vector<std::uint8_t> claimed(logical_pages, 0);
    for (const RelocatedSpan& span : spans) {
        if (span.tokens == 0 || span.tokens % kPromptEditBlockTokens != 0 ||
            span.source_begin % kPromptEditBlockTokens != 0 ||
            span.dest_begin % kPromptEditBlockTokens != 0) {
            return {};
        }
        for (std::uint32_t offset = 0; offset < span.tokens; offset += kPromptEditBlockTokens) {
            const std::uint32_t source_page = (span.source_begin + offset) / kPromptEditBlockTokens;
            const std::uint32_t dest_page   = (span.dest_begin + offset) / kPromptEditBlockTokens;
            if (source_page >= current_pages.size() || dest_page >= mapping.size()) { return {}; }
            mapping[dest_page] = current_pages[source_page];
            claimed[dest_page]  = 1;
        }
    }

    // The reused prefix keeps its own pages, including the one straddling the frontier.
    const std::uint32_t prefix_page = prefix_reuse_tokens / kPromptEditBlockTokens;
    if (prefix_page >= current_pages.size() || prefix_page >= mapping.size()) { return {}; }
    for (std::uint32_t page = 0; page <= prefix_page; ++page) {
        if (claimed[page] == 0) { mapping[page] = current_pages[page]; }
    }

    // One physical page must not be named twice. Reassigning rejects that, and it would mean the
    // prefix and a surviving region disagree about which page holds their content.
    std::vector<std::int32_t> carried;
    carried.reserve(mapping.size());
    for (const std::int32_t page : mapping) {
        if (page != kRemapAcquirePage) { carried.push_back(page); }
    }
    std::sort(carried.begin(), carried.end());
    if (std::adjacent_find(carried.begin(), carried.end()) != carried.end()) { return {}; }
    return mapping;
}

std::uint32_t remap_pages_from_pool(std::span<const std::int32_t> current_pages,
                                    std::span<const std::int32_t> mapping) {
    // Mirrors PagedKVAllocation::reassign_pages's own arithmetic, which this exists to price before
    // that call: acquire slots are filled from the pages the mapping drops before the pool is asked
    // for anything. A page the mapping keeps is consumed once, so an owned page named twice (which
    // reassign_pages rejects anyway) is not double counted here.
    std::uint32_t acquires = 0;
    for (const std::int32_t page : mapping) {
        if (page == kRemapAcquirePage) { ++acquires; }
    }
    const std::uint64_t named = static_cast<std::uint64_t>(mapping.size()) - acquires;
    const std::uint64_t surplus =
        static_cast<std::uint64_t>(current_pages.size()) > named
            ? static_cast<std::uint64_t>(current_pages.size()) - named
            : 0;
    return acquires - std::min<std::uint64_t>(acquires, surplus);
}

std::vector<std::int32_t> plan_copied_page_remap(std::span<const std::int32_t> current_pages,
                                                 std::span<const RelocatedSpan> spans,
                                                 std::uint32_t prefix_reuse_tokens,
                                                 std::uint32_t prompt_tokens) {
    std::vector<std::int32_t> mapping;
    if (current_pages.empty() || spans.empty() || prompt_tokens == 0) { return mapping; }

    const std::uint32_t logical_pages =
        (prompt_tokens + kPromptEditBlockTokens - 1) / kPromptEditBlockTokens;
    mapping.assign(logical_pages, kRemapAcquirePage);

    // Every destination page a span covers is acquired and never carried: its content arrives by a
    // copy rather than by being pointed at, and the page that holds that content today is one of the
    // copy's sources, which this mapping deliberately does not name.
    std::vector<std::uint8_t> copied(logical_pages, 0);
    for (const RelocatedSpan& span : spans) {
        if (span.tokens == 0 || span.dest_end() > prompt_tokens) { return {}; }
        const std::uint32_t first_page = span.dest_begin / kPromptEditBlockTokens;
        const std::uint32_t last_page = (span.dest_end() - 1) / kPromptEditBlockTokens;
        if (last_page >= logical_pages) { return {}; }
        for (std::uint32_t page = first_page; page <= last_page; ++page) {
            copied[page] = 1;
        }
    }

    // The reused prefix keeps its own pages, including the one straddling the frontier. A page a span
    // copies into is never carried back over that: the prefix pass would claim it twice.
    const std::uint32_t prefix_page = prefix_reuse_tokens / kPromptEditBlockTokens;
    if (prefix_page >= current_pages.size() || prefix_page >= mapping.size()) { return {}; }
    for (std::uint32_t page = 0; page <= prefix_page; ++page) {
        if (copied[page] == 0) { mapping[page] = current_pages[page]; }
    }

    // One physical page must not be named twice, for the same reason `plan_page_remap` checks it.
    std::vector<std::int32_t> carried;
    carried.reserve(mapping.size());
    for (const std::int32_t page : mapping) {
        if (page != kRemapAcquirePage) { carried.push_back(page); }
    }
    std::sort(carried.begin(), carried.end());
    if (std::adjacent_find(carried.begin(), carried.end()) != carried.end()) { return {}; }
    return mapping;
}

std::uint32_t shifted_relocation_pages_from_pool(std::span<const std::int32_t> current_pages,
                                                 std::span<const std::int32_t> mapping,
                                                 std::span<const std::uint32_t> source_pages) {
    std::uint32_t acquires = 0;
    for (const std::int32_t page : mapping) {
        if (page == kRemapAcquirePage) { ++acquires; }
    }
    if (acquires == 0) { return 0; }

    // Occurrences of the pages the mapping keeps; the rest of the ownership is the surplus the
    // acquire slots are otherwise filled from. Mirrors `reassign_pages`' own bookkeeping, by
    // occurrence, because a page named twice consumes two slots.
    std::unordered_map<std::int32_t, std::uint32_t> remaining;
    remaining.reserve(current_pages.size());
    for (const std::int32_t page : current_pages) { ++remaining[page]; }
    for (const std::int32_t page : mapping) {
        if (page == kRemapAcquirePage) { continue; }
        const auto found = remaining.find(page);
        if (found != remaining.end() && found->second != 0) {
            if (--found->second == 0) { remaining.erase(found); }
        }
    }

    // A page a copy reads is spoken for until the copy has moved its bytes, so it cannot serve as a
    // destination. Count those occurrences per physical page: the same page can back more than one
    // logical page on the source side of a relocation only if the mapping already named it twice,
    // which the expressibility rules reject.
    std::unordered_map<std::int32_t, std::uint32_t> blocked;
    for (const std::uint32_t logical : source_pages) {
        if (logical < current_pages.size()) { ++blocked[current_pages[logical]]; }
    }

    std::uint64_t usable = 0;
    for (const auto& entry : remaining) {
        const auto blocked_at = blocked.find(entry.first);
        const std::uint32_t unusable = blocked_at == blocked.end() ? 0u : blocked_at->second;
        usable += entry.second > unusable ? entry.second - unusable : 0u;
    }
    const std::uint64_t reused = std::min<std::uint64_t>(acquires, usable);
    return acquires - static_cast<std::uint32_t>(reused);
}

std::vector<PrefillRange> relocation_prefill_ranges(std::span<const RelocatedSpan> spans,
                                                    std::uint32_t prefix_reuse_tokens,
                                                    std::uint32_t prompt_tokens) {
    std::vector<PrefillRange> ranges;
    if (prompt_tokens <= prefix_reuse_tokens) { return ranges; }

    // Walk the spans in prompt order and emit the gaps before, between and after them. The planner
    // hands them over sorted by destination, but the max() keeps this correct for any order: an
    // overlapping or out-of-order span cannot manufacture a gap, it can only shrink one.
    std::uint32_t cursor = prefix_reuse_tokens;
    for (const RelocatedSpan& span : spans) {
        if (span.dest_begin > cursor) { ranges.push_back(PrefillRange{cursor, span.dest_begin}); }
        cursor = std::max(cursor, span.dest_end());
    }
    if (prompt_tokens > cursor) { ranges.push_back(PrefillRange{cursor, prompt_tokens}); }
    return ranges;
}

std::uint64_t relocation_prefill_splits(std::span<const PrefillRange> ranges) {
    // One range's partial chunk is already in the projection; every further range is a step the
    // single-span formula does not count.
    return ranges.size() > 1 ? static_cast<std::uint64_t>(ranges.size() - 1) : 0ULL;
}bool relocation_is_expressible(std::span<const RelocatedSpan> spans,
                              std::span<const PrefillRange> ranges,
                              std::uint32_t prefix_reuse_tokens,
                              std::uint32_t prompt_tokens,
                              bool source_may_shift) {
    if (prompt_tokens <= prefix_reuse_tokens || ranges.empty()) { return false; }

    std::vector<RelocatedSpan> sorted(spans.begin(), spans.end());
    std::sort(sorted.begin(), sorted.end(), [](const RelocatedSpan& a, const RelocatedSpan& b) {
        return a.dest_begin < b.dest_begin;
    });
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        const RelocatedSpan& span = sorted[i];
        if (span.tokens == 0) { return false; }
        // The re-pointing path moves the page table, so the source has to name a page that already
        // holds the content and must be a whole page. The copying path moves the payload, so its
        // source is just a token offset -- the freedom a shifted edit needs.
        if (!source_may_shift &&
            (span.source_begin % kPromptEditBlockTokens != 0 ||
             span.dest_begin % kPromptEditBlockTokens != 0 ||
             span.tokens % kPromptEditBlockTokens != 0)) {
            return false;
        }
        if (span.dest_begin < prefix_reuse_tokens || span.dest_end() > prompt_tokens - 1) {
            return false;
        }
        if (source_may_shift && prefix_reuse_tokens % kPromptEditBlockTokens != 0 &&
            span.dest_begin / kPromptEditBlockTokens ==
                prefix_reuse_tokens / kPromptEditBlockTokens) {
            return false;
        }
        if (i != 0 && span.dest_begin < sorted[i - 1].dest_end()) { return false; }
        if (source_may_shift && i != 0 &&
            span.dest_begin / kPromptEditBlockTokens <=
                (sorted[i - 1].dest_end() - 1) / kPromptEditBlockTokens) {
            return false;
        }
    }

    // Ranges and spans have to tile the suffix exactly: every token prefilled or relocated, none
    // twice, none missed. The walk takes whichever of the next range and the next span begins at the
    // cursor, so an interleaved plan -- range, run, gap, run, range -- tiles the same way a
    // contiguous one does.
    std::uint32_t cursor = prefix_reuse_tokens;
    std::size_t next_range = 0;
    std::size_t next_span  = 0;
    while (next_range < ranges.size() || next_span < sorted.size()) {
        const bool take_range =
            next_span >= sorted.size() ||
            (next_range < ranges.size() && ranges[next_range].begin <= sorted[next_span].dest_begin);
        if (take_range) {
            if (next_range >= ranges.size() || ranges[next_range].empty() ||
                ranges[next_range].begin != cursor) {
                return false;
            }
            cursor = ranges[next_range++].end;
        } else {
            if (sorted[next_span].dest_begin != cursor) { return false; }
            cursor = sorted[next_span++].dest_end();
        }
    }
    return cursor == prompt_tokens;
}

PromptEditDiff diff_prompt_edit(std::span<const TokenId> previous, std::span<const TokenId> current,
                                std::uint32_t block_tokens) {
    if (block_tokens == 0) {
        throw std::invalid_argument("prompt edit block size must be positive");
    }

    const std::uint32_t previous_blocks = checked_block_count(previous.size(), block_tokens,
                                                              "previous prompt");
    const std::uint32_t current_blocks =
        checked_block_count(current.size(), block_tokens, "current prompt");

    PromptEditDiff diff;
    if (previous_blocks == 0 || current_blocks == 0) { return diff; }

    // Index every whole block of the previous prompt by hash. Candidate lists stay in
    // ascending block order so the scan below always finds the earliest usable match.
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> index;
    index.reserve(previous_blocks);
    for (std::uint32_t block = 0; block < previous_blocks; ++block) {
        index[hash_block(previous, block, block_tokens)].push_back(block);
    }

    std::uint64_t reusable = 0;
    std::uint32_t old_cursor = 0;
    std::uint32_t new_block  = 0;
    while (new_block < current_blocks) {
        const auto found = index.find(hash_block(current, new_block, block_tokens));
        if (found == index.end()) {
            ++new_block;
            continue;
        }

        const std::vector<std::uint32_t>& candidates = found->second;
        std::size_t probes                           = 0;
        bool matched                                 = false;
        for (const std::uint32_t old_block : candidates) {
            // Spans never reuse an old block, which keeps both ranges disjoint.
            if (old_block < old_cursor) { continue; }
            if (probes == kMaxCandidateProbes) { break; }
            ++probes;
            if (!blocks_equal(previous, old_block, current, new_block, block_tokens)) { continue; }

            std::uint32_t run = 1;
            while (old_block + run < previous_blocks && new_block + run < current_blocks &&
                   blocks_equal(previous, old_block + run, current, new_block + run, block_tokens)) {
                ++run;
            }

            diff.spans.push_back(SurvivingSpan{.old_begin = old_block * block_tokens,
                                               .new_begin = new_block * block_tokens,
                                               .tokens    = run * block_tokens});
            reusable += static_cast<std::uint64_t>(run) * block_tokens;
            old_cursor = old_block + run;
            new_block += run;
            matched = true;
            break;
        }
        if (!matched) { ++new_block; }
    }

    if (reusable > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt edit reusable token count does not fit uint32");
    }
    diff.reusable_tokens = static_cast<std::uint32_t>(reusable);
    return diff;
}

PromptEditDiff diff_prompt_edit_shifted(std::span<const TokenId> previous,
                                        std::span<const TokenId> current,
                                        std::uint32_t anchor_tokens) {
    if (anchor_tokens == 0) {
        throw std::invalid_argument("prompt edit anchor size must be positive");
    }

    PromptEditDiff diff;
    if (previous.size() < anchor_tokens || current.size() < anchor_tokens) { return diff; }

    // Every offset, not every block: the whole point is to match content that moved by a count the
    // page grid cannot express.
    const std::uint32_t previous_anchors =
        static_cast<std::uint32_t>(previous.size()) - anchor_tokens + 1;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> index;
    index.reserve(previous_anchors);
    for (std::uint32_t offset = 0; offset < previous_anchors; ++offset) {
        index[hash_tokens(previous, offset, anchor_tokens)].push_back(offset);
    }

    std::uint64_t reusable   = 0;
    std::uint32_t old_cursor = 0;  // no run reuses a token of the previous prompt twice
    std::uint32_t new_cursor = 0;
    std::uint32_t at         = 0;
    const std::uint32_t current_anchors =
        static_cast<std::uint32_t>(current.size()) - anchor_tokens + 1;
    while (at < current_anchors) {
        const auto found = index.find(hash_tokens(current, at, anchor_tokens));
        if (found == index.end()) {
            ++at;
            continue;
        }

        bool matched    = false;
        std::size_t probes = 0;
        for (const std::uint32_t old_begin : found->second) {
            if (old_begin < old_cursor) { continue; }
            if (probes == kMaxCandidateProbes) { break; }
            ++probes;
            if (!tokens_equal(previous, old_begin, current, at, anchor_tokens)) { continue; }

            std::uint32_t right = anchor_tokens;
            while (old_begin + right < previous.size() && at + right < current.size() &&
                   previous[old_begin + right] == current[at + right]) {
                ++right;
            }
            // Grown left over the run's head, but never into the previous run or past the start of
            // either prompt: that is what keeps the runs disjoint and the walk monotone.
            std::uint32_t left = 0;
            while (old_begin > left && at > left && at - left - 1 >= new_cursor &&
                   old_begin - left - 1 >= old_cursor &&
                   previous[old_begin - left - 1] == current[at - left - 1]) {
                ++left;
            }

            const std::uint32_t begin_old = old_begin - left;
            const std::uint32_t begin_new = at - left;
            const std::uint32_t length    = left + right;
            diff.spans.push_back(
                SurvivingSpan{.old_begin = begin_old, .new_begin = begin_new, .tokens = length});
            reusable += length;
            old_cursor = begin_old + length;
            new_cursor = begin_new + length;
            at         = new_cursor;
            matched    = true;
            break;
        }
        if (!matched) { ++at; }
    }

    if (reusable > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt edit reusable token count does not fit uint32");
    }
    diff.reusable_tokens = static_cast<std::uint32_t>(reusable);
    return diff;
}

std::vector<RelocatedSpan> plan_shifted_relocations(const PromptEditDiff& diff,
                                                    std::uint32_t resident_kv_tokens,
                                                    std::uint32_t prefix_reuse_tokens,
                                                    std::uint32_t prompt_tokens,
                                                    std::optional<std::uint32_t> turn_rewrite_boundary,
                                                    std::uint32_t max_spans,
                                                    bool destination_may_shift) {
    std::vector<RelocatedSpan> candidates;
    if (max_spans == 0 || prompt_tokens == 0 || prompt_tokens <= prefix_reuse_tokens) {
        return candidates;
    }

    const std::uint32_t limit = relocation_limit(prompt_tokens, turn_rewrite_boundary);

    for (const SurvivingSpan& span : diff.spans) {
        if (span.old_begin == 0 && span.new_begin == 0) { continue; }
        if (span.old_end() > resident_kv_tokens) { continue; }
        if (span.new_begin < prefix_reuse_tokens) { continue; }
        if (span.tokens == 0) { continue; }

        const std::uint32_t reach = std::min(span.new_begin + span.tokens, limit);
        const std::uint32_t dest_begin =
            destination_may_shift
                ? span.new_begin
                : ((span.new_begin + kPromptEditBlockTokens - 1) / kPromptEditBlockTokens) *
                      kPromptEditBlockTokens;
        const std::uint32_t dest_end = destination_may_shift
                                           ? reach
                                           : (reach / kPromptEditBlockTokens) *
                                                 kPromptEditBlockTokens;
        if (dest_begin >= dest_end) { continue; }

        candidates.push_back(
            RelocatedSpan{.source_begin = span.old_begin + (dest_begin - span.new_begin),
                          .dest_begin   = dest_begin,
                          .tokens       = dest_end - dest_begin});
    }

    // The cap and the order match `plan_relocations`: longest survivors first, so the cap discards
    // the least useful reuse, then prompt order, which is how the runtime consumes them.
    const std::size_t keep = std::min<std::size_t>(max_spans, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates.end(), [](const RelocatedSpan& a, const RelocatedSpan& b) {
                          return a.tokens > b.tokens;
                      });
    candidates.resize(keep);
    std::sort(candidates.begin(), candidates.end(),
              [](const RelocatedSpan& a, const RelocatedSpan& b) {
                  return a.dest_begin < b.dest_begin;
              });
    return candidates;
}

std::uint32_t longest_shared_prefix_tokens(std::span<const TokenId> resident_tokens,
                                           std::span<const TokenId> prompt_tokens,
                                           std::uint32_t resident_kv_tokens,
                                           std::uint32_t limit) noexcept {
    const std::size_t reach = std::min(
        {resident_tokens.size(), prompt_tokens.size(), static_cast<std::size_t>(resident_kv_tokens),
         static_cast<std::size_t>(limit)});
    std::size_t shared = 0;
    while (shared < reach && resident_tokens[shared] == prompt_tokens[shared]) { ++shared; }
    // Whole blocks only, so every page the base retains is entirely valid rather than valid up to
    // its last tokens. A short trailing run is dropped rather than rounded up: rounding up would
    // claim KV the comparison never confirmed.
    return static_cast<std::uint32_t>((shared / kPromptEditBlockTokens) * kPromptEditBlockTokens);
}

} // namespace ninfer::targets::qwen3_6::detail
