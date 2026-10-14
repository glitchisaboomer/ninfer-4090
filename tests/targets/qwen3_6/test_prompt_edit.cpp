// Phase 1-2 SCR checks: surviving-span detection over an edited prompt.
//
// The detector is host-only and dependency-free, so this file also builds with a plain
// host compiler:
//   g++ -std=c++20 -I src -I include
//       src/targets/qwen3_6/impl/runtime/prompt_edit.cpp
//       tests/targets/qwen3_6/test_prompt_edit.cpp -o /tmp/test_prompt_edit

#include "targets/qwen3_6/impl/runtime/prompt_edit.h"

#include "core/paged_kv_delta_layout.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::targets::qwen3_6::detail::diff_prompt_edit;
using ninfer::targets::qwen3_6::detail::kPromptEditBlockTokens;
using ninfer::targets::qwen3_6::detail::PromptEditDiff;
using ninfer::targets::qwen3_6::detail::SurvivingSpan;

using Tokens = std::vector<TokenId>;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Callable>
void expect_throws_std_invalid_argument(Callable&& callable, std::string_view message) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception&) {
        expect(false, message);
        return;
    }
    expect(false, message);
}

// One block whose tokens are `value * 1000 + block_index`, so a segment is identified by
// its label and its blocks stay distinct from each other.
Tokens segment(std::int32_t label, std::uint32_t blocks) {
    Tokens out;
    out.reserve(static_cast<std::size_t>(blocks) * kPromptEditBlockTokens);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t i = 0; i < kPromptEditBlockTokens; ++i) {
            out.push_back(label * 1000 + static_cast<std::int32_t>(block));
        }
    }
    return out;
}

// `blocks` identical blocks, so the same block value appears more than once.
Tokens repeated(std::int32_t value, std::uint32_t blocks) {
    Tokens out;
    out.reserve(static_cast<std::size_t>(blocks) * kPromptEditBlockTokens);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t i = 0; i < kPromptEditBlockTokens; ++i) { out.push_back(value); }
    }
    return out;
}

void append(Tokens& out, const Tokens& part) {
    out.insert(out.end(), part.begin(), part.end());
}

std::uint32_t block_tokens(std::uint32_t blocks) {
    return blocks * kPromptEditBlockTokens;
}

bool spans_equal(const std::vector<SurvivingSpan>& left, const std::vector<SurvivingSpan>& right) {
    if (left.size() != right.size()) { return false; }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].old_begin != right[i].old_begin || left[i].new_begin != right[i].new_begin ||
            left[i].tokens != right[i].tokens) {
            return false;
        }
    }
    return true;
}

// Spans must be ordered and disjoint in both prompts. Every reuse path depends on this.
void expect_disjoint(const PromptEditDiff& diff, std::string_view message) {
    for (std::size_t i = 0; i < diff.spans.size(); ++i) {
        const SurvivingSpan& span = diff.spans[i];
        if (span.tokens == 0 || span.tokens % kPromptEditBlockTokens != 0) {
            expect(false, message);
            return;
        }
        if (i == 0) { continue; }
        const SurvivingSpan& previous = diff.spans[i - 1];
        if (span.old_begin < previous.old_end() || span.new_begin < previous.new_end()) {
            expect(false, message);
            return;
        }
    }
}

void test_identical_prompts() {
    const Tokens prompt = segment(1, 3);
    const PromptEditDiff diff = diff_prompt_edit(prompt, prompt);

    expect(diff.spans.size() == 1, "identical prompts produce one span");
    expect(diff.reusable_tokens == block_tokens(3), "identical prompts reuse every whole block");
    expect(diff.leading_prefix_tokens() == block_tokens(3), "identical prompts are all prefix");
    expect(diff.relocated_tokens() == 0, "identical prompts relocate nothing");
    expect_disjoint(diff, "identical prompt spans are disjoint");
}

void test_reasoning_stripping() {
    // Scope A shape: [P R S] -> [P S]. The stripped run is `R`; `S` survives and moves.
    Tokens previous;
    append(previous, segment(1, 2)); // prefix
    append(previous, segment(2, 2)); // stripped reasoning
    append(previous, segment(3, 2)); // preserved suffix

    Tokens current;
    append(current, segment(1, 2));
    append(current, segment(3, 2));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);

    const std::vector<SurvivingSpan> expected{
        SurvivingSpan{.old_begin = 0, .new_begin = 0, .tokens = block_tokens(2)},
        SurvivingSpan{.old_begin = block_tokens(4), .new_begin = block_tokens(2),
                      .tokens = block_tokens(2)},
    };
    expect(spans_equal(diff.spans, expected), "reasoning stripping yields prefix plus relocated suffix");
    expect(diff.reusable_tokens == block_tokens(4), "reasoning stripping reuses prefix and suffix");
    expect(diff.leading_prefix_tokens() == block_tokens(2), "reasoning stripping prefix length");
    expect(diff.relocated_tokens() == block_tokens(2), "reasoning stripping relocates the suffix");
    expect_disjoint(diff, "reasoning stripping spans are disjoint");
}

void test_in_place_replacement() {
    // Paper example: [A B C] -> [A B' C]. Only A and C survive.
    Tokens previous;
    append(previous, segment(1, 2));
    append(previous, segment(2, 1));
    append(previous, segment(3, 2));

    Tokens current;
    append(current, segment(1, 2));
    append(current, segment(9, 1));
    append(current, segment(3, 2));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);

    const std::vector<SurvivingSpan> expected{
        SurvivingSpan{.old_begin = 0, .new_begin = 0, .tokens = block_tokens(2)},
        SurvivingSpan{.old_begin = block_tokens(3), .new_begin = block_tokens(3),
                      .tokens = block_tokens(2)},
    };
    expect(spans_equal(diff.spans, expected), "in-place replacement keeps the surviving suffix at its old offset");
    expect(diff.reusable_tokens == block_tokens(4), "in-place replacement reuses both surviving regions");
    expect(diff.relocated_tokens() == block_tokens(2), "in-place replacement relocates the suffix");
}

void test_multiple_surviving_spans() {
    // Appendix B: one edit can leave several surviving spans after the edit point.
    Tokens previous;
    append(previous, segment(1, 1));
    append(previous, segment(2, 1));
    append(previous, segment(3, 1));
    append(previous, segment(4, 1));
    append(previous, segment(5, 1));

    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(8, 1));
    append(current, segment(3, 1));
    append(current, segment(9, 1));
    append(current, segment(5, 1));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);

    const std::vector<SurvivingSpan> expected{
        SurvivingSpan{.old_begin = 0, .new_begin = 0, .tokens = block_tokens(1)},
        SurvivingSpan{.old_begin = block_tokens(2), .new_begin = block_tokens(2),
                      .tokens = block_tokens(1)},
        SurvivingSpan{.old_begin = block_tokens(4), .new_begin = block_tokens(4),
                      .tokens = block_tokens(1)},
    };
    expect(spans_equal(diff.spans, expected), "three surviving spans are detected");
    expect(diff.reusable_tokens == block_tokens(3), "three surviving spans reuse three blocks");
    expect(diff.relocated_tokens() == block_tokens(2), "first span is the prefix, the rest are relocated");
    expect_disjoint(diff, "multiple spans are disjoint");
}

void test_block_split_forfeits_partial_block() {
    // A one-token insertion inside the second block misaligns the rest of that block. The
    // partial block is forfeited, which is the accepted cost of block-granular matching.
    Tokens previous = segment(1, 2);
    Tokens current  = segment(1, 2);
    current.insert(current.begin() + static_cast<std::ptrdiff_t>(kPromptEditBlockTokens + 3), 777);

    const PromptEditDiff diff = diff_prompt_edit(previous, current);

    expect(diff.spans.size() == 1, "block split leaves one aligned span");
    expect(diff.reusable_tokens == block_tokens(1), "block split forfeits the partial block");
    expect(diff.reusable_tokens % kPromptEditBlockTokens == 0, "reuse is block aligned");
    expect_disjoint(diff, "block split spans are disjoint");
}

void test_partial_tail_is_never_reused() {
    Tokens previous = segment(1, 1);
    Tokens current  = segment(1, 1);
    for (std::int32_t i = 0; i < 5; ++i) {
        previous.push_back(40 + i);
        current.push_back(40 + i);
    }

    const PromptEditDiff diff = diff_prompt_edit(previous, current);

    expect(diff.reusable_tokens == block_tokens(1), "trailing partial block is not reused");
    expect(diff.spans.size() == 1, "trailing partial block does not add a span");
}

void test_repeated_blocks_stay_monotone() {
    // Duplicate block values exercise the monotone walk: a block must never be reused
    // twice, and the shortened current prompt must not claim a third block.
    const Tokens previous = repeated(7, 3);
    const Tokens current  = repeated(7, 2);

    const PromptEditDiff diff = diff_prompt_edit(previous, current);

    expect(diff.spans.size() == 1, "repeated blocks collapse into one span");
    expect(diff.spans.front().old_begin == 0, "repeated blocks start at zero");
    expect(diff.spans.front().tokens == block_tokens(2), "repeated blocks stop at the shorter prompt");
    expect(diff.reusable_tokens == block_tokens(2), "repeated blocks never reuse a block twice");
    expect_disjoint(diff, "repeated block spans are disjoint");

    // The current prompt repeats the block more often than the previous prompt holds it.
    // The single old block may feed exactly one new position; the surplus must be treated
    // as new content instead of re-pointing at an already-used block.
    const Tokens narrow = repeated(7, 1);
    const Tokens wider  = repeated(7, 3);

    const PromptEditDiff grown = diff_prompt_edit(narrow, wider);

    expect(grown.spans.size() == 1, "a repeated block is reclaimed at most once");
    expect(grown.reusable_tokens == block_tokens(1), "surplus repeated blocks are not rewritten as reuse");
    if (!grown.spans.empty()) {
        expect(grown.spans.front().old_begin == 0 && grown.spans.front().new_begin == 0,
               "the reclaimed repeated block maps to the first new position");
    }
    expect_disjoint(grown, "grown repeated block spans are disjoint");
}

void test_empty_and_degenerate_inputs() {
    const Tokens prompt = segment(1, 2);

    expect(diff_prompt_edit({}, {}).spans.empty(), "empty prompts have no spans");
    expect(diff_prompt_edit(prompt, {}).spans.empty(), "empty current prompt has no spans");
    expect(diff_prompt_edit({}, prompt).spans.empty(), "empty previous prompt has no spans");

    const Tokens shorter_than_one_block{kPromptEditBlockTokens - 1, 1};
    expect(diff_prompt_edit(shorter_than_one_block, shorter_than_one_block).spans.empty(),
           "prompts shorter than one block have no spans");

    expect_throws_std_invalid_argument(
        [&] { (void)diff_prompt_edit(prompt, prompt, 0); },
        "zero block size must be rejected");
}

void test_plan_relocations() {
    using ninfer::targets::qwen3_6::detail::plan_relocations;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    // Scope A shape: [P R S] -> [P S]. The prefix span is ordinary prefix reuse, so only the
    // surviving suffix should be reported as a relocation. The trailing block is new content, so
    // the suffix does not run to the prompt end and the final-token limit does not trim it; that
    // rule is `test_plan_relocations_respects_limits`'s subject.
    Tokens previous;
    append(previous, segment(1, 2));
    append(previous, segment(2, 2));
    append(previous, segment(3, 2));
    Tokens current;
    append(current, segment(1, 2));
    append(current, segment(3, 2));
    append(current, segment(4, 1));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto prompt          = static_cast<std::uint32_t>(current.size());
    const auto relocations     = plan_relocations(diff, resident, 0, prompt, std::nullopt, 6);

    expect(relocations.size() == 1, "only the surviving suffix is relocated");
    if (relocations.size() == 1) {
        expect(relocations.front().source_begin == block_tokens(4), "relocation source is the old suffix");
        expect(relocations.front().dest_begin == block_tokens(2), "relocation destination is the new suffix");
        expect(relocations.front().tokens == block_tokens(2), "relocation length is the suffix length");
    }

    // Zero cap means reuse nothing, which is how SCR is disabled per request.
    expect(plan_relocations(diff, resident, 0, prompt, std::nullopt, 0).empty(),
           "a zero span cap relocates nothing");

    // A span without resident KV cannot be relocated.
    expect(plan_relocations(diff, block_tokens(5), 0, prompt, std::nullopt, 6).empty(),
           "a span outside resident KV is not relocated");

    // The surviving suffix moves from block 4 to block 2. When ordinary prefix reuse already
    // covers the first four blocks, the suffix starts at the frontier and is still relocated.
    expect(plan_relocations(diff, resident, block_tokens(2), prompt, std::nullopt, 6).size() == 1,
           "a survivor exactly at the prefix frontier is relocated");

    // Once prefix reuse reaches into the survivor's destination, that region is already covered
    // and must not be claimed a second time.
    expect(plan_relocations(diff, resident, block_tokens(3), prompt, std::nullopt, 6).empty(),
           "a survivor inside the prefix-reuse region is not relocated");
    expect(plan_relocations(diff, resident, block_tokens(4), prompt, std::nullopt, 6).empty(),
           "prefix reuse covering the whole survivor leaves nothing to relocate");
}

void test_plan_relocations_keeps_longest() {
    using ninfer::targets::qwen3_6::detail::plan_relocations;

    // Three surviving regions after the prefix, each separated by a replaced block so the
    // diff cannot merge them into one run. Lengths run 1, 2, 3 in prompt order, so the
    // longest survivor is last: keeping the longest is distinguishable from keeping the
    // first N in prompt order. The replacements differ in length from what they replace,
    // so every survivor also moves, and source and destination offsets stay distinct.
    Tokens previous;
    append(previous, segment(1, 1)); // prefix
    append(previous, segment(2, 1)); // replaced by three blocks
    append(previous, segment(6, 1)); // survivor, shortest, first in prompt order
    append(previous, segment(4, 2)); // replaced by one block
    append(previous, segment(5, 2)); // survivor, middle
    append(previous, segment(7, 1)); // replaced by two blocks
    append(previous, segment(3, 3)); // survivor, longest, last in prompt order

    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(9, 3));
    append(current, segment(6, 1));
    append(current, segment(8, 1));
    append(current, segment(5, 2));
    append(current, segment(8, 2));
    append(current, segment(3, 3));
    // New trailing content, so the longest survivor does not run to the prompt end and the
    // final-token limit does not shorten it into a tie with the middle one.
    append(current, segment(10, 1));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident = static_cast<std::uint32_t>(previous.size());
    const auto prompt   = static_cast<std::uint32_t>(current.size());

    // All three survivors plus the prefix are detected before capping.
    expect(diff.spans.size() == 4, "the fixture keeps its three survivors separate");
    expect(diff.relocated_tokens() == block_tokens(6),
           "the fixture relocates one, two, and three blocks");

    // The cap is smaller than the number of candidates; the dropped span is the one-block
    // survivor, not the earliest candidate.
    const auto capped = plan_relocations(diff, resident, 0, prompt, std::nullopt, 2);
    expect(capped.size() == 2, "the span cap is honored");
    if (capped.size() == 2) {
        expect(capped[0].tokens == block_tokens(2) && capped[1].tokens == block_tokens(3),
               "the cap keeps the longest survivors");
        expect(capped[0].dest_begin < capped[1].dest_begin, "relocations are returned in prompt order");
        expect(capped[0].source_begin == block_tokens(5) && capped[0].dest_begin == block_tokens(6),
               "a capped survivor reports its own source and destination");
    }

    const auto all = plan_relocations(diff, resident, 0, prompt, std::nullopt, 6);
    expect(all.size() == 3, "every surviving region is relocated when the cap allows");
    if (all.size() == 3) {
        expect(all[0].dest_begin < all[1].dest_begin && all[1].dest_begin < all[2].dest_begin,
               "uncapped relocations are in prompt order");
        expect(all[0].tokens == block_tokens(1) && all[1].tokens == block_tokens(2) &&
                   all[2].tokens == block_tokens(3),
               "prompt order is independent of survivor length");
        expect(all[0].source_begin == block_tokens(2) && all[0].dest_begin == block_tokens(4),
               "the shortest survivor moves forward");
        expect(all[2].source_begin == block_tokens(8) && all[2].dest_begin == block_tokens(10),
               "the longest survivor moves forward");
    }
}

void test_plan_relocations_respects_limits() {
    using ninfer::targets::qwen3_6::detail::plan_relocations;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    // Scope A shape: [P R S] -> [P S], with S three blocks long and ending at the prompt end.
    // previous is one block of P, one of R and three of S; current is P followed by S.
    Tokens previous;
    append(previous, segment(1, 1)); // P
    append(previous, segment(2, 1)); // R, replaced
    append(previous, segment(3, 3)); // S, survives and moves left, reaching the prompt end
    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(3, 3));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto prompt          = static_cast<std::uint32_t>(current.size());
    const auto unlimited       = plan_relocations(diff, resident, 0, prompt, std::nullopt, 6);
    expect(unlimited.size() == 1, "the survivor is selected without a boundary");
    if (unlimited.size() == 1) {
        // The final prompt token must be prefilled, so a run reaching the prompt end keeps only
        // the blocks that end before it. The tail planner trims the same shape the same way; the
        // two selectors must not disagree about which blocks are relocatable.
        expect(unlimited.front().tokens == block_tokens(2),
               "a run reaching the prompt end loses its final block to the sampler");
        expect(unlimited.front().dest_begin == block_tokens(1),
               "the trimmed run still starts at the frontier");
    }

    // A boundary inside the survivor trims it back to the last whole block ending before the
    // boundary, exactly as it does for the final prompt token.
    const auto inside = plan_relocations(diff, resident, 0, prompt, block_tokens(1), 6);
    expect(inside.empty(), "a boundary at the survivor's start leaves no whole block to relocate");
    const auto mid = plan_relocations(diff, resident, 0, prompt, block_tokens(2), 6);
    expect(mid.size() == 1 && mid.front().tokens == block_tokens(1),
           "a boundary after one block trims the run to that block");
    const auto off_edge =
        plan_relocations(diff, resident, 0, prompt, block_tokens(2) + 7, 6);
    expect(off_edge.size() == 1 && off_edge.front().tokens == block_tokens(1),
           "a boundary mid-block trims to that block's edge, not past it");
    const auto past_two = plan_relocations(diff, resident, 0, prompt, block_tokens(3), 6);
    expect(past_two.size() == 1 && past_two.front().tokens == block_tokens(2),
           "a boundary past two blocks trims the run to those two");

    // A boundary at or beyond the final prompt token is the same limit as no boundary at all, and
    // the plan must not change because the chat template happened to supply one.
    const auto at_end = plan_relocations(diff, resident, 0, prompt, prompt - 1, 6);
    expect(at_end.size() == 1 && at_end.front().tokens == block_tokens(2),
           "a boundary at the final token matches no boundary");
    const auto beyond_end = plan_relocations(diff, resident, 0, prompt, prompt, 6);
    expect(beyond_end.size() == 1 && beyond_end.front().tokens == block_tokens(2),
           "a boundary past the final token matches no boundary");
}

void test_relocation_prefill_ranges() {
    using ninfer::targets::qwen3_6::detail::PrefillRange;
    using ninfer::targets::qwen3_6::detail::relocation_prefill_ranges;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    const auto span = [](std::uint32_t dest, std::uint32_t tokens) {
        return RelocatedSpan{.source_begin = 0, .dest_begin = dest, .tokens = tokens};
    };
    const auto same = [](const std::vector<PrefillRange>& ranges, std::uint32_t count) {
        return ranges.size() == count;
    };

    // No spans: the whole suffix is prefilled, which is the ordinary no-relocation shape.
    const std::vector<RelocatedSpan> none;
    const auto single = relocation_prefill_ranges(none, block_tokens(1), block_tokens(4));
    expect(single.size() == 1 && single[0].begin == block_tokens(1) && single[0].end == block_tokens(4),
           "no spans prefills the whole suffix as one range");

    // One run in the middle: a head and a tail, which is what the tail planner reports.
    const std::vector<RelocatedSpan> middle{span(block_tokens(2), block_tokens(2))};
    const auto around = relocation_prefill_ranges(middle, block_tokens(1), block_tokens(5));
    expect(same(around, 2), "a middle run yields a head and a tail");
    if (same(around, 2)) {
        expect(around[0].begin == block_tokens(1) && around[0].end == block_tokens(2),
               "the head runs from the frontier to the run");
        expect(around[1].begin == block_tokens(4) && around[1].end == block_tokens(5),
               "the tail runs from the run to the prompt end");
    }

    // A run starting at the frontier has no head, which is the empty-head case the tail planner
    // produces and the runtime treats as "nothing deleted ahead of the run".
    const std::vector<RelocatedSpan> at_frontier{span(block_tokens(1), block_tokens(2))};
    const auto no_head = relocation_prefill_ranges(at_frontier, block_tokens(1), block_tokens(5));
    expect(same(no_head, 1) && no_head[0].begin == block_tokens(3),
           "a run at the frontier leaves no head range");

    // Two adjacent spans, which is one tail run expressed as several blocks: the gaps between them
    // must collapse, so the ranges stay head and tail rather than gaining an empty middle.
    const std::vector<RelocatedSpan> adjacent{span(block_tokens(2), block_tokens(1)),
                                              span(block_tokens(3), block_tokens(1))};
    const auto merged = relocation_prefill_ranges(adjacent, block_tokens(1), block_tokens(5));
    expect(same(merged, 2) && merged[1].begin == block_tokens(4),
           "adjacent runs collapse into the tail range");

    // Interleaved survivors: one range per gap, in prompt order. This is the shape only the
    // segmented planner produces, and the reason the range list exists.
    const std::vector<RelocatedSpan> interleaved{span(block_tokens(2), block_tokens(1)),
                                                 span(block_tokens(4), block_tokens(1))};
    const auto gaps = relocation_prefill_ranges(interleaved, block_tokens(1), block_tokens(6));
    expect(same(gaps, 3), "interleaved survivors leave one range per gap");
    if (same(gaps, 3)) {
        expect(gaps[0].end == block_tokens(2) && gaps[1].begin == block_tokens(3) &&
                   gaps[1].end == block_tokens(4) && gaps[2].begin == block_tokens(5),
               "the gaps sit between the survivors in prompt order");
    }

    // A frontier at the prompt end leaves nothing to prefill.
    expect(relocation_prefill_ranges(none, block_tokens(3), block_tokens(3)).empty(),
           "a frontier at the prompt end prefills nothing");
}

void test_relocation_prefill_splits() {
    using ninfer::targets::qwen3_6::detail::PrefillRange;
    using ninfer::targets::qwen3_6::detail::relocation_prefill_splits;

    const std::vector<PrefillRange> none;
    const std::vector<PrefillRange> one{PrefillRange{0, 64}};
    const std::vector<PrefillRange> two{PrefillRange{0, 64}, PrefillRange{128, 192}};
    const std::vector<PrefillRange> four{PrefillRange{0, 64},   PrefillRange{128, 192},
                                        PrefillRange{256, 320}, PrefillRange{384, 448}};

    // The projection already accounts for one range's partial chunks; every further range is a
    // whole extra step, and leaving one out is how a request dies one quantum short mid-prefill.
    expect(relocation_prefill_splits(none) == 0, "no ranges cost no extra split");
    expect(relocation_prefill_splits(one) == 0, "one range is already projected");
    expect(relocation_prefill_splits(two) == 1, "two ranges cost one extra split");
    expect(relocation_prefill_splits(four) == 3, "four ranges cost three extra splits");
}

void test_relocation_is_expressible() {
    using ninfer::targets::qwen3_6::detail::PrefillRange;
    using ninfer::targets::qwen3_6::detail::relocation_is_expressible;
    using ninfer::targets::qwen3_6::detail::relocation_prefill_ranges;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    const auto span = [](std::uint32_t dest, std::uint32_t tokens) {
        return RelocatedSpan{.source_begin = block_tokens(2), .dest_begin = dest, .tokens = tokens};
    };
    const auto check = [](const std::vector<RelocatedSpan>& spans, std::uint32_t base,
                          std::uint32_t prompt) {
        return relocation_is_expressible(spans,
                                         relocation_prefill_ranges(spans, base, prompt), base,
                                         prompt);
    };

    // The shapes the planners produce: a contiguous run, a run at the frontier, interleaved runs.
    expect(check({span(block_tokens(2), block_tokens(2))}, block_tokens(1), block_tokens(5)),
           "a contiguous run with a head and a tail is expressible");
    expect(check({span(block_tokens(1), block_tokens(2))}, block_tokens(1), block_tokens(5)),
           "a run at the frontier with only a tail is expressible");
    expect(check({span(block_tokens(2), block_tokens(1)), span(block_tokens(4), block_tokens(1))},
                 block_tokens(1), block_tokens(6)),
           "interleaved runs with one range per gap are expressible");

    // A span the plan does not cover: below the frontier, reaching the final token, misaligned, or
    // overlapping a neighbour. Each is a plan the runtime cannot execute, and the point of the
    // check is that it says so before the summary counts the tokens as reused.
    expect(!check({span(0, block_tokens(1))}, block_tokens(2), block_tokens(5)),
           "a span below the frontier is not expressible");
    expect(!check({span(block_tokens(4), block_tokens(1))}, block_tokens(1), block_tokens(5)),
           "a span reaching the final prompt token is not expressible");
    expect(!check({RelocatedSpan{.source_begin = 7, .dest_begin = block_tokens(2), .tokens = 64}},
                  block_tokens(1), block_tokens(5)),
           "a span that is not block aligned is not expressible");
    expect(!check({span(block_tokens(2), block_tokens(3))}, block_tokens(1), block_tokens(5)),
           "a span running into the next range is not expressible");
    expect(!check({span(block_tokens(2), block_tokens(1)), span(block_tokens(2), block_tokens(1))},
                  block_tokens(1), block_tokens(5)),
           "two spans claiming the same destination are not expressible");

    // Ranges that do not tile the suffix, whether by a gap or past its end.
    const std::vector<RelocatedSpan> run{span(block_tokens(2), block_tokens(2))};
    const std::vector<PrefillRange> gapped{PrefillRange{block_tokens(1), block_tokens(2)}};
    expect(!relocation_is_expressible(run, gapped, block_tokens(1), block_tokens(5)),
           "ranges that miss the tail are not expressible");
    const std::vector<PrefillRange> overrun{PrefillRange{block_tokens(1), block_tokens(2)},
                                            PrefillRange{block_tokens(4), block_tokens(9)}};
    expect(!relocation_is_expressible(run, overrun, block_tokens(1), block_tokens(5)),
           "ranges that run past the prompt are not expressible");
    expect(!relocation_is_expressible(run, {}, block_tokens(1), block_tokens(5)),
           "a relocation with no prefilled range is not expressible");
}

void test_plan_tail_relocation() {
    using ninfer::targets::qwen3_6::detail::plan_tail_relocation;

    // Scope A shape: [P R S] -> [P S]. S is the entire tail, so it is trimmed back to the last whole
    // block ending before the final prompt token, which is always prefilled.
    Tokens previous;
    append(previous, segment(1, 1)); // P
    append(previous, segment(2, 2)); // R, replaced
    append(previous, segment(3, 2)); // S, survives and moves left
    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(3, 2));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto prompt          = static_cast<std::uint32_t>(current.size());

    const auto plan = plan_tail_relocation(diff, resident, 0, prompt, 6);
    expect(plan.viable, "a Scope A edit yields a tail relocation");
    expect(plan.spans.size() == 1, "a Scope A edit relocates a single run");
    if (plan.spans.size() == 1) {
        expect(plan.spans.front().source_begin == block_tokens(3),
               "the relocation source is the old suffix");
        expect(plan.spans.front().dest_begin == block_tokens(1),
               "the relocation destination is the new suffix");
        expect(plan.spans.front().tokens == block_tokens(1),
               "the final block is trimmed and stays prefilled");
    }
    expect(plan.head.begin == 0 && plan.head.end == block_tokens(1), "the head covers the prefix");
    expect(plan.tail.begin == block_tokens(2) && plan.tail.end == prompt,
           "the tail covers the final block");
    expect(!plan.tail.empty(), "the plan's tail is never empty");
    expect(plan.relocated_tokens() == block_tokens(1), "the relocated token count is reported");

    // A reuse frontier that lands exactly on the destination leaves an empty head.
    const auto at_frontier = plan_tail_relocation(diff, resident, block_tokens(1), prompt, 6);
    expect(at_frontier.viable, "a survivor exactly at the reuse frontier still relocates");
    expect(at_frontier.head.empty(), "an empty head means nothing is prefilled before the run");

    // A destination already covered by prefix reuse must not be claimed a second time.
    expect(!plan_tail_relocation(diff, resident, block_tokens(2), prompt, 6).viable,
           "a destination inside the prefix-reuse region is not relocated");

    // Caps and degenerate prompts.
    expect(!plan_tail_relocation(diff, resident, 0, prompt, 0).viable,
           "a zero span cap relocates nothing");
    expect(!plan_tail_relocation(diff, resident, prompt, prompt, 6).viable,
           "a prompt that is entirely prefix reuse relocates nothing");
    expect(!plan_tail_relocation(diff, block_tokens(4), 0, prompt, 6).viable,
           "a survivor without resident KV is not relocated");
}

// Every chat request carries a turn-rewrite boundary, because the chat template always appends a
// generation prompt. A boundary sitting at the end of the user content must therefore not disable
// relocation; it must only bound how far a span may reach.
void test_plan_tail_relocation_respects_turn_rewrite_boundary() {
    using ninfer::targets::qwen3_6::detail::plan_tail_relocation;

    // [P R S S S] -> [P S S S]: the middle is deleted and three blocks survive to the end.
    Tokens previous;
    append(previous, segment(1, 1)); // P
    append(previous, segment(2, 2)); // R, deleted by the edit
    append(previous, segment(3, 3)); // S, survives
    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(3, 3));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto prompt          = static_cast<std::uint32_t>(current.size());

    // Without a boundary the survivor is trimmed back from the final prompt token, which is the
    // original behaviour.
    const auto unbounded = plan_tail_relocation(diff, resident, 0, prompt, 6, std::nullopt);
    expect(unbounded.viable, "no boundary relocates as before");
    expect(unbounded.spans.front().tokens == block_tokens(2),
           "without a boundary the final block is trimmed off");
    expect(unbounded.spans.back().dest_end() == block_tokens(3),
           "without a boundary the span stops at the last whole block before the final token");

    // A boundary past the survivor leaves it untouched. This is the single-turn chat case: the
    // boundary is the generation prompt at the very end of the user content.
    const auto after = plan_tail_relocation(diff, resident, 0, prompt, 6, prompt);
    expect(after.viable, "a boundary past the survivor still relocates");
    expect(after.spans.size() == 1 && after.spans.front().tokens == block_tokens(2),
           "a boundary past the survivor leaves its extent unchanged");

    // A boundary before the survivor truncates it to whole blocks ending before the boundary. The
    // survivor starts at block 1, so a boundary at block 2 leaves exactly one whole block.
    const auto before = plan_tail_relocation(diff, resident, 0, prompt, 6, block_tokens(2));
    expect(before.viable, "a boundary inside the survivor still relocates what precedes it");
    expect(before.spans.front().tokens == block_tokens(1),
           "a span is trimmed back to a whole block ending before the boundary");
    expect(before.spans.front().dest_end() <= block_tokens(2),
           "a truncated span never reaches past the boundary");
    expect(before.tail.begin == before.spans.back().dest_end() && before.tail.end == prompt,
           "the prefilled tail covers the boundary through the end of the prompt");

    // A boundary at the survivor's first token leaves nothing to relocate.
    expect(!plan_tail_relocation(diff, resident, 0, prompt, 6, block_tokens(1)).viable,
           "a boundary at the start of the survivor relocates nothing");
    expect(!plan_tail_relocation(diff, resident, 0, prompt, 6, 0).viable,
           "a zero boundary relocates nothing");
}

void test_plan_tail_relocation_picks_one_run() {
    using ninfer::targets::qwen3_6::detail::plan_tail_relocation;

    // Two survivors separated by a replaced block cannot both be relocated, because the gap between
    // them would have to be prefilled. The longer run wins and the head absorbs the gap.
    Tokens previous;
    append(previous, segment(1, 1)); // prefix
    append(previous, segment(2, 2)); // replaced by a single block
    append(previous, segment(6, 1)); // survivor, shorter
    append(previous, segment(4, 2)); // replaced by four blocks
    append(previous, segment(5, 3)); // survivor, longer and last in prompt order
    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(6, 1));
    append(current, segment(7, 4));
    append(current, segment(5, 3));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto prompt          = static_cast<std::uint32_t>(current.size());

    const auto plan = plan_tail_relocation(diff, resident, 0, prompt, 6);
    expect(plan.viable, "a tail relocation is found across non-adjacent survivors");
    expect(plan.spans.size() == 1, "only the chosen run is relocated");
    if (plan.spans.size() == 1) {
        expect(plan.spans.front().dest_begin == block_tokens(6),
               "the relocated run sits at its destination");
        expect(plan.spans.front().tokens == block_tokens(2),
               "the longer survivor is the one relocated, trimmed before the final token");
    }
    expect(plan.head.begin == 0 && plan.head.end == block_tokens(6),
           "the head absorbs everything before the run");
    expect(plan.tail.begin == block_tokens(8) && plan.tail.end == prompt,
           "the tail covers the final block");
}

void test_plan_tail_relocation_merges_adjacent_survivors() {
    using ninfer::targets::qwen3_6::detail::plan_tail_relocation;

    // Two survivors separated in the source but adjacent in the destination form a single run even
    // though the detector reports them separately, so both are relocated. The span cap then keeps
    // only the best window inside the run, and equal windows keep the earliest.
    Tokens previous;
    append(previous, segment(1, 1)); // prefix
    append(previous, segment(2, 1)); // replaced, keeps the next survivor off the prefix
    append(previous, segment(3, 1)); // survivor, destination follows the prefix
    append(previous, segment(4, 1)); // replaced, so the two survivors stay separate spans
    append(previous, segment(5, 1)); // survivor, destination abuts the previous one
    append(previous, segment(6, 2)); // replaced by a different two blocks
    Tokens current;
    append(current, segment(1, 1));
    append(current, segment(3, 1));
    append(current, segment(5, 1));
    append(current, segment(7, 2));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto prompt          = static_cast<std::uint32_t>(current.size());

    const auto plan = plan_tail_relocation(diff, resident, 0, prompt, 6);
    expect(plan.viable, "a destination-adjacent run yields a tail relocation");
    expect(plan.spans.size() == 2, "both spans of the run are relocated");
    expect(plan.head.begin == 0 && plan.head.end == block_tokens(1),
           "the head stops at the start of the run");
    expect(plan.tail.begin == block_tokens(3) && plan.tail.end == prompt,
           "the tail covers everything after the run");

    const auto capped = plan_tail_relocation(diff, resident, 0, prompt, 1);
    expect(capped.viable, "a one-span cap still relocates");
    expect(capped.spans.size() == 1, "the span cap keeps a single span");
    if (capped.spans.size() == 1) {
        expect(capped.spans.front().dest_begin == block_tokens(1),
               "equal spans keep the earliest window");
    }
}

void test_plan_page_remap() {
    using ninfer::targets::qwen3_6::detail::kRemapAcquirePage;
    using ninfer::targets::qwen3_6::detail::plan_page_remap;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    // Scope A shift: the surviving suffix moves from logical page 3 to logical page 1 of a three-page
    // prompt whose first page is reused prefix. Only the relocated destination and the prefix are
    // carried; the remaining slot is left for prefill and paid for by the dropped pages.
    const std::vector<std::int32_t> resident{10, 11, 12, 13, 14};
    const std::vector<RelocatedSpan> shift{
        RelocatedSpan{.source_begin = block_tokens(3), .dest_begin = block_tokens(1), .tokens = block_tokens(1)}};

    const auto remap = plan_page_remap(resident, shift, block_tokens(1), block_tokens(3));
    expect(remap.size() == 3, "the remap covers every logical page of the new prompt");
    if (remap.size() == 3) {
        expect(remap[0] == 10, "the reused prefix keeps its page");
        expect(remap[1] == 13, "the relocated block keeps the page that already holds it");
        expect(remap[2] == kRemapAcquirePage, "a prefilled page is left to acquire");
    }

    // A partially reused frontier block is still carried, because prefill overwrites only its tail.
    const auto partial = plan_page_remap(resident, shift, block_tokens(1) + 7, block_tokens(3));
    expect(partial.size() == 3 && partial[0] == 10, "a frontier inside a block still keeps that page");

    // A span whose source page is also a prefix page would name one page twice, which reassigning
    // rejects, so the remap must refuse rather than hand it over.
    const std::vector<std::int32_t> tight{10, 11, 12};
    const std::vector<RelocatedSpan> overlapping{
        RelocatedSpan{.source_begin = 0, .dest_begin = block_tokens(2), .tokens = block_tokens(1)}};
    expect(plan_page_remap(tight, overlapping, block_tokens(2), block_tokens(3)).empty(),
           "a remap that would reuse a page twice is refused");

    // Degenerate and out-of-range inputs fall back rather than producing a bad mapping.
    const std::vector<RelocatedSpan> unaligned{
        RelocatedSpan{.source_begin = block_tokens(3) + 1, .dest_begin = block_tokens(1), .tokens = block_tokens(1)}};
    expect(plan_page_remap(resident, unaligned, block_tokens(1), block_tokens(3)).empty(),
           "an unaligned source is refused");
    const std::vector<RelocatedSpan> beyond{
        RelocatedSpan{.source_begin = block_tokens(9), .dest_begin = block_tokens(1), .tokens = block_tokens(1)}};
    expect(plan_page_remap(resident, beyond, block_tokens(1), block_tokens(3)).empty(),
           "a source page past the resident mapping is refused");
    const std::vector<std::int32_t> no_pages;
    const std::vector<RelocatedSpan> no_spans;
    expect(plan_page_remap(resident, shift, block_tokens(1), 0).empty(),
           "an empty prompt is refused");
    expect(plan_page_remap(no_pages, shift, 0, block_tokens(3)).empty(),
           "an empty resident mapping is refused");
    expect(plan_page_remap(resident, no_spans, 0, block_tokens(3)).empty(),
           "a remap with no relocated spans is refused");
    expect(plan_page_remap(resident, shift, block_tokens(4), block_tokens(3)).empty(),
           "a prefix reaching past the resident mapping is refused");
}

// The second allocation's relocation. MTP keeps its KV in its own pool (`sequence.kv->backend`), so
// its physical pages are numbered independently of the Text allocation's even though its logical
// pages cover the same tokens. `start_prefill_lane` derives one mapping per allocation from the same
// spans, so it is sound only if the spans select the same *logical* page in both. This checks the
// property that makes it sound -- same source logical page to same destination logical page,
// whatever the physical ids are -- and that an allocation which cannot express a span is refused
// outright rather than handed back a mapping of a different length.
void test_plan_page_remap_second_allocation() {
    using ninfer::targets::qwen3_6::detail::kRemapAcquirePage;
    using ninfer::targets::qwen3_6::detail::plan_page_remap;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    const std::vector<std::int32_t> text_resident{10, 11, 12, 13, 14};
    const std::vector<std::int32_t> backend_resident{70, 71, 72, 73, 74};
    const std::vector<RelocatedSpan> shift{RelocatedSpan{.source_begin = block_tokens(3),
                                                        .dest_begin = block_tokens(1),
                                                        .tokens = block_tokens(1)}};

    const auto text_remap =
        plan_page_remap(text_resident, shift, block_tokens(1), block_tokens(3));
    const auto backend_remap =
        plan_page_remap(backend_resident, shift, block_tokens(1), block_tokens(3));
    expect(text_remap.size() == backend_remap.size(),
           "both allocations are mapped over the same logical page count");

    // The logical page an entry came from, -1 for a page prefill will write, and -2 for an entry
    // naming a page the allocation does not own.
    const auto carried_from = [](const std::vector<std::int32_t>& resident,
                                 std::int32_t id) -> std::int32_t {
        if (id == kRemapAcquirePage) { return -1; }
        for (std::size_t page = 0; page < resident.size(); ++page) {
            if (resident[page] == id) { return static_cast<std::int32_t>(page); }
        }
        return -2;
    };
    if (text_remap.size() == backend_remap.size()) {
        for (std::size_t page = 0; page < text_remap.size(); ++page) {
            expect(carried_from(text_resident, text_remap[page]) ==
                       carried_from(backend_resident, backend_remap[page]),
                   "both allocations carry the same source logical page to the same destination");
        }
        expect(carried_from(text_resident, text_remap[1]) == 3,
               "the relocated block still comes from the source allocation's page 3");
    }

    // A second allocation too short to name the spans' source pages cannot be relocated at all. The
    // runtime relies on this being a refusal: a mapping of a different length would leave one cache
    // moved and the other not, which is the corruption the paired derivation exists to prevent.
    const std::vector<std::int32_t> short_backend{70, 71, 72};
    expect(plan_page_remap(short_backend, shift, block_tokens(1), block_tokens(3)).empty(),
           "a second allocation too short for the spans is refused");
}

// The pool demand of a remap: the free pages `reassign_pages` must report, priced before either of a
// paired relocation's two publishes. The hand-computed cases are the shapes a relocation actually
// produces. The randomized half replays the allocation's documented steps -- consume each named page,
// then spend the leftovers on acquire slots -- rather than reusing the closed form, so an off-by-one
// in the surplus shows up as a disagreement instead of a shared mistake.
void test_remap_pages_from_pool() {
    using ninfer::targets::qwen3_6::detail::kRemapAcquirePage;
    using ninfer::targets::qwen3_6::detail::remap_pages_from_pool;

    // The Scope A shift of `test_plan_page_remap`: page 3 is promoted to page 1 and the tail is
    // acquired. Three pages are dropped and one is acquired, so the surplus pays for the acquire and
    // the pool is never asked.
    const std::vector<std::int32_t> five_pages{10, 11, 12, 13, 14};
    const std::vector<std::int32_t> promoted{10, 13, kRemapAcquirePage};
    expect(remap_pages_from_pool(five_pages, promoted) == 0,
           "a dropped page pays for an acquired one");

    // Nothing kept: every slot is acquired, both dropped pages are reused for the job, and the third
    // still has to come from the pool.
    const std::vector<std::int32_t> two_pages{10, 11};
    const std::vector<std::int32_t> all_acquired{kRemapAcquirePage, kRemapAcquirePage,
                                                 kRemapAcquirePage};
    expect(remap_pages_from_pool(two_pages, all_acquired) == 1,
           "acquires past the surplus reach the pool");

    // An identity mapping keeps everything and acquires nothing.
    const std::vector<std::int32_t> three_pages{10, 11, 12};
    expect(remap_pages_from_pool(three_pages, three_pages) == 0,
           "an identity remap costs the pool nothing");

    // Growth against a surplus too small to matter: one page kept, three acquired.
    const std::vector<std::int32_t> one_page{10};
    const std::vector<std::int32_t> one_kept_three_acquired{10, kRemapAcquirePage,
                                                            kRemapAcquirePage, kRemapAcquirePage};
    expect(remap_pages_from_pool(one_page, one_kept_three_acquired) == 3,
           "growth past a single kept page needs every page from the pool");

    // A partially reused frontier block is carried, because prefill overwrites only its tail, so it
    // is not surplus and cannot pay for an acquire: two acquires against one carried page and one
    // dropped page needs one page from the pool. Counting the carried page as surplus instead would
    // report zero and let the second publish fail after the first.
    const std::vector<std::int32_t> frontier_carried{10, kRemapAcquirePage, kRemapAcquirePage};
    expect(remap_pages_from_pool(two_pages, frontier_carried) == 1,
           "a carried frontier page is not surplus");

    // `reassign_pages` rejects a mapping naming unowned pages before this arithmetic matters; the
    // helper must still not underflow on one (owned 1, named 2, surplus clamps to zero).
    const std::vector<std::int32_t> more_named_than_owned{10, 11, kRemapAcquirePage};
    expect(remap_pages_from_pool(one_page, more_named_than_owned) == 1,
           "a mapping naming more pages than owned clamps its surplus to zero");

    std::uint32_t seed = 0x9e3779b9u;
    const auto next    = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };
    std::uint32_t mismatches = 0;
    std::uint32_t cases      = 0;
    for (int iteration = 0; iteration < 5000; ++iteration) {
        const std::uint32_t owned_count   = 1 + (next() % 12);
        const std::uint32_t keep_count    = next() % (owned_count + 1);
        const std::uint32_t acquire_count = next() % 6;

        std::vector<std::int32_t> owned;
        owned.reserve(owned_count);
        for (std::uint32_t i = 0; i < owned_count; ++i) {
            owned.push_back(static_cast<std::int32_t>(100 + i));
        }
        // Shuffle the owned pages, so the kept set is not always a prefix of them.
        for (std::uint32_t i = owned_count; i > 1; --i) {
            const std::uint32_t j = next() % i;
            std::swap(owned[i - 1], owned[j]);
        }

        std::vector<std::int32_t> mapping(owned.begin(),
                                          owned.begin() + static_cast<std::ptrdiff_t>(keep_count));
        for (std::uint32_t i = 0; i < acquire_count; ++i) {
            mapping.push_back(kRemapAcquirePage);
        }
        // Interleave, so carried and acquired slots alternate the way a real remap lays them out.
        for (std::size_t i = mapping.size(); i > 1; --i) {
            const std::size_t j = next() % i;
            std::swap(mapping[i - 1], mapping[j]);
        }

        std::vector<std::int32_t> remaining(owned);
        std::uint32_t acquires = 0;
        for (const std::int32_t page : mapping) {
            if (page == kRemapAcquirePage) {
                ++acquires;
                continue;
            }
            const auto found = std::find(remaining.begin(), remaining.end(), page);
            if (found != remaining.end()) { remaining.erase(found); }
        }
        const std::uint32_t surplus = static_cast<std::uint32_t>(remaining.size());
        const std::uint32_t oracle  = acquires > surplus ? acquires - surplus : 0;
        if (remap_pages_from_pool(owned, mapping) != oracle) { ++mismatches; }
        ++cases;
    }
    expect(mismatches == 0, "every randomized remap's pool demand matches the oracle");
    expect(cases == 5000, "the randomized remap sweep ran every case");
}

void test_determinism() {
    Tokens previous;
    append(previous, segment(1, 2));
    append(previous, segment(2, 3));
    Tokens current;
    append(current, segment(1, 2));
    append(current, segment(2, 1));

    const PromptEditDiff first  = diff_prompt_edit(previous, current);
    const PromptEditDiff second = diff_prompt_edit(previous, current);

    expect(spans_equal(first.spans, second.spans), "detection is deterministic");
    expect(first.reusable_tokens == second.reusable_tokens, "reusable token count is deterministic");
}

// The source-shifting path. Whole-block detection can only match content that sits on the same
// 64-token grid in both prompts, so a run shifted by a count the page cannot express is invisible to
// it -- which is the shape of nearly every edit a user types. These pin the two halves that fix it:
// a detector that anchors at every offset, and a selector that quantizes only the destination, since
// a page carries one RoPE delta and the destination is the end that has to stay whole pages.
Tokens varied(std::int32_t label, std::uint32_t blocks) {
    // Unique per (block, position), unlike `segment`: a block whose tokens are all equal would still
    // match after an arbitrary shift, which is exactly the property these tests must not rely on.
    Tokens out;
    out.reserve(static_cast<std::size_t>(blocks) * kPromptEditBlockTokens);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t i = 0; i < kPromptEditBlockTokens; ++i) {
            out.push_back(label * 100000 +
                          static_cast<std::int32_t>(block * kPromptEditBlockTokens + i));
        }
    }
    return out;
}

Tokens shifted_scenario(Tokens& previous, Tokens& current) {
    append(previous, varied(1, 3));
    append(previous, varied(2, 5));
    append(previous, varied(3, 1));
    append(current, varied(1, 3));
    for (std::int32_t junk = 0; junk < 3; ++junk) { current.push_back(-7 - junk); }
    append(current, varied(2, 5));
    append(current, varied(4, 1));
    return previous;
}

void test_shifted_detector_finds_a_run_off_the_block_grid() {
    using ninfer::targets::qwen3_6::detail::diff_prompt_edit_shifted;

    Tokens previous;
    Tokens current;
    shifted_scenario(previous, current);

    const PromptEditDiff aligned = diff_prompt_edit(previous, current);
    expect(aligned.spans.size() == 1, "a 3-token insert leaves the block-aligned detector one span");
    if (aligned.spans.size() == 1) {
        expect(aligned.spans.front().old_begin == 0 && aligned.spans.front().new_begin == 0,
               "and that span is the leading prefix, which relocation never moves");
    }

    const PromptEditDiff shifted = diff_prompt_edit_shifted(previous, current);
    expect(shifted.spans.size() == 2, "the shifted detector finds the prefix and the moved run");
    if (shifted.spans.size() == 2) {
        expect(shifted.spans[1].old_begin == block_tokens(3),
               "the moved run's source is where it was");
        expect(shifted.spans[1].new_begin == block_tokens(3) + 3,
               "and its destination moved by the three inserted tokens");
        expect(shifted.spans[1].tokens == block_tokens(5), "grown to the run's true length");
    }
    expect(shifted.reusable_tokens == block_tokens(3) + block_tokens(5),
           "reuse is counted over both runs");
}

void test_segmented_plan_needs_the_shifted_detector() {
    using ninfer::targets::qwen3_6::detail::diff_prompt_edit_shifted;
    using ninfer::targets::qwen3_6::detail::plan_relocations;
    using ninfer::targets::qwen3_6::detail::plan_shifted_relocations;
    using ninfer::targets::qwen3_6::detail::relocation_is_expressible;
    using ninfer::targets::qwen3_6::detail::relocation_prefill_ranges;

    // The wiring rather than the arithmetic: the runtime builds one diff and hands it to whichever
    // selector the mode names, so a detector that cannot see the edit's survivors empties every plan
    // built from it. That is the failure this pins -- an off-grid edit looked exactly like SCR
    // declining to engage, when in fact nothing downstream was ever given anything to keep.
    Tokens previous;
    Tokens current;
    shifted_scenario(previous, current);

    const PromptEditDiff aligned = diff_prompt_edit(previous, current);
    const PromptEditDiff shifted = diff_prompt_edit_shifted(previous, current);
    const auto resident = static_cast<std::uint32_t>(previous.size());
    const auto prompt   = static_cast<std::uint32_t>(current.size());

    expect(plan_relocations(aligned, resident, 0, prompt, std::nullopt, 6).empty(),
           "the segmented selector keeps nothing from the aligned diff: its only survivor was the "
           "leading prefix, which is ordinary prefix reuse");
    expect(plan_shifted_relocations(aligned, resident, 0, prompt, std::nullopt, 6, true).empty(),
           "and the shifted fallback cannot recover what that detector never found");

    const auto spans =
        plan_shifted_relocations(shifted, resident, 0, prompt, std::nullopt, 6, true);
    expect(!spans.empty(), "the all-offset diff hands the fallback the run the edit moved");
    const auto ranges = relocation_prefill_ranges(spans, 0, prompt);
    expect(relocation_is_expressible(spans, ranges, 0, prompt, true),
           "and the plan built from it is one the copied route can execute");
}

void test_shifted_selector_quantizes_only_the_destination() {
    using ninfer::targets::qwen3_6::detail::diff_prompt_edit_shifted;
    using ninfer::targets::qwen3_6::detail::plan_shifted_relocations;
    using ninfer::targets::qwen3_6::detail::relocation_is_expressible;
    using ninfer::targets::qwen3_6::detail::relocation_prefill_ranges;

    Tokens previous;
    Tokens current;
    shifted_scenario(previous, current);

    const PromptEditDiff diff = diff_prompt_edit_shifted(previous, current);
    const auto resident = static_cast<std::uint32_t>(previous.size());
    const auto prompt   = static_cast<std::uint32_t>(current.size());

    const auto spans = plan_shifted_relocations(diff, resident, 0, prompt, std::nullopt, 6);
    expect(spans.size() == 1, "the moved run is selected");
    if (spans.size() == 1) {
        expect(spans.front().dest_begin == block_tokens(4),
               "the destination is trimmed up to the first whole page in the run");
        expect(spans.front().tokens == block_tokens(4),
               "and back to the last whole page before the final token");
        expect(spans.front().source_begin == block_tokens(3) + 61,
               "the source keeps the run's own offset, which is what the page grid could not express");
        expect(spans.front().source_begin % kPromptEditBlockTokens != 0,
               "so an off-grid source is the normal case here, not an error");
    }

    const auto ranges = relocation_prefill_ranges(spans, 0, prompt);
    expect(relocation_is_expressible(spans, ranges, 0, prompt, true),
           "a whole-page destination with an off-grid source is expressible when the payload is copied");
    expect(!relocation_is_expressible(spans, ranges, 0, prompt),
           "and refused by the re-pointing path, whose source has to name a page that holds it");

    const auto destination_free =
        plan_shifted_relocations(diff, resident, 0, prompt, std::nullopt, 6, true);
    expect(destination_free.size() == 1, "boundary-aware attention keeps the untrimmed run");
    if (destination_free.size() == 1) {
        expect(destination_free.front().dest_begin == block_tokens(3) + 3,
               "destination freedom preserves the run's exact token offset");
         expect(destination_free.front().tokens == block_tokens(5),
             "the full surviving run is retained when it ends before the prompt tail");
        const auto destination_free_ranges =
            relocation_prefill_ranges(destination_free, 0, prompt);
        expect(relocation_is_expressible(destination_free, destination_free_ranges, 0, prompt, true),
               "token-granular copied ranges still tile the prompt");
    }
}

void test_plan_copied_page_remap_acquires_every_relocated_page() {
    using ninfer::targets::qwen3_6::detail::kRemapAcquirePage;
    using ninfer::targets::qwen3_6::detail::plan_copied_page_remap;
    using ninfer::targets::qwen3_6::detail::plan_page_remap;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;

    // The same shift `plan_page_remap` carries by re-pointing a page, but with an off-grid source:
    // every copied destination page is acquired (the copy writes it) and the source page the
    // re-pointing version would have named is simply absent, which is what drops it.
    const std::vector<std::int32_t> resident{10, 11, 12, 13, 14};
    const std::vector<RelocatedSpan> shift{RelocatedSpan{.source_begin = block_tokens(3) + 61,
                                                        .dest_begin = block_tokens(4),
                                                        .tokens = block_tokens(2)}};

    const auto remap = plan_copied_page_remap(resident, shift, 0, block_tokens(6));
    expect(remap.size() == 6, "the copied mapping covers every logical page of the new prompt");
    if (remap.size() == 6) {
        expect(remap[0] == 10, "the reused prefix still keeps its own page");
        expect(remap[1] == kRemapAcquirePage && remap[2] == kRemapAcquirePage,
               "a prefilled page is acquired");
        expect(remap[4] == kRemapAcquirePage && remap[5] == kRemapAcquirePage,
               "and a copied destination page is acquired too, never pointed at its own source");
    }

    // The re-pointing mapping refuses this shape outright. That difference is what the copy buys.
    expect(plan_page_remap(resident, shift, 0, block_tokens(6)).empty(),
           "the re-pointing mapping cannot express an off-grid source");

    const std::vector<RelocatedSpan> partial{
        RelocatedSpan{.source_begin = block_tokens(3), .dest_begin = block_tokens(2) + 1,
                      .tokens = block_tokens(2)}};
    const auto partial_remap = plan_copied_page_remap(resident, partial, 0, block_tokens(6));
    expect(partial_remap.size() == 6, "a partial copied span still has a complete page mapping");
    if (partial_remap.size() == 6) {
        expect(partial_remap[2] == kRemapAcquirePage && partial_remap[3] == kRemapAcquirePage,
               "both pages touched by a partial destination span are acquired");
    }
    const std::vector<RelocatedSpan> beyond{
        RelocatedSpan{.source_begin = 0, .dest_begin = block_tokens(6), .tokens = block_tokens(2)}};
    expect(plan_copied_page_remap(resident, beyond, 0, block_tokens(6)).empty(),
           "a destination reaching past the prompt is refused");
}

void test_shifted_pricing_excludes_the_copy_sources() {
    using ninfer::targets::qwen3_6::detail::kRemapAcquirePage;
    using ninfer::targets::qwen3_6::detail::remap_pages_from_pool;
    using ninfer::targets::qwen3_6::detail::shifted_relocation_pages_from_pool;

    // Five owned pages, none named by the mapping, so every acquire slot has a dropped page to take
    // and the re-pointing price is zero pool pages. The copies read logical pages 0 and 1, whose
    // physical pages have to survive until the copy has moved their bytes, so those two cannot be
    // destinations and the pool must supply them instead. Without this the substitution hands back
    // the very pages being read, and a device copy of overlapping ranges is undefined.
    const std::vector<std::int32_t> owned{10, 11, 12, 13, 14};
    const std::vector<std::int32_t> all_acquired{
        kRemapAcquirePage, kRemapAcquirePage, kRemapAcquirePage, kRemapAcquirePage,
        kRemapAcquirePage};
    const std::vector<std::uint32_t> sources{0, 1};

    expect(remap_pages_from_pool(owned, all_acquired) == 0,
           "without copies the surplus pays for every acquire slot");
    expect(shifted_relocation_pages_from_pool(owned, all_acquired, sources) == 2,
           "a copy source cannot also be a destination, so the pool pays for those slots");

    const std::vector<std::uint32_t> no_sources;
    expect(shifted_relocation_pages_from_pool(owned, all_acquired, no_sources) == 0,
           "no sources means no exclusion");
    const std::vector<std::uint32_t> unnamed{5, 6};
    expect(shifted_relocation_pages_from_pool(owned, all_acquired, unnamed) == 0,
           "a logical page the mapping does not name blocks nothing");
}

// The planner prices a copied relocation so it can refuse one the pool cannot afford, and the copier
// performs a substitution that spends the pages it priced. Those are two functions in two layers,
// written from the same rules but not from one expression, and the failure of the pair is not
// symmetric: a price that *under*-states the demand is refused by the allocation's own check after
// the planner has already reported the reuse, which is a 500 on a prompt shape the pool can serve.
//
// So the two are compared here over shapes that put the copied destinations, the pages they read and
// the pages the mapping drops in different relations to each other. The price is allowed to be
// larger -- a source the mapping still names blocks budget it never could have spent -- but never
// smaller, and where every source is a dropped page the two have to agree exactly.
void test_copied_price_covers_the_substitution_the_copier_performs() {
    using ninfer::kPagedKVAquirePage;
    using ninfer::plan_relocated_page_moves;
    using ninfer::plan_relocated_page_substitution;
    using ninfer::RelocatedCopySpan;
    using ninfer::RelocatedPageMove;
    using ninfer::targets::qwen3_6::detail::kRemapAcquirePage;
    using ninfer::targets::qwen3_6::detail::plan_copied_page_remap;
    using ninfer::targets::qwen3_6::detail::remap_pages_from_pool;
    using ninfer::targets::qwen3_6::detail::RelocatedSpan;
    using ninfer::targets::qwen3_6::detail::shifted_relocation_pages_from_pool;

    expect(kRemapAcquirePage == kPagedKVAquirePage,
           "the planner's acquire sentinel is the one the core substitution spells");

    bool saw_a_strict_case = false;
    bool saw_an_exact_case = false;
    for (std::uint32_t resident = 12; resident <= 20; resident += 4) {
        for (std::uint32_t offset = 0; offset < kPromptEditBlockTokens; offset += 13) {
            for (std::uint32_t dest_page = 1; dest_page <= 3; ++dest_page) {
                // An identity mapping, so a logical page and its physical page are told apart by the
                // arithmetic under test rather than by the ids looking different.
                std::vector<std::int32_t> current(resident);
                for (std::uint32_t page = 0; page < resident; ++page) {
                    current[page] = static_cast<std::int32_t>(30 + page);
                }
                const std::uint32_t prompt_tokens = resident * kPromptEditBlockTokens;
                const std::uint32_t prefix_reuse  = kPromptEditBlockTokens;
                const std::vector<RelocatedSpan> spans{
                    RelocatedSpan{.source_begin = 4 * kPromptEditBlockTokens + offset,
                                  .dest_begin   = dest_page * kPromptEditBlockTokens,
                                  .tokens       = 3 * kPromptEditBlockTokens}};

                const std::vector<std::int32_t> remap =
                    plan_copied_page_remap(current, spans, prefix_reuse, prompt_tokens);
                expect(!remap.empty(), "a whole-page destination is an expressible copied relocation");
                if (remap.empty()) { continue; }

                const std::vector<RelocatedCopySpan> copy_spans{RelocatedCopySpan{
                    spans[0].source_begin, spans[0].dest_begin, spans[0].tokens}};
                const std::vector<RelocatedPageMove> moves =
                    plan_relocated_page_moves(copy_spans, current.size());

                // What the runtime prices: the pages the moves read, in logical coordinates. Not a
                // page-by-page walk of the source range -- that misses the page holding the span's
                // last token whenever the span does not end on a page boundary, which is every
                // off-grid span, and a price one page short is a refusal after the fact.
                std::vector<std::uint32_t> source_logical;
                for (const RelocatedPageMove& move : moves) {
                    source_logical.push_back(static_cast<std::uint32_t>(move.source_page));
                }
                std::sort(source_logical.begin(), source_logical.end());
                source_logical.erase(std::unique(source_logical.begin(), source_logical.end()),
                                     source_logical.end());
                const std::uint32_t price =
                    shifted_relocation_pages_from_pool(current, remap, source_logical);

                std::vector<std::int32_t> protected_pages;
                protected_pages.reserve(moves.size());
                for (const RelocatedPageMove& move : moves) {
                    protected_pages.push_back(
                        current[static_cast<std::size_t>(move.source_page)]);
                }
                const ninfer::RelocatedPageSubstitution substitution =
                    plan_relocated_page_substitution(current, remap, protected_pages);

                if (price < substitution.from_pool) {
                    std::cerr << "FAIL: resident=" << resident << " offset=" << offset
                              << " dest_page=" << dest_page << " priced " << price
                              << " but the copier spends " << substitution.from_pool << '\n';
                    ++failures;
                }
                if (source_logical.front() > prefix_reuse / kPromptEditBlockTokens) {
                    // Every page the copies read is above the carried prefix, so all of them are
                    // dropped and the two functions are counting the same occurrences.
                    saw_an_exact_case = true;
                    if (price != substitution.from_pool) {
                        std::cerr << "FAIL: resident=" << resident << " offset=" << offset
                                  << " dest_page=" << dest_page << " priced " << price
                                  << " against a demand of " << substitution.from_pool << '\n';
                        ++failures;
                    }
                }
                if (remap_pages_from_pool(current, remap) < substitution.from_pool) {
                    saw_a_strict_case = true;
                }
            }
        }
    }
    expect(saw_an_exact_case, "the sweep reached a shape where the two agree exactly");
    expect(saw_a_strict_case,
           "and one where the re-pointing price would have understated the copying one");
}

} // namespace

int main() {
    test_identical_prompts();
    test_reasoning_stripping();
    test_in_place_replacement();
    test_multiple_surviving_spans();
    test_block_split_forfeits_partial_block();
    test_partial_tail_is_never_reused();
    test_repeated_blocks_stay_monotone();
    test_empty_and_degenerate_inputs();
    test_plan_relocations();
    test_plan_relocations_keeps_longest();
    test_plan_relocations_respects_limits();
    test_relocation_prefill_ranges();
    test_relocation_prefill_splits();
    test_relocation_is_expressible();
    test_plan_tail_relocation();
    test_plan_tail_relocation_respects_turn_rewrite_boundary();
    test_plan_tail_relocation_picks_one_run();
    test_plan_tail_relocation_merges_adjacent_survivors();
    test_plan_page_remap();
    test_plan_page_remap_second_allocation();
    test_remap_pages_from_pool();
    test_shifted_detector_finds_a_run_off_the_block_grid();
    test_segmented_plan_needs_the_shifted_detector();
    test_shifted_selector_quantizes_only_the_destination();
    test_plan_copied_page_remap_acquires_every_relocated_page();
    test_shifted_pricing_excludes_the_copy_sources();
    test_copied_price_covers_the_substitution_the_copier_performs();
    test_determinism();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 prompt edit checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 prompt edit checks passed\n";
    return 0;
}
