// Phase 1-2 SCR checks: surviving-span detection over an edited prompt.
//
// The detector is host-only and dependency-free, so this file also builds with a plain
// host compiler:
//   g++ -std=c++20 -I src -I include
//       src/targets/qwen3_6/impl/runtime/prompt_edit.cpp
//       tests/targets/qwen3_6/test_prompt_edit.cpp -o /tmp/test_prompt_edit

#include "targets/qwen3_6/impl/runtime/prompt_edit.h"

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
    // surviving suffix should be reported as a relocation.
    Tokens previous;
    append(previous, segment(1, 2));
    append(previous, segment(2, 2));
    append(previous, segment(3, 2));
    Tokens current;
    append(current, segment(1, 2));
    append(current, segment(3, 2));

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident        = static_cast<std::uint32_t>(previous.size());
    const auto relocations     = plan_relocations(diff, resident, 0, 6);

    expect(relocations.size() == 1, "only the surviving suffix is relocated");
    if (relocations.size() == 1) {
        expect(relocations.front().source_begin == block_tokens(4), "relocation source is the old suffix");
        expect(relocations.front().dest_begin == block_tokens(2), "relocation destination is the new suffix");
        expect(relocations.front().tokens == block_tokens(2), "relocation length is the suffix length");
    }

    // Zero cap means reuse nothing, which is how SCR is disabled per request.
    expect(plan_relocations(diff, resident, 0, 0).empty(), "a zero span cap relocates nothing");

    // A span without resident KV cannot be relocated.
    expect(plan_relocations(diff, block_tokens(5), 0, 6).empty(),
           "a span outside resident KV is not relocated");

    // The surviving suffix moves from block 4 to block 2. When ordinary prefix reuse already
    // covers the first four blocks, the suffix starts at the frontier and is still relocated.
    expect(plan_relocations(diff, resident, block_tokens(2), 6).size() == 1,
           "a survivor exactly at the prefix frontier is relocated");

    // Once prefix reuse reaches into the survivor's destination, that region is already covered
    // and must not be claimed a second time.
    expect(plan_relocations(diff, resident, block_tokens(3), 6).empty(),
           "a survivor inside the prefix-reuse region is not relocated");
    expect(plan_relocations(diff, resident, block_tokens(4), 6).empty(),
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

    const PromptEditDiff diff = diff_prompt_edit(previous, current);
    const auto resident = static_cast<std::uint32_t>(previous.size());

    // All three survivors plus the prefix are detected before capping.
    expect(diff.spans.size() == 4, "the fixture keeps its three survivors separate");
    expect(diff.relocated_tokens() == block_tokens(6),
           "the fixture relocates one, two, and three blocks");

    // The cap is smaller than the number of candidates; the dropped span is the one-block
    // survivor, not the earliest candidate.
    const auto capped = plan_relocations(diff, resident, 0, 2);
    expect(capped.size() == 2, "the span cap is honored");
    if (capped.size() == 2) {
        expect(capped[0].tokens == block_tokens(2) && capped[1].tokens == block_tokens(3),
               "the cap keeps the longest survivors");
        expect(capped[0].dest_begin < capped[1].dest_begin, "relocations are returned in prompt order");
        expect(capped[0].source_begin == block_tokens(5) && capped[0].dest_begin == block_tokens(6),
               "a capped survivor reports its own source and destination");
    }

    const auto all = plan_relocations(diff, resident, 0, 6);
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
    test_plan_tail_relocation();
    test_plan_tail_relocation_picks_one_run();
    test_plan_tail_relocation_merges_adjacent_survivors();
    test_determinism();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 prompt edit checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 prompt edit checks passed\n";
    return 0;
}
