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

std::vector<RelocatedSpan> plan_relocations(const PromptEditDiff& diff,
                                           std::uint32_t resident_kv_tokens,
                                           std::uint32_t prefix_reuse_tokens,
                                           std::uint32_t max_spans) {
    std::vector<RelocatedSpan> candidates;
    if (max_spans == 0) { return candidates; }

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
        candidates.push_back(RelocatedSpan{.source_begin = span.old_begin,
                                           .dest_begin   = span.new_begin,
                                           .tokens       = span.tokens});
    }

    // Longest survivors first so the cap discards the least useful reuse.
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

} // namespace ninfer::targets::qwen3_6::detail
